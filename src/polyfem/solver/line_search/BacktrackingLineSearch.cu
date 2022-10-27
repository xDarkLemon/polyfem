#pragma once

// #include <polyfem/utils/Logger.hpp>
// #include <polyfem/utils/Timer.hpp>

#include <polyfem/solver/NLProblem.hpp>
#include <polyfem/solver/FullNLProblem.hpp>
#include <polyfem/solver/line_search/LineSearch.hpp>
#include <polyfem/solver/line_search/BacktrackingLineSearch.hpp>

#include <cuda.h>
#include <cuda_runtime_api.h>
#include "polyfem/utils/CUDA_utilities.cuh"
#include "cublas_v2.h"

#include <cfenv>

using namespace std;

namespace polyfem
{
	namespace solver
	{
		namespace line_search
		{
            template <typename ProblemType>
            double BacktrackingLineSearch<ProblemType>::compute_descent_step_size_gpu(
                const Eigen::Matrix<double, -1, 1> &x,
                const Eigen::Matrix<double, -1, 1> &delta_x,
                ProblemType &objFunc,
                const double old_energy_in,
                const double starting_step_size)
            {
				double step_size = starting_step_size;

                TVector grad(x.rows());
                {
                    POLYFEM_SCOPED_TIMER("compute grad in LS", this->compute_grad_time);
                    objFunc.gradient(x, grad);
                }

                const int N = x.rows();
                double *x_dev, *delta_x_dev, *grad_dev, *new_x_dev;
                double *new_x_host = new double[N];

                grad_dev = ALLOCATE_GPU<double>(grad_dev, N*sizeof(double));
                new_x_dev = ALLOCATE_GPU<double>(new_x_dev, N*sizeof(double));
                x_dev = ALLOCATE_GPU<double>(x_dev, N*sizeof(double));
                delta_x_dev = ALLOCATE_GPU<double>(delta_x_dev, N*sizeof(double));

                {
                    POLYFEM_SCOPED_TIMER("move data between device and host in LS", this->move_data_time);
                    COPYDATATOGPU<double>(grad_dev, grad.data(), N*sizeof(double));
                    COPYDATATOGPU<double>(new_x_dev, x.data(), N*sizeof(double));
                    COPYDATATOGPU<double>(x_dev, x.data(), N*sizeof(double));
                    COPYDATATOGPU<double>(delta_x_dev, delta_x.data(), N*sizeof(double));
                }

                cublasHandle_t handle;
                cublasCreate(&handle);

                bool use_grad_norm;
                double old_energy;
                {
                    POLYFEM_SCOPED_TIMER("compute grad norm in LS", this->compute_grad_norm_time);
                    // use_grad_norm = grad.norm() < this->use_grad_norm_tol;  // NonlinearSolver json parameter, init to -1 in LineSearch
                    // old_energy = use_grad_norm ? grad.squaredNorm() : old_energy_in;
                    double grad_norm;
                    cublasDnrm2(handle, N, grad_dev, 1, &grad_norm);
                    use_grad_norm = grad_norm < this->use_grad_norm_tol; 
                    if(use_grad_norm==true)
                        cublasDdot(handle, N, grad_dev, 1, grad_dev, 1, &old_energy);
                    else
                        old_energy=old_energy_in;
                }

                // Find step that reduces the energy
                double cur_energy = std::nan("");
                bool is_step_valid = false;
                while (step_size > this->min_step_size && this->cur_iter < this->max_step_size_iter)
                {
                    this->iterations++;

                    // TVector new_x = x + step_size * delta_x;
					{
                        POLYFEM_SCOPED_TIMER("move data between device and host in LS", this->move_data_time);
                        cudaMemcpy(new_x_dev, x_dev, N*sizeof(double), cudaMemcpyDeviceToDevice);
                    }
                    {
                        POLYFEM_SCOPED_TIMER("compute new_x in LS", this->compute_new_x_time);
                        cublasDaxpy(handle, N, &step_size, delta_x_dev, 1, new_x_dev, 1);
                    }
                    {
                        POLYFEM_SCOPED_TIMER("move data between device and host in LS", this->move_data_time);
                        COPYDATATOHOST<double>(new_x_host, new_x_dev, N*sizeof(double));
                    }
                    Eigen::Matrix<double, -1, 1> new_x(Eigen::Map<Eigen::Matrix<double, -1, 1>>(new_x_host,N));

                    {
                        POLYFEM_SCOPED_TIMER("constraint set update in LS", this->constraint_set_update_time);
                        objFunc.solution_changed(new_x);
                    }

                    {
                        POLYFEM_SCOPED_TIMER("compute grad/value in LS", this->compute_grad_or_value_time);
                        if (use_grad_norm)
                        {
                            {
                                POLYFEM_SCOPED_TIMER("compute grad in LS", this->compute_grad_time);
                                objFunc.gradient(new_x, grad);
                            }
                            // cur_energy = grad.squaredNorm();
                            {
                                POLYFEM_SCOPED_TIMER("move data between device and host in LS", this->move_data_time);
                                COPYDATATOGPU<double>(grad_dev, grad.data(), N*sizeof(double));
                            }
                            {
                                POLYFEM_SCOPED_TIMER("compute grad norm in LS", this->compute_grad_norm_time);
                                cublasDdot(handle, N, grad_dev, 1, grad_dev, 1, &cur_energy);
                            }
                        }
                        else
                        {
                            POLYFEM_SCOPED_TIMER("compute value in LS", this->compute_value_time);
                            cur_energy = objFunc.value(new_x);
                        }
                    }

                    {
                        POLYFEM_SCOPED_TIMER("is step valid in LS", this->is_step_valid_time);
                        is_step_valid = objFunc.is_step_valid(x, new_x);
                    }
			        // TVector grad = TVector::Zero(objFunc.reduced_size);
			        // gradient(new_x, grad, true);

                    logger().trace("ls it: {} delta: {} invalid: {} ", this->cur_iter, (cur_energy - old_energy), !is_step_valid);

                    // if (!std::isfinite(cur_energy) || (cur_energy >= old_energy && fabs(cur_energy - old_energy) > 1e-12) || !is_step_valid)
                    if (!std::isfinite(cur_energy) || cur_energy > old_energy || !is_step_valid)
                    {
                        step_size /= 2.0;
                        // max_step_size should return a collision free step
                        // assert(objFunc.is_step_collision_free(x, new_x));
                    }
                    else
                    {
                        break;
                    }
                    this->cur_iter++;
                }

                if (this->cur_iter >= this->max_step_size_iter || step_size <= this->min_step_size)
                {
                    logger().warn(
                        "Line search failed to find descent step (f(x)={:g} f(x+αΔx)={:g} α_CCD={:g} α={:g}, ||Δx||={:g} is_step_valid={} iter={:d})",
                        old_energy, cur_energy, starting_step_size, step_size, delta_x.norm(),
                        is_step_valid ? "true" : "false", this->cur_iter);
                    objFunc.line_search_end();
                    return std::nan("");
                }

                cudaFree(grad_dev);
                cudaFree(new_x_dev);
                cudaFree(x_dev);
                cudaFree(delta_x_dev);
                cublasDestroy(handle);

                return step_size;
            }
            template class BacktrackingLineSearch<polyfem::solver::NLProblem>;
            template class BacktrackingLineSearch<polyfem::solver::FullNLProblem>;
        }
    }
}