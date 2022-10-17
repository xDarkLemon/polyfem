#pragma once

// #include "LineSearch.hpp"

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

// #include <thrust/host_vector.h>
// #include <thrust/device_vector.h>
// #include <thrust/copy.h>
// #include <thrust/inner_product.h>

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
                objFunc.gradient(x, grad);
                const bool use_grad_norm = grad.norm() < this->use_grad_norm_tol;  // NonlinearSolver json parameter, init to -1 in LineSearch

                const double old_energy = use_grad_norm ? grad.squaredNorm() : old_energy_in;
                
                // move data to gpu
                const int N = x.rows();
                double *x_dev, *delta_x_dev, *grad_dev, *new_x_dev;
                double *grad_host = grad.data();
                const double *x_host = x.data();
                const double *delta_x_host = delta_x.data();
                double *new_x_host = new double[N];

                grad_dev = ALLOCATE_GPU<double>(grad_dev, N*sizeof(double));
                new_x_dev = ALLOCATE_GPU<double>(new_x_dev, N*sizeof(double));
                x_dev = ALLOCATE_GPU<double>(x_dev, N*sizeof(double));
                delta_x_dev = ALLOCATE_GPU<double>(delta_x_dev, N*sizeof(double));

                COPYDATATOGPU<double>(grad_dev, grad_host, N*sizeof(double));
                cudaMemset(new_x_dev, 0, N*sizeof(double));
                COPYDATATOGPU<double>(x_dev, x_host, N*sizeof(double));
                COPYDATATOGPU<double>(delta_x_dev, delta_x_host, N*sizeof(double));

                cublasHandle_t handle;
                cublasCreate(&handle);

                // Find step that reduces the energy
                double cur_energy = std::nan("");
                bool is_step_valid = false;
                while (step_size > this->min_step_size && this->cur_iter < this->max_step_size_iter)
                {
                    this->iterations++;

                    // TVector new_x = x + step_size * delta_x;

                    COPYDATATOGPU<double>(new_x_dev, x_host, N*sizeof(double));
                    cublasDaxpy(handle, N, &step_size, delta_x_dev, 1, new_x_dev, 1);
                    COPYDATATOHOST<double>(new_x_host, new_x_dev, N*sizeof(double));
                    Eigen::Matrix<double, -1, 1> new_x(Eigen::Map<Eigen::Matrix<double, -1, 1>>(new_x_host,N));

                    {
                        POLYFEM_SCOPED_TIMER("constraint set update in LS", this->constraint_set_update_time);
                        objFunc.solution_changed(new_x);
                        // new_x_host = new_x.data();
                    }

                    if (use_grad_norm)
                    {
                        objFunc.gradient(new_x, grad);
                        // cur_energy = grad.squaredNorm();
                        grad_host = grad.data();
                        COPYDATATOGPU<double>(grad_dev, grad_host, N*sizeof(double));
                        cublasDdot(handle, N, grad_dev, 1, grad_dev, 1, &cur_energy);
                    }
                    else
                        cur_energy = objFunc.value(new_x);

                    is_step_valid = objFunc.is_step_valid(x, new_x);
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