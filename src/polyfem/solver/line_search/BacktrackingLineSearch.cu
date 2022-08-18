#pragma once

// #include "LineSearch.hpp"

// #include <polyfem/utils/Logger.hpp>
// #include <polyfem/utils/Timer.hpp>

#include <polyfem/solver/NLProblem.hpp>
#include <polyfem/solver/ALNLProblem.hpp>
#include <polyfem/solver/line_search/LineSearch.hpp>
#include <polyfem/solver/line_search/BacktrackingLineSearch.hpp>

#include <cuda.h>
#include <cuda_runtime_api.h>
#include "polyfem/assembler/CUDA_utilities.cuh"
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
                const bool use_grad_norm = grad.norm() < this->use_grad_norm_tol;

                const double old_energy = use_grad_norm ? grad.squaredNorm() : old_energy_in;
                
                // move data to gpu
                int N = x.rows();
                double *x_dev, *delta_x_dev;
                double *step_size_dev, *cur_energy_dev, *old_energy_dev, *grad_dev, *new_x_iter_dev, *tmp_dev;
                double *grad_host = grad.data();
                double *x_host = x.data();
                double *delta_x_host = delta_x.data();
                double *new_x_host = new double[N];

                step_size_dev = ALLOCATE_GPU<double>(step_size_dev, sizeof(double));
                grad_dev = ALLOCATE_GPU<double>(grad_dev, N*sizeof(double));
                old_energy_dev = ALLOCATE_GPU<double>(old_energy_dev, sizeof(double));
                // new_x_iter_dev = ALLOCATE_GPU<double>(new_x_iter_dev, N*sizeof(double));
                tmp_dev = ALLOCATE_GPU<double>(tmp_dev, N*sizeof(double));
                x_dev = ALLOCATE_GPU<double>(x_dev, N*sizeof(double));
                delta_x_dev = ALLOCATE_GPU<double>(delta_x_dev, N*sizeof(double));

                cudaMemset(step_size_dev, step_size, sizeof(double));
                COPYDATATOGPU<double>(grad_dev, grad_host, N*sizeof(double));
                cudaMemset(old_energy_dev, old_energy, sizeof(double));
                cudaMemset(new_x_iter_dev, 0, N*sizeof(double));
                cudaMemset(tmp_dev, 0, N*sizeof(double));
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
                    COPYDATATOGPU<double>(tmp_dev, x_host, N*sizeof(double));
                    cublasDaxpy(handle, N, step_size_dev, delta_x_dev, 1, tmp_dev, 1);
                    COPYDATATOHOST<double>(new_x_host, tmp_dev, N*sizeof(double));
                    Eigen::Matrix<double, -1, 1> new_x(new_x_host, N);

                    {
                        POLYFEM_SCOPED_TIMER("constraint set update in LS", this->constraint_set_update_time);
                        objFunc.solution_changed(new_x);
                    }

                    if (use_grad_norm)
                    {
                        objFunc.gradient(new_x, grad);
                        cur_energy = grad.squaredNorm();
                    }
                    else
                        cur_energy = objFunc.value(new_x);

                    is_step_valid = objFunc.is_step_valid(x, new_x);

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

                cudaFree(step_size_dev);
                cudaFree(cur_energy_dev);
                cudaFree(old_energy_dev);
                cudaFree(grad_dev);
                cudaFree(tmp_dev);
                cudaFree(x_dev);
                cudaFree(delta_x_dev);
                cublasDestroy(handle);

                return step_size;
            }
            template class BacktrackingLineSearch<polyfem::solver::NLProblem>;
            template class BacktrackingLineSearch<polyfem::solver::ALNLProblem>;
        }
    }
}