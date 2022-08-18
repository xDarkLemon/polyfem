#pragma once

#include <polyfem/utils/Logger.hpp>
#include <polyfem/utils/Timer.hpp>
#include <polyfem/Common.hpp>
// #include "NonlinearSolver.hpp"
#include <polysolve/LinearSolver.hpp>
#include <polyfem/utils/MatrixUtils.hpp>

#include <polyfem/solver/NLProblem.hpp>
#include <polyfem/solver/ALNLProblem.hpp>
#include <polyfem/solver/SparseNewtonDescentSolver.hpp>

#include <cuda.h>
#include <cuda_runtime_api.h>
#include "polyfem/assembler/CUDA_utilities.cuh"
#include "cublas_v2.h"

namespace cppoptlib
{
	template <typename ProblemType>
	bool SparseNewtonDescentSolver<ProblemType>::compute_update_direction_cuda(
			ProblemType &objFunc,
			const Eigen::Matrix<double, -1, 1> &x,
			const Eigen::Matrix<double, -1, 1> &grad,
			Eigen::Matrix<double, -1, 1> &direction)
	{
        if (this->descent_strategy == 2)
        {
            direction = -grad;
            return true;
        }

        {
            POLYFEM_SCOPED_TIMER("assembly time", this->assembly_time);

            if (this->descent_strategy == 1)
                objFunc.set_project_to_psd(true);
            else if (this->descent_strategy == 0)
                objFunc.set_project_to_psd(false);
            else
                assert(false);

            objFunc.hessian(x, hessian);

            if (reg_weight > 0)
            {
                hessian += reg_weight * polyfem::utils::sparse_identity(hessian.rows(), hessian.cols());
            }
        }

        {
            POLYFEM_SCOPED_TIMER("linear solve", this->inverting_time);
            // TODO: get the correct size
            linear_solver->analyzePattern(hessian, hessian.rows());

            try
            {
                linear_solver->factorize(hessian);
            }
            catch (const std::runtime_error &err)
            {
                increase_descent_strategy();
                // warn if using gradient descent
                polyfem::logger().log(
                    this->descent_strategy == 2 ? spdlog::level::warn : spdlog::level::debug,
                    "Unable to factorize Hessian: \"{}\"; reverting to {}",
                    err.what(), this->descent_strategy_name());
                // polyfem::write_sparse_matrix_csv("problematic_hessian.csv", hessian);
                return compute_update_direction(objFunc, x, grad, direction);
            }

            linear_solver->solve(-grad, direction); // H Δx = -g
        }

        // gradient descent, check descent direction
        const double residual = (hessian * direction + grad).norm(); // H Δx + g = 0
        if (std::isnan(residual))
        {
            increase_descent_strategy();
            polyfem::logger().log(
                this->descent_strategy == 2 ? spdlog::level::warn : spdlog::level::debug,
                "nan linear solve residual {} (||∇f||={}); reverting to {}",
                residual, grad.norm(), this->descent_strategy_name());
            return compute_update_direction(objFunc, x, grad, direction);
        }
        else if (residual > std::max(1e-8 * grad.norm(), 1e-5))
        {
            increase_descent_strategy();
            polyfem::logger().log(
                this->descent_strategy == 2 ? spdlog::level::warn : spdlog::level::debug,
                "large linear solve residual {} (||∇f||={}); reverting to {}",
                residual, grad.norm(), this->descent_strategy_name());
            return compute_update_direction(objFunc, x, grad, direction);
        }
        else
        {
            polyfem::logger().trace("linear solve residual {}", residual);
        }

        // do this check here because we need to repeat the solve without resetting reg_weight
        if (grad.dot(direction) >= 0)
        {
            increase_descent_strategy();
            polyfem::logger().log(
                this->descent_strategy == 2 ? spdlog::level::warn : spdlog::level::debug,
                "[{}] direction is not a descent direction (Δx⋅g={}≥0); reverting to {}",
                name(), direction.dot(grad), descent_strategy_name());
            return compute_update_direction(objFunc, x, grad, direction);
        }

        json info;
        linear_solver->getInfo(info);
        internal_solver_info.push_back(info);

        reg_weight /= reg_weight_dec;
        if (reg_weight < reg_weight_min)
            reg_weight = 0;

        return true;
    }
    template class SparseNewtonDescentSolver<polyfem::solver::NLProblem>;
    template class SparseNewtonDescentSolver<polyfem::solver::ALNLProblem>;
}