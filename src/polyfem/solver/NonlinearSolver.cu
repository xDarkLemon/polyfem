#pragma once

#include <polyfem/Common.hpp>

#include <polyfem/utils/Logger.hpp>
#include <polyfem/utils/Timer.hpp>

// Line search methods
// #include "line_search/LineSearch.hpp"

#include <polyfem/solver/NLProblem.hpp>
#include <polyfem/solver/NonlinearSolver.hpp>

#include <cppoptlib/solver/isolver.h>

#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/copy.h>
#include <thrust/inner_product.h>

#include <iostream>

namespace cppoptlib
{
    // typedef Eigen::Matrix<double, -1, 1> TVector;

    template <typename ProblemType /*, int Ord*/>
    void NonlinearSolver<ProblemType>::minimize_gpu(ProblemType &objFunc, Eigen::Matrix<double, -1, 1> &x)
    {
			using namespace polyfem;

			// ---------------------------
			// Initialize the minimization
			// ---------------------------

			reset(objFunc, x); // place for children to initialize their fields

			TVector grad = TVector::Zero(x.rows());
			TVector delta_x = TVector::Zero(x.rows());

			// double factor = 1e-5;

			// Set these to nan to indicate they have not been computed yet
			double old_energy = std::nan("");

			{
				POLYFEM_SCOPED_TIMER("constraint set update", constraint_set_update_time);
				objFunc.solution_changed(x);
			}

			{
				POLYFEM_SCOPED_TIMER("compute gradient", grad_time);
				objFunc.gradient(x, grad);
			}
			double first_grad_norm = grad.norm();
			if (std::isnan(first_grad_norm))
			{
				this->m_status = Status::UserDefined;
				polyfem::logger().error("[{}] Initial gradient is nan; stopping", name());
				m_error_code = ErrorCode::NanEncountered;
				throw std::runtime_error("Gradient is nan; stopping");
				return;
			}
			this->m_current.gradNorm = first_grad_norm / (normalize_gradient ? first_grad_norm : 1);
			this->m_current.fDelta = old_energy;

			this->m_status = checkConvergence(this->m_stop, this->m_current);
			if (this->m_status != Status::Continue)
			{
				POLYFEM_SCOPED_TIMER("compute objective function", obj_fun_time);
				this->m_current.fDelta = objFunc.value(x);
				polyfem::logger().log(
					spdlog::level::info, "[{}] {} (f={} ||∇f||={} g={} tol={})",
					name(), "Not even starting, grad is small enough", this->m_current.fDelta, first_grad_norm,
					this->m_current.gradNorm, this->m_stop.gradNorm);
				update_solver_info();
				return;
			}

			utils::Timer timer("non-linear solver", this->total_time);
			timer.start();

			m_line_search->use_grad_norm_tol = use_grad_norm_tol;

			do
			{
				{
					POLYFEM_SCOPED_TIMER("constraint set update", constraint_set_update_time);
					objFunc.solution_changed(x);
				}

				double energy;
				{
					POLYFEM_SCOPED_TIMER("compute objective function", obj_fun_time);
					energy = objFunc.value(x);
				}
				if (!std::isfinite(energy))
				{
					this->m_status = Status::UserDefined;
					polyfem::logger().error("[{}] f(x) is nan or inf; stopping", name());
					m_error_code = ErrorCode::NanEncountered;
					throw std::runtime_error("f(x) is nan or inf; stopping");
					break;
				}

				{
					POLYFEM_SCOPED_TIMER("compute gradient", grad_time);
					objFunc.gradient(x, grad);
				}

				const double grad_norm = grad.norm();
				if (std::isnan(grad_norm))
				{
					this->m_status = Status::UserDefined;
					polyfem::logger().error("[{}] Gradient is nan; stopping", name());
					m_error_code = ErrorCode::NanEncountered;
					throw std::runtime_error("Gradient is nan; stopping");
					break;
				}

				// ------------------------
				// Compute update direction
				// ------------------------

				// Compute a Δx to update the variable
				if (!compute_update_direction(objFunc, x, grad, delta_x))
				{
					this->m_status = Status::Continue;
					continue;
				}

				if (grad_norm != 0 && delta_x.dot(grad) >= 0)
				{
					increase_descent_strategy();
					polyfem::logger().log(
						spdlog::level::debug,
						"[{}] direction is not a descent direction (Δx⋅g={}≥0); reverting to {}",
						name(), delta_x.dot(grad), descent_strategy_name());
					this->m_status = Status::Continue;
					continue;
				}

				const double delta_x_norm = delta_x.norm();
				if (std::isnan(delta_x_norm))
				{
					increase_descent_strategy();
					this->m_status = Status::UserDefined;
					polyfem::logger().debug("[{}] Δx is nan; reverting to {}", name(), descent_strategy_name());
					this->m_status = Status::Continue;
					continue;
				}

				if (!use_gradient_norm)
				{
					//TODO, we shold remove this
					// Use the maximum absolute displacement value divided by the timestep,
					// so the units are in velocity units.
					// TODO: Set this to the actual timestep
					double dt = 1;
					// TODO: Also divide by the world scale to make this criteria scale invariant.
					this->m_current.gradNorm = delta_x.template lpNorm<Eigen::Infinity>() / dt;
				}
				else
				{
					//if normalize_gradient, use relative to first norm
					this->m_current.gradNorm = grad_norm / (normalize_gradient ? first_grad_norm : 1);
				}
				this->m_current.fDelta = std::abs(old_energy - energy); // / std::abs(old_energy);

				this->m_status = checkConvergence(this->m_stop, this->m_current);

				old_energy = energy;

				// ---------------
				// Variable update
				// ---------------

				// Perform a line_search to compute step scale
				double rate = line_search(x, delta_x, objFunc);
				if (std::isnan(rate))
				{
					// descent_strategy set by line_search upon failure
					if (this->m_status == Status::Continue)
						continue;
					else
						break;
				}

				x += rate * delta_x;

				// -----------
				// Post update
				// -----------

				descent_strategy = default_descent_strategy(); // Reset this for the next iterations

				const double step = (rate * delta_x).norm();

				if (objFunc.stop(x))
				{
					this->m_status = Status::UserDefined;
					m_error_code = ErrorCode::Success;
					polyfem::logger().debug("[{}] Objective decided to stop", name());
				}

				objFunc.post_step(this->m_current.iterations, x);

				polyfem::logger().debug(
					"[{}] iter={:} f={} ‖∇f‖={} ‖Δx‖={} Δx⋅∇f(x)={} g={} tol={} rate={} ‖step‖={}",
					name(), this->m_current.iterations, energy, grad_norm, delta_x_norm, delta_x.dot(grad),
					this->m_current.gradNorm, this->m_stop.gradNorm, rate, step);
				++this->m_current.iterations;
			} while (objFunc.callback(this->m_current, x) && (this->m_status == Status::Continue));

			timer.stop();

			// -----------
			// Log results
			// -----------

			std::string msg = "Finished";
			spdlog::level::level_enum level = spdlog::level::info;
			if (this->m_status == Status::IterationLimit)
			{
				const std::string msg = fmt::format("[{}] Reached iteration limit", name());
				polyfem::logger().error(msg);
				throw std::runtime_error(msg);
				level = spdlog::level::err;
			}
			else if (this->m_current.iterations == 0)
			{
				const std::string msg = fmt::format("[{}] Unable to take a step", name());
				polyfem::logger().error(msg);
				throw std::runtime_error(msg);
				level = this->m_status == Status::UserDefined ? spdlog::level::err : spdlog::level::warn;
			}
			else if (this->m_status == Status::UserDefined)
			{
				const std::string msg = fmt::format("[{}] Failed to find minimizer", name());
				polyfem::logger().error(msg);
				throw std::runtime_error(msg);
				level = spdlog::level::err;
			}
			polyfem::logger().log(
				level, "[{}] {}, took {}s (niters={} f={} ||∇f||={} ||Δx||={} Δx⋅∇f(x)={} g={} tol={})",
				name(), msg, timer.getElapsedTimeInSec(), this->m_current.iterations, old_energy, grad.norm(), delta_x.norm(),
				delta_x.dot(grad), this->m_current.gradNorm, this->m_stop.gradNorm);

			log_times();
			update_solver_info();
    }
    template class NonlinearSolver<polyfem::solver::NLProblem>;
}