#pragma once

#include "LineSearch.hpp"

#include <polyfem/utils/Timer.hpp>
#include <igl/Timer.h>
#include <cfenv>

namespace polyfem
{
	namespace solver
	{
		namespace line_search
		{
			template <typename ProblemType>
			class BacktrackingLineSearch : public LineSearch<ProblemType>
			{
			public:
				using Superclass = LineSearch<ProblemType>;
				using typename Superclass::Scalar;
				using typename Superclass::TVector;

				BacktrackingLineSearch()
				{
					this->min_step_size = 0;
					this->max_step_size_iter = 100; // std::numeric_limits<int>::max();
				}

				double line_search(
					const TVector &x,
					const TVector &delta_x,
					ProblemType &objFunc) override
				{
					// ----------------
					// Begin linesearch
					// ----------------

					double old_energy, step_size;
					{
						POLYFEM_SCOPED_TIMER("LS begin");

						this->cur_iter = 0;

						{
							POLYFEM_SCOPED_TIMER("compute value in LS", this->compute_value_time);
							old_energy = objFunc.value(x);
						}
						if (std::isnan(old_energy))
						{
							logger().error("Original energy in line search is nan!");
							return std::nan("");
						}

						step_size = 1;
						// TODO: removed feature
						// objFunc.heuristic_max_step(delta_x);
					}

					// ----------------------------
					// Find finite energy step size
					// ----------------------------

					{
						POLYFEM_SCOPED_TIMER("LS compute finite energy step size", this->checking_for_nan_inf_time);
						step_size = this->compute_nan_free_step_size(x, delta_x, objFunc, step_size, 0.5);
						if (std::isnan(step_size))
							return std::nan("");
					}

					const double nan_free_step_size = step_size;

					// -----------------------------
					// Find collision-free step size
					// -----------------------------

					TVector new_x;
					{
						POLYFEM_SCOPED_TIMER("compute new_x in LS", this->compute_new_x_time);
						new_x = x + step_size * delta_x;
					}

					{
						POLYFEM_SCOPED_TIMER("line search begin in LS", this->ls_begin_time);
						objFunc.line_search_begin(x, new_x);
					}

					// {
					// 	POLYFEM_SCOPED_TIMER("CCD broad-phase", this->broad_phase_ccd_time);
					// 	TVector new_x = x + step_size * delta_x;
					// 	objFunc.line_search_begin(x, new_x);
					// }

					{
						POLYFEM_SCOPED_TIMER("CCD narrow-phase", this->ccd_time);
						logger().trace("Performing narrow-phase CCD");
						step_size = this->compute_collision_free_step_size(x, delta_x, objFunc, step_size);
						if (std::isnan(step_size))
							return std::nan("");
					}

					const double collision_free_step_size = step_size;

					// ----------------------
					// Find descent step size
					// ----------------------

					{
						POLYFEM_SCOPED_TIMER("energy min in LS", this->classical_line_search_time);
#ifdef USE_NONLINEAR_GPU
						step_size = compute_descent_step_size_gpu(x, delta_x, objFunc, old_energy, step_size);
#endif
#ifndef USE_NONLINEAR_GPU
						step_size = compute_descent_step_size(x, delta_x, objFunc, old_energy, step_size);
#endif
						if (std::isnan(step_size))
						{
							// Superclass::save_sampled_values("failed-line-search-values.csv", x, delta_x, objFunc);
							return std::nan("");
						}
					}

					const double descent_step_size = step_size;

					// #ifndef NDEBUG
					// 					// -------------
					// 					// CCD safeguard
					// 					// -------------

					// 					{
					// 						POLYFEM_SCOPED_TIMER("safeguard in LS");
					// 						step_size = this->compute_debug_collision_free_step_size(x, delta_x, objFunc, step_size, 0.5);
					// 					}

					// 					const double debug_collision_free_step_size = step_size;
					// #endif

					{
						POLYFEM_SCOPED_TIMER("LS end");
						objFunc.line_search_end();
					}

					logger().debug(
						"Line search finished (nan_free_step_size={} collision_free_step_size={} descent_step_size={} final_step_size={})",
						nan_free_step_size, collision_free_step_size, descent_step_size, step_size);

					return step_size;
				}

			protected:
				double compute_descent_step_size_gpu(
					const Eigen::Matrix<double, -1, 1> &x,
					const Eigen::Matrix<double, -1, 1> &delta_x,
					ProblemType &objFunc,
					const double old_energy_in,
					const double starting_step_size = 1);

				double compute_descent_step_size(
					const TVector &x,
					const TVector &delta_x,
					ProblemType &objFunc,
					const double old_energy_in,
					const double starting_step_size = 1)
				{

					igl::Timer timerg;
					double step_size = starting_step_size;

					TVector grad(x.rows());
					{
						POLYFEM_SCOPED_TIMER("compute grad in LS", this->compute_grad_time);
						objFunc.gradient(x, grad);
					}

					bool use_grad_norm;
					double old_energy;
					{
						POLYFEM_SCOPED_TIMER("compute grad norm in LS", this->compute_grad_norm_time);
						use_grad_norm = grad.norm() < this->use_grad_norm_tol;
						old_energy = use_grad_norm ? grad.squaredNorm() : old_energy_in;
					}
					// Find step that reduces the energy
					double cur_energy = std::nan("");
					bool is_step_valid = false;
					while (step_size > this->min_step_size && this->cur_iter < this->max_step_size_iter)
					{
						this->iterations++;

						TVector new_x;
						{
							POLYFEM_SCOPED_TIMER("compute new_x in LS", this->compute_new_x_time);
							new_x = x + step_size * delta_x;
						}
						{
							POLYFEM_SCOPED_TIMER("constraint set update in LS", this->constraint_set_update_time);
							objFunc.solution_changed(new_x);
						}

						timerg.start();
						{
							POLYFEM_SCOPED_TIMER("compute grad/value in LS", this->compute_grad_or_value_time);
							if (use_grad_norm)
							{
								{
									POLYFEM_SCOPED_TIMER("compute grad in LS", this->compute_grad_time);
									objFunc.gradient(new_x, grad);
								}
								{
									POLYFEM_SCOPED_TIMER("compute grad norm in LS", this->compute_grad_norm_time);
									cur_energy = grad.squaredNorm();
								}
							}
							else
							{
								POLYFEM_SCOPED_TIMER("compute value in LS", this->compute_value_time);
								cur_energy = objFunc.value(new_x);
							}
						}
						timerg.stop();
						logger().trace("done obj.value/grad for LS {}s...", timerg.getElapsedTime());

						timerg.start();
						{
							POLYFEM_SCOPED_TIMER("is step valid in LS", this->is_step_valid_time);
							is_step_valid = objFunc.is_step_valid(x, new_x);
						}
						timerg.stop();
						logger().trace("done is_step_valid for LS {}s...", timerg.getElapsedTime());

						logger().trace("ls it: {} delta: {} invalid: {} ", this->cur_iter, (cur_energy - old_energy), !is_step_valid);
						//  if (!std::isfinite(cur_energy) || (cur_energy >= old_energy && fabs(cur_energy - old_energy) > 1e-12) || !is_step_valid)
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

					return step_size;
				}
			};
		} // namespace line_search
	}     // namespace solver
} // namespace polyfem
