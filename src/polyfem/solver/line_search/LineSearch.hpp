#pragma once

namespace polyfem
{
	namespace solver
	{
		namespace line_search
		{
			template <typename ProblemType>
			class LineSearch
			{
			public:
				using Scalar = typename ProblemType::Scalar;
				using TVector = typename ProblemType::TVector;

				LineSearch() {}
				virtual ~LineSearch() = default;

				virtual double line_search(
					const TVector &x,
					const TVector &grad,
					ProblemType &objFunc) = 0;

				static std::shared_ptr<LineSearch<ProblemType>> construct_line_search(const std::string &name);

				static void save_sampled_values(const std::string &filename,
												const TVector &x,
												const TVector &grad,
												ProblemType &objFunc,
												const double starting_step_size = 1e-1,
												const int num_samples = 1000);

				virtual void reset_times()
				{
					iterations = 0;
					checking_for_nan_inf_time = 0;
					broad_phase_ccd_time = 0;
					ccd_time = 0;
					constraint_set_update_time = 0;
					classical_line_search_time = 0;
					compute_grad_norm_time = 0;
					compute_grad_time = 0;
					compute_value_time = 0;
					compute_grad_or_value_time = 0;
					is_step_valid_time = 0;
					compute_new_x_time = 0;
					move_data_time = 0;
					max_step_size_time = 0;
					ls_begin_time = 0;
					ls_end_time = 0;
				}

				int iterations; ///< total number of backtracking iterations done
				double checking_for_nan_inf_time;
				double broad_phase_ccd_time;
				double ccd_time;
				double constraint_set_update_time;
				double classical_line_search_time;
				double compute_grad_norm_time;
				double compute_grad_time;
				double compute_value_time;
				double compute_grad_or_value_time;
				double is_step_valid_time;
				double compute_new_x_time;
				double move_data_time;
				double max_step_size_time;
				double ls_begin_time;
				double ls_end_time;

				double use_grad_norm_tol = -1;
				
				// tmporary pointers for gpu
				double *x_dev, *delta_x_dev;

			protected:
				double min_step_size = 0;
				int max_step_size_iter = 100;
				int cur_iter = 0;

				double compute_nan_free_step_size(
					const TVector &x,
					const TVector &delta_x,
					ProblemType &objFunc,
					const double starting_step_size, const double rate);

				double compute_collision_free_step_size(
					const TVector &x,
					const TVector &delta_x,
					ProblemType &objFunc,
					const double starting_step_size);
				// #ifndef NDEBUG
				// 				double compute_debug_collision_free_step_size(
				// 					const typename ProblemType::TVector &x,
				// 					const typename ProblemType::TVector &delta_x,
				// 					ProblemType &objFunc,
				// 					const double starting_step_size,
				// 					const double rate);
				// #endif
			};
		} // namespace line_search
	}     // namespace solver
} // namespace polyfem

#include "LineSearch.tpp"
