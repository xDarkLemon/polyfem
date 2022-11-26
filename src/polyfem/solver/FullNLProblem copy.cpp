#include "FullNLProblem.hpp"
#include <polyfem/utils/Logger.hpp>
//#include <polyfem/MaybeParallelFor.hpp>

#include <igl/Timer.h>

#include "polyfem/utils/CUDA_utilities.cuh"
#include "polyfem/utils/CuSparseUtils.cuh"

namespace polyfem::solver
{
	FullNLProblem::FullNLProblem(std::vector<std::shared_ptr<Form>> &forms)
		: forms_(forms)
	{
	}

	void FullNLProblem::init(const TVector &x)
	{
		for (auto &f : forms_)
			f->init(x);
	}

	void FullNLProblem::set_project_to_psd(bool project_to_psd)
	{
		for (auto &f : forms_)
			f->set_project_to_psd(project_to_psd);
	}

	void FullNLProblem::init_lagging(const TVector &x)
	{
		for (auto &f : forms_)
			f->init_lagging(x);
	}

	void FullNLProblem::update_lagging(const TVector &x, const int iter_num)
	{
		for (auto &f : forms_)
			f->update_lagging(x, iter_num);
	}

	int FullNLProblem::max_lagging_iterations() const
	{
		int max_lagging_iterations = std::numeric_limits<int>::max();
		for (auto &f : forms_)
			max_lagging_iterations = std::min(max_lagging_iterations, f->max_lagging_iterations());
		return max_lagging_iterations;
	}

	bool FullNLProblem::uses_lagging() const
	{
		for (auto &f : forms_)
			if (f->uses_lagging())
				return true;
		return false;
	}

	void FullNLProblem::line_search_begin(const TVector &x0, const TVector &x1)
	{
		for (auto &f : forms_)
			f->line_search_begin(x0, x1);
	}

	void FullNLProblem::line_search_end()
	{
		for (auto &f : forms_)
			f->line_search_end();
	}

	double FullNLProblem::max_step_size(const TVector &x0, const TVector &x1) const
	{
		double step = 1;
		for (auto &f : forms_)
			if (f->enabled())
				step = std::min(step, f->max_step_size(x0, x1));
		return step;
	}

	bool FullNLProblem::is_step_valid(const TVector &x0, const TVector &x1) const
	{
		for (auto &f : forms_)
			if (f->enabled() && !f->is_step_valid(x0, x1))
				return false;
		return true;
	}

	bool FullNLProblem::is_step_collision_free(const TVector &x0, const TVector &x1) const
	{
		for (auto &f : forms_)
			if (f->enabled() && !f->is_step_collision_free(x0, x1))
				return false;
		return true;
	}

	double FullNLProblem::value(const TVector &x)
	{
		double val = 0;
		for (auto &f : forms_)
			if (f->enabled())
				val += f->value(x);
		return val;
	}

	void FullNLProblem::gradient(const TVector &x, TVector &grad)
	{
		grad = TVector::Zero(x.size());
		for (auto &f : forms_)
		{
			if (!f->enabled())
				continue;
			TVector tmp;
			f->first_derivative(x, tmp);
			grad += tmp;
		}
	}

	void FullNLProblem::hessian(const TVector &x, THessian &hessian)
	{

		igl::Timer timerg;
		hessian.resize(x.size(), x.size());

#ifndef USE_GPU
		for (auto &f : forms_)
		{
			if (!f->enabled())
				continue;
			THessian tmp;

			f->second_derivative(x, tmp);
			timerg.start();
			hessian += tmp;
			timerg.stop();
			logger().trace("done partial sum hessian -- {}s...", timerg.getElapsedTime());
		}
#endif
// #ifdef USE_GPU
		/* another approach */
		// std::vector<Eigen::SparseMatrix<double>> tmp_all;
		// for (auto &f : forms_)
		// {
		// 	if (!f->enabled())
		// 		continue;
		// 	THessian tmp;

		// 	f->second_derivative(x, tmp);

		// 	tmp_all.push_back(tmp);
		// }
		// PartialHessianSum(tmp_all, hessian);
		/* another approach */

		int cnt = 0;
		for (auto &f : forms_)
		{
			if (!f->enabled())
				continue;
			THessian tmp;

			f->second_derivative(x, tmp);

			// printf("AFTER computing tmp at round %d\n", cnt);
			// printf("\ntmp eigen matrix:");
			// printCSRMatrix(tmp.valuePtr(), tmp.outerIndexPtr(), tmp.innerIndexPtr(), tmp.cols(), tmp.nonZeros());
			// printf("\nhessian eigen matrix:");
			// printCSRMatrix(hessian.valuePtr(), hessian.outerIndexPtr(), hessian.innerIndexPtr(), hessian.cols(), hessian.nonZeros());

			if (tmp.nonZeros()==0)
				continue;

			if (!hessian.nonZeros()>0)
			{
				hessian = tmp;

				// printf("AFTER copying tmp to hessian\n");
				// printf("\ntmp eigen matrix:");
				// printCSRMatrix(tmp.valuePtr(), tmp.outerIndexPtr(), tmp.innerIndexPtr(), tmp.cols(), tmp.nonZeros());
				// printf("\nhessian eigen matrix:");
				// printCSRMatrix(hessian.valuePtr(), hessian.outerIndexPtr(), hessian.innerIndexPtr(), hessian.cols(), hessian.nonZeros());
			}
			else
			{
				// printf("BEFORE CuSparseHessianSum\n");
				// printf("\ntmp eigen matrix:");
				// printCSRMatrix(tmp.valuePtr(), tmp.outerIndexPtr(), tmp.innerIndexPtr(), tmp.cols(), tmp.nonZeros());
				// printf("\nhessian eigen matrix:");
				// printCSRMatrix(hessian.valuePtr(), hessian.outerIndexPtr(), hessian.innerIndexPtr(), hessian.cols(), hessian.nonZeros());
				// printf("IN CuSparseHessianSum\n");

				timerg.start();
				CuSparseHessianSum(tmp, hessian);
				timerg.stop();
				logger().trace("done partial sum hessian -- {}s...", timerg.getElapsedTime());
			}
			cnt++;
		}
// #endif
	}

	void FullNLProblem::solution_changed(const TVector &x)
	{
		for (auto &f : forms_)
			f->solution_changed(x);
	}

	void FullNLProblem::post_step(const int iter_num, const TVector &x)
	{
		for (auto &f : forms_)
			f->post_step(iter_num, x);
	}
} // namespace polyfem::solver
