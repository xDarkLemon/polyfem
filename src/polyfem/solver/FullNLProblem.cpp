#include "FullNLProblem.hpp"
#include <polyfem/utils/Logger.hpp>
//#include <polyfem/MaybeParallelFor.hpp>

#include <igl/Timer.h>

#include "polyfem/utils/CuSparseUtils.cuh"

#define DEBUG false

namespace polyfem::solver
{
	FullNLProblem::FullNLProblem(const std::vector<std::shared_ptr<Form>> &forms)
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
		int max_lagging_iterations = 1;
		for (auto &f : forms_)
			max_lagging_iterations = std::max(max_lagging_iterations, f->max_lagging_iterations());
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
		igl::Timer timerg2;

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
			printf("[2022-11-30 04:53:24.274] [polyfem] [trace] [timing] CPU HESSIAN SUM TOT %lfs\n", timerg.getElapsedTime());
		}
#endif

		int nnz_max = 0;
		int cnt = 0;
		THessian all_tmp[7];
		for (auto &f : forms_)
		{
			if (!f->enabled())
				continue;
			THessian tmp;

			f->second_derivative(x, tmp);

			all_tmp[cnt] = tmp;
			nnz_max += tmp.nonZeros();
			cnt += 1;
		}

		timerg2.start();
		double *hessian_val;
		int *hessian_row, *hessian_col, hessian_nnz = 0;
		timerg.start();
		hessian_row = ALLOCATE_GPU<int>(hessian_row, (hessian.cols()+1)*sizeof(int));
		hessian_col = ALLOCATE_GPU<int>(hessian_col, nnz_max*sizeof(int));
		hessian_val = ALLOCATE_GPU<double>(hessian_val, nnz_max*sizeof(double));
		timerg.stop();
		// logger().trace("CUDA MALLOC HESSIAN {}s", timerg.getElapsedTime());
		printf("[2022-11-30 04:53:24.274] [polyfem] [trace] [timing] CUDA MALLOC HESSIAN %lfs\n", timerg.getElapsedTime());
		if (hessian.nonZeros()>0)
		{
			timerg.start();
			cudaMemcpy(hessian_row, hessian.outerIndexPtr(), sizeof(int) * (hessian.cols()+1), cudaMemcpyHostToDevice);
			cudaMemcpy(hessian_col, hessian.innerIndexPtr(), sizeof(int) * hessian.nonZeros(), cudaMemcpyHostToDevice);
			cudaMemcpy(hessian_val, hessian.valuePtr(), sizeof(double) * hessian.nonZeros(), cudaMemcpyHostToDevice);
			timerg.stop();
			printf("[2022-11-30 04:53:24.274] [polyfem] [trace] [timing] DATA MOVING HTOD (HESSIAN) %lfs\n", timerg.getElapsedTime());
		}

		for (int i=0; i<cnt; i++)
		{
			THessian tmp = all_tmp[i];

			if (tmp.nonZeros()==0)
				continue;

			if (hessian_nnz==0)
			{
				// hessian = tmp;
				timerg.start();
				cudaMemcpy(hessian_row, tmp.outerIndexPtr(), sizeof(int) * (tmp.cols()+1), cudaMemcpyHostToDevice);
				cudaMemcpy(hessian_col, tmp.innerIndexPtr(), sizeof(int) * tmp.nonZeros(), cudaMemcpyHostToDevice);
				cudaMemcpy(hessian_val, tmp.valuePtr(), sizeof(double) * tmp.nonZeros(), cudaMemcpyHostToDevice);
				timerg.stop();
				// logger().trace("DATA MOVING HTOD {}s", timerg.getElapsedTime());
				printf("[2022-11-30 04:53:24.274] [polyfem] [trace] [timing] DATA MOVING HTOD %lfs\n", timerg.getElapsedTime());
				printf("[2022-11-30 04:53:24.274] [polyfem] [trace] [timing] TMP NNZ %ds\n", tmp.nonZeros());		

				hessian_nnz = tmp.nonZeros();

				if(DEBUG)
				{
					printf("\nAFTER copying: hessian dev matrix:");
					printCSRMatrixGPU(hessian_val, hessian_row, hessian_col, hessian.cols(), hessian_nnz);
				}
			}
			else
			{
				if(DEBUG)
				{
					printf("\nBEFORE: hessian dev matrix: hessian_nnz=%d", hessian_nnz);
					printCSRMatrixGPU(hessian_val, hessian_row, hessian_col, hessian.cols(), hessian_nnz);
				}
				
				timerg.start();
				CuSparseHessianSum(tmp, hessian_nnz, &hessian_val, &hessian_row, &hessian_col);
				timerg.stop();
				printf("[2022-11-30 04:53:24.274] [polyfem] [trace] [timing] CUSP HESSIAN SUM %lfs\n", timerg.getElapsedTime());
				
				if(DEBUG)
				{
					printf("\nAFTER: hessian dev matrix: hessian_nnz=%d", hessian_nnz);
					printCSRMatrixGPU(hessian_val, hessian_row, hessian_col, hessian.cols(), hessian_nnz);
				}
			}
		}
		timerg.start();
		hessian.reserve(hessian_nnz);     
		cudaMemcpy(hessian.outerIndexPtr(), hessian_row, sizeof(int) * (hessian.cols() + 1), cudaMemcpyDeviceToHost);
		cudaMemcpy(hessian.innerIndexPtr(), hessian_col, sizeof(int) * hessian_nnz, cudaMemcpyDeviceToHost);
		cudaMemcpy(hessian.valuePtr(), hessian_val, sizeof(double) * hessian_nnz, cudaMemcpyDeviceToHost);
		// CuSparseTransposeToEigenSparse(hessian_row, hessian_col, hessian_val, hessian_nnz, hessian.rows(), hessian.cols(), hessian);
		timerg.stop();
		printf("[2022-11-30 04:53:24.274] [polyfem] [trace] [timing] DATA MOVING DTOH %lfs\n", timerg.getElapsedTime());		
		printf("[2022-11-30 04:53:24.274] [polyfem] [trace] [timing] HESSIAN NNZ %ds\n", hessian_nnz);		
		timerg.start();
		cudaFree(hessian_val);
		cudaFree(hessian_row);
		cudaFree(hessian_col);
		timerg.stop();
		timerg2.stop();
		// logger().trace("FREE HESSIAN(LAST) {}s", timerg.getElapsedTime());
		// logger().trace("HESSIAN SUM TOT {}s", timerg.getElapsedTime());
		printf("[2022-11-30 04:53:24.274] [polyfem] [trace] [timing] FREE HESSIAN %lfs\n", timerg.getElapsedTime());
		printf("[2022-11-30 04:53:24.274] [polyfem] [trace] [timing] HESSIAN SUM TOT %lfs\n", timerg2.getElapsedTime());
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
