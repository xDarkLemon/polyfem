#pragma once

#include "SparseNewtonDescentSolver.hpp"

#ifdef USE_GPU
#include "polyfem/utils/CuSparseUtils.cuh"
#endif

#include <polyfem/utils/save_problem.hpp>
#include <polyfem/utils/Nullspace.hpp>

namespace cppoptlib
{
	template <typename ProblemType>
	SparseNewtonDescentSolver<ProblemType>::SparseNewtonDescentSolver(
		const json &solver_params, const json &linear_solver_params, const double dt)
		: Superclass(solver_params, dt)
	{
		linear_solver = polysolve::LinearSolver::create(
			linear_solver_params["solver"], linear_solver_params["precond"]);
#ifdef POLYSOLVE_WITH_PETSC
		if (linear_solver_params["solver"] == "PETSC_Solver")
			its_petsc = 1;
#endif
		linear_solver->setParameters(linear_solver_params);
		force_psd_projection = solver_params["force_psd_projection"];
	}

	// =======================================================================

	template <typename ProblemType>
	std::string SparseNewtonDescentSolver<ProblemType>::descent_strategy_name(int descent_strategy) const
	{
		switch (descent_strategy)
		{
		case 0:
			return "Newton";
		case 1:
			if (reg_weight == 0)
				return "projected Newton";
			return fmt::format("projected Newton w/ regularization weight={}", reg_weight);
		case 2:
			return "gradient descent";
		default:
			throw std::invalid_argument("invalid descent strategy");
		}
	}

	// =======================================================================

	template <typename ProblemType>
	void SparseNewtonDescentSolver<ProblemType>::increase_descent_strategy()
	{
		if (this->descent_strategy == 0 || reg_weight > reg_weight_max)
			this->descent_strategy++;
		else
			reg_weight = std::max(reg_weight_inc * reg_weight, reg_weight_min);
		assert(this->descent_strategy <= 2);
	}

	// =======================================================================

	template <typename ProblemType>
	void SparseNewtonDescentSolver<ProblemType>::reset(const int ndof)
	{
		Superclass::reset(ndof);
		assert(linear_solver != nullptr);
		reg_weight = 0;
		internal_solver_info = json::array();
	}

	// =======================================================================

	template <typename ProblemType>
	bool SparseNewtonDescentSolver<ProblemType>::compute_update_direction(
		ProblemType &objFunc,
		const TVector &x,
		const TVector &grad,
		TVector &direction)
	{
		if (this->descent_strategy == 2)
		{
			direction = -grad;
			return true;
		}

		polyfem::StiffnessMatrix hessian;

		assemble_hessian(objFunc, x, hessian);

		if (!solve_linear_system(hessian, grad, direction))
			// solve_linear_system will increase descent_strategy if needed
			return compute_update_direction(objFunc, x, grad, direction);

#ifdef USE_NONLINEAR_GPU
		if (!check_direction_gpu(hessian, grad, direction))
			return compute_update_direction(objFunc, x, grad, direction);
#endif
#ifndef USE_NONLINEAR_GPU
		if (!check_direction(hessian, grad, direction))
			// check_direction will increase descent_strategy if needed
			return compute_update_direction(objFunc, x, grad, direction);
#endif

		json info;
		linear_solver->getInfo(info);
		internal_solver_info.push_back(info);

		reg_weight /= reg_weight_dec;
		if (reg_weight < reg_weight_min)
			reg_weight = 0;

		return true;
	}

	// =======================================================================

	template <typename ProblemType>
	void SparseNewtonDescentSolver<ProblemType>::assemble_hessian(
		ProblemType &objFunc, const TVector &x, polyfem::StiffnessMatrix &hessian)
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

	// =======================================================================

	template <typename ProblemType>
	bool SparseNewtonDescentSolver<ProblemType>::solve_linear_system(
		polyfem::StiffnessMatrix &hessian, const TVector &grad, TVector &direction)
	{
		using namespace benchy::io;
		benchy::io::Problem<double> probleminfo;
		probleminfo.A = hessian;
		probleminfo.b = -grad;
		benchy::io::iter_global = benchy::io::iter_global+1;  // starts from 1
		std::cout << "TIME STEP: " << benchy::io::ts_global << " ITER: " << benchy::io::iter_global << std::endl;
		probleminfo.nullspace = remove_boundary_vertices(test_vertices, test_boundary_nodes);
		benchy::io::save_problem(probleminfo);

		POLYFEM_SCOPED_TIMER("linear solve", this->inverting_time);
		// TODO: get the correct size
		linear_solver->analyzePattern(hessian, hessian.rows());

		// Initializes linear solver for PETSC
		// TODO: To create a manager for choosing the external linear solver by using json
		/*
		/// @param[in] hessian : Sparse Matrix A
					   MAT_AIJ : 0 (sequential), 1(CuSPARSE AIJ)
					   SOLVER_INDEX : see below
		0 = PARDISO
		1 = SUPERLU_DIST
		2 = CHOLMOD
		3 = MUMPS
		4 = CUSPARSE
		5 = STRUMPACK
		6 = HYPRE // NOT FULLY IMPLEMENTED YET
		(ANY)DEFAULT - 99 = PETSC NATIVE SOLVER
		*/

		try
		{
			if (!its_petsc)
				linear_solver->factorize(hessian);
#ifdef POLYSOLVE_WITH_PETSC
			else
				linear_solver->factorize(hessian, 0, 99);
#endif
		}
		catch (const std::runtime_error &err)
		{
			increase_descent_strategy();

			// warn if using gradient descent
			polyfem::logger().log(
				log_level(), "Unable to factorize Hessian: \"{}\"; reverting to {}",
				err.what(), this->descent_strategy_name());

			// polyfem::write_sparse_matrix_csv("problematic_hessian.csv", hessian);
			return false;
		}
		
		linear_solver->solve(-grad, direction); // H Δx = -g

		return true;
	}

	// =======================================================================
	template <typename ProblemType>
	bool SparseNewtonDescentSolver<ProblemType>::check_direction(
		const polyfem::StiffnessMatrix &hessian, const TVector &grad, const TVector &direction)
	{
		POLYFEM_SCOPED_TIMER("checking direction", this->checking_direction_time);
		// gradient descent, check descent direction
		const double residual = (hessian * direction + grad).norm(); // H Δx + g = 0
		if (std::isnan(residual) || residual > std::max(1e-8 * grad.norm(), 1e-5))
		{
			increase_descent_strategy();

			polyfem::logger().log(
				log_level(),
				"[{}] large (or nan) linear solve residual {} (||∇f||={}); reverting to {}",
				name(), residual, grad.norm(), this->descent_strategy_name());

			return false;
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
				log_level(), "[{}] direction is not a descent direction (Δx⋅g={}≥0); reverting to {}",
				name(), direction.dot(grad), descent_strategy_name());
			return false;
		}

		return true;
	}

#ifdef USE_NONLINEAR_GPU
	template <typename ProblemType>
	bool SparseNewtonDescentSolver<ProblemType>::check_direction_gpu(
		const polyfem::StiffnessMatrix &hessian,
		const Eigen::Matrix<double, -1, 1> &grad,
		const Eigen::Matrix<double, -1, 1> &direction)
	{
		POLYFEM_SCOPED_TIMER("check direction", this->checking_direction_time);

		int N = hessian.cols();

		double *hessian_dev, *direction_dev, *grad_dev, *tmp_dev, *residual_dev; // to compute residual
		double *grad_norm_dev, *grad_dir_dot_dev;                                // to compute grad norm and grad dot direction

		double *residual_h = new double[1];
		double *grad_dir_dot_h = new double[1];
		double *grad_norm_h = new double[1];

		// move direction, grad to gpu
		direction_dev = ALLOCATE_GPU<double>(direction_dev, N * sizeof(double));
		grad_dev = ALLOCATE_GPU<double>(grad_dev, N * sizeof(double));
		COPYDATATOGPU<double>(direction_dev, direction.data(), N * sizeof(double));
		COPYDATATOGPU<double>(grad_dev, grad.data(), N * sizeof(double));

		// move hessian to gpu (compressed format)
		const int non0 = hessian.nonZeros();
		polyfem::logger().trace("non0: {}, cols: {}, rows: {}, allocating size: {} bytes", non0, hessian.cols(), hessian.rows(), non0 * sizeof(double));
		int *row_dev, *col_dev;
		// row_dev = ALLOCATE_GPU<int>(row_dev, (N+1)*sizeof(int));
		// col_dev = ALLOCATE_GPU<int>(col_dev, non0*sizeof(int));
		// hessian_dev = ALLOCATE_GPU<double>(hessian_dev, non0*sizeof(double));
		EigenSparseToCuSparseTranspose(hessian, row_dev, col_dev, hessian_dev);

		// compute residual
		// const double residual = (hessian * direction + grad).norm(); // H Δx + g = 0
		tmp_dev = ALLOCATE_GPU<double>(tmp_dev, N * sizeof(double));
		COPYDATATOGPU<double>(tmp_dev, grad.data(), N * sizeof(double));
		residual_dev = ALLOCATE_GPU<double>(residual_dev, sizeof(double));

		cusparseStatus_t status;
		cusparseHandle_t handle = 0;
		cusparseMatDescr_t descr = 0;
		status = cusparseCreate(&handle);
		status = cusparseCreateMatDescr(&descr);
		cusparseSetMatType(descr, CUSPARSE_MATRIX_TYPE_GENERAL);
		cusparseSetMatIndexBase(descr, CUSPARSE_INDEX_BASE_ZERO);
		double *buffer;
		buffer = ALLOCATE_GPU<double>(buffer, 2 * non0 * sizeof(double));
		double alpha = 1.0;
		double beta = 1.0;
		status = cusparseCsrmvEx(handle, CUSPARSE_ALG_MERGE_PATH, CUSPARSE_OPERATION_NON_TRANSPOSE, N, N, non0, &alpha, CUDA_R_64F, descr, hessian_dev, CUDA_R_64F, row_dev, col_dev, direction_dev, CUDA_R_64F, &beta, CUDA_R_64F, tmp_dev, CUDA_R_64F, CUDA_R_64F, buffer);
		cusparseDestroyMatDescr(descr);
		cusparseDestroy(handle);
		cudaFree(row_dev);
		cudaFree(col_dev);
		cudaFree(buffer);

		cublasHandle_t handle2;
		cublasCreate(&handle2);
		cublasDnrm2(handle2, N, tmp_dev, 1, residual_h);

		// compute grad norm, grad direction dot product
		grad_norm_dev = ALLOCATE_GPU<double>(grad_norm_dev, sizeof(double));
		grad_dir_dot_dev = ALLOCATE_GPU<double>(grad_dir_dot_dev, sizeof(double));

		cublasDnrm2(handle2, N, grad_dev, 1, grad_norm_h);
		cublasDdot(handle2, N, grad_dev, 1, direction_dev, 1, grad_dir_dot_h);
		cublasDestroy(handle2);

		cudaFree(hessian_dev);
		cudaFree(direction_dev);
		cudaFree(grad_dev);
		cudaFree(tmp_dev);
		cudaFree(residual_dev);
		cudaFree(grad_norm_dev);
		cudaFree(grad_dir_dot_dev);

		const double residual = *residual_h;
		const double grad_norm = *grad_norm_h;
		const double grad_dir_dot = *grad_dir_dot_h;

		delete[] residual_h;
		delete[] grad_norm_h;
		delete[] grad_dir_dot_h;

		// gradient descent, check descent direction

		if (std::isnan(residual))
		{
			increase_descent_strategy();
			polyfem::logger().log(
				this->descent_strategy == 2 ? spdlog::level::warn : spdlog::level::debug,
				"nan linear solve residual {} (||∇f||={}); reverting to {}",
				residual, grad_norm, this->descent_strategy_name());
			return false;
		}
		else if (residual > std::max(1e-8 * grad_norm, 1e-5))
		{
			increase_descent_strategy();
			polyfem::logger().log(
				this->descent_strategy == 2 ? spdlog::level::warn : spdlog::level::debug,
				"large linear solve residual {} (||∇f||={}); reverting to {}",
				residual, grad_norm, this->descent_strategy_name());
			return false;
		}
		else
		{
			polyfem::logger().trace("linear solve residual {}", residual);
		}

		// do this check here because we need to repeat the solve without resetting reg_weight
		if (grad_dir_dot >= 0)
		{
			increase_descent_strategy();
			polyfem::logger().log(
				this->descent_strategy == 2 ? spdlog::level::warn : spdlog::level::debug,
				"[{}] direction is not a descent direction (Δx⋅g={}≥0); reverting to {}",
				name(), grad_dir_dot, descent_strategy_name());
			return false;
		}

		return true;
	}
#endif
	// =======================================================================

	template <typename ProblemType>
	void SparseNewtonDescentSolver<ProblemType>::update_solver_info()
	{
		Superclass::update_solver_info();
		this->solver_info["internal_solver"] = internal_solver_info;
	}

	// =======================================================================

	template <typename ProblemType>
	static bool has_hessian_nans(const polyfem::StiffnessMatrix &hessian)
	{
		for (int k = 0; k < hessian.outerSize(); ++k)
		{
			for (polyfem::StiffnessMatrix::InnerIterator it(hessian, k); it; ++it)
			{
				if (std::isnan(it.value()))
					return true;
			}
		}

		return false;
	}
} // namespace cppoptlib
