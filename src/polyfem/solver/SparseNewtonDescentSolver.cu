#pragma once

#include <polyfem/Common.hpp>
#include <polyfem/utils/Logger.hpp>
#include <polyfem/utils/Timer.hpp>
#include <polyfem/Common.hpp>

#include <polyfem/solver/NLProblem.hpp>
#include <polyfem/solver/FullNLProblem.hpp>
#include <polyfem/solver/SparseNewtonDescentSolver.hpp>
#include "polyfem/utils/CUDA_utilities.cuh"

#include <cuda.h>
#include <cuda_runtime_api.h>
#include <cuda_runtime.h>
#include "cublas_v2.h"
#include "cusparse.h"
#include "library_types.h"

namespace cppoptlib
{
    void EigenSparseToCuSparseTranspose(const Eigen::SparseMatrix<double> &mat, int *row, int *col, double *val)
    {
        const int num_non0  = mat.nonZeros();
        const int num_outer = mat.cols() + 1;
        cudaMemcpy(row, mat.outerIndexPtr(), sizeof(int) * num_outer, cudaMemcpyHostToDevice);
        cudaMemcpy(col, mat.innerIndexPtr(), sizeof(int) * num_non0, cudaMemcpyHostToDevice);
        cudaMemcpy(val, mat.valuePtr(), sizeof(double) * num_non0, cudaMemcpyHostToDevice);
    }

	template <typename ProblemType>
	bool SparseNewtonDescentSolver<ProblemType>::check_direction_gpu(
		const polyfem::StiffnessMatrix &hessian,
		const Eigen::Matrix<double, -1, 1> &grad,
		const Eigen::Matrix<double, -1, 1> &direction)
	{
        int N = hessian.cols();
        
        double *hessian_dev, *direction_dev, *grad_dev, *tmp_dev, *residual_dev;  // to compute residual
        double *grad_norm_dev, *grad_dir_dot_dev;  // to compute grad norm and grad dot direction
        
        double *residual_h = new double[1];
        double *grad_dir_dot_h = new double[1];
        double *grad_norm_h = new double[1];
        
        // move direction, grad to gpu
        direction_dev = ALLOCATE_GPU<double>(direction_dev, N*sizeof(double));
        grad_dev = ALLOCATE_GPU<double>(grad_dev, N*sizeof(double));
        COPYDATATOGPU<double>(direction_dev, direction.data(), N*sizeof(double));
        COPYDATATOGPU<double>(grad_dev, grad.data(), N*sizeof(double));

        // move hessian to gpu (compressed format)
        const int non0 = hessian.nonZeros();
        polyfem::logger().trace("non0: {}, cols: {}, rows: {}, allocating size: {} bytes", non0,  hessian.cols(), hessian.rows(), non0*sizeof(double));
        int *row_dev, *col_dev;
        row_dev = ALLOCATE_GPU<int>(row_dev, (N+1)*sizeof(int));
        col_dev = ALLOCATE_GPU<int>(col_dev, non0*sizeof(int));
        hessian_dev = ALLOCATE_GPU<double>(hessian_dev, non0*sizeof(double));
        EigenSparseToCuSparseTranspose(hessian, row_dev, col_dev, hessian_dev);
        
        // compute residual
        // const double residual = (hessian * direction + grad).norm(); // H Δx + g = 0
        tmp_dev = ALLOCATE_GPU<double>(tmp_dev, N*sizeof(double));
        COPYDATATOGPU<double>(tmp_dev, grad.data(), N*sizeof(double));
        residual_dev = ALLOCATE_GPU<double>(residual_dev, sizeof(double));

        cusparseStatus_t status;
        cusparseHandle_t handle=0;
        cusparseMatDescr_t descr=0;
        status= cusparseCreate(&handle);
        status= cusparseCreateMatDescr(&descr);
        cusparseSetMatType(descr , CUSPARSE_MATRIX_TYPE_GENERAL);
        cusparseSetMatIndexBase(descr, CUSPARSE_INDEX_BASE_ZERO); 
        double *buffer;
        buffer = ALLOCATE_GPU<double>(buffer, 2*non0*sizeof(double));
        double alpha = 1.0;
        double beta = 1.0;
        status=cusparseCsrmvEx(handle, CUSPARSE_ALG_MERGE_PATH, CUSPARSE_OPERATION_NON_TRANSPOSE, N, N, non0, &alpha , CUDA_R_64F, descr , hessian_dev, CUDA_R_64F, row_dev , col_dev , direction_dev , CUDA_R_64F, &beta , CUDA_R_64F, tmp_dev, CUDA_R_64F, CUDA_R_64F, buffer);
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
    template class SparseNewtonDescentSolver<polyfem::solver::NLProblem>;
    template class SparseNewtonDescentSolver<polyfem::solver::FullNLProblem>;
}
