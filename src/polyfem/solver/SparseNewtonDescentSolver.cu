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
#include "polyfem/assembler/CUDA_utilities.cuh"

#include <cuda.h>
#include <cuda_runtime_api.h>
#include <cuda_runtime.h>
#include "cublas_v2.h"
#include "cusparse.h"
#include "library_types.h"

namespace cppoptlib
{
    void check_cuda_mem()
    {
        static int flag_gpu_settings = 0;
        if (!flag_gpu_settings)
        {
            size_t free_bytes = 0, total_bytes = 0;
            cudaMemGetInfo(&free_bytes, &total_bytes);
            std::cout << "Mem GPU Free : " << free_bytes << " bytes" << std::endl;
            std::cout << "Mem GPU Total: " << total_bytes << " bytes" << std::endl;
            size_t sizeLimit = 0;
            cudaDeviceGetLimit(&sizeLimit, cudaLimitMallocHeapSize);
            std::cout << "Original device heap sizeLimit: " << sizeLimit << std::endl;

            //cudaDeviceSetLimit( cudaLimitMallocHeapSize, free_bytes );

            //cudaDeviceGetLimit( &sizeLimit, cudaLimitMallocHeapSize );
            //std::cout << "Current device heap sizeLimit: " << sizeLimit << std::endl;
            /*
            std::cout << "SharedMemoryRequired: "
            << ":" << size_inner_index*sizeof(double)
            << std::endl;
            */
            flag_gpu_settings++;
        }
    }

    void EigenSparseToCuSparseTranspose(const Eigen::SparseMatrix<double> &mat, int *row, int *col, double *val)
    {
        const int num_non0  = mat.nonZeros();
        const int num_outer = mat.cols() + 1;
        cudaMemcpy(row, mat.outerIndexPtr(), sizeof(int) * num_outer, cudaMemcpyHostToDevice);
        cudaMemcpy(col, mat.innerIndexPtr(), sizeof(int) * num_non0, cudaMemcpyHostToDevice);
        cudaMemcpy(val, mat.valuePtr(), sizeof(double) * num_non0, cudaMemcpyHostToDevice);
    }

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
                return compute_update_direction_cuda(objFunc, x, grad, direction);
            }

            linear_solver->solve(-grad, direction); // H Δx = -g
            // will change direction = -inv(H) * g
        }

        // gradient descent, check descent direction
        // const double residual = (hessian * direction + grad).norm(); // H Δx + g = 0
        // const double residual_cpu = (hessian * direction + grad).norm(); // H Δx + g = 0

        int N = hessian.cols();
        
        double *hessian_dev, *direction_dev, *grad_dev, *tmp_dev, *res_dev;  // to compute residual
        double *grad_grad_dev, *grad_direc_dev;  // to compute grad norm and grad dot direction
        
        // double *hessian_host = hessian.valuePtr();
        double *direction_host = direction.data();
        const double *grad_host = grad.data();
        double *tmp_host = new double[N];
        double *res_host = new double[1];
        double *grad_grad_host = new double[1];
        double *grad_direc_host = new double[1];
        
        double alpha = 1.0;
        double beta = 1.0;  // 0.0 for cublas
        
        printf("before allocating hessian_dev:\n");
        check_cuda_mem();

        const int non0 = hessian.nonZeros();
        polyfem::logger().trace("non0: {}, cols: {}, rows: {}, allocating size: {} bytes", non0,  hessian.cols(), hessian.rows(), non0*sizeof(double));
        // std::cout << "non0: " << non0 << " , cols: "<< hessian.cols() <<  ", rows: " << hessian.rows() << ", allocating size : " << non0*sizeof(double) << " bytes" << std::endl;
        int *row_dev;
        int *col_dev;
        row_dev = ALLOCATE_GPU<int>(row_dev, (N+1)*sizeof(int));
        col_dev = ALLOCATE_GPU<int>(col_dev, non0*sizeof(int));
        hessian_dev = ALLOCATE_GPU<double>(hessian_dev, non0*sizeof(double));
        EigenSparseToCuSparseTranspose(hessian, row_dev, col_dev, hessian_dev);
        
        int *row_host = new int[N+1]; 
        int *col_host = new int[non0]; 
        double *hessian_host = new double[non0];
        COPYDATATOHOST<int>(row_host, row_dev, (N+1)*sizeof(int));
        COPYDATATOHOST<int>(col_host, col_dev, (non0)*sizeof(int));
        COPYDATATOHOST<double>(hessian_host, hessian_dev, (non0)*sizeof(double));
        int base = row_host[N]-non0;

        // printf("hessian:\n");
        // std::cout << MatrixXd(hessian) << std::endl;
        // polyfem::logger().trace("row_host[N](none zero elements): {}", row_host[N]);
        // printf("\nrow_host[N]: %d\n", row_host[N]);
        // printf("cusparse index base: %d\n", base);
        // printf("\ncol_host:\n");
        // for(int i=0;i<non0;i++)
        //     printf("%d, ",col_host[i]);
        // printf("\nrow_host:\n");
        // for(int i=0;i<N+1;i++)
        //     printf("%d, ",row_host[i]);
        // printf("\nhessian_host:\n");
        // for(int i=0;i<non0;i++)
        //     printf("%lf, ",hessian_host[i]);
        
        // printf("\nafter allocating and copying hessian_dev:\n");
        // check_cuda_mem();

        // hessian_dev = ALLOCATE_GPU<double>(hessian_dev, N*N*sizeof(double));
        direction_dev = ALLOCATE_GPU<double>(direction_dev, N*sizeof(double));
        grad_dev = ALLOCATE_GPU<double>(grad_dev, N*sizeof(double));
        tmp_dev = ALLOCATE_GPU<double>(tmp_dev, N*sizeof(double));
        res_dev = ALLOCATE_GPU<double>(res_dev, sizeof(double));
        grad_grad_dev = ALLOCATE_GPU<double>(grad_grad_dev, sizeof(double));
        grad_direc_dev = ALLOCATE_GPU<double>(grad_grad_dev, sizeof(double));

        // COPYDATATOGPU<double>(hessian_dev, hessian_host, N*N*sizeof(double));
        COPYDATATOGPU<double>(direction_dev, direction_host, N*sizeof(double));
        COPYDATATOGPU<double>(grad_dev, grad_host, N*sizeof(double));
        
        // cublasHandle_t handle;
        // cublasCreate(&handle);
        
        // cudaDeviceSynchronize();
        // cublasDgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, 1, N, N, &alpha, direction_dev, 1, hessian_dev, N, &beta, tmp_dev, 1);  // hessian * direction
        // cudaDeviceSynchronize();
        // cublasDaxpy(handle, N, &alpha, grad_dev, 1, tmp_dev, 1); // vector add, hessian * direction + grad
        // cudaDeviceSynchronize();
        // cublasDdot(handle, N, tmp_dev, 1, tmp_dev, 1, res_dev);  // dot product
        // cudaDeviceSynchronize();
        
        ////
        COPYDATATOGPU<double>(tmp_dev, grad_host, N*sizeof(double));

        cusparseStatus_t status;
        cusparseHandle_t handle=0;
        cusparseMatDescr_t descr=0;
        status= cusparseCreate(&handle);
        status= cusparseCreateMatDescr(&descr);
        cusparseSetMatType(descr , CUSPARSE_MATRIX_TYPE_GENERAL);
        cusparseIndexBase_t cubase;
        cusparseSetMatIndexBase(descr, CUSPARSE_INDEX_BASE_ZERO); 
        double *buffer;
        buffer = ALLOCATE_GPU<double>(buffer, 2*non0*sizeof(double));
        status=cusparseCsrmvEx(handle, CUSPARSE_ALG_MERGE_PATH, CUSPARSE_OPERATION_NON_TRANSPOSE, N, N, non0, &alpha , CUDA_R_64F, descr , hessian_dev, CUDA_R_64F, row_dev , col_dev , direction_dev , CUDA_R_64F, &beta , CUDA_R_64F, tmp_dev, CUDA_R_64F, CUDA_R_64F, buffer);
        cusparseDestroyMatDescr(descr);
        cusparseDestroy(handle); 
        cudaFree(row_dev);
        cudaFree(col_dev);
        cudaFree(buffer);

        cublasHandle_t handle2;
        cublasCreate(&handle2);
        cublasDdot(handle2, N, tmp_dev, 1, tmp_dev, 1, res_dev);  // dot product
        COPYDATATOHOST<double>(res_host, res_dev, sizeof(double));
        const double residual = std::sqrt(*res_host);  // norm

        // polyfem::logger().trace("residual_cpu: {}, residual_gpu: {}, diff: {}", residual_cpu, residual, residual-residual_cpu);
        ////
        
        // cublasHandle_t handle2;
        // cublasCreate(&handle2);

        cublasDdot(handle2, N, grad_dev, 1, grad_dev, 1, grad_grad_dev);
        cublasDdot(handle2, N, grad_dev, 1, direction_dev, 1, grad_direc_dev);
        COPYDATATOHOST<double>(grad_grad_host, grad_grad_dev, sizeof(double));
        COPYDATATOHOST<double>(grad_direc_host, grad_direc_dev, sizeof(double));
        const double grad_norm = std::sqrt(*grad_grad_host);
        const double grad_direc_prod = *grad_direc_host;
        
        delete[] res_host;
        delete[] tmp_host;
        delete[] grad_grad_host;
        delete[] grad_direc_host;
        cudaFree(hessian_dev);
        cudaFree(direction_dev);
        cudaFree(grad_dev);
        cudaFree(tmp_dev);
        cudaFree(res_dev);
        cublasDestroy(handle2);

        // gradient descent, check descent direction

        if (std::isnan(residual))
        {
            increase_descent_strategy();
            polyfem::logger().log(
                this->descent_strategy == 2 ? spdlog::level::warn : spdlog::level::debug,
                "nan linear solve residual {} (||∇f||={}); reverting to {}",
                residual, grad_norm, this->descent_strategy_name());
            return compute_update_direction_cuda(objFunc, x, grad, direction);
        }
        else if (residual > std::max(1e-8 * grad_norm, 1e-5))
        {
            increase_descent_strategy();
            polyfem::logger().log(
                this->descent_strategy == 2 ? spdlog::level::warn : spdlog::level::debug,
                "large linear solve residual {} (||∇f||={}); reverting to {}",
                residual, grad_norm, this->descent_strategy_name());
            return compute_update_direction_cuda(objFunc, x, grad, direction);
        }
        else
        {
            polyfem::logger().trace("linear solve residual {}", residual);
        }

        // do this check here because we need to repeat the solve without resetting reg_weight
        if (grad_direc_prod >= 0)
        {
            increase_descent_strategy();
            polyfem::logger().log(
                this->descent_strategy == 2 ? spdlog::level::warn : spdlog::level::debug,
                "[{}] direction is not a descent direction (Δx⋅g={}≥0); reverting to {}",
                name(), direction.dot(grad), descent_strategy_name());
            return compute_update_direction_cuda(objFunc, x, grad, direction);
        }

        json info;
        linear_solver->getInfo(info);
        internal_solver_info.push_back(info);

        reg_weight /= reg_weight_dec;
        if (reg_weight < reg_weight_min)
            reg_weight = 0;
        polyfem::logger().trace("reg_weight: {}", reg_weight);

        return true;
    }
    template class SparseNewtonDescentSolver<polyfem::solver::NLProblem>;
    template class SparseNewtonDescentSolver<polyfem::solver::ALNLProblem>;
}