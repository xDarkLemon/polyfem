#pragma once
#include <cuda.h>
#include <cuda_runtime.h>
#include <stdio.h>
#include "cusparse.h"
#include "CUDA_utilities.cuh"

#include <igl/Timer.h>

using namespace std;
#include <vector>

// Eigen's sparse matrix is in CSC format, whereas cuSparse is in CSR format. Therefore, need a transpose.
// https://stackoverflow.com/questions/57334742/convert-eigensparsematrix-to-cusparse-and-vice-versa
inline void EigenSparseToCuSparseTranspose(const Eigen::SparseMatrix<double> &mat, int *&row, int *&col, double *&val)
{
	using namespace polyfem;
    igl::Timer timerg;

	int num_non0  = mat.nonZeros();
	int num_outer = mat.cols() + 1;

    timerg.start();
	row = ALLOCATE_GPU<int>(row, (mat.cols()+1)*sizeof(int));
	col = ALLOCATE_GPU<int>(col, mat.nonZeros()*sizeof(int));
	val = ALLOCATE_GPU<double>(val, mat.nonZeros()*sizeof(double));
    timerg.stop();
    logger().trace("CUDA MALLOC {}s", timerg.getElapsedTime());

    timerg.start();
    cudaMemcpy(row, mat.outerIndexPtr(), sizeof(int) * num_outer, cudaMemcpyHostToDevice);
	cudaMemcpy(col, mat.innerIndexPtr(), sizeof(int) * num_non0, cudaMemcpyHostToDevice);
	cudaMemcpy(val, mat.valuePtr(), sizeof(double) * num_non0, cudaMemcpyHostToDevice);
    timerg.stop();
    logger().trace("DATA MOVING HTOD {}s", timerg.getElapsedTime());
}

inline void CuSparseMatrixAdd(
    int m, int n,
    int nnzA, const double *csrValA, const int *csrRowPtrA, const int *csrColIndA,     
    int nnzB, const double *csrValB, const int *csrRowPtrB, const int *csrColIndB,     
    int &nnzCret, double *&csrValC, int *&csrRowPtrC, int *&csrColIndC     
)
{
	cusparseStatus_t status;
	cusparseHandle_t handle=0;
	cusparseMatDescr_t descr=0;
    status= cusparseCreate(&handle);
	status= cusparseCreateMatDescr(&descr);
	cusparseSetMatType(descr, CUSPARSE_MATRIX_TYPE_GENERAL);
	cusparseSetMatIndexBase(descr, CUSPARSE_INDEX_BASE_ZERO); 	
    const double a = 1.0;
    const double b = 1.0;
	const double *alpha = &a;
	const double *beta = &b;
    int baseC, nnzC;
    /* alpha, nnzTotalDevHostPtr points to host memory */
    int *nnzTotalDevHostPtr = &nnzC;
    cusparseSetPointerMode(handle, CUSPARSE_POINTER_MODE_HOST);
    cudaMalloc((void**)&csrRowPtrC, sizeof(int)*(m+1));

    double *buffer;
	buffer = ALLOCATE_GPU<double>(buffer, 4*(nnzA+nnzB)*sizeof(double));    

    cusparseXcsrgeam2Nnz(handle, m, n,
            descr, /*descrA*/ nnzA, csrRowPtrA, csrColIndA,
            descr, /*descrB*/ nnzB, csrRowPtrB, csrColIndB,
            descr, /*descrC*/ csrRowPtrC, nnzTotalDevHostPtr,
            buffer);
    if (NULL != nnzTotalDevHostPtr){
        nnzC = *nnzTotalDevHostPtr;
    }else{
        cudaMemcpy(&nnzC, csrRowPtrC+m, sizeof(int), cudaMemcpyDeviceToHost);
        cudaMemcpy(&baseC, csrRowPtrC, sizeof(int), cudaMemcpyDeviceToHost);
        nnzC -= baseC;
    }
    cudaMalloc((void**)&csrColIndC, sizeof(int)*nnzC);
    cudaMalloc((void**)&csrValC, sizeof(double)*nnzC);
    cusparseDcsrgeam2(handle, m, n,
            alpha,
            descr, /*descrA*/ nnzA,
            csrValA, csrRowPtrA, csrColIndA,
            beta,
            descr, /*descrB*/ nnzB,
            csrValB, csrRowPtrB, csrColIndB,
            descr, /*descrC*/
            csrValC, csrRowPtrC, csrColIndC,
            buffer);

	cusparseDestroyMatDescr(descr);
	cusparseDestroy(handle); 
	cudaFree(buffer);

    nnzCret = nnzC;
}

inline void CuSparseHessianSum(
    const Eigen::SparseMatrix<double> &tmp, 
    int &hessian_nnz, double **hessian_val, int **hessian_row, int **hessian_col)
{
    using namespace polyfem;
    igl::Timer timerg;

    double *tmp_val;
    int *tmp_row, *tmp_col;
    timerg.start();
    EigenSparseToCuSparseTranspose(tmp, tmp_row, tmp_col, tmp_val);
    timerg.stop();
    logger().trace("EIGEN TO CUSPARSE {}s", timerg.getElapsedTime());

    double *new_hessian_val;
    int *new_hessian_row;
    int *new_hessian_col;
    int new_hessian_nnz;

    timerg.start();
    CuSparseMatrixAdd(
        tmp.cols(), tmp.cols(),
        tmp.nonZeros(), tmp_val, tmp_row, tmp_col, 
        hessian_nnz, *hessian_val, *hessian_row, *hessian_col,
        new_hessian_nnz, new_hessian_val, new_hessian_row, new_hessian_col);
    timerg.stop();
    logger().trace("CUSPARSE ADD {}s", timerg.getElapsedTime());

    hessian_nnz = new_hessian_nnz;

    timerg.start();
    cudaMemcpy(*hessian_row, new_hessian_row, sizeof(int)*(tmp.cols()+1), cudaMemcpyDeviceToDevice);    
    cudaMemcpy(*hessian_col, new_hessian_col, sizeof(int)*hessian_nnz, cudaMemcpyDeviceToDevice);    
    cudaMemcpy(*hessian_val, new_hessian_val, sizeof(double)*hessian_nnz, cudaMemcpyDeviceToDevice);      
    timerg.stop();
    logger().trace("DATA MOVING DTOD {}s", timerg.getElapsedTime());

    timerg.start();
    cudaFree(new_hessian_val);
    cudaFree(new_hessian_row);
    cudaFree(new_hessian_col);
    timerg.stop();
    logger().trace("FREE NEW HESSIAN {}s", timerg.getElapsedTime());

    timerg.start();
    cudaFree(tmp_val);
    cudaFree(tmp_row);
    cudaFree(tmp_col);
    timerg.stop();
    logger().trace("FREE TMP {}s", timerg.getElapsedTime());
}
