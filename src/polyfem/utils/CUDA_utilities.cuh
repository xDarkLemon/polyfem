#pragma once
#include <cuda.h>
#include <cuda_runtime.h>
#include <stdio.h>

#define NUMBER_THREADS 32
#define gpuErrchk(ans)                        \
	{                                         \
		gpuAssert((ans), __FILE__, __LINE__); \
	}
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort = true)
{
	if (code != cudaSuccess)
	{
		fprintf(stderr, "GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
		if (abort)
			exit(code);
	}
}

#define CHECK_CUDA_ERROR(val) check((val), #val, __FILE__, __LINE__)
template <typename T>
void check(T err, char const *const func, char const *const file,
		   int const line)
{
	if (err != cudaSuccess)
	{
		std::cerr << "CUDA Runtime Error at: " << file << ":" << line
				  << std::endl;
		std::cerr << cudaGetErrorString(err) << " " << func << std::endl;
		std::exit(EXIT_FAILURE);
	}
}

template <typename T>
T *ALLOCATE_GPU(T *d_A, int _size)
{
	if (cudaMalloc(reinterpret_cast<void **>(&d_A), _size) != cudaSuccess)
	{
		cudaError_t err = cudaGetLastError(); // Get error code
		printf("CUDA Error: %s\n", cudaGetErrorString(err));
		printf("Error allocating GPU\n");
		abort();
	}
	else
	{
		return d_A;
	}
}
template <typename T>
void COPYDATATOGPU(T *d_A, T *A, int _size)
{
	if (cudaMemcpy(d_A, A, _size, cudaMemcpyHostToDevice) != cudaSuccess)
	{
		cudaError_t err = cudaGetLastError(); // Get error code
		printf("CUDA Error: %s\n", cudaGetErrorString(err));
		printf("Error copying to GPU\n");
		abort();
	}
}

template <typename X>
void COPYDATATOGPU(X *d_A, const X *A, int _size)
{
	if (cudaMemcpy(d_A, A, _size, cudaMemcpyHostToDevice) != cudaSuccess)
	{
		cudaError_t err = cudaGetLastError(); // Get error code
		printf("CUDA Error: %s\n", cudaGetErrorString(err));
		printf("Error copying to GPU\n");
		abort();
	}
}

template <typename T>
void COPYDATATOHOST(T *A, T *d_A, int _size)
{
	if (cudaMemcpy(A, d_A, _size, cudaMemcpyDeviceToHost) != cudaSuccess)
	{
		cudaError_t err = cudaGetLastError(); // Get error code
		printf("CUDA Error: %s\n", cudaGetErrorString(err));
		printf("Error copying to CPU\n");
		abort();
	}
}

namespace polyfem
{
	class DATA_POINTERS_GPU
	{
	public:
		DATA_POINTERS_GPU()
		{
			n_elements = 0;
			jac_it_size = 0;
			n_loc_bases = 0;
			global_vector_size = 0;
			n_pts = 0;

			jac_it_dev_ptr = nullptr;
			global_data_dev_ptr = nullptr;
			da_dev_ptr = nullptr;
			grad_dev_ptr = nullptr;
			mu_ptr = nullptr;
			lambda_ptr = nullptr;
		}

		int n_elements;
		int jac_it_size;
		int n_loc_bases;
		int global_vector_size;
		int n_pts;

		Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, 0, 3, 3> *jac_it_dev_ptr;
		basis::Local2Global_GPU *global_data_dev_ptr;
		Eigen::Matrix<double, -1, 1, 0, 3, 1> *da_dev_ptr;
		Eigen::Matrix<double, -1, -1, 0, 3, 3> *grad_dev_ptr;
		double *mu_ptr;
		double *lambda_ptr;
	};
} // namespace polyfem

template <typename T>
__device__ void DYNAMIC_GPU_ALLOC(T *A, int _size)
{
	A = new T[_size];
}

template <typename T>
__device__ void DYNAMIC_GPU_FREE(T *A)
{
	delete[] A;
}

void CATCHCUDAERROR(void);

__device__ void clock_block(int count);

template <typename T>
void mult_trace(T *, int, double *);

template <typename T>
__global__ void multiplicative_trace(T *mat, int n, double *result);

template <typename T>
__device__ void mat_mul_cache(T *, T *, T *, int);

/*
void CATCHCUDAERROR(void){
cudaError_t err = cudaGetLastError();        // Get error code
	   if ( err != cudaSuccess )
	   {
			  printf("CUDA Error: %s\n", cudaGetErrorString(err));
			  exit(-1);
	   }
}

template <typename T>
void CUDA_FREE(T *d_A){
cudaFree(d_A);
}

void CUDA_DEV_SYNCHRONIZE(void){
   cudaDeviceSynchronize();
}
*/
