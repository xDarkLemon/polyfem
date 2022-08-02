#pragma once

#include "CUDA_utilities.cuh"

void CATCHCUDAERROR(void)
{
	cudaError_t err = cudaGetLastError(); // Get error code
	if (err != cudaSuccess)
	{
		printf("CUDA Error: %s\n", cudaGetErrorString(err));
		exit(-1);
	}
}

// ONLY CALL WITH ONE BLOCK
template <typename T>
__device__ void mat_mul_cache(T *d_A, T *d_B, T *d_C, int M_size)
{
	__shared__ T Mds[NUMBER_THREADS][NUMBER_THREADS];
	__shared__ T Nds[NUMBER_THREADS][NUMBER_THREADS];

	int tx = threadIdx.x;
	int ty = threadIdx.y;
	int Row = ty;
	int Col = tx;
	int tile_width = (NUMBER_THREADS > M_size) ? M_size : NUMBER_THREADS;

	int xy_A, xy_B, x_A, y_B;
	T cumulative_value = 0;

	x_A = tx;
	y_B = ty;
	xy_A = Row * tile_width + x_A;
	xy_B = Col + y_B * tile_width;

	if (x_A < M_size)
	{
		Mds[ty][tx] = d_A[xy_A];
	}
	else
	{
		Mds[ty][tx] = 0;
	}

	if (y_B < M_size)
	{
		Nds[ty][tx] = d_B[xy_B];
	}
	else
	{
		Nds[ty][tx] = 0;
	}

	__syncthreads();
	for (int k = 0; k < tile_width; ++k)
		cumulative_value += Mds[ty][k] * Nds[k][tx];

	if (Row < M_size && Col < M_size)
		d_C[Row * M_size + Col] = cumulative_value;
}

template <typename T>
__global__ void multiplicative_trace(T *mat, int n, double *result)
{
	int bx = blockIdx.x;
	int by = blockIdx.y;
	int tx = threadIdx.x;
	int ty = threadIdx.y;
	int inner_index_x = bx * NUMBER_THREADS + tx;
	int inner_index_y = by * NUMBER_THREADS + ty;
	result[0] = 1.0;
	if (inner_index_x == 0 && inner_index_y == 0)
	{
		for (int i = 0; i < n; i++)
		{
			result[0] *= mat[i * (n + 1)];
		}
		result[0] = log(result[0]);
	}
	return;
}

template <typename T>
void mult_trace(T *mat, int n, double *result)
{
	multiplicative_trace<<<1, 1>>>(mat, n, result);
}

__device__ void clock_block(int count)
{
	clock_t start_clock = clock();
	clock_t clock_offset = 0;
	while (clock_offset < count)
	{
		clock_offset = clock() - start_clock;
	}
	return;
}
template void mult_trace<double>(double *, int, double *);
/*
template <typename T>
void CUDA_FREE(T *d_A){
cudaFree(d_A);
}

void CUDA_DEV_SYNCHRONIZE(void){
   cudaDeviceSynchronize();
}
*/
