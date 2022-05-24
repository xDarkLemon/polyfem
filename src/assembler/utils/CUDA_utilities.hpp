#pragma once
#include <cuda.h>
#include <cuda_runtime.h>
#include <stdio.h>

#define NUMBER_THREADS  32
/*
template <typename T>
T* ALLOCATE_GPU(T *d_A, int _size);

template <typename T>
void COPYDATATOGPU(T *d_A,T *A,int _size);

template <typename T>
void COPYDATATOHOST(T *A,T *d_A,int _size);
*/

template <typename T>
T* ALLOCATE_GPU(T *d_A, int _size) {
  if (cudaMalloc((void **)&d_A, _size) != cudaSuccess)
  {
      printf("Error allocating GPU\n");
      abort();
  }
  else{
          return d_A;
  }
}
template <typename T>
void COPYDATATOGPU(T *d_A,T *A,int _size){
    if(cudaMemcpy(d_A, A, _size, cudaMemcpyHostToDevice) != cudaSuccess)
    {
      printf("Error copying to GPU\n");
      abort();
    }
}

template <typename X>
void COPYDATATOGPU(X *d_A,const X *A,int _size){
    if(cudaMemcpy(d_A, A, _size, cudaMemcpyHostToDevice) != cudaSuccess)
    {
      printf("Error copying to GPU\n");
      abort();
    }
}

template <typename T>
void COPYDATATOHOST(T *A,T *d_A,int _size){
    if(cudaMemcpy(A,d_A,_size,cudaMemcpyDeviceToHost) != cudaSuccess)
    {
      printf("Error copying to CPU\n");
      abort();
    }
}


/*
template <typename T>
void CUDA_FREE(T *d_A){
cudaFree(d_A);
}

void CUDA_DEV_SYNCHRONIZE(void){
   cudaDeviceSynchronize();
}
*/