#pragma once

#include <cuda_runtime.h>

#define CUDA_CHECK_ERROR(x) error = x; if(!cutil::checkError(error)) return -1; 

#define CUDA_TIME_IT(f, grid, block,msg, ...) { \
  printf("\nStarting Kernel Function %s\n", #f); \
  cudaEvent_t tstart, tstop; \
  float time; \
  cudaEventCreate(&tstart); \
  cudaEventCreate(&tstop); \
  cudaEventRecord(tstart, 0); \
  f<<<grid,block>>>(__VA_ARGS__); \
  cudaEventRecord(tstop, 0); \
  cudaEventSynchronize(tstop); \
  cudaEventElapsedTime(&time, tstart, tstop); \
  printf("%s: %f ms\n", msg, time); \
  cudaEventDestroy(tstart); \
  cudaEventDestroy(tstop); \
}

namespace cutil
{
  bool checkError(cudaError_t error);
}