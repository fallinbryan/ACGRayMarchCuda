
#include "cutils.cuh"

#include <iostream>

#include <cuda_runtime.h>



bool cutil::checkError(cudaError_t error) {
  if (error != cudaSuccess) {
    std::cout << "CUDA error: " << cudaGetErrorString(error) << std::endl;
    return false;
  }
  return true;
};