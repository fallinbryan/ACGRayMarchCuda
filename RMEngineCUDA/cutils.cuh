#pragma once

#include <cuda_runtime.h>

#define CUDA_CHECK_ERROR(x) error = x; if(!cutil::checkError(error)) return -1; 

namespace cutil
{
  bool checkError(cudaError_t error);
}