#ifndef CUDANET_POOLING_H
#define CUDANET_POOLING_H

#include <cuda_runtime.h>

namespace CUDANet::Kernels {

__global__ void max_pooling(
    const float* __restrict__ d_input,
    float* __restrict__ d_output,
    const int inputSize,
    const int nChannels,
    const int poolingSize,
    const int stride
);

__global__ void avg_pooling(
    const float* __restrict__ d_input,
    float* __restrict__ d_output,
    const int inputSize,
    const int nChannels,
    const int poolingSize,
    const int stride
);

}  // namespace CUDANet::Kernels

#endif // CUDANET_POOLING_H