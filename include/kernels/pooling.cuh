#ifndef CUDANET_POOLING_H
#define CUDANET_POOLING_H

#include <cuda_runtime.h>
#include "layer.cuh"

namespace CUDANet::Kernels {

__global__ void max_pooling(
    const float* __restrict__ d_input,
    float* __restrict__ d_output,
    const dim2d inputSize,
    const dim2d outputSize,
    const int nChannels,
    const dim2d poolingSize,
    const dim2d stride
);

__global__ void avg_pooling(
    const float* __restrict__ d_input,
    float* __restrict__ d_output,
    const dim2d inputSize,
    const dim2d outputSize,
    const int nChannels,
    const dim2d poolingSize,
    const dim2d stride
);

}  // namespace CUDANet::Kernels

#endif // CUDANET_POOLING_H