#ifndef CUDANET_POOLING_H
#define CUDANET_POOLING_H

#include <cuda_runtime.h>
#include "layer.cuh"

namespace CUDANet::Kernels {

__global__ void max_pooling(
    const float* __restrict__ d_input,
    float* __restrict__ d_output,
    const shape2d inputSize,
    const shape2d outputSize,
    const int nChannels,
    const shape2d poolingSize,
    const shape2d stride,
    const shape2d padding
);

__global__ void avg_pooling(
    const float* __restrict__ d_input,
    float* __restrict__ d_output,
    const shape2d inputSize,
    const shape2d outputSize,
    const int nChannels,
    const shape2d poolingSize,
    const shape2d stride,
    const shape2d padding
);

}  // namespace CUDANet::Kernels

#endif // CUDANET_POOLING_H