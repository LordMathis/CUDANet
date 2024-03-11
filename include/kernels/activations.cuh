#ifndef ACTIVATIONS_H
#define ACTIVATIONS_H

namespace Kernels {

__global__ void
sigmoid(const float* __restrict__ src, float* __restrict__ dst, int len);

__global__ void
relu(const float* __restrict__ src, float* __restrict__ dst, int len);

}  // namespace Kernels

#endif  // ACTIVATIONS_H