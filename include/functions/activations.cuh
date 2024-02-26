#include <functional>

#ifndef ACTIVATIONS_H
#define ACTIVATIONS_H

__device__ float sigmoid(float a);
__device__ float relu(float a);
__device__ float linear(float a);

__global__ void sigmoid_kernel(const float* __restrict__ src, float* __restrict__ dst, int len);
__global__ void relu_kernel(const float* __restrict__ src, float* __restrict__ dst, int len);
__global__ void linear_kernel(const float* __restrict__ src, float* __restrict__ dst, int len);

#endif // ACTIVATIONS_H