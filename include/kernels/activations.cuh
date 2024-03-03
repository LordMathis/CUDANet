#ifndef ACTIVATIONS_H
#define ACTIVATIONS_H

__global__ void
sigmoid_kernel(const float* __restrict__ src, float* __restrict__ dst, int len);

__global__ void
relu_kernel(const float* __restrict__ src, float* __restrict__ dst, int len);

__global__ void
linear_kernel(const float* __restrict__ src, float* __restrict__ dst, int len);

enum Activation {
    SIGMOID,
    RELU,
    LINEAR
};

#endif  // ACTIVATIONS_H