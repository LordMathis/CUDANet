#ifndef CUDANET_MAX_POOLING_H
#define CUDANET_MAX_POOLING_H

#include <cuda_runtime.h>

#include "layer.cuh"
#include "activation.cuh"

namespace CUDANet::Layers {

class MaxPooling2D : public SequentialLayer {
  public:
    MaxPooling2D(
        int            inputSize,
        int            nChannels,
        int            poolingSize,
        int            stride,
        ActivationType activationType
    );
    ~MaxPooling2D();

    float* forward(const float* d_input);

  private:
    int inputSize;
    int nChannels;
    int poolingSize;
    int stride;

    int outputSize;
    int gridSize;

    float* d_output;

    Activation activation;
};

}  // namespace CUDANet::Layers

#endif  // CUDANET_MAX_POOLING_H