#ifndef CONV_LAYER_H
#define CONV_LAYER_H

#include <cublas_v2.h>

#include <string>
#include <vector>

#include "activations.cuh"

namespace Layers {

class Conv2d {
  public:
    Conv2d(
        int            inputSize,
        int            inputChannels,
        int            kernelSize,
        int            stride,
        std::string    padding,
        int            numFilters,
        Activation     activation,
        cublasHandle_t cublasHandle
    );
    ~Conv2d();

    void forward(const float* d_input, float* d_output);

  private:
    // Inputs
    int inputSize;
    int inputChannels;

    // Kernel
    int kernelSize;
    int stride;
    int paddingSize;
    int numFilters;

    // Outputs
    int outputSize;

    // Kernels
    std::vector<float> kernels;

    // Cuda
    cublasHandle_t cublasHandle;
    float*         d_kernels;
    float*         d_padded;

    // Kernels
    Activation activation;

    void initializeKernels();
    void toCuda();
};

}  // namespace Layers

#endif  // CONV_LAYER_H
