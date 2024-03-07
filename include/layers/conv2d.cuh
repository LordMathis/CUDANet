#ifndef CONV_LAYER_H
#define CONV_LAYER_H

#include <string>
#include <vector>

#include "activations.cuh"

namespace Layers {

class Conv2d {
  public:
    Conv2d(
        int         inputSize,
        int         inputChannels,
        int         kernelSize,
        int         stride,
        std::string padding,
        int         numFilters,
        Activation  activation
    );
    ~Conv2d();

    // Outputs
    int outputSize;

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

    // Kernels
    std::vector<float> kernels;

    // Cuda
    float* d_kernels;
    float* d_padded;

    // Kernels
    Activation activation;

    void initializeKernels();
    void toCuda();

    void setKernels(const std::vector<float>& kernels_input);

    void host_conv(const float* input, float* output);
};

}  // namespace Layers

#endif  // CONV_LAYER_H
