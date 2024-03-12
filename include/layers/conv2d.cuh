#ifndef CONV_LAYER_H
#define CONV_LAYER_H

#include <string>
#include <vector>

#include "activations.cuh"
#include "convolution.cuh"
#include "ilayer.cuh"

namespace Layers {

class Conv2d : public ILayer {
  public:
    Conv2d(
        int                inputSize,
        int                inputChannels,
        int                kernelSize,
        int                stride,
        Layers::Padding    padding,
        int                numFilters,
        Layers::Activation activation
    );
    ~Conv2d();

    // Outputs
    int outputSize;

    float* forward(const float* d_input);
    void   setWeights(const float* weights_input);
    void   setBiases(const float* biases_input);
    void   host_conv(const float* input, float* output);

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
    std::vector<float> weights;
    std::vector<float> biases;

    // Cuda
    float* d_output;
    float* d_weights;
    float* d_biases;
    float* d_padded;

    // Kernels
    Layers::Activation activation;

    void initializeWeights();
    void initializeBiases();
    void toCuda();
};

}  // namespace Layers

#endif  // CONV_LAYER_H
