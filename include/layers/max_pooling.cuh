#ifndef CUDANET_MAX_POOLING_H
#define CUDANET_MAX_POOLING_H

#include "activation.cuh"
#include "layer.cuh"

namespace CUDANet::Layers {

class MaxPooling2d : public SequentialLayer {
  public:
    MaxPooling2d(
        dim2d          inputSize,
        int            nChannels,
        dim2d          poolingSize,
        dim2d          stride,
        ActivationType activationType
    );
    ~MaxPooling2d();

    float* forward(const float* d_input);

    /**
     * @brief Get output size
     *
     * @return int output size
     */
    int getOutputSize();

    /**
     * @brief Get input size
     *
     * @return int input size
     */
    int getInputSize();

  private:
    dim2d inputSize;
    int   nChannels;
    dim2d poolingSize;
    dim2d stride;

    dim2d outputSize;

    float* d_output;

    Activation* activation;
};

}  // namespace CUDANet::Layers

#endif  // CUDANET_MAX_POOLING_H