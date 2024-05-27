#ifndef CUDANET_MAX_POOLING_H
#define CUDANET_MAX_POOLING_H

#include "activation.cuh"
#include "layer.cuh"

namespace CUDANet::Layers {

class MaxPooling2d : public SequentialLayer, public TwoDLayer {
  public:
    MaxPooling2d(
        shape2d        inputSize,
        int            nChannels,
        shape2d        poolingSize,
        shape2d        stride,
        shape2d        padding,
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

    shape2d getOutputDims();

  private:
    shape2d inputSize;
    int     nChannels;
    shape2d poolingSize;
    shape2d stride;
    shape2d padding;

    shape2d outputSize;

    float* d_output;

    Activation* activation;
};

}  // namespace CUDANet::Layers

#endif  // CUDANET_MAX_POOLING_H