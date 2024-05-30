#ifndef CUDANET_AVG_POOLING_H
#define CUDANET_AVG_POOLING_H

#include "activation.cuh"
#include "layer.cuh"

namespace CUDANet::Layers {

class AvgPooling2d : public SequentialLayer, public TwoDLayer {
  public:
    AvgPooling2d(
        shape2d        inputSize,
        int            nChannels,
        shape2d        poolingSize,
        shape2d        stride,
        shape2d        padding,
        ActivationType activationType
    );
    ~AvgPooling2d();

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

  protected:
    shape2d inputSize;
    int     nChannels;
    shape2d poolingSize;
    shape2d stride;
    shape2d padding;

    shape2d outputSize;

    float* d_output;

    Activation* activation;
};

class AdaptiveAvgPooling2d : public AvgPooling2d {
  public:
    AdaptiveAvgPooling2d(shape2d inputShape, int nChannels, shape2d outputShape, ActivationType activationType);
};

}  // namespace CUDANet::Layers

#endif  // CUDANET_AVG_POOLING_H
