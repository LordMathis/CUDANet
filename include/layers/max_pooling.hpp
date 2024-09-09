#ifndef CUDANET_MAX_POOLING_H
#define CUDANET_MAX_POOLING_H

#include "activation.hpp"
#include "layer.hpp"

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

    float* forward(const float* input);

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

    Activation* activation;

    float* forwardCPU(const float* input);

#ifdef USE_CUDA
    float* d_output;
    float* forwardCUDA(const float* d_input);

    void initCUDA();
    void delCUDA();
#endif
};

}  // namespace CUDANet::Layers

#endif  // CUDANET_MAX_POOLING_H