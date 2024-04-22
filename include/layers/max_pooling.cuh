#ifndef CUDANET_MAX_POOLING_H
#define CUDANET_MAX_POOLING_H

#include "activation.cuh"
#include "layer.cuh"

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
    int inputSize;
    int nChannels;
    int poolingSize;
    int stride;

    int outputSize;

    float* d_output;

    Activation* activation;
};

}  // namespace CUDANet::Layers

#endif  // CUDANET_MAX_POOLING_H