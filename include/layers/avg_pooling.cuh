#ifndef CUDANET_AVG_POOLING_H
#define CUDANET_AVG_POOLING_H

#include "activation.cuh"
#include "layer.cuh"

namespace CUDANet::Layers {

class AvgPooling2D : public SequentialLayer {
  public:
    AvgPooling2D(
        int            inputSize,
        int            nChannels,
        int            poolingSize,
        int            stride,
        ActivationType activationType
    );
    ~AvgPooling2D();

    float* forward(const float* d_input);

    /**
     * @brief Get the output width (/ height) of the layer
     *
     * @return int
     */
    int getOutputSize() {
        return outputSize;
    }

  private:
    int inputSize;
    int nChannels;
    int poolingSize;
    int stride;

    int outputSize;

    float* d_output;

    Activation activation;
};

}  // namespace CUDANet::Layers

#endif  // CUDANET_AVG_POOLING_H
