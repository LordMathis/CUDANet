#ifndef CUDANET_INPUT_LAYER_H
#define CUDANET_INPUT_LAYER_H

#include "layer.cuh"

namespace CUDANet::Layers {

/**
 * @brief Input layer, just copies the input to the device
 *
 */
class Input : public SequentialLayer {
  public:
    /**
     * @brief Create a new Input layer
     * 
     * @param inputSize Size of the input vector
     */
    explicit Input(int inputSize);

    /**
     * @brief Destroy the Input layer
     * 
     */
    ~Input();

    /**
     * @brief Forward pass of the input layer. Just copies the input to the device
     * 
     * @param input Host pointer to the input vector
     * @return Device pointer to the output vector
     */
    float* forward(const float* input);

  private:
    int    inputSize;
    float* d_output;
};

}  // namespace CUDANet::Layers

#endif  // CUDANET_INPUT_LAYER_H