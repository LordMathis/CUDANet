#ifndef CUDANET_INPUT_LAYER_H
#define CUDANET_INPUT_LAYER_H

#include "layer.hpp"

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
     * @brief Forward pass of the input layer. Just copies the input to the
     * device
     *
     * @param input Host pointer to the input vector
     * @return Device pointer to the output vector
     */
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

  private:
    int inputSize;

    float* forwardCPU(const float* input);

#ifdef USE_CUDA
    float* d_output;

    float* forwardCUDA(const float* input);
    void   initCUDA();
    void   delCUDA();
#endif
};

}  // namespace CUDANet::Layers

#endif  // CUDANET_INPUT_LAYER_H