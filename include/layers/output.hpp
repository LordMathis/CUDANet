#ifndef CUDANET_OUTPUT_LAYER_H
#define CUDANET_OUTPUT_LAYER_H

#include "layer.hpp"

namespace CUDANet::Layers {

class Output : public SequentialLayer {
  public:
    /**
     * @brief Create a new Output layer
     *
     * @param inputSize Size of the input vector
     */
    explicit Output(int inputSize);

    /**
     * @brief Destroy the Output layer
     *
     */
    ~Output();

    /**
     * @brief Forward pass of the output layer. Just copies the input from
     * device to host
     *
     * @param input Device pointer to the input vector
     * @return Host pointer to the output vector
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
    int    inputSize;
    float* h_output;

    float* forwardCPU(const float* input);

#ifdef USE_CUDA
    float* forwardCUDA(const float* input);
#endif
};

}  // namespace CUDANet::Layers

#endif  // CUDANET_OUTPUT_LAYER_H