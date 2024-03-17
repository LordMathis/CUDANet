#ifndef CUDANET_INPUT_LAYER_H
#define CUDANET_INPUT_LAYER_H

#include <ilayer.cuh>

namespace CUDANet::Layers {

/**
 * @brief Input layer, just copies the input to the device
 *
 */
class Input : public ILayer {
  public:
    /**
     * @brief Create a new Input layer
     * 
     * @param inputSize Size of the input vector
     */
    Input(int inputSize);

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

    void setWeights(const float* weights);
    void setBiases(const float* biases);

  private:
    void initializeWeights();
    void initializeBiases();

    void toCuda();

    int    inputSize;
    float* d_output;
};

}  // namespace CUDANet::Layers

#endif  // CUDANET_INPUT_LAYER_H