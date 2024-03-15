#ifndef CUDANET_DENSE_LAYER_H
#define CUDANET_DENSE_LAYER_H

#include <functional>
#include <string>
#include <vector>

#include "ilayer.cuh"

namespace Layers {

/**
 * @brief Dense (fully connected) layer
 *
 */
class Dense : public ILayer {
  public:
    /**
     * @brief Construct a new Dense layer
     *
     * @param inputSize Size of the input vector
     * @param outputSize Size of the output vector
     * @param activation Activation function ('RELU', 'SIGMOID' or 'NONE')
     */
    Dense(int inputSize, int outputSize, Layers::Activation activation);

    /**
     * @brief Destroy the Dense layer
     *
     */
    ~Dense();

    /**
     * @brief Forward pass of the dense layer
     *
     * @param d_input Device pointer to the input vector
     * @return Device pointer to the output vector
     */
    float* forward(const float* d_input);

    /**
     * @brief Set the weights of the layer
     *
     * @param weights Pointer to vector of weights
     */
    void setWeights(const float* weights);

    /**
     * @brief Set the biases of the layer
     *
     * @param biases Pointer to vector of biases
     */
    void setBiases(const float* biases);

  private:
    int inputSize;
    int outputSize;

    float* d_output;

    float* d_weights;
    float* d_biases;

    std::vector<float> weights;
    std::vector<float> biases;

    Layers::Activation activation;

    // Precompute kernel launch parameters
    int forwardGridSize;
    int biasGridSize;

    /**
     * @brief Initialize the weights to zeros
     *
     */
    void initializeWeights();

    /**
     * @brief Initialize the biases to zeros
     *
     */
    void initializeBiases();

    /**
     * @brief Copy the weights and biases to the device
     *
     */
    void toCuda();
};

}  // namespace Layers

#endif  // CUDANET_DENSE_LAYER_H
