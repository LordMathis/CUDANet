#ifndef CUDANET_DENSE_LAYER_H
#define CUDANET_DENSE_LAYER_H

#include <vector>

#include "activation.hpp"
#include "layer.hpp"

namespace CUDANet::Layers {

/**
 * @brief Dense (fully connected) layer
 *
 */
class Dense : public WeightedLayer {
  public:
    /**
     * @brief Construct a new Dense layer
     *
     * @param inputSize Size of the input vector
     * @param outputSize Size of the output vector
     * @param activationType Activation function type ('RELU', 'SIGMOID',
     * 'SOFTMAX' or 'NONE')
     */
    Dense(int inputSize, int outputSize, Layers::ActivationType activationType);

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
     * @brief Get the weights of the layer
     *
     * @return Vector of weights
     */
    std::vector<float> getWeights();

    /**
     * @brief Set the biases of the layer
     *
     * @param biases Pointer to vector of biases
     */
    void setBiases(const float* biases);

    /**
     * @brief Get the biases of the layer
     *
     * @return Vector of biases
     */
    std::vector<float> getBiases();

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
    int outputSize;

    std::vector<float> weights;
    std::vector<float> biases;

    Layers::Activation* activation;

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

    float* forwardCPU(const float* input);

#ifdef USE_CUDA
    float* d_output;

    float* d_weights;
    float* d_biases;

    // Precompute kernel launch parameters
    int forwardGridSize;
    int biasGridSize;

    /**
     * @brief Copy the weights and biases to the device
     *
     */
    void toCuda();

    void initCUDA();
    void delCUDA();

    float* forwardCUDA(const float* d_input);
#endif

};

}  // namespace CUDANet::Layers

#endif  // CUDANET_DENSE_LAYER_H
