
#ifndef CUDANET_I_LAYER_H
#define CUDANET_I_LAYER_H

#include <vector>

namespace CUDANet::Layers {

/**
 * @brief Activation functions
 * 
 * SIGMOID: Sigmoid
 * RELU: Rectified Linear Unit
 *
 */
enum Activation { SIGMOID, RELU, NONE };

/**
 * @brief Padding types
 * 
 * SAME: Zero padding such that the output size is the same as the input
 * VALID: No padding
 *
 */
enum Padding { SAME, VALID };

/**
 * @brief Base class for all layers
 */
class ILayer {
  public:
    /**
     * @brief Destroy the ILayer object
     * 
     */
    virtual ~ILayer() {}

    /**
     * @brief Virtual function for forward pass
     * 
     * @param input (Device) Pointer to the input
     * @return float* Device pointer to the output
     */
    virtual float* forward(const float* input)      = 0;

    /**
     * @brief Virtual function for setting weights
     * 
     * @param weights Pointer to the weights
     */
    virtual void   setWeights(const float* weights) = 0;

    /**
     * @brief Virtual function for setting biases
     * 
     * @param biases Pointer to the biases
     */
    virtual void   setBiases(const float* biases)   = 0;

  private:

    /**
     * @brief Initialize the weights
     */
    virtual void initializeWeights() = 0;

    /**
     * @brief Initialize the biases
     */
    virtual void initializeBiases()  = 0;

    /**
     * @brief Copy the weights and biases to the device
     */
    virtual void toCuda() = 0;

    int inputSize;
    int outputSize;

    float* d_output;

    float* d_weights;
    float* d_biases;

    std::vector<float> weights;
    std::vector<float> biases;

    Layers::Activation activation;
};

}  // namespace CUDANet::Layers

#endif  // CUDANET_I_LAYERH