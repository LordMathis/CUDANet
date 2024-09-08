#ifndef CUDANET_CONV_LAYER_H
#define CUDANET_CONV_LAYER_H

#include <vector>

#include "activation.hpp"
#include "convolution.cuh"
#include "layer.hpp"

namespace CUDANet::Layers {

/**
 * @brief 2D convolutional layer
 *
 */
class Conv2d : public WeightedLayer, public TwoDLayer {
  public:
    /**
     * @brief Construct a new Conv 2d layer
     *
     * @param inputSize Width and height of the input matrix
     * @param inputChannels Number of channels in the input matrix
     * @param kernelSize Width and height of the convolution kernel
     * @param stride Convolution stride
     * @param numFilters Number of output filters
     * @param paddingSize Padding size
     * @param activationType Activation function type ('RELU', 'SIGMOID',
     * 'SOFTMAX' or 'NONE')
     */
    Conv2d(
        shape2d          inputSize,
        int            inputChannels,
        shape2d          kernelSize,
        shape2d          stride,
        int            numFilters,
        shape2d          paddingSize,
        ActivationType activationType
    );

    /**
     * @brief Destroy the Conv 2d object
     *
     */
    ~Conv2d();

    /**
     * @brief Forward pass of the convolutional layer
     *
     * @param d_input Device pointer to the input matrix
     * @return Device pointer to the output matrix
     */
    float* forward(const float* d_input);

    /**
     * @brief Set the weights of the convolutional layer
     *
     * @param weights_input Pointer to the weights
     */
    void setWeights(const float* weights_input);

    /**
     * @brief Get the weights of the convolutional layer
     *
     * @return std::vector<float>
     */
    std::vector<float> getWeights();

    /**
     * @brief Set the biases of the convolutional layer
     *
     * @param biases_input Pointer to the biases
     */
    void setBiases(const float* biases_input);

    /**
     * @brief Get the biases of the convolutional layer
     *
     * @return std::vector<float>
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

    /**
     * @brief Get the padding size of the layer
     *
     * @return int
     */
    shape2d getPaddingSize() {
        return paddingSize;
    }

    shape2d getOutputDims();

  private:
    // Inputs
    shape2d inputSize;
    int   inputChannels;

    // Outputs
    shape2d outputSize;

    // Kernel
    shape2d kernelSize;
    shape2d stride;
    shape2d paddingSize;
    int   numFilters;

    // Kernels
    std::vector<float> weights;
    std::vector<float> biases;

    // Cuda
    float* d_output;
    float* d_weights;
    float* d_biases;

    Activation* activation;

    /**
     * @brief Initialize weights of the convolutional layer with zeros
     *
     */
    void initializeWeights();

    /**
     * @brief Initialize biases of the convolutional layer with zeros
     *
     */
    void initializeBiases();

    /**
     * @brief Copy weights and biases to the device
     *
     */
    void toCuda();
};

}  // namespace CUDANet::Layers

#endif  // CUDANET_CONV_LAYER_H
