#ifndef CUDANET_CONV_LAYER_H
#define CUDANET_CONV_LAYER_H

#include <vector>

#include "activation.cuh"
#include "convolution.cuh"
#include "layer.cuh"

namespace CUDANet::Layers {

/**
 * @brief 2D convolutional layer
 *
 */
class Conv2d : public WeightedLayer {
  public:
    /**
     * @brief Construct a new Conv 2d layer
     *
     * @param inputSize Width and height of the input matrix
     * @param inputChannels Number of channels in the input matrix
     * @param kernelSize Width and height of the convolution kernel
     * @param stride Convolution stride
     * @param numFilters Number of output filters
     * @param padding Padding type ('SAME' or 'VALID')
     * @param activationType Activation function type ('RELU', 'SIGMOID',
     * 'SOFTMAX' or 'NONE')
     */
    Conv2d(
        int            inputSize,
        int            inputChannels,
        int            kernelSize,
        int            stride,
        int            numFilters,
        Padding        padding,
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
     * @brief Set the biases of the convolutional layer
     *
     * @param biases_input Pointer to the biases
     */
    void setBiases(const float* biases_input);

    /**
     * @brief Get the output width (/ height) of the layer
     *
     * @return int
     */
    int getOutputSize() {
        return outputSize;
    }

    /**
     * @brief Get the padding size of the layer
     *
     * @return int
     */
    int getPaddingSize() {
        return paddingSize;
    }

  private:
    // Inputs
    int inputSize;
    int inputChannels;

    // Outputs
    int outputSize;

    // Kernel
    int kernelSize;
    int stride;
    int paddingSize;
    int numFilters;

    // Kernels
    std::vector<float> weights;
    std::vector<float> biases;

    // Cuda
    float* d_output;
    float* d_weights;
    float* d_biases;

    // Kernels
    Activation activation;

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
