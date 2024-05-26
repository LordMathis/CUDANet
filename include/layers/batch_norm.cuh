#ifndef CUDANET_BATCH_NORM_H
#define CUDANET_BATCH_NORM_H

#include <vector>

#include "activation.cuh"
#include "layer.cuh"

namespace CUDANet::Layers {

class BatchNorm2d : public WeightedLayer, public TwoDLayer {
  public:
    BatchNorm2d(dim2d inputSize, int inputChannels, float epsilon, ActivationType activationType);

    ~BatchNorm2d();

    /**
     * @brief Compute the forward pass of the batchnorm layer
     *
     * @param d_input Device pointer to the input
     * @return float* Device pointer to the output
     */
    float* forward(const float* d_input);

    /**
     * @brief Set the weights of the batchnorm layer
     *
     * @param weights_input Pointer to the weights
     */
    void setWeights(const float* weights_input);

    /**
     * @brief Get the weights of the batchnorm layer
     *
     * @return std::vector<float>
     */
    std::vector<float> getWeights();

    /**
     * @brief Set the biases of the batchnorm layer
     *
     * @param biases_input Pointer to the biases
     */
    void setBiases(const float* biases_input);

    /**
     * @brief Get the biases of the batchnorm layer
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

    dim2d getOutputDims();

  private:

    dim2d inputSize;
    int inputChannels;

    int gridSize;

    float* d_output;

    float* d_mean;
    float* d_mean_sub;
    float* d_sqrt_var;

    float* d_length;
    float* d_epsilon;

    float* d_weights;
    float* d_biases;

    std::vector<float> weights;
    std::vector<float> biases;

    std::vector<float> mean;
    std::vector<float> sqrt_var;

    Activation* activation;

    /**
     * @brief Initialize weights of the batchnorm layer with zeros
     *
     */
    void initializeWeights();

    /**
     * @brief Initialize biases of the batchnorm layer with zeros
     *
     */
    void initializeBiases();

    /**
     * @brief Initialize mean of the batchnorm layer with zeros
     *
     */
    void initializeMean();

    /**
     * @brief Initialize sqrt of variance of the batchnorm layer with ones
     *
     */
    void initializeSqrtVar();

    /**
     * @brief Copy weights and biases to the device
     *
     */
    void toCuda();
};

}  // namespace CUDANet::Layers

#endif  // CUDANET_BATCH_NORM_H