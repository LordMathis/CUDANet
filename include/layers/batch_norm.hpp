#ifndef CUDANET_BATCH_NORM_H
#define CUDANET_BATCH_NORM_H

#include <vector>

#include "activation.hpp"
#include "layer.hpp"

namespace CUDANet::Layers {

class BatchNorm2d : public WeightedLayer, public TwoDLayer {
  public:
    BatchNorm2d(
        shape2d        inputSize,
        int            inputChannels,
        float          epsilon,
        ActivationType activationType
    );

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
     * @brief Set the Running Mean
     *
     * @param running_mean_input
     */
    void setRunningMean(const float* running_mean_input);

    /**
     * @brief Get the Running Mean
     *
     */
    std::vector<float> getRunningMean();

    /**
     * @brief Set the Running Var
     *
     * @param running_mean_input
     */
    void setRunningVar(const float* running_mean_input);

    /**
     * @brief Get the Running Var
     *
     */
    std::vector<float> getRunningVar();

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

    shape2d getOutputDims();

  private:
    shape2d inputSize;
    int     inputChannels;
    float   epsilon;

    int gridSize;

#ifdef USE_CUDA

    float* d_output;

    float* d_running_mean;
    float* d_running_var;

    float* d_length;
    float* d_epsilon;

    float* d_weights;
    float* d_biases;

    void initCUDA();
    void delCUDA();

    /**
     * @brief Copy weights and biases to the device
     *
     */
    void toCuda();

    float* forwardCUDA(const float* d_input);

#endif

    std::vector<float> weights;
    std::vector<float> biases;

    std::vector<float> running_mean;
    std::vector<float> running_var;

    Activation* activation;

    float* forwardCPU(const float* input);

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
    void initializeRunningMean();

    /**
     * @brief Initialize sqrt of variance of the batchnorm layer with ones
     *
     */
    void initializeRunningVar();
};

}  // namespace CUDANet::Layers

#endif  // CUDANET_BATCH_NORM_H