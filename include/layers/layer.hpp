
#ifndef CUDANET_I_LAYER_H
#define CUDANET_I_LAYER_H

#include <vector>

#define CUDANET_SAME_PADDING(inputSize, kernelSize, stride) \
    ((stride - 1) * inputSize - stride + kernelSize) / 2;

typedef std::pair<int, int> shape2d;

namespace CUDANet::Layers {


class TwoDLayer {

  public:
    virtual shape2d getOutputDims() = 0;

};


/**
 * @brief Basic Sequential Layer
 *
 */
class SequentialLayer {
  public:
    /**
     * @brief Destroy the Sequential Layer
     *
     */
    virtual ~SequentialLayer(){};

    /**
     * @brief Forward propagation virtual function
     *
     * @param input Device pointer to the input
     * @return float* Device pointer to the output
     */
    virtual float* forward(const float* input) = 0;

    /**
     * @brief Get output size
     *
     * @return int output size
     */
    virtual int getOutputSize() = 0;

    /**
     * @brief Get input size
     * 
     * @return int input size
     */
    virtual int getInputSize() = 0;
};

/**
 * @brief Base class for layers with weights and biases
 */
class WeightedLayer : public SequentialLayer {
  public:
    /**
     * @brief Destroy the ILayer object
     *
     */
    virtual ~WeightedLayer(){};

    /**
     * @brief Virtual function for forward pass
     *
     * @param input (Device) Pointer to the input
     * @return float* Device pointer to the output
     */
    virtual float* forward(const float* input) = 0;

    /**
     * @brief Virtual function for setting weights
     *
     * @param weights Pointer to the weights
     */
    virtual void setWeights(const float* weights) = 0;

    /**
     * @brief Virtual function for getting weights
     *
     */
    virtual std::vector<float> getWeights() = 0;

    /**
     * @brief Virtual function for setting biases
     *
     * @param biases Pointer to the biases
     */
    virtual void setBiases(const float* biases) = 0;

    /**
     * @brief Virtual function for getting biases
     *
     */
    virtual std::vector<float> getBiases() = 0;

  private:
    /**
     * @brief Initialize the weights
     */
    virtual void initializeWeights() = 0;

    /**
     * @brief Initialize the biases
     */
    virtual void initializeBiases() = 0;

#ifdef USE_CUDA
    /**
     * @brief Copy the weights and biases to the device
     */
    virtual void toCuda() = 0;
#endif
};

}  // namespace CUDANet::Layers

#endif  // CUDANET_I_LAYERH