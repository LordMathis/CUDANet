#ifndef CUDANET_ACTIVATION_H
#define CUDANET_ACTIVATION_H

namespace CUDANet::Layers {

/**
 * @brief Activation functions
 * 
 * SIGMOID: Sigmoid
 * RELU: Rectified Linear Unit
 * SOFTMAX: Softmax
 *
 */
enum ActivationType { SIGMOID, RELU, SOFTMAX, NONE };

class Activation {
  public:

    Activation() = default;

    /**
     * @brief Construct a new Activation object
     * 
     * @param activation Type of activation
     * @param length     Length of the input
     */
    Activation(ActivationType activation, const unsigned int length);

    /**
     * @brief Destroy the Activation object
     * 
     */
    ~Activation();

    /**
     * @brief Run the activation function on the input
     * 
     * @param d_input Pointer to the input vector on the device
     */
    void activate(float* d_input);


  private:
    ActivationType activationType;
    unsigned int length;
    unsigned int gridSize;

    float* d_softmax_sum;

};


}  // namespace CUDANet::Layers

#endif  // CUDANET_ACTIVATION_H