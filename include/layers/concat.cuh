#ifndef CUDANET_CONCAT_LAYER_H
#define CUDANET_CONCAT_LAYER_H

namespace CUDANet::Layers {

/**
 * @brief Concatenate layers
 *
 */
class Concat {
  public:
    /**
     * @brief Create a new Concat layer
     * 
     * @param inputASize Size of the first input
     * @param inputBSize Size of the second input
     */
    Concat(const unsigned int inputASize, const unsigned int inputBSize);

    /**
     * @brief Destroy the Concat layer
     * 
     */
    ~Concat();

    /**
     * @brief Concatenates the two inputs
     *
     * @param d_input_A Device pointer to the first input
     * @param d_input_B Device pointer to the second input
     * 
     * @return Device pointer to the output
     */
    float* forward(const float* d_input_A, const float* d_input_B);

  private:
    unsigned int inputASize;
    unsigned int inputBSize;

    float* d_output;
};

}  // namespace CUDANet::Layers

#endif  // CUDANET_CONCAT_LAYER_H
