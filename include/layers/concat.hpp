#ifndef CUDANET_CONCAT_LAYER_H
#define CUDANET_CONCAT_LAYER_H

#include "layer.hpp"

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
    Concat(const int inputASize, const int inputBSize);

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

    int getOutputSize();

  private:
    int inputASize;
    int inputBSize;

    float* forwardCPU(const float* input_A, const float* input_B);

#ifdef USE_CUDA
    float* d_output;
    float* forwardCUDA(const float* d_input_A, const float* d_input_B);

    void initCUDA();
    void delCUDA();
#endif
};

}  // namespace CUDANet::Layers

#endif  // CUDANET_CONCAT_LAYER_H
