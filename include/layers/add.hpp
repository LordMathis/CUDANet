#ifndef CUDANET_ADD_LAYER_H
#define CUDANET_ADD_LAYER_H

namespace CUDANet::Layers {

class Add {
  public:
    /**
     * @brief Create a new Add layer
     *
     * @param inputSize Size of the input arrays
     */
    Add(int inputSize);

    /**
     * @brief Destroy the Add layer
     *
     */
    ~Add();

    /**
     * @brief Adds first input to second input
     *
     * @param d_inputA Device pointer to the first input
     * @param d_inputB Device pointer to the second input
     *
     */
    float* forward(const float* inputA, const float* inputB);

  private:
    int inputSize;

    float* output;

    float* forwardCPU(const float* inputA, const float* inputB);

#ifdef USE_CUDA
    float* d_output;
    int gridSize;

    float* forwardCUDA(const float* d_inputA, const float* d_inputB);
    void initCUDA();
    void delCUDA();
#endif
};

}  // namespace CUDANet::Layers

#endif  // CUDANET_ADD_LAYER_H