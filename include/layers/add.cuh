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
    void forward(const float* d_inputA, const float* d_inputB);

  private:
    int inputSize;
    int gridSize;

    float* d_output;
};

}  // namespace CUDANet::Layers

#endif  // CUDANET_ADD_LAYER_H