#include "gtest/gtest.h"
#include <cuda_runtime_api.h>
#include <driver_types.h>
#include "dense.h"
#include "test_cublas_fixture.h"

class DenseLayerTest : public CublasTestFixture {
protected:
};


TEST_F(DenseLayerTest, Forward) {

    cudaError_t cudaStatus;
    cublasStatus_t cublasStatus;

    int inputSize = 3;
    int outputSize = 3;

    Layers::Dense denseLayer(inputSize, outputSize, cublasHandle);

    // Initialize a weight matrix
    std::vector<std::vector<float>> weights(inputSize, std::vector<float>(outputSize, 0.0f));
    for (int i = 0; i < inputSize; ++i) {
        for (int j = 0; j < outputSize; ++j) {
            if (i == j) {
                weights[i][j] = 1.0f;
            }
        }
    }

    // Set the weights
    denseLayer.setWeights(weights);

    // Initialize and set a bias vector
    std::vector<float> biases(outputSize, 1.0f);
    denseLayer.setBiases(biases);

    std::vector<float> input = {1.0f, 2.0f, 3.0f};
    std::vector<float> output(outputSize);

    float* d_input;
    float* d_output;

    cudaStatus =cudaMalloc((void**)&d_input, sizeof(float) * input.size());
    EXPECT_EQ(cudaStatus, cudaSuccess);

    cudaStatus = cudaMalloc((void**)&d_output, sizeof(float) * outputSize);
    EXPECT_EQ(cudaStatus, cudaSuccess);

    cublasStatus =cublasSetVector(input.size(), sizeof(float), input.data(), 1, d_input, 1);
    EXPECT_EQ(cublasStatus, CUBLAS_STATUS_SUCCESS);

    // Perform forward pass
    denseLayer.forward(d_input, d_output);

    cublasStatus = cublasGetVector(outputSize, sizeof(float), d_output, 1, output.data(), 1);
    EXPECT_EQ(cublasStatus, CUBLAS_STATUS_SUCCESS);

    // Check if the output is a zero vector
    EXPECT_FLOAT_EQ(output[0], 2.0f);
    EXPECT_FLOAT_EQ(output[1], 3.0f);
    EXPECT_FLOAT_EQ(output[2], 4.0f);

    cudaFree(d_input);
    cudaFree(d_output);
}
