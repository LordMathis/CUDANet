#include "activation.cuh"
#include <gtest/gtest.h>
#include <cuda_runtime.h>
#include <vector>


TEST(ActivationTest, SoftmaxTest1) {
    const int inputSize = 5;
    cudaError_t cudaStatus;

    CUDANet::Layers::Activation activation(
        CUDANet::Layers::ActivationType::SOFTMAX, inputSize
    );

    std::vector<float> input = {0.573f, 0.619f, 0.732f, 0.055f, 0.243f};

    float* d_input;
    cudaStatus = cudaMalloc((void**)&d_input, sizeof(float) * inputSize);
    EXPECT_EQ(cudaStatus, cudaSuccess);

    cudaStatus = cudaMemcpy(d_input, input.data(), sizeof(float) * inputSize, cudaMemcpyHostToDevice);
    EXPECT_EQ(cudaStatus, cudaSuccess);

    activation.activate(d_input);
    std::vector<float> output(5);
    cudaStatus = cudaMemcpy(
        output.data(), d_input, sizeof(float) * inputSize, cudaMemcpyDeviceToHost
    );
    EXPECT_EQ(cudaStatus, cudaSuccess);
    
    float sum = 0.0f;

    std::vector<float> expected = {0.22055f, 0.23094f, 0.25856f, 0.13139f, 0.15856f};
    for (int i = 0; i < inputSize; ++i) {
        sum += output[i];
        EXPECT_NEAR(output[i], expected[i], 1e-5f);
    }

    EXPECT_NEAR(sum, 1.0f, 1e-5f);

    cudaStatus = cudaFree(d_input);
    EXPECT_EQ(cudaStatus, cudaSuccess);
}

TEST(ActivationTest, SoftmaxTest2) {
    const int inputSize = 6;
    cudaError_t cudaStatus;

    CUDANet::Layers::Activation activation(
        CUDANet::Layers::ActivationType::SOFTMAX, inputSize
    );

    cudaStatus = cudaGetLastError();
    EXPECT_EQ(cudaStatus, cudaSuccess);

    std::vector<float> input = {22.496f, 36.9006f, 30.9904f, 28.4213f, 26.4541f, 31.7887f};

    float* d_input;
    cudaStatus = cudaMalloc((void**)&d_input, sizeof(float) * inputSize);
    EXPECT_EQ(cudaStatus, cudaSuccess);

    cudaStatus = cudaMemcpy(d_input, input.data(), sizeof(float) * inputSize, cudaMemcpyHostToDevice);
    EXPECT_EQ(cudaStatus, cudaSuccess);

    activation.activate(d_input);
    std::vector<float> output(inputSize);
    cudaStatus = cudaMemcpy(
        output.data(), d_input, sizeof(float) * inputSize, cudaMemcpyDeviceToHost
    );
    EXPECT_EQ(cudaStatus, cudaSuccess);
    
    float sum = 0.0f;

    std::vector<float> expected = {0.0f, 0.99111f, 0.00269f, 0.00021f, 3e-05f, 0.00597f};
    for (int i = 0; i < inputSize; ++i) {
        sum += output[i];
        EXPECT_NEAR(output[i], expected[i], 1e-5f);
    }
    EXPECT_NEAR(sum, 1.0f, 1e-5f);

    // Cleanup
    cudaStatus = cudaFree(d_input);
    EXPECT_EQ(cudaStatus, cudaSuccess);
}