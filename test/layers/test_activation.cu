#include "activation.cuh"
#include <gtest/gtest.h>
#include <cuda_runtime.h>
#include <vector>

TEST(ActivationTest, SoftmaxTest1) {
    CUDANet::Layers::Activation activation(
        CUDANet::Layers::ActivationType::SOFTMAX, 5
    );

    std::vector<float> input = {0.573f, 0.619f, 0.732f, 0.055f, 0.243f};

    float* d_input;
    cudaMalloc((void**)&d_input, sizeof(float) * 5);
    cudaMemcpy(d_input, input.data(), sizeof(float) * 5, cudaMemcpyHostToDevice);

    activation.activate(d_input);
    std::vector<float> output(5);
    cudaMemcpy(
        output.data(), d_input, sizeof(float) * 5, cudaMemcpyDeviceToHost
    );
    
    float sum = 0.0f;

    std::vector<float> expected = {0.22055f, 0.23094f, 0.25856f, 0.13139f, 0.15856f};
    for (int i = 0; i < 5; ++i) {
        sum += output[i];
        EXPECT_NEAR(output[i], expected[i], 1e-5f);
    }

    EXPECT_NEAR(sum, 1.0f, 1e-5f);

    cudaFree(d_input);
}

TEST(ActivationTest, SoftmaxTest2) {
    CUDANet::Layers::Activation activation(
        CUDANet::Layers::ActivationType::SOFTMAX, 6
    );

    std::vector<float> input = {22.496f, 36.9006f, 30.9904f, 28.4213f, 26.4541f, 31.7887f};

    float* d_input;
    cudaMalloc((void**)&d_input, sizeof(float) * 6);
    cudaMemcpy(d_input, input.data(), sizeof(float) * 6, cudaMemcpyHostToDevice);

    activation.activate(d_input);
    std::vector<float> output(6);
    cudaMemcpy(
        output.data(), d_input, sizeof(float) * 6, cudaMemcpyDeviceToHost
    );
    
    float sum = 0.0f;

    std::vector<float> expected = {0.0f, 0.99111f, 0.00269f, 0.00021f, 3e-05f, 0.00597f};
    for (int i = 0; i < 5; ++i) {
        sum += output[i];
        EXPECT_NEAR(output[i], expected[i], 1e-5f);
    }

    std::cout << sum << std::endl;

    EXPECT_NEAR(sum, 1.0f, 1e-5f);

    cudaFree(d_input);
}