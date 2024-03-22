#include <cuda_runtime_api.h>
#include <gtest/gtest.h>

#include <iostream>

#include "activation_functions.cuh"

TEST(ActivationFunctionsTest, SigmoidSanityCheck) {
    cudaError_t cudaStatus;

    float input[3] = {-100.0f, 0.0f, 100.0f};

    std::vector<float> expected_output = {0.0f, 0.5f, 1.0f};

    float* d_input;
    float* d_output;

    cudaStatus = cudaMalloc((void**)&d_input, sizeof(float) * 3);
    EXPECT_EQ(cudaStatus, cudaSuccess);

    cudaStatus = cudaMalloc((void**)&d_output, sizeof(float) * 3);
    EXPECT_EQ(cudaStatus, cudaSuccess);

    cudaStatus =
        cudaMemcpy(d_input, input, sizeof(float) * 3, cudaMemcpyHostToDevice);
    EXPECT_EQ(cudaStatus, cudaSuccess);

    CUDANet::Kernels::sigmoid<<<1, 3>>>(d_input, d_output, 3);
    cudaStatus = cudaDeviceSynchronize();
    EXPECT_EQ(cudaStatus, cudaSuccess);

    std::vector<float> output(3);

    cudaStatus = cudaMemcpy(
        output.data(), d_output, sizeof(float) * 3, cudaMemcpyDeviceToHost
    );
    EXPECT_EQ(cudaStatus, cudaSuccess);

    for (int i = 0; i < 3; i++) {
        EXPECT_NEAR(expected_output[i], output[i], 1e-5);
    }

    cudaFree(d_input);
    cudaFree(d_output);
}

TEST(ActivationFunctionsTest, SoftmaxExpTest) {
    cudaError_t cudaStatus;

    float input[6] = {22.496f,  36.9006f, 30.9904f,
                      28.4213f, 26.4541f, 31.7887f};

    std::vector<float> expected = {5886928896.0f,     1.06102872080384e+16f,
                                   28771323215872.0f, 2204012904448.0f,
                                   308226162688.0f,   63922983927808.0f};

    float* d_input;
    float* d_output;

    cudaStatus = cudaMalloc((void**)&d_input, sizeof(float) * 6);
    EXPECT_EQ(cudaStatus, cudaSuccess);

    cudaStatus = cudaMalloc((void**)&d_output, sizeof(float) * 6);
    EXPECT_EQ(cudaStatus, cudaSuccess);

    cudaStatus =
        cudaMemcpy(d_input, input, sizeof(float) * 6, cudaMemcpyHostToDevice);
    EXPECT_EQ(cudaStatus, cudaSuccess);

    CUDANet::Kernels::softmax_exp<<<1, 6>>>(d_input, d_output, 6);
    cudaStatus = cudaDeviceSynchronize();
    EXPECT_EQ(cudaStatus, cudaSuccess);

    std::vector<float> output(6);

    cudaStatus = cudaMemcpy(
        output.data(), d_output, sizeof(float) * 6, cudaMemcpyDeviceToHost
    );
    EXPECT_EQ(cudaStatus, cudaSuccess);

    for (int i = 0; i < 6; i++) {
        EXPECT_NEAR(expected[i], output[i], 1e7);
    }

    cudaFree(d_input);
    cudaFree(d_output);
}

TEST(ActivationFunctionsTest, SoftmaxSumTest) {
    cudaError_t cudaStatus;

    std::vector<float> input = {5886928896.0f,     1.06102872080384e+16f,
                                28771323215872.0f, 2204012904448.0f,
                                308226162688.0f,   63922983927808.0f};

    float* d_input;

    cudaStatus = cudaMalloc((void**)&d_input, sizeof(float) * 6);
    EXPECT_EQ(cudaStatus, cudaSuccess);
}