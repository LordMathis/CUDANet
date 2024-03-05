#include <cuda_runtime_api.h>
#include <gtest/gtest.h>

#include <iostream>

#include "activations.cuh"

TEST(ActivationsTest, SigmoidSanityCheck) {

    cudaError_t cudaStatus;

    float input[3] = {-100.0f, 0.0f, 100.0f};

    std::vector<float> expected_output = {0.0f, 0.5f, 1.0f};

    float* d_input;
    float* d_output;

    cudaStatus = cudaMalloc((void**)&d_input, sizeof(float) * 3);
    EXPECT_EQ(cudaStatus, cudaSuccess);

    cudaStatus = cudaMalloc((void**)&d_output, sizeof(float) * 3);
    EXPECT_EQ(cudaStatus, cudaSuccess);

    cudaStatus = cudaMemcpy(d_input, input, sizeof(float) * 3, cudaMemcpyHostToDevice);
    EXPECT_EQ(cudaStatus, cudaSuccess);

    sigmoid_kernel<<<1, 3>>>(d_input, d_output, 3);
    cudaStatus = cudaDeviceSynchronize();
    EXPECT_EQ(cudaStatus, cudaSuccess);

    std::vector<float> output(3);

    cudaStatus = cudaMemcpy(output.data(), d_output, sizeof(float) * 3, cudaMemcpyDeviceToHost);
    EXPECT_EQ(cudaStatus, cudaSuccess);

    for (int i = 0; i < 3; i++) {
        EXPECT_NEAR(expected_output[i], output[i], 1e-5);
    }

    cudaFree(d_input);
    cudaFree(d_output);
}