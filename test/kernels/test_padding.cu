#include <cuda_runtime_api.h>
#include <gtest/gtest.h>

#include <iostream>

#include "padding.cuh"

TEST(PaddingTest, SimplePaddingTest) {
    cudaError_t cudaStatus;

    int w = 2;
    int h = 3;
    int n = 2;
    int p = 1;

    float* d_input;
    float* d_padded;

    int inputSize  = w * h * n;
    int paddedSize = (w + 2 * p) * (h + 2 * p) * n;

    cudaStatus = cudaMalloc((void**)&d_input, sizeof(float) * inputSize);
    EXPECT_EQ(cudaStatus, cudaSuccess);

    cudaStatus = cudaMalloc((void**)&d_padded, sizeof(float) * paddedSize);
    EXPECT_EQ(cudaStatus, cudaSuccess);

    /*
    Matrix channel 0:
    0  1
    2  3
    4  5
    Matrix channel 1:
    6  7
    8  9
    10 11

    Represented as a vector:

    0 1 2 3 4 5 6 7 8 9 10 11
    */

    std::vector<float> input = {0.0f, 1.0f, 2.0f,  3.0f, 4.0f, 5.0f,
                                6.0f, 7.0f, 8.0f, 9.0f, 10.0f, 11.0f};

    cudaStatus = cudaMemcpy(
        d_input, input.data(), sizeof(float) * inputSize, cudaMemcpyHostToDevice
    );
    EXPECT_EQ(cudaStatus, cudaSuccess);

    int THREADS_PER_BLOCK = 64;
    int BLOCKS            = paddedSize / THREADS_PER_BLOCK + 1;

    pad_matrix_kernel<<<BLOCKS, THREADS_PER_BLOCK>>>(
        d_input, d_padded, w, h, n, p
    );
    cudaStatus = cudaDeviceSynchronize();
    EXPECT_EQ(cudaStatus, cudaSuccess);

    // clang-format off
    std::vector<float> expectedOutput = {
        // channel 0
        0.0f, 0.0f, 0.0f, 0.0f,
        0.0f, 0.0f, 1.0f, 0.0f,
        0.0f, 2.0f, 3.0f, 0.0f,
        0.0f, 4.0f, 5.0f, 0.0f,
        0.0f, 0.0f, 0.0f, 0.0f,
        // channel 1
        0.0f, 0.0f, 0.0f, 0.0f,
        0.0f, 6.0f, 7.0f, 0.0f,
        0.0f, 8.0f, 9.0f, 0.0f,
        0.0f, 10.0f, 11.0f, 0.0f,
        0.0f, 0.0f, 0.0f, 0.0f
    };
    // clang-format on

    std::vector<float> output(paddedSize);

    cudaStatus = cudaMemcpy(
        output.data(), d_padded, sizeof(float) * paddedSize,
        cudaMemcpyDeviceToHost
    );
    EXPECT_EQ(cudaStatus, cudaSuccess);

    for (int i = 0; i < paddedSize; i++) {
        EXPECT_NEAR(expectedOutput[i], output[i], 1e-5);
    }


    cudaFree(d_input);
    cudaFree(d_padded);
}
