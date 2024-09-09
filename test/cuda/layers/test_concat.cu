#include "concat.hpp"
#include <gtest/gtest.h>
#include <cuda_runtime.h>
#include <vector>

TEST(ConcatLayerTest, Init) {

    cudaError_t cudaStatus;

    std::vector<float> inputA = {0.573f, 0.619f, 0.732f, 0.055f, 0.243f};
    std::vector<float> inputB = {0.123f, 0.321f, 0.456f, 0.789f, 0.654f, 0.123f};

    CUDANet::Layers::Concat concat(5, 6);   

    float* d_inputA;
    float* d_inputB;

    cudaStatus = cudaMalloc((void**)&d_inputA, sizeof(float) * 5);
    EXPECT_EQ(cudaStatus, cudaSuccess);

    cudaStatus = cudaMalloc((void**)&d_inputB, sizeof(float) * 6);
    EXPECT_EQ(cudaStatus, cudaSuccess);

    cudaStatus = cudaMemcpy(
        d_inputA, inputA.data(), sizeof(float) * 5, cudaMemcpyHostToDevice
    );
    EXPECT_EQ(cudaStatus, cudaSuccess);

    cudaStatus = cudaMemcpy(
        d_inputB, inputB.data(), sizeof(float) * 6, cudaMemcpyHostToDevice
    );
    EXPECT_EQ(cudaStatus, cudaSuccess);

    float* d_output = concat.forward(d_inputA, d_inputB);

    std::vector<float> output(11);
    cudaStatus = cudaMemcpy(
        output.data(), d_output, sizeof(float) * 11, cudaMemcpyDeviceToHost
    );
    EXPECT_EQ(cudaStatus, cudaSuccess);

    for (int i = 0; i < 5; ++i) {
        EXPECT_EQ(output[i], inputA[i]);
    }

    for (int i = 0; i < 6; ++i) {
        EXPECT_EQ(output[i + 5], inputB[i]);
    }

    cudaStatus = cudaFree(d_inputA);
    EXPECT_EQ(cudaStatus, cudaSuccess);
    cudaStatus = cudaFree(d_inputB);
    EXPECT_EQ(cudaStatus, cudaSuccess);
}