#include <gtest/gtest.h>

#include "vector.cuh"

TEST(VectorTest, TestVectorMean) {

    cudaError_t cudaStatus;
    float length = 10;

    std::vector<float> input = {0.44371f, 0.20253f, 0.73232f, 0.40378f, 0.93348f, 0.72756f, 0.63388f, 0.5251f, 0.23973f, 0.52233f};

    float* d_vec = nullptr;
    cudaStatus = cudaMalloc((void **)&d_vec, sizeof(float) * length);
    EXPECT_EQ(cudaStatus, cudaSuccess);

    float* d_mean = nullptr;
    cudaStatus = cudaMalloc((void **)&d_mean, sizeof(float) * length);
    EXPECT_EQ(cudaStatus, cudaSuccess);
    
    float* d_length = nullptr;
    cudaStatus = cudaMalloc((void **)&d_length, sizeof(float));
    EXPECT_EQ(cudaStatus, cudaSuccess);

    cudaStatus = cudaMemcpy(d_vec, input.data(), sizeof(float) * length, cudaMemcpyHostToDevice);
    EXPECT_EQ(cudaStatus, cudaSuccess);

    cudaStatus = cudaMemcpy(d_length, &length, sizeof(float), cudaMemcpyHostToDevice);
    EXPECT_EQ(cudaStatus, cudaSuccess);

    CUDANet::Utils::mean(d_vec, d_mean, d_length, length);

    std::vector<float> mean(length);
    cudaStatus = cudaMemcpy(mean.data(), d_mean, sizeof(float) * length, cudaMemcpyDeviceToHost);
    EXPECT_EQ(cudaStatus, cudaSuccess);

    float expected_mean = 0.5364f;
    EXPECT_NEAR(mean[0], expected_mean, 1e-4);

}