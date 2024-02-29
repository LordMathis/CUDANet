#include <cuda_runtime_api.h>
#include <gtest/gtest.h>

#include <iostream>

#include "activations.cuh"
#include "test_cublas_fixture.cuh"

class ActivationsTest : public CublasTestFixture {
  protected:
    cudaError_t    cudaStatus;
    cublasStatus_t cublasStatus;
};

TEST_F(ActivationsTest, SigmoidSanityCheck) {
    float input[3] = {-100.0f, 0.0f, 100.0f};

    std::vector<float> expected_output = {0.0f, 0.5f, 1.0f};

    float* d_input;
    float* d_output;

    cudaStatus = cudaMalloc((void**)&d_input, sizeof(float) * 3);
    EXPECT_EQ(cudaStatus, cudaSuccess);

    cudaStatus = cudaMalloc((void**)&d_output, sizeof(float) * 3);
    EXPECT_EQ(cudaStatus, cudaSuccess);

    cublasStatus = cublasSetVector(3, sizeof(float), input, 1, d_input, 1);
    EXPECT_EQ(cublasStatus, CUBLAS_STATUS_SUCCESS);

    sigmoid_kernel<<<1, 3>>>(d_input, d_output, 3);
    cudaStatus = cudaDeviceSynchronize();
    EXPECT_EQ(cudaStatus, cudaSuccess);

    std::vector<float> output(3);

    cublasStatus =
        cublasGetVector(3, sizeof(float), d_output, 1, output.data(), 1);
    EXPECT_EQ(cublasStatus, CUBLAS_STATUS_SUCCESS);

    for (int i = 0; i < 3; i++) {
        EXPECT_NEAR(expected_output[i], output[i], 1e-5);
    }

    cudaFree(d_input);
    cudaFree(d_output);
}