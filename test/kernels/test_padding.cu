#include <cuda_runtime_api.h>
#include <gtest/gtest.h>

#include <iostream>

#include "padding.cuh"
#include "test_cublas_fixture.cuh"

class PaddingTest : public CublasTestFixture {
  protected:
    cudaError_t    cudaStatus;
    cublasStatus_t cublasStatus;
};

TEST_F(PaddingTest, SimplePaddingTest) {
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

    cudaStatus = cudaMalloc(
        (void**)&d_padded, sizeof(float) * paddedSize
    );
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

    Represented as column major vector:

    0 2 4 1 3 5 6 8 10 7 9 11
    */

    std::vector<float> input = {0.0f, 2.0f, 4.0f,  1.0f, 3.0f, 5.0f,
                                6.0f, 8.0f, 10.0f, 7.0f, 9.0f, 11.0f};

    cublasStatus =
        cublasSetVector(inputSize, sizeof(float), input.data(), 1, d_input, 1);
    EXPECT_EQ(cublasStatus, CUBLAS_STATUS_SUCCESS);

    pad_matrix_kernel<<<1, 1>>>(d_input, d_padded, w, h, n, p);
    cudaStatus = cudaDeviceSynchronize();
    EXPECT_EQ(cudaStatus, cudaSuccess);

    std::vector<float> expectedOutput = {
        0.0f, 0.0f, 0.0f, 0.0f,  0.0f, 0.0f, 0.0f, 2.0f, 4.0f,  0.0f,
        0.0f, 1.0f, 3.0f, 5.0f,  0.0f, 0.0f, 0.0f, 0.0f, 0.0f,  0.0f,
        0.0f, 0.0f, 0.0f, 0.0f,  0.0f, 0.0f, 6.0f, 8.0f, 10.0f, 0.0f,
        0.0f, 7.0f, 9.0f, 11.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f,  0.0f
    };

    std::vector<float> output(paddedSize);
    cublasStatus = cublasGetVector(
        paddedSize, sizeof(float), d_padded, 1, output.data(), 1
    );

    std::cout << "Actual output: " << std::endl;
    for (int i = 0; i < paddedSize; i++) {
        std::cout << output[i] << " ";
    }
    std::cout << std::endl;

    for (int i = 0; i < paddedSize; i++) {
        EXPECT_NEAR(expectedOutput[i], output[i], 1e-5);
    }
}