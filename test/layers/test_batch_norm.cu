#include <cuda_runtime.h>
#include <gtest/gtest.h>

#include <vector>

#include "activation.cuh"
#include "batch_norm.cuh"

TEST(BatchNormLayerTest, BatchNormSmallForwardTest) {
    dim2d inputSize = {4, 4};
    int   nChannels = 2;

    cudaError_t cudaStatus;

    CUDANet::Layers::BatchNorm2D batchNorm(
        inputSize, nChannels, 1e-5f, CUDANet::Layers::ActivationType::NONE
    );

    std::vector<float> weights = {0.63508f, 0.64903f};
    std::vector<float> biases  = {0.25079f, 0.66841f};

    batchNorm.setWeights(weights.data());
    batchNorm.setBiases(biases.data());

    cudaStatus = cudaGetLastError();
    EXPECT_EQ(cudaStatus, cudaSuccess);

    // clang-format off
    std::vector<float> input = {
        // Channel 0
        0.38899f, 0.80478f, 0.48836f, 0.97381f,
        0.57508f, 0.60835f, 0.65467f, 0.00168f,
        0.65869f, 0.74235f, 0.17928f, 0.70349f,
        0.15524f, 0.38664f, 0.23411f, 0.7137f,
        // Channel 1
        0.32473f, 0.15698f, 0.314f, 0.60888f,
        0.80268f, 0.99766f, 0.93694f, 0.89237f,
        0.13449f, 0.27367f, 0.53036f, 0.18962f,
        0.57672f, 0.48364f, 0.10863f, 0.0571f
    };
    // clang-format on

    std::vector<float> output(input.size());

    float* d_input;
    cudaStatus = cudaMalloc((void**)&d_input, sizeof(float) * input.size());
    EXPECT_EQ(cudaStatus, cudaSuccess);

    cudaStatus = cudaMemcpy(
        d_input, input.data(), sizeof(float) * input.size(),
        cudaMemcpyHostToDevice
    );
    EXPECT_EQ(cudaStatus, cudaSuccess);

    float* d_output = batchNorm.forward(d_input);

    cudaStatus = cudaMemcpy(
        output.data(), d_output, sizeof(float) * output.size(),
        cudaMemcpyDeviceToHost
    );
    EXPECT_EQ(cudaStatus, cudaSuccess);

    std::vector<float> expected = {-0.06007f, 0.951f,    0.18157f,  1.36202f,
                                   0.39244f,  0.47335f,  0.58598f,  -1.00188f,
                                   0.59576f,  0.79919f,  -0.57001f, 0.70469f,
                                   -0.62847f, -0.06578f, -0.43668f, 0.72952f,
                                   0.37726f,  0.02088f,  0.35446f,  0.98092f,
                                   1.39264f,  1.80686f,  1.67786f,  1.58318f,
                                   -0.0269f,  0.26878f,  0.81411f,  0.09022f,
                                   0.9126f,   0.71485f,  -0.08184f, -0.19131f};

    // std::cout << "BatchNorm2D: " << std::endl;
    for (int i = 0; i < output.size(); i++) {
        EXPECT_NEAR(output[i], expected[i], 1e-5);
        // std::cout << output[i] << " ";
    }
    // std::cout << std::endl;
    cudaFree(d_input);
}