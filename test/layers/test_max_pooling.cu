#include <cuda_runtime.h>
#include <gtest/gtest.h>

#include <vector>

#include "max_pooling.cuh"

TEST(MaxPoolingLayerTest, MaxPoolForwardTest) {
    dim2d inputSize   = {4, 4};
    int   nChannels   = 2;
    dim2d poolingSize = {2, 2};
    dim2d stride      = {2, 2};

    cudaError_t cudaStatus;

    std::vector<float> input = {
        // clang-format off
        // Channel 0
        0.573f, 0.619f, 0.732f, 0.055f,
        0.243f, 0.316f, 0.573f, 0.619f,
        0.712f, 0.055f, 0.243f, 0.316f,
        0.573f, 0.619f, 0.742f, 0.055f,
        // Channel 1
        0.473f, 0.919f, 0.107f, 0.073f,
        0.073f, 0.362f, 0.973f, 0.059f,
        0.473f, 0.455f, 0.283f, 0.416f,
        0.532f, 0.819f, 0.732f, 0.850f
        // clang-format on
    };

    CUDANet::Layers::MaxPooling2d maxPoolingLayer(
        inputSize, nChannels, poolingSize, stride,
        CUDANet::Layers::ActivationType::NONE
    );

    float *d_input;

    cudaStatus = cudaMalloc(
        (void **)&d_input, sizeof(float) * inputSize.first * inputSize.second * nChannels
    );
    EXPECT_EQ(cudaStatus, cudaSuccess);

    cudaStatus = cudaMemcpy(
        d_input, input.data(),
        sizeof(float) * inputSize.first * inputSize.second * nChannels,
        cudaMemcpyHostToDevice
    );
    EXPECT_EQ(cudaStatus, cudaSuccess);

    float *d_output = maxPoolingLayer.forward(d_input);

    int outputSize = maxPoolingLayer.getOutputSize();

    std::vector<float> output(outputSize);
    cudaStatus = cudaMemcpy(
        output.data(), d_output, sizeof(float) * outputSize,
        cudaMemcpyDeviceToHost
    );
    EXPECT_EQ(cudaStatus, cudaSuccess);

    std::vector<float> expected = {0.619f, 0.732f, 0.712f, 0.742f,
                                   0.919f, 0.973f, 0.819f, 0.85f};

    for (int i = 0; i < output.size(); ++i) {
        EXPECT_FLOAT_EQ(expected[i], output[i]);
    }

    cudaStatus = cudaFree(d_input);
    EXPECT_EQ(cudaStatus, cudaSuccess);
}
