#include <cuda_runtime.h>
#include <gtest/gtest.h>

#include <vector>

#include "avg_pooling.cuh"

TEST(AvgPoolingLayerTest, AvgPoolForwardTest) {
    int inputSize   = 4;
    int nChannels   = 2;
    int poolingSize = 2;
    int stride      = 2;

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

    CUDANet::Layers::AvgPooling2D avgPoolingLayer(
        inputSize, nChannels, poolingSize, stride,
        CUDANet::Layers::ActivationType::NONE
    );

    float *d_input;

    cudaStatus = cudaMalloc(
        (void **)&d_input, sizeof(float) * inputSize * inputSize * nChannels
    );
    EXPECT_EQ(cudaStatus, cudaSuccess);

    cudaStatus = cudaMemcpy(
        d_input, input.data(),
        sizeof(float) * inputSize * inputSize * nChannels,
        cudaMemcpyHostToDevice
    );
    EXPECT_EQ(cudaStatus, cudaSuccess);

    float *d_output = avgPoolingLayer.forward(d_input);

    int outputSize = avgPoolingLayer.getOutputSize();

    std::vector<float> output(outputSize * outputSize * nChannels);
    cudaStatus = cudaMemcpy(
        output.data(), d_output,
        sizeof(float) * outputSize * outputSize * nChannels,
        cudaMemcpyDeviceToHost
    );
    EXPECT_EQ(cudaStatus, cudaSuccess);

    std::vector<float> expected = {0.43775f, 0.49475f, 0.48975f, 0.339f, 0.45675f, 0.303f, 0.56975f, 0.57025f};

    for (int i = 0; i < output.size(); ++i) {
        EXPECT_NEAR(expected[i], output[i], 1e-4);
    }

    cudaFree(d_input);
    cudaFree(d_output);
}
