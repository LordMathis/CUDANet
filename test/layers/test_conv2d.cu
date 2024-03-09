#include <cuda_runtime_api.h>
#include <gtest/gtest.h>

#include <iostream>

#include "conv2d.cuh"

class Conv2dTest : public::testing::Test {
    protected:
        cudaError_t    cudaStatus;
};


TEST_F(Conv2dTest, SimpleExample) {

    int inputSize = 4;
    int inputChannels = 1;
    int kernelSize = 2;
    int stride = 1;
    std::string padding = "VALID";
    int numFilters = 1;
    Activation activation = LINEAR;

    Layers::Conv2d conv2d(
        inputSize,
        inputChannels,
        kernelSize,
        stride,
        padding,
        numFilters,
        activation
    );

    int outputSize = (inputSize - kernelSize) / stride + 1;
    EXPECT_EQ(outputSize, conv2d.outputSize);

    std::vector<float> input = {
        1.0f, 2.0f, 3.0f, 4.0f,
        5.0f, 6.0f, 7.0f, 8.0f,
        9.0f, 10.0f, 11.0f, 12.0f,
        13.0f, 14.0f, 15.0f, 16.0f
    };
    std::vector<float> kernels = {
        1.0f, 2.0f, 3.0f, 4.0f,
    };

    float* d_input;
    float* d_output;

    conv2d.setKernels(kernels);

            // Allocate device memory
    cudaStatus = cudaMalloc((void**)&d_input, sizeof(float) * inputSize * inputSize * inputChannels);
    EXPECT_EQ(cudaStatus, cudaSuccess);

    cudaStatus = cudaMalloc((void**)&d_output, sizeof(float) * outputSize * outputSize * numFilters);
    EXPECT_EQ(cudaStatus, cudaSuccess);

    // // Copy input to device
    cudaStatus = cudaMemcpy(
        d_input, input.data(), sizeof(float) * input.size(), cudaMemcpyHostToDevice
    );
    EXPECT_EQ(cudaStatus, cudaSuccess);

    conv2d.forward(d_input, d_output);

    std::vector<float> expected = {
        44.0f, 54.0f, 64.0f,
        84.0f, 94.0f, 104.0f,
        124.0f, 134.0f, 144.0f
    };
    std::vector<float> output(outputSize * outputSize * numFilters);

    cudaStatus = cudaMemcpy(
        output.data(), d_output, sizeof(float) * output.size(),
        cudaMemcpyDeviceToHost
    );
    EXPECT_EQ(cudaStatus, cudaSuccess);

    for (int i = 0; i < output.size(); ++i) {
        EXPECT_FLOAT_EQ(expected[i], output[i]);
    }

}
