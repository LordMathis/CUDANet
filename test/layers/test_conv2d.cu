#include <cuda_runtime_api.h>
#include <gtest/gtest.h>

#include <iostream>

#include "conv2d.cuh"

class Conv2dTest : public ::testing::Test {
  protected:
    CUDANet::Layers::Conv2d commonTestSetup(
        int                         inputSize,
        int                         inputChannels,
        int                         kernelSize,
        int                         stride,
        CUDANet::Layers::Padding    padding,
        int                         numFilters,
        CUDANet::Layers::Activation activation,
        std::vector<float>&         input,
        float*                      kernels,
        float*&                     d_input
    ) {
        // Create Conv2d layer
        CUDANet::Layers::Conv2d conv2d(
            inputSize, inputChannels, kernelSize, stride, padding, numFilters,
            activation
        );

        conv2d.setWeights(kernels);

        // Allocate device memory
        cudaStatus = cudaMalloc(
            (void**)&d_input,
            sizeof(float) * inputSize * inputSize * inputChannels
        );
        EXPECT_EQ(cudaStatus, cudaSuccess);

        // // Copy input to device
        cudaStatus = cudaMemcpy(
            d_input, input.data(), sizeof(float) * input.size(),
            cudaMemcpyHostToDevice
        );
        EXPECT_EQ(cudaStatus, cudaSuccess);

        return conv2d;
    }

    void commonTestTeardown(float* d_input) {
        // Free device memory
        cudaFree(d_input);
    }

    cudaError_t cudaStatus;
};

TEST_F(Conv2dTest, SimpleTest) {
    int                         inputSize     = 4;
    int                         inputChannels = 1;
    int                         kernelSize    = 2;
    int                         stride        = 1;
    CUDANet::Layers::Padding    padding       = CUDANet::Layers::Padding::VALID;
    int                         numFilters    = 1;
    CUDANet::Layers::Activation activation = CUDANet::Layers::Activation::NONE;

    std::vector<float> input   = {1.0f,  2.0f,  3.0f,  4.0f,  5.0f,  6.0f,
                                  7.0f,  8.0f,  9.0f,  10.0f, 11.0f, 12.0f,
                                  13.0f, 14.0f, 15.0f, 16.0f};
    std::vector<float> kernels = {
        1.0f,
        2.0f,
        3.0f,
        4.0f,
    };

    float* d_input;
    float* d_output;

    CUDANet::Layers::Conv2d conv2d = commonTestSetup(
        inputSize, inputChannels, kernelSize, stride, padding, numFilters,
        activation, input, kernels.data(), d_input
    );

    int outputSize = (inputSize - kernelSize) / stride + 1;
    EXPECT_EQ(outputSize, conv2d.getOutputSize());

    d_output = conv2d.forward(d_input);

    std::vector<float> expected = {44.0f,  54.0f,  64.0f,  84.0f, 94.0f,
                                   104.0f, 124.0f, 134.0f, 144.0f};
    std::vector<float> output(outputSize * outputSize * numFilters);

    cudaStatus = cudaMemcpy(
        output.data(), d_output, sizeof(float) * output.size(),
        cudaMemcpyDeviceToHost
    );
    EXPECT_EQ(cudaStatus, cudaSuccess);

    for (int i = 0; i < output.size(); ++i) {
        EXPECT_FLOAT_EQ(expected[i], output[i]);
    }

    commonTestTeardown(d_input);
}

TEST_F(Conv2dTest, PaddedTest) {
    int                         inputSize     = 5;
    int                         inputChannels = 3;
    int                         kernelSize    = 3;
    int                         stride        = 1;
    CUDANet::Layers::Padding    padding       = CUDANet::Layers::Padding::SAME;
    int                         numFilters    = 2;
    CUDANet::Layers::Activation activation = CUDANet::Layers::Activation::NONE;

    // clang-format off
    std::vector<float> input = {
        // Channel 1
        0.823f, 0.217f, 0.435f, 0.981f, 0.742f,
        0.109f, 0.518f, 0.374f, 0.681f, 0.147f,
        0.956f, 0.729f, 0.654f, 0.087f, 0.392f,
        0.784f, 0.921f, 0.543f, 0.231f, 0.816f,
        0.472f, 0.614f, 0.102f, 0.987f, 0.398f,
        // Channel 2
        0.051f, 0.756f, 0.841f, 0.293f, 0.128f,
        0.417f, 0.632f, 0.095f, 0.184f, 0.529f,
        0.871f, 0.958f, 0.213f, 0.347f, 0.725f,
        0.461f, 0.012f, 0.278f, 0.195f, 0.649f,
        0.853f, 0.707f, 0.988f, 0.988f, 0.322f,
        // Channel 3
        0.345f, 0.123f, 0.789f, 0.123f, 0.456f,
        0.456f, 0.789f, 0.123f, 0.345f, 0.123f,
        0.789f, 0.123f, 0.345f, 0.123f, 0.456f,
        0.123f, 0.345f, 0.123f, 0.789f, 0.123f,
        0.345f, 0.123f, 0.789f, 0.123f, 0.456f
    };

    std::vector<float> kernels = {
        // Filter 1, Channel 1
        0.128f, 0.754f, 0.987f,
        0.321f, 0.412f, 0.635f,
        0.298f, 0.017f, 0.845f,
        // Filter 1, Channel 2
        0.514f, 0.729f, 0.952f,
        0.684f, 0.378f, 0.159f,
        0.823f, 0.547f, 0.216f,
        // Filter 1, Channel 3
        0.983f, 0.231f, 0.456f,
        0.178f,  0.654f, 0.821f,
        0.345f, 0.987f, 0.123f,
        // Filter 2, Channel 1
        0.789f, 0.543f, 0.210f,
        0.012f, 0.371f, 0.638f,
        0.456f, 0.198f, 0.907f,
        // Filter 2, Channel 2
        0.101f, 0.432f, 0.759f,
        0.234f, 0.567f, 0.890f,
        0.543f, 0.876f, 0.219f,
        // Filter 2, Channel 3
        0.345f, 0.678f, 0.011f,
        0.678f, 0.011f, 0.345f,
        0.011f, 0.345f, 0.678f
    };
    // clang-format on

    float* d_input;
    float* d_output;

    CUDANet::Layers::Conv2d conv2d = commonTestSetup(
        inputSize, inputChannels, kernelSize, stride, padding, numFilters,
        activation, input, kernels.data(), d_input
    );

    EXPECT_EQ(inputSize, conv2d.getOutputSize());

    d_output = conv2d.forward(d_input);

    std::vector<float> output(
        conv2d.getOutputSize() * conv2d.getOutputSize() * numFilters
    );
    cudaMemcpy(
        output.data(), d_output,
        sizeof(float) * conv2d.getOutputSize() * conv2d.getOutputSize() *
            numFilters,
        cudaMemcpyDeviceToHost
    );

    // Generated by tools/generate_conv2d_test.py
    std::vector<float> expected = {
        // Channel 1
        2.29426f, 3.89173f, 4.17634f, 3.25501f, 2.07618f, 5.41483f, 7.09971f,
        6.39811f, 5.71432f, 3.10928f, 5.12973f, 6.29638f, 5.26962f, 5.21997f,
        3.05852f, 6.17517f, 7.19311f, 6.69771f, 6.2142f, 4.03242f, 3.3792f,
        4.36444f, 4.396f, 4.69905f, 3.62061f,
        // Channel 2
        2.87914f, 3.71743f, 3.51854f, 2.98413f, 1.46579f, 4.94951f, 6.18983f,
        4.98187f, 4.38372f, 3.35386f, 5.0364f, 5.3756f, 4.05993f, 4.89299f,
        2.78625f, 5.33763f, 5.80899f, 5.89785f, 5.51095f, 3.74287f, 2.64053f,
        4.05895f, 3.96482f, 4.30177f, 1.94269f
    };
    for (int i = 0; i < output.size(); i++) {
        EXPECT_NEAR(output[i], expected[i], 0.0001f);
    }

    commonTestTeardown(d_input);
}

TEST_F(Conv2dTest, StridedPaddedConvolution) {
    int                         inputSize     = 5;
    int                         inputChannels = 2;
    int                         kernelSize    = 3;
    int                         stride        = 2;
    int                         numFilters    = 2;
    CUDANet::Layers::Padding    padding       = CUDANet::Layers::Padding::SAME;
    CUDANet::Layers::Activation activation = CUDANet::Layers::Activation::RELU;

    // clang-format off
    std::vector<float> input = {
        // Channel 1
        0.946f, 0.879f, 0.382f, 0.542f, 0.453f,
        0.128f, 0.860f, 0.778f, 0.049f, 0.974f,
        0.400f, 0.874f, 0.161f, 0.271f, 0.580f,
        0.373f, 0.078f, 0.366f, 0.396f, 0.181f,
        0.246f, 0.112f, 0.179f, 0.979f, 0.026f,
        // Channel 2
        0.598f, 0.458f, 0.776f, 0.213f, 0.199f,
        0.853f, 0.170f, 0.609f, 0.269f, 0.777f,
        0.776f, 0.694f, 0.430f, 0.238f, 0.968f,
        0.473f, 0.303f, 0.084f, 0.785f, 0.444f,
        0.464f, 0.413f, 0.779f, 0.298f, 0.783f
    };
    std::vector<float> kernels = {
        // Filter 1, Channel 1
        0.744f, 0.745f, 0.641f,
        0.164f, 0.157f, 0.127f,
        0.732f, 0.761f, 0.601f,
        // Filter 1, Channel 2
        0.475f, 0.335f, 0.499f,
        0.833f, 0.793f, 0.176f,
        0.822f, 0.163f, 0.175f,
        // Filter 2, Channel 1
        0.918f, 0.340f, 0.497f,
        0.233f, 0.218f, 0.847f,
        0.931f, 0.926f, 0.199f,
        // Filter 2, Channel 2
        0.510f, 0.432f, 0.567f,
        0.236f, 0.397f, 0.739f,
        0.939f, 0.891f, 0.006f
    };
    // clang-format on

    float* d_input;
    float* d_output;

    CUDANet::Layers::Conv2d conv2d = commonTestSetup(
        inputSize, inputChannels, kernelSize, stride, padding, numFilters,
        activation, input, kernels.data(), d_input
    );

    EXPECT_EQ(inputSize, conv2d.getOutputSize());

    d_output = conv2d.forward(d_input);

    std::vector<float> output(
        conv2d.getOutputSize() * conv2d.getOutputSize() * numFilters
    );
    cudaMemcpy(
        output.data(), d_output,
        sizeof(float) * conv2d.getOutputSize() * conv2d.getOutputSize() *
            numFilters,
        cudaMemcpyDeviceToHost
    );

    // Generated by tools/generate_conv2d_test.py
    std::vector<float> expected = {
        // Channel 1
        0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 1.59803f, 2.84444f, 1.6201f, 0.0f,
        0.0f, 2.38937f, 3.80762f, 3.39679f, 0.0f, 0.0f, 1.13102f, 2.33335f,
        1.98488f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f,
        // Channel 2
        0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 2.57732f, 3.55543f, 2.24675f, 0.0f,
        0.0f, 3.36842f, 3.41373f, 3.14804f, 0.0f, 0.0f, 1.17963f, 2.55005f,
        1.63218f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f
    };

    for (int i = 0; i < output.size(); i++) {
        EXPECT_NEAR(output[i], expected[i], 0.0001f);
    }

    commonTestTeardown(d_input);
}
