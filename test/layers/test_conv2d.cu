#include <cuda_runtime_api.h>
#include <gtest/gtest.h>

#include <iostream>

#include "conv2d.cuh"

class Conv2dTest : public ::testing::Test {
  protected:
    shape2d                           inputSize;
    int                             inputChannels;
    shape2d                           kernelSize;
    shape2d                           stride;
    int                             numFilters;
    shape2d                           paddingSize;
    CUDANet::Layers::ActivationType activationType;
    std::vector<float>              input;
    std::vector<float>              kernels;
    std::vector<float>              expected;

    float                   *d_input;
    float                   *d_output;
    CUDANet::Layers::Conv2d *conv2dLayer;

    virtual void SetUp() override {
        d_input     = nullptr;
        d_output    = nullptr;
        conv2dLayer = nullptr;
    }

    virtual void TearDown() override {
        if (d_input) {
            cudaFree(d_input);
        }
        delete conv2dLayer;
    }

    void runTest() {
        cudaError_t cudaStatus;

        conv2dLayer = new CUDANet::Layers::Conv2d(
            inputSize, inputChannels, kernelSize, stride, numFilters,
            paddingSize, activationType
        );

        conv2dLayer->setWeights(kernels.data());

        cudaStatus =
            cudaMalloc((void **)&d_input, sizeof(float) * input.size());
        EXPECT_EQ(cudaStatus, cudaSuccess);

        cudaStatus = cudaMemcpy(
            d_input, input.data(), sizeof(float) * input.size(),
            cudaMemcpyHostToDevice
        );
        EXPECT_EQ(cudaStatus, cudaSuccess);

        d_output = conv2dLayer->forward(d_input);

        int outputHeight =
            (inputSize.first - kernelSize.first + 2 * paddingSize.first) /
                stride.first +
            1;
        int outputWidth =
            (inputSize.second - kernelSize.second + 2 * paddingSize.second) /
                stride.second +
            1;
        int outputSize = outputHeight * outputWidth * numFilters;
        EXPECT_EQ(outputSize, conv2dLayer->getOutputSize());

        std::vector<float> output(outputSize);
        cudaStatus = cudaMemcpy(
            output.data(), d_output, sizeof(float) * output.size(),
            cudaMemcpyDeviceToHost
        );
        EXPECT_EQ(cudaStatus, cudaSuccess);

        for (int i = 0; i < output.size(); ++i) {
            EXPECT_NEAR(expected[i], output[i], 1e-5f);
        }
    }
};

TEST_F(Conv2dTest, SimpleTest) {
    inputSize      = {4, 4};
    inputChannels  = 1;
    kernelSize     = {2, 2};
    stride         = {1, 1};
    numFilters     = 1;
    paddingSize    = {0, 0};
    activationType = CUDANet::Layers::ActivationType::NONE;

    input = {
        // clang-format off
        1.0f, 2.0f, 3.0f, 4.0f,
        5.0f, 6.0f, 7.0f, 8.0f,
        9.0f, 10.0f, 11.0f, 12.0f,
        13.0f, 14.0f, 15.0f, 16.0f
        // clang-format on
    };
    kernels = {
        // clang-format off
        1.0f,2.0f,
        3.0f, 4.0f
        // clang-format on
    };
    expected = {44.0f,  54.0f,  64.0f,  84.0f, 94.0f,
                104.0f, 124.0f, 134.0f, 144.0f};

    runTest();
}

TEST_F(Conv2dTest, PaddedTest) {
    inputSize     = {5, 5};
    inputChannels = 3;
    kernelSize    = {3, 3};
    stride        = {1, 1};
    numFilters    = 2;

    int paddingFirst =
        CUDANET_SAME_PADDING(inputSize.first, kernelSize.first, stride.first);
    int paddingSecond = CUDANET_SAME_PADDING(
        inputSize.second, kernelSize.second, stride.second
    );
    paddingSize = {paddingFirst, paddingSecond};

    activationType = CUDANet::Layers::ActivationType::NONE;

    // clang-format off
    input = {
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

    kernels = {
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

    // Generated by tools/generate_conv2d_test.py
    expected = {
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

    runTest();
}

TEST_F(Conv2dTest, StridedPaddedTest) {
    inputSize     = {5, 5};
    inputChannels = 2;
    kernelSize    = {3, 3};
    stride        = {2, 2};
    numFilters    = 2;

    int paddingFirst =
        CUDANET_SAME_PADDING(inputSize.first, kernelSize.second, stride.first);
    int paddingSecond = CUDANET_SAME_PADDING(
        inputSize.second, kernelSize.second, stride.second
    );
    paddingSize = {paddingFirst, paddingSecond};

    activationType = CUDANet::Layers::ActivationType::RELU;

    // clang-format off
    input = {
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
    kernels = {
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

    expected = {// Channel 1
                0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 1.59803f, 2.84444f, 1.6201f,
                0.0f, 0.0f, 2.38937f, 3.80762f, 3.39679f, 0.0f, 0.0f, 1.13102f,
                2.33335f, 1.98488f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f,
                // Channel 2
                0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 2.57732f, 3.55543f,
                2.24675f, 0.0f, 0.0f, 3.36842f, 3.41373f, 3.14804f, 0.0f, 0.0f,
                1.17963f, 2.55005f, 1.63218f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f
    };

    runTest();
}

TEST_F(Conv2dTest, NonSquareInputTest) {
    inputSize      = {4, 6};  // Non-square input
    inputChannels  = 1;
    kernelSize     = {2, 2};
    stride         = {1, 1};
    numFilters     = 1;
    paddingSize    = {0, 0};
    activationType = CUDANet::Layers::ActivationType::NONE;

    input = {
        // clang-format off
        0.946f, 0.879f, 0.382f, 0.542f, 0.453f, 0.128f,
        0.128f, 0.860f, 0.778f, 0.049f, 0.974f, 0.400f,
        0.400f, 0.874f, 0.161f, 0.271f, 0.580f, 0.373f,
        0.078f, 0.366f, 0.396f, 0.181f, 0.246f, 0.112f
        // clang-format on
    };
    kernels  = {0.744f, 0.745f, 0.164f, 0.157f};
    expected = {1.51469f, 1.20175f, 0.82328f, 0.90169f, 0.65493f,
                0.93875f, 1.38806f, 0.68429f, 0.89759f, 1.17634f,
                1.01898f, 0.8924f,  0.41504f, 0.70203f, 0.76733f};

    runTest();
}

TEST_F(Conv2dTest, NonSquareKernelTest) {
    inputSize      = {4, 4};
    inputChannels  = 1;
    kernelSize     = {1, 3};  // Non-square kernel
    stride         = {1, 1};
    numFilters     = 1;
    paddingSize    = {0, 0};
    activationType = CUDANet::Layers::ActivationType::NONE;

    input = {
        // clang-format off
        0.946f, 0.879f, 0.382f, 0.542f,
        0.128f, 0.860f, 0.778f, 0.049f,
        0.400f, 0.874f, 0.161f, 0.271f,
        0.078f, 0.366f, 0.396f, 0.181f
        // clang-format on
    };
    kernels  = {0.744f, 0.745f, 0.164f};
    expected = {1.42133f, 1.02745f, 0.86352f, 1.22749f,
                0.97513f, 0.81465f, 0.39565f, 0.59701f};

    runTest();
}

TEST_F(Conv2dTest, NonSquareStrideTest) {
    inputSize      = {4, 4};
    inputChannels  = 1;
    kernelSize     = {2, 2};
    stride         = {1, 2};  // Non-square stride
    numFilters     = 1;
    paddingSize    = {0, 0};
    activationType = CUDANet::Layers::ActivationType::NONE;

    input = {
        // clang-format off
        0.946f, 0.879f, 0.382f, 0.542f,
        0.128f, 0.860f, 0.778f, 0.049f,
        0.400f, 0.874f, 0.161f, 0.271f,
        0.078f, 0.366f, 0.396f, 0.181f
        // clang-format on
    };
    kernels  = {0.144f, 0.745f, 0.964f, 0.164f};
    expected = {1.05551f, 1.21683f, 1.18807f, 0.34818f, 0.84395f, 0.63651f};

    runTest();
}

TEST_F(Conv2dTest, NonSquarePaddingTest) {
    inputSize      = {4, 4};
    inputChannels  = 1;
    kernelSize     = {2, 2};
    stride         = {1, 1};
    numFilters     = 1;
    paddingSize    = {1, 2};  // Non-square padding
    activationType = CUDANet::Layers::ActivationType::NONE;

    input = {
        // clang-format off
        0.946f, 0.879f, 0.382f, 0.542f,
        0.128f, 0.860f, 0.778f, 0.049f,
        0.400f, 0.874f, 0.161f, 0.271f,
        0.078f, 0.366f, 0.396f, 0.181f
        // clang-format on
    };
    kernels  = {0.144f, 0.745f, 0.964f, 0.164f};
    expected = {0.0f, 0.15514f, 1.0561f,  0.91f,    0.45714f, 0.52249f, 0.0f,
                0.0f, 0.72576f, 1.05551f, 1.3678f,  1.21683f, 0.12528f, 0.0f,
                0.0f, 0.16096f, 1.18807f, 1.57239f, 0.34818f, 0.2683f,  0.0f,
                0.0f, 0.31079f, 0.84395f, 0.66357f, 0.63651f, 0.21351f, 0.0f,
                0.0f, 0.05811f, 0.2839f,  0.34772f, 0.19187f, 0.02606f, 0.0f};
}