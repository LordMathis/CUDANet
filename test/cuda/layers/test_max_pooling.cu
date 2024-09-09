#include <cuda_runtime.h>
#include <gtest/gtest.h>

#include <vector>

#include "max_pooling.hpp"

class MaxPoolingLayerTest : public ::testing::Test {
  protected:
    shape2d              inputSize;
    int                nChannels;
    shape2d              poolingSize;
    shape2d              stride;
    shape2d              padding;
    std::vector<float> input;
    std::vector<float> expected;

    float                         *d_input;
    float                         *d_output;
    CUDANet::Layers::MaxPooling2d *maxPoolingLayer;

    virtual void SetUp() override {
        d_input         = nullptr;
        d_output        = nullptr;
        maxPoolingLayer = nullptr;
    }

    virtual void TearDown() override {
        if (d_input) {
            cudaFree(d_input);
        }
        delete maxPoolingLayer;
    }

    void runTest() {
        cudaError_t cudaStatus;

        maxPoolingLayer = new CUDANet::Layers::MaxPooling2d(
            inputSize, nChannels, poolingSize, stride, padding,
            CUDANet::Layers::ActivationType::NONE
        );

        cudaStatus =
            cudaMalloc((void **)&d_input, sizeof(float) * input.size());
        EXPECT_EQ(cudaStatus, cudaSuccess);

        cudaStatus = cudaMemcpy(
            d_input, input.data(), sizeof(float) * input.size(),
            cudaMemcpyHostToDevice
        );
        EXPECT_EQ(cudaStatus, cudaSuccess);

        d_output = maxPoolingLayer->forward(d_input);

        int outputSize = maxPoolingLayer->getOutputSize();

        std::vector<float> output(outputSize);
        cudaStatus = cudaMemcpy(
            output.data(), d_output, sizeof(float) * output.size(),
            cudaMemcpyDeviceToHost
        );
        EXPECT_EQ(cudaStatus, cudaSuccess);

        for (int i = 0; i < output.size(); ++i) {
            EXPECT_FLOAT_EQ(expected[i], output[i]);
        }
    }
};

TEST_F(MaxPoolingLayerTest, MaxPoolForwardTest) {
    inputSize   = {4, 4};
    nChannels   = 2;
    poolingSize = {2, 2};
    stride      = {2, 2};
    padding     = {0, 0};

    input = {
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

    expected = {0.619f, 0.732f, 0.712f, 0.742f, 0.919f, 0.973f, 0.819f, 0.85f};

    runTest();
}

TEST_F(MaxPoolingLayerTest, MaxPoolForwardNonSquareInputTest) {
    inputSize   = {4, 6};  // Non-square input
    nChannels   = 2;
    poolingSize = {2, 2};
    stride      = {2, 2};
    padding     = {0, 0};

    input = {// Channel 0
             0.573f, 0.619f, 0.732f, 0.055f, 0.123f, 0.234f, 0.243f, 0.316f,
             0.573f, 0.619f, 0.456f, 0.789f, 0.712f, 0.055f, 0.243f, 0.316f,
             0.654f, 0.987f, 0.573f, 0.619f, 0.742f, 0.055f, 0.321f, 0.654f,
             // Channel 1
             0.473f, 0.919f, 0.107f, 0.073f, 0.321f, 0.654f, 0.073f, 0.362f,
             0.973f, 0.059f, 0.654f, 0.987f, 0.473f, 0.455f, 0.283f, 0.416f,
             0.789f, 0.123f, 0.532f, 0.819f, 0.732f, 0.850f, 0.987f, 0.321f
    };

    expected = {0.619f, 0.732f, 0.789f, 0.712f, 0.742f, 0.987f, 0.919f, 0.973f, 0.987f, 0.819f, 0.85f, 0.987f};

    runTest();
}

TEST_F(MaxPoolingLayerTest, MaxPoolForwardNonSquarePoolSizeTest) {
    inputSize   = {4, 4};
    nChannels   = 2;
    poolingSize = {2, 3};  // Non-square pooling size
    stride      = {2, 2};
    padding     = {0, 0};

    input = {
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

    expected = {0.732f, 0.742f, 0.973f, 0.819f};

    runTest();

}

TEST_F(MaxPoolingLayerTest, MaxPoolForwardNonSquareStrideTest) {
    inputSize   = {4, 4};
    nChannels   = 2;
    poolingSize = {2, 2};
    stride      = {1, 2};  // Non-square stride
    padding     = {0, 0};

    input = {
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

    expected = {0.619f, 0.732f, 0.712f, 0.619f, 0.712f, 0.742f, 0.919f, 0.973f, 0.473f, 0.973f, 0.819f, 0.85f};

    runTest();

}

TEST_F(MaxPoolingLayerTest, MaxPoolForwardNonSquarePaddingTest) {
    inputSize   = {4, 4};
    nChannels   = 2;
    poolingSize = {2, 2};
    stride      = {2, 2};  // Non-square stride
    padding     = {0, 1};

    input = {
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

    expected = {0.573f, 0.732f, 0.619f, 0.712f, 0.742f, 0.316f, 0.473f, 0.973f, 0.073f, 0.532f, 0.819f, 0.85f};

    runTest();

}