#include <cuda_runtime.h>
#include <gtest/gtest.h>

#include <vector>

#include "avg_pooling.cuh"

class AvgPoolingLayerTest : public ::testing::Test {
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
    CUDANet::Layers::AvgPooling2d *avgPoolingLayer;

    virtual void SetUp() override {
        d_input         = nullptr;
        d_output        = nullptr;
        avgPoolingLayer = nullptr;
    }

    virtual void TearDown() override {
        if (d_input) {
            cudaFree(d_input);
        }
    }

    void runTest() {
        cudaError_t cudaStatus;

        avgPoolingLayer = new CUDANet::Layers::AvgPooling2d(
            inputSize, nChannels, poolingSize, stride, padding,
            CUDANet::Layers::ActivationType::NONE
        );

        cudaStatus = cudaMalloc(
            (void **)&d_input,
            sizeof(float) * inputSize.first * inputSize.second * nChannels
        );
        EXPECT_EQ(cudaStatus, cudaSuccess);

        cudaStatus = cudaMemcpy(
            d_input, input.data(),
            sizeof(float) * inputSize.first * inputSize.second * nChannels,
            cudaMemcpyHostToDevice
        );
        EXPECT_EQ(cudaStatus, cudaSuccess);

        d_output = avgPoolingLayer->forward(d_input);

        int outputSize = avgPoolingLayer->getOutputSize();

        std::vector<float> output(outputSize);
        cudaStatus = cudaMemcpy(
            output.data(), d_output, sizeof(float) * outputSize,
            cudaMemcpyDeviceToHost
        );
        EXPECT_EQ(cudaStatus, cudaSuccess);

        for (int i = 0; i < output.size(); ++i) {
            EXPECT_NEAR(expected[i], output[i], 1e-4);
        }

        delete avgPoolingLayer;
    }
};

TEST_F(AvgPoolingLayerTest, AvgPoolForwardTest) {
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

    expected = {0.43775f, 0.49475f, 0.48975f, 0.339f,
                0.45675f, 0.303f,   0.56975f, 0.57025f};

    runTest();
}

TEST_F(AvgPoolingLayerTest, AvgPoolForwardNonSquareInputTest) {
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

    expected = {0.43775f, 0.49475f, 0.4005f, 0.48975f, 0.339f,   0.654f,
                0.45675f, 0.303f,   0.654f,  0.56975f, 0.57025f, 0.555f};

    runTest();
}

TEST_F(AvgPoolingLayerTest, AvgPoolForwardNonSquarePoolingTest) {
    inputSize   = {4, 4};
    nChannels   = 2;
    poolingSize = {2, 3};  // Non-square pooling
    stride      = {2, 2};
    padding     = {0, 0};

    input = {// Channel 0
             0.573f, 0.619f, 0.732f, 0.055f, 0.243f, 0.316f, 0.573f, 0.619f,
             0.712f, 0.055f, 0.243f, 0.316f, 0.573f, 0.619f, 0.742f, 0.055f,
             // Channel 1
             0.473f, 0.919f, 0.107f, 0.073f, 0.073f, 0.362f, 0.973f, 0.059f,
             0.473f, 0.455f, 0.283f, 0.416f, 0.532f, 0.819f, 0.732f, 0.850f
    };

    expected = {0.50933f, 0.49067f, 0.4845f, 0.549f};

    runTest();
}

TEST_F(AvgPoolingLayerTest, AvgPoolForwardNonSquareStrideTest) {
    inputSize   = {4, 4};
    nChannels   = 2;
    poolingSize = {2, 2};
    stride      = {1, 2};  // Non-square stride
    padding     = {0, 0};

    input = {// Channel 0
             0.573f, 0.619f, 0.732f, 0.055f, 0.243f, 0.316f, 0.573f, 0.619f,
             0.712f, 0.055f, 0.243f, 0.316f, 0.573f, 0.619f, 0.742f, 0.055f,
             // Channel 1
             0.473f, 0.919f, 0.107f, 0.073f, 0.073f, 0.362f, 0.973f, 0.059f,
             0.473f, 0.455f, 0.283f, 0.416f, 0.532f, 0.819f, 0.732f, 0.850f
    };

    expected = {0.43775f, 0.49475f, 0.3315f,  0.43775f, 0.48975f, 0.339f,
                0.45675f, 0.303f,   0.34075f, 0.43275f, 0.56975f, 0.57025f};

    runTest();
}

TEST_F(AvgPoolingLayerTest, AvgPoolForwardNonSquarePaddingTest) {
    inputSize   = {4, 4};
    nChannels   = 2;
    poolingSize = {2, 2};
    stride      = {2, 2};
    padding     = {1, 0};  // Non-square padding

    input = {// Channel 0
             0.573f, 0.619f, 0.732f, 0.055f, 0.243f, 0.316f, 0.573f, 0.619f,
             0.712f, 0.055f, 0.243f, 0.316f, 0.573f, 0.619f, 0.742f, 0.055f,
             // Channel 1
             0.473f, 0.919f, 0.107f, 0.073f, 0.073f, 0.362f, 0.973f, 0.059f,
             0.473f, 0.455f, 0.283f, 0.416f, 0.532f, 0.819f, 0.732f, 0.850f
    };

    expected = {0.298f, 0.19675f, 0.3315f,  0.43775f, 0.298f,   0.19925f,
                0.348f, 0.045f,   0.34075f, 0.43275f, 0.33775f, 0.3955f};

    runTest();
}