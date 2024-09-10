#include <cuda_runtime.h>
#include <gtest/gtest.h>

#include <vector>

#include "activation.hpp"
#include "batch_norm.hpp"

class BatchNormLayerTest : public ::testing::Test {
  protected:
    shape2d            inputSize;
    int                nChannels;
    std::vector<float> weights;
    std::vector<float> biases;

    std::vector<float> runningMean;
    std::vector<float> runningVar;

    std::vector<float> input;
    std::vector<float> expected;

    float                        *d_input;
    float                        *d_output;
    CUDANet::Layers::BatchNorm2d *batchNorm;

    virtual void SetUp() override {
        d_input   = nullptr;
        d_output  = nullptr;
        batchNorm = nullptr;
    }

    virtual void TearDown() override {
        if (d_input) {
            cudaFree(d_input);
        }
    }

    void runTest() {
        cudaError_t cudaStatus;

        batchNorm = new CUDANet::Layers::BatchNorm2d(
            inputSize, nChannels, 1e-5f, CUDANet::Layers::ActivationType::NONE
        );

        batchNorm->setWeights(weights.data());
        batchNorm->setBiases(biases.data());

        batchNorm->setRunningMean(runningMean.data());
        batchNorm->setRunningVar(runningVar.data());

        cudaStatus = cudaGetLastError();
        EXPECT_EQ(cudaStatus, cudaSuccess);

        cudaStatus =
            cudaMalloc((void **)&d_input, sizeof(float) * input.size());
        EXPECT_EQ(cudaStatus, cudaSuccess);

        cudaStatus = cudaMemcpy(
            d_input, input.data(), sizeof(float) * input.size(),
            cudaMemcpyHostToDevice
        );
        EXPECT_EQ(cudaStatus, cudaSuccess);

        d_output = batchNorm->forward(d_input);

        std::vector<float> output(input.size());
        cudaStatus = cudaMemcpy(
            output.data(), d_output, sizeof(float) * output.size(),
            cudaMemcpyDeviceToHost
        );
        EXPECT_EQ(cudaStatus, cudaSuccess);

        for (int i = 0; i < output.size(); ++i) {
            EXPECT_NEAR(output[i], expected[i], 1e-5);
        }

        delete batchNorm;
    }
};

TEST_F(BatchNormLayerTest, BatchNormSmallForwardTest) {
    inputSize = {4, 4};
    nChannels = 2;

    weights = {0.63508f, 0.64903f};
    biases  = {0.25079f, 0.66841f};

    runningMean = {0.5f, 0.5f};
    runningVar  = {1.0f, 1.0f};

    // clang-format off
    input = {
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

    expected = {0.18029f, 0.44435f,  0.2434f,  0.5517f,  0.29847f, 0.3196f,
                0.34902f, -0.06568f, 0.35157f, 0.4047f,  0.04711f, 0.38002f,
                0.03184f, 0.1788f,   0.08193f, 0.38651f, 0.55466f, 0.44578f,
                0.54769f, 0.73908f,  0.86486f, 0.9914f,  0.952f,   0.92307f,
                0.43118f, 0.52152f,  0.68811f, 0.46697f, 0.7182f,  0.65779f,
                0.4144f,  0.38096f};

    runTest();
}

TEST_F(BatchNormLayerTest, BatchNormNonSquareInputTest) {
    inputSize = {4, 6};  // Non-square input
    nChannels = 2;
    weights   = {0.63508f, 0.64903f};
    biases    = {0.25079f, 0.66841f};

    runningMean = {0.5f, 0.5f};
    runningVar  = {1.0f, 1.0f};

    input = {// Channel 0
             0.38899f, 0.80478f, 0.48836f, 0.97381f, 0.21567f, 0.92312f,
             0.57508f, 0.60835f, 0.65467f, 0.00168f, 0.31567f, 0.71345f,
             0.65869f, 0.74235f, 0.17928f, 0.70349f, 0.12856f, 0.95645f,
             0.15524f, 0.38664f, 0.23411f, 0.7137f, 0.26789f, 0.83412f,
             // Channel 1
             0.32473f, 0.15698f, 0.314f, 0.60888f, 0.23145f, 0.78945f, 0.80268f,
             0.99766f, 0.93694f, 0.89237f, 0.61234f, 0.92314f, 0.13449f,
             0.27367f, 0.53036f, 0.18962f, 0.45623f, 0.14523f, 0.57672f,
             0.48364f, 0.10863f, 0.0571f, 0.78934f, 0.67545f
    };

    expected = {0.18029f, 0.44435f, 0.2434f,  0.5517f,   0.07022f, 0.5195f,
                0.29847f, 0.3196f,  0.34902f, -0.06568f, 0.13373f, 0.38635f,
                0.35157f, 0.4047f,  0.04711f, 0.38002f,  0.0149f,  0.54067f,
                0.03184f, 0.1788f,  0.08193f, 0.38651f,  0.10338f, 0.46298f,
                0.55466f, 0.44578f, 0.54769f, 0.73908f,  0.49411f, 0.85627f,
                0.86486f, 0.9914f,  0.952f,   0.92307f,  0.74132f, 0.94304f,
                0.43118f, 0.52152f, 0.68811f, 0.46697f,  0.64f,    0.43815f,
                0.7182f,  0.65779f, 0.4144f,  0.38096f,  0.8562f,  0.78228f};

    runTest();
}