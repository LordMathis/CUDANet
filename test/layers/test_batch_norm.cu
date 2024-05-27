#include <cuda_runtime.h>
#include <gtest/gtest.h>

#include <vector>

#include "activation.cuh"
#include "batch_norm.cuh"

class BatchNormLayerTest : public ::testing::Test {
  protected:
    shape2d              inputSize;
    int                nChannels;
    std::vector<float> weights;
    std::vector<float> biases;
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

    expected = {-0.06007f, 0.951f,    0.18157f,  1.36202f, 0.39244f,  0.47335f,
                0.58598f,  -1.00188f, 0.59576f,  0.79919f, -0.57001f, 0.70469f,
                -0.62847f, -0.06578f, -0.43668f, 0.72952f, 0.37726f,  0.02088f,
                0.35446f,  0.98092f,  1.39264f,  1.80686f, 1.67786f,  1.58318f,
                -0.0269f,  0.26878f,  0.81411f,  0.09022f, 0.9126f,   0.71485f,
                -0.08184f, -0.19131f};

    runTest();
}

TEST_F(BatchNormLayerTest, BatchNormNonSquareInputTest) {
    inputSize = {4, 6};  // Non-square input
    nChannels = 2;
    weights   = {0.63508f, 0.64903f};
    biases    = {0.25079f, 0.66841f};

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

    expected = {-0.05598f, 0.87495f,  0.1665f,   1.2534f,   -0.44404f,
                1.13991f,  0.36066f,  0.43515f,  0.53886f,  -0.92315f,
                -0.22014f, 0.67047f,  0.54786f,  0.73517f,  -0.52552f,
                0.64817f,  -0.63907f, 1.21453f,  -0.57934f, -0.06124f,
                -0.40275f, 0.67103f,  -0.32712f, 0.94064f,  0.28344f,
                -0.08405f, 0.25993f,  0.90592f,  0.07909f,  1.30149f,
                1.33047f,  1.7576f,   1.62459f,  1.52695f,  0.9135f,
                1.59436f,  -0.13331f, 0.17158f,  0.73391f,  -0.01254f,
                0.57151f,  -0.10979f, 0.83546f,  0.63156f,  -0.18996f,
                -0.30285f, 1.30124f,  1.05175f};

    runTest();
}