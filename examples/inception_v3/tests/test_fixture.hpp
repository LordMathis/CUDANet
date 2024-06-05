#ifndef TEST_FIXTURE_H
#define TEST_FIXTURE_H

#include <cmath>
#include <gtest/gtest.h>

#include "inception_v3.hpp"

class InceptionBlockTest : public ::testing::Test {
  protected:
    CUDANet::Model *model;

    cudaError_t cudaStatus;

    shape2d inputShape;
    int     inputChannels;

    int outputSize;

    std::vector<float> input;
    std::vector<float> expected;

    virtual void SetUp() override {
        model = nullptr;
    }

    virtual void TearDown() override {
        // Clean up
        delete model;
    }

    void runTest() {
        EXPECT_EQ(
            input.size(), inputShape.first * inputShape.second * inputChannels
        );

        float *output = model->predict(input.data());

        cudaStatus = cudaGetLastError();
        EXPECT_EQ(cudaStatus, cudaSuccess);

        EXPECT_EQ(outputSize, expected.size());

        for (int i = 0; i < outputSize; ++i) {
            EXPECT_NEAR(expected[i], output[i], 1e-3f);
        }
    }
};

#endif