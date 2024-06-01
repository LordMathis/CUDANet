#include <gtest/gtest.h>

#include <inception_v3.hpp>

class BasicConv2dTest : public ::testing::Test {
  protected:
    BasicConv2d *basic_conv2d;

    shape2d     inputShape;
    int         inputChannels;
    int         outputChannels;
    shape2d     kernelSize;
    shape2d     stride;
    shape2d     padding;
    std::string prefix = "test";

    float *d_input;
    float *d_output;

    std::vector<float> input;
    std::vector<float> output;

    std::vector<float> weights;
    std::vector<float> biases;

    virtual void SetUp() override {
        basic_conv2d = nullptr;
    }

    virtual void TearDown() override {
        // Clean up
        delete basic_conv2d;
    }
};

TEST_F(BasicConv2dTest, BasicConv2dTest1) {
    inputShape     = {28, 28};
    inputChannels  = 1;
    outputChannels = 32;
    kernelSize     = {3, 3};
    stride         = {1, 1};
    padding        = {1, 1};
};