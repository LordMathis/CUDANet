#include <gtest/gtest.h>

#include <inception_v3.hpp>

class InceptionATest : public ::testing::Test {
  protected:
    InceptionA *inception_a;

    cudaError_t cudaStatus;

    shape2d inputShape;
    int     inputChannels;
    int     poolFeatures;
    std::string prefix = "test";

    float *d_input;
    float *d_output;

    std::vector<float> input;
    std::vector<float> expected;

    std::vector<float> branch1x1_conv_weights;
    std::vector<float> branch1x1_conv_biases;
    std::vector<float> branch1x1_bn_weights;
    std::vector<float> branch1x1_bn_biases;

    std::vector<float> branch5x5_1_conv_weights;
    std::vector<float> branch5x5_1_conv_biases;
    std::vector<float> branch5x5_1_bn_weights;
    std::vector<float> branch5x5_1_bn_biases;
    std::vector<float> branch5x5_2_conv_weights;
    std::vector<float> branch5x5_2_conv_biases;
    std::vector<float> branch5x5_2_bn_weights;
    std::vector<float> branch5x5_2_bn_biases;

    std::vector<float> branch3x3dbl_1_conv_weights;
    std::vector<float> branch3x3dbl_1_conv_biases;
    std::vector<float> branch3x3dbl_1_bn_weights;
    std::vector<float> branch3x3dbl_1_bn_biases;
    std::vector<float> branch3x3dbl_2_conv_weights;
    std::vector<float> branch3x3dbl_2_conv_biases;
    std::vector<float> branch3x3dbl_2_bn_weights;
    std::vector<float> branch3x3dbl_2_bn_biases;
    std::vector<float> branch3x3dbl_3_conv_weights;
    std::vector<float> branch3x3dbl_3_conv_biases;
    std::vector<float> branch3x3dbl_3_bn_weights;
    std::vector<float> branch3x3dbl_3_bn_biases;

    std::vector<float> branchPool_2_conv_weights;
    std::vector<float> branchPool_2_conv_biases;
    std::vector<float> branchPool_2_bn_weights;
    std::vector<float> branchPool_2_bn_biases;

    virtual void SetUp() override {
        inception_a = nullptr;
    }

    virtual void TearDown() override {
        // Clean up
        delete inception_a;
    }

    void setBasicConv2dWeights(
        BasicConv2d *basic_conv2d,
        const std::vector<float> &conv_weights,
        const std::vector<float> &conv_biases,
        const std::vector<float> &bn_weights,
        const std::vector<float> &bn_biases
    ) {
        std::pair<std::string, CUDANet::Layers::SequentialLayer *> layerPair =
            basic_conv2d->getLayers()[0];

        ASSERT_EQ(layerPair.first, prefix + ".conv");

        CUDANet::Layers::Conv2d *conv =
            dynamic_cast<CUDANet::Layers::Conv2d *>(layerPair.second);
        conv->setWeights(conv_weights.data());
        conv->setBiases(conv_biases.data());

        ASSERT_EQ(conv->getWeights().size(), conv_weights.size());
        ASSERT_EQ(conv->getBiases().size(), conv_biases.size());

        cudaStatus = cudaGetLastError();
        EXPECT_EQ(cudaStatus, cudaSuccess);

        layerPair = basic_conv2d->getLayers()[1];
        ASSERT_EQ(layerPair.first, prefix + ".bn");

        CUDANet::Layers::BatchNorm2d *bn =
            dynamic_cast<CUDANet::Layers::BatchNorm2d *>(layerPair.second);
        bn->setWeights(bn_weights.data());
        bn->setBiases(bn_biases.data());

        ASSERT_EQ(bn->getWeights().size(), bn_weights.size());
        ASSERT_EQ(bn->getBiases().size(), bn_biases.size());

        cudaStatus = cudaGetLastError();
        EXPECT_EQ(cudaStatus, cudaSuccess);
    }

    void runTest() {
        inception_a = new InceptionA(
            inputShape, inputChannels, poolFeatures, prefix
        );

        // Set up layer weights and biases
        // Branch 1x1
        std::pair<std::string, CUDANet::Layers::SequentialLayer *> layerPair =
            inception_a->getLayers()[0];
        ASSERT_EQ(layerPair.first, prefix + ".branch1x1");
        BasicConv2d *basic_conv2d =
            dynamic_cast<BasicConv2d *>(layerPair.second);
        setBasicConv2dWeights(
            basic_conv2d,
            branch1x1_conv_weights,
            branch1x1_conv_biases,
            branch1x1_bn_weights,
            branch1x1_bn_biases
        );

        // Branch 5x5
        layerPair = inception_a->getLayers()[1];
        ASSERT_EQ(layerPair.first, prefix + ".branch5x5_1");
        basic_conv2d = dynamic_cast<BasicConv2d *>(layerPair.second);
        setBasicConv2dWeights(
            basic_conv2d,
            branch5x5_1_conv_weights,
            branch5x5_1_conv_biases,
            branch5x5_1_bn_weights,
            branch5x5_1_bn_biases
        );
        layerPair = inception_a->getLayers()[2];
        ASSERT_EQ(layerPair.first, prefix + ".branch5x5_2");
        basic_conv2d = dynamic_cast<BasicConv2d *>(layerPair.second);
        setBasicConv2dWeights(
            basic_conv2d,
            branch5x5_2_conv_weights,
            branch5x5_2_conv_biases,
            branch5x5_2_bn_weights,
            branch5x5_2_bn_biases
        );

        // Branch 3x3dbl
        layerPair = inception_a->getLayers()[3];
        ASSERT_EQ(layerPair.first, prefix + ".branch3x3dbl_1");
        basic_conv2d = dynamic_cast<BasicConv2d *>(layerPair.second);
        setBasicConv2dWeights(
            basic_conv2d,
            branch3x3dbl_1_conv_weights,
            branch3x3dbl_1_conv_biases,
            branch3x3dbl_1_bn_weights,
            branch3x3dbl_1_bn_biases
        );
        layerPair = inception_a->getLayers()[4];
        ASSERT_EQ(layerPair.first, prefix + ".branch3x3dbl_2");
        basic_conv2d = dynamic_cast<BasicConv2d *>(layerPair.second);
        setBasicConv2dWeights(
            basic_conv2d,
            branch3x3dbl_2_conv_weights,
            branch3x3dbl_2_conv_biases,
            branch3x3dbl_2_bn_weights,
            branch3x3dbl_2_bn_biases
        );
        layerPair = inception_a->getLayers()[5];
        ASSERT_EQ(layerPair.first, prefix + ".branch3x3dbl_3");
        basic_conv2d = dynamic_cast<BasicConv2d *>(layerPair.second);
        setBasicConv2dWeights(
            basic_conv2d,
            branch3x3dbl_3_conv_weights,
            branch3x3dbl_3_conv_biases,
            branch3x3dbl_3_bn_weights,
            branch3x3dbl_3_bn_biases
        );

        // Pool
        layerPair = inception_a->getLayers()[7]; // 6 is a pool layer without weights
        ASSERT_EQ(layerPair.first, prefix + ".branch_pool");
        basic_conv2d = dynamic_cast<BasicConv2d *>(layerPair.second);
        setBasicConv2dWeights(
            basic_conv2d,
            branchPool_2_conv_weights,
            branchPool_2_conv_biases,
            branchPool_2_bn_weights,
            branchPool_2_bn_biases
        );

        cudaStatus =
            cudaMalloc((void **)&d_input, sizeof(float) * input.size());
        EXPECT_EQ(cudaStatus, cudaSuccess);

        cudaStatus = cudaMemcpy(
            d_input, input.data(), sizeof(float) * input.size(),
            cudaMemcpyHostToDevice
        );
        EXPECT_EQ(cudaStatus, cudaSuccess);

        d_output = inception_a->forward(d_input);

        cudaStatus = cudaGetLastError();
        EXPECT_EQ(cudaStatus, cudaSuccess);

        int                outputSize = inception_a->getOutputSize();
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