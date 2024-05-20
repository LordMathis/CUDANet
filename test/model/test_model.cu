#include <gtest/gtest.h>

#include "conv2d.cuh"
#include "dense.cuh"
#include "max_pooling.cuh"
#include "model.hpp"

class ModelTest : public ::testing::Test {
  protected:
    CUDANet::Model *commonTestSetup(
        bool setWeights = true,

        dim2d inputSize     = {6, 6},
        int   inputChannels = 2,
        int   outputSize    = 3,

        dim2d kernelSize = {3, 3},
        dim2d stride     = {1, 1},
        int   numFilters = 2,

        dim2d poolingSize   = {2, 2},
        dim2d poolingStride = {2, 2}
    ) {
        CUDANet::Model *model =
            new CUDANet::Model(inputSize, inputChannels, outputSize);

        dim2d paddingSize = {0, 0};

        // Conv2d
        CUDANet::Layers::Conv2d *conv2d = new CUDANet::Layers::Conv2d(
            inputSize, inputChannels, kernelSize, stride, numFilters,
            paddingSize, CUDANet::Layers::ActivationType::NONE
        );

        if (setWeights) {
            conv2d->setWeights(getConv1Weights().data());
        }
        model->addLayer("conv1", conv2d);

        // maxpool2d
        dim2d poolingInput = {
            inputSize.first - kernelSize.first + 1,
            inputSize.second - kernelSize.second + 1
        };
        CUDANet::Layers::MaxPooling2d *maxpool2d =
            new CUDANet::Layers::MaxPooling2d(
                poolingInput, numFilters, poolingSize,
                poolingStride, CUDANet::Layers::ActivationType::RELU
            );
        model->addLayer("maxpool1", maxpool2d);

        // dense
        CUDANet::Layers::Dense *dense = new CUDANet::Layers::Dense(
            8, 3, CUDANet::Layers::ActivationType::SOFTMAX
        );

        if (setWeights) {
            dense->setWeights(getDenseWeights().data());
        }
        model->addLayer("linear", dense);

        return model;
    }

    std::vector<float> getConv1Weights() {
        return std::vector<float>{
            0.18313f, 0.53363f, 0.39527f, 0.27575f, 0.3433f,  0.41746f,
            0.16831f, 0.61693f, 0.54599f, 0.99692f, 0.77127f, 0.25146f,
            0.4206f,  0.16291f, 0.93484f, 0.79765f, 0.74982f, 0.78336f,
            0.6386f,  0.87744f, 0.33587f, 0.9691f,  0.68437f, 0.65098f,
            0.48153f, 0.97546f, 0.8026f,  0.36689f, 0.98152f, 0.37351f,
            0.68407f, 0.2684f,  0.2855f,  0.76195f, 0.67828f, 0.603f
        };
    }

    std::vector<float> getDenseWeights() {
        return std::vector<float>{
            0.36032f, 0.33115f, 0.02948f, 0.09802f, 0.45072f, 0.56266f,
            0.43514f, 0.80946f, 0.43439f, 0.90916f, 0.08605f, 0.07473f,
            0.94788f, 0.66168f, 0.34927f, 0.09464f, 0.61963f, 0.73775f,
            0.51559f, 0.81916f, 0.64915f, 0.03934f, 0.87608f, 0.68364f,
        };
    }

    void commonTestTeardown(CUDANet::Model *model) {
        delete model;
    }

    cudaError_t cudaStatus;
};

TEST_F(ModelTest, TestModelPredict) {
    int             outputSize = 3;
    CUDANet::Model *model      = commonTestSetup();

    // input 6*6*2
    std::vector<float> input = {
        0.12762f, 0.99056f, 0.77565f, 0.29058f, 0.29787f, 0.58415f, 0.20484f,
        0.05415f, 0.60593f, 0.3162f,  0.08198f, 0.92749f, 0.72392f, 0.91786f,
        0.65266f, 0.80908f, 0.53389f, 0.36069f, 0.18614f, 0.52381f, 0.08525f,
        0.43054f, 0.3355f,  0.96587f, 0.98194f, 0.71336f, 0.78392f, 0.50648f,
        0.40355f, 0.31863f, 0.54686f, 0.1836f,  0.77171f, 0.01262f, 0.41108f,
        0.53467f, 0.3553f,  0.42808f, 0.45798f, 0.29958f, 0.3923f,  0.98277f,
        0.02033f, 0.99868f, 0.90584f, 0.57554f, 0.15957f, 0.91273f, 0.38901f,
        0.27097f, 0.64788f, 0.84272f, 0.42984f, 0.07466f, 0.53658f, 0.83388f,
        0.28232f, 0.48046f, 0.85626f, 0.04721f, 0.36139f, 0.6123f,  0.56991f,
        0.84854f, 0.61415f, 0.2466f,  0.20017f, 0.78952f, 0.93797f, 0.27884f,
        0.30514f, 0.23521f
    };

    std::vector<float> expected = {2e-05f, 0.00021f, 0.99977f};

    // predict
    const float *output = model->predict(input.data());

    float sum = 0.0f;
    for (int i = 0; i < outputSize; ++i) {
        EXPECT_NEAR(expected[i], output[i], 1e-5f);
        sum += output[i];
    }

    EXPECT_NEAR(sum, 1.0f, 1e-5f);

    commonTestTeardown(model);
}

TEST_F(ModelTest, TestModelPredictMultiple) {
    int             outputSize = 3;
    CUDANet::Model *model      = commonTestSetup();

    std::vector<float> input_1 = {
        0.81247f, 0.03579f, 0.26577f, 0.80374f, 0.64584f, 0.19658f, 0.04817f,
        0.50769f, 0.33502f, 0.01739f, 0.32263f, 0.69625f, 0.07433f, 0.98283f,
        0.21217f, 0.48437f, 0.58012f, 0.86991f, 0.81879f, 0.63589f, 0.30264f,
        0.90318f, 0.12978f, 0.35972f, 0.95847f, 0.58633f, 0.55025f, 0.68302f,
        0.61422f, 0.79524f, 0.7205f,  0.72481f, 0.51553f, 0.83032f, 0.23561f,
        0.80631f, 0.23548f, 0.84634f, 0.05839f, 0.76526f, 0.39654f, 0.95635f,
        0.75422f, 0.75341f, 0.82431f, 0.79371f, 0.72413f, 0.88557f, 0.33594f,
        0.56363f, 0.12415f, 0.05635f, 0.15952f, 0.27887f, 0.05417f, 0.58474f,
        0.75129f, 0.1788f,  0.88958f, 0.49793f, 0.85386f, 0.86262f, 0.05568f,
        0.16811f, 0.72188f, 0.08683f, 0.66985f, 0.62707f, 0.4035f,  0.51822f,
        0.46545f, 0.88722f
    };

    std::vector<float> expected_1 = {5e-05f, 0.00033f, 0.99962f};

    // predict
    const float *output_1 = model->predict(input_1.data());

    float sum_1 = 0.0f;
    for (int i = 0; i < outputSize; ++i) {
        EXPECT_NEAR(expected_1[i], output_1[i], 1e-5f);
        sum_1 += output_1[i];
    }

    EXPECT_NEAR(sum_1, 1.0f, 1e-5f);

    std::vector<float> input_2 = {
        0.83573f, 0.19191f, 0.16004f, 0.27137f, 0.64768f, 0.38417f, 0.02167f,
        0.28834f, 0.21401f, 0.16624f, 0.12037f, 0.12706f, 0.3588f,  0.10685f,
        0.49224f, 0.71267f, 0.17677f, 0.29276f, 0.92467f, 0.76689f, 0.8209f,
        0.82226f, 0.11581f, 0.6698f,  0.01109f, 0.47085f, 0.44912f, 0.45936f,
        0.83645f, 0.83272f, 0.81693f, 0.97726f, 0.60649f, 0.9f,     0.37f,
        0.20517f, 0.81921f, 0.83573f, 0.00271f, 0.30453f, 0.78925f, 0.8453f,
        0.80416f, 0.09041f, 0.0802f,  0.98408f, 0.19746f, 0.25598f, 0.09437f,
        0.27681f, 0.92053f, 0.35385f, 0.17389f, 0.14293f, 0.60151f, 0.12338f,
        0.81858f, 0.56294f, 0.97378f, 0.93272f, 0.36075f, 0.64944f, 0.2433f,
        0.66075f, 0.64496f, 0.1191f,  0.66261f, 0.63431f, 0.7137f,  0.14851f,
        0.84456f, 0.44482f
    };

    std::vector<float> expected_2 = {5e-05f, 0.0001f, 0.99985f};

    // predict
    const float *output_2 = model->predict(input_2.data());

    float sum_2 = 0.0f;
    for (int i = 0; i < outputSize; ++i) {
        EXPECT_NEAR(expected_2[i], output_2[i], 1e-5f);
        sum_2 += output_2[i];
    }

    EXPECT_NEAR(sum_2, 1.0f, 1e-5f);

    commonTestTeardown(model);
}

TEST_F(ModelTest, TestLoadWeights) {
    CUDANet::Model *model = commonTestSetup();

    model->loadWeights("../test/resources/model.bin");

    CUDANet::Layers::WeightedLayer *convLayer =
        dynamic_cast<CUDANet::Layers::WeightedLayer *>(model->getLayer("conv1")
        );
    EXPECT_NE(convLayer, nullptr);

    std::vector<float> convWeights  = convLayer->getWeights();
    std::vector<float> convExpected = getConv1Weights();

    for (int i = 0; i < convExpected.size(); ++i) {
        EXPECT_FLOAT_EQ(convExpected[i], convWeights[i]);
    }

    CUDANet::Layers::WeightedLayer *denseLayer =
        dynamic_cast<CUDANet::Layers::WeightedLayer *>(model->getLayer("linear")
        );
    EXPECT_NE(denseLayer, nullptr);

    std::vector<float> denseWeights  = denseLayer->getWeights();
    std::vector<float> denseExpected = getDenseWeights();

    for (int i = 0; i < denseExpected.size(); ++i) {
        EXPECT_FLOAT_EQ(denseExpected[i], denseWeights[i]);
    }

    commonTestTeardown(model);
}