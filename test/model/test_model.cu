#include <gtest/gtest.h>

#include "conv2d.cuh"
#include "dense.cuh"
#include "max_pooling.cuh"
#include "model.hpp"

TEST(Model, TestModelPredict) {
    int inputSize     = 6;
    int inputChannels = 2;
    int outputSize    = 3;

    int kernelSize = 3;
    int stride     = 1;
    int numFilters = 2;

    int poolingSize = 2;
    int poolingStride = 2;

    CUDANet::Model model(inputSize, inputChannels, outputSize);

    // Conv2d
    CUDANet::Layers::Conv2d conv2d(
        inputSize, inputChannels, kernelSize, stride, numFilters, CUDANet::Layers::Padding::VALID,
        CUDANet::Layers::ActivationType::NONE
    );
    // weights 6*6*2*2
    std::vector<float> conv2dWeights = {
        0.18313f, 0.53363f, 0.39527f, 0.27575f, 0.3433f,  0.41746f,
        0.16831f, 0.61693f, 0.54599f, 0.99692f, 0.77127f, 0.25146f,
        0.4206f,  0.16291f, 0.93484f, 0.79765f, 0.74982f, 0.78336f,
        0.6386f,  0.87744f, 0.33587f, 0.9691f,  0.68437f, 0.65098f,
        0.48153f, 0.97546f, 0.8026f,  0.36689f, 0.98152f, 0.37351f,
        0.68407f, 0.2684f,  0.2855f,  0.76195f, 0.67828f, 0.603f
    };
    conv2d.setWeights(conv2dWeights.data());
    model.addLayer("conv2d", &conv2d);

    // maxpool2d
    CUDANet::Layers::MaxPooling2D maxpool2d(
        inputSize - kernelSize + 1, numFilters, poolingSize, poolingStride, CUDANet::Layers::ActivationType::RELU
    );
    model.addLayer("maxpool2d", &maxpool2d);

    // dense
    CUDANet::Layers::Dense dense(
        8, 3, CUDANet::Layers::ActivationType::SOFTMAX
    );
    // dense weights 18*6
    std::vector<float> denseWeights = {
        0.36032f, 0.33115f, 0.02948f,
        0.09802f, 0.45072f, 0.56266f,
        0.43514f, 0.80946f, 0.43439f,
        0.90916f, 0.08605f, 0.07473f,
        0.94788f, 0.66168f, 0.34927f,
        0.09464f, 0.61963f, 0.73775f,
        0.51559f, 0.81916f, 0.64915f,
        0.03934f, 0.87608f, 0.68364f,
    };
    dense.setWeights(denseWeights.data());
    model.addLayer("dense", &dense);

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
    const float* output = model.predict(input.data());

    float sum = 0.0f;
    for (int i = 0; i < outputSize; ++i) {
        EXPECT_NEAR(expected[i], output[i], 1e-5f);
        sum += output[i];
    }

    EXPECT_NEAR(sum, 1.0f, 1e-5f);

    cudaDeviceReset();
}