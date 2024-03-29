#include <gtest/gtest.h>

#include "conv2d.cuh"
#include "dense.cuh"
#include "max_pooling.cuh"
#include "model.hpp"

TEST(Model, TestModelPredict) {
    int inputSize     = 6;
    int inputChannels = 2;
    int outputSize    = 6;

    CUDANet::Model model(inputSize, inputChannels, outputSize);

    // Conv2d
    CUDANet::Layers::Conv2d conv2d(
        inputSize, inputChannels, 3, 1, 2, CUDANet::Layers::Padding::VALID,
        CUDANet::Layers::ActivationType::NONE
    );
    // weights 6*6*2*2
    std::vector<float> conv2dWeights = {
        0.18313f, 0.53363f, 0.39527f, 0.27575f, 0.3433f,  0.41746f, 0.16831f,
        0.61693f, 0.54599f, 0.99692f, 0.77127f, 0.25146f, 0.4206f,  0.16291f,
        0.93484f, 0.79765f, 0.74982f, 0.78336f, 0.6386f,  0.87744f, 0.33587f,
        0.9691f,  0.68437f, 0.65098f, 0.48153f, 0.97546f, 0.8026f,  0.36689f,
        0.98152f, 0.37351f, 0.68407f, 0.2684f,  0.2855f,  0.76195f, 0.67828f,
        0.57567f, 0.6752f,  0.83416f, 0.62217f, 0.92441f, 0.96507f, 0.11171f,
        0.52438f, 0.90435f, 0.46854f, 0.59987f, 0.21747f, 0.82223f, 0.2709f,
        0.69207f, 0.16794f, 0.26679f, 0.49572f, 0.42392f, 0.49808f, 0.33058f,
        0.54071f, 0.83304f, 0.03446f, 0.65041f, 0.58601f, 0.7208f,  0.49659f,
        0.60447f, 0.70867f, 0.33336f, 0.0199f,  0.53188f, 0.15774f, 0.31791f,
        0.2611f,  0.66174f, 0.22588f, 0.95612f, 0.01209f, 0.2239f,  0.51731f,
        0.80405f, 0.09126f, 0.85215f, 0.01911f, 0.7448f,  0.61376f, 0.22161f,
        0.71483f, 0.36953f, 0.67528f, 0.77609f, 0.5933f,  0.66035f, 0.79205f,
        0.04973f, 0.78845f, 0.4023f,  0.55086f, 0.03391f, 0.06616f, 0.45673f,
        0.24497f, 0.87024f, 0.43097f, 0.24168f, 0.66931f, 0.03734f, 0.31513f,
        0.46383f, 0.74909f, 0.57525f, 0.5295f,  0.48876f, 0.89053f, 0.31964f,
        0.87242f, 0.08605f, 0.30465f, 0.8018f,  0.53419f, 0.73341f, 0.12307f,
        0.70645f, 0.40608f, 0.89397f, 0.97853f, 0.67084f, 0.47644f, 0.39974f,
        0.97879f, 0.86642f, 0.20244f, 0.66219f, 0.11623f, 0.18979f, 0.52886f,
        0.44583f, 0.41313f, 0.19766f, 0.47676f, 0.48318f, 0.02079f, 0.83777f,
        0.41167f, 0.57684f, 0.79578f, 0.17775f
    };
    conv2d.setWeights(conv2dWeights.data());
    model.addLayer("conv2d", &conv2d);

    // maxpool2d
    CUDANet::Layers::MaxPooling2D maxpool2d(
        6, 2, 2, 2, CUDANet::Layers::ActivationType::RELU
    );
    model.addLayer("maxpool2d", &maxpool2d);

    // dense
    CUDANet::Layers::Dense dense(
        18, 6, CUDANet::Layers::ActivationType::NONE
    );
    // dense weights 18*6
    std::vector<float> denseWeights = {
        0.36032f, 0.33115f, 0.02948f, 0.56265f, 0.23524f, 0.96589f, 0.09802f,
        0.45072f, 0.56266f, 0.5246f,  0.86663f, 0.30252f, 0.43514f, 0.80946f,
        0.43439f, 0.95206f, 0.5658f,  0.16344f, 0.90916f, 0.08605f, 0.07473f,
        0.95572f, 0.9127f,  0.96697f, 0.94788f, 0.66168f, 0.34927f, 0.86927f,
        0.10111f, 0.67001f, 0.09464f, 0.61963f, 0.73775f, 0.15255f, 0.37537f,
        0.72831f, 0.51559f, 0.81916f, 0.64915f, 0.23607f, 0.88699f, 0.39844f,
        0.03934f, 0.87608f, 0.68364f, 0.03633f, 0.11632f, 0.99925f, 0.86102f,
        0.6659f,  0.11022f, 0.47878f, 0.92411f, 0.38027f, 0.06771f, 0.99645f,
        0.47783f, 0.54653f, 0.41552f, 0.61055f, 0.50326f, 0.79817f, 0.20008f,
        0.32929f, 0.23562f, 0.0033f,  0.46628f, 0.04958f, 0.05235f, 0.28102f,
        0.45705f, 0.78327f, 0.91427f, 0.41122f, 0.08883f, 0.43558f, 0.14724f,
        0.74515f, 0.98215f, 0.50503f, 0.02887f, 0.25426f, 0.3463f,  0.81567f,
        0.84608f, 0.15469f, 0.6069f,  0.54311f, 0.77967f, 0.50657f, 0.18208f,
        0.7969f,  0.48401f, 0.36097f, 0.7563f,  0.50316f, 0.1134f,  0.98089f,
        0.97041f, 0.4832f,  0.79216f, 0.06572f, 0.09688f, 0.51555f, 0.1652f,
        0.73933f, 0.44365f, 0.66949f
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

    // predict
    const float* output = model.predict(input.data());

    // float sum = 0.0f;
    for (int i = 0; i < outputSize; ++i) {
        // sum += output[i];
        std::cout << output[i] << " ";
    }
    // EXPECT_NEAR(sum, 1.0f, 1e-5f);

    std::cout << std::endl;
}