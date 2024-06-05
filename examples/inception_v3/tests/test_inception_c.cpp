#include <gtest/gtest.h>

#include <inception_v3.hpp>

#include "test_fixture.hpp"

class InceptionCModel : public CUDANet::Model {
  public:
    InceptionCModel(
        const shape2d inputShape,
        const int     inputChannels,
        const int     outputSize
    )
        : CUDANet::Model(inputShape, inputChannels, outputSize) {
        inception_c =
            new InceptionC(inputShape, inputChannels, 64, "inception_block.");
        addLayer("", inception_c);
        fc = new CUDANet::Layers::Dense(
            inception_c->getOutputSize(), 50,
            CUDANet::Layers::ActivationType::NONE
        );
        addLayer("fc", fc);
    };

    float *predict(const float *input) override {
        float *d_input = inputLayer->forward(input);
        d_input        = inception_c->forward(d_input);
        d_input        = fc->forward(d_input);
        return outputLayer->forward(d_input);
    }

  private:
    InceptionC             *inception_c;
    CUDANet::Layers::Dense *fc;
};

TEST_F(InceptionBlockTest, InceptionCTest) {
    inputShape    = {4, 4};
    inputChannels = 3;
    outputSize    = 50;

    model = new InceptionCModel(inputShape, inputChannels, outputSize);
    model->loadWeights("../tests/resources/inception_c.bin");

    input = {-0.50514f, 1.61672f,  -0.53398f, -0.00431f, 0.30868f,  -1.17392f,
             -2.15633f, -1.34838f, 0.02578f,  1.33409f,  0.25805f,  -0.93777f,
             0.5875f,   -1.2065f,  0.15659f,  -1.8198f,  0.22255f,  0.52929f,
             -0.83366f, -0.38078f, 0.02898f,  0.48225f,  0.14351f,  -1.08989f,
             -2.06598f, -0.25796f, -0.79819f, 0.38487f,  -1.02079f, 0.41253f,
             0.53986f,  -0.65411f, -0.32057f, 2.15608f,  -0.99935f, 0.10825f,
             -1.60163f, -0.27932f, -0.20508f, 1.31193f,  -0.7601f,  -0.0586f,
             -0.21923f, -0.85385f, -1.10512f, -0.22181f, 0.94507f,  -0.09808f};

    expected = {-9231.45508f,  -11854.50684f, -2690.15942f, -6366.60303f,
                -6953.01855f,  -2204.80371f,  1670.89551f,  18207.81641f,
                -8896.50977f,  10661.94434f,  3338.31055f,  -3853.95947f,
                1445.87354f,   -9627.54297f,  4166.00635f,  -22477.38477f,
                11400.2207f,   8139.3877f,    8114.41602f,  -2006.37793f,
                -9130.33398f,  10554.69824f,  5194.41016f,  -7031.67969f,
                -10880.09277f, -4093.95068f,  6500.65967f,  -459.13672f,
                -10640.70215f, 6096.37842f,   12178.46094f, 5894.95117f,
                -3034.80225f,  -5177.80518f,  -6112.60449f, -7296.75879f,
                -1134.77344f,  -13472.27637f, 8982.56543f,  -3773.67334f,
                -4207.74609f,  -4001.82129f,  -6682.51953f, -12314.57617f,
                -6180.21875f,  -886.62231f,   5490.0752f,   4868.64893f,
                -12725.73633f, -3121.33716f};

    runTest();
}
