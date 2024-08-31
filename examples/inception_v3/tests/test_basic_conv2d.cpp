#include <gtest/gtest.h>

#include <inception_v3.hpp>

#include "test_fixture.hpp"

class BasicConv2dModel : public CUDANet::Model {
  public:
    BasicConv2dModel(
        const shape2d inputShape,
        const int     inputChannels,
        const int     outputChannels,
        const int     outputSize
    )
        : CUDANet::Model(inputShape, inputChannels, outputSize) {
        basic_conv2d = new BasicConv2d(
            inputShape, inputChannels, outputChannels, {3, 3}, {2, 2}, {0, 0},
            "inception_block."
        );
        addLayer("", basic_conv2d);
        fc = new CUDANet::Layers::Dense(
            basic_conv2d->getOutputSize(), 50,
            CUDANet::Layers::ActivationType::NONE
        );
        addLayer("fc", fc);
    };

    float *predict(const float *input) override {
        float *d_input = inputLayer->forward(input);
        d_input        = basic_conv2d->forward(d_input);
        d_input        = fc->forward(d_input);
        return outputLayer->forward(d_input);
    }

  private:
    BasicConv2d            *basic_conv2d;
    CUDANet::Layers::Dense *fc;
};

TEST_F(InceptionBlockTest, BasicConv2dTest) {
    inputShape    = {4, 4};
    inputChannels = 3;
    outputSize    = 50;

    int outputChannels = 32;

    model = new BasicConv2dModel(
        inputShape, inputChannels, outputChannels, outputSize
    );
    model->loadWeights("../tests/resources/basic_conv2d.bin");

    input = {3.11462f,  -0.81077f, -1.10521f, -0.72331f, 0.78823f,  1.36453f,
             0.37365f,  -1.00043f, 0.00156f,  -0.13156f, 0.10315f,  -0.36979f,
             -0.7116f,  -0.1203f,  1.23831f,  0.19852f,  -0.79851f, 0.27605f,
             0.09819f,  2.48209f,  -1.20067f, 1.02096f,  -0.38697f, -0.05689f,
             -0.27344f, -0.06105f, 0.53209f,  -0.89718f, -1.30166f, -1.37283f,
             1.69093f,  -0.4622f,  0.20359f,  -1.03283f, 1.13048f,  -0.5703f,
             -2.10094f, 0.38992f,  0.08734f,  -0.85736f, -0.27462f, 0.44321f,
             0.95911f,  1.33195f,  0.77331f,  3.0567f,   -2.4878f,  -1.69617f};

    expected = {-31.72671f, -26.44757f, -10.69324f, 34.00166f,  17.61435f,
                27.69349f,  3.65285f,   -23.72797f, -4.0109f,   15.94939f,
                -3.76765f,  24.18737f,  9.72213f,   0.57065f,   29.3591f,
                3.71433f,   18.16798f,  -6.33768f,  -21.24062f, 1.64767f,
                -9.12255f,  -17.8523f,  7.95207f,   5.59372f,   20.16168f,
                2.86041f,   -16.32113f, 29.07518f,  4.09777f,   8.65735f,
                1.00664f,   -23.30697f, 7.38348f,   24.17699f,  -15.56942f,
                9.82751f,   -8.22256f,  0.88987f,   4.39324f,   11.62732f,
                -6.18404f,  11.78396f,  -37.57986f, 28.56649f,  -16.00127f,
                -30.68929f, 30.1704f,   14.04265f,  -2.70738f,  4.89308f};
    runTest();
}