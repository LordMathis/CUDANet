#include <gtest/gtest.h>

#include <inception_v3.hpp>

#include "test_fixture.hpp"

class InceptionAModel : public CUDANet::Model {
  public:
    InceptionAModel(
        const shape2d inputShape,
        const int     inputChannels,
        const int     outputSize
    )
        : CUDANet::Model(inputShape, inputChannels, outputSize) {
        inception_a =
            new InceptionA(inputShape, inputChannels, 6, "inception_block.");
        addLayer("", inception_a);
        fc = new CUDANet::Layers::Dense(
            inception_a->getOutputSize(), 50,
            CUDANet::Layers::ActivationType::NONE
        );
        addLayer("fc", fc);
    };

    float *predict(const float *input) override {
        float *d_input = inputLayer->forward(input);
        d_input        = inception_a->forward(d_input);
        d_input        = fc->forward(d_input);
        return outputLayer->forward(d_input);
    }

  private:
    InceptionA             *inception_a;
    CUDANet::Layers::Dense *fc;
};

TEST_F(InceptionBlockTest, InceptionATest) {
    inputShape    = {4, 4};
    inputChannels = 3;
    outputSize    = 50;

    model = new InceptionAModel(inputShape, inputChannels, outputSize);
    model->loadWeights("../tests/resources/inception_a.bin");

    input = {0.0961f,   -2.01883f, -1.09473f, -0.86616f, -0.11743f, 0.30927f,
             0.89226f,  -0.86077f, -0.54149f, 0.06902f,  0.41018f,  1.12389f,
             -1.94536f, -1.01233f, -1.93574f, -0.37831f, 0.30134f,  -0.71024f,
             -0.85796f, -1.60188f, 0.16672f,  0.04074f,  -0.17714f, -0.45344f,
             -0.67299f, 0.39537f,  1.36158f,  -0.04113f, 0.04399f,  0.08246f,
             0.28653f,  0.00399f,  -0.76861f, -0.1379f,  1.23108f,  1.15452f,
             -1.67101f, 0.892f,    -0.22458f, 1.08748f,  0.63386f,  1.91399f,
             0.69495f,  1.39091f,  -0.38628f, -1.3974f,  -0.74191f, 0.40352f};

    expected = {
        5525.64062f,  -761.25549f,  -2780.82837f, 1123.72534f,  -5405.26465f,
        -840.91406f,  3590.53394f,  -3732.77344f, 945.03845f,   1172.73401f,
        -1085.39026f, 1690.71399f,  2042.38208f,  -5948.82129f, 2648.69897f,
        6884.2876f,   -1833.52173f, 3289.4668f,   110.44409f,   -1192.91907f,
        6087.70117f,  8234.74316f,  4488.75488f,  7.75244f,     -2987.04834f,
        -5129.2124f,  -3235.24585f, -336.58179f,  2506.66943f,  598.82483f,
        488.68921f,   913.30005f,   -6063.51318f, 3496.71753f,  -4504.59473f,
        1082.13867f,  3889.91968f,  1248.47168f,  -742.7981f,   -2244.45215f,
        -1985.24561f, 14646.59766f, -310.81506f,  4763.40527f,  -3007.1792f,
        382.23853f,   -2357.31445f, 3503.68457f,  -5159.74902f, -5777.59863f
    };

    runTest();
}