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

    input = {-0.57666f, -1.62078f, -1.33839f, -0.84871f, 0.95678f,  0.5095f,
             0.46427f,  1.63513f,  -0.86174f, 1.37982f,  2.11955f,  -0.56532f,
             -1.41889f, -0.24865f, 0.75658f,  1.41115f,  -0.04036f, -0.13206f,
             0.54325f,  -0.90184f, -0.30188f, -2.06574f, -0.12676f, 0.38189f,
             1.7959f,   0.24076f,  1.17587f,  -0.21496f, 0.55819f,  0.21572f,
             -1.66043f, 1.24566f,  0.837f,    0.13259f,  -0.73019f, 0.87461f,
             1.38548f,  -0.48258f, -0.11748f, 0.4244f,   1.14489f,  0.28394f,
             -0.46594f, 1.18402f,  -0.91973f, 0.63682f,  -0.31897f, 0.80855f};

    expected = {4.80097f,    47.97113f,  2.84091f,   -8.17906f,  -27.73839f,
                -20.00487f,  25.13571f,  55.96156f,  -57.32637f, 46.79503f,
                30.84768f,   -2.27363f,  11.58069f,  -84.01064f, -86.74448f,
                -34.90844f,  -31.9425f,  -26.9795f,  43.22921f,  18.58556f,
                19.94732f,   9.99053f,   77.01399f,  -29.40551f, 22.79751f,
                -25.38616f,  78.7154f,   -3.62437f,  -7.37189f,  -37.58518f,
                28.78344f,   46.85378f,  -84.57623f, 0.10005f,   64.6466f,
                -10.21144f,  51.44754f,  15.37502f,  24.96819f,  -34.59124f,
                2.73933f,    -32.52842f, -1.32425f,  41.48183f,  -12.74939f,
                -102.07105f, 5.58513f,   9.73683f,   -25.97733f, -24.79673f};

    runTest();
}