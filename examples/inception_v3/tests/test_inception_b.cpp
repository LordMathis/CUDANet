#include <gtest/gtest.h>

#include <inception_v3.hpp>

#include "test_fixture.hpp"

class InceptionBModel : public CUDANet::Model {
  public:
    InceptionBModel(
        const shape2d inputShape,
        const int     inputChannels,
        const int     outputSize
    )
        : CUDANet::Model(inputShape, inputChannels, outputSize) {
        inception_b =
            new InceptionB(inputShape, inputChannels, "inception_block.");
        addLayer("", inception_b);
        fc = new CUDANet::Layers::Dense(
            inception_b->getOutputSize(), 50,
            CUDANet::Layers::ActivationType::NONE
        );
        addLayer("fc", fc);
    };

    float *predict(const float *input) override {
        float *d_input = inputLayer->forward(input);
        d_input        = inception_b->forward(d_input);
        d_input        = fc->forward(d_input);
        return outputLayer->forward(d_input);
    }

  private:
    InceptionB             *inception_b;
    CUDANet::Layers::Dense *fc;
};

TEST_F(InceptionBlockTest, InceptionBTest) {
    inputShape    = {4, 4};
    inputChannels = 3;
    outputSize    = 50;

    model = new InceptionBModel(inputShape, inputChannels, outputSize);
    model->loadWeights("../tests/resources/inception_b.bin");

    input = {-0.74714f, 0.70954f,  1.21276f,  0.55136f,  -2.02672f, 0.42895f,
             -0.61635f, -0.37651f, -2.0056f,  2.06836f,  -0.62239f, -1.18187f,
             1.73166f,  -0.44721f, 1.42673f,  -0.75294f, 2.85223f,  1.33567f,
             -1.28691f, -0.59681f, 0.49173f,  -0.7806f,  -0.28427f, 0.01525f,
             -1.70295f, -0.0927f,  -0.04545f, -1.0465f,  0.22519f,  1.69716f,
             -0.33748f, 0.15713f,  -0.83215f, -0.54749f, -0.66072f, -0.18568f,
             0.98517f,  0.32883f,  -0.28338f, 0.81102f,  0.70454f,  0.84246f,
             0.93766f,  -0.83322f, 0.58987f,  1.23888f,  -0.6962f,  0.68079f};

    expected = {-19.94974f,  93.30141f,   85.76035f,   214.1964f,   -4.30855f,
                -92.65581f,  -37.0993f,   -96.92029f,  -145.99411f, 177.49068f,
                185.37115f,  263.2403f,   -158.78972f, -435.17844f, 208.36617f,
                -481.91907f, 179.2296f,   42.68469f,   661.90039f,  192.62759f,
                146.78622f,  59.37774f,   -107.44885f, 578.51874f,  745.61536f,
                528.30847f,  -54.60599f,  237.63603f,  198.97778f,  9.95003f,
                61.68781f,   -156.87708f, -166.12646f, 294.20853f,  382.63782f,
                53.15688f,   -74.18913f,  7.70657f,    202.17197f,  -121.06818f,
                32.45838f,   -401.04041f, 94.91491f,   240.54332f,  171.52502f,
                -121.58651f, 419.89447f,  161.91119f,  -13.53201f,  -74.76675f};

    runTest();
}