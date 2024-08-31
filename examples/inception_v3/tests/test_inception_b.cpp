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

    expected = {
        -3051.75049f, -4.80383f,    -1978.21191f, -522.09924f,  1021.27625f,
        2102.80273f,  875.50775f,   -466.79095f,  -706.03009f,  2394.21826f,
        1953.83984f,  -1130.63367f, 1569.5769f,   -12.87457f,   -502.60977f,
        593.30615f,   104.63843f,   -1463.10815f, -1655.98389f, -1414.4104f,
        -366.82794f,  -2672.62769f, -1057.31287f, 832.19531f,   -116.99335f,
        476.57092f,   -1208.35327f, 357.08228f,   2724.59375f,  1238.1272f,
        1124.98877f,  566.73798f,   -2852.34058f, -98.82605f,   -4457.8584f,
        1228.86597f,  1112.53467f,  2053.17212f,  396.1055f,    -2534.39136f,
        1349.99756f,  -792.96722f,  -1477.9967f,  450.82751f,   -297.31879f,
        294.22925f,   3548.19165f,  -63.16211f,   -1651.43335f, 51.88046f
    };

    runTest();
}