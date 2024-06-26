#include <gtest/gtest.h>

#include <inception_v3.hpp>

#include "test_fixture.hpp"

class InceptionDModel : public CUDANet::Model {
  public:
    InceptionDModel(
        const shape2d inputShape,
        const int     inputChannels,
        const int     outputSize
    )
        : CUDANet::Model(inputShape, inputChannels, outputSize) {
        inception_d =
            new InceptionD(inputShape, inputChannels, "inception_block.");
        addLayer("", inception_d);
        fc = new CUDANet::Layers::Dense(
            inception_d->getOutputSize(), 50,
            CUDANet::Layers::ActivationType::NONE
        );
        addLayer("fc", fc);
    };

    float *predict(const float *input) override {
        float *d_input = inputLayer->forward(input);
        d_input        = inception_d->forward(d_input);
        d_input        = fc->forward(d_input);
        return outputLayer->forward(d_input);
    }

  private:
    InceptionD             *inception_d;
    CUDANet::Layers::Dense *fc;
};

TEST_F(InceptionBlockTest, InceptionDTest) {
    inputShape    = {4, 4};
    inputChannels = 3;
    outputSize    = 50;

    model = new InceptionDModel(inputShape, inputChannels, outputSize);
    model->loadWeights("../tests/resources/inception_d.bin");

    input = {0.11909f,  0.39688f,  -0.48224f, 2.5266f,   -0.72496f, -0.4192f,
             0.74736f,  1.15859f,  1.83703f,  -0.68566f, -1.18465f, 2.22915f,
             -0.48233f, -1.38496f, -0.23985f, -2.01891f, -0.61103f, -0.52143f,
             1.33207f,  0.11988f,  0.57453f,  0.26697f,  -0.4713f,  0.26025f,
             -0.55807f, -0.02672f, -0.18203f, 0.59285f,  0.28061f,  -0.69015f,
             -0.54148f, -1.64196f, -1.34975f, 0.4348f,   0.57761f,  1.47873f,
             -0.89471f, -0.0348f,  -1.49654f, -1.18578f, -2.013f,   -0.47656f,
             -0.16578f, 0.21603f,  -0.23605f, -0.53382f, -0.25789f, 2.30887f};

    expected = {
        -778.66046f,  2780.01416f, -908.03717f,  720.61572f,   975.1803f,
        -2017.04016f, 2678.03955f, -2089.99609f, -1231.16272f, 4078.28247f,
        -765.89209f,  -2531.9021f, -1590.11182f, 6677.42822f,  174.45618f,
        -1065.43262f, 4505.68066f, 3798.1748f,   1419.7229f,   2433.96948f,
        355.61597f,   1356.61279f, -2179.37061f, -973.08789f,  2414.1543f,
        -2190.11792f, -157.86133f, 1810.07166f,  2140.48706f,  8073.00488f,
        2629.58789f,  4686.91992f, -3285.09985f, 5723.23584f,  1181.26648f,
        -5476.90723f, 4895.85547f, -1787.32935f, 2138.9646f,   1336.84277f,
        -3492.97656f, 3706.74121f, 703.98871f,   -2263.92188f, 4441.91016f,
        -3471.9314f,  3354.59106f, 5038.75928f,  -3676.13037f, 563.34637f
    };

    runTest();
}