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

    expected = {-21052.13086f,  97856.53125f,  114996.78125f,  -26694.66602f,
                -51989.99219f,  -41073.51562f, 52375.89844f,   -101566.1875f,
                110595.30469f,  -34081.6875f,  41151.85938f,   -116816.51562f,
                12594.64941f,   -86867.95312f, -103277.80469f, -31095.63281f,
                30530.58984f,   -47046.89844f, 94815.74219f,   -24208.12891f,
                -50130.52734f,  38272.71094f,  102970.35938f,  92221.41406f,
                20659.89258f,   -60365.08984f, 10940.85938f,   -48804.74219f,
                119315.45312f,  -49296.32031f, -113509.04688f, -19691.87305f,
                -62688.6875f,   -94743.73438f, -77935.0f,      -84231.10156f,
                58992.52344f,   -23301.23828f, -34058.94531f,  -27215.86328f,
                -103682.59375f, 13735.66992f,  7671.27002f,    -68139.50781f,
                -59972.78125f,  -7613.14844f,  -34182.88281f,  29532.60352f,
                -71745.53906f,  -137596.1875f};

    runTest();
}