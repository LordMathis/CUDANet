#include <gtest/gtest.h>

#include <inception_v3.hpp>

#include "test_fixture.hpp"

class InceptionEModel : public CUDANet::Model {
  public:
    InceptionEModel(
        const shape2d inputShape,
        const int     inputChannels,
        const int     outputSize
    )
        : CUDANet::Model(inputShape, inputChannels, outputSize) {
        inception_e =
            new InceptionE(inputShape, inputChannels, "inception_block.");
        addLayer("", inception_e);
        fc = new CUDANet::Layers::Dense(
            inception_e->getOutputSize(), 50,
            CUDANet::Layers::ActivationType::NONE
        );
        addLayer("fc", fc);
    };

    float *predict(const float *input) override {
        float *d_input = inputLayer->forward(input);
        d_input        = inception_e->forward(d_input);
        d_input        = fc->forward(d_input);
        return outputLayer->forward(d_input);
    }

  private:
    InceptionE             *inception_e;
    CUDANet::Layers::Dense *fc;
};

TEST_F(InceptionBlockTest, InceptionETest) {
    inputShape    = {4, 4};
    inputChannels = 3;
    outputSize    = 50;

    model = new InceptionEModel(inputShape, inputChannels, outputSize);
    model->loadWeights("../tests/resources/inception_e.bin");

    input = {1.85083f,  0.11234f,  0.05994f,  -1.02453f, 0.21205f,  -0.67387f,
             0.66981f,  -0.40378f, 0.34194f,  0.92048f,  0.87556f,  0.81094f,
             -1.55728f, -0.70326f, -0.25078f, -0.10276f, 1.10463f,  -2.40992f,
             -1.7226f,  -0.18546f, 0.14397f,  -1.24784f, -0.35248f, -1.28729f,
             0.44803f,  1.68539f,  -1.05037f, 0.32115f,  -0.12896f, 1.02391f,
             0.95329f,  -0.81876f, -1.05828f, 0.09282f,  -0.38344f, 2.05074f,
             2.1034f,   1.65832f,  1.63788f,  -1.32596f, -1.43412f, -1.28353f,
             0.70226f,  0.9459f,   0.8579f,   0.15361f,  0.34449f,  -1.70587f};

    expected = {-52475.21094f,  -45850.59766f,  25258.94727f,  -123668.88281f,
                -124592.32812f, 120878.47656f,  69247.67188f,  3390.39258f,
                -17620.58594f,  5239.70117f,    -30841.2793f,  -134645.84375f,
                -71254.0f,      -69958.625f,    27372.9668f,   -10891.0293f,
                52875.20703f,   810.01172f,     -57457.20312f, -26664.05469f,
                -8147.90527f,   -139440.09375f, -71311.84375f, -53446.54688f,
                25358.27148f,   -42854.97656f,  57698.98438f,  63391.79688f,
                54427.98438f,   89160.73438f,   79430.96094f,  -51700.30469f,
                29048.21094f,   -28000.3418f,   -29570.61133f, -16047.83691f,
                -69285.42188f,  -13865.00391f,  17681.38672f,  -45284.46484f,
                42490.97656f,   30390.58203f,   21886.40039f,  -89973.20312f,
                75571.00781f,   19183.16797f,   -37130.51562f, 12787.17383f,
                59336.42578f,   -88201.78125f};

    runTest();
}