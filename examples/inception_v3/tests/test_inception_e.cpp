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

    expected = {1614.15283f,   -11319.01855f, 614.40479f,    5280.0293f,
                1914.45007f,   -2937.50317f,  -11177.16113f, 3215.01245f,
                6249.16992f,   5654.91357f,   -11702.27148f, 13057.32422f,
                8665.35742f,   3911.11743f,   5239.45947f,   -11552.88477f,
                -8056.7666f,   -16426.19922f, -1383.04346f,  6573.53125f,
                -12226.16992f, -6641.0957f,   -9614.80078f,  -9313.30273f,
                7023.68848f,   2089.5752f,    1095.53369f,   -1387.65698f,
                -7928.21729f,  -9489.18848f,  4159.78613f,   -690.03442f,
                -8356.81738f,  12364.08203f,  8226.95703f,   8822.66602f,
                -5462.90381f,  -1037.42773f,  12958.68555f,  -666.58423f,
                2032.38574f,   -9534.14062f,  -947.41333f,   689.37158f,
                4585.76465f,   -23245.36719f, 975.83398f,    -1253.45703f,
                -14745.35059f, -2588.05493f};

    runTest();
}