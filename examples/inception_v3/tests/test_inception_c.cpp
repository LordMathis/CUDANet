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

    expected = {386048.9375f,   -721189.75f,    1039515.0625f, -92812.95312f,
                533437.75f,     -617244.0f,     -81946.21094f, -775994.25f,
                -653376.0f,     -690453.25f,    218790.28125f, 454025.3125f,
                947592.375f,    280879.25f,     -61118.59375f, -88742.75781f,
                -458026.0625f,  82204.71875f,   -297425.9375f, 114420.0625f,
                397277.71875f,  593181.375f,    582754.125f,   -614345.1875f,
                -173317.15625f, -220982.48438f, -932588.5625f, 339467.5625f,
                917578.125f,    -95884.16406f,  83229.875f,    434552.375f,
                231232.1875f,   142239.71875f,  -264704.5f,    854149.0f,
                462348.21875f,  33728.0f,       24409.39062f,  -509526.3125f,
                -279235.5625f,  570330.0f,      103149.71875f, 26780.33984f,
                -328880.71875f, 1027994.8125f,  -585315.0f,    -210921.71875f,
                492957.53125f,  -122604.57031f};

    runTest();
}
