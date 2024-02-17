#include "gtest/gtest.h"
#include "dense.h"
#include "test_cublas_fixture.h"

class DenseLayerTest : public CublasTestFixture {
protected:
};


TEST_F(DenseLayerTest, Forward) {

    Layers::Dense denseLayer(3, 2, cublasHandle);

    // Create input and output arrays
    float input[3] = {1.0f, 2.0f, 3.0f};
    float output[2] = {0.0f, 0.0f};

    // Perform forward pass
    denseLayer.forward(input, output);

    // Check if the output is a zero vector
    EXPECT_FLOAT_EQ(output[0], 0.0f);
    EXPECT_FLOAT_EQ(output[1], 0.0f);
}
