#include <gtest/gtest.h>

#include "cuda_helper.cuh"
#include "input.cuh"

TEST(InputLayerTest, Init) {
    std::vector<float> input = {0.573f, 0.619f, 0.732f, 0.055f, 0.243f, 0.316f};
    CUDANet::Layers::Input inputLayer(6);
    float*                 d_output = inputLayer.forward(input.data());

    std::vector<float> output(6);
    CUDA_CHECK(cudaMemcpy(
        output.data(), d_output, sizeof(float) * 6, cudaMemcpyDeviceToHost
    ));
    EXPECT_EQ(input, output);
}