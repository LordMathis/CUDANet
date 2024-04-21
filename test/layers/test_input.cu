#include <cuda_runtime_api.h>
#include <gtest/gtest.h>

#include "input.cuh"

TEST(InputLayerTest, InputForward) {
    std::vector<float> input = {0.573f, 0.619f, 0.732f, 0.055f, 0.243f, 0.316f};
    CUDANet::Layers::Input inputLayer(6);
    float*                 d_output = inputLayer.forward(input.data());

    std::vector<float> output(6);
    cudaError_t        cudaStatus = cudaMemcpy(
        output.data(), d_output, sizeof(float) * 6, cudaMemcpyDeviceToHost
    );
    EXPECT_EQ(cudaStatus, cudaSuccess);
    EXPECT_EQ(input, output);

    
}