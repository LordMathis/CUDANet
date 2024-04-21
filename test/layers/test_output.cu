#include <cuda_runtime_api.h>
#include <gtest/gtest.h>

#include "output.cuh"

TEST(OutputLayerTest, OutputForward) {
    cudaError_t cudaStatus;

    std::vector<float> input = {0.573f, 0.619f, 0.732f, 0.055f, 0.243f, 0.316f};
    float*             d_input;
    cudaStatus = cudaMalloc((void**)&d_input, sizeof(float) * 6);
    EXPECT_EQ(cudaStatus, cudaSuccess);
    cudaStatus = cudaMemcpy(
        d_input, input.data(), sizeof(float) * 6, cudaMemcpyHostToDevice
    );
    EXPECT_EQ(cudaStatus, cudaSuccess);

    CUDANet::Layers::Output outputLayer(6);
    float* h_output = outputLayer.forward(d_input);

    for (int i = 0; i < 6; ++i) {
        EXPECT_EQ(input[i], h_output[i]);
    }

    cudaFree(d_input);
    
}