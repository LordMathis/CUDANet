#include "avg_pooling.cuh"
#include "cuda_helper.cuh"
#include "pooling.cuh"

using namespace CUDANet::Layers;

AvgPooling2D::AvgPooling2D(
    int            inputSize,
    int            nChannels,
    int            poolingSize,
    int            stride,
    ActivationType activationType
)
    : inputSize(inputSize),
      nChannels(nChannels),
      poolingSize(poolingSize),
      stride(stride) {
    outputSize = (inputSize - poolingSize) / stride + 1;

    activation =
        Activation(activationType, outputSize * outputSize * nChannels);

    d_output = nullptr;
    CUDA_CHECK(cudaMalloc(
        (void**)&d_output, sizeof(float) * outputSize * outputSize * nChannels
    ));

    gridSize =
        (outputSize * outputSize * nChannels + BLOCK_SIZE - 1) / BLOCK_SIZE;
}

AvgPooling2D::~AvgPooling2D() {
    cudaFree(d_output);
}

float* AvgPooling2D::forward(const float* d_input) {

    dim3 block(8, 8, 8);
    dim3 grid(
        (outputSize + block.x - 1) / block.x,
        (outputSize + block.y - 1) / block.y,
        (nChannels + block.z - 1) / block.z
    );

    Kernels::avg_pooling<<<grid, block>>>(
        d_input, d_output, inputSize, outputSize, nChannels, poolingSize, stride
    );

    return d_output;
}