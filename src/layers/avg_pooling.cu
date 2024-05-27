#include "avg_pooling.cuh"
#include "cuda_helper.cuh"
#include "pooling.cuh"

using namespace CUDANet::Layers;

AvgPooling2d::AvgPooling2d(
    shape2d          inputSize,
    int            nChannels,
    shape2d          poolingSize,
    shape2d          stride,
    shape2d          padding,
    ActivationType activationType
)
    : inputSize(inputSize),
      nChannels(nChannels),
      poolingSize(poolingSize),
      stride(stride),
      padding(padding) {
    outputSize = {
        (inputSize.first + 2 * padding.first - poolingSize.first) / stride.first + 1,
        (inputSize.second + 2 * padding.second - poolingSize.second) / stride.second + 1
    };

    activation = new Activation(
        activationType, outputSize.first * outputSize.second * nChannels
    );

    d_output = nullptr;
    CUDA_CHECK(cudaMalloc(
        (void**)&d_output,
        sizeof(float) * outputSize.first * outputSize.second * nChannels
    ));
}

AvgPooling2d::~AvgPooling2d() {
    cudaFree(d_output);
    delete activation;
}

float* AvgPooling2d::forward(const float* d_input) {
    dim3 block(8, 8, 8);
    dim3 grid(
        (outputSize.first + block.x - 1) / block.x,
        (outputSize.second + block.y - 1) / block.y,
        (nChannels + block.z - 1) / block.z
    );

    Kernels::avg_pooling<<<grid, block>>>(
        d_input, d_output, inputSize, outputSize, nChannels, poolingSize,
        stride, padding
    );
    CUDA_CHECK(cudaGetLastError());

    activation->activate(d_output);
    CUDA_CHECK(cudaDeviceSynchronize());

    return d_output;
}

int AvgPooling2d::getOutputSize() {
    return outputSize.first * outputSize.second * nChannels;
}

int AvgPooling2d::getInputSize() {
    return inputSize.first * inputSize.second * nChannels;
}

shape2d AvgPooling2d::getOutputDims() {
    return outputSize;
}