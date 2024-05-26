#include "cuda_helper.cuh"
#include "max_pooling.cuh"
#include "pooling.cuh"

using namespace CUDANet::Layers;

MaxPooling2d::MaxPooling2d(
    dim2d          inputSize,
    int            nChannels,
    dim2d          poolingSize,
    dim2d          stride,
    dim2d          padding,
    ActivationType activationType
)
    : inputSize(inputSize),
      nChannels(nChannels),
      poolingSize(poolingSize),
      stride(stride),
      padding(padding) {
    outputSize = {
        (inputSize.first + 2 * padding.first - poolingSize.first) /
                stride.first +
            1,
        (inputSize.second + 2 * padding.second - poolingSize.second) /
                stride.second +
            1
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

MaxPooling2d::~MaxPooling2d() {
    cudaFree(d_output);
    delete activation;
}

float* MaxPooling2d::forward(const float* d_input) {
    dim3 block(8, 8, 8);
    dim3 grid(
        (outputSize.first + block.x - 1) / block.x,
        (outputSize.second + block.y - 1) / block.y,
        (nChannels + block.z - 1) / block.z
    );

    Kernels::max_pooling<<<grid, block>>>(
        d_input, d_output, inputSize, outputSize, nChannels, poolingSize,
        stride, padding
    );
    CUDA_CHECK(cudaGetLastError());

    activation->activate(d_output);
    CUDA_CHECK(cudaDeviceSynchronize());

    return d_output;
}

int MaxPooling2d::getOutputSize() {
    return outputSize.first * outputSize.second * nChannels;
}

int MaxPooling2d::getInputSize() {
    return inputSize.first * inputSize.second * nChannels;
}

dim2d MaxPooling2d::getOutputDims() {
    return outputSize;
}