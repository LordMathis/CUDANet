#include "cuda_helper.cuh"
#include "max_pooling.cuh"
#include "pooling.cuh"

using namespace CUDANet::Layers;

MaxPooling2D::MaxPooling2D(
    dim2d          inputSize,
    int            nChannels,
    dim2d          poolingSize,
    dim2d          stride,
    ActivationType activationType
)
    : inputSize(inputSize),
      nChannels(nChannels),
      poolingSize(poolingSize),
      stride(stride) {
    outputSize = {
        (inputSize.first - poolingSize.first) / stride.first + 1,
        (inputSize.second - poolingSize.second) / stride.second + 1
    };

    activation =
        new Activation(activationType, outputSize.first * outputSize.second * nChannels);

    d_output = nullptr;
    CUDA_CHECK(cudaMalloc(
        (void**)&d_output, sizeof(float) * outputSize.first * outputSize.second * nChannels
    ));
}

MaxPooling2D::~MaxPooling2D() {
    cudaFree(d_output);
    delete activation;
}

float* MaxPooling2D::forward(const float* d_input) {
    dim3 block(8, 8, 8);
    dim3 grid(
        (outputSize.first + block.x - 1) / block.x,
        (outputSize.second + block.y - 1) / block.y,
        (nChannels + block.z - 1) / block.z
    );

    Kernels::max_pooling<<<grid, block>>>(
        d_input, d_output, inputSize, outputSize, nChannels, poolingSize, stride
    );
    CUDA_CHECK(cudaGetLastError());

    activation->activate(d_output);
    CUDA_CHECK(cudaDeviceSynchronize());

    return d_output;
}

int MaxPooling2D::getOutputSize() {
    return outputSize.first * outputSize.second * nChannels;
}

int MaxPooling2D::getInputSize() {
    return inputSize.first * inputSize.second * nChannels;
}