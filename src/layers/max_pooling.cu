#include "max_pooling.cuh"
#include "cuda_helper.cuh"
#include "pooling.cuh"

using namespace CUDANet::Layers;


MaxPooling2D::MaxPooling2D(
        int            inputSize,
        int            nChannels,
        int            poolingSize,
        int            stride,
        ActivationType activationType
    )
    : inputSize(inputSize), nChannels(nChannels), poolingSize(poolingSize), stride(stride) {


    outputSize  = (inputSize - 1) / stride + 1;

    activation = Activation(
        activationType, outputSize * outputSize * nChannels
    );

    d_output = nullptr;
    CUDA_CHECK(cudaMalloc(
        (void**)&d_output, sizeof(float) * outputSize * outputSize * nChannels
    ));
}


MaxPooling2D::~MaxPooling2D() {
    cudaFree(d_output);
}


float* MaxPooling2D::forward(const float* d_input) {

    dim3 block(8,8,8);
    dim3 grid(
        (outputSize + block.x - 1) / block.x,
        (outputSize + block.y - 1) / block.y,
        (nChannels + block.z - 1) / block.z
    );

    Kernels::max_pooling<<<grid, block>>>(
        d_input, d_output, inputSize, outputSize, nChannels, poolingSize, stride
    );
    CUDA_CHECK(cudaGetLastError());

    activation.activate(d_output);
    CUDA_CHECK(cudaDeviceSynchronize());

    return d_output;
}