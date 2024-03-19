#include "max_pooling.cuh"
#include "cuda_helper.cuh"
#include "pooling.cuh"

using namespace CUDANet::Layers;


MaxPooling2D::MaxPooling2D(
        int            inputSize,
        int            nChannels,
        int            poolingSize,
        int            stride,
        Padding        padding,
        ActivationType activationType
    )
    : inputSize(inputSize), nChannels(nChannels), poolingSize(poolingSize), stride(stride) {


    switch (padding) {
        case SAME:
            outputSize  = inputSize;
            paddingSize = ((stride - 1) * inputSize - stride + poolingSize) / 2;
            break;

        case VALID:
            paddingSize = 0;
            outputSize  = (inputSize - poolingSize) / stride + 1;
            break;

        default:
            break;
    }

    activation = Activation(
        activationType, outputSize * outputSize * nChannels
    );

    d_output = nullptr;
    CUDA_CHECK(cudaMalloc(
        (void**)&d_output, sizeof(float) * outputSize * outputSize * nChannels
    ));

    gridSize = (outputSize * outputSize * nChannels + BLOCK_SIZE - 1) / BLOCK_SIZE;

}


MaxPooling2D::~MaxPooling2D() {
    cudaFree(d_output);
}


float* MaxPooling2D::forward(const float* d_input) {
    Kernels::max_pooling<<<gridSize, BLOCK_SIZE>>>(
        d_input, d_output, inputSize, nChannels, poolingSize, stride, paddingSize
    );

    return d_output;
}