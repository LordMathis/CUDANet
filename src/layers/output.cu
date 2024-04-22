#include "output.cuh"

#include "cuda_helper.cuh"

using namespace CUDANet::Layers;


Output::Output(int inputSize) : inputSize(inputSize) {
    h_output = (float*) malloc(sizeof(float) * inputSize);
}

Output::~Output() {
    free(h_output);
}

float* Output::forward(const float* input) {
    CUDA_CHECK(cudaMemcpy(
        h_output, input, sizeof(float) * inputSize, cudaMemcpyDeviceToHost
    ));
    CUDA_CHECK(cudaDeviceSynchronize());

    return h_output;
}

int Output::getOutputSize() {
    return inputSize;
}


int Output::getInputSize() {
    return inputSize;
}