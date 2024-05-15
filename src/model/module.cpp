#include "module.hpp"

#include "cuda_helper.cuh"

using namespace CUDANet;

Module::Module(
    const int inputSize,
    const int inputChannels,
    const int outputSize,
    const int outputChannels
)
    : inputSize(inputSize),
      inputChannels(inputChannels),
      outputSize(outputSize),
      outputChannels(outputChannels),
      layers(std::vector<std::pair<std::string, Layers::SequentialLayer*>>()),
      layerMap(std::unordered_map<std::string, Layers::SequentialLayer*>()) {
    d_output = nullptr;
    CUDA_CHECK(cudaMalloc(
        (void**)&d_output,
        sizeof(float) * outputSize * outputSize * outputChannels
    ));
}

Module::~Module() {
    cudaFree(d_output);
}

void Module::addLayer(const std::string& name, Layers::SequentialLayer* layer) {
    layers.push_back({ name, layer });
    layerMap[name] = layer;
}

Layers::SequentialLayer* Module::getLayer(const std::string& name) {
    return layerMap[name];
}

const std::unordered_map<std::string, Layers::SequentialLayer*>& Module::getLayers() const {
    return layerMap;
}

int Module::getInputSize() {
    return inputSize * inputSize * inputChannels;
}

int Module::getOutputSize() {
    return outputSize * outputSize * outputChannels;
}