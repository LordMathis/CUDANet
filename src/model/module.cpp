#include "module.hpp"

#include "cuda_helper.cuh"

using namespace CUDANet;

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