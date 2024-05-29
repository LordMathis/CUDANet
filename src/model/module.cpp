#include "module.hpp"

#include "cuda_helper.cuh"

using namespace CUDANet;

void Module::addLayer(const std::string& name, Layers::SequentialLayer* layer) {
    const Module* module = dynamic_cast<Module*>(layer);

    if (module != nullptr) {
        for (const auto& moduleLayer : module->getLayers()) {
            layerMap[moduleLayer.first] = moduleLayer.second;
            layers.push_back({moduleLayer.first, moduleLayer.second});
        }

        return;
    }

    layers.push_back({name, layer});
    layerMap[name] = layer;

    // std::cout << "Wat?! - module" << name << std::endl;
}

Layers::SequentialLayer* Module::getLayer(const std::string& name) {
    return layerMap[name];
}

const std::unordered_map<std::string, Layers::SequentialLayer*>&
Module::getLayers() const {
    return layerMap;
}

int Module::getInputSize() {
    return inputSize * inputSize * inputChannels;
}

int Module::getOutputSize() {
    return outputSize * outputSize * outputChannels;
}