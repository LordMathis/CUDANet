#include "module.hpp"

#include <algorithm>

#include "cuda_helper.cuh"

using namespace CUDANet;

void Module::addLayer(const std::string& name, Layers::SequentialLayer* layer) {
    const Module* module = dynamic_cast<Module*>(layer);

    if (module != nullptr) {
        for (const auto& moduleLayer : module->getLayers()) {
            layers.push_back({moduleLayer.first, moduleLayer.second});
        }

        return;
    }

    layers.push_back({name, layer});
}

const std::vector<std::pair<std::string, Layers::SequentialLayer*>>&
Module::getLayers() const {
    return layers;
}

int Module::getInputSize() {
    return inputSize;
}

int Module::getOutputSize() {
    return outputSize;
}