#include "model.hpp"
#include "layer.cuh"
#include "input.cuh"

using namespace CUDANet;

Model::Model(const int inputSize, const int inputChannels)
    : inputSize(inputSize), inputChannels(inputChannels) {

    layerMap = std::map<std::string, Layers::WeightedLayer*>();
    layers   = std::vector<Layers::SequentialLayer*>();


    const int inputLayerSize = inputSize * inputSize * inputChannels;
    inputLayer = new Layers::Input(inputLayerSize);
};

Model::~Model(){};

float* Model::predict(const float* input) {

    float* d_input = inputLayer->forward(input);
    
    for (auto& layer : layers) {
        d_input = layer->forward(d_input);
    }

    return d_input;
}

void Model::addLayer(const std::string& name, Layers::SequentialLayer* layer) {
    layers.push_back(layer);

    if (dynamic_cast<Layers::WeightedLayer*>(layer) != nullptr) {
        layerMap[name] = dynamic_cast<Layers::WeightedLayer*>(layer);
    }
}