#include "model.hpp"
#include "layer.cuh"
#include "input.cuh"

using namespace CUDANet;

Model::Model(const int inputSize, const int inputChannels)
    : inputSize(inputSize), inputChannels(inputChannels) {

    layerMap = std::map<std::string, Layers::WeightedLayer*>();
    layers   = std::vector<Layers::SequentialLayer*>();


    const int inputLayerSize = inputSize * inputSize * inputChannels;
    Layers::Input* inputLayer = new Layers::Input(inputLayerSize);
    
    layers.push_back(inputLayer);
};

Model::~Model(){};

float* Model::predict(const float* input) {
    
    for (auto& layer : layers) {
        input = layer->forward(input);
    }

}

void Model::addLayer(const std::string& name, Layers::SequentialLayer* layer) {
    layers.push_back(layer);

    if (dynamic_cast<Layers::WeightedLayer*>(layer) != nullptr) {
        layerMap[name] = dynamic_cast<Layers::WeightedLayer*>(layer);
    }
}