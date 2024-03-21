#include "model.hpp"

#include "input.cuh"
#include "layer.cuh"

using namespace CUDANet;

Model::Model(const int inputSize, const int inputChannels, const int outputSize)
    : inputSize(inputSize),
      inputChannels(inputChannels),
      outputSize(outputSize),
      layers(std::vector<Layers::SequentialLayer*>()),
      layerMap(std::unordered_map<std::string, Layers::WeightedLayer*>()) {
    inputLayer  = new Layers::Input(inputSize * inputSize * inputChannels);
    outputLayer = new Layers::Output(outputSize);
};

Model::Model(const Model& other)
    : inputSize(other.inputSize),
      inputChannels(other.inputChannels),
      outputSize(other.outputSize),
      layers(std::vector<Layers::SequentialLayer*>()),
      layerMap(std::unordered_map<std::string, Layers::WeightedLayer*>()) {
    inputLayer  = new Layers::Input(*other.inputLayer);
    outputLayer = new Layers::Output(*other.outputLayer);
}

Model::~Model(){};

float* Model::predict(const float* input) {
    float* d_input = inputLayer->forward(input);

    for (auto& layer : layers) {
        d_input = layer->forward(d_input);
    }

    return outputLayer->forward(d_input);
}

void Model::addLayer(const std::string& name, Layers::SequentialLayer* layer) {
    layers.push_back(layer);

    Layers::WeightedLayer* wLayer = dynamic_cast<Layers::WeightedLayer*>(layer);

    if (wLayer != nullptr) {
        layerMap[name] = wLayer;
    }
}