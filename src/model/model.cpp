#include "model.hpp"

#include <fstream>
#include <iostream>
#include <string>
#include <unordered_map>
#include <vector>

#include "input.cuh"
#include "layer.cuh"

using namespace CUDANet;

Model::Model(const int inputSize, const int inputChannels, const int outputSize)
    : inputSize(inputSize),
      inputChannels(inputChannels),
      outputSize(outputSize),
      layers(std::vector<Layers::SequentialLayer*>()),
      layerMap(std::unordered_map<std::string, Layers::SequentialLayer*>()) {
    inputLayer  = new Layers::Input(inputSize * inputSize * inputChannels);
    outputLayer = new Layers::Output(outputSize);
};

Model::Model(const Model& other)
    : inputSize(other.inputSize),
      inputChannels(other.inputChannels),
      outputSize(other.outputSize),
      layers(std::vector<Layers::SequentialLayer*>()),
      layerMap(std::unordered_map<std::string, Layers::SequentialLayer*>()) {
    inputLayer  = new Layers::Input(*other.inputLayer);
    outputLayer = new Layers::Output(*other.outputLayer);
}

Model::~Model() {
    delete inputLayer;
    delete outputLayer;
    for (auto layer : layers) {
        delete layer;
    }
};

float* Model::predict(const float* input) {
    float* d_input = inputLayer->forward(input);

    for (auto& layer : layers) {
        d_input = layer->forward(d_input);
    }

    return outputLayer->forward(d_input);
}

void Model::addLayer(const std::string& name, Layers::SequentialLayer* layer) {
    layers.push_back(layer);
    layerMap[name] = layer;
}

Layers::SequentialLayer* Model::getLayer(const std::string& name) {
    return layerMap[name];
}

void Model::loadWeights(const std::string& path) {
    std::ifstream file(path, std::ios::binary);

    if (!file.is_open()) {
        std::cerr << "Failed to open file: " << path << std::endl;
        return;
    }

    int64_t headerSize;
    file.read(reinterpret_cast<char*>(&headerSize), sizeof(headerSize));

    std::string header(headerSize, '\0');
    file.read(&header[0], headerSize);

    std::vector<TensorInfo> tensorInfos;
    size_t pos = 0;

    while (pos < header.size()) {
        size_t nextPos = header.find('\n', pos);
        if (nextPos == std::string::npos)
            break;
        
        std::string line = header.substr(pos, nextPos - pos);
        pos = nextPos + 1;

        size_t commaPos = line.find(',');
        if (commaPos == std::string::npos)
            continue;

        // Parse tensor name into name and type
        std::string nameStr = line.substr(0, commaPos);
        size_t dotPos = nameStr.find('.');
        if (dotPos == std::string::npos)
            continue;
        std::string name = nameStr.substr(0, dotPos);
        TensorType type = nameStr.substr(dotPos + 1) == "w" ? TensorType::WEIGHT : TensorType::BIAS;

        line = line.substr(commaPos + 1);

        commaPos = line.find(',');
        if (commaPos == std::string::npos)
            continue;

        int size = std::stoi(line.substr(0, commaPos));
        int offset = std::stoi(line.substr(commaPos + 1));

        tensorInfos.push_back({name, type, size, offset});
    }

    for (const auto& tensorInfo : tensorInfos) {
        std::vector<float> values(tensorInfo.size);

        file.seekg(tensorInfo.offset);
        file.read(reinterpret_cast<char*>(values.data()), tensorInfo.size * sizeof(float));

        if (layerMap.find(tensorInfo.name) != layerMap.end()) {

            Layers::WeightedLayer* wLayer = dynamic_cast<Layers::WeightedLayer*>(layerMap[tensorInfo.name]);

            if (wLayer == nullptr) {
                std::cerr << "Layer: " << tensorInfo.name << "does not have weights, skipping" << std::endl;
                continue;
            }

            if (tensorInfo.type == TensorType::WEIGHT) {
                wLayer->setWeights(values.data());
            } else if (tensorInfo.type == TensorType::BIAS) {
                wLayer->setBiases(values.data());
            }
        }
    }

    file.close();
}