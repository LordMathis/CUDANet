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
      layers(std::vector<std::pair<std::string, Layers::SequentialLayer*>>()),
      layerMap(std::unordered_map<std::string, Layers::SequentialLayer*>()) {
    inputLayer  = new Layers::Input(inputSize * inputSize * inputChannels);
    outputLayer = new Layers::Output(outputSize);
};

Model::Model(const Model& other)
    : inputSize(other.inputSize),
      inputChannels(other.inputChannels),
      outputSize(other.outputSize),
      layers(std::vector<std::pair<std::string, Layers::SequentialLayer*>>()),
      layerMap(std::unordered_map<std::string, Layers::SequentialLayer*>()) {
    inputLayer  = new Layers::Input(*other.inputLayer);
    outputLayer = new Layers::Output(*other.outputLayer);
}

Model::~Model() {
    delete inputLayer;
    delete outputLayer;
    for (const auto& layer : layers) {
        delete layer.second;
    }
};

float* Model::predict(const float* input) {
    float* d_input = inputLayer->forward(input);

    for (auto& layer : layers) {
        d_input = layer.second->forward(d_input);
    }

    return outputLayer->forward(d_input);
}

void Model::addLayer(const std::string& name, Layers::SequentialLayer* layer) {

    Module* module = dynamic_cast<Module*>(layer);

    if (module != nullptr) {
        layers.push_back({ name, module });
        for (const auto& moduleLayer : module->getLayers()) {
            layerMap[moduleLayer.first] = moduleLayer.second;
        }

        return;
    }

    layers.push_back({ name, layer });
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

    u_short version;
    file.read(reinterpret_cast<char*>(&version), sizeof(version));

    if (version != 1) {
        std::cerr << "Unsupported model version: " << version << std::endl;
        return;
    }

    u_int64_t headerSize;
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
        size_t dotPos = nameStr.find_last_of('.');
        if (dotPos == std::string::npos)
            continue;
        std::string name = nameStr.substr(0, dotPos);
        TensorType type = nameStr.substr(dotPos + 1) == "weight" ? TensorType::WEIGHT : TensorType::BIAS;

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

        file.seekg(sizeof(version) + sizeof(headerSize) + header.size() + tensorInfo.offset);
        file.read(reinterpret_cast<char*>(values.data()), tensorInfo.size * sizeof(float));

        if (layerMap.find(tensorInfo.name) != layerMap.end()) {

            Layers::WeightedLayer* wLayer = dynamic_cast<Layers::WeightedLayer*>(layerMap[tensorInfo.name]);

            if (wLayer == nullptr) {
                std::cerr << "Layer: " << tensorInfo.name << " does not have weights" << std::endl;
                continue;
            }

            if (tensorInfo.type == TensorType::WEIGHT) {

                if (wLayer->getWeights().size() != values.size()) {
                    std::cerr << "Layer: " << tensorInfo.name << " has incorrect number of weights, expected "
                              << wLayer->getWeights().size() << " but got " << values.size() << ", skipping" << std::endl;
                    continue;
                }

                wLayer->setWeights(values.data());
            } else if (tensorInfo.type == TensorType::BIAS) {

                if (wLayer->getBiases().size() != values.size()) {
                    std::cerr << "Layer: " << tensorInfo.name << " has incorrect number of biases, expected " 
                              << wLayer->getBiases().size() << " but got " << values.size() << ", skipping" << std::endl;
                    continue;
                }

                wLayer->setBiases(values.data());
            }
        } else {
            std::cerr << "Layer: " << tensorInfo.name << " does not exist, skipping" << std::endl;
        }
    }

    file.close();
}

bool Model::validate() {
    
    bool valid = true;
    int size = inputLayer->getInputSize();

    for (const auto& layer : layers) {
        if (layer.second->getInputSize() != size) {
            valid = false;
            std::cerr << "Layer: " << layer.first << " has incorrect input size, expected " << size << " but got "
                      << layer.second->getInputSize() << std::endl;
            break;
        }

        size = layer.second->getOutputSize();
    }

    return valid;
}