#ifndef CUDANET_MODEL_H
#define CUDANET_MODEL_H

#include <string>
#include <unordered_map>
#include <vector>

#include "input.cuh"
#include "layer.cuh"
#include "output.cuh"

namespace CUDANet {

enum TensorType {
    WEIGHT,
    BIAS,
};

struct TensorInfo {
    std::string name;
    TensorType  type;
    int         size;
    int         offset;
};

class Model {
  public:
    Model(const int inputSize, const int inputChannels, const int outputSize);
    Model(const Model& other);
    ~Model();

    float* predict(const float* input);

    void addLayer(const std::string& name, Layers::SequentialLayer* layer);
    Layers::SequentialLayer* getLayer(const std::string& name);

    void loadWeights(const std::string& path);

  private:
    Layers::Input*  inputLayer;
    Layers::Output* outputLayer;

    int inputSize;
    int inputChannels;

    int outputSize;

    std::vector<std::pair<std::string, Layers::SequentialLayer*>> layers;
    std::unordered_map<std::string, Layers::SequentialLayer*>     layerMap;
};

}  // namespace CUDANet

#endif  // CUDANET_MODEL_H