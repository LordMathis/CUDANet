#ifndef CUDANET_MODEL_H
#define CUDANET_MODEL_H

#include <string>
#include <unordered_map>
#include <vector>

#include "input.hpp"
#include "layer.hpp"
#include "module.hpp"
#include "output.hpp"

namespace CUDANet {

enum TensorType {
    WEIGHT,
    BIAS,
    RUNNING_MEAN,
    RUNNING_VAR
};

struct TensorInfo {
    std::string name;
    TensorType  type;
    int         size;
    int         offset;
};

class Model {
  public:
    Model(const shape2d inputSize, const int inputChannels, const int outputSize);
    Model(const Model& other);
    ~Model();

    virtual float* predict(const float* input);

    void addLayer(const std::string& name, Layers::SequentialLayer* layer);
    Layers::SequentialLayer* getLayer(const std::string& name);

    void loadWeights(const std::string& path);

    bool validate();

    void printSummary();

  protected:
    Layers::Input*  inputLayer;
    Layers::Output* outputLayer;

    shape2d inputSize;
    int inputChannels;

    int outputSize;

    std::vector<std::pair<std::string, Layers::SequentialLayer*>> layers;
    std::unordered_map<std::string, Layers::SequentialLayer*>     layerMap;
};

}  // namespace CUDANet

#endif  // CUDANET_MODEL_H