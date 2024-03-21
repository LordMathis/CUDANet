#ifndef CUDANET_MODEL_H
#define CUDANET_MODEL_H

#include <string>
#include <vector>
#include <unordered_map>

#include "layer.cuh"
#include "input.cuh"
#include "output.cuh"

namespace CUDANet {

class Model {
  public:
    Model(const int inputSize, const int inputChannels, const int outputSize);
    Model(const Model& other);
    ~Model();

    float* predict(const float* input);

    void addLayer(const std::string& name, Layers::SequentialLayer* layer);

  private:

    Layers::Input *inputLayer;
    Layers::Output *outputLayer;

    int inputSize;
    int inputChannels;

    int outputSize;
    
    std::vector<Layers::SequentialLayer*> layers;
    std::unordered_map<std::string, Layers::WeightedLayer*> layerMap;

};

}  // namespace CUDANet

#endif  // CUDANET_MODEL_H