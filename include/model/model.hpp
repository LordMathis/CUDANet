#ifndef CUDANET_MODEL_H
#define CUDANET_MODEL_H

#include <string>
#include <vector>
#include <map>

#include "layer.cuh"
#include "input.cuh"

namespace CUDANet {

class Model {
  public:
    Model(const int inputSize, const int inputChannels);
    ~Model();

    float* predict(const float* input);

    void addLayer(const std::string& name, Layers::SequentialLayer* layer);

  private:

    Layers::Input *inputLayer;

    int inputSize;
    int inputChannels;

    int outputSize;
    
    std::vector<Layers::SequentialLayer*> layers;
    std::map<std::string, Layers::WeightedLayer*> layerMap;

};

}  // namespace CUDANet

#endif  // CUDANET_MODEL_H