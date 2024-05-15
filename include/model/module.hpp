#ifndef CUDANET_MODULE_H
#define CUDANET_MODULE_H

#include <string>
#include <unordered_map>
#include <vector>

#include "layer.cuh"

namespace CUDANet {

class Module : public Layers::SequentialLayer {
  public:
    Module(const int inputSize, const int inputChannels, const int outputSize, const int outputChannels);
    ~Module();

    virtual float* forward(const float* d_input) = 0;

    int getOutputSize();
    int getInputSize();

    void addLayer(const std::string& name, Layers::SequentialLayer* layer);
    Layers::SequentialLayer* getLayer(const std::string& name);

    const std::unordered_map<std::string, Layers::SequentialLayer*>& getLayers() const;

  private:
    int inputSize;
    int inputChannels;

    int outputSize;
    int outputChannels;

    float *d_output;

    std::vector<std::pair<std::string, Layers::SequentialLayer*>> layers;
    std::unordered_map<std::string, Layers::SequentialLayer*>     layerMap;
};

}  // namespace CUDANet

#endif