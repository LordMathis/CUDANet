#ifndef CUDANET_MODULE_H
#define CUDANET_MODULE_H

#include <string>
#include <unordered_map>
#include <vector>

#include "layer.cuh"

namespace CUDANet {

class Module : public Layers::SequentialLayer {
  public:
    virtual ~Module() = 0;

    virtual float* forward(const float* d_input) = 0;

    int getOutputSize();
    int getInputSize();

    void addLayer(const std::string& name, Layers::SequentialLayer* layer);
    Layers::SequentialLayer* getLayer(const std::string& name);

    const std::unordered_map<std::string, Layers::SequentialLayer*>& getLayers() const;

  protected:
    int inputSize;
    int inputChannels;

    int outputSize;
    int outputChannels;

    std::vector<std::pair<std::string, Layers::SequentialLayer*>> layers;
    std::unordered_map<std::string, Layers::SequentialLayer*>     layerMap;
};

}  // namespace CUDANet

#endif