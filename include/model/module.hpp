#ifndef CUDANET_MODULE_H
#define CUDANET_MODULE_H

#include <string>
#include <unordered_map>
#include <vector>

#include "layer.cuh"

namespace CUDANet {

class Module : public Layers::SequentialLayer {
  public:
    virtual float* forward(const float* d_input) = 0;

    int getOutputSize();
    int getInputSize();

    void addLayer(const std::string& name, Layers::SequentialLayer* layer);

    const std::vector<std::pair<std::string, Layers::SequentialLayer*>>& getLayers() const;

  protected:
    std::vector<std::pair<std::string, Layers::SequentialLayer*>> layers;

    int outputSize;
    int inputSize;
};

}  // namespace CUDANet

#endif