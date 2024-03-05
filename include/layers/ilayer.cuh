
#ifndef I_LAYER_H
#define I_LAYER_H

#include <vector>

namespace Layers {

class ILayer {
  public:
    virtual ~ILayer() {}

    virtual void forward(const float* input, float* output)                 = 0;
    virtual void setWeights(const std::vector<std::vector<float>>& weights) = 0;
    virtual void setBiases(const std::vector<float>& biases)                = 0;
};

}  // namespace Layers

#endif  // I_LAYERH