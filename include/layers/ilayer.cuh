
#ifndef I_LAYER_H
#define I_LAYER_H

#include <vector>

namespace Layers {

class ILayer {
  public:
    virtual ~ILayer() {}

    virtual void forward(const float* input, float* output) = 0;
    virtual void setWeights(const float* weights)           = 0;
    virtual void setBiases(const float* biases)             = 0;

  private:
    virtual void initializeWeights() = 0;
    virtual void initializeBiases()  = 0;

    virtual void toCuda() = 0;

    int inputSize;
    int outputSize;

    float* d_weights;
    float* d_biases;

    std::vector<float> weights;
    std::vector<float> biases;

    Activation activation;
};

}  // namespace Layers

#endif  // I_LAYERH