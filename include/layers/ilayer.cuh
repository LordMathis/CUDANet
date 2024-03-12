
#ifndef I_LAYER_H
#define I_LAYER_H

#include <vector>

namespace Layers {

enum Activation { SIGMOID, RELU, NONE };

enum Padding { SAME, VALID };

class ILayer {
  public:
    virtual ~ILayer() {}

    virtual float* forward(const float* input)      = 0;
    virtual void   setWeights(const float* weights) = 0;
    virtual void   setBiases(const float* biases)   = 0;

  private:
    virtual void initializeWeights() = 0;
    virtual void initializeBiases()  = 0;

    virtual void toCuda() = 0;

    int inputSize;
    int outputSize;

    float* d_output;

    float* d_weights;
    float* d_biases;

    std::vector<float> weights;
    std::vector<float> biases;

    Layers::Activation activation;
};

}  // namespace Layers

#endif  // I_LAYERH