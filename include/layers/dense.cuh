#ifndef DENSE_LAYER_H
#define DENSE_LAYER_H

#include <functional>
#include <string>
#include <vector>

#include "ilayer.cuh"

namespace Layers {

class Dense : public ILayer {
  public:
    Dense(
        int            inputSize,
        int            outputSize,
        Layers::Activation     activation
    );
    ~Dense();

    float* forward(const float* d_input);
    void setWeights(const float* weights);
    void setBiases(const float* biases);

  private:
    int inputSize;
    int outputSize;

    float* d_output;

    float* d_weights;
    float* d_biases;

    std::vector<float> weights;
    std::vector<float> biases;

    Layers::Activation activation;

    void initializeWeights();
    void initializeBiases();
    void toCuda();
};

}  // namespace Layers

#endif  // DENSE_LAYER_H
