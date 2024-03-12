#ifndef INPUT_LAYER_H
#define INPUT_LAYER_H

#include <ilayer.cuh>

namespace Layers {

class Input : public ILayer {
  public:
    Input(int inputSize);
    ~Input();

    float* forward(const float* input);

    void setWeights(const float* weights);
    void setBiases(const float* biases);

  private:
    void initializeWeights();
    void initializeBiases();

    void toCuda();

    int    inputSize;
    float* d_output;
};

}  // namespace Layers

#endif  // INPUT_LAYER_H