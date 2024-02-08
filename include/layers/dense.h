// fully_connected_layer.h

#ifndef DENSE_LAYER_H
#define DENSE_LAYER_H

#include <vector>
#include <cublas_v2.h>

namespace Layers {

    class Dense {
    public:
        Dense(int inputSize, int outputSize, cublasHandle_t cublasHandle);
        ~Dense();

        void forward(const float* input, float* output);

    private:
        int inputSize;
        int outputSize;

        cublasHandle_t cublasHandle;

        float* d_weights;
        float* d_biases;

        std::vector<std::vector<float>> weights;
        std::vector<float> biases;

        void initializeWeights();
        void initializeBiases();
    };

} // namespace Layers

#endif // DENSE_LAYER_H
