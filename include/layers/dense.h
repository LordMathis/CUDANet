// fully_connected_layer.h

#ifndef DENSE_LAYER_H
#define DENSE_LAYER_H

#include <vector>
#include <cublas_v2.h>
#include <ilayer.h>

namespace Layers {

    class Dense : public ILayer {
    public:
        Dense(int inputSize, int outputSize, cublasHandle_t cublasHandle);
        ~Dense();

        void forward(const float* input, float* output);
        virtual void setWeights(const std::vector<std::vector<float>>& weights) = 0;
        virtual void setBiases(const std::vector<float>& biases) = 0;

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
        void toCuda();
    };

} // namespace Layers

#endif // DENSE_LAYER_H
