#ifndef CONV_LAYER_H
#define CONV_LAYER_H

#include <cublas_v2.h>

namespace Layers {

    class Conv {
    public:
        Conv(int inputSize, int outputSize, int kernelSize, cublasHandle_t cublasHandle);
        ~Conv();

        void forward(const float* input, float* output);

    private:
        int inputSize;
        int outputSize;
        int kernelSize;
        cublasHandle_t cublasHandle;
        float* d_weights;
        float* d_biases;
    };

} // namespace Layers

#endif // CONV_LAYER_H
