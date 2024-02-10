
#ifndef I_LAYER_H
#define I_LAYER_H

#include <cublas_v2.h>

namespace Layers {

    class ILayer {
    public:
        virtual ~ILayer() {}

        virtual void forward(const float* input, float* output) = 0;
    };

} // namespace Layers

#endif // I_LAYERH