#ifndef MATMUL_H
#define MATMUL_H

namespace Kernels {

__global__ void mat_vec_mul(
    const float* d_matrix,
    const float* d_vector,
    float*       d_output,
    int          w,
    int          h
);

__global__ void vec_vec_add(
    const float* d_vector1,
    const float* d_vector2,
    float*       d_output,
    int          w
);

}  // namespace Kernels

#endif  // MATMUL_H