#ifndef MATRIX_MATH_H
#define MATRIX_MATH_H

__global__ void mat_vec_mul_kernel(
    const float* d_matrix,
    const float* d_vector,
    float*       d_output,
    int          w,
    int          h
);

__global__ void vec_vec_add_kernel(
    const float* d_vector1,
    const float* d_vector2,
    float*       d_output,
    int          w
);

#endif  // MATRIX_MATH_H