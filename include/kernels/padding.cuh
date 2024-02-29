#ifndef PADDING_H
#define PADDING_H

__global__ void pad_matrix_kernel(
    const float* d_input,
    float*       d_padded,
    int          w,
    int          h,
    int          n,
    int          p
);

#endif  // PADDING_H