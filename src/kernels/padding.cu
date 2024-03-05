#include <vector>

/*
Pads matrix width x height x n_channels to width + 2 * padding x height + 2 *
padding x n_channels Matrix is represented as a pointer to a vector

For example:

w = 2
h = 3
n = 2
p = 1

Channel 0:
  0  1
  2  3
  4  5
Channel 1:
  6  7
  8  9
 10 11

Is represented as:

0 1 2 3 4 5 6 7 8 9 10 11

Padded result (as a continuous vector):

0.0f, 0.0f, 0.0f, 0.0f,
0.0f, 0.0f, 1.0f, 0.0f,
0.0f, 2.0f, 3.0f, 0.0f,
0.0f, 4.0f, 5.0f, 0.0f,
0.0f, 0.0f, 0.0f, 0.0f,
0.0f, 0.0f, 0.0f, 0.0f,
0.0f, 6.0f, 7.0f, 0.0f,
0.0f, 8.0f, 9.0f, 0.0f,
9.0f, 10.0f, 11.0f, 0.0f,
0.0f, 0.0f, 0.0f, 0.0f

Args:
  d_input: Pointer to input vector representing matrix
  d_padded: Pointer to output vector representing padded matrix (needs to be
pre-allocated)
  w: Width of input matrix
  h: Height of input matrix
  n: Number of channels in input matrix
  p: Padding
*/
__global__ void pad_matrix_kernel(
    const float* d_input,
    float*       d_padded,
    int          w,
    int          h,
    int          n,
    int          p
) {
    int tid = blockDim.x * blockIdx.x + threadIdx.x;

    if (tid >= (w + 2 * p) * (h + 2 * p) * n) {
        return;
    }

    int idx = tid;

    // unravel index into padded matrix
    int i_n = idx / ((w + 2 * p) * (h + 2 * p));
    int i_h = idx % ((w + 2 * p) * (h + 2 * p)) / (w + 2 * p);
    int i_w = idx % (w + 2 * p);

    // if i is in the padding region
    if (i_w < p || i_w >= (w + p) || i_h < p || i_h >= (h + p)) {
        d_padded[tid] = 0.0f;
    } else {
        // Get index into input vector
        int i_orig    = i_n * w * h + (i_h - p) * w + (i_w - p);
        d_padded[tid] = d_input[i_orig];
    }
}
