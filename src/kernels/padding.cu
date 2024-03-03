#include <vector>

/*
Pads matrix width x height x n_channels to width + 2 * padding x height + 2 *
padding x n_channels Matrix is represented as a pointer to column major vector

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

0 2 4 1 3 5 6 8 10 7 9 11

Padded result (as a continuous vector):

0 0 0 0 0 0 0 2 4 0
0 1 3 5 0 0 0 0 0 0
0 0 0 0 0 0 6 8 10 0
0 7 9 11 0 0 0 0 0 0

Args:
  d_input: Pointer to input vector representing matrix
  d_padded: Pointer to output vector representing padded matrix (needs to be
pre-allocated) w: Width of input matrix h: Height of input matrix n: Number of
channels in input matrix p: Padding
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

    // unravel index
    int i_h = idx % (h + 2 * p);
    idx /= (h + 2 * p);

    int i_w = idx % (w + 2 * p);
    idx /= (w + 2 * p);

    int i_n = idx % n;

    // if i is in the padding region
    if (i_w < p || i_w >= (w + p) || i_h < p || i_h >= (h + p)) {
        d_padded[tid] = 0.0f;
    } else {
        // Get index into input vector
        int i_orig    = i_n * w * h + (i_w - p) * h + (i_h - p);
        d_padded[tid] = d_input[i_orig];
    }
}
