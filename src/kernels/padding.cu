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

Padded result:

0 0 0 0 0 0 0 2 4 0 0 1 3 5 0 0 0 0 0 0 0 0 0 0 0 0 6 8 10 0 0 7 9 11 0 0 0 0 0 0


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
    int stride = gridDim.x * blockDim.x;
    int tid    = blockDim.x * blockIdx.x + threadIdx.x;

	for (int i = tid; i < (w + 2 * p) * (h + 2 * p) * n; i += stride) {
		
		// if i is in the padding region
		if (i < p * (h + 2 * p) * n || i >= (w + p) * (h + 2 * p) * n) {
			d_padded[i] = 0.0f;
		} else {
			// if i is in the original region
			d_padded[i] = d_input[(i - p * (h + 2 * p) * n) / (h + 2 * p) * w + (i - p * (h + 2 * p) * n) % (h + 2 * p)];
		}
	}
}