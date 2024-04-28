#include "cuda_helper.cuh"
#include "matmul.cuh"

using namespace CUDANet;

__global__ void Kernels::mat_vec_mul(
    const float* __restrict__ d_matrix,
    const float* __restrict__ d_vector,
    float* __restrict__ d_output,
    const unsigned int w,
    const unsigned int h
) {
    int tid = blockDim.x * blockIdx.x + threadIdx.x;

    if (tid < h) {
        float temp = 0.0f;

        for (unsigned int j = 0; j < w; j++) {
            temp += d_matrix[tid * w + j] * d_vector[j];
        }

        d_output[tid] = temp;
    }
}

__global__ void Kernels::vec_vec_add(
    const float* __restrict__ d_vector1,
    const float* __restrict__ d_vector2,
    float* __restrict__ d_output,
    const unsigned int w
) {
    int tid = blockDim.x * blockIdx.x + threadIdx.x;
    if (tid >= w) {
        return;
    }
    d_output[tid] = d_vector1[tid] + d_vector2[tid];
}

__global__ void Kernels::vec_scalar_sub(
    const float* __restrict__ d_src,
    float* __restrict__ d_out,
    const float* __restrict__ d_scalar,
    const unsigned int len
) {
    int tid = blockDim.x * blockIdx.x + threadIdx.x;
    if (tid >= len) {
        return;
    }
    d_out[tid] = d_src[tid] - *d_scalar;
}

__global__ void Kernels::vec_scalar_div(
    const float* __restrict__ d_src,
    float* __restrict__ d_out,
    const float* __restrict__ d_scalar,
    const unsigned int len
) {
    int tid = blockDim.x * blockIdx.x + threadIdx.x;
    if (tid >= len) {
        return;
    }
    d_out[tid] = d_src[tid] / *d_scalar;
}

__global__ void Kernels::vec_exp(
    const float* __restrict__ src,
    float* __restrict__ dst,
    const unsigned int len
) {
    int stride = gridDim.x * blockDim.x;
    int tid    = blockDim.x * blockIdx.x + threadIdx.x;

    for (int i = tid; i < len; i += stride) {
        dst[i] = expf(src[i]);
    }
}


__global__ void Kernels::max_reduce(
    const float* __restrict__ d_vector,
    float* __restrict__ d_output,
    const unsigned int len
) {
    __shared__ float shared_max[BLOCK_SIZE];
    int i       = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < len) {
        shared_max[threadIdx.x] = d_vector[i];
    } else {
        shared_max[threadIdx.x] = -INFINITY;
    }    

    __syncthreads();

    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) {
            shared_max[threadIdx.x] = fmaxf(shared_max[threadIdx.x], shared_max[threadIdx.x + s]);
        }
        __syncthreads();
    }

    if (threadIdx.x == 0) {
        d_output[blockIdx.x] = shared_max[0];
    }
}

__global__ void Kernels::sum_reduce(
    const float* __restrict__ d_vector,
    float* __restrict__ d_output,
    const unsigned int len
) {
    __shared__ float partial_sum[BLOCK_SIZE];
    int              i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < len) {
        partial_sum[threadIdx.x] = d_vector[i];
    } else {
        partial_sum[threadIdx.x] = 0.0f;
    }

    __syncthreads();

    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) {
            partial_sum[threadIdx.x] += partial_sum[threadIdx.x + s];
        }
        __syncthreads();
    }

    if (threadIdx.x == 0) {
        d_output[blockIdx.x] = partial_sum[0];
    }
}