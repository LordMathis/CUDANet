#include <cuda_runtime_api.h>
#include <gtest/gtest.h>

#include <vector>

#include "cuda_helper.cuh"
#include "matmul.cuh"

TEST(MatMulTest, MatVecMulTest) {
    cudaError_t cudaStatus;

    int w = 10;
    int h = 5;

    float* d_matrix;
    float* d_vector;
    float* d_output;

    cudaStatus = cudaMalloc((void**)&d_matrix, sizeof(float) * w * h);
    EXPECT_EQ(cudaStatus, cudaSuccess);

    cudaStatus = cudaMalloc((void**)&d_vector, sizeof(float) * w);
    EXPECT_EQ(cudaStatus, cudaSuccess);

    cudaStatus = cudaMalloc((void**)&d_output, sizeof(float) * h);
    EXPECT_EQ(cudaStatus, cudaSuccess);

    std::vector<float> matrix = {
        0.643f, 0.912f, 0.723f, 0.587f, 0.155f, 0.932f, 0.391f, 0.279f, 0.846f, 0.788f,
        0.641f, 0.445f, 0.528f, 0.316f, 0.247f, 0.181f, 0.549f, 0.328f, 0.919f, 0.405f,
        0.733f, 0.287f, 0.901f, 0.602f, 0.816f, 0.495f, 0.797f, 0.210f, 0.305f, 0.613f,
        0.178f, 0.856f, 0.724f, 0.263f, 0.559f, 0.677f, 0.193f, 0.389f, 0.488f, 0.848f,
        0.121f, 0.734f, 0.587f, 0.904f, 0.312f, 0.672f, 0.807f, 0.478f, 0.581f, 0.964f
    };
    std::vector<float> vector = {
        0.643f, 0.912f, 0.723f, 0.587f, 0.155f, 0.932f, 0.391f, 0.279f, 0.846f, 0.788f
    };

    cudaStatus = cudaMemcpy(d_matrix, matrix.data(), sizeof(float) * w * h, cudaMemcpyHostToDevice);
    EXPECT_EQ(cudaStatus, cudaSuccess);

    cudaStatus = cudaMemcpy(d_vector, vector.data(), sizeof(float) * w, cudaMemcpyHostToDevice);
    EXPECT_EQ(cudaStatus, cudaSuccess);

    int THREADS_PER_BLOCK = std::max(w, h);
    int BLOCKS            = 1;

    CUDANet::Kernels::clear<<<BLOCKS, h>>>(d_output, h);

    CUDANet::Kernels::mat_vec_mul<<<BLOCKS, THREADS_PER_BLOCK, sizeof(float) * w>>>(d_matrix, d_vector, d_output, w, h);
    cudaStatus = cudaDeviceSynchronize();
    EXPECT_EQ(cudaStatus, cudaSuccess);

    std::vector<float> output_gpu(h);
    cudaStatus = cudaMemcpy(output_gpu.data(), d_output, sizeof(float) * h, cudaMemcpyDeviceToHost);
    EXPECT_EQ(cudaStatus, cudaSuccess);

    for (int i = 0; i < h; i++) {
        float sum = 0.0f;
        for (int j = 0; j < w; j++) {
            sum += matrix[i * w + j] * vector[j];
        }
        EXPECT_NEAR(sum, output_gpu[i], 1e-5);
    }

    cudaFree(d_matrix);
    cudaFree(d_vector);
    cudaFree(d_output);

    cudaDeviceReset();
}

TEST(MatMulTest, MaxReduceTest) {
    cudaError_t cudaStatus;

    std::vector<float> input = {0.643f, 0.912f, 0.723f, 0.587f, 0.155f, 0.932f, 0.391f, 0.279f, 0.846f, 0.788f};

    float* d_input;
    float* d_output;

    cudaStatus = cudaMalloc((void**)&d_input, sizeof(float) * 10);
    EXPECT_EQ(cudaStatus, cudaSuccess);

    cudaStatus = cudaMalloc((void**)&d_output, sizeof(float));
    EXPECT_EQ(cudaStatus, cudaSuccess);

    cudaStatus = cudaMemcpy(d_input, input.data(), sizeof(float) * 10, cudaMemcpyHostToDevice);
    EXPECT_EQ(cudaStatus, cudaSuccess);

    const int grid_size = (10 + BLOCK_SIZE - 1) / BLOCK_SIZE;

    CUDANet::Kernels::max_reduce<<<grid_size, BLOCK_SIZE>>>(d_input, d_output);
    CUDANet::Kernels::max_reduce<<<1, BLOCK_SIZE>>>(d_output, d_output);

    std::vector<float> output(10);
    cudaStatus = cudaMemcpy(output.data(), d_output, sizeof(float), cudaMemcpyDeviceToHost);
    EXPECT_EQ(cudaStatus, cudaSuccess);

    EXPECT_EQ(output[0], 0.932f);

    cudaFree(d_input);
    cudaFree(d_output);

    cudaDeviceReset();
}