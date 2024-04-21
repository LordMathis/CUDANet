#include <cuda_runtime_api.h>
#include <gtest/gtest.h>

#include <vector>

#include "cuda_helper.cuh"
#include "vector.cuh"
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

    int grid_size = (std::max(w, h) + BLOCK_SIZE - 1) / BLOCK_SIZE;


    CUDANet::Utils::clear(d_output, h);

    CUDANet::Kernels::mat_vec_mul<<<grid_size, BLOCK_SIZE>>>(d_matrix, d_vector, d_output, w, h);
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
        EXPECT_NEAR(sum, output_gpu[i], 1e-5f);
    }

    cudaFree(d_matrix);
    cudaFree(d_vector);
    cudaFree(d_output);

    
}

TEST(MatMulTest, MaxReduceTest) {
    cudaError_t cudaStatus;

    const int n = 1 << 16;

    std::vector<float> input(n);
    for (int i = 0; i < n; i++) {
        input[i] = i;
    }

    float* d_input;
    float* d_output;

    cudaStatus = cudaMalloc((void**)&d_input, sizeof(float) * n);
    EXPECT_EQ(cudaStatus, cudaSuccess);

    cudaStatus = cudaMalloc((void**)&d_output, sizeof(float) * n);
    EXPECT_EQ(cudaStatus, cudaSuccess);

    cudaStatus = cudaMemcpy(d_input, input.data(), sizeof(float) * n, cudaMemcpyHostToDevice);
    EXPECT_EQ(cudaStatus, cudaSuccess);

    const int grid_size = (n + BLOCK_SIZE - 1) / BLOCK_SIZE;

    CUDANet::Kernels::max_reduce<<<grid_size, BLOCK_SIZE>>>(d_input, d_output, n);

    int remaining = grid_size;
    while (remaining > 1) {
        int blocks_needed = (remaining + BLOCK_SIZE - 1) / BLOCK_SIZE;
        CUDANet::Kernels::max_reduce<<<blocks_needed, BLOCK_SIZE>>>(d_output, d_output, remaining);
        remaining = blocks_needed;
    }

    std::vector<float> output(n);
    cudaStatus = cudaMemcpy(output.data(), d_output, sizeof(float), cudaMemcpyDeviceToHost);
    EXPECT_EQ(cudaStatus, cudaSuccess);

    EXPECT_EQ(output[0], 65535.0f);

    cudaFree(d_input);
    cudaFree(d_output);

    
}

TEST(MatMulTest, VecExpTest) {
    cudaError_t cudaStatus;

    float input[6] = {22.496f,  36.9006f, 30.9904f,
                      28.4213f, 26.4541f, 31.7887f};

    std::vector<float> expected = {5886928896.0f,     1.06102872080384e+16f,
                                   28771323215872.0f, 2204012904448.0f,
                                   308226162688.0f,   63922983927808.0f};

    float* d_input;
    float* d_output;

    cudaStatus = cudaMalloc((void**)&d_input, sizeof(float) * 6);
    EXPECT_EQ(cudaStatus, cudaSuccess);

    cudaStatus = cudaMalloc((void**)&d_output, sizeof(float) * 6);
    EXPECT_EQ(cudaStatus, cudaSuccess);

    cudaStatus =
        cudaMemcpy(d_input, input, sizeof(float) * 6, cudaMemcpyHostToDevice);
    EXPECT_EQ(cudaStatus, cudaSuccess);

    CUDANet::Kernels::vec_exp<<<1, 6>>>(d_input, d_output, 6);
    cudaStatus = cudaDeviceSynchronize();
    EXPECT_EQ(cudaStatus, cudaSuccess);

    std::vector<float> output(6);

    cudaStatus = cudaMemcpy(
        output.data(), d_output, sizeof(float) * 6, cudaMemcpyDeviceToHost
    );
    EXPECT_EQ(cudaStatus, cudaSuccess);

    for (int i = 0; i < 6; i++) {
        EXPECT_NEAR(expected[i], output[i], 1e7f);
    }

    cudaFree(d_input);
    cudaFree(d_output);

    
}

TEST(MatMulTest, SumReduceTest) {
    cudaError_t cudaStatus;

    const int n = 1 << 16;

    std::vector<float> input(n);
    for (int i = 0; i < n; i++) {
        input[i] = 1.0f;
    }

    const float expected = n;

    float* d_input = nullptr;
    float* d_sum = nullptr;

    const int gridSize = (n + BLOCK_SIZE - 1) / BLOCK_SIZE;

    cudaStatus = cudaMalloc((void**)&d_input, sizeof(float) * n);
    EXPECT_EQ(cudaStatus, cudaSuccess);

    cudaStatus = cudaMalloc((void**)&d_sum, sizeof(float) * n);
    EXPECT_EQ(cudaStatus, cudaSuccess);

    cudaStatus =
        cudaMemcpy(d_input, input.data(), sizeof(float) * n, cudaMemcpyHostToDevice);
    EXPECT_EQ(cudaStatus, cudaSuccess);

    CUDANet::Kernels::sum_reduce<<<gridSize, BLOCK_SIZE>>>(
        d_input, d_sum, n
    );

    int remaining = gridSize;
    while (remaining > 1) {
        int blocks_needed = (remaining + BLOCK_SIZE - 1) / BLOCK_SIZE;
        CUDANet::Kernels::sum_reduce<<<blocks_needed, BLOCK_SIZE>>>(d_sum, d_sum, remaining);
        remaining = blocks_needed;
    }


    std::vector<float> sum(n);
    cudaStatus = cudaMemcpy(
        sum.data(), d_sum, sizeof(float) * n, cudaMemcpyDeviceToHost
    );
    EXPECT_EQ(cudaStatus, cudaSuccess);

    EXPECT_FLOAT_EQ(expected, sum[0]);    

    cudaFree(d_input);
    cudaFree(d_sum);

    
}