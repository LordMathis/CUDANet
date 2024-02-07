#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include "cublas_v2.h"
#include "cuda_helper.h"

int main() {
    // Initialize CUDA and get device properties
    cudaDeviceProp deviceProp = initializeCUDA();

    // Specify vector size
    const int N = 5;

    // Host vectors
    float *h_A, *h_B, *h_C;

    // Allocate host memory
    h_A = (float*)malloc(N * sizeof(float));
    h_B = (float*)malloc(N * sizeof(float));
    h_C = (float*)malloc(N * sizeof(float));

    // Initialize host vectors
    for (int i = 0; i < N; ++i) {
        h_A[i] = static_cast<float>(i);
        h_B[i] = static_cast<float>(2 * i);
    }

    // Allocate device memory
    float *d_A, *d_B, *d_C;
    cudaMalloc((void**)&d_A, N * sizeof(float));
    cudaMalloc((void**)&d_B, N * sizeof(float));
    cudaMalloc((void**)&d_C, N * sizeof(float));

    // Copy host vectors to device
    cudaMemcpy(d_A, h_A, N * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, N * sizeof(float), cudaMemcpyHostToDevice);

    // Create cuBLAS handle
    cublasHandle_t handle;
    cublasCreate(&handle);

    // Perform vector addition: C = A + B
    const float alpha = 1.0f;
    const float beta = 1.0f;
    cublasSaxpy(handle, N, &alpha, d_A, 1, d_B, 1);
    cublasSaxpy(handle, N, &beta, d_B, 1, d_C, 1);

    // Copy result from device to host
    cudaMemcpy(h_C, d_C, N * sizeof(float), cudaMemcpyDeviceToHost);

    // Display result
    printf("Result: ");
    for (int i = 0; i < N; ++i) {
        printf("%f ", h_C[i]);
    }
    printf("\n");

    // Clean up
    free(h_A);
    free(h_B);
    free(h_C);
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    cublasDestroy(handle);

    return 0;
}