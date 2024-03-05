#include <cuda_runtime_api.h>
#include <gtest/gtest.h>

#include <iostream>

#include "activations.cuh"
#include "dense.cuh"


class DenseLayerTest : public::testing::Test {
  protected:
    Layers::Dense commonTestSetup(
        int                              inputSize,
        int                              outputSize,
        std::vector<float>&              input,
        std::vector<std::vector<float>>& weights,
        std::vector<float>&              biases,
        float*&                          d_input,
        float*&                          d_output,
        Activation                       activation
    ) {
        // Create Dense layer
        Layers::Dense denseLayer(
            inputSize, outputSize, activation
        );

        // Set weights and biases
        denseLayer.setWeights(weights);
        denseLayer.setBiases(biases);

        // Allocate device memory
        cudaStatus = cudaMalloc((void**)&d_input, sizeof(float) * input.size());
        EXPECT_EQ(cudaStatus, cudaSuccess);

        cudaStatus = cudaMalloc((void**)&d_output, sizeof(float) * outputSize);
        EXPECT_EQ(cudaStatus, cudaSuccess);

        // Copy input to device
        cudaStatus = cudaMemcpy(
            d_input, input.data(), sizeof(float) * input.size(), cudaMemcpyHostToDevice
        );
        EXPECT_EQ(cudaStatus, cudaSuccess);


        return denseLayer;
    }

    void commonTestTeardown(float* d_input, float* d_output) {
        // Free device memory
        cudaFree(d_input);
        cudaFree(d_output);
    }

    cudaError_t    cudaStatus;
};

TEST_F(DenseLayerTest, Init) {
    for (int i = 1; i < 100; ++i) {
        for (int j = 1; j < 100; ++j) {
            int inputSize  = i;
            int outputSize = j;

            Layers::Dense denseLayer(
                inputSize, outputSize, SIGMOID
            );
        }
    }
}

TEST_F(DenseLayerTest, setWeights) {
    int inputSize  = 4;
    int outputSize = 5;

    std::vector<std::vector<float>> weights = {
        {0.5f, 1.0f, 0.2f, 0.8f},
        {1.2f, 0.3f, 1.5f, 0.4f},
        {0.7f, 1.8f, 0.9f, 0.1f},
        {0.4f, 2.0f, 0.6f, 1.1f},
        {1.3f, 0.5f, 0.0f, 1.7f}
    };

    Layers::Dense denseLayer(inputSize, outputSize, SIGMOID);

    denseLayer.setWeights(weights);
}

TEST_F(DenseLayerTest, ForwardUnitWeightMatrixLinear) {
    int inputSize  = 3;
    int outputSize = 3;

    std::vector<float> input = {1.0f, 2.0f, 3.0f};

    std::vector<std::vector<float>> weights(
        inputSize, std::vector<float>(outputSize, 0.0f)
    );
    for (int i = 0; i < inputSize; ++i) {
        for (int j = 0; j < outputSize; ++j) {
            if (i == j) {
                weights[i][j] = 1.0f;
            }
        }
    }
    std::vector<float> biases(outputSize, 1.0f);

    float* d_input;
    float* d_output;

    Layers::Dense denseLayer = commonTestSetup(
        inputSize, outputSize, input, weights, biases, d_input, d_output, LINEAR
    );
    denseLayer.forward(d_input, d_output);

    std::vector<float> output(outputSize);
    cudaStatus = cudaMemcpy(
        output.data(), d_output, sizeof(float) * outputSize, cudaMemcpyDeviceToHost
    );
    EXPECT_EQ(cudaStatus, cudaSuccess);

    // Check if the output is a zero vector
    EXPECT_FLOAT_EQ(output[0], 2.0f);
    EXPECT_FLOAT_EQ(output[1], 3.0f);
    EXPECT_FLOAT_EQ(output[2], 4.0f);

    commonTestTeardown(d_input, d_output);
}

TEST_F(DenseLayerTest, ForwardRandomWeightMatrixRelu) {
    int inputSize  = 5;
    int outputSize = 4;

    std::vector<float> input = {1.0f, 2.0f, 3.0f, 4.0f, -5.0f};

    std::vector<std::vector<float>> weights = {
        {0.5f, 1.2f, 0.7f, 0.4f, 1.3f},
        {1.0f, 0.3f, 1.8f, 2.0f, 0.5f},
        {0.2f, 1.5f, 0.9f, 0.6f, 0.0f},
        {0.8f, 0.4f, 0.1f, 1.1f, 1.7f}
    };
    std::vector<float> biases = {0.2f, 0.5f, 0.7f, -1.1f};

    float* d_input;
    float* d_output;

    Layers::Dense denseLayer = commonTestSetup(
        inputSize, outputSize, input, weights, biases, d_input, d_output, RELU
    );

    denseLayer.forward(d_input, d_output);

    std::vector<float> output(outputSize);
    cudaStatus = cudaMemcpy(
        output.data(), d_output, sizeof(float) * outputSize, cudaMemcpyDeviceToHost
    );
    EXPECT_EQ(cudaStatus, cudaSuccess);

    // weights * inputs = 0.1, 12.5, 8.3, -2.2
    // + biases = 0.3, 13, 9, -3.3

    std::vector<float> expectedOutput = {0.3f, 13.0f, 9.0f, 0.0f};
    for (int i = 0; i < outputSize; ++i) {
        EXPECT_NEAR(
            output[i], expectedOutput[i], 1e-4
        );  // Allow small tolerance for floating-point comparison
    }

    commonTestTeardown(d_input, d_output);
}

TEST_F(DenseLayerTest, ForwardRandomWeightMatrixSigmoid) {
    int inputSize  = 5;
    int outputSize = 4;

    std::vector<float> input = {0.1f, 0.2f, 0.3f, 0.4f, 0.5f};

    std::vector<std::vector<float>> weights = {
        {0.8f, 0.7f, 0.7f, 0.3f, 0.8f},
        {0.1f, 0.4f, 0.8f, 0.0f, 0.2f},
        {0.2f, 0.5f, 0.7f, 0.3f, 0.0f},
        {0.1f, 0.7f, 0.6f, 1.0f, 0.4f}
    };
    std::vector<float> biases = {0.1f, 0.2f, 0.3f, 0.4f};

    float* d_input;
    float* d_output;

    Layers::Dense denseLayer = commonTestSetup(
        inputSize, outputSize, input, weights, biases, d_input, d_output,
        SIGMOID
    );

    denseLayer.forward(d_input, d_output);

    std::vector<float> output(outputSize);
    cudaStatus = cudaMemcpy(
        output.data(), d_output, sizeof(float) * outputSize, cudaMemcpyDeviceToHost
    );
    EXPECT_EQ(cudaStatus, cudaSuccess);

    // weights * input = 0.95, 0.43, 0.45, 0.93
    // + biases = 1.05, 0.63, 0.75, 1.33
    // sigmoid = 0.740775, 0.652489, 0.679179, 0.790841

    std::vector<float> expectedOutput = {
        0.740775f, 0.652489f, 0.679179f, 0.790841f
    };

    for (int i = 0; i < outputSize; ++i) {
        EXPECT_NEAR(output[i], expectedOutput[i], 1e-5);
    }

    commonTestTeardown(d_input, d_output);
}
