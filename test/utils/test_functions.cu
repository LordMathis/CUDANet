#include "gtest/gtest.h"
#include <cuda_runtime_api.h>
#include <driver_types.h>
#include <iostream>
#include "functions.cuh"
#include "test_cublas_fixture.cuh"

class FunctionsTest : public CublasTestFixture {
protected:
    cudaError_t cudaStatus;
    cublasStatus_t cublasStatus;
};

TEST_F(FunctionsTest, sigmoid) {

}