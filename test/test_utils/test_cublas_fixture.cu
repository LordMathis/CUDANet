#include <cublas_v2.h>
#include <gtest/gtest.h>

#include "test_cublas_fixture.cuh"

cublasHandle_t CublasTestFixture::cublasHandle;

void CublasTestFixture::SetUpTestSuite() {
    cublasCreate(&cublasHandle);
}

void CublasTestFixture::TearDownTestSuite() {
    cublasDestroy(cublasHandle);
}
