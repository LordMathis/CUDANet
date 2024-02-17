#include "gtest/gtest.h"
#include "cublas_v2.h"
#include "test_cublas_fixture.h"

cublasHandle_t CublasTestFixture::cublasHandle;

void CublasTestFixture::SetUpTestSuite() {
    cublasCreate(&cublasHandle);
}

void CublasTestFixture::TearDownTestSuite() {
    cublasDestroy(cublasHandle);
}
