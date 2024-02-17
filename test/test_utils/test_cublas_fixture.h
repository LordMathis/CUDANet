#include "gtest/gtest.h"
#include "cublas_v2.h"

class CublasTestFixture : public ::testing::Test {
protected:
    static cublasHandle_t cublasHandle;

    static void SetUpTestSuite();
    static void TearDownTestSuite();
};