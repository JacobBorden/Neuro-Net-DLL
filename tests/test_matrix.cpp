#include <gtest/gtest.h>
#include "../src/math/matrix.h"

class MatrixTest : public ::testing::Test {
protected:
    void SetUp() override {
        // Setup code if needed
    }
};

TEST_F(MatrixTest, OperatorBracket_OutOfBounds_ThrowsException) {
    Matrix::Matrix<float> mat(3, 3);
    EXPECT_THROW(mat[3], std::out_of_range);
    EXPECT_THROW(mat[4], std::out_of_range);
}

TEST_F(MatrixTest, OperatorBracket_InBounds_DoesNotThrow) {
    Matrix::Matrix<float> mat(3, 3);
    EXPECT_NO_THROW(mat[0]);
    EXPECT_NO_THROW(mat[1]);
    EXPECT_NO_THROW(mat[2]);
}
