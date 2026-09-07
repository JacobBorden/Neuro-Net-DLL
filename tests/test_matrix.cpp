#include <gtest/gtest.h>
#include "math/matrix.h"

class MatrixTest : public ::testing::Test {
protected:
    void SetUp() override {
    }
};

TEST_F(MatrixTest, SplitHorizontalErrorInvalidDivisor) {
    Matrix::Matrix<float> mat(2, 5); // 5 columns
    EXPECT_THROW(mat.SplitHorizontal(2), std::invalid_argument); // 5 % 2 != 0
    EXPECT_THROW(mat.SplitHorizontal(3), std::invalid_argument); // 5 % 3 != 0
}

TEST_F(MatrixTest, SplitHorizontalErrorZeroSplits) {
    Matrix::Matrix<float> mat(2, 4);
    EXPECT_THROW(mat.SplitHorizontal(0), std::invalid_argument);
}

TEST_F(MatrixTest, SplitHorizontalValid) {
    Matrix::Matrix<float> mat(2, 6);
    auto splits = mat.SplitHorizontal(2);
    EXPECT_EQ(splits.size(), 2);
    EXPECT_EQ(splits[0].rows(), 2);
    EXPECT_EQ(splits[0].cols(), 3);
}

TEST_F(MatrixTest, SplitHorizontalValidDefault) {
    Matrix::Matrix<float> mat(2, 6);
    auto splits = mat.SplitHorizontal();
    EXPECT_EQ(splits.size(), 2);
    EXPECT_EQ(splits[0].rows(), 2);
    EXPECT_EQ(splits[0].cols(), 3);
}

TEST_F(MatrixTest, SplitHorizontalDefaultErrorInvalidDivisor) {
    Matrix::Matrix<float> mat(2, 5);
    EXPECT_THROW(mat.SplitHorizontal(), std::invalid_argument);
}
