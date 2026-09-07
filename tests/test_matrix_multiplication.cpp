#include <gtest/gtest.h>

#include "math/matrix.h"

TEST(MatrixMultiplicationTest, MultipliesRectangularMatrixByColumnVector) {
    Matrix::Matrix<float> left(3, 4);
    Matrix::Matrix<float> right(4, 1);

    const float left_values[3][4] = {
        {1.0f, -2.0f, 3.0f, 4.0f},
        {-1.0f, 0.5f, 2.0f, -3.0f},
        {0.0f, 7.0f, -2.0f, 1.0f},
    };
    const float right_values[4] = {2.0f, -1.0f, 0.5f, 3.0f};

    for (size_t row = 0; row < left.rows(); ++row) {
        for (size_t column = 0; column < left.cols(); ++column) {
            left[row][column] = left_values[row][column];
        }
    }
    for (size_t row = 0; row < right.rows(); ++row) {
        right[row][0] = right_values[row];
    }

    const Matrix::Matrix<float> result = left * right;

    ASSERT_EQ(result.rows(), 3U);
    ASSERT_EQ(result.cols(), 1U);
    EXPECT_FLOAT_EQ(result[0][0], 17.5f);
    EXPECT_FLOAT_EQ(result[1][0], -10.5f);
    EXPECT_FLOAT_EQ(result[2][0], -5.0f);
}

TEST(MatrixMultiplicationTest, ColumnVectorUsesLeftToRightDotProductAccumulation) {
    Matrix::Matrix<float> left(1, 3);
    Matrix::Matrix<float> right(3, 1);

    left[0][0] = 1.0e20f;
    left[0][1] = -1.0e20f;
    left[0][2] = 3.25f;
    right[0][0] = 1.0f;
    right[1][0] = 1.0f;
    right[2][0] = 1.0f;

    float expected = 0.0f;
    for (size_t column = 0; column < left.cols(); ++column) {
        expected += left[0][column] * right[column][0];
    }

    const Matrix::Matrix<float> result = left * right;

    ASSERT_EQ(result.rows(), 1U);
    ASSERT_EQ(result.cols(), 1U);
    EXPECT_FLOAT_EQ(result[0][0], expected);
}
