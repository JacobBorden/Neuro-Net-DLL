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

TEST(MatrixMultiplicationTest, MatchesReferenceImplementation) {
    auto reference_multiply = [](const Matrix::Matrix<float>& A, const Matrix::Matrix<float>& B) {
        Matrix::Matrix<float> C(A.rows(), B.cols());
        for (size_t i = 0; i < A.rows(); ++i) {
            for (size_t j = 0; j < B.cols(); ++j) {
                float sum = 0.0f;
                for (size_t k = 0; k < A.cols(); ++k) {
                    sum += A[i][k] * B[k][j];
                }
                C[i][j] = sum;
            }
        }
        return C;
    };

    struct Workload {
        size_t m, k, n;
    };
    std::vector<Workload> workloads = {
        {10, 10, 10},    // Square
        {5, 20, 5},      // Rectangular
        {20, 5, 20},     // Rectangular
        {10, 10, 1},     // Vector
        {1, 10, 10}      // Row Vector
    };

    for (const auto& w : workloads) {
        Matrix::Matrix<float> A(w.m, w.k);
        Matrix::Matrix<float> B(w.k, w.n);

        for (size_t i = 0; i < w.m; ++i) {
            for (size_t k = 0; k < w.k; ++k) {
                A[i][k] = static_cast<float>(rand()) / RAND_MAX;
            }
        }
        for (size_t k = 0; k < w.k; ++k) {
            for (size_t j = 0; j < w.n; ++j) {
                B[k][j] = static_cast<float>(rand()) / RAND_MAX;
            }
        }

        Matrix::Matrix<float> C_actual = A * B;
        Matrix::Matrix<float> C_expected = reference_multiply(A, B);

        ASSERT_EQ(C_actual.rows(), C_expected.rows());
        ASSERT_EQ(C_actual.cols(), C_expected.cols());

        for (size_t i = 0; i < C_actual.rows(); ++i) {
            for (size_t j = 0; j < C_actual.cols(); ++j) {
                EXPECT_NEAR(C_actual[i][j], C_expected[i][j], 1e-4f);
            }
        }
    }
}
