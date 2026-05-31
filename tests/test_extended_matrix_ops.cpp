#include <gtest/gtest.h>
#include "math/extended_matrix_ops.h"
#include <cmath>
#include <stdexcept>

const float TOLERANCE = 1e-4f;

class SoftmaxTest : public ::testing::Test {
protected:
    Matrix::Matrix<float> create_test_matrix() {
        Matrix::Matrix<float> mat(2, 3);
        mat[0][0] = 1.0f; mat[0][1] = 2.0f; mat[0][2] = 3.0f;
        mat[1][0] = 4.0f; mat[1][1] = 5.0f; mat[1][2] = 6.0f;
        return mat;
    }
};

TEST_F(SoftmaxTest, RowWiseSoftmaxSumsToOne) {
    auto input = create_test_matrix();
    auto result = NeuroNet::MathUtils::softmax(input, 1);

    EXPECT_EQ(result.rows(), 2);
    EXPECT_EQ(result.cols(), 3);

    for (size_t i = 0; i < result.rows(); ++i) {
        float sum = 0.0f;
        for (size_t j = 0; j < result.cols(); ++j) {
            sum += result[i][j];
            EXPECT_GE(result[i][j], 0.0f); // probabilities are >= 0
            EXPECT_LE(result[i][j], 1.0f); // probabilities are <= 1
        }
        EXPECT_NEAR(sum, 1.0f, TOLERANCE);
    }
}

TEST_F(SoftmaxTest, RowWiseSoftmaxNegativeOneAxisSumsToOne) {
    auto input = create_test_matrix();
    auto result = NeuroNet::MathUtils::softmax(input, -1);

    EXPECT_EQ(result.rows(), 2);
    EXPECT_EQ(result.cols(), 3);

    for (size_t i = 0; i < result.rows(); ++i) {
        float sum = 0.0f;
        for (size_t j = 0; j < result.cols(); ++j) {
            sum += result[i][j];
            EXPECT_GE(result[i][j], 0.0f);
            EXPECT_LE(result[i][j], 1.0f);
        }
        EXPECT_NEAR(sum, 1.0f, TOLERANCE);
    }
}

TEST_F(SoftmaxTest, ColumnWiseSoftmaxSumsToOne) {
    auto input = create_test_matrix();
    auto result = NeuroNet::MathUtils::softmax(input, 0);

    EXPECT_EQ(result.rows(), 2);
    EXPECT_EQ(result.cols(), 3);

    for (size_t j = 0; j < result.cols(); ++j) {
        float sum = 0.0f;
        for (size_t i = 0; i < result.rows(); ++i) {
            sum += result[i][j];
            EXPECT_GE(result[i][j], 0.0f);
            EXPECT_LE(result[i][j], 1.0f);
        }
        EXPECT_NEAR(sum, 1.0f, TOLERANCE);
    }
}

TEST_F(SoftmaxTest, InvalidAxisThrows) {
    auto input = create_test_matrix();
    EXPECT_THROW(NeuroNet::MathUtils::softmax(input, 2), std::invalid_argument);
    EXPECT_THROW(NeuroNet::MathUtils::softmax(input, -2), std::invalid_argument);
}

TEST_F(SoftmaxTest, EmptyMatrixThrows) {
    Matrix::Matrix<float> empty_mat(0, 0);
    EXPECT_THROW(NeuroNet::MathUtils::softmax(empty_mat, 1), std::invalid_argument);
    EXPECT_THROW(NeuroNet::MathUtils::softmax(empty_mat, 0), std::invalid_argument);
}
