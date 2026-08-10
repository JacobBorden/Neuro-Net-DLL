#include <gtest/gtest.h>
#include "math/extended_matrix_ops.h"
#include <cmath>
#include <stdexcept>
#include <vector>

const float TOLERANCE = 1e-4f;

class ExtendedMatrixOpsTest : public ::testing::Test {
protected:
    bool is_approx(float a, float b, float epsilon = 1e-5f) {
        return std::abs(a - b) < epsilon;
    }
};

TEST_F(ExtendedMatrixOpsTest, GeluStandardValues) {
    Matrix::Matrix<float> input(1, 4);
    input[0][0] = 0.0f;
    input[0][1] = 1.0f;
    input[0][2] = -1.0f;
    input[0][3] = 2.0f;

    Matrix::Matrix<float> result = NeuroNet::MathUtils::gelu(input);

    ASSERT_EQ(result.rows(), 1);
    ASSERT_EQ(result.cols(), 4);
    EXPECT_TRUE(is_approx(result[0][0], 0.0f));
    EXPECT_TRUE(is_approx(result[0][1], 0.84119f));
    EXPECT_TRUE(is_approx(result[0][2], -0.15881f));
    EXPECT_TRUE(is_approx(result[0][3], 1.9546f, 1e-4f));
}

TEST_F(ExtendedMatrixOpsTest, LayerNormStandard) {
    Matrix::Matrix<float> input(2, 3);
    input[0][0] = 1.0f; input[0][1] = 2.0f; input[0][2] = 3.0f;
    input[1][0] = 4.0f; input[1][1] = 4.0f; input[1][2] = 4.0f;

    float epsilon = 1e-5f;
    Matrix::Matrix<float> result = NeuroNet::MathUtils::layer_norm(input, epsilon);

    ASSERT_EQ(result.rows(), 2);
    ASSERT_EQ(result.cols(), 3);

    float expected_std_1 = std::sqrt(2.0f / 3.0f + epsilon);
    EXPECT_TRUE(is_approx(result[0][0], (1.0f - 2.0f) / expected_std_1));
    EXPECT_TRUE(is_approx(result[0][1], (2.0f - 2.0f) / expected_std_1));
    EXPECT_TRUE(is_approx(result[0][2], (3.0f - 2.0f) / expected_std_1));

    EXPECT_TRUE(is_approx(result[1][0], 0.0f));
    EXPECT_TRUE(is_approx(result[1][1], 0.0f));
    EXPECT_TRUE(is_approx(result[1][2], 0.0f));
}

TEST_F(ExtendedMatrixOpsTest, SplitMatrixByColsSuccess) {
    Matrix::Matrix<float> input(2, 4);
    input[0][0] = 1; input[0][1] = 2; input[0][2] = 3; input[0][3] = 4;
    input[1][0] = 5; input[1][1] = 6; input[1][2] = 7; input[1][3] = 8;

    std::vector<Matrix::Matrix<float>> result = NeuroNet::MathUtils::split_matrix_by_cols(input, 2);

    ASSERT_EQ(result.size(), 2);
    ASSERT_EQ(result[0].rows(), 2);
    ASSERT_EQ(result[0].cols(), 2);
    EXPECT_EQ(result[0][0][0], 1);
    EXPECT_EQ(result[0][0][1], 2);
    EXPECT_EQ(result[0][1][0], 5);
    EXPECT_EQ(result[0][1][1], 6);

    ASSERT_EQ(result[1].rows(), 2);
    ASSERT_EQ(result[1].cols(), 2);
    EXPECT_EQ(result[1][0][0], 3);
    EXPECT_EQ(result[1][0][1], 4);
    EXPECT_EQ(result[1][1][0], 7);
    EXPECT_EQ(result[1][1][1], 8);
}

TEST_F(ExtendedMatrixOpsTest, SplitMatrixByColsInvalid) {
    Matrix::Matrix<float> input(2, 4);
    EXPECT_THROW(NeuroNet::MathUtils::split_matrix_by_cols(input, 3), std::invalid_argument);
    EXPECT_THROW(NeuroNet::MathUtils::split_matrix_by_cols(input, 0), std::invalid_argument);
}

TEST_F(ExtendedMatrixOpsTest, CombineMatricesByColsSuccess) {
    Matrix::Matrix<float> m1(2, 2);
    m1[0][0] = 1; m1[0][1] = 2;
    m1[1][0] = 5; m1[1][1] = 6;

    Matrix::Matrix<float> m2(2, 2);
    m2[0][0] = 3; m2[0][1] = 4;
    m2[1][0] = 7; m2[1][1] = 8;

    std::vector<Matrix::Matrix<float>> inputs = {m1, m2};
    Matrix::Matrix<float> result = NeuroNet::MathUtils::combine_matrices_by_cols(inputs);

    ASSERT_EQ(result.rows(), 2);
    ASSERT_EQ(result.cols(), 4);
    EXPECT_EQ(result[0][0], 1);
    EXPECT_EQ(result[0][1], 2);
    EXPECT_EQ(result[0][2], 3);
    EXPECT_EQ(result[0][3], 4);
    EXPECT_EQ(result[1][0], 5);
    EXPECT_EQ(result[1][1], 6);
    EXPECT_EQ(result[1][2], 7);
    EXPECT_EQ(result[1][3], 8);
}

TEST_F(ExtendedMatrixOpsTest, CombineMatricesByColsEmpty) {
    std::vector<Matrix::Matrix<float>> inputs;
    Matrix::Matrix<float> result = NeuroNet::MathUtils::combine_matrices_by_cols(inputs);

    EXPECT_EQ(result.rows(), 0);
    EXPECT_EQ(result.cols(), 0);
}

TEST_F(ExtendedMatrixOpsTest, CombineMatricesByColsMismatchedRows) {
    Matrix::Matrix<float> m1(2, 2);
    Matrix::Matrix<float> m2(3, 2);
    std::vector<Matrix::Matrix<float>> inputs = {m1, m2};

    EXPECT_THROW(NeuroNet::MathUtils::combine_matrices_by_cols(inputs), std::invalid_argument);
}

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
