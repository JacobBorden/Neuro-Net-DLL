#include <gtest/gtest.h>
#include "math/matrix.h"
#include <stdexcept>

class MatrixTest : public ::testing::Test {
protected:
    Matrix::Matrix<float> create_test_matrix() {
        Matrix::Matrix<float> mat(2, 3);
        mat[0][0] = 1.0f; mat[0][1] = 2.0f; mat[0][2] = 3.0f;
        mat[1][0] = 4.0f; mat[1][1] = 5.0f; mat[1][2] = 6.0f;
        return mat;
    }
};

TEST_F(MatrixTest, DivisionByZeroThrows) {
    auto mat = create_test_matrix();

    // Division by zero
    EXPECT_THROW(mat / 0.0f, std::runtime_error);
    EXPECT_THROW(mat /= 0.0f, std::runtime_error);

    // Division by very small number (< 1e-9)
    EXPECT_THROW(mat / 1e-10f, std::runtime_error);
    EXPECT_THROW(mat /= 1e-10f, std::runtime_error);

    // Division by valid number (>= 1e-9)
    EXPECT_NO_THROW({
        (void)(mat / 1e-8f);
    });
    EXPECT_NO_THROW({
        mat /= 1e-8f;
    });
}
