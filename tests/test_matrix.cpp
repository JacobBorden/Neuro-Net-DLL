#include <gtest/gtest.h>
#include "math/matrix.h"
#include <stdexcept>

TEST(MatrixTest, DeterminantNonSquareThrows) {
    Matrix::Matrix<float> mat(2, 3);
    EXPECT_THROW(mat.Determinant(), std::invalid_argument);
}
