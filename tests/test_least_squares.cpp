#include "gtest/gtest.h"
#include "math/least_squares.h" // This should find solve_least_squares in Math namespace
#include "math/matrix.h"      // For Matrix::Matrix

const double TOLERANCE = 1e-3; // Adjusted tolerance for floating point comparisons

// Helper function to create a matrix from a 2D vector
Matrix::Matrix<double> create_matrix(const std::vector<std::vector<double>>& data) {
    if (data.empty() || data[0].empty()) {
        return Matrix::Matrix<double>(0, 0);
    }
    size_t rows = data.size();
    size_t cols = data[0].size();
    Matrix::Matrix<double> mat(rows, cols);
    for (size_t i = 0; i < rows; ++i) {
        if (data[i].size() != cols) {
            throw std::invalid_argument("All rows must have the same number of columns.");
        }
        for (size_t j = 0; j < cols; ++j) {
            mat[i][j] = data[i][j];
        }
    }
    return mat;
}

TEST(LeastSquaresTest, SimpleCase) {
    // A = [[1, 1], [1, 2], [1, 3]]
    // b = [[2], [3], [5]]
    // Expected x = [[0.333...], [1.5]]
    Matrix::Matrix<double> A = create_matrix({{1, 1}, {1, 2}, {1, 3}});
    Matrix::Matrix<double> b = create_matrix({{2}, {3}, {5}});

    Matrix::Matrix<double> x = Math::solve_least_squares(A, b);

    ASSERT_EQ(x.rows(), 2);
    ASSERT_EQ(x.cols(), 1);
    EXPECT_NEAR(x[0][0], 1.0/3.0, TOLERANCE);
    EXPECT_NEAR(x[1][0], 1.5, TOLERANCE);
}

TEST(LeastSquaresTest, OverdeterminedSystem) {
    // A = [[1, 0], [0, 1], [1, 1]]
    // b = [[1], [1], [3]]
    // A^T A = [[2, 1], [1, 2]]
    // A^T b = [[4], [4]]
    // (A^T A)^-1 = (1/3) * [[2, -1], [-1, 2]]
    // x = (1/3) * [[2, -1], [-1, 2]] * [[4], [4]] = (1/3) * [[4], [4]] = [[4/3], [4/3]]
    Matrix::Matrix<double> A = create_matrix({{1, 0}, {0, 1}, {1, 1}});
    Matrix::Matrix<double> b = create_matrix({{1}, {1}, {3}});

    Matrix::Matrix<double> x = Math::solve_least_squares(A, b);

    ASSERT_EQ(x.rows(), 2);
    ASSERT_EQ(x.cols(), 1);
    EXPECT_NEAR(x[0][0], 4.0/3.0, TOLERANCE);
    EXPECT_NEAR(x[1][0], 4.0/3.0, TOLERANCE);
}

TEST(LeastSquaresTest, PerfectFit) {
    // A = [[1, 2], [3, 4]]
    // b = [[5], [11]]
    // Ax = b has an exact solution x = [[1], [2]]
    // Least squares should find this exact solution.
    Matrix::Matrix<double> A = create_matrix({{1, 2}, {3, 4}});
    Matrix::Matrix<double> b = create_matrix({{5}, {11}});

    Matrix::Matrix<double> x = Math::solve_least_squares(A, b);

    ASSERT_EQ(x.rows(), 2);
    ASSERT_EQ(x.cols(), 1);
    EXPECT_NEAR(x[0][0], 1.0, TOLERANCE);
    EXPECT_NEAR(x[1][0], 2.0, TOLERANCE);
}

TEST(LeastSquaresTest, SingleUnknown) {
    // A = [[1], [2], [3]]
    // b = [[2], [4.5], [5.5]]
    // Expected x is approx mean of (b_i / A_i), which is not how LS works, but it's a simple system.
    // A^T A = [1*1 + 2*2 + 3*3] = [1+4+9] = [14]
    // A^T b = [1*2 + 2*4.5 + 3*5.5] = [2 + 9 + 16.5] = [27.5]
    // x = (1/14) * 27.5 = 27.5 / 14 = 1.9642857
    Matrix::Matrix<double> A = create_matrix({{1}, {2}, {3}});
    Matrix::Matrix<double> b = create_matrix({{2}, {4.5}, {5.5}});

    Matrix::Matrix<double> x = Math::solve_least_squares(A, b);

    ASSERT_EQ(x.rows(), 1);
    ASSERT_EQ(x.cols(), 1);
    EXPECT_NEAR(x[0][0], 27.5 / 14.0, TOLERANCE);
}

TEST(LeastSquaresTest, ErrorHandling_SingularAtA) {
    // A = [[1, 1], [1, 1], [1, 1]] (columns are linearly dependent)
    // A^T A will be singular
    // A^T A = [[1,1,1],[1,1,1]] * [[1,1],[1,1],[1,1]] = [[3,3],[3,3]] which is singular
    Matrix::Matrix<double> A = create_matrix({{1, 1}, {1, 1}, {1, 1}});
    Matrix::Matrix<double> b = create_matrix({{1}, {2}, {3}});

    EXPECT_THROW(Math::solve_least_squares(A, b), std::runtime_error);
}

TEST(LeastSquaresTest, ErrorHandling_MismatchedRows) {
    Matrix::Matrix<double> A = create_matrix({{1, 0}, {0, 1}});
    Matrix::Matrix<double> b = create_matrix({{1}}); // b has 1 row, A has 2
    EXPECT_THROW(Math::solve_least_squares(A, b), std::invalid_argument);
}

TEST(LeastSquaresTest, ErrorHandling_bNotColumnVector) {
    Matrix::Matrix<double> A = create_matrix({{1, 0}, {0, 1}});
    Matrix::Matrix<double> b = create_matrix({{1, 2}, {3, 4}}); // b has 2 columns
    EXPECT_THROW(Math::solve_least_squares(A, b), std::invalid_argument);
}

TEST(LeastSquaresTest, ErrorHandling_AEmpty) {
    Matrix::Matrix<double> A = create_matrix({});
    Matrix::Matrix<double> b = create_matrix({{1},{2}});
    EXPECT_THROW(Math::solve_least_squares(A, b), std::invalid_argument);
}

TEST(LeastSquaresTest, ErrorHandling_bEmpty) {
    Matrix::Matrix<double> A = create_matrix({{1,2},{3,4}});
    Matrix::Matrix<double> b = create_matrix({});
    EXPECT_THROW(Math::solve_least_squares(A, b), std::invalid_argument);
}
