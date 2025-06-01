#ifndef LEAST_SQUARES_H
#define LEAST_SQUARES_H

#include "matrix.h" // Defines Matrix::Matrix

namespace Math {
    /**
     * @brief Solves the least squares problem Ax = b.
     *
     * Given a design matrix A and a vector of observed values b, this function
     * finds the vector x that minimizes the squared Euclidean norm ||Ax - b||^2.
     * The solution is found by solving the normal equations: (A^T * A) * x = A^T * b.
     *
     * @param A The design matrix (m x n), assumed to be of type Matrix::Matrix<double>.
     * @param b The vector of observed values (m x 1), assumed to be of type Matrix::Matrix<double>.
     * @return Matrix::Matrix<double> The solution vector x (n x 1).
     * @throws std::runtime_error if A^T * A is singular or if matrix dimensions are incompatible.
     */
    Matrix::Matrix<double> solve_least_squares(const Matrix::Matrix<double>& A, const Matrix::Matrix<double>& b);
} // namespace Math

#endif // LEAST_SQUARES_H
