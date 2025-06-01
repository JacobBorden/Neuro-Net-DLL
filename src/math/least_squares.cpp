#include "least_squares.h"
#include "matrix.h" // Defines Matrix::Matrix
#include <stdexcept> // For std::runtime_error

namespace Math {

    Matrix::Matrix<double> solve_least_squares(const Matrix::Matrix<double>& A, const Matrix::Matrix<double>& b) {
        if (A.rows() == 0 || A.cols() == 0) {
            throw std::invalid_argument("Input matrix A cannot be empty.");
        }
        if (b.rows() == 0 || b.cols() == 0) {
            throw std::invalid_argument("Input matrix b cannot be empty.");
        }
        if (A.rows() != b.rows()) {
            throw std::invalid_argument("Matrix A and matrix b must have the same number of rows.");
        }
        if (b.cols() != 1) {
            throw std::invalid_argument("Matrix b must be a column vector (have only 1 column).");
        }

        // Normal equations: (A^T * A) * x = A^T * b
        // Let At = A^T
        // Let AtA = At * A
        // Let Atb = At * b
        // Then AtA * x = Atb
        // So, x = (AtA)^-1 * Atb

        Matrix::Matrix<double> At = A.Transpose();
        Matrix::Matrix<double> AtA = At * A;
        Matrix::Matrix<double> Atb = At * b;

        // Check for singularity of AtA before attempting inverse
        // A simple check could be if determinant is close to zero,
        // but the Inverse() method already throws if singular.
        // For numerical stability, one might prefer other methods like SVD or QR decomposition,
        // but for this exercise, direct inverse is assumed as per the problem description.

        Matrix::Matrix<double> AtA_inv;
        try {
            AtA_inv = AtA.Inverse();
        } catch (const std::runtime_error& e) {
            // Re-throw with a more specific message or handle as per requirements
            throw std::runtime_error("Failed to solve least squares: A^T * A is singular. " + std::string(e.what()));
        }

        Matrix::Matrix<double> x = AtA_inv * Atb;

        return x;
    }

} // namespace Math
