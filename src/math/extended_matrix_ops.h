#pragma once

#include "matrix.h" // Assuming this is the correct path to the existing matrix library
#include <cmath>   // For std::tanh, std::sqrt, std::pow

namespace NeuroNet {
namespace MathUtils {

/**
 * @brief Applies the GELU (Gaussian Error Linear Unit) activation function element-wise.
 * GELU(x) = 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
 * @param input The input matrix.
 * @return Matrix::Matrix<float> A new matrix with GELU applied.
 */
Matrix::Matrix<float> gelu(const Matrix::Matrix<float>& input);

/**
 * @brief Applies Layer Normalization to the input matrix.
 * Normalization is applied row-wise. Each row is treated as a separate sample/embedding.
 * Formula for each row x: y = (x - mean(x)) / sqrt(variance(x) + epsilon)
 * @param input The input matrix (e.g., batch_size x features or seq_len x embedding_dim).
 * @param epsilon A small value added to the variance for numerical stability.
 * @return Matrix::Matrix<float> The normalized matrix.
 */
Matrix::Matrix<float> layer_norm(const Matrix::Matrix<float>& input, float epsilon = 1e-5f);

/**
 * @brief Applies the Softmax function along a specified axis for numerical stability.
 * @param input The input matrix.
 * @param axis The axis along which to apply Softmax.
 *             axis = 0: column-wise (each column becomes a probability distribution).
 *             axis = 1 or -1: row-wise (each row becomes a probability distribution).
 * @return Matrix::Matrix<float> A new matrix with Softmax applied.
 * @throws std::invalid_argument if axis is not 0, 1, or -1, or if input matrix is empty along the specified axis.
 */
Matrix::Matrix<float> softmax(const Matrix::Matrix<float>& input, int axis = 1);

#include <vector> // For std::vector

/**
 * @brief Splits a matrix into multiple smaller matrices by dividing its columns.
 * The number of columns in the input matrix must be divisible by num_splits.
 * Each resulting matrix will have the same number of rows as the input.
 * @param input The matrix to split.
 * @param num_splits The number of ways to split the columns.
 * @return std::vector<Matrix::Matrix<float>> A vector of matrices, each representing a split.
 * @throws std::invalid_argument if input.cols() is not divisible by num_splits or if num_splits is zero.
 */
std::vector<Matrix::Matrix<float>> split_matrix_by_cols(const Matrix::Matrix<float>& input, int num_splits);

/**
 * @brief Combines a vector of matrices into a single matrix by concatenating them column-wise.
 * All input matrices in the vector must have the same number of rows.
 * If the input vector is empty, an empty matrix is returned.
 * If the vector contains one matrix, a copy of that matrix is returned.
 * @param inputs A vector of matrices to combine.
 * @return Matrix::Matrix<float> The resulting combined matrix.
 * @throws std::invalid_argument if matrices in the input vector have differing numbers of rows.
 */
Matrix::Matrix<float> combine_matrices_by_cols(const std::vector<Matrix::Matrix<float>>& inputs);

} // namespace MathUtils
} // namespace NeuroNet
