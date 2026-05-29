#include "extended_matrix_ops.h"
#include <numeric> // For std::accumulate (though manual sum is often clearer for matrices)
#include <stdexcept> // For std::runtime_error

namespace NeuroNet {
namespace MathUtils {

Matrix::Matrix<float> gelu(const Matrix::Matrix<float>& input) {
    if (input.rows() == 0 || input.cols() == 0) {
        return Matrix::Matrix<float>(input.rows(), input.cols()); // Return empty/original if input is empty
    }
    Matrix::Matrix<float> output(input.rows(), input.cols());
    constexpr float M_SQRT2_OVER_PI = 0.7978845608028654f; // sqrt(2/PI)

    for (size_t i = 0; i < input.rows(); ++i) {
        for (size_t j = 0; j < input.cols(); ++j) {
            float x = input[i][j];
            float x_cubed = x * x * x;
            float inner = M_SQRT2_OVER_PI * (x + 0.044715f * x_cubed);
            output[i][j] = 0.5f * x * (1.0f + std::tanh(inner));
        }
    }
    return output;
}

Matrix::Matrix<float> softmax(const Matrix::Matrix<float>& input, int axis) {
    if (axis != 0 && axis != 1 && axis != -1) {
        throw std::invalid_argument("Softmax axis must be 0 (column-wise) or 1/-1 (row-wise).");
    }

    size_t rows = input.rows();
    size_t cols = input.cols();

    if (rows == 0 || cols == 0) {
        return Matrix::Matrix<float>(rows, cols); // Return empty/original if input is empty
    }

    Matrix::Matrix<float> output(rows, cols);

    if (axis == 1 || axis == -1) { // Row-wise Softmax
        for (size_t i = 0; i < rows; ++i) {
            float max_val = input[i][0];
            for (size_t j = 1; j < cols; ++j) {
                if (input[i][j] > max_val) {
                    max_val = input[i][j];
                }
            }

            float sum_exp = 0.0f;
            for (size_t j = 0; j < cols; ++j) {
                output[i][j] = std::exp(input[i][j] - max_val);
                sum_exp += output[i][j];
            }

            if (sum_exp == 0.0f) { // Avoid division by zero; should be rare with exp
                 // This case implies all exp(input[i][j] - max_val) were zero, which means
                 // all input[i][j] - max_val were very small negative numbers.
                 // Assign uniform probability if sum_exp is zero.
                for (size_t j = 0; j < cols; ++j) {
                    output[i][j] = 1.0f / static_cast<float>(cols);
                }
            } else {
                for (size_t j = 0; j < cols; ++j) {
                    output[i][j] /= sum_exp;
                }
            }
        }
    } else { // Column-wise Softmax (axis == 0)
        for (size_t j = 0; j < cols; ++j) {
            float max_val = input[0][j];
            for (size_t i = 1; i < rows; ++i) {
                if (input[i][j] > max_val) {
                    max_val = input[i][j];
                }
            }

            float sum_exp = 0.0f;
            for (size_t i = 0; i < rows; ++i) {
                // Store intermediate exp values in output matrix temporarily
                output[i][j] = std::exp(input[i][j] - max_val);
                sum_exp += output[i][j];
            }

            if (sum_exp == 0.0f) {
                for (size_t i = 0; i < rows; ++i) {
                    output[i][j] = 1.0f / static_cast<float>(rows);
                }
            } else {
                for (size_t i = 0; i < rows; ++i) {
                    output[i][j] /= sum_exp;
                }
            }
        }
    }
    return output;
}

Matrix::Matrix<float> layer_norm(const Matrix::Matrix<float>& input, float epsilon) {
    if (input.rows() == 0) { // Handle empty input (no rows)
        return Matrix::Matrix<float>(0, input.cols());
    }
    if (input.cols() == 0) { // Handle input with no features/columns
        return Matrix::Matrix<float>(input.rows(), 0);
    }

    Matrix::Matrix<float> output(input.rows(), input.cols());

    for (size_t i = 0; i < input.rows(); ++i) {
        float sum = 0.0f;
        for (size_t j = 0; j < input.cols(); ++j) {
            sum += input[i][j];
        }
        float mean = sum / static_cast<float>(input.cols());

        float sq_sum_diff = 0.0f;
        for (size_t j = 0; j < input.cols(); ++j) {
            float diff = input[i][j] - mean;
            sq_sum_diff += diff * diff;
        }
        float variance = sq_sum_diff / static_cast<float>(input.cols());
        float inv_std_dev = 1.0f / std::sqrt(variance + epsilon);

        for (size_t j = 0; j < input.cols(); ++j) {
            output[i][j] = (input[i][j] - mean) * inv_std_dev;
        }
    }
    return output;
}

#include <vector> // For std::vector (already included but good for clarity)

std::vector<Matrix::Matrix<float>> split_matrix_by_cols(const Matrix::Matrix<float>& input, int num_splits) {
    if (num_splits <= 0) {
        throw std::invalid_argument("Number of splits must be greater than zero.");
    }
    if (input.cols() == 0 && num_splits > 0) { // Handle splitting an empty matrix
        std::vector<Matrix::Matrix<float>> splits(num_splits);
        for(int i=0; i < num_splits; ++i) {
            splits[i].resize(input.rows(), 0);
        }
        return splits;
    }
    if (input.cols() % num_splits != 0) {
        throw std::invalid_argument("Number of columns in input matrix must be divisible by num_splits.");
    }

    std::vector<Matrix::Matrix<float>> splits;
    splits.reserve(num_splits);
    size_t original_rows = input.rows();
    size_t split_cols = input.cols() / num_splits;

    for (int k = 0; k < num_splits; ++k) {
        Matrix::Matrix<float> current_split(original_rows, split_cols);
        size_t start_col_original = k * split_cols;
        for (size_t i = 0; i < original_rows; ++i) {
            for (size_t j = 0; j < split_cols; ++j) {
                current_split[i][j] = input[i][start_col_original + j];
            }
        }
        splits.push_back(current_split);
    }
    return splits;
}

Matrix::Matrix<float> combine_matrices_by_cols(const std::vector<Matrix::Matrix<float>>& inputs) {
    if (inputs.empty()) {
        return Matrix::Matrix<float>(0, 0);
    }
    if (inputs.size() == 1) {
        return inputs[0]; // Return a copy
    }

    size_t num_rows = inputs[0].rows();
    size_t total_cols = 0;
    for (const auto& m : inputs) {
        if (m.rows() != num_rows) {
            throw std::invalid_argument("All matrices to be combined must have the same number of rows.");
        }
        total_cols += m.cols();
    }

    if (num_rows == 0) { // All inputs are empty row-wise, but might have columns
        return Matrix::Matrix<float>(0, total_cols);
    }
    if (total_cols == 0) { // All inputs are empty column-wise
         return Matrix::Matrix<float>(num_rows, 0);
    }


    Matrix::Matrix<float> combined_matrix(num_rows, total_cols);
    size_t current_col_offset = 0;

    for (const auto& input_matrix : inputs) {
        for (size_t i = 0; i < num_rows; ++i) {
            for (size_t j = 0; j < input_matrix.cols(); ++j) {
                combined_matrix[i][current_col_offset + j] = input_matrix[i][j];
            }
        }
        current_col_offset += input_matrix.cols();
    }
    return combined_matrix;
}

} // namespace MathUtils
} // namespace NeuroNet
