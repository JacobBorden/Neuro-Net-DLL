#pragma once

#include "../math/matrix.h"
#include "../math/extended_matrix_ops.h" // For MathUtils::softmax
#include <cmath> // For std::sqrt
#include <string> // For std::to_string in exceptions

namespace NeuroNet {
namespace Transformer {

struct AttentionOutput {
    Matrix::Matrix<float> output;            // Shape: (seq_len_q, d_v)
    Matrix::Matrix<float> attention_weights; // Shape: (seq_len_q, seq_len_k)
};

class ScaledDotProductAttention {
public:
    /**
     * @brief Constructor for ScaledDotProductAttention.
     * @param dropout_rate Rate for dropout (0.0 to 1.0). Not implemented in this version, placeholder for future.
     */
    explicit ScaledDotProductAttention(float dropout_rate = 0.0f); // dropout_rate currently unused

    /**
     * @brief Performs the forward pass for scaled dot-product attention.
     * Calculates: softmax((Q * K^T) / sqrt(d_k) + mask) * V
     * @param query The Query matrix, shape (seq_len_q, d_k).
     * @param key The Key matrix, shape (seq_len_k, d_k).
     * @param value The Value matrix, shape (seq_len_v, d_v), where seq_len_k typically equals seq_len_v.
     * @param mask Optional mask matrix, shape (seq_len_q, seq_len_k).
     *             Values in the mask are added to the attention scores before softmax.
     *             Masked positions (e.g., padding) should have large negative values (like -1e9f).
     *             If mask.rows() or mask.cols() is 0, it's ignored.
     * @return AttentionOutput struct containing the output matrix and attention weights.
     * @throws std::invalid_argument if matrix dimensions are incompatible.
     */
    AttentionOutput forward(
        const Matrix::Matrix<float>& query,
        const Matrix::Matrix<float>& key,
        const Matrix::Matrix<float>& value,
        const Matrix::Matrix<float>& mask = Matrix::Matrix<float>(0,0) // Default empty matrix
    );

private:
    float dropout_rate_; // Placeholder, not currently used in implementation
};

} // namespace Transformer
} // namespace NeuroNet
