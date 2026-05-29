#include "attention.h"
#include <stdexcept> // For std::invalid_argument
#include <iostream>  // For debugging (optional)

namespace NeuroNet {
namespace Transformer {

ScaledDotProductAttention::ScaledDotProductAttention(float dropout_rate)
    : dropout_rate_(dropout_rate) {
    // Dropout is not implemented in this version.
    // If it were, we might initialize a random number generator or similar here.
}

AttentionOutput ScaledDotProductAttention::forward(
    const Matrix::Matrix<float>& query,
    const Matrix::Matrix<float>& key,
    const Matrix::Matrix<float>& value,
    const Matrix::Matrix<float>& mask) {

    // Validate dimensions
    // Q: (seq_len_q, d_k)
    // K: (seq_len_k, d_k)
    // V: (seq_len_v, d_v)
    // Mask: (seq_len_q, seq_len_k)
    // Output: (seq_len_q, d_v)
    // Attn Weights: (seq_len_q, seq_len_k)

    if (query.cols() != key.cols()) {
        throw std::invalid_argument(
            "Query and Key must have the same feature dimension (d_k). Query_cols: " +
            std::to_string(query.cols()) + ", Key_cols: " + std::to_string(key.cols()));
    }
    if (key.rows() != value.rows()) { // seq_len_k must equal seq_len_v
        throw std::invalid_argument(
            "Key and Value must have the same sequence length (seq_len_k == seq_len_v). Key_rows: " +
            std::to_string(key.rows()) + ", Value_rows: " + std::to_string(value.rows()));
    }

    size_t d_k = query.cols();
    if (d_k == 0) { // Cannot compute scale factor if d_k is 0
        throw std::invalid_argument("Feature dimension d_k cannot be zero.");
    }

    // 1. Calculate scores = Q * K^T
    // K is (seq_len_k, d_k), K.Transpose() is (d_k, seq_len_k)
    // Q is (seq_len_q, d_k)
    // scores will be (seq_len_q, seq_len_k)
    Matrix::Matrix<float> key_transposed = key.Transpose();
    Matrix::Matrix<float> scores = query * key_transposed; // Uses Matrix::operator*

    // 2. Scale scores
    float scale_factor = 1.0f / std::sqrt(static_cast<float>(d_k));
    Matrix::Matrix<float> scaled_scores = scores * scale_factor; // Uses Matrix::operator*(scalar)

    // 3. Apply mask (if provided)
    // Mask should have dimensions (seq_len_q, seq_len_k)
    bool use_mask = (mask.rows() > 0 && mask.cols() > 0);
    if (use_mask) {
        if (mask.rows() != scaled_scores.rows() || mask.cols() != scaled_scores.cols()) {
            throw std::invalid_argument(
                "Mask dimensions (" + std::to_string(mask.rows()) + "x" + std::to_string(mask.cols()) +
                ") must match attention score dimensions (" + std::to_string(scaled_scores.rows()) + "x" +
                std::to_string(scaled_scores.cols()) + ").");
        }
        // Element-wise addition
        scaled_scores = scaled_scores + mask; // Assumes Matrix::operator+ for element-wise addition
    }

    // 4. Calculate attention_weights = softmax(scaled_scores (or masked_scores), axis=1)
    // Softmax along the last dimension (cols of scaled_scores, which is seq_len_k)
    Matrix::Matrix<float> attention_weights = MathUtils::softmax(scaled_scores, 1); // axis=1 for row-wise

    // Dropout on attention_weights is not implemented in this version.

    // 5. Calculate output = attention_weights * V
    // attention_weights: (seq_len_q, seq_len_k)
    // V: (seq_len_v, d_v) where seq_len_v = seq_len_k
    // output: (seq_len_q, d_v)
    Matrix::Matrix<float> output_matrix = attention_weights * value; // Uses Matrix::operator*

    return {output_matrix, attention_weights};
}

} // namespace Transformer
} // namespace NeuroNet
