#include "multi_head_attention.h"
#include <iostream> // For debugging (optional)

namespace NeuroNet {
namespace Transformer {

MultiHeadAttention::MultiHeadAttention(int num_heads, int d_model, float dropout_rate)
    : num_heads_(num_heads), d_model_(d_model), dropout_rate_(dropout_rate) {
    if (d_model <= 0 || num_heads <= 0) {
        throw std::invalid_argument("d_model and num_heads must be positive.");
    }
    if (d_model % num_heads != 0) {
        throw std::invalid_argument("d_model must be divisible by num_heads.");
    }
    d_head_ = d_model / num_heads;

    // Initialize projection matrices
    Wq_.resize(d_model_, d_model_);
    Wk_.resize(d_model_, d_model_);
    Wv_.resize(d_model_, d_model_);
    Wo_.resize(d_model_, d_model_);
    initialize_weights();

    // attention_module_ is default constructed (dropout_rate can be passed if it's used there)
    // For this version, ScaledDotProductAttention's dropout is also a placeholder.
    attention_module_ = ScaledDotProductAttention(dropout_rate_);
}

void MultiHeadAttention::initialize_weights() {
    Wq_.Randomize();
    Wk_.Randomize();
    Wv_.Randomize();
    Wo_.Randomize();
}

Matrix::Matrix<float> MultiHeadAttention::forward(
    const Matrix::Matrix<float>& query_input,    // (seq_len_q, d_model)
    const Matrix::Matrix<float>& key_input,      // (seq_len_k, d_model)
    const Matrix::Matrix<float>& value_input,    // (seq_len_v, d_model)
    const Matrix::Matrix<float>& mask) {

    if (query_input.cols() != static_cast<size_t>(d_model_) ||
        key_input.cols() != static_cast<size_t>(d_model_) ||
        value_input.cols() != static_cast<size_t>(d_model_)) {
        throw std::invalid_argument("Input matrix column count must match d_model (" + std::to_string(d_model_) + ").");
    }

    // 1. Linear Projections
    // Q_proj = query_input * Wq_ : (seq_len_q, d_model) * (d_model, d_model) -> (seq_len_q, d_model)
    Matrix::Matrix<float> Q_projected = query_input * Wq_;
    Matrix::Matrix<float> K_projected = key_input * Wk_;
    Matrix::Matrix<float> V_projected = value_input * Wv_;

    // 2. Split Q, K, V into heads
    // Each is split from (seq_len, d_model) into num_heads_ matrices of (seq_len, d_head_)
    // The split_matrix_by_cols function splits based on columns.
    // This means we project first to (seq_len, d_model) and then view this as (seq_len, num_heads * d_head).
    // We then want to process each head: (seq_len, d_head).
    // This requires a conceptual transpose or careful handling if we were in a true tensor library.
    // With 2D matrices, Q_projected (seq_len_q, d_model) is what we have.
    // We need Q_h (seq_len_q, d_head) for each head.
    // The most straightforward way with current tools is to split the *projected* Q, K, V.

    std::vector<Matrix::Matrix<float>> Q_heads = MathUtils::split_matrix_by_cols(Q_projected, num_heads_);
    std::vector<Matrix::Matrix<float>> K_heads = MathUtils::split_matrix_by_cols(K_projected, num_heads_);
    std::vector<Matrix::Matrix<float>> V_heads = MathUtils::split_matrix_by_cols(V_projected, num_heads_);

    // Each Q_heads[h] is (seq_len_q, d_head), K_heads[h] is (seq_len_k, d_head), V_heads[h] is (seq_len_v, d_head)

    // 3. Apply attention for each head
    std::vector<Matrix::Matrix<float>> head_outputs;
    head_outputs.reserve(num_heads_);

    for (int h = 0; h < num_heads_; ++h) {
        // The mask (if provided) applies to the attention scores within each head.
        // Its dimensions should be (seq_len_q, seq_len_k).
        AttentionOutput single_head_attention_output = attention_module_.forward(
            Q_heads[h], K_heads[h], V_heads[h], mask
        );
        head_outputs.push_back(single_head_attention_output.output); // Each is (seq_len_q, d_head)
    }

    // 4. Concatenate head outputs
    // head_outputs contains num_heads_ matrices, each of shape (seq_len_q, d_head).
    // Combining them by columns results in (seq_len_q, num_heads_ * d_head) which is (seq_len_q, d_model).
    Matrix::Matrix<float> concatenated_output;
    if (!head_outputs.empty()) {
        concatenated_output = MathUtils::combine_matrices_by_cols(head_outputs);
    } else {
        // Should not happen if num_heads > 0. Handle defensively.
        // Output shape should be (seq_len_q, d_model)
        concatenated_output.resize(query_input.rows(), d_model_);
        concatenated_output.assign(0.0f); // Fill with zeros
    }


    // 5. Final linear projection
    // Output = concatenated_output * Wo_ : (seq_len_q, d_model) * (d_model, d_model) -> (seq_len_q, d_model)
    Matrix::Matrix<float> final_output = concatenated_output * Wo_;

    return final_output;
}

// --- Weight Accessors ---
void MultiHeadAttention::set_wq(const Matrix::Matrix<float>& wq) {
    if (wq.rows() != static_cast<size_t>(d_model_) || wq.cols() != static_cast<size_t>(d_model_))
        throw std::invalid_argument("Wq dimensions mismatch.");
    Wq_ = wq;
}
void MultiHeadAttention::set_wk(const Matrix::Matrix<float>& wk) {
    if (wk.rows() != static_cast<size_t>(d_model_) || wk.cols() != static_cast<size_t>(d_model_))
        throw std::invalid_argument("Wk dimensions mismatch.");
    Wk_ = wk;
}
void MultiHeadAttention::set_wv(const Matrix::Matrix<float>& wv) {
    if (wv.rows() != static_cast<size_t>(d_model_) || wv.cols() != static_cast<size_t>(d_model_))
        throw std::invalid_argument("Wv dimensions mismatch.");
    Wv_ = wv;
}
void MultiHeadAttention::set_wo(const Matrix::Matrix<float>& wo) {
    if (wo.rows() != static_cast<size_t>(d_model_) || wo.cols() != static_cast<size_t>(d_model_))
        throw std::invalid_argument("Wo dimensions mismatch.");
    Wo_ = wo;
}

} // namespace Transformer
} // namespace NeuroNet
