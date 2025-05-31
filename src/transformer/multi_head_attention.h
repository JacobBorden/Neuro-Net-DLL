#pragma once

#include "attention.h" // For ScaledDotProductAttention and AttentionOutput
#include "../math/matrix.h"
#include "../math/extended_matrix_ops.h" // For split_matrix_by_cols, combine_matrices_by_cols
#include <vector>
#include <stdexcept> // For std::invalid_argument

namespace NeuroNet {
namespace Transformer {

class MultiHeadAttention {
public:
    /**
     * @brief Constructor for MultiHeadAttention.
     * @param num_heads Number of attention heads.
     * @param d_model Dimensionality of the input/output model. Must be divisible by num_heads.
     * @param dropout_rate Dropout rate (currently unused, placeholder).
     */
    MultiHeadAttention(int num_heads, int d_model, float dropout_rate = 0.0f);

    /**
     * @brief Initializes the weight matrices for projections.
     * Weights are initialized randomly using Matrix::Randomize().
     */
    void initialize_weights();

    /**
     * @brief Performs the forward pass for multi-head attention.
     * @param query_input Query input matrix, shape (seq_len_q, d_model).
     * @param key_input Key input matrix, shape (seq_len_k, d_model).
     * @param value_input Value input matrix, shape (seq_len_v, d_model).
     *                    (seq_len_k typically equals seq_len_v).
     * @param mask Optional attention mask, shape (seq_len_q, seq_len_k).
     *             Applied to each head's scaled dot-product attention.
     * @return Matrix::Matrix<float> The output matrix, shape (seq_len_q, d_model).
     *         (Note: Does not return individual head attention weights in this version for simplicity).
     * @throws std::invalid_argument if d_model is not divisible by num_heads or other dimension errors.
     */
    Matrix::Matrix<float> forward(
        const Matrix::Matrix<float>& query_input,
        const Matrix::Matrix<float>& key_input,
        const Matrix::Matrix<float>& value_input,
        const Matrix::Matrix<float>& mask = Matrix::Matrix<float>(0,0)
    );

    // --- Weight Accessors for Serialization/Training ---
    const Matrix::Matrix<float>& get_wq() const { return Wq_; }
    const Matrix::Matrix<float>& get_wk() const { return Wk_; }
    const Matrix::Matrix<float>& get_wv() const { return Wv_; }
    const Matrix::Matrix<float>& get_wo() const { return Wo_; }

    void set_wq(const Matrix::Matrix<float>& wq);
    void set_wk(const Matrix::Matrix<float>& wk);
    void set_wv(const Matrix::Matrix<float>& wv);
    void set_wo(const Matrix::Matrix<float>& wo);

    int get_num_heads() const { return num_heads_; }
    int get_d_model() const { return d_model_; }
    int get_d_head() const { return d_head_; }


private:
    int num_heads_;
    int d_model_;
    int d_head_; // d_model / num_heads

    // Projection weight matrices
    Matrix::Matrix<float> Wq_; // Shape: (d_model, d_model)
    Matrix::Matrix<float> Wk_; // Shape: (d_model, d_model)
    Matrix::Matrix<float> Wv_; // Shape: (d_model, d_model)
    Matrix::Matrix<float> Wo_; // Shape: (d_model, d_model)

    ScaledDotProductAttention attention_module_; // Each head uses this
    float dropout_rate_; // Placeholder
};

} // namespace Transformer
} // namespace NeuroNet
