#pragma once

#include "multi_head_attention.h"
#include "transformer_ffn.h"
#include "../math/matrix.h"
#include "../math/extended_matrix_ops.h" // For MathUtils::layer_norm
#include <stdexcept>

namespace NeuroNet {
namespace Transformer {

class TransformerEncoderLayer {
public:
    /**
     * @brief Constructor for TransformerEncoderLayer.
     * @param d_model Dimensionality of the input and output.
     * @param num_heads Number of attention heads for MultiHeadAttention.
     * @param d_ff Dimensionality of the inner feed-forward layer in TransformerFFN.
     * @param attention_dropout_rate Dropout rate for multi-head attention (currently unused).
     * @param ffn_dropout_rate Dropout rate for FFN (currently unused).
     * @param layer_norm_epsilon Epsilon value for Layer Normalization.
     */
    TransformerEncoderLayer(
        int d_model,
        int num_heads,
        int d_ff,
        float attention_dropout_rate = 0.0f, // Passed to MHA
        float ffn_dropout_rate = 0.0f,       // Passed to FFN
        float layer_norm_epsilon = 1e-5f
    );

    /**
     * @brief Initializes weights for sub-modules (MultiHeadAttention and TransformerFFN).
     * This method is called by the constructor.
     */
    void initialize_weights(); // Not strictly needed if sub-modules init themselves

    /**
     * @brief Performs the forward pass for the Transformer Encoder Layer.
     * Consists of: Multi-Head Self-Attention -> Add & Norm -> FFN -> Add & Norm.
     * @param input Input matrix, shape (seq_len, d_model).
     * @param attention_mask Optional mask for self-attention, shape (seq_len, seq_len).
     * @return Matrix::Matrix<float> The output matrix, shape (seq_len, d_model).
     * @throws std::invalid_argument if input dimensions are incorrect.
     */
    Matrix::Matrix<float> forward(
        const Matrix::Matrix<float>& input,
        const Matrix::Matrix<float>& attention_mask = Matrix::Matrix<float>(0,0)
    );

    // --- Accessors for sub-modules (useful for inspection, serialization, or fine-tuning) ---
    MultiHeadAttention& get_multi_head_attention_module() { return multi_head_attention_; }
    const MultiHeadAttention& get_multi_head_attention_module() const { return multi_head_attention_; }

    TransformerFFN& get_ffn_module() { return transformer_ffn_; }
    const TransformerFFN& get_ffn_module() const { return transformer_ffn_; }

    int get_d_model() const { return d_model_; }

private:
    int d_model_;
    float layer_norm_epsilon_;

    MultiHeadAttention multi_head_attention_;
    TransformerFFN transformer_ffn_;

    // Dropout layers are placeholders in MHA and FFN for now.
    // If implemented, they would be members here too, e.g., Dropout dropout_mha_, dropout_ffn_;
};

} // namespace Transformer
} // namespace NeuroNet
