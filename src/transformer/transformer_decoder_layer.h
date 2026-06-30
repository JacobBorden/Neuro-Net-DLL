#pragma once

#include "multi_head_attention.h"
#include "transformer_ffn.h"
#include "../math/matrix.h"
#include "../math/extended_matrix_ops.h" // For MathUtils::layer_norm
#include <stdexcept>

namespace NeuroNet {
namespace Transformer {

class TransformerDecoderLayer {
public:
    /**
     * @brief Constructor for TransformerDecoderLayer.
     * @param d_model Dimensionality of the input and output.
     * @param num_heads Number of attention heads for MultiHeadAttention.
     * @param d_ff Dimensionality of the inner feed-forward layer in TransformerFFN.
     * @param attention_dropout_rate Dropout rate for multi-head attention (currently unused).
     * @param ffn_dropout_rate Dropout rate for FFN (currently unused).
     * @param layer_norm_epsilon Epsilon value for Layer Normalization.
     */
    TransformerDecoderLayer(
        int d_model,
        int num_heads,
        int d_ff,
        float attention_dropout_rate = 0.0f,
        float ffn_dropout_rate = 0.0f,
        float layer_norm_epsilon = 1e-5f
    );

    /**
     * @brief Initializes weights for sub-modules.
     */
    void initialize_weights();

    /**
     * @brief Performs the forward pass for the Transformer Decoder Layer.
     * @param input Input matrix, shape (tgt_seq_len, d_model).
     * @param encoder_output Encoder output matrix, shape (src_seq_len, d_model).
     * @param self_attention_mask Optional mask for self-attention, shape (tgt_seq_len, tgt_seq_len).
     * @param cross_attention_mask Optional mask for cross-attention, shape (tgt_seq_len, src_seq_len).
     * @return Matrix::Matrix<float> The output matrix, shape (tgt_seq_len, d_model).
     * @throws std::invalid_argument if input dimensions are incorrect.
     */
    Matrix::Matrix<float> forward(
        const Matrix::Matrix<float>& input,
        const Matrix::Matrix<float>& encoder_output,
        const Matrix::Matrix<float>& self_attention_mask = Matrix::Matrix<float>(0,0),
        const Matrix::Matrix<float>& cross_attention_mask = Matrix::Matrix<float>(0,0)
    );

    MultiHeadAttention& get_self_attention_module() { return self_attention_; }
    const MultiHeadAttention& get_self_attention_module() const { return self_attention_; }

    MultiHeadAttention& get_cross_attention_module() { return cross_attention_; }
    const MultiHeadAttention& get_cross_attention_module() const { return cross_attention_; }

    TransformerFFN& get_ffn_module() { return transformer_ffn_; }
    const TransformerFFN& get_ffn_module() const { return transformer_ffn_; }

    int get_d_model() const { return d_model_; }

private:
    int d_model_;
    float layer_norm_epsilon_;

    MultiHeadAttention self_attention_;
    MultiHeadAttention cross_attention_;
    TransformerFFN transformer_ffn_;
};

} // namespace Transformer
} // namespace NeuroNet
