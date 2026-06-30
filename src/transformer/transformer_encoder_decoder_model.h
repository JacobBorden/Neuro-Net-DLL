#pragma once

#include "embedding.h"
#include "positional_encoding.h"
#include "transformer_encoder_layer.h"
#include "transformer_decoder_layer.h"
#include "../math/matrix.h"
#include "../math/extended_matrix_ops.h"
#include <vector>
#include <stdexcept>

namespace NeuroNet {
namespace Transformer {

class TransformerEncoderDecoderModel {
public:
    /**
     * @brief Constructor for the full Transformer Encoder-Decoder model.
     * @param src_vocab_size Size of the vocabulary for the encoder embedding layer.
     * @param tgt_vocab_size Size of the vocabulary for the decoder embedding layer.
     * @param max_seq_len Maximum sequence length for positional encoding.
     * @param d_model Dimensionality of embeddings and model layers.
     * @param num_encoder_layers Number of TransformerEncoderLayer to stack.
     * @param num_decoder_layers Number of TransformerDecoderLayer to stack.
     * @param num_heads Number of attention heads in each layer.
     * @param d_ff Dimensionality of the feed-forward network within each layer.
     * @param dropout_rate General dropout rate (placeholder).
     * @param layer_norm_epsilon Epsilon for LayerNormalization.
     */
    TransformerEncoderDecoderModel(
        int src_vocab_size,
        int tgt_vocab_size,
        int max_seq_len,
        int d_model,
        int num_encoder_layers,
        int num_decoder_layers,
        int num_heads,
        int d_ff,
        float dropout_rate = 0.0f,
        float layer_norm_epsilon = 1e-5f
    );

    /**
     * @brief Performs the forward pass of the encoder-decoder model.
     * @param src_input_token_ids Matrix of source token IDs, shape (1, src_seq_len).
     * @param tgt_input_token_ids Matrix of target token IDs, shape (1, tgt_seq_len).
     * @param src_attention_mask Optional mask for encoder self-attention.
     * @param tgt_self_attention_mask Optional mask for decoder self-attention.
     * @param tgt_cross_attention_mask Optional mask for decoder cross-attention.
     * @return Matrix::Matrix<float> Output matrix from the final decoder layer,
     *                               after final layer normalization. Shape (tgt_seq_len, d_model).
     */
    Matrix::Matrix<float> forward(
        const Matrix::Matrix<float>& src_input_token_ids,
        const Matrix::Matrix<float>& tgt_input_token_ids,
        const Matrix::Matrix<float>& src_attention_mask = Matrix::Matrix<float>(0,0),
        const Matrix::Matrix<float>& tgt_self_attention_mask = Matrix::Matrix<float>(0,0),
        const Matrix::Matrix<float>& tgt_cross_attention_mask = Matrix::Matrix<float>(0,0)
    );

    // --- Accessors ---
    EmbeddingLayer& get_src_embedding_layer() { return src_embedding_layer_; }
    EmbeddingLayer& get_tgt_embedding_layer() { return tgt_embedding_layer_; }
    PositionalEncoding& get_positional_encoding_module() { return positional_encoding_; }
    std::vector<TransformerEncoderLayer>& get_encoder_layers() { return encoder_layers_; }
    std::vector<TransformerDecoderLayer>& get_decoder_layers() { return decoder_layers_; }

    int get_src_vocab_size() const { return src_vocab_size_; }
    int get_tgt_vocab_size() const { return tgt_vocab_size_; }
    int get_d_model() const { return d_model_; }
    int get_num_encoder_layers() const { return num_encoder_layers_; }
    int get_num_decoder_layers() const { return num_decoder_layers_; }

private:
    int src_vocab_size_;
    int tgt_vocab_size_;
    int max_seq_len_;
    int d_model_;
    int num_encoder_layers_;
    int num_decoder_layers_;
    int num_heads_;
    int d_ff_;
    float dropout_rate_;
    float layer_norm_epsilon_;

    EmbeddingLayer src_embedding_layer_;
    EmbeddingLayer tgt_embedding_layer_;
    PositionalEncoding positional_encoding_;
    std::vector<TransformerEncoderLayer> encoder_layers_;
    std::vector<TransformerDecoderLayer> decoder_layers_;
};

} // namespace Transformer
} // namespace NeuroNet
