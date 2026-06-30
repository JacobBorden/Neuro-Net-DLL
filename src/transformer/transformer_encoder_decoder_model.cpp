#include "transformer_encoder_decoder_model.h"

namespace NeuroNet {
namespace Transformer {

TransformerEncoderDecoderModel::TransformerEncoderDecoderModel(
    int src_vocab_size,
    int tgt_vocab_size,
    int max_seq_len,
    int d_model,
    int num_encoder_layers,
    int num_decoder_layers,
    int num_heads,
    int d_ff,
    float dropout_rate,
    float layer_norm_epsilon)
    : src_vocab_size_(src_vocab_size),
      tgt_vocab_size_(tgt_vocab_size),
      max_seq_len_(max_seq_len),
      d_model_(d_model),
      num_encoder_layers_(num_encoder_layers),
      num_decoder_layers_(num_decoder_layers),
      num_heads_(num_heads),
      d_ff_(d_ff),
      dropout_rate_(dropout_rate),
      layer_norm_epsilon_(layer_norm_epsilon),
      src_embedding_layer_(src_vocab_size, d_model),
      tgt_embedding_layer_(tgt_vocab_size, d_model),
      positional_encoding_(max_seq_len, d_model) {

    if (d_model <= 0 || num_encoder_layers <= 0 || num_decoder_layers <= 0) {
        throw std::invalid_argument("Invalid hyperparameters for TransformerEncoderDecoderModel.");
    }

    encoder_layers_.reserve(num_encoder_layers_);
    for (int i = 0; i < num_encoder_layers_; ++i) {
        encoder_layers_.emplace_back(
            d_model_, num_heads_, d_ff_, dropout_rate_, dropout_rate_, layer_norm_epsilon_
        );
    }

    decoder_layers_.reserve(num_decoder_layers_);
    for (int i = 0; i < num_decoder_layers_; ++i) {
        decoder_layers_.emplace_back(
            d_model_, num_heads_, d_ff_, dropout_rate_, dropout_rate_, layer_norm_epsilon_
        );
    }
}

Matrix::Matrix<float> TransformerEncoderDecoderModel::forward(
    const Matrix::Matrix<float>& src_input_token_ids,
    const Matrix::Matrix<float>& tgt_input_token_ids,
    const Matrix::Matrix<float>& src_attention_mask,
    const Matrix::Matrix<float>& tgt_self_attention_mask,
    const Matrix::Matrix<float>& tgt_cross_attention_mask) {

    if (src_input_token_ids.rows() != 1 || tgt_input_token_ids.rows() != 1) {
        throw std::invalid_argument("Input token IDs must have exactly 1 row.");
    }

    // Encoder
    Matrix::Matrix<float> src_embeddings = src_embedding_layer_.forward(src_input_token_ids);
    Matrix::Matrix<float> src_pos_embeddings = positional_encoding_.forward(src_embeddings);
    Matrix::Matrix<float> encoder_output = src_pos_embeddings;

    for (int i = 0; i < num_encoder_layers_; ++i) {
        encoder_output = encoder_layers_[i].forward(encoder_output, src_attention_mask);
    }

    // Decoder
    Matrix::Matrix<float> tgt_embeddings = tgt_embedding_layer_.forward(tgt_input_token_ids);
    Matrix::Matrix<float> tgt_pos_embeddings = positional_encoding_.forward(tgt_embeddings);
    Matrix::Matrix<float> decoder_output = tgt_pos_embeddings;

    for (int i = 0; i < num_decoder_layers_; ++i) {
        decoder_output = decoder_layers_[i].forward(
            decoder_output, encoder_output, tgt_self_attention_mask, tgt_cross_attention_mask
        );
    }

    Matrix::Matrix<float> final_norm_output = MathUtils::layer_norm(decoder_output, layer_norm_epsilon_);
    return final_norm_output;
}

} // namespace Transformer
} // namespace NeuroNet
