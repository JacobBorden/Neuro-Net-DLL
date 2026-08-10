#include "transformer_decoder_layer.h"
#include <iostream>

namespace NeuroNet {
namespace Transformer {

TransformerDecoderLayer::TransformerDecoderLayer(
    int d_model,
    int num_heads,
    int d_ff,
    float attention_dropout_rate,
    float ffn_dropout_rate,
    float layer_norm_epsilon)
    : d_model_(d_model),
      layer_norm_epsilon_(layer_norm_epsilon),
      self_attention_(num_heads, d_model, attention_dropout_rate),
      cross_attention_(num_heads, d_model, attention_dropout_rate),
      transformer_ffn_(d_model, d_ff, ffn_dropout_rate) {

    if (d_model <= 0) {
        throw std::invalid_argument("d_model must be positive for TransformerDecoderLayer.");
    }
}

void TransformerDecoderLayer::initialize_weights() {
    // sub-modules self initialize
}

Matrix::Matrix<float> TransformerDecoderLayer::forward(
    const Matrix::Matrix<float>& input,
    const Matrix::Matrix<float>& encoder_output,
    const Matrix::Matrix<float>& self_attention_mask,
    const Matrix::Matrix<float>& cross_attention_mask) {

    if (input.cols() != static_cast<size_t>(d_model_)) {
        throw std::invalid_argument("Input matrix column count (" + std::to_string(input.cols()) +
                                    ") must match TransformerDecoderLayer d_model (" + std::to_string(d_model_) + ").");
    }
    if (encoder_output.cols() != static_cast<size_t>(d_model_)) {
        throw std::invalid_argument("Encoder output matrix column count (" + std::to_string(encoder_output.cols()) +
                                    ") must match TransformerDecoderLayer d_model (" + std::to_string(d_model_) + ").");
    }
    if (input.rows() == 0) {
        return Matrix::Matrix<float>(0, d_model_);
    }

    // 1. Masked Multi-Head Self-Attention Block
    Matrix::Matrix<float> normed_input1 = MathUtils::layer_norm(input, layer_norm_epsilon_);
    Matrix::Matrix<float> self_attention_output = self_attention_.forward(
        normed_input1, normed_input1, normed_input1, self_attention_mask
    );
    Matrix::Matrix<float> residual_output1 = input + self_attention_output;

    // 2. Multi-Head Cross-Attention Block
    Matrix::Matrix<float> normed_input2 = MathUtils::layer_norm(residual_output1, layer_norm_epsilon_);
    Matrix::Matrix<float> cross_attention_output = cross_attention_.forward(
        normed_input2, encoder_output, encoder_output, cross_attention_mask
    );
    Matrix::Matrix<float> residual_output2 = residual_output1 + cross_attention_output;

    // 3. Feed-Forward Network Block
    Matrix::Matrix<float> normed_input3 = MathUtils::layer_norm(residual_output2, layer_norm_epsilon_);
    Matrix::Matrix<float> ffn_output = transformer_ffn_.forward(normed_input3);
    Matrix::Matrix<float> final_output = residual_output2 + ffn_output;

    return final_output;
}

} // namespace Transformer
} // namespace NeuroNet
