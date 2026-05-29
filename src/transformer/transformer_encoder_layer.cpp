#include "transformer_encoder_layer.h"
#include <iostream> // For debugging (optional)

namespace NeuroNet {
namespace Transformer {

TransformerEncoderLayer::TransformerEncoderLayer(
    int d_model,
    int num_heads,
    int d_ff,
    float attention_dropout_rate,
    float ffn_dropout_rate,
    float layer_norm_epsilon)
    : d_model_(d_model),
      layer_norm_epsilon_(layer_norm_epsilon),
      multi_head_attention_(num_heads, d_model, attention_dropout_rate),
      transformer_ffn_(d_model, d_ff, ffn_dropout_rate) {

    if (d_model <= 0) {
        throw std::invalid_argument("d_model must be positive for TransformerEncoderLayer.");
    }
    // Sub-modules (MHA, FFN) constructors already validate their specific parameters (num_heads, d_ff)
    // and initialize their own weights.
}

// initialize_weights() is not strictly needed here as MHA and FFN constructors call their own init.
// If there were weights directly in this class, this method would handle them.
void TransformerEncoderLayer::initialize_weights() {
    // multi_head_attention_.initialize_weights(); // Already done in MHA constructor
    // transformer_ffn_.initialize_weights();    // Already done in FFN constructor
}

Matrix::Matrix<float> TransformerEncoderLayer::forward(
    const Matrix::Matrix<float>& input,
    const Matrix::Matrix<float>& attention_mask) {

    if (input.cols() != static_cast<size_t>(d_model_)) {
        throw std::invalid_argument("Input matrix column count (" + std::to_string(input.cols()) +
                                    ") must match TransformerEncoderLayer d_model (" + std::to_string(d_model_) + ").");
    }
     if (input.rows() == 0) { // Handle empty sequence
        return Matrix::Matrix<float>(0, d_model_);
    }

    // 1. Multi-Head Self-Attention Block
    // 1a. Layer Normalization before attention
    Matrix::Matrix<float> normed_input1 = MathUtils::layer_norm(input, layer_norm_epsilon_);

    // 1b. Multi-Head Attention
    // Input to MHA is (seq_len, d_model). Output is also (seq_len, d_model).
    Matrix::Matrix<float> attention_output = multi_head_attention_.forward(
        normed_input1, normed_input1, normed_input1, attention_mask // Self-attention: Q, K, V are the same
    );

    // Dropout after attention_output (not implemented)

    // 1c. Residual Connection (Add)
    // Output = Input + AttentionOutput
    // Assumes Matrix class supports element-wise addition via operator+
    Matrix::Matrix<float> residual_output1 = input + attention_output;


    // 2. Feed-Forward Network Block
    // 2a. Layer Normalization before FFN
    Matrix::Matrix<float> normed_input2 = MathUtils::layer_norm(residual_output1, layer_norm_epsilon_);

    // 2b. FFN
    // Input to FFN is (seq_len, d_model). Output is also (seq_len, d_model).
    Matrix::Matrix<float> ffn_output = transformer_ffn_.forward(normed_input2);

    // Dropout after ffn_output (not implemented)

    // 2c. Residual Connection (Add)
    // Output = PreviousBlockOutput + FFNOutput
    Matrix::Matrix<float> final_output = residual_output1 + ffn_output;

    return final_output;
}

} // namespace Transformer
} // namespace NeuroNet
