#include "transformer_ffn.h"
#include <iostream> // For debugging (optional)

namespace NeuroNet {
namespace Transformer {

TransformerFFN::TransformerFFN(int d_model, int d_ff, float dropout_rate)
    : d_model_(d_model), d_ff_(d_ff), dropout_rate_(dropout_rate) {
    if (d_model <= 0 || d_ff <= 0) {
        throw std::invalid_argument("d_model and d_ff must be positive.");
    }

    // Initialize weight and bias matrices
    W1_.resize(d_model_, d_ff_);
    b1_.resize(1, d_ff_); // Bias is a row vector, to be broadcasted
    W2_.resize(d_ff_, d_model_);
    b2_.resize(1, d_model_); // Bias is a row vector

    initialize_weights();
}

void TransformerFFN::initialize_weights() {
    W1_.Randomize();
    W2_.Randomize();
    b1_.assign(0.0f); // Initialize biases to zero
    b2_.assign(0.0f);
}

Matrix::Matrix<float> TransformerFFN::forward(const Matrix::Matrix<float>& input) {
    if (input.cols() != static_cast<size_t>(d_model_)) {
        throw std::invalid_argument("Input matrix column count (" + std::to_string(input.cols()) +
                                    ") must match FFN d_model (" + std::to_string(d_model_) + ").");
    }
    if (input.rows() == 0) { // Handle empty sequence
        return Matrix::Matrix<float>(0, d_model_);
    }

    // Layer 1: input * W1
    Matrix::Matrix<float> hidden_linear = input * W1_; // (seq_len, d_model) * (d_model, d_ff) -> (seq_len, d_ff)

    // Add bias b1 (broadcasting)
    // The Matrix library might not support direct broadcasting of (1, d_ff) to (seq_len, d_ff).
    // We need to manually add b1 to each row of hidden_linear.
    Matrix::Matrix<float> hidden_biased(hidden_linear.rows(), hidden_linear.cols());
    for(size_t r = 0; r < hidden_linear.rows(); ++r) {
        for(size_t c = 0; c < hidden_linear.cols(); ++c) {
            hidden_biased[r][c] = hidden_linear[r][c] + b1_[0][c];
        }
    }

    // Activation: GELU
    Matrix::Matrix<float> hidden_activated = MathUtils::gelu(hidden_biased);

    // Dropout is not implemented here.

    // Layer 2: hidden_activated * W2
    Matrix::Matrix<float> output_linear = hidden_activated * W2_; // (seq_len, d_ff) * (d_ff, d_model) -> (seq_len, d_model)

    // Add bias b2 (broadcasting)
    Matrix::Matrix<float> output_biased(output_linear.rows(), output_linear.cols());
     for(size_t r = 0; r < output_linear.rows(); ++r) {
        for(size_t c = 0; c < output_linear.cols(); ++c) {
            output_biased[r][c] = output_linear[r][c] + b2_[0][c];
        }
    }

    return output_biased;
}

// --- Weight and Bias Accessors ---
void TransformerFFN::set_W1(const Matrix::Matrix<float>& w1) {
    if (w1.rows() != static_cast<size_t>(d_model_) || w1.cols() != static_cast<size_t>(d_ff_))
        throw std::invalid_argument("W1 dimensions mismatch.");
    W1_ = w1;
}
void TransformerFFN::set_b1(const Matrix::Matrix<float>& b1) {
    if (b1.rows() != 1 || b1.cols() != static_cast<size_t>(d_ff_))
        throw std::invalid_argument("b1 dimensions mismatch (must be 1x" + std::to_string(d_ff_) + ").");
    b1_ = b1;
}
void TransformerFFN::set_W2(const Matrix::Matrix<float>& w2) {
    if (w2.rows() != static_cast<size_t>(d_ff_) || w2.cols() != static_cast<size_t>(d_model_))
        throw std::invalid_argument("W2 dimensions mismatch.");
    W2_ = w2;
}
void TransformerFFN::set_b2(const Matrix::Matrix<float>& b2) {
    if (b2.rows() != 1 || b2.cols() != static_cast<size_t>(d_model_))
        throw std::invalid_argument("b2 dimensions mismatch (must be 1x" + std::to_string(d_model_) + ").");
    b2_ = b2;
}

} // namespace Transformer
} // namespace NeuroNet
