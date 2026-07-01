#include "rnn_layer.h"
#include <cmath> // For std::tanh
#include <random>

namespace NeuroNet {

RNNLayer::RNNLayer(int input_size, int hidden_size)
    : input_size_(input_size), hidden_size_(hidden_size) {

    // Initialize weights and biases
    W_xh.resize(input_size_, hidden_size_);
    W_xh.Randomize();

    W_hh.resize(hidden_size_, hidden_size_);
    W_hh.Randomize();

    b_h.resize(1, hidden_size_);
    b_h.assign(0.0f); // Initialize bias to 0

    // Initialize hidden state
    h_prev.resize(1, hidden_size_);
    h_prev.assign(0.0f);
}

Matrix::Matrix<float> RNNLayer::Forward(const Matrix::Matrix<float>& input) {
    if (input.cols() != input_size_) {
        throw std::invalid_argument("Input size mismatch in RNNLayer.");
    }

    // h_t = tanh(W_xh * x_t + W_hh * h_{t-1} + b_h)
    Matrix::Matrix<float> h_t = (input * W_xh) + (h_prev * W_hh) + b_h;

    // Apply tanh activation manually (since it's a basic implementation)
    for (size_t r = 0; r < h_t.rows(); ++r) {
        for (size_t c = 0; c < h_t.cols(); ++c) {
            h_t[r][c] = std::tanh(h_t[r][c]);
        }
    }

    h_prev = h_t; // Update hidden state
    return h_t;
}

void RNNLayer::ResetState() {
    h_prev.assign(0.0f);
}

} // namespace NeuroNet
