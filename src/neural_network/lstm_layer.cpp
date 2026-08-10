#include "lstm_layer.h"
#include <cmath> // For std::tanh, std::exp
#include <stdexcept>

namespace NeuroNet {

float LSTMLayer::sigmoid(float x) {
    return 1.0f / (1.0f + std::exp(-x));
}

LSTMLayer::LSTMLayer(int input_size, int hidden_size)
    : input_size_(input_size), hidden_size_(hidden_size) {
    if (input_size_ <= 0 || hidden_size_ <= 0) {
        throw std::invalid_argument("LSTMLayer dimensions must be positive.");
    }

    // Initialize forget gate weights and biases
    W_xf.resize(input_size_, hidden_size_); W_xf.Randomize();
    W_hf.resize(hidden_size_, hidden_size_); W_hf.Randomize();
    b_f.resize(1, hidden_size_); b_f.assign(1.0f); // Common to initialize forget gate bias to 1

    // Initialize input gate weights and biases
    W_xi.resize(input_size_, hidden_size_); W_xi.Randomize();
    W_hi.resize(hidden_size_, hidden_size_); W_hi.Randomize();
    b_i.resize(1, hidden_size_); b_i.assign(0.0f);

    // Initialize cell gate (candidate) weights and biases
    W_xc.resize(input_size_, hidden_size_); W_xc.Randomize();
    W_hc.resize(hidden_size_, hidden_size_); W_hc.Randomize();
    b_c.resize(1, hidden_size_); b_c.assign(0.0f);

    // Initialize output gate weights and biases
    W_xo.resize(input_size_, hidden_size_); W_xo.Randomize();
    W_ho.resize(hidden_size_, hidden_size_); W_ho.Randomize();
    b_o.resize(1, hidden_size_); b_o.assign(0.0f);

    // Initialize states
    h_prev.resize(1, hidden_size_); h_prev.assign(0.0f);
    c_prev.resize(1, hidden_size_); c_prev.assign(0.0f);
}

Matrix::Matrix<float> LSTMLayer::Forward(const Matrix::Matrix<float>& input) {
    if (input.rows() != 1 || input.cols() != static_cast<size_t>(input_size_)) {
        throw std::invalid_argument("Input size mismatch in LSTMLayer.");
    }

    // 1. Forget Gate: f_t = sigmoid(W_xf * x_t + W_hf * h_{t-1} + b_f)
    Matrix::Matrix<float> f_t = (input * W_xf) + (h_prev * W_hf) + b_f;
    for (size_t c = 0; c < f_t.cols(); ++c) {
        f_t[0][c] = sigmoid(f_t[0][c]);
    }

    // 2. Input Gate: i_t = sigmoid(W_xi * x_t + W_hi * h_{t-1} + b_i)
    Matrix::Matrix<float> i_t = (input * W_xi) + (h_prev * W_hi) + b_i;
    for (size_t c = 0; c < i_t.cols(); ++c) {
        i_t[0][c] = sigmoid(i_t[0][c]);
    }

    // 3. Cell Gate (Candidate): c_tilde_t = tanh(W_xc * x_t + W_hc * h_{t-1} + b_c)
    Matrix::Matrix<float> c_tilde_t = (input * W_xc) + (h_prev * W_hc) + b_c;
    for (size_t c = 0; c < c_tilde_t.cols(); ++c) {
        c_tilde_t[0][c] = std::tanh(c_tilde_t[0][c]);
    }

    // 4. Output Gate: o_t = sigmoid(W_xo * x_t + W_ho * h_{t-1} + b_o)
    Matrix::Matrix<float> o_t = (input * W_xo) + (h_prev * W_ho) + b_o;
    for (size_t c = 0; c < o_t.cols(); ++c) {
        o_t[0][c] = sigmoid(o_t[0][c]);
    }

    // 5. Update Cell State: c_t = f_t * c_{t-1} + i_t * c_tilde_t (element-wise multiplication)
    Matrix::Matrix<float> c_t(1, hidden_size_);
    for (size_t c = 0; c < c_t.cols(); ++c) {
        c_t[0][c] = f_t[0][c] * c_prev[0][c] + i_t[0][c] * c_tilde_t[0][c];
    }

    // 6. Update Hidden State: h_t = o_t * tanh(c_t) (element-wise multiplication)
    Matrix::Matrix<float> h_t(1, hidden_size_);
    for (size_t c = 0; c < h_t.cols(); ++c) {
        h_t[0][c] = o_t[0][c] * std::tanh(c_t[0][c]);
    }

    // Save states for next step
    c_prev = c_t;
    h_prev = h_t;

    return h_t;
}

void LSTMLayer::ResetState() {
    h_prev.assign(0.0f);
    c_prev.assign(0.0f);
}

} // namespace NeuroNet
