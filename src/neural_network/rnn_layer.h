#pragma once
#include "../math/matrix.h"

namespace NeuroNet {

class RNNLayer {
public:
    RNNLayer(int input_size, int hidden_size);

    // Forward pass for a single time step
    Matrix::Matrix<float> Forward(const Matrix::Matrix<float>& input);

    // Reset the hidden state (e.g., at the start of a new sequence)
    void ResetState();

    // Getters for weights and biases
    Matrix::Matrix<float>& GetW_xh() { return W_xh; }
    Matrix::Matrix<float>& GetW_hh() { return W_hh; }
    Matrix::Matrix<float>& Getb_h() { return b_h; }
    Matrix::Matrix<float> GetHiddenState() const { return h_prev; }

private:
    int input_size_;
    int hidden_size_;

    Matrix::Matrix<float> W_xh; // Input to hidden weights
    Matrix::Matrix<float> W_hh; // Hidden to hidden weights
    Matrix::Matrix<float> b_h;  // Hidden bias

    Matrix::Matrix<float> h_prev; // Previous hidden state
};

} // namespace NeuroNet
