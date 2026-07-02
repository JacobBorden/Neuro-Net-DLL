#pragma once
#include "../math/matrix.h"

namespace NeuroNet {

/**
 * @brief Long Short-Term Memory (LSTM) recurrent layer.
 *
 * LSTMLayer processes one time step at a time. Each forward pass combines the
 * input with the previous hidden state and cell state, updating and storing them
 * for the next call.
 */
class LSTMLayer {
public:
    /**
     * @brief Constructs an LSTM layer.
     * @param input_size Number of input features per time step. Must be positive.
     * @param hidden_size Number of hidden units. Must be positive.
     * @throws std::invalid_argument if either dimension is non-positive.
     */
    LSTMLayer(int input_size, int hidden_size);

    /**
     * @brief Runs the layer for a single time step.
     * @param input A 1 x input_size matrix containing the time-step input.
     * @return The updated 1 x hidden_size hidden state.
     * @throws std::invalid_argument if input dimensions do not match the layer.
     */
    Matrix::Matrix<float> Forward(const Matrix::Matrix<float>& input);

    /**
     * @brief Resets the hidden and cell states to all zeros.
     */
    void ResetState();

    // Getters for states
    Matrix::Matrix<float> GetHiddenState() const { return h_prev; }
    Matrix::Matrix<float> GetCellState() const { return c_prev; }

    // Getters for weights and biases
    Matrix::Matrix<float>& GetW_xf() { return W_xf; }
    Matrix::Matrix<float>& GetW_hf() { return W_hf; }
    Matrix::Matrix<float>& Getb_f() { return b_f; }

    Matrix::Matrix<float>& GetW_xi() { return W_xi; }
    Matrix::Matrix<float>& GetW_hi() { return W_hi; }
    Matrix::Matrix<float>& Getb_i() { return b_i; }

    Matrix::Matrix<float>& GetW_xc() { return W_xc; }
    Matrix::Matrix<float>& GetW_hc() { return W_hc; }
    Matrix::Matrix<float>& Getb_c() { return b_c; }

    Matrix::Matrix<float>& GetW_xo() { return W_xo; }
    Matrix::Matrix<float>& GetW_ho() { return W_ho; }
    Matrix::Matrix<float>& Getb_o() { return b_o; }

private:
    int input_size_;
    int hidden_size_;

    // Forget gate
    Matrix::Matrix<float> W_xf;
    Matrix::Matrix<float> W_hf;
    Matrix::Matrix<float> b_f;

    // Input gate
    Matrix::Matrix<float> W_xi;
    Matrix::Matrix<float> W_hi;
    Matrix::Matrix<float> b_i;

    // Cell gate (candidate)
    Matrix::Matrix<float> W_xc;
    Matrix::Matrix<float> W_hc;
    Matrix::Matrix<float> b_c;

    // Output gate
    Matrix::Matrix<float> W_xo;
    Matrix::Matrix<float> W_ho;
    Matrix::Matrix<float> b_o;

    Matrix::Matrix<float> h_prev; // Previous hidden state
    Matrix::Matrix<float> c_prev; // Previous cell state

    // Helper functions for activations
    float sigmoid(float x);
};

} // namespace NeuroNet
