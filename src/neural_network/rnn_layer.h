#pragma once
#include "../math/matrix.h"

namespace NeuroNet {

/**
 * @brief Basic recurrent neural-network layer with tanh hidden-state updates.
 *
 * RNNLayer processes one time step at a time. Each forward pass combines the
 * input with the previous hidden state and stores the resulting hidden state
 * for the next call.
 */
class RNNLayer {
public:
    /**
     * @brief Constructs a recurrent layer.
     * @param input_size Number of input features per time step. Must be positive.
     * @param hidden_size Number of hidden units. Must be positive.
     * @throws std::invalid_argument if either dimension is non-positive.
     */
    RNNLayer(int input_size, int hidden_size);

    /**
     * @brief Runs the layer for a single time step.
     * @param input A 1 x input_size matrix containing the time-step input.
     * @return The updated 1 x hidden_size hidden state.
     * @throws std::invalid_argument if input dimensions do not match the layer.
     */
    Matrix::Matrix<float> Forward(const Matrix::Matrix<float>& input);

    /**
     * @brief Resets the hidden state to all zeros.
     */
    void ResetState();

    /** @brief Returns the input-to-hidden weights. */
    Matrix::Matrix<float>& GetW_xh() { return W_xh; }

    /** @brief Returns the hidden-to-hidden weights. */
    Matrix::Matrix<float>& GetW_hh() { return W_hh; }

    /** @brief Returns the hidden bias vector. */
    Matrix::Matrix<float>& Getb_h() { return b_h; }

    /** @brief Returns a copy of the current hidden state. */
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
