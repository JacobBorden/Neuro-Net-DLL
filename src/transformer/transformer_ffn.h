#pragma once

#include "../math/matrix.h"
#include "../math/extended_matrix_ops.h" // For MathUtils::gelu
#include <stdexcept> // For std::invalid_argument

namespace NeuroNet {
namespace Transformer {

class TransformerFFN {
public:
    /**
     * @brief Constructor for TransformerFFN.
     * Typically consists of two linear layers with a GELU activation in between.
     * Output = GELU(input * W1 + b1) * W2 + b2
     * @param d_model Dimensionality of the input and output.
     * @param d_ff Dimensionality of the inner feed-forward layer (hidden layer).
     * @param dropout_rate Dropout rate (currently unused, placeholder).
     */
    TransformerFFN(int d_model, int d_ff, float dropout_rate = 0.0f);

    /**
     * @brief Initializes the weight and bias matrices.
     * Weights are initialized randomly; biases are initialized to zero.
     */
    void initialize_weights();

    /**
     * @brief Performs the forward pass for the FFN.
     * @param input Input matrix, shape (seq_len, d_model).
     * @return Matrix::Matrix<float> The output matrix, shape (seq_len, d_model).
     * @throws std::invalid_argument if input dimensions are incorrect.
     */
    Matrix::Matrix<float> forward(const Matrix::Matrix<float>& input);

    // --- Weight and Bias Accessors for Serialization/Training ---
    const Matrix::Matrix<float>& get_W1() const { return W1_; }
    const Matrix::Matrix<float>& get_b1() const { return b1_; }
    const Matrix::Matrix<float>& get_W2() const { return W2_; }
    const Matrix::Matrix<float>& get_b2() const { return b2_; }

    void set_W1(const Matrix::Matrix<float>& w1);
    void set_b1(const Matrix::Matrix<float>& b1);
    void set_W2(const Matrix::Matrix<float>& w2);
    void set_b2(const Matrix::Matrix<float>& b2);

    int get_d_model() const { return d_model_; }
    int get_d_ff() const { return d_ff_; }

private:
    int d_model_;
    int d_ff_;

    Matrix::Matrix<float> W1_; // Shape: (d_model, d_ff)
    Matrix::Matrix<float> b1_; // Shape: (1, d_ff) - broadcasted
    Matrix::Matrix<float> W2_; // Shape: (d_ff, d_model)
    Matrix::Matrix<float> b2_; // Shape: (1, d_model) - broadcasted

    float dropout_rate_; // Placeholder
};

} // namespace Transformer
} // namespace NeuroNet
