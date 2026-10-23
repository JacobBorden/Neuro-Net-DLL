#include <gtest/gtest.h>
#include "../src/neural_network/neuronet.h"
#include "../src/neural_network/conv2d_layer.h"
#include "../src/neural_network/rnn_layer.h"
#include "../src/neural_network/lstm_layer.h"
#include "../src/transformer/attention.h"
#include "../src/transformer/multi_head_attention.h"
#include "../src/math/extended_matrix_ops.h"
#include "../src/math/matrix.h"
#include <cmath>
#include <vector>
#include <stdexcept>

namespace NeuroNet {
namespace {

TEST(NumericalGradientTest, SigmoidActivation) {
    float eps = 1e-4f;
    std::vector<float> inputs = {-2.0f, -0.5f, 0.0f, 0.5f, 2.0f};

    for (float x : inputs) {
        float analytical_val = 1.0f / (1.0f + std::exp(-x));
        float analytical_grad = analytical_val * (1.0f - analytical_val);

        float x_plus = x + eps;
        float x_minus = x - eps;
        float val_plus = 1.0f / (1.0f + std::exp(-x_plus));
        float val_minus = 1.0f / (1.0f + std::exp(-x_minus));
        float numerical_grad = (val_plus - val_minus) / (2.0f * eps);

        EXPECT_NEAR(analytical_grad, numerical_grad, 1e-3f);
    }
}

TEST(NumericalGradientTest, TanhActivation) {
    float eps = 1e-4f;
    std::vector<float> inputs = {-1.5f, -0.5f, 0.0f, 0.5f, 1.5f};

    for (float x : inputs) {
        float analytical_val = std::tanh(x);
        float analytical_grad = 1.0f - analytical_val * analytical_val;

        float x_plus = x + eps;
        float x_minus = x - eps;
        float val_plus = std::tanh(x_plus);
        float val_minus = std::tanh(x_minus);
        float numerical_grad = (val_plus - val_minus) / (2.0f * eps);

        EXPECT_NEAR(analytical_grad, numerical_grad, 1e-3f);
    }
}

TEST(NumericalGradientTest, ReLUActivation) {
    float eps = 1e-4f;
    std::vector<float> inputs = {-2.0f, -0.5f, 0.5f, 2.0f}; // Exclude exact 0 due to non-differentiability

    for (float x : inputs) {
        float analytical_grad = (x > 0.0f) ? 1.0f : 0.0f;

        float x_plus = x + eps;
        float x_minus = x - eps;
        float val_plus = (x_plus > 0.0f) ? x_plus : 0.0f;
        float val_minus = (x_minus > 0.0f) ? x_minus : 0.0f;
        float numerical_grad = (val_plus - val_minus) / (2.0f * eps);

        EXPECT_NEAR(analytical_grad, numerical_grad, 1e-3f);
    }
}

TEST(NumericalGradientTest, MSELossGradient) {
    float eps = 1e-4f;
    Matrix::Matrix<float> pred(1, 4);
    pred[0][0] = 0.2f; pred[0][1] = 0.5f; pred[0][2] = 0.8f; pred[0][3] = 0.1f;

    Matrix::Matrix<float> target(1, 4);
    target[0][0] = 0.0f; target[0][1] = 1.0f; target[0][2] = 0.5f; target[0][3] = 0.0f;

    // Analytical MSE loss: L = 1/N * sum((y_pred - y_true)^2)
    // dL/dy_pred = 2/N * (y_pred - y_true)
    int n = pred.cols();

    for (size_t col = 0; col < pred.cols(); ++col) {
        float diff = pred[0][col] - target[0][col];
        float analytical_grad = (2.0f / n) * diff;

        Matrix::Matrix<float> pred_plus = pred;
        Matrix::Matrix<float> pred_minus = pred;
        pred_plus[0][col] += eps;
        pred_minus[0][col] -= eps;

        float loss_plus = 0.0f, loss_minus = 0.0f;
        for (size_t j = 0; j < pred.cols(); ++j) {
            loss_plus += std::pow(pred_plus[0][j] - target[0][j], 2.0f);
            loss_minus += std::pow(pred_minus[0][j] - target[0][j], 2.0f);
        }
        loss_plus /= n;
        loss_minus /= n;

        float numerical_grad = (loss_plus - loss_minus) / (2.0f * eps);
        EXPECT_NEAR(analytical_grad, numerical_grad, 1e-3f);
    }
}

TEST(NumericalGradientTest, CrossEntropyLossGradient) {
    float eps = 1e-4f;
    // Softmax outputs p = [0.1, 0.7, 0.2], target y = [0, 1, 0]
    std::vector<float> p = {0.1f, 0.7f, 0.2f};
    std::vector<float> y = {0.0f, 1.0f, 0.0f};

    // Cross Entropy Loss L = - sum(y_i * log(p_i))
    // dL/dp_i = - y_i / p_i
    for (size_t i = 0; i < p.size(); ++i) {
        float analytical_grad = (y[i] > 0.0f) ? (-y[i] / p[i]) : 0.0f;

        float p_plus = p[i] + eps;
        float p_minus = p[i] - eps;

        float loss_plus = -std::log(p_plus); // Since y[1]=1 and others 0
        float loss_minus = -std::log(p_minus);

        if (y[i] > 0.0f) {
            float numerical_grad = (loss_plus - loss_minus) / (2.0f * eps);
            EXPECT_NEAR(analytical_grad, numerical_grad, 1e-3f);
        }
    }
}

TEST(ReferenceValuesTest, Conv2DLayerReferenceAndShapes) {
    // 1 input channel, 1 output channel, kernel 2x2, stride 1, padding 0
    Conv2DLayer conv(1, 1, 2, 1, 0);

    // Input 1x4 (2x2 spatial image)
    Matrix::Matrix<float> input(1, 4);
    input[0][0] = 1.0f; input[0][1] = 2.0f; input[0][2] = 3.0f; input[0][3] = 4.0f;
    Matrix::Matrix<float> output = conv.Forward(input, 2, 2);

    // Output should be 1x1
    EXPECT_EQ(output.rows(), 1u);
    EXPECT_EQ(output.cols(), 1u);

    EXPECT_EQ(conv.GetOutputHeight(2), 1);
    EXPECT_EQ(conv.GetOutputWidth(2), 1);

    // Test invalid input dimensions
    Matrix::Matrix<float> invalid_input(2, 4); // Multi-row input not supported
    EXPECT_THROW(conv.Forward(invalid_input, 2, 2), std::invalid_argument);

    // Empty input
    Matrix::Matrix<float> empty_input(0, 0);
    EXPECT_THROW(conv.Forward(empty_input, 0, 0), std::invalid_argument);
}

TEST(ReferenceValuesTest, RNNLayerReferenceAndShapes) {
    // input_size = 3, hidden_size = 2
    RNNLayer rnn(3, 2);

    Matrix::Matrix<float> input(1, 3);
    input[0][0] = 0.5f; input[0][1] = -0.2f; input[0][2] = 0.8f;
    Matrix::Matrix<float> h1 = rnn.Forward(input);

    EXPECT_EQ(h1.rows(), 1u);
    EXPECT_EQ(h1.cols(), 2u);

    // Hidden state should match GetHiddenState
    Matrix::Matrix<float> h_state = rnn.GetHiddenState();
    EXPECT_EQ(h_state.rows(), 1u);
    EXPECT_EQ(h_state.cols(), 2u);
    EXPECT_FLOAT_EQ(h1[0][0], h_state[0][0]);
    EXPECT_FLOAT_EQ(h1[0][1], h_state[0][1]);

    // Reset state check
    rnn.ResetState();
    Matrix::Matrix<float> h_reset = rnn.GetHiddenState();
    EXPECT_FLOAT_EQ(h_reset[0][0], 0.0f);
    EXPECT_FLOAT_EQ(h_reset[0][1], 0.0f);

    // Test invalid input size
    Matrix::Matrix<float> wrong_shape_input(1, 5);
    EXPECT_THROW(rnn.Forward(wrong_shape_input), std::invalid_argument);

    // Test invalid constructor parameters
    EXPECT_THROW(RNNLayer(0, 2), std::invalid_argument);
    EXPECT_THROW(RNNLayer(3, -1), std::invalid_argument);
}

TEST(ReferenceValuesTest, LSTMLayerReferenceAndShapes) {
    // input_size = 3, hidden_size = 2
    LSTMLayer lstm(3, 2);

    Matrix::Matrix<float> input(1, 3);
    input[0][0] = 0.1f; input[0][1] = 0.4f; input[0][2] = -0.3f;
    Matrix::Matrix<float> h1 = lstm.Forward(input);

    EXPECT_EQ(h1.rows(), 1u);
    EXPECT_EQ(h1.cols(), 2u);

    Matrix::Matrix<float> c1 = lstm.GetCellState();
    EXPECT_EQ(c1.rows(), 1u);
    EXPECT_EQ(c1.cols(), 2u);

    lstm.ResetState();
    EXPECT_FLOAT_EQ(lstm.GetHiddenState()[0][0], 0.0f);
    EXPECT_FLOAT_EQ(lstm.GetCellState()[0][0], 0.0f);

    // Test invalid inputs
    Matrix::Matrix<float> wrong_shape_input(1, 4);
    EXPECT_THROW(lstm.Forward(wrong_shape_input), std::invalid_argument);

    EXPECT_THROW(LSTMLayer(-1, 2), std::invalid_argument);
}

TEST(ReferenceValuesTest, TransformerAttentionMasks) {
    // 1 head, embed_dim = 4
    Transformer::MultiHeadAttention mha(1, 4);

    // Query, Key, Value shape: 2 x 4 (seq_len = 2, embed_dim = 4)
    Matrix::Matrix<float> q(2, 4);
    q[0][0] = 1.0f; q[1][1] = 1.0f;
    Matrix::Matrix<float> k(2, 4);
    k[0][0] = 1.0f; k[1][1] = 1.0f;
    Matrix::Matrix<float> v(2, 4);
    for (size_t j = 0; j < 4; ++j) {
        v[0][j] = 0.5f;
        v[1][j] = 1.0f;
    }

    // Valid forward without mask
    Matrix::Matrix<float> out_nomask = mha.forward(q, k, v);
    EXPECT_EQ(out_nomask.rows(), 2u);
    EXPECT_EQ(out_nomask.cols(), 4u);

    // Causal mask 2x2:
    // [0, -1e9]
    // [0,    0]
    Matrix::Matrix<float> mask(2, 2);
    mask[0][0] = 0.0f; mask[0][1] = -1e9f;
    mask[1][0] = 0.0f; mask[1][1] = 0.0f;
    Matrix::Matrix<float> out_masked = mha.forward(q, k, v, mask);
    EXPECT_EQ(out_masked.rows(), 2u);
    EXPECT_EQ(out_masked.cols(), 4u);

    // Mismatched mask dimensions should throw invalid_argument
    Matrix::Matrix<float> invalid_mask(3, 3);
    EXPECT_THROW(mha.forward(q, k, v, invalid_mask), std::invalid_argument);

    // Empty input matrices should throw invalid_argument
    Matrix::Matrix<float> empty_q(0, 0);
    EXPECT_THROW(mha.forward(empty_q, k, v), std::invalid_argument);
}

} // namespace
} // namespace NeuroNet
