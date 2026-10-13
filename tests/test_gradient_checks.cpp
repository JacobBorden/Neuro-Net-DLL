#include <gtest/gtest.h>
#include <cmath>
#include <vector>
#include <stdexcept>

#include "../src/neural_network/neuronet.h"
#include "../src/neural_network/conv2d_layer.h"
#include "../src/neural_network/rnn_layer.h"
#include "../src/neural_network/lstm_layer.h"
#include "../src/transformer/attention.h"
#include "../src/math/matrix.h"

using namespace NeuroNet;

// Helper to calculate MSE loss L = 0.5 * sum((actual - target)^2)
static float compute_mse_loss(const Matrix::Matrix<float>& actual, const Matrix::Matrix<float>& target) {
    float loss = 0.0f;
    for (size_t r = 0; r < actual.rows(); ++r) {
        for (size_t c = 0; c < actual.cols(); ++c) {
            float diff = actual[r][c] - target[r][c];
            loss += 0.5f * diff * diff;
        }
    }
    return loss;
}

// -----------------------------------------------------------------------------
// 1. Numerical Gradient Checks for Dense Activations and Backpropagation
// -----------------------------------------------------------------------------

class DenseGradientCheckTest : public ::testing::TestWithParam<ActivationFunctionType> {};

INSTANTIATE_TEST_SUITE_P(
    ActivationFunctions,
    DenseGradientCheckTest,
    ::testing::Values(
        ActivationFunctionType::None,
        ActivationFunctionType::ReLU,
        ActivationFunctionType::LeakyReLU,
        ActivationFunctionType::ELU,
        ActivationFunctionType::Sigmoid,
        ActivationFunctionType::Tanh
    )
);

TEST_P(DenseGradientCheckTest, NumericalGradientCheckSingleLayer) {
    ActivationFunctionType act_func = GetParam();

    NeuroNet::NeuroNet net(1);
    net.SetInputSize(3);
    net.ResizeLayer(0, 2);
    net.getLayer(0).SetActivationFunction(act_func);

    // Set non-zero initial weights and biases
    LayerWeights lw;
    lw.WeightCount = 6;
    lw.WeightsVector = {0.2f, -0.4f, 0.5f, 0.1f, -0.3f, 0.6f};
    net.getLayer(0).SetWeights(lw);

    LayerBiases lb;
    lb.BiasCount = 2;
    lb.BiasVector = {0.1f, -0.2f};
    net.getLayer(0).SetBiases(lb);

    Matrix::Matrix<float> input(1, 3);
    input[0][0] = 0.5f; input[0][1] = -0.8f; input[0][2] = 1.2f;

    Matrix::Matrix<float> target(1, 2);
    target[0][0] = 1.0f; target[0][1] = 0.0f;

    // Run forward pass and backpropagation
    net.SetInput(input);
    Matrix::Matrix<float> actual = net.GetOutput();
    net.Backpropagate(actual, target);

    Matrix::Matrix<float> dLdW_analytical = net.getLayer(0).get_dLdW();
    Matrix::Matrix<float> dLdB_analytical = net.getLayer(0).get_dLdB();

    const float epsilon = 1e-3f;

    // Numerical gradient check for Weights
    for (size_t r = 0; r < dLdW_analytical.rows(); ++r) {
        for (size_t c = 0; c < dLdW_analytical.cols(); ++c) {
            // W + epsilon
            LayerWeights lw_plus = lw;
            size_t idx = r * dLdW_analytical.cols() + c;
            lw_plus.WeightsVector[idx] += epsilon;
            net.getLayer(0).SetWeights(lw_plus);
            net.SetInput(input);
            float loss_plus = compute_mse_loss(net.GetOutput(), target);

            // W - epsilon
            LayerWeights lw_minus = lw;
            lw_minus.WeightsVector[idx] -= epsilon;
            net.getLayer(0).SetWeights(lw_minus);
            net.SetInput(input);
            float loss_minus = compute_mse_loss(net.GetOutput(), target);

            // Restore original weights
            net.getLayer(0).SetWeights(lw);

            float num_grad = (loss_plus - loss_minus) / (2.0f * epsilon);
            float ana_grad = dLdW_analytical[r][c];

            EXPECT_NEAR(num_grad, ana_grad, 1e-2f)
                << "Weight gradient mismatch at (" << r << ", " << c << ")"
                << " for activation " << net.getLayer(0).get_activation_function_name();
        }
    }

    // Numerical gradient check for Biases
    for (size_t c = 0; c < dLdB_analytical.cols(); ++c) {
        // B + epsilon
        LayerBiases lb_plus = lb;
        lb_plus.BiasVector[c] += epsilon;
        net.getLayer(0).SetBiases(lb_plus);
        net.SetInput(input);
        float loss_plus = compute_mse_loss(net.GetOutput(), target);

        // B - epsilon
        LayerBiases lb_minus = lb;
        lb_minus.BiasVector[c] -= epsilon;
        net.getLayer(0).SetBiases(lb_minus);
        net.SetInput(input);
        float loss_minus = compute_mse_loss(net.GetOutput(), target);

        // Restore original biases
        net.getLayer(0).SetBiases(lb);

        float num_grad = (loss_plus - loss_minus) / (2.0f * epsilon);
        float ana_grad = dLdB_analytical[0][c];

        EXPECT_NEAR(num_grad, ana_grad, 1e-2f)
            << "Bias gradient mismatch at column " << c
            << " for activation " << net.getLayer(0).get_activation_function_name();
    }
}

TEST(DenseGradientCheckTest, TwoLayerNetworkGradientCheck) {
    NeuroNet::NeuroNet net(2);
    net.SetInputSize(2);
    net.ResizeLayer(0, 3);
    net.ResizeLayer(1, 2);
    net.getLayer(0).SetActivationFunction(ActivationFunctionType::Sigmoid);
    net.getLayer(1).SetActivationFunction(ActivationFunctionType::Tanh);

    LayerWeights lw0; lw0.WeightCount = 6; lw0.WeightsVector = {0.1f, -0.2f, 0.3f, 0.4f, -0.5f, 0.6f};
    LayerBiases  lb0; lb0.BiasCount = 3;   lb0.BiasVector = {0.05f, -0.1f, 0.15f};
    net.getLayer(0).SetWeights(lw0);
    net.getLayer(0).SetBiases(lb0);

    LayerWeights lw1; lw1.WeightCount = 6; lw1.WeightsVector = {-0.3f, 0.5f, 0.2f, -0.1f, 0.4f, -0.6f};
    LayerBiases  lb1; lb1.BiasCount = 2;   lb1.BiasVector = {-0.05f, 0.1f};
    net.getLayer(1).SetWeights(lw1);
    net.getLayer(1).SetBiases(lb1);

    Matrix::Matrix<float> input(1, 2);
    input[0][0] = 0.6f; input[0][1] = -0.4f;

    Matrix::Matrix<float> target(1, 2);
    target[0][0] = 0.5f; target[0][1] = -0.5f;

    net.SetInput(input);
    Matrix::Matrix<float> actual = net.GetOutput();
    net.Backpropagate(actual, target);

    const float epsilon = 1e-3f;

    // Check layer 0 weights gradient
    Matrix::Matrix<float> dLdW0_ana = net.getLayer(0).get_dLdW();
    for (size_t r = 0; r < dLdW0_ana.rows(); ++r) {
        for (size_t c = 0; c < dLdW0_ana.cols(); ++c) {
            size_t idx = r * dLdW0_ana.cols() + c;

            LayerWeights lw0_p = lw0; lw0_p.WeightsVector[idx] += epsilon;
            net.getLayer(0).SetWeights(lw0_p); net.SetInput(input);
            float loss_p = compute_mse_loss(net.GetOutput(), target);

            LayerWeights lw0_m = lw0; lw0_m.WeightsVector[idx] -= epsilon;
            net.getLayer(0).SetWeights(lw0_m); net.SetInput(input);
            float loss_m = compute_mse_loss(net.GetOutput(), target);

            net.getLayer(0).SetWeights(lw0);

            float num_grad = (loss_p - loss_m) / (2.0f * epsilon);
            EXPECT_NEAR(num_grad, dLdW0_ana[r][c], 1e-2f);
        }
    }
}

// -----------------------------------------------------------------------------
// 2. Reference Value Tests & Shape Validation for CNN (Conv2DLayer)
// -----------------------------------------------------------------------------

TEST(Conv2DReferenceTest, ForwardExactReferenceValue) {
    // 1 input channel, 1 output channel, 2x2 kernel, stride 1, padding 0
    Conv2DLayer conv(1, 1, 2, 1, 0);

    // Set filter weights (1 filter of size 1x2x2) and bias
    Matrix::Matrix<float> filter_weights(1, 4);
    filter_weights[0][0] = 1.0f; filter_weights[0][1] = 2.0f;
    filter_weights[0][2] = 3.0f; filter_weights[0][3] = 4.0f;
    conv.GetFilters() = filter_weights;

    Matrix::Matrix<float> bias(1, 1);
    bias[0][0] = 0.5f;
    conv.GetBiases() = bias;

    // 3x3 input image:
    // [ 1, 2, 3 ]
    // [ 4, 5, 6 ]
    // [ 7, 8, 9 ]
    Matrix::Matrix<float> input(1, 9);
    for (int i = 0; i < 9; ++i) {
        input[0][i] = static_cast<float>(i + 1);
    }

    Matrix::Matrix<float> output = conv.Forward(input, 3, 3);

    // Output dimension: (3 - 2)/1 + 1 = 2x2 -> 4 elements
    ASSERT_EQ(output.rows(), 1);
    ASSERT_EQ(output.cols(), 4);

    // Expected window 0: top-left [1, 2; 4, 5] -> 1*1 + 2*2 + 4*3 + 5*4 + 0.5 = 1 + 4 + 12 + 20 + 0.5 = 37.5
    // Expected window 1: top-right [2, 3; 5, 6] -> 2*1 + 3*2 + 5*3 + 6*4 + 0.5 = 2 + 6 + 15 + 24 + 0.5 = 47.5
    // Expected window 2: bottom-left [4, 5; 7, 8] -> 4*1 + 5*2 + 7*3 + 8*4 + 0.5 = 4 + 10 + 21 + 32 + 0.5 = 67.5
    // Expected window 3: bottom-right [5, 6; 8, 9] -> 5*1 + 6*2 + 8*3 + 9*4 + 0.5 = 5 + 12 + 24 + 36 + 0.5 = 77.5
    EXPECT_FLOAT_EQ(output[0][0], 37.5f);
    EXPECT_FLOAT_EQ(output[0][1], 47.5f);
    EXPECT_FLOAT_EQ(output[0][2], 67.5f);
    EXPECT_FLOAT_EQ(output[0][3], 77.5f);
}

TEST(Conv2DReferenceTest, InvalidAndEmptyShapes) {
    Conv2DLayer conv(1, 1, 3, 1, 0);

    // Mismatched image dimensions (e.g., input has 8 elements but 3x3 expects 9)
    Matrix::Matrix<float> invalid_input(1, 8);
    EXPECT_THROW(conv.Forward(invalid_input, 3, 3), std::invalid_argument);

    // Multi-row input (batching not supported by Conv2DLayer::Forward)
    Matrix::Matrix<float> multi_row_input(2, 9);
    EXPECT_THROW(conv.Forward(multi_row_input, 3, 3), std::invalid_argument);

    // Zero dimension height/width
    Matrix::Matrix<float> valid_size_input(1, 9);
    EXPECT_THROW(conv.Forward(valid_size_input, 0, 3), std::invalid_argument);
    EXPECT_THROW(conv.Forward(valid_size_input, 3, 0), std::invalid_argument);
}

// -----------------------------------------------------------------------------
// 3. Reference Value Tests & Shape Validation for RNN (RNNLayer)
// -----------------------------------------------------------------------------

TEST(RNNReferenceTest, ForwardExactReferenceValue) {
    RNNLayer rnn(2, 2);

    // Set known weights and bias
    rnn.GetW_xh()[0][0] = 0.5f; rnn.GetW_xh()[0][1] = -0.2f;
    rnn.GetW_xh()[1][0] = 0.1f; rnn.GetW_xh()[1][1] = 0.8f;

    rnn.GetW_hh()[0][0] = 0.3f; rnn.GetW_hh()[0][1] = 0.1f;
    rnn.GetW_hh()[1][0] = -0.4f; rnn.GetW_hh()[1][1] = 0.2f;

    rnn.Getb_h()[0][0] = 0.1f; rnn.Getb_h()[0][1] = -0.1f;

    // Time step 1: h_prev = [0, 0]
    Matrix::Matrix<float> x1(1, 2);
    x1[0][0] = 1.0f; x1[0][1] = 0.5f;

    Matrix::Matrix<float> h1 = rnn.Forward(x1);

    // z1 = x1 * W_xh + h_prev * W_hh + b_h
    // z1[0] = 1.0*0.5 + 0.5*0.1 + 0 + 0.1 = 0.65
    // z1[1] = 1.0*(-0.2) + 0.5*0.8 + 0 - 0.1 = 0.10
    // h1[0] = tanh(0.65), h1[1] = tanh(0.10)
    EXPECT_NEAR(h1[0][0], std::tanh(0.65f), 1e-5f);
    EXPECT_NEAR(h1[0][1], std::tanh(0.10f), 1e-5f);

    // Time step 2: h_prev = h1
    Matrix::Matrix<float> x2(1, 2);
    x2[0][0] = -0.5f; x2[0][1] = 1.0f;

    Matrix::Matrix<float> h2 = rnn.Forward(x2);

    float h1_0 = std::tanh(0.65f);
    float h1_1 = std::tanh(0.10f);

    // z2[0] = (-0.5)*0.5 + 1.0*0.1 + h1_0*0.3 + h1_1*(-0.4) + 0.1
    // z2[1] = (-0.5)*(-0.2) + 1.0*0.8 + h1_0*0.1 + h1_1*0.2 - 0.1
    float z2_0 = -0.25f + 0.1f + h1_0 * 0.3f - h1_1 * 0.4f + 0.1f;
    float z2_1 = 0.1f + 0.8f + h1_0 * 0.1f + h1_1 * 0.2f - 0.1f;

    EXPECT_NEAR(h2[0][0], std::tanh(z2_0), 1e-5f);
    EXPECT_NEAR(h2[0][1], std::tanh(z2_1), 1e-5f);
}

TEST(RNNReferenceTest, InvalidAndEmptyShapes) {
    RNNLayer rnn(3, 4);

    // Invalid columns
    Matrix::Matrix<float> invalid_cols(1, 2);
    EXPECT_THROW(rnn.Forward(invalid_cols), std::invalid_argument);

    // Invalid rows (must be 1 row)
    Matrix::Matrix<float> invalid_rows(2, 3);
    EXPECT_THROW(rnn.Forward(invalid_rows), std::invalid_argument);

    // Empty matrix
    Matrix::Matrix<float> empty_mat(0, 0);
    EXPECT_THROW(rnn.Forward(empty_mat), std::invalid_argument);
}

// -----------------------------------------------------------------------------
// 4. Reference Value Tests & Shape Validation for LSTM (LSTMLayer)
// -----------------------------------------------------------------------------

TEST(LSTMReferenceTest, ForwardExactReferenceValue) {
    LSTMLayer lstm(1, 1);

    // Forget gate
    lstm.GetW_xf()[0][0] = 0.0f; lstm.GetW_hf()[0][0] = 0.0f; lstm.Getb_f()[0][0] = 0.0f; // sigmoid(0) = 0.5
    // Input gate
    lstm.GetW_xi()[0][0] = 0.0f; lstm.GetW_hi()[0][0] = 0.0f; lstm.Getb_i()[0][0] = 0.0f; // sigmoid(0) = 0.5
    // Cell candidate
    lstm.GetW_xc()[0][0] = 1.0f; lstm.GetW_hc()[0][0] = 0.0f; lstm.Getb_c()[0][0] = 0.0f; // tanh(1*x)
    // Output gate
    lstm.GetW_xo()[0][0] = 0.0f; lstm.GetW_ho()[0][0] = 0.0f; lstm.Getb_o()[0][0] = 0.0f; // sigmoid(0) = 0.5

    Matrix::Matrix<float> input(1, 1);
    input[0][0] = 0.5f;

    Matrix::Matrix<float> h1 = lstm.Forward(input);

    // f_t = sigmoid(0) = 0.5
    // i_t = sigmoid(0) = 0.5
    // c_tilde = tanh(0.5)
    // c_t = f_t * c_prev + i_t * c_tilde = 0 + 0.5 * tanh(0.5)
    // o_t = sigmoid(0) = 0.5
    // h_t = o_t * tanh(c_t) = 0.5 * tanh(0.5 * tanh(0.5))

    float expected_c = 0.5f * std::tanh(0.5f);
    float expected_h = 0.5f * std::tanh(expected_c);

    EXPECT_NEAR(lstm.GetCellState()[0][0], expected_c, 1e-5f);
    EXPECT_NEAR(h1[0][0], expected_h, 1e-5f);
}

TEST(LSTMReferenceTest, InvalidAndEmptyShapes) {
    LSTMLayer lstm(2, 3);

    Matrix::Matrix<float> invalid_cols(1, 1);
    EXPECT_THROW(lstm.Forward(invalid_cols), std::invalid_argument);

    Matrix::Matrix<float> invalid_rows(3, 2);
    EXPECT_THROW(lstm.Forward(invalid_rows), std::invalid_argument);

    Matrix::Matrix<float> empty_mat(0, 0);
    EXPECT_THROW(lstm.Forward(empty_mat), std::invalid_argument);
}

// -----------------------------------------------------------------------------
// 5. Reference Value Tests & Shape Validation for Transformer Attention & Masks
// -----------------------------------------------------------------------------

TEST(AttentionReferenceTest, ScaledDotProductAttentionNoMaskReference) {
    Transformer::ScaledDotProductAttention attn;

    // Q: (1, 2), K: (2, 2), V: (2, 2)
    Matrix::Matrix<float> Q(1, 2); Q[0][0] = 1.0f; Q[0][1] = 0.0f;
    Matrix::Matrix<float> K(2, 2); K[0][0] = 1.0f; K[0][1] = 0.0f;
                                  K[1][0] = 0.0f; K[1][1] = 1.0f;
    Matrix::Matrix<float> V(2, 2); V[0][0] = 10.0f; V[0][1] = 20.0f;
                                  V[1][0] = 30.0f; V[1][1] = 40.0f;

    auto result = attn.forward(Q, K, V);

    // Q * K^T = [1, 0] * [[1, 0], [0, 1]] = [1, 0]
    // d_k = 2 -> sqrt(2) = 1.4142135
    // scores = [1 / sqrt(2), 0 / sqrt(2)] = [0.707107, 0]
    // softmax([0.707107, 0]):
    // exp(0.707107) = 2.028115, exp(0) = 1.0
    // sum = 3.028115
    // attn_weights = [0.66978, 0.33022]
    // output = 0.66978 * [10, 20] + 0.33022 * [30, 40]
    // output[0] = 6.6978 + 9.9066 = 16.6044
    // output[1] = 13.3956 + 13.2088 = 26.6044

    ASSERT_EQ(result.output.rows(), 1);
    ASSERT_EQ(result.output.cols(), 2);

    float exp0 = std::exp(1.0f / std::sqrt(2.0f));
    float exp1 = std::exp(0.0f);
    float sum_exp = exp0 + exp1;
    float w0 = exp0 / sum_exp;
    float w1 = exp1 / sum_exp;

    EXPECT_NEAR(result.attention_weights[0][0], w0, 1e-4f);
    EXPECT_NEAR(result.attention_weights[0][1], w1, 1e-4f);

    EXPECT_NEAR(result.output[0][0], w0 * 10.0f + w1 * 30.0f, 1e-4f);
    EXPECT_NEAR(result.output[0][1], w0 * 20.0f + w1 * 40.0f, 1e-4f);
}

TEST(AttentionReferenceTest, ScaledDotProductAttentionWithMaskReference) {
    Transformer::ScaledDotProductAttention attn;

    Matrix::Matrix<float> Q(1, 2); Q[0][0] = 1.0f; Q[0][1] = 0.0f;
    Matrix::Matrix<float> K(2, 2); K[0][0] = 1.0f; K[0][1] = 0.0f;
                                  K[1][0] = 0.0f; K[1][1] = 1.0f;
    Matrix::Matrix<float> V(2, 2); V[0][0] = 10.0f; V[0][1] = 20.0f;
                                  V[1][0] = 30.0f; V[1][1] = 40.0f;

    // Mask second key with large negative value (-1e9f)
    Matrix::Matrix<float> mask(1, 2);
    mask[0][0] = 0.0f;
    mask[0][1] = -1e9f;

    auto result = attn.forward(Q, K, V, mask);

    // After mask, weight for key 1 is ~1.0 and key 2 is ~0.0
    EXPECT_NEAR(result.attention_weights[0][0], 1.0f, 1e-4f);
    EXPECT_NEAR(result.attention_weights[0][1], 0.0f, 1e-4f);

    // Output should match value 0: [10, 20]
    EXPECT_NEAR(result.output[0][0], 10.0f, 1e-4f);
    EXPECT_NEAR(result.output[0][1], 20.0f, 1e-4f);
}

TEST(AttentionReferenceTest, InvalidAndEmptyShapes) {
    Transformer::ScaledDotProductAttention attn;

    Matrix::Matrix<float> Q(1, 2);
    Matrix::Matrix<float> K(2, 3); // Mismatched d_k dimension (Q has 2, K has 3)
    Matrix::Matrix<float> V(2, 2);

    EXPECT_THROW(attn.forward(Q, K, V), std::invalid_argument);

    // Empty Q, K, V
    Matrix::Matrix<float> empty_Q(0, 0);
    Matrix::Matrix<float> empty_K(0, 0);
    Matrix::Matrix<float> empty_V(0, 0);
    EXPECT_THROW(attn.forward(empty_Q, empty_K, empty_V), std::invalid_argument);

    // Mismatched mask dimensions (mask should be 1x2, give 1x3)
    Matrix::Matrix<float> valid_Q(1, 2);
    Matrix::Matrix<float> valid_K(2, 2);
    Matrix::Matrix<float> valid_V(2, 2);
    Matrix::Matrix<float> bad_mask(1, 3);
    EXPECT_THROW(attn.forward(valid_Q, valid_K, valid_V, bad_mask), std::invalid_argument);
}
