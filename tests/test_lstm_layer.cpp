#include <gtest/gtest.h>
#include <stdexcept>
#include "../src/neural_network/lstm_layer.h"

using namespace NeuroNet;

TEST(LSTMLayerTest, Initialization) {
    LSTMLayer lstm(10, 20);
    EXPECT_EQ(lstm.GetW_xf().rows(), 10);
    EXPECT_EQ(lstm.GetW_xf().cols(), 20);
    EXPECT_EQ(lstm.GetW_hf().rows(), 20);
    EXPECT_EQ(lstm.GetW_hf().cols(), 20);
    EXPECT_EQ(lstm.Getb_f().rows(), 1);
    EXPECT_EQ(lstm.Getb_f().cols(), 20);
    EXPECT_EQ(lstm.GetHiddenState().rows(), 1);
    EXPECT_EQ(lstm.GetHiddenState().cols(), 20);
    EXPECT_EQ(lstm.GetCellState().rows(), 1);
    EXPECT_EQ(lstm.GetCellState().cols(), 20);
}

TEST(LSTMLayerTest, RejectsInvalidDimensions) {
    EXPECT_THROW(LSTMLayer(0, 20), std::invalid_argument);
    EXPECT_THROW(LSTMLayer(10, 0), std::invalid_argument);
    EXPECT_THROW(LSTMLayer(-1, 20), std::invalid_argument);
    EXPECT_THROW(LSTMLayer(10, -1), std::invalid_argument);
}

TEST(LSTMLayerTest, ForwardPass) {
    LSTMLayer lstm(5, 10);
    Matrix::Matrix<float> input(1, 5);
    input.assign(1.0f);

    Matrix::Matrix<float> output = lstm.Forward(input);
    EXPECT_EQ(output.rows(), 1);
    EXPECT_EQ(output.cols(), 10);
}

TEST(LSTMLayerTest, RejectsInvalidInputShape) {
    LSTMLayer lstm(5, 10);
    Matrix::Matrix<float> wrong_columns(1, 4);
    Matrix::Matrix<float> wrong_rows(2, 5);

    EXPECT_THROW(lstm.Forward(wrong_columns), std::invalid_argument);
    EXPECT_THROW(lstm.Forward(wrong_rows), std::invalid_argument);
}

TEST(LSTMLayerTest, ResetState) {
    LSTMLayer lstm(5, 10);
    lstm.GetW_xf().assign(0.25f);
    lstm.GetW_hf().assign(0.0f);
    lstm.Getb_f().assign(0.0f);

    Matrix::Matrix<float> input(1, 5);
    input.assign(1.0f);

    lstm.Forward(input);

    // Check that state is not all zeros
    bool has_non_zero = false;
    Matrix::Matrix<float> state = lstm.GetHiddenState();
    for(size_t c = 0; c < state.cols(); ++c) {
        if(state[0][c] != 0.0f) {
            has_non_zero = true;
            break;
        }
    }
    // We expect the state to change after forward pass
    EXPECT_TRUE(has_non_zero);

    lstm.ResetState();

    // Check that state is all zeros again
    state = lstm.GetHiddenState();
    for(size_t c = 0; c < state.cols(); ++c) {
        EXPECT_FLOAT_EQ(state[0][c], 0.0f);
    }
}

TEST(LSTMLayerTest, ExactNumericalReferenceGatesAndState) {
    LSTMLayer lstm(1, 1);
    lstm.ResetState();

    lstm.GetW_xf()[0][0] = 0.0f; lstm.GetW_hf()[0][0] = 0.0f; lstm.Getb_f()[0][0] = 0.0f;
    lstm.GetW_xi()[0][0] = 1.0f; lstm.GetW_hi()[0][0] = 0.0f; lstm.Getb_i()[0][0] = 0.0f;
    lstm.GetW_xc()[0][0] = 2.0f; lstm.GetW_hc()[0][0] = 0.0f; lstm.Getb_c()[0][0] = 0.0f;
    lstm.GetW_xo()[0][0] = 0.5f; lstm.GetW_ho()[0][0] = 0.0f; lstm.Getb_o()[0][0] = 0.0f;

    Matrix::Matrix<float> input(1, 1);
    input[0][0] = 1.0f;

    Matrix::Matrix<float> h1 = lstm.Forward(input);

    float f1 = 1.0f / (1.0f + std::exp(0.0f)); // 0.5
    float i1 = 1.0f / (1.0f + std::exp(-1.0f)); // ~0.73105858
    float c_tilde1 = std::tanh(2.0f); // ~0.96402758
    float o1 = 1.0f / (1.0f + std::exp(-0.5f)); // ~0.62245933

    float expected_c1 = f1 * 0.0f + i1 * c_tilde1;
    float expected_h1 = o1 * std::tanh(expected_c1);

    EXPECT_NEAR(lstm.GetCellState()[0][0], expected_c1, 1e-5f);
    EXPECT_NEAR(h1[0][0], expected_h1, 1e-5f);
    EXPECT_NEAR(lstm.GetHiddenState()[0][0], expected_h1, 1e-5f);
}

TEST(LSTMLayerTest, RejectsEmptyAndInvalidMatrixInputs) {
    LSTMLayer lstm(2, 2);

    Matrix::Matrix<float> empty_matrix(0, 0);
    EXPECT_THROW(lstm.Forward(empty_matrix), std::invalid_argument);

    Matrix::Matrix<float> empty_cols(1, 0);
    EXPECT_THROW(lstm.Forward(empty_cols), std::invalid_argument);

    Matrix::Matrix<float> zero_rows(0, 2);
    EXPECT_THROW(lstm.Forward(zero_rows), std::invalid_argument);
}
