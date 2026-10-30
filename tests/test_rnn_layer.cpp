#include <gtest/gtest.h>
#include <stdexcept>
#include "../src/neural_network/rnn_layer.h"

using namespace NeuroNet;

TEST(RNNLayerTest, Initialization) {
    RNNLayer rnn(10, 20);
    EXPECT_EQ(rnn.GetW_xh().rows(), 10);
    EXPECT_EQ(rnn.GetW_xh().cols(), 20);
    EXPECT_EQ(rnn.GetW_hh().rows(), 20);
    EXPECT_EQ(rnn.GetW_hh().cols(), 20);
    EXPECT_EQ(rnn.Getb_h().rows(), 1);
    EXPECT_EQ(rnn.Getb_h().cols(), 20);
    EXPECT_EQ(rnn.GetHiddenState().rows(), 1);
    EXPECT_EQ(rnn.GetHiddenState().cols(), 20);
}

TEST(RNNLayerTest, RejectsInvalidDimensions) {
    EXPECT_THROW(RNNLayer(0, 20), std::invalid_argument);
    EXPECT_THROW(RNNLayer(10, 0), std::invalid_argument);
    EXPECT_THROW(RNNLayer(-1, 20), std::invalid_argument);
    EXPECT_THROW(RNNLayer(10, -1), std::invalid_argument);
}

TEST(RNNLayerTest, ForwardPass) {
    RNNLayer rnn(5, 10);
    Matrix::Matrix<float> input(1, 5);
    input.assign(1.0f);

    Matrix::Matrix<float> output = rnn.Forward(input);
    EXPECT_EQ(output.rows(), 1);
    EXPECT_EQ(output.cols(), 10);
}

TEST(RNNLayerTest, RejectsInvalidInputShape) {
    RNNLayer rnn(5, 10);
    Matrix::Matrix<float> wrong_columns(1, 4);
    Matrix::Matrix<float> wrong_rows(2, 5);

    EXPECT_THROW(rnn.Forward(wrong_columns), std::invalid_argument);
    EXPECT_THROW(rnn.Forward(wrong_rows), std::invalid_argument);
}

TEST(RNNLayerTest, ResetState) {
    RNNLayer rnn(5, 10);
    rnn.GetW_xh().assign(0.25f);
    rnn.GetW_hh().assign(0.0f);
    rnn.Getb_h().assign(0.0f);

    Matrix::Matrix<float> input(1, 5);
    input.assign(1.0f);

    rnn.Forward(input);

    // Check that state is not all zeros
    bool has_non_zero = false;
    Matrix::Matrix<float> state = rnn.GetHiddenState();
    for(size_t c = 0; c < state.cols(); ++c) {
        if(state[0][c] != 0.0f) {
            has_non_zero = true;
            break;
        }
    }
    // We expect the state to change after forward pass
    EXPECT_TRUE(has_non_zero);

    rnn.ResetState();

    // Check that state is all zeros again
    state = rnn.GetHiddenState();
    for(size_t c = 0; c < state.cols(); ++c) {
        EXPECT_FLOAT_EQ(state[0][c], 0.0f);
    }
}

TEST(RNNLayerTest, HandCalculatedReferenceValue) {
    RNNLayer rnn(2, 2);
    rnn.ResetState();

    // Set weights and biases
    // W_xh: [[0.5, -0.2], [0.1, 0.4]]
    rnn.GetW_xh()[0][0] = 0.5f;  rnn.GetW_xh()[0][1] = -0.2f;
    rnn.GetW_xh()[1][0] = 0.1f;  rnn.GetW_xh()[1][1] = 0.4f;

    // W_hh: [[0.2, 0.3], [-0.1, 0.5]]
    rnn.GetW_hh()[0][0] = 0.2f;  rnn.GetW_hh()[0][1] = 0.3f;
    rnn.GetW_hh()[1][0] = -0.1f; rnn.GetW_hh()[1][1] = 0.5f;

    // b_h: [0.1, -0.05]
    rnn.Getb_h()[0][0] = 0.1f;   rnn.Getb_h()[0][1] = -0.05f;

    // Step 1: Input x1 = [1.0, 2.0]
    Matrix::Matrix<float> x1(1, 2);
    x1[0][0] = 1.0f; x1[0][1] = 2.0f;

    Matrix::Matrix<float> h1 = rnn.Forward(x1);

    // Expected h1: [tanh(0.8), tanh(0.55)]
    float expected_h1_0 = std::tanh(0.8f);
    float expected_h1_1 = std::tanh(0.55f);

    ASSERT_EQ(h1.rows(), 1);
    ASSERT_EQ(h1.cols(), 2);
    EXPECT_NEAR(h1[0][0], expected_h1_0, 1e-5f);
    EXPECT_NEAR(h1[0][1], expected_h1_1, 1e-5f);

    // Step 2: Input x2 = [-0.5, 1.0]
    Matrix::Matrix<float> x2(1, 2);
    x2[0][0] = -0.5f; x2[0][1] = 1.0f;

    Matrix::Matrix<float> h2 = rnn.Forward(x2);

    float z2_0 = -0.15f + (expected_h1_0 * 0.2f + expected_h1_1 * (-0.1f)) + 0.1f;
    float z2_1 = 0.5f + (expected_h1_0 * 0.3f + expected_h1_1 * 0.5f) - 0.05f;
    float expected_h2_0 = std::tanh(z2_0);
    float expected_h2_1 = std::tanh(z2_1);

    ASSERT_EQ(h2.rows(), 1);
    ASSERT_EQ(h2.cols(), 2);
    EXPECT_NEAR(h2[0][0], expected_h2_0, 1e-5f);
    EXPECT_NEAR(h2[0][1], expected_h2_1, 1e-5f);
}

TEST(RNNLayerTest, InvalidAndEmptyInputShapes) {
    RNNLayer rnn(2, 2);

    Matrix::Matrix<float> empty_0x0(0, 0);
    Matrix::Matrix<float> empty_1x0(1, 0);
    Matrix::Matrix<float> empty_0x2(0, 2);
    Matrix::Matrix<float> wrong_cols(1, 3);

    EXPECT_THROW(rnn.Forward(empty_0x0), std::invalid_argument);
    EXPECT_THROW(rnn.Forward(empty_1x0), std::invalid_argument);
    EXPECT_THROW(rnn.Forward(empty_0x2), std::invalid_argument);
    EXPECT_THROW(rnn.Forward(wrong_cols), std::invalid_argument);
}
