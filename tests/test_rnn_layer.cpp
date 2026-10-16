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

TEST(RNNLayerTest, ForwardPassHandCalculatedReferenceValues) {
    RNNLayer rnn(2, 2);
    rnn.ResetState();

    // Set known weights W_xh
    rnn.GetW_xh()[0][0] = 0.5f;  rnn.GetW_xh()[0][1] = -0.5f;
    rnn.GetW_xh()[1][0] = 0.2f;  rnn.GetW_xh()[1][1] = 0.8f;

    // Set known weights W_hh
    rnn.GetW_hh()[0][0] = 0.1f;  rnn.GetW_hh()[0][1] = 0.4f;
    rnn.GetW_hh()[1][0] = -0.3f; rnn.GetW_hh()[1][1] = 0.2f;

    // Set known biases b_h
    rnn.Getb_h()[0][0] = 0.1f;   rnn.Getb_h()[0][1] = -0.1f;

    // Time step 1: x1 = [1.0, 0.5]
    Matrix::Matrix<float> x1(1, 2);
    x1[0][0] = 1.0f; x1[0][1] = 0.5f;

    Matrix::Matrix<float> h1 = rnn.Forward(x1);
    EXPECT_NEAR(h1[0][0], std::tanh(0.7f), 1e-5f);
    EXPECT_NEAR(h1[0][1], std::tanh(-0.2f), 1e-5f);

    // Time step 2: x2 = [-0.5, 1.0]
    Matrix::Matrix<float> x2(1, 2);
    x2[0][0] = -0.5f; x2[0][1] = 1.0f;

    Matrix::Matrix<float> h2 = rnn.Forward(x2);
    float expected_z2_0 = -0.05f + (std::tanh(0.7f) * 0.1f + std::tanh(-0.2f) * -0.3f) + 0.1f;
    float expected_z2_1 = 1.05f + (std::tanh(0.7f) * 0.4f + std::tanh(-0.2f) * 0.2f) - 0.1f;
    EXPECT_NEAR(h2[0][0], std::tanh(expected_z2_0), 1e-5f);
    EXPECT_NEAR(h2[0][1], std::tanh(expected_z2_1), 1e-5f);
}

TEST(RNNLayerTest, EmptyAndInvalidShapeHandling) {
    RNNLayer rnn(2, 2);

    Matrix::Matrix<float> empty_0x0(0, 0);
    EXPECT_THROW(rnn.Forward(empty_0x0), std::invalid_argument);

    Matrix::Matrix<float> empty_1x0(1, 0);
    EXPECT_THROW(rnn.Forward(empty_1x0), std::invalid_argument);

    Matrix::Matrix<float> empty_0x2(0, 2);
    EXPECT_THROW(rnn.Forward(empty_0x2), std::invalid_argument);
}
