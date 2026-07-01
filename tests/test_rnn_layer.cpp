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
