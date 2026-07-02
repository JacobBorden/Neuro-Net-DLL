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
