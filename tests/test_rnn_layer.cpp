#include <gtest/gtest.h>
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

TEST(RNNLayerTest, ForwardPass) {
    RNNLayer rnn(5, 10);
    Matrix::Matrix<float> input(1, 5);
    input.assign(1.0f);

    Matrix::Matrix<float> output = rnn.Forward(input);
    EXPECT_EQ(output.rows(), 1);
    EXPECT_EQ(output.cols(), 10);
}

TEST(RNNLayerTest, ResetState) {
    RNNLayer rnn(5, 10);
    Matrix::Matrix<float> input(1, 5);
    input.assign(1.0f);

    rnn.Forward(input);

    // Check that state is not all zeros
    bool has_non_zero = false;
    for(size_t c = 0; c < rnn.GetHiddenState().cols(); ++c) {
        if(rnn.GetHiddenState()[0][c] != 0.0f) {
            has_non_zero = true;
            break;
        }
    }
    // We expect the state to change after forward pass
    EXPECT_TRUE(has_non_zero);

    rnn.ResetState();

    // Check that state is all zeros again
    for(size_t c = 0; c < rnn.GetHiddenState().cols(); ++c) {
        EXPECT_FLOAT_EQ(rnn.GetHiddenState()[0][c], 0.0f);
    }
}
