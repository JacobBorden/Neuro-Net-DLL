#include <gtest/gtest.h>
#include "../src/neural_network/conv2d_layer.h"

using namespace NeuroNet;

TEST(Conv2DLayerTest, Initialization) {
    EXPECT_NO_THROW(Conv2DLayer(1, 1, 3));
    EXPECT_THROW(Conv2DLayer(0, 1, 3), std::invalid_argument);
    EXPECT_THROW(Conv2DLayer(1, 0, 3), std::invalid_argument);
    EXPECT_THROW(Conv2DLayer(1, 1, 0), std::invalid_argument);
}

TEST(Conv2DLayerTest, OutputDimensions) {
    Conv2DLayer layer(1, 1, 3, 1, 0);
    EXPECT_EQ(layer.GetOutputHeight(5), 3);
    EXPECT_EQ(layer.GetOutputWidth(5), 3);

    Conv2DLayer layer_pad(1, 1, 3, 1, 1);
    EXPECT_EQ(layer_pad.GetOutputHeight(5), 5);
    EXPECT_EQ(layer_pad.GetOutputWidth(5), 5);

    Conv2DLayer layer_stride(1, 1, 3, 2, 0);
    EXPECT_EQ(layer_stride.GetOutputHeight(5), 2);
    EXPECT_EQ(layer_stride.GetOutputWidth(5), 2);

    Conv2DLayer layer_large_kernel(1, 1, 5, 4, 0);
    EXPECT_EQ(layer_large_kernel.GetOutputHeight(2), 0);
    EXPECT_EQ(layer_large_kernel.GetOutputWidth(2), 0);
}

TEST(Conv2DLayerTest, ForwardThrowsWhenKernelDoesNotFitInput) {
    Conv2DLayer layer(1, 1, 5, 4, 0);
    Matrix::Matrix<float> input(1, 4);
    input.assign(1.0f);

    EXPECT_THROW(layer.Forward(input, 2, 2), std::invalid_argument);
}

TEST(Conv2DLayerTest, ForwardRequiresSingleFlattenedInputRow) {
    Conv2DLayer layer(1, 1, 3, 1, 0);

    Matrix::Matrix<float> batched_input(2, 9);
    batched_input.assign(1.0f);
    EXPECT_THROW(layer.Forward(batched_input, 3, 3), std::invalid_argument);

    Matrix::Matrix<float> empty_input(0, 9);
    EXPECT_THROW(layer.Forward(empty_input, 3, 3), std::invalid_argument);
}

TEST(Conv2DLayerTest, ForwardPass) {
    Conv2DLayer layer(1, 1, 2, 1, 0); // 1 in_channel, 1 out_channel, 2x2 kernel
    // Create a 3x3 input matrix (1 channel, flattened)
    Matrix::Matrix<float> input(1, 9);
    for (int i = 0; i < 9; ++i) input[0][i] = 1.0f;

    Matrix::Matrix<float> output = layer.Forward(input, 3, 3);
    EXPECT_EQ(output.rows(), 1);
    EXPECT_EQ(output.cols(), 4); // 2x2 output flattened
}

TEST(Conv2DLayerTest, DeterministicReferenceValue) {
    Conv2DLayer layer(1, 1, 2, 1, 0); // 1 in_channel, 1 out_channel, 2x2 kernel

    // Set explicit filter and bias
    // filter: [1.0, 0.0, 0.0, -1.0]
    layer.GetFilters()[0][0] = 1.0f;
    layer.GetFilters()[0][1] = 0.0f;
    layer.GetFilters()[0][2] = 0.0f;
    layer.GetFilters()[0][3] = -1.0f;

    // bias: [0.5]
    layer.GetBiases()[0][0] = 0.5f;

    // 3x3 input: 1,2,3, 4,5,6, 7,8,9
    Matrix::Matrix<float> input(1, 9);
    for (int i = 0; i < 9; ++i) {
        input[0][i] = static_cast<float>(i + 1);
    }

    Matrix::Matrix<float> output = layer.Forward(input, 3, 3);
    ASSERT_EQ(output.rows(), 1);
    ASSERT_EQ(output.cols(), 4);

    // Expected output values: -3.5, -3.5, -3.5, -3.5
    for (int i = 0; i < 4; ++i) {
        EXPECT_NEAR(output[0][i], -3.5f, 1e-5f);
    }
}

TEST(Conv2DLayerTest, InvalidAndEmptyShapeHandling) {
    Conv2DLayer layer(1, 1, 3, 1, 0);

    // GetOutputHeight/Width with non-positive dimensions
    EXPECT_EQ(layer.GetOutputHeight(0), 0);
    EXPECT_EQ(layer.GetOutputWidth(-1), 0);

    // Empty input matrix (0x0)
    Matrix::Matrix<float> empty_input(0, 0);
    EXPECT_THROW(layer.Forward(empty_input, 3, 3), std::invalid_argument);

    // Mismatched flattened input length (expected 9 cols, provided 8)
    Matrix::Matrix<float> wrong_len_input(1, 8);
    EXPECT_THROW(layer.Forward(wrong_len_input, 3, 3), std::invalid_argument);

    // Output dimension <= 0 (input size 2 is smaller than kernel 3 with 0 padding)
    Matrix::Matrix<float> small_input(1, 4);
    EXPECT_THROW(layer.Forward(small_input, 2, 2), std::invalid_argument);
}
