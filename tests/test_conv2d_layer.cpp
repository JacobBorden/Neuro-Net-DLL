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

TEST(Conv2DLayerTest, ExactNumericalReferenceSingleChannel) {
    // 1 in_channel, 1 out_channel, 2x2 kernel, stride 1, padding 0
    Conv2DLayer layer(1, 1, 2, 1, 0);

    // Filter 2x2: [1, 0, 0, -1]
    layer.GetFilters()[0][0] = 1.0f;
    layer.GetFilters()[0][1] = 0.0f;
    layer.GetFilters()[0][2] = 0.0f;
    layer.GetFilters()[0][3] = -1.0f;
    layer.GetBiases()[0][0] = 0.5f;

    // Input 3x3 flattened
    Matrix::Matrix<float> input(1, 9);
    float input_vals[9] = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f, 9.0f};
    for (int i = 0; i < 9; ++i) input[0][i] = input_vals[i];

    Matrix::Matrix<float> output = layer.Forward(input, 3, 3);
    EXPECT_EQ(output.rows(), 1);
    EXPECT_EQ(output.cols(), 4);

    EXPECT_FLOAT_EQ(output[0][0], -3.5f);
    EXPECT_FLOAT_EQ(output[0][1], -3.5f);
    EXPECT_FLOAT_EQ(output[0][2], -3.5f);
    EXPECT_FLOAT_EQ(output[0][3], -3.5f);
}

TEST(Conv2DLayerTest, ExactNumericalReferenceMultiChannel) {
    // 2 in_channels, 1 out_channel, 2x2 kernel, stride 1, padding 0
    Conv2DLayer layer(2, 1, 2, 1, 0);

    // Filter for Ch0: [1, 1, 1, 1], Ch1: [2, 0, 0, -1]
    float f_vals[8] = {1.0f, 1.0f, 1.0f, 1.0f, 2.0f, 0.0f, 0.0f, -1.0f};
    for (int i = 0; i < 8; ++i) layer.GetFilters()[0][i] = f_vals[i];
    layer.GetBiases()[0][0] = 1.0f;

    // Input: 2 channels of 2x2 image = 8 elements total
    Matrix::Matrix<float> input(1, 8);
    float in_vals[8] = {1.0f, 2.0f, 3.0f, 4.0f, 0.5f, 1.0f, 1.5f, 2.0f};
    for (int i = 0; i < 8; ++i) input[0][i] = in_vals[i];

    Matrix::Matrix<float> output = layer.Forward(input, 2, 2);
    EXPECT_EQ(output.rows(), 1);
    EXPECT_EQ(output.cols(), 1);
    EXPECT_FLOAT_EQ(output[0][0], 10.0f);
}

TEST(Conv2DLayerTest, InvalidAndEmptyShapeErrors) {
    Conv2DLayer layer(1, 1, 3, 1, 0);

    Matrix::Matrix<float> empty_input(0, 0);
    EXPECT_THROW(layer.Forward(empty_input, 3, 3), std::invalid_argument);

    Matrix::Matrix<float> invalid_size_input(1, 8); // Expected 1x9 for 1x3x3
    EXPECT_THROW(layer.Forward(invalid_size_input, 3, 3), std::invalid_argument);

    Matrix::Matrix<float> input(1, 9);
    EXPECT_THROW(layer.Forward(input, 0, 3), std::invalid_argument);
    EXPECT_THROW(layer.Forward(input, 3, 0), std::invalid_argument);
    EXPECT_THROW(layer.Forward(input, -1, 3), std::invalid_argument);
}
