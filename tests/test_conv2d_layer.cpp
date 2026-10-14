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

TEST(Conv2DLayerTest, ForwardPassReferenceValues) {
    // 1 in_channel, 1 out_channel, 2x2 kernel, stride 1, padding 0
    Conv2DLayer layer(1, 1, 2, 1, 0);

    // Set filter weights: [[1, 0], [0, 1]]
    layer.GetFilters()[0][0] = 1.0f;
    layer.GetFilters()[0][1] = 0.0f;
    layer.GetFilters()[0][2] = 0.0f;
    layer.GetFilters()[0][3] = 1.0f;

    // Set bias: [1.0]
    layer.GetBiases()[0][0] = 1.0f;

    // 3x3 input: [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
    Matrix::Matrix<float> input(1, 9);
    for (int i = 0; i < 9; ++i) input[0][i] = static_cast<float>(i + 1);

    Matrix::Matrix<float> output = layer.Forward(input, 3, 3);

    ASSERT_EQ(output.rows(), 1);
    ASSERT_EQ(output.cols(), 4);
    EXPECT_FLOAT_EQ(output[0][0], 7.0f);  // (1*1 + 2*0 + 4*0 + 5*1) + 1.0 = 7.0
    EXPECT_FLOAT_EQ(output[0][1], 9.0f);  // (2*1 + 3*0 + 5*0 + 6*1) + 1.0 = 9.0
    EXPECT_FLOAT_EQ(output[0][2], 13.0f); // (4*1 + 5*0 + 7*0 + 8*1) + 1.0 = 13.0
    EXPECT_FLOAT_EQ(output[0][3], 15.0f); // (5*1 + 6*0 + 8*0 + 9*1) + 1.0 = 15.0
}

TEST(Conv2DLayerTest, InvalidAndEmptyShapes) {
    Conv2DLayer layer(1, 1, 3, 1, 0);

    Matrix::Matrix<float> valid_input(1, 9);
    valid_input.assign(1.0f);

    // Invalid input height or width (<= 0)
    EXPECT_THROW(layer.Forward(valid_input, 0, 3), std::invalid_argument);
    EXPECT_THROW(layer.Forward(valid_input, 3, 0), std::invalid_argument);
    EXPECT_THROW(layer.Forward(valid_input, -1, 3), std::invalid_argument);

    // Empty input matrices (0x0, 1x0, 0x9)
    Matrix::Matrix<float> empty_0x0(0, 0);
    EXPECT_THROW(layer.Forward(empty_0x0, 3, 3), std::invalid_argument);

    Matrix::Matrix<float> empty_1x0(1, 0);
    EXPECT_THROW(layer.Forward(empty_1x0, 3, 3), std::invalid_argument);

    Matrix::Matrix<float> empty_0x9(0, 9);
    EXPECT_THROW(layer.Forward(empty_0x9, 3, 3), std::invalid_argument);

    // Input dimensions too small for kernel (e.g. 2x2 input for 3x3 kernel with 0 padding)
    Matrix::Matrix<float> small_input(1, 4);
    EXPECT_THROW(layer.Forward(small_input, 2, 2), std::invalid_argument);
}
