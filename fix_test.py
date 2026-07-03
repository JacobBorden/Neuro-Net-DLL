import sys

def main():
    with open('tests/test_conv2d_layer.cpp', 'r') as f:
        content = f.read()

    old_test = """    Conv2DLayer layer_stride(1, 1, 3, 2, 0);
    EXPECT_EQ(layer_stride.GetOutputHeight(5), 2);
    EXPECT_EQ(layer_stride.GetOutputWidth(5), 2);
}"""

    new_test = """    Conv2DLayer layer_stride(1, 1, 3, 2, 0);
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
}"""

    content = content.replace(old_test, new_test)

    with open('tests/test_conv2d_layer.cpp', 'w') as f:
        f.write(content)

    print("test_conv2d_layer.cpp fixed")

if __name__ == '__main__':
    main()
