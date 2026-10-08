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

#ifdef _OPENMP
#include <omp.h>

TEST(Conv2DLayerTest, SerialAndParallelWorkloadsAgreeAcrossThreadCounts) {
    const int saved_threads = omp_get_max_threads();
    const int saved_dynamic = omp_get_dynamic();
    omp_set_dynamic(0);
    // Tiny, padded/strided, and sufficiently large parallel workloads.
    for (int size : {1, 8, 32}) {
        for (int stride : {1, 2}) {
            SCOPED_TRACE(size);
            SCOPED_TRACE(stride);
            const int channels = size == 32 ? 4 : 1;
            Conv2DLayer layer(channels, channels + 1, 3, stride, 1);
            Matrix::Matrix<float> input(1, channels * size * size);
            for (size_t j = 0; j < input.cols(); ++j)
                input[0][j] = (static_cast<int>(j % 19) - 9) / 16.0f;
            omp_set_num_threads(1);
            auto serial = layer.Forward(input, size, size);
            omp_set_num_threads(4);
            for (int repeat = 0; repeat < 3; ++repeat) {
                auto parallel = layer.Forward(input, size, size);
                EXPECT_EQ(parallel.rows(), serial.rows());
                EXPECT_EQ(parallel.cols(), serial.cols());
                for (size_t j = 0; j < serial.cols(); ++j)
                    EXPECT_FLOAT_EQ(parallel[0][j], serial[0][j]);
            }
        }
    }
    omp_set_num_threads(saved_threads);
    omp_set_dynamic(saved_dynamic);
}
#endif
