#pragma once
#include "../math/matrix.h"

namespace NeuroNet {

/**
 * @brief A single-sample 2D convolution layer for flattened image inputs.
 */
class Conv2DLayer {
public:
    /**
     * @brief Creates a convolution layer with square kernels.
     *
     * @param input_channels Number of channels in the flattened input image.
     * @param output_channels Number of convolution filters to apply.
     * @param kernel_size Width and height of each square kernel.
     * @param stride Convolution stride in both spatial dimensions.
     * @param padding Zero-padding applied around the input image.
     */
    Conv2DLayer(int input_channels, int output_channels, int kernel_size, int stride = 1, int padding = 0);

    /**
     * @brief Applies the layer to one flattened image.
     *
     * @param input Matrix with exactly one row containing channel-major flattened image data.
     * @param input_height Height of the unflattened input image.
     * @param input_width Width of the unflattened input image.
     * @return A single-row matrix containing flattened channel-major output feature maps.
     */
    Matrix::Matrix<float> Forward(const Matrix::Matrix<float>& input, int input_height, int input_width);

    /**
     * @brief Computes the output feature-map height for a given input height.
     */
    int GetOutputHeight(int input_height) const;

    /**
     * @brief Computes the output feature-map width for a given input width.
     */
    int GetOutputWidth(int input_width) const;
private:
    int input_channels_;
    int output_channels_;
    int kernel_size_;
    int stride_;
    int padding_;
    Matrix::Matrix<float> filters_;
    Matrix::Matrix<float> biases_;
};

} // namespace NeuroNet
