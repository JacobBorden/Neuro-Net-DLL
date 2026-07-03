#pragma once
#include "../math/matrix.h"

namespace NeuroNet {

class Conv2DLayer {
public:
    Conv2DLayer(int input_channels, int output_channels, int kernel_size, int stride = 1, int padding = 0);
    Matrix::Matrix<float> Forward(const Matrix::Matrix<float>& input, int input_height, int input_width);
    int GetOutputHeight(int input_height) const;
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
