#include "conv2d_layer.h"
#include <stdexcept>
#include <cmath>

namespace NeuroNet {

Conv2DLayer::Conv2DLayer(int input_channels, int output_channels, int kernel_size, int stride, int padding)
    : input_channels_(input_channels), output_channels_(output_channels), kernel_size_(kernel_size), stride_(stride), padding_(padding) {
    if (input_channels <= 0 || output_channels <= 0 || kernel_size <= 0 || stride <= 0 || padding < 0) {
        throw std::invalid_argument("Invalid arguments for Conv2DLayer");
    }
    filters_.resize(output_channels, input_channels * kernel_size * kernel_size);
    filters_.Randomize();
    biases_.resize(1, output_channels);
    biases_.assign(0.0f);
}

int Conv2DLayer::GetOutputHeight(int input_height) const {
    const int effective_height = input_height + 2 * padding_;
    if (input_height <= 0 || effective_height < kernel_size_) {
        return 0;
    }
    return (effective_height - kernel_size_) / stride_ + 1;
}

int Conv2DLayer::GetOutputWidth(int input_width) const {
    const int effective_width = input_width + 2 * padding_;
    if (input_width <= 0 || effective_width < kernel_size_) {
        return 0;
    }
    return (effective_width - kernel_size_) / stride_ + 1;
}

Matrix::Matrix<float> Conv2DLayer::Forward(const Matrix::Matrix<float>& input, int input_height, int input_width) {
    int out_h = GetOutputHeight(input_height);
    int out_w = GetOutputWidth(input_width);
    if (out_h <= 0 || out_w <= 0) {
        throw std::invalid_argument("Invalid output dimensions in Conv2DLayer");
    }
    if (input.rows() != 1 ||
        static_cast<int>(input.cols()) != input_channels_ * input_height * input_width) {
        throw std::invalid_argument("Invalid input dimensions in Conv2DLayer");
    }

    Matrix::Matrix<float> output(1, output_channels_ * out_h * out_w);
    output.assign(0.0f);

#ifdef _OPENMP
    #pragma omp parallel for collapse(2) schedule(static)
#endif
    for (int oc = 0; oc < output_channels_; ++oc) {
        for (int oh = 0; oh < out_h; ++oh) {
            int ih_base = oh * stride_ - padding_;
            int out_idx_base = oc * (out_h * out_w) + oh * out_w;

            for (int ow = 0; ow < out_w; ++ow) {
                int iw_base = ow * stride_ - padding_;
                float val = biases_[0][oc];

                for (int ic = 0; ic < input_channels_; ++ic) {
                    int in_c_offset = ic * (input_height * input_width);
                    int filter_c_offset = ic * (kernel_size_ * kernel_size_);

                    for (int kh = 0; kh < kernel_size_; ++kh) {
                        int ih = ih_base + kh;
                        if (ih >= 0 && ih < input_height) {
                            int in_h_offset = in_c_offset + ih * input_width;
                            int filter_h_offset = filter_c_offset + kh * kernel_size_;

                            for (int kw = 0; kw < kernel_size_; ++kw) {
                                int iw = iw_base + kw;
                                if (iw >= 0 && iw < input_width) {
                                    val += input[0][in_h_offset + iw] * filters_[oc][filter_h_offset + kw];
                                }
                            }
                        }
                    }
                }
                output[0][out_idx_base + ow] = val;
            }
        }
    }
    return output;
}

} // namespace NeuroNet
