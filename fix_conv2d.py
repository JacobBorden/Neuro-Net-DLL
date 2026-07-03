import sys

def main():
    with open('src/neural_network/conv2d_layer.cpp', 'r') as f:
        content = f.read()

    # Fix GetOutputHeight
    old_h = """int Conv2DLayer::GetOutputHeight(int input_height) const {
    return (input_height + 2 * padding_ - kernel_size_) / stride_ + 1;
}"""
    new_h = """int Conv2DLayer::GetOutputHeight(int input_height) const {
    const int effective_height = input_height + 2 * padding_;
    if (input_height <= 0 || effective_height < kernel_size_) {
        return 0;
    }
    return (effective_height - kernel_size_) / stride_ + 1;
}"""
    content = content.replace(old_h, new_h)

    # Fix GetOutputWidth
    old_w = """int Conv2DLayer::GetOutputWidth(int input_width) const {
    return (input_width + 2 * padding_ - kernel_size_) / stride_ + 1;
}"""
    new_w = """int Conv2DLayer::GetOutputWidth(int input_width) const {
    const int effective_width = input_width + 2 * padding_;
    if (input_width <= 0 || effective_width < kernel_size_) {
        return 0;
    }
    return (effective_width - kernel_size_) / stride_ + 1;
}"""
    content = content.replace(old_w, new_w)

    # Fix Forward
    old_f = """    if (static_cast<int>(input.cols()) != input_channels_ * input_height * input_width) {"""
    new_f = """    if (input.rows() != 1 ||
        static_cast<int>(input.cols()) != input_channels_ * input_height * input_width) {"""
    content = content.replace(old_f, new_f)

    with open('src/neural_network/conv2d_layer.cpp', 'w') as f:
        f.write(content)

    print("conv2d_layer.cpp fixed")

if __name__ == '__main__':
    main()
