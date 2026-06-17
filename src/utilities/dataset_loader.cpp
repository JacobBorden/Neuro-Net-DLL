#include "dataset_loader.h"
#include <fstream>
#include <iostream>

namespace Utilities {
namespace Dataset {

uint32_t MNISTLoader::SwapEndian(uint32_t val) {
    return ((val << 24) & 0xff000000) |
           ((val <<  8) & 0x00ff0000) |
           ((val >>  8) & 0x0000ff00) |
           ((val >> 24) & 0x000000ff);
}

bool MNISTLoader::LoadImages(const std::string& filepath, Matrix::Matrix<float>& outImages) {
    std::ifstream file(filepath, std::ios::binary);
    if (!file.is_open()) return false;

    uint32_t magic_number = 0;
    uint32_t number_of_images = 0;
    uint32_t n_rows = 0;
    uint32_t n_cols = 0;

    file.read(reinterpret_cast<char*>(&magic_number), sizeof(magic_number));
    magic_number = SwapEndian(magic_number);
    if (magic_number != 2051) return false;

    file.read(reinterpret_cast<char*>(&number_of_images), sizeof(number_of_images));
    number_of_images = SwapEndian(number_of_images);

    file.read(reinterpret_cast<char*>(&n_rows), sizeof(n_rows));
    n_rows = SwapEndian(n_rows);

    file.read(reinterpret_cast<char*>(&n_cols), sizeof(n_cols));
    n_cols = SwapEndian(n_cols);

    outImages.resize(number_of_images, n_rows * n_cols);

    for (size_t i = 0; i < number_of_images; ++i) {
        for (size_t r = 0; r < n_rows * n_cols; ++r) {
            unsigned char temp = 0;
            file.read(reinterpret_cast<char*>(&temp), sizeof(temp));
            outImages[i][r] = static_cast<float>(temp) / 255.0f;
        }
    }

    return true;
}

bool MNISTLoader::LoadLabels(const std::string& filepath, Matrix::Matrix<float>& outLabels) {
    std::ifstream file(filepath, std::ios::binary);
    if (!file.is_open()) return false;

    uint32_t magic_number = 0;
    uint32_t number_of_items = 0;

    file.read(reinterpret_cast<char*>(&magic_number), sizeof(magic_number));
    magic_number = SwapEndian(magic_number);
    if (magic_number != 2049) return false;

    file.read(reinterpret_cast<char*>(&number_of_items), sizeof(number_of_items));
    number_of_items = SwapEndian(number_of_items);

    outLabels.resize(number_of_items, 10);
    outLabels.assign(number_of_items, 10, 0.0f);

    for (size_t i = 0; i < number_of_items; ++i) {
        unsigned char temp = 0;
        file.read(reinterpret_cast<char*>(&temp), sizeof(temp));
        if (temp < 10) {
            outLabels[i][temp] = 1.0f;
        }
    }

    return true;
}

} // namespace Dataset
} // namespace Utilities