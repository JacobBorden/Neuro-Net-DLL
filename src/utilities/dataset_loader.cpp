#include "dataset_loader.h"
#include <fstream>

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

    if (!file.read(reinterpret_cast<char*>(&magic_number), sizeof(magic_number))) return false;
    magic_number = SwapEndian(magic_number);
    if (magic_number != 2051) return false;

    if (!file.read(reinterpret_cast<char*>(&number_of_images), sizeof(number_of_images))) return false;
    number_of_images = SwapEndian(number_of_images);

    if (!file.read(reinterpret_cast<char*>(&n_rows), sizeof(n_rows))) return false;
    n_rows = SwapEndian(n_rows);

    if (!file.read(reinterpret_cast<char*>(&n_cols), sizeof(n_cols))) return false;
    n_cols = SwapEndian(n_cols);

    const size_t image_size = static_cast<size_t>(n_rows) * static_cast<size_t>(n_cols);
    Matrix::Matrix<float> images(number_of_images, image_size);

    for (size_t i = 0; i < number_of_images; ++i) {
        for (size_t r = 0; r < image_size; ++r) {
            unsigned char temp = 0;
            if (!file.read(reinterpret_cast<char*>(&temp), sizeof(temp))) return false;
            images[i][r] = static_cast<float>(temp) / 255.0f;
        }
    }

    outImages = images;
    return true;
}

bool MNISTLoader::LoadLabels(const std::string& filepath, Matrix::Matrix<float>& outLabels) {
    std::ifstream file(filepath, std::ios::binary);
    if (!file.is_open()) return false;

    uint32_t magic_number = 0;
    uint32_t number_of_items = 0;

    if (!file.read(reinterpret_cast<char*>(&magic_number), sizeof(magic_number))) return false;
    magic_number = SwapEndian(magic_number);
    if (magic_number != 2049) return false;

    if (!file.read(reinterpret_cast<char*>(&number_of_items), sizeof(number_of_items))) return false;
    number_of_items = SwapEndian(number_of_items);

    Matrix::Matrix<float> labels;
    labels.assign(number_of_items, 10, 0.0f);

    for (size_t i = 0; i < number_of_items; ++i) {
        unsigned char temp = 0;
        if (!file.read(reinterpret_cast<char*>(&temp), sizeof(temp))) return false;
        if (temp < 10) {
            labels[i][temp] = 1.0f;
        }
    }

    outLabels = labels;
    return true;
}

} // namespace Dataset
} // namespace Utilities
