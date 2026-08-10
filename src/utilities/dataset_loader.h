#pragma once
#include <string>
#include <cstdint>
#include "../math/matrix.h"

namespace Utilities {
namespace Dataset {

/**
 * @brief Loads datasets stored in the MNIST IDX binary file format.
 */
class MNISTLoader {
public:
    /**
     * @brief Loads and normalizes MNIST image data from an IDX image file.
     * @param filepath Path to the IDX image file.
     * @param outImages Output matrix where each row is one flattened image.
     * @return True when the file is well-formed and fully read; false otherwise.
     */
    static bool LoadImages(const std::string& filepath, Matrix::Matrix<float>& outImages);

    /**
     * @brief Loads MNIST labels from an IDX label file as one-hot rows.
     * @param filepath Path to the IDX label file.
     * @param outLabels Output matrix where each row is a one-hot label vector.
     * @return True when the file is well-formed and fully read; false otherwise.
     */
    static bool LoadLabels(const std::string& filepath, Matrix::Matrix<float>& outLabels);
private:
    static uint32_t SwapEndian(uint32_t val);
};

} // namespace Dataset
} // namespace Utilities
