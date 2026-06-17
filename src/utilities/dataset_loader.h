#pragma once
#include <string>
#include <vector>
#include <cstdint>
#include "../math/matrix.h"

namespace Utilities {
namespace Dataset {

class MNISTLoader {
public:
    static bool LoadImages(const std::string& filepath, Matrix::Matrix<float>& outImages);
    static bool LoadLabels(const std::string& filepath, Matrix::Matrix<float>& outLabels);
private:
    static uint32_t SwapEndian(uint32_t val);
};

} // namespace Dataset
} // namespace Utilities