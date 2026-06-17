#include <gtest/gtest.h>
#include "../src/utilities/dataset_loader.h"
#include "../src/math/matrix.h"
#include <fstream>
#include <cstdio>

class MNISTLoaderTest : public ::testing::Test {
protected:
    void SetUp() override {
        // Create a dummy MNIST images file
        std::ofstream imgFile("dummy_images-idx3-ubyte", std::ios::binary);
        uint32_t magic = 2051;
        uint32_t num = 2;
        uint32_t rows = 2;
        uint32_t cols = 2;

        auto writeSwapped = [&imgFile](uint32_t val) {
            uint32_t swapped = ((val << 24) & 0xff000000) | ((val << 8) & 0x00ff0000) | ((val >> 8) & 0x0000ff00) | ((val >> 24) & 0x000000ff);
            imgFile.write(reinterpret_cast<char*>(&swapped), 4);
        };

        writeSwapped(magic);
        writeSwapped(num);
        writeSwapped(rows);
        writeSwapped(cols);

        unsigned char pixels[8] = {0, 127, 255, 0, 10, 20, 30, 40};
        imgFile.write(reinterpret_cast<char*>(pixels), 8);
        imgFile.close();

        // Create a dummy MNIST labels file
        std::ofstream lblFile("dummy_labels-idx1-ubyte", std::ios::binary);
        magic = 2049;
        auto writeSwappedLbl = [&lblFile](uint32_t val) {
            uint32_t swapped = ((val << 24) & 0xff000000) | ((val << 8) & 0x00ff0000) | ((val >> 8) & 0x0000ff00) | ((val >> 24) & 0x000000ff);
            lblFile.write(reinterpret_cast<char*>(&swapped), 4);
        };
        writeSwappedLbl(magic);
        writeSwappedLbl(num);
        unsigned char labels[2] = {5, 0};
        lblFile.write(reinterpret_cast<char*>(labels), 2);
        lblFile.close();
    }

    void TearDown() override {
        std::remove("dummy_images-idx3-ubyte");
        std::remove("dummy_labels-idx1-ubyte");
    }
};

TEST_F(MNISTLoaderTest, LoadImagesSuccess) {
    Matrix::Matrix<float> images;
    bool success = Utilities::Dataset::MNISTLoader::LoadImages("dummy_images-idx3-ubyte", images);
    EXPECT_TRUE(success);
    EXPECT_EQ(images.rows(), 2);
    EXPECT_EQ(images.cols(), 4);
    EXPECT_NEAR(images[0][1], 127.0f / 255.0f, 1e-4f);
    EXPECT_NEAR(images[0][2], 1.0f, 1e-4f);
}

TEST_F(MNISTLoaderTest, LoadLabelsSuccess) {
    Matrix::Matrix<float> labels;
    bool success = Utilities::Dataset::MNISTLoader::LoadLabels("dummy_labels-idx1-ubyte", labels);
    EXPECT_TRUE(success);
    EXPECT_EQ(labels.rows(), 2);
    EXPECT_EQ(labels.cols(), 10);
    EXPECT_EQ(labels[0][5], 1.0f);
    EXPECT_EQ(labels[0][0], 0.0f);
    EXPECT_EQ(labels[1][0], 1.0f);
}

TEST_F(MNISTLoaderTest, FileNotFound) {
    Matrix::Matrix<float> images;
    bool success = Utilities::Dataset::MNISTLoader::LoadImages("nonexistent_file", images);
    EXPECT_FALSE(success);
}