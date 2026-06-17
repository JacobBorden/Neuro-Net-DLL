#include <gtest/gtest.h>
#include "../src/utilities/dataset_loader.h"
#include "../src/math/matrix.h"
#include <cstdint>
#include <fstream>
#include <cstdio>

class MNISTLoaderTest : public ::testing::Test {
protected:
    static void WriteBigEndian(std::ofstream& file, uint32_t val) {
        uint32_t swapped = ((val << 24) & 0xff000000) |
                           ((val << 8) & 0x00ff0000) |
                           ((val >> 8) & 0x0000ff00) |
                           ((val >> 24) & 0x000000ff);
        file.write(reinterpret_cast<char*>(&swapped), 4);
    }

    void SetUp() override {
        // Create a dummy MNIST images file
        std::ofstream imgFile("dummy_images-idx3-ubyte", std::ios::binary);
        uint32_t magic = 2051;
        uint32_t num = 2;
        uint32_t rows = 2;
        uint32_t cols = 2;

        WriteBigEndian(imgFile, magic);
        WriteBigEndian(imgFile, num);
        WriteBigEndian(imgFile, rows);
        WriteBigEndian(imgFile, cols);

        unsigned char pixels[8] = {0, 127, 255, 0, 10, 20, 30, 40};
        imgFile.write(reinterpret_cast<char*>(pixels), 8);
        imgFile.close();

        // Create a dummy MNIST labels file
        std::ofstream lblFile("dummy_labels-idx1-ubyte", std::ios::binary);
        magic = 2049;
        WriteBigEndian(lblFile, magic);
        WriteBigEndian(lblFile, num);
        unsigned char labels[2] = {5, 0};
        lblFile.write(reinterpret_cast<char*>(labels), 2);
        lblFile.close();
    }

    void TearDown() override {
        std::remove("dummy_images-idx3-ubyte");
        std::remove("dummy_labels-idx1-ubyte");
        std::remove("truncated_images-idx3-ubyte");
        std::remove("truncated_labels-idx1-ubyte");
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

TEST_F(MNISTLoaderTest, LoadImagesRejectsTruncatedPayload) {
    std::ofstream imgFile("truncated_images-idx3-ubyte", std::ios::binary);
    WriteBigEndian(imgFile, 2051);
    WriteBigEndian(imgFile, 1);
    WriteBigEndian(imgFile, 2);
    WriteBigEndian(imgFile, 2);
    unsigned char pixels[3] = {0, 127, 255};
    imgFile.write(reinterpret_cast<char*>(pixels), 3);
    imgFile.close();

    Matrix::Matrix<float> images(1, 1);
    images[0][0] = 42.0f;

    bool success = Utilities::Dataset::MNISTLoader::LoadImages("truncated_images-idx3-ubyte", images);
    EXPECT_FALSE(success);
    EXPECT_EQ(images.rows(), 1);
    EXPECT_EQ(images.cols(), 1);
    EXPECT_EQ(images[0][0], 42.0f);
}

TEST_F(MNISTLoaderTest, LoadLabelsRejectsTruncatedPayload) {
    std::ofstream lblFile("truncated_labels-idx1-ubyte", std::ios::binary);
    WriteBigEndian(lblFile, 2049);
    WriteBigEndian(lblFile, 2);
    unsigned char labels[1] = {3};
    lblFile.write(reinterpret_cast<char*>(labels), 1);
    lblFile.close();

    Matrix::Matrix<float> labels_matrix(1, 1);
    labels_matrix[0][0] = 7.0f;

    bool success = Utilities::Dataset::MNISTLoader::LoadLabels("truncated_labels-idx1-ubyte", labels_matrix);
    EXPECT_FALSE(success);
    EXPECT_EQ(labels_matrix.rows(), 1);
    EXPECT_EQ(labels_matrix.cols(), 1);
    EXPECT_EQ(labels_matrix[0][0], 7.0f);
}
