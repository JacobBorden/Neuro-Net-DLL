#include <gtest/gtest.h>
#include "../src/transformer/positional_encoding.h"
#include "../src/math/matrix.h"
#include <stdexcept>

TEST(PositionalEncodingTest, BasicForward) {
    const int max_seq_len = 10;
    const int emb_dim = 4;
    NeuroNet::Transformer::PositionalEncoding pe(max_seq_len, emb_dim);

    Matrix::Matrix<float> input(2, emb_dim);
    input[0][0] = 1.0f; input[0][1] = 2.0f; input[0][2] = 3.0f; input[0][3] = 4.0f;
    input[1][0] = 5.0f; input[1][1] = 6.0f; input[1][2] = 7.0f; input[1][3] = 8.0f;

    Matrix::Matrix<float> output = pe.forward(input);
    EXPECT_EQ(output.rows(), 2);
    EXPECT_EQ(output.cols(), emb_dim);

    const auto& table = pe.get_pe_table();
    for (size_t i = 0; i < 2; ++i) {
        for (size_t j = 0; j < static_cast<size_t>(emb_dim); ++j) {
            EXPECT_FLOAT_EQ(output[i][j], input[i][j] + table[i][j]);
        }
    }
}

TEST(PositionalEncodingTest, EmptyInput) {
    NeuroNet::Transformer::PositionalEncoding pe(10, 4);
    Matrix::Matrix<float> input(0, 4);
    Matrix::Matrix<float> output = pe.forward(input);
    EXPECT_EQ(output.rows(), 0);
    EXPECT_EQ(output.cols(), 4);
}

TEST(PositionalEncodingTest, DimensionMismatch) {
    NeuroNet::Transformer::PositionalEncoding pe(10, 4);
    Matrix::Matrix<float> input(2, 5);
    EXPECT_THROW(pe.forward(input), std::invalid_argument);
}

TEST(PositionalEncodingTest, SequenceLengthExceeded) {
    NeuroNet::Transformer::PositionalEncoding pe(5, 4);
    Matrix::Matrix<float> input(6, 4);
    EXPECT_THROW(pe.forward(input), std::invalid_argument);
}
