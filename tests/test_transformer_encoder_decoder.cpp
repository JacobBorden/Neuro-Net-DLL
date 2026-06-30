#include <gtest/gtest.h>
#include "../src/transformer/transformer_encoder_decoder_model.h"

using namespace NeuroNet::Transformer;

class TransformerEncoderDecoderTest : public ::testing::Test {
protected:
    void SetUp() override {
        // Initialize basic parameters for tests
    }
};

TEST_F(TransformerEncoderDecoderTest, Initialization) {
    int src_vocab_size = 100;
    int tgt_vocab_size = 150;
    int max_seq_len = 50;
    int d_model = 64;
    int num_encoder_layers = 2;
    int num_decoder_layers = 2;
    int num_heads = 8;
    int d_ff = 128;

    EXPECT_NO_THROW({
        TransformerEncoderDecoderModel model(
            src_vocab_size, tgt_vocab_size, max_seq_len, d_model,
            num_encoder_layers, num_decoder_layers, num_heads, d_ff
        );
    });
}

TEST_F(TransformerEncoderDecoderTest, ForwardPassBasic) {
    int src_vocab_size = 100;
    int tgt_vocab_size = 150;
    int max_seq_len = 50;
    int d_model = 64;
    int num_encoder_layers = 1;
    int num_decoder_layers = 1;
    int num_heads = 4;
    int d_ff = 128;

    TransformerEncoderDecoderModel model(
        src_vocab_size, tgt_vocab_size, max_seq_len, d_model,
        num_encoder_layers, num_decoder_layers, num_heads, d_ff
    );

    int src_seq_len = 5;
    int tgt_seq_len = 6;
    Matrix::Matrix<float> src_input_ids(1, src_seq_len);
    Matrix::Matrix<float> tgt_input_ids(1, tgt_seq_len);
    for (int i = 0; i < src_seq_len; ++i) src_input_ids[0][i] = i;
    for (int i = 0; i < tgt_seq_len; ++i) tgt_input_ids[0][i] = i + 10;

    Matrix::Matrix<float> output;
    EXPECT_NO_THROW({
        output = model.forward(src_input_ids, tgt_input_ids);
    });

    EXPECT_EQ(output.rows(), static_cast<size_t>(tgt_seq_len));
    EXPECT_EQ(output.cols(), static_cast<size_t>(d_model));
}

TEST_F(TransformerEncoderDecoderTest, DefaultDecoderMaskPreventsFutureTokenLeakage) {
    TransformerEncoderDecoderModel model(
        100, 100, 10, 16,
        1, 1, 4, 32
    );

    Matrix::Matrix<float> src_input_ids(1, 4);
    for (int i = 0; i < 4; ++i) {
        src_input_ids[0][i] = i + 1;
    }

    Matrix::Matrix<float> tgt_input_ids_a(1, 3);
    Matrix::Matrix<float> tgt_input_ids_b(1, 3);
    tgt_input_ids_a[0][0] = 10;
    tgt_input_ids_a[0][1] = 11;
    tgt_input_ids_a[0][2] = 12;
    tgt_input_ids_b[0][0] = 10;
    tgt_input_ids_b[0][1] = 11;
    tgt_input_ids_b[0][2] = 35;

    Matrix::Matrix<float> output_a = model.forward(src_input_ids, tgt_input_ids_a);
    Matrix::Matrix<float> output_b = model.forward(src_input_ids, tgt_input_ids_b);

    for (size_t row = 0; row < 2; ++row) {
        for (size_t col = 0; col < output_a.cols(); ++col) {
            EXPECT_NEAR(output_a[row][col], output_b[row][col], 1e-5f);
        }
    }
}

TEST_F(TransformerEncoderDecoderTest, InvalidInputHandling) {
    int src_vocab_size = 100;
    int tgt_vocab_size = 150;
    int max_seq_len = 50;
    int d_model = 64;
    int num_encoder_layers = 1;
    int num_decoder_layers = 1;
    int num_heads = 4;
    int d_ff = 128;

    TransformerEncoderDecoderModel model(
        src_vocab_size, tgt_vocab_size, max_seq_len, d_model,
        num_encoder_layers, num_decoder_layers, num_heads, d_ff
    );

    Matrix::Matrix<float> invalid_src_input(2, 5); // Should be 1 row
    Matrix::Matrix<float> tgt_input(1, 6);
    EXPECT_THROW(model.forward(invalid_src_input, tgt_input), std::invalid_argument);
}
