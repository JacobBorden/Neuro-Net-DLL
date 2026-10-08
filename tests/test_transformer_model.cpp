#include "gtest/gtest.h"
#include "../src/transformer/transformer_model.h"
#include "../src/math/matrix.h"
#include <vector>
#include <string>
#include <stdexcept>

// Test fixture for TransformerModel tests
class TransformerModelTest : public ::testing::Test {
protected:
    // NeuroNet::Transformer::TransformerModel model; // Will be initialized in each test
};

// Test case for the default constructor - REMOVED as there is no default constructor
// TEST_F(TransformerModelTest, DefaultConstructor) {
//     // Depending on the default behavior, add assertions here.
//     // For example, if it initializes with default layers or a specific state:
//     // EXPECT_EQ(model.get_num_layers(), DEFAULT_NUM_LAYERS);
//     // EXPECT_EQ(model.get_model_dim(), DEFAULT_MODEL_DIM);
//     // For now, just ensure it doesn't crash
//     // ASSERT_NE(&model, nullptr);
// }

TEST_F(TransformerModelTest, LoadModelRejectsUnsafePaths) {
    EXPECT_THROW(NeuroNet::Transformer::TransformerModel::load_model("../transformer_model.json"), std::runtime_error);
    EXPECT_THROW(NeuroNet::Transformer::TransformerModel::load_model("/tmp/transformer_model.json"), std::runtime_error);
    EXPECT_THROW(NeuroNet::Transformer::TransformerModel::load_model("C:\\temp\\transformer_model.json"), std::runtime_error);
}

// Test case for initialization with parameters
TEST_F(TransformerModelTest, Initialization) {
    const int vocab_size = 1000;
    const int max_seq_len = 50;
    const int d_model = 512;
    const int num_encoder_layers = 6;
    const int num_heads = 8;
    const int d_ff = 2048;
    const float dropout_rate = 0.1f; // MHA_dropout_rate and FFN_dropout_rate

    NeuroNet::Transformer::TransformerModel model(
        vocab_size, max_seq_len, d_model, num_encoder_layers, num_heads, d_ff, dropout_rate, dropout_rate
    );

    // Add assertions to check if the model is initialized correctly
    // These depend on available getter methods in TransformerModel
    EXPECT_EQ(model.get_vocab_size(), vocab_size);
    EXPECT_EQ(model.get_max_seq_len(), max_seq_len);
    EXPECT_EQ(model.get_d_model(), d_model);
    EXPECT_EQ(model.get_num_encoder_layers(), num_encoder_layers);
    EXPECT_EQ(model.get_num_heads(), num_heads);
    EXPECT_EQ(model.get_d_ff(), d_ff);
    // EXPECT_EQ(model.get_MHA_dropout_rate(), dropout_rate); // Getter does not exist
    // EXPECT_EQ(model.get_FFN_dropout_rate(), dropout_rate); // Getter does not exist

    // For now, we'll assume initialization is successful if no errors are thrown.
    // More detailed checks require inspecting the internal state or behavior.
    SUCCEED();
}

// Test case for forward pass (basic check)
TEST_F(TransformerModelTest, ForwardPassBasic) {
    const int vocab_size_test = 100;
    const int max_seq_len_test = 10;
    const int d_model_test = 64;
    const int num_layers_test = 2; // Smaller model for faster testing
    const int num_heads_test = 4;
    const int d_ff_test = 128;
    const float dropout_rate_test = 0.0f; // Disable dropout for deterministic testing

    NeuroNet::Transformer::TransformerModel model(
        vocab_size_test, max_seq_len_test, d_model_test, num_layers_test, num_heads_test, d_ff_test, dropout_rate_test, dropout_rate_test
    );

    // Create a dummy input matrix (batch_size=1, seq_len=5)
    // Values are token IDs (integers converted to float for the model)
    const int current_seq_len = 5;
    Matrix::Matrix<float> input_sequence(1, current_seq_len);
    for (int j = 0; j < current_seq_len; ++j) {
        input_sequence[0][j] = static_cast<float>(j + 1); // Token IDs 1.0, 2.0, 3.0, 4.0, 5.0
    }

    // Create a dummy attention mask (float matrix)
    // For this basic test, let's assume no mask or a full mask (all 1.0s).
    // The mask should be (seq_len, seq_len) for self-attention.
    Matrix::Matrix<float> attention_mask(current_seq_len, current_seq_len);
    attention_mask.assign(1.0f); // All elements to 1.0f, indicating allow attention for all pairs

    Matrix::Matrix<float> output_matrix;
    // The forward pass takes float matrices.
    ASSERT_NO_THROW(output_matrix = model.forward(input_sequence, attention_mask));

    // Check output dimensions
    // Expected: (batch_size, seq_len, model_dim) - but output is likely 2D (batch_size * seq_len, model_dim) or (batch_size, seq_len * model_dim)
    // Or, if it's probabilities over vocab: (batch_size, seq_len, vocab_size)
    // This needs clarification based on TransformerModel's actual output structure.
    // For now, let's assume the output is (batch_size, seq_len, model_dim) flattened or processed.
    // Without knowing the exact output structure of `model.forward`, we can only make basic checks.

    // Example: If output is (batch_size, seq_len * model_dim)
    // EXPECT_EQ(output_matrix.rows(), 1); // batch_size
    // EXPECT_EQ(output_matrix.cols(), 5 * model_dim); // seq_len * model_dim

    // Example: If output is (batch_size * seq_len, model_dim)
    // EXPECT_EQ(output_matrix.rows(), 1 * 5); // batch_size * seq_len
    // EXPECT_EQ(output_matrix.cols(), model_dim);

    // For now, just check that the output matrix is not empty if the forward pass succeeded.
    EXPECT_GT(output_matrix.rows(), 0);
    EXPECT_GT(output_matrix.cols(), 0);
}

// Test for handling invalid input (e.g., empty sequence)
TEST_F(TransformerModelTest, ForwardPassEmptyInput) {
    const int vocab_size_test = 50;
    const int max_seq_len_test = 5;
    const int d_model_test = 32;
    const int num_layers_test = 1;
    const int num_heads_test = 2;
    const int d_ff_test = 64;

    NeuroNet::Transformer::TransformerModel model(
        vocab_size_test, max_seq_len_test, d_model_test, num_layers_test, num_heads_test, d_ff_test, 0.0f, 0.0f
    );

    Matrix::Matrix<float> empty_input_sequence(0, 0); // Empty input
    Matrix::Matrix<float> empty_mask(0,0); // Empty mask, matching forward signature

    // Behavior for empty input depends on implementation.
    // It might throw an error, or return an empty/specific matrix.
    // For this example, let's assume it should throw std::invalid_argument.
    // Adjust if the actual error type or behavior is different.
    EXPECT_THROW(model.forward(empty_input_sequence, empty_mask), std::invalid_argument);
}

// Test for input sequence exceeding max_seq_len
TEST_F(TransformerModelTest, ForwardPassInputTooLong) {
    const int vocab_size_test = 50;
    const int max_seq_len_test = 5; // Max sequence length is 5
    const int d_model_test = 32;
    const int num_layers_test = 1;
    const int num_heads_test = 2;
    const int d_ff_test = 64;

    NeuroNet::Transformer::TransformerModel model(
        vocab_size_test, max_seq_len_test, d_model_test, num_layers_test, num_heads_test, d_ff_test, 0.0f, 0.0f
    );

    const int current_seq_len = max_seq_len_test + 1; // Sequence length 6
    Matrix::Matrix<float> long_input_sequence(1, current_seq_len);
    for (size_t j = 0; j < long_input_sequence.cols(); ++j) {
        long_input_sequence[0][j] = static_cast<float>(j + 1);
    }
    Matrix::Matrix<float> mask(1, current_seq_len);
    mask.assign(1.0f); // Fill with 1.0f


    // Behavior for input exceeding max_seq_len.
    // It might truncate, throw an error, or handle it in another way.
    // Assuming it throws std::invalid_argument if not automatically truncated.
    // If truncation is the expected behavior, this test needs to be adjusted
    // to check that the output corresponds to a truncated input.
    EXPECT_THROW(model.forward(long_input_sequence, mask), std::invalid_argument);
}

TEST(ScaledDotProductAttentionTest, MaskedAndUnmaskedReferenceValues) {
    NeuroNet::Transformer::ScaledDotProductAttention attn;

    Matrix::Matrix<float> query(1, 2);
    query[0][0] = 1.0f; query[0][1] = 0.0f;

    Matrix::Matrix<float> key(2, 2);
    key[0][0] = 1.0f; key[0][1] = 0.0f;
    key[1][0] = 0.0f; key[1][1] = 1.0f;

    Matrix::Matrix<float> value(2, 2);
    value[0][0] = 10.0f; value[0][1] = 20.0f;
    value[1][0] = 30.0f; value[1][1] = 40.0f;

    // 1. Masked attention test: mask out index 1 with large negative value (-1e9)
    Matrix::Matrix<float> mask(1, 2);
    mask[0][0] = 0.0f; mask[0][1] = -1e9f;

    auto res_masked = attn.forward(query, key, value, mask);
    EXPECT_NEAR(res_masked.attention_weights[0][0], 1.0f, 1e-4f);
    EXPECT_NEAR(res_masked.attention_weights[0][1], 0.0f, 1e-4f);
    EXPECT_NEAR(res_masked.output[0][0], 10.0f, 1e-4f);
    EXPECT_NEAR(res_masked.output[0][1], 20.0f, 1e-4f);

    // 2. Unmasked attention test
    auto res_unmasked = attn.forward(query, key, value);
    float exp_s0 = std::exp(1.0f / std::sqrt(2.0f));
    float exp_s1 = std::exp(0.0f);
    float sum_exp = exp_s0 + exp_s1;
    float w0 = exp_s0 / sum_exp;
    float w1 = exp_s1 / sum_exp;

    EXPECT_NEAR(res_unmasked.attention_weights[0][0], w0, 1e-4f);
    EXPECT_NEAR(res_unmasked.attention_weights[0][1], w1, 1e-4f);
    EXPECT_NEAR(res_unmasked.output[0][0], w0 * 10.0f + w1 * 30.0f, 1e-4f);
    EXPECT_NEAR(res_unmasked.output[0][1], w0 * 20.0f + w1 * 40.0f, 1e-4f);
}

TEST(ScaledDotProductAttentionTest, InvalidAndEmptyShapeHandling) {
    NeuroNet::Transformer::ScaledDotProductAttention attn;

    Matrix::Matrix<float> q(1, 2); q.assign(1.0f);
    Matrix::Matrix<float> k_mismatched_cols(2, 3); k_mismatched_cols.assign(1.0f);
    Matrix::Matrix<float> v(2, 2); v.assign(1.0f);

    // Query and Key column mismatch
    EXPECT_THROW(attn.forward(q, k_mismatched_cols, v), std::invalid_argument);

    // Key and Value row mismatch
    Matrix::Matrix<float> k_valid(2, 2); k_valid.assign(1.0f);
    Matrix::Matrix<float> v_mismatched_rows(3, 2); v_mismatched_rows.assign(1.0f);
    EXPECT_THROW(attn.forward(q, k_valid, v_mismatched_rows), std::invalid_argument);

    // Zero feature dimension d_k
    Matrix::Matrix<float> q_zero_col(1, 0);
    Matrix::Matrix<float> k_zero_col(2, 0);
    Matrix::Matrix<float> v_zero_col(2, 2);
    EXPECT_THROW(attn.forward(q_zero_col, k_zero_col, v_zero_col), std::invalid_argument);

    // Mismatched mask dimensions
    Matrix::Matrix<float> invalid_mask(2, 2); // Expected 1x2 to match query rows and key rows
    EXPECT_THROW(attn.forward(q, k_valid, v, invalid_mask), std::invalid_argument);
}
