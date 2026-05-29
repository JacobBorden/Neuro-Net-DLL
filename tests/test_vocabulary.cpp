#include "gtest/gtest.h"
#include "../src/utilities/vocabulary.h" // Adjust path as necessary
#include <fstream>      // For creating temporary test vocab files
#include <vector>
#include <string>
#include <cstdio> // For std::remove

// Helper function to create a temporary vocabulary JSON file for tests
void CreateTempVocabFile(const std::string& filepath, const std::string& content) {
    std::ofstream ofs(filepath);
    ofs << content;
    ofs.close();
}

// Test Fixture for Vocabulary Tests (optional, but good for setup/teardown if needed)
class VocabularyTest : public ::testing::Test {
protected:
    const std::string test_vocab_path = "temp_test_vocab.json";
    NeuroNet::Vocabulary vocab;

    void TearDown() override {
        std::remove(test_vocab_path.c_str()); // Clean up temp file
    }
};

TEST_F(VocabularyTest, LoadFromJson_Valid) {
    CreateTempVocabFile(test_vocab_path, R"({
        "word_to_token": { "hello": 0, "world": 1, "<UNK>": 2, "<PAD>": 3 },
        "token_to_word": { "0": "hello", "1": "world", "2": "<UNK>", "3": "<PAD>" },
        "special_tokens": { "unknown_token": "<UNK>", "padding_token": "<PAD>" },
        "config": { "max_sequence_length": 10 }
    })");
    ASSERT_TRUE(vocab.load_from_json(test_vocab_path));
    EXPECT_EQ(vocab.get_token_id("hello"), 0);
    EXPECT_EQ(vocab.get_token_id("world"), 1);
    EXPECT_EQ(vocab.get_unknown_token_id(), 2);
    EXPECT_EQ(vocab.get_padding_token_id(), 3);
    EXPECT_EQ(vocab.get_max_sequence_length(), 10);
    EXPECT_EQ(vocab.get_word(0), "hello");
    EXPECT_EQ(vocab.get_word(3), "<PAD>");
}

TEST_F(VocabularyTest, LoadFromJson_MalformedJson) {
    CreateTempVocabFile(test_vocab_path, R"({ "word_to_token": { "hello": 0, ... )"); // Malformed
    EXPECT_FALSE(vocab.load_from_json(test_vocab_path));
}

TEST_F(VocabularyTest, LoadFromJson_MissingWordToToken) {
    CreateTempVocabFile(test_vocab_path, R"({
        "token_to_word": { "0": "hello" },
        "special_tokens": { "unknown_token": "<UNK>", "padding_token": "<PAD>" }
    })");
    EXPECT_FALSE(vocab.load_from_json(test_vocab_path));
}

TEST_F(VocabularyTest, LoadFromJson_MissingSpecialTokens) {
    CreateTempVocabFile(test_vocab_path, R"({
        "word_to_token": { "hello": 0, "<UNK>": 1, "<PAD>": 2 },
        "token_to_word": { "0": "hello", "1": "<UNK>", "2": "<PAD>" }
    })");
    EXPECT_FALSE(vocab.load_from_json(test_vocab_path));
}

TEST_F(VocabularyTest, LoadFromJson_SpecialTokenNotInWordMap) {
    CreateTempVocabFile(test_vocab_path, R"({
        "word_to_token": { "hello": 0 },
        "token_to_word": { "0": "hello" },
        "special_tokens": { "unknown_token": "<UNK>", "padding_token": "<PAD>" }
    })");
    EXPECT_FALSE(vocab.load_from_json(test_vocab_path)); // <UNK> not in word_to_token
}


TEST_F(VocabularyTest, GetTokenId) {
    CreateTempVocabFile(test_vocab_path, R"({
        "word_to_token": { "hello": 0, "world": 1, "<unk>": 2, "<pad>": 3 },
        "token_to_word": { "0": "hello", "1": "world", "2": "<unk>", "3": "<pad>" },
        "special_tokens": { "unknown_token": "<unk>", "padding_token": "<pad>" }
    })");
    ASSERT_TRUE(vocab.load_from_json(test_vocab_path));
    EXPECT_EQ(vocab.get_token_id("hello"), 0);
    EXPECT_EQ(vocab.get_token_id("HELLO"), 0); // Test case insensitivity (due to to_lowercase)
    EXPECT_EQ(vocab.get_token_id("unknownword"), 2); // unknown_token_id
}

TEST_F(VocabularyTest, GetWord) {
     CreateTempVocabFile(test_vocab_path, R"({
        "word_to_token": { "hello": 0, "world": 1, "<unk>": 2, "<pad>": 3 },
        "token_to_word": { "0": "hello", "1": "world", "2": "<unk>", "3": "<pad>" },
        "special_tokens": { "unknown_token": "<unk>", "padding_token": "<pad>" }
    })");
    ASSERT_TRUE(vocab.load_from_json(test_vocab_path));
    EXPECT_EQ(vocab.get_word(0), "hello");
    EXPECT_EQ(vocab.get_word(1), "world");
    EXPECT_EQ(vocab.get_word(100), "<unk>"); // unknown_token_str
}

TEST_F(VocabularyTest, TokenizeSequence) {
    CreateTempVocabFile(test_vocab_path, R"({
        "word_to_token": { "this":0, "is":1, "a":2, "test":3, "<unk>":4, "<pad>":5 },
        "token_to_word": { "0":"this", "1":"is", "2":"a", "3":"test", "4":"<unk>", "5":"<pad>" },
        "special_tokens": { "unknown_token": "<unk>", "padding_token": "<pad>" }
    })");
    ASSERT_TRUE(vocab.load_from_json(test_vocab_path));
    std::vector<int> expected_ids = {0, 1, 2, 3}; // this is a test
    EXPECT_EQ(vocab.tokenize_sequence("This is a test"), expected_ids);
    std::vector<int> expected_ids_with_oov = {0, 1, 4}; // this is <unk>
    EXPECT_EQ(vocab.tokenize_sequence("This is OOV"), expected_ids_with_oov);
    EXPECT_TRUE(vocab.tokenize_sequence("").empty());
}

TEST_F(VocabularyTest, PrepareBatchMatrix_PadToMaxLenParam) {
    CreateTempVocabFile(test_vocab_path, R"({
        "word_to_token": { "hello":0, "world":1, "test":2, "<unk>":3, "<pad>":4 },
        "token_to_word": { "0":"hello", "1":"world", "2":"test", "3":"<unk>", "4":"<pad>" },
        "special_tokens": { "unknown_token": "<unk>", "padding_token": "<pad>" }
    })");
    ASSERT_TRUE(vocab.load_from_json(test_vocab_path));

    std::vector<std::string> batch = {"hello world", "hello"};
    Matrix::Matrix<float> matrix = vocab.prepare_batch_matrix(batch, 3); // max_len_param = 3

    ASSERT_EQ(matrix.rows(), 2);
    ASSERT_EQ(matrix.cols(), 3);
    EXPECT_FLOAT_EQ(matrix[0][0], 0); // hello
    EXPECT_FLOAT_EQ(matrix[0][1], 1); // world
    EXPECT_FLOAT_EQ(matrix[0][2], 4); // <pad>
    EXPECT_FLOAT_EQ(matrix[1][0], 0); // hello
    EXPECT_FLOAT_EQ(matrix[1][1], 4); // <pad>
    EXPECT_FLOAT_EQ(matrix[1][2], 4); // <pad>
}

TEST_F(VocabularyTest, PrepareBatchMatrix_PadToInternalMaxSeqLen) {
    CreateTempVocabFile(test_vocab_path, R"({
        "word_to_token": { "hello":0, "<pad>":1, "<unk>":2 },
        "token_to_word": { "0":"hello", "1":"<pad>", "2":"<unk>" },
        "special_tokens": { "unknown_token": "<unk>", "padding_token": "<pad>" },
        "config": { "max_sequence_length": 3 }
    })");
    ASSERT_TRUE(vocab.load_from_json(test_vocab_path)); // This loads max_sequence_length = 3

    std::vector<std::string> batch = {"hello"};
    Matrix::Matrix<float> matrix = vocab.prepare_batch_matrix(batch); // Use internal max_seq_len

    ASSERT_EQ(matrix.rows(), 1);
    ASSERT_EQ(matrix.cols(), 3);
    EXPECT_FLOAT_EQ(matrix[0][0], 0); // hello
    EXPECT_FLOAT_EQ(matrix[0][1], 1); // <pad>
    EXPECT_FLOAT_EQ(matrix[0][2], 1); // <pad>
}

TEST_F(VocabularyTest, PrepareBatchMatrix_PadToMaxInBatch) {
    CreateTempVocabFile(test_vocab_path, R"({
        "word_to_token": { "a":0, "b":1, "c":2, "<pad>":3, "<unk>":4 },
        "token_to_word": { "0":"a", "1":"b", "2":"c", "3":"<pad>", "4":"<unk>" },
        "special_tokens": { "unknown_token": "<unk>", "padding_token": "<pad>" }
    })"); // No max_sequence_length in config
    ASSERT_TRUE(vocab.load_from_json(test_vocab_path));

    std::vector<std::string> batch = {"a b c", "a b"}; // Longest is 3
    Matrix::Matrix<float> matrix = vocab.prepare_batch_matrix(batch, -1, true);

    ASSERT_EQ(matrix.rows(), 2);
    ASSERT_EQ(matrix.cols(), 3);
    EXPECT_FLOAT_EQ(matrix[0][0], 0); EXPECT_FLOAT_EQ(matrix[0][1], 1); EXPECT_FLOAT_EQ(matrix[0][2], 2);
    EXPECT_FLOAT_EQ(matrix[1][0], 0); EXPECT_FLOAT_EQ(matrix[1][1], 1); EXPECT_FLOAT_EQ(matrix[1][2], 3); // <pad>
}

TEST_F(VocabularyTest, PrepareBatchMatrix_Truncate) {
    CreateTempVocabFile(test_vocab_path, R"({
        "word_to_token": { "a":0, "b":1, "c":2, "<pad>":3, "<unk>":4 },
        "token_to_word": { "0":"a", "1":"b", "2":"c", "3":"<pad>", "4":"<unk>" },
        "special_tokens": { "unknown_token": "<unk>", "padding_token": "<pad>" }
    })");
    ASSERT_TRUE(vocab.load_from_json(test_vocab_path));

    std::vector<std::string> batch = {"a b c"};
    Matrix::Matrix<float> matrix = vocab.prepare_batch_matrix(batch, 2); // max_len_param = 2

    ASSERT_EQ(matrix.rows(), 1);
    ASSERT_EQ(matrix.cols(), 2);
    EXPECT_FLOAT_EQ(matrix[0][0], 0); // a
    EXPECT_FLOAT_EQ(matrix[0][1], 1); // b (c is truncated)
}

TEST_F(VocabularyTest, PrepareBatchMatrix_EmptyBatch) {
    CreateTempVocabFile(test_vocab_path, R"({
        "word_to_token": { "<pad>":0, "<unk>":1 },
        "token_to_word": { "0":"<pad>", "1":"<unk>" },
        "special_tokens": { "unknown_token": "<unk>", "padding_token": "<pad>" }
    })");
    ASSERT_TRUE(vocab.load_from_json(test_vocab_path));
    std::vector<std::string> batch = {};
    Matrix::Matrix<float> matrix = vocab.prepare_batch_matrix(batch);
    EXPECT_EQ(matrix.rows(), 0);
    EXPECT_EQ(matrix.cols(), 0);
}

TEST_F(VocabularyTest, PrepareBatchMatrix_BatchOfEmptyStrings) {
    CreateTempVocabFile(test_vocab_path, R"({
        "word_to_token": { "<pad>":0, "<unk>":1 },
        "token_to_word": { "0":"<pad>", "1":"<unk>" },
        "special_tokens": { "unknown_token": "<unk>", "padding_token": "<pad>" }
    })");
    ASSERT_TRUE(vocab.load_from_json(test_vocab_path));
    std::vector<std::string> batch = {"", ""}; // Longest sequence is 0 tokens
    // prepare_batch_matrix ensures current_max_len is at least 1 if batch is not empty.
    Matrix::Matrix<float> matrix = vocab.prepare_batch_matrix(batch, -1, true);
    ASSERT_EQ(matrix.rows(), 2);
    ASSERT_EQ(matrix.cols(), 1); // Padded to 1 column of <pad>
    EXPECT_FLOAT_EQ(matrix[0][0], 0); // <pad>
    EXPECT_FLOAT_EQ(matrix[1][0], 0); // <pad>
}
