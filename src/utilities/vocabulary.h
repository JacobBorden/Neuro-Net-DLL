#pragma once

#include <string>
#include <vector>
#include <unordered_map>
#include "../math/matrix.h" // For Matrix::Matrix<float>
#include "json/json.hpp"    // For JsonValue, JsonParser

namespace NeuroNet { // Assuming NeuroNet is the root namespace used elsewhere

class Vocabulary {
public:
    Vocabulary();

    /**
     * @brief Loads vocabulary data from a JSON file.
     * The JSON file should contain 'word_to_token', 'token_to_word', and 'special_tokens' fields.
     * @param filepath Path to the vocabulary JSON file.
     * @return True if loading was successful, false otherwise.
     */
    bool load_from_json(const std::string& filepath);

    /**
     * @brief Gets the token ID for a given word.
     * Converts the word to lowercase before lookup.
     * @param word The word to look up.
     * @return The token ID, or the unknown_token_id if the word is not found.
     */
    int get_token_id(const std::string& word) const;

    /**
     * @brief Gets the word for a given token ID.
     * @param token_id The token ID to look up.
     * @return The word string, or the unknown_token_str if the ID is not found.
     */
    std::string get_word(int token_id) const;

    /**
     * @brief Tokenizes a single raw string sequence into a vector of token IDs.
     * Converts the sequence to lowercase and splits by space.
     * @param sequence_str The raw string sequence.
     * @return A vector of token IDs.
     */
    std::vector<int> tokenize_sequence(const std::string& sequence_str) const;

    /**
     * @brief Tokenizes a batch of raw string sequences.
     * @param batch_sequences A vector of raw string sequences.
     * @return A vector where each element is a vector of token IDs for the corresponding input sequence.
     */
    std::vector<std::vector<int>> tokenize_batch(const std::vector<std::string>& batch_sequences) const;

    /**
     * @brief Converts a batch of raw string sequences into a padded matrix of token IDs (as floats).
     * @param batch_sequences Vector of raw string sequences.
     * @param max_len The maximum length to pad/truncate sequences to.
     *                If -1, uses max_sequence_length_internal_ if set,
     *                otherwise pads to the longest sequence in the current batch if pad_to_max_in_batch is true.
     * @param pad_to_max_in_batch If true and max_len is -1 and max_sequence_length_internal_ is not set,
     *                            sequences are padded to the length of the longest sequence in the current batch.
     * @return A Matrix::Matrix<float> where each row is a processed sequence.
     * @throws std::runtime_error if padding cannot be determined or other errors occur.
     */
    Matrix::Matrix<float> prepare_batch_matrix(
        const std::vector<std::string>& batch_sequences,
        int max_len = -1,
        bool pad_to_max_in_batch = true) const;

    int get_unknown_token_id() const { return unknown_token_id_internal; }
    int get_padding_token_id() const { return padding_token_id_internal; }
    std::string get_unknown_token_str() const { return unknown_token_str_internal; }
    std::string get_padding_token_str() const { return padding_token_str_internal; }

    void set_max_sequence_length(int length) { max_sequence_length_internal_ = length; }
    int get_max_sequence_length() const { return max_sequence_length_internal_; }


private:
    std::unordered_map<std::string, int> word_to_token_map;
    std::unordered_map<int, std::string> token_to_word_map;

    std::string unknown_token_str_internal = "<UNK>";
    std::string padding_token_str_internal = "<PAD>";
    int unknown_token_id_internal = -1;
    int padding_token_id_internal = -1;
    int max_sequence_length_internal_ = -1; // Default: not set

    std::string to_lowercase(const std::string& str) const;
    std::vector<std::string> split_by_space(const std::string& str) const;
};

} // namespace NeuroNet
