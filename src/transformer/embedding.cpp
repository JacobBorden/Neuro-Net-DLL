#include "embedding.h"
#include <stdexcept> // For std::out_of_range, std::invalid_argument
#include <iostream> // For potential debug cout

namespace NeuroNet {
namespace Transformer {

EmbeddingLayer::EmbeddingLayer(int vocab_size, int embedding_dim)
    : vocab_size_(vocab_size), embedding_dim_(embedding_dim) {
    if (vocab_size <= 0 || embedding_dim <= 0) {
        throw std::invalid_argument("Vocabulary size and embedding dimension must be positive.");
    }
    embedding_table_.resize(vocab_size_, embedding_dim_);
    initialize_weights(true); // Initialize with random weights by default
}

void EmbeddingLayer::initialize_weights(bool random) {
    if (random) {
        embedding_table_.Randomize(); // Assumes Matrix::Randomize exists and works as expected
    } else {
        embedding_table_.assign(0.0f); // Assumes Matrix::assign(value) sets all elements
    }
}

Matrix::Matrix<float> EmbeddingLayer::forward(const Matrix::Matrix<float>& input_token_ids) {
    if (input_token_ids.rows() != 1) {
        // This simplified version expects a single sequence (1 row of token IDs).
        // For batch processing (multiple sequences), this logic would need extension,
        // potentially returning a list of matrices or a 3D tensor if the matrix lib supported it.
        // Current plan is to process one sequence at a time if batching is needed later.
        throw std::invalid_argument("EmbeddingLayer::forward expects input_token_ids to have exactly 1 row (a single sequence).");
    }

    size_t seq_len = input_token_ids.cols();
    if (seq_len == 0) {
        return Matrix::Matrix<float>(0, embedding_dim_); // Return empty if sequence is empty
    }

    Matrix::Matrix<float> output_embeddings(seq_len, embedding_dim_);

    for (size_t i = 0; i < seq_len; ++i) {
        int token_id = static_cast<int>(input_token_ids[0][i]); // Get token ID from the input row

        if (token_id < 0 || token_id >= vocab_size_) {
            // Consider how to handle out-of-vocabulary tokens.
            // Option 1: Throw error (current).
            // Option 2: Use a default <UNK> embedding if one is designated and handled.
            // For now, strict error.
            throw std::out_of_range("Token ID " + std::to_string(token_id) +
                                    " is out of bounds for embedding table (vocab_size: " +
                                    std::to_string(vocab_size_) + ").");
        }

        // Copy the embedding vector (row) for the token_id from embedding_table_
        for (int j = 0; j < embedding_dim_; ++j) {
            output_embeddings[i][j] = embedding_table_[token_id][j];
        }
    }
    return output_embeddings;
}

const Matrix::Matrix<float>& EmbeddingLayer::get_weights() const {
    return embedding_table_;
}

void EmbeddingLayer::set_weights(const Matrix::Matrix<float>& weights) {
    if (weights.rows() != static_cast<size_t>(vocab_size_) || weights.cols() != static_cast<size_t>(embedding_dim_)) {
        throw std::invalid_argument("Dimensions of provided weights (" +
                                    std::to_string(weights.rows()) + "x" + std::to_string(weights.cols()) +
                                    ") do not match EmbeddingLayer's expected dimensions (" +
                                    std::to_string(vocab_size_) + "x" + std::to_string(embedding_dim_) + ").");
    }
    embedding_table_ = weights;
}

} // namespace Transformer
} // namespace NeuroNet
