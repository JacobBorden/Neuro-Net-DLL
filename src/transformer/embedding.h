#pragma once

#include "../math/matrix.h" // Path to your Matrix library
#include <vector>
#include <string> // For std::string in weight serialization (optional now)

namespace NeuroNet {
namespace Transformer {

class EmbeddingLayer {
public:
    /**
     * @brief Constructs an EmbeddingLayer.
     * @param vocab_size The total number of unique tokens in the vocabulary.
     * @param embedding_dim The dimensionality of the embedding vectors.
     */
    EmbeddingLayer(int vocab_size, int embedding_dim);

    /**
     * @brief Initializes the embedding weights.
     * Weights are initialized randomly by default using the Matrix::Randomize() method,
     * which typically initializes between -1 and 1.
     * @param random If true (default), initializes with random values. If false, initializes to zero.
     */
    void initialize_weights(bool random = true);

    /**
     * @brief Performs the forward pass of the embedding layer.
     * Converts a matrix of token IDs into a matrix of corresponding embedding vectors.
     * Input is assumed to be a 2D matrix where each row is a sequence of token IDs.
     * Output will be a 2D matrix where each row corresponds to an input row,
     * and columns are the concatenated embeddings of tokens in that input row.
     * For current 2D matrix lib: if input is (1, seq_len), output is (seq_len, embedding_dim).
     * If input is (N, seq_len), output is (N * seq_len, embedding_dim) - this will need careful handling by caller.
     * Let's simplify for now: input (1, seq_len) -> output (seq_len, embedding_dim).
     *
     * @param input_token_ids A Matrix::Matrix<float> containing token IDs.
     *                        Expected to have 1 row, where cols = sequence length.
     *                        Values should be valid token IDs (indices for the embedding table).
     * @return Matrix::Matrix<float> The resulting matrix of embedding vectors.
     *         Dimensions: (sequence_length, embedding_dim).
     * @throws std::out_of_range if a token ID is out of bounds for the embedding table.
     * @throws std::invalid_argument if input_token_ids has more than 1 row (for this simplified version).
     */
    Matrix::Matrix<float> forward(const Matrix::Matrix<float>& input_token_ids);

    /**
     * @brief Gets the embedding table (weights).
     * @return const Matrix::Matrix<float>& The embedding table.
     */
    const Matrix::Matrix<float>& get_weights() const;

    /**
     * @brief Sets the embedding table (weights).
     * @param weights The new embedding table. Must match expected dimensions.
     * @throws std::invalid_argument if dimensions of weights do not match.
     */
    void set_weights(const Matrix::Matrix<float>& weights);

    int get_vocab_size() const { return vocab_size_; }
    int get_embedding_dim() const { return embedding_dim_; }


private:
    int vocab_size_;
    int embedding_dim_;
    Matrix::Matrix<float> embedding_table_; // vocab_size x embedding_dim
};

} // namespace Transformer
} // namespace NeuroNet
