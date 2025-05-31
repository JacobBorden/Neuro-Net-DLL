#pragma once

#include "../math/matrix.h" // Path to your Matrix library
#include <cmath>           // For std::sin, std::cos, std::pow

namespace NeuroNet {
namespace Transformer {

class PositionalEncoding {
public:
    /**
     * @brief Constructs a PositionalEncoding layer.
     * Pre-calculates sinusoidal positional encodings.
     * @param max_seq_len The maximum sequence length for which to generate encodings.
     * @param embedding_dim The dimensionality of the embeddings (must match input embeddings).
     */
    PositionalEncoding(int max_seq_len, int embedding_dim);

    /**
     * @brief Adds positional encodings to the input embedding matrix.
     * @param input_embeddings A Matrix::Matrix<float> of shape (sequence_length, embedding_dim).
     *                         The sequence_length must be less than or equal to max_seq_len
     *                         specified in the constructor.
     * @return Matrix::Matrix<float> The input embeddings with positional encodings added.
     *                               Shape: (sequence_length, embedding_dim).
     * @throws std::invalid_argument if input_embeddings.cols() does not match embedding_dim_
     *         or if input_embeddings.rows() exceeds max_seq_len_.
     */
    Matrix::Matrix<float> forward(const Matrix::Matrix<float>& input_embeddings);

    /**
     * @brief Returns the pre-calculated positional encoding table.
     * @return const Matrix::Matrix<float>& The PE table of shape (max_seq_len, embedding_dim).
     */
    const Matrix::Matrix<float>& get_pe_table() const;

private:
    int max_seq_len_;
    int embedding_dim_;
    Matrix::Matrix<float> pe_table_; // Stores the pre-calculated positional encodings
};

} // namespace Transformer
} // namespace NeuroNet
