#include "positional_encoding.h"
#include <stdexcept> // For std::invalid_argument

namespace NeuroNet {
namespace Transformer {

PositionalEncoding::PositionalEncoding(int max_seq_len, int embedding_dim)
    : max_seq_len_(max_seq_len), embedding_dim_(embedding_dim) {
    if (max_seq_len <= 0 || embedding_dim <= 0) {
        throw std::invalid_argument("Max sequence length and embedding dimension must be positive.");
    }

    pe_table_.resize(max_seq_len_, embedding_dim_);
    pe_table_.assign(0.0f); // Initialize with zeros

    for (int pos = 0; pos < max_seq_len_; ++pos) {
        for (int i = 0; i < embedding_dim_; ++i) {
            float angle_denominator = std::pow(10000.0f, static_cast<float>(2 * (i / 2)) / static_cast<float>(embedding_dim_));
            float angle = static_cast<float>(pos) / angle_denominator;
            if (i % 2 == 0) { // Even index: sin
                pe_table_[pos][i] = std::sin(angle);
            } else { // Odd index: cos
                pe_table_[pos][i] = std::cos(angle);
            }
        }
    }
}

Matrix::Matrix<float> PositionalEncoding::forward(const Matrix::Matrix<float>& input_embeddings) {
    size_t seq_len = input_embeddings.rows();
    size_t emb_dim = input_embeddings.cols();

    if (emb_dim != static_cast<size_t>(embedding_dim_)) {
        throw std::invalid_argument("Input embedding dimension (" + std::to_string(emb_dim) +
                                    ") does not match PositionalEncoding's embedding_dim (" +
                                    std::to_string(embedding_dim_) + ").");
    }
    if (seq_len > static_cast<size_t>(max_seq_len_)) {
        throw std::invalid_argument("Input sequence length (" + std::to_string(seq_len) +
                                    ") exceeds PositionalEncoding's max_seq_len (" +
                                    std::to_string(max_seq_len_) + ").");
    }

    if (seq_len == 0) { // Handle empty sequence input
        return Matrix::Matrix<float>(0, embedding_dim_);
    }

    // Create a slice of pe_table_ matching the input sequence length
    Matrix::Matrix<float> relevant_pe(seq_len, embedding_dim_);
    for(size_t i = 0; i < seq_len; ++i) {
        for(size_t j = 0; j < emb_dim; ++j) {
            relevant_pe[i][j] = pe_table_[i][j];
        }
    }

    // Add positional encodings to input embeddings
    // Assumes Matrix class supports element-wise addition via operator+
    Matrix::Matrix<float> output = input_embeddings + relevant_pe;
    return output;
}

const Matrix::Matrix<float>& PositionalEncoding::get_pe_table() const {
    return pe_table_;
}

} // namespace Transformer
} // namespace NeuroNet
