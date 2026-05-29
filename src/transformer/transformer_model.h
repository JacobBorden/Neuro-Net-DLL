#pragma once

#include "embedding.h"
#include "positional_encoding.h"
#include "transformer_encoder_layer.h"
#include "../math/matrix.h"
#include "../math/extended_matrix_ops.h" // For MathUtils::layer_norm
#include "../utilities/vocabulary.h"     // For NeuroNet::Vocabulary (optional, if model manages vocab loading)
#include "../utilities/json/json.hpp"     // For JsonValue, JsonParser
#include "../utilities/json/json_exception.hpp" // For JsonParseException
#include <fstream> // For file operations
#include <vector>
#include <stdexcept>
#include <string> // For future serialization method signatures

namespace NeuroNet {
namespace Transformer {

class TransformerModel {
public:
    /**
     * @brief Constructor for the TransformerModel (Encoder-Only).
     * @param vocab_size Size of the vocabulary for the embedding layer.
     * @param max_seq_len Maximum sequence length for positional encoding.
     * @param d_model Dimensionality of embeddings and model layers.
     * @param num_encoder_layers Number of TransformerEncoderLayer to stack.
     * @param num_heads Number of attention heads in each encoder layer.
     * @param d_ff Dimensionality of the feed-forward network within each encoder layer.
     * @param MHA_dropout_rate Dropout rate for MultiHeadAttention in encoder layers (placeholder).
     * @param FFN_dropout_rate Dropout rate for TransformerFFN in encoder layers (placeholder).
     * @param layer_norm_epsilon Epsilon for LayerNormalization.
     */
    TransformerModel(
        int vocab_size,
        int max_seq_len,
        int d_model,
        int num_encoder_layers,
        int num_heads,
        int d_ff,
        float MHA_dropout_rate = 0.0f,
        float FFN_dropout_rate = 0.0f,
        float layer_norm_epsilon = 1e-5f
    );

    /**
     * @brief Performs the forward pass of the Transformer model.
     * @param input_token_ids Matrix of token IDs, shape (1, seq_len).
     *                        seq_len must be <= max_seq_len.
     * @param attention_mask Optional mask for self-attention in encoder layers,
     *                       shape (seq_len, seq_len) or (1, seq_len) for some types.
     *                       For self-attention, typically (seq_len, seq_len).
     * @return Matrix::Matrix<float> Output matrix from the final encoder layer,
     *                               after final layer normalization. Shape (seq_len, d_model).
     * @throws std::invalid_argument for dimension mismatches or invalid inputs.
     */
    Matrix::Matrix<float> forward(
        const Matrix::Matrix<float>& input_token_ids,
        const Matrix::Matrix<float>& attention_mask = Matrix::Matrix<float>(0,0)
    );

    // --- Accessors for sub-modules (for inspection, serialization, fine-tuning) ---
    EmbeddingLayer& get_embedding_layer() { return embedding_layer_; }
    const EmbeddingLayer& get_embedding_layer() const { return embedding_layer_; }

    PositionalEncoding& get_positional_encoding_module() { return positional_encoding_; }
    const PositionalEncoding& get_positional_encoding_module() const { return positional_encoding_; }

    std::vector<TransformerEncoderLayer>& get_encoder_layers() { return encoder_layers_; }
    const std::vector<TransformerEncoderLayer>& get_encoder_layers() const { return encoder_layers_; }

    // --- Model Parameters ---
    int get_vocab_size() const { return vocab_size_; }
    int get_max_seq_len() const { return max_seq_len_; }
    int get_d_model() const { return d_model_; }
    int get_num_encoder_layers() const { return num_encoder_layers_; }
    int get_num_heads() const { return num_heads_; }
    int get_d_ff() const { return d_ff_; }
    float get_layer_norm_epsilon() const { return layer_norm_epsilon_; }


    // --- Serialization (to be implemented later) ---
    // bool save_model(const std::string& filename) const;
    // static TransformerModel load_model(const std::string& filename);
    // std::string to_json_string() const; // For custom JSON

    // --- Serialization ---
    /**
     * @brief Saves the TransformerModel's architecture and weights to a JSON file.
     * @param filename The path to the file where the model will be saved.
     * @return True if saving was successful, false otherwise.
     */
    bool save_model(const std::string& filename) const;

    /**
     * @brief Loads a TransformerModel from a JSON file.
     * @param filename The path to the file from which the model will be loaded.
     * @return A TransformerModel object populated with the loaded data.
     * @throws std::runtime_error if loading fails (e.g., file not found, JSON parsing error, invalid format).
     */
    static TransformerModel load_model(const std::string& filename);

private:
    int vocab_size_;
    int max_seq_len_;
    int d_model_;
    int num_encoder_layers_;
    int num_heads_;
    int d_ff_;
    float MHA_dropout_rate_; // Stored, but dropout not fully implemented in sub-modules
    float FFN_dropout_rate_; // Stored
    float layer_norm_epsilon_;

    EmbeddingLayer embedding_layer_;
    PositionalEncoding positional_encoding_;
    std::vector<TransformerEncoderLayer> encoder_layers_;
    // Final LayerNorm is often applied outside the loop of encoders
};

} // namespace Transformer
} // namespace NeuroNet
