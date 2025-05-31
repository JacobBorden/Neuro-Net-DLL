# Transformer Module

This document provides an overview of the Transformer module components, which together form an encoder-only Transformer model. This architecture is suitable for tasks like text classification, feature extraction, or language modeling where a full encoder-decoder setup is not required.

## Overview

The Transformer module implements key building blocks of the Transformer architecture, as introduced in "Attention Is All You Need" by Vaswani et al. This implementation focuses on the encoder part.

The main components include:
*   Input Embedding
*   Positional Encoding
*   Scaled Dot-Product Attention
*   Multi-Head Attention
*   Position-wise Feed-Forward Networks (FFN)
*   Transformer Encoder Layers (combining Attention and FFN)
*   The main `TransformerModel` class that stacks these layers.

## Key Components

### 1. `EmbeddingLayer`
*   **Purpose:** Converts input token IDs into dense vector representations (embeddings).
*   **Functionality:** Maintains an embedding table (matrix of size `vocab_size` x `embedding_dim`). Given a sequence of token IDs, it looks up their corresponding vectors.
*   **Key Methods:**
    *   `EmbeddingLayer(int vocab_size, int embedding_dim)`: Constructor.
    *   `forward(const Matrix::Matrix<float>& input_token_ids)`: Performs the embedding lookup.
    *   `get_weights()`, `set_weights()`: Access or modify the embedding table.
*   **Source:** `src/transformer/embedding.h`

### 2. `PositionalEncoding`
*   **Purpose:** Injects information about the position of tokens in a sequence, as standard Transformers do not inherently process sequential order otherwise.
*   **Functionality:** Uses sinusoidal functions of different frequencies to generate a unique positional encoding vector for each position up to `max_seq_len`. This encoding is added to the input token embeddings.
*   **Key Methods:**
    *   `PositionalEncoding(int max_seq_len, int embedding_dim)`: Constructor, pre-calculates the encoding table.
    *   `forward(const Matrix::Matrix<float>& input_embeddings)`: Adds positional encodings to the embeddings.
*   **Source:** `src/transformer/positional_encoding.h`

### 3. `ScaledDotProductAttention`
*   **Purpose:** The core attention mechanism that allows the model to weigh the importance of different tokens when processing a given token.
*   **Functionality:** Computes attention scores as: `softmax((Q * K^T) / sqrt(d_k) + mask) * V`.
    *   `Q` (Query), `K` (Key), `V` (Value) are input matrices.
    *   `d_k` is the dimension of the key vectors.
    *   `mask` is an optional matrix to prevent attention to certain positions (e.g., padding tokens or future tokens in self-attention).
*   **Key Methods:**
    *   `ScaledDotProductAttention(float dropout_rate)`: Constructor (dropout currently a placeholder).
    *   `forward(query, key, value, mask)`: Performs the attention calculation, returning an `AttentionOutput` struct containing the context vector and attention weights.
*   **Source:** `src/transformer/attention.h`

### 4. `MultiHeadAttention`
*   **Purpose:** Extends Scaled Dot-Product Attention by running it multiple times in parallel ("heads") with different, learned linear projections of Q, K, and V. This allows the model to jointly attend to information from different representation subspaces at different positions.
*   **Functionality:**
    1.  Linearly project Q, K, V `num_heads` times with different weight matrices (`Wq_`, `Wk_`, `Wv_`).
    2.  Pass each projected set to `ScaledDotProductAttention`.
    3.  Concatenate the outputs of all heads.
    4.  Apply a final linear projection (`Wo_`).
*   **Key Methods:**
    *   `MultiHeadAttention(int num_heads, int d_model, float dropout_rate)`: Constructor. `d_model` must be divisible by `num_heads`.
    *   `forward(query_input, key_input, value_input, mask)`: Performs multi-head attention.
    *   Accessors for weight matrices (`get_wq`, `set_wq`, etc.).
*   **Source:** `src/transformer/multi_head_attention.h`

### 5. `TransformerFFN` (Position-wise Feed-Forward Network)
*   **Purpose:** A fully connected feed-forward network applied independently to each position in the sequence after the attention mechanism.
*   **Functionality:** Consists of two linear transformations with a GELU activation in between: `GELU(input * W1 + b1) * W2 + b2`.
*   **Key Methods:**
    *   `TransformerFFN(int d_model, int d_ff, float dropout_rate)`: Constructor. `d_ff` is the dimensionality of the inner layer.
    *   `forward(const Matrix::Matrix<float>& input)`: Applies the FFN transformations.
    *   Accessors for weight and bias matrices.
*   **Source:** `src/transformer/transformer_ffn.h`

### 6. `TransformerEncoderLayer`
*   **Purpose:** A single layer of the Transformer encoder stack.
*   **Functionality:** Comprises two main sub-layers:
    1.  A multi-head self-attention mechanism.
    2.  A position-wise feed-forward network.
    *   Residual connections are used around each of the two sub-layers, followed by layer normalization (`Add & Norm`).
*   **Key Methods:**
    *   `TransformerEncoderLayer(int d_model, int num_heads, int d_ff, ...)`: Constructor.
    *   `forward(const Matrix::Matrix<float>& input, const Matrix::Matrix<float>& attention_mask)`: Processes input through the layer.
    *   Accessors for internal `MultiHeadAttention` and `TransformerFFN` modules.
*   **Source:** `src/transformer/transformer_encoder_layer.h`

### 7. `TransformerModel`
*   **Purpose:** The main class that assembles the complete encoder-only Transformer model.
*   **Functionality:**
    1.  Applies input embedding (`EmbeddingLayer`).
    2.  Adds positional encodings (`PositionalEncoding`).
    3.  Passes the sequence through a stack of `N` `TransformerEncoderLayer`s.
    4.  Applies a final layer normalization (not explicitly shown in `TransformerModel`'s forward but typically done after the loop or as part of the last layer).
*   **Key Methods:**
    *   `TransformerModel(int vocab_size, int max_seq_len, int d_model, int num_encoder_layers, ...)`: Constructor.
    *   `forward(const Matrix::Matrix<float>& input_token_ids, const Matrix::Matrix<float>& attention_mask)`: Full forward pass of the model.
    *   Accessors for sub-modules.
    *   `save_model(const std::string& filename) const`, `static TransformerModel load_model(const std::string& filename)`: For model persistence.
*   **Source:** `src/transformer/transformer_model.h`

## Basic Usage Example: Initializing and Using the Model

```cpp
#include "transformer/transformer_model.h" // Main model header
#include "math/matrix.h" // For Matrix::Matrix
#include <iostream>
#include <vector>

int main() {
    // Model parameters
    int vocab_size = 1000;       // Example vocabulary size
    int max_seq_len = 50;        // Max sequence length the model can handle
    int d_model = 128;           // Embedding dimension, model dimension
    int num_encoder_layers = 3;  // Number of encoder layers
    int num_heads = 4;           // Number of attention heads
    int d_ff = 256;              // Dimension of FFN inner layer

    // Create the Transformer model
    NeuroNet::Transformer::TransformerModel model(
        vocab_size,
        max_seq_len,
        d_model,
        num_encoder_layers,
        num_heads,
        d_ff
    );

    std::cout << "TransformerModel created." << std::endl;
    std::cout << "Vocab Size: " << model.get_vocab_size() << ", Max Seq Len: " << model.get_max_seq_len() << std::endl;
    std::cout << "d_model: " << model.get_d_model() << ", Num Layers: " << model.get_num_encoder_layers() << std::endl;

    // Example input: a sequence of token IDs (batch size 1, sequence length 10)
    // Values should be < vocab_size
    Matrix::Matrix<float> input_ids(1, 10);
    for (int i = 0; i < 10; ++i) {
        input_ids[0][i] = static_cast<float>((i * 10) % vocab_size); // Dummy token IDs
    }
    std::cout << "Input token IDs (1x10):" << std::endl;
    input_ids.print();

    // Optional: Create an attention mask (e.g., to ignore padding)
    // For this example, no mask is used, so all tokens attend to all others.
    Matrix::Matrix<float> attention_mask(0,0); // Empty mask

    try {
        // Perform a forward pass
        Matrix::Matrix<float> output = model.forward(input_ids, attention_mask);
        std::cout << "Output matrix shape: (" << output.rows() << "x" << output.cols() << ")" << std::endl;
        // The output would be of shape (seq_len, d_model), so (10, 128) in this case.
        // output.print(); // Potentially large output

    } catch (const std::exception& e) {
        std::cerr << "Error during forward pass: " << e.what() << std::endl;
        return 1;
    }

    // Model can be saved and loaded
    // model.save_model("transformer_model.json");
    // NeuroNet::Transformer::TransformerModel loaded_model = NeuroNet::Transformer::TransformerModel::load_model("transformer_model.json");

    return 0;
}

```
This example demonstrates basic initialization and a forward pass. For actual use, the model's weights (embeddings, projection matrices in attention, FFN layers) would need to be trained on a specific task.

(Further details on specific layer configurations, masking strategies, or training considerations can be added.)
