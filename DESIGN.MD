# NeuroNet Design Document

## Architecture Overview

NeuroNet is a C++ library designed for building and training neural networks and related machine learning models. The core philosophy is to provide a clean, customizable, and dependency-free (other than standard library) framework from the ground up, avoiding external math libraries like BLAS or Eigen where possible in favor of an internal implementation.

The primary modules of the library are:

1.  **Math Module (`src/math/`)**
2.  **Neural Network Module (`src/neural_network/`)**
3.  **Optimization Module (`src/optimization/`)**
4.  **Transformer Module (`src/transformer/`)**
5.  **Utilities Module (`src/utilities/`)**

---

### 1. Math Module
The math module forms the computational backbone.
*   **`Matrix<T>`:** A generic template class for 2D matrix operations. It handles matrix allocation, arithmetic (`+`, `-`, `*`), and common transformations (transpose). Matrix multiplication is optimized using OpenMP when available.
*   **Extended Operations:** Found in `extended_matrix_ops.h`, this provides operations specific to deep learning such as Layer Normalization, Softmax, and GELU activation.

### 2. Neural Network Module
This module implements feed-forward neural networks.
*   **`NeuroNetLayer`:** Represents a single layer (fully connected). It stores weights, biases, and an activation function type (ReLU, Softmax, etc.). It calculates the output as `Activation(Input * Weights + Biases)`.
*   **`NeuroNet`:** Manages the entire network. It consists of a sequence of `NeuroNetLayer` objects. It is responsible for the overall forward pass, moving output from one layer as input to the next. It also handles model serialization to custom JSON formats.

### 3. Optimization Module
Since the library doesn't strictly depend on backpropagation by default, it includes genetic algorithms for training.
*   **`GeneticAlgorithm`:** Trains a `NeuroNet` by evolving a population of network configurations. It uses fitness functions, tournament selection, crossover, and mutation operators to search the parameter space (weights and biases).

### 4. Transformer Module
A more recent addition to support modern NLP architectures, specifically an Encoder-only Transformer model.
*   **Components:** `EmbeddingLayer`, `PositionalEncoding`, `ScaledDotProductAttention`, `MultiHeadAttention`, and `TransformerFFN` represent the core building blocks.
*   **`TransformerEncoderLayer`:** Combines Multi-Head Attention and the FFN with residual connections and layer normalization.
*   **`TransformerModel`:** The top-level class that aggregates embedding, positional encoding, and multiple encoder layers into a complete runnable model.

### 5. Utilities
*   **JSON (`src/utilities/json/`):** A custom, lightweight JSON parser used heavily for model saving and loading (`NeuroNet::save_model`, `TransformerModel::save_model`).
*   **Vocabulary (`src/utilities/vocabulary.h`):** Handles tokenization mapping for NLP tasks.

## Design Patterns & Decisions

*   **Header-only Math:** `matrix.h` is designed to be easily included and instantiated for different types (primarily `float`).
*   **Custom JSON Parser:** To keep the library self-contained without needing things like `nlohmann/json`, a custom parser and AST (`JsonValue`) were written.
*   **Serialization Strategy:** Both `NeuroNet` and `TransformerModel` implement methods to serialize their architecture parameters and flattened weights/biases into `JsonValue` trees, which are then written to strings/files.
