#include <iostream>
#include <vector>
#include <fstream> // For std::ofstream, std::ifstream
#include <string>
#include <iomanip> // For std::fixed, std::setprecision (for printing floats)
#include <cstdio>  // For std::remove (to clean up temp files)

#include "transformer/transformer_model.h" // Adjust path as needed
#include "utilities/vocabulary.h"      // Adjust path as needed
#include "math/matrix.h"               // Adjust path as needed

// Helper to print a matrix (subset for brevity)
void print_matrix_summary(const Matrix::Matrix<float>& m, const std::string& title) {
    std::cout << title << " (Shape: " << m.rows() << "x" << m.cols() << "):" << std::endl;
    if (m.rows() == 0 || m.cols() == 0) {
        std::cout << "  [Empty Matrix]" << std::endl;
        return;
    }
    for (size_t i = 0; i < std::min((size_t)2, m.rows()); ++i) { // Print max 2 rows
        std::cout << "  Row " << i << ": [";
        for (size_t j = 0; j < std::min((size_t)5, m.cols()); ++j) { // Print max 5 cols
            std::cout << std::fixed << std::setprecision(4) << m[i][j] << (j == std::min((size_t)5, m.cols()) - 1 ? "" : ", ");
        }
        if (m.cols() > 5) std::cout << "...";
        std::cout << "]" << std::endl;
    }
    if (m.rows() > 2) std::cout << "  ..." << std::endl;
}

// Helper to create a dummy vocabulary JSON file for the example
bool create_dummy_vocab_file(const std::string& filepath, int vocab_size, int& pad_id, int& unk_id) {
    pad_id = vocab_size - 1; // Assign last ID to PAD
    unk_id = vocab_size - 2; // Assign second to last ID to UNK

    std::ofstream vocab_file(filepath);
    if (!vocab_file.is_open()) {
        std::cerr << "ERROR: Failed to create dummy vocabulary file at " << filepath << std::endl;
        return false;
    }
    vocab_file << "{
";
    vocab_file << "  \"word_to_token\": {
";
    for (int i = 0; i < vocab_size - 2; ++i) {
        vocab_file << "    \"token" << i << "\": " << i << (i == vocab_size - 3 ? "" : ",") << "
";
    }
    vocab_file << "    \"<UNK>\": " << unk_id << ",
";
    vocab_file << "    \"<PAD>\": " << pad_id << "
";
    vocab_file << "  },
";
    vocab_file << "  \"token_to_word\": {
";
    for (int i = 0; i < vocab_size - 2; ++i) {
        vocab_file << "    \"" << i << "\": \"token" << i << "\",
";
    }
    vocab_file << "    \"" << unk_id << "\": \"<UNK>\",
";
    vocab_file << "    \"" << pad_id << "\": \"<PAD>\"
";
    vocab_file << "  },
";
    vocab_file << "  \"special_tokens\": {
";
    vocab_file << "    \"unknown_token\": \"<UNK>\",
";
    vocab_file << "    \"padding_token\": \"<PAD>\"
";
    vocab_file << "  },
";
    vocab_file << "  \"config\": {
";
    vocab_file << "    \"max_sequence_length\": 10
"; // Default max_seq_len for vocab
    vocab_file << "  }
";
    vocab_file << "}
";
    vocab_file.close();
    std::cout << "Dummy vocabulary file created: " << filepath << std::endl;
    return true;
}


int main() {
    std::cout << "--- Transformer Model Usage Example ---" << std::endl;

    // --- 1. Model Hyperparameters & Instantiation ---
    const int vocab_size_param = 50; // Example vocab size
    const int max_seq_len_param = 10; // Max sequence length the model can handle
    const int d_model_param = 32;    // Embedding dimension, model dimension
    const int num_encoder_layers_param = 2;
    const int num_heads_param = 4;     // d_model must be divisible by num_heads (32/4=8)
    const int d_ff_param = 64;       // Feed-forward inner dimension
    const std::string vocab_filepath = "example_transformer_vocab.json";
    const std::string model_save_filepath = "example_transformer_model.json";

    int pad_token_id = -1, unk_token_id = -1;
    if (!create_dummy_vocab_file(vocab_filepath, vocab_size_param, pad_token_id, unk_token_id)) {
        return 1;
    }

    NeuroNet::Transformer::TransformerModel model(
        vocab_size_param, max_seq_len_param, d_model_param,
        num_encoder_layers_param, num_heads_param, d_ff_param
    );
    std::cout << "
1. TransformerModel instantiated." << std::endl;
    std::cout << "   Vocab Size: " << model.get_vocab_size() << std::endl;
    std::cout << "   Max Seq Len: " << model.get_max_seq_len() << std::endl;
    std::cout << "   D_Model: " << model.get_d_model() << std::endl;
    std::cout << "   Encoder Layers: " << model.get_num_encoder_layers() << std::endl;
    std::cout << "   Heads: " << model.get_num_heads() << std::endl;
    std::cout << "   D_FF: " << model.get_d_ff() << std::endl;

    // --- 2. Vocabulary Loading ---
    NeuroNet::Vocabulary vocab;
    if (!vocab.load_from_json(vocab_filepath)) {
        std::cerr << "ERROR: Failed to load vocabulary from " << vocab_filepath << std::endl;
        std::remove(vocab_filepath.c_str()); // Clean up
        return 1;
    }
    std::cout << "
2. Vocabulary loaded from " << vocab_filepath << "." << std::endl;
    std::cout << "   Vocab max_seq_len (from file): " << vocab.get_max_sequence_length() << std::endl;
    std::cout << "   Padding token ID: " << vocab.get_padding_token_id() << std::endl;

    // --- 3. String Input Processing ---
    std::vector<std::string> text_batch = {
        "hello world token0 token1", // 4 tokens
        "token2 token3 unknownword"  // 3 tokens, "unknownword" -> <UNK>
    };
    std::cout << "
3. Processing string input batch:" << std::endl;
    for(const auto&s : text_batch) std::cout << "   \"" << s << "\"" << std::endl;

    // `prepare_batch_matrix` pads/truncates to `max_len`.
    // If max_len=-1, it uses vocab's internal max_seq_len (10 here) or pads to max in batch.
    // Let's use the vocab's max_seq_len.
    Matrix::Matrix<float> token_id_batch_matrix = vocab.prepare_batch_matrix(text_batch, vocab.get_max_sequence_length());
    print_matrix_summary(token_id_batch_matrix, "Token ID Batch Matrix (from vocab.prepare_batch_matrix)");


    // --- 4. Forward Pass (one sequence at a time, as model.forward expects 1xN) ---
    std::cout << "
4. Performing forward pass (one sequence at a time):" << std::endl;
    if (token_id_batch_matrix.rows() > 0) {
        for (size_t i = 0; i < token_id_batch_matrix.rows(); ++i) {
            // Create a (1, seq_len) matrix for the current sequence
            Matrix::Matrix<float> single_sequence_tokens(1, token_id_batch_matrix.cols());
            for(size_t j=0; j < token_id_batch_matrix.cols(); ++j) {
                single_sequence_tokens[0][j] = token_id_batch_matrix[i][j];
            }

            std::cout << "  Forward pass for sequence " << i << ":" << std::endl;
            print_matrix_summary(single_sequence_tokens, "  Input Token IDs for sequence " + std::to_string(i));

            // Create a dummy attention mask (no masking) for this example
            // A real mask might be (seq_len, seq_len)
            Matrix::Matrix<float> dummy_attention_mask(0,0); // Empty mask = no mask in attention layer

            try {
                Matrix::Matrix<float> output_embeddings = model.forward(single_sequence_tokens, dummy_attention_mask);
                print_matrix_summary(output_embeddings, "  Output Embeddings for sequence " + std::to_string(i));
            } catch (const std::exception& e) {
                std::cerr << "  ERROR during forward pass for sequence " << i << ": " << e.what() << std::endl;
            }
        }
    }


    // --- 5. Save Model ---
    std::cout << "
5. Saving model to " << model_save_filepath << "..." << std::endl;
    if (model.save_model(model_save_filepath)) {
        std::cout << "   Model saved successfully." << std::endl;

        // --- 6. Load Model ---
        std::cout << "
6. Loading model from " << model_save_filepath << "..." << std::endl;
        try {
            NeuroNet::Transformer::TransformerModel loaded_model = NeuroNet::Transformer::TransformerModel::load_model(model_save_filepath);
            std::cout << "   Model loaded successfully." << std::endl;
            std::cout << "   Loaded Model Vocab Size: " << loaded_model.get_vocab_size() << std::endl;
            std::cout << "   Loaded Model D_Model: " << loaded_model.get_d_model() << std::endl;

            // --- Optional: Test loaded model with the first sequence ---
            if (token_id_batch_matrix.rows() > 0) {
                 Matrix::Matrix<float> first_sequence_tokens(1, token_id_batch_matrix.cols());
                 for(size_t j=0; j < token_id_batch_matrix.cols(); ++j) {
                     first_sequence_tokens[0][j] = token_id_batch_matrix[0][j];
                 }
                std::cout << "   Testing loaded model with first sequence..." << std::endl;
                Matrix::Matrix<float> loaded_model_output = loaded_model.forward(first_sequence_tokens);
                print_matrix_summary(loaded_model_output, "   Output from loaded model (first sequence)");
                // For a true test, one would compare this output to the original model's output
                // if the random initialization was seeded or if weights were deterministic.
            }

        } catch (const std::exception& e) {
            std::cerr << "   ERROR: Failed to load or test model: " << e.what() << std::endl;
        }
        std::remove(model_save_filepath.c_str()); // Clean up saved model file
        std::cout << "   Cleaned up temporary model file: " << model_save_filepath << std::endl;

    } else {
        std::cerr << "   ERROR: Failed to save model." << std::endl;
    }

    // --- Cleanup ---
    std::remove(vocab_filepath.c_str()); // Clean up dummy vocab file
    std::cout << "
Cleaned up temporary vocabulary file: " << vocab_filepath << std::endl;
    std::cout << "
--- Example Finished ---" << std::endl;
    return 0;
}
