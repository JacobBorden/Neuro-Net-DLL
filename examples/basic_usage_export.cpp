#include <iostream>
#include <vector>
#include <cmath>
#include <functional>
#include <fstream> // Required for std::ofstream
#include <iomanip> // Required for std::put_time (though not directly used here, good for completeness)

#include "neural_network/neuronet.h"
#include "optimization/genetic_algorithm.h"
#include "math/matrix.h"
#include <string> // For std::to_string, std::string
#include <cstdio> // For std::remove
// #include "src/utilities/json/json.hpp" // For nlohmann::json - no longer directly used for model export here, but training_metrics.json uses it.
                                       // NeuroNet.h includes it for its own to_nlohmann_json, which is not used here.
                                       // TrainingRunMetrics uses nlohmann::json so it will be pulled by genetic_algorithm.h

// Fitness function for XOR
// Inputs: {0,0}, {0,1}, {1,0}, {1,1}
// Outputs: {{0}, {1}, {1}, {0}}
std::function<double(NeuroNet::NeuroNet&)> xor_fitness_function = 
    [](NeuroNet::NeuroNet& nn) -> double {
    
    std::vector<std::vector<float>> inputs = {{0.0f, 0.0f}, {0.0f, 1.0f}, {1.0f, 0.0f}, {1.0f, 1.0f}};
    std::vector<std::vector<float>> target_outputs = {{0.0f}, {1.0f}, {1.0f}, {0.0f}};
    double total_error = 0.0;

    if (nn.getLayerCount() == 0) {
        // Return worst fitness if network is not configured
        return 0.0; 
    }

    for (size_t i = 0; i < inputs.size(); ++i) {
        Matrix::Matrix<float> input_matrix(1, inputs[i].size());
        for(size_t j = 0; j < inputs[i].size(); ++j) {
            input_matrix[0][j] = inputs[i][j];
        }

        Matrix::Matrix<float> target_output_matrix(1, target_outputs[i].size());
        for(size_t j = 0; j < target_outputs[i].size(); ++j) {
            target_output_matrix[0][j] = target_outputs[i][j];
        }

        nn.SetInput(input_matrix);
        Matrix::Matrix<float> output_matrix = nn.GetOutput();

        if (output_matrix.rows() != 1 || output_matrix.cols() != target_outputs[i].size()) {
            // Output dimension mismatch, penalize heavily
            // This might happen if network structure is wrong
            total_error += 100.0 * inputs.size(); // High penalty
            continue; 
        }

        for (size_t j = 0; j < target_outputs[i].size(); ++j) {
            float error_val = output_matrix[0][j] - target_output_matrix[0][j];
            total_error += error_val * error_val;
        }
    }

    return 1.0 / (1.0 + total_error);
};


int main() {
    std::cout << "Starting NeuroNet Genetic Algorithm XOR Example..." << std::endl;

    // 1. Define NeuroNet Structure
    NeuroNet::NeuroNet template_network;
    template_network.SetInputSize(2); // 2 input features for XOR

    // Add layers:
    // Layer 0: Hidden Layer
    template_network.ResizeNeuroNet(2); // 1 hidden, 1 output layer initially
    template_network.ResizeLayer(0, 3); // Hidden layer with 3 neurons
    template_network.getLayer(0).SetActivationFunction(NeuroNet::ActivationFunctionType::ReLU);

    // Layer 1: Output Layer
    // Input size for layer 1 is automatically set by ResizeLayer(0,...) if layer 1 already exists,
    // or by the output of layer 0.
    // We need to ensure layer 1 is configured correctly for 1 output neuron.
    template_network.ResizeLayer(1, 1); // Output layer with 1 neuron
    template_network.getLayer(1).SetActivationFunction(NeuroNet::ActivationFunctionType::None); // Or Sigmoid for 0-1 range

    std::cout << "Template network configured." << std::endl;
    std::cout << "  Input Size: " << template_network.getLayer(0).WeightCount() / template_network.getLayer(0).LayerSize() << std::endl; // A bit indirect
    std::cout << "  Layer 0 (Hidden): " << template_network.getLayer(0).LayerSize() << " neurons, Activation: " << template_network.getLayer(0).get_activation_function_name() << std::endl;
    std::cout << "  Layer 1 (Output): " << template_network.getLayer(1).LayerSize() << " neurons, Activation: " << template_network.getLayer(1).get_activation_function_name() << std::endl;


    // 2. Instantiate Genetic Algorithm
    const int population_size = 20; // Small for quick demo
    const double mutation_rate = 0.1;
    const double crossover_rate = 0.7;
    const int num_generations = 50; // Small for quick demo

    Optimization::GeneticAlgorithm ga_instance(
        population_size,
        mutation_rate,
        crossover_rate,
        num_generations,
        template_network
    );
    std::cout << "Genetic Algorithm instance created." << std::endl;

    // 3. Run Evolution
    std::cout << "Running evolution for " << num_generations << " generations..." << std::endl;
    ga_instance.run_evolution(xor_fitness_function);
    std::cout << "Evolution finished." << std::endl;

    // 4. Export Training Metrics
    const std::string metrics_filename = "training_metrics.json";
    ga_instance.export_training_metrics_json(metrics_filename);
    std::cout << "Training metrics exported to: " << metrics_filename << std::endl;

    // 5. Get Best Model
    NeuroNet::NeuroNet best_model = ga_instance.get_best_individual();
    std::cout << "Best individual retrieved from GA." << std::endl;

    if (best_model.getLayerCount() > 0) {
        // 6. Save Best Model (using the library's custom JSON format)
        // This format is primarily intended for use with NeuroNet::load_model().
        const std::string custom_model_filename = "best_model_custom_format.json";
        best_model.save_model(custom_model_filename); 
        std::cout << "Best model saved in custom JSON format to: " << custom_model_filename << std::endl;

        // Note: The training_metrics.json file (exported earlier) contains the
        // best model's architecture as a JSON string (using the custom format)
        // embedded within its structure. For external analysis, that embedded string
        // or this dedicated custom format file can be used.

        // Optional: Test the best model
        std::cout << "\nTesting the best model on XOR inputs:" << std::endl;
        std::vector<std::vector<float>> test_inputs = {{0.0f, 0.0f}, {0.0f, 1.0f}, {1.0f, 0.0f}, {1.0f, 1.0f}};
        std::vector<float> expected_outputs = {0.0f, 1.0f, 1.0f, 0.0f};

        for (size_t i = 0; i < test_inputs.size(); ++i) {
            Matrix::Matrix<float> input_matrix(1, test_inputs[i].size());
            for(size_t j=0; j<test_inputs[i].size(); ++j) input_matrix[0][j] = test_inputs[i][j];
            
            best_model.SetInput(input_matrix);
            Matrix::Matrix<float> output = best_model.GetOutput();
            std::cout << "Input: (" << test_inputs[i][0] << ", " << test_inputs[i][1] << ") -> Output: ";
            if (output.cols() > 0) {
                 std::cout << output[0][0] << " (Expected: " << expected_outputs[i] << ")" << std::endl;
            } else {
                 std::cout << "[empty output]" << std::endl;
            }
        }

        // Test with JSON input/output
        std::cout << "\nTesting the best model using JSON input/output:" << std::endl;
        for (size_t i = 0; i < test_inputs.size(); ++i) {
            // Construct JSON input string
            std::string json_input_str = "{ \"input_matrix\": [[";
            json_input_str += std::to_string(test_inputs[i][0]);
            json_input_str += ", ";
            json_input_str += std::to_string(test_inputs[i][1]);
            json_input_str += "]] }";

            std::cout << "Input JSON: " << json_input_str << std::endl;

            try {
                if (!best_model.SetInputJSON(json_input_str)) {
                    std::cerr << "Call to SetInputJSON failed for: " << json_input_str << std::endl;
                    continue;
                }

                std::string output_json_str = best_model.GetOutputJSON();
                std::cout << "Output JSON: " << output_json_str << std::endl;

                // Parse output_json_str to verify the numeric value
                JsonValue output_json_val = JsonParser::Parse(output_json_str);
                if (output_json_val.type == JsonValueType::Object &&
                    output_json_val.GetObject().count("output_matrix")) {
                    const auto& matrix_val = output_json_val.GetObject().at("output_matrix");
                    if (matrix_val->type == JsonValueType::Array && !matrix_val->GetArray().empty()) {
                        const auto& row_val = matrix_val->GetArray()[0];
                        if (row_val.type == JsonValueType::Array && !row_val.GetArray().empty()) {
                            const auto& cell_val = row_val.GetArray()[0];
                            if (cell_val.type == JsonValueType::Number) {
                                std::cout << "Parsed Output Value: " << cell_val.GetNumber()
                                          << " (Expected: " << expected_outputs[i] << ")" << std::endl;
                            }
                        }
                    }
                }
                // Cleanup parsed JSON (important for the custom JSON library)
                if (output_json_val.type == JsonValueType::Object) {
                    for (auto& pair : output_json_val.GetObject()) {
                        delete pair.second; // Delete the JsonValue*
                    }
                    output_json_val.GetObject().clear(); // Clear the map
                }

            } catch (const JsonParseException& e) {
                std::cerr << "Error with JSON processing: " << e.what() << std::endl;
            } catch (const std::runtime_error& e) {
                std::cerr << "Runtime error during JSON processing: " << e.what() << std::endl;
            }
            std::cout << "----" << std::endl;
        }

        std::cout << "\n--- Demonstrating String Input and Vocabulary Features ---" << std::endl;

        // 1. Define and create a sample vocabulary JSON file for the example
        const std::string example_vocab_filepath = "example_vocabulary.json";
        std::ofstream vocab_file(example_vocab_filepath);
        if (vocab_file.is_open()) {
            vocab_file << R"({
                "word_to_token": {
                    "hello": 0, "world": 1, "neuronet": 2, "example": 3,
                    "<UNK>": 4, "<PAD>": 5
                },
                "token_to_word": {
                    "0": "hello", "1": "world", "2": "neuronet", "3": "example",
                    "4": "<UNK>", "5": "<PAD>"
                },
                "special_tokens": {
                    "unknown_token": "<UNK>",
                    "padding_token": "<PAD>"
                },
                "config": {
                    "max_sequence_length": 5
                }
            })";
            vocab_file.close();
            std::cout << "Sample vocabulary file created: " << example_vocab_filepath << std::endl;
        } else {
            std::cerr << "Failed to create sample vocabulary file for example." << std::endl;
        }

        // We'll use the 'best_model' from the GA, but reconfigure its InputSize
        // and load the new vocabulary for demonstration purposes.
        // Its existing weights won't be meaningful for string inputs with this new vocab.
        // Note: best_model is a copy of the GA's best individual. Modifying it here is fine for demo.

        // 2. Load the vocabulary
        bool vocab_loaded = false;
        try {
            vocab_loaded = best_model.LoadVocabulary(example_vocab_filepath);
        } catch (const std::exception& e) {
            std::cerr << "Exception during LoadVocabulary: " << e.what() << std::endl;
        }

        if (vocab_loaded) {
            std::cout << "Vocabulary loaded successfully." << std::endl;
            const auto& loaded_vocab_obj = best_model.getVocabulary(); // Use the new getter
            std::cout << "  Vocabulary's max_sequence_length: " << loaded_vocab_obj.get_max_sequence_length() << std::endl;
            std::cout << "  <UNK> token ID: " << loaded_vocab_obj.get_unknown_token_id() << std::endl;
            std::cout << "  <PAD> token ID: " << loaded_vocab_obj.get_padding_token_id() << std::endl;


            // 3. Reconfigure network's InputSize to match vocabulary's max_sequence_length for this demo
            int required_input_size = loaded_vocab_obj.get_max_sequence_length();
            if (required_input_size <= 0) {
                 std::cout << "Vocabulary max_sequence_length is not positive, defaulting to 5 for demo." << std::endl;
                 required_input_size = 5; // Fallback for demo if not in vocab file
            }

            std::cout << "Setting network InputSize to: " << required_input_size << " to match vocab's max_sequence_length for demo." << std::endl;
            best_model.SetInputSize(required_input_size);

            // Resizing layers to be compatible with the new InputSize.
            // For simplicity, let's assume the first layer's number of neurons can remain the same.
            // The internal weight matrix of the first layer will be re-initialized by ResizeLayer.
            if (best_model.getLayerCount() > 0) {
                 std::cout << "Re-initializing first layer with new InputSize." << std::endl;
                 best_model.ResizeLayer(0, best_model.getLayer(0).LayerSize());
            }


            // 4. Prepare and set string input using SetStringsInput
            const std::string string_input_json = R"({
                "input_batch": [
                    "hello world neuronet",   // 3 tokens
                    "an unknown example",     // 3 tokens, "an" will be <UNK>
                    "hello world again test"  // 4 tokens, "again", "test" are <UNK>
                ]
            })";
            std::cout << "Attempting to set string input: " << string_input_json << std::endl;

            try {
                // SetStringsInput will use the max_sequence_length from the loaded vocabulary (5)
                // So, sequences will be padded/truncated to 5 tokens.
                if (best_model.SetStringsInput(string_input_json)) {
                    std::cout << "SetStringsInput successful." << std::endl;

                    // 5. Get and display output (as JSON, for consistency with other examples)
                    // The actual numerical output will be based on the (probably unsuitable) XOR weights
                    // or re-initialized weights, but this demonstrates the pipeline.
                    std::string string_net_output_json = best_model.GetOutputJSON();
                    std::cout << "Output from network after string input (JSON): " << string_net_output_json << std::endl;
                } else {
                    std::cerr << "SetStringsInput failed." << std::endl;
                }
            } catch (const std::exception& e) {
                std::cerr << "Exception during SetStringsInput or GetOutputJSON: " << e.what() << std::endl;
            }
        } else {
            std::cerr << "Failed to load vocabulary for string input demo." << std::endl;
        }
        // Clean up the example vocabulary file
        std::remove(example_vocab_filepath.c_str());
        std::cout << "Cleaned up temporary vocabulary file: " << example_vocab_filepath << std::endl;

    } else {
        std::cout << "Best model has no layers, cannot save or test." << std::endl;
    }

    std::cout << "\nExample finished." << std::endl;
    return 0;
}
