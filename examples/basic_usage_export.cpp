#include <iostream>
#include <vector>
#include <cmath>
#include <functional>
#include <fstream> // Required for std::ofstream
#include <iomanip> // Required for std::put_time (though not directly used here, good for completeness)

#include "neural_network/neuronet.h"
#include "optimization/genetic_algorithm.h"
#include "math/matrix.h"
#include "optimization/neural_pathfinder.h" // For NeuralPathfinder
#include <string> // For std::to_string, std::string
#include <cstdio> // For std::remove
// #include "src/utilities/json/json.hpp" // For nlohmann::json - no longer directly used for model export here, but training_metrics.json uses it.
                                       // NeuroNet.h includes it for its own to_nlohmann_json, which is not used here.
                                       // TrainingRunMetrics uses nlohmann::json so it will be pulled by genetic_algorithm.h


// Helper function to print a matrix (already exists if merged correctly, but ensure it's available)
void print_matrix_astar(const Matrix::Matrix<float>& mat, const std::string& title) {
    std::cout << title << " (" << mat.rows() << "x" << mat.cols() << "):\n";
    if (mat.rows() == 0 || mat.cols() == 0) {
        std::cout << "  [Empty Matrix]\n";
        return;
    }
    for (size_t i = 0; i < mat.rows(); ++i) {
        std::cout << "  [";
        for (size_t j = 0; j < mat.cols(); ++j) {
            std::cout << mat[i][j] << (j == mat.cols() - 1 ? "" : ", ");
        }
        std::cout << "]\n";
    }
    std::cout << std::endl;
}


void run_astar_pathfinding_example() {
    std::cout << "\n--- A* Pathfinding Example ---" << std::endl;
    NeuroNet::NeuroNet path_network;
    // Network input size doesn't directly affect A* pathfinding through weights of hidden/output layers,
    // but it defines the input dimensionality for the first layer's WeightMatrix.
    path_network.SetInputSize(2);
    path_network.ResizeNeuroNet(3); // 3 layers: L0 (2 neurons), L1 (3 neurons), L2 (2 neurons - output)

    // Layer 0: InputSize=2 (from network input), LayerSize=2 neurons
    path_network.ResizeLayer(0, 2);
    // Activation functions don't affect this A* version, which only uses weights.
    path_network.getLayer(0).SetActivationFunction(NeuroNet::ActivationFunctionType::None);

    // Layer 1: InputSize=2 (from L0's 2 neurons), LayerSize=3 neurons
    path_network.ResizeLayer(1, 3);
    path_network.getLayer(1).SetActivationFunction(NeuroNet::ActivationFunctionType::None);

    // Layer 2: InputSize=3 (from L1's 3 neurons), LayerSize=2 neurons (output layer)
    path_network.ResizeLayer(2, 2);
    path_network.getLayer(2).SetActivationFunction(NeuroNet::ActivationFunctionType::None);

    // Set weights for Layer 0 (Network Input -> Layer 0 neurons)
    // WeightMatrix in Layer 0 is 2 (network inputs) x 2 (L0 neurons)
    NeuroNet::LayerWeights weights_L0_for_astar;
    weights_L0_for_astar.WeightCount = 2 * 2;
    weights_L0_for_astar.WeightsVector = {
        1.0f, 0.5f,  // From Network Input 0 to L0 Neurons 0,1
        0.2f, 2.0f   // From Network Input 1 to L0 Neurons 0,1
    };
    path_network.getLayer(0).SetWeights(weights_L0_for_astar);
    // Note: The A* pathfinding as defined starts from neurons in the first hidden layer (Layer 0).
    // The weights above (Network Input -> Layer 0) are not part of the A* path cost calculation itself,
    // but they are part of the network definition. The first set of weights A* will use are those
    // connecting Layer 0 neurons to Layer 1 neurons (i.e., the weights belonging to Layer 1).

    // Set weights for Layer 1 (Layer 0 neurons -> Layer 1 neurons)
    // WeightMatrix in Layer 1 is 2 (L0 neurons) x 3 (L1 neurons)
    NeuroNet::LayerWeights weights_L1_for_astar;
    weights_L1_for_astar.WeightCount = 2 * 3;
    weights_L1_for_astar.WeightsVector = {
        1.0f, 2.0f, 0.1f,  // From L0 Neuron 0 to L1 Neurons 0,1,2
        0.5f, 0.5f, 3.0f   // From L0 Neuron 1 to L1 Neurons 0,1,2
    };
    path_network.getLayer(1).SetWeights(weights_L1_for_astar);

    // Set weights for Layer 2 (Layer 1 neurons -> Layer 2 neurons)
    // WeightMatrix in Layer 2 is 3 (L1 neurons) x 2 (L2 neurons)
    NeuroNet::LayerWeights weights_L2_for_astar;
    weights_L2_for_astar.WeightCount = 3 * 2;
    weights_L2_for_astar.WeightsVector = {
        4.0f, 0.2f,  // From L1 Neuron 0 to L2 Neurons 0,1
        0.3f, 5.0f,  // From L1 Neuron 1 to L2 Neurons 0,1
        0.4f, 0.1f   // From L1 Neuron 2 to L2 Neurons 0,1
    };
    path_network.getLayer(2).SetWeights(weights_L2_for_astar);

    std::cout << "A* example network configured with specific weights." << std::endl;
    // Expected path for max product of abs weights:
    // Start L0N0:
    //   L0N0 -> L1N0 (w:1.0) -> L2N0 (w:4.0). Prod: 1*4 = 4. Path: (0,0)->(1,0)->(2,0)
    //   L0N0 -> L1N1 (w:2.0) -> L2N1 (w:5.0). Prod: 2*5 = 10. Path: (0,0)->(1,1)->(2,1) <-- Best overall
    // Start L0N1:
    //   L0N1 -> L1N2 (w:3.0) -> L2N0 (w:0.4). Prod: 3*0.4 = 1.2 Path: (0,1)->(1,2)->(2,0)
    // The pathfinder should identify (0,0) -> (1,1) -> (2,1)

    try {
        NeuroNet::Optimization::NeuralPathfinder pathfinder(path_network);
        std::vector<NeuroNet::Optimization::AStarPathNode> optimal_path = pathfinder.FindOptimalPathAStar();

        if (optimal_path.empty()) {
            std::cout << "A* Pathfinder: No optimal path found." << std::endl;
        } else {
            std::cout << "A* Pathfinder: Optimal path found:" << std::endl;
            double calculated_product = 1.0;
            for (size_t i = 0; i < optimal_path.size(); ++i) {
                const auto& node = optimal_path[i];
                std::cout << "  Layer " << node.layer_idx << ", Neuron " << node.neuron_idx;
                if (i < optimal_path.size() - 1) {
                    const auto& next_node = optimal_path[i+1];
                    // Get weight from current_node.neuron_idx in layer current_node.layer_idx
                    // to next_node.neuron_idx in layer next_node.layer_idx.
                    // Weights are stored in the layer *receiving* the input.
                    float weight = path_network.getLayer(next_node.layer_idx).get_weight(node.neuron_idx, next_node.neuron_idx);
                    std::cout << " --(w:" << weight << ")--> ";
                    calculated_product *= std::abs(weight);
                }
            }
            std::cout << "\nCalculated Path Product (abs weights): " << calculated_product << std::endl;

            // Verification for the specific setup:
            bool path_matches = optimal_path.size() == 3 &&
                                optimal_path[0].layer_idx == 0 && optimal_path[0].neuron_idx == 0 &&
                                optimal_path[1].layer_idx == 1 && optimal_path[1].neuron_idx == 1 &&
                                optimal_path[2].layer_idx == 2 && optimal_path[2].neuron_idx == 1;
            bool product_matches = std::abs(calculated_product - 10.0) < NeuroNet::Optimization::NeuralPathfinder::EPSILON * 100;


            if (path_matches && product_matches) {
                std::cout << "Path and product match expected: (0,0) -> (1,1) -> (2,1), Product ~10.0" << std::endl;
            } else {
                std::cerr << "Path or product DOES NOT match expected!" << std::endl;
                if (!path_matches) std::cerr << "  Path mismatch." << std::endl;
                if (!product_matches) std::cerr << "  Product mismatch (got " << calculated_product << ", expected 10.0)." << std::endl;
            }
        }
    } catch (const std::exception& e) {
        std::cerr << "Exception during A* pathfinding: " << e.what() << std::endl;
    }
    std::cout << "--- A* Pathfinding Example End ---" << std::endl;
}


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

    // Run the A* pathfinding example
    run_astar_pathfinding_example();

    return 0;
}
