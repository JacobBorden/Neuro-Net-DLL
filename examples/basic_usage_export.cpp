#include <iostream>
#include <vector>
#include <cmath>
#include <functional>
#include <fstream> // Required for std::ofstream
#include <iomanip> // Required for std::put_time (though not directly used here, good for completeness)

#include "src/neural_network/neuronet.h"
#include "src/optimization/genetic_algorithm.h"
#include "src/math/matrix.h"
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

    } else {
        std::cout << "Best model has no layers, cannot save or test." << std::endl;
    }

    std::cout << "\nExample finished." << std::endl;
    return 0;
}
