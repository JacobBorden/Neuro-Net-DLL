// Catch2 directives and include removed.

#include "../src/neural_network/neuronet.h" // Adjusted path
#include "../src/math/matrix.h"           // Adjusted path
#include <vector>
#include <cmath> // For std::exp, std::max


// --- Existing Google Test code commented out or removed ---
// Removed /* block comment marker

#include "gtest/gtest.h"
#include "neural_network/neuronet.h" // Access to NeuroNet and NeuroNetLayer
#include "math/matrix.h"   // For creating Matrix objects for input
#include <fstream>      // For std::ofstream
#include <cstdio>       // For std::remove
#include <stdexcept>    // For std::runtime_error
#include "../src/utilities/json/json.hpp" // For custom JsonValue, JsonParser
#include "../src/utilities/json/json_exception.hpp" // For JsonParseException

// Test fixture for NeuroNet tests
class NeuroNetTest : public ::testing::Test {
protected:
    NeuroNet::NeuroNet net;
    NeuroNet::NeuroNetLayer layer;
};

// --- NeuroNetLayer Tests ---

TEST_F(NeuroNetTest, NeuroNetLayerConstructor) {
    NeuroNet::NeuroNetLayer test_layer;
    EXPECT_EQ(test_layer.LayerSize(), 0);
    EXPECT_EQ(test_layer.WeightCount(), 0);
    EXPECT_EQ(test_layer.BiasCount(), 0);
}

TEST_F(NeuroNetTest, NeuroNetLayerResizeLayer) {
    layer.ResizeLayer(3, 5); // 3 inputs, 5 neurons
    EXPECT_EQ(layer.LayerSize(), 5);
    EXPECT_EQ(layer.WeightCount(), 3 * 5);
    EXPECT_EQ(layer.BiasCount(), 5);

    // Check internal matrix dimensions (conceptual, direct access might not be public)
    // This would typically be verified by SetInput, CalculateOutput behavior
}

TEST_F(NeuroNetTest, NeuroNetLayerSetInput) {
    layer.ResizeLayer(3, 2); // 3 inputs, 2 neurons
    Matrix::Matrix<float> input_matrix(1, 3);
    input_matrix[0][0] = 1.0f;
    input_matrix[0][1] = 2.0f;
    input_matrix[0][2] = 3.0f;
    EXPECT_TRUE(layer.SetInput(input_matrix));

    Matrix::Matrix<float> wrong_size_input(1, 2);
    EXPECT_FALSE(layer.SetInput(wrong_size_input));
}

TEST_F(NeuroNetTest, NeuroNetLayerWeightsAndBiases) {
    layer.ResizeLayer(2, 3); // 2 inputs, 3 neurons
    EXPECT_EQ(layer.WeightCount(), 2 * 3);
    EXPECT_EQ(layer.BiasCount(), 3);

    NeuroNet::LayerWeights weights_to_set;
    weights_to_set.WeightCount = 2 * 3;
    weights_to_set.WeightsVector.resize(weights_to_set.WeightCount);
    for(int i = 0; i < weights_to_set.WeightCount; ++i) {
        weights_to_set.WeightsVector[i] = static_cast<float>(i) * 0.1f;
    }
    EXPECT_TRUE(layer.SetWeights(weights_to_set));

    NeuroNet::LayerWeights retrieved_weights = layer.get_weights();
    EXPECT_EQ(retrieved_weights.WeightCount, weights_to_set.WeightCount);
    for(int i = 0; i < weights_to_set.WeightCount; ++i) {
        EXPECT_FLOAT_EQ(retrieved_weights.WeightsVector[i], weights_to_set.WeightsVector[i]);
    }

    NeuroNet::LayerBiases biases_to_set;
    biases_to_set.BiasCount = 3;
    biases_to_set.BiasVector.resize(biases_to_set.BiasCount);
    for(int i = 0; i < biases_to_set.BiasCount; ++i) {
        biases_to_set.BiasVector[i] = static_cast<float>(i) * 0.2f;
    }
    EXPECT_TRUE(layer.SetBiases(biases_to_set));

    NeuroNet::LayerBiases retrieved_biases = layer.get_biases();
    EXPECT_EQ(retrieved_biases.BiasCount, biases_to_set.BiasCount);
    for(int i = 0; i < biases_to_set.BiasCount; ++i) {
        EXPECT_FLOAT_EQ(retrieved_biases.BiasVector[i], biases_to_set.BiasVector[i]);
    }
}

// Basic test for CalculateOutput - more detailed tests would involve known weights/biases
TEST_F(NeuroNetTest, NeuroNetLayerCalculateOutput) {
    layer.ResizeLayer(2, 1); // 2 inputs, 1 neuron
    
    NeuroNet::LayerWeights weights;
    weights.WeightCount = 2;
    weights.WeightsVector = {0.5f, 0.5f};
    layer.SetWeights(weights);

    NeuroNet::LayerBiases biases;
    biases.BiasCount = 1;
    biases.BiasVector = {0.1f};
    layer.SetBiases(biases);

    Matrix::Matrix<float> input_matrix(1, 2);
    input_matrix[0][0] = 1.0f;
    input_matrix[0][1] = 1.0f;
    layer.SetInput(input_matrix);

    Matrix::Matrix<float> output = layer.CalculateOutput();
    EXPECT_EQ(output.rows(), 1);
    EXPECT_EQ(output.cols(), 1);
    // Expected: (1.0*0.5 + 1.0*0.5) + 0.1 = 1.0 + 0.1 = 1.1
    EXPECT_FLOAT_EQ(output[0][0], 1.1f);
}


// --- NeuroNet Tests ---

TEST_F(NeuroNetTest, NeuroNetConstructor) {
    NeuroNet::NeuroNet test_net_default; // Default constructor
    // No direct way to check layer count without a getter, rely on ResizeNeuroNet

    NeuroNet::NeuroNet test_net_param(3); // Constructor with layer count
    // Again, rely on other methods to verify internal state post-construction
}

TEST_F(NeuroNetTest, NeuroNetConfiguration) {
    net.SetInputSize(4); // 4 input features
    net.ResizeNeuroNet(2); // 2 layers
    
    // Configure layer 0: 4 inputs (from SetInputSize), 5 neurons
    EXPECT_TRUE(net.ResizeLayer(0, 5)); 
    // Configure layer 1: 5 inputs (from layer 0), 3 neurons
    EXPECT_TRUE(net.ResizeLayer(1, 3)); 

    // Attempt to resize non-existent layer
    EXPECT_FALSE(net.ResizeLayer(2, 2));
}

TEST_F(NeuroNetTest, NeuroNetSetAndGetInputOutput) {
    net.SetInputSize(2);
    net.ResizeNeuroNet(1);
    net.ResizeLayer(0, 1); // Layer 0: 2 inputs, 1 output

    // Set weights and biases for the single layer
    // Accessing layer 0 directly is not typical, but we can use get_all_layer_weights/biases
    // For simplicity here, we'd need a way to set them. Let's assume some default init or test with flat setters.
    std::vector<float> weights_flat = {0.5f, 0.3f}; // For 2 inputs, 1 neuron
    std::vector<float> biases_flat = {0.1f};       // For 1 neuron
    
    EXPECT_TRUE(net.set_all_weights_flat(weights_flat));
    EXPECT_TRUE(net.set_all_biases_flat(biases_flat));

    Matrix::Matrix<float> input_matrix(1, 2);
    input_matrix[0][0] = 1.0f;
    input_matrix[0][1] = 2.0f;
    EXPECT_TRUE(net.SetInput(input_matrix));

    Matrix::Matrix<float> output = net.GetOutput();
    EXPECT_EQ(output.rows(), 1);
    EXPECT_EQ(output.cols(), 1); 
    // Expected: (1.0 * 0.5 + 2.0 * 0.3) + 0.1 = 0.5 + 0.6 + 0.1 = 1.2
    EXPECT_FLOAT_EQ(output[0][0], 1.2f);
}


TEST_F(NeuroNetTest, NeuroNetAllWeightsBiasesFlat) {
    net.SetInputSize(2);
    net.ResizeNeuroNet(2); // 2 layers
    net.ResizeLayer(0, 3); // Layer 0: 2 inputs, 3 neurons (6 weights, 3 biases)
    net.ResizeLayer(1, 1); // Layer 1: 3 inputs, 1 neuron  (3 weights, 1 bias)
                           // Total weights: 6 + 3 = 9
                           // Total biases: 3 + 1 = 4

    std::vector<float> weights_to_set(9);
    for(size_t i = 0; i < weights_to_set.size(); ++i) weights_to_set[i] = static_cast<float>(i) * 0.1f;
    
    std::vector<float> biases_to_set(4);
    for(size_t i = 0; i < biases_to_set.size(); ++i) biases_to_set[i] = static_cast<float>(i) * -0.1f;

    EXPECT_TRUE(net.set_all_weights_flat(weights_to_set));
    EXPECT_TRUE(net.set_all_biases_flat(biases_to_set));

    std::vector<float> retrieved_weights = net.get_all_weights_flat();
    EXPECT_EQ(retrieved_weights.size(), weights_to_set.size());
    for(size_t i = 0; i < weights_to_set.size(); ++i) {
        EXPECT_FLOAT_EQ(retrieved_weights[i], weights_to_set[i]);
    }

    std::vector<float> retrieved_biases = net.get_all_biases_flat();
    EXPECT_EQ(retrieved_biases.size(), biases_to_set.size());
    for(size_t i = 0; i < biases_to_set.size(); ++i) {
        EXPECT_FLOAT_EQ(retrieved_biases[i], biases_to_set[i]);
    }

    // Test setting with wrong sizes
    std::vector<float> wrong_size_weights(5);
    EXPECT_FALSE(net.set_all_weights_flat(wrong_size_weights));
    std::vector<float> wrong_size_biases(2);
    EXPECT_FALSE(net.set_all_biases_flat(wrong_size_biases));
}

TEST_F(NeuroNetTest, NeuroNetAllLayerWeightsBiases) {
    net.SetInputSize(1);
    net.ResizeNeuroNet(2);
    net.ResizeLayer(0, 2); // L0: 1 in, 2 out (2W, 2B)
    net.ResizeLayer(1, 1); // L1: 2 in, 1 out (2W, 1B)

    std::vector<NeuroNet::LayerWeights> all_w(2);
    all_w[0].WeightCount = 2; all_w[0].WeightsVector = {0.1f, 0.2f};
    all_w[1].WeightCount = 2; all_w[1].WeightsVector = {0.3f, 0.4f};
    
    std::vector<NeuroNet::LayerBiases> all_b(2);
    all_b[0].BiasCount = 2; all_b[0].BiasVector = {-0.1f, -0.2f};
    all_b[1].BiasCount = 1; all_b[1].BiasVector = {-0.3f};

    EXPECT_TRUE(net.set_all_layer_weights(all_w));
    EXPECT_TRUE(net.set_all_layer_biases(all_b));

    std::vector<NeuroNet::LayerWeights> retrieved_all_w = net.get_all_layer_weights();
    ASSERT_EQ(retrieved_all_w.size(), 2);
    EXPECT_EQ(retrieved_all_w[0].WeightCount, 2);
    EXPECT_FLOAT_EQ(retrieved_all_w[0].WeightsVector[0], 0.1f);
    EXPECT_FLOAT_EQ(retrieved_all_w[0].WeightsVector[1], 0.2f);
    EXPECT_EQ(retrieved_all_w[1].WeightCount, 2);
    EXPECT_FLOAT_EQ(retrieved_all_w[1].WeightsVector[0], 0.3f);
    EXPECT_FLOAT_EQ(retrieved_all_w[1].WeightsVector[1], 0.4f);

    std::vector<NeuroNet::LayerBiases> retrieved_all_b = net.get_all_layer_biases();
    ASSERT_EQ(retrieved_all_b.size(), 2);
    EXPECT_EQ(retrieved_all_b[0].BiasCount, 2);
    EXPECT_FLOAT_EQ(retrieved_all_b[0].BiasVector[0], -0.1f);
    EXPECT_FLOAT_EQ(retrieved_all_b[0].BiasVector[1], -0.2f);
    EXPECT_EQ(retrieved_all_b[1].BiasCount, 1);
    EXPECT_FLOAT_EQ(retrieved_all_b[1].BiasVector[0], -0.3f);
}

// Removed */ block comment marker


// --- Activation Function Tests ---

TEST_F(NeuroNetTest, ActivationReLU) {
    const int num_outputs = 3;
    layer.ResizeLayer(1, num_outputs); // InputSize=1, LayerSize=num_outputs

    Matrix::Matrix<float> input_to_layer(1, 1);
    input_to_layer[0][0] = 1.0f;
    layer.SetInput(input_to_layer);

    NeuroNet::LayerBiases biases_config; // Renamed to avoid conflict with fixture member if any
    biases_config.BiasCount = num_outputs;
    biases_config.BiasVector.assign(num_outputs, 0.0f);
    layer.SetBiases(biases_config);

    NeuroNet::LayerWeights weights_config; // Renamed
    weights_config.WeightCount = num_outputs;
    weights_config.WeightsVector = {1.0f, -2.0f, 0.0f};
    layer.SetWeights(weights_config);

    layer.SetActivationFunction(NeuroNet::ActivationFunctionType::ReLU);
    Matrix::Matrix<float> output = layer.CalculateOutput();

    EXPECT_EQ(output.rows(), 1);
    EXPECT_EQ(output.cols(), num_outputs);
    EXPECT_FLOAT_EQ(output[0][0], 1.0f);
    EXPECT_FLOAT_EQ(output[0][1], 0.0f);
    EXPECT_FLOAT_EQ(output[0][2], 0.0f);
}

TEST_F(NeuroNetTest, ActivationLeakyReLU) {
    const int num_outputs = 3;
    layer.ResizeLayer(1, num_outputs);

    Matrix::Matrix<float> input_to_layer(1, 1);
    input_to_layer[0][0] = 1.0f;
    layer.SetInput(input_to_layer);

    NeuroNet::LayerBiases biases_config;
    biases_config.BiasCount = num_outputs;
    biases_config.BiasVector.assign(num_outputs, 0.0f);
    layer.SetBiases(biases_config);

    NeuroNet::LayerWeights weights_config;
    weights_config.WeightCount = num_outputs;
    weights_config.WeightsVector = {1.0f, -2.0f, -10.0f};
    layer.SetWeights(weights_config);

    layer.SetActivationFunction(NeuroNet::ActivationFunctionType::LeakyReLU);
    Matrix::Matrix<float> output = layer.CalculateOutput();

    EXPECT_EQ(output.rows(), 1);
    EXPECT_EQ(output.cols(), num_outputs);
    EXPECT_FLOAT_EQ(output[0][0], 1.0f);
    EXPECT_FLOAT_EQ(output[0][1], -0.02f);
    EXPECT_FLOAT_EQ(output[0][2], -0.1f);
}

TEST_F(NeuroNetTest, ActivationELU) {
    const int num_outputs = 3;
    layer.ResizeLayer(1, num_outputs);

    Matrix::Matrix<float> input_to_layer(1, 1);
    input_to_layer[0][0] = 1.0f;
    layer.SetInput(input_to_layer);

    NeuroNet::LayerBiases biases_config;
    biases_config.BiasCount = num_outputs;
    biases_config.BiasVector.assign(num_outputs, 0.0f);
    layer.SetBiases(biases_config);

    NeuroNet::LayerWeights weights_config;
    weights_config.WeightCount = num_outputs;
    weights_config.WeightsVector = {1.0f, -2.0f, 0.0f};
    layer.SetWeights(weights_config);

    layer.SetActivationFunction(NeuroNet::ActivationFunctionType::ELU);
    Matrix::Matrix<float> output = layer.CalculateOutput();

    EXPECT_EQ(output.rows(), 1);
    EXPECT_EQ(output.cols(), num_outputs);
    EXPECT_FLOAT_EQ(output[0][0], 1.0f);
    EXPECT_FLOAT_EQ(output[0][1], 1.0f * (std::exp(-2.0f) - 1.0f));
    EXPECT_FLOAT_EQ(output[0][2], 0.0f);
}

TEST_F(NeuroNetTest, ActivationSoftmax) {
    const int num_outputs = 3;
    layer.ResizeLayer(1, num_outputs);

    Matrix::Matrix<float> input_to_layer(1, 1);
    input_to_layer[0][0] = 1.0f;
    layer.SetInput(input_to_layer);

    NeuroNet::LayerBiases biases_config;
    biases_config.BiasCount = num_outputs;
    biases_config.BiasVector.assign(num_outputs, 0.0f);
    layer.SetBiases(biases_config);

    NeuroNet::LayerWeights weights_config;
    weights_config.WeightCount = num_outputs;
    weights_config.WeightsVector = {1.0f, 2.0f, 3.0f};
    layer.SetWeights(weights_config);

    layer.SetActivationFunction(NeuroNet::ActivationFunctionType::Softmax);
    Matrix::Matrix<float> output = layer.CalculateOutput();

    EXPECT_EQ(output.rows(), 1);
    EXPECT_EQ(output.cols(), num_outputs);

    float e1 = std::exp(1.0f);
    float e2 = std::exp(2.0f);
    float e3 = std::exp(3.0f);
    float sum_exp = e1 + e2 + e3;

    EXPECT_FLOAT_EQ(output[0][0], e1 / sum_exp);
    EXPECT_FLOAT_EQ(output[0][1], e2 / sum_exp);
    EXPECT_FLOAT_EQ(output[0][2], e3 / sum_exp);
}

TEST_F(NeuroNetTest, ActivationNone) {
    const int num_outputs = 3;
    layer.ResizeLayer(1, num_outputs);

    Matrix::Matrix<float> input_to_layer(1, 1);
    input_to_layer[0][0] = 1.0f;
    layer.SetInput(input_to_layer);

    NeuroNet::LayerBiases biases_config;
    biases_config.BiasCount = num_outputs;
    biases_config.BiasVector.assign(num_outputs, 0.0f);
    layer.SetBiases(biases_config);

    NeuroNet::LayerWeights weights_config;
    weights_config.WeightCount = num_outputs;
    weights_config.WeightsVector = {1.0f, -2.0f, 0.5f};
    layer.SetWeights(weights_config);

    layer.SetActivationFunction(NeuroNet::ActivationFunctionType::None);
    Matrix::Matrix<float> output = layer.CalculateOutput();

    EXPECT_EQ(output.rows(), 1);
    EXPECT_EQ(output.cols(), num_outputs);
    EXPECT_FLOAT_EQ(output[0][0], 1.0f);
    EXPECT_FLOAT_EQ(output[0][1], -2.0f);
    EXPECT_FLOAT_EQ(output[0][2], 0.5f);
}

// --- End of Activation Function Tests ---

TEST_F(NeuroNetTest, Serialization) {
    const std::string test_filename = "test_model_serialization.json";
    const std::string empty_filename = "empty_test.json";
    const std::string malformed_filename = "malformed_test.json";

    // 1. Create and configure the original model
    NeuroNet::NeuroNet original_model;
    original_model.SetInputSize(10);
    original_model.ResizeNeuroNet(2); // 2 layers

    // Configure layer 0
    original_model.ResizeLayer(0, 5); // Layer 0: 10 inputs, 5 outputs
    original_model.NeuroNetVector[0].SetActivationFunction(NeuroNet::ActivationFunctionType::ReLU);
    NeuroNet::LayerWeights lw0;
    lw0.WeightCount = 10 * 5;
    for(int k=0; k < lw0.WeightCount; ++k) lw0.WeightsVector.push_back(static_cast<float>(k + 1) * 0.05f);
    original_model.NeuroNetVector[0].SetWeights(lw0);
    NeuroNet::LayerBiases lb0;
    lb0.BiasCount = 5;
    for(int k=0; k < lb0.BiasCount; ++k) lb0.BiasVector.push_back(static_cast<float>(k + 1) * 0.1f);
    original_model.NeuroNetVector[0].SetBiases(lb0);

    // Configure layer 1
    original_model.ResizeLayer(1, 3); // Layer 1: 5 inputs (from L0), 3 outputs
    original_model.NeuroNetVector[1].SetActivationFunction(NeuroNet::ActivationFunctionType::Softmax);
    NeuroNet::LayerWeights lw1;
    lw1.WeightCount = 5 * 3;
    for(int k=0; k < lw1.WeightCount; ++k) lw1.WeightsVector.push_back(static_cast<float>(k + 1) * -0.03f);
    original_model.NeuroNetVector[1].SetWeights(lw1);
    NeuroNet::LayerBiases lb1;
    lb1.BiasCount = 3;
    for(int k=0; k < lb1.BiasCount; ++k) lb1.BiasVector.push_back(static_cast<float>(k + 1) * -0.05f);
    original_model.NeuroNetVector[1].SetBiases(lb1);

    // Store original properties for comparison
    // Note: Direct access to InputSize and LayerCount is not available,
    // we verify these by checking the structure of the loaded model (number of layers, their sizes)
    std::vector<NeuroNet::LayerWeights> original_weights_list = original_model.get_all_layer_weights();
    std::vector<NeuroNet::LayerBiases> original_biases_list = original_model.get_all_layer_biases();
    std::vector<int> original_layer_sizes;
    std::vector<NeuroNet::ActivationFunctionType> original_activations;
    original_layer_sizes.push_back(original_model.NeuroNetVector[0].LayerSize());
    original_layer_sizes.push_back(original_model.NeuroNetVector[1].LayerSize());
    original_activations.push_back(original_model.NeuroNetVector[0].get_activation_type());
    original_activations.push_back(original_model.NeuroNetVector[1].get_activation_type());
    int original_overall_input_size = 10; // This was set initially

    // 2. Save the model
    EXPECT_TRUE(original_model.save_model(test_filename));

    // 3. Load the model
    NeuroNet::NeuroNet loaded_model = NeuroNet::NeuroNet::load_model(test_filename);

    // 4. Compare
    // Verify overall structure implicitly by checking layer details
    ASSERT_EQ(loaded_model.NeuroNetVector.size(), 2) << "Loaded model should have 2 layers.";
    EXPECT_EQ(loaded_model.InputSize, original_overall_input_size) << "Loaded model input size mismatch.";


    std::vector<NeuroNet::LayerWeights> loaded_weights_list = loaded_model.get_all_layer_weights();
    std::vector<NeuroNet::LayerBiases> loaded_biases_list = loaded_model.get_all_layer_biases();

    ASSERT_EQ(loaded_weights_list.size(), original_weights_list.size());
    for (size_t i = 0; i < original_weights_list.size(); ++i) {
        EXPECT_EQ(loaded_model.NeuroNetVector[i].LayerSize(), original_layer_sizes[i]);
        EXPECT_EQ(loaded_model.NeuroNetVector[i].get_activation_type(), original_activations[i]);
        
        ASSERT_EQ(loaded_weights_list[i].WeightsVector.size(), original_weights_list[i].WeightsVector.size());
        for (size_t j = 0; j < original_weights_list[i].WeightsVector.size(); ++j) {
            EXPECT_FLOAT_EQ(loaded_weights_list[i].WeightsVector[j], original_weights_list[i].WeightsVector[j]);
        }

        ASSERT_EQ(loaded_biases_list[i].BiasVector.size(), original_biases_list[i].BiasVector.size());
        for (size_t j = 0; j < original_biases_list[i].BiasVector.size(); ++j) {
            EXPECT_FLOAT_EQ(loaded_biases_list[i].BiasVector[j], original_biases_list[i].BiasVector[j]);
        }
    }
    
    // Check layer input sizes (derived check)
    // Layer 0 input should be the network's input size
    // Layer 1 input should be Layer 0's output size
    // This is implicitly tested if SetWeights and SetBiases succeed during load, as they depend on correct
    // internal matrix sizes which NeuroNetLayer::ResizeLayer sets up.
    // NeuroNet::ResizeLayer inside load_model is responsible for setting layer input sizes correctly.

    // 5. Test error handling for load_model
    EXPECT_THROW(NeuroNet::NeuroNet::load_model("non_existent_file.json"), std::runtime_error);

    // Test with an empty file
    std::ofstream ofs_empty(empty_filename);
    ofs_empty.close();
    EXPECT_THROW(NeuroNet::NeuroNet::load_model(empty_filename), std::runtime_error);
    std::remove(empty_filename.c_str());

    // Test with a malformed JSON file (e.g., missing layer_count)
    std::ofstream ofs_malformed(malformed_filename);
    ofs_malformed << "{ \"input_size\": 5 }"; // Missing layer_count
    ofs_malformed.close();
    EXPECT_THROW(NeuroNet::NeuroNet::load_model(malformed_filename), std::runtime_error);
    std::remove(malformed_filename.c_str());
    
    // Test with a JSON file with missing weights data
    std::ofstream ofs_missing_weights(malformed_filename); // Reuse filename
    ofs_missing_weights << R"({
        "input_size": 10,
        "layer_count": 1,
        "layers": [
            {
                "input_size": 10,
                "layer_size": 5,
                "activation_function": "ReLU", // Now string
                "biases": {"rows":1.0, "cols":5.0, "data": [0.1,0.1,0.1,0.1,0.1]}
            }
        ]
    })";
    ofs_missing_weights.close();
    EXPECT_THROW(NeuroNet::NeuroNet::load_model(malformed_filename), std::runtime_error);
    std::remove(malformed_filename.c_str());


    // 6. Cleanup
    std::remove(test_filename.c_str());
}

// Helper function for creating temporary vocabulary files specifically for NeuroNet string input tests
static void CreateTempVocabFileForNeuroNetTest(const std::string& filepath, const std::string& content) {
    std::ofstream ofs(filepath);
    ofs << content;
    ofs.close();
}
const std::string nn_test_vocab_path = "nn_temp_vocab.json"; // Unique name for these tests

// --- NeuroNetStringInputTest Tests ---

TEST(NeuroNetStringInputTest, LoadVocabulary_Success) {
    NeuroNet::NeuroNet net;
    CreateTempVocabFileForNeuroNetTest(nn_test_vocab_path, R"({
        "word_to_token": { "hello": 0, "<UNK>": 1, "<PAD>": 2 },
        "token_to_word": { "0": "hello", "1": "<UNK>", "2": "<PAD>" },
        "special_tokens": { "unknown_token": "<UNK>", "padding_token": "<PAD>" }
    })");
    ASSERT_TRUE(net.LoadVocabulary(nn_test_vocab_path));
    std::remove(nn_test_vocab_path.c_str());
}

TEST(NeuroNetStringInputTest, LoadVocabulary_Fail_NonExistentFile) {
    NeuroNet::NeuroNet net;
    ASSERT_FALSE(net.LoadVocabulary("non_existent_vocab.json"));
}

TEST(NeuroNetStringInputTest, SetStringsInput_Success) {
    NeuroNet::NeuroNet net;
    net.SetInputSize(3); // Network expects input vector of size 3
    // Configure network layers BEFORE SetStringsInput
    net.ResizeNeuroNet(1);
    net.ResizeLayer(0, 1); // Layer 0: InputSize=3, LayerSize=1

    CreateTempVocabFileForNeuroNetTest(nn_test_vocab_path, R"({
        "word_to_token": { "a":0, "b":1, "<UNK>":2, "<PAD>":3 },
        "token_to_word": { "0":"a", "1":"b", "2":"<UNK>", "3":"<PAD>" },
        "special_tokens": { "unknown_token": "<UNK>", "padding_token": "<PAD>" }
    })");
    ASSERT_TRUE(net.LoadVocabulary(nn_test_vocab_path));

    // Changed to single string in batch to align with layer's SetInput expectation
    const std::string json_input = R"({ "input_batch": ["a b a"] })";
    // max_len_override = 3 to match net.SetInputSize(3)
    ASSERT_TRUE(net.SetStringsInput(json_input, 3));

    ASSERT_NO_THROW(net.GetOutput()); // GetOutput should now work

    std::remove(nn_test_vocab_path.c_str());
}

TEST(NeuroNetStringInputTest, SetStringsInput_Fail_VocabNotLoaded) {
    NeuroNet::NeuroNet net;
    net.SetInputSize(3);
    const std::string json_input = R"({ "input_batch": ["a b"] })";
    ASSERT_THROW(net.SetStringsInput(json_input, 3), std::runtime_error);
}

TEST(NeuroNetStringInputTest, SetStringsInput_Fail_JsonParseError) {
    NeuroNet::NeuroNet net;
    CreateTempVocabFileForNeuroNetTest(nn_test_vocab_path, R"({
        "word_to_token": { "a":0, "<PAD>":1, "<UNK>":2 },
        "token_to_word": { "0":"a", "1":"<PAD>", "2":"<UNK>" },
        "special_tokens": { "unknown_token": "<UNK>", "padding_token": "<PAD>" }
    })");
    ASSERT_TRUE(net.LoadVocabulary(nn_test_vocab_path));
    const std::string malformed_json = R"({ "input_batch": ["a b"], ... )"; // Malformed
    ASSERT_THROW(net.SetStringsInput(malformed_json, 2), JsonParseException);
    std::remove(nn_test_vocab_path.c_str());
}

TEST(NeuroNetStringInputTest, SetStringsInput_Fail_MatrixColsMismatchNetworkInputSize) {
    NeuroNet::NeuroNet net;
    net.SetInputSize(5); // Network expects 5 features
    CreateTempVocabFileForNeuroNetTest(nn_test_vocab_path, R"({
        "word_to_token": { "a":0, "b":1, "<UNK>":2, "<PAD>":3 },
        "token_to_word": { "0":"a", "1":"b", "2":"<UNK>", "3":"<PAD>" },
        "special_tokens": { "unknown_token": "<UNK>", "padding_token": "<PAD>" }
    })");
    ASSERT_TRUE(net.LoadVocabulary(nn_test_vocab_path));

    const std::string json_input = R"({ "input_batch": ["a b a"] })";
    // prepare_batch_matrix with max_len_override = 3 will create matrix with 3 cols.
    // This will mismatch net.InputSize = 5.
    ASSERT_THROW(net.SetStringsInput(json_input, 3), std::runtime_error);
    std::remove(nn_test_vocab_path.c_str());
}

TEST(NeuroNetStringInputTest, SetStringsInput_CorrectPaddingToMatchNetworkInputSize) {
    NeuroNet::NeuroNet net;
    net.SetInputSize(5); // Network expects 5 features
    CreateTempVocabFileForNeuroNetTest(nn_test_vocab_path, R"({
        "word_to_token": { "a":0, "b":1, "<UNK>":2, "<PAD>":3 },
        "token_to_word": { "0":"a", "1":"b", "2":"<UNK>", "3":"<PAD>" },
        "special_tokens": { "unknown_token": "<UNK>", "padding_token": "<PAD>" }
    })");
    ASSERT_TRUE(net.LoadVocabulary(nn_test_vocab_path));

    const std::string json_input = R"({ "input_batch": ["a b a", "a b a b c d e f"] })";
    // max_len_override = 5 to match net.InputSize. First sequence will be padded, second truncated.
    ASSERT_NO_THROW(net.SetStringsInput(json_input, 5));
    std::remove(nn_test_vocab_path.c_str());
}

TEST(NeuroNetStringInputTest, SaveLoadModel_VocabularyMaxSeqLen) {
    NeuroNet::NeuroNet net_to_save;
    net_to_save.SetInputSize(7); // Configure InputSize for the network to be saved
    net_to_save.ResizeNeuroNet(1); // Add a layer
    net_to_save.ResizeLayer(0, 2); // Layer 0: 7 inputs, 2 outputs (example)

    const std::string model_path = "test_model_vocab_cfg.json";
    const std::string vocab_path_for_test = "test_model_vocab_file.json"; // Renamed to avoid conflict

    // Create vocab with max_sequence_length in its config
    CreateTempVocabFileForNeuroNetTest(vocab_path_for_test, R"({
        "word_to_token": { "a":0, "<UNK>":1, "<PAD>":2 },
        "token_to_word": { "0":"a", "1":"<UNK>", "2":"<PAD>" },
        "special_tokens": { "unknown_token": "<UNK>", "padding_token": "<PAD>" },
        "config": { "max_sequence_length": 7 }
    })");
    ASSERT_TRUE(net_to_save.LoadVocabulary(vocab_path_for_test));
    EXPECT_EQ(net_to_save.getVocabulary().get_max_sequence_length(), 7);
    // Also set the vocabulary's max_sequence_length in the NeuroNet object itself for saving.
    // This happens if LoadVocabulary also updates the NeuroNet's vocabulary member's max_sequence_length.
    // The save_model function saves `this->vocabulary.get_max_sequence_length()`.
    // And load_model sets `model.vocabulary.set_max_sequence_length(max_len);`

    ASSERT_TRUE(net_to_save.save_model(model_path));

    NeuroNet::NeuroNet net_to_load;
    ASSERT_NO_THROW(net_to_load = NeuroNet::NeuroNet::load_model(model_path));

    // Direct check using the new getter
    // The loaded model's internal vocabulary object should have max_sequence_length set from the file.
    EXPECT_EQ(net_to_load.getVocabulary().get_max_sequence_length(), 7);

    // Indirect test: Load the actual vocabulary words and see if SetStringsInput behaves as if max_len is 7.
    ASSERT_TRUE(net_to_load.LoadVocabulary(vocab_path_for_test)); // Load the actual words into the loaded network's vocab object
    // The network structure (InputSize, layers) should be restored by load_model.
    // So net_to_load.InputSize should be 7.
    EXPECT_EQ(net_to_load.GetInputSize(), 7); // Verify this assumption
    ASSERT_EQ(net_to_load.getLayerCount(), 1);
    ASSERT_EQ(net_to_load.getLayer(0).LayerSize(), 2);


    const std::string json_input = R"({ "input_batch": ["a a a"] })"; // Single sequence
    // SetStringsInput should use the loaded max_sequence_length of 7 from net_to_load's vocabulary object.
    // The input matrix will be 1x7. The layer expects 7 inputs. This should work.
    EXPECT_TRUE(net_to_load.SetStringsInput(json_input));


    std::remove(model_path.c_str());
    std::remove(vocab_path_for_test.c_str());
}

TEST(NeuroNetStringInputTest, LoadModel_NoVocabConfig_Defaults) {
    NeuroNet::NeuroNet net_to_save;
    net_to_save.SetInputSize(1); // Example initial InputSize for the network to be saved
    net_to_save.ResizeNeuroNet(1); // Add a layer
    net_to_save.ResizeLayer(0,1);  // Layer 0: 1 input, 1 output

    const std::string model_path = "test_model_no_vocab_cfg.json";
    // Save a model without loading any vocabulary, so no vocab_config will be saved.
    // The internal vocabulary's max_sequence_length will be default -1.
    ASSERT_TRUE(net_to_save.save_model(model_path));
    EXPECT_EQ(net_to_save.getVocabulary().get_max_sequence_length(), -1);


    NeuroNet::NeuroNet net_to_load;
    ASSERT_NO_THROW(net_to_load = NeuroNet::NeuroNet::load_model(model_path));
    // After loading, net_to_load.InputSize should be 1 (from the saved model).
    EXPECT_EQ(net_to_load.GetInputSize(), 1);

    // max_sequence_length on its vocab should be -1 (default) as it was not in the saved model.
    EXPECT_EQ(net_to_load.getVocabulary().get_max_sequence_length(), -1);

    CreateTempVocabFileForNeuroNetTest(nn_test_vocab_path, R"({
        "word_to_token": { "a":0, "<UNK>":1, "<PAD>":2 },
        "token_to_word": { "0":"a", "1":"<UNK>", "2":"<PAD>" },
        "special_tokens": { "unknown_token": "<UNK>", "padding_token": "<PAD>" }
    })"); // No config.max_sequence_length here
    ASSERT_TRUE(net_to_load.LoadVocabulary(nn_test_vocab_path));
    // After loading this vocab, max_sequence_length should still be -1 as vocab file doesn't set it.
    EXPECT_EQ(net_to_load.getVocabulary().get_max_sequence_length(), -1);

    // This configures the loaded network to expect 3 input features.
    // Crucially, SetInputSize also calls ResizeLayer(0, current_output_size) on the first layer,
    // adapting its input expectations (WeightMatrix, internal InputSize member).
    net_to_load.SetInputSize(3);
    EXPECT_EQ(net_to_load.GetInputSize(), 3); // Verify InputSize is updated
    ASSERT_EQ(net_to_load.getLayerCount(), 1); // Still 1 layer
    // Layer 0 should now expect 3 inputs. Its internal InputMatrix will be 1x3.

    // Changed to single string. Vocab has no max_len, SetStringsInput has no override.
    // prepare_batch_matrix will pad to max in batch. Here, "a a a" has length 3.
    // Resulting input matrix will be 1x3. This matches the layer's expectation (1x3).
    const std::string json_input = R"({ "input_batch": ["a a a"] })";
    EXPECT_TRUE(net_to_load.SetStringsInput(json_input));

    std::remove(model_path.c_str());
    std::remove(nn_test_vocab_path.c_str());
}

TEST(NeuroNetStringInputTest, EndToEnd_StringInput_SimpleNet) {
    NeuroNet::NeuroNet net;
    net.SetInputSize(2);
    net.ResizeNeuroNet(1);
    net.ResizeLayer(0, 1);
    net.getLayer(0).SetActivationFunction(NeuroNet::ActivationFunctionType::None);

    NeuroNet::LayerWeights weights;
    weights.WeightCount = 2;
    weights.WeightsVector.push_back(1.0f);
    weights.WeightsVector.push_back(1.0f);
    ASSERT_TRUE(net.getLayer(0).SetWeights(weights));
    NeuroNet::LayerBiases biases;
    biases.BiasCount = 1;
    biases.BiasVector.push_back(0.0f);
    ASSERT_TRUE(net.getLayer(0).SetBiases(biases));

    CreateTempVocabFileForNeuroNetTest(nn_test_vocab_path, R"({
        "word_to_token": { "cat":5, "dog":10, "<UNK>":0, "<PAD>":1 },
        "token_to_word": { "5":"cat", "10":"dog", "0":"<UNK>", "1":"<PAD>" },
        "special_tokens": { "unknown_token": "<UNK>", "padding_token": "<PAD>" }
    })");
    ASSERT_TRUE(net.LoadVocabulary(nn_test_vocab_path));

    const std::string json_input = R"({ "input_batch": ["cat dog"] })";
    ASSERT_TRUE(net.SetStringsInput(json_input, 2)); // max_len_override = 2

    Matrix::Matrix<float> output = net.GetOutput();
    ASSERT_EQ(output.rows(), 1);
    ASSERT_EQ(output.cols(), 1);
    EXPECT_FLOAT_EQ(output[0][0], 15.0f); // 5.0*1.0 + 10.0*1.0 + 0.0 = 15.0

    std::remove(nn_test_vocab_path.c_str());
}

TEST_F(NeuroNetTest, ToCustomJsonString) {
    NeuroNet::NeuroNet model;
    model.SetInputSize(2);
    model.ResizeNeuroNet(2);
    model.ResizeLayer(0, 3); // Layer 0: 2 in, 3 out
    model.getLayer(0).SetActivationFunction(NeuroNet::ActivationFunctionType::ReLU);
    model.ResizeLayer(1, 1); // Layer 1: 3 in, 1 out
    model.getLayer(1).SetActivationFunction(NeuroNet::ActivationFunctionType::Softmax);

    // Add some dummy weights/biases for completeness
    std::vector<float> l0_weights(2 * 3, 0.1f);
    std::vector<float> l0_biases(3, 0.01f);
    NeuroNet::LayerWeights lw0; lw0.WeightCount = 2*3; lw0.WeightsVector = l0_weights;
    NeuroNet::LayerBiases lb0; lb0.BiasCount = 3; lb0.BiasVector = l0_biases;
    model.getLayer(0).SetWeights(lw0);
    model.getLayer(0).SetBiases(lb0);

    std::vector<float> l1_weights(3 * 1, 0.2f);
    std::vector<float> l1_biases(1, 0.02f);
    NeuroNet::LayerWeights lw1; lw1.WeightCount = 3*1; lw1.WeightsVector = l1_weights;
    NeuroNet::LayerBiases lb1; lb1.BiasCount = 1; lb1.BiasVector = l1_biases;
    model.getLayer(1).SetWeights(lw1);
    model.getLayer(1).SetBiases(lb1);

    std::string json_str = model.to_custom_json_string();

    ASSERT_FALSE(json_str.empty());
    EXPECT_EQ(json_str.front(), '{'); // Basic check
    EXPECT_EQ(json_str.back(), '}');  // Basic check

    JsonValue parsed_json;
    ASSERT_NO_THROW({
        parsed_json = JsonParser::Parse(json_str);
    }) << "Failed to parse JSON string: " << json_str;
    
    ASSERT_EQ(parsed_json.type, JsonValueType::Object);
    const auto& root_obj = parsed_json.GetObject();

    ASSERT_TRUE(root_obj.count("input_size"));
    EXPECT_EQ(root_obj.at("input_size")->type, JsonValueType::Number);
    EXPECT_DOUBLE_EQ(root_obj.at("input_size")->GetNumber(), 2.0);

    ASSERT_TRUE(root_obj.count("layer_count"));
    EXPECT_EQ(root_obj.at("layer_count")->type, JsonValueType::Number);
    EXPECT_DOUBLE_EQ(root_obj.at("layer_count")->GetNumber(), 2.0);

    ASSERT_TRUE(root_obj.count("layers"));
    ASSERT_EQ(root_obj.at("layers")->type, JsonValueType::Array);
    const auto& layers_array = root_obj.at("layers")->GetArray();
    ASSERT_EQ(layers_array.size(), 2);

    // Validate Layer 0
    ASSERT_FALSE(layers_array.empty());
    ASSERT_EQ(layers_array[0].type, JsonValueType::Object);
    const auto& layer0_obj_map = layers_array[0].GetObject(); // Correctly get the map
    ASSERT_TRUE(layer0_obj_map.count("activation_function"));
    EXPECT_EQ(layer0_obj_map.at("activation_function")->type, JsonValueType::String);
    EXPECT_EQ(layer0_obj_map.at("activation_function")->GetString(), "ReLU");
    ASSERT_TRUE(layer0_obj_map.count("input_size"));
    EXPECT_DOUBLE_EQ(layer0_obj_map.at("input_size")->GetNumber(), 2.0);
    ASSERT_TRUE(layer0_obj_map.count("layer_size"));
    EXPECT_DOUBLE_EQ(layer0_obj_map.at("layer_size")->GetNumber(), 3.0);
    ASSERT_TRUE(layer0_obj_map.count("weights"));
    EXPECT_EQ(layer0_obj_map.at("weights")->type, JsonValueType::Object);
    ASSERT_TRUE(layer0_obj_map.count("biases"));
    EXPECT_EQ(layer0_obj_map.at("biases")->type, JsonValueType::Object);


    // Validate Layer 1
    ASSERT_EQ(layers_array.size(), 2); // Ensure second layer exists
    ASSERT_EQ(layers_array[1].type, JsonValueType::Object);
    const auto& layer1_obj_map = layers_array[1].GetObject(); // Correctly get the map
    ASSERT_TRUE(layer1_obj_map.count("activation_function"));
    EXPECT_EQ(layer1_obj_map.at("activation_function")->type, JsonValueType::String);
    EXPECT_EQ(layer1_obj_map.at("activation_function")->GetString(), "Softmax");
    ASSERT_TRUE(layer1_obj_map.count("input_size"));
    EXPECT_DOUBLE_EQ(layer1_obj_map.at("input_size")->GetNumber(), 3.0); // Input to layer 1 is output of layer 0
    ASSERT_TRUE(layer1_obj_map.count("layer_size"));
    EXPECT_DOUBLE_EQ(layer1_obj_map.at("layer_size")->GetNumber(), 1.0);
    ASSERT_TRUE(layer1_obj_map.count("weights"));
    EXPECT_EQ(layer1_obj_map.at("weights")->type, JsonValueType::Object);
    ASSERT_TRUE(layer1_obj_map.count("biases"));
    EXPECT_EQ(layer1_obj_map.at("biases")->type, JsonValueType::Object);
}

TEST_F(NeuroNetTest, SaveLoadStringActivations) {
    const std::string test_filename = "test_model_str_act.json";
    NeuroNet::NeuroNet original_model;
    original_model.SetInputSize(2);
    original_model.ResizeNeuroNet(2);

    original_model.ResizeLayer(0, 3);
    original_model.getLayer(0).SetActivationFunction(NeuroNet::ActivationFunctionType::ReLU);
    original_model.ResizeLayer(1, 1);
    original_model.getLayer(1).SetActivationFunction(NeuroNet::ActivationFunctionType::None);
    
    // Add minimal weights/biases for valid structure
    NeuroNet::LayerWeights lw0; lw0.WeightCount = 2*3; lw0.WeightsVector.assign(2*3, 0.1f);
    original_model.getLayer(0).SetWeights(lw0);
    NeuroNet::LayerBiases lb0; lb0.BiasCount = 3; lb0.BiasVector.assign(3, 0.01f);
    original_model.getLayer(0).SetBiases(lb0);

    NeuroNet::LayerWeights lw1; lw1.WeightCount = 3*1; lw1.WeightsVector.assign(3*1, 0.2f);
    original_model.getLayer(1).SetWeights(lw1);
    NeuroNet::LayerBiases lb1; lb1.BiasCount = 1; lb1.BiasVector.assign(1, 0.02f);
    original_model.getLayer(1).SetBiases(lb1);


    ASSERT_TRUE(original_model.save_model(test_filename));

    NeuroNet::NeuroNet loaded_model;
    ASSERT_NO_THROW(loaded_model = NeuroNet::NeuroNet::load_model(test_filename));

    ASSERT_EQ(loaded_model.getLayerCount(), 2);
    EXPECT_EQ(loaded_model.getLayer(0).get_activation_function_name(), "ReLU");
    EXPECT_EQ(loaded_model.getLayer(1).get_activation_function_name(), "None");
    
    // Ensure structure is otherwise intact
    EXPECT_EQ(loaded_model.getLayer(0).LayerSize(), 3);
    EXPECT_EQ(loaded_model.getLayer(1).LayerSize(), 1);

    std::remove(test_filename.c_str());
}

// --- NeuroNetJSONTest Tests ---

TEST(NeuroNetJSONTest, SetInputJSON_Valid) {
    NeuroNet::NeuroNet net;
    net.SetInputSize(3); // Configure network to accept 3 inputs
    net.ResizeNeuroNet(1); // Add one layer
    net.ResizeLayer(0, 2); // First layer has 2 neurons

    const std::string valid_json_input = R"({
        "input_matrix": [
            [1.0, 2.0, 3.0]
        ]
    })";

    ASSERT_NO_THROW(net.SetInputJSON(valid_json_input));

    // Optional: Further verify the input matrix within the network if NeuroNet had a GetInputMatrix() method.
    // For now, we rely on SetInput not throwing and GetOutput producing expected results later.
    Matrix::Matrix<float> output = net.GetOutput(); // Will use default weights/biases (usually zero)
    ASSERT_EQ(output.rows(), 1);
    ASSERT_EQ(output.cols(), 2);
}

TEST(NeuroNetJSONTest, SetInputJSON_MalformedJSON) {
    NeuroNet::NeuroNet net;
    net.SetInputSize(2);
    const std::string malformed_json = R"({ "input_matrix": [[1.0, 2.0], )"; // Missing closing bracket and brace
    ASSERT_THROW(net.SetInputJSON(malformed_json), JsonParseException);
}

TEST(NeuroNetJSONTest, SetInputJSON_InvalidStructure_NotAnObject) {
    NeuroNet::NeuroNet net;
    net.SetInputSize(2);
    const std::string wrong_structure_json = R"([[1.0, 2.0]])"; // Array instead of object
    ASSERT_THROW(net.SetInputJSON(wrong_structure_json), std::runtime_error);
}

TEST(NeuroNetJSONTest, SetInputJSON_InvalidStructure_MissingKey) {
    NeuroNet::NeuroNet net;
    net.SetInputSize(2);
    const std::string missing_key_json = R"({ "data": [[1.0, 2.0]] })";
    ASSERT_THROW(net.SetInputJSON(missing_key_json), std::runtime_error);
}

TEST(NeuroNetJSONTest, SetInputJSON_InvalidStructure_MatrixValueNotArray) {
    NeuroNet::NeuroNet net;
    net.SetInputSize(2);
    const std::string value_not_array_json = R"({ "input_matrix": "not an array" })";
    ASSERT_THROW(net.SetInputJSON(value_not_array_json), std::runtime_error);
}

TEST(NeuroNetJSONTest, SetInputJSON_InvalidStructure_RowNotArray) {
    NeuroNet::NeuroNet net;
    net.SetInputSize(2);
    const std::string row_not_array_json = R"({ "input_matrix": [[1.0, 2.0], "not a row array"] })";
    ASSERT_THROW(net.SetInputJSON(row_not_array_json), std::runtime_error);
}

TEST(NeuroNetJSONTest, SetInputJSON_DataTypeError) {
    NeuroNet::NeuroNet net;
    net.SetInputSize(2);
    const std::string data_type_error_json = R"({ "input_matrix": [[1.0, "not a number"]] })";
    ASSERT_THROW(net.SetInputJSON(data_type_error_json), std::runtime_error);
}

TEST(NeuroNetJSONTest, SetInputJSON_InconsistentColumns) {
    NeuroNet::NeuroNet net;
    net.SetInputSize(2); // Input size here is for the network, not directly enforced by SetInputJSON on matrix shape
    const std::string inconsistent_cols_json = R"({
        "input_matrix": [
            [1.0, 2.0],
            [3.0]
        ]
    })";
    ASSERT_THROW(net.SetInputJSON(inconsistent_cols_json), std::runtime_error);
}

TEST(NeuroNetJSONTest, GetOutputJSON_Basic) {
    NeuroNet::NeuroNet net;
    net.SetInputSize(1);
    net.ResizeNeuroNet(1);
    net.ResizeLayer(0, 1); // 1 input, 1 neuron layer

    // Manually set weights and biases for predictable output
    NeuroNet::LayerWeights weights;
    weights.WeightCount = 1; // 1 input * 1 neuron
    weights.WeightsVector.push_back(2.0f);
    ASSERT_TRUE(net.getLayer(0).SetWeights(weights));

    NeuroNet::LayerBiases biases;
    biases.BiasCount = 1; // 1 neuron
    biases.BiasVector.push_back(0.5f);
    ASSERT_TRUE(net.getLayer(0).SetBiases(biases));

    net.getLayer(0).SetActivationFunction(NeuroNet::ActivationFunctionType::None); // No activation

    const std::string input_json = R"({ "input_matrix": [[3.0]] })"; // Input X = 3.0
    ASSERT_NO_THROW(net.SetInputJSON(input_json));

    // Expected output: (2.0 * 3.0) + 0.5 = 6.5
    std::string output_json_str;
    ASSERT_NO_THROW(output_json_str = net.GetOutputJSON());

    // Parse the output JSON to verify its content
    JsonValue output_json_val;
    ASSERT_NO_THROW(output_json_val = JsonParser::Parse(output_json_str));

    ASSERT_EQ(output_json_val.type, JsonValueType::Object);
    ASSERT_TRUE(output_json_val.GetObject().count("output_matrix"));

    const JsonValue* matrix_val = output_json_val.GetObject().at("output_matrix");
    ASSERT_EQ(matrix_val->type, JsonValueType::Array);
    ASSERT_EQ(matrix_val->GetArray().size(), 1); // 1 row

    const JsonValue& row_val = matrix_val->GetArray()[0];
    ASSERT_EQ(row_val.type, JsonValueType::Array);
    ASSERT_EQ(row_val.GetArray().size(), 1); // 1 col

    const JsonValue& cell_val = row_val.GetArray()[0];
    ASSERT_EQ(cell_val.type, JsonValueType::Number);
    EXPECT_FLOAT_EQ(static_cast<float>(cell_val.GetNumber()), 6.5f);

    // Cleanup for parsed JSON object
    if (output_json_val.type == JsonValueType::Object) {
        for (auto& pair : output_json_val.GetObject()) {
            delete pair.second; // Delete the JsonValue*
        }
        output_json_val.GetObject().clear(); // Clear map to remove dangling pointers
    }
}

TEST(NeuroNetJSONTest, GetOutputJSON_NoLayers) {
    NeuroNet::NeuroNet net; // No layers added
    std::string output_json_str;
    ASSERT_NO_THROW(output_json_str = net.GetOutputJSON());

    JsonValue output_json_val;
    ASSERT_NO_THROW(output_json_val = JsonParser::Parse(output_json_str));
    ASSERT_EQ(output_json_val.type, JsonValueType::Object);
    ASSERT_TRUE(output_json_val.GetObject().count("output_matrix"));
    const JsonValue* matrix_val = output_json_val.GetObject().at("output_matrix");
    ASSERT_EQ(matrix_val->type, JsonValueType::Array);
    ASSERT_TRUE(matrix_val->GetArray().empty()); // Expect empty array for data

    // Cleanup for parsed JSON object
    if (output_json_val.type == JsonValueType::Object) {
        for (auto& pair : output_json_val.GetObject()) {
            delete pair.second;
        }
        output_json_val.GetObject().clear(); // Clear map
    }
}
