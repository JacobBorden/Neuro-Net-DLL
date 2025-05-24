#include "gtest/gtest.h"
#include "neural_network/neuronet.h" // Access to NeuroNet and NeuroNetLayer
#include "math/matrix.h"   // For creating Matrix objects for input

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

// Main function for running tests (needed if not using gtest_main)
// int main(int argc, char **argv) {
//     ::testing::InitGoogleTest(&argc, argv);
//     return RUN_ALL_TESTS();
// }
