#include "gtest/gtest.h"
#include "optimization/neural_pathfinder.h"
#include "neural_network/neuronet.h"

using namespace NeuroNet;
using namespace NeuroNet::Optimization;

class NeuralPathfinderTest : public ::testing::Test {
protected:
    void SetUp() override {
        // Common setup if needed
    }
};

TEST_F(NeuralPathfinderTest, EmptyNetwork_ThrowsException) {
    NeuroNet::NeuroNet net;
    // Network is empty by default (0 layers)
    EXPECT_THROW({
        NeuralPathfinder pathfinder(net);
        pathfinder.FindOptimalPathAStar();
    }, std::runtime_error);
}

TEST_F(NeuralPathfinderTest, SingleLayerNetwork_ReturnsSingleNode) {
    NeuroNet::NeuroNet net;
    net.SetInputSize(2);
    net.ResizeNeuroNet(1);
    net.ResizeLayer(0, 3); // 1 layer, 3 neurons

    NeuralPathfinder pathfinder(net);
    auto path = pathfinder.FindOptimalPathAStar();

    ASSERT_EQ(path.size(), 1);
    EXPECT_EQ(path[0].layer_idx, 0);
    // Since weights are zero initially, it will probably pick the first node
    EXPECT_EQ(path[0].neuron_idx, 0);
}

TEST_F(NeuralPathfinderTest, MultiLayerNetwork_FindsCorrectPath) {
    NeuroNet::NeuroNet net;
    net.SetInputSize(2);
    net.ResizeNeuroNet(3);
    net.ResizeLayer(0, 2);
    net.ResizeLayer(1, 2);
    net.ResizeLayer(2, 2);

    // Initialize all weights to 0.1 to avoid 0s and have a baseline
    std::vector<float> flat_weights(net.get_all_weights_flat().size(), 0.1f);

    // Let's set a specific path to have large weights.
    // The weights vector format is sequential by layer, then by (prev_neuron * current_layer_size + current_neuron)

    // Layer 0 size is 2, input size is 2. (4 weights)
    // Layer 1 size is 2, input size is 2. (4 weights)
    // Layer 2 size is 2, input size is 2. (4 weights)
    // Total weights: 12

    // Set weight for path: Layer 0, Node 1 -> Layer 1, Node 0 -> Layer 2, Node 1
    // We don't have a direct setter for specific weights, so let's modify the flat array
    // Layer 0: Input(size 2) to Layer0(size 2). Let's make node 1 a good start? Wait, A* doesn't evaluate input-to-Layer0 weights.
    // A* evaluates weights from Layer 0 to Layer 1, and Layer 1 to Layer 2.
    // So the heuristic looks at the weights *between* the network's layers.
    // The path starts AT Layer 0. So we care about Layer 0 -> Layer 1 weights.

    // Layer 0 -> Layer 1 weights (4 of them, starting at index 4)
    // prev_n_idx (Layer 0) -> curr_n_idx (Layer 1)
    // L0_0 -> L1_0: index 4
    // L0_0 -> L1_1: index 5
    // L0_1 -> L1_0: index 6
    // L0_1 -> L1_1: index 7

    flat_weights[6] = 10.0f; // L0_Node1 to L1_Node0 has large weight

    // Layer 1 -> Layer 2 weights (4 of them, starting at index 8)
    // L1_0 -> L2_0: index 8
    // L1_0 -> L2_1: index 9
    // L1_1 -> L2_0: index 10
    // L1_1 -> L2_1: index 11

    flat_weights[9] = 10.0f; // L1_Node0 to L2_Node1 has large weight

    net.set_all_weights_flat(flat_weights);

    NeuralPathfinder pathfinder(net);
    auto path = pathfinder.FindOptimalPathAStar();

    ASSERT_EQ(path.size(), 3);
    EXPECT_EQ(path[0].layer_idx, 0);
    EXPECT_EQ(path[0].neuron_idx, 1);

    EXPECT_EQ(path[1].layer_idx, 1);
    EXPECT_EQ(path[1].neuron_idx, 0);

    EXPECT_EQ(path[2].layer_idx, 2);
    EXPECT_EQ(path[2].neuron_idx, 1);
}

TEST_F(NeuralPathfinderTest, AllZeroWeights_FindsAnyPath) {
    NeuroNet::NeuroNet net;
    net.SetInputSize(2);
    net.ResizeNeuroNet(2);
    net.ResizeLayer(0, 2);
    net.ResizeLayer(1, 2);

    std::vector<float> flat_weights(net.get_all_weights_flat().size(), 0.0f);
    net.set_all_weights_flat(flat_weights);

    NeuralPathfinder pathfinder(net);
    auto path = pathfinder.FindOptimalPathAStar();

    ASSERT_EQ(path.size(), 2);
    // As long as it returns a valid path, it's correct.
    EXPECT_EQ(path[0].layer_idx, 0);
    EXPECT_EQ(path[1].layer_idx, 1);
}
