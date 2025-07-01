#pragma once

#include "../neural_network/neuronet.h" // For NeuroNet class
#include "astar_path_node.h"          // For AStarPathNode
#include <vector>
#include <string>

namespace NeuroNet {
namespace Optimization {

/**
 * @brief Finds optimal paths through a NeuroNet model using A* search.
 *
 * The "optimality" is defined by minimizing a cost function, which can be
 * configured to achieve goals like maximizing the product of weights along a path.
 */
class NeuralPathfinder {
public:
    /**
     * @brief Constructs a NeuralPathfinder for a given neural network.
     * @param network A const reference to the NeuroNet model to analyze.
     */
    NeuralPathfinder(const NeuroNet::NeuroNet& network);

    /**
     * @brief Finds the optimal path based on maximizing the product of absolute weights.
     *
     * The path is a sequence of neurons, one from each layer, starting from
     * the first hidden layer (layer 0) to any neuron in the output layer.
     *
     * @return A vector of AStarPathNode representing the neurons in the optimal path.
     *         Returns an empty vector if no path is found or if the network is unsuitable
     *         (e.g., too few layers).
     * @throws std::runtime_error if the network structure is invalid (e.g., no layers).
     */
    std::vector<AStarPathNode> FindOptimalPathAStar();

private:
    const NeuroNet::NeuroNet& network_; // Reference to the neural network
    static constexpr double EPSILON = 1e-9; // Small constant for log calculations

    // Helper function to reconstruct the path from came_from map
    std::vector<AStarPathNode> ReconstructPath(
        const std::unordered_map<AStarPathNode, AStarPathNode, std::hash<AStarPathNode>>& came_from,
        AStarPathNode current_node);

    // Heuristic function for A*
    double CalculateHeuristic(const AStarPathNode& node, int goal_layer_idx, double min_single_edge_cost) const;

    // Helper to get overall max absolute weight for heuristic calculation
    double GetMaxAbsoluteWeight() const;
};

} // namespace Optimization
} // namespace NeuroNet
