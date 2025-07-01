#include "neural_pathfinder.h"
#include <cmath>      // For std::log, std::abs
#include <limits>     // For std::numeric_limits
#include <stdexcept>  // For std::runtime_error
#include <algorithm>  // For std::reverse, std::max
#include <queue>      // For std::priority_queue
#include <map>        // For std::map (can be alt to unordered_map if issues with hash)

namespace NeuroNet {
namespace Optimization {

NeuralPathfinder::NeuralPathfinder(const NeuroNet::NeuroNet& network) : network_(network) {
    if (network_.getLayerCount() == 0) {
        throw std::runtime_error("NeuralPathfinder: Network has no layers.");
    }
}

// Helper function to reconstruct the path from came_from map
std::vector<AStarPathNode> NeuralPathfinder::ReconstructPath(
    const std::unordered_map<AStarPathNode, AStarPathNode, std::hash<AStarPathNode>>& came_from,
    AStarPathNode current_node) {

    std::vector<AStarPathNode> total_path;
    total_path.push_back(current_node);
    while (came_from.count(current_node)) {
        current_node = came_from.at(current_node);
        total_path.push_back(current_node);
    }
    std::reverse(total_path.begin(), total_path.end());
    return total_path;
}

// Heuristic function for A*
// h(node) = (num_remaining_layers_from_node_to_output) * min_possible_edge_cost
// min_possible_edge_cost = -log(max_abs_overall_weight + epsilon)
double NeuralPathfinder::CalculateHeuristic(const AStarPathNode& node, int goal_layer_idx, double min_single_edge_cost) const {
    if (node.layer_idx > goal_layer_idx) { // Should not happen if called correctly
        return 0.0;
    }
    int remaining_layers = goal_layer_idx - node.layer_idx;
    return static_cast<double>(remaining_layers) * min_single_edge_cost;
}

// Helper to get overall max absolute weight for heuristic calculation
double NeuralPathfinder::GetMaxAbsoluteWeight() const {
    double max_abs_weight = 0.0;
    bool weight_found = false;

    for (int i = 0; i < network_.getLayerCount(); ++i) {
        const auto& layer = network_.getLayer(i);

        // Determine the number of inputs to this layer to iterate prev_neuron_idx
        // For layer 0, input_size is network_.GetInputSize()
        // For layer > 0, input_size is network_.getLayer(i-1).LayerSize()
        // However, NeuroNetLayer::get_input_size() should give the correct number of rows for its WeightMatrix.
        int current_layer_input_size = layer.get_input_size();
        int current_layer_neuron_count = layer.LayerSize();

        if (current_layer_input_size == 0 || current_layer_neuron_count == 0) {
            continue; // Skip layers with no inputs or no neurons, as they won't have weights.
        }

        for (int prev_n_idx = 0; prev_n_idx < current_layer_input_size; ++prev_n_idx) {
            for (int curr_n_idx = 0; curr_n_idx < current_layer_neuron_count; ++curr_n_idx) {
                try {
                    float weight = layer.get_weight(prev_n_idx, curr_n_idx);
                    max_abs_weight = std::max(max_abs_weight, std::abs(static_cast<double>(weight)));
                    weight_found = true;
                } catch (const std::out_of_range& e) {
                    // This might happen if layer configuration is unusual or if there's a bug.
                    // For robustness, could log this error. For now, skip this weight.
                    // std::cerr << "Warning: Out of range accessing weight in GetMaxAbsoluteWeight: " << e.what() << std::endl;
                    continue;
                }
            }
        }
    }

    if (!weight_found) {
        // If there are no weights in the network (e.g. all layers are size 0, or no connections)
        // return EPSILON to avoid log(0) issues. The cost -log(EPSILON) will be high.
        return EPSILON;
    }
    // If max_abs_weight is still 0 (all weights are zero), return EPSILON to ensure -log(max_abs_weight + EPSILON) is defined.
    return (max_abs_weight == 0.0) ? EPSILON : max_abs_weight;
}


std::vector<AStarPathNode> NeuralPathfinder::FindOptimalPathAStar() {
    if (network_.getLayerCount() < 1) { // Need at least one layer.
        // If network has 0 layers, constructor should have thrown.
        // This check is for safety or if definition of path changes.
        return {};
    }
    // Path from first hidden layer (layer 0) to output layer (layer_count - 1).
    // If only 1 layer, it's both start and end. Path is just a single node.

    // A* data structures
    std::priority_queue<std::pair<double, AStarPathNode>,
                        std::vector<std::pair<double, AStarPathNode>>,
                        std::greater<std::pair<double, AStarPathNode>>> open_set;

    std::unordered_map<AStarPathNode, AStarPathNode, std::hash<AStarPathNode>> came_from;
    std::unordered_map<AStarPathNode, double, std::hash<AStarPathNode>> g_score;

    // --- Initialization for Heuristic ---
    double max_abs_overall_weight = GetMaxAbsoluteWeight();
    // min_single_edge_cost will be -log(max_abs_overall_weight + EPSILON)
    // If max_abs_overall_weight is EPSILON (meaning no weights or all zero), then cost is -log(2*EPSILON)
    double min_single_edge_cost = -std::log(max_abs_overall_weight + EPSILON);


    // --- A* Algorithm ---
    int start_layer_idx = 0;
    int goal_layer_idx = network_.getLayerCount() - 1;

    // Initialize g_score for all potential nodes to infinity
    // Not strictly necessary to pre-fill g_score, can check map.count() instead.
    // Using map.count() or find() is generally cleaner.

    const auto& first_layer_ref = network_.getLayer(start_layer_idx);
    int num_neurons_in_start_layer = first_layer_ref.LayerSize();

    if (num_neurons_in_start_layer == 0 && network_.getLayerCount() > 0) {
        return {}; // Start layer has no neurons.
    }

    // If network has only one layer, any neuron in it is a "path" of length 1.
    // A* will correctly handle this: start_node is goal_node.
    // The loop for neighbors won't run if current_node.layer_idx == goal_layer_idx.

    for (int i = 0; i < num_neurons_in_start_layer; ++i) {
        AStarPathNode start_node(start_layer_idx, i);
        g_score[start_node] = 0.0; // Cost to reach start_node from itself is 0
        double h_val = CalculateHeuristic(start_node, goal_layer_idx, min_single_edge_cost);
        open_set.push({0.0 + h_val, start_node}); // f_score = g_score + h_score
    }

    while (!open_set.empty()) {
        AStarPathNode current_node = open_set.top().second;
        // double current_node_f_score_in_pq = open_set.top().first; // For debugging
        open_set.pop();

        // If this node was already processed with a better or equal path, skip.
        // This check is particularly useful if we can update priorities in PQ,
        // but with std::priority_queue, we add duplicates.
        // Check against g_score which stores the best known path cost.
        // double current_node_h_val = CalculateHeuristic(current_node, goal_layer_idx, min_single_edge_cost);
        // if (current_node_f_score_in_pq > g_score.at(current_node) + current_node_h_val + EPSILON) { // Add EPSILON for float comparisons
        //    continue; // Stale entry in PQ
        // }
        // A simpler check: if a node is popped, its g_score is final for standard A* with consistent heuristic.

        if (current_node.layer_idx == goal_layer_idx) {
            return ReconstructPath(came_from, current_node); // Goal reached
        }

        // Explore neighbors (neurons in the next layer)
        int next_layer_idx = current_node.layer_idx + 1;
        // This check should be redundant if goal_layer_idx logic is correct, but good for safety.
        if (next_layer_idx > goal_layer_idx || next_layer_idx >= network_.getLayerCount()) {
            continue;
        }

        const auto& next_layer_ref = network_.getLayer(next_layer_idx);
        int num_neurons_in_next_layer = next_layer_ref.LayerSize();
        if (num_neurons_in_next_layer == 0) { // Next layer has no neurons to connect to
            continue;
        }

        // The input to 'next_layer_ref' is the output of 'current_node.layer_idx'.
        // The 'prev_neuron_idx' for next_layer_ref.get_weight is current_node.neuron_idx.
        for (int neighbor_neuron_idx_in_layer = 0; neighbor_neuron_idx_in_layer < num_neurons_in_next_layer; ++neighbor_neuron_idx_in_layer) {
            AStarPathNode neighbor_node(next_layer_idx, neighbor_neuron_idx_in_layer);
            double weight_val;
            try {
                weight_val = static_cast<double>(
                    next_layer_ref.get_weight(current_node.neuron_idx, neighbor_neuron_idx_in_layer)
                );
            } catch (const std::out_of_range& e) {
                // Should not happen if layer sizes and indices are correct.
                // std::cerr << "Error accessing weight: " << e.what() << std::endl;
                continue; // Skip this potential connection
            }

            double edge_cost = -std::log(std::abs(weight_val) + EPSILON);
            double tentative_g_score = g_score.at(current_node) + edge_cost;

            // Check if this path to neighbor is better or if neighbor hasn't been visited
            if (g_score.find(neighbor_node) == g_score.end() || tentative_g_score < g_score.at(neighbor_node)) {
                came_from[neighbor_node] = current_node;
                g_score[neighbor_node] = tentative_g_score;
                double h_val_neighbor = CalculateHeuristic(neighbor_node, goal_layer_idx, min_single_edge_cost);
                open_set.push({tentative_g_score + h_val_neighbor, neighbor_node});
            }
        }
    }

    return {}; // No path found
}

} // namespace Optimization
} // namespace NeuroNet
