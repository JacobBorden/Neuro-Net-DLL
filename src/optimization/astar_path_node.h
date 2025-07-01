#pragma once

#include <functional> // For std::hash

namespace NeuroNet {
namespace Optimization {

/**
 * @brief Represents a node in the A* search, corresponding to a specific neuron
 *        in the neural network.
 */
struct AStarPathNode {
    int layer_idx;    ///< Index of the layer in the network.
    int neuron_idx;   ///< Index of the neuron within its layer.

    AStarPathNode(int l_idx = -1, int n_idx = -1) : layer_idx(l_idx), neuron_idx(n_idx) {}

    /**
     * @brief Equality operator.
     */
    bool operator==(const AStarPathNode& other) const {
        return layer_idx == other.layer_idx && neuron_idx == other.neuron_idx;
    }

    /**
     * @brief Inequality operator.
     */
    bool operator!=(const AStarPathNode& other) const {
        return !(*this == other);
    }

    /**
     * @brief Less-than operator, primarily for ordering in structures like std::map
     *        or for consistent ordering if needed, though priority queue will use its own comparator.
     *        Defines an arbitrary consistent order.
     */
    bool operator<(const AStarPathNode& other) const {
        if (layer_idx != other.layer_idx) {
            return layer_idx < other.layer_idx;
        }
        return neuron_idx < other.neuron_idx;
    }
};

} // namespace Optimization
} // namespace NeuroNet

// Custom hash function for AStarPathNode to be used with std::unordered_map/set
namespace std {
template <>
struct hash<NeuroNet::Optimization::AStarPathNode> {
    std::size_t operator()(const NeuroNet::Optimization::AStarPathNode& node) const noexcept {
        // A simple hash combination approach
        std::size_t h1 = std::hash<int>{}(node.layer_idx);
        std::size_t h2 = std::hash<int>{}(node.neuron_idx);
        return h1 ^ (h2 << 1); // Combine hashes. Left shift h2 to ensure bits are mixed well.
    }
};
} // namespace std
