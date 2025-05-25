/**
 * @file neuronet.cpp
 * @author Jacob Borden (amenra.beats@gmail.com)
 * @brief Implements the NeuroNet and NeuroNetLayer classes for the neural network library.
 * @version 0.2.0
 * @date 2023-10-27 (Last major update date)
 *
 * @copyright Copyright (c) 2021-2023 Jacob Borden
 *
 */

#include "neuronet.h"
#include <stdexcept> // For std::runtime_error or other exceptions if needed
#include <cmath>     // For std::exp and std::max
// #include "pch.h" // Precompiled header (if used, ensure it's appropriate for the project) - REMOVED

// --- NeuroNetLayer Method Implementations ---

NeuroNet::NeuroNetLayer::NeuroNetLayer() : vLayerSize(0), InputSize(0), vActivationFunction(ActivationFunctionType::None) {
    // Constructor body can remain empty if all initialization is done by member initializers
    // or if default member initializers in the header are sufficient.
}

NeuroNet::NeuroNetLayer::~NeuroNetLayer() {
    // Destructor body - typically empty unless managing raw pointers or other resources.
}

void NeuroNet::NeuroNetLayer::SetActivationFunction(ActivationFunctionType pActivationFunction) {
	this->vActivationFunction = pActivationFunction;
}

/**
 * @brief Resizes the layer and its internal matrices.
 *
 * Also initializes the Weights and Biases structs with correct counts.
 * The actual vector data within Weights and Biases is not initialized here,
 * it's expected to be set via SetWeights/SetBiases or through a random initialization process.
 */
void NeuroNet::NeuroNetLayer::ResizeLayer(int pInputSize, int pLayerSize) {
	this->vLayerSize = pLayerSize;
	this->InputSize = pInputSize;
	// Input matrix is assumed to be 1 row by pInputSize columns.
	this->InputMatrix.resize(1, this->InputSize);
	// Weight matrix is pInputSize rows by pLayerSize columns.
	this->WeightMatrix.resize(this->InputSize, this->vLayerSize);
	// Bias matrix is 1 row by pLayerSize columns.
	this->BiasMatrix.resize(1, this->vLayerSize);
	// Output matrix is 1 row by pLayerSize columns.
	this->OutputMatrix.resize(1, this->vLayerSize);

	// Update weight and bias counts. The actual vectors need to be populated by SetWeights/SetBiases.
	this->Weights.WeightCount = this->vLayerSize * this->InputSize;
	this->Weights.WeightsVector.clear(); // Clear old data; new data must be set explicitly.
	this->Biases.BiasCount = this->vLayerSize;
	this->Biases.BiasVector.clear(); // Clear old data.
}

Matrix::Matrix<float> NeuroNet::NeuroNetLayer::ApplyReLU(const Matrix::Matrix<float>& input) {
    Matrix::Matrix<float> output = input; // Make a copy
    for (int i = 0; i < output.rows(); ++i) {
        for (int j = 0; j < output.cols(); ++j) {
            output[i][j] = std::max(0.0f, output[i][j]);
        }
    }
    return output;
}

Matrix::Matrix<float> NeuroNet::NeuroNetLayer::ApplyLeakyReLU(const Matrix::Matrix<float>& input) {
    Matrix::Matrix<float> output = input; // Make a copy
    const float alpha = 0.01f;
    for (int i = 0; i < output.rows(); ++i) {
        for (int j = 0; j < output.cols(); ++j) {
            if (output[i][j] < 0) {
                output[i][j] = alpha * output[i][j];
            }
        }
    }
    return output;
}

Matrix::Matrix<float> NeuroNet::NeuroNetLayer::ApplyELU(const Matrix::Matrix<float>& input) {
    Matrix::Matrix<float> output = input; // Make a copy
    const float alpha = 1.0f;
    for (int i = 0; i < output.rows(); ++i) {
        for (int j = 0; j < output.cols(); ++j) {
            if (output[i][j] < 0) {
                output[i][j] = alpha * (std::exp(output[i][j]) - 1.0f);
            }
        }
    }
    return output;
}

Matrix::Matrix<float> NeuroNet::NeuroNetLayer::ApplySoftmax(const Matrix::Matrix<float>& input) {
    Matrix::Matrix<float> output = input; // Make a copy
    float sum_exp = 0.0f;
    // Calculate sum of exponents for normalization.
    // This implementation assumes input is a 1xN matrix (a single row vector),
    // which is typical for the output of a layer before activation.
    if (output.rows() != 1) {
        // For a more general Softmax that could operate column-wise on a batch of outputs,
        // this logic would need to be adjusted. For now, it processes a single output vector.
    }

    for (int j = 0; j < output.cols(); ++j) {
        output[0][j] = std::exp(output[0][j]);
        sum_exp += output[0][j];
    }

    // Normalize
    if (sum_exp != 0.0f) { // Avoid division by zero
        for (int j = 0; j < output.cols(); ++j) {
            output[0][j] /= sum_exp;
        }
    }
    return output;
}


Matrix::Matrix<float> NeuroNet::NeuroNetLayer::CalculateOutput() {
	if (InputMatrix.cols() == 0 || WeightMatrix.rows() == 0 || BiasMatrix.cols() == 0) {
		// Or throw an exception, or return an empty matrix with error status
		// For now, returning the current (possibly uninitialized) OutputMatrix.
		// Consider adding error handling or preconditions.
	}
	// Calculate the linear transformation part: (InputMatrix * WeightMatrix) + BiasMatrix
	this->OutputMatrix = (this->InputMatrix * this->WeightMatrix) + this->BiasMatrix;

    // OutputMatrix now holds the result of the linear transformation.
    // Apply the selected activation function.
    switch (this->vActivationFunction) {
        case ActivationFunctionType::ReLU:
            this->OutputMatrix = ApplyReLU(this->OutputMatrix);
            break;
        case ActivationFunctionType::LeakyReLU:
            this->OutputMatrix = ApplyLeakyReLU(this->OutputMatrix);
            break;
        case ActivationFunctionType::ELU:
            this->OutputMatrix = ApplyELU(this->OutputMatrix);
            break;
        case ActivationFunctionType::Softmax:
            this->OutputMatrix = ApplySoftmax(this->OutputMatrix);
            break;
        case ActivationFunctionType::None:
            // No activation function applied, do nothing.
            break;
        default:
            // Optional: Handle unknown activation type, e.g., log a warning or do nothing.
            break;
    }

	return this->OutputMatrix;
}

Matrix::Matrix<float> NeuroNet::NeuroNetLayer::ReturnOutputMatrix() {
	return this->OutputMatrix;
}

bool NeuroNet::NeuroNetLayer::SetInput(const Matrix::Matrix<float>& pInputMatrix) {
	// Expecting a 1xN matrix where N is InputSize.
	if (pInputMatrix.rows() != 1 || pInputMatrix.cols() != this->InputMatrix.cols()) {
		return false; // Input dimensions do not match layer's expected input dimensions.
	}
	this->InputMatrix = pInputMatrix;
	return true;
}

int NeuroNet::NeuroNetLayer::WeightCount() {
	return this->Weights.WeightCount;
}

int NeuroNet::NeuroNetLayer::BiasCount() {
	return this->Biases.BiasCount;
}

int NeuroNet::NeuroNetLayer::LayerSize() {
	return this->vLayerSize;
}

/**
 * @brief Sets the layer's weights from a LayerWeights struct.
 *
 * The internal WeightMatrix is updated based on the values in pWeights.WeightsVector.
 * The values are mapped from the flat vector to the 2D WeightMatrix.
 */
bool NeuroNet::NeuroNetLayer::SetWeights(LayerWeights pWeights) {
	if (pWeights.WeightCount != this->Weights.WeightCount || 
		pWeights.WeightsVector.size() != static_cast<size_t>(this->Weights.WeightCount)) {
		return false; // Mismatch in expected number of weights.
	}
	this->Weights = pWeights; // Store the provided weights struct.

	// Populate the internal WeightMatrix from the WeightsVector.
	// Assumes row-major order for weights in the vector if mapping to a 2D matrix.
	// Or, more directly, the Matrix class might handle this if it can be constructed from a flat vector.
	// Here, we map it assuming WeightMatrix is InputSize x vLayerSize.
	int k = 0; // Index for the flat WeightsVector.
	for (int i = 0; i < this->WeightMatrix.rows(); i++) { // Iterating through rows (inputs)
		for (int j = 0; j < this->WeightMatrix.cols(); j++) { // Iterating through columns (neurons)
			if (k < this->Weights.WeightCount) {
				this->WeightMatrix[i][j] = this->Weights.WeightsVector[k];
				k++;
			} else {
				// This case should not be reached if WeightCount and vector size are validated.
				return false; // Error: not enough weights in vector.
			}
		}
	}
	return true;
}

/**
 * @brief Sets the layer's biases from a LayerBiases struct.
 *
 * The internal BiasMatrix is updated based on the values in pBiases.BiasVector.
 */
bool NeuroNet::NeuroNetLayer::SetBiases(LayerBiases pBiases) {
	if (pBiases.BiasCount != this->Biases.BiasCount ||
		pBiases.BiasVector.size() != static_cast<size_t>(this->Biases.BiasCount)) {
		return false; // Mismatch in expected number of biases.
	}
	this->Biases = pBiases; // Store the provided biases struct.

	// Populate the internal BiasMatrix from the BiasVector.
	// BiasMatrix is 1 x vLayerSize.
	int k = 0; // Index for the flat BiasVector.
	for (int i = 0; i < this->BiasMatrix.rows(); i++) { // Should only be 1 row.
		for (int j = 0; j < this->BiasMatrix.cols(); j++) { // Iterating through columns (neurons)
			if (k < this->Biases.BiasCount) {
				this->BiasMatrix[i][j] = this->Biases.BiasVector[k];
				k++;
			} else {
				// This case should not be reached if BiasCount and vector size are validated.
				return false; // Error: not enough biases in vector.
			}
		}
	}
	return true;
}

// --- NeuroNet Method Implementations ---

NeuroNet::NeuroNet::NeuroNet() : InputSize(0), LayerCount(0) {
    // Default constructor
}

NeuroNet::NeuroNet::NeuroNet(int pLayerCount) : InputSize(0), LayerCount(pLayerCount) {
	this->NeuroNetVector.resize(pLayerCount);
    // Note: Layers are default-constructed and need further configuration (e.g., ResizeLayer).
}

NeuroNet::NeuroNet::~NeuroNet() {
    // Destructor
}

bool NeuroNet::NeuroNet::ResizeLayer(int pLayerIndex, int pLayerSize) {
	if (pLayerIndex < 0 || static_cast<size_t>(pLayerIndex) >= this->NeuroNetVector.size()) {
		return false; // Index out of bounds.
	}

	int currentLayerInputSize;
	if (pLayerIndex > 0) {
		// Input size for this layer is the output size (LayerSize) of the previous layer.
		currentLayerInputSize = this->NeuroNetVector[pLayerIndex - 1].LayerSize();
	} else {
		// Input size for the first layer (index 0) is the network's overall input size.
		currentLayerInputSize = this->InputSize;
	}
    
    if (currentLayerInputSize == 0 && pLayerIndex > 0) {
        // Previous layer is not configured, cannot reliably set current layer's input size.
        // Or, if InputSize for the network is 0 for the first layer.
        // Consider how to handle this: error, or allow it if user configures later.
        // For now, proceed, but this layer might not be usable until previous is sized.
    }

	this->NeuroNetVector[pLayerIndex].ResizeLayer(currentLayerInputSize, pLayerSize);

	// If this is not the last layer, update the input size of the next layer.
	if (static_cast<size_t>(pLayerIndex + 1) < this->NeuroNetVector.size()) {
		// The next layer's input size is the current layer's output size (pLayerSize).
        // We need to re-call ResizeLayer on the next layer to update its internal matrices.
        // This creates a cascading resize if layer sizes change.
        int nextLayerCurrentOutputSize = this->NeuroNetVector[pLayerIndex + 1].LayerSize();
		this->NeuroNetVector[pLayerIndex + 1].ResizeLayer(pLayerSize, nextLayerCurrentOutputSize);
	}
	return true;
}

void NeuroNet::NeuroNet::SetInputSize(int pInputSize) {
	this->InputSize = pInputSize;
	// If there's at least one layer, its input size needs to be updated.
	if (!this->NeuroNetVector.empty()) {
        int firstLayerCurrentOutputSize = this->NeuroNetVector[0].LayerSize();
		this->NeuroNetVector[0].ResizeLayer(this->InputSize, firstLayerCurrentOutputSize);
        // If the first layer's output size changes as a result of some internal logic in ResizeLayer (it shouldn't here),
        // then subsequent layers might also need updating. The current ResizeLayer handles this cascade.
	}
}

void NeuroNet::NeuroNet::ResizeNeuroNet(int pLayerCount) {
    if (pLayerCount < 0) return; // Or throw error
	this->NeuroNetVector.resize(pLayerCount);
    this->LayerCount = pLayerCount; // Update the LayerCount member
}

bool NeuroNet::NeuroNet::SetInput(const Matrix::Matrix<float>& pInputMatrix) {
	if (this->NeuroNetVector.empty()) {
		return false; // No layers to process input.
	}
	return this->NeuroNetVector[0].SetInput(pInputMatrix);
}

Matrix::Matrix<float> NeuroNet::NeuroNet::GetOutput() {
	if (this->NeuroNetVector.empty()) {
		return Matrix::Matrix<float>(); // Return an empty matrix if no layers.
	}
    if (this->LayerCount == 0 && !this->NeuroNetVector.empty()){
        // This state implies NeuroNetVector was resized but LayerCount wasn't updated.
        // This indicates an internal inconsistency. For safety, use NeuroNetVector.size().
        // However, the design intends LayerCount to be the authority.
        // This situation should be fixed by ensuring LayerCount is always consistent.
        // For now, let's trust NeuroNetVector for iteration if LayerCount is 0 but vector isn't empty.
        // A better fix is to ensure LayerCount is always accurate.
        // The ResizeNeuroNet method should be the primary way to change layer count.
    }


	// Process first layer
	this->NeuroNetVector[0].CalculateOutput();

	// Process subsequent layers
    // Use NeuroNetVector.size() for safety if LayerCount might be out of sync.
    // However, the design relies on LayerCount. If ResizeNeuroNet is used correctly, they should match.
	for (size_t i = 1; i < this->NeuroNetVector.size(); i++) { // Iterate up to actual number of layers present
		this->NeuroNetVector[i].SetInput(this->NeuroNetVector[i - 1].ReturnOutputMatrix());
		this->NeuroNetVector[i].CalculateOutput();
	}
	return this->NeuroNetVector.back().ReturnOutputMatrix(); // Output of the last layer
}

// --- Helper Method Implementations for NeuroNetLayer ---
// These are defined in neuronet.h as they are simple getters,
// but if they had more complex logic, they'd be here.
// For Doxygen, their detailed comments are in the header.

NeuroNet::LayerWeights NeuroNet::NeuroNetLayer::get_weights() const {
    LayerWeights current_weights_struct;
    // Use the WeightCount already stored in this->Weights, which ResizeLayer correctly sets.
    current_weights_struct.WeightCount = this->Weights.WeightCount; 
    if (this->WeightMatrix.rows() * this->WeightMatrix.cols() != current_weights_struct.WeightCount) {
        // Optional: Add error handling or log if counts mismatch,
        // but this->Weights.WeightCount should be authoritative if ResizeLayer is always used.
    }

    current_weights_struct.WeightsVector.clear(); // Ensure vector is empty before filling
    current_weights_struct.WeightsVector.reserve(current_weights_struct.WeightCount);

    for (int i = 0; i < this->WeightMatrix.rows(); ++i) {
        for (int j = 0; j < this->WeightMatrix.cols(); ++j) {
            current_weights_struct.WeightsVector.push_back(this->WeightMatrix[i][j]);
        }
    }
    return current_weights_struct;
}

NeuroNet::LayerBiases NeuroNet::NeuroNetLayer::get_biases() const {
    LayerBiases current_biases_struct;
    // Use the BiasCount already stored in this->Biases, which ResizeLayer correctly sets.
    current_biases_struct.BiasCount = this->Biases.BiasCount;
    if (this->BiasMatrix.cols() != current_biases_struct.BiasCount && this->BiasMatrix.rows() == 1) {
         // Optional: Add error handling or log if counts mismatch
    }

    current_biases_struct.BiasVector.clear(); // Ensure vector is empty before filling
    current_biases_struct.BiasVector.reserve(current_biases_struct.BiasCount);

    // BiasMatrix is 1xN (1 row, N columns where N is number of neurons/biases)
    for (int j = 0; j < this->BiasMatrix.cols(); ++j) {
        current_biases_struct.BiasVector.push_back(this->BiasMatrix[0][j]);
    }
    return current_biases_struct;
}

// --- Helper Method Implementations for NeuroNet ---

std::vector<NeuroNet::LayerWeights> NeuroNet::NeuroNet::get_all_layer_weights() {
	std::vector<LayerWeights> all_weights;
	all_weights.reserve(this->NeuroNetVector.size());
	for (auto& layer : this->NeuroNetVector) {
		all_weights.push_back(layer.get_weights());
	}
	return all_weights;
}

bool NeuroNet::NeuroNet::set_all_layer_weights(const std::vector<LayerWeights>& all_weights) {
	if (all_weights.size() != this->NeuroNetVector.size()) {
		return false; // Mismatch in the number of layers.
	}
	for (size_t i = 0; i < this->NeuroNetVector.size(); ++i) {
		if (!this->NeuroNetVector[i].SetWeights(all_weights[i])) {
			// Log error or handle partial update? For now, return false on first failure.
			return false; // Failed to set weights for a layer.
		}
	}
	return true;
}

std::vector<NeuroNet::LayerBiases> NeuroNet::NeuroNet::get_all_layer_biases() {
	std::vector<LayerBiases> all_biases;
	all_biases.reserve(this->NeuroNetVector.size());
	for (auto& layer : this->NeuroNetVector) {
		all_biases.push_back(layer.get_biases());
	}
	return all_biases;
}

bool NeuroNet::NeuroNet::set_all_layer_biases(const std::vector<LayerBiases>& all_biases) {
	if (all_biases.size() != this->NeuroNetVector.size()) {
		return false; // Mismatch in the number of layers.
	}
	for (size_t i = 0; i < this->NeuroNetVector.size(); ++i) {
		if (!this->NeuroNetVector[i].SetBiases(all_biases[i])) {
			return false; // Failed to set biases for a layer.
		}
	}
	return true;
}

std::vector<float> NeuroNet::NeuroNet::get_all_weights_flat() const {
	std::vector<float> flat_weights;
	for (const auto& layer : this->NeuroNetVector) {
		LayerWeights lw = layer.get_weights();
		if (lw.WeightCount > 0 && !lw.WeightsVector.empty()) {
			flat_weights.insert(flat_weights.end(), lw.WeightsVector.begin(), lw.WeightsVector.end());
		}
	}
	return flat_weights;
}

bool NeuroNet::NeuroNet::set_all_weights_flat(const std::vector<float>& all_weights_flat) {
	size_t current_idx = 0;
	for (auto& layer : this->NeuroNetVector) {
		LayerWeights current_lw_template = layer.get_weights(); // Used to get expected WeightCount
		int expected_layer_weight_count = layer.WeightCount(); // More direct
        
        // If get_weights() doesn't fill WeightsVector but only WeightCount,
        // this logic needs layer.WeightCount() to determine how many weights to take.
        // The current LayerWeights struct has both.

		if (current_idx + expected_layer_weight_count > all_weights_flat.size()) {
			return false; // Not enough weights in the flat vector for this layer.
		}

		LayerWeights new_lw;
		new_lw.WeightCount = expected_layer_weight_count;
        if (expected_layer_weight_count > 0) {
		    new_lw.WeightsVector.assign(all_weights_flat.begin() + current_idx, 
                                    all_weights_flat.begin() + current_idx + expected_layer_weight_count);
        } else {
            new_lw.WeightsVector.clear();
        }

		if (!layer.SetWeights(new_lw)) {
			return false; // Failed to set weights for the current layer.
		}
		current_idx += expected_layer_weight_count;
	}

	if (current_idx != all_weights_flat.size()) {
		// This means the flat vector had more weights than the network expected, or fewer were consumed.
		return false;
	}
	return true;
}

std::vector<float> NeuroNet::NeuroNet::get_all_biases_flat() const {
	std::vector<float> flat_biases;
	for (const auto& layer : this->NeuroNetVector) {
		LayerBiases lb = layer.get_biases();
		if (lb.BiasCount > 0 && !lb.BiasVector.empty()) {
			flat_biases.insert(flat_biases.end(), lb.BiasVector.begin(), lb.BiasVector.end());
		}
	}
	return flat_biases;
}

bool NeuroNet::NeuroNet::set_all_biases_flat(const std::vector<float>& all_biases_flat) {
	size_t current_idx = 0;
	for (auto& layer : this->NeuroNetVector) {
		LayerBiases current_lb_template = layer.get_biases(); // Used to get expected BiasCount
        int expected_layer_bias_count = layer.BiasCount();

		if (current_idx + expected_layer_bias_count > all_biases_flat.size()) {
			return false; // Not enough biases in the flat vector for this layer.
		}

		LayerBiases new_lb;
		new_lb.BiasCount = expected_layer_bias_count;
        if (expected_layer_bias_count > 0) {
		    new_lb.BiasVector.assign(all_biases_flat.begin() + current_idx, 
                                 all_biases_flat.begin() + current_idx + expected_layer_bias_count);
        } else {
            new_lb.BiasVector.clear();
        }
        
		if (!layer.SetBiases(new_lb)) {
			return false; // Failed to set biases for the current layer.
		}
		current_idx += expected_layer_bias_count;
	}

	if (current_idx != all_biases_flat.size()) {
		// This means the flat vector had more biases than the network expected.
		return false;
	}
	return true;
}
