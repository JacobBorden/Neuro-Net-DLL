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
#include <fstream> // For std::ofstream
#include <vector>  // For std::vector, though often included via neuronet.h indirectly
#include <iostream>  // For std::cout (used in benchmarking)
#include "../utilities/timer.h" // For Timer class
#include "../utilities/json/json.hpp"
#include "../utilities/json/json_exception.hpp" // Added for JsonParseException

// Define ENABLE_BENCHMARKING to enable timing of neural network operations.
// This can be defined in project settings or uncommented here for testing.
// #define ENABLE_BENCHMARKING

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

	// Also resize gradient matrices
	this->dLdW.resize(this->InputSize, this->vLayerSize);
	this->dLdW.assign(0.0f); // Initialize with zeros
	this->dLdB.resize(1, this->vLayerSize);
	this->dLdB.assign(0.0f); // Initialize with zeros
}

NeuroNet::ActivationFunctionType NeuroNet::NeuroNetLayer::get_activation_type() const {
	return this->vActivationFunction;
}

std::string NeuroNet::NeuroNetLayer::get_activation_function_name() const {
    switch (this->vActivationFunction) {
        case ActivationFunctionType::None: return "None";
        case ActivationFunctionType::ReLU: return "ReLU";
        case ActivationFunctionType::LeakyReLU: return "LeakyReLU";
        case ActivationFunctionType::ELU: return "ELU";
        case ActivationFunctionType::Softmax: return "Softmax";
        default: return "Unknown";
    }
}

void NeuroNet::NeuroNet::UpdateWeights(float learning_rate) {
    if (learning_rate <= 0.0f) {
        // Optional: throw std::invalid_argument("Learning rate must be positive.");
        // For now, just return to avoid issues with negative/zero learning rate.
        return;
    }

    for (int i = 0; i < this->LayerCount; ++i) {
        // Ensure layer index is valid for NeuroNetVector (safety check)
        if (static_cast<size_t>(i) >= this->NeuroNetVector.size()) {
            throw std::out_of_range("Layer index " + std::to_string(i) +
                                    " is out of bounds for NeuroNetVector with size " + std::to_string(this->NeuroNetVector.size()) +
                                    " during UpdateWeights.");
        }
        NeuroNetLayer& layer = this->NeuroNetVector[i];

        int layer_input_size;
        if (i == 0) {
            layer_input_size = this->GetInputSize();
        } else {
            if (static_cast<size_t>(i-1) >= this->NeuroNetVector.size()){ // Should not happen if LayerCount is correct
                 throw std::out_of_range("Previous layer index " + std::to_string(i-1) + " is out of bounds during UpdateWeights.");
            }
            layer_input_size = this->NeuroNetVector[i-1].LayerSize();
        }
        int layer_output_size = layer.LayerSize();

        // --- Handle Weights ---
        LayerWeights current_lw_struct = layer.get_weights();
        Matrix::Matrix<float> dLdW_matrix = layer.get_dLdW();

        if (current_lw_struct.WeightCount > 0) {
            if (dLdW_matrix.rows() == 0 && dLdW_matrix.cols() == 0 && layer_input_size == 0 && layer_output_size == 0) {
                // Special case: layer has 0 input/output size, dLdW might be 0x0 legitimately. Skip update.
            } else if (dLdW_matrix.rows() != static_cast<size_t>(layer_input_size) || dLdW_matrix.cols() != static_cast<size_t>(layer_output_size)) {
                 throw std::runtime_error("Dimension mismatch for weight gradients in layer " + std::to_string(i) +
                                          ". Expected (" + std::to_string(layer_input_size) + "," + std::to_string(layer_output_size) +
                                          "), Got (" + std::to_string(dLdW_matrix.rows()) + "," + std::to_string(dLdW_matrix.cols()) + ")");
            }
            if (static_cast<int>(dLdW_matrix.rows() * dLdW_matrix.cols()) != current_lw_struct.WeightCount) {
                 throw std::runtime_error("Mismatch between dLdW_matrix total elements (" + std::to_string(dLdW_matrix.rows() * dLdW_matrix.cols()) +
                                          ") and layer's WeightCount (" + std::to_string(current_lw_struct.WeightCount) +") for layer " + std::to_string(i));
            }

            Matrix::Matrix<float> current_weights_matrix(layer_input_size, layer_output_size);
            if (layer_input_size > 0 && layer_output_size > 0) { // Only construct if dimensions are valid
                int k_w = 0;
                for (int r = 0; r < layer_input_size; ++r) {
                    for (int c = 0; c < layer_output_size; ++c) {
                        if (k_w < current_lw_struct.WeightCount) {
                            current_weights_matrix[r][c] = current_lw_struct.WeightsVector[k_w++];
                        } else {
                            throw std::runtime_error("WeightCount mismatch during weight matrix reconstruction for layer " + std::to_string(i));
                        }
                    }
                }
            }

            Matrix::Matrix<float> updated_weights_matrix = current_weights_matrix - (dLdW_matrix * learning_rate);

            LayerWeights new_lw_struct;
            new_lw_struct.WeightCount = current_lw_struct.WeightCount;
            if (new_lw_struct.WeightCount > 0) { // Only fill vector if there are weights
                new_lw_struct.WeightsVector.reserve(new_lw_struct.WeightCount);
                for (size_t r = 0; r < updated_weights_matrix.rows(); ++r) {
                    for (size_t c = 0; c < updated_weights_matrix.cols(); ++c) {
                        new_lw_struct.WeightsVector.push_back(updated_weights_matrix[r][c]);
                    }
                }
            }
            if (!layer.SetWeights(new_lw_struct)) {
                throw std::runtime_error("Failed to set updated weights for layer " + std::to_string(i));
            }
        }

        // --- Handle Biases ---
        LayerBiases current_lb_struct = layer.get_biases();
        Matrix::Matrix<float> dLdB_matrix = layer.get_dLdB();

        if (current_lb_struct.BiasCount > 0) {
             if (dLdB_matrix.rows() == 0 && dLdB_matrix.cols() == 0 && layer_output_size == 0) {
                // Special case: layer has 0 output size, dLdB might be 0x0 legitimately. Skip update.
             } else if (dLdB_matrix.rows() != 1 || dLdB_matrix.cols() != static_cast<size_t>(layer_output_size)) {
                 throw std::runtime_error("Dimension mismatch for bias gradients in layer " + std::to_string(i) +
                                          ". Expected (1," + std::to_string(layer_output_size) +
                                          "), Got (" + std::to_string(dLdB_matrix.rows()) + "," + std::to_string(dLdB_matrix.cols()) + ")");
            }
            if (static_cast<int>(dLdB_matrix.rows() * dLdB_matrix.cols()) != current_lb_struct.BiasCount) {
                 throw std::runtime_error("Mismatch between dLdB_matrix total elements (" + std::to_string(dLdB_matrix.rows() * dLdB_matrix.cols()) +
                                          ") and layer's BiasCount (" + std::to_string(current_lb_struct.BiasCount) + ") for layer " + std::to_string(i));
            }

            Matrix::Matrix<float> current_biases_matrix(1, layer_output_size);
            if (layer_output_size > 0) { // Only construct if dimensions are valid
                int k_b = 0;
                for (int c = 0; c < layer_output_size; ++c) {
                     if (k_b < current_lb_struct.BiasCount) {
                        current_biases_matrix[0][c] = current_lb_struct.BiasVector[k_b++];
                     } else {
                        throw std::runtime_error("BiasCount mismatch during bias matrix reconstruction for layer " + std::to_string(i));
                     }
                }
            }

            Matrix::Matrix<float> updated_biases_matrix = current_biases_matrix - (dLdB_matrix * learning_rate);

            LayerBiases new_lb_struct;
            new_lb_struct.BiasCount = current_lb_struct.BiasCount;
            if (new_lb_struct.BiasCount > 0) { // Only fill vector if there are biases
                new_lb_struct.BiasVector.reserve(new_lb_struct.BiasCount);
                for (size_t c = 0; c < updated_biases_matrix.cols(); ++c) {
                    new_lb_struct.BiasVector.push_back(updated_biases_matrix[0][c]);
                }
            }
            if (!layer.SetBiases(new_lb_struct)) {
                throw std::runtime_error("Failed to set updated biases for layer " + std::to_string(i));
            }
        }
    }
}

void NeuroNet::NeuroNet::Train(const std::vector<Matrix::Matrix<float>>& training_inputs,
                               const std::vector<Matrix::Matrix<float>>& training_targets,
                               float learning_rate,
                               int epochs) {
    // Step a: Basic validation
    if (training_inputs.empty()) { // Combined check for empty inputs/targets
        throw std::invalid_argument("Training inputs cannot be empty.");
    }
    if (training_targets.empty()) {
        throw std::invalid_argument("Training targets cannot be empty.");
    }
    if (training_inputs.size() != training_targets.size()) {
        throw std::invalid_argument("Number of training inputs (" + std::to_string(training_inputs.size()) +
                                    ") must match number of training targets (" + std::to_string(training_targets.size()) + ").");
    }
    if (learning_rate <= 0.0f) {
        throw std::invalid_argument("Learning rate must be positive. Got: " + std::to_string(learning_rate));
    }
    if (epochs <= 0) {
        throw std::invalid_argument("Number of epochs must be positive. Got: " + std::to_string(epochs));
    }
    if (this->LayerCount == 0) {
        throw std::runtime_error("Cannot train an empty network (0 layers). Initialize layers first.");
    }
     if (this->NeuroNetVector.empty()) { // Should be redundant if LayerCount is managed well
        throw std::runtime_error("Cannot train a network with an empty NeuroNetVector. Initialize layers first.");
    }


    // Step b: Loop for epochs
    for (int epoch = 0; epoch < epochs; ++epoch) {
        // Optional: Add logging for epoch number, e.g., std::cout << "Epoch " << epoch + 1 << "/" << epochs << std::endl;

        // Step c: Iterate through each training sample
        for (size_t i = 0; i < training_inputs.size(); ++i) {
            const Matrix::Matrix<float>& current_input_sample = training_inputs[i];
            const Matrix::Matrix<float>& current_target_sample = training_targets[i];

            // Validate sample dimensions against network's expected input/output sizes
            if (this->GetInputSize() < 0) { // Check if network input size is even set
                 throw std::runtime_error("Network input size is not set or invalid (<0). Call SetInputSize() before training.");
            }
            if (static_cast<int>(current_input_sample.cols()) != this->GetInputSize()) {
                 throw std::runtime_error("Input sample " + std::to_string(i) + " has " + std::to_string(current_input_sample.cols()) +
                                          " columns, but network expects " + std::to_string(this->GetInputSize()) + " input features.");
            }
            // Assuming network processes one instance at a time (1 row input). SetInput in NeuroNetLayer checks this.
            if (current_input_sample.rows() != 1 && this->GetInputSize() > 0) {
                 // This check is active based on NeuroNetLayer::SetInput expecting 1 row.
                 throw std::runtime_error("Input sample " + std::to_string(i) + " must have 1 row for current network setup. Got " + std::to_string(current_input_sample.rows()) + " rows.");
            }


            if (this->NeuroNetVector.back().LayerSize() <= 0 && current_target_sample.cols() > 0) { // Check if last layer size is valid
                 throw std::runtime_error("Network output layer size is not set or invalid (<=0), but target sample " + std::to_string(i) + " has columns. Configure network layers.");
            }
            if (static_cast<int>(current_target_sample.cols()) != this->NeuroNetVector.back().LayerSize()) {
                 throw std::runtime_error("Target sample " + std::to_string(i) + " has " + std::to_string(current_target_sample.cols()) +
                                          " columns, but network output layer has " + std::to_string(this->NeuroNetVector.back().LayerSize()) + " neurons.");
            }
            // Assuming 1 row target per sample.
            if (current_target_sample.rows() != 1 && this->NeuroNetVector.back().LayerSize() > 0) {
                 throw std::runtime_error("Target sample " + std::to_string(i) + " must have 1 row. Got " + std::to_string(current_target_sample.rows()) + " rows.");
            }


            // Step c.i: Set input
            if (!this->SetInput(current_input_sample)) {
                throw std::runtime_error("Failed to set input for sample " + std::to_string(i) + " in epoch " + std::to_string(epoch) +
                                         ". Input dimensions: " + std::to_string(current_input_sample.rows()) + "x" + std::to_string(current_input_sample.cols()) +
                                         ", Network expected input columns: " + std::to_string(this->GetInputSize()));
            }

            // Step c.ii: Forward pass
            Matrix::Matrix<float> actual_output = this->GetOutput();

            // Step c.iii: Backpropagation
            this->Backpropagate(actual_output, current_target_sample);

            // Step c.iv: Update weights
            this->UpdateWeights(learning_rate);
        }
    }
}

NeuroNet::ActivationFunctionType NeuroNet::NeuroNetLayer::activation_type_from_string(const std::string& name) {
    if (name == "None") return ActivationFunctionType::None;
    if (name == "ReLU") return ActivationFunctionType::ReLU;
    if (name == "LeakyReLU") return ActivationFunctionType::LeakyReLU;
    if (name == "ELU") return ActivationFunctionType::ELU;
    if (name == "Softmax") return ActivationFunctionType::Softmax;
    throw std::invalid_argument("Unknown activation function name: " + name);
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

Matrix::Matrix<float> NeuroNet::NeuroNetLayer::DerivativeReLU(const Matrix::Matrix<float>& activated_output) const {
    Matrix::Matrix<float> derivative = activated_output; // Copy dimensions and initial values
    for (int i = 0; i < derivative.rows(); ++i) {
        for (int j = 0; j < derivative.cols(); ++j) {
            derivative[i][j] = (activated_output[i][j] > 0.0f) ? 1.0f : 0.0f;
        }
    }
    return derivative;
}

Matrix::Matrix<float> NeuroNet::NeuroNetLayer::DerivativeLeakyReLU(const Matrix::Matrix<float>& activated_output) const {
    Matrix::Matrix<float> derivative = activated_output; // Copy dimensions and initial values
    const float alpha = 0.01f; // Ensure this matches the alpha in ApplyLeakyReLU
    for (int i = 0; i < derivative.rows(); ++i) {
        for (int j = 0; j < derivative.cols(); ++j) {
            derivative[i][j] = (activated_output[i][j] > 0.0f) ? 1.0f : alpha;
        }
    }
    return derivative;
}

Matrix::Matrix<float> NeuroNet::NeuroNetLayer::DerivativeELU(const Matrix::Matrix<float>& activated_output) const {
    Matrix::Matrix<float> derivative = activated_output; // Copy dimensions and initial values
    const float alpha = 1.0f; // Ensure this matches the alpha in ApplyELU
    for (int i = 0; i < derivative.rows(); ++i) {
        for (int j = 0; j < derivative.cols(); ++j) {
            if (activated_output[i][j] > 0.0f) {
                derivative[i][j] = 1.0f;
            } else {
                // If output = alpha * (exp(x) - 1), then derivative = alpha * exp(x) = output + alpha
                derivative[i][j] = activated_output[i][j] + alpha;
            }
        }
    }
    return derivative;
}

// Computes S_i * (1 - S_i) element-wise, where S is the softmax output (activated_output).
// This is the diagonal of the Jacobian dS/dZ, commonly used with Cross-Entropy loss.
Matrix::Matrix<float> NeuroNet::NeuroNetLayer::DerivativeSoftmax(const Matrix::Matrix<float>& activated_output) const {
    Matrix::Matrix<float> derivative = activated_output; // Copy dimensions and initial values
    for (int i = 0; i < derivative.rows(); ++i) {
        for (int j = 0; j < derivative.cols(); ++j) {
            float s_ij = activated_output[i][j];
            derivative[i][j] = s_ij * (1.0f - s_ij);
        }
    }
    return derivative;
}

Matrix::Matrix<float> NeuroNet::NeuroNetLayer::BackwardPass(const Matrix::Matrix<float>& dLdOutput, const Matrix::Matrix<float>& input_to_this_layer) {
    // Ensure gradient matrices are initialized/resized correctly
    // This check is a safeguard; ResizeLayer should have already initialized them.
    if (this->dLdW.rows() != this->WeightMatrix.rows() || this->dLdW.cols() != this->WeightMatrix.cols()) {
        this->dLdW.resize(this->WeightMatrix.rows(), this->WeightMatrix.cols());
        this->dLdW.assign(0.0f); // Initialize gradients to zero
    }
    if (this->dLdB.rows() != this->BiasMatrix.rows() || this->dLdB.cols() != this->BiasMatrix.cols()) {
        this->dLdB.resize(this->BiasMatrix.rows(), this->BiasMatrix.cols());
        this->dLdB.assign(0.0f); // Initialize gradients to zero
    }

    // Step a: Calculate dLdActivationInput = dLdOutput * f'(OutputMatrix)
    // f'(OutputMatrix) is the derivative of the activation function w.r.t. its input,
    // evaluated at the pre-activation output, but our derivative functions take activated_output.
    // Let Z = WX + B (pre-activation), A = f(Z) (OutputMatrix).
    // We have dLdA (dLdOutput). We need dAdZ.
    // Our derivative functions compute dAdZ based on A.
    Matrix::Matrix<float> dAdZ; // Derivative of activation function output w.r.t its input
    switch (this->vActivationFunction) {
        case ActivationFunctionType::ReLU:
            dAdZ = DerivativeReLU(this->OutputMatrix); // OutputMatrix stores f(Z)
            break;
        case ActivationFunctionType::LeakyReLU:
            dAdZ = DerivativeLeakyReLU(this->OutputMatrix);
            break;
        case ActivationFunctionType::ELU:
            dAdZ = DerivativeELU(this->OutputMatrix);
            break;
        case ActivationFunctionType::Softmax:
            // For Softmax with Cross-Entropy loss, dL/dZ = A - Y (output - target).
            // If dLdOutput *is* already (A-Y) for the output layer, then dLdActivationInput = dLdOutput.
            // But the plan is generic. dLdOutput is dL/dA.
            // dL/dZ_i = sum_j (dL/dA_j * dA_j/dZ_i).
            // If using the simplified S_i(1-S_i) from DerivativeSoftmax:
            dAdZ = DerivativeSoftmax(this->OutputMatrix);
            break;
        case ActivationFunctionType::None:
            // If no activation, f(Z) = Z, so f'(Z) = 1.
            // dAdZ should be a matrix of ones with the same dimensions as OutputMatrix.
            dAdZ.resize(this->OutputMatrix.rows(), this->OutputMatrix.cols());
            dAdZ.assign(1.0f);
            break;
        default:
            // Should not happen or throw error
            // For safety, resize and fill, but consider throwing an exception for unsupported types.
            dAdZ.resize(this->OutputMatrix.rows(), this->OutputMatrix.cols());
            dAdZ.assign(1.0f); // Fallback, or throw std::runtime_error("Unsupported activation function in BackwardPass");
            break;
    }

    // dLdActivationInput (dLdZ) = dLdOutput (dLdA) element-wise_multiply dAdZ
    Matrix::Matrix<float> dLdZ = dLdOutput; // Copy dimensions
    if (dLdOutput.rows() != dAdZ.rows() || dLdOutput.cols() != dAdZ.cols()) {
        // Error handling: dimensions must match for element-wise product
        throw std::runtime_error("Dimension mismatch for element-wise multiplication in BackwardPass (dLdOutput vs dAdZ). "
                                 "dLdOutput: (" + std::to_string(dLdOutput.rows()) + "," + std::to_string(dLdOutput.cols()) + ") "
                                 "dAdZ: (" + std::to_string(dAdZ.rows()) + "," + std::to_string(dAdZ.cols()) + ")");
    }
    for (int i = 0; i < dLdZ.rows(); ++i) {
        for (int j = 0; j < dLdZ.cols(); ++j) {
            dLdZ[i][j] = dLdOutput[i][j] * dAdZ[i][j];
        }
    }

    // Step b: Calculate dLdW = input_to_this_layer.transpose() * dLdZ
    this->dLdW = input_to_this_layer.Transpose() * dLdZ;

    // Step c: Calculate dLdB = dLdZ
    // This assumes dLdZ is (1, num_neurons) for a single instance.
    // If batch processing were implemented where dLdZ could be (batch_size, num_neurons),
    // dLdB would require summing dLdZ across the batch dimension.
    // For now, direct assignment is consistent with other matrix shapes (e.g. BiasMatrix is 1xN).
    this->dLdB = dLdZ;

    // Step d: Calculate dLdInput (dLdX for this layer, which is dLdA_prev for previous layer)
    Matrix::Matrix<float> dLdInput = dLdZ * this->WeightMatrix.Transpose();

    // Step e: Return dLdInput
    return dLdInput;
}

// Implementations for gradient getters
Matrix::Matrix<float> NeuroNet::NeuroNetLayer::get_dLdW() const {
    return this->dLdW;
}

Matrix::Matrix<float> NeuroNet::NeuroNetLayer::get_dLdB() const {
    return this->dLdB;
}

const Matrix::Matrix<float>& NeuroNet::NeuroNetLayer::get_input_matrix() const {
    return this->InputMatrix;
}

int NeuroNet::NeuroNetLayer::get_input_size() const {
    return this->InputSize;
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
#ifdef ENABLE_BENCHMARKING
    utilities::Timer layer_timer;
    layer_timer.start();
#endif

	if (InputMatrix.cols() == 0 || WeightMatrix.rows() == 0 || BiasMatrix.cols() == 0) {
		// Or throw an exception, or return an empty matrix with error status
		// For now, returning the current (possibly uninitialized) OutputMatrix.
		// Consider adding error handling or preconditions.
#ifdef ENABLE_BENCHMARKING
        // Even if returning early due to uninitialized state, stop the timer.
        layer_timer.stop();
        // Layer index is not directly available here without modification to the function signature
        // or making NeuroNetLayer aware of its index.
        // For now, a generic message or one using LayerSize as a proxy identifier.
        std::cout << "NeuroNetLayer::CalculateOutput() (Layer with output size " << this->vLayerSize 
                  << ", potentially uninitialized) took: " << layer_timer.elapsed_microseconds() << " us" << std::endl;
#endif
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

#ifdef ENABLE_BENCHMARKING
    layer_timer.stop();
    // As above, layer index isn't directly available. Using LayerSize as a proxy.
    std::cout << "NeuroNetLayer::CalculateOutput() (Layer with output size " << this->vLayerSize 
              << ") took: " << layer_timer.elapsed_microseconds() << " us" << std::endl;
#endif
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

NeuroNet::NeuroNetLayer& NeuroNet::NeuroNet::getLayer(int index) {
    if (index < 0 || static_cast<size_t>(index) >= this->NeuroNetVector.size()) {
        throw std::out_of_range("Layer index out of bounds in getLayer(). Requested index: " + std::to_string(index) + ", Layer count: " + std::to_string(this->NeuroNetVector.size()));
    }
    return this->NeuroNetVector[index];
}

const NeuroNet::NeuroNetLayer& NeuroNet::NeuroNet::getLayer(int index) const {
    if (index < 0 || static_cast<size_t>(index) >= this->NeuroNetVector.size()) {
        throw std::out_of_range("Layer index out of bounds in getLayer() const. Requested index: " + std::to_string(index) + ", Layer count: " + std::to_string(this->NeuroNetVector.size()));
    }
    return this->NeuroNetVector[index];
}

int NeuroNet::NeuroNet::getLayerCount() const {
    return this->LayerCount;
}

// Helper to create and set a number JsonValue in an object
// Ensures dynamic allocation for JsonValue members in objects, as per custom library's object_value map.
static void SetJsonNumber(JsonValue& parent_object, const std::string& key, double number_value) {
    JsonValue* val = new JsonValue(); // Dynamically allocate
    val->SetNumber(number_value);
    parent_object.InsertIntoObject(key, val); // InsertIntoObject stores the pointer
}

// Helper to create and set an object JsonValue in an object
// Returns a pointer to the created object for further modification
static JsonValue* CreateJsonObjectInObject(JsonValue& parent_object, const std::string& key) {
    JsonValue* obj_val = new JsonValue(); // Dynamically allocate
    obj_val->SetObject();
    parent_object.InsertIntoObject(key, obj_val);
    return obj_val;
}

// Helper to create and set an array JsonValue in an object
// Returns a pointer to the created array for further modification
static JsonValue* CreateJsonArrayInObject(JsonValue& parent_object, const std::string& key) {
    JsonValue* arr_val = new JsonValue(); // Dynamically allocate
    arr_val->SetArray();
    parent_object.InsertIntoObject(key, arr_val);
    return arr_val; 
}

std::string NeuroNet::NeuroNet::to_custom_json_string() const {
    JsonValue root; 
    root.SetObject();

    // 1. Serialize NeuroNet global parameters
    SetJsonNumber(root, "input_size", static_cast<double>(this->InputSize));
    SetJsonNumber(root, "layer_count", static_cast<double>(this->LayerCount));

    // 2. Serialize Layers
    JsonValue* layers_array_json_val_ptr = CreateJsonArrayInObject(root, "layers");

    for (int i = 0; i < this->LayerCount; ++i) {
        const NeuroNetLayer& layer = this->NeuroNetVector[i];
        JsonValue layer_json_val; 
        layer_json_val.SetObject();

        int current_layer_input_size = (i == 0) ? this->InputSize : this->NeuroNetVector[i-1].LayerSize();
        SetJsonNumber(layer_json_val, "input_size", static_cast<double>(current_layer_input_size));
        SetJsonNumber(layer_json_val, "layer_size", static_cast<double>(layer.LayerSize()));
        
        JsonValue* act_str_val = new JsonValue(); 
        act_str_val->SetString(layer.get_activation_function_name());
        layer_json_val.InsertIntoObject("activation_function", act_str_val);

        // Weights
        JsonValue* weights_obj_ptr = CreateJsonObjectInObject(layer_json_val, "weights");
        const auto& weights_data = layer.get_weights(); 
        SetJsonNumber(*weights_obj_ptr, "rows", static_cast<double>(current_layer_input_size));
        SetJsonNumber(*weights_obj_ptr, "cols", static_cast<double>(layer.LayerSize()));
        JsonValue* weights_data_arr_ptr = CreateJsonArrayInObject(*weights_obj_ptr, "data");
        for (float w : weights_data.WeightsVector) {
            JsonValue w_val; w_val.SetNumber(w);
            weights_data_arr_ptr->GetArray().push_back(w_val);
        }
        
        // Biases
        JsonValue* biases_obj_ptr = CreateJsonObjectInObject(layer_json_val, "biases");
        const auto& biases_data = layer.get_biases(); 
        SetJsonNumber(*biases_obj_ptr, "rows", 1.0);
        SetJsonNumber(*biases_obj_ptr, "cols", static_cast<double>(layer.LayerSize()));
        JsonValue* biases_data_arr_ptr = CreateJsonArrayInObject(*biases_obj_ptr, "data");
        for (float b : biases_data.BiasVector) {
            JsonValue b_val; b_val.SetNumber(b);
            biases_data_arr_ptr->GetArray().push_back(b_val);
        }
        
        layers_array_json_val_ptr->GetArray().push_back(layer_json_val);
    }

    std::string result_string = root.ToString();

    // IMPORTANT: Clean up dynamically allocated JsonValue objects.
    for (auto& pair : root.GetObject()) { 
        if (pair.first == "layers") {
            JsonValue* layers_array = pair.second;
            for (JsonValue& layer_val : layers_array->GetArray()) { 
                for (auto& layer_prop_pair : layer_val.GetObject()) {
                    if (layer_prop_pair.first == "weights" || layer_prop_pair.first == "biases") {
                        JsonValue* wb_object = layer_prop_pair.second; 
                        for (auto& wb_prop_pair : wb_object->GetObject()) { 
                             delete wb_prop_pair.second; 
                        }
                    }
                    delete layer_prop_pair.second; 
                }
            }
        }
        delete pair.second; 
    }
    // Clear the root object's map to prevent double deletion if root itself is destroyed later by a caller that manages it.
    // However, since root is a local stack variable, its map will be cleared upon exiting scope.
    // The pointers in the map are what need deletion.
    root.GetObject().clear(); 


    return result_string;
}

NeuroNet::NeuroNet NeuroNet::NeuroNet::load_model(const std::string& filename)
{
	std::ifstream ifs(filename);
	if (!ifs.is_open()) {
		throw std::runtime_error("Failed to open model file: " + filename);
	}
	std::string content((std::istreambuf_iterator<char>(ifs)), std::istreambuf_iterator<char>());
	ifs.close(); // Close the file stream after reading

	JsonValue root;
	try {
		root = JsonParser::Parse(content);
	} catch (const JsonParseException& e) {
		throw std::runtime_error("Failed to parse JSON from model file: " + filename + "\nErrors: " + e.what());
	}

	NeuroNet model; // Correctly use NeuroNet, not NeuroNet::NeuroNet for local variable

	// 1. Deserialize NeuroNet global parameters
	if (root.type != JsonValueType::Object ||
		root.GetObject().count("input_size") == 0 || root.GetObject().at("input_size")->type != JsonValueType::Number ||
		root.GetObject().count("layer_count") == 0 || root.GetObject().at("layer_count")->type != JsonValueType::Number) {
		throw std::runtime_error("Invalid JSON format: missing or invalid input_size or layer_count.");
	}
	int input_size = static_cast<int>(root.GetObject().at("input_size")->GetNumber());
	int layer_count = static_cast<int>(root.GetObject().at("layer_count")->GetNumber());

	model.SetInputSize(input_size);
	model.ResizeNeuroNet(layer_count); // Resizes NeuroNetVector

    // Load Vocabulary Configuration (optional field)
    if (root.GetObject().count("vocabulary_config")) {
        const JsonValue* vocab_config_json_val_ptr = root.GetObject().at("vocabulary_config");
        if (vocab_config_json_val_ptr->type == JsonValueType::Object) {
            const auto& vocab_config_obj = vocab_config_json_val_ptr->GetObject();
            if (vocab_config_obj.count("max_sequence_length") && vocab_config_obj.at("max_sequence_length")->type == JsonValueType::Number) {
                int max_len = static_cast<int>(vocab_config_obj.at("max_sequence_length")->GetNumber());
                model.vocabulary.set_max_sequence_length(max_len);
            }
        }
    }

	// 2. Deserialize Layers
	if (root.GetObject().count("layers") == 0 || root.GetObject().at("layers")->type != JsonValueType::Array) {
		throw std::runtime_error("Invalid JSON format: missing 'layers' array.");
	}
	const std::vector<JsonValue>& layers_array = root.GetObject().at("layers")->GetArray();
	if (layers_array.size() != static_cast<size_t>(layer_count)) { // Use size_t for comparison with vector::size()
		throw std::runtime_error("Layer count mismatch in JSON data.");
	}

	for (int i = 0; i < layer_count; ++i)
	{
		const JsonValue& layer_json = layers_array[i];
		if (layer_json.type != JsonValueType::Object) {
			throw std::runtime_error("Invalid layer format (not an object) in JSON for layer " + std::to_string(i));
		}
		
		const auto& layer_obj = layer_json.GetObject(); // Use a reference for convenience
		if (layer_obj.count("layer_size") == 0 || layer_obj.at("layer_size")->type != JsonValueType::Number ||
			layer_obj.count("input_size") == 0 || layer_obj.at("input_size")->type != JsonValueType::Number ||
			layer_obj.count("activation_function") == 0 || layer_obj.at("activation_function")->type != JsonValueType::String || // Expect String now
			layer_obj.count("weights") == 0 || layer_obj.at("weights")->type != JsonValueType::Object ||
			layer_obj.count("biases") == 0 || layer_obj.at("biases")->type != JsonValueType::Object) {
			throw std::runtime_error("Invalid layer format in JSON for layer " + std::to_string(i) + ": missing or invalid type for key members (activation_function should be string).");
		}

		int layer_output_size = static_cast<int>(layer_obj.at("layer_size")->GetNumber());
		model.ResizeLayer(i, layer_output_size); 
		
		NeuroNetLayer& current_layer = model.getLayer(i);

		//int activation_int = static_cast<int>(layer_obj.at("activation_function")->GetNumber()); // Old way
		//current_layer.SetActivationFunction(static_cast<ActivationFunctionType>(activation_int)); // Old way
        std::string activation_str = layer_obj.at("activation_function")->GetString(); // New: get as string
        current_layer.SetActivationFunction(NeuroNetLayer::activation_type_from_string(activation_str)); // New: convert string to enum

		// --- Weights ---
		const JsonValue& weights_json_val = *layer_obj.at("weights"); // Dereference pointer
		if (weights_json_val.type != JsonValueType::Object) throw std::runtime_error("Weights is not an object for layer " + std::to_string(i));
		const auto& weights_obj = weights_json_val.GetObject();

		if (weights_obj.count("rows") == 0 || weights_obj.at("rows")->type != JsonValueType::Number ||
			weights_obj.count("cols") == 0 || weights_obj.at("cols")->type != JsonValueType::Number ||
			weights_obj.count("data") == 0 || weights_obj.at("data")->type != JsonValueType::Array) {
			throw std::runtime_error("Invalid weights format for layer " + std::to_string(i));
		}
		const std::vector<JsonValue>& weights_data_array = weights_obj.at("data")->GetArray();
		
		LayerWeights layer_weights;
		layer_weights.WeightCount = weights_data_array.size();
		for (const auto& w_val_json : weights_data_array) { // Iterate over JsonValue
			if (w_val_json.type != JsonValueType::Number) throw std::runtime_error("Non-numeric weight value in layer " + std::to_string(i));
			layer_weights.WeightsVector.push_back(static_cast<float>(w_val_json.GetNumber()));
		}
		if (!current_layer.SetWeights(layer_weights)) {
			 throw std::runtime_error("Failed to set weights for layer " + std::to_string(i) + ". Count mismatch or other error.");
		}

		// --- Biases ---
		const JsonValue& biases_json_val = *layer_obj.at("biases"); // Dereference pointer
		if (biases_json_val.type != JsonValueType::Object) throw std::runtime_error("Biases is not an object for layer " + std::to_string(i));
		const auto& biases_obj = biases_json_val.GetObject();

		if (biases_obj.count("rows") == 0 || biases_obj.at("rows")->type != JsonValueType::Number ||
			biases_obj.count("cols") == 0 || biases_obj.at("cols")->type != JsonValueType::Number ||
			biases_obj.count("data") == 0 || biases_obj.at("data")->type != JsonValueType::Array) {
			throw std::runtime_error("Invalid biases format for layer " + std::to_string(i));
		}
		const std::vector<JsonValue>& biases_data_array = biases_obj.at("data")->GetArray();

		LayerBiases layer_biases;
		layer_biases.BiasCount = biases_data_array.size();
		for (const auto& b_val_json : biases_data_array) { // Iterate over JsonValue
			 if (b_val_json.type != JsonValueType::Number) throw std::runtime_error("Non-numeric bias value in layer " + std::to_string(i));
			layer_biases.BiasVector.push_back(static_cast<float>(b_val_json.GetNumber()));
		}
		if (!current_layer.SetBiases(layer_biases)) {
			 throw std::runtime_error("Failed to set biases for layer " + std::to_string(i) + ". Count mismatch or other error.");
		}
	}
	return model;
}

// Static helper functions (SetJsonNumber, CreateJsonObjectInObject, CreateJsonArrayInObject)
// were moved before to_custom_json_string() and save_model()


bool NeuroNet::NeuroNet::save_model(const std::string& filename) const
{
	JsonValue root; 
    root.SetObject();

	// 1. Serialize NeuroNet global parameters
    SetJsonNumber(root, "input_size", static_cast<double>(this->InputSize));
    SetJsonNumber(root, "layer_count", static_cast<double>(this->LayerCount));

    // Add Vocabulary Configuration
    if (this->vocabulary.get_max_sequence_length() > 0) { // Only save if it's meaningfully set
        JsonValue* vocab_config_obj_ptr = CreateJsonObjectInObject(root, "vocabulary_config");
        SetJsonNumber(*vocab_config_obj_ptr, "max_sequence_length", static_cast<double>(this->vocabulary.get_max_sequence_length()));
    }

	// 2. Serialize Layers
    // Create the main 'layers' array within the root object
    JsonValue* layers_array_json_val_ptr = CreateJsonArrayInObject(root, "layers");

	for (int i = 0; i < this->LayerCount; ++i)
	{
		const NeuroNetLayer& layer = this->NeuroNetVector[i];
		JsonValue layer_json_val; // This will be an element of layers_array_json_val_ptr
        layer_json_val.SetObject(); // This layer_json_val itself is an object

		int current_layer_input_size = (i == 0) ? this->InputSize : this->NeuroNetVector[i-1].LayerSize();
        SetJsonNumber(layer_json_val, "input_size", static_cast<double>(current_layer_input_size));
        SetJsonNumber(layer_json_val, "layer_size", static_cast<double>(layer.LayerSize()));
		//SetJsonNumber(layer_json_val, "activation_function", static_cast<double>(layer.get_activation_type())); // Old way
        JsonValue* act_str_val = new JsonValue(); 
        act_str_val->SetString(layer.get_activation_function_name()); // New: store as string
        layer_json_val.InsertIntoObject("activation_function", act_str_val);

		// --- Weights ---
        // Create 'weights' object within layer_json_val
		JsonValue* weights_obj_ptr = CreateJsonObjectInObject(layer_json_val, "weights");
		const auto& weights_data = layer.get_weights(); 
        SetJsonNumber(*weights_obj_ptr, "rows", static_cast<double>(current_layer_input_size));
        SetJsonNumber(*weights_obj_ptr, "cols", static_cast<double>(layer.LayerSize()));
        
        // Create 'data' array within 'weights' object
        JsonValue* weights_data_arr_ptr = CreateJsonArrayInObject(*weights_obj_ptr, "data");
		for (float w : weights_data.WeightsVector) {
            JsonValue w_val; w_val.SetNumber(w); // w_val is temporary, its value copied
			weights_data_arr_ptr->GetArray().push_back(w_val); // push_back copies w_val
		}
        
		// --- Biases ---
        // Create 'biases' object within layer_json_val
		JsonValue* biases_obj_ptr = CreateJsonObjectInObject(layer_json_val, "biases");
		const auto& biases_data = layer.get_biases(); 
        SetJsonNumber(*biases_obj_ptr, "rows", 1.0); // Biases typically have 1 row
        SetJsonNumber(*biases_obj_ptr, "cols", static_cast<double>(layer.LayerSize()));
        
        // Create 'data' array within 'biases' object
        JsonValue* biases_data_arr_ptr = CreateJsonArrayInObject(*biases_obj_ptr, "data");
		for (float b : biases_data.BiasVector) {
            JsonValue b_val; b_val.SetNumber(b);
			biases_data_arr_ptr->GetArray().push_back(b_val);
		}
		
        // Add the fully constructed layer_json_val to the main 'layers' array
		layers_array_json_val_ptr->GetArray().push_back(layer_json_val);
	}

	// 3. Write to file
	std::ofstream ofs(filename);
	if (!ofs.is_open()) {
        // NOTE: Potential memory leak here if we return early, as dynamically allocated
        // JsonValue objects (via new in SetJsonNumber, CreateJsonObjectInObject, etc.)
        // are not cleaned up by this function. A robust solution would require
        // a RAII wrapper or a recursive deletion function for the JsonValue structure
        // if an error occurs after allocations have begun.
		return false; 
	}
	
    ofs << root.ToString(); // Use the ToString method from custom JsonValue
	
	ofs.close();

    // IMPORTANT: Clean up dynamically allocated JsonValue objects.
    // The custom JsonValue::object_value stores JsonValue*. These were allocated with 'new'.
    // This is a simplified cleanup; a real scenario needs a recursive destructor in JsonValue
    // or a dedicated cleanup utility.
    for (auto& pair : root.GetObject()) { // For "input_size", "layer_count", "layers", "vocabulary_config"
        if (pair.first == "layers") {
            JsonValue* layers_array = pair.second;
            for (JsonValue& layer_val : layers_array->GetArray()) { // Each layer_val is an object
                for (auto& layer_prop_pair : layer_val.GetObject()) {
                    if (layer_prop_pair.first == "weights" || layer_prop_pair.first == "biases") {
                        JsonValue* wb_object = layer_prop_pair.second; // This is the object for weights/biases
                        for (auto& wb_prop_pair : wb_object->GetObject()) { // rows, cols, data
                             delete wb_prop_pair.second; // Delete JsonValue* for rows, cols, data array
                        }
                    }
                    delete layer_prop_pair.second; // Delete JsonValue* for input_size, layer_size, activation_function, weights obj, biases obj
                }
            }
        } else if (pair.first == "vocabulary_config") {
            JsonValue* vocab_config_object = pair.second;
            for (auto& vocab_prop_pair : vocab_config_object->GetObject()) { // e.g., "max_sequence_length"
                delete vocab_prop_pair.second; // Delete the JsonValue* for "max_sequence_length" value
            }
        }
        delete pair.second; // Delete JsonValue* for top-level keys like "input_size", "layer_count", "layers" array itself, "vocabulary_config" object itself
    }
    root.GetObject().clear(); // Clear the map in root to remove dangling pointers
	return true;
}

int NeuroNet::NeuroNetLayer::WeightCount() {
	return this->Weights.WeightCount;
}

int NeuroNet::NeuroNetLayer::BiasCount() {
	return this->Biases.BiasCount;
}

int NeuroNet::NeuroNetLayer::LayerSize() const {
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

void NeuroNet::NeuroNet::Backpropagate(const Matrix::Matrix<float>& actual_output, const Matrix::Matrix<float>& target_output) {
    if (this->NeuroNetVector.empty()) {
        return; // No layers to backpropagate through.
    }

    // Ensure target_output and actual_output have the same dimensions.
    if (actual_output.rows() != target_output.rows() || actual_output.cols() != target_output.cols()) {
        throw std::runtime_error("Dimension mismatch between actual_output (" +
                                 std::to_string(actual_output.rows()) + "x" + std::to_string(actual_output.cols()) +
                                 ") and target_output (" +
                                 std::to_string(target_output.rows()) + "x" + std::to_string(target_output.cols()) +
                                 ") in Backpropagate.");
    }

    // Step a: Calculate initial dLdOutput for the last layer.
    // This is typically (actual_output - target_output) for MSE loss,
    // or (softmax_output - target_one_hot) if used as dL/dZ for softmax + CCE.
    // Our Layer::BackwardPass expects dL/dActivation_Output.
    Matrix::Matrix<float> dLdOutput_current = actual_output - target_output; // Element-wise subtraction

    // Step b: Iterate backward through layers
    // Using LayerCount as it's explicitly managed by ResizeNeuroNet and represents the intended number of layers.
    for (int i = this->LayerCount - 1; i >= 0; --i) {
        // Ensure layer index is valid for NeuroNetVector (safety check, though LayerCount should be consistent)
        if (static_cast<size_t>(i) >= this->NeuroNetVector.size()) {
            throw std::out_of_range("Layer index " + std::to_string(i) +
                                    " is out of bounds for NeuroNetVector with size " + std::to_string(this->NeuroNetVector.size()) +
                                    " during backpropagation.");
        }
        NeuroNetLayer& current_layer = this->NeuroNetVector[i];

        // Get the input that was fed to this layer during the forward pass.
        // Use the new public getter.
        const Matrix::Matrix<float>& input_to_current_layer = current_layer.get_input_matrix();

        // Validate InputMatrix
        if (input_to_current_layer.rows() == 0 && current_layer.LayerSize() > 0) { // LayerSize > 0 implies it's a configured layer
             throw std::runtime_error("Layer " + std::to_string(i) +
                                     " InputMatrix has 0 rows during backpropagation. Layer output size: " + std::to_string(current_layer.LayerSize()) +
                                     ". Forward pass might be incomplete or network input not set.");
        }
        // Use the new public getter for InputSize.
        if (input_to_current_layer.cols() == 0 && current_layer.get_input_size() > 0 ) { // InputSize > 0 implies it expects input
             throw std::runtime_error("Layer " + std::to_string(i) +
                                     " InputMatrix has 0 columns but layer expects input size " + std::to_string(current_layer.get_input_size()) +
                                     ". Forward pass might be incomplete or network input not set.");
        }


        dLdOutput_current = current_layer.BackwardPass(dLdOutput_current, input_to_current_layer);
        // The returned dLdOutput_current is now dL/dInput for the current layer,
        // which is dL/dOutput for the *previous* layer (i-1).
    }
}

bool NeuroNet::NeuroNet::SetInputJSON(const std::string& json_input) {
    JsonValue parsed_json_input;
    try {
        parsed_json_input = JsonParser::Parse(json_input);
    } catch (const JsonParseException& e) {
        // Optionally log the error e.what()
        throw; // Re-throw the JsonParseException
    }

    if (parsed_json_input.type != JsonValueType::Object ||
        parsed_json_input.GetObject().find("input_matrix") == parsed_json_input.GetObject().end()) {
        throw std::runtime_error("JSON input must be an object with an 'input_matrix' key.");
    }

    const JsonValue* matrix_json_val_ptr = parsed_json_input.GetObject().at("input_matrix");
    if (matrix_json_val_ptr->type != JsonValueType::Array) {
        throw std::runtime_error("'input_matrix' value must be an array of arrays.");
    }

    const std::vector<JsonValue>& rows_json = matrix_json_val_ptr->GetArray();
    if (rows_json.empty()) {
        // Allow empty matrix if SetInput can handle it, or throw error
        // For now, let's assume an empty matrix is valid if SetInput allows 0 rows/cols
        Matrix::Matrix<float> empty_matrix(0,0);
        return this->SetInput(empty_matrix);
    }

    // Determine matrix dimensions
    size_t num_rows = rows_json.size();
    size_t num_cols = 0;
    if (num_rows > 0) {
        const JsonValue& first_row_json = rows_json[0];
        if (first_row_json.type != JsonValueType::Array) {
            throw std::runtime_error("Each element of 'input_matrix' must be an array (a row).");
        }
        num_cols = first_row_json.GetArray().size();
    }

    Matrix::Matrix<float> input_matrix(num_rows, num_cols);

    for (size_t i = 0; i < num_rows; ++i) {
        const JsonValue& row_json_val = rows_json[i];
        if (row_json_val.type != JsonValueType::Array) {
            throw std::runtime_error("Row " + std::to_string(i) + " is not a JSON array.");
        }
        const std::vector<JsonValue>& cols_json = row_json_val.GetArray();
        if (cols_json.size() != num_cols) {
            throw std::runtime_error("Row " + std::to_string(i) + " has incorrect number of columns. Expected " + std::to_string(num_cols) + ", got " + std::to_string(cols_json.size()));
        }
        for (size_t j = 0; j < num_cols; ++j) {
            const JsonValue& val_json = cols_json[j];
            if (val_json.type != JsonValueType::Number) {
                throw std::runtime_error("Matrix element at (" + std::to_string(i) + "," + std::to_string(j) + ") is not a number.");
            }
            input_matrix[i][j] = static_cast<float>(val_json.GetNumber());
        }
    }
    return this->SetInput(input_matrix);
}

std::string NeuroNet::NeuroNet::GetOutputJSON() {
    Matrix::Matrix<float> output_matrix = this->GetOutput(); // Get the output matrix

    // Check if the output matrix is empty or invalid (e.g., if GetOutput returns an empty matrix on error)
    if (output_matrix.rows() == 0 && output_matrix.cols() == 0 && this->LayerCount > 0) {
         // This case might indicate an issue with GetOutput() or an uninitialized network,
         // but GetOutput() itself might return an empty matrix if no layers.
         // If LayerCount is 0, GetOutput() already returns an empty matrix.
         // If LayerCount > 0 and we still get 0x0, it's potentially an issue or just an empty output.
         // For now, we'll serialize an empty array for the matrix data.
    }

    if (this->LayerCount == 0 && output_matrix.rows() == 0 && output_matrix.cols() == 0) {
        // If there are no layers, GetOutput() returns an empty matrix.
        // We should return a valid JSON representing an empty matrix or handle as an error.
        // Let's return a JSON with an empty "output_matrix" array.
    }


    JsonValue root_json;
    root_json.SetObject();

    JsonValue* matrix_data_json_ptr = new JsonValue(); // Dynamically allocate
    matrix_data_json_ptr->SetArray();                 // This will be an array of arrays

    for (size_t i = 0; i < output_matrix.rows(); ++i) { // Use size_t for loop
        JsonValue row_json; // Temporary JsonValue for the row, will be copied into matrix_data_json_ptr
        row_json.SetArray();
        std::vector<JsonValue>& row_array_ref = row_json.GetArray(); // Get reference to internal vector

        for (size_t j = 0; j < output_matrix.cols(); ++j) { // Use size_t for loop
            JsonValue val_json; // Temporary JsonValue for the number
            val_json.SetNumber(static_cast<double>(output_matrix[i][j]));
            row_array_ref.push_back(val_json); // Copy val_json into the row array
        }
        matrix_data_json_ptr->GetArray().push_back(row_json); // Copy row_json into the main matrix data array
    }

    root_json.InsertIntoObject("output_matrix", matrix_data_json_ptr);

    std::string output_json_string = root_json.ToString();

    // Cleanup dynamically allocated JsonValue pointed to by matrix_data_json_ptr
    // The JsonValue stored in root_json.object_value["output_matrix"] is matrix_data_json_ptr
    // JsonValue's destructor or clear method should ideally handle this if it's designed to own its children.
    // Based on json.hpp, object_value stores JsonValue*, and JsonValue::SetObject clears object_value
    // but does not delete the pointed-to objects.
    // JsonValue::InsertIntoObject also doesn't manage memory of previous value if key existed.
    // The `save_model` function had manual cleanup. We need similar here.
    delete matrix_data_json_ptr; // matrix_data_json_ptr itself
    root_json.GetObject().clear(); // Clear the map to remove the dangling pointer entry

    return output_json_string;
}

// --- NeuroNet Method Implementations ---
// (This is a comment, ensure the new method is outside other function bodies)

bool NeuroNet::NeuroNet::LoadVocabulary(const std::string& filepath) {
    return this->vocabulary.load_from_json(filepath);
}

bool NeuroNet::NeuroNet::SetStringsInput(const std::string& json_string_input, int max_len_override, bool pad_to_max_in_batch_override) {
    if (this->vocabulary.get_padding_token_id() == -1 || this->vocabulary.get_unknown_token_id() == -1) {
        throw std::runtime_error("Vocabulary not loaded or not properly initialized. Call LoadVocabulary() first.");
    }

    JsonValue parsed_json_input;
    try {
        parsed_json_input = JsonParser::Parse(json_string_input);
    } catch (const JsonParseException& e) {
        // Consider logging e.what()
        throw; // Re-throw
    }

    if (parsed_json_input.type != JsonValueType::Object) {
        // Cleanup for parsed_json_input if it allocated anything
        if (parsed_json_input.type == JsonValueType::Object) {
            for(auto& pair : parsed_json_input.GetObject()) delete pair.second;
            parsed_json_input.GetObject().clear();
        } else if (parsed_json_input.type == JsonValueType::Array) {
            // If it was an array of objects/arrays, deeper cleanup might be needed
            // For now, assume if not object, it's a simple type or an array of simple types
            // that JsonParser::Parse might not allocate deeply for, or that this path is an error anyway.
        }
        throw std::runtime_error("JSON input must be an object.");
    }

    const auto& root_obj = parsed_json_input.GetObject();
    if (root_obj.find("input_batch") == root_obj.end() || root_obj.at("input_batch")->type != JsonValueType::Array) {
        for(auto& pair : root_obj) delete pair.second;
        parsed_json_input.GetObject().clear();
        throw std::runtime_error("JSON input must be an object with an 'input_batch' key, and its value must be an array of strings.");
    }

    const std::vector<JsonValue>& string_json_array = root_obj.at("input_batch")->GetArray();
    std::vector<std::string> batch_sequences;
    batch_sequences.reserve(string_json_array.size());

    for (const JsonValue& str_val : string_json_array) {
        if (str_val.type != JsonValueType::String) {
            for(auto& pair : root_obj) delete pair.second;
            parsed_json_input.GetObject().clear();
            throw std::runtime_error("All elements in 'input_batch' array must be strings.");
        }
        batch_sequences.push_back(str_val.GetString());
    }

    // Cleanup parsed_json_input *before* potential throw in prepare_batch_matrix or SetInput
    for(auto& pair : root_obj) {
        // The "input_batch" key points to an array (JsonValue*). This array's internal vector
        // holds JsonValue objects (not pointers) if they are simple types like strings.
        // So, deleting pair.second (which is the JsonValue* for the array) is correct.
        // The JsonValue destructor for the array should handle its own vector of JsonValues.
        // If JsonValue stored JsonValue* in its array_value, then deeper cleanup would be needed here.
        // Given current Vocabulary::load_from_json cleanup, this seems consistent.
        delete pair.second;
    }
    parsed_json_input.GetObject().clear();


    Matrix::Matrix<float> input_matrix;
    try {
        input_matrix = this->vocabulary.prepare_batch_matrix(batch_sequences, max_len_override, pad_to_max_in_batch_override);
    } catch (const std::runtime_error& e) {
        // Re-throw or wrap if more context is needed
        throw;
    }

    if (input_matrix.rows() == 0 && !batch_sequences.empty()) {
        // This case might occur if all strings in batch are empty AND max_len results in 0 columns.
        // SetInput might not handle a 0-column matrix if rows > 0.
        // Or if batch_sequences was empty, prepare_batch_matrix returns 0x0, which SetInput should handle.
         if (this->InputSize > 0 && input_matrix.cols() == 0) {
              throw std::runtime_error("Prepared matrix has 0 columns, but network input size is > 0.");
         }
    }
     // Check if network's InputSize is configured (i.e. > 0). If it is, the matrix columns must match.
     // If InputSize is 0, it means the network's input size is flexible or not yet defined,
     // so we don't enforce a specific column count from the prepared matrix.
     // The first layer's input size would be set by this matrix's column count.
     if (this->InputSize > 0 && static_cast<int>(input_matrix.cols()) != this->InputSize) {
        throw std::runtime_error("Prepared matrix column count (" + std::to_string(input_matrix.cols()) +
                                 ") does not match network InputSize (" + std::to_string(this->InputSize) + "). " +
                                 "Ensure vocabulary's max_sequence_length or override matches network's expected sequence length for tokenized input.");
     }


    return this->SetInput(input_matrix);
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
#ifdef ENABLE_BENCHMARKING
    utilities::Timer total_forward_pass_timer;
    total_forward_pass_timer.start();
#endif

	if (this->NeuroNetVector.empty()) {
#ifdef ENABLE_BENCHMARKING
        total_forward_pass_timer.stop();
        std::cout << "NeuroNet::GetOutput() (Total Forward Pass - No Layers) took: " 
                  << total_forward_pass_timer.elapsed_milliseconds() << " ms" << std::endl;
#endif
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
	// Note: NeuroNetLayer::CalculateOutput() will print its own timing if ENABLE_BENCHMARKING is defined.
	this->NeuroNetVector[0].CalculateOutput();

	// Process subsequent layers
    // Use NeuroNetVector.size() for safety if LayerCount might be out of sync.
    // However, the design relies on LayerCount. If ResizeNeuroNet is used correctly, they should match.
	for (size_t i = 1; i < this->NeuroNetVector.size(); i++) { // Iterate up to actual number of layers present
		this->NeuroNetVector[i].SetInput(this->NeuroNetVector[i - 1].ReturnOutputMatrix());
		this->NeuroNetVector[i].CalculateOutput(); // This will also print its timing.
	}
    
    Matrix::Matrix<float> final_output = this->NeuroNetVector.back().ReturnOutputMatrix();

#ifdef ENABLE_BENCHMARKING
    total_forward_pass_timer.stop();
    std::cout << "NeuroNet::GetOutput() (Total Forward Pass for " << this->NeuroNetVector.size() << " layers) took: " 
              << total_forward_pass_timer.elapsed_milliseconds() << " ms" << std::endl;
#endif
	return final_output; // Output of the last layer
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
