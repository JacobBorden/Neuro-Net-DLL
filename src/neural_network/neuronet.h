/**
 * @file neuronet.h
 * @author Jacob Borden (amenra.beats@@gmail.com)
 * @brief Defines the core classes for the NeuroNet neural network library, including NeuroNetLayer and NeuroNet.
 * @version 0.2.0
 * @date 2023-10-27 (Last major update date)
 *
 * @copyright Copyright (c) 2021-2023 Jacob Borden
 *
 */

#pragma once

// Standard library includes
#include <string>
#include <vector>
#include <stdexcept>

// Project-specific dependencies
#include "../math/matrix.h"
#include "../utilities/json/json.hpp"
#include "../utilities/vocabulary.h"

// Forward declare GTest class (if it was present immediately after includes)
class NeuroNetTest_Serialization_Test; 

namespace NeuroNet
{
	/**
	 * @brief Specifies the type of activation function to be used in a NeuroNetLayer.
	 */
	enum class ActivationFunctionType
	{
		None,      ///< No activation function. Output is the raw linear transformation.
		ReLU,      ///< Rectified Linear Unit. Output is max(0, x).
		LeakyReLU, ///< Leaky Rectified Linear Unit. Output is x if x > 0, otherwise alpha*x.
		ELU,       ///< Exponential Linear Unit. Output is x if x > 0, otherwise alpha*(exp(x)-1).
		Softmax    ///< Softmax function. Normalizes outputs to a probability distribution.
	};

	/**
	 * @brief Structure to hold the weights for a neural network layer.
	 *
	 * Contains the count of weights and a vector storing the actual weight values as floats.
	 */
	struct LayerWeights
	{
		int WeightCount = 0; ///< The total number of weights in this layer.
		std::vector<float> WeightsVector; ///< Vector containing the weight values.
	};

	/**
	 * @brief Structure to hold the biases for a neural network layer.
	 *
	 * Contains the count of biases and a vector storing the actual bias values as floats.
	 */
	struct LayerBiases
	{
		int BiasCount = 0; ///< The total number of biases in this layer (typically equals the number of neurons).
		std::vector<float> BiasVector; ///< Vector containing the bias values.
	};

	/**
	 * @brief Represents an individual layer within a neural network.
	 *
	 * Manages its own weights, biases, activation function, and calculates its output based on inputs.
	 */
	class NeuroNetLayer
	{
	public:
		/**
		 * @brief Default constructor for NeuroNetLayer.
		 * Initializes an empty layer with ActivationFunctionType::None by default.
		 */
		NeuroNetLayer();

		/**
		 * @brief Destructor for NeuroNetLayer.
		 */
		~NeuroNetLayer();

		/**
		 * @brief Resizes the layer to specified input and layer sizes.
		 *
		 * This function configures the dimensions of the weight and bias matrices.
		 * Weights are typically initialized to small random values or zeros by default by the Matrix class,
		 * and biases are initialized to zeros.
		 * @param pInputSize The number of inputs to this layer.
		 * @param pLayerSize The number of neurons in this layer (output size of this layer).
		 */
		void ResizeLayer(int pInputSize, int pLayerSize);

		/**
		 * @brief Calculates the output of this layer.
		 *
		 * The output is computed as (InputMatrix * WeightMatrix) + BiasMatrix.
		 * @return Matrix::Matrix<float> A matrix representing the layer's output.
		 */
		Matrix::Matrix<float> CalculateOutput();

		/**
		 * @brief Returns the most recently calculated output matrix of this layer.
		 * @return Matrix::Matrix<float> The output matrix.
		 */
		Matrix::Matrix<float> ReturnOutputMatrix();

		/**
		 * @brief Sets the input for this layer.
		 * @param pInputMatrix A const reference to a matrix containing the input values. Must match the expected input dimensions.
		 * @return bool True if the input was successfully set, false otherwise (e.g., size mismatch).
		 */
		bool SetInput(const Matrix::Matrix<float>& pInputMatrix);

		/**
		 * @brief Gets the total number of weights in this layer.
		 * @return int The count of weights.
		 */
		int WeightCount();

		/**
		 * @brief Gets the total number of biases in this layer.
		 * @return int The count of biases.
		 */
		int BiasCount();

		/**
		 * @brief Gets the size of this layer (number of neurons).
		 * @return int The number of neurons in this layer.
		 */
		int LayerSize() const;

		/**
		 * @brief Sets the weights for this layer.
		 * @param pWeights A LayerWeights struct containing the weights to set.
		 * @return bool True if weights were successfully set, false otherwise (e.g., count mismatch).
		 */
		bool SetWeights(LayerWeights pWeights);

		/**
		 * @brief Sets the biases for this layer.
		 * @param pBiases A LayerBiases struct containing the biases to set.
		 * @return bool True if biases were successfully set, false otherwise (e.g., count mismatch).
		 */
		bool SetBiases(LayerBiases pBiases);

		/**
		 * @brief Retrieves the current weights of the layer.
		 * @return LayerWeights A struct containing the layer's weights.
		 */
		LayerWeights get_weights() const;

		/**
		 * @brief Retrieves the current biases of the layer.
		 * @return LayerBiases A struct containing the layer's biases.
		 */
		LayerBiases get_biases() const;

		/**
		 * @brief Sets the activation function for this layer.
		 * @param pActivationFunction The type of activation function to use (e.g., ReLU, Softmax).
		 */
		void SetActivationFunction(ActivationFunctionType pActivationFunction);

		/**
		 * @brief Gets the activation function type for this layer.
		 * @return ActivationFunctionType The activation function type.
		 */
		ActivationFunctionType get_activation_type() const;

		/**
		 * @brief Gets the string name of the activation function for this layer.
		 * @return std::string The name of the activation function (e.g., "ReLU", "Softmax").
		 */
		std::string get_activation_function_name() const;

		/**
		 * @brief Converts a string name to an ActivationFunctionType enum value.
		 * @param name The string name of the activation function.
		 * @return ActivationFunctionType The corresponding enum value.
		 * @throws std::invalid_argument if the name is not recognized.
		 */
		static ActivationFunctionType activation_type_from_string(const std::string& name);

    /**
     * @brief Computes the element-wise derivative of the ReLU activation function.
     * The derivative of ReLU is 1 if the input (pre-activation) was > 0, else 0.
     * This function takes the *activated* output and infers the derivative: 1 if activated_output > 0, else 0.
     * @param activated_output The matrix of outputs after ReLU activation was applied.
     * @return Matrix::Matrix<float> A matrix containing the derivatives (1s and 0s).
     */
    Matrix::Matrix<float> DerivativeReLU(const Matrix::Matrix<float>& activated_output) const;

    /**
     * @brief Computes the element-wise derivative of the Leaky ReLU activation function.
     * Derivative is 1 if input (pre-activation) was > 0, else alpha.
     * This function takes the *activated* output: 1 if activated_output > 0, else alpha.
     * @param activated_output The matrix of outputs after Leaky ReLU activation was applied.
     * @return Matrix::Matrix<float> A matrix containing the derivatives (1s and alpha values).
     */
    Matrix::Matrix<float> DerivativeLeakyReLU(const Matrix::Matrix<float>& activated_output) const;

    /**
     * @brief Computes the element-wise derivative of the ELU activation function.
     * Derivative is 1 if input (pre-activation) was > 0, else ELU_output + alpha (where ELU_output is alpha*(exp(input)-1)).
     * This function takes the *activated* output. If activated_output > 0, derivative is 1.
     * If activated_output <= 0, derivative is activated_output + alpha.
     * @param activated_output The matrix of outputs after ELU activation was applied.
     * @return Matrix::Matrix<float> A matrix containing the derivatives.
     */
    Matrix::Matrix<float> DerivativeELU(const Matrix::Matrix<float>& activated_output) const;

    /**
     * @brief Computes the element-wise derivative of the Softmax activation function.
     * This typically refers to the diagonal elements of the Jacobian matrix, dS_i/dZ_i = S_i * (1 - S_i),
     * where S is the softmax output. This form is commonly used when the loss function (e.g., Cross-Entropy)
     * and Softmax derivative are combined, leading to a simpler dL/dZ.
     * @param activated_output The matrix of outputs after Softmax activation was applied (i.e., the S matrix).
     * @return Matrix::Matrix<float> A matrix where each element is S_ij * (1 - S_ij).
     */
    Matrix::Matrix<float> DerivativeSoftmax(const Matrix::Matrix<float>& activated_output) const;

    /**
     * @brief Performs the backward pass for this layer.
     *
     * Calculates the gradients of the loss with respect to the layer's weights (dLdW),
     * biases (dLdB), and input (dLdInput_prev_layer_output). The calculated dLdW and dLdB
     * are stored internally in the layer.
     *
     * @param dLdOutput Gradient of the loss with respect to this layer's activated output (dL/dA).
     *                  Dimensions should match this layer's output dimensions.
     * @param input_to_this_layer The input matrix that was fed to this layer during the forward pass (X, or A_prev).
     *                            Dimensions should match this layer's input dimensions.
     * @return Matrix::Matrix<float> Gradient of the loss with respect to this layer's input (dL/dX_prev),
     *                               to be passed to the previous layer.
     * @throws std::runtime_error if dimension mismatches occur during calculations.
     */
    Matrix::Matrix<float> BackwardPass(const Matrix::Matrix<float>& dLdOutput, const Matrix::Matrix<float>& input_to_this_layer);

    /**
     * @brief Retrieves the stored gradient of the loss with respect to the layer's weights (dL/dW).
     * This gradient is computed and stored during the BackwardPass.
     * @return Matrix::Matrix<float> The dL/dW matrix. Dimensions match the layer's weight matrix.
     */
    Matrix::Matrix<float> get_dLdW() const;

    /**
     * @brief Retrieves the stored gradient of the loss with respect to the layer's biases (dL/dB).
     * This gradient is computed and stored during the BackwardPass.
     * @return Matrix::Matrix<float> The dL/dB matrix. Dimensions match the layer's bias matrix.
     */
    Matrix::Matrix<float> get_dLdB() const;

	private:
		int vLayerSize = 0; ///< Number of neurons in this layer.
		int InputSize = 0; ///< Number of inputs expected by this layer.
		ActivationFunctionType vActivationFunction; ///< The activation function type for this layer.
		Matrix::Matrix<float> InputMatrix; ///< Matrix storing the inputs to this layer.
		Matrix::Matrix<float> WeightMatrix; ///< Matrix storing the weights of this layer.
		Matrix::Matrix<float> BiasMatrix;   ///< Matrix storing the biases of this layer.
		Matrix::Matrix<float> OutputMatrix; ///< Matrix storing the calculated outputs of this layer.
		LayerWeights Weights; ///< Struct holding the layer's weights.
		LayerBiases Biases;   ///< Struct holding the layer's biases.
    Matrix::Matrix<float> dLdW; // Gradient of Loss w.r.t Weights
    Matrix::Matrix<float> dLdB; // Gradient of Loss w.r.t Biases

    // Private helper methods for activation functions
    /**
     * @brief Applies the ReLU activation function element-wise to the input matrix.
     * @param input The matrix resulting from the linear transformation (Wx + b).
     * @return Matrix::Matrix<float> The matrix after applying ReLU.
     */
    Matrix::Matrix<float> ApplyReLU(const Matrix::Matrix<float>& input);
    /**
     * @brief Applies the LeakyReLU activation function element-wise to the input matrix.
     * @param input The matrix resulting from the linear transformation (Wx + b).
     * @return Matrix::Matrix<float> The matrix after applying LeakyReLU.
     */
    Matrix::Matrix<float> ApplyLeakyReLU(const Matrix::Matrix<float>& input);
    /**
     * @brief Applies the ELU activation function element-wise to the input matrix.
     * @param input The matrix resulting from the linear transformation (Wx + b).
     * @return Matrix::Matrix<float> The matrix after applying ELU.
     */
    Matrix::Matrix<float> ApplyELU(const Matrix::Matrix<float>& input);
    /**
     * @brief Applies the Softmax activation function to the input matrix.
     * Typically used for the output layer in classification tasks.
     * @param input The matrix resulting from the linear transformation (Wx + b).
     * @return Matrix::Matrix<float> The matrix after applying Softmax.
     */
    Matrix::Matrix<float> ApplySoftmax(const Matrix::Matrix<float>& input);
	};

	/**
	 * @brief Represents a complete neural network, composed of multiple NeuroNetLayer objects.
	 *
	 * This class manages the overall structure of the network, including its layers,
	 * input size, and provides methods for configuration, data processing, and
	 * accessing network parameters (weights and biases).
	 */
	class NeuroNet
	{
	public:
		// Friend the specific GTest generated class
		friend class ::NeuroNetTest_Serialization_Test; // Use :: for global scope

		/**
		 * @brief Default constructor for NeuroNet.
		 * Initializes an empty network with no layers.
		 */
		NeuroNet();

		/**
		 * @brief Constructs a NeuroNet with a specified number of layers.
		 * Layers are initialized but not yet sized.
		 * @param pLayerCount The initial number of layers in the network.
		 */
		NeuroNet(int pLayerCount);

		/**
		 * @brief Destructor for NeuroNet.
		 */
		~NeuroNet();

		/**
		 * @brief Resizes a specific layer within the network.
		 *
		 * If resizing layer 0, its input size is determined by `SetInputSize`.
		 * For subsequent layers, the input size is determined by the output size of the previous layer.
		 * This method also handles updating the input size of the next layer if it exists.
		 * @param pLayerIndex The index of the layer to resize.
		 * @param pLayerSize The new number of neurons for the specified layer.
		 * @return bool True if resizing was successful, false otherwise (e.g., invalid index).
		 */
		bool ResizeLayer(int pLayerIndex, int pLayerSize);

		/**
		 * @brief Sets the input size for the entire neural network (i.e., for the first layer).
		 * This also resizes the first layer's input connections.
		 * @param pInputSize The number of input features for the network.
		 */
		void SetInputSize(int pInputSize);

		/**
		 * @brief Resizes the entire neural network to have a new number of layers.
		 * If the new count is smaller, excess layers are removed.
		 * If larger, new default-constructed layers are added.
		 * @param pLayerCount The new total number of layers.
		 */
		void ResizeNeuroNet(int pLayerCount);

		/**
		 * @brief Sets the input for the first layer of the network.
		 * @param pInputMatrix A const reference to a matrix containing the input values for the network.
		 * @return bool True if input was successfully set, false otherwise (e.g., network has no layers).
		 */
		bool SetInput(const Matrix::Matrix<float>& pInputMatrix);

		/**
		 * @brief Calculates and returns the output of the entire network.
		 * This involves feeding the input through all layers sequentially.
		 * @return Matrix::Matrix<float> The final output matrix from the last layer.
		 */
		Matrix::Matrix<float> GetOutput();

		/**
		 * @brief Sets the input for the first layer of the network from a JSON string.
		 * The JSON string should represent an object with a key (e.g., "input_matrix")
		 * whose value is an array of arrays of numbers.
		 * @param json_input A const reference to a string containing the JSON input.
		 * @return bool True if input was successfully parsed and set, false otherwise.
		 * @throws JsonParseException if JSON parsing fails.
		 * @throws std::runtime_error if JSON structure is invalid or data conversion fails.
		 */
		bool SetInputJSON(const std::string& json_input);

		/**
		 * @brief Gets the output of the network and returns it as a JSON string.
		 * The JSON string will represent an object with a key (e.g., "output_matrix")
		 * whose value is an array of arrays of numbers.
		 * @return std::string A JSON string representing the network's output matrix.
		 * @throws std::runtime_error if the network has no layers or output cannot be generated.
		 */
		std::string GetOutputJSON();

		/**
		 * @brief Loads a vocabulary for string tokenization from a JSON file.
		 * @param filepath Path to the vocabulary JSON file.
		 * @return True if loading was successful, false otherwise.
		 */
		bool LoadVocabulary(const std::string& filepath);

		/**
		 * @brief Sets the input for the network from a batch of strings using JSON.
		 * The JSON string should contain a key (e.g., "input_batch") whose value is an array of strings.
		 * Uses the loaded vocabulary for tokenization and padding.
		 * @param json_string_input A const reference to a string containing the JSON input.
		 * @param max_len_override Optional: Override max sequence length for this batch.
		 *                         If -1, uses vocabulary's configured max length or pads to max in batch.
		 * @param pad_to_max_in_batch_override Optional: Override padding to max in batch if max_len_override and vocab max_len are not set.
		 * @return bool True if input was successfully parsed, tokenized, and set, false otherwise.
		 * @throws JsonParseException if JSON parsing fails.
		 * @throws std::runtime_error if vocabulary is not loaded, JSON structure is invalid, or data conversion/padding fails.
		 */
		bool SetStringsInput(const std::string& json_string_input, int max_len_override = -1, bool pad_to_max_in_batch_override = true);

		/**
		 * @brief Retrieves the weights for all layers in the network.
		 * @return std::vector<LayerWeights> A vector where each element contains the weights for a layer.
		 */
		std::vector<LayerWeights> get_all_layer_weights();

		/**
		 * @brief Sets the weights for all layers in the network.
		 * @param all_weights A vector of LayerWeights structs. The size must match the number of layers.
		 * @return bool True if all weights were set successfully, false otherwise (e.g., size mismatch).
		 */
		bool set_all_layer_weights(const std::vector<LayerWeights>& all_weights);

		/**
		 * @brief Retrieves the biases for all layers in the network.
		 * @return std::vector<LayerBiases> A vector where each element contains the biases for a layer.
		 */
		std::vector<LayerBiases> get_all_layer_biases();

		/**
		 * @brief Sets the biases for all layers in the network.
		 * @param all_biases A vector of LayerBiases structs. The size must match the number of layers.
		 * @return bool True if all biases were set successfully, false otherwise (e.g., size mismatch).
		 */
		bool set_all_layer_biases(const std::vector<LayerBiases>& all_biases);

		/**
		 * @brief Retrieves all weights from all layers, flattened into a single vector.
		 * This is useful for genetic algorithms or other optimization techniques.
		 * @return std::vector<float> A flat vector containing all weight values.
		 */
		std::vector<float> get_all_weights_flat() const;

		/**
		 * @brief Sets all weights for all layers from a single flattened vector.
		 * The order of weights in the vector must correspond to the order of layers and then
		 * the internal order of weights within each layer.
		 * @param all_weights_flat A flat vector containing all weight values.
		 * @return bool True if weights were set successfully, false if the vector size doesn't match total weights.
		 */
		bool set_all_weights_flat(const std::vector<float>& all_weights_flat);

		/**
		 * @brief Retrieves all biases from all layers, flattened into a single vector.
		 * Useful for genetic algorithms or similar optimization methods.
		 * @return std::vector<float> A flat vector containing all bias values.
		 */
		std::vector<float> get_all_biases_flat() const;

		/**
		 * @brief Sets all biases for all layers from a single flattened vector.
		 * The order of biases in the vector must correspond to the order of layers and then
		 * the internal order of biases within each layer.
		 * @param all_biases_flat A flat vector containing all bias values.
		 * @return bool True if biases were set successfully, false if the vector size doesn't match total biases.
		 */
		bool set_all_biases_flat(const std::vector<float>& all_biases_flat);

		/**
		 * @brief Saves the neural network model to a JSON file.
		 * @param filename The path to the file where the model will be saved.
		 * @return True if saving was successful, false otherwise.
		 */
		bool save_model(const std::string& filename) const;

		/**
		 * @brief Loads a neural network model from a JSON file.
		 * @param filename The path to the file from which the model will be loaded.
		 * @return A NeuroNet object populated with the loaded data.
		 * @throws std::runtime_error if loading fails (e.g., file not found, JSON parsing error, invalid format).
		 */
		static NeuroNet load_model(const std::string& filename);

		/**
		 * @brief Gets a reference to a specific layer in the network.
		 * @param index The index of the layer to retrieve.
		 * @return NeuroNetLayer& A reference to the layer.
		 * @throws std::out_of_range if the index is out of bounds.
		 */
		NeuroNetLayer& getLayer(int index);

		/**
		 * @brief Gets a const reference to a specific layer in the network.
		 * @param index The index of the layer to retrieve.
		 * @return const NeuroNetLayer& A const reference to the layer.
		 * @throws std::out_of_range if the index is out of bounds.
		 */
		const NeuroNetLayer& getLayer(int index) const;

		/**
		 * @brief Serializes the neural network to a JSON string using the custom JsonValue library.
		 * This method is responsible for its own memory management of temporary JsonValue objects.
		 * @return std::string A JSON string representing the network.
		 */
		std::string to_custom_json_string() const;

		/**
		 * @brief Gets the total number of layers in the network.
		 * @return int The number of layers.
		 */
		int getLayerCount() const;

		/**
		 * @brief Gets the input size for the entire neural network.
		 * @return int The number of input features.
		 */
		int GetInputSize() const;

		// Getter for vocabulary - DECLARATION ONLY
		const Vocabulary& getVocabulary() const;

    /**
     * @brief Performs backpropagation through the entire network.
     *
     * This method computes the gradient of the loss function with respect to the network's output
     * (actual_output - target_output, assuming MSE-like loss for the initial gradient calculation)
     * and then propagates this gradient backward through all layers. Each layer computes and stores
     * its own weight and bias gradients (dL/dW, dL/dB) internally.
     *
     * @param actual_output The matrix representing the actual output produced by the network for a given input.
     * @param target_output The matrix representing the target (true) output for that input.
     * @throws std::runtime_error if actual_output and target_output dimensions mismatch, or if issues occur
     *                            during a layer's BackwardPass (e.g., empty input matrix for a layer).
     * @throws std::out_of_range if layer indexing fails during backpropagation.
     */
    void Backpropagate(const Matrix::Matrix<float>& actual_output, const Matrix::Matrix<float>& target_output);

    /**
     * @brief Updates the weights and biases of all layers in the network.
     *
     * This method applies the gradients (dL/dW, dL/dB) that were computed and stored by
     * the `Backpropagate` method. The update rule used is:
     * new_weight = old_weight - learning_rate * dL/dW
     * new_bias   = old_bias   - learning_rate * dL/dB
     *
     * @param learning_rate The learning rate to scale the gradients. Must be positive.
     * @throws std::runtime_error if issues occur during weight/bias reconstruction or setting,
     *                            or if gradient dimensions are inconsistent.
     * @throws std::out_of_range if layer indexing fails.
     */
    void UpdateWeights(float learning_rate);

    /**
     * @brief Trains the neural network using stochastic gradient descent (SGD).
     *
     * This method iterates over the training data for a specified number of epochs.
     * In each iteration, for each training sample, it performs a forward pass,
     * then backpropagation to compute gradients, and finally updates the network weights.
     *
     * @param training_inputs A vector of matrices, where each matrix represents a single training input sample.
     *                        Each input sample matrix is typically 1xN, where N is the network's input size.
     * @param training_targets A vector of matrices, where each matrix represents the corresponding target output
     *                         for a training input sample. Each target sample matrix is typically 1xM, where M
     *                         is the network's output size.
     * @param learning_rate The learning rate to be used for updating weights. Must be positive.
     * @param epochs The total number of epochs (passes over the entire training dataset) to train for. Must be positive.
     * @throws std::invalid_argument if parameters are invalid (e.g., empty data, mismatched sizes, non-positive learning rate/epochs).
     * @throws std::runtime_error if the network is not configured (e.g., no layers) or other runtime issues occur during training,
     *                            such as dimension mismatches in input/target samples or issues during forward/backward passes.
     */
    void Train(const std::vector<Matrix::Matrix<float>>& training_inputs,
               const std::vector<Matrix::Matrix<float>>& training_targets,
               float learning_rate,
               int epochs);

	private:
		int InputSize = 0; ///< Number of input features for the entire network.
		int LayerCount = 0; ///< Total number of layers in the network.
		std::vector<NeuroNetLayer> NeuroNetVector; ///< Vector storing all layers of the network.
		Vocabulary vocabulary; // Vocabulary for string processing
	};

// Inline definition for getVocabulary (outside the class body but in the header for inlining)
// This ensures NeuroNet class is fully defined before its methods are defined.
inline const Vocabulary& NeuroNet::getVocabulary() const {
    return vocabulary;
}

inline int NeuroNet::GetInputSize() const {
    return InputSize;
}

} // namespace NeuroNet