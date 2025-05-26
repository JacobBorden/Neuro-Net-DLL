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

#include "../math/matrix.h" // For Matrix class usage
#include <vector>   // For std::vector usage
#include <string>   // For potential string usage in future extensions
#include "../utilities/json/json.hpp" // For Json::Value

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

	private:
		int InputSize = 0; ///< Number of input features for the entire network.
		int LayerCount = 0; ///< Total number of layers in the network.
		std::vector<NeuroNetLayer> NeuroNetVector; ///< Vector storing all layers of the network.
	};
} // namespace NeuroNet