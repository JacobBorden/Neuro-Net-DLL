# Neural Network Module (`NeuroNet`)

This document details the core components of the neural network module, primarily focusing on the `NeuroNet` and `NeuroNetLayer` classes.

## Overview

The neural network module forms the backbone of this library, providing the tools to define, configure, and run feed-forward neural networks. It includes functionalities for layer management, weight and bias handling, activation functions, and model serialization.

## Key Classes and Structs

### 1. `NeuroNet::NeuroNet` Class

*   **Purpose:** Acts as the main container and manager for a neural network. It encapsulates all layers and provides an interface for the network as a whole.
*   **Core Responsibilities:**
    *   Managing the overall architecture (number and size of layers).
    *   Handling data flow from input to output.
    *   Saving and loading model architecture and parameters.
*   **Key Functionalities (refer to Doxygen API docs for full details):**
    *   `SetInputSize(int pInputSize)`: Defines the network's input feature count.
    *   `ResizeNeuroNet(int pLayerCount)`: Adjusts the total number of layers.
    *   `ResizeLayer(int pLayerIndex, int pLayerSize)`: Configures neuron count for a specific layer.
    *   `SetInput(const Matrix::Matrix<float>& pInputMatrix)`: Feeds input data to the network.
    *   `GetOutput()`: Retrieves the network's output after processing the input.
    *   `set_all_weights_flat(const std::vector<float>& all_weights_flat)`, `get_all_weights_flat() const`: Manage all network weights as a single vector.
    *   `set_all_biases_flat(const std::vector<float>& all_biases_flat)`, `get_all_biases_flat() const`: Manage all network biases as a single vector.
    *   `save_model(const std::string& filename) const`: Saves the model to a JSON file.
    *   `static NeuroNet load_model(const std::string& filename)`: Loads a model from a JSON file.
    *   `LoadVocabulary(const std::string& filepath)`: Loads a vocabulary for text processing.
    *   `SetStringsInput(const std::string& json_string_input, ...)`: Processes string inputs using the vocabulary.
    *   `Backpropagate(const Matrix::Matrix<float>& actual_output, const Matrix::Matrix<float>& target_output)`: Computes and propagates gradients backward through the network.
    *   `UpdateWeights(float learning_rate)`: Updates network weights and biases based on computed gradients.
    *   `Train(const std::vector<Matrix::Matrix<float>>& training_inputs, ...)`: Trains the network using specified data, learning rate, and epochs.
*   **Source:** `src/neural_network/neuronet.h`

### 2. `NeuroNet::NeuroNetLayer` Class

*   **Purpose:** Represents an individual layer within a `NeuroNet`.
*   **Core Responsibilities:**
    *   Managing its own weights and biases.
    *   Performing forward propagation: `output = activation(input * weights + biases)`.
    *   Applying a configurable activation function.
*   **Key Functionalities (refer to Doxygen API docs for full details):**
    *   `ResizeLayer(int pInputSize, int pLayerSize)`: Configures layer dimensions.
    *   `SetInput(const Matrix::Matrix<float>& pInputMatrix)`: Sets input for the layer.
    *   `CalculateOutput()`: Computes and returns the layer's output.
    *   `SetWeights(LayerWeights pWeights)`, `get_weights() const`: Manage layer weights.
    *   `SetBiases(LayerBiases pBiases)`, `get_biases() const`: Manage layer biases.
    *   `SetActivationFunction(ActivationFunctionType pActivationFunction)`: Sets the activation function for the layer.
    *   `get_activation_type() const`: Gets the current activation function.
    *   `DerivativeReLU(const Matrix::Matrix<float>& activated_output) const`: Computes derivative of ReLU.
    *   `DerivativeLeakyReLU(const Matrix::Matrix<float>& activated_output) const`: Computes derivative of Leaky ReLU.
    *   `DerivativeELU(const Matrix::Matrix<float>& activated_output) const`: Computes derivative of ELU.
    *   `DerivativeSoftmax(const Matrix::Matrix<float>& activated_output) const`: Computes derivative of Softmax.
    *   `BackwardPass(const Matrix::Matrix<float>& dLdOutput, ...)`: Performs backpropagation for the layer.
    *   `get_dLdW() const`: Retrieves stored weight gradients (dL/dW).
    *   `get_dLdB() const`: Retrieves stored bias gradients (dL/dB).
*   **Source:** `src/neural_network/neuronet.h`

### 3. `NeuroNet::LayerWeights` and `NeuroNet::LayerBiases` Structs

*   **Purpose:** Simple data structures for packaging weight and bias values for a single `NeuroNetLayer`.
    *   `LayerWeights`: Contains `WeightCount` and `std::vector<float> WeightsVector`.
    *   `LayerBiases`: Contains `BiasCount` and `std::vector<float> BiasVector`.
*   **Usage:** Used by `NeuroNetLayer` getter/setter methods and for network-wide parameter management.
*   **Source:** `src/neural_network/neuronet.h`

### 4. Activation Functions (`NeuroNet::ActivationFunctionType` Enum)

*   **Purpose:** `NeuroNetLayer` supports various activation functions applied after the linear transformation.
*   **Supported Functions:**
    *   `None`: Linear output.
    *   `ReLU`: Rectified Linear Unit.
    *   `LeakyReLU`: Leaky Rectified Linear Unit.
    *   `ELU`: Exponential Linear Unit.
    *   `Softmax`: Normalizes outputs to a probability distribution.
*   **Configuration:** Set per-layer using `NeuroNetLayer::SetActivationFunction()`.
*   **Source:** `src/neural_network/neuronet.h`

## Basic Usage Example: Creating and Configuring a Network

```cpp
#include "neural_network/neuronet.h" // For NeuroNet::NeuroNet
#include <iostream>

int main() {
    // Create a NeuroNet instance
    NeuroNet::NeuroNet myNetwork;

    // Define network parameters
    int inputFeatures = 10;
    int hiddenNeurons = 20;
    int outputNeurons = 5;

    // Configure the network
    myNetwork.SetInputSize(inputFeatures);
    myNetwork.ResizeNeuroNet(2); // 1 hidden layer, 1 output layer

    // Configure hidden layer (layer 0)
    myNetwork.ResizeLayer(0, hiddenNeurons);
    myNetwork.getLayer(0).SetActivationFunction(NeuroNet::ActivationFunctionType::ReLU);

    // Configure output layer (layer 1)
    myNetwork.ResizeLayer(1, outputNeurons);
    myNetwork.getLayer(1).SetActivationFunction(NeuroNet::ActivationFunctionType::Softmax); // Example for classification

    std::cout << "Network configured with:" << std::endl;
    std::cout << "Input Size: " << myNetwork.GetInputSize() << std::endl;
    std::cout << "Layer 0 (Hidden): " << myNetwork.getLayer(0).LayerSize() << " neurons, Activation: " << myNetwork.getLayer(0).get_activation_function_name() << std::endl;
    std::cout << "Layer 1 (Output): " << myNetwork.getLayer(1).LayerSize() << " neurons, Activation: " << myNetwork.getLayer(1).get_activation_function_name() << std::endl;

    // Further steps would involve setting weights/biases (e.g., by training or loading)
    // and processing data using myNetwork.SetInput() and myNetwork.GetOutput().

    return 0;
}
```

## Model Serialization

The `NeuroNet` class supports saving its current state (architecture, weights, biases, vocabulary if loaded) to a JSON file and loading it back.

*   `myNetwork.save_model("model.json");`
*   `NeuroNet loadedNetwork = NeuroNet::NeuroNet::load_model("model.json");`

Refer to the main `README.md` for details on the JSON format.

(This is a starting point. More details on specific functionalities or advanced usage can be added.)
