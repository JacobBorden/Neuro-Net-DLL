# NeuroNet with Genetic Algorithm

This project provides a C++ library with tools for creating and training neural networks.
The core components of the library are:
*   The `NeuroNet` class (from `src/neural_network/neuronet.h`): Used for defining the structure and operations of neural networks.
*   The `GeneticAlgorithm` class (from `src/optimization/genetic_algorithm.h`): Used for training and optimizing these neural networks using evolutionary strategies.
It also includes a matrix library for underlying mathematical operations.

## Class Explanations

This section provides details on the core classes and data structures used in the NeuroNet library.

### `NeuroNet::NeuroNet` Class

*   **Role:** The `NeuroNet` class is the main container and manager for a neural network. It encapsulates all the layers and provides an interface for interacting with the network as a whole.
*   **Capabilities:** It is responsible for managing the overall architecture of the network, including the number and size of its layers. It handles the flow of data through the network, from input to output.
*   **Key Functionalities:**
    *   `ResizeNeuroNet(int pLayerCount)`: Adjusts the total number of layers in the network.
    *   `ResizeLayer(int pLayerIndex, int pLayerSize)`: Modifies the number of neurons in a specific layer. The input size of the first layer is set by `SetInputSize`, and for subsequent layers, it's determined by the output size of the preceding layer.
    *   `SetInputSize(int pInputSize)`: Defines the number of input features the network expects.
    *   `SetInput(const Matrix::Matrix<float>& pInputMatrix)`: Provides an input matrix to the first layer of the network.
    *   `GetOutput()`: Processes the current input through all layers and returns the final output matrix from the last layer.
    *   `get_all_layer_weights()`, `set_all_layer_weights(...)`: Access and modify the weights of all layers, typically as a vector of `LayerWeights` structs.
    *   `get_all_layer_biases()`, `set_all_layer_biases(...)`: Access and modify the biases of all layers, typically as a vector of `LayerBiases` structs.
    *   `get_all_weights_flat()`, `set_all_weights_flat(...)`: Get or set all network weights as a single flat vector of floats, useful for optimization algorithms.
    *   `get_all_biases_flat()`, `set_all_biases_flat(...)`: Get or set all network biases as a single flat vector of floats.
    *   `save_model(const std::string& filename) const`: Saves the network structure and parameters to a JSON file.
    *   `static NeuroNet load_model(const std::string& filename)`: Loads a network from a JSON file.
*   **Source:** Defined in `src/neural_network/neuronet.h`.

### `NeuroNet::NeuroNetLayer` Class

*   **Role:** The `NeuroNetLayer` class represents an individual layer within a `NeuroNet`. It's a fundamental building block of the network.
*   **Responsibilities:** Each `NeuroNetLayer` instance manages its own set of weights and biases. Its primary responsibility is to perform the forward propagation calculation: taking an input matrix, multiplying it by its weight matrix, adding its bias matrix, and producing an output matrix. It also handles applying a configured activation function.
*   **Interaction:** The `NeuroNet` class contains a `std::vector<NeuroNetLayer>` to store and manage all the layers that constitute the network. When `NeuroNet::GetOutput()` is called, it iteratively passes the output of one layer as the input to the next.
*   **Key Functionalities:**
    *   `ResizeLayer(int pInputSize, int pLayerSize)`: Configures the dimensions of the layer's weight and bias matrices.
    *   `SetInput(const Matrix::Matrix<float>& pInputMatrix)`: Sets the input for this specific layer.
    *   `CalculateOutput()`: Computes the layer's output based on its current input, weights, biases, and selected activation function.
    *   `ReturnOutputMatrix()`: Returns the last computed output matrix.
    *   `SetWeights(LayerWeights pWeights)`, `get_weights()`: Set or get the layer's weights.
    *   `SetBiases(LayerBiases pBiases)`, `get_biases()`: Set or get the layer's biases.
    *   `SetActivationFunction(ActivationFunctionType pActivationFunction)`: Sets the activation function for the layer.
    *   `get_activation_type() const`: Gets the currently configured activation function type.
*   **Source:** Defined in `src/neural_network/neuronet.h`.

### `NeuroNet::LayerWeights` and `NeuroNet::LayerBiases` Structs

*   **Purpose:** These are simple C++ structures designed for data aggregation.
    *   `LayerWeights`: Holds an `int WeightCount` and a `std::vector<float> WeightsVector`. It's used to package all weight values for a single `NeuroNetLayer`.
    *   `LayerBiases`: Holds an `int BiasCount` and a `std::vector<float> BiasVector`. It's used to package all bias values for a single `NeuroNetLayer`.
*   **Usage:** They provide a convenient way to get and set weights and biases for layers, facilitating the transfer of these parameters, for example, when initializing a network or when a genetic algorithm modifies them.
*   **Source:** Defined in `src/neural_network/neuronet.h`.

### `Optimization::GeneticAlgorithm` Class

*   **Role:** The `Optimization::GeneticAlgorithm` class is used to train or evolve the parameters (weights and biases) of `NeuroNet` instances. It employs evolutionary strategies to search for optimal network configurations.
*   **Concepts:**
    *   **Population:** The GA maintains a collection (`std::vector<NeuroNet::NeuroNet>`) of `NeuroNet` individuals. Each individual represents a potential solution (a specific set of weights and biases).
    *   **Fitness Function:** The user must provide a fitness function (`std::function<double(NeuroNet::NeuroNet&)>`). This function evaluates a given `NeuroNet` individual and returns a score (typically, higher is better) indicating how well it performs a target task.
    *   **Selection:** Based on their fitness scores, individuals are selected to become "parents" for the next generation. This class uses tournament selection and elitism (ensuring the best individual from one generation is carried over to the next).
    *   **Crossover:** Selected parents are combined to produce "offspring." This involves exchanging genetic material (parts of their flattened weight and bias vectors) with a certain `crossover_rate`. The goal is to combine beneficial traits from different parents.
    *   **Mutation:** Each weight and bias in an offspring's network has a small chance (`mutation_rate`) of being randomly altered. This introduces new genetic variations into the population, helping to avoid premature convergence and explore more of the solution space.
*   **Key Functionalities:**
    *   Constructor `GeneticAlgorithm(...)`: Initializes the GA with parameters like population size, mutation rate, crossover rate, number of generations, and a `template_network` which defines the architecture of the individuals.
    *   `initialize_population()`: Creates the initial population of random `NeuroNet` individuals based on the template network.
    *   `run_evolution(const std::function<double(NeuroNet::NeuroNet&)>& fitness_function)`: Executes the main evolutionary loop for a specified number of generations. This involves repeated cycles of fitness evaluation, selection, crossover, and mutation.
    *   `evolve_one_generation(...)`: Carries out a single step of the evolutionary process.
    *   `get_best_individual()`: After the evolution process, this function returns the `NeuroNet` individual that achieved the highest fitness score.
*   **Source:** Defined in `src/optimization/genetic_algorithm.h`.

## Activation Functions

`NeuroNetLayer` now supports pluggable activation functions. You can set an activation function for each layer individually. After the standard linear transformation `(InputMatrix * WeightMatrix) + BiasMatrix`, the selected activation function is applied to the result.

Supported activation functions (defined in `NeuroNet::ActivationFunctionType`):
- `None` (0): No activation (linear output). This is the default.
- `ReLU` (1): Rectified Linear Unit. Output is `max(0, x)`.
- `LeakyReLU` (2): Leaky Rectified Linear Unit. Output is `x` if `x > 0`, otherwise `alpha*x` (current alpha = 0.01).
- `ELU` (3): Exponential Linear Unit. Output is `x` if `x > 0`, otherwise `alpha*(exp(x)-1)` (current alpha = 1.0).
- `Softmax` (4): Softmax function. Normalizes outputs to a probability distribution, suitable for output layers in classification tasks.

### Example Usage:

```cpp
#include "neural_network/neuronet.h" // Or appropriate path
#include "math/matrix.h" // For Matrix::Matrix

int main() {
    // Create a layer
    NeuroNet::NeuroNetLayer myLayer;
    myLayer.ResizeLayer(10, 5); // 10 inputs, 5 neurons

    // Set ReLU activation for this layer
    myLayer.SetActivationFunction(NeuroNet::ActivationFunctionType::ReLU);

    // --- Setup: Provide dummy input, weights, and biases for demonstration ---
    Matrix::Matrix<float> input(1, 10); // 1 sample, 10 features
    for(int i=0; i<10; ++i) input[0][i] = (i % 3) - 1.0f; // Some -1, 0, 1 values

    NeuroNet::LayerWeights weights;
    weights.WeightCount = 10 * 5;
    weights.WeightsVector.assign(weights.WeightCount, 0.1f); // All weights 0.1

    NeuroNet::LayerBiases biases;
    biases.BiasCount = 5;
    biases.BiasVector.assign(biases.BiasCount, 0.05f); // All biases 0.05

    myLayer.SetInput(input);
    myLayer.SetWeights(weights);
    myLayer.SetBiases(biases);
    // --- End Setup ---

    // Calculate output - it will have ReLU applied
    Matrix::Matrix<float> layerOutput = myLayer.CalculateOutput(); 

    // Example with Softmax for an output layer
    NeuroNet::NeuroNetLayer outputLayer;
    outputLayer.ResizeLayer(20, 3); // 20 inputs from previous layer, 3 output classes
    outputLayer.SetActivationFunction(NeuroNet::ActivationFunctionType::Softmax);
    // ... further setup for outputLayer (input, weights, biases) ...
    
    // (Code to print layerOutput would go here if desired)

    return 0;
}

```

## Model Serialization

This library supports saving and loading NeuroNet models to and from a human-readable JSON format using its internal custom JSON utilities. This allows for model persistence, inspection, and transfer primarily within the library's ecosystem.

## Exporting Training Metrics and Model Structure

The library supports exporting detailed training progress and model architectures using its **internal custom JSON tools**.

### 1. Overview
- **`training_metrics.json`**: This file is generated by `GeneticAlgorithm::export_training_metrics_json()` and uses the library's custom JSON format. It captures:
    - Overall statistics for a training run (start/end times, total generations, overall best fitness).
    - The best model's architecture, embedded as an *escaped JSON string* (which itself is also in the custom library format).
    - Per-generation metrics (average fitness, best fitness, loss, accuracy).
- **Model Files (e.g., `best_model_custom_format.json`)**: These files are generated by `NeuroNet::save_model()` and also use the library's custom JSON format. They are primarily intended for use with `NeuroNet::load_model()`.

### 2. Generating Exported Files
An example script `examples/basic_usage_export.cpp` (target name `BasicUsageExport` after building) demonstrates how to generate these files. This script:
1. Configures and trains a simple `NeuroNet` using `GeneticAlgorithm`.
2. Calls `ga_instance.export_training_metrics_json("training_metrics.json")` to save the training log.
3. Retrieves the best model and saves it using `best_model.save_model("best_model_custom_format.json")`.

The key output files demonstrated are:
- `training_metrics.json`
- `best_model_custom_format.json`

### 3. `training_metrics.json` Format (Custom JSON)
This entire file is generated as a string by the custom `JsonValue::ToString()` method.

```json
// Content of training_metrics.json 
{
    "start_time": "YYYY-MM-DDTHH:MM:SSZ",
    "end_time": "YYYY-MM-DDTHH:MM:SSZ",
    "total_generations": 100.0, // Custom JSON often outputs numbers as doubles
    "overall_best_fitness": 0.95,
    "best_model_architecture_params_custom_json_string": "{\"input_size\":2.0,\"layer_count\":2.0,\"layers\":[...]}", // Escaped JSON string
    "generation_data": [
        { 
            "generation_number": 1.0, 
            "average_fitness": 0.85, 
            "best_fitness": 0.92,
            "loss": null, // Custom JSON null
            "accuracy": null // Custom JSON null
        }
        // ... more generations
    ]
}
```
- `start_time`/`end_time`: Timestamps for the training run (string).
- `total_generations`: The number of generations run (number, likely double).
- `overall_best_fitness`: The best fitness score achieved during the run (number, likely double).
- `best_model_architecture_params_custom_json_string`: An **escaped JSON string** representing the best model's architecture, generated by `NeuroNet::to_custom_json_string()`. This string itself is in the custom JSON format.
- `generation_data`: An array of objects, where each object represents metrics for one generation.
    - `loss`/`accuracy`: Placeholders; default to `null` if not explicitly computed by the fitness function.
- Note: The custom JSON library might have specific output characteristics (e.g., numbers as doubles `100.0`, `null` representation as the string "null").

### 4. Model Structure (from Custom JSON Library)
This refers to the content of the `model_custom_json_string` (once unescaped and parsed from `training_metrics.json`) and the format of files saved by `NeuroNet::save_model()` (e.g., `best_model_custom_format.json`).

```json
// Content of model_custom_json_string (after parsing the string)
// or content of a file from NeuroNet::save_model()
{
    "input_size": 2.0,
    "layer_count": 2.0,
    "layers": [
        {
            "input_size": 2.0,
            "layer_size": 3.0,
            "activation_function": "ReLU", // String representation
            "weights": {
                "rows": 2.0,
                "cols": 3.0,
                "data": [0.1, -0.2, /* ... */, 0.5]
            },
            "biases": {
                "rows": 1.0,
                "cols": 3.0,
                "data": [0.01, 0.02, 0.03]
            }
        }
        // ... more layers
    ]
}
```
- `activation_function`: Stored as a string (e.g., "ReLU", "Softmax", "None").
- Numeric values (like sizes, rows, cols) are typically output as doubles (e.g., `2.0`) by the custom JSON library.
- The structure is defined by `NeuroNet::to_custom_json_string()` and used by `NeuroNet::save_model()`/`NeuroNet::load_model()`.

### 5. Visualizing and Parsing Exported Data (Python Example)

#### Training Metrics and Embedded Model
The following Python script demonstrates how to load `training_metrics.json`, parse its content (including the embedded model string), and visualize training progress.

```python
import json
import matplotlib.pyplot as plt
import pandas as pd

file_path = "training_metrics.json"
metrics_data = None

# Load and parse the main training metrics file
try:
    with open(file_path, "r") as f:
        # The entire file is a string output from the custom JSON library.
        # We assume Python's json.loads() can handle this format.
        # If the custom format is very non-standard (e.g., uses single quotes, unquoted keys),
        # this step might require custom preprocessing or a more lenient parser.
        metrics_data_str = f.read()
        metrics_data = json.loads(metrics_data_str) 
except json.JSONDecodeError as e:
    print(f"Error decoding {file_path}: {e}")
    print("Note: The custom JSON format from the C++ library might require specific parsing "
          "if it's not strictly standard (e.g., regarding quotes, trailing commas, number formats).")
    exit()
except FileNotFoundError:
    print(f"Error: {file_path} not found.")
    exit()

if not metrics_data:
    exit()

# Extract and parse the embedded custom JSON model string
model_json_str_key = "best_model_architecture_params_custom_json_string"
overall_fitness_key = "overall_best_fitness"

if model_json_str_key in metrics_data:
    best_model_json_string = metrics_data[model_json_str_key]
    if isinstance(best_model_json_string, str) and best_model_json_string: # Check if it's a non-empty string
        try:
            # Python's json.loads handles unescaping of strings from a parent JSON object.
            best_model_structure = json.loads(best_model_json_string)
            print("Best Model Structure (from custom JSON string):")
            print(json.dumps(best_model_structure, indent=4))
        except json.JSONDecodeError as e:
            print(f"Error decoding embedded model string: {e}")
            print(f"Model string was: {best_model_json_string}")
    elif best_model_json_string: # It might already be parsed if not properly escaped, or be an error object
        print("Embedded model data (potentially pre-parsed or error object):")
        print(json.dumps(best_model_json_string, indent=4)) # Print what we got
    else:
        print("Embedded model string is empty or missing.")
else:
    print(f"Key '{model_json_str_key}' not found in metrics_data.")

if overall_fitness_key in metrics_data:
    print(f"Overall best fitness: {metrics_data[overall_fitness_key]}")
else:
    print(f"Key '{overall_fitness_key}' not found in metrics_data.")

# Convert generation data to a Pandas DataFrame for plotting
generation_data_list = metrics_data.get("generation_data", [])
if generation_data_list: # Check if list is not empty
    df_generations = pd.DataFrame(generation_data_list)
    required_cols = ["generation_number", "average_fitness", "best_fitness"]
    if all(col in df_generations.columns for col in required_cols):
        plt.figure(figsize=(10, 6))
        plt.plot(df_generations["generation_number"], df_generations["average_fitness"], label="Average Fitness")
        plt.plot(df_generations["generation_number"], df_generations["best_fitness"], label="Best Fitness")
        plt.xlabel("Generation")
        plt.ylabel("Fitness")
        plt.title("Training Progress")
        plt.legend()
        plt.grid(True)
        plt.show()
    else:
        print("Generation data is missing required columns for plotting.")
else:
    print("No generation data to plot or 'generation_data' key missing.")

```
Ensure you have `matplotlib` and `pandas` installed (`pip install matplotlib pandas`).

#### Direct Model Files (Custom Format)
Files saved by `NeuroNet::save_model()` (e.g., `best_model_custom_format.json`) are also in the custom JSON format. They can be loaded and parsed in Python similarly:

```python
# Example for loading a model file directly (e.g., best_model_custom_format.json)
# with open("best_model_custom_format.json", "r") as f:
#     try:
#         model_file_str = f.read()
#         direct_model_data = json.loads(model_file_str)
#         print("\nDirectly Loaded Model Data (from custom format file):")
#         print(json.dumps(direct_model_data, indent=4))
#     except json.JSONDecodeError as e:
#         print(f"\nError decoding direct model file: {e}")
#     except FileNotFoundError:
#         print("\nError: Direct model file not found.")
```
**Note on Parsing Custom JSON:** The library's custom `JsonValue::ToString()` method aims to produce valid JSON. However, standard Python JSON parsers are strict. If the custom output has minor non-standard features (e.g., specific number formatting like always including `.0`, handling of trailing commas, or non-standard `null` representation if it differs from JSON `null`), direct parsing with `json.loads()` might require adjustments or a more lenient parser. For use within the C++ library, `JsonParser::Parse()` is designed to understand the output of `JsonValue::ToString()`.

## Performance Benchmarking and Optimization

This library includes tools and features for performance analysis and optimization:

*   **Timing Instrumentation:** Instrumentation for core operations such as matrix multiplication, neural network forward pass (full pass and per-layer), and key genetic algorithm steps (selection, crossover, mutation, evaluation) has been added. This allows for detailed performance monitoring.
*   **Activation:** To enable the timing output from this instrumentation, the project must be compiled with the `ENABLE_BENCHMARKING` macro defined. You can define this in your build system or by adding `#define ENABLE_BENCHMARKING` at the top of specific `.cpp` files before including library headers.
*   **Benchmark Scenarios:** The file `tests/test_benchmarks.cpp` provides example usage of the instrumented features and serves as a suite for running these benchmarks. It demonstrates how to set up and execute tests for matrix operations, neural network forward passes of varying sizes, and genetic algorithm operations.
*   **OpenMP Optimization:** Matrix multiplication (`Matrix<T>::operator*`) has been parallelized using OpenMP to leverage multi-core processors for improved performance, especially with large matrices.
*   **Compiling with OpenMP:** To benefit from this parallelization, the project must be compiled with OpenMP support enabled. This typically involves setting a compiler flag (e.g., `/openmp` for MSVC, `-fopenmp` for GCC/Clang).

Refer to `tests/test_benchmarks.cpp` for an example of how to run these benchmarks.

### JSON Format

The NeuroNet model is serialized into a JSON object with the following structure:

```json
{
  "input_size": <integer>,      // Number of input features for the network
  "layer_count": <integer>,     // Total number of layers in the network
  "layers": [                   // Array of layer objects
    {
      "input_size": <integer>,  // Number of inputs for this specific layer
      "layer_size": <integer>,  // Number of neurons (outputs) in this layer
      "activation_function": <integer>, // Integer code for ActivationFunctionType (0: None, 1: ReLU, 2: LeakyReLU, 3: ELU, 4: Softmax - refer to src/neural_network/neuronet.h for exact mapping)
      "weights": {
        "rows": <integer>,      // Number of rows in the weight matrix (equal to layer's input_size)
        "cols": <integer>,      // Number of columns in the weight matrix (equal to layer_size)
        "data": [<float>, ...]  // Flattened array of weight values (row-major order)
      },
      "biases": {
        "rows": <integer>,      // Number of rows in the bias matrix (typically 1)
        "cols": <integer>,      // Number of columns in the bias matrix (equal to layer_size)
        "data": [<float>, ...]  // Flattened array of bias values
      }
    },
    // ... more layers ...
  ]
}
```

### Usage

#### Saving a Model

To save a `NeuroNet` model to a file, use the `save_model` member function.

```cpp
#include "neural_network/neuronet.h" // Adjust path as needed
#include <string>
#include <iostream> // For std::cout, std::cerr

// Assuming 'my_model' is an instance of NeuroNet::NeuroNet
// and is already configured and trained.

int main() {
    // Example: Create a simple model to save
    NeuroNet::NeuroNet my_model;
    my_model.SetInputSize(2);
    my_model.ResizeNeuroNet(1);
    my_model.ResizeLayer(0,1);
    my_model.NeuroNetVector[0].SetActivationFunction(NeuroNet::ActivationFunctionType::ReLU); // Example activation

    // Add some dummy weights/biases for a complete file
    NeuroNet::LayerWeights lw; 
    lw.WeightCount = 2*1; 
    lw.WeightsVector = {0.5f, -0.5f};
    my_model.NeuroNetVector[0].SetWeights(lw);

    NeuroNet::LayerBiases lb; 
    lb.BiasCount = 1; 
    lb.BiasVector = {0.1f};
    my_model.NeuroNetVector[0].SetBiases(lb);

    std::string filename = "my_neural_network.json";
    bool success = my_model.save_model(filename);

    if (success) {
        std::cout << "Model saved successfully to " << filename << std::endl;
    } else {
        std::cerr << "Error saving model to " << filename << std::endl;
    }
    return 0;
}
```

#### Loading a Model

To load a `NeuroNet` model from a file, use the static `load_model` function. This function returns a new `NeuroNet` object.

```cpp
#include "neural_network/neuronet.h" // Adjust path as needed
#include <string>
#include <stdexcept> // For std::runtime_error
#include <iostream>  // For std::cout, std::cerr

// Assuming "my_neural_network.json" exists from the saving example.
int main() {
    std::string load_filename = "my_neural_network.json"; 
    NeuroNet::NeuroNet loaded_model;

    try {
        loaded_model = NeuroNet::NeuroNet::load_model(load_filename);
        std::cout << "Model loaded successfully from " << load_filename << std::endl;
        
        // Model loaded successfully, use 'loaded_model'
        // Example: Print basic info about the loaded model
        std::cout << "Loaded model input size: " << loaded_model.InputSize << std::endl;
        std::cout << "Loaded model has " << loaded_model.NeuroNetVector.size() << " layers." << std::endl;
        if (!loaded_model.NeuroNetVector.empty()) {
            std::cout << "First layer output size: " << loaded_model.NeuroNetVector[0].LayerSize() << std::endl;
            std::cout << "First layer activation: " << static_cast<int>(loaded_model.NeuroNetVector[0].get_activation_type()) << std::endl;
        }

    } catch (const std::runtime_error& e) {
        std::cerr << "Error loading model: " << e.what() << std::endl;
    }
    return 0;
}
```

## Usage Examples

This section provides C++ code examples to illustrate how to use the NeuroNet library for common tasks.

### 1. Creating a Neural Network

This example shows how to instantiate and configure a `NeuroNet::NeuroNet`.

```cpp
#include "neural_network/neuronet.h" // For NeuroNet::NeuroNet
#include <iostream>
#include <vector>

int main() {
    // Create a NeuroNet instance
    NeuroNet::NeuroNet myNetwork;

    // Set the input size for the network (e.g., 5 input features)
    int inputFeatureCount = 5;
    myNetwork.SetInputSize(inputFeatureCount);

    // Configure the network architecture: 2 layers (1 hidden, 1 output)
    myNetwork.ResizeNeuroNet(2); // Total number of layers

    // Define layer sizes
    int hiddenLayerNeurons = 10;
    int outputLayerNeurons = 2;

    // Size the first layer (hidden layer)
    // Input size for layer 0 is taken from myNetwork.SetInputSize()
    // Output size is hiddenLayerNeurons
    myNetwork.ResizeLayer(0, hiddenLayerNeurons);

    // Size the second layer (output layer)
    // Input size for layer 1 is taken from layer 0's output size (hiddenLayerNeurons)
    // Output size is outputLayerNeurons
    myNetwork.ResizeLayer(1, outputLayerNeurons);

    std::cout << "Neural network created with " << inputFeatureCount << " inputs, "
              << "1 hidden layer with " << hiddenLayerNeurons << " neurons, and "
              << "1 output layer with " << outputLayerNeurons << " neurons." << std::endl;

    return 0;
}
```

### 2. Setting Weights and Biases Manually (Optional)

You might want to set weights and biases manually if you're loading a pre-trained model or for specific testing scenarios. You can do this per layer or for the entire network using flattened vectors.

```cpp
#include "neural_network/neuronet.h"
#include <vector>
#include <iostream>

// Helper function to calculate total weights for a given architecture
// (This is a simplified example; NeuroNet does this internally when you use flat vectors)
int calculate_total_weights(int input_size, const std::vector<int>& layer_sizes) {
    int total_weights = 0;
    int current_input_size = input_size;
    for (int neurons_in_layer : layer_sizes) {
        total_weights += current_input_size * neurons_in_layer;
        current_input_size = neurons_in_layer; // Output of current layer is input to next
    }
    return total_weights;
}

// Helper function to calculate total biases
int calculate_total_biases(const std::vector<int>& layer_sizes) {
    int total_biases = 0;
    for (int neurons_in_layer : layer_sizes) {
        total_biases += neurons_in_layer;
    }
    return total_biases;
}


int main() {
    NeuroNet::NeuroNet network;
    int inputSize = 3;
    network.SetInputSize(inputSize);

    std::vector<int> layer_neuron_counts = {4, 2}; // Hidden layer: 4 neurons, Output layer: 2 neurons
    network.ResizeNeuroNet(layer_neuron_counts.size());
    network.ResizeLayer(0, layer_neuron_counts[0]); // Hidden layer
    network.ResizeLayer(1, layer_neuron_counts[1]); // Output layer

    // Example using flat vectors (set_all_weights_flat / set_all_biases_flat)
    // This is generally more convenient if you have all parameters in a single sequence.

    int total_weights_count = calculate_total_weights(inputSize, layer_neuron_counts);
    std::vector<float> all_weights_flat(total_weights_count);
    // Populate all_weights_flat with your desired weight values
    // For demonstration, let's fill with 0.1
    for(size_t i = 0; i < all_weights_flat.size(); ++i) all_weights_flat[i] = 0.1f * (i + 1);

    int total_biases_count = calculate_total_biases(layer_neuron_counts);
    std::vector<float> all_biases_flat(total_biases_count);
    // Populate all_biases_flat with your desired bias values
    // For demonstration, let's fill with 0.05
    for(size_t i = 0; i < all_biases_flat.size(); ++i) all_biases_flat[i] = 0.05f * (i + 1);

    bool weights_set = network.set_all_weights_flat(all_weights_flat);
    bool biases_set = network.set_all_biases_flat(all_biases_flat);

    if (weights_set && biases_set) {
        std::cout << "Weights and biases set successfully using flat vectors." << std::endl;
    } else {
        std::cerr << "Error setting weights or biases using flat vectors." << std::endl;
    }

    // Alternatively, using LayerWeights and LayerBiases structs (per-layer basis)
    // This requires more manual construction of vectors for each layer.
    // NeuroNet::LayerWeights layer0_weights;
    // layer0_weights.WeightCount = inputSize * layer_neuron_counts[0];
    // layer0_weights.WeightsVector.assign(layer0_weights.WeightCount, 0.1f);
    // // ... and so on for other layers and biases ...
    // std::vector<NeuroNet::LayerWeights> all_layer_weights_structs;
    // std::vector<NeuroNet::LayerBiases> all_layer_biases_structs;
    // // ... populate these vectors ...
    // network.set_all_layer_weights(all_layer_weights_structs);
    // network.set_all_layer_biases(all_layer_biases_structs);

    return 0;
}
```

### 3. Processing Data (Forward Pass)

This example demonstrates how to feed input data to the network and get its output.
It assumes `myNetwork` is an already configured `NeuroNet::NeuroNet` object (e.g., from Example 1).

```cpp
#include "neural_network/neuronet.h"
#include "math/matrix.h" // For Matrix::Matrix<float>. Ensure "math/matrix.h" is accessible (e.g., by adding the "src/" directory to your include paths).
#include <iostream>
#include <vector>

int main() {
    // Assume myNetwork is already created and configured (e.g., from Example 1)
    NeuroNet::NeuroNet myNetwork;
    int inputFeatureCount = 5;
    myNetwork.SetInputSize(inputFeatureCount);
    myNetwork.ResizeNeuroNet(2);
    myNetwork.ResizeLayer(0, 10); // Hidden layer
    myNetwork.ResizeLayer(1, 2);  // Output layer
    // For this example, weights and biases would be randomly initialized by default
    // or could be set as in Example 2.

    // Create an input matrix (1 sample, inputFeatureCount features)
    Matrix::Matrix<float> inputData(1, inputFeatureCount);

    // Populate the input matrix with some data
    std::cout << "Input data: ";
    for (int i = 0; i < inputFeatureCount; ++i) {
        inputData[0][i] = static_cast<float>(i + 1) * 0.5f;
        std::cout << inputData[0][i] << " ";
    }
    std::cout << std::endl;

    // Set the input to the network
    if (myNetwork.SetInput(inputData)) {
        // Get the output from the network
        Matrix::Matrix<float> outputData = myNetwork.GetOutput();

        // Access the output data
        // outputData is a matrix, typically 1xN where N is the number of neurons in the output layer.
        std::cout << "Network output: ";
        for (int i = 0; i < outputData.cols(); ++i) {
            std::cout << outputData[0][i] << " ";
        }
        std::cout << std::endl;
    } else {
        std::cerr << "Error: Failed to set input to the network." << std::endl;
        std::cerr << "Check if the network has layers and input size is configured correctly." << std::endl;
    }

    return 0;
}
```

### 4. Training with `GeneticAlgorithm`

This example outlines the steps to train a neural network using the `Optimization::GeneticAlgorithm`.

```cpp
#include "optimization/genetic_algorithm.h" // For Optimization::GeneticAlgorithm
#include "neural_network/neuronet.h"        // For NeuroNet::NeuroNet
#include "math/matrix.h"                    // For Matrix::Matrix (used in fitness function)
#include <iostream>
#include <vector>
#include <numeric> // For std::accumulate in more complex fitness functions
#include <cmath>   // For std::fabs or other math in fitness functions
#include <limits>  // For std::numeric_limits

// --- Main Application ---
int main() {
    // a. Create a Template Network for the GA
    NeuroNet::NeuroNet templateNetwork;
    int inputSize = 3;  // Example: 3 input features
    int outputSize = 1; // Example: 1 output neuron
    templateNetwork.SetInputSize(inputSize);
    templateNetwork.ResizeNeuroNet(2);      // 1 hidden layer, 1 output layer
    templateNetwork.ResizeLayer(0, 5);    // Hidden layer with 5 neurons
    templateNetwork.ResizeLayer(1, outputSize); // Output layer with 1 neuron

    // b. Define a Fitness Function
    // This function evaluates a NeuroNet individual and returns a score (higher is better).
    // The logic is highly problem-specific.
    // This conceptual example tries to make the network output the sum of its inputs.
    auto fitness_function = [&](NeuroNet::NeuroNet& nn) -> double {
        // The 'inputSize' variable (captured by the lambda) defines the expected input dimension
        // for this specific fitness evaluation. The network 'nn' should be configured
        // (e.g., via its template in the GA) to match this inputSize.
        // A public NeuroNet::GetInputSize() method is not available, but the network's
        // internal InputSize member is set via SetInputSize().

        // Create some test input data
        Matrix::Matrix<float> testInput(1, inputSize);
        float expectedOutputSum = 0.0f;
        for (int i = 0; i < inputSize; ++i) {
            testInput[0][i] = static_cast<float>(rand() % 100) / 10.0f; // Random input 0.0 to 9.9
            expectedOutputSum += testInput[0][i];
        }

        nn.SetInput(testInput);
        Matrix::Matrix<float> output = nn.GetOutput();

        if (output.cols() != outputSize) {
            // Output layer not configured as expected or GetOutput() failed
            return 0.0; // Low fitness for misconfigured network
        }

        // Calculate fitness: inverse of absolute error from the expected sum
        // (assuming output[0][0] is the relevant output for this problem)
        double error = std::fabs(output[0][0] - expectedOutputSum);
        
        // Add 1.0 to avoid division by zero and to ensure fitness is positive.
        // Smaller error -> higher fitness.
        return 1.0 / (1.0 + error);
    };

    // c. Instantiate GeneticAlgorithm
    int populationSize = 100;       // Number of NeuroNet individuals in the population
    double mutationRate = 0.05;     // Probability of mutating a weight/bias
    double crossoverRate = 0.7;     // Probability of parents creating offspring
    int numGenerations = 200;       // Number of generations to evolve

    Optimization::GeneticAlgorithm ga(
        populationSize,
        mutationRate,
        crossoverRate,
        numGenerations,
        templateNetwork // The configured template network
    );

    // d. Run Evolution
    std::cout << "Starting genetic algorithm evolution..." << std::endl;
    ga.run_evolution(fitness_function);
    std::cout << "Evolution finished." << std::endl;

    // e. Get Best Individual
    NeuroNet::NeuroNet bestNetwork = ga.get_best_individual();

    // Now, 'bestNetwork' contains the weights and biases of the network
    // that performed best according to your fitness function.
    // You can use it for inference, save its parameters, etc.
    std::cout << "Best network obtained. Evaluating its fitness one last time:" << std::endl;
    double bestFitness = fitness_function(bestNetwork); // Re-evaluate or use stored best fitness
    std::cout << "Fitness of the best individual: " << bestFitness << std::endl;

    // Example: Test the best network with a new input
    Matrix::Matrix<float> finalTestInput(1, inputSize);
    float finalExpectedSum = 0.0f;
    std::cout << "Testing best network with new input: ";
    for (int i = 0; i < inputSize; ++i) {
        finalTestInput[0][i] = static_cast<float>(i + 1) * 1.1f;
        finalExpectedSum += finalTestInput[0][i];
        std::cout << finalTestInput[0][i] << " ";
    }
    std::cout << std::endl;
    std::cout << "Expected sum for this input: " << finalExpectedSum << std::endl;

    bestNetwork.SetInput(finalTestInput);
    Matrix::Matrix<float> finalOutput = bestNetwork.GetOutput();
    std::cout << "Actual output from best network: " << finalOutput[0][0] << std::endl;

    return 0;
}
```

These examples should provide a solid starting point for using the NeuroNet library. Remember to adapt the network architectures, fitness functions, and data handling to your specific application.

## Extension and Integration Guidelines

This section provides guidelines for users who want to extend the NeuroNet library or integrate it into their own C++ projects.

### 1. Extending the Library

Before making significant modifications, it's recommended to thoroughly understand the existing class structures, particularly `NeuroNet::NeuroNet` and `NeuroNet::NeuroNetLayer`.

*   **Activation Functions:** (This part is now updated by the library)
    *   The `NeuroNetLayer::CalculateOutput` method now applies the configured activation function after the linear transformation.
    *   Supported functions are listed in the "Activation Functions" section.
    *   To add new activation functions:
        1.  Add a new value to the `ActivationFunctionType` enum in `src/neural_network/neuronet.h`.
        2.  Implement the corresponding private helper method in `NeuroNetLayer` (e.g., `ApplyMyNewFunction`).
        3.  Add a case for your new function type in `NeuroNetLayer::CalculateOutput`.
        4.  Update documentation and serialization logic if necessary.

*   **New Layer Types:**
    *   The current `NeuroNetLayer` provides a standard fully connected layer. To introduce specialized layer behaviors (e.g., convolutional layers, recurrent layers):
        *   You could define new classes that are distinct from `NeuroNetLayer` and manage them within the `NeuroNet` class. This would likely require modifications to `NeuroNet` to handle different layer types.
        *   Alternatively, if a common interface for layers can be established (e.g., an abstract `BaseLayer` class), `NeuroNetLayer` and your new layer types could inherit from it. `NeuroNet` would then manage a collection of `BaseLayer` pointers or references. This is a more significant architectural change.

*   **Custom Genetic Algorithm Components:**
    *   The `Optimization::GeneticAlgorithm` class provides a standard implementation. If you require different selection strategies (e.g., roulette wheel instead of tournament), crossover methods (e.g., multi-point crossover), or mutation operators, you might consider:
        *   Modifying the existing `GeneticAlgorithm` class methods (`selection`, `crossover`, `mutate`).
        *   Creating a new class, perhaps inheriting from `GeneticAlgorithm` if you want to reuse some of its functionality (though C++ inheritance might not be the most straightforward approach here without a base class designed for extension). More likely, you would write a new GA class inspired by the current one.

*   **General Advice:**
    *   Start by understanding the data flow: how `NeuroNet` manages `NeuroNetLayer` instances, how weights and biases are stored and updated, and how the `GeneticAlgorithm` interacts with `NeuroNet` individuals.
    *   Refer to the existing code for patterns on matrix operations, parameter handling, and class interactions.

### 2. Integrating into Other C++ Projects

*   **Linking:**
    *   The NeuroNet library is built as a **static library** by default (as per `CMakeLists.txt`, which specifies `add_library(neuronet STATIC ...)`).
    *   When you build this project, it will produce a static library file (e.g., `libneuronet.a` on Linux/macOS, `neuronet.lib` on Windows).
    *   Your C++ project will need to link against this compiled static library file. How you do this depends on your build system (e.g., for g++, you'd use the `-L` option for the directory and `-lneuronet` to link).

*   **Header Files:**
    *   Your project must include the relevant header files to use the library's classes and functions. Primarily, these are:
        *   `neural_network/neuronet.h` (for `NeuroNet::NeuroNet`, `NeuroNet::NeuroNetLayer`, etc.)
        *   `optimization/genetic_algorithm.h` (for `Optimization::GeneticAlgorithm`)
        *   `math/matrix.h` (as this is used in the interfaces, e.g. `NeuroNet::SetInput()`)
        *   `utilities/json/json.hpp` (if you interact with serialization or need JSON utilities, though it's included by `neuronet.h`)

*   **Include Paths:**
    *   You need to configure your compiler's include paths so it can find the library's header files. This typically involves adding the `src` directory (or a custom install directory if you set one up) of the NeuroNet library to your project's include directories.
    *   For example, if you clone this repository into `libs/neuronet-ga`, you might add `libs/neuronet-ga/src` to your include paths.

*   **Runtime (for DLL - Not applicable by default):**
    *   Since the library is built as a static library by default, there's no separate DLL file to distribute or manage at runtime in the same way as a dynamic library. The library's code is linked directly into your executable. If you were to change it to a DLL, then the DLL would need to be accessible at runtime.

*   **CMake Integration (Recommended):**
    *   If your project uses CMake, you can integrate the NeuroNet library more easily. After building the NeuroNet library (or if you have it as a pre-built package):
        1.  Ensure CMake can find the NeuroNet library. This might involve setting `CMAKE_PREFIX_PATH` if installed to a non-standard location, or using `find_package` if NeuroNet provided a CMake config file (it currently doesn't, so manual path setup is more likely).
        2.  Use `find_library` to locate the compiled `neuronet` static library if needed, or directly specify the path to the `.a`/`.lib` file.
        3.  Use `target_link_libraries` in your `CMakeLists.txt` to link your executable or library against `neuronet`.
            ```cmake
            # Example (assuming NeuroNet library is found or path is known)
            # target_link_libraries(YourTarget PRIVATE path/to/libneuronet.a)
            ```
        4.  Use `target_include_directories` to add the path to the NeuroNet header files (e.g., the `src` directory).
            ```cmake
            # Example
            # target_include_directories(YourTarget PRIVATE path/to/neuronet-ga/src)
            ```
    *   If you are building NeuroNet as part of a larger CMake project (e.g., using `add_subdirectory`), you can directly link against the `neuronet` target:
        ```cmake
        # In your main project's CMakeLists.txt, after add_subdirectory(path/to/neuronet-ga)
        # target_link_libraries(YourExecutableTarget PRIVATE neuronet)
        ```
        This automatically handles linking and include directories if NeuroNet's `CMakeLists.txt` correctly defines its `INTERFACE_INCLUDE_DIRECTORIES`.

### 3. Dependencies

*   **Matrix Library:**
        *   The NeuroNet library relies on an internal header-only `Matrix` class, located in `src/math/matrix.h`.
        *   This `matrix.h` file provides the `Matrix::Matrix<T>` template class used for all underlying mathematical operations and is included as part of the NeuroNet library's source.
*   **Custom JSON Library:**
        *   The library utilizes an internal custom JSON parsing and manipulation library located in `src/utilities/json/` (specifically `json.hpp` for the interface and `json.cpp` for the implementation).
        *   This library is used for model serialization/deserialization and other JSON tasks. It is compiled directly into the NeuroNet library.
*   **Setup:**
    *   **If building NeuroNet library yourself:** `matrix.h` (header-only) and the custom JSON library (`json.hpp`, `json.cpp`) are part of this repository. `matrix.h` is included directly, and the custom JSON implementation (`json.cpp`) is compiled into the NeuroNet library.
    *   **If integrating NeuroNet into your project:** You need to ensure that your compiler can find the NeuroNet header files (which include `src/math/matrix.h` and `src/utilities/json/json.hpp`). This means the `src` directory from the NeuroNet library (or a relevant install path for headers) should be in your include paths. The compiled code for the JSON library and the definitions for the matrix library will be part of the NeuroNet static library you link against.

## Project Documentation

This `README.md` provides a general overview of the project. For more detailed information, please refer to the following:

*   **Module-specific Documentation:**
    *   [Neural Network (`NeuroNet` & `NeuroNetLayer`)](./docs/modules/neural_network.md)
    *   [Genetic Algorithm (`GeneticAlgorithm`)](./docs/modules/optimization.md)
    *   [Transformer Architecture](./docs/modules/transformer.md)
    *   [Matrix Library](./docs/modules/math.md)
    *   [JSON Utilities](./docs/modules/json_utilities.md)
*   **API Reference (Doxygen):**
    *   Due to potential environment differences, Doxygen HTML generation might not work reliably in all cloud-based IDEs.
    *   The `Doxyfile` in the root of the repository is configured to generate API documentation from the source code comments.
    *   To generate the API documentation locally:
        1.  Ensure you have Doxygen installed (see [doxygen.nl](https://www.doxygen.nl/download.html)).
        2.  Navigate to the root directory of this repository in your terminal.
        3.  Run the command: `doxygen Doxyfile`
        4.  The HTML documentation will be generated in the `docs/api/html` directory. Open `docs/api/html/index.html` in your browser to view it.
    *   The generated API documentation (`docs/api/`) should not be committed to the repository (it's included in `.gitignore`).
*   **Contributing Guidelines:** See [CONTRIBUTING.md](CONTRIBUTING.md) for details on how to contribute to this project.
*   **Changelog:** See [CHANGELOG.md](CHANGELOG.md) for the history of changes.

## Features

*   **Core Neural Network (`NeuroNet`, `NeuroNetLayer`):**
    *   Customizable feed-forward neural network architecture.
    *   Layer-wise configuration of neuron counts and activation functions.
    *   Support for various activation functions: None (Linear), ReLU, LeakyReLU, ELU, Softmax.
    *   Serialization/deserialization of network architecture and parameters to/from JSON.
    *   Vocabulary loading and string input processing for NLP tasks.
*   **Optimization (`GeneticAlgorithm`):**
    *   Genetic Algorithm for evolving `NeuroNet` weights and biases.
    *   Configurable population size, mutation rate, crossover rate, and generations.
    *   Tournament selection and elitism.
    *   Exportable training metrics (fitness per generation, best model architecture).
*   **Transformer Module (Encoder-Only):**
    *   Implementation of core Transformer components:
        *   `EmbeddingLayer` for token embeddings.
        *   `PositionalEncoding` for sequence position information.
        *   `ScaledDotProductAttention` and `MultiHeadAttention`.
        *   `TransformerFFN` (Position-wise Feed-Forward Network).
        *   `TransformerEncoderLayer` stacking attention and FFN.
    *   `TransformerModel` class to build encoder-only transformer architectures.
    *   Serialization/deserialization for `TransformerModel`.
*   **Math Utilities:**
    *   Generic `Matrix<T>` class for 2D matrix operations.
        *   Standard arithmetic (+, -, *), scalar operations.
        *   Transpose, determinant, inverse.
        *   OpenMP parallelization for matrix multiplication.
    *   Extended math functions: GELU, Layer Normalization, Softmax.
    *   Matrix splitting and combining utilities.
*   **JSON Utilities:**
    *   Custom lightweight JSON parser (`JsonParser`) and data representation (`JsonValue`).
    *   Supports parsing JSON strings into a tree of `JsonValue` objects.
    *   Serialization of `JsonValue` trees back to JSON strings.
    *   Handles basic JSON types: null, boolean, number, string, array, object.
    *   Includes support for skipping comments in JSON files.
*   **Build and Testing:**
    *   Built with CMake for cross-platform compilation.
    *   Unit tests using Google Test framework (`ctest` runnable).
*   **Benchmarking:**
    *   Optional timing instrumentation for core operations (e.g., matrix multiplication, GA steps) via `ENABLE_BENCHMARKING` macro.
    *   Example benchmark tests in `tests/test_benchmarks.cpp`.

## Prerequisites

*   C++ Compiler (supporting C++17 or later)
*   CMake (version 3.10 or later)

## Building the Project

1.  **Clone the repository:**
    ```bash
    git clone <repository_url>
    cd <repository_name>
    ```

2.  **Create a build directory:**
    ```bash
    mkdir build
    cd build
    ```

3.  **Configure with CMake:**
    ```bash
    cmake ..
    ```

4.  **Build the project (library and tests):**
    ```bash
    make 
    # Or, on Windows with MSVC
    # cmake --build .
    ```
    **Note on OpenMP:** To enable OpenMP for parallelized matrix multiplication, ensure your compiler supports OpenMP and set the appropriate compiler flags (e.g., `/openmp` for MSVC, `-fopenmp` for GCC/Clang). This might be done by modifying the `CMAKE_CXX_FLAGS` in CMake or setting environment variables before configuring.

This will build the `neuronet` static library and the test executables.

## Running Tests

The project includes a suite of unit tests to ensure functionality and correctness. Tests are located in the `tests/` directory, primarily in `tests/test_neuronet.cpp`.
The tests are built using Google Test.

**General Steps to Build and Run Tests:**

1.  Ensure your development environment is set up with a C++ compiler and CMake.
2.  Navigate to your build directory (or create one if it doesn't exist):
    ```bash
    # From the project root
    mkdir -p build
    cd build
    ```
3.  Configure the project with CMake:
    ```bash
    cmake ..
    ```
4.  Build the project, including the test executables:
    ```bash
    cmake --build . 
    # Alternatively, use 'make' if that's your build tool
    # make
    ```
5.  Run the tests:
    *   The primary test executable defined in `tests/CMakeLists.txt` is `test_neuronet`. If compiled successfully, it would be located in the `tests` subdirectory of your build folder.
    *   `ctest` can be used if the tests are correctly registered with CTest in CMake (which they are):
        ```bash
        # From the build directory
        ctest
        ```
        Alternatively, you can run the test executable directly (e.g., `tests/runTests` on Linux/macOS or `tests\Debug\runTests.exe` on Windows, path may vary based on build type).
The previous mention of Catch2 has been removed as the tests primarily use Google Test.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
