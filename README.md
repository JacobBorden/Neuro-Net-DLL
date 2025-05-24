# NeuroNet with Genetic Algorithm

This project provides a C++ library for creating and training neural networks using a genetic algorithm. It includes a matrix library for underlying mathematical operations.

## Features

*   Customizable Neural Network architecture (`NeuroNet`, `NeuroNetLayer`).
*   Genetic Algorithm (`GeneticAlgorithm`) for evolving network weights and biases.
*   Matrix library (`Matrix`) for numerical computations.
*   Built with CMake.
*   Unit tests using Google Test.

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

This will build the `neuronet` static library and the test executables.

## Running Tests

After building the project, you can run the tests from the `build` directory:

```bash
ctest

# Or, if 'make test' is available
# make test
```
Google Test is used as the testing framework and is fetched automatically by CMake during the configuration step.

Using the NeuroNet Library
Here's a minimal example of how to use the NeuroNet library:

```cpp
#include "neural_network/neuronet.h" // Assumes include paths are set up
#include "math/matrix.h"             // For Matrix::Matrix
#include <iostream>
#include <vector>

int main() {
    // 1. Create a NeuroNet instance
    NeuroNet::NeuroNet myNetwork;

    // 2. Configure the network architecture
    int inputSize = 10;
    myNetwork.SetInputSize(inputSize); // Set the size of the input vector

    // Add layers: ResizeNeuroNet(numberOfLayers)
    // Then ResizeLayer(layerIndex, numberOfNeuronsInLayer)
    myNetwork.ResizeNeuroNet(2); // Create a network with 2 layers
    myNetwork.ResizeLayer(0, 15); // Layer 0 has 15 neurons
    myNetwork.ResizeLayer(1, 5);  // Layer 1 has 5 neurons (output layer)

    // (Optional) Initialize weights and biases if not using GA
    // You can use set_all_weights_flat() and set_all_biases_flat()
    // For example, to randomize:
    // NeuroNet::NeuroNetLayer tempLayer; // Use to get counts
    // if (myNetwork.GetLayerCount() > 0) { // Make sure network is configured
    //     std::vector<float> random_weights;
    //     std::vector<float> random_biases;
    //     // This is a simplified way to get total counts. 
    //     // In a real scenario, sum counts from each layer.
    //     // The GA handles this initialization internally for its population.
    // }


    // 3. Create an input matrix
    // Matrix is in the Matrix namespace, NeuroNet in NeuroNet namespace
    Matrix::Matrix<float> inputMatrix(1, inputSize); 
    for (int i = 0; i < inputSize; ++i) {
        inputMatrix[0][i] = static_cast<float>(i) * 0.1f; // Example input
    }

    // 4. Set input to the network
    if (myNetwork.SetInput(inputMatrix)) {
        // 5. Get the output
        Matrix::Matrix<float> outputMatrix = myNetwork.GetOutput();

        // Print the output
        std::cout << "Network Output: " << std::endl;
        for (int i = 0; i < outputMatrix.cols(); ++i) {
            std::cout << outputMatrix[0][i] << " ";
        }
        std::cout << std::endl;
    } else {
        std::cerr << "Error setting input to the network." << std::endl;
    }

    return 0;
}
Using the GeneticAlgorithm
Here's a minimal example of how to use the GeneticAlgorithm:

#include "optimization/genetic_algorithm.h" // For Optimization::GeneticAlgorithm
#include "neural_network/neuronet.h"        // For NeuroNet::NeuroNet
#include "math/matrix.h"                    // For Matrix::Matrix
#include <iostream>
#include <vector>
#include <numeric> // For std::accumulate
#include <cmath>   // For std::fabs

// 1. Define a fitness function
//    This function evaluates a NeuroNet (from NeuroNet namespace) and returns a fitness score (higher is better).
double simpleFitnessFunction(NeuroNet::NeuroNet& network) {
    // Example: Try to get the network to output a sum of its inputs.
    // This is a toy problem.
    int inputSize = network.GetInputSize();
    if (inputSize == 0) return 0.0;

    Matrix::Matrix<float> testInput(1, inputSize); // Matrix from Matrix namespace
    float expectedSum = 0.0f;
    for (int i = 0; i < inputSize; ++i) {
        testInput[0][i] = static_cast<float>(i + 1); // e.g., 1, 2, 3...
        expectedSum += testInput[0][i];
    }

    network.SetInput(testInput);
    Matrix::Matrix<float> output = network.GetOutput(); // Matrix from Matrix namespace

    if (output.cols() == 0) return 0.0; // No output produced

    // Let's assume the network is expected to produce the sum in its first output neuron
    float actualOutput = output[0][0]; 
    
    // Fitness: inverse of the absolute difference. Maximize by minimizing difference.
    double error = std::fabs(actualOutput - expectedSum);
    return 1.0 / (1.0 + error); // Add 1 to avoid division by zero and normalize
}

int main() {
    // 2. Create a template NeuroNet for the GA population
    NeuroNet::NeuroNet templateNetwork; // NeuroNet from NeuroNet namespace
    int inputSize = 5;
    templateNetwork.SetInputSize(inputSize);
    templateNetwork.ResizeNeuroNet(2);    // 2 layers
    templateNetwork.ResizeLayer(0, 8);  // Layer 0 with 8 neurons
    templateNetwork.ResizeLayer(1, 1);  // Layer 1 with 1 neuron (output layer for our fitness function)

    // 3. Instantiate the GeneticAlgorithm
    int populationSize = 50;
    double mutationRate = 0.1;
    double crossoverRate = 0.7;
    int numGenerations = 100;

    // GeneticAlgorithm is in the Optimization namespace
    Optimization::GeneticAlgorithm ga(
        populationSize, 
        mutationRate, 
        crossoverRate, 
        numGenerations, 
        templateNetwork // Pass the template network
    );

    // 4. Run the evolution process
    std::cout << "Starting evolution..." << std::endl;
    ga.run_evolution(simpleFitnessFunction);
    std::cout << "Evolution finished." << std::endl;

    // 5. Get the best individual
    NeuroNet::NeuroNet bestNetwork = ga.get_best_individual(); // NeuroNet from NeuroNet namespace

    // You can now use the bestNetwork, e.g., test its output or save its weights
    std::cout << "Best individual's fitness can be re-evaluated (or stored during GA run): " 
              << simpleFitnessFunction(bestNetwork) << std::endl;

    // Example: Print weights of the first layer of the best network
    if (bestNetwork.GetLayerCount() > 0) {
        // LayerWeights is part of NeuroNet namespace (defined in neuronet.h)
        NeuroNet::LayerWeights weights = bestNetwork.get_all_layer_weights()[0]; 
        std::cout << "Best network, Layer 0, first few weights: ";
        for(int i=0; i < 5 && i < weights.WeightsVector.size(); ++i) {
            std::cout << weights.WeightsVector[i] << " ";
        }
        std::cout << std::endl;
    }

    return 0;
}
```
Key Components
NeuroNet (src/neural_network/neuronet.h, src/neural_network/neuronet.cpp): The core class representing a neural network, part of the NeuralNetwork module. Manages layers and network-level operations.
NeuroNetLayer (src/neural_network/neuronet.h, src/neural_network/neuronet.cpp): Represents a single layer within a NeuroNet, part of the NeuralNetwork module. Handles calculations for that layer.
GeneticAlgorithm (src/optimization/genetic_algorithm.h, src/optimization/genetic_algorithm.cpp): Implements the genetic algorithm to train/evolve NeuroNet instances. Part of the Optimization module (namespace Optimization).
Matrix (src/math/matrix.h): A header-only template library for matrix operations, part of the Math module (namespace Matrix). Used by NeuroNetLayer for calculations.
Dependencies
Google Test: Used for unit testing. It is fetched automatically by CMake via FetchContent during the build configuration. No manual installation is required.
