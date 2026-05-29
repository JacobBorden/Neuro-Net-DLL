# Optimization Module (`GeneticAlgorithm`)

This document describes the `Optimization::GeneticAlgorithm` class, which is used for training or evolving the parameters (weights and biases) of `NeuroNet` instances using evolutionary strategies.

## Overview

The `GeneticAlgorithm` class implements a standard evolutionary algorithm to search for optimal neural network configurations. It manages a population of `NeuroNet` individuals, iteratively applying genetic operators like selection, crossover, and mutation to improve fitness over generations.

## Key Concepts

The `GeneticAlgorithm` relies on several core evolutionary concepts:

*   **Population:** A collection of `NeuroNet` individuals (`std::vector<NeuroNet::NeuroNet>`). Each individual represents a potential solution (a specific set of weights and biases for a neural network of a fixed architecture).
*   **Fitness Function:** A user-provided function (`std::function<double(NeuroNet::NeuroNet&)>`) that evaluates a given `NeuroNet` individual and returns a numerical score. Higher scores typically indicate better performance on the target task. This function is crucial as it guides the evolution.
*   **Selection:** The process of choosing individuals ("parents") from the current population to produce the next generation. This implementation uses tournament selection, where a small subset of individuals is chosen randomly, and the best among them is selected. It also incorporates elitism, ensuring the best individual from one generation is carried over to the next.
*   **Crossover:** The process of combining genetic material from two selected parents to create "offspring." This involves exchanging parts of their flattened weight and bias vectors. The `crossover_rate_` parameter controls the probability of crossover occurring.
*   **Mutation:** The process of introducing small, random changes to an individual's genetic material (weights and biases). Each weight and bias has a chance (`mutation_rate_`) of being altered. Mutation helps introduce new genetic variations and explore the solution space.

## `Optimization::GeneticAlgorithm` Class

### Purpose

To provide a ready-to-use genetic algorithm for optimizing the parameters of `NeuroNet` objects.

### Core Responsibilities

*   Initializing and managing a population of `NeuroNet` individuals.
*   Executing the evolutionary loop: evaluation, selection, crossover, mutation.
*   Tracking the best individual found during the evolution.
*   Exporting training metrics.

### Key Functionalities (refer to Doxygen API docs for full details)

*   **Constructor `GeneticAlgorithm(int population_size, double mutation_rate, double crossover_rate, int num_generations, const NeuroNet::NeuroNet& template_network)`:**
    *   Initializes the GA with essential parameters.
    *   `template_network`: A crucial `NeuroNet` object that defines the fixed architecture (input size, layer count, layer sizes) for all individuals in the population.
*   **`initialize_population()`:** Creates the initial population. Each `NeuroNet` individual is a copy of the `template_network`'s architecture but with randomly initialized weights and biases.
*   **`evaluate_fitness(const std::function<double(NeuroNet::NeuroNet&)>& fitness_function)`:** Calculates and stores the fitness score for each individual in the population using the provided fitness function.
*   **`selection()`:** Selects individuals for the next generation using tournament selection and elitism.
*   **`crossover(const NeuroNet::NeuroNet& parent1, const NeuroNet::NeuroNet& parent2)`:** Combines two parent networks to produce two offspring by swapping segments of their flattened weights and biases.
*   **`mutate(NeuroNet::NeuroNet& individual)`:** Applies random changes to the weights and biases of an individual network.
*   **`evolve_one_generation(const std::function<double(NeuroNet::NeuroNet&)>& fitness_function, int current_generation_number)`:** Runs a single cycle of evaluation, selection, crossover, and mutation.
*   **`run_evolution(const std::function<double(NeuroNet::NeuroNet&)>& fitness_function)`:** Executes the main evolutionary loop for the configured number of generations.
*   **`get_best_individual() const`:** Returns the `NeuroNet` individual with the highest fitness score achieved during the evolution.
*   **`export_training_metrics_json(const std::string& filename) const`:** Saves training metrics (e.g., fitness per generation) to a JSON file.

### Source

*   `src/optimization/genetic_algorithm.h`
*   `src/optimization/genetic_algorithm.cpp`

## Basic Usage Example: Training a Network

The main `README.md` provides a comprehensive example of training a `NeuroNet` using `GeneticAlgorithm`. Key steps involve:

1.  **Creating a Template Network:** Define the architecture of the `NeuroNet` instances you want to train.
    ```cpp
    NeuroNet::NeuroNet templateNetwork;
    // ... configure templateNetwork (SetInputSize, ResizeNeuroNet, ResizeLayer) ...
    ```

2.  **Defining a Fitness Function:** This function is problem-specific and crucial for success.
    ```cpp
    auto fitness_function = [&](NeuroNet::NeuroNet& nn) -> double {
        // ... evaluate nn, return a score (higher is better) ...
        // Example: calculate error against target data, fitness = 1 / (1 + error)
        Matrix::Matrix<float> input = /* your input data */;
        Matrix::Matrix<float> expected_output = /* your expected output */;
        nn.SetInput(input);
        Matrix::Matrix<float> actual_output = nn.GetOutput();
        double error = 0.0;
        // ... calculate error between actual_output and expected_output ...
        return 1.0 / (1.0 + error); // Ensure fitness is positive and higher is better
    };
    ```

3.  **Instantiating `GeneticAlgorithm`:**
    ```cpp
    int populationSize = 100;
    double mutationRate = 0.05;
    double crossoverRate = 0.7;
    int numGenerations = 200;

    Optimization::GeneticAlgorithm ga(
        populationSize,
        mutationRate,
        crossoverRate,
        numGenerations,
        templateNetwork
    );
    ```

4.  **Running Evolution:**
    ```cpp
    ga.run_evolution(fitness_function);
    ```

5.  **Getting the Best Individual:**
    ```cpp
    NeuroNet::NeuroNet bestNetwork = ga.get_best_individual();
    // ... use bestNetwork for inference or save it ...
    bestNetwork.save_model("best_model.json");
    ```

6.  **Exporting Metrics (Optional):**
    ```cpp
    ga.export_training_metrics_json("training_run_metrics.json");
    ```

Refer to the "Training with `GeneticAlgorithm`" example in the main `README.md` for a more complete, runnable code snippet.

(This is a starting point. More details on specific parameters or advanced usage can be added.)
