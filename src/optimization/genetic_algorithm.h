/**
 * @file genetic_algorithm.h
 * @author Your Name (youremail@example.com)
 * @brief Defines the GeneticAlgorithm class for evolving NeuroNet individuals.
 * @version 0.1.0
 * @date 2023-10-27
 *
 * @copyright Copyright (c) 2023
 */

#ifndef GENETIC_ALGORITHM_H
#define GENETIC_ALGORITHM_H

#include <vector>
#include <functional> // For std::function
#include <algorithm>  // For std::sort, std::transform, std::max_element
#include <random>     // For std::mt19937, std::uniform_real_distribution, etc.
#include <limits>     // For std::numeric_limits
#include "../neural_network/neuronet.h" // NeuroNet class header
#include "optimization/training_metrics.h" // For training metrics

namespace Optimization {

/**
 * @brief Implements a genetic algorithm to evolve NeuroNet individuals.
 *
 * This class manages a population of NeuroNet objects, applying genetic operators
 * such as selection, crossover, and mutation over a number of generations
 * to find an individual that performs well according to a user-defined fitness function.
 */
class GeneticAlgorithm {
public:
    /**
     * @brief Constructs a GeneticAlgorithm instance.
     * @param population_size The number of individuals in the population.
     * @param mutation_rate The probability of mutating a gene (e.g., a weight or bias).
     * @param crossover_rate The probability of performing crossover between two parents.
     * @param num_generations The total number of generations to run the evolution.
     * @param template_network A NeuroNet object configured with the desired layer structure
     *                         (input size, layer sizes). This template is used to create
     *                         new individuals in the population.
     */
    GeneticAlgorithm(
        int population_size,
        double mutation_rate,
        double crossover_rate,
        int num_generations,
        const NeuroNet::NeuroNet& template_network
    );

    /**
     * @brief Initializes the population with random NeuroNet individuals.
     * Each individual is created based on the structure of the template_network
     * provided in the constructor, but with randomized weights and biases.
     */
    void initialize_population();

    /**
     * @brief Evaluates the fitness of each individual in the current population.
     * @param fitness_function A function that takes a NeuroNet individual by reference
     *                         and returns its fitness score (double). Higher scores are better.
     */
    void evaluate_fitness(const std::function<double(NeuroNet::NeuroNet&)>& fitness_function);

    /**
     * @brief Performs selection to choose individuals for the next generation.
     * This implementation uses tournament selection and elitism (best individual carries over).
     * It modifies the internal population and fitness_scores_ vectors.
     * Crossover and mutation are typically applied to the selected individuals afterwards.
     */
    void selection();

    /**
     * @brief Performs crossover between two parent NeuroNet individuals.
     * Creates two offspring by exchanging genetic material (weights and biases).
     * This implementation uses single-point crossover on the flattened weight and bias vectors.
     * @param parent1 The first parent NeuroNet.
     * @param parent2 The second parent NeuroNet.
     * @return std::vector<NeuroNet> A vector containing two offspring NeuroNet individuals.
     */
    std::vector<NeuroNet::NeuroNet> crossover(const NeuroNet::NeuroNet& parent1, const NeuroNet::NeuroNet& parent2);

    /**
     * @brief Applies mutation to an individual NeuroNet.
     * Each weight and bias in the individual has a chance (defined by mutation_rate_)
     * to be altered by a small random amount.
     * @param individual The NeuroNet individual to mutate (modified in place).
     */
    void mutate(NeuroNet::NeuroNet& individual);

    /**
     * @brief Evolves the population for a single generation.
     * This involves evaluating fitness, performing selection, crossover, and mutation.
     * @param fitness_function The fitness function to evaluate individuals.
     * @param current_generation_number The current generation number for metrics.
     */
    void evolve_one_generation(const std::function<double(NeuroNet::NeuroNet&)>& fitness_function, int current_generation_number);

    /**
     * @brief Runs the complete evolution process for the specified number of generations.
     * Initializes the population and then iteratively calls evolve_one_generation.
     * @param fitness_function The fitness function to evaluate individuals.
     */
    void run_evolution(const std::function<double(NeuroNet::NeuroNet&)>& fitness_function);

    /**
     * @brief Retrieves the best NeuroNet individual found during the evolution process.
     * The "best" individual is the one with the highest fitness score encountered.
     * @return NeuroNet The best performing NeuroNet individual.
     *                Returns a default-constructed NeuroNet if evolution hasn't run or population is empty.
     */
    NeuroNet::NeuroNet get_best_individual() const;

    /**
     * @brief Exports the collected training metrics to a JSON file.
     * @param filename The path to the file where metrics should be saved.
     */
    void export_training_metrics_json(const std::string& filename) const;

private:
    int population_size_;       ///< Number of individuals in the population.
    double mutation_rate_;      ///< Probability of mutation for each gene.
    double crossover_rate_;     ///< Probability of performing crossover.
    int num_generations_;       ///< Total number of generations for evolution.
    NeuroNet::NeuroNet template_network_; ///< Template NeuroNet defining the structure of individuals.

    std::vector<NeuroNet::NeuroNet> population_;     ///< Current population of NeuroNet individuals.
    std::vector<double> fitness_scores_;   ///< Fitness scores corresponding to the population_.
    NeuroNet::NeuroNet best_individual_;             ///< The best individual found so far across all generations.
    double best_fitness_score_;            ///< The fitness score of the best_individual_.
    int current_generation_;               ///< Current generation number, used for metrics.

    TrainingRunMetrics current_run_metrics_; ///< Metrics collected during the current training run.

    mutable std::mt19937 random_engine_; ///< Mersenne Twister random number engine for GA operations.

    /**
     * @brief Creates a random NeuroNet individual based on the template_network_.
     * Weights and biases are initialized to small random values.
     * @return NeuroNet A new NeuroNet individual with randomized parameters.
     */
    NeuroNet::NeuroNet create_random_individual() const;

    /**
     * @brief Selects an individual from the population using tournament selection.
     * @param tournament_size The number of individuals to randomly pick for the tournament.
     * @return const NeuroNet& A reference to the winning individual from the tournament.
     *                         Returns a reference to a fallback network if population is empty.
     */
    const NeuroNet::NeuroNet& tournament_selection(int tournament_size = 5) const;
};

} // namespace Optimization

#endif // GENETIC_ALGORITHM_H
