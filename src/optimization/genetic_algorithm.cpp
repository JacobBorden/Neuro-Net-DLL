/**
 * @file genetic_algorithm.cpp
 * @author Your Name (youremail@example.com)
 * @brief Implements the GeneticAlgorithm class for evolving NeuroNet individuals.
 * @version 0.1.0
 * @date 2023-10-27
 *
 * @copyright Copyright (c) 2023
 */

#include "genetic_algorithm.h"
#include <iostream>  // For potential debugging output (e.g., during run_evolution)
#include <algorithm> // For std::shuffle, std::transform, std::min_element, std::max_element, std::sort
#include <limits>    // For std::numeric_limits
#include <stdexcept> // For std::runtime_error (optional, for error handling)
#include "../utilities/timer.h" // For Timer class

// Define ENABLE_BENCHMARKING to enable timing of genetic algorithm operations.
// This can be defined in project settings or uncommented here for testing.
// #define ENABLE_BENCHMARKING

namespace Optimization {

/**
 * @brief Constructs a GeneticAlgorithm instance.
 *
 * Initializes GA parameters and seeds the random number generator.
 * The template_network is copied and used as a blueprint for all individuals
 * in the population, ensuring consistent network architecture.
 */
Optimization::GeneticAlgorithm::GeneticAlgorithm(
    int population_size,
    double mutation_rate,
    double crossover_rate,
    int num_generations,
    const NeuroNet::NeuroNet& template_network)
    : population_size_(population_size),
      mutation_rate_(mutation_rate),
      crossover_rate_(crossover_rate),
      num_generations_(num_generations),
      template_network_(template_network), // Make a copy of the template network
      best_individual_(template_network), // Initialize best_individual_ with template structure
      best_fitness_score_(std::numeric_limits<double>::lowest()) { // Initialize best score to a very low value
    
    // Seed the random number generator for reproducibility during a run,
    // but different seeds across different program executions.
    std::random_device rd;
    random_engine_.seed(rd());
    
    // It's crucial that the template_network is properly configured (input size, layer sizes set)
    // before being passed to this constructor, as create_random_individual relies on this structure.
}

/**
 * @brief Creates a new NeuroNet individual with randomized weights and biases.
 *
 * The structure (number of layers, neurons per layer, input size) is copied
 * from the template_network_. Weights and biases are initialized to small random
 * values, typically between -1.0 and 1.0.
 * @return NeuroNet A new, randomly initialized NeuroNet individual.
 */
NeuroNet::NeuroNet Optimization::GeneticAlgorithm::create_random_individual() const {
    NeuroNet::NeuroNet individual = template_network_; // Start with the template structure.

    // Retrieve the total number of weights and biases from the template structure.
    // This uses the NeuroNet's flat accessor methods, which is convenient.
    std::vector<float> weights = individual.get_all_weights_flat();
    std::vector<float> biases = individual.get_all_biases_flat();

    // Define a distribution for random weights/biases (e.g., between -1.0 and 1.0).
    std::uniform_real_distribution<float> dist_params(-1.0f, 1.0f);

    // Randomize weights.
    for (float& weight : weights) {
        weight = dist_params(random_engine_);
    }
    individual.set_all_weights_flat(weights); // Apply the new random weights.

    // Randomize biases.
    for (float& bias : biases) {
        bias = dist_params(random_engine_);
    }
    individual.set_all_biases_flat(biases); // Apply the new random biases.

    return individual;
}

/**
 * @brief Clears and re-initializes the population with new random individuals.
 *
 * The population vector is cleared and then filled with `population_size_`
 * new individuals, each created by `create_random_individual()`.
 * Fitness scores are also cleared, as they will need to be re-evaluated.
 */
void Optimization::GeneticAlgorithm::initialize_population() {
    population_.clear();
    fitness_scores_.clear(); // Fitness scores are now invalid.
    population_.reserve(population_size_); // Pre-allocate memory.
    fitness_scores_.reserve(population_size_);

    for (int i = 0; i < population_size_; ++i) {
        population_.push_back(create_random_individual());
    }
    // Fitness scores will be calculated when evaluate_fitness is called.
    // Reset overall best fitness score for a new evolution run.
    best_fitness_score_ = std::numeric_limits<double>::lowest();
    // best_individual_ can be left as is, or reset to a default NeuroNet,
    // as it will be updated when a better individual is found.
}

/**
 * @brief Calculates and stores the fitness score for each individual in the population.
 *
 * Iterates through the current population, applying the provided `fitness_function`
 * to each NeuroNet individual. The scores are stored in `fitness_scores_`.
 * This function also identifies the best fitness score *within the current generation*
 * but does not update the overall `best_individual_` or `best_fitness_score_` (across all generations);
 * that update is handled in `evolve_one_generation` after selection and other operations.
 */
void Optimization::GeneticAlgorithm::evaluate_fitness(const std::function<double(NeuroNet::NeuroNet&)>& fitness_function) {
#ifdef ENABLE_BENCHMARKING
    utilities::Timer eval_timer;
    eval_timer.start();
#endif
    if (population_.empty()) {
        // Or throw std::runtime_error("Population is empty, cannot evaluate fitness.");
#ifdef ENABLE_BENCHMARKING
        eval_timer.stop();
        std::cout << "GeneticAlgorithm::evaluate_fitness() (Population Empty) took: " << eval_timer.elapsed_milliseconds() << " ms" << std::endl;
#endif
        return;
    }
    fitness_scores_.resize(population_.size());

    double current_generation_best_fitness = std::numeric_limits<double>::lowest();
    // int current_generation_best_idx = -1; // If needed for specific tracking

    for (size_t i = 0; i < population_.size(); ++i) {
        fitness_scores_[i] = fitness_function(population_[i]);
        if (fitness_scores_[i] > current_generation_best_fitness) {
            current_generation_best_fitness = fitness_scores_[i];
            // current_generation_best_idx = i;
        }
    }
    // Note: The overall best_individual_ and best_fitness_score_ (across all generations)
    // are updated in evolve_one_generation after new individuals might be created.
#ifdef ENABLE_BENCHMARKING
    eval_timer.stop();
    std::cout << "GeneticAlgorithm::evaluate_fitness() (Population Size: " << population_.size() 
              << ") took: " << eval_timer.elapsed_milliseconds() << " ms" << std::endl;
#endif
}

/**
 * @brief Selects an individual from the population using tournament selection.
 *
 * Randomly selects `tournament_size` individuals from the current population
 * and returns a reference to the one with the highest fitness score among them.
 * @param tournament_size The number of participants in each tournament.
 * @return const NeuroNet& A reference to the selected parent individual.
 * @throws std::runtime_error if the population is empty.
 */
const NeuroNet::NeuroNet& Optimization::GeneticAlgorithm::tournament_selection(int tournament_size) const {
    if (population_.empty()) {
        throw std::runtime_error("Tournament selection called on an empty population.");
    }
    if (fitness_scores_.size() != population_.size()) {
        throw std::runtime_error("Fitness scores are not aligned with the population for tournament selection.");
    }
    
    std::uniform_int_distribution<int> dist(0, population_.size() - 1);
    const NeuroNet::NeuroNet* best_participant = nullptr;
    double max_fitness_in_tournament = std::numeric_limits<double>::lowest();

    for (int i = 0; i < tournament_size; ++i) {
        int idx = dist(random_engine_);
        if (fitness_scores_[idx] > max_fitness_in_tournament) {
            max_fitness_in_tournament = fitness_scores_[idx];
            best_participant = &population_[idx];
        }
    }
    // If all selected participants have the lowest possible fitness, best_participant might still be one of them.
    // If tournament_size is 0 or population is small, this might need adjustment, but typical use has tournament_size > 0.
    if (!best_participant) {
        // Fallback: if for some reason no best participant was chosen (e.g., all fitnesses are -infinity and tournament_size=0),
        // pick a random one. This case should be rare with typical parameters.
        return population_[dist(random_engine_)];
    }
    return *best_participant;
}

/**
 * @brief Creates a new generation of individuals through selection, crossover, and mutation.
 *
 * 1. Elitism: The best individual from the current generation is carried over to the new population.
 * 2. The rest of the new population is filled by:
 *    a. Selecting two parents using tournament selection.
 *    b. With probability `crossover_rate_`, performing crossover to produce offspring.
 *       These offspring are then mutated.
 *    c. Otherwise (no crossover), the selected parents are mutated and added.
 * The `population_` member is updated with the new generation.
 */
void Optimization::GeneticAlgorithm::selection() {
#ifdef ENABLE_BENCHMARKING
    utilities::Timer selection_timer;
    selection_timer.start();
#endif
    if (population_.empty() || fitness_scores_.empty() || fitness_scores_.size() != population_.size()) {
        // Or throw std::runtime_error("Population or fitness scores are not ready for selection.");
#ifdef ENABLE_BENCHMARKING
        selection_timer.stop();
        std::cout << "GeneticAlgorithm::selection() (Population/Fitness Scores Empty or Mismatched) took: " << selection_timer.elapsed_milliseconds() << " ms" << std::endl;
#endif
        return;
    }

    std::vector<NeuroNet::NeuroNet> new_population;
    new_population.reserve(population_size_);

    // Elitism: Carry over the best individual from the current population.
    auto it_best_current_gen = std::max_element(fitness_scores_.begin(), fitness_scores_.end());
    if (it_best_current_gen != fitness_scores_.end()) {
        int best_idx = std::distance(fitness_scores_.begin(), it_best_current_gen);
        if (best_idx >=0 && static_cast<size_t>(best_idx) < population_.size()){ // Ensure index is valid
             new_population.push_back(population_[best_idx]); // Add a copy of the best
        }
    } else if (!population_.empty()){ // Fallback if max_element fails unexpectedly but population exists
        new_population.push_back(population_[0]); // Add first element as a fallback
    }


    // Fill the rest of the new population.
    while (new_population.size() < static_cast<size_t>(population_size_)) {
        const NeuroNet::NeuroNet& parent1 = tournament_selection();
        const NeuroNet::NeuroNet& parent2 = tournament_selection();

        std::uniform_real_distribution<double> cross_dist(0.0, 1.0);
        if (cross_dist(random_engine_) < crossover_rate_) {
            std::vector<NeuroNet::NeuroNet> offspring = crossover(parent1, parent2); // Get two offspring
            mutate(offspring[0]); // Mutate first offspring
            if (new_population.size() < static_cast<size_t>(population_size_)) {
                new_population.push_back(offspring[0]);
            }
            if (new_population.size() < static_cast<size_t>(population_size_)) { // Check size again for second offspring
                mutate(offspring[1]); // Mutate second offspring
                new_population.push_back(offspring[1]);
            }
        } else {
            // No crossover: clone parents, mutate, and add to new population.
            NeuroNet::NeuroNet p1_copy = parent1;
            mutate(p1_copy);
            if (new_population.size() < static_cast<size_t>(population_size_)) {
                new_population.push_back(p1_copy);
            }

            if (new_population.size() < static_cast<size_t>(population_size_)) { // Check size again
                NeuroNet::NeuroNet p2_copy = parent2;
                mutate(p2_copy);
                new_population.push_back(p2_copy);
            }
        }
    }
    population_ = new_population; // Replace old population with the new one.
#ifdef ENABLE_BENCHMARKING
    selection_timer.stop();
    std::cout << "GeneticAlgorithm::selection() took: " << selection_timer.elapsed_milliseconds() << " ms" << std::endl;
#endif
}

/**
 * @brief Performs single-point crossover on the flattened weight and bias vectors of two parents.
 *
 * Offspring inherit parts of their genetic material (weights/biases) from each parent.
 * A crossover point is chosen randomly for weights and biases independently.
 * @param parent1 The first parent NeuroNet.
 * @param parent2 The second parent NeuroNet.
 * @return std::vector<NeuroNet> A vector containing two new offspring NeuroNet individuals.
 */
std::vector<NeuroNet::NeuroNet> Optimization::GeneticAlgorithm::crossover(const NeuroNet::NeuroNet& parent1, const NeuroNet::NeuroNet& parent2) {
#ifdef ENABLE_BENCHMARKING
    utilities::Timer crossover_timer;
    crossover_timer.start();
#endif
    NeuroNet::NeuroNet offspring1 = template_network_; // Ensure offspring have the correct structure.
    NeuroNet::NeuroNet offspring2 = template_network_;

    // Crossover for weights
    std::vector<float> p1_weights = parent1.get_all_weights_flat();
    std::vector<float> p2_weights = parent2.get_all_weights_flat();
    
    if (p1_weights.size() == p2_weights.size() && !p1_weights.empty()) {
        std::uniform_int_distribution<int> dist(0, p1_weights.size() - 1); // Crossover point index
        int crossover_point = dist(random_engine_);

        std::vector<float> o1_weights = p1_weights; // Start with parent1's weights
        std::vector<float> o2_weights = p2_weights; // Start with parent2's weights

        // Swap genetic material after the crossover point
        for (size_t i = crossover_point; i < p1_weights.size(); ++i) {
            std::swap(o1_weights[i], o2_weights[i]);
        }
        offspring1.set_all_weights_flat(o1_weights);
        offspring2.set_all_weights_flat(o2_weights);
    } else { 
        // Fallback: If sizes mismatch or weights are empty (shouldn't happen with proper template use),
        // offspring are effectively clones of parents in terms of weights.
        offspring1.set_all_weights_flat(p1_weights);
        offspring2.set_all_weights_flat(p2_weights);
    }

    // Crossover for biases (similarly to weights)
    std::vector<float> p1_biases = parent1.get_all_biases_flat();
    std::vector<float> p2_biases = parent2.get_all_biases_flat();

    if (p1_biases.size() == p2_biases.size() && !p1_biases.empty()) {
        std::uniform_int_distribution<int> dist(0, p1_biases.size() - 1);
        int crossover_point = dist(random_engine_);
        
        std::vector<float> o1_biases = p1_biases;
        std::vector<float> o2_biases = p2_biases;

        for (size_t i = crossover_point; i < p1_biases.size(); ++i) {
            std::swap(o1_biases[i], o2_biases[i]);
        }
        offspring1.set_all_biases_flat(o1_biases);
        offspring2.set_all_biases_flat(o2_biases);
    } else {
        offspring1.set_all_biases_flat(p1_biases);
        offspring2.set_all_biases_flat(p2_biases);
    }

#ifdef ENABLE_BENCHMARKING
    crossover_timer.stop();
    std::cout << "GeneticAlgorithm::crossover() took: " << crossover_timer.elapsed_microseconds() << " us" << std::endl;
#endif
    return {offspring1, offspring2};
}

/**
 * @brief Applies random mutations to the weights and biases of an individual.
 *
 * Each weight and bias has a `mutation_rate_` chance of being perturbed
 * by a small random value (e.g., adding a value between -0.1 and 0.1).
 * @param individual The NeuroNet to mutate (modified in place).
 */
void Optimization::GeneticAlgorithm::mutate(NeuroNet::NeuroNet& individual) {
#ifdef ENABLE_BENCHMARKING
    utilities::Timer mutate_timer;
    mutate_timer.start();
#endif
    std::uniform_real_distribution<double> prob_dist(0.0, 1.0); // For checking against mutation_rate_
    std::uniform_real_distribution<float> mutation_val_dist(-0.1f, 0.1f); // Magnitude of mutation

    // Mutate weights
    std::vector<float> weights = individual.get_all_weights_flat();
    for (float& weight : weights) {
        if (prob_dist(random_engine_) < mutation_rate_) {
            weight += mutation_val_dist(random_engine_);
            // Optional: Clamp weights to a specific range, e.g., [-1.0f, 1.0f]
            // weight = std::max(-1.0f, std::min(1.0f, weight));
        }
    }
    individual.set_all_weights_flat(weights);

    // Mutate biases
    std::vector<float> biases = individual.get_all_biases_flat();
    for (float& bias : biases) {
        if (prob_dist(random_engine_) < mutation_rate_) {
            bias += mutation_val_dist(random_engine_);
            // Optional: Clamp biases
            // bias = std::max(-1.0f, std::min(1.0f, bias));
        }
    }
    individual.set_all_biases_flat(biases);
#ifdef ENABLE_BENCHMARKING
    mutate_timer.stop();
    std::cout << "GeneticAlgorithm::mutate() took: " << mutate_timer.elapsed_microseconds() << " us" << std::endl;
#endif
}

/**
 * @brief Executes one full cycle of the genetic algorithm (one generation).
 *
 * This involves:
 * 1. Evaluating the fitness of the current population.
 * 2. Updating the record of the best individual found so far if a better one emerges in this generation.
 * 3. Performing selection (which internally handles crossover and mutation for the new population).
 * @param fitness_function The function used to evaluate the fitness of each individual.
 */
void Optimization::GeneticAlgorithm::evolve_one_generation(const std::function<double(NeuroNet::NeuroNet&)>& fitness_function) {
    if (population_.empty()) {
        // This might happen if initialize_population was not called before the first evolution step.
        // Or if population_size_ is 0.
        initialize_population(); 
        if(population_.empty() && population_size_ > 0) { // If still empty after trying to init
             throw std::runtime_error("Population is empty even after initialization attempt in evolve_one_generation.");
        } else if (population_size_ == 0) { // If population_size_ is 0
            return; // Nothing to evolve
        }
    }
    
    evaluate_fitness(fitness_function); // Fitness scores are now up-to-date for the current population.

    // Identify the best individual in the current generation and update overall best if it's better.
    // This ensures best_individual_ always reflects the absolute best found up to this point.
    auto it_current_gen_best = std::max_element(fitness_scores_.begin(), fitness_scores_.end());
    if (it_current_gen_best != fitness_scores_.end()) {
        double current_gen_best_fitness = *it_current_gen_best;
        if (current_gen_best_fitness > best_fitness_score_) {
            best_fitness_score_ = current_gen_best_fitness;
            int best_idx = std::distance(fitness_scores_.begin(), it_current_gen_best);
            if (best_idx >= 0 && static_cast<size_t>(best_idx) < population_.size()){
                 best_individual_ = population_[best_idx]; // Store a copy of the new best
            }
        }
    } else if (best_individual_.get_all_weights_flat().empty() && !population_.empty()) {
        // Handle case for the very first generation where best_fitness_score_ is lowest_double
        // and no overall best is yet set.
        // This logic is mostly covered by the above, but as a safeguard for first run:
        if (!fitness_scores_.empty()){ //Ensure fitness_scores_ is not empty before accessing
            best_fitness_score_ = fitness_scores_[0]; // Initialize with first individual
            best_individual_ = population_[0];
            for(size_t i = 1; i < population_.size(); ++i) {
                if(fitness_scores_[i] > best_fitness_score_){
                    best_fitness_score_ = fitness_scores_[i];
                    best_individual_ = population_[i];
                }
            }
        }
    }
    
    selection(); // Create the next generation (population_ is updated internally).
                 // selection() method as implemented also calls crossover and mutate.
}

/**
 * @brief Runs the genetic algorithm for the configured number of generations.
 *
 * It first initializes the population, then iteratively calls `evolve_one_generation`.
 * @param fitness_function The function to evaluate individual fitness.
 */
void Optimization::GeneticAlgorithm::run_evolution(const std::function<double(NeuroNet::NeuroNet&)>& fitness_function) {
    initialize_population(); // Prepare the initial random population.

    for (int i = 0; i < num_generations_; ++i) {
        evolve_one_generation(fitness_function);
        // Optional: Add logging here to track progress, e.g., best fitness per generation.
        // std::cout << "Generation " << (i + 1) << "/" << num_generations_
        //           << " - Best Fitness: " << best_fitness_score_ << std::endl;
    }
}

/**
 * @brief Returns the best NeuroNet individual found across all generations.
 *
 * If the evolution process has not yet run or if the population is empty,
 * it might return a default-constructed or the initially stored `best_individual_`
 * (which could be default if no better one was found).
 * @return NeuroNet A copy of the best performing individual.
 */
NeuroNet::NeuroNet Optimization::GeneticAlgorithm::get_best_individual() const {
    // If best_individual_ was never updated (e.g. evolution didn't run or no valid individuals),
    // it might be default-constructed.
    // A check could be added: if best_fitness_score_ is still numeric_limits::lowest(),
    // it implies no valid individual was successfully evaluated and stored as best.
    if (best_fitness_score_ == std::numeric_limits<double>::lowest() && !population_.empty() && !fitness_scores_.empty()) {
        // Fallback: if evolution ran but best_individual_ somehow wasn't updated properly,
        // try to return the best from the current population as a last resort.
        // This situation should ideally not occur if evolve_one_generation is correct.
        auto it_best = std::max_element(fitness_scores_.begin(), fitness_scores_.end());
        if (it_best != fitness_scores_.end()) {
            int best_idx = std::distance(fitness_scores_.begin(), it_best);
            if (best_idx >= 0 && static_cast<size_t>(best_idx) < population_.size()){
                return population_[best_idx]; // Return best of current population
            }
        }
    }
    return best_individual_; // Return the best found across generations
}

} // namespace Optimization
