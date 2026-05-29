#define ENABLE_BENCHMARKING

#include "../src/math/matrix.h"
#include "../src/neural_network/neuronet.h"
#include "../src/optimization/genetic_algorithm.h"
#include "../src/utilities/timer.h" // For completeness, though timing is internal

#include <iostream>
#include <vector>
#include <functional>
#include <random> // For populating matrices with random data
#include <numeric> // For std::iota if needed

// Helper function to fill a matrix with random data
void fill_matrix_random(Matrix::Matrix<float>& mat) {
    if (mat.rows() == 0 || mat.cols() == 0) return;
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    for (size_t i = 0; i < mat.rows(); ++i) {
        for (size_t j = 0; j < mat.cols(); ++j) {
            mat[i][j] = dist(gen);
        }
    }
}

// Helper function to fill a matrix with sequential data (for simplicity)
void fill_matrix_sequential(Matrix::Matrix<float>& mat) {
    if (mat.rows() == 0 || mat.cols() == 0) return;
    float val = 0.0f;
    for (size_t i = 0; i < mat.rows(); ++i) {
        for (size_t j = 0; j < mat.cols(); ++j) {
            mat[i][j] = val++;
        }
    }
}


int main() {
    std::cout << "=============== Starting Benchmarks ===============" << std::endl;
    std::cout << "#define ENABLE_BENCHMARKING is active." << std::endl << std::endl;

    // Benchmark 1: Matrix Multiplication
    std::cout << "------- Benchmark 1: Matrix Multiplication -------" << std::endl;
    std::vector<int> matrix_sizes = {10, 50, 100, 200, 500};

    for (int size : matrix_sizes) {
        std::cout << "\n--- Benchmarking Matrix Multiplication " << size << "x" << size << " ---" << std::endl;
        Matrix::Matrix<float> A(size, size);
        Matrix::Matrix<float> B(size, size);
        
        fill_matrix_random(A);
        fill_matrix_random(B);

        // The actual multiplication will trigger the internal timer in matrix.h
        Matrix::Matrix<float> C = A * B; 
        std::cout << "Matrix C created with rows: " << C.rows() << ", cols: " << C.cols() << " (result not printed)" << std::endl;
        std::cout << "--- Finished Matrix Multiplication " << size << "x" << size << " ---" << std::endl;
    }
    std::cout << "------- Finished Benchmark 1: Matrix Multiplication -------" << std::endl << std::endl;

    // Benchmark 2: Neural Network Forward Pass
    std::cout << "------- Benchmark 2: Neural Network Forward Pass -------" << std::endl;

    // Scenario 2a: Small network
    std::cout << "\n--- Scenario 2a: Small Neural Network ---" << std::endl;
    NeuroNet::NeuroNet small_nn;
    small_nn.SetInputSize(10);
    small_nn.ResizeNeuroNet(2); // 2 layers (1 hidden, 1 output)
    small_nn.ResizeLayer(0, 5);  // Hidden layer: 5 neurons
    small_nn.ResizeLayer(1, 2);  // Output layer: 2 neurons
    // Optionally set activation functions if desired for specific behavior, though not critical for benchmark
    small_nn.getLayer(0).SetActivationFunction(NeuroNet::ActivationFunctionType::ReLU);
    small_nn.getLayer(1).SetActivationFunction(NeuroNet::ActivationFunctionType::Softmax);


    Matrix::Matrix<float> small_input(1, 10);
    fill_matrix_random(small_input);
    small_nn.SetInput(small_input);
    Matrix::Matrix<float> small_output = small_nn.GetOutput();
    std::cout << "Small NN Output (rows: " << small_output.rows() << ", cols: " << small_output.cols() << ") obtained." << std::endl;
    std::cout << "--- Finished Scenario 2a ---" << std::endl;

    // Scenario 2b: Medium network
    std::cout << "\n--- Scenario 2b: Medium Neural Network ---" << std::endl;
    NeuroNet::NeuroNet medium_nn;
    medium_nn.SetInputSize(50);
    medium_nn.ResizeNeuroNet(3); // 3 layers (2 hidden, 1 output)
    medium_nn.ResizeLayer(0, 25); // Hidden layer 1: 25 neurons
    medium_nn.ResizeLayer(1, 25); // Hidden layer 2: 25 neurons
    medium_nn.ResizeLayer(2, 10); // Output layer: 10 neurons
    medium_nn.getLayer(0).SetActivationFunction(NeuroNet::ActivationFunctionType::ReLU);
    medium_nn.getLayer(1).SetActivationFunction(NeuroNet::ActivationFunctionType::ReLU);
    medium_nn.getLayer(2).SetActivationFunction(NeuroNet::ActivationFunctionType::None);


    Matrix::Matrix<float> medium_input(1, 50);
    fill_matrix_random(medium_input);
    medium_nn.SetInput(medium_input);
    Matrix::Matrix<float> medium_output = medium_nn.GetOutput();
    std::cout << "Medium NN Output (rows: " << medium_output.rows() << ", cols: " << medium_output.cols() << ") obtained." << std::endl;
    std::cout << "--- Finished Scenario 2b ---" << std::endl;

    // Scenario 2c: Large network
    std::cout << "\n--- Scenario 2c: Large Neural Network ---" << std::endl;
    NeuroNet::NeuroNet large_nn;
    large_nn.SetInputSize(100);
    large_nn.ResizeNeuroNet(4); // 4 layers (3 hidden, 1 output)
    large_nn.ResizeLayer(0, 50); // Hidden layer 1: 50 neurons
    large_nn.ResizeLayer(1, 50); // Hidden layer 2: 50 neurons
    large_nn.ResizeLayer(2, 50); // Hidden layer 3: 50 neurons
    large_nn.ResizeLayer(3, 20); // Output layer: 20 neurons
    // Set some activation functions
    for(int i=0; i<3; ++i) large_nn.getLayer(i).SetActivationFunction(NeuroNet::ActivationFunctionType::LeakyReLU);
    large_nn.getLayer(3).SetActivationFunction(NeuroNet::ActivationFunctionType::None);


    Matrix::Matrix<float> large_input(1, 100);
    fill_matrix_random(large_input);
    large_nn.SetInput(large_input);
    Matrix::Matrix<float> large_output = large_nn.GetOutput();
    std::cout << "Large NN Output (rows: " << large_output.rows() << ", cols: " << large_output.cols() << ") obtained." << std::endl;
    std::cout << "--- Finished Scenario 2c ---" << std::endl;
    std::cout << "------- Finished Benchmark 2: Neural Network Forward Pass -------" << std::endl << std::endl;

    // Benchmark 3: Genetic Algorithm Operations
    std::cout << "------- Benchmark 3: Genetic Algorithm Operations -------" << std::endl;
    // Use the small_nn structure as a template for GA
    // Ensure template_network is fully configured before passing to GA constructor.
    // Small_nn already has its layers sized. We can also randomize its weights/biases first.
    // This step is not strictly necessary for GA to function as GA randomizes individuals,
    // but a fully initialized template is good practice.
    std::vector<float> template_weights = small_nn.get_all_weights_flat();
    std::vector<float> template_biases = small_nn.get_all_biases_flat();
    std::uniform_real_distribution<float> dist_params(-0.5f, 0.5f);
    std::mt19937 ga_gen(std::random_device{}());
    for(float& w : template_weights) w = dist_params(ga_gen);
    for(float& b : template_biases) b = dist_params(ga_gen);
    small_nn.set_all_weights_flat(template_weights);
    small_nn.set_all_biases_flat(template_biases);


    int population_size = 50; // Example population size
    double mutation_rate = 0.05;
    double crossover_rate = 0.7;
    int num_generations_for_benchmark = 1; // For single evolution step benchmark

    Optimization::GeneticAlgorithm ga(
        population_size,
        mutation_rate,
        crossover_rate,
        num_generations_for_benchmark, // Will run only one generation via evolve_one_generation
        small_nn // Use the configured small_nn as template
    );

    // Dummy fitness function
    std::function<double(NeuroNet::NeuroNet&)> fitness_func = 
        [](NeuroNet::NeuroNet& nn) {
            // Simple dummy: get output, sum its elements. More complex might be needed for real scenarios.
            Matrix::Matrix<float> temp_input(1, nn.getLayer(0).InputSize > 0 ? nn.getLayer(0).InputSize : 10); // Use actual input size
            fill_matrix_random(temp_input); // Create some input
            nn.SetInput(temp_input);
            Matrix::Matrix<float> output = nn.GetOutput(); // This itself is timed internally by NeuroNet
            double sum = 0.0;
            if (output.rows() > 0 && output.cols() > 0) {
                for (size_t i = 0; i < output.rows(); ++i) {
                    for (size_t j = 0; j < output.cols(); ++j) {
                        sum += output[i][j];
                    }
                }
            }
            // Add a small random element to ensure fitness values vary a bit
            std::uniform_real_distribution<double> fit_rand(-0.1, 0.1);
            std::mt19937 fit_gen(std::random_device{}());
            return sum + fit_rand(fit_gen);
        };

    std::cout << "\n--- Benchmarking GA: initialize_population ---" << std::endl;
    // Note: initialize_population itself is not instrumented, but it calls create_random_individual,
    // which might be instrumented in future if it becomes complex. For now, timing the call itself.
    utilities::Timer init_pop_timer;
    init_pop_timer.start();
    ga.initialize_population();
    init_pop_timer.stop();
    std::cout << "GA initialize_population() for population size " << population_size 
              << " took: " << init_pop_timer.elapsed_milliseconds() << " ms (external timer)" << std::endl;
    std::cout << "--- Finished GA: initialize_population ---" << std::endl;


    std::cout << "\n--- Benchmarking GA: evaluate_fitness (population size: " << population_size << ") ---" << std::endl;
    // evaluate_fitness is instrumented internally.
    ga.evaluate_fitness(fitness_func);
    std::cout << "--- Finished GA: evaluate_fitness ---" << std::endl;

    std::cout << "\n--- Benchmarking GA: evolve_one_generation (includes selection, crossover, mutate) ---" << std::endl;
    // evolve_one_generation calls evaluate_fitness, selection, etc.
    // evaluate_fitness will be timed again here.
    // selection, crossover, and mutate are timed internally.
    ga.evolve_one_generation(fitness_func);
    std::cout << "--- Finished GA: evolve_one_generation ---" << std::endl;
    
    std::cout << "\n--- Benchmarking GA: run_evolution (for " << num_generations_for_benchmark << " generation(s)) ---" << std::endl;
    // For a slightly longer run, let's re-initialize GA with more generations
    // and call run_evolution.
    int few_generations = 3;
    Optimization::GeneticAlgorithm ga_run(
        population_size, mutation_rate, crossover_rate, few_generations, small_nn
    );
    // run_evolution calls initialize_population and then evolve_one_generation multiple times.
    // The internal timers for evaluate_fitness, selection, crossover, mutate will be called for each generation.
    utilities::Timer run_evol_timer;
    run_evol_timer.start();
    ga_run.run_evolution(fitness_func);
    run_evol_timer.stop();
     std::cout << "GA run_evolution() for " << few_generations << " generations (pop size " << population_size 
              << ") took: " << run_evol_timer.elapsed_milliseconds() << " ms (external timer)" << std::endl;
    std::cout << "--- Finished GA: run_evolution ---" << std::endl;


    std::cout << "------- Finished Benchmark 3: Genetic Algorithm Operations -------" << std::endl << std::endl;

    std::cout << "=============== All Benchmarks Finished ===============" << std::endl;

    return 0;
}
