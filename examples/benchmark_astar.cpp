#include "neural_network/neuronet.h"
#include "optimization/neural_pathfinder.h"
#include <iostream>
#include <chrono>

#ifdef _OPENMP
#include <omp.h>
#endif

int main() {
    std::cout << "=============== Starting A* Pathfinder Benchmark ===============" << std::endl;

#if defined(__clang__)
    std::cout << "Compiler: Clang " << __clang_version__ << std::endl;
#elif defined(__GNUC__)
    std::cout << "Compiler: GCC " << __VERSION__ << std::endl;
#elif defined(_MSC_VER)
    std::cout << "Compiler: MSVC " << _MSC_VER << std::endl;
#else
    std::cout << "Compiler: Unknown" << std::endl;
#endif

#ifdef _OPENMP
    std::cout << "Thread Count (OpenMP): " << omp_get_max_threads() << std::endl;
#else
    std::cout << "Thread Count: 1 (OpenMP Disabled)" << std::endl;
#endif

    constexpr int input_size = 100;
    constexpr int num_layers = 10;
    constexpr int layer_neurons = 100;
    constexpr int repetitions = 100;

    std::cout << "Dimensions: Input Size " << input_size << ", Layers " << num_layers
              << " (" << layer_neurons << " neurons/layer)" << std::endl;
    std::cout << "Repetitions: " << repetitions << std::endl;

    NeuroNet::NeuroNet nn;
    nn.SetInputSize(input_size);
    nn.ResizeNeuroNet(num_layers);
    for (int i = 0; i < num_layers; ++i) {
        nn.ResizeLayer(i, layer_neurons);
    }

    NeuroNet::Optimization::NeuralPathfinder pathfinder(nn);

    auto start = std::chrono::high_resolution_clock::now();
    for(int i = 0; i < repetitions; ++i) {
        auto path = pathfinder.FindOptimalPathAStar();
    }
    auto end = std::chrono::high_resolution_clock::now();

    auto elapsed_ms = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
    std::cout << "Time: " << elapsed_ms << " ms ("
              << static_cast<double>(elapsed_ms) / repetitions << " ms/iteration)" << std::endl;
    std::cout << "=============== Finished A* Pathfinder Benchmark ===============" << std::endl;
    return 0;
}
