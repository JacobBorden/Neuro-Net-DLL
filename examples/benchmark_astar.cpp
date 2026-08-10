#include "../src/neural_network/neuronet.h"
#include "../src/optimization/neural_pathfinder.h"
#include <iostream>
#include <chrono>

int main() {
    NeuroNet::NeuroNet nn;
    nn.SetInputSize(100);
    nn.ResizeNeuroNet(10);
    for (int i = 0; i < 10; ++i) {
        nn.ResizeLayer(i, 100);
    }

    NeuroNet::Optimization::NeuralPathfinder pathfinder(nn);

    auto start = std::chrono::high_resolution_clock::now();
    for(int i = 0; i < 100; ++i) {
        pathfinder.FindOptimalPathAStar();
    }
    auto end = std::chrono::high_resolution_clock::now();

    std::cout << "Time: " << std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count() << " ms\n";
    return 0;
}
