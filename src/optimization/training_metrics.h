#pragma once

#include <string>
#include <vector>
#include <limits> // Required for std::numeric_limits
// #include "utilities/json/json.hpp" // Removed nlohmann::json include
// #include "neural_network/neuronet.h" // For NeuroNet::NeuroNet - Removed as NeuroNet object is no longer stored

namespace Optimization {

    struct GenerationMetrics {
        int generation_number = 0;
        double average_fitness = 0.0;
        double best_fitness = 0.0;
        double loss = std::numeric_limits<double>::quiet_NaN(); // Default to NaN
        double accuracy = std::numeric_limits<double>::quiet_NaN(); // Default to NaN

        // nlohmann::json related methods and free functions removed
    };

    struct TrainingRunMetrics {
        std::string start_time;
        std::string end_time;
        int total_generations = 0;
        std::vector<GenerationMetrics> generation_data;
        
        // Changed from nlohmann::json to std::string
        std::string best_model_architecture_params_custom_json_string; 
        
        // Added to store the fitness score previously part of the nlohmann::json object
        double overall_best_fitness = std::numeric_limits<double>::lowest(); 

        // nlohmann::json related methods and free functions removed
    };

} // namespace Optimization
