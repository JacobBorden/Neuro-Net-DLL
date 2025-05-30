#pragma once

#include <string>
#include <vector>
#include <limits> // Required for std::numeric_limits

#include "utilities/json/json.hpp"
#include "neural_network/neuronet.h" // For NeuroNet::NeuroNet

namespace Optimization {

    // Forward declaration of NeuroNet if necessary, assuming NeuroNet might include this header
    // namespace NeuroNet {
    //     class NeuroNet; 
    // }

    struct GenerationMetrics {
        int generation_number = 0;
        double average_fitness = 0.0;
        double best_fitness = 0.0;
        double loss = std::numeric_limits<double>::quiet_NaN(); // Default to NaN
        double accuracy = std::numeric_limits<double>::quiet_NaN(); // Default to NaN

        // Method to convert GenerationMetrics to nlohmann::json
        nlohmann::json to_json() const {
            nlohmann::json j;
            j["generation_number"] = generation_number;
            j["average_fitness"] = average_fitness;
            j["best_fitness"] = best_fitness;
            j["loss"] = loss;
            j["accuracy"] = accuracy;
            return j;
        }

        // Method to convert nlohmann::json to GenerationMetrics
        static GenerationMetrics from_json(const nlohmann::json& j) {
            GenerationMetrics m;
            if (j.contains("generation_number") && j["generation_number"].is_number_integer()) {
                m.generation_number = j["generation_number"].get<int>();
            }
            if (j.contains("average_fitness") && j["average_fitness"].is_number()) {
                m.average_fitness = j["average_fitness"].get<double>();
            }
            if (j.contains("best_fitness") && j["best_fitness"].is_number()) {
                m.best_fitness = j["best_fitness"].get<double>();
            }
            if (j.contains("loss") && j["loss"].is_number()) {
                m.loss = j["loss"].get<double>();
            }
            if (j.contains("accuracy") && j["accuracy"].is_number()) {
                m.accuracy = j["accuracy"].get<double>();
            }
            return m;
        }
    };

    // Non-member functions for nlohmann::json serialization (alternative to member functions)
    inline void to_json(nlohmann::json& j, const GenerationMetrics& m) {
        j = m.to_json();
    }

    inline void from_json(const nlohmann::json& j, GenerationMetrics& m) {
        m = GenerationMetrics::from_json(j);
    }

    struct TrainingRunMetrics {
        std::string start_time;
        std::string end_time;
        int total_generations = 0;
        std::vector<GenerationMetrics> generation_data;
        // Placeholder for best model architecture - using nlohmann::json to store model parameters
        nlohmann::json best_model_architecture_params; 

        // Method to convert TrainingRunMetrics to nlohmann::json
        nlohmann::json to_json() const {
            nlohmann::json j;
            j["start_time"] = start_time;
            j["end_time"] = end_time;
            j["total_generations"] = total_generations;
            j["generation_data"] = nlohmann::json::array();
            for (const auto& gen_metric : generation_data) {
                j["generation_data"].push_back(gen_metric.to_json());
            }
            j["best_model_architecture_params"] = best_model_architecture_params;
            return j;
        }

        // Method to convert nlohmann::json to TrainingRunMetrics
        static TrainingRunMetrics from_json(const nlohmann::json& j) {
            TrainingRunMetrics trm;
            if (j.contains("start_time") && j["start_time"].is_string()) {
                trm.start_time = j["start_time"].get<std::string>();
            }
            if (j.contains("end_time") && j["end_time"].is_string()) {
                trm.end_time = j["end_time"].get<std::string>();
            }
            if (j.contains("total_generations") && j["total_generations"].is_number_integer()) {
                trm.total_generations = j["total_generations"].get<int>();
            }
            if (j.contains("generation_data") && j["generation_data"].is_array()) {
                for (const auto& item : j["generation_data"]) {
                    trm.generation_data.push_back(GenerationMetrics::from_json(item));
                }
            }
            if (j.contains("best_model_architecture_params")) { // Can be any type of json
                trm.best_model_architecture_params = j["best_model_architecture_params"];
            }
            return trm;
        }
    };

    // Non-member functions for nlohmann::json serialization
    inline void to_json(nlohmann::json& j, const TrainingRunMetrics& trm) {
        j = trm.to_json();
    }

    inline void from_json(const nlohmann::json& j, TrainingRunMetrics& trm) {
        trm = TrainingRunMetrics::from_json(j);
    }

} // namespace Optimization
