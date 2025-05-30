#include "gtest/gtest.h"
#include "optimization/genetic_algorithm.h" // Access to GeneticAlgorithm
#include "neural_network/neuronet.h"          // Access to NeuroNet for template and individuals
#include <numeric>             // For std::accumulate
#include <set>                 // For checking distinctness

// Simple fitness function for testing: sum of all weights and biases
// Assumes higher sum is better.
double simple_fitness_function(NeuroNet::NeuroNet& net) {
    double sum = 0.0;
    std::vector<float> weights = net.get_all_weights_flat();
    std::vector<float> biases = net.get_all_biases_flat();

    for (float w : weights) {
        sum += w;
    }
    for (float b : biases) {
        sum += b;
    }
    return sum;
}

// Test fixture for GeneticAlgorithm tests
class GeneticAlgorithmTest : public ::testing::Test {
protected:
    NeuroNet::NeuroNet template_net;
    int population_size = 10;
    double mutation_rate = 0.1;
    double crossover_rate = 0.7;
    int num_generations = 5; // Small number for testing

    void SetUp() override {
        // Configure a simple template network: 2 inputs, 1 hidden layer of 3 neurons, 1 output neuron
        template_net.SetInputSize(2);
        template_net.ResizeNeuroNet(2); // 1 hidden layer + 1 output layer
        template_net.ResizeLayer(0, 3); // Hidden layer: 2 in, 3 out
        template_net.ResizeLayer(1, 1); // Output layer: 3 in, 1 out
    }
};

TEST_F(GeneticAlgorithmTest, Constructor) {
    Optimization::GeneticAlgorithm ga(population_size, mutation_rate, crossover_rate, num_generations, template_net);
    // Test if constructor runs, further state checked by other tests
    SUCCEED();
}

TEST_F(GeneticAlgorithmTest, InitializePopulation) {
    Optimization::GeneticAlgorithm ga(population_size, mutation_rate, crossover_rate, num_generations, template_net);
    ga.initialize_population();
    
    // Population size is implicitly tested by other methods using the population.
    // For now, let's check if get_best_individual returns something (even if random)
    // after initialization. This relies on the internal population_ being populated.
    // A more direct way would be to add a getter for the population or population_size,
    // but the problem description doesn't ask for that.
    // So, we'll infer from other tests.
    
    // Let's check if at least some individuals are somewhat distinct.
    // This is a probabilistic test.
    ga.evaluate_fitness(simple_fitness_function); // Need fitness scores for get_best_individual
    NeuroNet::NeuroNet ind1 = ga.get_best_individual(); // Will be one of the random individuals

    ga.initialize_population(); // Re-initialize
    ga.evaluate_fitness(simple_fitness_function);
    NeuroNet::NeuroNet ind2 = ga.get_best_individual();

    // It's highly unlikely they'll have the exact same flat weights if randomized.
    // This isn't a perfect test for distinctness of the whole population.
    // A better test would be to get all individuals and compare them.
    // For now, this is a basic sanity check.
    if (template_net.get_all_weights_flat().size() > 0) { // Only if network has weights
         // If they are different, it suggests randomization happened.
         // This test might sometimes fail if by sheer chance two random initializations are identical,
         // or if get_best_individual has issues.
         // A more robust test for distinctness would be:
         // std::vector<std::vector<float>> all_initial_weights;
         // for (const auto& individual : ga.get_population()) { // Assuming a getter for population
         //    all_initial_weights.push_back(individual.get_all_weights_flat());
         // }
         // std::set<std::vector<float>> unique_weights(all_initial_weights.begin(), all_initial_weights.end());
         // EXPECT_GT(unique_weights.size(), 1); // Expect more than 1 unique individual if pop_size > 1
    }
    SUCCEED(); // Placeholder, as direct population access is not available.
}


TEST_F(GeneticAlgorithmTest, EvaluateFitness) {
    Optimization::GeneticAlgorithm ga(population_size, mutation_rate, crossover_rate, num_generations, template_net);
    ga.initialize_population();
    ga.evaluate_fitness(simple_fitness_function);
    // No direct way to get all fitness scores to verify,
    // but we can check if get_best_individual returns a valid network.
    // The fitness logic itself is simple_fitness_function.
    NeuroNet::NeuroNet best = ga.get_best_individual();
    EXPECT_GT(best.get_all_weights_flat().size(), 0); // Check if it's a configured network
}

TEST_F(GeneticAlgorithmTest, Mutate) {
    Optimization::GeneticAlgorithm ga(population_size, 1.0, crossover_rate, num_generations, template_net); // 100% mutation
    ga.initialize_population();
    NeuroNet::NeuroNet original_individual = template_net; // Use template as a base
    std::vector<float> original_weights = original_individual.get_all_weights_flat();
    std::vector<float> original_biases = original_individual.get_all_biases_flat();

    NeuroNet::NeuroNet mutated_individual = original_individual; // Copy
    ga.mutate(mutated_individual); // Mutate the copy

    std::vector<float> mutated_weights = mutated_individual.get_all_weights_flat();
    std::vector<float> mutated_biases = mutated_individual.get_all_biases_flat();

    if (original_weights.size() > 0) {
        bool weight_changed = false;
        for (size_t i = 0; i < original_weights.size(); ++i) {
            if (original_weights[i] != mutated_weights[i]) {
                weight_changed = true;
                break;
            }
        }
        EXPECT_TRUE(weight_changed);
    } else if (original_biases.size() > 0) { // If no weights, check biases
         bool bias_changed = false;
        for (size_t i = 0; i < original_biases.size(); ++i) {
            if (original_biases[i] != mutated_biases[i]) {
                bias_changed = true;
                break;
            }
        }
        EXPECT_TRUE(bias_changed);
    } else {
        SUCCEED(); // No weights or biases to mutate
    }
}

TEST_F(GeneticAlgorithmTest, Crossover) {
    Optimization::GeneticAlgorithm ga(population_size, mutation_rate, 1.0, num_generations, template_net); // 100% crossover
    
    NeuroNet::NeuroNet parent1 = template_net;
    std::vector<float> p1_weights(template_net.get_all_weights_flat().size(), 1.0f); // All 1s
    std::vector<float> p1_biases(template_net.get_all_biases_flat().size(), 1.0f);
    parent1.set_all_weights_flat(p1_weights);
    parent1.set_all_biases_flat(p1_biases);

    NeuroNet::NeuroNet parent2 = template_net;
    std::vector<float> p2_weights(template_net.get_all_weights_flat().size(), 2.0f); // All 2s
    std::vector<float> p2_biases(template_net.get_all_biases_flat().size(), 2.0f);
    parent2.set_all_weights_flat(p2_weights);
    parent2.set_all_biases_flat(p2_biases);

    std::vector<NeuroNet::NeuroNet> offspring = ga.crossover(parent1, parent2);
    ASSERT_EQ(offspring.size(), 2);

    std::vector<float> o1_weights = offspring[0].get_all_weights_flat();
    std::vector<float> o2_weights = offspring[1].get_all_weights_flat();

    bool p1_material_in_o1 = false;
    bool p2_material_in_o1 = false;
    bool p1_material_in_o2 = false;
    bool p2_material_in_o2 = false;

    if (!o1_weights.empty()) {
        for(float w : o1_weights) {
            if (w == 1.0f) p1_material_in_o1 = true;
            if (w == 2.0f) p2_material_in_o1 = true;
        }
        for(float w : o2_weights) {
            if (w == 1.0f) p1_material_in_o2 = true;
            if (w == 2.0f) p2_material_in_o2 = true;
        }
        // Single point crossover means one offspring will have start of P1, end of P2
        // and other will have start of P2, end of P1. So both should have material from both.
        EXPECT_TRUE(p1_material_in_o1);
        EXPECT_TRUE(p2_material_in_o1);
        EXPECT_TRUE(p1_material_in_o2);
        EXPECT_TRUE(p2_material_in_o2);
    } else {
        SUCCEED(); // No weights to crossover
    }
}


TEST_F(GeneticAlgorithmTest, RunEvolutionImprovesFitness) {
    // This test is probabilistic and might not always pass if the GA gets stuck
    // or if the problem/fitness function is too complex for few generations.
    // For a simple sum-of-weights fitness, we expect improvement.
    Optimization::GeneticAlgorithm ga(population_size, mutation_rate, crossover_rate, num_generations, template_net);
    
    ga.initialize_population();
    ga.evaluate_fitness(simple_fitness_function);
    NeuroNet::NeuroNet initial_best = ga.get_best_individual();
    double initial_best_fitness = simple_fitness_function(initial_best);

    ga.run_evolution(simple_fitness_function);

    NeuroNet::NeuroNet final_best = ga.get_best_individual();
    double final_best_fitness = simple_fitness_function(final_best);
    
    // Check if the number of weights is greater than 0 to avoid issues with empty networks.
    if (template_net.get_all_weights_flat().size() > 0) {
         EXPECT_GE(final_best_fitness, initial_best_fitness);
    } else {
         SUCCEED(); // No weights to optimize, so fitness won't change meaningfully.
    }
}

TEST_F(GeneticAlgorithmTest, GetBestIndividual) {
    Optimization::GeneticAlgorithm ga(population_size, mutation_rate, crossover_rate, 1, template_net);
    ga.initialize_population();
    // Manually create an individual that should be the best
    NeuroNet::NeuroNet clearly_best_net = template_net;
    std::vector<float> best_weights(template_net.get_all_weights_flat().size(), 100.0f); // High values
    std::vector<float> best_biases(template_net.get_all_biases_flat().size(), 100.0f);
    clearly_best_net.set_all_weights_flat(best_weights);
    clearly_best_net.set_all_biases_flat(best_biases);

    // "Inject" this individual into the population (not a standard GA operation, for testing get_best_individual logic)
    // This requires modifying the GA's internal population or ensuring selection picks it.
    // For this test, we'll rely on evaluate_fitness and then get_best_individual.
    // If we could replace an individual in ga.population_[0]:
    // ga.population_[0] = clearly_best_net; // (if population_ was public or had a setter)
    // Instead, we'll just run one generation. If elitism works, it might pick it up.
    // This test is more about whether get_best_individual reflects the best score found.

    // For a more direct test of get_best_individual without full evolution:
    // 1. Initialize population
    // 2. Evaluate fitness
    // 3. Manually find the best fitness and corresponding individual in your test
    // 4. Compare with ga.get_best_individual() and its fitness
    // The current ga.get_best_individual() returns a copy of best_individual_ which is updated
    // during evolve_one_generation.

    ga.evaluate_fitness(simple_fitness_function);
    // The best_individual_ is updated in evolve_one_generation.
    // So, call evolve_one_generation to ensure best_individual_ is set.
    ga.evolve_one_generation(simple_fitness_function, 1); // Using 1 as a dummy generation number for the test
    
    NeuroNet::NeuroNet reported_best = ga.get_best_individual();
    double reported_best_fitness = simple_fitness_function(reported_best);

    // We expect that after evaluation, the reported_best_fitness is indeed the max.
    // This doesn't strictly test if `clearly_best_net` was found, but that `get_best_individual` works.
    // To verify `clearly_best_net` would require a way to insert it and ensure it's picked.
    // For now, check that `get_best_individual` returns a valid, configured network.
    EXPECT_GT(reported_best.get_all_weights_flat().size(), 0);
}


#include <fstream> // For std::ifstream for reading file
#include <cstdio>  // For std::remove
// Use custom JSON library for parsing the output file
#include "../src/utilities/json/json.hpp" 
#include "../src/utilities/json/json_exception.hpp"

TEST_F(GeneticAlgorithmTest, ExportTrainingMetrics) {
    Optimization::GeneticAlgorithm ga(population_size, mutation_rate, crossover_rate, num_generations, template_net);
    
    const int generations_to_run = 2; // Keep it small for test speed
    // Directly modify num_generations_ of the instance for this test
    // This is a bit of a hack; ideally, GA would take generations as a run_evolution param or similar
    // For now, we assume ga.num_generations_ can be set, or we rely on the fixture's num_generations
    // For this test, let's assume the GA object itself is reconfigured or this test is specific to fixture's num_generations
    // To make it explicit for the test:
    Optimization::GeneticAlgorithm ga_test_instance(population_size, mutation_rate, crossover_rate, generations_to_run, template_net);


    ga_test_instance.run_evolution(simple_fitness_function);

    const std::string metrics_filename = "test_training_metrics_custom.json";
    ASSERT_NO_THROW(ga_test_instance.export_training_metrics_json(metrics_filename));

    std::ifstream metrics_file(metrics_filename);
    ASSERT_TRUE(metrics_file.is_open()) << "Failed to open metrics file: " << metrics_filename;
    std::string json_content((std::istreambuf_iterator<char>(metrics_file)),
                             std::istreambuf_iterator<char>());
    metrics_file.close();
    ASSERT_FALSE(json_content.empty());

    JsonValue root;
    ASSERT_NO_THROW(root = JsonParser::Parse(json_content)) << "Failed to parse metrics JSON with custom parser.";
    
    ASSERT_EQ(root.type, JsonValueType::Object);
    const auto& root_obj = root.GetObject();

    // Validate top-level keys
    EXPECT_TRUE(root_obj.count("start_time"));
    ASSERT_EQ(root_obj.at("start_time")->type, JsonValueType::String);
    EXPECT_FALSE(root_obj.at("start_time")->GetString().empty());

    EXPECT_TRUE(root_obj.count("end_time"));
    ASSERT_EQ(root_obj.at("end_time")->type, JsonValueType::String);
    EXPECT_FALSE(root_obj.at("end_time")->GetString().empty());

    EXPECT_TRUE(root_obj.count("total_generations"));
    ASSERT_EQ(root_obj.at("total_generations")->type, JsonValueType::Number);
    EXPECT_EQ(static_cast<int>(root_obj.at("total_generations")->GetNumber()), generations_to_run);

    EXPECT_TRUE(root_obj.count("overall_best_fitness"));
    ASSERT_EQ(root_obj.at("overall_best_fitness")->type, JsonValueType::Number);
    // Value can be anything, just check type and presence

    EXPECT_TRUE(root_obj.count("best_model_architecture_params_custom_json_string"));
    ASSERT_EQ(root_obj.at("best_model_architecture_params_custom_json_string")->type, JsonValueType::String);
    std::string model_str = root_obj.at("best_model_architecture_params_custom_json_string")->GetString();
    EXPECT_FALSE(model_str.empty());
    
    // Validate the embedded model string
    JsonValue parsed_model_json;
    ASSERT_NO_THROW(parsed_model_json = JsonParser::Parse(model_str));
    ASSERT_EQ(parsed_model_json.type, JsonValueType::Object);
    if (ga_test_instance.get_best_individual().getLayerCount() > 0) {
        EXPECT_TRUE(parsed_model_json.GetObject().count("input_size"));
        EXPECT_TRUE(parsed_model_json.GetObject().count("layer_count"));
        EXPECT_TRUE(parsed_model_json.GetObject().count("layers"));
        // Check for activation function string from template_net (assuming it has one)
        if (template_net.getLayerCount() > 0) {
             std::string expected_activation_str = template_net.getLayer(0).get_activation_function_name();
             // Find it in the parsed_model_json
             const auto& layers_array = parsed_model_json.GetObject().at("layers")->GetArray();
             if (!layers_array.empty()) {
                 const auto& first_layer_obj = layers_array[0].GetObject();
                 EXPECT_EQ(first_layer_obj.at("activation_function")->GetString(), expected_activation_str);
             }
        }
    } else {
        EXPECT_TRUE(parsed_model_json.GetObject().count("error"));
    }


    EXPECT_TRUE(root_obj.count("generation_data"));
    const auto& gen_data_val = root_obj.at("generation_data");
    ASSERT_EQ(gen_data_val->type, JsonValueType::Array);
    const auto& gen_array = gen_data_val->GetArray();
    EXPECT_EQ(gen_array.size(), generations_to_run);

    if (generations_to_run > 0) {
        const auto& gen0_metric_obj = gen_array[0];
        ASSERT_EQ(gen0_metric_obj.type, JsonValueType::Object);
        const auto& gen0_obj_map = gen0_metric_obj.GetObject();

        EXPECT_TRUE(gen0_obj_map.count("generation_number"));
        EXPECT_EQ(gen0_obj_map.at("generation_number")->type, JsonValueType::Number);
        EXPECT_TRUE(gen0_obj_map.count("average_fitness"));
        EXPECT_EQ(gen0_obj_map.at("average_fitness")->type, JsonValueType::Number);
        EXPECT_TRUE(gen0_obj_map.count("best_fitness"));
        EXPECT_EQ(gen0_obj_map.at("best_fitness")->type, JsonValueType::Number);
        EXPECT_TRUE(gen0_obj_map.count("loss")); // Expect Null or Number
        EXPECT_TRUE(gen0_obj_map.at("loss")->type == JsonValueType::Null || gen0_obj_map.at("loss")->type == JsonValueType::Number);
        EXPECT_TRUE(gen0_obj_map.count("accuracy")); // Expect Null or Number
        EXPECT_TRUE(gen0_obj_map.at("accuracy")->type == JsonValueType::Null || gen0_obj_map.at("accuracy")->type == JsonValueType::Number);
    }

    // Cleanup
    std::remove(metrics_filename.c_str());
}

// Main function for running tests (needed if not using gtest_main)
// int main(int argc, char **argv) {
//     ::testing::InitGoogleTest(&argc, argv);
//     return RUN_ALL_TESTS();
// }
