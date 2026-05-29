#include "vocabulary.h"
#include <fstream>      // For std::ifstream
#include <sstream>      // For std::istringstream (used in split_by_space)
#include <algorithm>    // For std::transform (used in to_lowercase) and std::max
#include <stdexcept>    // For std::runtime_error
#include "json/json_exception.hpp" // For JsonParseException

namespace NeuroNet {

Vocabulary::Vocabulary() :
    unknown_token_id_internal(-1),
    padding_token_id_internal(-1),
    max_sequence_length_internal_(-1) {}

std::string Vocabulary::to_lowercase(const std::string& str) const {
    std::string lower_str = str;
    std::transform(lower_str.begin(), lower_str.end(), lower_str.begin(),
                   [](unsigned char c){ return std::tolower(c); });
    return lower_str;
}

std::vector<std::string> Vocabulary::split_by_space(const std::string& str) const {
    std::vector<std::string> tokens;
    std::string token;
    std::istringstream tokenStream(str);
    while (std::getline(tokenStream, token, ' ')) {
        if (!token.empty()) { // Avoid empty tokens if there are multiple spaces
            tokens.push_back(token);
        }
    }
    return tokens;
}

bool Vocabulary::load_from_json(const std::string& filepath) {
    std::ifstream ifs(filepath);
    if (!ifs.is_open()) {
        // Consider logging an error here
        return false;
    }
    std::string content((std::istreambuf_iterator<char>(ifs)), std::istreambuf_iterator<char>());
    ifs.close();

    JsonValue root;
    try {
        root = JsonParser::Parse(content);
    } catch (const JsonParseException& e) {
        // Log e.what()
        return false;
    }

    if (root.type != JsonValueType::Object) return false;

    // Clear existing maps
    word_to_token_map.clear();
    token_to_word_map.clear();

    // Load word_to_token
    if (root.GetObject().count("word_to_token") && root.GetObject().at("word_to_token")->type == JsonValueType::Object) {
        const auto& w2t_obj = root.GetObject().at("word_to_token")->GetObject();
        for (const auto& pair : w2t_obj) {
            if (pair.second->type == JsonValueType::Number) {
                word_to_token_map[pair.first] = static_cast<int>(pair.second->GetNumber());
            }
        }
    } else return false; // word_to_token is mandatory

    // Load token_to_word
    if (root.GetObject().count("token_to_word") && root.GetObject().at("token_to_word")->type == JsonValueType::Object) {
        const auto& t2w_obj = root.GetObject().at("token_to_word")->GetObject();
        for (const auto& pair : t2w_obj) {
            try {
                int token_id = std::stoi(pair.first); // Keys in JSON object are strings
                if (pair.second->type == JsonValueType::String) {
                    token_to_word_map[token_id] = pair.second->GetString();
                }
            } catch (const std::invalid_argument& ia) {
                // key is not a valid integer, log or handle
            } catch (const std::out_of_range& oor) {
                // key is out of range for int, log or handle
            }
        }
    } else return false; // token_to_word is mandatory

    // Load special tokens
    if (root.GetObject().count("special_tokens") && root.GetObject().at("special_tokens")->type == JsonValueType::Object) {
        const auto& st_obj = root.GetObject().at("special_tokens")->GetObject();
        if (st_obj.count("unknown_token") && st_obj.at("unknown_token")->type == JsonValueType::String) {
            unknown_token_str_internal = st_obj.at("unknown_token")->GetString();
        } else return false; // unknown_token is mandatory in special_tokens

        if (st_obj.count("padding_token") && st_obj.at("padding_token")->type == JsonValueType::String) {
            padding_token_str_internal = st_obj.at("padding_token")->GetString();
        } else return false; // padding_token is mandatory

        // Get IDs for special tokens from the loaded word_to_token_map
        if (word_to_token_map.count(unknown_token_str_internal)) {
            unknown_token_id_internal = word_to_token_map.at(unknown_token_str_internal);
        } else return false; // unknown_token must exist in word_to_token

        if (word_to_token_map.count(padding_token_str_internal)) {
            padding_token_id_internal = word_to_token_map.at(padding_token_str_internal);
        } else return false; // padding_token must exist in word_to_token

    } else return false; // special_tokens block is mandatory

    // Optional: Load config like max_sequence_length
    if (root.GetObject().count("config") && root.GetObject().at("config")->type == JsonValueType::Object) {
        const auto& config_obj = root.GetObject().at("config")->GetObject();
        if (config_obj.count("max_sequence_length") && config_obj.at("max_sequence_length")->type == JsonValueType::Number) {
            max_sequence_length_internal_ = static_cast<int>(config_obj.at("max_sequence_length")->GetNumber());
        }
    }

    // Cleanup for JsonParser::Parse, as it dynamically allocates members for objects
    // This is crucial for the custom JSON parser
    if (root.type == JsonValueType::Object) {
        for (auto& pair : root.GetObject()) {
            if(pair.second->type == JsonValueType::Object){
                 for(auto& inner_pair : pair.second->GetObject()){
                    delete inner_pair.second; // delete JsonValue* from inner objects
                 }
            }
            delete pair.second; // delete JsonValue* from root object
        }
        root.GetObject().clear();
    }
    return true;
}

int Vocabulary::get_token_id(const std::string& word) const {
    std::string lower_word = to_lowercase(word);
    auto it = word_to_token_map.find(lower_word);
    if (it != word_to_token_map.end()) {
        return it->second;
    }
    return unknown_token_id_internal; // Should be set during load_from_json
}

std::string Vocabulary::get_word(int token_id) const {
    auto it = token_to_word_map.find(token_id);
    if (it != token_to_word_map.end()) {
        return it->second;
    }
    return unknown_token_str_internal; // Should be set
}

std::vector<int> Vocabulary::tokenize_sequence(const std::string& sequence_str) const {
    std::string processed_str = to_lowercase(sequence_str);
    // Basic pre-processing: could expand to remove punctuation if desired
    // For now, just lowercase and split
    std::vector<std::string> words = split_by_space(processed_str);
    std::vector<int> token_ids;
    token_ids.reserve(words.size());
    for (const std::string& word : words) {
        token_ids.push_back(get_token_id(word));
    }
    return token_ids;
}

std::vector<std::vector<int>> Vocabulary::tokenize_batch(const std::vector<std::string>& batch_sequences) const {
    std::vector<std::vector<int>> batch_token_ids;
    batch_token_ids.reserve(batch_sequences.size());
    for (const std::string& seq_str : batch_sequences) {
        batch_token_ids.push_back(tokenize_sequence(seq_str));
    }
    return batch_token_ids;
}

Matrix::Matrix<float> Vocabulary::prepare_batch_matrix(
    const std::vector<std::string>& batch_sequences,
    int max_len_param,
    bool pad_to_max_in_batch) const {

    if (padding_token_id_internal == -1) {
        throw std::runtime_error("Padding token ID is not set. Load vocabulary first.");
    }

    std::vector<std::vector<int>> batch_token_ids = tokenize_batch(batch_sequences);

    size_t current_max_len = 0;
    if (max_len_param > 0) {
        current_max_len = static_cast<size_t>(max_len_param);
    } else if (max_sequence_length_internal_ > 0) {
        current_max_len = static_cast<size_t>(max_sequence_length_internal_);
    } else if (pad_to_max_in_batch) {
        if (batch_token_ids.empty()) {
            current_max_len = 0; // Or a default small value like 1?
        } else {
            for (const auto& seq : batch_token_ids) {
                if (seq.size() > current_max_len) {
                    current_max_len = seq.size();
                }
            }
        }
    } else {
         // If no max length specified and not padding to max in batch, sequences must be same length or error.
         // For now, let's default to padding to max in batch if no other length is given.
         // If batch_token_ids is empty, current_max_len will be 0.
        if (!batch_token_ids.empty()) {
             for (const auto& seq : batch_token_ids) {
                if (seq.size() > current_max_len) {
                    current_max_len = seq.size();
                }
            }
        }
    }

    if (batch_sequences.empty()) {
        return Matrix::Matrix<float>(0, 0); // Return empty matrix for empty batch
    }
    // If current_max_len is still 0 (e.g. batch of empty strings, or single empty string)
    // we might want a minimum length of 1 for the matrix.
    if (current_max_len == 0 && !batch_sequences.empty()) {
        current_max_len = 1; // Ensure matrix has at least one column if there are sequences
    }


    Matrix::Matrix<float> output_matrix(batch_token_ids.size(), current_max_len);

    for (size_t i = 0; i < batch_token_ids.size(); ++i) {
        const std::vector<int>& seq_ids = batch_token_ids[i];
        for (size_t j = 0; j < current_max_len; ++j) {
            if (j < seq_ids.size()) {
                output_matrix[i][j] = static_cast<float>(seq_ids[j]);
            } else {
                output_matrix[i][j] = static_cast<float>(padding_token_id_internal);
            }
        }
    }
    return output_matrix;
}

} // namespace NeuroNet
