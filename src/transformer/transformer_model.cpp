#include "transformer_model.h"
#include <iostream> // For debugging (optional)
#include <string>
#include <vector>
#include <stdexcept>
#include <fstream>
#include <iterator> // For std::istreambuf_iterator
#include <algorithm> // For std::min, std::copy_n if needed elsewhere, though not directly in errors yet

namespace NeuroNet {
namespace Transformer {

TransformerModel::TransformerModel(
    int vocab_size,
    int max_seq_len,
    int d_model,
    int num_encoder_layers,
    int num_heads,
    int d_ff,
    float MHA_dropout_rate,
    float FFN_dropout_rate,
    float layer_norm_epsilon)
    : vocab_size_(vocab_size),
      max_seq_len_(max_seq_len),
      d_model_(d_model),
      num_encoder_layers_(num_encoder_layers),
      num_heads_(num_heads),
      d_ff_(d_ff),
      MHA_dropout_rate_(MHA_dropout_rate),
      FFN_dropout_rate_(FFN_dropout_rate),
      layer_norm_epsilon_(layer_norm_epsilon),
      embedding_layer_(vocab_size, d_model), // EmbeddingLayer constructor
      positional_encoding_(max_seq_len, d_model) // PositionalEncoding constructor
{
    if (vocab_size <= 0 || max_seq_len <= 0 || d_model <= 0 || num_encoder_layers < 0 || num_heads <= 0 || d_ff <= 0) {
        throw std::invalid_argument("Invalid parameters for TransformerModel constructor. Dimensions must be positive, num_encoder_layers non-negative."); // Fixed std::invalid_argument
    }
    if (d_model % num_heads != 0) {
        // This check is also in MHA, but good to have at model level too.
        throw std::invalid_argument("d_model must be divisible by num_heads for TransformerModel."); // Fixed std::invalid_argument
    }

    encoder_layers_.reserve(num_encoder_layers_);
    for (int i = 0; i < num_encoder_layers_; ++i) {
        encoder_layers_.emplace_back(
            d_model,
            num_heads,
            d_ff,
            MHA_dropout_rate,
            FFN_dropout_rate,
            layer_norm_epsilon
        );
    }
}

Matrix::Matrix<float> TransformerModel::forward(
    const Matrix::Matrix<float>& input_token_ids,
    const Matrix::Matrix<float>& attention_mask) {

    // Validate input_token_ids: should be (1, seq_len)
    if (input_token_ids.rows() != 1) {
        throw std::invalid_argument("TransformerModel::forward expects input_token_ids to have exactly 1 row (a single sequence).");
    }
    size_t seq_len = input_token_ids.cols();
    if (seq_len == 0) { // Handle empty sequence
        return Matrix::Matrix<float>(0, d_model_);
    }
    if (seq_len > static_cast<size_t>(max_seq_len_)) {
         throw std::invalid_argument("Input sequence length (" + std::to_string(seq_len) + // Fixed std::to_string
                                    ") exceeds TransformerModel's max_seq_len (" +
                                    std::to_string(max_seq_len_) + ")."); // Fixed std::to_string
    }

    // 1. Embedding
    // input_token_ids: (1, seq_len) -> embeddings: (seq_len, d_model)
    Matrix::Matrix<float> embeddings = embedding_layer_.forward(input_token_ids);

    // 2. Positional Encoding
    // embeddings: (seq_len, d_model) -> pos_embeddings: (seq_len, d_model)
    Matrix::Matrix<float> pos_embeddings = positional_encoding_.forward(embeddings);

    // Dropout on pos_embeddings (not implemented)

    // 3. Pass through Encoder Layers
    Matrix::Matrix<float> current_sequence_output = pos_embeddings;
    for (int i = 0; i < num_encoder_layers_; ++i) {
        current_sequence_output = encoder_layers_[i].forward(current_sequence_output, attention_mask);
    }

    // 4. Final Layer Normalization (applied to the output of the last encoder layer)
    // This is a common practice.
    Matrix::Matrix<float> final_norm_output = MathUtils::layer_norm(current_sequence_output, layer_norm_epsilon_);

    return final_norm_output;
}

// --- Serialization methods (save_model, load_model, to_json_string) ---
// To be implemented later.

// #include <iomanip> // For std::setprecision when writing floats (optional) - REMOVED due to compile issues

// Helper function to serialize a Matrix::Matrix<float> to a JsonValue object
// This object will contain "rows", "cols", and "data" (array of floats)
static JsonValue serialize_matrix_to_json(const Matrix::Matrix<float>& matrix) { // Assuming Matrix::Matrix is already fully qualified or in global/NeuroNet scope
    JsonValue matrix_json;
    matrix_json.SetObject();

    JsonValue* rows_val = new JsonValue(); rows_val->SetNumber(static_cast<double>(matrix.rows()));
    matrix_json.InsertIntoObject("rows", rows_val);

    JsonValue* cols_val = new JsonValue(); cols_val->SetNumber(static_cast<double>(matrix.cols()));
    matrix_json.InsertIntoObject("cols", cols_val);

    JsonValue* data_array_val = new JsonValue(); data_array_val->SetArray();
    if (matrix.rows() > 0 && matrix.cols() > 0) { // Only add data if matrix is not empty
        for (size_t r = 0; r < matrix.rows(); ++r) {
            for (size_t c = 0; c < matrix.cols(); ++c) {
                JsonValue val; val.SetNumber(static_cast<double>(matrix[r][c]));
                data_array_val->GetArray().push_back(val); // Pushes a copy
            }
        }
    }
    matrix_json.InsertIntoObject("data", data_array_val);
    return matrix_json; // Returns a copy
}

// Helper function to deserialize a Matrix::Matrix<float> from a JsonValue object
static Matrix::Matrix<float> deserialize_matrix_from_json(const JsonValue* matrix_json_val_ptr) { // Assuming Matrix::Matrix is already fully qualified
    if (!matrix_json_val_ptr || matrix_json_val_ptr->type != JsonValueType::Object) {
        throw std::runtime_error("Invalid JSON format for matrix: not an object.");
    }
    const auto& matrix_obj = matrix_json_val_ptr->GetObject();

    if (matrix_obj.find("rows") == matrix_obj.end() || matrix_obj.at("rows")->type != JsonValueType::Number ||
        matrix_obj.find("cols") == matrix_obj.end() || matrix_obj.at("cols")->type != JsonValueType::Number ||
        matrix_obj.find("data") == matrix_obj.end() || matrix_obj.at("data")->type != JsonValueType::Array) {
        throw std::runtime_error("Invalid JSON format for matrix: missing rows, cols, or data array.");
    }

    int rows = static_cast<int>(matrix_obj.at("rows")->GetNumber());
    int cols = static_cast<int>(matrix_obj.at("cols")->GetNumber());
    const std::vector<JsonValue>& data_array = matrix_obj.at("data")->GetArray();

    if (rows < 0 || cols < 0) {
         throw std::runtime_error("Matrix dimensions (rows, cols) cannot be negative.");
    }
    if (static_cast<size_t>(rows * cols) != data_array.size() && (rows > 0 && cols > 0)) {
         // Allow empty data array if rows or cols is 0
        throw std::runtime_error("Matrix data size mismatch. Expected " + std::to_string(rows * cols) +
                                 " elements, got " + std::to_string(data_array.size()));
    }

    Matrix::Matrix<float> matrix(rows, cols); // Assuming Matrix::Matrix is already fully qualified
    if (rows > 0 && cols > 0) {
        for (int r = 0; r < rows; ++r) {
            for (int c = 0; c < cols; ++c) {
                size_t flat_idx = r * cols + c;
                if (data_array[flat_idx].type != JsonValueType::Number) {
                    throw std::runtime_error("Non-numeric value in matrix data array.");
                }
                matrix[r][c] = static_cast<float>(data_array[flat_idx].GetNumber());
            }
        }
    }
    return matrix;
}


// Manual cleanup for JsonValue objects created by serialize_matrix_to_json
// This is needed because JsonValue::InsertIntoObject takes ownership of the pointer,
// but the returned JsonValue from serialize_matrix_to_json is a copy, so its internal
// pointers would leak if not managed.
// A better JsonValue would handle this with RAII or shared_ptr.
static void cleanup_serialized_matrix_json(JsonValue& matrix_json) {
    if (matrix_json.type == JsonValueType::Object) {
        auto& obj = matrix_json.GetObject();
        if (obj.count("rows")) { delete obj["rows"]; obj.erase("rows"); }
        if (obj.count("cols")) { delete obj["cols"]; obj.erase("cols"); }
        if (obj.count("data")) { delete obj["data"]; obj.erase("data"); } // Data array's elements are copies, not ptrs
    }
}


bool TransformerModel::save_model(const std::string& filename) const { // Fixed std::string
    JsonValue root;
    root.SetObject();

    // Save hyperparameters
    JsonValue* vs_val = new JsonValue(); vs_val->SetNumber(vocab_size_); root.InsertIntoObject("vocab_size", vs_val);
    JsonValue* msl_val = new JsonValue(); msl_val->SetNumber(max_seq_len_); root.InsertIntoObject("max_seq_len", msl_val);
    JsonValue* dm_val = new JsonValue(); dm_val->SetNumber(d_model_); root.InsertIntoObject("d_model", dm_val);
    JsonValue* nel_val = new JsonValue(); nel_val->SetNumber(num_encoder_layers_); root.InsertIntoObject("num_encoder_layers", nel_val);
    JsonValue* nh_val = new JsonValue(); nh_val->SetNumber(num_heads_); root.InsertIntoObject("num_heads", nh_val);
    JsonValue* dff_val = new JsonValue(); dff_val->SetNumber(d_ff_); root.InsertIntoObject("d_ff", dff_val);
    JsonValue* mha_do_val = new JsonValue(); mha_do_val->SetNumber(MHA_dropout_rate_); root.InsertIntoObject("MHA_dropout_rate", mha_do_val);
    JsonValue* ffn_do_val = new JsonValue(); ffn_do_val->SetNumber(FFN_dropout_rate_); root.InsertIntoObject("FFN_dropout_rate", ffn_do_val);
    JsonValue* lne_val = new JsonValue(); lne_val->SetNumber(layer_norm_epsilon_); root.InsertIntoObject("layer_norm_epsilon", lne_val);

    // Save EmbeddingLayer weights
    // Need to use a JsonValue* for the object that serialize_matrix_to_json returns, then cleanup.
    JsonValue embedding_weights_json_obj = serialize_matrix_to_json(embedding_layer_.get_weights());
    JsonValue* embedding_weights_json_ptr = new JsonValue(embedding_weights_json_obj); // Copy constructor
    root.InsertIntoObject("embedding_weights", embedding_weights_json_ptr);
    // No need to call cleanup_serialized_matrix_json on embedding_weights_json_obj as its members were copied.
    // The pointers within embedding_weights_json_ptr will be cleaned up at the end.


    // Save EncoderLayers weights
    JsonValue* encoder_layers_array_val = new JsonValue();
    encoder_layers_array_val->SetArray();
    for (const auto& layer : encoder_layers_) {
        JsonValue encoder_layer_json; // This will be an object for one layer
        encoder_layer_json.SetObject();

        // MHA weights
        JsonValue mha_wq_json = serialize_matrix_to_json(layer.get_multi_head_attention_module().get_wq());
        JsonValue* mha_wq_ptr = new JsonValue(mha_wq_json);
        encoder_layer_json.InsertIntoObject("mha_Wq", mha_wq_ptr);

        JsonValue mha_wk_json = serialize_matrix_to_json(layer.get_multi_head_attention_module().get_wk());
        JsonValue* mha_wk_ptr = new JsonValue(mha_wk_json);
        encoder_layer_json.InsertIntoObject("mha_Wk", mha_wk_ptr);

        JsonValue mha_wv_json = serialize_matrix_to_json(layer.get_multi_head_attention_module().get_wv());
        JsonValue* mha_wv_ptr = new JsonValue(mha_wv_json);
        encoder_layer_json.InsertIntoObject("mha_Wv", mha_wv_ptr);

        JsonValue mha_wo_json = serialize_matrix_to_json(layer.get_multi_head_attention_module().get_wo());
        JsonValue* mha_wo_ptr = new JsonValue(mha_wo_json);
        encoder_layer_json.InsertIntoObject("mha_Wo", mha_wo_ptr);

        // FFN weights
        JsonValue ffn_w1_json = serialize_matrix_to_json(layer.get_ffn_module().get_W1());
        JsonValue* ffn_w1_ptr = new JsonValue(ffn_w1_json);
        encoder_layer_json.InsertIntoObject("ffn_W1", ffn_w1_ptr);

        JsonValue ffn_b1_json = serialize_matrix_to_json(layer.get_ffn_module().get_b1());
        JsonValue* ffn_b1_ptr = new JsonValue(ffn_b1_json);
        encoder_layer_json.InsertIntoObject("ffn_b1", ffn_b1_ptr);

        JsonValue ffn_w2_json = serialize_matrix_to_json(layer.get_ffn_module().get_W2());
        JsonValue* ffn_w2_ptr = new JsonValue(ffn_w2_json);
        encoder_layer_json.InsertIntoObject("ffn_W2", ffn_w2_ptr);

        JsonValue ffn_b2_json = serialize_matrix_to_json(layer.get_ffn_module().get_b2());
        JsonValue* ffn_b2_ptr = new JsonValue(ffn_b2_json);
        encoder_layer_json.InsertIntoObject("ffn_b2", ffn_b2_ptr);

        encoder_layers_array_val->GetArray().push_back(encoder_layer_json); // Pushes a copy
    }
    root.InsertIntoObject("encoder_layers_weights", encoder_layers_array_val);

    // Write to file
    std::ofstream ofs(filename); // Fixed std::ofstream
    if (!ofs.is_open()) {
        // Cleanup allocated JsonValues before returning
        for (auto& pair : root.GetObject()) {
            if (pair.first == "encoder_layers_weights") {
                JsonValue* layers_array = pair.second;
                for (JsonValue& layer_val : layers_array->GetArray()) {
                    for (auto& layer_prop_pair : layer_val.GetObject()) {
                        cleanup_serialized_matrix_json(*layer_prop_pair.second); // Cleanup matrix object
                        delete layer_prop_pair.second; // Delete the JsonValue* itself
                    }
                }
            } else if (pair.first == "embedding_weights") {
                 cleanup_serialized_matrix_json(*pair.second);
            }
            delete pair.second;
        }
        root.GetObject().clear();
        return false;
    }
    ofs << root.ToString();
    ofs.close();

    // Cleanup allocated JsonValues
    // This is tricky with the custom library. The JsonValue objects pointed to by the map in 'root'
    // and nested objects/arrays need their own pointed-to members deleted if they were also objects/arrays.
    // The serialize_matrix_to_json creates JsonValue that owns its internal pointers.
    // When we do `new JsonValue(mha_wq_json)`, the new JsonValue copies mha_wq_json.
    // The map in `root` and `encoder_layer_json` now store these `new JsonValue*`.
    for (auto& pair : root.GetObject()) { // Top-level properties of root
        if (pair.first == "encoder_layers_weights") {
            JsonValue* layers_array = pair.second; // This is the JsonValue* for the array itself
            for (JsonValue& layer_val_obj : layers_array->GetArray()) { // layer_val_obj is a copy of an object from the array
                for (auto& layer_prop_pair : layer_val_obj.GetObject()) { // layer_prop_pair.second is JsonValue* for a matrix
                    cleanup_serialized_matrix_json(*layer_prop_pair.second); // Cleanup matrix object's internal JsonValue*s
                    delete layer_prop_pair.second; // Delete the JsonValue* for the matrix object itself
                }
                // layer_val_obj.GetObject().clear(); // Not strictly needed as layer_val_obj is a copy
            }
        } else if (pair.first == "embedding_weights") {
             cleanup_serialized_matrix_json(*pair.second); // Cleanup matrix object's internals
        }
        delete pair.second; // Delete the top-level JsonValue* (e.g., for "vocab_size", "embedding_weights" object, "encoder_layers_weights" array)
    }
    root.GetObject().clear(); // Clear the map in root

    return true;
}


TransformerModel TransformerModel::load_model(const std::string& filename) { // Fixed std::string
    std::ifstream ifs(filename); // Fixed std::ifstream
    if (!ifs.is_open()) {
        throw std::runtime_error("Failed to open model file: " + filename); // Fixed std::runtime_error, std::string
    }
    std::string content((std::istreambuf_iterator<char>(ifs)), std::istreambuf_iterator<char>()); // Fixed std::string, std::istreambuf_iterator
    ifs.close();

    JsonValue root_json_val;
    try {
        root_json_val = JsonParser::Parse(content);
    } catch (const JsonParseException& e) {
        throw std::runtime_error("Failed to parse JSON from model file: " + filename + "\nError: " + e.what()); // Fixed std::runtime_error, std::string
    }

    if (root_json_val.type != JsonValueType::Object) {
        throw std::runtime_error("Model JSON root is not an object."); // Fixed std::runtime_error
    }
    const auto& root_obj = root_json_val.GetObject();

    // Helper to get a number or throw
    auto get_num = [&](const std::string& key) {
        if (root_obj.find(key) == root_obj.end() || root_obj.at(key)->type != JsonValueType::Number)
            throw std::runtime_error("Missing or invalid hyperparameter in JSON: " + key);
        return root_obj.at(key)->GetNumber();
    };

    int vocab_size = static_cast<int>(get_num("vocab_size"));
    int max_seq_len = static_cast<int>(get_num("max_seq_len"));
    int d_model = static_cast<int>(get_num("d_model"));
    int num_encoder_layers = static_cast<int>(get_num("num_encoder_layers"));
    int num_heads = static_cast<int>(get_num("num_heads"));
    int d_ff = static_cast<int>(get_num("d_ff"));
    float mha_dropout_rate = static_cast<float>(get_num("MHA_dropout_rate"));
    float ffn_dropout_rate = static_cast<float>(get_num("FFN_dropout_rate"));
    float layer_norm_epsilon = static_cast<float>(get_num("layer_norm_epsilon"));

    TransformerModel model(vocab_size, max_seq_len, d_model, num_encoder_layers, num_heads, d_ff,
                           mha_dropout_rate, ffn_dropout_rate, layer_norm_epsilon);

    // Load EmbeddingLayer weights
    if (root_obj.find("embedding_weights") == root_obj.end()) throw std::runtime_error("Missing 'embedding_weights' in JSON.");
    model.embedding_layer_.set_weights(deserialize_matrix_from_json(root_obj.at("embedding_weights")));

    // Load EncoderLayers weights
    if (root_obj.find("encoder_layers_weights") == root_obj.end() || root_obj.at("encoder_layers_weights")->type != JsonValueType::Array) {
        throw std::runtime_error("Missing or invalid 'encoder_layers_weights' array in JSON.");
    }
    const auto& layers_array_json = root_obj.at("encoder_layers_weights")->GetArray();
    if (layers_array_json.size() != static_cast<size_t>(num_encoder_layers)) {
        throw std::runtime_error("Mismatch in number of encoder layers in JSON and model constructor.");
    }

    for (int i = 0; i < num_encoder_layers; ++i) {
        const JsonValue& layer_json_val = layers_array_json[i];
        if (layer_json_val.type != JsonValueType::Object) throw std::runtime_error("Encoder layer JSON is not an object for layer " + std::to_string(i));
        const auto& layer_obj = layer_json_val.GetObject();

        auto load_sub_matrix = [&](const std::string& key) {
            if (layer_obj.find(key) == layer_obj.end()) throw std::runtime_error("Missing matrix '" + key + "' in encoder layer " + std::to_string(i));
            return deserialize_matrix_from_json(layer_obj.at(key));
        };

        model.encoder_layers_[i].get_multi_head_attention_module().set_wq(load_sub_matrix("mha_Wq"));
        model.encoder_layers_[i].get_multi_head_attention_module().set_wk(load_sub_matrix("mha_Wk"));
        model.encoder_layers_[i].get_multi_head_attention_module().set_wv(load_sub_matrix("mha_Wv"));
        model.encoder_layers_[i].get_multi_head_attention_module().set_wo(load_sub_matrix("mha_Wo"));

        model.encoder_layers_[i].get_ffn_module().set_W1(load_sub_matrix("ffn_W1"));
        model.encoder_layers_[i].get_ffn_module().set_b1(load_sub_matrix("ffn_b1"));
        model.encoder_layers_[i].get_ffn_module().set_W2(load_sub_matrix("ffn_W2"));
        model.encoder_layers_[i].get_ffn_module().set_b2(load_sub_matrix("ffn_b2"));
    }

    // Cleanup for JsonParser::Parse result (root_json_val)
    // Similar to NeuroNet::load_model cleanup for its custom Json library
    if (root_json_val.type == JsonValueType::Object) {
        for (auto& pair : root_obj) { // pair.first is string, pair.second is JsonValue*
            if (pair.second->type == JsonValueType::Object) {
                for (auto& inner_pair : pair.second->GetObject()) delete inner_pair.second; // For matrix objects
                pair.second->GetObject().clear();
            } else if (pair.second->type == JsonValueType::Array) {
                 for (JsonValue& array_item_val : pair.second->GetArray()) { // array_item_val is a copy
                    if (array_item_val.type == JsonValueType::Object) { // This is for encoder_layers_weights
                        for (auto& el_pair : array_item_val.GetObject()) delete el_pair.second; // Delete matrix JsonValue*
                        // array_item_val.GetObject().clear(); // Not needed as array_item_val is a copy
                    }
                 }
                 // pair.second->GetArray().clear(); // Not needed
            }
            delete pair.second; // Delete the JsonValue* itself
        }
        // root_json_val.GetObject().clear(); // The map in root_json_val will be cleared when it goes out of scope
                                           // but the pointers it holds need to be deleted.
    }


    return model;
}

} // namespace Transformer
} // namespace NeuroNet
