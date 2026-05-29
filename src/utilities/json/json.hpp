#pragma once
#ifndef _JSON_
#define _JSON_

/**
 * @file json.hpp
 * @brief Defines a simple JSON parser and data structures for representing JSON values.
 *
 * This file provides the `JsonValue` struct to represent different JSON types (null, boolean,
 * number, string, array, object) and the `JsonParser` class, a static utility class,
 * for parsing JSON strings into `JsonValue` objects.
 */

#include <string>
#include <unordered_map>
#include <vector>
#include <stdexcept> // For potential use in JsonValue, though JsonParseException is primary
#include <cmath>     // For std::to_string on numbers, potentially for number parsing aspects
#include <climits>   // For LLONG_MIN and LLONG_MAX
#include "json_exception.hpp" // For JsonParseException

/**
 * @enum JsonValueType
 * @brief Enumerates the possible types a JSON value can represent.
 */
enum class JsonValueType{
	Null,    ///< Represents a JSON null value.
	Boolean, ///< Represents a JSON boolean value (true or false).
	Number,  ///< Represents a JSON number (stored as double).
	String,  ///< Represents a JSON string.
	Array,   ///< Represents a JSON array (a sequence of JsonValue objects).
	Object   ///< Represents a JSON object (a collection of key-value pairs where keys are strings and values are JsonValue objects).
};

/**
 * @struct JsonValue
 * @brief Represents a JSON value, which can be of various types defined by JsonValueType.
 *
 * This struct uses a union-like approach with a type discriminator (`type`) to store
 * one of several possible JSON data types.
 */
struct JsonValue{

    /**
     * @brief Constructs a JsonValue.
     * @param type_ The initial type of the JsonValue. Defaults to JsonValueType::Null.
     */
	JsonValue(JsonValueType type_ = JsonValueType::Null) : type(type_){}

	JsonValueType type; ///< The actual type of this JSON value, determining which member field is valid.
	
	bool boolean_value;   ///< Valid and used if `type` is JsonValueType::Boolean. Stores the boolean state.
	double number_value;  ///< Valid and used if `type` is JsonValueType::Number. Stores the numeric value as a double.
	std::string string_value; ///< Valid and used if `type` is JsonValueType::String. Stores the string content.
	
    /**
     * @brief Valid and used if `type` is JsonValueType::Array. Stores JsonValue objects in sequence.
     * The `JsonValue` objects within the vector are owned by the vector (stack-allocated or moved).
     */
	std::vector<JsonValue> array_value; 
    
    /**
     * @brief Valid and used if `type` is JsonValueType::Object. Stores key-value pairs.
     * Keys are strings, and values are *pointers* to `JsonValue` objects.
     * @warning Users of this map are responsible for the memory management of the `JsonValue`
     * objects pointed to if they are dynamically allocated (e.g., using `new`). The map itself
     * does not automatically deallocate these pointers upon destruction of the containing `JsonValue`
     * or when elements are removed from the map.
     */
	std::unordered_map<std::string, JsonValue* > object_value;

    /**
     * @brief Serializes the JsonValue to its JSON string representation.
     * For Array and Object types, this method recursively calls ToString() on their elements.
     * @return A std::string containing the JSON representation of this value.
     */
	std::string ToString() const{
		switch (type)
		{
			case JsonValueType::Null:
				return "null";
			case JsonValueType::Boolean:
				return boolean_value ? "true":"false";
			case JsonValueType::Number:
				// Note: std::to_string for double might have precision issues or fixed notation.
                // For full JSON compliance, a more sophisticated number-to-string conversion might be needed.
                // Let's try to remove trailing zeros for a cleaner output for whole numbers
                if (std::fmod(number_value, 1.0) == 0.0) {
                    // Check if the number is too large for long long to avoid overflow
                    if (number_value >= static_cast<double>(LLONG_MIN) && number_value <= static_cast<double>(LLONG_MAX)) {
                        return std::to_string(static_cast<long long>(number_value));
                    }
                }
				return std::to_string(number_value);
			case JsonValueType::String:
				{
					std::string escaped_string = "\"";
					for (char c : string_value) {
						switch (c) {
							case '"':  escaped_string += "\\\""; break;
							case '\\': escaped_string += "\\\\"; break;
							case '\b': escaped_string += "\\b"; break;
							case '\f': escaped_string += "\\f"; break;
							case '\n': escaped_string += "\\n"; break;
							case '\r': escaped_string += "\\r"; break;
							case '\t': escaped_string += "\\t"; break;
							default:
								if (c >= 0 && c < 0x20) { // Control characters
									char buf[7];
									snprintf(buf, sizeof(buf), "\\u%04X", static_cast<unsigned char>(c)); // Cast to unsigned char for snprintf
									escaped_string += buf;
								} else {
									escaped_string += c;
								}
								break;
						}
					}
					escaped_string += "\"";
					return escaped_string;
				}
			case JsonValueType::Array:
				return ArrayToString(array_value);
			case JsonValueType::Object:
				return ObjectToString(object_value);
            default: // Should not happen with a valid JsonValueType
                return ""; 
		}
	}

    /**
     * @brief Sets this JsonValue to a boolean type and assigns the given value.
     * @param value The boolean value to set.
     */
	void SetBoolean(bool value){type = JsonValueType::Boolean; boolean_value = value;}
    /**
     * @brief Gets the boolean value.
     * @pre The JsonValue must be of type JsonValueType::Boolean. Behavior is undefined otherwise.
     * @return The boolean value.
     */
	bool GetBoolean() const {return boolean_value;}

    /**
     * @brief Sets this JsonValue to a number type and assigns the given value.
     * All numbers are stored internally as double.
     * @param value The double value to set.
     */
	void SetNumber(double value){type = JsonValueType::Number; number_value = value;}
    /**
     * @brief Gets the number value.
     * @pre The JsonValue must be of type JsonValueType::Number. Behavior is undefined otherwise.
     * @return The numeric value as a double.
     */
	double GetNumber() const {return number_value;}

    /**
     * @brief Sets this JsonValue to a string type and assigns the given value.
     * @param value The string value to set.
     */
	void SetString(const std::string& value){ type = JsonValueType::String; string_value = value;}
    /**
     * @brief Gets the string value (const version).
     * @pre The JsonValue must be of type JsonValueType::String. Behavior is undefined otherwise.
     * @return A const reference to the string value.
     */
	const std::string& GetString() const { return string_value;}
    /**
     * @brief Gets the string value (non-const version).
     * @pre The JsonValue must be of type JsonValueType::String. Behavior is undefined otherwise.
     * @return A non-const reference to the string value.
     */
	std::string& GetString() {return string_value;}

    /**
     * @brief Sets this JsonValue to an Array type. Initializes an empty array.
     */
	void SetArray(){type = JsonValueType::Array; array_value.clear(); } // Ensure it's clean
    /**
     * @brief Checks if this JsonValue is an Array.
     * @return True if the type is JsonValueType::Array, false otherwise.
     */
	bool IsArray() const { return type == JsonValueType::Array;}
    /**
     * @brief Gets the underlying std::vector<JsonValue> for an Array type (non-const version).
     * @pre The JsonValue must be of type JsonValueType::Array. Behavior is undefined otherwise.
     * @return A reference to the vector of JsonValues.
     */
	std::vector<JsonValue>& GetArray(){ return array_value;}
    /**
     * @brief Gets the underlying std::vector<JsonValue> for an Array type (const version).
     * @pre The JsonValue must be of type JsonValueType::Array. Behavior is undefined otherwise.
     * @return A const reference to the vector of JsonValues.
     */
	const std::vector<JsonValue>& GetArray() const{return array_value;}

    /**
     * @brief Sets this JsonValue to an Object type. Initializes an empty object.
     * @warning If this JsonValue previously held an object with dynamically allocated
     * `JsonValue*` members, those pointers are NOT deallocated by this call. Memory
     * management of previous object members is the caller's responsibility.
     */
	void SetObject(){type = JsonValueType::Object; object_value.clear(); } // Ensure it's clean, but doesn't delete pointed-to values
    /**
     * @brief Checks if this JsonValue is an Object.
     * @return True if the type is JsonValueType::Object, false otherwise.
     */
	bool IsObject() const { return type == JsonValueType::Object;}
    /**
     * @brief Gets the underlying std::unordered_map<std::string, JsonValue*> for an Object type (non-const version).
     * @pre The JsonValue must be of type JsonValueType::Object. Behavior is undefined otherwise.
     * @return A reference to the map.
     * @warning The map stores raw pointers (`JsonValue*`). Callers are responsible for memory management
     * of these pointed-to `JsonValue` objects if they were dynamically allocated.
     */
	std::unordered_map<std::string, JsonValue*>& GetObject(){return object_value;} // Typo fixed: GetObJect -> GetObject
    /**
     * @brief Gets the underlying std::unordered_map<std::string, JsonValue*> for an Object type (const version).
     * @pre The JsonValue must be of type JsonValueType::Object. Behavior is undefined otherwise.
     * @return A const reference to the map.
     * @warning The map stores raw pointers (`JsonValue*`). Callers are responsible for memory management
     * of these pointed-to `JsonValue` objects if they were dynamically allocated.
     */
	const std::unordered_map<std::string, JsonValue*> &GetObject() const {return object_value;}
    
    /**
     * @brief Inserts a key-value pair into this JsonValue, assuming it's an Object.
     * If a key already exists, its associated `JsonValue*` is overwritten.
     * @param key The string key for the member to insert.
     * @param value A pointer to the JsonValue to be associated with the key.
     * @throw JsonParseException if this JsonValue is not of type JsonValueType::Object.
     * @warning This JsonValue (the object) takes ownership of the key string, but NOT the
     * `JsonValue` pointed to by `value`. If `value` was dynamically allocated (e.g., `new JsonValue()`),
     * the caller is responsible for ensuring its lifetime is managed correctly. If the key
     * previously existed, the old `JsonValue*` is overwritten and NOT deallocated by this method,
     * potentially leading to memory leaks if not handled by the caller.
     */
	void InsertIntoObject(const std::string& key , JsonValue* value){
		if (type != JsonValueType::Object){
			throw JsonParseException("Cannot insert into a non-object value.");
		}
        // Potential memory leak: if object_value[key] previously held a pointer, it's overwritten.
		object_value[key] = value;
	}


 
	private:
	
    /**
     * @brief Helper function to convert a vector of JsonValues to its JSON string representation.
     * Called by ToString() when the JsonValue is an Array.
     * @param array The vector of JsonValues to serialize.
     * @return A std::string containing the JSON array representation.
     */
	std::string ArrayToString(const std::vector<JsonValue>& array) const {
		std::string result = "[";
		for(size_t i=0; i < array.size(); i++)
		{
			if(i >0){
				result += ", ";
			}
			result += array[i].ToString(); // Recursive call
		}
		result +="]";
		return result;
	}
	
    /**
     * @brief Helper function to convert a map of JsonValue pointers to its JSON string representation.
     * Called by ToString() when the JsonValue is an Object.
     * @param object The map of string keys to JsonValue pointers to serialize.
     * @return A std::string containing the JSON object representation.
     * @warning This method assumes that the `JsonValue*` in the map are valid pointers.
     */
	std::string ObjectToString(const std::unordered_map<std::string, JsonValue*>& object) const{
		std::string result ="{";
		bool first = true;
		for(const auto& kv: object)
		{
			if(!first){
				result += ", ";
			}
			
			first =false;
            // Assumes kv.second points to a valid JsonValue
			result +="\""+ kv.first + "\": " + (kv.second ? kv.second->ToString() : "null"); // Recursive call, with null check for safety
		}
		result +="}";
		return result;
	}
};

/**
 * @class JsonParser
 * @brief A static utility class for parsing JSON strings into JsonValue objects.
 *
 * This class provides a single public static method, `Parse()`, to initiate parsing.
 * All other methods are private helper functions used internally by the parsing logic.
 */
class JsonParser{
	public:
        /**
         * @brief Parses a JSON formatted string and constructs a JsonValue object tree.
         * This is the main entry point for parsing JSON data.
         * @param json_string The JSON string to parse.
         * @return A JsonValue object representing the root of the parsed JSON structure.
         * @throw JsonParseException if the input string is not valid JSON, contains syntax errors,
         *        or if any other parsing error occurs (e.g., unexpected end of input).
         */
		static JsonValue Parse(const std::string& json_string);
	private:
        /// @brief Parses a generic JSON value from the input string at the current index.
		static JsonValue ParseValue(const std::string& json_string, size_t& index);
        /// @brief Parses a JSON string literal (enclosed in double quotes) from the input. Handles basic escape sequences.
		static std::string ParseString(const std::string& json_string, size_t& index);
        /// @brief Parses a JSON number (integer or floating-point) from the input.
		static double ParseNumber(const std::string& json_string, size_t& index);
        /// @brief Parses a JSON array from the input string.
		static std::vector<JsonValue> ParseArray(const std::string& json_string, size_t& index);
        /// @brief Parses a JSON object from the input string.
        /// @warning Dynamically allocates JsonValue objects for the object's members. The caller (ultimately ParseValue for an object)
        /// is responsible for ensuring these are correctly managed by the returned JsonValue's object_value map.
		static std::unordered_map<std::string, JsonValue*> ParseObject(const std::string& json_string, size_t& index);
        /// @brief Parses a JSON boolean literal (true or false) from the input.
		static bool ParseBoolean(const std::string& json_string, size_t& index);
        /// @brief Parses a JSON null literal from the input.
		static void ParseNull(const std::string& json_string, size_t& index);
        /// @brief Skips whitespace characters (spaces, tabs, newlines, carriage returns) in the input string.
		static void SkipWhitespace(const std::string& json_string, size_t& index);
        /// @brief Expects and consumes a specific string from the input at the current index. Throws if not found.
		static void ExpectString(const std::string& json_string, size_t& index, const std::string& expected_string);
        /// @brief Expects and consumes a specific character from the input at the current index. Throws if not found.
		static void ExpectChar(const std::string& json_string, size_t& index, char expected_char);
        /// @brief Converts a Unicode code point (integer) to its UTF-8 string representation.
		static std::string UnicodeCodePointToUtf8(int code_point);
        /// @brief Checks if a character is a digit ('0'-'9').
		static bool IsDigit(char c);
        /// @brief Skips single-line (//) and multi-line (/* ... */) comments in the input string.
		static void SkipComment(const std::string& json_string, size_t& index);
};
#endif