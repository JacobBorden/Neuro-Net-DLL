# JSON Utilities

This document outlines the custom JSON processing utilities provided by the library, primarily focused on parsing and representing JSON data. These utilities are used internally, for example, in model serialization and deserialization.

## Overview

The JSON utilities consist of:
*   `JsonValueType` enum: Defines the types a JSON value can hold (Null, Boolean, Number, String, Array, Object).
*   `JsonValue` struct: A variant-like structure to represent any JSON value.
*   `JsonParser` class: A static utility class to parse JSON strings into `JsonValue` objects.
*   `JsonParseException`: An exception class for errors during parsing.

## `JsonValueType` Enum

Defines the kind of data a `JsonValue` instance holds.
```cpp
enum class JsonValueType {
    Null,
    Boolean,
    Number,  // Stored as double
    String,
    Array,   // std::vector<JsonValue>
    Object   // std::unordered_map<std::string, JsonValue*>
};
```

## `JsonValue` Struct

Represents a single JSON value.

### Key Members
*   `type`: A `JsonValueType` indicating the current type of data stored.
*   `boolean_value`: `bool`, used if `type` is `Boolean`.
*   `number_value`: `double`, used if `type` is `Number`.
*   `string_value`: `std::string`, used if `type` is `String`.
*   `array_value`: `std::vector<JsonValue>`, used if `type` is `Array`. Stores `JsonValue` objects directly.
*   `object_value`: `std::unordered_map<std::string, JsonValue*>`, used if `type` is `Object`.
    *   **Important Memory Note:** The map stores raw pointers (`JsonValue*`). The `JsonValue` struct itself (and the `JsonParser` when creating objects) **does not manage the memory of these pointed-to `JsonValue` objects if they are dynamically allocated.** This is a crucial consideration if manipulating `JsonValue` objects directly, especially `Object` types. For objects returned by `JsonParser::Parse`, the parser handles allocation, and the top-level `JsonValue` effectively owns the tree (though explicit recursive deletion would be needed if not using RAII for the top-level object).

### Key Methods
*   `JsonValue(JsonValueType type_ = JsonValueType::Null)`: Constructor.
*   `ToString() const`: Serializes the `JsonValue` and its children (for arrays/objects) into a JSON formatted string.
*   `SetBoolean(bool value)`, `GetBoolean() const`
*   `SetNumber(double value)`, `GetNumber() const`
*   `SetString(const std::string& value)`, `GetString() const` (and non-const overload)
*   `SetArray()`, `IsArray() const`, `GetArray()` (const and non-const overloads)
*   `SetObject()`, `IsObject() const`, `GetObject()` (const and non-const overloads)
    *   `SetObject()` clears the map but does not deallocate existing pointed-to values.
*   `InsertIntoObject(const std::string& key, JsonValue* value)`:
    *   Inserts a key-value pair into an object.
    *   Throws `JsonParseException` if the `JsonValue` is not an object.
    *   **Memory Warning:** Does not take ownership of the `value` pointer. Overwrites (and potentially leaks) previous pointer if key existed.

## `JsonParser` Class

A static utility class for parsing JSON strings.

### Key Static Method
*   **`static JsonValue Parse(const std::string& json_string)`:**
    *   The main entry point for parsing.
    *   Takes a JSON formatted string as input.
    *   Returns a `JsonValue` object representing the root of the parsed JSON structure.
    *   Internally, when parsing objects, the `JsonParser` dynamically allocates `JsonValue` instances for the object's members and stores pointers to them in the `object_value` map of the parent `JsonValue` (object). The lifetime of these dynamically allocated `JsonValue` members is tied to the lifetime of the root `JsonValue` returned by `Parse()` *if managed correctly by the caller*. Typically, the entire structure is used and then discarded, or specific parts are extracted.
    *   Throws `JsonParseException` for syntax errors, invalid JSON structure, or other parsing issues.

### Internal Helper Methods
The `JsonParser` uses several private static helper methods for parsing specific JSON constructs (values, strings, numbers, arrays, objects, booleans, nulls), skipping whitespace, and handling comments (single-line `//` and multi-line `/* ... */`).

## `JsonParseException` Class

A custom exception class derived from `std::exception`.
*   `JsonParseException(const std::string& message)`: Constructor.
*   `what() const noexcept override`: Returns the error message.

## Basic Usage Example: Parsing a JSON String

```cpp
#include "utilities/json/json.hpp" // Includes JsonValue, JsonParser
#include "utilities/json/json_exception.hpp" // For JsonParseException
#include <iostream>
#include <string>

void print_json_value(const JsonValue& val, int indent = 0) {
    std::string prefix(indent * 2, ' ');
    switch (val.type) {
        case JsonValueType::Null:
            std::cout << prefix << "Null" << std::endl;
            break;
        case JsonValueType::Boolean:
            std::cout << prefix << "Boolean: " << (val.GetBoolean() ? "true" : "false") << std::endl;
            break;
        case JsonValueType::Number:
            std::cout << prefix << "Number: " << val.GetNumber() << std::endl;
            break;
        case JsonValueType::String:
            std::cout << prefix << "String: \"" << val.GetString() << "\"" << std::endl;
            break;
        case JsonValueType::Array:
            std::cout << prefix << "Array: [" << std::endl;
            for (const auto& item : val.GetArray()) {
                print_json_value(item, indent + 1);
            }
            std::cout << prefix << "]" << std::endl;
            break;
        case JsonValueType::Object:
            std::cout << prefix << "Object: {" << std::endl;
            for (const auto& pair : val.GetObject()) {
                std::cout << prefix << "  \"" << pair.first << "\":" << std::endl;
                if (pair.second) { // Check if pointer is not null
                    print_json_value(*(pair.second), indent + 2);
                } else {
                    std::cout << prefix << "    Null (pointer)" << std::endl;
                }
            }
            std::cout << prefix << "}" << std::endl;
            break;
    }
}

// Simple recursive function to free memory for dynamically allocated JsonValues in an object
void free_json_object_memory(JsonValue& val) {
    if (val.IsObject()) {
        for (auto& pair : val.GetObject()) {
            if (pair.second) {
                free_json_object_memory(*(pair.second)); // Recurse for nested objects/arrays
                delete pair.second; // Delete the JsonValue pointed to
                pair.second = nullptr; // Optional: nullify pointer after deletion
            }
        }
        val.GetObject().clear(); // Clear the map
    } else if (val.IsArray()) {
        for (auto& item : val.GetArray()) {
            free_json_object_memory(item); // Recurse for items in array
        }
        val.GetArray().clear();
    }
}


int main() {
    std::string json_str = R"({
        "name": "NeuroNet Library",
        "version": 0.2,
        "is_beta": true,
        "features": ["Core NN", "Genetic Algorithm", "Transformers"],
        "details": {
            "author": "Jacob Borden",
            "license": null
        }
        // This is a comment
    })";

    try {
        JsonValue parsed_json = JsonParser::Parse(json_str);
        std::cout << "Parsed JSON:" << std::endl;
        print_json_value(parsed_json);

        std::cout << "\nSerialized JSON: " << parsed_json.ToString() << std::endl;

        // IMPORTANT: Clean up dynamically allocated memory for object members
        // The JsonValue returned by Parse is stack-allocated, but its 'object_value' map
        // contains pointers to heap-allocated JsonValues if it's an object.
        free_json_object_memory(parsed_json);

    } catch (const JsonParseException& e) {
        std::cerr << "JSON Parsing Error: " << e.what() << std::endl;
    } catch (const std::exception& e) {
        std::cerr << "Standard Exception: " << e.what() << std::endl;
    }

    return 0;
}
```

### Memory Management Considerations

*   When `JsonParser::Parse` creates a JSON object, the `JsonValue` instances for the members of that object are dynamically allocated (`new JsonValue()`). The `object_value` map within the parent `JsonValue` stores raw pointers to these members.
*   The `JsonValue` struct itself (including its `std::vector` for arrays and `std::unordered_map` for objects) does **not** automatically deallocate these raw pointers when it goes out of scope or when elements are removed/cleared.
*   **If you parse a JSON string and the root is an object or contains objects, you are responsible for recursively deleting the dynamically allocated `JsonValue` members to avoid memory leaks.** The provided `free_json_object_memory` function in the example is a basic illustration of how this might be done.
*   When constructing `JsonValue` objects manually (especially objects), if you dynamically allocate `JsonValue` instances to be pointed to by an object's map, you must manage their memory.

### Source
*   `src/utilities/json/json.hpp` (main definitions)
*   `src/utilities/json/json.cpp` (parser implementation)
*   `src/utilities/json/json_exception.hpp` (exception class)

(This provides a good overview. More specific details on parsing nuances or advanced construction could be added if needed.)
