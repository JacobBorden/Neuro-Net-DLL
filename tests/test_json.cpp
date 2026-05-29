#include "gtest/gtest.h"
#include "../src/utilities/json/json.hpp"
#include "../src/utilities/json/json_exception.hpp"
#include <string>
#include <vector>
#include <stdexcept> // For std::runtime_error (if testing exceptions)

// Basic Test Fixture (optional, but good practice if common setup/teardown is needed later)
class JsonLibTest : public ::testing::Test {
protected:
    void SetUp() override {
        // Common setup for tests, if any
    }

    void TearDown() override {
        // Common teardown for tests, if any
    }
};

// Placeholder for the first test
TEST_F(JsonLibTest, InitialTest) {
    // Basic test to ensure parsing a simple valid JSON doesn't throw.
    EXPECT_NO_THROW({
        JsonValue val = JsonParser::Parse("null");
        ASSERT_EQ(val.type, JsonValueType::Null);
    });
    SUCCEED(); // Indicates the test setup is working, and basic parsing works
}

// Example of how to start adding actual tests (will be fleshed out in next steps)
TEST_F(JsonLibTest, ParsingNull) {
    try {
        JsonValue root = JsonParser::Parse("null");
        ASSERT_EQ(root.type, JsonValueType::Null);
    } catch (const JsonParseException& e) {
        FAIL() << "Parsing null threw an exception: " << e.what();
    }
}

// --- Basic JSON Type Parsing Tests ---

TEST_F(JsonLibTest, ParseStringValues) {
    // Simple string
    const std::string json_simple_string = "\"hello world\"";
    JsonValue root_simple = JsonParser::Parse(json_simple_string);
    ASSERT_EQ(root_simple.type, JsonValueType::String);
    EXPECT_EQ(root_simple.GetString(), "hello world");

    // Empty string
    const std::string json_empty_string = "\"\"";
    JsonValue root_empty = JsonParser::Parse(json_empty_string);
    ASSERT_EQ(root_empty.type, JsonValueType::String);
    EXPECT_EQ(root_empty.GetString(), "");

    // String with escapes
    // The custom parser needs to correctly handle standard JSON escapes.
    // Assuming ParseString correctly unescapes:
    const std::string json_escaped_string = "\"line1\\nline2\\t\\\"quoted\\\"\"";
    JsonValue root_escaped = JsonParser::Parse(json_escaped_string);
    ASSERT_EQ(root_escaped.type, JsonValueType::String);
    EXPECT_EQ(root_escaped.GetString(), "line1\nline2\t\"quoted\"");
}

TEST_F(JsonLibTest, ParseNumericValues) {
    // Integer
    JsonValue root_int = JsonParser::Parse("123");
    ASSERT_EQ(root_int.type, JsonValueType::Number);
    EXPECT_DOUBLE_EQ(root_int.GetNumber(), 123.0);

    // Negative Integer
    JsonValue root_neg_int = JsonParser::Parse("-45");
    ASSERT_EQ(root_neg_int.type, JsonValueType::Number);
    EXPECT_DOUBLE_EQ(root_neg_int.GetNumber(), -45.0);
    
    // Zero
    JsonValue root_zero = JsonParser::Parse("0");
    ASSERT_EQ(root_zero.type, JsonValueType::Number);
    EXPECT_DOUBLE_EQ(root_zero.GetNumber(), 0.0);

    // Floating-point
    JsonValue root_float = JsonParser::Parse("3.141");
    ASSERT_EQ(root_float.type, JsonValueType::Number);
    EXPECT_DOUBLE_EQ(root_float.GetNumber(), 3.141);

    // Negative Floating-point
    JsonValue root_neg_float = JsonParser::Parse("-0.001");
    ASSERT_EQ(root_neg_float.type, JsonValueType::Number);
    EXPECT_DOUBLE_EQ(root_neg_float.GetNumber(), -0.001);

    // Scientific notation
    JsonValue root_sci_pos = JsonParser::Parse("1.2e5");
    ASSERT_EQ(root_sci_pos.type, JsonValueType::Number);
    EXPECT_DOUBLE_EQ(root_sci_pos.GetNumber(), 120000.0);

    JsonValue root_sci_neg = JsonParser::Parse("1.23e-2");
    ASSERT_EQ(root_sci_neg.type, JsonValueType::Number);
    EXPECT_DOUBLE_EQ(root_sci_neg.GetNumber(), 0.0123);
}

TEST_F(JsonLibTest, ParseBooleanValues) {
    JsonValue root_true = JsonParser::Parse("true");
    ASSERT_EQ(root_true.type, JsonValueType::Boolean);
    EXPECT_EQ(root_true.GetBoolean(), true);
    EXPECT_TRUE(root_true.GetBoolean());

    JsonValue root_false = JsonParser::Parse("false");
    ASSERT_EQ(root_false.type, JsonValueType::Boolean);
    EXPECT_EQ(root_false.GetBoolean(), false);
    EXPECT_FALSE(root_false.GetBoolean());
}

// --- JSON Array Parsing Tests ---

TEST_F(JsonLibTest, ParseEmptyArray) {
    const std::string json_array = "[]";
    JsonValue root = JsonParser::Parse(json_array);
    ASSERT_EQ(root.type, JsonValueType::Array);
    EXPECT_EQ(root.GetArray().size(), 0);
}

TEST_F(JsonLibTest, ParseArrayOfNumbers) {
    const std::string json_array = "[1, 2, 3, -50, 0]";
    JsonValue root = JsonParser::Parse(json_array);
    ASSERT_EQ(root.type, JsonValueType::Array);
    ASSERT_EQ(root.GetArray().size(), 5);
    EXPECT_DOUBLE_EQ(root.GetArray()[0].GetNumber(), 1.0);
    EXPECT_DOUBLE_EQ(root.GetArray()[1].GetNumber(), 2.0);
    EXPECT_DOUBLE_EQ(root.GetArray()[2].GetNumber(), 3.0);
    EXPECT_DOUBLE_EQ(root.GetArray()[3].GetNumber(), -50.0);
    EXPECT_DOUBLE_EQ(root.GetArray()[4].GetNumber(), 0.0);
}

TEST_F(JsonLibTest, ParseArrayOfStrings) {
    const std::string json_array = "[\"a\", \"b\", \"c\", \"hello world\", \"\"]";
    JsonValue root = JsonParser::Parse(json_array);
    ASSERT_EQ(root.type, JsonValueType::Array);
    ASSERT_EQ(root.GetArray().size(), 5);
    EXPECT_EQ(root.GetArray()[0].GetString(), "a");
    EXPECT_EQ(root.GetArray()[1].GetString(), "b");
    EXPECT_EQ(root.GetArray()[2].GetString(), "c");
    EXPECT_EQ(root.GetArray()[3].GetString(), "hello world");
    EXPECT_EQ(root.GetArray()[4].GetString(), "");
}

TEST_F(JsonLibTest, ParseArrayOfMixedTypes) {
    const std::string json_array = "[1, \"hello\", true, null, 3.14]";
    JsonValue root = JsonParser::Parse(json_array);
    ASSERT_EQ(root.type, JsonValueType::Array);
    ASSERT_EQ(root.GetArray().size(), 5);

    EXPECT_EQ(root.GetArray()[0].type, JsonValueType::Number);
    EXPECT_DOUBLE_EQ(root.GetArray()[0].GetNumber(), 1.0);

    EXPECT_EQ(root.GetArray()[1].type, JsonValueType::String);
    EXPECT_EQ(root.GetArray()[1].GetString(), "hello");

    EXPECT_EQ(root.GetArray()[2].type, JsonValueType::Boolean);
    EXPECT_EQ(root.GetArray()[2].GetBoolean(), true);

    EXPECT_EQ(root.GetArray()[3].type, JsonValueType::Null);

    EXPECT_EQ(root.GetArray()[4].type, JsonValueType::Number);
    EXPECT_DOUBLE_EQ(root.GetArray()[4].GetNumber(), 3.14);
}

TEST_F(JsonLibTest, ParseArrayWithNestedArrays) {
    const std::string json_array = "[[1, 2], [3, 4], []]";
    JsonValue root = JsonParser::Parse(json_array);
    ASSERT_EQ(root.type, JsonValueType::Array);
    ASSERT_EQ(root.GetArray().size(), 3);

    const JsonValue& first_nested_array = root.GetArray()[0];
    ASSERT_EQ(first_nested_array.type, JsonValueType::Array);
    ASSERT_EQ(first_nested_array.GetArray().size(), 2);
    EXPECT_EQ(first_nested_array.GetArray()[0].type, JsonValueType::Number);
    EXPECT_DOUBLE_EQ(first_nested_array.GetArray()[0].GetNumber(), 1.0);
    EXPECT_EQ(first_nested_array.GetArray()[1].type, JsonValueType::Number);
    EXPECT_DOUBLE_EQ(first_nested_array.GetArray()[1].GetNumber(), 2.0);

    const JsonValue& second_nested_array = root.GetArray()[1];
    ASSERT_EQ(second_nested_array.type, JsonValueType::Array);
    ASSERT_EQ(second_nested_array.GetArray().size(), 2);
    EXPECT_EQ(second_nested_array.GetArray()[0].type, JsonValueType::Number);
    EXPECT_DOUBLE_EQ(second_nested_array.GetArray()[0].GetNumber(), 3.0);
    EXPECT_EQ(second_nested_array.GetArray()[1].type, JsonValueType::Number);
    EXPECT_DOUBLE_EQ(second_nested_array.GetArray()[1].GetNumber(), 4.0);
    
    const JsonValue& third_nested_array = root.GetArray()[2];
    ASSERT_EQ(third_nested_array.type, JsonValueType::Array);
    EXPECT_EQ(third_nested_array.GetArray().size(), 0);
}

TEST_F(JsonLibTest, ParseArrayWithNestedObjects) {
    const std::string json_array = "[{\"key1\": \"value1\"}, {\"key2\": 123, \"key3\": true}]";
    JsonValue root = JsonParser::Parse(json_array);
    ASSERT_EQ(root.type, JsonValueType::Array);
    ASSERT_EQ(root.GetArray().size(), 2);

    const JsonValue& first_nested_object_val = root.GetArray()[0];
    ASSERT_EQ(first_nested_object_val.type, JsonValueType::Object);
    ASSERT_GT(first_nested_object_val.GetObject().count("key1"), 0);
    EXPECT_EQ(first_nested_object_val.GetObject().at("key1")->type, JsonValueType::String);
    EXPECT_EQ(first_nested_object_val.GetObject().at("key1")->GetString(), "value1");

    const JsonValue& second_nested_object_val = root.GetArray()[1];
    ASSERT_EQ(second_nested_object_val.type, JsonValueType::Object);
    ASSERT_GT(second_nested_object_val.GetObject().count("key2"), 0);
    EXPECT_EQ(second_nested_object_val.GetObject().at("key2")->type, JsonValueType::Number);
    EXPECT_DOUBLE_EQ(second_nested_object_val.GetObject().at("key2")->GetNumber(), 123.0);
    ASSERT_GT(second_nested_object_val.GetObject().count("key3"), 0);
    EXPECT_EQ(second_nested_object_val.GetObject().at("key3")->type, JsonValueType::Boolean);
    EXPECT_EQ(second_nested_object_val.GetObject().at("key3")->GetBoolean(), true);
}

// --- JSON Object Parsing Tests ---

TEST_F(JsonLibTest, ParseEmptyObject) {
    const std::string json_object = "{}";
    JsonValue root = JsonParser::Parse(json_object);
    ASSERT_EQ(root.type, JsonValueType::Object);
    EXPECT_EQ(root.GetObject().size(), 0);
}

TEST_F(JsonLibTest, ParseObjectSimpleKeyValuePairs) {
    const std::string json_object = "{\"name\": \"John Doe\", \"age\": 30, \"isStudent\": false, \"car\": null, \"score\": 95.5}";
    JsonValue root = JsonParser::Parse(json_object);
    ASSERT_EQ(root.type, JsonValueType::Object);
    
    ASSERT_GT(root.GetObject().count("name"), 0);
    EXPECT_EQ(root.GetObject().at("name")->type, JsonValueType::String);
    EXPECT_EQ(root.GetObject().at("name")->GetString(), "John Doe");

    ASSERT_GT(root.GetObject().count("age"), 0);
    EXPECT_EQ(root.GetObject().at("age")->type, JsonValueType::Number);
    EXPECT_DOUBLE_EQ(root.GetObject().at("age")->GetNumber(), 30.0);

    ASSERT_GT(root.GetObject().count("isStudent"), 0);
    EXPECT_EQ(root.GetObject().at("isStudent")->type, JsonValueType::Boolean);
    EXPECT_EQ(root.GetObject().at("isStudent")->GetBoolean(), false);

    ASSERT_GT(root.GetObject().count("car"), 0);
    EXPECT_EQ(root.GetObject().at("car")->type, JsonValueType::Null);
    
    ASSERT_GT(root.GetObject().count("score"), 0);
    EXPECT_EQ(root.GetObject().at("score")->type, JsonValueType::Number);
    EXPECT_DOUBLE_EQ(root.GetObject().at("score")->GetNumber(), 95.5);
}

TEST_F(JsonLibTest, ParseObjectWithNestedObjects) {
    const std::string json_object = "{\"person\": {\"name\": \"Jane\", \"age\": 25, \"address\": {\"street\": \"123 Main St\", \"city\": \"Anytown\"}}, \"city\": \"New York\"}";
    JsonValue root = JsonParser::Parse(json_object);
    ASSERT_EQ(root.type, JsonValueType::Object);

    ASSERT_GT(root.GetObject().count("person"), 0);
    const JsonValue& person = *root.GetObject().at("person");
    ASSERT_EQ(person.type, JsonValueType::Object);
    EXPECT_EQ(person.GetObject().at("name")->GetString(), "Jane");
    EXPECT_DOUBLE_EQ(person.GetObject().at("age")->GetNumber(), 25.0);

    ASSERT_GT(person.GetObject().count("address"), 0);
    const JsonValue& address = *person.GetObject().at("address");
    ASSERT_EQ(address.type, JsonValueType::Object);
    EXPECT_EQ(address.GetObject().at("street")->GetString(), "123 Main St");
    EXPECT_EQ(address.GetObject().at("city")->GetString(), "Anytown");
    
    ASSERT_GT(root.GetObject().count("city"), 0);
    EXPECT_EQ(root.GetObject().at("city")->type, JsonValueType::String);
    EXPECT_EQ(root.GetObject().at("city")->GetString(), "New York");
}

TEST_F(JsonLibTest, ParseObjectWithNestedArrays) {
    const std::string json_object = "{\"data\": [1, 2, 3, 4.5], \"info\": {\"status\": \"active\", \"codes\": [\"X\", \"Y\"]}}";
    JsonValue root = JsonParser::Parse(json_object);
    ASSERT_EQ(root.type, JsonValueType::Object);

    ASSERT_GT(root.GetObject().count("data"), 0);
    const JsonValue& data_array = *root.GetObject().at("data");
    ASSERT_EQ(data_array.type, JsonValueType::Array);
    ASSERT_EQ(data_array.GetArray().size(), 4);
    EXPECT_DOUBLE_EQ(data_array.GetArray()[0].GetNumber(), 1.0);
    EXPECT_DOUBLE_EQ(data_array.GetArray()[1].GetNumber(), 2.0);
    EXPECT_DOUBLE_EQ(data_array.GetArray()[2].GetNumber(), 3.0);
    EXPECT_DOUBLE_EQ(data_array.GetArray()[3].GetNumber(), 4.5);

    ASSERT_GT(root.GetObject().count("info"), 0);
    const JsonValue& info_object = *root.GetObject().at("info");
    ASSERT_EQ(info_object.type, JsonValueType::Object);
    EXPECT_EQ(info_object.GetObject().at("status")->GetString(), "active");
    
    ASSERT_GT(info_object.GetObject().count("codes"), 0);
    const JsonValue& codes_array = *info_object.GetObject().at("codes");
    ASSERT_EQ(codes_array.type, JsonValueType::Array);
    ASSERT_EQ(codes_array.GetArray().size(), 2);
    EXPECT_EQ(codes_array.GetArray()[0].GetString(), "X");
    EXPECT_EQ(codes_array.GetArray()[1].GetString(), "Y");
}

TEST_F(JsonLibTest, ParseComplexNestedStructure) {
    const std::string json_complex = R"({
        "id": "user123",
        "profile": {
            "name": "Alice Wonderland",
            "email": "alice@example.com",
            "roles": ["admin", "editor"],
            "preferences": {
                "theme": "dark",
                "notifications": true,
                "max_items": 100
            }
        },
        "activity_log": [
            {"action": "login", "timestamp": "2023-01-15T10:00:00Z"},
            {"action": "view_page", "page": "/home", "timestamp": "2023-01-15T10:01:00Z"},
            {"action": "update_settings", "settings": {"theme": "dark"}, "timestamp": "2023-01-15T10:05:00Z"}
        ],
        "status": null
    })";
    
    JsonValue root = JsonParser::Parse(json_complex);
    ASSERT_EQ(root.type, JsonValueType::Object);
    
    EXPECT_EQ(root.GetObject().at("id")->GetString(), "user123");
    
    const JsonValue& profile = *root.GetObject().at("profile");
    ASSERT_EQ(profile.type, JsonValueType::Object);
    EXPECT_EQ(profile.GetObject().at("name")->GetString(), "Alice Wonderland");
    EXPECT_EQ(profile.GetObject().at("email")->GetString(), "alice@example.com");
    
    const JsonValue& roles = *profile.GetObject().at("roles");
    ASSERT_EQ(roles.type, JsonValueType::Array);
    EXPECT_EQ(roles.GetArray().size(), 2);
    EXPECT_EQ(roles.GetArray()[0].GetString(), "admin");
    EXPECT_EQ(roles.GetArray()[1].GetString(), "editor");
    
    const JsonValue& preferences = *profile.GetObject().at("preferences");
    ASSERT_EQ(preferences.type, JsonValueType::Object);
    EXPECT_EQ(preferences.GetObject().at("theme")->GetString(), "dark");
    EXPECT_TRUE(preferences.GetObject().at("notifications")->GetBoolean());
    EXPECT_DOUBLE_EQ(preferences.GetObject().at("max_items")->GetNumber(), 100.0);
    
    const JsonValue& activity_log = *root.GetObject().at("activity_log");
    ASSERT_EQ(activity_log.type, JsonValueType::Array);
    ASSERT_EQ(activity_log.GetArray().size(), 3);
    
    const JsonValue& activity_log_0 = activity_log.GetArray()[0];
    ASSERT_EQ(activity_log_0.type, JsonValueType::Object);
    EXPECT_EQ(activity_log_0.GetObject().at("action")->GetString(), "login");
    EXPECT_EQ(activity_log_0.GetObject().at("timestamp")->GetString(), "2023-01-15T10:00:00Z");
    
    const JsonValue& activity_log_2_settings = *activity_log.GetArray()[2].GetObject().at("settings");
    ASSERT_EQ(activity_log_2_settings.type, JsonValueType::Object);
    EXPECT_EQ(activity_log_2_settings.GetObject().at("theme")->GetString(), "dark");
    
    EXPECT_EQ(root.GetObject().at("status")->type, JsonValueType::Null);
}

// --- JSON Serialization Tests ---

TEST_F(JsonLibTest, SerializeStringValue) {
    JsonValue val;
    val.SetString("hello world");
    std::string json_string = val.ToString();
    
    JsonValue parsed_back = JsonParser::Parse(json_string);
    ASSERT_EQ(parsed_back.type, JsonValueType::String);
    EXPECT_EQ(parsed_back.GetString(), "hello world");

    JsonValue empty_val;
    empty_val.SetString("");
    json_string = empty_val.ToString();
    parsed_back = JsonParser::Parse(json_string);
    ASSERT_EQ(parsed_back.type, JsonValueType::String);
    EXPECT_EQ(parsed_back.GetString(), "");

    JsonValue special_chars_val;
    // Storing the actual characters, ToString should escape them.
    special_chars_val.SetString("line1\nline2\t\"quoted\""); 
    json_string = special_chars_val.ToString();
    // Expected: "\"line1\\nline2\\t\\\"quoted\\\"\""
    parsed_back = JsonParser::Parse(json_string);
    ASSERT_EQ(parsed_back.type, JsonValueType::String);
    // The GetString() should return the unescaped string.
    EXPECT_EQ(parsed_back.GetString(), "line1\nline2\t\"quoted\"");
}

TEST_F(JsonLibTest, SerializeNumericValues) {
    JsonValue num_val;
    num_val.SetNumber(123);
    std::string json_string = num_val.ToString();
    JsonValue parsed_back = JsonParser::Parse(json_string);
    ASSERT_EQ(parsed_back.type, JsonValueType::Number);
    EXPECT_DOUBLE_EQ(parsed_back.GetNumber(), 123.0);

    num_val.SetNumber(-45.67);
    json_string = num_val.ToString();
    parsed_back = JsonParser::Parse(json_string);
    ASSERT_EQ(parsed_back.type, JsonValueType::Number);
    EXPECT_DOUBLE_EQ(parsed_back.GetNumber(), -45.67);

    num_val.SetNumber(0);
    json_string = num_val.ToString();
    parsed_back = JsonParser::Parse(json_string);
    ASSERT_EQ(parsed_back.type, JsonValueType::Number);
    EXPECT_DOUBLE_EQ(parsed_back.GetNumber(), 0.0);
}

TEST_F(JsonLibTest, SerializeBooleanValues) {
    JsonValue bool_val;
    bool_val.SetBoolean(true);
    std::string json_string = bool_val.ToString();
    JsonValue parsed_back = JsonParser::Parse(json_string);
    ASSERT_EQ(parsed_back.type, JsonValueType::Boolean);
    EXPECT_EQ(parsed_back.GetBoolean(), true);

    bool_val.SetBoolean(false);
    json_string = bool_val.ToString();
    parsed_back = JsonParser::Parse(json_string);
    ASSERT_EQ(parsed_back.type, JsonValueType::Boolean);
    EXPECT_EQ(parsed_back.GetBoolean(), false);
}

TEST_F(JsonLibTest, SerializeNullValue) {
    JsonValue null_val; // Default constructor is Null
    ASSERT_EQ(null_val.type, JsonValueType::Null);
    std::string json_string = null_val.ToString(); // Should be "null"
    JsonValue parsed_back = JsonParser::Parse(json_string);
    ASSERT_EQ(parsed_back.type, JsonValueType::Null);
}

TEST_F(JsonLibTest, SerializeEmptyArray) {
    JsonValue arr_val;
    arr_val.SetArray(); // Initialize as an empty array
    std::string json_string = arr_val.ToString(); // Should be "[]"
    JsonValue parsed_back = JsonParser::Parse(json_string);
    ASSERT_EQ(parsed_back.type, JsonValueType::Array);
    EXPECT_EQ(parsed_back.GetArray().size(), 0);
}

TEST_F(JsonLibTest, SerializeArrayOfNumbers) {
    JsonValue arr_val;
    arr_val.SetArray();
    JsonValue num1; num1.SetNumber(10);
    JsonValue num2; num2.SetNumber(-20.5);
    JsonValue num3; num3.SetNumber(0);
    arr_val.GetArray().push_back(num1);
    arr_val.GetArray().push_back(num2);
    arr_val.GetArray().push_back(num3);
    
    std::string json_string = arr_val.ToString();
    JsonValue parsed_back = JsonParser::Parse(json_string);
    ASSERT_EQ(parsed_back.type, JsonValueType::Array);
    ASSERT_EQ(parsed_back.GetArray().size(), 3);
    EXPECT_DOUBLE_EQ(parsed_back.GetArray()[0].GetNumber(), 10.0);
    EXPECT_DOUBLE_EQ(parsed_back.GetArray()[1].GetNumber(), -20.5);
    EXPECT_DOUBLE_EQ(parsed_back.GetArray()[2].GetNumber(), 0.0);
}

TEST_F(JsonLibTest, SerializeArrayOfStrings) {
    JsonValue arr_val;
    arr_val.SetArray();
    JsonValue str1; str1.SetString("apple");
    JsonValue str2; str2.SetString("");
    JsonValue str3; str3.SetString("banana split");
    arr_val.GetArray().push_back(str1);
    arr_val.GetArray().push_back(str2);
    arr_val.GetArray().push_back(str3);
        
    std::string json_string = arr_val.ToString();
    JsonValue parsed_back = JsonParser::Parse(json_string);
    ASSERT_EQ(parsed_back.type, JsonValueType::Array);
    ASSERT_EQ(parsed_back.GetArray().size(), 3);
    EXPECT_EQ(parsed_back.GetArray()[0].GetString(), "apple");
    EXPECT_EQ(parsed_back.GetArray()[1].GetString(), "");
    EXPECT_EQ(parsed_back.GetArray()[2].GetString(), "banana split");
}

TEST_F(JsonLibTest, SerializeArrayOfMixedTypes) {
    JsonValue arr_val;
    arr_val.SetArray();
    JsonValue item1; item1.SetNumber(1);
    JsonValue item2; item2.SetString("test");
    JsonValue item3; item3.SetBoolean(true);
    JsonValue item4; // Null by default
    JsonValue item5; item5.SetNumber(12.34);

    arr_val.GetArray().push_back(item1);
    arr_val.GetArray().push_back(item2);
    arr_val.GetArray().push_back(item3);
    arr_val.GetArray().push_back(item4);
    arr_val.GetArray().push_back(item5);

    std::string json_string = arr_val.ToString();
    JsonValue parsed_back = JsonParser::Parse(json_string);
    ASSERT_EQ(parsed_back.type, JsonValueType::Array);
    ASSERT_EQ(parsed_back.GetArray().size(), 5);
    EXPECT_DOUBLE_EQ(parsed_back.GetArray()[0].GetNumber(), 1.0);
    EXPECT_EQ(parsed_back.GetArray()[1].GetString(), "test");
    EXPECT_EQ(parsed_back.GetArray()[2].GetBoolean(), true);
    EXPECT_EQ(parsed_back.GetArray()[3].type, JsonValueType::Null);
    EXPECT_DOUBLE_EQ(parsed_back.GetArray()[4].GetNumber(), 12.34);
}

TEST_F(JsonLibTest, SerializeArrayWithNestedStructure) {
    JsonValue root_array_val;
    root_array_val.SetArray();
    
    JsonValue nested_array_val;
    nested_array_val.SetArray();
    JsonValue num100; num100.SetNumber(100);
    JsonValue num200; num200.SetNumber(200);
    nested_array_val.GetArray().push_back(num100);
    nested_array_val.GetArray().push_back(num200);
    root_array_val.GetArray().push_back(nested_array_val);

    JsonValue nested_object_val;
    nested_object_val.SetObject();
    JsonValue prop_val; prop_val.SetString("value_in_nested_object");
    // For objects, InsertIntoObject takes a pointer. Manage memory carefully.
    // For testing, it's safer to build string and parse, or use stack objects if parser supports it.
    // The current ToString() and Parse() approach for other tests is better.
    // Let's build this via string for robust testing of ToString for objects within arrays.
    // This part is complex due to JsonValue* in map.
    // Re-evaluate how to test serialization of complex nested objects/arrays.
    // The current ToString will handle it if structure is built.
    // The custom library requires manual memory management for JsonValue* in object maps.
    // This test will be easier by constructing the string and parsing it.
    // Let's re-construct the string and parse it to verify ToString.
    // Simpler approach: build the object, then ToString, then parse back.
    // The following is how one might build it, but is tricky with memory for test.
    // nested_object_val.InsertIntoObject("prop", new JsonValue(prop_val));
    // root_array_val.GetArray().push_back(nested_object_val);
    //
    // Safer: Construct expected JSON string then parse, or parse a string and then ToString it.
    // Let's parse a string, then ToString it, then parse it back.
    std::string original_json_str = "[[100, 200], {\"prop\": \"value_in_nested_object\"}]";
    JsonValue original_parsed = JsonParser::Parse(original_json_str);

    std::string json_string = original_parsed.ToString();
    JsonValue parsed_back = JsonParser::Parse(json_string);

    ASSERT_EQ(parsed_back.type, JsonValueType::Array);
    ASSERT_EQ(parsed_back.GetArray().size(), 2);

    const auto& p_nested_array = parsed_back.GetArray()[0];
    ASSERT_EQ(p_nested_array.type, JsonValueType::Array);
    ASSERT_EQ(p_nested_array.GetArray().size(), 2);
    EXPECT_DOUBLE_EQ(p_nested_array.GetArray()[0].GetNumber(), 100.0);
    EXPECT_DOUBLE_EQ(p_nested_array.GetArray()[1].GetNumber(), 200.0);

    const auto& p_nested_object = parsed_back.GetArray()[1];
    ASSERT_EQ(p_nested_object.type, JsonValueType::Object);
    ASSERT_TRUE(p_nested_object.GetObject().count("prop") > 0);
    EXPECT_EQ(p_nested_object.GetObject().at("prop")->GetString(), "value_in_nested_object");
}


TEST_F(JsonLibTest, SerializeEmptyObject) {
    JsonValue obj_val;
    obj_val.SetObject(); // Initialize as an empty object
    std::string json_string = obj_val.ToString(); // Should be "{}"
    JsonValue parsed_back = JsonParser::Parse(json_string);
    ASSERT_EQ(parsed_back.type, JsonValueType::Object);
    EXPECT_EQ(parsed_back.GetObject().size(), 0);
}

TEST_F(JsonLibTest, SerializeSimpleObject) {
    // Build through string and parse, then ToString and parse back.
    // This avoids manual JsonValue* management for object values in test setup.
    std::string original_json_str = "{\"name\": \"Alice\", \"age\": 30, \"active\": true, \"city\": null, \"score\": 99.9}";
    JsonValue original_parsed = JsonParser::Parse(original_json_str);

    std::string json_string = original_parsed.ToString();
    JsonValue parsed_back = JsonParser::Parse(json_string);

    ASSERT_EQ(parsed_back.type, JsonValueType::Object);
    ASSERT_GT(parsed_back.GetObject().count("name"), 0);
    EXPECT_EQ(parsed_back.GetObject().at("name")->GetString(), "Alice");
    ASSERT_GT(parsed_back.GetObject().count("age"), 0);
    EXPECT_DOUBLE_EQ(parsed_back.GetObject().at("age")->GetNumber(), 30.0);
    ASSERT_GT(parsed_back.GetObject().count("active"), 0);
    EXPECT_EQ(parsed_back.GetObject().at("active")->GetBoolean(), true);
    ASSERT_GT(parsed_back.GetObject().count("city"), 0);
    EXPECT_EQ(parsed_back.GetObject().at("city")->type, JsonValueType::Null);
    ASSERT_GT(parsed_back.GetObject().count("score"), 0);
    EXPECT_DOUBLE_EQ(parsed_back.GetObject().at("score")->GetNumber(), 99.9);
}

TEST_F(JsonLibTest, SerializeObjectWithNestedStructure) {
    std::string original_json_str = R"({
        "id": "item123",
        "details": {"color": "blue", "quantity": 50},
        "tags": ["electronics", "consumer", {"special_tag": "clearance"}]
    })";
    JsonValue original_parsed = JsonParser::Parse(original_json_str);
    std::string json_string = original_parsed.ToString();
    JsonValue parsed_back = JsonParser::Parse(json_string);

    ASSERT_EQ(parsed_back.type, JsonValueType::Object);
    EXPECT_EQ(parsed_back.GetObject().at("id")->GetString(), "item123");
    
    const auto& details = *parsed_back.GetObject().at("details");
    ASSERT_EQ(details.type, JsonValueType::Object);
    EXPECT_EQ(details.GetObject().at("color")->GetString(), "blue");
    EXPECT_DOUBLE_EQ(details.GetObject().at("quantity")->GetNumber(), 50.0);

    const auto& tags = *parsed_back.GetObject().at("tags");
    ASSERT_EQ(tags.type, JsonValueType::Array);
    ASSERT_EQ(tags.GetArray().size(), 3);
    EXPECT_EQ(tags.GetArray()[0].GetString(), "electronics");
    EXPECT_EQ(tags.GetArray()[1].GetString(), "consumer");
    const auto& nested_tag_obj = tags.GetArray()[2];
    ASSERT_EQ(nested_tag_obj.type, JsonValueType::Object);
    EXPECT_EQ(nested_tag_obj.GetObject().at("special_tag")->GetString(), "clearance");
}

TEST_F(JsonLibTest, SerializeComplexNestedStructureRoundTrip) { 
    std::string original_json_str = R"({
        "id": 123,
        "data": {
            "points": [10, 20, 30],
            "valid": true
        },
        "tags": ["TagA", "TagB"],
        "name": "Complex Test Object",
        "value": null
    })";
    JsonValue original_parsed = JsonParser::Parse(original_json_str);
    std::string json_string = original_parsed.ToString();
    JsonValue parsed_back = JsonParser::Parse(json_string);

    ASSERT_EQ(parsed_back.type, JsonValueType::Object);
    EXPECT_DOUBLE_EQ(parsed_back.GetObject().at("id")->GetNumber(), 123.0);
    EXPECT_EQ(parsed_back.GetObject().at("name")->GetString(), "Complex Test Object");
    EXPECT_EQ(parsed_back.GetObject().at("value")->type, JsonValueType::Null);
    
    const auto& data = *parsed_back.GetObject().at("data");
    ASSERT_EQ(data.type, JsonValueType::Object);
    const auto& points = *data.GetObject().at("points");
    ASSERT_EQ(points.type, JsonValueType::Array);
    EXPECT_EQ(points.GetArray().size(), 3);
    EXPECT_DOUBLE_EQ(points.GetArray()[0].GetNumber(), 10.0);
    EXPECT_DOUBLE_EQ(points.GetArray()[1].GetNumber(), 20.0);
    EXPECT_DOUBLE_EQ(points.GetArray()[2].GetNumber(), 30.0);
    EXPECT_TRUE(data.GetObject().at("valid")->GetBoolean());

    const auto& tags = *parsed_back.GetObject().at("tags");
    ASSERT_EQ(tags.type, JsonValueType::Array);
    EXPECT_EQ(tags.GetArray().size(), 2);
    EXPECT_EQ(tags.GetArray()[0].GetString(), "TagA");
    EXPECT_EQ(tags.GetArray()[1].GetString(), "TagB");
}

// --- JsonValue API Tests (adapted from Json::Value) ---

TEST_F(JsonLibTest, TypeCheckingMethods) {
    JsonValue str_val; str_val.SetString("hello");
    EXPECT_EQ(str_val.type, JsonValueType::String);
    EXPECT_NE(str_val.type, JsonValueType::Number);
    EXPECT_NE(str_val.type, JsonValueType::Boolean);
    EXPECT_NE(str_val.type, JsonValueType::Array);
    EXPECT_NE(str_val.type, JsonValueType::Object);
    EXPECT_NE(str_val.type, JsonValueType::Null);

    JsonValue num_val; num_val.SetNumber(123);
    EXPECT_NE(num_val.type, JsonValueType::String);
    EXPECT_EQ(num_val.type, JsonValueType::Number);
    EXPECT_NE(num_val.type, JsonValueType::Boolean);
    EXPECT_NE(num_val.type, JsonValueType::Array);
    EXPECT_NE(num_val.type, JsonValueType::Object);
    EXPECT_NE(num_val.type, JsonValueType::Null);

    // Note: Custom library has only JsonValueType::Number (double)
    // No specific Int/UInt type checks like jsoncpp's isInt/isUInt/isDouble separately
    // isNumeric() concept is implicitly covered by type == JsonValueType::Number

    JsonValue bool_val; bool_val.SetBoolean(true);
    EXPECT_NE(bool_val.type, JsonValueType::String);
    EXPECT_NE(bool_val.type, JsonValueType::Number);
    EXPECT_EQ(bool_val.type, JsonValueType::Boolean);
    EXPECT_NE(bool_val.type, JsonValueType::Array);
    EXPECT_NE(bool_val.type, JsonValueType::Object);
    EXPECT_NE(bool_val.type, JsonValueType::Null);

    JsonValue arr_val; arr_val.SetArray();
    EXPECT_NE(arr_val.type, JsonValueType::String);
    EXPECT_NE(arr_val.type, JsonValueType::Number);
    EXPECT_NE(arr_val.type, JsonValueType::Boolean);
    EXPECT_EQ(arr_val.type, JsonValueType::Array);
    EXPECT_NE(arr_val.type, JsonValueType::Object);
    EXPECT_NE(arr_val.type, JsonValueType::Null);
    EXPECT_TRUE(arr_val.IsArray()); // Test helper method too

    JsonValue obj_val; obj_val.SetObject();
    EXPECT_NE(obj_val.type, JsonValueType::String);
    EXPECT_NE(obj_val.type, JsonValueType::Number);
    EXPECT_NE(obj_val.type, JsonValueType::Boolean);
    EXPECT_NE(obj_val.type, JsonValueType::Array);
    EXPECT_EQ(obj_val.type, JsonValueType::Object);
    EXPECT_NE(obj_val.type, JsonValueType::Null);
    EXPECT_TRUE(obj_val.IsObject()); // Test helper method too

    JsonValue null_val; // Default is Null
    EXPECT_NE(null_val.type, JsonValueType::String);
    EXPECT_NE(null_val.type, JsonValueType::Number);
    EXPECT_NE(null_val.type, JsonValueType::Boolean);
    EXPECT_NE(null_val.type, JsonValueType::Array);
    EXPECT_NE(null_val.type, JsonValueType::Object);
    EXPECT_EQ(null_val.type, JsonValueType::Null);
}

TEST_F(JsonLibTest, ValueRetrievalMethods) { // Renamed from ValueRetrievalAsType
    JsonValue str_val; str_val.SetString("hello");
    EXPECT_EQ(str_val.GetString(), "hello");
    // For non-string types, GetString() behavior is undefined by current .hpp; assume it might throw or return empty.
    // For other GetType() methods, they should only be called if type matches.
    // Example: EXPECT_THROW(str_val.GetNumber(), std::logic_error); // Or similar if getters assert type

    JsonValue num_val; num_val.SetNumber(123.45);
    EXPECT_DOUBLE_EQ(num_val.GetNumber(), 123.45);
    // GetString() for number: ToString() handles this.
    // EXPECT_EQ(num_val.ToString(), "123.450000"); // Exact string form of double.

    JsonValue bool_true_val; bool_true_val.SetBoolean(true);
    EXPECT_EQ(bool_true_val.GetBoolean(), true);
    // EXPECT_EQ(bool_true_val.ToString(), "true");

    JsonValue bool_false_val; bool_false_val.SetBoolean(false);
    EXPECT_EQ(bool_false_val.GetBoolean(), false);
    // EXPECT_EQ(bool_false_val.ToString(), "false");
    
    JsonValue null_val; // Default Null
    // GetString(), GetNumber(), GetBoolean() on a Null type: behavior is undefined by .hpp.
    // Test ToString() for null
    EXPECT_EQ(null_val.ToString(), "null");

    // Array and Object retrieval:
    // GetArray() and GetObject() are tested in their respective parsing/serialization tests.
    // Attempting to GetArray() on a non-array type should ideally throw.
    JsonValue not_array; not_array.SetString("test");
    // EXPECT_THROW(not_array.GetArray(), JsonValueException); // Or std::logic_error
}

TEST_F(JsonLibTest, ObjectMemberHandlingAndSize) {
    JsonValue obj_val;
    obj_val.SetObject();
    EXPECT_EQ(obj_val.GetObject().size(), 0u);

    // For setting members, we'd typically parse a string or use InsertIntoObject.
    // Since InsertIntoObject requires new JsonValue*, let's parse.
    obj_val = JsonParser::Parse("{\"name\": \"Test Object\", \"count\": 101, \"valid\": true}");
    
    EXPECT_GT(obj_val.GetObject().count("name"), 0u);
    EXPECT_GT(obj_val.GetObject().count("count"), 0u);
    EXPECT_GT(obj_val.GetObject().count("valid"), 0u);
    EXPECT_EQ(obj_val.GetObject().count("non_existent_key"), 0u);
    EXPECT_EQ(obj_val.GetObject().count("Name"), 0u); // Case-sensitive

    EXPECT_EQ(obj_val.GetObject().size(), 3u);
    EXPECT_EQ(obj_val.GetObject().at("name")->GetString(), "Test Object");
    EXPECT_DOUBLE_EQ(obj_val.GetObject().at("count")->GetNumber(), 101.0);
    EXPECT_EQ(obj_val.GetObject().at("valid")->GetBoolean(), true);

    // Accessing non-existent key with .at() throws std::out_of_range
    EXPECT_THROW(obj_val.GetObject().at("new_key_access"), std::out_of_range);
    // Count remains the same, no null member is added automatically like jsoncpp
    EXPECT_EQ(obj_val.GetObject().size(), 3u); 
    EXPECT_EQ(obj_val.GetObject().count("new_key_access"), 0u);
}

TEST_F(JsonLibTest, ArrayAppendAndSize) {
    JsonValue arr_val;
    arr_val.SetArray();
    EXPECT_EQ(arr_val.GetArray().size(), 0u);
    // EXPECT_TRUE(arr_val.GetArray().empty()); // Standard vector method

    JsonValue str_el; str_el.SetString("first_element");
    arr_val.GetArray().push_back(str_el);
    EXPECT_EQ(arr_val.GetArray().size(), 1u);
    // EXPECT_FALSE(arr_val.GetArray().empty());
    EXPECT_EQ(arr_val.GetArray()[0].GetString(), "first_element");

    JsonValue num_el; num_el.SetNumber(202);
    arr_val.GetArray().push_back(num_el);
    EXPECT_EQ(arr_val.GetArray().size(), 2u);
    EXPECT_DOUBLE_EQ(arr_val.GetArray()[1].GetNumber(), 202.0);

    // For nested object:
    // JsonValue nested_obj_el; nested_obj_el.SetObject();
    // JsonValue id_val; id_val.SetString("nested_id_001");
    // nested_obj_el.InsertIntoObject("id", new JsonValue(id_val)); // Memory management
    // arr_val.GetArray().push_back(nested_obj_el);
    // Instead, parse a string for complex elements to avoid manual `new` in tests
    JsonValue nested_obj_parsed = JsonParser::Parse("{\"id\": \"nested_id_001\"}");
    arr_val.GetArray().push_back(nested_obj_parsed);

    EXPECT_EQ(arr_val.GetArray().size(), 3u);
    ASSERT_EQ(arr_val.GetArray()[2].type, JsonValueType::Object);
    EXPECT_EQ(arr_val.GetArray()[2].GetObject().at("id")->GetString(), "nested_id_001");

    JsonValue null_el; // Default null
    arr_val.GetArray().push_back(null_el);
    EXPECT_EQ(arr_val.GetArray().size(), 4u);
    EXPECT_EQ(arr_val.GetArray()[3].type, JsonValueType::Null);
}

TEST_F(JsonLibTest, NullValueSpecifics) {
    JsonValue default_constructed_val; // Default constructor creates a null value
    EXPECT_EQ(default_constructed_val.type, JsonValueType::Null);
    // GetString(), GetNumber() etc. on Null is undefined by hpp, check ToString()
    EXPECT_EQ(default_constructed_val.ToString(), "null");

    JsonValue explicit_null_val(JsonValueType::Null); // Explicit constructor
    EXPECT_EQ(explicit_null_val.type, JsonValueType::Null);

    JsonValue num_val; num_val.SetNumber(55);
    EXPECT_NE(num_val.type, JsonValueType::Null);
    
    num_val = JsonValue(); // Assigning a default-constructed (null) value
    EXPECT_EQ(num_val.type, JsonValueType::Null);

    num_val.SetNumber(77); // Reassign to non-null
    EXPECT_NE(num_val.type, JsonValueType::Null);
    num_val = JsonValue(JsonValueType::Null); // Assigning explicit null type
    EXPECT_EQ(num_val.type, JsonValueType::Null);
}

// --- JSON Parsing Error Handling Tests ---

TEST_F(JsonLibTest, ParseErrorIncompleteObject) {
    // Missing closing brace
    EXPECT_THROW(JsonParser::Parse("{\"key\": \"value\""), JsonParseException);
    // Missing closing brace and value is incomplete
    EXPECT_THROW(JsonParser::Parse("{\"key\": \"value"), JsonParseException);
    // Missing key, only colon
    EXPECT_THROW(JsonParser::Parse("{: \"value\"}"), JsonParseException);
    // Missing value after key
    EXPECT_THROW(JsonParser::Parse("{\"key\": }"), JsonParseException);
}

TEST_F(JsonLibTest, ParseErrorIncompleteArray) {
    // Missing closing bracket
    EXPECT_THROW(JsonParser::Parse("[1, 2, 3"), JsonParseException);
    // Missing closing bracket and value is incomplete (string context)
    EXPECT_THROW(JsonParser::Parse("[1, \"hello"), JsonParseException);
}

TEST_F(JsonLibTest, ParseErrorMissingComma) {
    // Missing comma in object
    EXPECT_THROW(JsonParser::Parse("{\"key1\": \"v1\" \"key2\": \"v2\"}"), JsonParseException);
    // Missing comma in array
    EXPECT_THROW(JsonParser::Parse("[1 2]"), JsonParseException);
    EXPECT_THROW(JsonParser::Parse("[\"a\" \"b\"]"), JsonParseException);
}

TEST_F(JsonLibTest, ParseBehaviorTrailingComma) {
    // Standard JSON does not allow trailing commas. The custom parser should enforce this.
    const std::string json_object_trailing_comma = "{\"key\": \"value\",}";
    EXPECT_THROW(JsonParser::Parse(json_object_trailing_comma), JsonParseException);

    const std::string json_array_trailing_comma = "[1, 2, 3,]";
    EXPECT_THROW(JsonParser::Parse(json_array_trailing_comma), JsonParseException);
}

TEST_F(JsonLibTest, ParseErrorUnquotedKey) {
    // Standard JSON requires keys to be strings (quoted).
    EXPECT_THROW(JsonParser::Parse("{key: \"value\"}"), JsonParseException);
}

TEST_F(JsonLibTest, ParseErrorInvalidString) {
    // Unterminated string
    EXPECT_THROW(JsonParser::Parse("\"unterminated string"), JsonParseException);
    // Invalid escape sequence (e.g., \x is not standard JSON)
    // The custom parser's ParseString should handle this.
    EXPECT_THROW(JsonParser::Parse("\"\\x\""), JsonParseException); // Example of an invalid escape
    // Invalid unicode escape
    EXPECT_THROW(JsonParser::Parse("\"\\u123X\""), JsonParseException);
}

TEST_F(JsonLibTest, ParseErrorInvalidNumber) {
    EXPECT_THROW(JsonParser::Parse("1.2.3"), JsonParseException);      // Multiple decimal points
    EXPECT_THROW(JsonParser::Parse("1ee4"), JsonParseException);       // Invalid scientific notation (double 'e')
    EXPECT_THROW(JsonParser::Parse("--5"), JsonParseException);        // Double negative
    EXPECT_THROW(JsonParser::Parse("1.e"), JsonParseException);        // Incomplete scientific notation
    // Standard JSON does not allow leading zeros on non-zero numbers (e.g. 0123 is not valid for number 123)
    // The custom parser should enforce this if it's strictly following JSON.
    EXPECT_THROW(JsonParser::Parse("0123"), JsonParseException);
    // NaN and Infinity are not standard JSON.
    EXPECT_THROW(JsonParser::Parse("NaN"), JsonParseException);
    EXPECT_THROW(JsonParser::Parse("Infinity"), JsonParseException);
}

TEST_F(JsonLibTest, ParseErrorInvalidKeyword) {
    EXPECT_THROW(JsonParser::Parse("ture"), JsonParseException);  // Misspelled true
    EXPECT_THROW(JsonParser::Parse("flase"), JsonParseException); // Misspelled false
    EXPECT_THROW(JsonParser::Parse("nul"), JsonParseException);   // Misspelled null
    EXPECT_THROW(JsonParser::Parse("TRUE"), JsonParseException);  // Keywords are case-sensitive
    EXPECT_THROW(JsonParser::Parse("False"), JsonParseException);
    EXPECT_THROW(JsonParser::Parse("Null"), JsonParseException);
}

TEST_F(JsonLibTest, ParseErrorMismatchedBrackets) {
    EXPECT_THROW(JsonParser::Parse("{\"key\": [1, 2)}"), JsonParseException); // Mismatched array closing in object
    // This one is valid: "[{\"key\": \"v\"}]" - an array with one object element.
    // Let's test actual mismatches:
    EXPECT_THROW(JsonParser::Parse("{\"a\": 1]"), JsonParseException); // Mismatched object closing
    EXPECT_THROW(JsonParser::Parse("[\"a\", 2}"), JsonParseException); // Mismatched array closing
}

TEST_F(JsonLibTest, ParseErrorUnexpectedToken) {
    EXPECT_THROW(JsonParser::Parse("{\"key\": \"value\"} unexpected_token"), JsonParseException); // Extra token after root object
    EXPECT_THROW(JsonParser::Parse("[1, 2] unexpected_token"), JsonParseException);             // Extra token after root array
    EXPECT_THROW(JsonParser::Parse("{\"key\": \"value\"} 123"), JsonParseException);
    EXPECT_THROW(JsonParser::Parse("[1,2] \"abc\""), JsonParseException);
    EXPECT_THROW(JsonParser::Parse("true false"), JsonParseException); // Multiple roots without being in an array/object
}

TEST_F(JsonLibTest, ParseErrorEmptyInput) {
    EXPECT_THROW(JsonParser::Parse(""), JsonParseException); // Empty string is not valid JSON
}

TEST_F(JsonLibTest, ParseErrorOnlyWhitespace) {
    // Whitespace is skipped, but then an empty string results, which is not valid JSON.
    EXPECT_THROW(JsonParser::Parse("   \t\n   "), JsonParseException);
}
