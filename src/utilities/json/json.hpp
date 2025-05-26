#pragma once
#ifndef _JSON_
#define _JSON_
#include <string>
#include <unordered_map>
#include <vector>
#include <stdexcept>
#include <cmath>
#include "json_exception.hpp"
enum class JsonValueType{
	Null,
	Boolean,
	Number,
	String,
	Array,
	Object
};


struct JsonValue{

	JsonValue(JsonValueType type_ = JsonValueType::Null) : type(type_){}
	JsonValueType type;
	bool boolean_value;
	double number_value;
	std::string string_value;
	std::vector<JsonValue> array_value;
	std::unordered_map<std::string, JsonValue* > object_value;
	std::string ToString() const{
		switch (type)
		{
			case JsonValueType::Null:
				return "null";
			case JsonValueType::Boolean:
				return boolean_value ? "true":"false";
			case JsonValueType::Number:
				return std::to_string(number_value);
			case JsonValueType::String:
				return "\"" +string_value +"\"";
			case JsonValueType::Array:
				return ArrayToString(array_value);
			case JsonValueType::Object:
				return ObjectToString(object_value);
		}
	}


	void SetBoolean(bool value){type = JsonValueType::Boolean; boolean_value = value;}
	bool GetBoolean() const {return boolean_value;}
	void SetNumber(double value){type = JsonValueType::Number; number_value = value;}
	double GetNumber() const {return number_value;}
	void SetString(const std::string& value){ type = JsonValueType::String; string_value = value;}
	const std::string& GetString() const { return string_value;}
	std::string& GetString() {return string_value;}
	void SetArray(){type = JsonValueType::Array;}
	bool IsArray() const { return type == JsonValueType::Array;}
	std::vector<JsonValue>& GetArray(){ return array_value;}
	const std::vector<JsonValue>& GetArray() const{return array_value;}
	void SetObject(){type = JsonValueType::Object;}
	bool IsObject() const { return type == JsonValueType::Object;}
	std::unordered_map<std::string, JsonValue*> &GetObJect(){return object_value;}
	const std::unordered_map<std::string, JsonValue*> &GetObject() const {return object_value;}
	void InsertIntoObject(const std::string& key , JsonValue* value){
		if (type != JsonValueType::Object){
			throw JsonParseException("Cannot insert into a non-object value.");
		}
		object_value[key] = value;
	}


 
	private:
	
	std::string ArrayToString(const std::vector<JsonValue>& array) const {
		std::string result = "[";
		for(size_t i=0; i < array.size(); i++)
		{
			if(i >0){
				result += ", ";
			}
			result += array[i].ToString();
		}
		result +="]";
		return result;
	}
	
	std::string ObjectToString(const std::unordered_map<std::string, JsonValue*>& object) const{
		std::string result ="{";
		bool first = true;
		for(const auto& kv: object)
		{
			if(!first){
				result += ", ";
			}
			
			first =false;
			result +="\""+ kv.first + "\": " + kv.second->ToString();
		}
		result +="}";
		return result;
	}
};

class JsonParser{
	public:
		static JsonValue Parse(const std::string& json_string);
	private:
		static JsonValue ParseValue(const std::string& json_string, size_t& index);
		static std::string ParseString(const std::string& json_string, size_t& index);
		static double ParseNumber(const std::string& json_string, size_t& index);
		static std::vector<JsonValue> ParseArray(const std::string& json_string, size_t& index);
		static std::unordered_map<std::string, JsonValue*> ParseObject(const std::string& json_string, size_t& index);
		static bool ParseBoolean(const std::string& json_string, size_t& index);
		static void ParseNull(const std::string& json_string, size_t& index);
		static void SkipWhitespace(const std::string& json_string, size_t& index);
		static void ExpectString(const std::string& json_string, size_t& index, const std::string& expected_string);
		static void ExpectChar(const std::string& json_string, size_t& index, char expected_char);
		static std::string UnicodeCodePointToUtf8(int code_point);
		static bool IsDigit(char c);
		static void SkipComment(const std::string& json_string, size_t& index);
};
#endif