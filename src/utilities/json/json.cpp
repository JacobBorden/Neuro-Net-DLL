#include "json.hpp"

JsonValue JsonParser::Parse(const std::string& json_string)
{
	size_t index =0;
	SkipComment(json_string, index);
	return ParseValue(json_string, index);
}

JsonValue JsonParser::ParseValue(const std::string& json_string, size_t& index)
{
	SkipWhitespace(json_string, index);
	switch(json_string[index])
	{
		case 'n':
		{
			ParseNull(json_string, index);
			JsonValue value(JsonValueType::Null);
			return value;
		}
		case 't':
		case 'f':
		{
			JsonValue value(JsonValueType::Boolean);
			value.boolean_value = ParseBoolean(json_string, index);
			return value;
		}
		case '"':
		{
			JsonValue value(JsonValueType::String);
			value.string_value = ParseString(json_string, index);
			return value;
		}
		case '{':
		{
			JsonValue value(JsonValueType::Object);
			value.object_value = ParseObject(json_string, index);
			return value;
		}
		case '[':
		{
			JsonValue value(JsonValueType::Array);
			value.array_value = ParseArray(json_string, index);
			return value;
		}
		default:
		{
			JsonValue value(JsonValueType::Number);
			value.number_value = ParseNumber(json_string, index);
			return value;
		}
	}
}
			
void JsonParser::SkipWhitespace(const std::string& json_string, size_t& index)
{
	while(index < json_string.length() && isspace(json_string[index])){
		++index;
	}
}

void JsonParser::ParseNull(const std::string& json_string, size_t& index){
	ExpectString(json_string, index, "null");
}

bool JsonParser::ParseBoolean(const std::string& json_string, size_t& index)
{
	if (json_string[index] == 't')
	{
		ExpectString(json_string, index, "true");
		return true;
	}
	else
	{
		ExpectString(json_string, index, "false");
		return false;
	}
}

std::string JsonParser::ParseString(const std::string& json_string, size_t& index)
{
	std::string result;
	
	// Parse opening quations mark.
	ExpectChar(json_string, index, '"');

	//Parse string characters.
	while(index < json_string.length() && json_string[index] != '"')
	{
		char c = json_string[index];
		if (c == '\\')
				{
				//Parse escape sequence.
				++index;
				if(index >= json_string.length())
				{
					throw JsonParseException("Unexpected end of character string");
				}
				c = json_string[index];
				switch(c)
				{
					case '"':
						result +='"';
						break;
					case '\\':
						result +='\\';
						break;
					case '/':
						result += '/';
						break;
					case 'b':
						result +='\b';
						break;
					case 'f':
						result +='\f';
						break;
					case 'n':
						result +='\n';
						break;
					case 'r':
						result += '\r';
						break;
					case 't':
						result += '\t';
						break;
					case 'u':
						{
						++index;
						if(index +4 >=json_string.length())
						{
							throw JsonParseException("Unexpected end of string");
						}
						std::string hex_string = json_string.substr(index, 4);
						try
						{
							int code_point = std::stoi(hex_string, nullptr, 16);
							result += UnicodeCodePointToUtf8(code_point);
						}
						catch (const std::invalid_argument& ex)
						{
							throw JsonParseException("Invalid Unicode escape sequence");
						}
						index +=4;
						break;
						}
					default:
						throw JsonParseException("Invalid escape sequence");
				}
			}	

		else
		{
			result += c;
		}
			
		++index;
	}			

	
	ExpectChar(json_string, index, '"');
	return result;
}

std::unordered_map<std::string, JsonValue*> JsonParser::ParseObject(const std::string& json_string, size_t& index)
{
	std::unordered_map <std::string, JsonValue*> object;
	SkipWhitespace(json_string, index);
	ExpectChar(json_string, index, '{');
	SkipWhitespace(json_string,index);
	while(index < json_string.length() && json_string[index] != '}')
	{
		std::string key = ParseString(json_string, index);
		SkipWhitespace(json_string, index);
		ExpectChar(json_string, index, ':');
		SkipWhitespace(json_string, index);
		JsonValue* val = new JsonValue(ParseValue(json_string,index));
		object[key] = val;
		SkipWhitespace(json_string, index);

		if(index < json_string.length())
		{
			char c = json_string[index];
			if(c==',')
			{
				++index;
				SkipWhitespace(json_string, index);
			}
			else if (c == '}')
				break;
			else throw JsonParseException("Expected ',' or '}' while parsing object");
		}
	}
	ExpectChar(json_string, index, '}');
	return object;
}

std::vector<JsonValue> JsonParser::ParseArray(const std::string& json_string, size_t& index)
{
	std::vector<JsonValue> array;
	SkipWhitespace(json_string, index);
	ExpectChar(json_string, index,'[');
	SkipWhitespace(json_string, index);
	while(index < json_string.length() && json_string[index] != '[')
	{
		JsonValue value = ParseValue(json_string, index);
		array.push_back(value);
		SkipWhitespace(json_string, index);

		if(index < json_string.length())
		{
			char c =json_string[index];
			if (c == ',')
			{
				++index;
				SkipWhitespace(json_string, index);
			}
			else if (c == ']')
				break;
			else throw JsonParseException("Expected ',' or ']' while parsing arraay");
		}
	}
	ExpectChar(json_string, index, ']');
	return array;
}

double JsonParser::ParseNumber(const std::string& json_string, size_t& index)
{
	SkipWhitespace(json_string, index);
	// Check to see if the number is negative
	bool negative = false;
	if(json_string[index] == '-')
	{
		negative = true;
		++index;
	}

	//Parse the integer part of the number
	
	unsigned long long integer = 0;
	while(index < json_string.length() && IsDigit(json_string[index]))
	{
		integer = integer * 10 + (json_string[index] - '0');
		++index;
	}

	//Check for decimal
	double fraction =0.0;
	if(index < json_string.length() && json_string[index] == '.')
	{
		double scale =0.1;
		while(index < json_string.length() && IsDigit(json_string[index]))
		{
			fraction = fraction + (scale * (json_string[index] - '0'));
			++index;
			scale *= 0.1;
		}
	}

	double exponent_value = 0;
	if(index < json_string.length() && (json_string[index] == 'e' || json_string[index] == 'E'))
	{
		bool exponent_negative = false;
		if(json_string[index] == '-')
		{
			exponent_negative = true;
			++index;
		}
		unsigned long long exponent =0;
		while(index < json_string.length() && IsDigit(json_string[index]))
		{
			exponent = exponent *10 + (json_string[index] - '0');
			++index;
		}

		if(exponent_negative)
			exponent_value = 1.0 / std::pow(10.0, exponent);
		else exponent_value = std::pow(10, exponent);

	}

	double value = integer + fraction;

	if(negative)
		value = -value;
	value *= exponent_value;

	return value;
}

bool JsonParser::IsDigit(char c)
{
	return c >='0' && c<= '9';
}

void JsonParser::ExpectChar(const std::string& json_string, size_t& index, char expected_char)
{
	if(index >= json_string.length() || json_string[index] != expected_char)
	{
		throw JsonParseException("Unexpected character");
	}
	++index;
}

void JsonParser::ExpectString(const std::string& json_string, size_t& index, const std::string& expected_string)
{
	for(char c: expected_string)
	{
		if(index >= json_string.length() || json_string[index] != c)
		{
			throw JsonParseException("Unexpected character");
		}
		++index;
	}
}

std::string JsonParser::UnicodeCodePointToUtf8(int code_point)
{
	if (code_point < 0 || code_point > 0x10FFFF) 
	{
    		throw JsonParseException("Invalid Unicode code point");
  	}
  	std::string result;
  	if (code_point <= 0x7F) {
    // One-byte encoding.
    		result += static_cast<char>(code_point);
  	} else if (code_point <= 0x7FF) {
    // Two-byte encoding.
    		result += static_cast<char>(0xC0 | (code_point >> 6));
    		result += static_cast<char>(0x80 | (code_point & 0x3F));
  	} else if (code_point <= 0xFFFF) {
    // Three-byte encoding.
    		result += static_cast<char>(0xE0 | (code_point >> 12));
    		result += static_cast<char>(0x80 | ((code_point >> 6) & 0x3F));
    		result += static_cast<char>(0x80 | (code_point & 0x3F));
  	} else {
    // Four-byte encoding.
    		result += static_cast<char>(0xF0 | (code_point >> 18));
    		result += static_cast<char>(0x80 | ((code_point >> 12) & 0x3F));
    		result += static_cast<char>(0x80 | ((code_point >> 6) & 0x3F));
    		result += static_cast<char>(0x80 | (code_point & 0x3F));
  	}
  	return result;
}

void JsonParser::SkipComment(const std::string& json_string, size_t& index)
{
	while (json_string[index] != '\0')
	{
		char c = json_string[index];
		if(c == '/' && json_string[index +1] == '/')
		{
			while (json_string[index] != '\0' && json_string[index]  != '\n')
					++index;
		}
		else if (c == '/' && json_string[index+1] =='*')
		{
			index +=2;
			while (json_string[index] != '\0' && !(json_string[index] == '*' && json_string[index +1] == '/'))
				++index;
			index +=2;
		}
		else break;
	}
}	