#include "json.hpp"

JsonValue JsonParser::Parse(const std::string& json_string)
{
	size_t index =0;
    // Skip initial comments and whitespace
    SkipWhitespace(json_string, index); // Skip any leading whitespace first
    SkipComment(json_string, index);    // Then skip comments (which might be preceded/followed by whitespace)
    SkipWhitespace(json_string, index); // Skip any whitespace after comments

    if (index >= json_string.length() && !json_string.empty()) { // String with only whitespace/comments
        // This case is fine if the string was *only* whitespace/comments.
        // But if it was truly empty, ParseValue will handle it.
        // If it was not empty but became empty after skipping, it implies only whitespace/comments.
        // Standard behavior is often to expect a value if the string is not empty.
        // For now, let ParseValue try and fail if it's truly empty after skips.
    } else if (json_string.empty()){
        throw JsonParseException("Cannot parse an empty string.");
    }


	JsonValue result = ParseValue(json_string, index);
    
    // After parsing a value, skip any trailing whitespace and comments
    SkipWhitespace(json_string, index);
    SkipComment(json_string, index); // Allow comments at the end
    SkipWhitespace(json_string, index);


	if(index < json_string.length())
	{
		throw JsonParseException("Unexpected trailing characters after JSON value: " + json_string.substr(index));
	}
	return result;
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
		// For numbers, check if it starts with a digit or '-'
		case '-':
		case '0': case '1': case '2': case '3': case '4':
		case '5': case '6': case '7': case '8': case '9':
		{
			JsonValue value(JsonValueType::Number);
			value.number_value = ParseNumber(json_string, index);
			return value;
		}
		default:
			// If it's none of the above, it's an unexpected character.
			throw JsonParseException("Unexpected character in ParseValue: " + std::string(1, json_string[index]));
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
    if (index < json_string.length() && json_string[index] == '}') { // Handle empty object
        ExpectChar(json_string, index, '}');
        return object;
    }
	while(index < json_string.length() && json_string[index] != '}')
	{
		if (index >= json_string.length()) throw JsonParseException("Unexpected end of object definition");
        std::string key = ParseString(json_string, index);
		SkipWhitespace(json_string, index);
		if (index >= json_string.length()) throw JsonParseException("Unexpected end of object definition, missing colon");
		ExpectChar(json_string, index, ':');
		SkipWhitespace(json_string, index);
		if (index >= json_string.length()) throw JsonParseException("Unexpected end of object definition, missing value");
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
                if (index >= json_string.length() || json_string[index] == '}') // Trailing comma or end after comma
                    throw JsonParseException("Trailing comma or unexpected end after comma in object");
			}
			else if (c == '}')
				break;
			else throw JsonParseException("Expected ',' or '}' while parsing object");
		} else { // End of string after a value, but no closing brace
             throw JsonParseException("Unexpected end of object definition, missing '}'");
        }
	}
	if (index >= json_string.length() || json_string[index] != '}') { // Check if loop exited due to end of string
        throw JsonParseException("Unexpected end of object definition or invalid character instead of '}'");
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
	if (index < json_string.length() && json_string[index] == ']') { // Handle empty array
        ExpectChar(json_string, index, ']');
        return array;
    }
	while(index < json_string.length() && json_string[index] != ']') // Corrected loop condition
	{
		if (index >= json_string.length()) throw JsonParseException("Unexpected end of array definition");
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
                if (index >= json_string.length() || json_string[index] == ']') // Trailing comma or end after comma
                     throw JsonParseException("Trailing comma or unexpected end after comma in array");
			}
			else if (c == ']')
				break;
			else throw JsonParseException("Expected ',' or ']' while parsing array");
		} else { // End of string after a value, but no closing bracket
            throw JsonParseException("Unexpected end of array definition, missing ']'");
        }
	}
    if (index >= json_string.length() || json_string[index] != ']') { // Check if loop exited due to end of string
        throw JsonParseException("Unexpected end of array definition or invalid character instead of ']'");
    }
	ExpectChar(json_string, index, ']');
	return array;
}

double JsonParser::ParseNumber(const std::string& json_string, size_t& index)
{
	SkipWhitespace(json_string, index);
    size_t start_index = index;
    bool negative = false;

    // Handle sign
    if (index < json_string.length() && json_string[index] == '-') {
        negative = true;
        index++;
    }

    // Check for just a '-' without digits
    if (index == start_index + 1 && negative && (index >= json_string.length() || !IsDigit(json_string[index]))) {
        throw JsonParseException("Invalid number format: '-' must be followed by digits.");
    }
    if (index >= json_string.length() || (!IsDigit(json_string[index]) && json_string[index] != '.')) {
         // Case where input is just "-" or starts with non-digit after sign
        if (index == start_index && (index >= json_string.length() || !IsDigit(json_string[index]))) {
             throw JsonParseException("Invalid character at start of number.");
        }
    }


    std::string num_str;
    // Integer part
    while (index < json_string.length() && IsDigit(json_string[index])) {
        num_str += json_string[index];
        index++;
    }

    // Check for leading zeros (e.g. "01", but "0" or "0.1" is fine)
    if (num_str.length() > 1 && num_str[0] == '0') {
         if (num_str.find('.') == std::string::npos) { // No decimal point found yet
            throw JsonParseException("Leading zeros are not allowed in numbers.");
        }
    }
     if (num_str.empty() && (index < json_string.length() && json_string[index] != '.')) {
        // This can happen if we only had a '-' and no digits followed.
        // Or if the first char is not a digit and not '.' (e.g. "abc") - already caught by ParseValue's default case
        throw JsonParseException("Number format error: missing integer part before potential fraction or exponent.");
    }


    // Fractional part
    if (index < json_string.length() && json_string[index] == '.') {
        num_str += json_string[index];
        index++;
        if (index >= json_string.length() || !IsDigit(json_string[index])) {
            throw JsonParseException("Decimal point must be followed by at least one digit.");
        }
        while (index < json_string.length() && IsDigit(json_string[index])) {
            num_str += json_string[index];
            index++;
        }
    }

    // Exponent part
    if (index < json_string.length() && (json_string[index] == 'e' || json_string[index] == 'E')) {
        num_str += json_string[index];
        index++;
        if (index < json_string.length() && (json_string[index] == '+' || json_string[index] == '-')) {
            num_str += json_string[index];
            index++;
        }
        if (index >= json_string.length() || !IsDigit(json_string[index])) {
            throw JsonParseException("Exponent 'e' or 'E' must be followed by at least one digit (after optional sign).");
        }
        while (index < json_string.length() && IsDigit(json_string[index])) {
            num_str += json_string[index];
            index++;
        }
    }
    
    // Make sure something was actually parsed as part of the number's magnitude
    if (num_str.empty() || (num_str.length() == 1 && (num_str[0] == '.' || num_str[0] == 'e' || num_str[0] == 'E' || num_str[0] == '+' ))) { // also check for num_str == "-" which is covered by negative flag
         if (negative && num_str.empty()) { // only a '-' was found
             throw JsonParseException("Invalid number: stranded '-' sign.");
         }
        throw JsonParseException("No valid numeric characters found for number parsing.");
    }
     if (num_str.back() == 'e' || num_str.back() == 'E' || num_str.back() == '+' || num_str.back() == '-') {
        throw JsonParseException("Number ends with incomplete exponent part.");
    }


    try {
        double value = std::stod(num_str);
        return negative ? -value : value;
    } catch (const std::out_of_range& oor) {
        throw JsonParseException("Number out of range for double precision.");
    } catch (const std::invalid_argument& ia) {
        // This should ideally be caught by prior checks, but as a fallback:
        throw JsonParseException("Invalid argument for number conversion (stod): " + num_str);
    }
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
    if (index + expected_string.length() > json_string.length()) {
        throw JsonParseException("Unexpected end of input while expecting string: " + expected_string);
    }
	for(char c: expected_string)
	{
		if(json_string[index] != c) // No need to check index >= json_string.length() due to above check
		{
			throw JsonParseException("Expected string '" + expected_string + "' but found different character starting at index " + std::to_string(index));
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
	// SkipWhitespace should be called by the parser logic before attempting to parse a value,
    // not at the beginning of ParseValue itself, to allow for leading whitespace in the overall JSON string.
    // SkipComment should ideally also be handled by a top-level parsing loop or before each value.
    // For now, Parse() calls SkipComment once at the beginning.
    while (index < json_string.length()) {
        // SkipWhitespace(json_string, index); // Moved to be called before each ParseValue attempt by caller or main loop
        bool comment_found = false;
        if (index + 1 < json_string.length()) {
            if (json_string[index] == '/' && json_string[index + 1] == '/') {
                index += 2; // Skip '//'
                while (index < json_string.length() && json_string[index] != '\n') {
                    ++index;
                }
                if (index < json_string.length() && json_string[index] == '\n') {
                    ++index; // Skip '\n'
                }
                continue; // Check for more comments or content
            } else if (json_string[index] == '/' && json_string[index + 1] == '*') {
                index += 2; // Skip '/*'
                while (index + 1 < json_string.length() && !(json_string[index] == '*' && json_string[index + 1] == '/')) {
                    ++index;
                }
                if (index + 1 < json_string.length()) {
                    index += 2; // Skip '*/'
                } else {
                    throw JsonParseException("Unterminated multi-line comment");
                }
                continue; // Check for more comments or content
            }
        }
        break; // No comment found at current position
    }
}