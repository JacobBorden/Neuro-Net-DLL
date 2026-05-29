#pragma once
#ifndef _JSON_ERROR_
#define _JSON_ERROR_
#include <exception>
#include <string>

class JsonParseException : public std::exception
{
	public:
		JsonParseException(const std::string& message): message_(message){}
		const char* what() const noexcept override { return message_.c_str();}
	private:
		std::string message_;
};

#endif