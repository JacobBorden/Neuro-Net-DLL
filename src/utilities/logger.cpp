#include "logger.h"
#include <chrono>
#include <ctime>
#include <iomanip>
#include <sstream>

namespace NeuroNet {

Logger::Logger() : m_Level(Level::INFO), m_LogToConsole(true) {}

Logger::~Logger() {
    if (m_FileStream && m_FileStream->is_open()) {
        m_FileStream->close();
    }
}

Logger& Logger::GetInstance() {
    static Logger instance;
    return instance;
}

void Logger::SetLevel(Level level) {
    std::lock_guard<std::mutex> lock(m_Mutex);
    m_Level = level;
}

Logger::Level Logger::GetLevel() const {
    return m_Level;
}

void Logger::SetOutputFile(const std::string& filename) {
    std::lock_guard<std::mutex> lock(m_Mutex);
    if (m_FileStream && m_FileStream->is_open()) {
        m_FileStream->close();
    }
    if (filename.empty()) {
        m_FileStream.reset();
        return;
    }
    m_FileStream = std::make_unique<std::ofstream>(filename, std::ios::app);
    if (!m_FileStream->is_open()) {
        std::cerr << "Failed to open log file: " << filename << std::endl;
        m_FileStream.reset();
    }
}

void Logger::CloseOutputFile() {
    std::lock_guard<std::mutex> lock(m_Mutex);
    if (m_FileStream && m_FileStream->is_open()) {
        m_FileStream->close();
    }
    m_FileStream.reset();
}

void Logger::LogToConsole(bool enable) {
    std::lock_guard<std::mutex> lock(m_Mutex);
    m_LogToConsole = enable;
}

std::string Logger::LevelToString(Level level) const {
    switch (level) {
        case Level::DEBUG:   return "DEBUG";
        case Level::INFO:    return "INFO";
        case Level::WARNING: return "WARN";
        case Level::ERROR:   return "ERROR";
        default:             return "UNKNOWN";
    }
}

void Logger::Log(Level level, const std::string& message, const char* file, int line) {
    if (level < m_Level || level == Level::NONE) {
        return;
    }

    std::lock_guard<std::mutex> lock(m_Mutex);

    auto now = std::chrono::system_clock::now();
    auto time_t_now = std::chrono::system_clock::to_time_t(now);
    std::tm tm_now;
#ifdef _WIN32
    localtime_s(&tm_now, &time_t_now);
#else
    localtime_r(&time_t_now, &tm_now);
#endif

    std::stringstream ss;
    ss << "[" << std::put_time(&tm_now, "%Y-%m-%d %H:%M:%S") << "] "
       << "[" << LevelToString(level) << "] ";

    if (file != nullptr) {
        // Optionally extract just the filename from the path
        std::string file_str(file);
        size_t last_slash = file_str.find_last_of("/\\");
        if (last_slash != std::string::npos) {
            file_str = file_str.substr(last_slash + 1);
        }
        ss << "[" << file_str << ":" << line << "] ";
    }

    ss << message;

    if (m_LogToConsole) {
        std::ostream& out = (level == Level::ERROR || level == Level::WARNING) ? std::cerr : std::cout;
        out << ss.str() << std::endl;
    }

    if (m_FileStream && m_FileStream->is_open()) {
        *m_FileStream << ss.str() << std::endl;
    }
}

void Logger::Debug(const std::string& message, const char* file, int line) {
    Log(Level::DEBUG, message, file, line);
}

void Logger::Info(const std::string& message, const char* file, int line) {
    Log(Level::INFO, message, file, line);
}

void Logger::Warning(const std::string& message, const char* file, int line) {
    Log(Level::WARNING, message, file, line);
}

void Logger::Error(const std::string& message, const char* file, int line) {
    Log(Level::ERROR, message, file, line);
}

} // namespace NeuroNet
