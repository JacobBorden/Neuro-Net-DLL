#include "logger.h"

namespace NeuroNet {
    Logger::Level Logger::currentLevel = Logger::Level::INFO;
    std::ostream* Logger::outputStream = &std::cout;
    std::mutex Logger::logMutex;

    void Logger::SetLevel(Level level) {
        currentLevel = level;
    }

    void Logger::SetStream(std::ostream* os) {
        std::lock_guard<std::mutex> lock(logMutex);
        outputStream = os;
    }
}
