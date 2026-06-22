#include "logger.h"

namespace NeuroNet {
    std::atomic<Logger::Level> Logger::currentLevel{Logger::Level::INFO};
    std::ostream* Logger::outputStream = &std::cout;
    std::mutex Logger::logMutex;

    void Logger::SetLevel(Level level) {
        currentLevel.store(level, std::memory_order_relaxed);
    }

    void Logger::SetStream(std::ostream* os) {
        std::lock_guard<std::mutex> lock(logMutex);
        outputStream = os;
    }
}
