#include "logger.h"

namespace NeuroNet {

    Logger::Level Logger::current_level = Logger::Level::INFO;

    void Logger::SetLevel(Level level) {
        current_level = level;
    }

    Logger::Level Logger::GetLevel() {
        return current_level;
    }

} // namespace NeuroNet
