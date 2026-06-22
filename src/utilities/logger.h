#pragma once

#include <atomic>
#include <iostream>
#include <mutex>

namespace NeuroNet {

    class Logger {
    public:
        enum class Level {
            DEBUG = 0,
            INFO,
            WARNING,
            ERROR,
            NONE
        };

        static void SetLevel(Level level);
        static void SetStream(std::ostream* os);

        template <typename... Args>
        static void Debug(Args... args) {
            if (currentLevel.load(std::memory_order_relaxed) <= Level::DEBUG) {
                Log("[DEBUG] ", args...);
            }
        }

        template <typename... Args>
        static void Info(Args... args) {
            if (currentLevel.load(std::memory_order_relaxed) <= Level::INFO) {
                Log("[INFO] ", args...);
            }
        }

        template <typename... Args>
        static void Warning(Args... args) {
            if (currentLevel.load(std::memory_order_relaxed) <= Level::WARNING) {
                Log("[WARNING] ", args...);
            }
        }

        template <typename... Args>
        static void Error(Args... args) {
            if (currentLevel.load(std::memory_order_relaxed) <= Level::ERROR) {
                Log("[ERROR] ", args...);
            }
        }

    private:
        static std::atomic<Level> currentLevel;
        static std::ostream* outputStream;
        static std::mutex logMutex;

        static void Print() {
            if (outputStream) {
                *outputStream << std::endl;
            }
        }

        template <typename T, typename... Args>
        static void Print(const T& first, const Args&... args) {
            if (outputStream) {
                *outputStream << first;
                Print(args...);
            }
        }

        template <typename... Args>
        static void Log(const char* prefix, const Args&... args) {
            std::lock_guard<std::mutex> lock(logMutex);
            if (outputStream) {
                *outputStream << prefix;
                Print(args...);
            }
        }
    };

}
