#pragma once

#include <iostream>

namespace NeuroNet {

    class Logger {
    public:
        enum class Level {
            DEBUG,
            INFO,
            WARNING,
            ERROR,
            NONE
        };

        static void SetLevel(Level level);
        static Level GetLevel();

        class LogStream {
        public:
            LogStream(Level level) : level_(level) {}

            // Allow copying
            LogStream(const LogStream&) = default;

            template<typename T>
            LogStream& operator<<(const T& value) {
                if (level_ >= Logger::GetLevel()) {
                    if (level_ == Level::ERROR) {
                        std::cerr << value;
                    } else {
                        std::cout << value;
                    }
                }
                return *this;
            }

            // Handle std::endl and other I/O manipulators
            LogStream& operator<<(std::ostream& (*manip)(std::ostream&)) {
                if (level_ >= Logger::GetLevel()) {
                    if (level_ == Level::ERROR) {
                        manip(std::cerr);
                    } else {
                        manip(std::cout);
                    }
                }
                return *this;
            }

        private:
            Level level_;
        };

        static LogStream Debug() { return LogStream(Level::DEBUG); }
        static LogStream Info() { return LogStream(Level::INFO); }
        static LogStream Warning() { return LogStream(Level::WARNING); }
        static LogStream Error() { return LogStream(Level::ERROR); }

    private:
        static Level current_level;
    };

} // namespace NeuroNet
