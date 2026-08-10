#ifndef TIMER_H
#define TIMER_H

#include <chrono>
#include <iomanip>
#include <sstream>
#include <string>

namespace utilities {

inline std::string get_current_time_string() {
    auto now = std::chrono::system_clock::now();
    auto in_time_t = std::chrono::system_clock::to_time_t(now);
    std::tm buf;
#ifdef _WIN32
    localtime_s(&buf, &in_time_t);
#else
    localtime_r(&in_time_t, &buf);
#endif
    std::ostringstream ss;
    ss << std::put_time(&buf, "%Y-%m-%dT%H:%M:%S%z");
    return ss.str();
}

class Timer {
public:
    Timer() : start_time_(), end_time_(), running_(false) {}

    void start() {
        start_time_ = std::chrono::high_resolution_clock::now();
        running_ = true;
    }

    void stop() {
        end_time_ = std::chrono::high_resolution_clock::now();
        running_ = false;
    }

    long long elapsed_milliseconds() const {
        std::chrono::time_point<std::chrono::high_resolution_clock> end = 
            running_ ? std::chrono::high_resolution_clock::now() : end_time_;
        return std::chrono::duration_cast<std::chrono::milliseconds>(end - start_time_).count();
    }

    long long elapsed_microseconds() const {
        std::chrono::time_point<std::chrono::high_resolution_clock> end = 
            running_ ? std::chrono::high_resolution_clock::now() : end_time_;
        return std::chrono::duration_cast<std::chrono::microseconds>(end - start_time_).count();
    }

private:
    std::chrono::time_point<std::chrono::high_resolution_clock> start_time_;
    std::chrono::time_point<std::chrono::high_resolution_clock> end_time_;
    bool running_;
};

} // namespace utilities

#endif // TIMER_H
