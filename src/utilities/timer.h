#ifndef TIMER_H
#define TIMER_H

#include <chrono>

namespace utilities {

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
