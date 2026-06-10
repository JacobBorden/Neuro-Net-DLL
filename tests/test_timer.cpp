#include <gtest/gtest.h>
#include "utilities/timer.h"
#include <thread>
#include <chrono>

using namespace utilities;

TEST(TimerTest, InitialState) {
    Timer timer;
    EXPECT_EQ(timer.elapsed_milliseconds(), 0);
    EXPECT_EQ(timer.elapsed_microseconds(), 0);
}

TEST(TimerTest, BasicTiming) {
    Timer timer;
    timer.start();
    std::this_thread::sleep_for(std::chrono::milliseconds(50));
    timer.stop();

    long long elapsed = timer.elapsed_milliseconds();
    EXPECT_GE(elapsed, 45);
}

TEST(TimerTest, MicrosecondsTiming) {
    Timer timer;
    timer.start();
    std::this_thread::sleep_for(std::chrono::microseconds(50000));
    timer.stop();

    long long elapsed = timer.elapsed_microseconds();
    EXPECT_GE(elapsed, 45000);
}

TEST(TimerTest, ElapsedWhileRunning) {
    Timer timer;
    timer.start();
    std::this_thread::sleep_for(std::chrono::milliseconds(50));

    long long elapsed = timer.elapsed_milliseconds();
    EXPECT_GE(elapsed, 45);

    timer.stop();
}

TEST(TimerTest, RestartTimer) {
    Timer timer;

    timer.start();
    std::this_thread::sleep_for(std::chrono::milliseconds(20));
    timer.stop();

    long long first_elapsed = timer.elapsed_milliseconds();

    timer.start();
    std::this_thread::sleep_for(std::chrono::milliseconds(50));
    timer.stop();

    long long second_elapsed = timer.elapsed_milliseconds();

    EXPECT_GE(first_elapsed, 15);
    EXPECT_GE(second_elapsed, 45);
}
