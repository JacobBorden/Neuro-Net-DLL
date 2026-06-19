#include <gtest/gtest.h>
#include "../src/utilities/logger.h"
#include <fstream>
#include <string>

using namespace NeuroNet;

class LoggerTest : public ::testing::Test {
protected:
    void SetUp() override {
        Logger::GetInstance().SetLevel(Logger::Level::DEBUG);
        Logger::GetInstance().LogToConsole(false); // Prevent cluttering test output
    }
    void TearDown() override {
        // Reset to some default
        Logger::GetInstance().SetLevel(Logger::Level::INFO);
        Logger::GetInstance().LogToConsole(true);
    }
};

TEST_F(LoggerTest, TestSingletonInstance) {
    Logger& logger1 = Logger::GetInstance();
    Logger& logger2 = Logger::GetInstance();
    EXPECT_EQ(&logger1, &logger2);
}

TEST_F(LoggerTest, TestLevelFiltering) {
    Logger::GetInstance().SetLevel(Logger::Level::WARNING);

    // Set a file to check output
    std::string test_log_file = "test_log_filtering.txt";
    std::remove(test_log_file.c_str());
    Logger::GetInstance().SetOutputFile(test_log_file);

    LOG_DEBUG("This is a debug message");
    LOG_INFO("This is an info message");
    LOG_WARN("This is a warning message");
    LOG_ERROR("This is an error message");

    // Close by setting empty or a different file
    Logger::GetInstance().CloseOutputFile();

    std::ifstream infile(test_log_file);
    std::string line;
    int line_count = 0;
    while (std::getline(infile, line)) {
        line_count++;
    }
    infile.close();
    std::remove(test_log_file.c_str());

    // Only WARN and ERROR should be logged
    EXPECT_EQ(line_count, 2);
}
