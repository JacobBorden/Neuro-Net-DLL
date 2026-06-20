#include <gtest/gtest.h>
#include "../src/utilities/logger.h"
#include <sstream>

class LoggerTest : public ::testing::Test {
protected:
    void SetUp() override {
        // Save original buffer
        old_cout = std::cout.rdbuf(cout_ss.rdbuf());
        old_cerr = std::cerr.rdbuf(cerr_ss.rdbuf());
        NeuroNet::Logger::SetLevel(NeuroNet::LogLevel::INFO);
    }

    void TearDown() override {
        // Restore original buffer
        std::cout.rdbuf(old_cout);
        std::cerr.rdbuf(old_cerr);
    }

    std::stringstream cout_ss;
    std::stringstream cerr_ss;
    std::streambuf* old_cout;
    std::streambuf* old_cerr;
};

TEST_F(LoggerTest, DefaultLevelIsInfo) {
    EXPECT_EQ(NeuroNet::Logger::GetLevel(), NeuroNet::LogLevel::INFO);
}

TEST_F(LoggerTest, DebugLogIgnoredAtInfoLevel) {
    NeuroNet::Logger::Debug("test debug");
    EXPECT_EQ(cout_ss.str(), "");
}

TEST_F(LoggerTest, InfoLogPrintedAtInfoLevel) {
    NeuroNet::Logger::Info("test info");
    EXPECT_EQ(cout_ss.str(), "[INFO] test info\n");
}

TEST_F(LoggerTest, ErrorLogPrintedToCerr) {
    NeuroNet::Logger::Error("test error");
    EXPECT_EQ(cerr_ss.str(), "[ERROR] test error\n");
    EXPECT_EQ(cout_ss.str(), "");
}

TEST_F(LoggerTest, ChangeLevelToDebug) {
    NeuroNet::Logger::SetLevel(NeuroNet::LogLevel::DEBUG);
    NeuroNet::Logger::Debug("test debug");
    EXPECT_EQ(cout_ss.str(), "[DEBUG] test debug\n");
}

TEST_F(LoggerTest, ChangeLevelToNone) {
    NeuroNet::Logger::SetLevel(NeuroNet::LogLevel::NONE);
    NeuroNet::Logger::Error("test error");
    EXPECT_EQ(cerr_ss.str(), "");
}
