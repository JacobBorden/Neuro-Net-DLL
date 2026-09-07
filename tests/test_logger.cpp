#include <gtest/gtest.h>
#include "../src/utilities/logger.h"
#include <sstream>

TEST(LoggerTest, DebugLevelOutput) {
    std::ostringstream oss;
    ::NeuroNet::Logger::SetStream(&oss);
    ::NeuroNet::Logger::SetLevel(::NeuroNet::Logger::Level::DEBUG);

    ::NeuroNet::Logger::Debug("Test ", 1, " debug");
    EXPECT_EQ(oss.str(), "[DEBUG] Test 1 debug\n");

    ::NeuroNet::Logger::SetStream(&std::cout);
}

TEST(LoggerTest, InfoLevelOutput) {
    std::ostringstream oss;
    ::NeuroNet::Logger::SetStream(&oss);
    ::NeuroNet::Logger::SetLevel(::NeuroNet::Logger::Level::INFO);

    ::NeuroNet::Logger::Debug("Should not print");
    ::NeuroNet::Logger::Info("Test ", 2, " info");
    EXPECT_EQ(oss.str(), "[INFO] Test 2 info\n");

    ::NeuroNet::Logger::SetStream(&std::cout);
}

TEST(LoggerTest, WarningLevelOutput) {
    std::ostringstream oss;
    ::NeuroNet::Logger::SetStream(&oss);
    ::NeuroNet::Logger::SetLevel(::NeuroNet::Logger::Level::WARNING);

    ::NeuroNet::Logger::Debug("Should not print");
    ::NeuroNet::Logger::Info("Should not print");
    ::NeuroNet::Logger::Warning("Test ", 3, " warning");
    EXPECT_EQ(oss.str(), "[WARNING] Test 3 warning\n");

    ::NeuroNet::Logger::SetStream(&std::cout);
}

TEST(LoggerTest, ErrorLevelOutput) {
    std::ostringstream oss;
    ::NeuroNet::Logger::SetStream(&oss);
    ::NeuroNet::Logger::SetLevel(::NeuroNet::Logger::Level::ERROR);

    ::NeuroNet::Logger::Debug("Should not print");
    ::NeuroNet::Logger::Info("Should not print");
    ::NeuroNet::Logger::Warning("Should not print");
    ::NeuroNet::Logger::Error("Test ", 4, " error");
    EXPECT_EQ(oss.str(), "[ERROR] Test 4 error\n");

    ::NeuroNet::Logger::SetStream(&std::cout);
}

TEST(LoggerTest, NoneLevelOutput) {
    std::ostringstream oss;
    ::NeuroNet::Logger::SetStream(&oss);
    ::NeuroNet::Logger::SetLevel(::NeuroNet::Logger::Level::NONE);

    ::NeuroNet::Logger::Debug("Should not print");
    ::NeuroNet::Logger::Info("Should not print");
    ::NeuroNet::Logger::Warning("Should not print");
    ::NeuroNet::Logger::Error("Should not print");
    EXPECT_EQ(oss.str(), "");

    ::NeuroNet::Logger::SetStream(&std::cout);
}
