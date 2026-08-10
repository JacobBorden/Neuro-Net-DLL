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
