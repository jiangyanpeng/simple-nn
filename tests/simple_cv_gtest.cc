#include "runtime/layer/linear.h"
#include "utils/test_util.h"

#include <fstream>
#include <gtest/gtest.h>

class SimpleNNTest : public ::testing::Test {
protected:
    void SetUp() override {
#ifdef CONFIG_SIMPLE_BASE_ENABLE_SPDLOG
#include <log.h>
        close_level();
#endif
    }
    void TearDown() override { Test::TearDown(); }
};


TEST_F(SimpleNNTest, nn_Linear) {
    auto linear = std::make_shared<nn::Linear>();
}