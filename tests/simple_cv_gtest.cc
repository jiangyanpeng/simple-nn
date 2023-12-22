#include "runtime/layer/linear.h"
#include "utils/test_util.h"

#include <fstream>
#include <gtest/gtest.h>
#include <unordered_map>

class SimpleNNTest : public ::testing::Test {
protected:
    void SetUp() override {
#ifdef CONFIG_SIMPLE_BASE_ENABLE_SPDLOG
#include <log.h>
        close_level();
#endif
        params["bias"]         = pnnx::Parameter(true);
        params["in_features"]  = pnnx::Parameter(in_count);
        params["out_features"] = pnnx::Parameter(out_count);

        bias.resize(out_count);
        weight.resize(out_count * in_count, 0);
        init_random<float>(bias.data(), out_count, 0, 1);
        init_random<float>(weight.data(), out_count * in_count, 0, 1);

        attrs["bias"]   = pnnx::Attribute({1, 1, 1, out_count}, bias);
        attrs["weight"] = pnnx::Attribute({1, 1, in_count, out_count}, weight);
    }
    void TearDown() override { Test::TearDown(); }

public:
    const int in_count = 512, out_count = 1000;
    std::vector<float> bias;
    std::vector<float> weight;
    std::map<std::string, pnnx::Parameter> params;
    std::map<std::string, pnnx::Attribute> attrs;
};


TEST_F(SimpleNNTest, pnnx_Parameter) {
    EXPECT_EQ(params.size(), 3);
    EXPECT_TRUE(params.count("bias"));
    EXPECT_TRUE(params.count("in_features"));
    EXPECT_TRUE(params.count("out_features"));
    EXPECT_EQ(params["bias"].type, 1);
    EXPECT_EQ(params["bias"].b, true);

    EXPECT_EQ(params["in_features"].type, 2);
    EXPECT_EQ(params["in_features"].i, in_count);
    EXPECT_EQ(params["out_features"].i, out_count);
}

TEST_F(SimpleNNTest, pnnx_Attribute) {
    EXPECT_EQ(attrs.size(), 2);
    EXPECT_TRUE(attrs.count("bias"));
    EXPECT_TRUE(attrs.count("weight"));
    EXPECT_EQ(bias.size(), attrs["bias"].elemcount());
    EXPECT_EQ(weight.size(), attrs["weight"].elemcount());

    auto attrs_bias = attrs["bias"].get_float32_data();
    for (int i = 0; i < 10; ++i) {
        EXPECT_NEAR(bias[i], attrs_bias[i], 0.0001);
    }

    auto attrs_weight = attrs["weight"].get_float32_data();
    for (int i = 0; i < weight.size(); ++i) {
        EXPECT_NEAR(weight[i], attrs_weight[i], 0.0001);
    }
}

TEST_F(SimpleNNTest, nn_Linear) {
    auto linear = std::make_shared<nn::Linear>();
}