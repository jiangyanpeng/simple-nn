#ifndef SIMPLE_NN_LAYEAR_REGISTER_H_
#define SIMPLE_NN_LAYEAR_REGISTER_H_

#include "runtime/layer/linear.h"
#include "runtime/layer/source.h"

#include <map>
#include <register.h>
using namespace nn;

REGISTER_COMMON_ENGINE(nn, Source, Layer, Source)
REGISTER_COMMON_ENGINE(nn, Linear, Layer, Linear)

// clang-format off
static const std::multimap<std::string, std::string> layer_map{
    {"pnnx.Input", "Source"}, {"pnnx.Output", "Source"},
    {"nn.Linear", "Linear"}};

#endif // SIMPLE_NN_LAYEAR_REGISTER_H_
