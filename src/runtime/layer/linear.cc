#include "runtime/layer/linear.h"

namespace nn {

MStatus Linear::Init(const std::map<std::string, pnnx::Parameter>& params) {
    for (const auto& param : params) {
        printf("%s\n", param.first.c_str());
    }
    return MStatus::M_OK;
}

MStatus Linear::Forward(const std::vector<TensorPtr>& input, std::vector<TensorPtr>& output) {
    return MStatus::M_OK;
}
} // namespace nn
