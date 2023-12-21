#include "source.h"

namespace nn {

MStatus Source::Init(const std::map<std::string, pnnx::Parameter>& params) {
    return MStatus::M_OK;
}

MStatus Source::Forward(const std::vector<TensorPtr>& input, std::vector<TensorPtr>& output) {
    output = input;
    return MStatus::M_OK;
}
} // namespace nn
