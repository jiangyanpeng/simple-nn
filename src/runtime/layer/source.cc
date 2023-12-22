#include "source.h"

namespace nn {

MStatus Source::Init(const std::map<std::string, pnnx::Parameter>& params,
                     const std::map<std::string, pnnx::Attribute>& attrs) {
    return MStatus::M_OK;
}

MStatus Source::Forward(const TensorPtr& input, TensorPtr& output) {
    output = input;
    return MStatus::M_OK;
}
} // namespace nn
