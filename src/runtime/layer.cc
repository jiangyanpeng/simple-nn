#include "runtime/layer.h"

#include <log.h>

namespace nn {

MStatus Layer::Init(const std::map<std::string, pnnx::Parameter>& params) {
    SIMPLE_LOG_DEBUG("{} Layer::Load Start\n", name_);
    SIMPLE_LOG_DEBUG("{} Layer::Load End\n", name_);
    return MStatus::M_OK;
}

MStatus Layer::Forward(const std::vector<TensorPtr>& input, std::vector<TensorPtr>& output) {
    SIMPLE_LOG_DEBUG("{} Layer::Load Start\n", name_);
    SIMPLE_LOG_DEBUG("{} Layer::Load End\n", name_);
    return MStatus::M_NOT_SUPPORT;
}

} // namespace nn
