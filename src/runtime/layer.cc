#include "runtime/layer.h"

#include <log.h>

namespace nn {

MStatus Layer::Init(const std::map<std::string, pnnx::Parameter>& params,
                    const std::map<std::string, pnnx::Attribute>& attrs) {
    SIMPLE_LOG_DEBUG("{} Layer::Load Start\n", name_);
    SIMPLE_LOG_DEBUG("{} Layer::Load End\n", name_);
    return MStatus::M_OK;
}

MStatus Layer::Forward(const TensorPtr& input, TensorPtr& output) {
    SIMPLE_LOG_DEBUG("{} Layer::Load Start\n", name_);
    SIMPLE_LOG_DEBUG("{} Layer::Load End\n", name_);
    return MStatus::M_NOT_SUPPORT;
}

} // namespace nn
