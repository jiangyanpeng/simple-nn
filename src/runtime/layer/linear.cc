#include "runtime/layer/linear.h"

namespace nn {

MStatus Linear::Init(const std::shared_ptr<LayerParam>& param,
                     const std::shared_ptr<ModelBin>& bin) {
    return MStatus::M_OK;
}

MStatus Linear::Forward(const std::vector<TensorPtr>& input, std::vector<TensorPtr>& output) {
    return MStatus::M_OK;
}
} // namespace nn
