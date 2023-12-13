#include "source.h"

namespace nn {

MStatus Source::Init(const std::shared_ptr<LayerParam>& param,
                     const std::shared_ptr<ModelBin>& bin) {
    return MStatus::M_OK;
}

MStatus Source::Forward(const std::vector<TensorPtr>& input, std::vector<TensorPtr>& output) {
    output = input;
    return MStatus::M_OK;
}
} // namespace nn
