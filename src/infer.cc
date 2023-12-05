#include "infer.h"

#include <log.h>

namespace nn {
MStatus InferBase::SetInputsShapeSize(const std::vector<uint32_t>& sizes) {
    SIMPLE_LOG_ERROR("InferBase::SetInputsShapeSize not implement");
    return MStatus::M_FAILED;
}
MStatus InferBase::SetOutputsShapeSize(const std::vector<uint32_t>& sizes) {
    SIMPLE_LOG_ERROR("InferBase::SetOutputsShapeSize not implement");
    return MStatus::M_FAILED;
}

MStatus InferBase::SetTensorByName(const NNTensorPtr& tensor, const std::string& name) {
    SIMPLE_LOG_ERROR("InferBase::SetTensorByName not implement");
    return MStatus::M_FAILED;
}

MStatus InferBase::GetTensorByName(NNTensorPtr& tensor, const std::string& name) {
    SIMPLE_LOG_ERROR("InferBase::GetTensorByName not implement");
    return MStatus::M_FAILED;
}

MStatus InferBase::SetInputs(std::vector<NNTensorPtr>& tensors) {
    SIMPLE_LOG_ERROR("InferBase::SetInputs not implement");
    return MStatus::M_FAILED;
}

MStatus InferBase::Run(void* stream,
                       std::vector<NNTensorPtr>& input,
                       std::shared_ptr<InferBase::InferContext>& infer_cxt,
                       const char* start,
                       const char* end) {
    SIMPLE_LOG_ERROR("InferBase::Run not implement");
    return MStatus::M_FAILED;
}

} // namespace nn
