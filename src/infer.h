#ifndef SIMPLE_NN_INFER_H_
#define SIMPLE_NN_INFER_H_

#include <common.h>
#include <map>
#include <memory>
#include <string>
#include <tensor/tensor.h>
#include <vector>

#include "loader/model_load.h"

namespace nn {
class InferBase {
public:
    using NNTensorPtr       = std::shared_ptr<base::Tensor>;
    using NNModelPackagePtr = std::shared_ptr<ModelPackage>;

public:
    class InferContext {
    public:
        InferContext() {}
        virtual ~InferContext() {}
    };

    virtual MStatus Init(const std::string& path, const ModelConfig& config)          = 0;
    virtual MStatus Init(NNModelPackagePtr model_resource, const ModelConfig& config) = 0;
    virtual uint32_t GetInputNum() const                                              = 0;
    virtual uint32_t GetOutputNum() const                                             = 0;

    virtual MStatus Run(std::vector<NNTensorPtr>& input,
                        std::vector<NNTensorPtr>& output,
                        const char* start = nullptr,
                        const char* end   = nullptr) = 0;
    virtual MStatus Run(void* stream,
                        std::vector<NNTensorPtr>& input,
                        std::shared_ptr<InferBase::InferContext>& infer_cxt,
                        const char* start = nullptr,
                        const char* end   = nullptr);
    virtual MStatus SetInputsShapeSize(const std::vector<uint32_t>& sizes);

    virtual MStatus SetOutputsShapeSize(const std::vector<uint32_t>& sizes);

    virtual MStatus SetTensorByName(const NNTensorPtr& tensor, const std::string& name);

    virtual MStatus GetTensorByName(NNTensorPtr& tensor, const std::string& name);

    virtual MStatus SetInputs(std::vector<NNTensorPtr>& tensors);


    /// @brief get the four-dimensional (nchw) of the idx-th input blob
    /// @param[in] idx blob的index
    virtual std::vector<uint32_t> GetInputDims(uint32_t idx) const = 0;

    /// @brief get the four-dimensional (nchw) of the idx-th output blob
    /// @param[in] idx blob的index
    virtual std::vector<uint32_t> GetOutputDims(uint32_t idx) const = 0;

protected:
    void SetModelName(const std::string& name) { model_name_ = name; }
    std::string model_name_{};
    std::vector<std::string> output_layer_name_{};

    std::shared_ptr<NNModel> row_model_{nullptr};
};

using InferBasePtr    = std::shared_ptr<InferBase>;
using InferContextPtr = std::shared_ptr<InferBase::InferContext>;

} // namespace nn
#endif // SIMPLE_NN_INFER_H_