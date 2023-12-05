#ifndef SIMPLE_NN_QNN_INFER_H_
#define SIMPLE_NN_QNN_INFER_H_

#include "infer.h"
#include "wrapper/qnn_wrapper.h"

namespace nn {
class InferQnn : public InferBase {
public:
    typedef struct QNNContext {
        bool is_use_signed_pd;
        bool is_use_vndk;

        const char* system_lib_path;
        unsigned int system_lib_path_len;

        const char* backend_lib_path;
        unsigned int backend_lib_path_len;
    } QNNContext;

public:
    InferQnn();
    ~InferQnn();

    MStatus Init(const std::string& path, const ModelConfig& config) override;
    MStatus Init(NNModelPackagePtr model_resource, const ModelConfig& config) override;

    MStatus Run(std::vector<NNTensorPtr>& input,
                std::vector<NNTensorPtr>& output,
                const char* start = nullptr,
                const char* end   = nullptr) override;

    uint32_t GetInputNum() const override;
    uint32_t GetOutputNum() const override;

    /// @brief get the four-dimensional (nchw) of the idx-th input blob
    /// @param[in] idx blob的index
    std::vector<uint32_t> GetInputDims(uint32_t idx) const override;

    /// @brief get the four-dimensional (nchw) of the idx-th output blob
    /// @param[in] idx blob的index
    std::vector<uint32_t> GetOutputDims(uint32_t idx) const override;

private:
    void SetConfig(const ModelConfig& config);
    MStatus ReArrangeInput(std::vector<NNTensorPtr>& input);
    MStatus ReArrangeOutput(std::vector<NNTensorPtr>& output);
    MStatus CreateNetOutput(const int batch_size, std::vector<NNTensorPtr>& output);
    MStatus CheckInputShape(std::vector<NNTensorPtr>& input);
    MStatus RunSingleBatch(std::vector<NNTensorPtr>& input,
                           std::vector<NNTensorPtr>& output,
                           uint32_t batch);

private:
    std::unique_ptr<wrap::QnnWrapperV1> qnn_wrapper_ptr_;
    std::vector<NNTensorPtr> output_tensors_;
    std::vector<std::vector<uint32_t>> output_layer_dims_;
    std::string backend_lib_path_;
    std::string system_lib_path_;
    bool is_use_signed_pd_;
    bool is_use_vndk_;
};
} // namespace nn

#endif // SIMPLE_NN_INFER_QNN_H_