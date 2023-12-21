#include "infer_qnn.h"

#include "wrapper/qnn_wrapper.h"

#include <algorithm>
#include <mutex>

namespace nn {
class QnnWrapperAllocator : public wrap::QnnWrapper::Allocator {
public:
    QnnWrapperAllocator() {}
    virtual void* Malloc(size_t size) override;
    virtual void Free(void* p) override;
    virtual ~QnnWrapperAllocator();

private:
    std::map<void*, size_t> mem_map_;
    static size_t capacity_;
    std::mutex mtx_;
};

size_t QnnWrapperAllocator::capacity_ = 0L;

void* QnnWrapperAllocator::Malloc(size_t size) {
    std::lock_guard<std::mutex> lck(mtx_);
    void* data     = malloc(size);
    mem_map_[data] = size;
    capacity_ += size;
    return data;
}

void QnnWrapperAllocator::Free(void* p) {
    std::lock_guard<std::mutex> lck(mtx_);
    auto it = mem_map_.find(p);
    if (it != mem_map_.end()) {
        capacity_ -= it->second;
        free(it->first);
        mem_map_.erase(it);
    }
}

QnnWrapperAllocator::~QnnWrapperAllocator() {
    if (mem_map_.size() > 0) {
        for (auto it = mem_map_.begin(); it != mem_map_.end(); ++it) {
            SIMPLE_LOG_WARN(
                "WrapperAllocator unfreed, addr: %p, size: %i\n", it->first, it->second);
            free(it->first);
            capacity_ -= it->second;
        }
    }
}

InferQnn::InferQnn() {}

InferQnn::~InferQnn() {
    if (nullptr != qnn_wrapper_ptr_) {
        qnn_wrapper_ptr_ = nullptr;
    }
}

void InferQnn::SetConfig(const ModelConfig& config) {
    SIMPLE_LOG_DEBUG("InferQnn::SetConfig : %p\n", config.engine_context);
    is_use_vndk_ = false;
    if (config.engine_context) {
        QNNContext* ctx = reinterpret_cast<QNNContext*>(const_cast<void*>(config.engine_context));
        if (ctx->backend_lib_path && ctx->backend_lib_path_len) {
            backend_lib_path_ = std::string(ctx->backend_lib_path, ctx->backend_lib_path_len);
        }
        SIMPLE_LOG_DEBUG("ctx->backend_lib_path: %s,ctx->backend_lib_path_len: %i\n",
                         ctx->backend_lib_path,
                         ctx->backend_lib_path_len);
        if (ctx->system_lib_path && ctx->system_lib_path_len) {
            system_lib_path_ = std::string(ctx->system_lib_path, ctx->system_lib_path_len);
        }
        SIMPLE_LOG_DEBUG("ctx->system_lib_path: %s,ctx->system_lib_path_len: %i\n",
                         ctx->system_lib_path,
                         ctx->system_lib_path_len);
        is_use_signed_pd_ = ctx->is_use_signed_pd;
        SIMPLE_LOG_DEBUG("ctx->is_use_signed_pd: %i\n", ctx->is_use_signed_pd);
#if (defined __ARM_NEON) && ((defined __arm64__) || (defined __aarch64__))
        is_use_vndk_ = ctx->is_use_vndk;
#endif

    } else {
        is_use_signed_pd_ = false;
        backend_lib_path_ = std::string("libQnnHtp.so");
        system_lib_path_  = std::string("libQnnSystem.so");

#if CONFIG_NN_QNN_VERSION_1_2_0
        system_lib_path_ = std::string("");
#if (defined __ARM_NEON) && ((defined __arm64__) || (defined __aarch64__))
        backend_lib_path_ = std::string("libQnnHtpStub.so");
#endif
#endif
    }
}

MStatus InferQnn::Init(const std::string& path, const ModelConfig& config) {
    SIMPLE_LOG_DEBUG("InferQnn::Init Start, %s\n", this->model_name_.c_str());
    MStatus ret = MStatus::M_OK;
    do {
        output_layer_name_.clear();
        SetConfig(config);

        row_model_ = std::make_shared<NNModel>(path);
        if (row_model_->Size() <= 0) {
            SIMPLE_LOG_ERROR("{} model load failed\n", path.c_str());
            ret = MStatus::M_FILE_NOT_FOUND;
            break;
        }

        const uint8_t* data   = reinterpret_cast<const uint8_t*>(row_model_->Data().get());
        const size_t data_len = row_model_->Size();
        SIMPLE_LOG_DEBUG("InferQnn::Init model_buffer size %i\n", data_len);

        bool ret         = true;
        qnn_wrapper_ptr_ = std::unique_ptr<wrap::QnnWrapperV1>(new wrap::QnnWrapperV1());
        qnn_wrapper_ptr_->setAllocator(std::make_shared<QnnWrapperAllocator>());

        ret = qnn_wrapper_ptr_->initForDevice(
            backend_lib_path_, system_lib_path_, is_use_signed_pd_, is_use_vndk_);
        if (ret != true) {
            SIMPLE_LOG_ERROR("initForDevice failed\n");
            ret = MStatus::M_FAILED;
            break;
        }

        ret = qnn_wrapper_ptr_->createGraphsFromBinary(data, data_len);
        if (ret != true) {
            SIMPLE_LOG_ERROR(
                "qnn createGraphsFromBinary failed, data: [%p], data_len: [%i]\n", data, data_len);
            ret = MStatus::M_FAILED;
            break;
        }

        auto net_output_dims  = qnn_wrapper_ptr_->getOutputDims();
        auto net_output_names = qnn_wrapper_ptr_->getOutputNames();
        if (output_layer_name_.empty()) {
            output_layer_name_ = net_output_names;
        }
        output_layer_dims_.clear();

        for (size_t i = 0; i < output_layer_name_.size(); ++i) {
            size_t idx =
                std::find(net_output_names.begin(), net_output_names.end(), output_layer_name_[i]) -
                net_output_names.begin();
            if (idx >= net_output_names.size()) {
                SIMPLE_LOG_ERROR("output layer name not found %s\n", output_layer_name_[i].c_str());
                ret = MStatus::M_FAILED;
                break;
            }
            output_layer_dims_.emplace_back(
                std::vector<uint32_t>{static_cast<uint32_t>(net_output_dims[idx][0]),
                                      static_cast<uint32_t>(net_output_dims[idx][1]),
                                      static_cast<uint32_t>(net_output_dims[idx][2]),
                                      static_cast<uint32_t>(net_output_dims[idx][3])});
        }

        // SIMPLE_LOG_DEBUG(
        //     "output names: {}",
        //     std::accumulate(std::next(output_layer_name_.begin()),
        //                     output_layer_name_.end(),
        //                     output_layer_name_[0],
        //                     [](const std::string& a, const std::string& b) -> std::string {
        //                         return a + "," + b;
        //                     }));
    } while (0);
    SIMPLE_LOG_DEBUG("InferQnn::Init End, %s\n", this->model_name_.c_str());
    return ret;
}

MStatus InferQnn::Init(NNModelPackagePtr model_resource, const ModelConfig& config) {
    return MStatus::M_FAILED;
}

MStatus InferQnn::ReArrangeInput(std::vector<NNTensorPtr>& input) {
    SIMPLE_LOG_DEBUG("InferQnn::ReArrangeInput Start\n");
    auto ret = MStatus::M_OK;
    do {
        if (input.empty()) {
            SIMPLE_LOG_ERROR("net input empty\n");
            ret = MStatus::M_FAILED;
            break;
        }

        for (size_t i = 0; i < input.size(); ++i) {
            if (nullptr == input[i]) {
                SIMPLE_LOG_ERROR("net %i input is nulptr\n", i);
                ret = MStatus::M_FAILED;
                break;
            }

            if (input[i]->GetElemType() != M_DATA_TYPE_FLOAT32) {
                SIMPLE_LOG_ERROR("net %i input not support %s data type\n",
                                 i,
                                 DataTypeStr[input[i]->GetElemType()]);
                ret = MStatus::M_NOT_SUPPORT;
                break;
            }

            if (input[i]->GetMemType() != M_MEM_ON_CPU) {
                SIMPLE_LOG_ERROR("net %i input not support %s memory type\n",
                                 i,
                                 MemTypeStr[input[i]->GetMemType()]);
                ret = MStatus::M_NOT_SUPPORT;
                break;
            }

            if (TENSOR_SHAPE_MODE_NHWC != input[i]->GetShapeMode()) {
                // reshape
                if (nullptr == input[i]) {
                    SIMPLE_LOG_ERROR("%i reshape failed\n", i);
                    ret = MStatus::M_FAILED;
                    break;
                }
            }
        }
    } while (0);
    SIMPLE_LOG_DEBUG("InferQnn::ReArrangeInput End\n");
}

MStatus InferQnn::ReArrangeOutput(std::vector<NNTensorPtr>& output) {
    SIMPLE_LOG_DEBUG("InferQnn::ReArrangeOutput Start\n");
    auto ret = MStatus::M_OK;
    do {
        for (size_t i = 0; i < output.size(); ++i) {
            if (std::find(output_layer_name_.begin(),
                          output_layer_name_.end(),
                          output[i]->GetName()) != output_layer_name_.end()) {
                SIMPLE_LOG_ERROR("%s tensor not find in output_layer_name\n",
                                 output[i]->GetName().c_str());
                ret = MStatus::M_FILE_NOT_FOUND;
                break;
            }

            if (TENSOR_SHAPE_MODE_NCHW != output[i]->GetShapeMode()) {
                // reshape
                if (nullptr == output[i]) {
                    SIMPLE_LOG_ERROR("%i reshape failed\n", i);
                    ret = MStatus::M_FAILED;
                    break;
                }
            }
            break;
        }
    } while (0);
    SIMPLE_LOG_DEBUG("InferQnn::ReArrangeOutput End\n");
    return ret;
}

MStatus InferQnn::CreateNetOutput(const int batch_size, std::vector<NNTensorPtr>& output) {
    SIMPLE_LOG_DEBUG("InferQnn::CreateNetOutput Start\n");
    auto ret = MStatus::M_OK;
    do {
        auto net_output_names = qnn_wrapper_ptr_->getOutputNames();
        auto net_output_dims  = qnn_wrapper_ptr_->getOutputDims();
        if (net_output_names.empty() || net_output_dims.empty()) {
            SIMPLE_LOG_ERROR("qnn net engine not init\n");
            ret = MStatus::M_FAILED;
            break;
        }
        output.resize(net_output_dims.size());
        for (size_t i = 0; i < net_output_dims.size(); ++i) {
            std::vector<uint32_t> tensor_shape = {
                static_cast<uint32_t>(batch_size * net_output_dims[i][0]),
                static_cast<uint32_t>(net_output_dims[i][1]),
                static_cast<uint32_t>(net_output_dims[i][2]),
                static_cast<uint32_t>(net_output_dims[i][3])};

            output[i] = std::make_shared<base::Tensor>(
                tensor_shape, M_LAYOUT_NCHW, M_MEM_ON_CPU, M_DATA_TYPE_FLOAT32);
            output[i]->SetName(net_output_names[i]);
        }
    } while (0);
    SIMPLE_LOG_DEBUG("InferQnn::CreateNetOutput End\n");
    return ret;
}

MStatus InferQnn::CheckInputShape(std::vector<NNTensorPtr>& input) {
    SIMPLE_LOG_DEBUG("InferQnn::CheckInputShape Start\n");
    auto ret = MStatus::M_OK;
    do {
        auto qnn_input_num = GetInputNum();
        if (qnn_input_num != input.size()) {
            SIMPLE_LOG_ERROR(
                "input tensor num not equal qnn input: %ivs%i\n", input.size(), qnn_input_num);
            ret = MStatus::M_FAILED;
            break;
        }

        for (size_t i = 0; i < qnn_input_num; ++i) {
            auto net_dim = GetInputDims(static_cast<uint32_t>(i));

            bool n_eq = (input[i]->GetShape(0) % net_dim[0]) == 0;
            bool c_eq = net_dim[1] == input[i]->GetShape(1);
            bool h_eq = net_dim[2] == input[i]->GetShape(2);
            bool w_eq = net_dim[3] == input[i]->GetShape(3);
            if (!n_eq || !c_eq || !h_eq || !w_eq) {
                SIMPLE_LOG_ERROR("input shape not qnn match: [%i,%i,%i,%i]vs[%i,%i,%i,%i]\n",
                                 input[i]->GetShape(0),
                                 input[i]->GetShape(1),
                                 input[i]->GetShape(2),
                                 input[i]->GetShape(3),
                                 net_dim[0],
                                 net_dim[1],
                                 net_dim[2],
                                 net_dim[3]);
                ret = MStatus::M_FAILED;
                break;
            }
        }
    } while (0);
    SIMPLE_LOG_DEBUG("InferQnn::CheckInputShape End\n");
}

MStatus InferQnn::RunSingleBatch(std::vector<NNTensorPtr>& input,
                                 std::vector<NNTensorPtr>& output,
                                 uint32_t batch) {
    SIMPLE_LOG_DEBUG("InferQnn::RunSingleBatch Start\n");

    std::vector<uint8_t*> net_input_buffers(input.size(), nullptr);
    std::vector<size_t> net_input_buffers_len(input.size(), 0L);
    for (size_t i = 0; i < input.size(); ++i) {
        auto net_dim = GetInputDims(static_cast<uint32_t>(i));
        net_input_buffers_len[i] =
            net_dim[0] * net_dim[1] * net_dim[2] * net_dim[3] * sizeof(float);
        net_input_buffers[i] = input[i]->GetData<uint8_t>() + batch * net_input_buffers_len[i];
    }

    std::vector<uint8_t*> net_output_buffers(output.size(), nullptr);
    std::vector<size_t> net_output_buffers_len(output.size(), 0L);
    for (size_t i = 0; i < output.size(); ++i) {
        auto net_dim = output[i]->GetShape();
        net_output_buffers_len[i] =
            net_dim[0] * net_dim[1] * net_dim[2] * net_dim[3] * sizeof(float);
        net_output_buffers[i] = output[i]->GetData<uint8_t>() + batch * net_output_buffers_len[i];
    }

    auto input_shape_mode = input[0]->GetShapeMode();
    bool run_success      = true;
    auto ret              = MStatus::M_OK;
    do {
        if (input_shape_mode == std::string(TENSOR_SHAPE_MODE_NCHW)) {
            // run_success = qnn_wrapper_ptr_->populateInputTensors(net_input_buffers,
            //                                              net_input_buffers_len,
            //                                              QnnWrapper::DataType::FLOAT,
            //                                              QnnWrapper::DataLayout::LAYOUT_NCHW);
        } else {
            // run_success = qnn_wrapper_ptr_->populateInputTensors(net_input_buffers,
            //                                              net_input_buffers_len,
            //                                              QnnWrapper::DataType::FLOAT,
            //                                              QnnWrapper::DataLayout::LAYOUT_NHWC);
        }
        if (run_success != true) {
            ret = MStatus::M_FAILED;
            SIMPLE_LOG_ERROR("qnn populateInputTensors failed\n");
            break;
        }

        run_success = qnn_wrapper_ptr_->executeGraphs();
        if (run_success != true) {
            ret = MStatus::M_FAILED;
            SIMPLE_LOG_ERROR("qnn executeGraphs failed\n");
            break;
        }

        run_success =
            qnn_wrapper_ptr_->populateOutputBuffer(net_output_buffers, net_output_buffers_len);
        if (run_success != true) {
            ret = MStatus::M_FAILED;
            SIMPLE_LOG_ERROR("qnn populateOutputBuffer failed\n");
            break;
        }
    } while (0);

    SIMPLE_LOG_DEBUG("InferQnn::RunSingleBatch End\n");
    return MStatus::M_OK;
}

MStatus InferQnn::Run(std::vector<NNTensorPtr>& input,
                      std::vector<NNTensorPtr>& output,
                      const char* start,
                      const char* end) {
    SIMPLE_LOG_DEBUG("InferQnn::Run Start\n");
    MStatus ret = MStatus::M_OK;
    do {
        auto net_input = input;
        ret            = CheckInputShape(net_input);
        if (ret != MStatus::M_OK) {
            SIMPLE_LOG_ERROR("CheckInputShape failed\n");
            break;
        }

        ret = ReArrangeInput(net_input);
        if (ret != MStatus::M_OK) {
            SIMPLE_LOG_ERROR("ReArrangeInput failed\n");
            break;
        }

        const int batch_size = net_input[0]->GetShape(0) / GetInputDims(0)[0];
        std::vector<NNTensorPtr> net_output;
        ret = CreateNetOutput(batch_size, net_output);
        if (ret != MStatus::M_OK) {
            SIMPLE_LOG_ERROR("CreateNetOutput failed\n");
            break;
        }

        for (int batch = 0; batch < batch_size; ++batch) {
            ret = RunSingleBatch(net_input, net_output, batch);
            if (ret != MStatus::M_OK) {
                SIMPLE_LOG_ERROR("RunSingleBatch failed\n");
                break;
            }
        }

        ret = ReArrangeOutput(net_output);
        if (ret != MStatus::M_OK) {
            SIMPLE_LOG_ERROR("ReArangeOutput failed\n");
            break;
        }
        output = output_tensors_;
    } while (0);
    SIMPLE_LOG_DEBUG("InferQnn::Run End\n");
    return MStatus::M_OK;
}

} // namespace nn
