#include "runtime/net.h"

#include <iomanip>
#include <sstream>

namespace nn {
Net::Net(const Net& net)
    : net_name_(net.net_name_),
      option_(net.option_),
      blobs_(net.blobs_),
      layers_(net.layers_),
      input_blob_index_(net.input_blob_index_),
      output_blob_index_(net.output_blob_index_) {}

Net& Net::operator=(const Net&) {
    return *this;
}

Net::Net(const std::string& name)
    : net_name_(name), option_(nullptr), graph_(nullptr), blobs_({}), layers_({}) {}


const std::string Net::Summary() const {
    std::vector<pnnx::Operator*> operators = this->graph_->ops;
    std::stringstream ss;
    auto get_shape_str = [](const std::vector<pnnx::Operand*>& op) -> std::string {
        std::stringstream ss;
        if (op.size()) {
            auto shape = op[0]->shape;
            for (size_t i = 0; i < shape.size(); ++i) {
                if (i == 0) {
                    ss << "[";
                }
                ss << shape[i];
                if (i != shape.size() - 1) {
                    ss << ", ";
                } else {
                    ss << "]";
                }
            }
        } else {
            ss << "----";
        }
        return ss.str();
    };

    auto get_param_str =
        [](const std::map<std::string, pnnx::Attribute>& attribute) -> std::string {
        u_int64_t param_size = 0;
        for (const auto& it : attribute) {
            param_size += it.second.elemcount() * it.second.elemsize();
        }
        return std::to_string(param_size);
    };

    for (size_t i = 0; i < operators.size(); ++i) {
        if (i == 0) {
            ss << "----------------------------------------------------------------------------"
                  "-------------------------------"
               << std::endl
               << std::left << std::setw(25) << "name" << std::left << std::setw(25) << "type"
               << std::left << std::setw(25) << "input_shape" << std::left << std::setw(25)
               << "output_shape" << std::left << std::setw(25) << "param" << std::endl
               << "============================================================================"
                  "==============================="
               << std::endl;
        }

        ss << std::left << std::setw(25) << operators[i]->name << std::left << std::setw(25)
           << operators[i]->type << std::left << std::setw(25)
           << get_shape_str(operators[i]->inputs) << std::left << std::setw(25)
           << get_shape_str(operators[i]->outputs) << std::left << std::setw(25)
           << get_param_str(operators[i]->attrs) << std::endl;

        if (i == operators.size() - 1) {
            ss << "============================================================================"
                  "===============================";
        }
    }
    return ss.str();
}

MStatus Net::Init(const std::string& param, const std::string& bin) {
    SIMPLE_LOG_DEBUG("Net::Init Start");
    MStatus ret = MStatus::M_OK;
    do {
        if (param.empty() || bin.empty()) {
            SIMPLE_LOG_ERROR("input param or bin path error! param:{}, bin:{}", param, bin);
            ret = MStatus::M_INVALID_ARG;
            break;
        }
        this->graph_.reset(new pnnx::Graph());
        if (nullptr == this->graph_) {
            SIMPLE_LOG_ERROR("pnnx::Graph create failed!");
            ret = MStatus::M_INTERNAL_FAILED;
            break;
        }
        SIMPLE_LOG_DEBUG("pnnx::Graph load {}, {}", param, bin);
        if (this->graph_->load(param, bin) < 0) {
            SIMPLE_LOG_ERROR("pnnx::Graph load param and bin failed!");
            ret = MStatus::M_FAILED;
            break;
        }

        std::vector<pnnx::Operator*> operators = this->graph_->ops;
        if (operators.empty()) {
            SIMPLE_LOG_ERROR("pnnx::Graph has no any operators!");
            ret = MStatus::M_FAILED;
            break;
        }

        // show net details
        printf("%s\n", Summary().c_str());

    } while (0);
    SIMPLE_LOG_DEBUG("Net::Init End");
    return ret;
}

MStatus Net::Forward(int layer_index, std::vector<TensorPtr>& blob_mats) const {
    if (layer_index < static_cast<int>(layers_.size()) || layer_index < 0) {
        SIMPLE_LOG_ERROR("Net::Forward index out of range, {}vs{}", layer_index, layers_.size());
        return MStatus::M_OUT_OF_MEMORY;
    }

    auto& layer = layers_[layer_index];
    for (size_t i = 0; i < layer->Bottom().size(); ++i) {
        int bottom_index = layer->Bottom()[i];
        auto ret         = Forward(blobs_[bottom_index].producer, blob_mats);
        if (ret != MStatus::M_OK) {
            SIMPLE_LOG_ERROR("Net::Forward failed, layer_name: {}, bottom_index: {}",
                             layer->Name(),
                             bottom_index);
            return ret;
        }
    }
}

int Net::find_blob_index_by_name(const std::string& name) {
    int index = -1;
    for (size_t i = 0; i < blobs_.size(); ++i) {
        const Blob& blob = blobs_[i];
        if (blob.name == name) {
            index = static_cast<int>(i);
            break;
        }
    }
    SIMPLE_LOG_DEBUG("Net::find_blob_index_by_name name:{}, index:{}", name, index);
}

int Net::find_layer_index_by_name(const std::string& name) {
    int index = -1;
    for (size_t i = 0; i < layers_.size(); ++i) {
        const auto& layer = layers_[i];
        if (layers_[i]->Name() == name) {
            index = static_cast<int>(i);
            break;
        }
    }
    SIMPLE_LOG_DEBUG("Net::find_layer_index_by_name name:{}, index:{}", name, index);
}
} // namespace nn
