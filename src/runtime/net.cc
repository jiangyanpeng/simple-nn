#include "runtime/net.h"

#include "runtime/layer_register.h"

#include <iomanip>
#include <regex>
#include <register.h>
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
    SIMPLE_LOG_DEBUG("Net::Init Start\n");
    MStatus ret = MStatus::M_OK;
    do {
        if (param.empty() || bin.empty()) {
            SIMPLE_LOG_ERROR(
                "input param or bin path error! param:%s, bin:%s\n", param.c_str(), bin.c_str());
            ret = MStatus::M_INVALID_ARG;
            break;
        }
        this->graph_.reset(new pnnx::Graph());
        if (nullptr == this->graph_) {
            SIMPLE_LOG_ERROR("pnnx::Graph create failed\n");
            ret = MStatus::M_INTERNAL_FAILED;
            break;
        }
        SIMPLE_LOG_DEBUG("pnnx::Graph load %s, %s\n", param.c_str(), bin.c_str());
        if (this->graph_->load(param, bin) < 0) {
            SIMPLE_LOG_ERROR("pnnx::Graph load param and bin failed\n");
            ret = MStatus::M_FAILED;
            break;
        }

        std::vector<pnnx::Operator*> operators = this->graph_->ops;
        if (operators.empty()) {
            SIMPLE_LOG_ERROR("pnnx::Graph has no any operators\n");
            ret = MStatus::M_FAILED;
            break;
        }

        // show net details
        printf("%s\n", Summary().c_str());

        int layer_count = static_cast<int>(this->graph_->ops.size());
        int blob_count  = static_cast<int>(this->graph_->operands.size());
        if (layer_count <= 0 || blob_count <= 0) {
            SIMPLE_LOG_ERROR(
                "layer_count[%i] or blob_count[%i] invalid\n", layer_count, blob_count);
            ret = MStatus::M_INVALID_ARG;
            break;
        }
        layers_.resize(layer_count);
        blobs_.resize(blob_count);

        // TODO: check model magic number

        for (int i = 0; i < layer_count; ++i) {
            int bottom_count       = static_cast<int>(this->graph_->ops[i]->inputs.size());
            int top_count          = static_cast<int>(this->graph_->ops[i]->outputs.size());
            std::string layer_name = this->graph_->ops[i]->name;
            std::string type       = this->graph_->ops[i]->type;
            SIMPLE_LOG_DEBUG("create [%s:%s] layer, bottom: %i, top: %i\n",
                             layer_name,
                             type,
                             bottom_count,
                             top_count);
            // find input and output
            if (!bottom_count) {
                input_names_.emplace_back(this->graph_->ops[i]->name);
            }
            if (!top_count) {
                output_names_.emplace_back(this->graph_->ops[i]->name);
            }

            auto layer_type_ptr = layer_map.find(this->graph_->ops[i]->type);
            if (layer_type_ptr == layer_map.end()) {
                SIMPLE_LOG_ERROR("layer map can't find %s layer\n",
                                 this->graph_->ops[i]->type.c_str());
                ret = MStatus::M_NOT_SUPPORT;
                break;
            }

            auto layer = RegisterBase<Layer>::GetInstance().Create(layer_type_ptr->second);
            if (!layer) {
                SIMPLE_LOG_ERROR("get [%s:%s] layer failed\n",
                                 this->graph_->ops[i]->name.c_str(),
                                 this->graph_->ops[i]->type.c_str());
                ret = MStatus::M_NOT_SUPPORT;
                break;
            }

            layer->bottom_.resize(bottom_count);
            for (int j = 0; j < bottom_count; ++j) {
                std::string input_idx = this->graph_->ops[i]->inputs[j]->name;
                std::regex rx("[0-9]+");
                if (!std::regex_match(input_idx.begin(), input_idx.end(), rx)) {
                    SIMPLE_LOG_ERROR("cread blob failed!, %s layer, %i blob is not number\n",
                                     this->graph_->ops[i]->name.c_str(),
                                     input_idx);
                    ret = MStatus::M_NOT_SUPPORT;
                    break;
                }
                int bottom_blob_index = std::atoi(input_idx.c_str());
                Blob& blob            = this->blobs_[bottom_blob_index];
                blob.consumer         = i;
                layer->bottom_[j]     = bottom_blob_index;
            }

            layer->top_.resize(top_count);
            for (int j = 0; j < top_count; ++j) {
                std::string output_idx = this->graph_->ops[i]->outputs[j]->name;
                std::regex rx("[0-9]+");
                if (!std::regex_match(output_idx.begin(), output_idx.end(), rx)) {
                    SIMPLE_LOG_ERROR("create blob failed!, %s layer, %i blob is not number\n",
                                     this->graph_->ops[i]->name.c_str(),
                                     output_idx);
                    ret = MStatus::M_NOT_SUPPORT;
                    break;
                }
                int top_blob_index = std::atoi(output_idx.c_str());
                Blob& blob         = this->blobs_[top_blob_index];
                blob.producer      = i;
                layer->top_[j]     = top_blob_index;
            }

            // load param to layer

            layers_[i] = std::move(layer);
        }

    } while (0);
    SIMPLE_LOG_DEBUG("Net::Init End\n");
    return ret;
}

MStatus Net::Forward(int layer_index, std::vector<TensorPtr>& blob_mats) const {
    if (layer_index < static_cast<int>(layers_.size()) || layer_index < 0) {
        SIMPLE_LOG_ERROR("Net::Forward index out of range, %ivs%i", layer_index, layers_.size());
        return MStatus::M_OUT_OF_MEMORY;
    }

    auto& layer = layers_[layer_index];
    for (size_t i = 0; i < layer->bottom_.size(); ++i) {
        int bottom_index = layer->bottom_[i];
        auto ret         = Forward(blobs_[bottom_index].producer, blob_mats);
        if (ret != MStatus::M_OK) {
            SIMPLE_LOG_ERROR("Net::Forward failed, layer_name: %s, bottom_index: %i\n",
                             layer->GetName().c_str(),
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
    SIMPLE_LOG_DEBUG("Net::find_blob_index_by_name name:%s, index:%i\n", name.c_str(), index);
}

int Net::find_layer_index_by_name(const std::string& name) {
    int index = -1;
    for (size_t i = 0; i < layers_.size(); ++i) {
        const auto& layer = layers_[i];
        if (layers_[i]->GetName() == name) {
            index = static_cast<int>(i);
            break;
        }
    }
    SIMPLE_LOG_DEBUG("Net::find_layer_index_by_name name:%s, index:%i\n", name.c_str(), index);
}
} // namespace nn
