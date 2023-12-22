#include "runtime/layer/linear.h"

static constexpr char kHasBias[]    = "bias";
static constexpr char kInFeature[]  = "in_features";
static constexpr char kOutFeature[] = "out_features";
namespace nn {
MStatus Linear::Init(const std::map<std::string, pnnx::Parameter>& params,
                     const std::map<std::string, pnnx::Attribute>& attrs) {
    SIMPLE_LOG_DEBUG("Linear::Init Start\n");
    MStatus ret = MStatus::M_OK;
    do {
        if ([this, params]() -> bool {
                return !(params.count(kHasBias) && params.count(kInFeature) &&
                         params.count(kOutFeature));
            }()) {
            std::string param_info;
            for (const auto& p : params) {
                param_info = p.first + "-";
            }
            SIMPLE_LOG_ERROR("Linear::Init failed, parma not support %s\n", param_info.c_str());
            ret = MStatus::M_FAILED;
            break;
        }
        has_bias_     = params.at(kHasBias).b;
        in_features_  = params.at(kInFeature).i;
        out_features_ = params.at(kOutFeature).i;
        SIMPLE_LOG_DEBUG("nn.Linear param (has_bias=%i, in_features=%i, out_features=%i)\n",
                         has_bias_,
                         in_features_,
                         out_features_);

        const auto& weight = attrs.at(kWeight).get_float32_data();
        SIMPLE_LOG_DEBUG("nn.Linear param (weight_size: %i)\n", weight.size());
        weight_.reset(new base::Tensor(weight.data(),
                                       {1, 1, in_features_, out_features_},
                                       M_LAYOUT_NCHW,
                                       M_MEM_ON_CPU,
                                       M_DATA_TYPE_FLOAT32));
        if (has_bias_) {
            if (!attrs.count(kBias)) {
                SIMPLE_LOG_ERROR("Linear::Init failed, attrs not bias data\n");
                ret = MStatus::M_FAILED;
                break;
            }

            const auto& bias = attrs.at(kBias).get_float32_data();
            SIMPLE_LOG_DEBUG("nn.Linear param (bias_size: %i)\n", bias.size());
            bias_.reset(new base::Tensor(bias.data(),
                                         {1, 1, 1, out_features_},
                                         M_LAYOUT_NCHW,
                                         M_MEM_ON_CPU,
                                         M_DATA_TYPE_FLOAT32));
        }

        if (!weight_ || (has_bias_ && !bias_)) {
            ret = MStatus::M_FAILED;
            SIMPLE_LOG_DEBUG("Linear::Init failed, weight or bias malloc failed\n");
            break;
        }
    } while (0);
    return ret;
    SIMPLE_LOG_DEBUG("Linear::Init End\n");
}

MStatus Linear::Forward(const TensorPtr& input, TensorPtr& output) {
    if (!input) {
        SIMPLE_LOG_DEBUG("Linear::Forward failed, input data nullptr\n");
        return MStatus::M_INTERNAL_FAILED;
    }
    if (input->GetShape(0) != 1) {
        SIMPLE_LOG_DEBUG("Linear::Forward failed, now can't support batch more than 1\n");
        return MStatus::M_NOT_SUPPORT;
    }
    auto check = [](const std::vector<uint32_t>& shape) -> bool {
        if (shape.size() < 2) {
            return false;
        }
        return !std::count(shape.rbegin() + 2, shape.rend(), 1);
    };
    if (check(input->GetShape())) {
        SIMPLE_LOG_ERROR("Linear::Forward failed, now can't support dims more than 2\n");
        return MStatus::M_FAILED;
    }

    if (input->GetShape(3) != weight_->GetShape(2)) {
        SIMPLE_LOG_ERROR("Linear::Forward failed, shape missmatch, [%i,%i]*[%i,%i]\n",
                         input->GetShape(0),
                         input->GetShape(1),
                         weight_->GetShape(0),
                         weight_->GetShape(1));
        return MStatus::M_FAILED;
    }
    std::vector<uint32_t> shape{1, 1, input->GetShape(2), weight_->GetShape(3)};
    if (!output || output->GetCount() != input->GetShape(2) * input->GetShape(3)) {
        output.reset(new base::Tensor(shape, M_LAYOUT_NCHW, M_MEM_ON_CPU, M_DATA_TYPE_FLOAT32));
    }

    const int rows = shape.at(2), cols = shape.at(3);
    for (int i = 0; i < rows; ++i) {

        const float* in_ptr_row = input->GetData<float>(i * rows);

        for (int j = 0; j < cols; ++j) {

            const float* weight_col = weight_->GetData<float>(0) + weight_->GetShape(3) * j;

            float sum = 0.f;

            if (has_bias_) {
                sum = bias_->GetData<float>(0)[j];
            }

            for (int i = 0; i < rows; i++) {
                sum += in_ptr_row[i] * weight_col[i];
            }
            output->GetData<float>(0)[i * rows + j] = sum;
        }
    }
    return MStatus::M_OK;
}
} // namespace nn
