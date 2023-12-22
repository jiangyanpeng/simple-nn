#ifndef SIMPLE_NN_LINEAR_H_
#define SIMPLE_NN_LINEAR_H_

#include "runtime/layer.h"

namespace nn {
constexpr char kLinearType[] = "nn.Linear";
class Linear : public Layer {
public:
    Linear()  = default;
    ~Linear() = default;

    MStatus Init(const std::map<std::string, pnnx::Parameter>& params,
                 const std::map<std::string, pnnx::Attribute>& attrs) override;

    MStatus Forward(const TensorPtr& input, TensorPtr& output) override;

private:
    uint32_t in_features_{0};
    uint32_t out_features_{0};
    bool has_bias_{false};
};
} // namespace nn

#endif // SIMPLE_NN_LINEAR_H_