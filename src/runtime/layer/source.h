#ifndef SIMPLE_NN_INPUT_H_
#define SIMPLE_NN_INPUT_H_

#include "runtime/layer.h"

namespace nn {
constexpr char kInputType[] = "pnnx.Input";
class Source : public Layer {

public:
    Source()  = default;
    ~Source() = default;

    MStatus Init(const std::map<std::string, pnnx::Parameter>& params,
                 const std::map<std::string, pnnx::Attribute>& attrs) override;

    MStatus Forward(const TensorPtr& input, TensorPtr& output) override;
};
} // namespace nn
#endif // SIMPLE_NN_INPUT_H_