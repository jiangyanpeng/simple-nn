#ifndef SIMPLE_NN_INPUT_H_
#define SIMPLE_NN_INPUT_H_

#include "runtime/layer.h"

namespace nn {
constexpr char kInputType[] = "pnnx.Input";
class Source : public Layer {

public:
    Source()  = default;
    ~Source() = default;

    MStatus Init(const std::map<std::string, pnnx::Parameter>& params) override;

    MStatus Forward(const std::vector<TensorPtr>& input, std::vector<TensorPtr>& output) override;
};
} // namespace nn
#endif // SIMPLE_NN_INPUT_H_