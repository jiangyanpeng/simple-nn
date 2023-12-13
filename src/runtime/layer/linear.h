#ifndef SIMPLE_NN_LINEAR_H_
#define SIMPLE_NN_LINEAR_H_

#include "runtime/layer.h"

namespace nn {
constexpr char kLinearType[] = "nn.Linear";
class Linear : public Layer {
public:
    Linear()  = default;
    ~Linear() = default;

    MStatus Init(const std::shared_ptr<LayerParam>& param,
                 const std::shared_ptr<ModelBin>& bin) override;

    MStatus Forward(const std::vector<TensorPtr>& input, std::vector<TensorPtr>& output) override;
};
} // namespace nn

#endif // SIMPLE_NN_LINEAR_H_