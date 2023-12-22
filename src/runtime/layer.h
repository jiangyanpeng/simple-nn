#ifndef SIMPLE_NN_LAYER_H_
#define SIMPLE_NN_LAYER_H_

#include "pnnx/ir.h"
#include "runtime/net.h"

#include <common.h>
#include <string>
#include <tensor/tensor.h>
namespace nn {

enum class LayerType {};
static constexpr char kWeight[] = "weight";
static constexpr char kBias[]   = "bias";
class Layer {
public:
    using TensorPtr = std::shared_ptr<base::Tensor>;

public:
    Layer() = default;
    virtual ~Layer() {}

    virtual MStatus Init(const std::map<std::string, pnnx::Parameter>& params,
                         const std::map<std::string, pnnx::Attribute>& attrs);

    virtual MStatus Forward(const TensorPtr& input, TensorPtr& output);

    const std::string GetName() const { return name_; }

protected:
    friend class Net;

protected:
    // layer name
    std::string name_;

    // layer type
    LayerType type_;

    // tensor shape which this layer needs as input
    std::vector<std::vector<uint8_t>> input_shape_{};

    // tensor shape which this layer produces as output
    std::vector<std::vector<uint8_t>> output_shape_{};

    // tensor index which this layer needs as input
    std::vector<int> bottom_{};

    // tensor index which this layer produces as output
    std::vector<int> top_{};

    // custom user data
    std::shared_ptr<uint8_t> data_{nullptr};

    TensorPtr weight_;
    TensorPtr bias_;
};
} // namespace nn


#endif // SIMPLE_NN_LAYER_H_