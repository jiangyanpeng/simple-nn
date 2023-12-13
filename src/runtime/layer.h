#ifndef SIMPLE_NN_LAYER_H_
#define SIMPLE_NN_LAYER_H_

#include "runtime/net.h"

#include <common.h>
#include <string>
#include <tensor/tensor.h>
namespace nn {

class ModelBin;
enum class LayerType {};

class Layer {
public:
    using TensorPtr = std::shared_ptr<base::Tensor>;

    class LayerParam {
        LayerParam() {}
        virtual ~LayerParam() {}
    };

public:
    Layer() = default;
    virtual ~Layer() {}

    virtual MStatus Init(const std::shared_ptr<LayerParam>& param,
                         const std::shared_ptr<ModelBin>& bin);

    virtual MStatus Forward(const std::vector<TensorPtr>& input, std::vector<TensorPtr>& output);

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
};
} // namespace nn


#endif // SIMPLE_NN_LAYER_H_