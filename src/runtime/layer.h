#ifndef SIMPLE_NN_LAYER_H_
#define SIMPLE_NN_LAYER_H_

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
    Layer(const std::string& name, const LayerType& type);
    virtual ~Layer() {}

    virtual MStatus Load(const std::shared_ptr<LayerParam>& param,
                         const std::shared_ptr<ModelBin>& bin);

    virtual MStatus Forward(const std::vector<TensorPtr>& input, std::vector<TensorPtr>& output);

    const std::string Name() const { return name_; }

    const std::vector<int> Bottom() const { return bottom_; }
    const std::vector<int> Top() const { return top_; }

protected:
    // layer name
    std::string name_;

    // layer type
    LayerType type_;

    // tensor shape which this layer needs as input
    std::vector<uint8_t> input_shape_{};

    // tensor shape which this layer produces as output
    std::vector<uint8_t> output_shape_{};

    // tensor index which this layer needs as input
    std::vector<int> bottom_{};

    // tensor index which this layer produces as output
    std::vector<int> top_{};

    // custom user data
    std::shared_ptr<uint8_t> data_{nullptr};
};
} // namespace nn


#endif // SIMPLE_NN_LAYER_H_