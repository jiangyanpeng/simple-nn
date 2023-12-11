#ifndef SIMPLE_NN_NET_H_
#define SIMPLE_NN_NET_H_

#include "runtime/blob.h"
#include "runtime/layer.h"
#include "runtime/net_option.h"
#include "runtime/pnnx/ir.h"

#include <bitset>
#include <string>
#include <tensor/tensor.h>

namespace nn {
constexpr int16_t MAX_NUM_LAYER = 2048;

class Layer;
class Net {
public:
    using TensorPtr = std::shared_ptr<base::Tensor>;

public:
    Net(const std::string& name = "");
    ~Net() = default;

    MStatus Init(const std::string& param, const std::string& bin);

    // MStatus Forward(const std::vector<TensorPtr>& input, std::vector<TensorPtr>& output);

    MStatus Forward(int layer_index, std::vector<TensorPtr>& blob_mats) const;

    const std::string Summary() const;

private:
    Net(const Net&);
    Net& operator=(const Net&);

    int find_blob_index_by_name(const std::string& name);
    int find_layer_index_by_name(const std::string& name);

private:
    std::string net_name_;
    std::shared_ptr<NetOption> option_{nullptr};

    std::unique_ptr<pnnx::Graph> graph_{nullptr};

    std::vector<Blob> blobs_;
    std::vector<std::shared_ptr<Layer>> layers_;
    std::bitset<MAX_NUM_LAYER> state_;

    std::vector<int> input_blob_index_;
    std::vector<int> output_blob_index_;
};
} // namespace nn


#endif // SIMPLE_NN_NET_H_