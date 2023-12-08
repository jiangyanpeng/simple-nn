#ifndef SIMPLE_NN_BLOB_H_
#define SIMPLE_NN_BLOB_H_

#include <string>
#include <vector>

namespace nn {
class Blob {
public:
    Blob();
    ~Blob() = default;

public:
    // blob name
    std::string name;
    // layer index which produce this blob as output
    int producer;
    // layer index which need this blob as input
    int consumer;
    // shape hint
    std::vector<uint8_t> shape;
};
} // namespace nn

#endif // SIMPLE_NN_BLOB_H_