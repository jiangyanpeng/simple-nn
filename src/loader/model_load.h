#ifndef SIMPLE_NN_MODEL_LOADER_H_
#define SIMPLE_NN_MODEL_LOADER_H_

#include <algorithm>
#include <common.h>
#include <memory>
#include <string>
#include <vector>

namespace nn {
typedef struct ModelConfig {
    const char* engine;
    unsigned int engine_len;
    void* engine_context;
} ModelConfig;


class NNModel {
public:
    enum class NNModelType {
        M_FILE,
        M_MEMORY,
        M_TARFILE,
        M_INVALID,
    };

public:
    NNModel(const std::string& path);
    ~NNModel();

    MStatus Init(const std::string& path);

    const std::string& Path() const { return path_; }
    NNModelType GetType() const { return type_; }

    size_t Size() const;

    const std::unique_ptr<uint8_t>& Data() const { return data_; }

private:
    MStatus Seek(size_t pos);
    size_t Read(void* buf, size_t size, size_t nmenb);

private:
    std::string path_;
    NNModelType type_;

    bool inited_;

    // file
    FILE* file_;
    size_t size_;

    std::unique_ptr<uint8_t> data_{nullptr};
};

class ModelPackage {
public:
    using NNModelPtr = std::shared_ptr<NNModel>;

public:
    ModelPackage() {}
    virtual ~ModelPackage() {}
    void Push(NNModelPtr base) { models_.push_back(base); }
    std::vector<NNModelPtr>& GetModels() { return models_; }

    NNModelPtr GetByPath(const std::string& name) const {
        auto it = std::find_if(models_.begin(), models_.end(), [name](const NNModelPtr& v) {
            return v->Path() == name;
        });
        return (it == models_.end()) ? nullptr : *it;
    }

private:
    std::vector<NNModelPtr> models_;
};
} // namespace nn

#endif // SIMPLE_NN_MODEL_LOADER_H_