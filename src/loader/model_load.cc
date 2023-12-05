#include "loader/model_load.h"

#include <log.h>

namespace nn {
NNModel::NNModel(const std::string& path) : path_(path), inited_(false), data_(nullptr) {}

MStatus NNModel::Init(const std::string& path) {
    SIMPLE_LOG_DEBUG("NNModel::NNModel {} Init Start", path.c_str());
    auto ret = MStatus::M_OK;
    do {
        file_ = fopen(path.c_str(), "rb");
        if (!file_) {
            SIMPLE_LOG_ERROR("failed open {}", path.c_str());
            ret = MStatus::M_FILE_NOT_FOUND;
            break;
        }
        fseek(file_, 0, SEEK_END);
        size_ = ftell(file_);
        fseek(file_, 0, SEEK_SET);
        type_ = NNModelType::M_FILE;

        data_ = std::unique_ptr<uint8_t>(new uint8_t[size_]);
        if (!Read(static_cast<void*>(data_.get()), size_, size_)) {
            SIMPLE_LOG_ERROR("read model buffer faile");
            ret = MStatus::M_FAILED;
            break;
        }

        inited_ = true;
    } while (0);
    SIMPLE_LOG_DEBUG("NNModel::NNModel {} Init End", path.c_str());
    return ret;
}

NNModel::~NNModel() {
    if (file_) {
        fclose(file_);
    }
}

MStatus NNModel::Seek(size_t pos) {
    if (pos >= size_) {
        SIMPLE_LOG_ERROR("NNModel::Seek failed, {} out of range : {} ", pos, size_);
        return MStatus::M_OUT_OF_MEMORY;
    }
    fseek(file_, static_cast<long>(pos), SEEK_SET);
}

size_t NNModel::Read(void* buf, size_t size, size_t nmenb) {
    if (size * nmenb < size_) {
        SIMPLE_LOG_ERROR("NNModel::Read failed, size : {} smaller than {} ", size, size_);
        return 0;
    }
    return fread(buf, size, nmenb, file_);
}

size_t NNModel::Size() const {
    return size_;
}

} // namespace nn
