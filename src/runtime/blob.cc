#include "runtime/blob.h"

namespace nn {
Blob::Blob() : name(""), producer(-1), consumer(-1), shape({}) {}
} // namespace nn