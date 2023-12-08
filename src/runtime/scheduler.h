#ifndef SIMPLE_NN_SCHEDULER_H_
#define SIMPLE_NN_SCHEDULER_H_

#include "runtime/net.h"

namespace nn {
class Scheduler {
public:
    Scheduler()  = default;
    ~Scheduler() = default;

private:
    std::shared_ptr<Net> net_{nullptr};
};
} // namespace nn

#endif // SIMPLE_NN_SCHEDULER_H_