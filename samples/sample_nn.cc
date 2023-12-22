#include "runtime/net.h"

#include <iostream>
#include <log.h>

int main(int argc, char* argv[]) {
    if (argc != 3) {
        printf("usage: ./bin/sample_nn "
               "{param} "
               "{bin} \n");
        return -1;
    }
#ifdef CONFIG_SIMPLE_BASE_ENABLE_SPDLOG
    std::cout << "enable" << std::endl;
#endif
    set_level(Loger::DEBUG);
    auto net = std::make_shared<nn::Net>();
    net->Init(argv[1], argv[2]);
    return 0;
}