#include "test_util.h"
#include "gtest/gtest.h"

#include <algorithm>
#include <fstream>
#include <iostream>

int read_binary_file(const std::string& file_path, void* buffer, long length) {
    std::ifstream ifs(file_path, std::ios::binary);

    if (!ifs) {
        std::cout << "can't read file: " << file_path << std::endl;
        return -1;
    }

    long file_len = 0;

    ifs.seekg(0, std::ios::end);
    file_len = ifs.tellg();
    ifs.seekg(0, std::ios::beg);

    ifs.read((char*)buffer, std::min(file_len, length));
    ifs.close();

    return 0;
}

int write_binary_file(const std::string& file_path, void* buffer, long length) {
    std::ofstream ofs(file_path.c_str(), std::ios::binary);

    if (!ofs) {
        printf("can't write file: %s", file_path.c_str());
        return -1;
    }

    ofs.write((char*)buffer, length);
    ofs.close();

    return 0;
}