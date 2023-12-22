#!/usr/bin/bash
rm -rf build && mkdir build && cd build
cmake -DCMAKE_BUILD_TYPE=DEBUG -DCMAKE_INSTALL_PREFIX=simple.nn/ -DBUILD_USE_AVX=ON -DBUILD_LOG=ON -DBUILD_QNN=ON -DBUILD_TEST=ON ..
make -j8 && make install
