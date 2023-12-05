IF(BUILD_QNN)
    # 下载第三方库
    pipe_download_dependency(
        "https://github.com/jiangyanpeng/simple.lib"
        qnn-1.11.2-release
        simple.lib
        ${PROJECT_SOURCE_DIR}/third_party)
    
    # 选择版本号    
    IF (BUILD_QNN_VERSION_1_2_0)
        SET(QNN_VERSION 1.2.0)
    ELSEIF (BUILD_QNN_VERSION_1_11_2)
        SET(QNN_VERSION 1.11.2)
    ELSEIF (BUILD_QNN_VERSION_1_12_4)
        SET(QNN_VERSION 1.12.4)
    ELSEIF (BUILD_QNN_VERSION_2_0_1)
        SET(QNN_VERSION 2.0.1)
    ELSEIF (BUILD_QNN_VERSION_2_3_5)
        SET(QNN_VERSION 2.3.5)
    ELSEIF (BUILD_QNN_VERSION_2_9_0)
        SET(QNN_VERSION 2.9.0)
    ELSEIF (BUILD_QNN_VERSION_2_9_1)
        SET(QNN_VERSION 2.9.1)
    ELSEIF (BUILD_QNN_VERSION_2_10_0)
        SET(QNN_VERSION 2.10.0)
    ELSEIF (BUILD_QNN_VERSION_2_12_1)
        SET(QNN_VERSION 2.12.1)
    ELSEIF (BUILD_QNN_VERSION_2_13_0)
        SET(QNN_VERSION 2.13.0)
    ELSEIF (BUILD_QNN_VERSION_2_14_2)
        SET(QNN_VERSION 2.14.2)
    ELSEIF (BUILD_QNN_VERSION_2_14_3)
        SET(QNN_VERSION 2.14.3)
    ENDIF()
    IF(NOT QNN_VERSION)
        SET(QNN_VERSION 1.11.2)
    ENDIF()

    # 解压第三库压缩文件
    SET(QNN_ROOT ${CMAKE_CURRENT_SOURCE_DIR}/third_party/simple.lib)
    IF(NOT EXISTS "${QNN_ROOT}/${QNN_VERSION}.tar.gz")
        MESSAGE(FATAL_ERROR "Not Found ${QNN_ROOT}/${QNN_VERSION}.tar.gz")
    ELSE()
        MESSAGE("-- tar -xzf ${QNN_ROOT}/${QNN_VERSION}.tar.gz")
        EXECUTE_PROCESS(COMMAND tar -xzf ${QNN_ROOT}/${QNN_VERSION}.tar.gz -C ${QNN_ROOT})
    ENDIF()

    # 配置相关路径
    SET(QNN_PATH ${QNN_ROOT}/${QNN_VERSION})
    IF(NOT EXISTS ${QNN_PATH})
        MESSAGE(FATAL_ERROR "Invalid QNN_PATH ${QNN_PATH}")
    ENDIF()

    SET(HPC_INC_PATH ${QNN_PATH}/include)
    SUBDIRLIST(HPC_INC_PATH_SUB ${QNN_PATH}/include "0")
    IF (QNN_VERSION STREQUAL "1.2.0" OR QNN_VERSION STREQUAL "1.11.2" OR QNN_VERSION STREQUAL "1.12.4")
        # 设置版本
        SET(FLATBUFFERS_VERSION flatbuffers-1.11.0)

        # 解压
        IF(NOT EXISTS "${QNN_ROOT}/${FLATBUFFERS_VERSION}.tar.gz")
            MESSAGE(FATAL_ERROR "Not Found ${QNN_ROOT}/${FLATBUFFERS_VERSION}.tar.gz")
        ELSE()
            MESSAGE("-- tar -xzf ${QNN_ROOT}/${FLATBUFFERS_VERSION}.tar.gz")
            EXECUTE_PROCESS(COMMAND tar -xzf ${QNN_ROOT}/${FLATBUFFERS_VERSION}.tar.gz -C ${QNN_ROOT})
        ENDIF()
        IF(NOT EXISTS ${QNN_ROOT}/${FLATBUFFERS_VERSION})
            MESSAGE(FATAL_ERROR "Invalid FLATBUFFERS_PATH ${QNN_ROOT}/${FLATBUFFERS_VERSION}")
        ENDIF()

        SET(CACHING_UTIL_PATH ${QNN_ROOT}/CachingUtil/include)
        SET(CACHING_UTIL_FILE_PATH ${CACHING_UTIL_PATH}/SampleAppContextCaching_generated.h)
        IF(NOT EXISTS ${CACHING_UTIL_FILE_PATH})
            SET(FLATC_FILE_PATH ${QNN_ROOT}/${FLATBUFFERS_VERSION}/build/flatc)
            SET(CACHING_FBS_FILE_PATH ${QNN_PATH}/examples/SampleApp/src/CachingUtil/SampleAppContextCaching.fbs)
            EXECUTE_PROCESS(COMMAND chmod +x ${FLATC_FILE_PATH})
            EXECUTE_PROCESS(COMMAND ${FLATC_FILE_PATH} --cpp --gen-object-api -o ${CACHING_UTIL_PATH} ${CACHING_FBS_FILE_PATH})
        ENDIF()

        LIST(APPEND HPC_INC_PATH ${QNN_ROOT}/${FLATBUFFERS_VERSION}/include)
        LIST(APPEND HPC_INC_PATH ${CACHING_UTIL_PATH})

        SUBDIRLIST(HPC_INC_PATH_SUB ${QNN_ROOT}/${FLATBUFFERS_VERSION}/include "0")
        SUBDIRLIST(HPC_INC_PATH_SUB ${CACHING_UTIL_PATH} "0")

    ENDIF()
ENDIF()
