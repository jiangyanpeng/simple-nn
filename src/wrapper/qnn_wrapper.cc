#include "wrapper/qnn_wrapper.h"

#include "HTA/QnnHtaBackend.h"
#include "HTP/QnnDspBackend.h"
#include "SampleAppContextCaching_generated.h"
#include "System/QnnSystemInterface.h"
#include "wrapper/utils/qnn_utils.h"

#include <algorithm>
#include <dlfcn.h>
#include <fstream>
#include <getopt.h>
#include <map>
#include <stdint.h>

namespace nn {
namespace wrap {
    void logQnnCallback(const char* fmt, QnnLog_Level_t level, uint64_t timestamp, va_list argp);
    bool QnnWrapperV1::initForDevice(const std::string& backendPath,
                                     const std::string& systemLibraryPath,
                                     bool is_use_signed_pd,
                                     bool is_use_vndk) {
        (void)(is_use_vndk); // unused
        memset(&m_qnnFunctionPointers, 0x00, sizeof(QnnFunctionPointers));

        if (!getQnnBackendFunctionPointers(backendPath)) {
            SIMPLE_LOG_ERROR("Load Qnn backend library error\n");
            return false;
        }

        // logInitialize will cause qnn crash in qnn 1.12.4.
        // if (QNN_SUCCESS !=
        //     m_qnnFunctionPointers.qnnInterface.logInitialize(logQnnCallback, QNN_LOG_LEVEL_WARN))
        //     { SIMPLE_LOG_WARN("Unable to initialize logging in the backend.");
        // }
        SIMPLE_LOG_DEBUG("qnn logInitialze is not called\n");

        QnnDspBackend_CustomConfig_t custom_config;
        custom_config.option = QNN_DSP_BACKEND_CONFIG_OPTION_USE_SIGNED_PROCESS_DOMAIN;
        custom_config.useSignedProcessDomain = true;

        QnnBackend_Config_t custom_dev_config1;
        custom_dev_config1.option       = QNN_BACKEND_CONFIG_OPTION_CUSTOM;
        custom_dev_config1.customConfig = &custom_config;

        QnnBackend_Config_t* m_backend_Config[] = {&custom_dev_config1, nullptr};
        SIMPLE_LOG_DEBUG("is_use_signed_pd: %i\n", is_use_signed_pd);
        if (is_use_signed_pd) { // use unsignedPD
            m_BackendConfig = (QnnBackend_Config_t**)m_backend_Config;
        } else { // use signedPD
            m_BackendConfig = nullptr;
        }

        auto status = m_qnnFunctionPointers.qnnInterface.backendInitialize(
            (const QnnBackend_Config_t**)m_BackendConfig);
        if (QNN_BACKEND_NO_ERROR != status && QNN_BACKEND_ERROR_ALREADY_INITIALIZED != status) {
            SIMPLE_LOG_ERROR("Could not initialize backend due to error = %i",
                             static_cast<int>(status));
            return false;
        }
        if (QNN_BACKEND_ERROR_ALREADY_INITIALIZED == status) {
            SIMPLE_LOG_WARN("A backend is already initialized and the error is ignored\n");
        }

        if (!getQnnSystemFunctionPointers(systemLibraryPath)) {
            SIMPLE_LOG_ERROR("Load Qnn system library error\n");
            return false;
        }

        SIMPLE_LOG_INFO("Initialize backend success\n");
        return true;
    }

    QnnWrapperV1::~QnnWrapperV1() {
        freeGraphsAndContext();
        if (m_libBackendHandle) {
            dlclose(m_libBackendHandle);
        }

        if (m_libSystemHandle) {
            dlclose(m_libSystemHandle);
        }
    }

    bool QnnWrapperV1::createGraphsFromBinary(const uint8_t* buffer, const size_t bufferSize) {
        if (nullptr == m_qnnFunctionPointers.qnnInterface.contextCreateFromBinary ||
            nullptr == m_qnnFunctionPointers.qnnInterface.graphRetrieve) {
            SIMPLE_LOG_ERROR("Qnn interface function hadnle is nullptr\n");
            return false;
        }

        if (nullptr == m_qnnFunctionPointers.qnnSystemInterface.systemContextCreate ||
            nullptr == m_qnnFunctionPointers.qnnSystemInterface.systemContextGetBinaryInfo ||
            nullptr == m_qnnFunctionPointers.qnnSystemInterface.systemContextFree) {
            SIMPLE_LOG_ERROR("QNN system function pointers are not populated\n");
            return false;
        }

        GraphTensorIdToNameMap graphTensorIdToNamesMap;
        uint32_t graphsCount     = 0;
        uint8_t* binaryCache     = nullptr;
        uint64_t binaryCacheSize = 0;
        if (!deserializeData(buffer,
                             bufferSize,
                             graphTensorIdToNamesMap,
                             &graphsCount,
                             binaryCache,
                             binaryCacheSize)) {
            SIMPLE_LOG_ERROR("Deserialize model data error\n");
            return false;
        }

        Qnn_ErrorHandle_t error                = QNN_SUCCESS;
        QnnSystemContext_Handle_t sysCtxHandle = nullptr;
        do {
            error = m_qnnFunctionPointers.qnnSystemInterface.systemContextCreate(&sysCtxHandle);
            if (QNN_SUCCESS != error) {
                SIMPLE_LOG_ERROR("Could not create system handle: error = %i\n",
                                 static_cast<int>(error));
                break;
            }

            QnnSystemContext_BinaryInfo_t* binary_info{nullptr};
            uint32_t binary_info_size{0};
            error = m_qnnFunctionPointers.qnnSystemInterface.systemContextGetBinaryInfo(
                sysCtxHandle,
                static_cast<void*>(binaryCache),
                binaryCacheSize,
                &binary_info,
                &binary_info_size);
            if (QNN_SUCCESS != error) {
                SIMPLE_LOG_ERROR("Failed to get context binary info: error = %i\n",
                                 static_cast<int>(error));
                break;
            }

            if (!copyMetadataToGraphsInfo(binary_info, m_GraphsInfo, m_GraphsCount)) {
                SIMPLE_LOG_ERROR("Failed to copy metadata\n");
                error = QNN_MIN_ERROR_GRAPH;
                break;
            }

            if (!populateTensorNamesFromMetadata(
                    graphTensorIdToNamesMap, m_GraphsInfo, m_GraphsCount)) {
                SIMPLE_LOG_ERROR("Failed to populate tensor names from metadata\n");
                error = QNN_MIN_ERROR_GRAPH;
                break;
            }

            error = m_qnnFunctionPointers.qnnInterface.contextCreateFromBinary(
                static_cast<void*>(binaryCache),
                binaryCacheSize,
                &m_context,
                m_profileBackendHandle);
            if (QNN_SUCCESS != error) {
                SIMPLE_LOG_ERROR("Could not create context from binary: error = %i\n",
                                 static_cast<int>(error));
                break;
            }

            for (size_t gIdx = 0; gIdx < m_GraphsCount; gIdx++) {
                error = m_qnnFunctionPointers.qnnInterface.graphRetrieve(
                    m_context, (*m_GraphsInfo)[gIdx].graphName, &((*m_GraphsInfo)[gIdx].graph));
                if (QNN_SUCCESS != error) {
                    SIMPLE_LOG_ERROR(
                        "Unable to retrieve graph handle for graph Idx: %i, error = %i\n",
                        gIdx,
                        static_cast<int>(error));
                    break;
                }
            }
            if (QNN_SUCCESS != error) {
                break;
            }

            if (!setupInputAndOutputTensors()) {
                error = QNN_MIN_ERROR_TENSOR;
                break;
            }

        } while (false);

        if (binaryCache != nullptr) {
            m_allocator_ptr->Free(binaryCache);
        }

        if (sysCtxHandle != nullptr) {
            m_qnnFunctionPointers.qnnSystemInterface.systemContextFree(sysCtxHandle);
            sysCtxHandle = nullptr;
        }

        if (QNN_SUCCESS != error) {
            freeGraphsAndContext();
            return false;
        }

        return true;
    }

    void QnnWrapperV1::freeGraphsAndContext() {
        tearDownInputAndOutputTensors();

        if (m_context != nullptr) {
            Qnn_ErrorHandle_t error =
                m_qnnFunctionPointers.qnnInterface.contextFree(m_context, m_profileBackendHandle);
            if (QNN_CONTEXT_NO_ERROR != error) {
                SIMPLE_LOG_ERROR("Could not free context: error = %i\n", static_cast<int>(error));
            }
            m_context = nullptr;
        }

        if (m_GraphsCount > 0) {
            if (!freeGraphsInfo(&m_GraphsInfo, m_GraphsCount)) {
                SIMPLE_LOG_ERROR("Could not free graphs\n");
            }
            m_GraphsCount = 0;
        }
    }

    std::vector<std::string> QnnWrapperV1::getInputNames() {
        GraphInfo_t graphInfo = (*m_GraphsInfo)[0];
        auto inputCount       = graphInfo.numInputTensors;
        std::vector<std::string> names;

        for (size_t idx = 0; idx < inputCount; idx++) {
            names.emplace_back(graphInfo.inputTensors[idx].name);
        }
        return names;
    }

    std::vector<std::string> QnnWrapperV1::getOutputNames() {
        GraphInfo_t graphInfo = (*m_GraphsInfo)[0];
        auto inputCount       = graphInfo.numOutputTensors;
        std::vector<std::string> names;

        for (size_t idx = 0; idx < inputCount; idx++) {
            names.emplace_back(graphInfo.outputTensors[idx].name);
        }
        return names;
    }

    std::vector<std::vector<size_t>> QnnWrapperV1::getInputDims() {
        GraphInfo_t graphInfo = (*m_GraphsInfo)[0];

        auto inputCount = graphInfo.numInputTensors;
        std::vector<std::vector<size_t>> inputDims;

        for (size_t idx = 0; idx < inputCount; idx++) {
            Qnn_Tensor_t& input = graphInfo.inputTensors[idx].tensor;
            std::vector<size_t> dims;
            fillDims(dims, input.currentDimensions, input.rank);
            if (dims.size() == 4) {
                // nhwc to nchw
                auto t  = dims[3];
                dims[3] = dims[2];
                dims[2] = dims[1];
                dims[1] = t;
            }
            while (dims.size() < 4) {
                dims.emplace_back(1);
            }
            inputDims.emplace_back(dims);
        }

        return inputDims;
    }

    std::vector<std::vector<size_t>> QnnWrapperV1::getOutputDims() {
        GraphInfo_t graphInfo = (*m_GraphsInfo)[0];

        auto outputCount = graphInfo.numOutputTensors;
        std::vector<std::vector<size_t>> outputDims;

        for (size_t idx = 0; idx < outputCount; idx++) {
            Qnn_Tensor_t& output = graphInfo.outputTensors[idx].tensor;
            std::vector<size_t> dims;
            fillDims(dims, output.currentDimensions, output.rank);
            if (dims.size() == 4) {
                // nhwc to nchw
                auto t  = dims[3];
                dims[3] = dims[2];
                dims[2] = dims[1];
                dims[1] = t;
            }
            while (dims.size() < 4) {
                dims.emplace_back(1);
            }
            outputDims.emplace_back(dims);
        }

        return outputDims;
    }

    bool QnnWrapperV1::setupInputAndOutputTensors() {
        auto returnStatus     = true;
        GraphInfo_t graphInfo = (*m_GraphsInfo)[0];

        if (!setupTensors(&m_input, graphInfo.numInputTensors, (graphInfo.inputTensors))) {
            SIMPLE_LOG_ERROR("Failure in setting up input tensors\n");
            returnStatus = false;
        }
        if (!setupTensors(&m_output, graphInfo.numOutputTensors, (graphInfo.outputTensors))) {
            SIMPLE_LOG_ERROR("Failure in setting up output tensors\n");
            returnStatus = false;
        }
        if (!returnStatus) {
            SIMPLE_LOG_ERROR("Failure in setupInputAndOutputTensors, cleaning up resources\n");
            tearDownInputAndOutputTensors();
            SIMPLE_LOG_ERROR("Failure in setupInputAndOutputTensors, done cleaning up resources\n");
        }
        return returnStatus;
    }

    bool QnnWrapperV1::populateInputTensors(std::vector<uint8_t*> inputBuffers,
                                            std::vector<size_t> inputBuffersLen,
                                            DataType dataType,
                                            DataLayout layout) {
        if (nullptr == m_input) {
            SIMPLE_LOG_ERROR("inputs is nullptr\n");
            return false;
        }

        GraphInfo_t graphInfo = (*m_GraphsInfo)[0];
        auto inputCount       = graphInfo.numInputTensors;

        if (inputBuffers.size() != inputCount) {
            SIMPLE_LOG_ERROR(
                "Incorrect amount of Input Buffers for graph. Expected: %i, received: %i\n",
                inputCount,
                inputBuffers.size());
            return false;
        }
        for (size_t inputIdx = 0; inputIdx < inputCount; inputIdx++) {
            if (!populateInputTensor(inputBuffers[inputIdx],
                                     inputBuffersLen[inputIdx],
                                     &(m_input[inputIdx]),
                                     dataType,
                                     layout)) {
                SIMPLE_LOG_ERROR("populateInputTensor failure for input: %i\n", inputIdx);
                return false;
            }
        }
        return true;
    }

    bool QnnWrapperV1::executeGraphs() {
        GraphInfo_t graphInfo = (*m_GraphsInfo)[0];
        Qnn_ErrorHandle_t executeStatus =
            m_qnnFunctionPointers.qnnInterface.graphExecute(graphInfo.graph,
                                                            m_input,
                                                            graphInfo.numInputTensors,
                                                            m_output,
                                                            graphInfo.numOutputTensors,
                                                            m_profileBackendHandle,
                                                            nullptr);

        return executeStatus == QNN_GRAPH_NO_ERROR;
    }

    bool QnnWrapperV1::populateOutputBuffer(std::vector<uint8_t*>& outputBuffers,
                                            std::vector<size_t> outBuffersLen,
                                            DataType dataType) {
        if (dataType != DataType::FLOAT) {
            SIMPLE_LOG_ERROR("Only suport populate FLOAT output\n");
            return false;
        }

        bool returnStatus     = true;
        GraphInfo_t graphInfo = (*m_GraphsInfo)[0];

        // Qnn_TensorWrapper_t* outputWrappers = graphInfo.outputTensors;
        uint32_t numOutputs = graphInfo.numOutputTensors;

        for (size_t outputIdx = 0; outputIdx < numOutputs; outputIdx++) {
            SIMPLE_LOG_DEBUG("populate output for outputIdx: %i\n", outputIdx);
            Qnn_Tensor_t& output = m_output[outputIdx];

            std::vector<size_t> dims;
            fillDims(dims, output.currentDimensions, output.rank);

            size_t length                  = 0;
            std::tie(returnStatus, length) = calculateLength(dims, QNN_DATATYPE_FLOAT_32);
            if (!returnStatus || length != outBuffersLen[outputIdx]) {
                SIMPLE_LOG_ERROR(
                    "Populate output length error: %i, %i\n", length, outBuffersLen[outputIdx]);
                return false;
            }

            if (output.dataType == QNN_DATATYPE_FLOAT_32) {
                memcpy(outputBuffers[outputIdx],
                       reinterpret_cast<uint8_t*>(output.clientBuf.data),
                       length);
            } else {
                returnStatus = convertToFloat_NCHW((float*)outputBuffers[outputIdx], &output);
                if (!returnStatus) {
                    return returnStatus;
                }
            }
        }
        return returnStatus;
    }

    bool QnnWrapperV1::tearDownInputAndOutputTensors() {
        if (nullptr == m_GraphsInfo) {
            return true;
        }
        GraphInfo_t graphInfo = (*m_GraphsInfo)[0];
        if (nullptr != m_input) {
            SIMPLE_LOG_INFO("cleaning up resources for input tensors\n");
            tearDownTensors(m_input, graphInfo.numInputTensors);
            m_input = nullptr;
        }
        if (nullptr != m_output) {
            SIMPLE_LOG_INFO("cleaning up resources for output tensors\n");
            tearDownTensors(m_output, graphInfo.numOutputTensors);
            m_output = nullptr;
        }
        return true;
    }

    template <class T>
    static inline T resolveSymbol(void* libHandle, const char* sym) {
        T ptr = (T)dlsym(libHandle, sym);
        if (ptr == nullptr) {
            SIMPLE_LOG_ERROR("Unable to access symbol [%s]. dlerror(): %s\n", sym, dlerror());
        }
        return ptr;
    }

    bool QnnWrapperV1::getQnnBackendFunctionPointers(std::string backendPath) {
        m_libBackendHandle = dlopen(backendPath.c_str(), RTLD_NOW | RTLD_LOCAL);
        if (nullptr == m_libBackendHandle) {
            SIMPLE_LOG_ERROR("Unable to load backend. dlerror(): %s\n", dlerror());
            return false;
        }

        // Get QNN Interface
        QnnInterfaceGetProvidersFn_t getInterfaceProviders{nullptr};
        getInterfaceProviders = resolveSymbol<QnnInterfaceGetProvidersFn_t>(
            m_libBackendHandle, "QnnInterface_getProviders\n");
        if (nullptr == getInterfaceProviders) {
            return false;
        }

        QnnInterface_t* interfaceProviders{nullptr};
        uint32_t numProviders{0};
        Qnn_ErrorHandle_t error =
            getInterfaceProviders((const QnnInterface_t**)&interfaceProviders, &numProviders);
        if (QNN_SUCCESS != error) {
            SIMPLE_LOG_ERROR("Failed to get interface providers. error = %i\n",
                             static_cast<int>(error));
            return false;
        }
        if (nullptr == interfaceProviders) {
            SIMPLE_LOG_ERROR(
                "Failed to get interface providers: null interface providers received\n");
            return false;
        }
        if (0 == numProviders) {
            SIMPLE_LOG_ERROR("Failed to get interface providers: 0 interface providers\n");
            return false;
        }

        bool foundValidInterface{false};
        for (size_t pIdx = 0; pIdx < numProviders; pIdx++) {
            if (QNN_API_VERSION_MAJOR == interfaceProviders[pIdx].apiVersion.coreApiVersion.major &&
                QNN_API_VERSION_MINOR <= interfaceProviders[pIdx].apiVersion.coreApiVersion.minor) {
                foundValidInterface = true;
                m_qnnFunctionPointers.qnnInterface =
                    interfaceProviders[pIdx].QNN_INTERFACE_VER_NAME;
                break;
            }
        }

        if (!foundValidInterface) {
            SIMPLE_LOG_ERROR("Unable to find a valid interface\n");
            return false;
        }

        return true;
    }

    bool QnnWrapperV1::getQnnSystemFunctionPointers(std::string systemLibraryPath) {
        m_libSystemHandle = dlopen(systemLibraryPath.c_str(), RTLD_NOW | RTLD_LOCAL);
        if (nullptr == m_libSystemHandle) {
            SIMPLE_LOG_ERROR("Unable to load system library. dlerror(): %s\n", dlerror());
            return false;
        }

        QnnSystemInterfaceGetProvidersFn_t getSystemInterfaceProviders{nullptr};
        getSystemInterfaceProviders = resolveSymbol<QnnSystemInterfaceGetProvidersFn_t>(
            m_libSystemHandle, "QnnSystemInterface_getProviders\n");
        if (nullptr == getSystemInterfaceProviders) {
            return false;
        }
        QnnSystemInterface_t* systemInterfaceProviders{nullptr};
        uint32_t numProviders{0};
        Qnn_ErrorHandle_t error = getSystemInterfaceProviders(
            (const QnnSystemInterface_t**)&systemInterfaceProviders, &numProviders);
        if (QNN_SUCCESS != error) {
            SIMPLE_LOG_ERROR("Failed to get system interface providers. error = %i\n",
                             static_cast<int>(error));
            return false;
        }
        if (nullptr == systemInterfaceProviders) {
            SIMPLE_LOG_ERROR(
                "Failed to get system interface providers: null interface providers received\n");
            return false;
        }
        if (0 == numProviders) {
            SIMPLE_LOG_ERROR("Failed to get interface providers: 0 interface providers\n");
            return false;
        }
        bool foundValidSystemInterface{false};
        for (size_t pIdx = 0; pIdx < numProviders; pIdx++) {
            if (QNN_SYSTEM_API_VERSION_MAJOR ==
                    systemInterfaceProviders[pIdx].systemApiVersion.major &&
                QNN_SYSTEM_API_VERSION_MINOR <=
                    systemInterfaceProviders[pIdx].systemApiVersion.minor) {
                foundValidSystemInterface = true;
                m_qnnFunctionPointers.qnnSystemInterface =
                    systemInterfaceProviders[pIdx].QNN_SYSTEM_INTERFACE_VER_NAME;
                break;
            }
        }
        if (!foundValidSystemInterface) {
            SIMPLE_LOG_ERROR("Unable to find a valid system interface\n");
            return false;
        }
        return true;
    }

    // Function to help extract tensorsInfos from flatbuffers structures.
    // bool QnnWrapperV1::extractTensorsInfo(
    //     const flatbuffers::Vector<flatbuffers::Offset<QnnTensorInfo>>* fbTensorInfosVector,
    //     std::string graphName,
    //     std::unordered_map<std::string, std::unordered_map<uint32_t, std::string>>&
    //         graphTensorIdToNamesMap,
    //     uint32_t tensorsCount) {
    //     auto returnStatus = true;
    //     if (true == returnStatus) {
    //         for (size_t tIdx = 0; tIdx < tensorsCount; tIdx++) {
    //             SIMPLE_LOG_DEBUG("Extracting tensorInfo for tensor Idx: {}", tIdx);
    //             if (graphTensorIdToNamesMap.find(graphName) == graphTensorIdToNamesMap.end()) {
    //                 graphTensorIdToNamesMap[graphName] =
    //                     std::unordered_map<uint32_t, std::string>();
    //             }
    //             auto fbTensorInfo = fbTensorInfosVector->Get(tIdx);
    //             if (fbTensorInfo->name() != nullptr) {
    //                 graphTensorIdToNamesMap[graphName][fbTensorInfo->id()] =
    //                     fbTensorInfo->name()->str();
    //             } else {
    //                 SIMPLE_LOG_DEBUG(
    //                     "fbTensorInfo->name() is nullptr for graph [{}] and tensorId [{}].",
    //                     graphName.c_str(),
    //                     fbTensorInfo->id());
    //                 graphTensorIdToNamesMap[graphName][fbTensorInfo->id()] = "";
    //             }
    //         }
    //     }
    //     return returnStatus;
    // }

    // Function to help extract graphs' metadata from loaded flatbuffers structure
    // in deserializeData().
    // bool QnnWrapperV1::extractGraphsInfo(
    //     const ContextCache* contextCache,
    //     std::unordered_map<std::string, std::unordered_map<uint32_t, std::string>>&
    //         graphTensorIdToNamesMap,
    //     uint32_t* graphsCount) {
    //     auto returnStatus   = true;
    //     *graphsCount        = contextCache->graphsCount();
    //     auto fbGraphsVector = contextCache->graphsInfo();
    //     if (returnStatus == true) {
    //         for (size_t gIdx = 0; gIdx < *graphsCount; gIdx++) {
    //             SIMPLE_LOG_DEBUG("Extracting graphsInfo for graph Idx: {}", gIdx);
    //             auto fbGraph = fbGraphsVector->Get(gIdx);
    //             if (true != extractTensorsInfo(fbGraph->inputTensorsInfo(),
    //                                            fbGraph->name()->str(),
    //                                            graphTensorIdToNamesMap,
    //                                            fbGraph->inputTensorsCount())) {
    //                 returnStatus = false;
    //                 break;
    //             }
    //             if (true != extractTensorsInfo(fbGraph->outputTensorsInfo(),
    //                                            fbGraph->name()->str(),
    //                                            graphTensorIdToNamesMap,
    //                                            fbGraph->outputTensorsCount())) {
    //                 returnStatus = false;
    //                 break;
    //             }
    //         }
    //     }
    //     return returnStatus;
    // }

    // Function to deserialize flatbuffers related to caching.
    //  1. Flatbuffers are loaded from a binary buffer.
    //  2. Metadata containing a map of tenor id to names is populated.
    //  3. Binary blob is retrieved and copied into a uint8_t buffer.
    // bool QnnWrapperV1::deserializeData(const uint8_t* buffer,
    //                                    const size_t bufferSize,
    //                                    GraphTensorIdToNameMap& graphTensorIdToNamesMap,
    //                                    uint32_t* graphsCount,
    //                                    uint8_t*& binaryCache,
    //                                    uint64_t& binaryCacheSize) {
    //     // Verify the buffer is well-formed
    //     flatbuffers::Verifier verifier(buffer, bufferSize);
    //     if (!VerifyContextCacheBuffer(verifier)) {
    //         SIMPLE_LOG_ERROR("Invalid flatbuffer binary");
    //         return false;
    //     }
    //     auto contextCache = GetContextCache(buffer);
    //     binaryCacheSize   = contextCache->binaryCacheSize();
    //     binaryCache =
    //         static_cast<uint8_t*>(m_allocator_ptr->Malloc(binaryCacheSize * sizeof(uint8_t)));
    //     if (nullptr == binaryCache) {
    //         SIMPLE_LOG_ERROR("Failed to allocate memory for binaryCache");
    //         return false;
    //     }
    //     memcpy(binaryCache, contextCache->binaryCache()->Data(), binaryCacheSize);

    //     if (true != extractGraphsInfo(contextCache, graphTensorIdToNamesMap, graphsCount)) {
    //         SIMPLE_LOG_ERROR("Failed to extract graphsInfo.");
    //         return false;
    //     }
    //     return true;
    // }

    bool QnnWrapperV1::copyTensorsInfo(const Qnn_Tensor_t* tensorsInfoSrc,
                                       Qnn_TensorWrapper_t*& tensorWrappers,
                                       uint32_t tensorsCount) {
        auto returnStatus = true;
        tensorWrappers    = (Qnn_TensorWrapper_t*)m_allocator_ptr->Malloc(
            static_cast<size_t>(tensorsCount * sizeof(Qnn_TensorWrapper_t)));
        m_allocator_ptr->Memset(tensorWrappers, 0x00, tensorsCount * sizeof(Qnn_TensorWrapper_t));
        if (nullptr == tensorWrappers) {
            SIMPLE_LOG_ERROR("Failed to allocate memory for tensorWrappers\n");
            return false;
        }
        if (returnStatus) {
            for (size_t tIdx = 0; tIdx < tensorsCount; tIdx++) {
                SIMPLE_LOG_DEBUG("Extracting tensorInfo for tensor Idx: %i\n", tIdx);
                tensorWrappers[tIdx].name              = nullptr;
                tensorWrappers[tIdx].tensor.id         = tensorsInfoSrc[tIdx].id;
                tensorWrappers[tIdx].tensor.type       = tensorsInfoSrc[tIdx].type;
                tensorWrappers[tIdx].tensor.dataFormat = tensorsInfoSrc[tIdx].dataFormat;
                tensorWrappers[tIdx].tensor.dataType   = tensorsInfoSrc[tIdx].dataType;
                tensorWrappers[tIdx].tensor.quantizeParams.quantizationEncoding =
                    QNN_QUANTIZATION_ENCODING_UNDEFINED;
                if (tensorsInfoSrc[tIdx].quantizeParams.quantizationEncoding ==
                    QNN_QUANTIZATION_ENCODING_SCALE_OFFSET) {
                    tensorWrappers[tIdx].tensor.quantizeParams.quantizationEncoding =
                        tensorsInfoSrc[tIdx].quantizeParams.quantizationEncoding;
                    tensorWrappers[tIdx].tensor.quantizeParams.scaleOffsetEncoding =
                        tensorsInfoSrc[tIdx].quantizeParams.scaleOffsetEncoding;
                } else if (tensorsInfoSrc[tIdx].quantizeParams.quantizationEncoding ==
                           QNN_QUANTIZATION_ENCODING_AXIS_SCALE_OFFSET) {
                    tensorWrappers[tIdx].tensor.quantizeParams.quantizationEncoding =
                        tensorsInfoSrc[tIdx].quantizeParams.quantizationEncoding;
                    tensorWrappers[tIdx].tensor.quantizeParams.axisScaleOffsetEncoding.axis =
                        tensorsInfoSrc[tIdx].quantizeParams.axisScaleOffsetEncoding.axis;
                    tensorWrappers[tIdx]
                        .tensor.quantizeParams.axisScaleOffsetEncoding.numScaleOffsets =
                        tensorsInfoSrc[tIdx].quantizeParams.axisScaleOffsetEncoding.numScaleOffsets;
                    if (tensorsInfoSrc[tIdx]
                            .quantizeParams.axisScaleOffsetEncoding.numScaleOffsets > 0) {
                        tensorWrappers[tIdx]
                            .tensor.quantizeParams.axisScaleOffsetEncoding.scaleOffset =
                            (Qnn_ScaleOffset_t*)m_allocator_ptr->Malloc(
                                tensorsInfoSrc[tIdx]
                                    .quantizeParams.axisScaleOffsetEncoding.numScaleOffsets *
                                sizeof(Qnn_ScaleOffset_t));
                        if (tensorWrappers[tIdx]
                                .tensor.quantizeParams.axisScaleOffsetEncoding.scaleOffset) {
                            for (size_t idx = 0;
                                 idx < tensorsInfoSrc[tIdx]
                                           .quantizeParams.axisScaleOffsetEncoding.numScaleOffsets;
                                 idx++) {
                                tensorWrappers[tIdx]
                                    .tensor.quantizeParams.axisScaleOffsetEncoding.scaleOffset[idx]
                                    .scale =
                                    tensorsInfoSrc[tIdx]
                                        .quantizeParams.axisScaleOffsetEncoding.scaleOffset[idx]
                                        .scale;
                                tensorWrappers[tIdx]
                                    .tensor.quantizeParams.axisScaleOffsetEncoding.scaleOffset[idx]
                                    .offset =
                                    tensorsInfoSrc[tIdx]
                                        .quantizeParams.axisScaleOffsetEncoding.scaleOffset[idx]
                                        .offset;
                            }
                        }
                    }
                }
                tensorWrappers[tIdx].tensor.rank              = tensorsInfoSrc[tIdx].rank;
                tensorWrappers[tIdx].tensor.maxDimensions     = nullptr;
                tensorWrappers[tIdx].tensor.currentDimensions = nullptr;
                if (tensorWrappers[tIdx].tensor.rank > 0) {
                    tensorWrappers[tIdx].tensor.maxDimensions =
                        (uint32_t*)m_allocator_ptr->Malloc(tensorsInfoSrc->rank * sizeof(uint32_t));
                    if (tensorWrappers[tIdx].tensor.maxDimensions) {
                        memcpy(tensorWrappers[tIdx].tensor.maxDimensions,
                               tensorsInfoSrc[tIdx].maxDimensions,
                               tensorsInfoSrc[tIdx].rank * sizeof(uint32_t));
                    }
                    tensorWrappers[tIdx].tensor.currentDimensions =
                        (uint32_t*)m_allocator_ptr->Malloc(tensorsInfoSrc->rank * sizeof(uint32_t));
                    if (tensorWrappers[tIdx].tensor.currentDimensions) {
                        memcpy(tensorWrappers[tIdx].tensor.currentDimensions,
                               tensorsInfoSrc[tIdx].currentDimensions,
                               tensorsInfoSrc[tIdx].rank * sizeof(uint32_t));
                    }
                }
            }
        }
        return returnStatus;
    }

    bool QnnWrapperV1::copyGraphsInfoV1(const QnnSystemContext_GraphInfoV1_t* graphInfoSrc,
                                        GraphInfo_t* graphInfoDst) {
        graphInfoDst->graphName = nullptr;
        if (graphInfoSrc->graphName) {
            graphInfoDst->graphName =
                strndup(graphInfoSrc->graphName, strlen(graphInfoSrc->graphName));
        }
        graphInfoDst->inputTensors    = nullptr;
        graphInfoDst->numInputTensors = 0;
        if (graphInfoSrc->graphInputs) {
            if (!copyTensorsInfo(graphInfoSrc->graphInputs,
                                 graphInfoDst->inputTensors,
                                 graphInfoSrc->numGraphInputs)) {
                return false;
            }
            graphInfoDst->numInputTensors = graphInfoSrc->numGraphInputs;
        }
        graphInfoDst->outputTensors    = nullptr;
        graphInfoDst->numOutputTensors = 0;
        if (graphInfoSrc->graphOutputs) {
            if (!copyTensorsInfo(graphInfoSrc->graphOutputs,
                                 graphInfoDst->outputTensors,
                                 graphInfoSrc->numGraphOutputs)) {
                return false;
            }
            graphInfoDst->numOutputTensors = graphInfoSrc->numGraphOutputs;
        }
        return true;
    }

    bool QnnWrapperV1::copyGraphsInfo(const QnnSystemContext_GraphInfo_t* graphsInput,
                                      const uint32_t numGraphs,
                                      GraphInfo_t**& graphsInfo) {
        if (!graphsInput) {
            SIMPLE_LOG_ERROR("Received nullptr for graphsInput\n");
            return false;
        }
        auto returnStatus = true;
        graphsInfo        = (GraphInfo_t**)m_allocator_ptr->Malloc(
            static_cast<size_t>(numGraphs * sizeof(GraphInfo_t*)));
        m_allocator_ptr->Memset(graphsInfo, 0x00, numGraphs * sizeof(GraphInfo_t*));
        GraphInfo_t* graphInfoArr = (GraphInfo_t*)m_allocator_ptr->Malloc(
            static_cast<size_t>(numGraphs * sizeof(GraphInfo_t)));
        m_allocator_ptr->Memset(graphInfoArr, 0x00, numGraphs * sizeof(GraphInfo_t));
        if (nullptr == graphsInfo || nullptr == graphInfoArr) {
            SIMPLE_LOG_ERROR("Failure to allocate memory for *graphInfo\n");
            returnStatus = false;
        }
        if (true == returnStatus) {
            for (size_t gIdx = 0; gIdx < numGraphs; gIdx++) {
                SIMPLE_LOG_DEBUG("Extracting graphsInfo for graph Idx: %i\n", gIdx);
                if (graphsInput[gIdx].version == QNN_SYSTEM_CONTEXT_GRAPH_INFO_VERSION_1) {
                    copyGraphsInfoV1(&graphsInput[gIdx].graphInfoV1, &graphInfoArr[gIdx]);
                }
                graphsInfo[gIdx] = graphInfoArr + gIdx;
            }
        }
        if (true != returnStatus) {
            SIMPLE_LOG_ERROR("Received an ERROR during extractGraphsInfo. Freeing resources\n");
            if (graphsInfo) {
                for (uint32_t gIdx = 0; gIdx < numGraphs; gIdx++) {
                    if (graphsInfo[gIdx]) {
                        if (nullptr != graphsInfo[gIdx]->graphName) {
                            m_allocator_ptr->Free(graphsInfo[gIdx]->graphName);
                            graphsInfo[gIdx]->graphName = nullptr;
                        }
                        freeQnnTensorWrappers(graphsInfo[gIdx]->inputTensors,
                                              graphsInfo[gIdx]->numInputTensors);
                        freeQnnTensorWrappers(graphsInfo[gIdx]->outputTensors,
                                              graphsInfo[gIdx]->numOutputTensors);
                    }
                }
                m_allocator_ptr->Free(*graphsInfo);
            }
            m_allocator_ptr->Free(graphsInfo);
            graphsInfo = nullptr;
        }
        return true;
    }

    bool QnnWrapperV1::copyMetadataToGraphsInfo(const QnnSystemContext_BinaryInfo_t* binaryInfo,
                                                GraphInfo_t**& graphsInfo,
                                                uint32_t& graphsCount) {
        if (nullptr == binaryInfo) {
            SIMPLE_LOG_ERROR("binaryInfo is nullptr\n");
            return false;
        }
        graphsCount = 0;
        if (binaryInfo->version == QNN_SYSTEM_CONTEXT_BINARY_INFO_VERSION_1) {
            if (binaryInfo->contextBinaryInfoV1.graphs) {
                if (!copyGraphsInfo(binaryInfo->contextBinaryInfoV1.graphs,
                                    binaryInfo->contextBinaryInfoV1.numGraphs,
                                    graphsInfo)) {
                    SIMPLE_LOG_ERROR("Failed while copying graphs Info\n");
                    return false;
                }
                graphsCount = binaryInfo->contextBinaryInfoV1.numGraphs;
                return true;
            }
        }
        SIMPLE_LOG_ERROR("Unrecognized system context binary info version\n");
        return false;
    }

    bool
    QnnWrapperV1::populateTensorNamesFromMetadata(GraphTensorIdToNameMap& graphTensorIdToNamesMap,
                                                  GraphInfo_t**& graphsInfo,
                                                  const uint32_t graphsCount) {
        for (uint32_t gIdx = 0; gIdx < graphsCount; gIdx++) {
            std::string graphName = std::string((*graphsInfo)[gIdx].graphName);
            if (graphTensorIdToNamesMap.find(graphName) == graphTensorIdToNamesMap.end()) {
                SIMPLE_LOG_ERROR("Graph [%s] not found in metadata\n", graphName.c_str());
                return false;
            }
            for (uint32_t tIdx = 0; tIdx < (*graphsInfo)[gIdx].numInputTensors; tIdx++) {
                auto tensorId = (*graphsInfo)[gIdx].inputTensors[tIdx].tensor.id;
                if (graphTensorIdToNamesMap[graphName].find(tensorId) ==
                    graphTensorIdToNamesMap[graphName].end()) {
                    SIMPLE_LOG_ERROR(
                        "Input tensor name for [%i] in graph [%s] not found in metadata\n",
                        tensorId,
                        graphName.c_str());
                    return false;
                }
                (*graphsInfo)[gIdx].inputTensors[tIdx].name =
                    strndup(graphTensorIdToNamesMap[graphName][tensorId].c_str(),
                            strlen(graphTensorIdToNamesMap[graphName][tensorId].c_str()));
            }
            for (uint32_t tIdx = 0; tIdx < (*graphsInfo)[gIdx].numOutputTensors; tIdx++) {
                auto tensorId = (*graphsInfo)[gIdx].outputTensors[tIdx].tensor.id;
                if (graphTensorIdToNamesMap[graphName].find(tensorId) ==
                    graphTensorIdToNamesMap[graphName].end()) {
                    SIMPLE_LOG_ERROR(
                        "Output tensor name for [%i] in graph [%s] not found in metadata\n",
                        tensorId,
                        graphName.c_str());
                    return false;
                }
                (*graphsInfo)[gIdx].outputTensors[tIdx].name =
                    strndup(graphTensorIdToNamesMap[graphName][tensorId].c_str(),
                            strlen(graphTensorIdToNamesMap[graphName][tensorId].c_str()));
            }
        }
        return true;
    }

    bool QnnWrapperV1::setupTensors(Qnn_Tensor_t** tensors,
                                    uint32_t tensorCount,
                                    Qnn_TensorWrapper_t* tensorWrappers) {
        if (nullptr == tensorWrappers) {
            SIMPLE_LOG_ERROR("tensorWrappers is nullptr\n");
            return false;
        }
        if (0 == tensorCount) {
            SIMPLE_LOG_INFO("tensor count is 0. Nothing to setup\n");
            return true;
        }
        auto returnStatus = true;
        *tensors          = (Qnn_Tensor_t*)m_allocator_ptr->Malloc(
            static_cast<size_t>(tensorCount * sizeof(Qnn_Tensor_t)));
        m_allocator_ptr->Memset(*tensors, 0x00, tensorCount * sizeof(Qnn_Tensor_t));
        if (nullptr == *tensors) {
            SIMPLE_LOG_ERROR("mem alloc failed for *tensors\n");
            return returnStatus;
        }

        for (size_t tensorIdx = 0; tensorIdx < tensorCount; tensorIdx++) {
            Qnn_Tensor_t wrapperTensor = tensorWrappers[tensorIdx].tensor;
            std::vector<size_t> dims;
            fillDims(dims, wrapperTensor.currentDimensions, wrapperTensor.rank);
            returnStatus = allocateBuffer(
                reinterpret_cast<uint8_t**>(&((*tensors) + tensorIdx)->clientBuf.data),
                dims,
                wrapperTensor.dataType);
            bool datautilStatus;
            size_t length;
            std::tie(datautilStatus, length) = calculateLength(dims, wrapperTensor.dataType);
            if (!datautilStatus) {
                returnStatus = datautilStatus;
            }
            ((*tensors) + tensorIdx)->clientBuf.dataSize = length;
            if (true == returnStatus) {
                SIMPLE_LOG_DEBUG("allocateBuffer successful\n");
                returnStatus = deepCopyQnnTensorInfo(((*tensors) + tensorIdx), &wrapperTensor);
            }
            if (true == returnStatus) {
                SIMPLE_LOG_DEBUG("deepCopyQnnTensorInfo successful\n");
                ((*tensors) + tensorIdx)->memType = QNN_TENSORMEMTYPE_RAW;
            }
            if (true != returnStatus) {
                SIMPLE_LOG_ERROR("Failure in setupTensors, cleaning up resources\n");
                if (nullptr != ((*tensors) + tensorIdx)->clientBuf.data) {
                    m_allocator_ptr->Free(((*tensors) + tensorIdx)->clientBuf.data);
                }
                tearDownTensors(*tensors, tensorIdx);
                *tensors     = nullptr;
                returnStatus = false;
                SIMPLE_LOG_ERROR("Failure in setupTensors, done cleaning up resources\n");
                return returnStatus;
            }
        }
        return returnStatus;
    }

    bool QnnWrapperV1::fillDims(std::vector<size_t>& dims, uint32_t* inDimensions, uint32_t rank) {
        if (nullptr == inDimensions) {
            SIMPLE_LOG_ERROR("input dimensions is nullptr\n");
            return false;
        }
        for (size_t r = 0; r < rank; r++) {
            dims.push_back(inDimensions[r]);
        }
        return true;
    }

    size_t QnnWrapperV1::calculateElementCount(std::vector<size_t> dims) {
        if (dims.size() == 0) {
            return 0;
        }
        return std::accumulate(dims.begin(), dims.end(), 1, std::multiplies<size_t>());
    }

    template <typename T>
    bool QnnWrapperV1::allocateBuffer(T** buffer, size_t& elementCount) {
        SIMPLE_LOG_DEBUG("ElementCount: %i, sizeof(T): %i, total size: %i\n",
                         elementCount,
                         sizeof(T),
                         elementCount * sizeof(T));
        *buffer = (T*)m_allocator_ptr->Malloc(elementCount * sizeof(T));
        if (nullptr == *buffer) {
            SIMPLE_LOG_ERROR("mem alloc failed for *buffer\n");
            return false;
        }
        return true;
    }

    bool QnnWrapperV1::allocateBuffer(uint8_t** buffer,
                                      std::vector<size_t> dims,
                                      Qnn_DataType_t dataType) {
        size_t elementCount = calculateElementCount(dims);
        auto returnStatus   = true;
        switch (dataType) {
            case QNN_DATATYPE_FLOAT_32:
                SIMPLE_LOG_DEBUG("allocating float buffer\n");
                returnStatus =
                    allocateBuffer<float>(reinterpret_cast<float**>(buffer), elementCount);
                break;

            case QNN_DATATYPE_UINT_8:
            case QNN_DATATYPE_UFIXED_POINT_8:
                SIMPLE_LOG_DEBUG("allocating uint8_t buffer\n");
                returnStatus =
                    allocateBuffer<uint8_t>(reinterpret_cast<uint8_t**>(buffer), elementCount);
                break;

            case QNN_DATATYPE_UINT_16:
            case QNN_DATATYPE_UFIXED_POINT_16:
                SIMPLE_LOG_DEBUG("allocating uint16_t buffer\n");
                returnStatus =
                    allocateBuffer<uint16_t>(reinterpret_cast<uint16_t**>(buffer), elementCount);
                break;

            case QNN_DATATYPE_UINT_32:
                SIMPLE_LOG_DEBUG("allocating uint32_t buffer\n");
                returnStatus =
                    allocateBuffer<uint32_t>(reinterpret_cast<uint32_t**>(buffer), elementCount);
                break;

            case QNN_DATATYPE_INT_8:
                SIMPLE_LOG_DEBUG("allocating int8_t buffer\n");
                returnStatus =
                    allocateBuffer<int8_t>(reinterpret_cast<int8_t**>(buffer), elementCount);
                break;

            case QNN_DATATYPE_INT_16:
                SIMPLE_LOG_DEBUG("allocating int16_t buffer\n");
                returnStatus =
                    allocateBuffer<int16_t>(reinterpret_cast<int16_t**>(buffer), elementCount);
                break;

            case QNN_DATATYPE_INT_32:
                SIMPLE_LOG_DEBUG("allocating int32_t buffer\n");
                returnStatus =
                    allocateBuffer<int32_t>(reinterpret_cast<int32_t**>(buffer), elementCount);
                break;

            case QNN_DATATYPE_BOOL_8:
                SIMPLE_LOG_DEBUG("allocating bool buffer\n");
                returnStatus =
                    allocateBuffer<uint8_t>(reinterpret_cast<uint8_t**>(buffer), elementCount);
                break;

            default:
                SIMPLE_LOG_ERROR("Datatype not supported yet\n");
                returnStatus = false;
                break;
        }

        return returnStatus;
    }

    std::tuple<bool, size_t> QnnWrapperV1::getDataTypeSizeInBytes(Qnn_DataType_t dataType) {
        const std::map<Qnn_DataType_t, size_t> g_dataTypeToSize = {
            {QNN_DATATYPE_INT_8, 1},
            {QNN_DATATYPE_INT_16, 2},
            {QNN_DATATYPE_INT_32, 4},
            {QNN_DATATYPE_INT_64, 8},
            {QNN_DATATYPE_UINT_8, 1},
            {QNN_DATATYPE_UINT_16, 2},
            {QNN_DATATYPE_UINT_32, 4},
            {QNN_DATATYPE_UINT_64, 8},
            {QNN_DATATYPE_FLOAT_16, 2},
            {QNN_DATATYPE_FLOAT_32, 4},
            {QNN_DATATYPE_SFIXED_POINT_8, 1},
            {QNN_DATATYPE_SFIXED_POINT_16, 2},
            {QNN_DATATYPE_SFIXED_POINT_32, 4},
            {QNN_DATATYPE_UFIXED_POINT_8, 1},
            {QNN_DATATYPE_UFIXED_POINT_16, 2},
            {QNN_DATATYPE_UFIXED_POINT_32, 4},
            {QNN_DATATYPE_BOOL_8, 1},
        };

        if (g_dataTypeToSize.find(dataType) == g_dataTypeToSize.end()) {
            SIMPLE_LOG_ERROR("Invalid qnn data type provided\n");
            return std::make_tuple(false, 0);
        }
        return std::make_tuple(true, g_dataTypeToSize.find(dataType)->second);
    }

    std::tuple<bool, size_t> QnnWrapperV1::calculateLength(std::vector<size_t> dims,
                                                           Qnn_DataType_t dataType) {
        if (dims.size() == 0) {
            SIMPLE_LOG_ERROR("dims.size() is zero\n");
            return std::make_tuple(false, 0);
        }

        bool returnStatus{true};
        size_t length{0};

        std::tie(returnStatus, length) = getDataTypeSizeInBytes(dataType);
        if (true != returnStatus) {
            return std::make_tuple(returnStatus, 0);
        }
        length *= calculateElementCount(dims);
        return std::make_tuple(true, length);
    }

    bool QnnWrapperV1::deepCopyQnnTensorInfo(Qnn_Tensor_t* dest, Qnn_Tensor_t* src) {
        if (nullptr == dest || nullptr == src) {
            SIMPLE_LOG_ERROR("Received nullptr\n");
            return false;
        }
        dest->id         = src->id;
        dest->type       = src->type;
        dest->dataFormat = src->dataFormat;
        dest->dataType   = src->dataType;
        dest->rank       = src->rank;
        memcpy(&(dest->quantizeParams), &(src->quantizeParams), sizeof(Qnn_QuantizeParams_t));
        size_t dimensionsSize   = src->rank * sizeof(uint32_t);
        dest->maxDimensions     = (uint32_t*)m_allocator_ptr->Malloc(dimensionsSize);
        dest->currentDimensions = (uint32_t*)m_allocator_ptr->Malloc(dimensionsSize);
        if (nullptr == dest->maxDimensions || nullptr == dest->currentDimensions) {
            if (dest->maxDimensions) {
                m_allocator_ptr->Free(dest->maxDimensions);
            }
            if (dest->currentDimensions) {
                m_allocator_ptr->Free(dest->currentDimensions);
            }
            return false;
        }
        memcpy(dest->maxDimensions, src->maxDimensions, dimensionsSize);
        memcpy(dest->currentDimensions, src->currentDimensions, dimensionsSize);
        return true;
    }

    void QnnWrapperV1::tearDownTensors(Qnn_Tensor_t* tensors, uint32_t tensorCount) {
        for (size_t tensorIdx = 0; tensorIdx < tensorCount; tensorIdx++) {
            SIMPLE_LOG_DEBUG("freeing resources for tensor: %i\n", tensorIdx);
            if (nullptr != tensors[tensorIdx].maxDimensions) {
                SIMPLE_LOG_DEBUG("freeing maxDimensions\n");
                m_allocator_ptr->Free(tensors[tensorIdx].maxDimensions);
            }
            if (nullptr != tensors[tensorIdx].currentDimensions) {
                SIMPLE_LOG_DEBUG("freeing currentDimensions\n");
                m_allocator_ptr->Free(tensors[tensorIdx].currentDimensions);
            }
            if (nullptr != tensors[tensorIdx].clientBuf.data) {
                SIMPLE_LOG_DEBUG("freeing clientBuf.data\n");
                m_allocator_ptr->Free(tensors[tensorIdx].clientBuf.data);
            }
        }
        m_allocator_ptr->Free(tensors);
    }

    bool QnnWrapperV1::populateInputTensor(uint8_t* buffer,
                                           size_t bufferLen,
                                           Qnn_Tensor_t* input,
                                           DataType inputDataType,
                                           DataLayout layout) {
        if (nullptr == input) {
            SIMPLE_LOG_ERROR("input is nullptr\n");
            return true;
        }

        std::vector<size_t> dims;
        fillDims(dims, input->currentDimensions, input->rank);

        size_t length                  = 0;
        bool returnStatus              = true;
        std::tie(returnStatus, length) = calculateLength(
            dims, inputDataType == DataType::FLOAT ? QNN_DATATYPE_FLOAT_32 : input->dataType);
        if (!returnStatus) {
            return returnStatus;
        }

        if (length != bufferLen) {
            SIMPLE_LOG_ERROR("populateInputTensor length error, %i %i\n", bufferLen, length);
            return false;
        }

        if (inputDataType == DataType::FLOAT && input->dataType != QNN_DATATYPE_FLOAT_32) {
            SIMPLE_LOG_DEBUG("Received FLOAT input, but model needs non-float input\n");
            if (layout == DataLayout::LAYOUT_NHWC) {
                if (!copyFromFloatToNative(reinterpret_cast<float*>(buffer), input)) {
                    SIMPLE_LOG_ERROR("copyFromFloatToNative failure\n");
                    return false;
                }
            } else { // layout == DataLayout::LAYOUT_NCHW
                if (!copyFromFloatToNative_NHWC(reinterpret_cast<float*>(buffer), input)) {
                    SIMPLE_LOG_ERROR("copyFromFloatToNative_NHWC failure\n");
                    return false;
                }
            }
        } else {
            memcpy(reinterpret_cast<uint8_t*>(input->clientBuf.data), buffer, length);
        }
        return true;
    }

    // Helper method to copy a float buffer, quantize it, and copy
    // it to a tensor (Qnn_Tensor_t) buffer.
    bool QnnWrapperV1::copyFromFloatToNative(float* floatBuffer, Qnn_Tensor_t* tensor) {
        if (nullptr == floatBuffer || nullptr == tensor) {
            SIMPLE_LOG_ERROR("copyFromFloatToNative(): received a nullptr\n");
            return false;
        }

        bool returnStatus = true;
        std::vector<size_t> dims;
        fillDims(dims, tensor->currentDimensions, tensor->rank);

        switch (tensor->dataType) {
            case QNN_DATATYPE_UFIXED_POINT_8:
                returnStatus =
                    floatToTfN<uint8_t>(static_cast<uint8_t*>(tensor->clientBuf.data),
                                        floatBuffer,
                                        tensor->quantizeParams.scaleOffsetEncoding.offset,
                                        tensor->quantizeParams.scaleOffsetEncoding.scale,
                                        calculateElementCount(dims));
                break;

            case QNN_DATATYPE_UFIXED_POINT_16:
                returnStatus =
                    floatToTfN<uint16_t>(static_cast<uint16_t*>(tensor->clientBuf.data),
                                         floatBuffer,
                                         tensor->quantizeParams.scaleOffsetEncoding.offset,
                                         tensor->quantizeParams.scaleOffsetEncoding.scale,
                                         calculateElementCount(dims));
                break;

            case QNN_DATATYPE_UINT_8:
                returnStatus = castFromFloat<uint8_t>(static_cast<uint8_t*>(tensor->clientBuf.data),
                                                      floatBuffer,
                                                      calculateElementCount(dims));
                break;

            case QNN_DATATYPE_UINT_16:
                returnStatus =
                    castFromFloat<uint16_t>(static_cast<uint16_t*>(tensor->clientBuf.data),
                                            floatBuffer,
                                            calculateElementCount(dims));
                break;

            case QNN_DATATYPE_UINT_32:
                returnStatus =
                    castFromFloat<uint32_t>(static_cast<uint32_t*>(tensor->clientBuf.data),
                                            floatBuffer,
                                            calculateElementCount(dims));
                break;

            case QNN_DATATYPE_INT_8:
                returnStatus = castFromFloat<int8_t>(static_cast<int8_t*>(tensor->clientBuf.data),
                                                     floatBuffer,
                                                     calculateElementCount(dims));
                break;

            case QNN_DATATYPE_INT_16:
                returnStatus = castFromFloat<int16_t>(static_cast<int16_t*>(tensor->clientBuf.data),
                                                      floatBuffer,
                                                      calculateElementCount(dims));
                break;

            case QNN_DATATYPE_INT_32:
                returnStatus = castFromFloat<int32_t>(static_cast<int32_t*>(tensor->clientBuf.data),
                                                      floatBuffer,
                                                      calculateElementCount(dims));
                break;

            case QNN_DATATYPE_BOOL_8:
                returnStatus = castFromFloat<uint8_t>(static_cast<uint8_t*>(tensor->clientBuf.data),
                                                      floatBuffer,
                                                      calculateElementCount(dims));
                break;

            default:
                SIMPLE_LOG_ERROR("Datatype not supported yet\n");
                returnStatus = false;
                break;
        }
        return returnStatus;
    }

    bool QnnWrapperV1::copyFromFloatToNative_NHWC(float* floatBuffer, Qnn_Tensor_t* tensor) {
        if (nullptr == floatBuffer || nullptr == tensor) {
            SIMPLE_LOG_ERROR("copyFromFloatToNative(): received a nullptr\n");
            return false;
        }

        bool returnStatus = true;
        std::vector<size_t> dims;
        fillDims(dims, tensor->currentDimensions, tensor->rank);

        switch (tensor->dataType) {
            case QNN_DATATYPE_UFIXED_POINT_8:
                returnStatus =
                    floatToTfN_NHWC<uint8_t>(static_cast<uint8_t*>(tensor->clientBuf.data),
                                             floatBuffer,
                                             tensor->quantizeParams.scaleOffsetEncoding.offset,
                                             tensor->quantizeParams.scaleOffsetEncoding.scale,
                                             dims);
                break;

            case QNN_DATATYPE_UFIXED_POINT_16:
                returnStatus =
                    floatToTfN_NHWC<uint16_t>(static_cast<uint16_t*>(tensor->clientBuf.data),
                                              floatBuffer,
                                              tensor->quantizeParams.scaleOffsetEncoding.offset,
                                              tensor->quantizeParams.scaleOffsetEncoding.scale,
                                              dims);
                break;

            case QNN_DATATYPE_UINT_8:
                returnStatus = castFromFloat_NHWC<uint8_t>(
                    static_cast<uint8_t*>(tensor->clientBuf.data), floatBuffer, dims);
                break;

            case QNN_DATATYPE_UINT_16:
                returnStatus = castFromFloat_NHWC<uint16_t>(
                    static_cast<uint16_t*>(tensor->clientBuf.data), floatBuffer, dims);
                break;

            case QNN_DATATYPE_UINT_32:
                returnStatus = castFromFloat_NHWC<uint32_t>(
                    static_cast<uint32_t*>(tensor->clientBuf.data), floatBuffer, dims);
                break;

            case QNN_DATATYPE_INT_8:
                returnStatus = castFromFloat_NHWC<int8_t>(
                    static_cast<int8_t*>(tensor->clientBuf.data), floatBuffer, dims);
                break;

            case QNN_DATATYPE_INT_16:
                returnStatus = castFromFloat_NHWC<int16_t>(
                    static_cast<int16_t*>(tensor->clientBuf.data), floatBuffer, dims);
                break;

            case QNN_DATATYPE_INT_32:
                returnStatus = castFromFloat_NHWC<int32_t>(
                    static_cast<int32_t*>(tensor->clientBuf.data), floatBuffer, dims);
                break;

            case QNN_DATATYPE_BOOL_8:
                returnStatus = castFromFloat_NHWC<uint8_t>(
                    static_cast<uint8_t*>(tensor->clientBuf.data), floatBuffer, dims);
                break;

            default:
                SIMPLE_LOG_ERROR("Datatype not supported yet\n");
                returnStatus = false;
                break;
        }
        return returnStatus;
    }

    // Convert data to float or de-quantization. This is used when
    // user requests for float output and the model produces
    // non-float output.
    bool QnnWrapperV1::convertToFloat_NCHW(float* out, Qnn_Tensor_t* tensor) {
        if (nullptr == tensor) {
            SIMPLE_LOG_ERROR("tensors is nullptr\n");
            return false;
        }
        std::vector<size_t> dims;
        fillDims(dims, tensor->currentDimensions, tensor->rank);
        auto returnStatus = true;
        switch (tensor->dataType) {
            case QNN_DATATYPE_UFIXED_POINT_8:
                if (!tfNToFloat_NCHW<uint8_t>(out,
                                              reinterpret_cast<uint8_t*>(tensor->clientBuf.data),
                                              tensor->quantizeParams.scaleOffsetEncoding.offset,
                                              tensor->quantizeParams.scaleOffsetEncoding.scale,
                                              dims)) {
                    SIMPLE_LOG_ERROR("failure in tfNToFloat<uint8_t>\n");
                    returnStatus = false;
                }
                break;

            case QNN_DATATYPE_UFIXED_POINT_16:
                if (!tfNToFloat_NCHW<uint16_t>(out,
                                               reinterpret_cast<uint16_t*>(tensor->clientBuf.data),
                                               tensor->quantizeParams.scaleOffsetEncoding.offset,
                                               tensor->quantizeParams.scaleOffsetEncoding.scale,
                                               dims)) {
                    SIMPLE_LOG_ERROR("failure in tfNToFloat<uint8_t>\n");
                    returnStatus = false;
                }
                break;

            case QNN_DATATYPE_UINT_8:
                if (!castToFloat_NCHW<uint8_t>(
                        out, reinterpret_cast<uint8_t*>(tensor->clientBuf.data), dims)) {
                    SIMPLE_LOG_ERROR("failure in castToFloat<uint8_t>\n");
                    returnStatus = false;
                }
                break;

            case QNN_DATATYPE_UINT_16:
                if (!castToFloat_NCHW<uint16_t>(
                        out, reinterpret_cast<uint16_t*>(tensor->clientBuf.data), dims)) {
                    SIMPLE_LOG_ERROR("failure in castToFloat<uint16_t>\n");
                    returnStatus = false;
                }
                break;

            case QNN_DATATYPE_UINT_32:
                if (!castToFloat_NCHW<uint32_t>(
                        out, reinterpret_cast<uint32_t*>(tensor->clientBuf.data), dims)) {
                    SIMPLE_LOG_ERROR("failure in castToFloat<uint32_t>\n");
                    returnStatus = false;
                }
                break;

            case QNN_DATATYPE_INT_8:
                if (!castToFloat_NCHW<int8_t>(
                        out, reinterpret_cast<int8_t*>(tensor->clientBuf.data), dims)) {
                    SIMPLE_LOG_ERROR("failure in castToFloat<int8_t>\n");
                    returnStatus = false;
                }
                break;

            case QNN_DATATYPE_INT_16:
                if (!castToFloat_NCHW<int16_t>(
                        out, reinterpret_cast<int16_t*>(tensor->clientBuf.data), dims)) {
                    SIMPLE_LOG_ERROR("failure in castToFloat<int16_t>\n");
                    returnStatus = false;
                }
                break;

            case QNN_DATATYPE_INT_32:
                if (!castToFloat_NCHW<int32_t>(
                        out, reinterpret_cast<int32_t*>(tensor->clientBuf.data), dims)) {
                    SIMPLE_LOG_ERROR("failure in castToFloat<int32_t>\n");
                    returnStatus = false;
                }
                break;

            case QNN_DATATYPE_BOOL_8:
                if (!castToFloat_NCHW<uint8_t>(
                        out, reinterpret_cast<uint8_t*>(tensor->clientBuf.data), dims)) {
                    SIMPLE_LOG_ERROR("failure in castToFloat<bool>\n");
                    returnStatus = false;
                }
                break;

            default:
                SIMPLE_LOG_ERROR("Datatype not supported yet\n");
                returnStatus = false;
                break;
        }
        return returnStatus;
    }

    void QnnWrapperV1::freeQnnTensorWrapper(Qnn_TensorWrapper_t& tensor) {
        // free all pointer allocations in struct
        m_allocator_ptr->Free(tensor.name);
        m_allocator_ptr->Free(tensor.tensor.maxDimensions);
        m_allocator_ptr->Free(tensor.tensor.currentDimensions);
    }

    void QnnWrapperV1::freeQnnTensorWrappers(Qnn_TensorWrapper_t*& tensors, uint32_t numTensors) {
        // free all pointer allocations in struct
        for (size_t i = 0; i < numTensors; i++) {
            freeQnnTensorWrapper(tensors[i]);
        }
        m_allocator_ptr->Free(tensors);
    }

    bool QnnWrapperV1::freeGraphsInfo(GraphInfoPtr_t** graphsInfo, uint32_t numGraphs) {
        if (graphsInfo == nullptr || *graphsInfo == nullptr) {
            SIMPLE_LOG_ERROR("freeGraphsInfo() invalid graphsInfo\n");
            return false;
        }
        for (uint32_t i = 0; i < numGraphs; i++) {
            m_allocator_ptr->Free((*graphsInfo)[i]->graphName);
            freeQnnTensorWrappers((*graphsInfo)[i]->inputTensors,
                                  (*graphsInfo)[i]->numInputTensors);
            freeQnnTensorWrappers((*graphsInfo)[i]->outputTensors,
                                  (*graphsInfo)[i]->numOutputTensors);
        }

        m_allocator_ptr->Free(**graphsInfo);
        m_allocator_ptr->Free(*graphsInfo);
        *graphsInfo = nullptr;

        return true;
    }

    void logQnnCallback(const char* fmt, QnnLog_Level_t level, uint64_t timestamp, va_list argp) {
#ifdef __ANDROID__
        static const char* kLogTag = "QnnWrapper";
        const int LEVEL_OFFSET[]   = {0, 4, 3, 2, 1, 0};
        int prio                   = ANDROID_LOG_VERBOSE;
        if (level <= QNN_LOG_LEVEL_VERBOSE) {
            prio += LEVEL_OFFSET[level];
        }
        __android_log_vprint(prio, kLogTag, fmt, argp);
#else
        const char* LEVEL_STR[] = {
            "UNKNOWN", " ERROR ", "WARNING", "  INFO ", "VERBOSE", " DEBUG "};
        const char* levelStr = "UNKNOWN";
        if (level <= QNN_LOG_LEVEL_VERBOSE) {
            levelStr = LEVEL_STR[level];
        }
        fprintf(stdout, "[%-7s] ", levelStr);
        vfprintf(stdout, fmt, argp);
        fprintf(stdout, "\n");
#endif
    }
} // namespace wrap
} // namespace nn
