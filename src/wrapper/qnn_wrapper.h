#ifndef SIMPLE_NN_QNN_WARPPER_H_
#define SIMPLE_NN_QNN_WARPPER_H_

#include "QnnInterface.h"
#include "System/QnnSystemInterface.h"
#include "wrapper/wrapper.h"

#include <string.h>

#if (defined __clang__) || (defined __GNUC__)
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wfloat-equal"
#endif
// #include "SampleAppContextCaching_generated.h"
#if (defined __clang__) || (defined __GNUC__)
#pragma GCC diagnostic pop
#endif

namespace nn {
namespace wrap {
    class QnnWrapperV1 {
    public:
        class Allocator {
        public:
            virtual void* Malloc(size_t size) { return malloc(size); }
            virtual void Free(void* p) { free(p); }
            virtual void Memset(void* p, int c, size_t size) { memset(p, c, size); }
            virtual ~Allocator() {}
        };

        enum class DataType { FLOAT, NATIVE, INVALID };
        enum class DataLayout { LAYOUT_NCHW, LAYOUT_NHWC };
        QnnWrapperV1() {
            if (m_allocator_ptr == nullptr) {
                m_allocator_ptr = std::make_shared<Allocator>();
            }
        }

        void setAllocator(std::shared_ptr<Allocator> allocator) { m_allocator_ptr = allocator; }

        bool initForDevice(const std::string& backendPath,
                           const std::string& systemLibraryPath,
                           bool is_use_signed_pd,
                           bool is_use_vndk);
        virtual ~QnnWrapperV1();

        bool createGraphsFromBinary(const uint8_t* buffer, const size_t bufferSize);
        void freeGraphsAndContext();

        bool setupInputAndOutputTensors();
        bool tearDownInputAndOutputTensors();

        std::vector<std::string> getInputNames();
        std::vector<std::string> getOutputNames();

        std::vector<std::vector<size_t>> getInputDims();
        std::vector<std::vector<size_t>> getOutputDims();

        bool populateInputTensors(std::vector<uint8_t*> inputBuffers,
                                  std::vector<size_t> inputBuffersLen,
                                  DataType dataType = DataType::FLOAT,
                                  DataLayout layout = DataLayout::LAYOUT_NHWC);

        bool executeGraphs();

        bool populateOutputBuffer(std::vector<uint8_t*>& outputBuffers,
                                  std::vector<size_t> outBuffersLen,
                                  DataType dataType = DataType::FLOAT);

    private:
        typedef struct QnnTensorWrapper {
            char* name;
            Qnn_Tensor_t tensor;
        } Qnn_TensorWrapper_t;
        typedef struct GraphInfo {
            Qnn_GraphHandle_t graph;
            char* graphName;
            Qnn_TensorWrapper_t* inputTensors;
            uint32_t numInputTensors;
            Qnn_TensorWrapper_t* outputTensors;
            uint32_t numOutputTensors;
        } GraphInfo_t;
        typedef GraphInfo_t* GraphInfoPtr_t;

        typedef struct QnnFunctionPointers {
            QNN_INTERFACE_VER_TYPE qnnInterface;
            QNN_SYSTEM_INTERFACE_VER_TYPE qnnSystemInterface;
        } QnnFunctionPointers;

        typedef Qnn_ErrorHandle_t (*QnnInterfaceGetProvidersFn_t)(
            const QnnInterface_t** providerList,
            uint32_t* numProviders);

        typedef Qnn_ErrorHandle_t (*QnnSystemInterfaceGetProvidersFn_t)(
            const QnnSystemInterface_t** providerList,
            uint32_t* numProviders);

        using GraphTensorIdToNameMap =
            std::unordered_map<std::string, std::unordered_map<uint32_t, std::string>>;

        QnnFunctionPointers m_qnnFunctionPointers;
        void* m_libBackendHandle              = nullptr;
        void* m_libSystemHandle               = nullptr;
        QnnBackend_Config_t** m_BackendConfig = nullptr;

        GraphInfo_t** m_GraphsInfo                 = nullptr;
        uint32_t m_GraphsCount                     = 0;
        Qnn_ContextHandle_t m_context              = nullptr;
        Qnn_ProfileHandle_t m_profileBackendHandle = nullptr;
        Qnn_Tensor_t* m_input                      = nullptr;
        Qnn_Tensor_t* m_output                     = nullptr;
        std::shared_ptr<Allocator> m_allocator_ptr;

        // load library
        bool getQnnBackendFunctionPointers(std::string backendPath);

        bool getQnnSystemFunctionPointers(std::string systemLibraryPath);

        // createGraphsFromBinary
        bool deserializeData(const uint8_t* buffer,
                             const size_t buffer_size,
                             GraphTensorIdToNameMap& graphTensorIdToNamesMap,
                             uint32_t* graphsCount,
                             uint8_t*& binaryCache,
                             uint64_t& binaryCacheSize);

        // bool extractTensorsInfo(
        //     const
        //     flatbuffers::Vector<flatbuffers::Offset<qnn::tools::sample_app::QnnTensorInfo>>*
        //         fbTensorInfosVector,
        //     std::string graphName,
        //     std::unordered_map<std::string, std::unordered_map<uint32_t, std::string>>&
        //         graphTensorIdToNamesMap,
        //     uint32_t tensorsCount);

        // bool extractGraphsInfo(
        //     const qnn::tools::sample_app::ContextCache* contextCache,
        //     std::unordered_map<std::string, std::unordered_map<uint32_t, std::string>>&
        //         graphTensorIdToNamesMap,
        //     uint32_t* graphsCount);

        bool copyTensorsInfo(const Qnn_Tensor_t* tensorsInfoSrc,
                             Qnn_TensorWrapper_t*& tensorWrappers,
                             uint32_t tensorsCount);

        bool copyGraphsInfoV1(const QnnSystemContext_GraphInfoV1_t* graphInfoSrc,
                              GraphInfo_t* graphInfoDst);

        bool copyGraphsInfo(const QnnSystemContext_GraphInfo_t* graphsInput,
                            const uint32_t numGraphs,
                            GraphInfo_t**& graphsInfo);

        bool copyMetadataToGraphsInfo(const QnnSystemContext_BinaryInfo_t* binaryInfo,
                                      GraphInfo_t**& graphsInfo,
                                      uint32_t& graphsCount);

        bool populateTensorNamesFromMetadata(GraphTensorIdToNameMap& graphTensorIdToNamesMap,
                                             GraphInfo_t**& graphsInfo,
                                             const uint32_t graphsCount);

        // setupInputAndOutputTensors
        bool setupTensors(Qnn_Tensor_t** tensors,
                          uint32_t tensorCount,
                          Qnn_TensorWrapper_t* tensorWrappers);

        bool fillDims(std::vector<size_t>& dims, uint32_t* inDimensions, uint32_t rank);

        bool deepCopyQnnTensorInfo(Qnn_Tensor_t* dest, Qnn_Tensor_t* src);

        template <typename T>
        bool allocateBuffer(T** buffer, size_t& elementCount);
        bool allocateBuffer(uint8_t** buffer, std::vector<size_t> dims, Qnn_DataType_t dataType);

        std::tuple<bool, size_t> getDataTypeSizeInBytes(Qnn_DataType_t dataType);

        size_t calculateElementCount(std::vector<size_t> dims);

        std::tuple<bool, size_t> calculateLength(std::vector<size_t> dims, Qnn_DataType_t dataType);

        void tearDownTensors(Qnn_Tensor_t* tensors, uint32_t tensorCount);

        // populateInputTensors
        bool populateInputTensor(uint8_t* buffer,
                                 size_t bufferLen,
                                 Qnn_Tensor_t* input,
                                 DataType inputDataType,
                                 DataLayout layout);

        bool copyFromFloatToNative(float* floatBuffer, Qnn_Tensor_t* tensor);

        bool copyFromFloatToNative_NHWC(float* floatBuffer, Qnn_Tensor_t* tensor);
        bool convertToFloat_NCHW(float* out, Qnn_Tensor_t* tensor);

        bool freeGraphsInfo(GraphInfoPtr_t** graphsInfo, uint32_t numGraphs);
        void freeQnnTensorWrapper(Qnn_TensorWrapper_t& tensor);
        void freeQnnTensorWrappers(Qnn_TensorWrapper_t*& tensors, uint32_t numTensors);
    };

    class QnnWrapper : public QnnWrapperV1 {};
} // namespace wrap
} // namespace nn

#endif // SIMPLE_NN_QNN_WARPPER_H_