#ifndef _PLUGIN_PRELU_H_
#define _PLUGIN_PRELU_H_

#include <iostream>
#include <map>
#include <cuda_runtime_api.h>

#include "NvInfer.h"
#include "NvInferPlugin.h"
#include "NvCaffeParser.h"

#include "plugin.h"

using namespace nvinfer1::plugin;
namespace nvinfer1
{
namespace plugin
{

struct PReLUParameters
{
    // Input dimensions
    int mC, mH, mW;
    // Output dimensions
    int mP, mQ;
    // Channel Shared
    bool mChannelShared;
};

// class PReLUPlugin : public IPluginV2IOExt
class PReLUPlugin : public IPluginV2Ext
{
public:

    PReLUPlugin(const std::string name, const nvinfer1::Weights& weights, const int nbWeights, const bool channelShared);
    PReLUPlugin(const std::string name, const nvinfer1::Weights& weights, const int nbWeights, const PReLUParameters& params);
    PReLUPlugin(const std::string name, const void* data, size_t length);

    PReLUPlugin() = delete;

    ~PReLUPlugin() override;

public:
    nvinfer1::IPluginV2Ext* clone() const override;
    nvinfer1::Dims getOutputDimensions(int outputIndex, const Dims* inputs, int nbInputs) override;
    // nvinfer1::DimsExprs getOutputDimensions(
    //     int outputIndex, const nvinfer1::DimsExprs* inputs, int nbInputs, nvinfer1::IExprBuilder& exprBuilder) override;
    // bool supportsFormatCombination(
    //     int pos, const PluginTensorDesc* inOut, int nbInputs, int nbOutputs) const override;
    // void configurePlugin(const PluginTensorDesc* in, int nbInput, 
    //     const PluginTensorDesc* out, int nbOutput) override;
    // size_t getWorkspaceSize(const nvinfer1::PluginTensorDesc* inputs, int nbInputs,
    //     const nvinfer1::PluginTensorDesc* outputs, int nbOutputs) const override;
    size_t getWorkspaceSize(int maxBatchSize) const override;
    // int enqueue(const nvinfer1::PluginTensorDesc* inputDesc, const nvinfer1::PluginTensorDesc* outputDesc,
    //     const void* const* inputs, void* const* outputs, void* workspace, cudaStream_t stream) override;
    int enqueue(int batchSize, const void*const * inputs, void** outputs, void* workspace, cudaStream_t stream) override;

    

    // void configureWithFormat(const Dims* inputDims, int nbInputs, const Dims* outputDims, int nbOutputs, DataType type, PluginFormat format, int maxBatchSize) override;

    bool supportsFormat(DataType type, PluginFormat format) const override;
    void configurePlugin(const Dims* inputDims, int nbInputs, const Dims* outputDims, int nbOutputs,
        const DataType* inputTypes, const DataType* outputTypes, const bool* inputIsBroadcast,
        const bool* outputIsBroadcast, PluginFormat floatFormat, int maxBatchSize) override;
    //! The combination of kLINEAR + kINT8/kHALF/kFLOAT is supported.

    nvinfer1::DataType getOutputDataType(int index, const nvinfer1::DataType* inputTypes, int nbInputs) const override;

    const char* getPluginType() const override;
    const char* getPluginVersion() const override;
    int getNbOutputs() const override;
    int initialize() override;
    void terminate() override;
    size_t getSerializationSize() const override;
    void serialize(void* buffer) const override;
    void destroy();
    void setPluginNamespace(const char* libNamespace) override;
    const char* getPluginNamespace() const override;
    



    bool isOutputBroadcastAcrossBatch(int outputIndex, const bool* inputIsBroadcasted, int nbInputs) const override;

    bool canBroadcastInputAcrossBatch(int inputIndex) const override;
    
private:
    // todo: need to replace with other
    size_t type2size(const DataType type) const;

    // todo: need to replace with other
    template <typename T>
    void write(char*& buffer, const T& val) const;

    // todo: need to replace with other
    template <typename T>
    T read(const char *&buffer) const;

    // todo: need to replace with common functions
    Weights copyToDevice(const void *data, const DataType dtype, size_t count);
    // void convertAndCopyToDevice(void *&deviceWeights, const Weights &weights);
    void serializeFromDevice(char*& hostBuffer, const nvinfer1::Weights& deviceWeights) const;

    Weights deserializeToDevice(const char *&hostBuffer, const DataType dtype, const size_t count);
    
    // size_t mNbInputChannels;
    // size_t mNbInputHeight;
    // size_t mNbInputWidth;
    // size_t mNbInputCount;

    std::string mPluginNamespace;
    const std::string mLayerName;
    DataType mDataType{DataType::kFLOAT};

    PReLUParameters mPReLUParams;
    Dims mInputDims{};
    Dims mOutputDims{};
    // bool mChannelShared{};
    nvinfer1::Weights mWeights{};

    // void* mDeviceKernel{nullptr};

    void dump(const char* filename, void* memblock, size_t size);
};

class PReLUPluginCreator : public nvinfer1::plugin::BaseCreator
{
public:
    PReLUPluginCreator();

    ~PReLUPluginCreator() override = default;

    const char* getPluginName() const override;

    const char* getPluginVersion() const override;
    
    const PluginFieldCollection* getFieldNames() override;
    
    IPluginV2* createPlugin(const char* name, const PluginFieldCollection* fc) override;
    
    IPluginV2* deserializePlugin(const char* name, const void* serialData, size_t serialLength) override;

    void setPluginNamespace(const char* libNamespace) override;

    const char* getPluginNamespace() const override;

private:
    static PluginFieldCollection mFC;
    static std::vector<PluginField> mPluginAttributes;

    bool mChannelShared{};
    int mNbWeights{};

    // Parameters for PReLUPlugin
    PReLUParameters params;
};
} // namespace plugin
} // namespace nvinfer1

#endif // _PLUGIN_PRELU_H_