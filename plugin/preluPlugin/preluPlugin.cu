#include <cassert>
#include "NvInfer.h"
#include "NvInferPlugin.h"
#include "NvCaffeParser.h"
#include "plugin.h" 
#include "common.h"
#include "serialize.hpp"

#include "cuda_fp16.h"
#include "half.h"
#include <iostream>

#include "preluPlugin.h"

using namespace nvinfer1;
using namespace plugin;
using nvinfer1::plugin::PReLUPlugin;
using nvinfer1::plugin::PReLUPluginCreator;

/******** plugin specific constants *********/
static const char* PRELU_PLUGIN_VERSION{"1"};
static const char* PRELU_PLUGIN_NAME{"PRELU_TRT"};

// Static class fields initialization
PluginFieldCollection PReLUPluginCreator::mFC{};
std::vector<PluginField> PReLUPluginCreator::mPluginAttributes;

// REGISTER_TENSORRT_PLUGIN(PReLUPluginCreator);

constexpr size_t maxWorkspaceBytes = 4194304; // 4MB

/******** PReLU CUDA function ********/
// CUDA kernels for PReLU forward
#define CUDA_KERNEL_LOOP(i, n) \
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; \
       i < (n); \
       i += blockDim.x * gridDim.x)

template <typename Ftype>
__global__ void PReLUForward(
    const int n, const int channels, const int dim,
    const Ftype* slope_data,
    const Ftype* input, Ftype* output,
    const Ftype zero,
    const int div_factor)
{
    if (threadIdx.x == 0 && threadIdx.y == 0)
        std::cout << "Unsupported type\n" << std::endl;
}
template<>
__global__ void PReLUForward(
    const int count, const int channels, const int dim,
    const float* slope_data,
    const float* input, float* output,
    const float zero,
    const int div_factor) {
    CUDA_KERNEL_LOOP(index, count) {
        int c = (index / dim) % channels / div_factor;
        output[index] = input[index] > 0 ? input[index] : input[index] * *(reinterpret_cast<const float*>(slope_data)+c);
    }
}
template<>
__global__ void PReLUForward(
    const int count, const int channels, const int dim,
    const half* slope_data,
    const half* input, half* output,
    const half zero,
    const int div_factor) {
    CUDA_KERNEL_LOOP(index, count) {
        int c = (index / dim) % channels / div_factor;
        const half in = input[index];
#if __CUDA_ARCH__ >= 530
        output[index] = (__hgt(in, zero)) ? in : in * *(reinterpret_cast<const float*>(slope_data)+c);
#else
        output[index] = (__half2float(in) > __half2float(zero)) ? in : __float2half(__half2float(in) * __half2float(*(reinterpret_cast<const float*>(slope_data)+c)));
#endif
    }
}

template <typename Ftype>
cudaError_t Forward_gpu(const int count, const int channels, const int dim,
                const Ftype* deviceBuffer,
                const Ftype* bottom_data, Ftype* top_data, 
                const Ftype zero,
                const int div_factor, const cudaStream_t stream) {
    cudaError_t err = cudaGetLastError();

    std::cout << "Not Supported Precision" << std::endl;

    return err;
}
template <>
cudaError_t Forward_gpu(const int count, const int channels, const int dim,
                const float* deviceBuffer,
                const float* bottom_data, float* top_data, 
                const float zero,
                const int div_factor, const cudaStream_t stream) {
    const int threads_per_block = 256;
    int blocks = (count + threads_per_block - 1) / threads_per_block;

    PReLUForward<float><<<blocks, threads_per_block, 0, stream>>>
        (count, channels, dim, deviceBuffer, bottom_data, top_data, zero, div_factor);
    cudaError_t err = cudaGetLastError();

    return err;
}
template <>
cudaError_t Forward_gpu(const int count, const int channels, const int dim,
                const half* deviceBuffer,
                const half* bottom_data, half* top_data, 
                const half zero,
                const int div_factor, const cudaStream_t stream) {
    const int threads_per_block = 512;
    int blocks = (count + threads_per_block - 1) / threads_per_block;
    PReLUForward<half><<<blocks, threads_per_block, 0, stream>>>
        (count, channels, dim, deviceBuffer, bottom_data, top_data, zero, div_factor);
    cudaError_t err = cudaGetLastError();

    return err;
}

/*******************/

using nvinfer1::plugin::PReLUPlugin;
using nvinfer1::plugin::PReLUPluginCreator;

/***** PReLU Plguin *****/
PReLUPlugin::PReLUPlugin(const std::string layerName, const nvinfer1::Weights& weights, const int nbWeights, const bool channelShared)
    : mLayerName(layerName)
{
    assert(nbWeights == 1);
    assert(mWeights.type == DataType::kFLOAT || mWeights.type == DataType::kHALF);
    
    mPReLUParams.mChannelShared = channelShared;
    mWeights = copyToDevice(weights.values, weights.type, weights.count);
}

PReLUPlugin::PReLUPlugin(const std::string layerName, const nvinfer1::Weights& weights, const int nbWeights, const PReLUParameters& params)
    : mLayerName(layerName)
    , mPReLUParams(params)
{
    assert(nbWeights == 1);
    assert(mWeights.type == DataType::kFLOAT || mWeights.type == DataType::kHALF);
    mWeights = copyToDevice(weights.values, weights.type, weights.count);
}

// create the plugin at runtime from a byte stream
PReLUPlugin::PReLUPlugin(const std::string layerName, const void *data, size_t length)
    : mLayerName(layerName)
{
    const char *d = static_cast<const char *>(data), *a = d;
    mPReLUParams = read<PReLUParameters>(d);
    mInputDims.nbDims = read<int>(d);
    for (int i = 0; i < mInputDims.nbDims; i++) {
        mInputDims.d[i] = read<int>(d);
    }
    mOutputDims.nbDims = read<int>(d);
    for (int i = 0; i < mOutputDims.nbDims; i++) {
        mOutputDims.d[i] = read<int>(d);
    }
    mDataType = static_cast<DataType>(read<int>(d));
    size_t count = read<int64_t>(d);
    DataType weights_dtype  = read<DataType>(d);
    mWeights = deserializeToDevice(d, weights_dtype, count);

    // printf("DES>>mC: %d, mH: %d, mW: %d, mP: %d, mQ: %d\n", mPReLUParams.mC, mPReLUParams.mH, mPReLUParams.mW, mPReLUParams.mP, mPReLUParams.mQ);

    assert(d == a + length);
}

PReLUPlugin::~PReLUPlugin()
{
    if (mWeights.values != nullptr) 
    {
        cudaFree(const_cast<void *>(mWeights.values));
        mWeights.values = nullptr;
    }
}

int PReLUPlugin::getNbOutputs() const
{
    return 1;
}

Dims PReLUPlugin::getOutputDimensions(int index, const Dims *inputs, int nbInputDims)
{
    assert(nbInputs == 1);
    assert(outputIndex == 0);
    Dims ret;
    ret.nbDims = 3;
    ret.d[0] = inputs[0].d[0];
    ret.d[1] = inputs[0].d[1];
    ret.d[2] = inputs[0].d[2];
    return ret;
}

bool PReLUPlugin::supportsFormat(DataType type, PluginFormat format) const 
{
    return (type == DataType::kFLOAT || type == DataType::kHALF) \
            && format == PluginFormat::kNCHW; 
}

int PReLUPlugin::initialize()
{
    return 0;
}

DataType PReLUPlugin::getOutputDataType(int index, const nvinfer1::DataType* inputTypes, int nbInputs) const
{
    assert(index == 0);
    assert(inputTypes && nbInputs == 1);
    return mDataType;
}

void PReLUPlugin::terminate()
{
    if (mWeights.values)
    {
        cudaFree(const_cast<void *>(mWeights.values));
        mWeights.values = nullptr;
    }
}

size_t PReLUPlugin::getWorkspaceSize(int maxBatchSize) const
{
    return maxWorkspaceBytes * maxBatchSize;
}

int PReLUPlugin::enqueue(int batchSize, const void *const *inputs, void **outputs, void *workspace, cudaStream_t stream)
{
    // const int count = batchSize * mWeights.count;
    // const int channels = mPReLUParams.mC;
    // const int dim = mPReLUParams.mH * mPReLUParams.mW;
    // const int div_factor = mPReLUParams.mChannelShared ? mPReLUParams.mC : 1; // mChannelShared default is false

    const int channels = mPReLUParams.mC;
    const int dim = mPReLUParams.mH * mPReLUParams.mW;
    const int count = batchSize * channels * dim;
    const int div_factor = mPReLUParams.mChannelShared ? mPReLUParams.mC : 1; // mChannelShared default is false

    if (mDataType == DataType::kFLOAT)
    {
        const float zerof{0.0f};
        Forward_gpu(count, channels, dim,
                                 reinterpret_cast<const float *>(mWeights.values),
                                 reinterpret_cast<const float *>(inputs[0]),
                                 reinterpret_cast<float *>(outputs[0]),
                                 zerof,
                                 div_factor,
                                 stream);
    }
    else if (mDataType == DataType::kHALF)
    {
        const half zeroh = __float2half(0.0f);
        Forward_gpu(count, channels, dim,
                                  reinterpret_cast<const half *>(mWeights.values),
                                  reinterpret_cast<const half *>(inputs[0]),
                                  reinterpret_cast<half *>(outputs[0]),
                                  zeroh,
                                  div_factor,
                                  stream);
    }
    else 
    {
        std::cout << "Unexpected precision request. Not implemented." << std::endl;
        abort();
    }
    CHECK(cudaGetLastError());

    return 0;
}

bool PReLUPlugin::isOutputBroadcastAcrossBatch(int outputIndex, const bool* inputIsBroadcasted, int nbInputs) const
{
    return false;
}

bool PReLUPlugin::canBroadcastInputAcrossBatch(int inputIndex) const
{
    return false;
}

void PReLUPlugin::configurePlugin(const Dims* inputDims, int nbInputs, const Dims* outputDims, int nbOutputs,
    const DataType* inputTypes, const DataType* outputTypes, const bool* inputIsBroadcast,
    const bool* outputIsBroadcast, PluginFormat floatFormat, int maxBatchSize)
{
    assert(inputTypes[0] == outputTypes[0]);

    nvinfer1::DataType type = inputTypes[0];
    assert((type == DataType::kFLOAT || type == DataType::kHALF) && format == PluginFormat::kNCHW);

    mDataType = type;
    mInputDims = inputDims[0];
    mOutputDims = outputDims[0];
    mPReLUParams.mC = mInputDims.d[0];
    mPReLUParams.mH = mInputDims.d[1];
    mPReLUParams.mW = mInputDims.d[2];
    mPReLUParams.mP = mOutputDims.d[1];
    mPReLUParams.mQ = mOutputDims.d[2];

    // printf("mC: %d, mH: %d, mW: %d, mP: %d, mQ: %d\n", mPReLUParams.mC, mPReLUParams.mH, mPReLUParams.mW, mPReLUParams.mP, mPReLUParams.mQ);
}

size_t PReLUPlugin::getSerializationSize() const
{
    size_t serializationSize = 0;
    serializationSize += sizeof(mPReLUParams);
    serializationSize += sizeof(mInputDims.nbDims);
    serializationSize += sizeof(mInputDims.d[0]) * mInputDims.nbDims;
    serializationSize += sizeof(mOutputDims.nbDims);
    serializationSize += sizeof(mOutputDims.d[0]) * mOutputDims.nbDims;
    serializationSize += sizeof(static_cast<int>(mDataType));
    serializationSize += sizeof(mWeights.count);
    serializationSize += sizeof(DataType);
    serializationSize += type2size(mWeights.type) * mWeights.count;
    return serializationSize;
}

void PReLUPlugin::serialize(void *buffer) const
{
    char *d = static_cast<char*>(buffer), *a = d;

    write<PReLUParameters>(d, mPReLUParams);
    write(d, mInputDims.nbDims);
    assert(mInputDims.nbDims <= mInputDims.MAX_DIMS);
    for (int i = 0; i < mInputDims.nbDims; i++) 
    {
        write(d, mInputDims.d[i]);
    }
    write(d, mOutputDims.nbDims);
    assert(mOutputDims.nbDims <= mOutputDims.MAX_DIMS);
    for (int i = 0; i < mOutputDims.nbDims; i++)
    {
        write(d, mOutputDims.d[i]);
    }
    write(d, static_cast<int>(mDataType));
    write<int64_t>(d, mWeights.count);
    write<DataType>(d, mWeights.type);
    serializeFromDevice(d, mWeights);
    assert(d == a + getSerializationSize());
}

const char *PReLUPlugin::getPluginType() const
{
    return PRELU_PLUGIN_NAME;
}

const char *PReLUPlugin::getPluginVersion() const
{
    return PRELU_PLUGIN_VERSION;
}

void PReLUPlugin::destroy()
{ 
    delete this; 
}

IPluginV2Ext *PReLUPlugin::clone() const
{
    IPluginV2Ext* plugin = new PReLUPlugin(mLayerName, mWeights, 1, mPReLUParams);
    plugin->setPluginNamespace(mPluginNamespace.c_str());
    return plugin;
}

void PReLUPlugin::setPluginNamespace(const char* libNamespace)
{
    mPluginNamespace = libNamespace;
}

const char* PReLUPlugin::getPluginNamespace() const
{
    return mPluginNamespace.c_str();
}

Weights PReLUPlugin::copyToDevice(const void *hostBuffer, const DataType dtype, const size_t count)
{
    void* deviceData;
    CUASSERT(cudaMalloc(&deviceData, count * type2size(dtype)));
    CUASSERT(cudaMemcpy(deviceData, hostBuffer, count * type2size(dtype), cudaMemcpyHostToDevice));
    return Weights{dtype, deviceData, int64_t(count)};
}

void PReLUPlugin::serializeFromDevice(char*& hostBuffer, const nvinfer1::Weights& deviceWeights) const
{
    CUASSERT(cudaMemcpy(hostBuffer, deviceWeights.values, deviceWeights.count * type2size(deviceWeights.type), cudaMemcpyDeviceToHost));
    hostBuffer += deviceWeights.count * type2size(deviceWeights.type);
}

Weights PReLUPlugin::deserializeToDevice(const char *&hostBuffer, const DataType dtype, const size_t count)
{
    Weights w = copyToDevice(hostBuffer, dtype, count);
    hostBuffer += count * sizeof(float);
    return w;
}

/***** Helpers *****/
template <typename T>
void PReLUPlugin::write(char*& buffer, const T& val) const
{
    *reinterpret_cast<T *>(buffer) = val;
    buffer += sizeof(T);
}

template <typename T>
T PReLUPlugin::read(const char*& buffer) const
{
    T val = *reinterpret_cast<const T *>(buffer);
    buffer += sizeof(T);
    return val;
}

size_t PReLUPlugin::type2size(DataType type) const
{
    // return type == DataType::kFLOAT ? sizeof(float) : (type == DataType::kHALF) ? sizeof(__half) : sizeof(int8_t); 
    return type == DataType::kFLOAT ? sizeof(float) : sizeof(half);
}

/*
 */
PReLUPluginCreator::PReLUPluginCreator()
{
    mPluginAttributes.emplace_back(PluginField("weights", nullptr, PluginFieldType::kFLOAT32, 1));
    mPluginAttributes.emplace_back(PluginField("channelShared", nullptr, PluginFieldType::kINT32, 1));
    mPluginAttributes.emplace_back(PluginField("nbWeights", nullptr, PluginFieldType::kINT32, 1));

    mFC.nbFields = mPluginAttributes.size();
    mFC.fields   = mPluginAttributes.data();
}

const char* PReLUPluginCreator::getPluginName() const
{
    return PRELU_PLUGIN_NAME;
}

const char* PReLUPluginCreator::getPluginVersion() const
{
    return PRELU_PLUGIN_VERSION;
}

const PluginFieldCollection* PReLUPluginCreator::getFieldNames()
{
    return &mFC;
}

IPluginV2* PReLUPluginCreator::createPlugin(const char* name, const PluginFieldCollection* fc)
{
    std::vector<float> weightValues;
    const PluginField* fields = fc->fields;
    for (int i = 0; i < fc->nbFields; i++)
    {
        const char* field_name = fc->fields[i].name;
        if (!strcmp(field_name, "nbWeights"))
        {
            ASSERT(fields[i].type == PluginFieldType::kINT32);
            mNbWeights = *(static_cast<const int*>(fields[i].data));
        }
        else if (!strcmp(field_name, "channelShared"))
        {
            ASSERT(fields[i].type == PluginFieldType::kINT32);
            mChannelShared = *(static_cast<const bool*>(fields[i].data));
        }
        else if (!strcmp(field_name, "weights"))
        {
            ASSERT(fields[i].type == PluginFieldType::kFLOAT32);
            int size = fields[i].length;
            weightValues.reserve(size);
            const auto* w = static_cast<const float*>(fields[i].data);
            for (int j = 0; j < size; j++)
            {
                weightValues.push_back(*w);
                w++;
            }
        }
    }
    Weights weights{DataType::kFLOAT, weightValues.data(), (int64_t) weightValues.size()};
    PReLUPlugin* obj = new PReLUPlugin(name, weights, mNbWeights, mChannelShared);
    obj->setPluginNamespace(mNamespace.c_str());
    return obj;
}

IPluginV2* PReLUPluginCreator::deserializePlugin(const char* name, const void* serialData, size_t serialLength)
{
    // This object will be deleted when the network is destroyed, which will
    PReLUPlugin* obj = new PReLUPlugin(name, serialData, serialLength);
    obj->setPluginNamespace(mNamespace.c_str());
    return obj;
}

void PReLUPluginCreator::setPluginNamespace(const char* libNamespace)
{
    mNamespace = libNamespace;
}

const char* PReLUPluginCreator::getPluginNamespace() const
{
    return mNamespace.c_str();
}
