#include <zeno/TinyCudaNNRuntime.h>
#include <tiny-cuda-nn/cpp_api.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <set>
#include <type_traits>
#include <utility>
#if defined(ZENO_WITH_NANOVDB_CUDA_BUILDERS)
#include <cub/cub.cuh>
#include <nanovdb/NanoVDB.h>
#include <nanovdb/GridHandle.h>
#include <nanovdb/cuda/Buffer.h>
#include <nanovdb/cuda/GridHandle.cuh>
#include <nanovdb/tools/CreateNanoGrid.h>
#include <nanovdb/cuda/HandleStorage.h>
#include <nanovdb/tools/cuda/IndexToGrid.cuh>
#include <nanovdb/tools/cuda/GridChecksum.cuh>
#endif
namespace zeno::neuralvolume {
namespace {
void checkCuda(cudaError_t status, const char* operation) {
    if (status != cudaSuccess) throw std::runtime_error(std::string(operation) + ": " + cudaGetErrorString(status));
}
template <typename T> class DeviceBuffer {
public:
    explicit DeviceBuffer(size_t count) {
        if (!count || count > std::numeric_limits<size_t>::max() / sizeof(T)) throw std::invalid_argument("invalid CUDA buffer size");
        checkCuda(cudaMalloc(reinterpret_cast<void**>(&ptr), count * sizeof(T)), "cudaMalloc");
    }
    ~DeviceBuffer() { if (ptr) cudaFree(ptr); }
    DeviceBuffer(const DeviceBuffer&) = delete;
    DeviceBuffer& operator=(const DeviceBuffer&) = delete;
    T* get() const { return ptr; }
private: T* ptr = nullptr;
};
__global__ void floatToHalf(const float* src, __half* dst, size_t n) {
    size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) dst[i] = __float2half_rn(src[i]);
}
__global__ void compactFloat(const float* src, float* dst, uint32_t n) {
    uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) dst[i] = src[i]; // first output dimension in column-major [dims, samples]
}
__global__ void compactHalf(const __half* src, float* dst, uint32_t n) {
    uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) dst[i] = __half2float(src[i]);
}
__global__ void assembleDecodedLeaf(
    const float* logits, const float* normalizedValues, bool hasRegressor,
    float valueCenter, float valueScale, float background,
    const uint16_t* occupancyOffsets, uint32_t occupancyCount,
    const uint16_t* exactOffsets, const float* exactValues, uint32_t exactCount,
    uint8_t* occupancyOut, float* valuesOut, int* validationError) {
    const uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= 512) return;
    uint8_t active = logits[i] >= 0.0f ? 1u : 0u;
    for (uint32_t c = 0; c < occupancyCount; ++c) {
        if (occupancyOffsets[c] == i) active = static_cast<uint8_t>(1u - active);
    }
    float value = (hasRegressor && active) ? normalizedValues[i] * valueScale + valueCenter : background;
    bool exactFound = false;
    for (uint32_t c = 0; c < exactCount; ++c) {
        if (exactOffsets[c] == i) { value = exactValues[c]; exactFound = true; }
    }
    if (active && !hasRegressor && !exactFound) atomicExch(validationError, 1);
    if (active && !isfinite(value)) atomicExch(validationError, 2);
    occupancyOut[i] = active;
    valuesOut[i] = value;
}
} // namespace

struct TinyCudaNNRuntime::Impl {
    std::unique_ptr<tcnn::cpp::Module> module;
    cudaStream_t stream = nullptr;
    void* params = nullptr;
    void* inputScratch = nullptr;
    void* outputScratch = nullptr;
    void* compactScratch = nullptr;
    size_t inputCapacity = 0;
    size_t outputCapacityBytes = 0;
    size_t compactCapacity = 0;
    uint32_t outputWidth = 0;
    bool fp16 = false;
    bool outputFp16 = false;

    void ensureBuffer(void** buffer, size_t& capacity, size_t required, const char* label) {
        if (required <= capacity) return;
        void* replacement = nullptr;
        checkCuda(cudaMalloc(&replacement, required), label);
        if (buffer && *buffer) cudaFree(*buffer);
        *buffer = replacement;
        capacity = required;
    }
    ~Impl() {
        if (params) cudaFree(params);
        if (inputScratch) cudaFree(inputScratch);
        if (outputScratch) cudaFree(outputScratch);
        if (compactScratch) cudaFree(compactScratch);
        if (stream) cudaStreamDestroy(stream);
    }
};
TinyCudaNNRuntime::TinyCudaNNRuntime(const std::string& jsonConfig, const std::vector<float>& weights)
    : m_impl(std::make_unique<Impl>()) {
    const auto config = tcnn::cpp::json::parse(jsonConfig);
    if (config.at("backend").get<std::string>() != "tiny-cuda-nn") throw std::runtime_error("unsupported ZNVM neural backend");
    const std::string storedPrecision = config.at("param_precision").get<std::string>();
    m_impl->module.reset(tcnn::cpp::create_network_with_input_encoding(3, 1, config.at("encoding"), config.at("network")));
    if (!m_impl->module) throw std::runtime_error("tiny-cuda-nn network creation failed");
    m_impl->fp16 = m_impl->module->param_precision() == tcnn::cpp::Precision::Fp16;
    m_impl->outputFp16 = m_impl->module->output_precision() == tcnn::cpp::Precision::Fp16;
    if (storedPrecision != "float16" && storedPrecision != "float32") throw std::runtime_error("unsupported ZNVM parameter precision");
    if ((storedPrecision == "float16") != m_impl->fp16) throw std::runtime_error("ZNVM precision differs from tiny-cuda-nn build precision");
    if (m_impl->module->n_params() != weights.size()) throw std::runtime_error("ZNVM parameter count differs from recreated network");
    for (float v : weights) if (!std::isfinite(v)) throw std::runtime_error("non-finite ZNVM model parameter");
    m_impl->outputWidth = m_impl->module->n_output_dims();
    if (!m_impl->outputWidth) throw std::runtime_error("invalid tiny-cuda-nn output width");
    checkCuda(cudaStreamCreateWithFlags(&m_impl->stream, cudaStreamNonBlocking), "cudaStreamCreate");
    const size_t n = weights.size();
    if (m_impl->fp16) {
        DeviceBuffer<float> staging(n);
        checkCuda(cudaMemcpyAsync(staging.get(), weights.data(), n * sizeof(float), cudaMemcpyHostToDevice, m_impl->stream), "upload parameter staging");
        checkCuda(cudaMalloc(&m_impl->params, n * sizeof(__half)), "allocate FP16 parameters");
        floatToHalf<<<static_cast<unsigned>((n + 255) / 256), 256, 0, m_impl->stream>>>(staging.get(), static_cast<__half*>(m_impl->params), n);
        checkCuda(cudaGetLastError(), "convert parameters to FP16");
        checkCuda(cudaStreamSynchronize(m_impl->stream), "synchronize FP16 parameter conversion");
    } else {
        checkCuda(cudaMalloc(&m_impl->params, n * sizeof(float)), "allocate FP32 parameters");
        checkCuda(cudaMemcpyAsync(m_impl->params, weights.data(), n * sizeof(float), cudaMemcpyHostToDevice, m_impl->stream), "upload FP32 parameters");
    }
    checkCuda(cudaStreamSynchronize(m_impl->stream), "synchronize parameter upload");
    m_impl->module->set_jit_fusion(tcnn::cpp::supports_jit_fusion());
}
TinyCudaNNRuntime::~TinyCudaNNRuntime() = default;
TinyCudaNNRuntime::TinyCudaNNRuntime(TinyCudaNNRuntime&&) noexcept = default;
TinyCudaNNRuntime& TinyCudaNNRuntime::operator=(TinyCudaNNRuntime&&) noexcept = default;
std::vector<float> TinyCudaNNRuntime::infer(const std::vector<std::array<float, 3>>& points) {
    if (points.empty()) return {};
    const uint32_t granularity = std::max<uint32_t>(1, tcnn::cpp::batch_size_granularity());
    constexpr uint32_t maxBatch = 65536;
    std::vector<float> result; result.reserve(points.size());
    for (size_t offset = 0; offset < points.size();) {
        uint32_t count = static_cast<uint32_t>(std::min<size_t>(maxBatch, points.size() - offset));
        uint32_t batch = ((count + granularity - 1) / granularity) * granularity;
        if (batch > maxBatch) batch = maxBatch;
        std::vector<float> hostInput(static_cast<size_t>(batch) * 3);
        for (uint32_t i = 0; i < batch; ++i) {
            const auto& p = points[offset + std::min(i, count - 1)];
            for (unsigned axis = 0; axis < 3; ++axis) {
                if (!std::isfinite(p[axis])) throw std::invalid_argument("non-finite inference coordinate");
                hostInput[static_cast<size_t>(axis) * batch + i] = p[axis];
            }
        }
        const size_t inputBytes = hostInput.size() * sizeof(float);
        const size_t outputElements = static_cast<size_t>(batch) * m_impl->outputWidth;
        const size_t outputBytes = outputElements * (m_impl->outputFp16 ? sizeof(__half) : sizeof(float));
        const size_t compactBytes = static_cast<size_t>(batch) * sizeof(float);
        m_impl->ensureBuffer(&m_impl->inputScratch, m_impl->inputCapacity, inputBytes, "allocate inference input scratch");
        m_impl->ensureBuffer(&m_impl->outputScratch, m_impl->outputCapacityBytes, outputBytes, "allocate inference output scratch");
        m_impl->ensureBuffer(&m_impl->compactScratch, m_impl->compactCapacity, compactBytes, "allocate compact output scratch");
        checkCuda(cudaMemcpyAsync(m_impl->inputScratch, hostInput.data(), inputBytes, cudaMemcpyHostToDevice, m_impl->stream), "upload inference coordinates");
        const uint32_t blocks = (batch + 255) / 256;
        std::vector<float> hostOutput(batch);
        if (m_impl->outputFp16) {
            m_impl->module->inference(m_impl->stream, batch, static_cast<const float*>(m_impl->inputScratch), m_impl->outputScratch, m_impl->params);
            compactHalf<<<blocks, 256, 0, m_impl->stream>>>(static_cast<const __half*>(m_impl->outputScratch), static_cast<float*>(m_impl->compactScratch), batch);
        } else {
            m_impl->module->inference(m_impl->stream, batch, static_cast<const float*>(m_impl->inputScratch), m_impl->outputScratch, m_impl->params);
            compactFloat<<<blocks, 256, 0, m_impl->stream>>>(static_cast<const float*>(m_impl->outputScratch), static_cast<float*>(m_impl->compactScratch), batch);
        }
        checkCuda(cudaGetLastError(), "compact inference output");
        checkCuda(cudaMemcpyAsync(hostOutput.data(), m_impl->compactScratch, compactBytes, cudaMemcpyDeviceToHost, m_impl->stream), "download inference output");
        checkCuda(cudaStreamSynchronize(m_impl->stream), "synchronize inference");
        result.insert(result.end(), hostOutput.begin(), hostOutput.begin() + count);
        offset += count;
    }
    return result;
}
} // namespace zeno::neuralvolume

#include <map>
#include <set>
#include <sstream>

namespace zeno::neuralvolume {

void TinyCudaNNRuntime::inferDevice(const std::vector<std::array<float, 3>>& points, void* deviceOutput) {
    if (!m_impl || !m_impl->module) throw std::logic_error("tiny-cuda-nn runtime is not initialized");
    if (points.empty()) return;
    if (!deviceOutput) throw std::invalid_argument("device inference output pointer is null");
    const uint32_t granularity = std::max<uint32_t>(1, tcnn::cpp::batch_size_granularity());
    constexpr uint32_t maxBatch = 65536;
    if (points.size() > maxBatch || granularity > maxBatch) {
        throw std::invalid_argument("device inference batch exceeds tiny-cuda-nn runtime limit");
    }
    const uint32_t count = static_cast<uint32_t>(points.size());
    const uint32_t batch = ((count + granularity - 1) / granularity) * granularity;
    std::vector<float> hostInput(static_cast<size_t>(batch) * 3);
    for (uint32_t i = 0; i < batch; ++i) {
        const auto& point = points[std::min(i, count - 1)];
        for (unsigned axis = 0; axis < 3; ++axis) {
            if (!std::isfinite(point[axis])) throw std::invalid_argument("non-finite inference coordinate");
            hostInput[static_cast<size_t>(axis) * batch + i] = point[axis];
        }
    }
    const size_t inputBytes = hostInput.size() * sizeof(float);
    const size_t outputElements = static_cast<size_t>(batch) * m_impl->outputWidth;
    const size_t outputBytes = outputElements * (m_impl->outputFp16 ? sizeof(__half) : sizeof(float));
    const size_t compactBytes = static_cast<size_t>(batch) * sizeof(float);
    m_impl->ensureBuffer(&m_impl->inputScratch, m_impl->inputCapacity, inputBytes, "allocate device inference input scratch");
    m_impl->ensureBuffer(&m_impl->outputScratch, m_impl->outputCapacityBytes, outputBytes, "allocate device inference output scratch");
    m_impl->ensureBuffer(&m_impl->compactScratch, m_impl->compactCapacity, compactBytes, "allocate device inference compact scratch");
    checkCuda(cudaMemcpyAsync(m_impl->inputScratch, hostInput.data(), inputBytes, cudaMemcpyHostToDevice, m_impl->stream), "upload device inference coordinates");
    m_impl->module->inference(m_impl->stream, batch, static_cast<const float*>(m_impl->inputScratch), m_impl->outputScratch, m_impl->params);
    const uint32_t blocks = (batch + 255) / 256;
    if (m_impl->outputFp16) {
        compactHalf<<<blocks, 256, 0, m_impl->stream>>>(static_cast<const __half*>(m_impl->outputScratch), static_cast<float*>(m_impl->compactScratch), batch);
    } else {
        compactFloat<<<blocks, 256, 0, m_impl->stream>>>(static_cast<const float*>(m_impl->outputScratch), static_cast<float*>(m_impl->compactScratch), batch);
    }
    checkCuda(cudaGetLastError(), "compact device inference output");
    checkCuda(cudaMemcpyAsync(deviceOutput, m_impl->compactScratch, static_cast<size_t>(count) * sizeof(float), cudaMemcpyDeviceToDevice, m_impl->stream), "copy device inference result");
    checkCuda(cudaStreamSynchronize(m_impl->stream), "synchronize device inference");
}

struct DecodedNeuralLeafDevice::Impl {
    std::array<int32_t, 3> leafOrigin{};
    cudaStream_t stream = nullptr;
    uint8_t* occupancy = nullptr;
    float* values = nullptr;
    int* validationError = nullptr;
    float* logits = nullptr;
    float* regressed = nullptr;
    uint16_t* occupancyOffsets = nullptr;
    uint16_t* exactOffsets = nullptr;
    float* exactValues = nullptr;
    ~Impl() {
        if (stream) cudaStreamSynchronize(stream);
        if (occupancy) cudaFree(occupancy);
        if (values) cudaFree(values);
        if (validationError) cudaFree(validationError);
        if (logits) cudaFree(logits);
        if (regressed) cudaFree(regressed);
        if (occupancyOffsets) cudaFree(occupancyOffsets);
        if (exactOffsets) cudaFree(exactOffsets);
        if (exactValues) cudaFree(exactValues);
        if (stream) cudaStreamDestroy(stream);    }
};
DecodedNeuralLeafDevice::DecodedNeuralLeafDevice() = default;
DecodedNeuralLeafDevice::~DecodedNeuralLeafDevice() = default;
DecodedNeuralLeafDevice::DecodedNeuralLeafDevice(DecodedNeuralLeafDevice&&) noexcept = default;
DecodedNeuralLeafDevice& DecodedNeuralLeafDevice::operator=(DecodedNeuralLeafDevice&&) noexcept = default;
const std::array<int32_t, 3>& DecodedNeuralLeafDevice::origin() const {
    if (!m_impl) throw std::logic_error("device leaf is not initialized");
    return m_impl->leafOrigin;
}
const void* DecodedNeuralLeafDevice::occupancyDevice() const {
    if (!m_impl) throw std::logic_error("device leaf is not initialized");
    return m_impl->occupancy;
}
const void* DecodedNeuralLeafDevice::valuesDevice() const {
    if (!m_impl) throw std::logic_error("device leaf is not initialized");
    return m_impl->values;
}
void* DecodedNeuralLeafDevice::streamHandle() const {
    if (!m_impl) throw std::logic_error("device leaf is not initialized");
    return reinterpret_cast<void*>(m_impl->stream);
}
void DecodedNeuralLeafDevice::wait() const {
    if (!m_impl) throw std::logic_error("device leaf is not initialized");
    checkCuda(cudaStreamSynchronize(m_impl->stream), "synchronize decoded device leaf");
    int error = 0;
    checkCuda(cudaMemcpy(&error, m_impl->validationError, sizeof(error), cudaMemcpyDeviceToHost), "read decoded leaf validation status");
    if (error == 1) throw std::runtime_error("leaf without a regressor is missing an exact active voxel value");
    if (error == 2) throw std::runtime_error("decoded active voxel value is not finite");
}

struct NeuralVolumeDecoder::Impl {
    struct DomainRuntime {
        std::unique_ptr<TinyCudaNNRuntime> classifier;
        std::unique_ptr<TinyCudaNNRuntime> regressor;
        float valueCenter = 0.0f;
        float valueScale = 1.0f;
        bool hasRegressor = false;
    };
    struct LeafRecord {
        size_t domainIndex = 0;
        size_t leafIndex = 0;
    };
    NeuralVolumeModel model;
    std::string configText;
    std::map<std::array<int32_t, 3>, size_t> domainByKey;
    std::map<std::array<int32_t, 3>, LeafRecord> leaves;
    std::unique_ptr<DomainRuntime> activeRuntime;
    size_t activeDomainIndex = std::numeric_limits<size_t>::max();

    DomainRuntime& runtimeFor(size_t index) {
        if (index >= model.domains.size()) throw std::out_of_range("ZNVM domain index is invalid");
        if (!activeRuntime || activeDomainIndex != index) {
            const auto& source = model.domains[index];
            auto runtime = std::make_unique<DomainRuntime>();
            runtime->classifier = std::make_unique<TinyCudaNNRuntime>(configText, source.classifierParams);
            runtime->hasRegressor = source.hasRegressor;
            runtime->valueCenter = source.valueCenter;
            runtime->valueScale = source.valueScale;
            if (source.hasRegressor) {
                runtime->regressor = std::make_unique<TinyCudaNNRuntime>(configText, source.regressorParams);
            }
            activeRuntime = std::move(runtime);
            activeDomainIndex = index;
        }
        return *activeRuntime;
    }
};

namespace {
std::array<int32_t, 3> domainKeyForLeaf(const std::array<int32_t, 3>& origin, uint32_t size) {
    std::array<int32_t, 3> key{};
    for (size_t axis = 0; axis < 3; ++axis) {
        key[axis] = static_cast<int32_t>(std::floor(static_cast<double>(origin[axis]) / size));
    }
    return key;
}
std::vector<std::array<float, 3>> normalizedLeafCoordinates(
    const std::array<int32_t, 3>& leafOrigin,
    const std::array<int32_t, 3>& domainOrigin,
    uint32_t domainSize) {
    std::vector<std::array<float, 3>> points(512);
    for (uint32_t offset = 0; offset < 512; ++offset) {
        const int64_t x = static_cast<int64_t>(leafOrigin[0]) + (offset >> 6);
        const int64_t y = static_cast<int64_t>(leafOrigin[1]) + ((offset >> 3) & 7);
        const int64_t z = static_cast<int64_t>(leafOrigin[2]) + (offset & 7);
        points[offset] = {{
            static_cast<float>(x - domainOrigin[0]) / domainSize,
            static_cast<float>(y - domainOrigin[1]) / domainSize,
            static_cast<float>(z - domainOrigin[2]) / domainSize
        }};
    }
    return points;
}
}

NeuralVolumeDecoder::NeuralVolumeDecoder(NeuralVolumeModel model)
    : m_impl(std::make_unique<Impl>()) {
    if (model.partial) throw std::invalid_argument("cannot construct a complete decoder from a partial ZNVM checkpoint");
    if (model.domainSizeVoxels == 0) throw std::invalid_argument("ZNVM has zero domain size");
    m_impl->model = std::move(model);
    const auto& ownedModel = m_impl->model;
    m_impl->configText = tcnn::cpp::json::parse(ownedModel.modelConfigJson).dump();
    for (size_t i = 0; i < ownedModel.domains.size(); ++i) {
        const auto& source = ownedModel.domains[i];
        if (!m_impl->domainByKey.emplace(source.key, i).second) {
            throw std::runtime_error("ZNVM contains duplicate domain keys");
        }
        for (size_t leafIndex = 0; leafIndex < source.leaves.size(); ++leafIndex) {
            const auto& leaf = source.leaves[leafIndex];
            if (!m_impl->leaves.emplace(leaf.origin, Impl::LeafRecord{i, leafIndex}).second) {
                throw std::runtime_error("ZNVM contains duplicate leaf origins");
            }
            if (domainKeyForLeaf(leaf.origin, ownedModel.domainSizeVoxels) != source.key) {
                throw std::runtime_error("ZNVM leaf origin does not belong to its recorded domain");
            }
        }
    }
    if (m_impl->leaves.size() != ownedModel.sourceLeafBlocks) {
        throw std::runtime_error("ZNVM leaf coverage does not match source leaf count");
    }
}
NeuralVolumeDecoder::~NeuralVolumeDecoder() = default;
NeuralVolumeDecoder::NeuralVolumeDecoder(NeuralVolumeDecoder&&) noexcept = default;
NeuralVolumeDecoder& NeuralVolumeDecoder::operator=(NeuralVolumeDecoder&&) noexcept = default;

const NeuralVolumeModel& NeuralVolumeDecoder::sourceModel() const {
    if (!m_impl) throw std::logic_error("neural volume decoder is not initialized");
    return m_impl->model;
}

bool NeuralVolumeDecoder::hasLeaf(const std::array<int32_t, 3>& origin) const {
    return m_impl && m_impl->leaves.find(origin) != m_impl->leaves.end();
}

DecodedNeuralLeaf NeuralVolumeDecoder::decodeLeaf(const std::array<int32_t, 3>& origin) {
    if (!m_impl) throw std::logic_error("neural volume decoder is not initialized");
    const auto leafIt = m_impl->leaves.find(origin);
    if (leafIt == m_impl->leaves.end()) throw std::out_of_range("requested leaf is not present in ZNVM");
    const auto& record = leafIt->second;
    const auto& ownedModel = m_impl->model;
    const auto& domain = m_impl->runtimeFor(record.domainIndex);
    const auto& sourceDomain = ownedModel.domains.at(record.domainIndex);
    const auto& leafCorrections = sourceDomain.leaves.at(record.leafIndex);
    const auto& domainKey = domainKeyForLeaf(origin, ownedModel.domainSizeVoxels);
    const auto domainIt = m_impl->domainByKey.find(domainKey);
    if (domainIt == m_impl->domainByKey.end() || domainIt->second != record.domainIndex) {
        throw std::runtime_error("leaf-to-domain lookup is inconsistent");
    }
    std::array<int32_t, 3> domainOrigin{};
    for (size_t axis = 0; axis < 3; ++axis) {
        const int64_t coordinate = static_cast<int64_t>(domainKey[axis]) * ownedModel.domainSizeVoxels;
        if (coordinate < std::numeric_limits<int32_t>::min() ||
            coordinate > std::numeric_limits<int32_t>::max()) {
            throw std::overflow_error("ZNVM domain origin exceeds index coordinate range");
        }
        domainOrigin[axis] = static_cast<int32_t>(coordinate);
    }
    const auto points = normalizedLeafCoordinates(origin, domainOrigin, ownedModel.domainSizeVoxels);
    const auto logits = domain.classifier->infer(points);
    if (logits.size() != 512) throw std::runtime_error("classifier returned an invalid leaf sample count");

    DecodedNeuralLeaf decoded;
    decoded.origin = origin;
    decoded.values.fill(ownedModel.background);
    std::vector<std::array<float, 3>> activePoints;
    std::vector<uint16_t> activeOffsets;
    for (uint16_t i = 0; i < 512; ++i) {
        decoded.occupancy[i] = logits[i] >= 0.0f ? 1u : 0u;
        if (decoded.occupancy[i] && domain.hasRegressor) {
            activePoints.push_back(points[i]);
            activeOffsets.push_back(i);
        }
    }
    if (domain.hasRegressor) {
        const auto normalizedValues = domain.regressor->infer(activePoints);
        if (normalizedValues.size() != activeOffsets.size()) {
            throw std::runtime_error("regressor returned an invalid active sample count");
        }
        for (size_t i = 0; i < activeOffsets.size(); ++i) {
            decoded.values[activeOffsets[i]] =
                normalizedValues[i] * domain.valueScale + domain.valueCenter;
        }
    }

    applyLeafCorrections(leafCorrections, decoded.occupancy, decoded.values);
    if (!sourceDomain.hasRegressor) {
        std::array<bool, 512> exactValuePresent{};
        for (const auto& correction : leafCorrections.exactValueOffsets) {
            exactValuePresent[correction.first] = true;
        }
        for (uint16_t i = 0; i < 512; ++i) {
            if (decoded.occupancy[i] && !exactValuePresent[i]) {
                throw std::runtime_error("leaf without a regressor is missing an exact active voxel value");
            }
        }
    }
    for (uint16_t i = 0; i < 512; ++i) {
        if (decoded.occupancy[i] && !std::isfinite(decoded.values[i])) {
            throw std::runtime_error("decoded active voxel value is not finite");
        }
    }
    return decoded;
}

DecodedNeuralLeafDevice NeuralVolumeDecoder::decodeLeafDevice(const std::array<int32_t, 3>& origin) {
    if (!m_impl) throw std::logic_error("neural volume decoder is not initialized");
    const auto leafIt = m_impl->leaves.find(origin);
    if (leafIt == m_impl->leaves.end()) throw std::out_of_range("requested leaf is not present in ZNVM");
    const auto& record = leafIt->second;
    const auto& ownedModel = m_impl->model;
    const auto& domain = m_impl->runtimeFor(record.domainIndex);
    const auto& sourceDomain = ownedModel.domains.at(record.domainIndex);
    const auto& corrections = sourceDomain.leaves.at(record.leafIndex);
    const auto key = domainKeyForLeaf(origin, ownedModel.domainSizeVoxels);
    std::array<int32_t, 3> domainOrigin{};
    for (size_t axis = 0; axis < 3; ++axis) {
        const int64_t coordinate = static_cast<int64_t>(key[axis]) * ownedModel.domainSizeVoxels;
        if (coordinate < std::numeric_limits<int32_t>::min() || coordinate > std::numeric_limits<int32_t>::max())
            throw std::overflow_error("ZNVM domain origin exceeds index coordinate range");
        domainOrigin[axis] = static_cast<int32_t>(coordinate);
    }
    const auto points = normalizedLeafCoordinates(origin, domainOrigin, ownedModel.domainSizeVoxels);
    DecodedNeuralLeafDevice result;
    result.m_impl = std::make_unique<DecodedNeuralLeafDevice::Impl>();
    result.m_impl->leafOrigin = origin;
    checkCuda(cudaStreamCreateWithFlags(&result.m_impl->stream, cudaStreamNonBlocking), "create decoded leaf stream");
    checkCuda(cudaMalloc(reinterpret_cast<void**>(&result.m_impl->occupancy), 512 * sizeof(uint8_t)), "allocate decoded occupancy");
    checkCuda(cudaMalloc(reinterpret_cast<void**>(&result.m_impl->values), 512 * sizeof(float)), "allocate decoded values");
    checkCuda(cudaMalloc(reinterpret_cast<void**>(&result.m_impl->validationError), sizeof(int)), "allocate decoded validation flag");
    checkCuda(cudaMemsetAsync(result.m_impl->validationError, 0, sizeof(int), result.m_impl->stream), "clear decoded validation flag");

    checkCuda(cudaMalloc(reinterpret_cast<void**>(&result.m_impl->logits), 512 * sizeof(float)), "allocate classifier logits");
    if (domain.hasRegressor)
        checkCuda(cudaMalloc(reinterpret_cast<void**>(&result.m_impl->regressed), 512 * sizeof(float)), "allocate regressor predictions");
    checkCuda(cudaMalloc(reinterpret_cast<void**>(&result.m_impl->occupancyOffsets), std::max<size_t>(1, corrections.occupancyOffsets.size()) * sizeof(uint16_t)), "allocate occupancy correction offsets");
    checkCuda(cudaMalloc(reinterpret_cast<void**>(&result.m_impl->exactOffsets), std::max<size_t>(1, corrections.exactValueOffsets.size()) * sizeof(uint16_t)), "allocate exact-value offsets");
    checkCuda(cudaMalloc(reinterpret_cast<void**>(&result.m_impl->exactValues), std::max<size_t>(1, corrections.exactValueOffsets.size()) * sizeof(float)), "allocate exact-value corrections");
    domain.classifier->inferDevice(points, result.m_impl->logits);
    if (domain.hasRegressor) domain.regressor->inferDevice(points, result.m_impl->regressed);

    if (!corrections.occupancyOffsets.empty())
        checkCuda(cudaMemcpy(result.m_impl->occupancyOffsets, corrections.occupancyOffsets.data(), corrections.occupancyOffsets.size() * sizeof(uint16_t), cudaMemcpyHostToDevice), "upload occupancy corrections");
    std::vector<uint16_t> exactOffsetHost;
    std::vector<float> exactValueHost;
    exactOffsetHost.reserve(corrections.exactValueOffsets.size());
    exactValueHost.reserve(corrections.exactValueOffsets.size());
    for (const auto& correction : corrections.exactValueOffsets) {
        exactOffsetHost.push_back(correction.first);
        exactValueHost.push_back(correction.second);    }
    if (!exactOffsetHost.empty()) {
        checkCuda(cudaMemcpy(result.m_impl->exactOffsets, exactOffsetHost.data(), exactOffsetHost.size() * sizeof(uint16_t), cudaMemcpyHostToDevice), "upload exact-value offsets");
        checkCuda(cudaMemcpy(result.m_impl->exactValues, exactValueHost.data(), exactValueHost.size() * sizeof(float), cudaMemcpyHostToDevice), "upload exact-value corrections");
    }
    assembleDecodedLeaf<<<2, 256, 0, result.m_impl->stream>>>(
        result.m_impl->logits, result.m_impl->regressed, domain.hasRegressor, domain.valueCenter, domain.valueScale, ownedModel.background,
        result.m_impl->occupancyOffsets, static_cast<uint32_t>(corrections.occupancyOffsets.size()),
        result.m_impl->exactOffsets, result.m_impl->exactValues, static_cast<uint32_t>(corrections.exactValueOffsets.size()),
        result.m_impl->occupancy, result.m_impl->values, result.m_impl->validationError);
    checkCuda(cudaGetLastError(), "assemble decoded device leaf");
    return result;
}

#if defined(ZENO_WITH_NANOVDB_CUDA_BUILDERS)
namespace {
struct Matrix4dRows {
    double row[4][4]{};
    const double* operator[](size_t i) const { return row[i]; }
    double* operator[](size_t i) { return row[i]; }
};

nanovdb::Map makeNanoMap(const NeuralVolumeModel& model) {
    Matrix4dRows matrix, inverse;
    for (size_t r = 0; r < 4; ++r) {
        for (size_t c = 0; c < 4; ++c) {
            const double v = model.indexToWorld[r * 4 + c];
            if (!std::isfinite(v)) throw std::runtime_error("ZNVM affine transform contains a non-finite value");
            matrix[r][c] = v;
        }
    }
    if (std::abs(matrix[0][3]) > 1e-10 || std::abs(matrix[1][3]) > 1e-10 ||
        std::abs(matrix[2][3]) > 1e-10 || std::abs(matrix[3][3] - 1.0) > 1e-10) {
        throw std::runtime_error("ZNVM affine transform is not in the supported affine row-vector form");
    }
    const double a = matrix[0][0], b = matrix[0][1], c = matrix[0][2];
    const double d = matrix[1][0], e = matrix[1][1], f = matrix[1][2];
    const double g = matrix[2][0], h = matrix[2][1], i = matrix[2][2];
    const double det = a * (e * i - f * h) - b * (d * i - f * g) + c * (d * h - e * g);
    if (!std::isfinite(det) || std::abs(det) < 1e-18) {
        throw std::runtime_error("ZNVM affine transform is singular");
    }
    const double invDet = 1.0 / det;
    inverse[0][0] =  (e * i - f * h) * invDet;
    inverse[0][1] =  (c * h - b * i) * invDet;
    inverse[0][2] =  (b * f - c * e) * invDet;
    inverse[1][0] =  (f * g - d * i) * invDet;
    inverse[1][1] =  (a * i - c * g) * invDet;
    inverse[1][2] =  (c * d - a * f) * invDet;
    inverse[2][0] =  (d * h - e * g) * invDet;
    inverse[2][1] =  (b * g - a * h) * invDet;
    inverse[2][2] =  (a * e - b * d) * invDet;
    nanovdb::Map map;
    map.set(matrix, inverse);
    return map;
}

__global__ void appendActiveLeafVoxels(
    const uint8_t* occupancy, const float* values,
    int32_t ox, int32_t oy, int32_t oz,
    nanovdb::Coord* coordsOut, float* valuesOut,
    uint64_t capacity, unsigned long long* count, int* error) {
    const uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= 512 || occupancy[i] == 0) return;
    const unsigned long long dst = atomicAdd(count, 1ull);
    if (dst >= capacity) {
        atomicExch(error, 1);
        return;
    }
    const int32_t x = ox + static_cast<int32_t>(i >> 6);
    const int32_t y = oy + static_cast<int32_t>((i >> 3) & 7);
    const int32_t z = oz + static_cast<int32_t>(i & 7);
    coordsOut[dst] = nanovdb::Coord(x, y, z);
    valuesOut[dst] = values[i];
}

struct TileDeviceRecord {
    int32_t x, y, z;
    uint32_t level;
    float value;
};
__global__ void initializeSidecar(float* sidecar, size_t count, float background) {
    const size_t i = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i < count) sidecar[i] = background;
}
__global__ void scatterTileValuesToIndexGrid(
    const nanovdb::NanoGrid<nanovdb::ValueOnIndex>* indexGrid,
    const TileDeviceRecord* tiles, uint64_t tileCount,
    uint64_t sidecarValueCount, float* sidecar, int* error) {
    const uint64_t i = static_cast<uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i >= tileCount) return;
    auto accessor = indexGrid->getAccessor();
    const auto& tile = tiles[i];
    const uint64_t index = accessor.getValue(nanovdb::Coord(tile.x, tile.y, tile.z));
    if (index == 0 || index > sidecarValueCount) {
        atomicExch(error, 3);
        return;
    }
    sidecar[index] = tile.value;
}

__global__ void scatterLeafValuesToIndexGrid(
    const nanovdb::NanoGrid<nanovdb::ValueOnIndex>* indexGrid,
    const nanovdb::Coord* coords, const float* values,
    uint64_t count, uint64_t sidecarValueCount, float* sidecar, int* error) {
    const uint64_t i = static_cast<uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i >= count) return;
    auto accessor = indexGrid->getAccessor();
    const uint64_t index = accessor.getValue(coords[i]);
    if (index == 0 || index > sidecarValueCount) {
        atomicExch(error, 2);
        return;
    }
    sidecar[index] = values[i];
}

__global__ void setNanoGridClass(nanovdb::NanoGrid<float>* grid, uint32_t gridClass) {
    if (blockIdx.x == 0 && threadIdx.x == 0) {
        grid->data()->mGridClass = static_cast<nanovdb::GridClass>(gridClass);
    }
}
} // namespace

struct NanoVDBDeviceGrid::Impl {
    using BufferT = nanovdb::cuda::Buffer<std::byte>;
    std::unique_ptr<nanovdb::GridHandle<BufferT>> handle;
    cudaStream_t stream = nullptr;
    uint64_t activeVoxels = 0;
    uint64_t sourceTiles = 0;
    ~Impl() {
        if (stream) cudaStreamSynchronize(stream);
        handle.reset();
        if (stream) cudaStreamDestroy(stream);
    }
};
#else
struct NanoVDBDeviceGrid::Impl {
    uint64_t activeVoxels = 0;
    uint64_t sourceTiles = 0;
    ActiveTile* sourceTilesDevice = nullptr;
};
#endif

NanoVDBDeviceGrid::NanoVDBDeviceGrid() = default;
NanoVDBDeviceGrid::~NanoVDBDeviceGrid() = default;
NanoVDBDeviceGrid::NanoVDBDeviceGrid(NanoVDBDeviceGrid&&) noexcept = default;
NanoVDBDeviceGrid& NanoVDBDeviceGrid::operator=(NanoVDBDeviceGrid&&) noexcept = default;
const void* NanoVDBDeviceGrid::gridDevice() const {
#if defined(ZENO_WITH_NANOVDB_CUDA_BUILDERS)
    if (!m_impl || !m_impl->handle) throw std::logic_error("NanoVDB device grid is not initialized");
    return m_impl->handle->template deviceGrid<float>();
#else
    throw std::runtime_error("NanoVDB CUDA GridBuilder/IndexToGrid headers were not available when Zeno was configured");
#endif
}
size_t NanoVDBDeviceGrid::bufferSize() const {
#if defined(ZENO_WITH_NANOVDB_CUDA_BUILDERS)
    if (!m_impl || !m_impl->handle) throw std::logic_error("NanoVDB device grid is not initialized");
    return m_impl->handle->bufferSize();
#else
    throw std::runtime_error("NanoVDB CUDA builders are unavailable in this build");
#endif
}
uint64_t NanoVDBDeviceGrid::activeVoxelCount() const {
    if (!m_impl) throw std::logic_error("NanoVDB device grid is not initialized");
    return m_impl->activeVoxels;
}
uint64_t NanoVDBDeviceGrid::sourceActiveTileCount() const {
    if (!m_impl) throw std::logic_error("NanoVDB device grid is not initialized");
    return m_impl->sourceTiles;
}
void NanoVDBDeviceGrid::wait() const {
#if defined(ZENO_WITH_NANOVDB_CUDA_BUILDERS)
    if (!m_impl || !m_impl->handle) throw std::logic_error("NanoVDB device grid is not initialized");
    checkCuda(cudaStreamSynchronize(m_impl->stream), "synchronize NanoVDB device grid");
#else
    throw std::runtime_error("NanoVDB CUDA builders are unavailable in this build");
#endif
}

NanoVDBDeviceGrid NeuralVolumeDecoder::buildNanoVDBDevice() {
    if (!m_impl) throw std::logic_error("neural volume decoder is not initialized");
#if !defined(ZENO_WITH_NANOVDB_CUDA_BUILDERS)
    throw std::runtime_error("This build lacks the pinned upstream NanoVDB GridBuilder/CreateNanoGrid/IndexToGrid headers; initialize projects/zenvdb/nanovdb-upstream and reconfigure Zeno");
#else
    const auto& model = m_impl->model;
    if (model.partial) throw std::invalid_argument("cannot build a complete NanoVDB grid from a partial ZNVM checkpoint");
    if (model.sourceActiveVoxels == 0 && model.activeTiles.empty()) {
        throw std::invalid_argument("cannot build an empty neural NanoVDB grid");
    }
    if (model.sourceActiveVoxels == 0) {
        throw std::invalid_argument("tile-only NanoVDB checkpoints are not supported by upstream IndexToGrid's zero-leaf path yet");
    }
    if (model.sourceActiveVoxels > std::numeric_limits<size_t>::max() / sizeof(nanovdb::Coord) ||
        model.sourceActiveVoxels > std::numeric_limits<size_t>::max() / sizeof(float)) {
        throw std::overflow_error("ZNVM active voxel count exceeds addressable GPU buffer size");
    }
    if (model.sourceActiveVoxels + model.activeTiles.size() >
        static_cast<uint64_t>(std::numeric_limits<uint32_t>::max())) {
        throw std::overflow_error("ZNVM leaf voxels plus active tiles exceed NanoVDB ValueOnIndex capacity");
    }

    NanoVDBDeviceGrid output;
    output.m_impl = std::make_unique<NanoVDBDeviceGrid::Impl>();
    output.m_impl->activeVoxels = model.sourceActiveVoxels;
    output.m_impl->sourceTiles = model.activeTiles.size();
    checkCuda(cudaStreamCreateWithFlags(&output.m_impl->stream, cudaStreamNonBlocking), "create NanoVDB build stream");
    const cudaStream_t stream = output.m_impl->stream;

    const size_t voxelCapacity = std::max<size_t>(1, static_cast<size_t>(model.sourceActiveVoxels));
    const size_t tileCapacity = std::max<size_t>(1, model.activeTiles.size());
    const uint64_t sidecarValueCount = model.sourceActiveVoxels + model.activeTiles.size();
    const size_t sidecarCapacity = static_cast<size_t>(sidecarValueCount) + 1;

    DeviceBuffer<nanovdb::Coord> dCoords(voxelCapacity);
    DeviceBuffer<float> dValues(voxelCapacity);
    DeviceBuffer<unsigned long long> dCount(1);
    DeviceBuffer<int> dError(1);
    DeviceBuffer<float> dSidecar(sidecarCapacity);
    DeviceBuffer<TileDeviceRecord> dTiles(tileCapacity);

    checkCuda(cudaMemsetAsync(dCount.get(), 0, sizeof(unsigned long long), stream), "clear NanoVDB active count");
    checkCuda(cudaMemsetAsync(dError.get(), 0, sizeof(int), stream), "clear NanoVDB bridge error");
    checkCuda(cudaStreamSynchronize(stream), "initialize NanoVDB append counters");

    // Construct only the sparse topology on the CPU. Each neural leaf contributes
    // its 512-byte occupancy mask; decoded float values remain on the GPU.
    using IndexBuildGrid = nanovdb::tools::build::Grid<nanovdb::ValueOnIndex>;
    constexpr int32_t leafDim = static_cast<int32_t>(IndexBuildGrid::Node0::DIM);
    constexpr int32_t lowerDim = static_cast<int32_t>(IndexBuildGrid::Node1::DIM);
    constexpr int32_t upperDim = static_cast<int32_t>(IndexBuildGrid::Node2::DIM);
    const auto alignDown = [](int32_t value, int32_t dim) {
        const int64_t v = value, d = dim;
        int64_t remainder = v % d;
        if (remainder < 0) remainder += d;
        return static_cast<int32_t>(v - remainder);
    };
    const auto nodeKey = [&](const nanovdb::Coord& coord, int32_t dim) {
        return std::array<int32_t, 3>{
            alignDown(coord.x(), dim), alignDown(coord.y(), dim), alignDown(coord.z(), dim)};
    };
    const auto tileContains = [&](const nanovdb::Coord& coord, const ActiveTile& tile) {
        const int32_t extent = tile.level == 1 ? leafDim : tile.level == 2 ? lowerDim : upperDim;        const int64_t x = coord.x(), y = coord.y(), z = coord.z();
        return x >= tile.coord[0] && x < int64_t(tile.coord[0]) + extent &&
               y >= tile.coord[1] && y < int64_t(tile.coord[1]) + extent &&
               z >= tile.coord[2] && z < int64_t(tile.coord[2]) + extent;
    };

    const std::string gridName = model.gridName.empty() ? "ZenoNeuralVolume" : model.gridName;
    IndexBuildGrid indexBuilder(1u, gridName);
    indexBuilder.mMap = makeNanoMap(model);
    if (model.gridClass < static_cast<uint32_t>(nanovdb::GridClass::End)) {
        indexBuilder.mGridClass = static_cast<nanovdb::GridClass>(model.gridClass);
    }
    std::set<std::array<int32_t, 3>> lowerParentKeys;

    unsigned long long topologyActiveCount = 0;
    for (const auto& domain : model.domains) {
        for (const auto& leafRecord : domain.leaves) {
            auto leaf = this->decodeLeafDevice(leafRecord.origin);
            leaf.wait();

            std::array<uint8_t, 512> hostOccupancy{};
            checkCuda(cudaMemcpy(hostOccupancy.data(), leaf.occupancyDevice(),
                                 hostOccupancy.size() * sizeof(uint8_t),
                                 cudaMemcpyDeviceToHost), "read compact neural leaf occupancy");
            bool leafHasActiveVoxel = false;
            for (uint32_t i = 0; i < 512; ++i) {
                if (hostOccupancy[i] == 0) continue;
                const int32_t x = leaf.origin()[0] + static_cast<int32_t>(i >> 6);
                const int32_t y = leaf.origin()[1] + static_cast<int32_t>((i >> 3) & 7);
                const int32_t z = leaf.origin()[2] + static_cast<int32_t>(i & 7);
                indexBuilder.setValue(nanovdb::Coord(x, y, z), 1u);
                leafHasActiveVoxel = true;
                ++topologyActiveCount;
            }
            if (leafHasActiveVoxel) {
                lowerParentKeys.insert(nodeKey(
                    nanovdb::Coord(leaf.origin()[0], leaf.origin()[1], leaf.origin()[2]), lowerDim));
            }

            appendActiveLeafVoxels<<<2, 256, 0, static_cast<cudaStream_t>(leaf.streamHandle())>>>(
                static_cast<const uint8_t*>(leaf.occupancyDevice()),
                static_cast<const float*>(leaf.valuesDevice()),
                leaf.origin()[0], leaf.origin()[1], leaf.origin()[2],
                dCoords.get(), dValues.get(), model.sourceActiveVoxels, dCount.get(), dError.get());
            checkCuda(cudaGetLastError(), "append active neural leaf voxels");
            checkCuda(cudaStreamSynchronize(static_cast<cudaStream_t>(leaf.streamHandle())),
                      "finish neural leaf append");
        }
    }
    if (topologyActiveCount != model.sourceActiveVoxels) {
        throw std::runtime_error("host occupancy metadata count does not match ZNVM source active-voxel count");
    }

    // Insert tiles only after leaf topology exists. A level-1 tile needs its
    // parent lower node to exist first; otherwise GridBuilder's convenience
    // addTile path seeds the newly-created parent with the tile's active value.
    for (const auto& tile : model.activeTiles) {
        const nanovdb::Coord coord(tile.coord[0], tile.coord[1], tile.coord[2]);
        switch (tile.level) {
        case 1: {
            const auto key = nodeKey(coord, lowerDim);
            if (lowerParentKeys.find(key) == lowerParentKeys.end()) {
                bool scaffolded = false;
                for (int32_t z = 0; z < lowerDim && !scaffolded; z += leafDim) {
                    for (int32_t y = 0; y < lowerDim && !scaffolded; y += leafDim) {
                        for (int32_t x = 0; x < lowerDim && !scaffolded; x += leafDim) {
                            const nanovdb::Coord candidate(key[0] + x, key[1] + y, key[2] + z);
                            if (candidate == coord) continue;
                            bool covered = false;
                            for (const auto& other : model.activeTiles) {
                                if (tileContains(candidate, other)) { covered = true; break; }
                            }
                            if (covered) continue;
                            indexBuilder.getAccessor().touchLeaf(candidate);
                            scaffolded = true;
                        }
                    }
                }
                if (!scaffolded) {
                    throw std::runtime_error("cannot create an empty parent leaf for an isolated level-1 NanoVDB tile");
                }
                lowerParentKeys.insert(key);
            }
            indexBuilder.root().template addTile<1>(coord, 1u, true);
            break;
        }
        case 2:
            indexBuilder.root().template addTile<2>(coord, 1u, true);
            break;
        case 3:
            indexBuilder.root().template addTile<3>(coord, 1u, true);
            break;
        default:
            throw std::runtime_error("ZNVM active tile level is outside NanoVDB's supported tree levels 1..3");
        }
    }

    unsigned long long activeCount = 0;
    int bridgeError = 0;
    checkCuda(cudaMemcpy(&activeCount, dCount.get(), sizeof(activeCount), cudaMemcpyDeviceToHost),
              "read NanoVDB active voxel count");
    checkCuda(cudaMemcpy(&bridgeError, dError.get(), sizeof(bridgeError), cudaMemcpyDeviceToHost),
              "read NanoVDB bridge status");
    if (bridgeError != 0) throw std::runtime_error("NanoVDB bridge active voxel buffer overflow");
    if (activeCount != model.sourceActiveVoxels) {
        throw std::runtime_error("decoded active voxel count does not match ZNVM source topology");
    }
    if (activeCount > static_cast<uint64_t>(std::numeric_limits<unsigned>::max()) * 256ull) {
        throw std::overflow_error("NanoVDB active voxel count exceeds one-dimensional CUDA launch limits");
    }

    // Create a sparse ValueOnIndex tree on the host, retaining native tile levels.
    // Only occupancy/topology metadata is converted; the scalar values are not.
    auto hostIndexHandle = nanovdb::tools::createNanoGrid(indexBuilder, 0u, false, true, 0);
    auto* hostIndexGrid = hostIndexHandle.template grid<nanovdb::ValueOnIndex>();
    if (!hostIndexGrid) throw std::runtime_error("NanoVDB GridBuilder did not produce a ValueOnIndex grid");
    if (hostIndexGrid->valueCount() != sidecarCapacity) {
        throw std::runtime_error("NanoVDB ValueOnIndex count does not match the ZNVM leaf-plus-tile value count");
    }
    auto hostIndexAccessor = hostIndexGrid->getAccessor();
    std::vector<uint64_t> hostTileIndices;
    hostTileIndices.reserve(model.activeTiles.size());
    for (const auto& tile : model.activeTiles) {
        const uint64_t index = hostIndexAccessor.getValue(
            nanovdb::Coord(tile.coord[0], tile.coord[1], tile.coord[2]));
        if (index == 0 || index > sidecarValueCount) {
            throw std::runtime_error("NanoVDB GridBuilder omitted an active internal tile from the index topology");
        }
        hostTileIndices.push_back(index);
    }
    std::sort(hostTileIndices.begin(), hostTileIndices.end());
    if (std::adjacent_find(hostTileIndices.begin(), hostTileIndices.end()) != hostTileIndices.end()) {
        throw std::runtime_error("NanoVDB assigned duplicate ValueOnIndex slots to distinct active tiles");
    }

    std::vector<TileDeviceRecord> hostTiles;
    hostTiles.reserve(model.activeTiles.size());
    for (const auto& tile : model.activeTiles) {
        hostTiles.push_back(TileDeviceRecord{tile.coord[0], tile.coord[1], tile.coord[2], tile.level, tile.value});
    }
    if (!hostTiles.empty()) {
        checkCuda(cudaMemcpyAsync(dTiles.get(), hostTiles.data(), hostTiles.size() * sizeof(TileDeviceRecord),
                                  cudaMemcpyHostToDevice, stream), "upload active tile values");
    }

    using BufferT = nanovdb::cuda::Buffer<std::byte>;
    auto indexDeviceHandle = nanovdb::cuda::copyTo<BufferT>(hostIndexHandle, stream);
    checkCuda(cudaStreamSynchronize(stream), "upload native NanoVDB topology");
    auto* indexGrid = indexDeviceHandle.template deviceGrid<nanovdb::ValueOnIndex>();
    if (!indexGrid) throw std::runtime_error("NanoVDB topology upload did not produce a device ValueOnIndex grid");

    initializeSidecar<<<static_cast<unsigned>((sidecarCapacity + 255) / 256), 256, 0, stream>>>(
        dSidecar.get(), sidecarCapacity, model.background);
    checkCuda(cudaGetLastError(), "initialize NanoVDB value sidecar");
    if (!hostTiles.empty()) {
        scatterTileValuesToIndexGrid<<<static_cast<unsigned>((hostTiles.size() + 127) / 128), 128, 0, stream>>>(
            indexGrid, dTiles.get(), hostTiles.size(), sidecarValueCount, dSidecar.get(), dError.get());
        checkCuda(cudaGetLastError(), "scatter native NanoVDB tile values");
    }
    if (activeCount != 0) {
        scatterLeafValuesToIndexGrid<<<static_cast<unsigned>((activeCount + 255) / 256), 256, 0, stream>>>(
            indexGrid, dCoords.get(), dValues.get(), activeCount, sidecarValueCount, dSidecar.get(), dError.get());
        checkCuda(cudaGetLastError(), "scatter neural values into NanoVDB index order");
    }
    checkCuda(cudaStreamSynchronize(stream), "finish NanoVDB value scatter");
    checkCuda(cudaMemcpy(&bridgeError, dError.get(), sizeof(bridgeError), cudaMemcpyDeviceToHost),
              "read NanoVDB value scatter status");
    if (bridgeError != 0) {
        throw std::runtime_error("NanoVDB index lookup failed while scattering a leaf or active-tile value");
    }

    nanovdb::tools::cuda::IndexToGrid<nanovdb::ValueOnIndex> converter(indexGrid, stream);
    converter.setGridName(gridName);
    auto valueHandle = converter.template getHandle<float>(dSidecar.get(), BufferT());
    checkCuda(cudaStreamSynchronize(stream), "finish NanoVDB IndexToGrid conversion");

    auto* valueGrid = const_cast<nanovdb::NanoGrid<float>*>(valueHandle.template deviceGrid<float>());
    if (!valueGrid) throw std::runtime_error("NanoVDB IndexToGrid did not produce a FloatGrid");
    if (model.gridClass < static_cast<uint32_t>(nanovdb::GridClass::End)) {
        setNanoGridClass<<<1, 1, 0, stream>>>(valueGrid, model.gridClass);
        checkCuda(cudaGetLastError(), "set NanoVDB grid class");
    }
    nanovdb::tools::cuda::updateChecksum(reinterpret_cast<nanovdb::GridData*>(valueGrid->data()), stream);
    checkCuda(cudaStreamSynchronize(stream), "finalize NanoVDB device grid");
    output.m_impl->handle = std::make_unique<nanovdb::GridHandle<BufferT>>(std::move(valueHandle));
    return output;
#endif
}
} // namespace zeno::neuralvolume