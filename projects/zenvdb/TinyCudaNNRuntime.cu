#include <zeno/TinyCudaNNRuntime.h>
#include <tiny-cuda-nn/cpp_api.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <utility>
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
    float value = hasRegressor ? normalizedValues[i] * valueScale + valueCenter : background;
    bool exactFound = false;
    for (uint32_t c = 0; c < exactCount; ++c) {
        if (exactOffsets[c] == i) { value = exactValues[c]; exactFound = true; }
    }
    if (active && !hasRegressor && !exactFound) atomicExch(validationError, 1);
    if (active && !isfinite(value)) atomicExch(validationError, 2);
    occupancyOut[i] = active;
    valuesOut[i] = value;
}

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
    mutable bool checked = false;
    ~Impl() {
        if (stream) cudaStreamSynchronize(stream);
        if (occupancy) cudaFree(occupancy);
        if (values) cudaFree(values);
        if (validationError) cudaFree(validationError);
        if (stream) cudaStreamDestroy(stream);
    }
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
void DecodedNeuralLeafDevice::wait() const {
    if (!m_impl) throw std::logic_error("device leaf is not initialized");
    checkCuda(cudaStreamSynchronize(m_impl->stream), "synchronize decoded device leaf");
    int error = 0;
    checkCuda(cudaMemcpy(&error, m_impl->validationError, sizeof(error), cudaMemcpyDeviceToHost), "read decoded leaf validation status");
    m_impl->checked = true;
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

    DeviceBuffer<float> logits(512);
    DeviceBuffer<float> regressed(512);
    domain.classifier->inferDevice(points, logits.get());
    if (domain.hasRegressor) domain.regressor->inferDevice(points, regressed.get());

    DeviceBuffer<uint16_t> occupancyOffsets(std::max<size_t>(1, corrections.occupancyOffsets.size()));
    DeviceBuffer<uint16_t> exactOffsets(std::max<size_t>(1, corrections.exactValueOffsets.size()));
    DeviceBuffer<float> exactValues(std::max<size_t>(1, corrections.exactValueOffsets.size()));
    if (!corrections.occupancyOffsets.empty())
        checkCuda(cudaMemcpy(occupancyOffsets.get(), corrections.occupancyOffsets.data(), corrections.occupancyOffsets.size() * sizeof(uint16_t), cudaMemcpyHostToDevice), "upload occupancy corrections");
    std::vector<uint16_t> exactOffsetHost;
    std::vector<float> exactValueHost;
    exactOffsetHost.reserve(corrections.exactValueOffsets.size());
    exactValueHost.reserve(corrections.exactValueOffsets.size());
    for (const auto& correction : corrections.exactValueOffsets) {
        exactOffsetHost.push_back(correction.first);
        exactValueHost.push_back(correction.second);
    }
    if (!exactOffsetHost.empty()) {
        checkCuda(cudaMemcpy(exactOffsets.get(), exactOffsetHost.data(), exactOffsetHost.size() * sizeof(uint16_t), cudaMemcpyHostToDevice), "upload exact-value offsets");
        checkCuda(cudaMemcpy(exactValues.get(), exactValueHost.data(), exactValueHost.size() * sizeof(float), cudaMemcpyHostToDevice), "upload exact-value corrections");
    }
    assembleDecodedLeaf<<<2, 256, 0, result.m_impl->stream>>>(
        logits.get(), regressed.get(), domain.hasRegressor, domain.valueCenter, domain.valueScale, ownedModel.background,
        occupancyOffsets.get(), static_cast<uint32_t>(corrections.occupancyOffsets.size()),
        exactOffsets.get(), exactValues.get(), static_cast<uint32_t>(corrections.exactValueOffsets.size()),
        result.m_impl->occupancy, result.m_impl->values, result.m_impl->validationError);
    checkCuda(cudaGetLastError(), "assemble decoded device leaf");
    result.wait();
    return result;
}
} // namespace zeno::neuralvolume
