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
__global__ void compactFloat(const float* src, float* dst, uint32_t width, uint32_t n) {
    uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) dst[i] = src[static_cast<size_t>(i) * width];
}
__global__ void compactHalf(const __half* src, float* dst, uint32_t width, uint32_t n) {
    uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) dst[i] = __half2float(src[static_cast<size_t>(i) * width]);
}
}
struct TinyCudaNNRuntime::Impl {
    std::unique_ptr<tcnn::cpp::Module> module;
    cudaStream_t stream = nullptr;
    void* params = nullptr;
    uint32_t outputWidth = 0;
    bool fp16 = false;
    ~Impl() { if (params) cudaFree(params); if (stream) cudaStreamDestroy(stream); }
};
TinyCudaNNRuntime::TinyCudaNNRuntime(const std::string& jsonConfig, const std::vector<float>& weights)
    : m_impl(std::make_unique<Impl>()) {
    const auto config = tcnn::cpp::json::parse(jsonConfig);
    if (config.at("backend").get<std::string>() != "tiny-cuda-nn") throw std::runtime_error("unsupported ZNVM neural backend");
    const std::string storedPrecision = config.at("param_precision").get<std::string>();
    m_impl->module.reset(tcnn::cpp::create_network_with_input_encoding(3, 1, config.at("encoding"), config.at("network")));
    if (!m_impl->module) throw std::runtime_error("tiny-cuda-nn network creation failed");
    m_impl->fp16 = m_impl->module->param_precision() == tcnn::cpp::Precision::Fp16;
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
    constexpr uint32_t granularity = 256, maxBatch = 65536;
    std::vector<float> result; result.reserve(points.size());
    for (size_t offset = 0; offset < points.size();) {
        uint32_t count = static_cast<uint32_t>(std::min<size_t>(maxBatch, points.size() - offset));
        uint32_t batch = ((count + granularity - 1) / granularity) * granularity;
        std::vector<float> hostInput(static_cast<size_t>(batch) * 3);
        for (uint32_t i = 0; i < batch; ++i) {
            const auto& p = points[offset + std::min(i, count - 1)];
            for (unsigned axis = 0; axis < 3; ++axis) {
                if (!std::isfinite(p[axis])) throw std::invalid_argument("non-finite inference coordinate");
                hostInput[static_cast<size_t>(i) * 3 + axis] = p[axis];
            }
        }
        DeviceBuffer<float> input(hostInput.size()), compact(batch);
        checkCuda(cudaMemcpyAsync(input.get(), hostInput.data(), hostInput.size() * sizeof(float), cudaMemcpyHostToDevice, m_impl->stream), "upload inference coordinates");
        const uint32_t blocks = (batch + 255) / 256;
        std::vector<float> hostOutput(batch);
        if (m_impl->fp16) {
            DeviceBuffer<__half> output(static_cast<size_t>(batch) * m_impl->outputWidth);
            m_impl->module->inference(m_impl->stream, batch, input.get(), output.get(), m_impl->params);
            compactHalf<<<blocks, 256, 0, m_impl->stream>>>(output.get(), compact.get(), m_impl->outputWidth, batch);
        } else {
            DeviceBuffer<float> output(static_cast<size_t>(batch) * m_impl->outputWidth);
            m_impl->module->inference(m_impl->stream, batch, input.get(), output.get(), m_impl->params);
            compactFloat<<<blocks, 256, 0, m_impl->stream>>>(output.get(), compact.get(), m_impl->outputWidth, batch);
        }
        checkCuda(cudaGetLastError(), "compact inference output");
        checkCuda(cudaMemcpyAsync(hostOutput.data(), compact.get(), batch * sizeof(float), cudaMemcpyDeviceToHost, m_impl->stream), "download inference output");
        checkCuda(cudaStreamSynchronize(m_impl->stream), "synchronize inference");
        result.insert(result.end(), hostOutput.begin(), hostOutput.begin() + count);
        offset += count;
    }
    return result;
}
} // namespace zeno::neuralvolume
