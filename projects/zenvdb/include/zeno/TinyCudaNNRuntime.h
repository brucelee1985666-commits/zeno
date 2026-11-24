#pragma once
#include <zeno/NeuralVolumeModel.h>
#include <array>
#include <memory>
#include <string>
#include <vector>
namespace zeno::neuralvolume {
class TinyCudaNNRuntime {
public:
    TinyCudaNNRuntime(const std::string& modelConfigJson, const std::vector<float>& serializedFloatParams);
    ~TinyCudaNNRuntime();
    TinyCudaNNRuntime(TinyCudaNNRuntime&&) noexcept;
    TinyCudaNNRuntime& operator=(TinyCudaNNRuntime&&) noexcept;
    TinyCudaNNRuntime(const TinyCudaNNRuntime&) = delete;
    TinyCudaNNRuntime& operator=(const TinyCudaNNRuntime&) = delete;
    std::vector<float> infer(const std::vector<std::array<float, 3>>& positions);
    // Writes scalar predictions directly to caller-owned CUDA memory; no D2H copy.
    // The output buffer must hold positions.size() float32 values.
    void inferDevice(const std::vector<std::array<float, 3>>& positions, void* deviceOutput);
private:
    struct Impl;
    std::unique_ptr<Impl> m_impl;
};
struct DecodedNeuralLeaf {
    std::array<int32_t, 3> origin{};
    std::array<uint8_t, 512> occupancy{};
    std::array<float, 512> values{};
};
class NanoVDBDeviceGrid {
public:
    NanoVDBDeviceGrid();
    ~NanoVDBDeviceGrid();
    NanoVDBDeviceGrid(NanoVDBDeviceGrid&&) noexcept;
    NanoVDBDeviceGrid& operator=(NanoVDBDeviceGrid&&) noexcept;
    NanoVDBDeviceGrid(const NanoVDBDeviceGrid&) = delete;
    NanoVDBDeviceGrid& operator=(const NanoVDBDeviceGrid&) = delete;
    // Opaque pointer to a device-resident nanovdb::NanoGrid<float>.
    const void* gridDevice() const;
    size_t bufferSize() const;
    uint64_t activeVoxelCount() const;
    uint64_t sourceActiveTileCount() const;
    void wait() const;
private:
    struct Impl;
    std::unique_ptr<Impl> m_impl;
    friend class NeuralVolumeDecoder;
};
class DecodedNeuralLeafDevice {
public:
    DecodedNeuralLeafDevice();
    ~DecodedNeuralLeafDevice();
    DecodedNeuralLeafDevice(DecodedNeuralLeafDevice&&) noexcept;
    DecodedNeuralLeafDevice& operator=(DecodedNeuralLeafDevice&&) noexcept;
    DecodedNeuralLeafDevice(const DecodedNeuralLeafDevice&) = delete;
    DecodedNeuralLeafDevice& operator=(const DecodedNeuralLeafDevice&) = delete;
    const std::array<int32_t, 3>& origin() const;
    // Opaque CUDA device pointers: occupancy is 512 uint8 values; values is 512 float32 values.
    const void* occupancyDevice() const;
    const void* valuesDevice() const;
    // Opaque cudaStream_t handle; enqueue dependent GPU work on this stream.
    void* streamHandle() const;
    // Synchronizes the producing stream and checks device-side validation flags.
    void wait() const;
private:
    struct Impl;
    std::unique_ptr<Impl> m_impl;
    friend class NeuralVolumeDecoder;
};
// Lazy leaf decoder: keeps neural weights on the GPU and decodes only the
// requested leaf. It does not materialize a complete OpenVDB grid.
class NeuralVolumeDecoder {
public:
    explicit NeuralVolumeDecoder(NeuralVolumeModel model);
    ~NeuralVolumeDecoder();
    NeuralVolumeDecoder(NeuralVolumeDecoder&&) noexcept;
    NeuralVolumeDecoder& operator=(NeuralVolumeDecoder&&) noexcept;
    NeuralVolumeDecoder(const NeuralVolumeDecoder&) = delete;
    NeuralVolumeDecoder& operator=(const NeuralVolumeDecoder&) = delete;
    const NeuralVolumeModel& sourceModel() const;
    bool hasLeaf(const std::array<int32_t, 3>& leafOrigin) const;
    DecodedNeuralLeaf decodeLeaf(const std::array<int32_t, 3>& leafOrigin);
    DecodedNeuralLeafDevice decodeLeafDevice(const std::array<int32_t, 3>& leafOrigin);
    // Builds one device-resident NanoVDB FloatGrid with native leaf and active-tile topology.
    NanoVDBDeviceGrid buildNanoVDBDevice();
private:
    struct Impl;
    std::unique_ptr<Impl> m_impl;
};
} // namespace zeno::neuralvolume
