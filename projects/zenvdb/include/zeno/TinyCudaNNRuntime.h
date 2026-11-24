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
    // Coordinates are normalized to [0,1] in the owning domain, as in train_znvds.py.
    std::vector<float> infer(const std::vector<std::array<float, 3>>& positions);
private:
    struct Impl;
    std::unique_ptr<Impl> m_impl;
};
} // namespace zeno::neuralvolume
