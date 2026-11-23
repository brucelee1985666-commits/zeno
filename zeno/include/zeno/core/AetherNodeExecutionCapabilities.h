#pragma once

#include <cstdint>
#include <string>
#include <vector>

namespace zeno {

// Conservative opt-in contract for future Aether scheduling.
// An unset capability always means "do not parallelize this node".
enum class AetherResourceAccessMode : std::uint8_t {
    Read,
    Write,
    ReadWrite
};

struct AetherResourceAccess {
    std::string resource;
    AetherResourceAccessMode mode = AetherResourceAccessMode::Read;
    std::uint64_t estimatedBytes = 0;
};

struct AetherNodeExecutionCapabilities {
    bool cpuThreadSafe = false;
    bool sideEffectFree = false;
    bool deterministic = false;
    bool hasCudaImplementation = false;
    bool explicitResourceAccess = false;
    std::vector<AetherResourceAccess> resources;

    bool isCpuParallelSafe() const noexcept {
        return cpuThreadSafe && sideEffectFree && deterministic;
    }

    bool isCudaDispatchReady() const noexcept {
        return isCpuParallelSafe() && hasCudaImplementation &&
               explicitResourceAccess;
    }
};

} // namespace zeno
