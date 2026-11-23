#pragma once

#include <zeno/utils/api.h>
#include <chrono>
#include <cstdint>
#include <string>
#include <vector>

namespace zeno {

struct NodeExecutionProfile {
    std::string node;
    std::uint64_t calls = 0;
    std::uint64_t failures = 0;
    double totalMilliseconds = 0.0;
    double averageMilliseconds = 0.0;
    double maximumMilliseconds = 0.0;
    std::uint64_t dependencyEdges = 0;
    double dependencyResolutionMilliseconds = 0.0;
};

struct GpuKernelExecutionProfile {
    std::string kernel;
    std::uint64_t calls = 0;
    double totalMilliseconds = 0.0;
    double averageMilliseconds = 0.0;
    double maximumMilliseconds = 0.0;
};

struct GpuTransferExecutionProfile {
    std::uint64_t hostToDeviceCalls = 0;
    std::uint64_t hostToDeviceBytes = 0;
    std::uint64_t deviceToHostCalls = 0;
    std::uint64_t deviceToHostBytes = 0;
    std::uint64_t deviceToDeviceCalls = 0;
    std::uint64_t deviceToDeviceBytes = 0;
    std::uint64_t genericCalls = 0;
    std::uint64_t genericBytes = 0;
};

// Reports explicit ZPC copy calls only; not Unified Memory page migration.
ZENO_API GpuTransferExecutionProfile getGpuTransferExecutionProfile();
ZENO_API void resetGpuTransferExecutionProfile();

// GPU kernel timings are collected from CUDA events on the launch stream.
ZENO_API std::vector<GpuKernelExecutionProfile> getGpuKernelExecutionProfile();
ZENO_API void resetGpuKernelExecutionProfile();
ZENO_API void recordGpuKernelExecution(const std::string &kernel,
                                       double elapsedMilliseconds);

// Thread-safe aggregate profiler. Snapshot is sorted by total time descending.
ZENO_API std::vector<NodeExecutionProfile> getNodeExecutionProfile();
ZENO_API void resetNodeExecutionProfile();
ZENO_API void recordNodeExecution(const std::string &node,
                                  double elapsedMilliseconds,
                                  bool failed);

class NodeDependencyProfileScope {
public:
    NodeDependencyProfileScope(std::string nodeName, std::uint64_t dependencyEdges);
    ~NodeDependencyProfileScope();

    NodeDependencyProfileScope(NodeDependencyProfileScope const &) = delete;
    NodeDependencyProfileScope &operator=(NodeDependencyProfileScope const &) = delete;

private:
    std::string nodeName;
    std::uint64_t dependencyEdges;
    std::chrono::steady_clock::time_point started;
};

class NodeExecutionProfileScope {
public:
    explicit NodeExecutionProfileScope(std::string nodeName);
    ~NodeExecutionProfileScope();

    NodeExecutionProfileScope(NodeExecutionProfileScope const &) = delete;
    NodeExecutionProfileScope &operator=(NodeExecutionProfileScope const &) = delete;

    void markFailed() noexcept { failed = true; }

private:
    std::string nodeName;
    std::chrono::steady_clock::time_point started;
    bool failed = false;
};

} // namespace zeno
