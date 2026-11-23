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


struct AetherExecutionReport {
    std::uint64_t executionId = 0;
    std::vector<NodeExecutionProfile> nodes;
    std::vector<GpuKernelExecutionProfile> gpuKernels;
    GpuTransferExecutionProfile gpuTransfers;
    double totalNodeApplyMilliseconds = 0.0;
    double totalDependencyResolutionMilliseconds = 0.0;
    double totalGpuKernelMilliseconds = 0.0;
    std::uint64_t totalGpuTransferCalls = 0;
    std::uint64_t totalGpuTransferBytes = 0;
    std::vector<std::string> findings;
};

// Snapshot of the current/most recently collected graph execution.
// Node apply time, dependency resolution time and GPU kernel time are separate
// measurements; do not add them as if they were mutually exclusive wall time.
ZENO_API AetherExecutionReport getAetherExecutionReport();
// Stable, dependency-free JSON export of the report.
ZENO_API std::string getAetherExecutionReportJson();

struct AetherMetricDelta {
    std::string name;
    std::uint64_t previousCalls = 0;
    std::uint64_t currentCalls = 0;
    double previousMilliseconds = 0.0;
    double currentMilliseconds = 0.0;
    double deltaMilliseconds = 0.0;
    // Zero when the baseline is zero (relative change is undefined).
    double deltaPercent = 0.0;
};

struct AetherExecutionReportComparison {
    std::uint64_t baselineExecutionId = 0;
    std::uint64_t candidateExecutionId = 0;
    double nodeApplyDeltaMilliseconds = 0.0;
    double nodeApplyDeltaPercent = 0.0;
    double gpuKernelDeltaMilliseconds = 0.0;
    double gpuKernelDeltaPercent = 0.0;
    std::int64_t gpuTransferBytesDelta = 0;
    std::vector<AetherMetricDelta> nodes;
    std::vector<AetherMetricDelta> gpuKernels;
};

ZENO_API AetherExecutionReportComparison compareAetherExecutionReports(
    AetherExecutionReport const &baseline,
    AetherExecutionReport const &candidate);

// Graph::applyNodes owns this scope. Nested graph applications on the same
// thread share the outer execution and do not reset its counters.
class AetherExecutionProfileScope {
public:
    AetherExecutionProfileScope();
    ~AetherExecutionProfileScope();
    AetherExecutionProfileScope(AetherExecutionProfileScope const &) = delete;
    AetherExecutionProfileScope &operator=(AetherExecutionProfileScope const &) = delete;
};

} // namespace zeno
