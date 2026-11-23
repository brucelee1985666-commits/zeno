#include <zeno/core/NodeExecutionProfiler.h>

#include <algorithm>
#include <mutex>
#include <unordered_map>
#include <utility>

namespace zeno {
namespace {

struct Aggregate {
    std::uint64_t calls = 0;
    std::uint64_t failures = 0;
    double totalMilliseconds = 0.0;
    double maximumMilliseconds = 0.0;
    std::uint64_t dependencyEdges = 0;
    double dependencyResolutionMilliseconds = 0.0;
};

std::mutex &profileMutex() {
    static std::mutex mutex;
    return mutex;
}

std::unordered_map<std::string, Aggregate> &profileData() {
    static std::unordered_map<std::string, Aggregate> data;
    return data;
}

struct GpuAggregate {
    std::uint64_t calls = 0;
    double totalMilliseconds = 0.0;
    double maximumMilliseconds = 0.0;
};

std::unordered_map<std::string, GpuAggregate> &gpuProfileData() {
    static std::unordered_map<std::string, GpuAggregate> data;
    return data;
}

} // namespace

void recordNodeExecution(const std::string &node, double elapsedMilliseconds, bool failed) {
    std::lock_guard<std::mutex> lock(profileMutex());
    auto &entry = profileData()[node];
    ++entry.calls;
    if (failed) {
        ++entry.failures;
    }
    entry.totalMilliseconds += elapsedMilliseconds;
    entry.maximumMilliseconds = std::max(entry.maximumMilliseconds, elapsedMilliseconds);
}

std::vector<NodeExecutionProfile> getNodeExecutionProfile() {
    std::vector<NodeExecutionProfile> result;
    {
        std::lock_guard<std::mutex> lock(profileMutex());
        result.reserve(profileData().size());
        for (auto const &item : profileData()) {
            NodeExecutionProfile row;
            row.node = item.first;
            row.calls = item.second.calls;
            row.failures = item.second.failures;
            row.totalMilliseconds = item.second.totalMilliseconds;
            row.averageMilliseconds = row.calls
                ? row.totalMilliseconds / static_cast<double>(row.calls) : 0.0;
            row.maximumMilliseconds = item.second.maximumMilliseconds;
            row.dependencyEdges = item.second.dependencyEdges;
            row.dependencyResolutionMilliseconds = item.second.dependencyResolutionMilliseconds;
            result.emplace_back(std::move(row));
        }
    }
    std::sort(result.begin(), result.end(),
        [](NodeExecutionProfile const &a, NodeExecutionProfile const &b) {
            return a.totalMilliseconds > b.totalMilliseconds;
        });
    return result;
}

void recordNodeDependencyResolution(const std::string &node,
                                    std::uint64_t dependencyEdges,
                                    double elapsedMilliseconds) {
    std::lock_guard<std::mutex> lock(profileMutex());
    auto &entry = profileData()[node];
    entry.dependencyEdges += dependencyEdges;
    entry.dependencyResolutionMilliseconds += elapsedMilliseconds;
}

void resetNodeExecutionProfile() {
    std::lock_guard<std::mutex> lock(profileMutex());
    profileData().clear();
}

void recordGpuKernelExecution(const std::string &kernel, double elapsedMilliseconds) {
    std::lock_guard<std::mutex> lock(profileMutex());
    auto &entry = gpuProfileData()[kernel];
    ++entry.calls;
    entry.totalMilliseconds += elapsedMilliseconds;
    entry.maximumMilliseconds = std::max(entry.maximumMilliseconds, elapsedMilliseconds);
}

std::vector<GpuKernelExecutionProfile> getGpuKernelExecutionProfile() {
    std::vector<GpuKernelExecutionProfile> result;
    {
        std::lock_guard<std::mutex> lock(profileMutex());
        result.reserve(gpuProfileData().size());
        for (auto const &item : gpuProfileData()) {
            GpuKernelExecutionProfile row;
            row.kernel = item.first;
            row.calls = item.second.calls;
            row.totalMilliseconds = item.second.totalMilliseconds;
            row.averageMilliseconds = row.calls
                ? row.totalMilliseconds / static_cast<double>(row.calls) : 0.0;
            row.maximumMilliseconds = item.second.maximumMilliseconds;
            result.emplace_back(std::move(row));
        }
    }
    std::sort(result.begin(), result.end(),
        [](GpuKernelExecutionProfile const &a, GpuKernelExecutionProfile const &b) {
            return a.totalMilliseconds > b.totalMilliseconds;
        });
    return result;
}

void resetGpuKernelExecutionProfile() {
    std::lock_guard<std::mutex> lock(profileMutex());
    gpuProfileData().clear();
}

NodeDependencyProfileScope::NodeDependencyProfileScope(
    std::string node, std::uint64_t edges)
    : nodeName(std::move(node)), dependencyEdges(edges),
      started(std::chrono::steady_clock::now()) {}

NodeDependencyProfileScope::~NodeDependencyProfileScope() {
    const auto elapsed = std::chrono::duration<double, std::milli>(
        std::chrono::steady_clock::now() - started).count();
    recordNodeDependencyResolution(nodeName, dependencyEdges, elapsed);
}

NodeExecutionProfileScope::NodeExecutionProfileScope(std::string node)
    : nodeName(std::move(node)), started(std::chrono::steady_clock::now()) {}

NodeExecutionProfileScope::~NodeExecutionProfileScope() {
    const auto elapsed = std::chrono::duration<double, std::milli>(
        std::chrono::steady_clock::now() - started).count();
    recordNodeExecution(nodeName, elapsed, failed);
}

} // namespace zeno
