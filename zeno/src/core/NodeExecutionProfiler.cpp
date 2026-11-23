#include <zeno/core/NodeExecutionProfiler.h>
#ifdef ZENO_AETHER_ENABLE_TRANSFER_PROFILER
#include <zensim/cuda/memory/TransferProfiler.hpp>
#endif

#include <algorithm>
#include <atomic>
#include <cmath>
#include <iomanip>
#include <mutex>
#include <sstream>
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

std::atomic<std::uint64_t> &executionSequence() {
    static std::atomic<std::uint64_t> sequence{0};
    return sequence;
}

std::uint64_t &activeExecutionId() {
    static std::uint64_t id = 0;
    return id;
}

thread_local unsigned int executionScopeDepth = 0;

std::string jsonEscape(const std::string &value) {
    std::ostringstream out;
    for (unsigned char ch : value) {
        switch (ch) {
        case '"': out << "\\\""; break;
        case '\\': out << "\\\\"; break;
        case '\b': out << "\\b"; break;
        case '\f': out << "\\f"; break;
        case '\n': out << "\\n"; break;
        case '\r': out << "\\r"; break;
        case '\t': out << "\\t"; break;
        default:
            if (ch < 0x20) {
                const char *hex = "0123456789abcdef";
                out << "\\u00" << hex[(ch >> 4) & 0x0f] << hex[ch & 0x0f];
            } else {
                out << static_cast<char>(ch);
            }
        }
    }
    return out.str();
}

void jsonNumber(std::ostringstream &out, double value) {
    if (std::isfinite(value)) out << std::setprecision(12) << value;
    else out << "null";
}

} // namespace

GpuTransferExecutionProfile getGpuTransferExecutionProfile() {
    GpuTransferExecutionProfile result;
#ifdef ZENO_AETHER_ENABLE_TRANSFER_PROFILER
    const auto snapshot = zs::transfer_profile::snapshot();
    result.hostToDeviceCalls = snapshot.h_to_d_calls;
    result.hostToDeviceBytes = snapshot.h_to_d_bytes;
    result.deviceToHostCalls = snapshot.d_to_h_calls;
    result.deviceToHostBytes = snapshot.d_to_h_bytes;
    result.deviceToDeviceCalls = snapshot.d_to_d_calls;
    result.deviceToDeviceBytes = snapshot.d_to_d_bytes;
    result.genericCalls = snapshot.generic_calls;
    result.genericBytes = snapshot.generic_bytes;
#endif
    return result;
}

void resetGpuTransferExecutionProfile() {
#ifdef ZENO_AETHER_ENABLE_TRANSFER_PROFILER
    zs::transfer_profile::reset();
#endif
}

void recordNodeExecution(const std::string &node, double elapsedMilliseconds, bool failed) {
    std::lock_guard<std::mutex> lock(profileMutex());
    auto &entry = profileData()[node];
    ++entry.calls;
    if (failed) ++entry.failures;
    entry.totalMilliseconds += elapsedMilliseconds;
    entry.maximumMilliseconds = std::max(entry.maximumMilliseconds, elapsedMilliseconds);
}

void recordNodeDependencyResolution(const std::string &node, std::uint64_t edges,
                                    double elapsedMilliseconds) {
    std::lock_guard<std::mutex> lock(profileMutex());
    auto &entry = profileData()[node];
    entry.dependencyEdges += edges;
    entry.dependencyResolutionMilliseconds += elapsedMilliseconds;
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

NodeDependencyProfileScope::NodeDependencyProfileScope(std::string node,
                                                         std::uint64_t edges)
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

AetherExecutionProfileScope::AetherExecutionProfileScope() {
    if (executionScopeDepth++ == 0) {
        resetNodeExecutionProfile();
        resetGpuKernelExecutionProfile();
        resetGpuTransferExecutionProfile();
        activeExecutionId() = executionSequence().fetch_add(1, std::memory_order_relaxed) + 1;
    }
}

AetherExecutionProfileScope::~AetherExecutionProfileScope() {
    if (executionScopeDepth > 0) --executionScopeDepth;
}

AetherExecutionReport getAetherExecutionReport() {
    AetherExecutionReport report;
    report.executionId = activeExecutionId();
    report.nodes = getNodeExecutionProfile();
    report.gpuKernels = getGpuKernelExecutionProfile();
    report.gpuTransfers = getGpuTransferExecutionProfile();

    for (auto const &node : report.nodes) {
        report.totalNodeApplyMilliseconds += node.totalMilliseconds;
        report.totalDependencyResolutionMilliseconds += node.dependencyResolutionMilliseconds;
    }
    for (auto const &kernel : report.gpuKernels)
        report.totalGpuKernelMilliseconds += kernel.totalMilliseconds;

    report.totalGpuTransferCalls =
        report.gpuTransfers.hostToDeviceCalls + report.gpuTransfers.deviceToHostCalls +
        report.gpuTransfers.deviceToDeviceCalls + report.gpuTransfers.genericCalls;
    report.totalGpuTransferBytes =
        report.gpuTransfers.hostToDeviceBytes + report.gpuTransfers.deviceToHostBytes +
        report.gpuTransfers.deviceToDeviceBytes + report.gpuTransfers.genericBytes;

    if (report.nodes.empty()) {
        report.findings.emplace_back(
            "No CPU node samples: enable ZENO_AETHER_ENABLE_PROFILER and execute a graph.");
    } else {
        const auto &slowest = report.nodes.front();
        std::ostringstream finding;
        finding << "Highest measured node apply time: " << slowest.node
                << " (" << std::setprecision(6) << slowest.totalMilliseconds
                << " ms total across " << slowest.calls << " call(s)).";
        report.findings.emplace_back(finding.str());
    }

    if (report.gpuKernels.empty()) {
        report.findings.emplace_back(
            "No GPU kernel samples: this build/run may not include the instrumented CUDA launch paths.");
    } else {
        const auto &slowest = report.gpuKernels.front();
        std::ostringstream finding;
        finding << "Highest measured instrumented GPU kernel: " << slowest.kernel
                << " (" << std::setprecision(6) << slowest.totalMilliseconds
                << " ms total).";
        report.findings.emplace_back(finding.str());
    }

    if (report.totalGpuTransferCalls == 0) {
        report.findings.emplace_back(
            "No explicit ZPC copy calls recorded; Unified Memory migration and uninstrumented CUDA copies are outside this counter.");
    } else {
        std::ostringstream finding;
        finding << "Recorded " << report.totalGpuTransferCalls << " explicit copy call(s), "
                << report.totalGpuTransferBytes << " requested byte(s); byte counts are not measured PCIe traffic.";
        report.findings.emplace_back(finding.str());
    }

    if (report.totalDependencyResolutionMilliseconds > 0.0) {
        report.findings.emplace_back(
            "Dependency resolution is reported separately; it can include upstream node execution and must not be added to node apply time as independent wall-clock cost.");
    }
    return report;
}

std::string getAetherExecutionReportJson() {
    const auto report = getAetherExecutionReport();
    std::ostringstream out;
    out << "{\n  \"schema\": \"zeno.aether.execution-report.v1\",\n";
    out << "  \"execution_id\": " << report.executionId << ",\n";
    out << "  \"totals\": {\n";
    out << "    \"node_apply_ms\": "; jsonNumber(out, report.totalNodeApplyMilliseconds); out << ",\n";
    out << "    \"dependency_resolution_ms\": "; jsonNumber(out, report.totalDependencyResolutionMilliseconds); out << ",\n";
    out << "    \"gpu_kernel_ms\": "; jsonNumber(out, report.totalGpuKernelMilliseconds); out << ",\n";
    out << "    \"gpu_transfer_calls\": " << report.totalGpuTransferCalls << ",\n";
    out << "    \"gpu_transfer_bytes\": " << report.totalGpuTransferBytes << "\n  },\n";

    out << "  \"nodes\": [";
    for (std::size_t i = 0; i < report.nodes.size(); ++i) {
        const auto &n = report.nodes[i];
        out << (i ? ",\n" : "\n") << "    {\"name\": \"" << jsonEscape(n.node)
            << "\", \"calls\": " << n.calls << ", \"failures\": " << n.failures
            << ", \"total_ms\": ";
        jsonNumber(out, n.totalMilliseconds);
        out << ", \"average_ms\": "; jsonNumber(out, n.averageMilliseconds);
        out << ", \"maximum_ms\": "; jsonNumber(out, n.maximumMilliseconds);
        out << ", \"dependency_edges\": " << n.dependencyEdges
            << ", \"dependency_resolution_ms\": ";
        jsonNumber(out, n.dependencyResolutionMilliseconds);
        out << "}";
    }
    if (!report.nodes.empty()) out << "\n  ";
    out << "],\n";

    out << "  \"gpu_kernels\": [";
    for (std::size_t i = 0; i < report.gpuKernels.size(); ++i) {
        const auto &k = report.gpuKernels[i];
        out << (i ? ",\n" : "\n") << "    {\"name\": \"" << jsonEscape(k.kernel)
            << "\", \"calls\": " << k.calls << ", \"total_ms\": ";
        jsonNumber(out, k.totalMilliseconds);
        out << ", \"average_ms\": "; jsonNumber(out, k.averageMilliseconds);
        out << ", \"maximum_ms\": "; jsonNumber(out, k.maximumMilliseconds);
        out << "}";
    }
    if (!report.gpuKernels.empty()) out << "\n  ";
    out << "],\n";

    const auto &t = report.gpuTransfers;
    out << "  \"gpu_transfers\": {"
        << "\"host_to_device\": {\"calls\": " << t.hostToDeviceCalls << ", \"bytes\": " << t.hostToDeviceBytes << "}, "
        << "\"device_to_host\": {\"calls\": " << t.deviceToHostCalls << ", \"bytes\": " << t.deviceToHostBytes << "}, "
        << "\"device_to_device\": {\"calls\": " << t.deviceToDeviceCalls << ", \"bytes\": " << t.deviceToDeviceBytes << "}, "
        << "\"generic\": {\"calls\": " << t.genericCalls << ", \"bytes\": " << t.genericBytes << "}},\n";

    out << "  \"findings\": [";
    for (std::size_t i = 0; i < report.findings.size(); ++i)
        out << (i ? ", " : "") << "\"" << jsonEscape(report.findings[i]) << "\"";
    out << "]\n}\n";
    return out.str();
}

} // namespace zeno
