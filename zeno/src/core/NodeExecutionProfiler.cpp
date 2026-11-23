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
#include <limits>
#include <unordered_map>
#include <unordered_set>
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

struct HistoricalAggregate {
    double meanApplyMilliseconds = 0.0;
    std::uint64_t samples = 0;
};

std::unordered_map<std::string, HistoricalAggregate> &historicalProfileData() {
    static std::unordered_map<std::string, HistoricalAggregate> data;
    return data;
}

std::vector<AetherScheduleEntry> &scheduleOrderData() {
    static std::vector<AetherScheduleEntry> data;
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

std::atomic<std::uint64_t> &activeExecutionId() {
    static std::atomic<std::uint64_t> id{0};
    return id;
}

thread_local unsigned int executionScopeDepth = 0;

std::string historicalNodeKey(const std::string &nodeName, const void *graphIdentity) {
    std::ostringstream key;
    key << reinterpret_cast<std::uintptr_t>(graphIdentity) << ':' << nodeName;
    return key.str();
}

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

void recordNodeExecution(const std::string &node, double elapsedMilliseconds, bool failed, const void *graphIdentity) {
    std::lock_guard<std::mutex> lock(profileMutex());
    auto &entry = profileData()[node];
    ++entry.calls;
    if (failed) ++entry.failures;
    entry.totalMilliseconds += elapsedMilliseconds;
    entry.maximumMilliseconds = std::max(entry.maximumMilliseconds, elapsedMilliseconds);

    // Keep one constant-space EWMA per observed (graph instance, node) pair.
    // Current-run counters may be reset; history survives those resets.
    if (!failed) {
        auto &history = historicalProfileData()[historicalNodeKey(node, graphIdentity)];
        constexpr double alpha = 0.20;
        history.meanApplyMilliseconds = history.samples == 0
            ? elapsedMilliseconds
            : alpha * elapsedMilliseconds + (1.0 - alpha) * history.meanApplyMilliseconds;
        ++history.samples;
    }
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


AetherNodeCostEstimate getAetherNodeCostEstimate(const std::string &nodeName, const void *graphIdentity) {
    std::lock_guard<std::mutex> lock(profileMutex());
    AetherNodeCostEstimate estimate;
    const auto it = historicalProfileData().find(historicalNodeKey(nodeName, graphIdentity));
    if (it == historicalProfileData().end()) return estimate;
    estimate.meanApplyMilliseconds = it->second.meanApplyMilliseconds;
    estimate.samples = it->second.samples;
    estimate.confidence = std::min(1.0,
        static_cast<double>(estimate.samples) / 8.0);
    return estimate;
}

void clearAetherNodeCostHistory(const void *graphIdentity) {
    std::ostringstream prefix;
    prefix << reinterpret_cast<std::uintptr_t>(graphIdentity) << ':';
    const std::string keyPrefix = prefix.str();
    std::lock_guard<std::mutex> lock(profileMutex());
    auto &history = historicalProfileData();
    for (auto it = history.begin(); it != history.end();) {
        if (it->first.compare(0, keyPrefix.size(), keyPrefix) == 0)
            it = history.erase(it);
        else
            ++it;
    }
}

void recordAetherScheduleOrder(const std::vector<AetherScheduleEntry> &entries) {
    std::lock_guard<std::mutex> lock(profileMutex());
    auto &schedule = scheduleOrderData();
    schedule.insert(schedule.end(), entries.begin(), entries.end());
}

std::vector<AetherScheduleEntry> getAetherScheduleOrder() {
    std::lock_guard<std::mutex> lock(profileMutex());
    return scheduleOrderData();
}

void resetAetherScheduleOrder() {
    std::lock_guard<std::mutex> lock(profileMutex());
    scheduleOrderData().clear();
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

NodeExecutionProfileScope::NodeExecutionProfileScope(std::string node, const void *identity)
    : nodeName(std::move(node)), started(std::chrono::steady_clock::now()), graphIdentity(identity) {}

NodeExecutionProfileScope::~NodeExecutionProfileScope() {
    const auto elapsed = std::chrono::duration<double, std::milli>(
        std::chrono::steady_clock::now() - started).count();
    recordNodeExecution(nodeName, elapsed, failed, graphIdentity);
}

AetherExecutionProfileScope::AetherExecutionProfileScope() {
    if (executionScopeDepth++ == 0) {
        resetNodeExecutionProfile();
        resetGpuKernelExecutionProfile();
        resetGpuTransferExecutionProfile();
        resetAetherScheduleOrder();
        activeExecutionId().store(
            executionSequence().fetch_add(1, std::memory_order_relaxed) + 1,
            std::memory_order_relaxed);
    }
}

AetherExecutionProfileScope::~AetherExecutionProfileScope() {
    if (executionScopeDepth > 0) --executionScopeDepth;
}

AetherExecutionReport getAetherExecutionReport() {
    AetherExecutionReport report;
    report.executionId = activeExecutionId().load(std::memory_order_relaxed);
#ifdef ZENO_AETHER_ENABLE_SMART_SCHEDULER
    report.schedulerMode = "history_critical_path_root_order";
#else
    report.schedulerMode = "legacy_input_order";
#endif
    report.nodes = getNodeExecutionProfile();
    report.gpuKernels = getGpuKernelExecutionProfile();
    report.schedule = getAetherScheduleOrder();
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

#ifdef ZENO_AETHER_ENABLE_SMART_SCHEDULER
    report.findings.emplace_back(
        "Smart scheduler reorders only requested graph roots using historical CPU apply EWMA and the longest known dependency path; dependencies remain on the existing recursive execution path.");
#else
    report.findings.emplace_back(
        "Smart scheduler is disabled; graph roots retain the legacy input order.");
#endif

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
    out << "  \"scheduler_mode\": \"" << jsonEscape(report.schedulerMode) << "\",\n";
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

    out << "  \"schedule\": [";
    for (std::size_t i = 0; i < report.schedule.size(); ++i) {
        const auto &entry = report.schedule[i];
        out << (i ? ",\n" : "\n") << "    {\"node\": \"" << jsonEscape(entry.node)
            << "\", \"estimated_critical_path_ms\": ";
        jsonNumber(out, entry.estimatedCriticalPathMilliseconds);
        out << ", \"historical_samples\": " << entry.historicalSamples
            << ", \"reason\": \"" << jsonEscape(entry.reason) << "\"}";
    }
    if (!report.schedule.empty()) out << "\n  ";
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


AetherExecutionReportComparison compareAetherExecutionReports(
    AetherExecutionReport const &baseline,
    AetherExecutionReport const &candidate) {
    AetherExecutionReportComparison comparison;
    comparison.baselineExecutionId = baseline.executionId;
    comparison.candidateExecutionId = candidate.executionId;
    comparison.nodeApplyDeltaMilliseconds =
        candidate.totalNodeApplyMilliseconds - baseline.totalNodeApplyMilliseconds;
    comparison.nodeApplyDeltaPercent = baseline.totalNodeApplyMilliseconds > 0.0
        ? comparison.nodeApplyDeltaMilliseconds / baseline.totalNodeApplyMilliseconds * 100.0
        : 0.0;
    comparison.gpuKernelDeltaMilliseconds =
        candidate.totalGpuKernelMilliseconds - baseline.totalGpuKernelMilliseconds;
    comparison.gpuKernelDeltaPercent = baseline.totalGpuKernelMilliseconds > 0.0
        ? comparison.gpuKernelDeltaMilliseconds / baseline.totalGpuKernelMilliseconds * 100.0
        : 0.0;

    const auto oldBytes = baseline.totalGpuTransferBytes;
    const auto newBytes = candidate.totalGpuTransferBytes;
    if (newBytes >= oldBytes) {
        const auto difference = newBytes - oldBytes;
        comparison.gpuTransferBytesDelta = difference >
            static_cast<std::uint64_t>(std::numeric_limits<std::int64_t>::max())
            ? std::numeric_limits<std::int64_t>::max()
            : static_cast<std::int64_t>(difference);
    } else {
        const auto difference = oldBytes - newBytes;
        comparison.gpuTransferBytesDelta = difference >
            static_cast<std::uint64_t>(std::numeric_limits<std::int64_t>::max())
            ? std::numeric_limits<std::int64_t>::min()
            : -static_cast<std::int64_t>(difference);
    }

    std::unordered_map<std::string, NodeExecutionProfile> oldNodes;
    std::unordered_map<std::string, NodeExecutionProfile> newNodes;
    std::unordered_set<std::string> nodeNames;
    for (const auto &entry : baseline.nodes) {
        oldNodes.emplace(entry.node, entry);
        nodeNames.insert(entry.node);
    }
    for (const auto &entry : candidate.nodes) {
        newNodes.emplace(entry.node, entry);
        nodeNames.insert(entry.node);
    }
    for (const auto &name : nodeNames) {
        const auto oldIt = oldNodes.find(name);
        const auto newIt = newNodes.find(name);
        const NodeExecutionProfile oldValue = oldIt == oldNodes.end()
            ? NodeExecutionProfile{} : oldIt->second;
        const NodeExecutionProfile newValue = newIt == newNodes.end()
            ? NodeExecutionProfile{} : newIt->second;
        AetherMetricDelta delta;
        delta.name = name;
        delta.previousCalls = oldValue.calls;
        delta.currentCalls = newValue.calls;
        delta.previousMilliseconds = oldValue.totalMilliseconds;
        delta.currentMilliseconds = newValue.totalMilliseconds;
        delta.deltaMilliseconds = delta.currentMilliseconds - delta.previousMilliseconds;
        delta.deltaPercent = delta.previousMilliseconds > 0.0
            ? delta.deltaMilliseconds / delta.previousMilliseconds * 100.0 : 0.0;
        comparison.nodes.emplace_back(std::move(delta));
    }
    std::sort(comparison.nodes.begin(), comparison.nodes.end(),
        [](AetherMetricDelta const &a, AetherMetricDelta const &b) {
            return a.deltaMilliseconds > b.deltaMilliseconds;
        });

    std::unordered_map<std::string, GpuKernelExecutionProfile> oldKernels;
    std::unordered_map<std::string, GpuKernelExecutionProfile> newKernels;
    std::unordered_set<std::string> kernelNames;
    for (const auto &entry : baseline.gpuKernels) {
        oldKernels.emplace(entry.kernel, entry);
        kernelNames.insert(entry.kernel);
    }
    for (const auto &entry : candidate.gpuKernels) {
        newKernels.emplace(entry.kernel, entry);
        kernelNames.insert(entry.kernel);
    }
    for (const auto &name : kernelNames) {
        const auto oldIt = oldKernels.find(name);
        const auto newIt = newKernels.find(name);
        const GpuKernelExecutionProfile oldValue = oldIt == oldKernels.end()
            ? GpuKernelExecutionProfile{} : oldIt->second;
        const GpuKernelExecutionProfile newValue = newIt == newKernels.end()
            ? GpuKernelExecutionProfile{} : newIt->second;
        AetherMetricDelta delta;
        delta.name = name;
        delta.previousCalls = oldValue.calls;
        delta.currentCalls = newValue.calls;
        delta.previousMilliseconds = oldValue.totalMilliseconds;
        delta.currentMilliseconds = newValue.totalMilliseconds;
        delta.deltaMilliseconds = delta.currentMilliseconds - delta.previousMilliseconds;
        delta.deltaPercent = delta.previousMilliseconds > 0.0
            ? delta.deltaMilliseconds / delta.previousMilliseconds * 100.0 : 0.0;
        comparison.gpuKernels.emplace_back(std::move(delta));
    }
    std::sort(comparison.gpuKernels.begin(), comparison.gpuKernels.end(),
        [](AetherMetricDelta const &a, AetherMetricDelta const &b) {
            return a.deltaMilliseconds > b.deltaMilliseconds;
        });
    return comparison;
}

} // namespace zeno
