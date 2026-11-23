#pragma once

#include <zeno/core/AetherNodeExecutionCapabilities.h>

#include <cstddef>
#include <set>
#include <string>
#include <utility>
#include <vector>

namespace zeno {
struct Graph;

struct AetherExecutionPlanNode {
    std::string id;
    AetherNodeExecutionCapabilities capabilities;
    bool cpuParallelSafe = false;
    bool cudaDispatchReady = false;
    std::vector<std::string> blockers;
};

struct AetherExecutionPlan {
    std::vector<AetherExecutionPlanNode> nodes; // dependency-first order
    std::vector<std::pair<std::size_t, std::size_t>> dependencyEdges;
    bool hasMissingNodes = false;
    bool hasDependencyCycle = false;
    bool allNodesCpuParallelSafe = false;
    bool allNodesCudaReady = false;
};

// Builds an immutable, diagnostic-only snapshot of the dependency closure.
// It never calls apply(), mutates Graph::ctx, or changes legacy execution.
AetherExecutionPlan buildAetherExecutionPlan(
    Graph const &graph, std::set<std::string> const &roots);

} // namespace zeno
