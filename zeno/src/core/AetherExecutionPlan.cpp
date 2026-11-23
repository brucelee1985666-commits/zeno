#include <zeno/core/AetherExecutionPlan.h>

#include <zeno/core/Graph.h>
#include <zeno/core/INode.h>

#include <algorithm>
#include <map>
#include <set>
#include <string>
#include <utility>
#include <vector>

namespace zeno {
namespace {

enum class VisitState { Unseen, Visiting, Complete };

void addCapabilityBlockers(AetherExecutionPlanNode &entry) {
    const auto &caps = entry.capabilities;
    if (!caps.cpuThreadSafe) entry.blockers.emplace_back("CPU thread safety not declared");
    if (!caps.sideEffectFree) entry.blockers.emplace_back("side-effect freedom not declared");
    if (!caps.deterministic) entry.blockers.emplace_back("deterministic behavior not declared");
    if (!caps.hasCudaImplementation) entry.blockers.emplace_back("CUDA implementation not declared");
    if (!caps.explicitResourceAccess) entry.blockers.emplace_back("explicit resource access not declared");
}

} // namespace

AetherExecutionPlan buildAetherExecutionPlan(
    Graph const &graph, std::set<std::string> const &roots) {
    AetherExecutionPlan plan;
    std::map<std::string, VisitState> states;
    std::map<std::string, std::size_t> indices;
    std::set<std::pair<std::string, std::string>> edges;

    const auto visit = [&](const auto &self, const std::string &id) -> void {
        const auto existing = graph.nodes.find(id);
        if (existing == graph.nodes.end()) {
            plan.hasMissingNodes = true;
            return;
        }

        const auto state = states.find(id);
        if (state != states.end()) {
            if (state->second == VisitState::Visiting) plan.hasDependencyCycle = true;
            return;
        }

        states[id] = VisitState::Visiting;
        const INode *node = existing->second.get();
        for (const auto &binding : node->inputBounds) {
            const std::string &dependency = binding.second.first;
            if (graph.nodes.find(dependency) == graph.nodes.end()) {
                plan.hasMissingNodes = true;
                continue;
            }
            self(self, dependency);
            edges.emplace(dependency, id);
        }

        AetherExecutionPlanNode entry;
        entry.id = id;
        entry.capabilities = node->getAetherExecutionCapabilities();
        entry.cpuParallelSafe = entry.capabilities.isCpuParallelSafe();
        entry.cudaDispatchReady = entry.capabilities.isCudaDispatchReady();
        addCapabilityBlockers(entry);
        indices[id] = plan.nodes.size();
        plan.nodes.emplace_back(std::move(entry));
        states[id] = VisitState::Complete;
    };

    for (const auto &root : roots) visit(visit, root);

    for (const auto &edge : edges) {
        const auto from = indices.find(edge.first);
        const auto to = indices.find(edge.second);
        if (from != indices.end() && to != indices.end())
            plan.dependencyEdges.emplace_back(from->second, to->second);
    }

    plan.allNodesCpuParallelSafe = !plan.nodes.empty() &&
        !plan.hasMissingNodes && !plan.hasDependencyCycle &&
        std::all_of(plan.nodes.begin(), plan.nodes.end(),
            [](const AetherExecutionPlanNode &node) { return node.cpuParallelSafe; });
    plan.allNodesCudaReady = !plan.nodes.empty() &&
        !plan.hasMissingNodes && !plan.hasDependencyCycle &&
        std::all_of(plan.nodes.begin(), plan.nodes.end(),
            [](const AetherExecutionPlanNode &node) { return node.cudaDispatchReady; });
    return plan;
}

} // namespace zeno
