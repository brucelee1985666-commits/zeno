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

bool writesResource(AetherResourceAccessMode mode) {
    return mode == AetherResourceAccessMode::Write ||
           mode == AetherResourceAccessMode::ReadWrite;
}

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

    // Conflicting accesses to a shared resource must be ordered by dataflow.
    const std::size_t count = plan.nodes.size();
    std::vector<std::vector<std::size_t>> adjacency(count);
    for (const auto &edge : plan.dependencyEdges)
        adjacency[edge.first].push_back(edge.second);

    const auto hasPath = [&](std::size_t from, std::size_t to) {
        std::vector<bool> seen(count, false);
        std::vector<std::size_t> stack{from};
        while (!stack.empty()) {
            const auto current = stack.back();
            stack.pop_back();
            if (current == to) return true;
            if (seen[current]) continue;
            seen[current] = true;
            for (const auto next : adjacency[current]) stack.push_back(next);
        }
        return false;
    };

    for (std::size_t i = 0; i < count; ++i) {
        for (std::size_t j = i + 1; j < count; ++j) {
            for (const auto &left : plan.nodes[i].capabilities.resources) {
                if (left.resource.empty()) continue;
                for (const auto &right : plan.nodes[j].capabilities.resources) {
                    if (left.resource != right.resource ||
                        (!writesResource(left.mode) && !writesResource(right.mode)))
                        continue;
                    if (!hasPath(i, j) && !hasPath(j, i)) {
                        plan.unorderedResourceHazards.push_back(
                            {plan.nodes[i].id, plan.nodes[j].id, left.resource});
                        plan.nodes[i].blockers.emplace_back(
                            "unordered conflicting access to resource: " + left.resource);
                        plan.nodes[j].blockers.emplace_back(
                            "unordered conflicting access to resource: " + left.resource);
                    }
                }
            }
        }
    }

    const bool noResourceHazards = plan.unorderedResourceHazards.empty();
    plan.allNodesCpuParallelSafe = !plan.nodes.empty() &&
        !plan.hasMissingNodes && !plan.hasDependencyCycle && noResourceHazards &&
        std::all_of(plan.nodes.begin(), plan.nodes.end(),
            [](const AetherExecutionPlanNode &node) { return node.cpuParallelSafe; });
    plan.allNodesCudaReady = !plan.nodes.empty() &&
        !plan.hasMissingNodes && !plan.hasDependencyCycle && noResourceHazards &&
        std::all_of(plan.nodes.begin(), plan.nodes.end(),
            [](const AetherExecutionPlanNode &node) { return node.cudaDispatchReady; });
    return plan;
}

} // namespace zeno
