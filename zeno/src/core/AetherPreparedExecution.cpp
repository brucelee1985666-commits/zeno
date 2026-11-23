#include <zeno/core/AetherPreparedExecution.h>

#include <zeno/core/Graph.h>
#include <zeno/core/INode.h>

#include <set>
#include <stdexcept>
#include <utility>
#include <vector>

#ifdef ZENO_AETHER_ENABLE_TASKFLOW
#include <zeno/core/AetherExecutionPlanTaskflow.h>
#endif

namespace zeno {
namespace {

bool sameCapabilities(
    const AetherNodeExecutionCapabilities &left,
    const AetherNodeExecutionCapabilities &right) {
    if (left.cpuThreadSafe != right.cpuThreadSafe ||
        left.sideEffectFree != right.sideEffectFree ||
        left.deterministic != right.deterministic ||
        left.hasCudaImplementation != right.hasCudaImplementation ||
        left.explicitResourceAccess != right.explicitResourceAccess ||
        left.resources.size() != right.resources.size()) {
        return false;
    }
    for (std::size_t i = 0; i < left.resources.size(); ++i) {
        const auto &a = left.resources[i];
        const auto &b = right.resources[i];
        if (a.resource != b.resource || a.mode != b.mode ||
            a.estimatedBytes != b.estimatedBytes) {
            return false;
        }
    }
    return true;
}

} // namespace

std::map<std::string, AetherPreparedNodeInputs>
prepareAetherExecutionInputs(Graph const &graph, AetherExecutionPlan const &plan) {
    if (plan.nodes.empty())
        throw std::invalid_argument("Aether prepared execution plan is empty");
    if (plan.hasMissingNodes || plan.hasDependencyCycle ||
        !plan.unorderedResourceHazards.empty() ||
        !plan.allNodesCpuParallelSafe) {
        throw std::invalid_argument("Aether input preparation requires a CPU-safe audited plan");
    }

    std::set<std::string> plannedIds;
    std::map<std::string, std::size_t> plannedIndices;
    for (std::size_t index = 0; index < plan.nodes.size(); ++index) {
        const auto &entry = plan.nodes[index];
        if (entry.id.empty() || !plannedIds.insert(entry.id).second)
            throw std::invalid_argument("Aether plan contains an empty or duplicate node ID");
        if (!entry.cpuParallelSafe || !entry.capabilities.isCpuParallelSafe())
            throw std::invalid_argument("Aether plan node lacks CPU safety capabilities: " + entry.id);
        const auto graphNode = graph.nodes.find(entry.id);
        if (graphNode == graph.nodes.end() || !graphNode->second)
            throw std::invalid_argument("Aether plan node no longer exists in graph: " + entry.id);
        const auto currentCapabilities = graphNode->second->getAetherExecutionCapabilities();
        if (!sameCapabilities(entry.capabilities, currentCapabilities))
            throw std::invalid_argument("Aether node capabilities changed after plan creation: " + entry.id);
        plannedIndices.emplace(entry.id, index);
    }

    std::set<std::pair<std::size_t, std::size_t>> auditedEdges;
    for (const auto &edge : plan.dependencyEdges) {
        if (edge.first >= plan.nodes.size() || edge.second >= plan.nodes.size() ||
            edge.first == edge.second ||
            !auditedEdges.emplace(edge.first, edge.second).second) {
            throw std::invalid_argument("Aether plan contains an invalid or duplicate dependency edge");
        }
    }

    std::map<std::string, AetherPreparedNodeInputs> prepared;
    for (const auto &entry : plan.nodes) {
        const auto nodeIt = graph.nodes.find(entry.id);
        const INode *node = nodeIt->second.get();
        if (!node)
            throw std::invalid_argument("Aether graph contains a null node: " + entry.id);
        if (!node->kframes.empty() || !node->formulas.empty())
            throw std::invalid_argument(
                "Aether static snapshot does not evaluate keyframes or formulas: " + entry.id);

        AetherPreparedNodeInputs snapshot;
        snapshot.nodeId = entry.id;
        snapshot.values = node->inputs;

        for (const auto &binding : node->inputBounds) {
            const auto &socket = binding.first;
            const auto &sourceNodeId = binding.second.first;
            const auto &sourceSocket = binding.second.second;
            const auto sourceIndex = plannedIndices.find(sourceNodeId);
            if (plannedIds.find(sourceNodeId) == plannedIds.end() ||
                sourceIndex == plannedIndices.end())
                throw std::invalid_argument(
                    "Aether plan omits upstream dependency " + sourceNodeId +
                    " required by " + entry.id);
            const auto targetIndex = plannedIndices.at(entry.id);
            if (auditedEdges.find({sourceIndex->second, targetIndex}) == auditedEdges.end())
                throw std::invalid_argument(
                    "Aether plan is stale: dependency edge is missing for " +
                    sourceNodeId + " -> " + entry.id);
            const auto sourceIt = graph.nodes.find(sourceNodeId);
            if (sourceIt == graph.nodes.end() || !sourceIt->second)
                throw std::invalid_argument("Aether upstream node is unavailable: " + sourceNodeId);

            zany value;
            if (sourceIt->second->muted_output) {
                value = sourceIt->second->muted_output;
            } else {
                const auto outputIt = sourceIt->second->outputs.find(sourceSocket);
                if (outputIt == sourceIt->second->outputs.end() || !outputIt->second)
                    throw std::invalid_argument(
                        "Aether upstream output is not materialized: " +
                        sourceNodeId + "." + sourceSocket);
                value = outputIt->second;
            }
            snapshot.values[socket] = std::move(value);
        }

        prepared.emplace(entry.id, std::move(snapshot));
    }
    return prepared;
}

#ifdef ZENO_AETHER_ENABLE_TASKFLOW
void executeAetherPreparedPlanTaskflow(
    Graph const &graph,
    AetherExecutionPlan const &plan,
    std::map<std::string, AetherPreparedNodeWork> const &preparedWork) {
    const auto preparedInputs = prepareAetherExecutionInputs(graph, plan);
    std::map<std::string, std::function<void()>> callbacks;

    for (const auto &entry : plan.nodes) {
        const auto workIt = preparedWork.find(entry.id);
        if (workIt == preparedWork.end() || !workIt->second)
            throw std::invalid_argument(
                "Aether prepared callback is missing for node: " + entry.id);
        const auto inputIt = preparedInputs.find(entry.id);
        if (inputIt == preparedInputs.end())
            throw std::logic_error("Aether prepared input snapshot is missing: " + entry.id);

        const auto *inputs = &inputIt->second;
        const auto *work = &workIt->second;
        callbacks.emplace(entry.id, [inputs, work] { (*work)(*inputs); });
    }

    executeAetherExecutionPlanTaskflow(plan, callbacks);
}
#endif

} // namespace zeno
