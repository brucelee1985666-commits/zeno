#include <zeno/AetherOldzenbasePreparedWork.h>
#include <zeno/core/Graph.h>

#include <stdexcept>

namespace zeno {

#ifdef ZENO_AETHER_ENABLE_TASKFLOW
std::map<std::string, AetherPreparedNodeWork>
makeAetherOldzenbasePreparedWorkForPlan(
    Graph const &graph, AetherExecutionPlan const &plan) {
    if (plan.nodes.empty() || !plan.allNodesCpuParallelSafe ||
        plan.hasMissingNodes || plan.hasDependencyCycle ||
        !plan.unorderedResourceHazards.empty()) {
        throw std::invalid_argument(
            "Aether oldzenbase dispatch requires a complete, CPU-safe, hazard-free plan");
    }

    std::map<std::string, AetherPreparedNodeWork> work;
    for (const auto &plannedNode : plan.nodes) {
        const auto graphNode = graph.nodes.find(plannedNode.id);
        if (graphNode == graph.nodes.end() || !graphNode->second)
            throw std::invalid_argument("Aether oldzenbase dispatch found a missing graph node");

        const auto &node = *graphNode->second;
        AetherPreparedNodeWork callback;
        if (isAetherTransformMeshNode(node)) {
            callback = makeAetherTransformMeshPreparedWork();
        } else if (isAetherGeoVertexVelNode(node)) {
            callback = makeAetherGeoVertexVelPreparedWork();
        } else if (isAetherPrimitiveToParticlesNode(node)) {
            callback = makeAetherPrimitiveToParticlesPreparedWork();
        } else if (isAetherParticlesToPrimitiveNode(node)) {
            callback = makeAetherParticlesToPrimitivePreparedWork();
        } else {
            throw std::invalid_argument(
                "Aether oldzenbase dispatch has no reviewed callback for node '" +
                plannedNode.id + "'");
        }
        if (!callback)
            throw std::invalid_argument("Aether oldzenbase dispatch produced an empty callback");
        work.emplace(plannedNode.id, std::move(callback));
    }
    return work;
}
#endif

} // namespace zeno
