#pragma once

#include <zeno/core/INode.h>
#include <zeno/core/AetherPreparedExecution.h>

namespace zeno {

// Identifies legacy nodes with reviewed prepared-callback adapters.
bool isAetherTransformMeshNode(INode const &node) noexcept;
bool isAetherGeoVertexVelNode(INode const &node) noexcept;
bool isAetherPrimitiveToParticlesNode(INode const &node) noexcept;
bool isAetherParticlesToPrimitiveNode(INode const &node) noexcept;

// Callbacks read prepared inputs only and return invocation-owned outputs.
// They never call INode::apply() or mutate shared input objects.
#ifdef ZENO_AETHER_ENABLE_TASKFLOW
AetherPreparedNodeWork makeAetherTransformMeshPreparedWork();
AetherPreparedNodeWork makeAetherGeoVertexVelPreparedWork();
AetherPreparedNodeWork makeAetherPrimitiveToParticlesPreparedWork();
AetherPreparedNodeWork makeAetherParticlesToPrimitivePreparedWork();

// Builds callbacks for every node in a validated CPU-safe plan. Unknown node
// types are rejected rather than falling back to arbitrary INode::apply().
std::map<std::string, AetherPreparedNodeWork>
makeAetherOldzenbasePreparedWorkForPlan(Graph const &graph, AetherExecutionPlan const &plan);
#endif

} // namespace zeno
