#pragma once

#include <zeno/core/INode.h>
#include <zeno/core/AetherPreparedExecution.h>

namespace zeno {

// Identifies the one legacy node with a reviewed prepared-callback adapter.
bool isAetherTransformMeshNode(INode const &node) noexcept;

// The callback reads prepared inputs only and returns a fresh MeshObject.
// It never calls INode::apply() or mutates the input mesh.
#ifdef ZENO_AETHER_ENABLE_TASKFLOW
AetherPreparedNodeWork makeAetherTransformMeshPreparedWork();
#endif

} // namespace zeno
