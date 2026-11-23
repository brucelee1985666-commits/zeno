#pragma once

#include <zeno/core/AetherExecutionPlan.h>
#include <zeno/core/AetherTaskflowExecutor.h>

#include <functional>
#include <map>
#include <string>

#ifdef ZENO_AETHER_ENABLE_TASKFLOW
namespace zeno {

// Executes a previously audited plan using caller-supplied, already-prepared
// callbacks. This deliberately does not call INode::apply() or access Graph::ctx.
// Every node must explicitly pass the CPU capability audit before any callback
// is dispatched. Callbacks must capture immutable inputs and isolated outputs.
ZENO_API void executeAetherExecutionPlanTaskflow(
    AetherExecutionPlan const &plan,
    std::map<std::string, std::function<void()>> const &preparedWork);

} // namespace zeno
#endif // ZENO_AETHER_ENABLE_TASKFLOW
