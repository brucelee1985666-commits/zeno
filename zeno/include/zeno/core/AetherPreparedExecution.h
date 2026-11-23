#pragma once

#include <zeno/utils/api.h>
#include <zeno/core/AetherExecutionPlan.h>
#include <zeno/core/IObject.h>

#include <functional>
#include <map>
#include <string>

namespace zeno {
struct Graph;

// A read-only-by-contract snapshot of values available before task dispatch.
// zany values are shared object references; callbacks must not mutate them.
struct AetherPreparedNodeInputs {
    std::string nodeId;
    std::map<std::string, zany> values;
};

// Copies static values and binds already-materialized upstream outputs without
// calling INode::requireInput/apply or modifying Graph, Context, or node maps.
// Rejects dynamic keyframe/formula inputs and unavailable upstream outputs.
ZENO_API std::map<std::string, AetherPreparedNodeInputs>
prepareAetherExecutionInputs(Graph const &graph, AetherExecutionPlan const &plan);

#ifdef ZENO_AETHER_ENABLE_TASKFLOW
using AetherPreparedNodeWork =
    std::function<void(AetherPreparedNodeInputs const &)>;

// Prepares all node inputs first, then dispatches the caller's specialized
// callbacks through Taskflow while preserving the audited dependency edges.
ZENO_API void executeAetherPreparedPlanTaskflow(
    Graph const &graph,
    AetherExecutionPlan const &plan,
    std::map<std::string, AetherPreparedNodeWork> const &preparedWork);
#endif

} // namespace zeno
