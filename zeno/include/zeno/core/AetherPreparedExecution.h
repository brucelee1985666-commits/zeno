#pragma once

#include <zeno/utils/api.h>
#include <zeno/core/AetherExecutionPlan.h>
#include <zeno/core/IObject.h>

#include <functional>
#include <map>
#include <string>
#include <vector>

namespace zeno {
struct Graph;

struct AetherPreparedInputBinding {
    std::string socket;
    std::string sourceNode;
    std::string sourceSocket;
    std::size_t sourceNodeIndex = 0;
    zany mutedSourceOutput;
};

// Static input values and immutable dependency metadata prepared before dispatch.
// zany values are shared references; callbacks must treat input objects as read-only.
struct AetherPreparedNodeInputs {
    std::string nodeId;
    std::map<std::string, zany> values;
    std::vector<AetherPreparedInputBinding> bindings;
};

// Copies static input values and dependency metadata without calling
// INode::requireInput/apply or modifying Graph, Context, or node maps.
// Rejects keyframe/formula inputs and stale/incomplete dependency plans.
ZENO_API std::map<std::string, AetherPreparedNodeInputs>
prepareAetherExecutionInputs(Graph const &graph, AetherExecutionPlan const &plan);

#ifdef ZENO_AETHER_ENABLE_TASKFLOW
using AetherPreparedNodeWork =
    std::function<std::map<std::string, zany>(AetherPreparedNodeInputs const &)>;

// Runs specialized callbacks as a DAG. Each callback returns its own output map;
// dependent callbacks receive those outputs only after Taskflow dependency completion.
// This does not invoke INode::apply() or mutate the source Graph.
ZENO_API std::map<std::string, std::map<std::string, zany>>
executeAetherPreparedPlanTaskflow(
    Graph const &graph,
    AetherExecutionPlan const &plan,
    std::map<std::string, AetherPreparedNodeWork> const &preparedWork);
#endif

} // namespace zeno
