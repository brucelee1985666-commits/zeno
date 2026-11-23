#include <zeno/core/AetherExecutionPlanTaskflow.h>

#ifdef ZENO_AETHER_ENABLE_TASKFLOW

#include <set>
#include <stdexcept>
#include <utility>
#include <vector>

namespace zeno {

void executeAetherExecutionPlanTaskflow(
    AetherExecutionPlan const &plan,
    std::map<std::string, std::function<void()>> const &preparedWork) {
    if (plan.nodes.empty())
        throw std::invalid_argument("Aether execution plan is empty");
    if (plan.hasMissingNodes || plan.hasDependencyCycle ||
        !plan.unorderedResourceHazards.empty() ||
        !plan.allNodesCpuParallelSafe) {
        throw std::invalid_argument(
            "Aether execution plan failed the CPU parallel-safety audit");
    }

    std::vector<AetherTaskflowTask> tasks;
    tasks.reserve(plan.nodes.size());
    std::set<std::string> nodeIds;
    for (const auto &node : plan.nodes) {
        if (!nodeIds.insert(node.id).second)
            throw std::invalid_argument("Aether execution plan contains a duplicate node ID: " + node.id);
        if (!node.cpuParallelSafe || !node.capabilities.isCpuParallelSafe())
            throw std::invalid_argument(
                "Aether execution plan contains a node without a complete CPU safety contract: " + node.id);
        const auto work = preparedWork.find(node.id);
        if (work == preparedWork.end() || !work->second)
            throw std::invalid_argument(
                "Aether prepared callback is missing for node: " + node.id);
        tasks.push_back({node.id, work->second});
    }

    AetherTaskflowExecutor executor;
    executor.run(tasks, plan.dependencyEdges);
}

} // namespace zeno

#endif // ZENO_AETHER_ENABLE_TASKFLOW
