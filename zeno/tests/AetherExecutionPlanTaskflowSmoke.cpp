#include <zeno/core/AetherExecutionPlanTaskflow.h>

#ifdef ZENO_AETHER_ENABLE_TASKFLOW

#include <atomic>
#include <cassert>
#include <functional>
#include <map>
#include <stdexcept>
#include <string>

int main() {
    zeno::AetherExecutionPlan plan;
    plan.allNodesCpuParallelSafe = true;
    for (const char *id : {"left", "right", "join"}) {
        zeno::AetherExecutionPlanNode node;
        node.id = id;
        node.cpuParallelSafe = true;
        node.capabilities.cpuThreadSafe = true;
        node.capabilities.sideEffectFree = true;
        node.capabilities.deterministic = true;
        plan.nodes.push_back(node);
    }
    plan.dependencyEdges = {{0, 2}, {1, 2}};

    std::atomic<int> completed{0};
    std::map<std::string, std::function<void()>> callbacks;
    callbacks["left"] = [&] { completed.fetch_add(1); };
    callbacks["right"] = [&] { completed.fetch_add(1); };
    callbacks["join"] = [&] {
        assert(completed.load() == 2);
        completed.fetch_add(1);
    };

    zeno::executeAetherExecutionPlanTaskflow(plan, callbacks);
    assert(completed.load() == 3);

    auto unsafe = plan;
    unsafe.allNodesCpuParallelSafe = false;
    bool rejected = false;
    try {
        zeno::executeAetherExecutionPlanTaskflow(unsafe, callbacks);
    } catch (const std::invalid_argument &) {
        rejected = true;
    }
    assert(rejected);
    return 0;
}

#else
int main() { return 0; }
#endif
