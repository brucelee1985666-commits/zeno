#include <zeno/core/AetherExecutionPlan.h>
#include <zeno/core/Graph.h>
#include <zeno/core/INode.h>

#include <cassert>
#include <memory>
#include <set>
#include <string>

namespace {
class TestNode final : public zeno::INode {
public:
    explicit TestNode(bool safe) : safe_(safe) {}
    zeno::AetherNodeExecutionCapabilities getAetherExecutionCapabilities() const override {
        zeno::AetherNodeExecutionCapabilities caps;
        caps.cpuThreadSafe = safe_;
        caps.sideEffectFree = safe_;
        caps.deterministic = safe_;
        caps.hasCudaImplementation = safe_;
        caps.explicitResourceAccess = safe_;
        return caps;
    }
private:
    void apply() override {}
    bool safe_;
};
}

int main() {
    zeno::Graph graph;
    auto source = std::make_unique<TestNode>(true);
    source->myname = "source";
    source->graph = &graph;
    graph.nodes.emplace("source", std::move(source));

    auto sink = std::make_unique<TestNode>(false);
    sink->myname = "sink";
    sink->graph = &graph;
    sink->inputBounds["in"] = {"source", "out"};
    graph.nodes.emplace("sink", std::move(sink));

    const auto plan = zeno::buildAetherExecutionPlan(graph, std::set<std::string>{"sink"});
    assert(plan.nodes.size() == 2);
    assert(plan.dependencyEdges.size() == 1);
    assert(!plan.hasMissingNodes);
    assert(!plan.hasDependencyCycle);
    assert(!plan.allNodesCpuParallelSafe);
    assert(!plan.allNodesCudaReady);
    assert(plan.nodes[0].id == "source");
    assert(plan.nodes[1].id == "sink");
    assert(plan.nodes[0].cpuParallelSafe);
    assert(!plan.nodes[1].cpuParallelSafe);
    return 0;
}
