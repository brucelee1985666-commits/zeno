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
    explicit TestNode(bool safe, bool writeShared = false) : safe_(safe), writeShared_(writeShared) {}
    zeno::AetherNodeExecutionCapabilities getAetherExecutionCapabilities() const override {
        zeno::AetherNodeExecutionCapabilities caps;
        caps.cpuThreadSafe = safe_;
        caps.sideEffectFree = safe_;
        caps.deterministic = safe_;
        caps.hasCudaImplementation = safe_;
        caps.explicitResourceAccess = safe_;
        if (writeShared_)
            caps.resources.push_back({"shared-buffer", zeno::AetherResourceAccessMode::Write, 4096});
        return caps;
    }
private:
    void apply() override {}
    bool safe_;
    bool writeShared_;
};

void addNode(zeno::Graph &graph, const std::string &id, bool safe, bool writeShared = false) {
    auto node = std::make_unique<TestNode>(safe, writeShared);
    node->myname = id;
    node->graph = &graph;
    graph.nodes.emplace(id, std::move(node));
}
}

int main() {
    zeno::Graph graph;
    addNode(graph, "source", true);
    addNode(graph, "sink", false);
    graph.nodes.at("sink")->inputBounds["in"] = {"source", "out"};
    addNode(graph, "writer-a", true, true);
    addNode(graph, "writer-b", true, true);
    addNode(graph, "cycle-a", true);
    addNode(graph, "cycle-b", true);
    graph.nodes.at("cycle-a")->inputBounds["in"] = {"cycle-b", "out"};
    graph.nodes.at("cycle-b")->inputBounds["in"] = {"cycle-a", "out"};

    const auto plan = zeno::buildAetherExecutionPlan(
        graph, std::set<std::string>{"sink", "writer-a", "writer-b"});
    assert(plan.nodes.size() == 4);
    assert(plan.dependencyEdges.size() == 1);
    assert(!plan.hasMissingNodes);
    assert(!plan.hasDependencyCycle);
    assert(!plan.allNodesCpuParallelSafe);
    assert(!plan.allNodesCudaReady);
    assert(plan.unorderedResourceHazards.size() == 1);
    assert(plan.unorderedResourceHazards.front().resource == "shared-buffer");

    const auto cycle = zeno::buildAetherExecutionPlan(
        graph, std::set<std::string>{"cycle-a"});
    assert(cycle.hasDependencyCycle);
    assert(!cycle.allNodesCpuParallelSafe);
    return 0;
}
