#include <zeno/core/AetherPreparedExecution.h>
#include <zeno/core/Graph.h>
#include <zeno/core/INode.h>

#include <cassert>
#include <memory>
#include <stdexcept>
#include <string>

namespace {
class PreparedTestNode final : public zeno::INode {
public:
    zeno::AetherNodeExecutionCapabilities getAetherExecutionCapabilities() const override {
        zeno::AetherNodeExecutionCapabilities caps;
        caps.cpuThreadSafe = true;
        caps.sideEffectFree = true;
        caps.deterministic = true;
        return caps;
    }
private:
    void apply() override {}
};

void addNode(zeno::Graph &graph, const std::string &id) {
    auto node = std::make_unique<PreparedTestNode>();
    node->myname = id;
    node->graph = &graph;
    graph.nodes.emplace(id, std::move(node));
}
}

int main() {
    zeno::Graph graph;
    addNode(graph, "source");
    addNode(graph, "consumer");

    auto staticValue = std::make_shared<zeno::IObject>();
    graph.nodes.at("consumer")->inputs["constant:"] = staticValue;
    graph.nodes.at("consumer")->inputBounds["linked"] = {"source", "result"};

    const auto plan = zeno::buildAetherExecutionPlan(graph, {"consumer"});
    assert(plan.allNodesCpuParallelSafe);
    assert(plan.nodes.size() == 2);
    assert(graph.ctx == nullptr);
    assert(!graph.nodes.at("consumer")->has_input("linked"));

    // Preparation records immutable wiring; it does not require producer
    // outputs to exist yet and does not mutate the graph.
    const auto snapshots = zeno::prepareAetherExecutionInputs(graph, plan);
    assert(snapshots.size() == 2);
    const auto &consumer = snapshots.at("consumer");
    assert(consumer.values.at("constant:") == staticValue);
    assert(consumer.bindings.size() == 1);
    assert(consumer.bindings.front().socket == "linked");
    assert(consumer.bindings.front().sourceNode == "source");
    assert(consumer.bindings.front().sourceSocket == "result");
    assert(consumer.bindings.front().sourceNodeIndex == 0);
    assert(!consumer.bindings.front().mutedSourceOutput);

    assert(!graph.nodes.at("consumer")->has_input("linked"));
    assert(graph.nodes.at("source")->outputs.empty());
    assert(graph.ctx == nullptr);

    graph.nodes.at("consumer")->formulas.insert("dynamic:");
    bool rejectedFormula = false;
    try {
        (void)zeno::prepareAetherExecutionInputs(graph, plan);
    } catch (const std::invalid_argument &) {
        rejectedFormula = true;
    }
    assert(rejectedFormula);
    graph.nodes.at("consumer")->formulas.clear();

    // A graph edit after plan creation invalidates the dependency snapshot.
    graph.nodes.at("consumer")->inputBounds["linked"] = {"not-in-plan", "result"};
    bool rejectedStalePlan = false;
    try {
        (void)zeno::prepareAetherExecutionInputs(graph, plan);
    } catch (const std::invalid_argument &) {
        rejectedStalePlan = true;
    }
    assert(rejectedStalePlan);
    return 0;
}
