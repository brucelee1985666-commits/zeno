#include <zeno/core/AetherPreparedExecution.h>
#include <zeno/core/Graph.h>
#include <zeno/core/INode.h>

#include <cassert>
#include <memory>
#include <set>
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

    auto upstream = std::make_shared<zeno::IObject>();
    auto staticValue = std::make_shared<zeno::IObject>();
    graph.nodes.at("source")->set_output("result", upstream);
    graph.nodes.at("consumer")->inputs["constant:"] = staticValue;
    graph.nodes.at("consumer")->inputBounds["linked"] = {"source", "result"};

    const auto plan = zeno::buildAetherExecutionPlan(graph, {"consumer"});
    assert(plan.allNodesCpuParallelSafe);
    assert(plan.nodes.size() == 2);
    assert(graph.ctx == nullptr);
    assert(!graph.nodes.at("consumer")->has_input("linked"));

    const auto snapshots = zeno::prepareAetherExecutionInputs(graph, plan);
    assert(snapshots.size() == 2);
    const auto &consumer = snapshots.at("consumer");
    assert(consumer.values.at("linked") == upstream);
    assert(consumer.values.at("constant:") == staticValue);

    // Preparation must not mutate the original node's input/output maps.
    assert(!graph.nodes.at("consumer")->has_input("linked"));
    assert(graph.nodes.at("source")->outputs.at("result") == upstream);
    assert(graph.ctx == nullptr);

    graph.nodes.at("source")->outputs.erase("result");
    bool rejectedMissingOutput = false;
    try {
        (void)zeno::prepareAetherExecutionInputs(graph, plan);
    } catch (const std::invalid_argument &) {
        rejectedMissingOutput = true;
    }
    assert(rejectedMissingOutput);

    graph.nodes.at("source")->set_output("result", upstream);
    graph.nodes.at("consumer")->formulas.insert("dynamic:");
    bool rejectedFormula = false;
    try {
        (void)zeno::prepareAetherExecutionInputs(graph, plan);
    } catch (const std::invalid_argument &) {
        rejectedFormula = true;
    }
    assert(rejectedFormula);
    return 0;
}
