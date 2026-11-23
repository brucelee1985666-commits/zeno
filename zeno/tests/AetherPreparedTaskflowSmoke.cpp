#include <zeno/core/AetherPreparedExecution.h>
#include <zeno/core/Graph.h>
#include <zeno/core/INode.h>

#ifdef ZENO_AETHER_ENABLE_TASKFLOW

#include <atomic>
#include <cassert>
#include <memory>
#include <string>

namespace {
class PreparedTaskNode final : public zeno::INode {
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
    auto node = std::make_unique<PreparedTaskNode>();
    node->myname = id;
    node->graph = &graph;
    graph.nodes.emplace(id, std::move(node));
}
}

int main() {
    zeno::Graph graph;
    addNode(graph, "producer");
    addNode(graph, "consumer");
    graph.nodes.at("consumer")->inputBounds["input"] = {"producer", "payload"};

    const auto plan = zeno::buildAetherExecutionPlan(graph, {"consumer"});
    assert(plan.allNodesCpuParallelSafe);
    assert(graph.nodes.at("producer")->outputs.empty());

    auto payload = std::make_shared<zeno::IObject>();
    std::atomic<bool> consumerRan{false};
    std::map<std::string, zeno::AetherPreparedNodeWork> work;
    work["producer"] = [payload](zeno::AetherPreparedNodeInputs const &) {
        return std::map<std::string, zeno::zany>{{"payload", payload}};
    };
    work["consumer"] = [&consumerRan, payload](zeno::AetherPreparedNodeInputs const &inputs) {
        assert(inputs.values.at("input") == payload);
        consumerRan.store(true);
        return std::map<std::string, zeno::zany>{{"done", payload}};
    };

    const auto outputs = zeno::executeAetherPreparedPlanTaskflow(graph, plan, work);
    assert(consumerRan.load());
    assert(outputs.at("producer").at("payload") == payload);
    assert(outputs.at("consumer").at("done") == payload);

    // The adapter returns isolated outputs and never writes Graph node outputs.
    assert(graph.nodes.at("producer")->outputs.empty());
    assert(graph.nodes.at("consumer")->outputs.empty());
    assert(graph.ctx == nullptr);
    return 0;
}

#else
int main() { return 0; }
#endif
