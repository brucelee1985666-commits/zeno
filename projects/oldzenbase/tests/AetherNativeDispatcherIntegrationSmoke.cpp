#include <zeno/AetherOldzenbasePreparedWork.h>
#include <zeno/ParticlesObject.h>
#include <zeno/PrimitiveObject.h>
#include <zeno/core/Graph.h>
#include <zeno/core/Session.h>

#include <cassert>
#include <cmath>
#include <memory>
#include <stdexcept>
#include <string>

namespace {
std::unique_ptr<zeno::INode> makeNode(
    zeno::Session &session, zeno::Graph &graph, const char *className,
    const char *id) {
    const auto it = session.nodeClasses.find(className);
    assert(it != session.nodeClasses.end());
    auto node = it->second->new_instance();
    node->graph = &graph;
    node->nodeClass = it->second.get();
    node->myname = id;
    return node;
}

std::shared_ptr<zeno::PrimitiveObject> makePrimitive(float x) {
    auto primitive = std::make_shared<zeno::PrimitiveObject>();
    primitive->verts.emplace_back(x, 2.0f, 3.0f);
    primitive->add_attr<zeno::vec3f>("vel", zeno::vec3f{0.0f, 1.0f, 0.0f});
    return primitive;
}
}

int main() {
    auto &session = zeno::getSession();
    zeno::Graph graph;
    graph.session = &session;

    auto sourcePrimitive = makePrimitive(1.0f);
    auto producer = makeNode(session, graph, "PrimitiveToParticles", "producer");
    producer->inputs["prim"] = sourcePrimitive;
    graph.nodes.emplace("producer", std::move(producer));

    auto consumer = makeNode(session, graph, "ParticlesToPrimitive", "consumer");
    consumer->inputBounds["pars"] = {"producer", "pars"};
    graph.nodes.emplace("consumer", std::move(consumer));

    const auto plan = zeno::buildAetherExecutionPlan(graph, {"consumer"});
    assert(plan.allNodesCpuParallelSafe);
    assert(plan.nodes.size() == 2);

    const auto work = zeno::makeAetherOldzenbasePreparedWorkForPlan(graph, plan);
    const auto outputs = zeno::executeAetherPreparedPlanTaskflow(graph, plan, work);
    auto result = std::dynamic_pointer_cast<zeno::PrimitiveObject>(
        outputs.at("consumer").at("prim"));
    assert(result && result->size() == 1);
    assert(std::abs(result->attr<zeno::vec3f>("pos")[0][0] - 1.0f) < 1.0e-5f);
    assert(std::abs(result->attr<zeno::vec3f>("vel")[0][1] - 1.0f) < 1.0e-5f);
    // Both conversion callbacks allocate outputs and leave shared source objects untouched.
    assert(sourcePrimitive->size() == 1);
    assert(std::abs(sourcePrimitive->verts[0][0] - 1.0f) < 1.0e-5f);
    assert(std::abs(sourcePrimitive->attr<zeno::vec3f>("vel")[0][1] - 1.0f) < 1.0e-5f);

    // Muted producer output must override the callback's produced value.
    auto mutedParticles = std::make_shared<zeno::ParticlesObject>();
    mutedParticles->pos.emplace_back(9.0f, 8.0f, 7.0f);
    mutedParticles->vel.emplace_back(0.0f, 0.0f, 1.0f);
    graph.nodes.at("producer")->muted_output = mutedParticles;
    const auto mutedOutputs = zeno::executeAetherPreparedPlanTaskflow(graph, plan, work);
    auto mutedResult = std::dynamic_pointer_cast<zeno::PrimitiveObject>(
        mutedOutputs.at("consumer").at("prim"));
    assert(mutedResult && mutedResult->size() == 1);
    assert(std::abs(mutedResult->attr<zeno::vec3f>("pos")[0][0] - 9.0f) < 1.0e-5f);
    assert(std::abs(mutedResult->attr<zeno::vec3f>("vel")[0][2] - 1.0f) < 1.0e-5f);
    assert(mutedParticles->pos.size() == 1 && mutedParticles->vel.size() == 1);
    assert(std::abs(mutedParticles->pos[0].x - 9.0f) < 1.0e-5f);
    assert(std::abs(mutedParticles->vel[0].z - 1.0f) < 1.0e-5f);

    // The explicit dispatcher and executor never write into legacy node outputs.
    assert(graph.nodes.at("producer")->outputs.empty());
    assert(graph.nodes.at("consumer")->outputs.empty());

    // A graph whose upstream binding changes after plan construction is rejected.
    graph.nodes.at("consumer")->inputBounds["pars"] = {"missing-node", "pars"};
    bool stalePlanRejected = false;
    try {
        const auto staleWork = zeno::makeAetherOldzenbasePreparedWorkForPlan(graph, plan);
        (void)zeno::executeAetherPreparedPlanTaskflow(graph, plan, staleWork);
    } catch (std::invalid_argument const &) {
        stalePlanRejected = true;
    }
    assert(stalePlanRejected);
    return 0;
}
