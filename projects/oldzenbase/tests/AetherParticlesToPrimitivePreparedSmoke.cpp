#include <zeno/AetherOldzenbasePreparedWork.h>
#include <zeno/ParticlesObject.h>
#include <zeno/PrimitiveObject.h>
#include <zeno/core/Graph.h>
#include <zeno/core/Session.h>

#include <cassert>
#include <cmath>
#include <memory>
#include <stdexcept>

int main() {
    auto &session = zeno::getSession();
    const auto classIt = session.nodeClasses.find("ParticlesToPrimitive");
    assert(classIt != session.nodeClasses.end());

    zeno::Graph graph;
    graph.session = &session;
    auto particles = std::make_shared<zeno::ParticlesObject>();
    particles->pos.emplace_back(1.0f, 2.0f, 3.0f);
    particles->pos.emplace_back(4.0f, 5.0f, 6.0f);
    particles->vel.emplace_back(0.0f, 1.0f, 0.0f);
    particles->vel.emplace_back(0.0f, 0.0f, 2.0f);

    auto node = classIt->second->new_instance();
    node->graph = &graph;
    node->nodeClass = classIt->second.get();
    node->myname = "convert";
    node->inputs["pars"] = particles;
    graph.nodes.emplace("convert", std::move(node));

    const auto &storedNode = *graph.nodes.at("convert");
    assert(zeno::isAetherParticlesToPrimitiveNode(storedNode));
    const auto plan = zeno::buildAetherExecutionPlan(graph, {"convert"});
    assert(plan.allNodesCpuParallelSafe);

    const auto work = zeno::makeAetherOldzenbasePreparedWorkForPlan(graph, plan);
    const auto outputs = zeno::executeAetherPreparedPlanTaskflow(graph, plan, work);
    const auto primitive = std::dynamic_pointer_cast<zeno::PrimitiveObject>(
        outputs.at("convert").at("prim"));
    assert(primitive && primitive->size() == 2);
    assert(primitive->attr<zeno::vec3f>("vel").size() == 2);
    assert(std::abs(primitive->attr<zeno::vec3f>("pos")[1][0] - 4.0f) < 1.0e-5f);
    assert(std::abs(primitive->attr<zeno::vec3f>("vel")[0][1] - 1.0f) < 1.0e-5f);
    assert(std::abs(primitive->attr<zeno::vec3f>("vel")[1][2] - 2.0f) < 1.0e-5f);
    assert(graph.nodes.at("convert")->outputs.empty());

    auto malformed = std::make_shared<zeno::ParticlesObject>();
    malformed->pos.emplace_back(0.0f, 0.0f, 0.0f);
    zeno::AetherPreparedNodeInputs malformedInputs;
    malformedInputs.nodeId = "malformed";
    malformedInputs.values["pars"] = malformed;
    bool rejectedMismatchedArrays = false;
    try {
        (void)zeno::makeAetherParticlesToPrimitivePreparedWork()(malformedInputs);
    } catch (std::invalid_argument const &) {
        rejectedMismatchedArrays = true;
    }
    assert(rejectedMismatchedArrays);
    return 0;
}
