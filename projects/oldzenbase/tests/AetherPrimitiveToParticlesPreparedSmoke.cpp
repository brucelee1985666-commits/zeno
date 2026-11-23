#include <zeno/AetherOldzenbasePreparedWork.h>
#include <zeno/PrimitiveObject.h>
#include <zeno/ParticlesObject.h>
#include <zeno/core/Graph.h>
#include <zeno/core/Session.h>

#include <cassert>
#include <cmath>
#include <memory>
#include <stdexcept>

int main() {
    auto &session = zeno::getSession();
    const auto classIt = session.nodeClasses.find("PrimitiveToParticles");
    assert(classIt != session.nodeClasses.end());

    zeno::Graph graph;
    graph.session = &session;
    auto primitive = std::make_shared<zeno::PrimitiveObject>();
    primitive->verts.emplace_back(1.0f, 2.0f, 3.0f);
    primitive->verts.emplace_back(4.0f, 5.0f, 6.0f);
    auto &velocity = primitive->add_attr<zeno::vec3f>("vel");
    velocity[0] = zeno::vec3f{0.0f, 1.0f, 0.0f};
    velocity[1] = zeno::vec3f{0.0f, 0.0f, 2.0f};

    auto node = classIt->second->new_instance();
    node->graph = &graph;
    node->nodeClass = classIt->second.get();
    node->myname = "convert";
    node->inputs["prim"] = primitive;
    graph.nodes.emplace("convert", std::move(node));

    const auto &storedNode = *graph.nodes.at("convert");
    assert(zeno::isAetherPrimitiveToParticlesNode(storedNode));
    const auto plan = zeno::buildAetherExecutionPlan(graph, {"convert"});
    assert(plan.allNodesCpuParallelSafe);

    std::map<std::string, zeno::AetherPreparedNodeWork> work;
    work.emplace("convert", zeno::makeAetherPrimitiveToParticlesPreparedWork());
    const auto outputs = zeno::executeAetherPreparedPlanTaskflow(graph, plan, work);
    const auto particles = std::dynamic_pointer_cast<zeno::ParticlesObject>(
        outputs.at("convert").at("pars"));
    assert(particles && particles->pos.size() == 2 && particles->vel.size() == 2);
    assert(std::abs(particles->pos[1].x - 4.0f) < 1.0e-5f);
    assert(std::abs(particles->vel[0].y - 1.0f) < 1.0e-5f);
    assert(std::abs(particles->vel[1].z - 2.0f) < 1.0e-5f);
    assert(graph.nodes.at("convert")->outputs.empty());

    auto noVelocityPrimitive = std::make_shared<zeno::PrimitiveObject>();
    noVelocityPrimitive->verts.emplace_back(7.0f, 8.0f, 9.0f);
    zeno::AetherPreparedNodeInputs noVelocityInputs;
    noVelocityInputs.nodeId = "no-velocity";
    noVelocityInputs.values["prim"] = noVelocityPrimitive;
    const auto noVelocityOutput =
        zeno::makeAetherPrimitiveToParticlesPreparedWork()(noVelocityInputs);
    const auto noVelocityParticles = std::dynamic_pointer_cast<zeno::ParticlesObject>(
        noVelocityOutput.at("pars"));
    assert(noVelocityParticles && noVelocityParticles->vel.size() == 1);
    assert(noVelocityParticles->vel[0] == glm::vec3(0.0f));

    auto malformedPrimitive = std::make_shared<zeno::PrimitiveObject>();
    malformedPrimitive->verts.emplace_back(0.0f, 0.0f, 0.0f);
    malformedPrimitive->add_attr<zeno::vec3f>("vel").clear();
    zeno::AetherPreparedNodeInputs malformedInputs;
    malformedInputs.nodeId = "malformed";
    malformedInputs.values["prim"] = malformedPrimitive;
    bool rejectedMismatchedAttribute = false;
    try {
        (void)zeno::makeAetherPrimitiveToParticlesPreparedWork()(malformedInputs);
    } catch (std::invalid_argument const &) {
        rejectedMismatchedAttribute = true;
    }
    assert(rejectedMismatchedAttribute);
    return 0;
}
