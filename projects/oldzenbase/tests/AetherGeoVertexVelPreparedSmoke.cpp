#include <zeno/AetherOldzenbasePreparedWork.h>
#include <zeno/MeshObject.h>
#include <zeno/ParticlesObject.h>
#include <zeno/NumericObject.h>
#include <zeno/core/Graph.h>
#include <zeno/core/Session.h>

#include <cassert>
#include <cmath>
#include <memory>
#include <stdexcept>

int main() {
    auto &session = zeno::getSession();
    const auto classIt = session.nodeClasses.find("GeoVertexVel");
    assert(classIt != session.nodeClasses.end());

    zeno::Graph graph;
    graph.session = &session;
    auto target = std::make_shared<zeno::MeshObject>();
    target->vertices.emplace_back(3.0f, 2.0f, 1.0f);
    auto origin = std::make_shared<zeno::MeshObject>();
    origin->vertices.emplace_back(1.0f, 0.0f, 1.0f);

    auto node = classIt->second->new_instance();
    node->graph = &graph;
    node->nodeClass = classIt->second.get();
    node->myname = "velocity";
    node->inputs["dt"] = std::make_shared<zeno::NumericObject>(2.0f);
    node->inputs["TargetMesh"] = target;
    node->inputs["OriginMesh"] = origin;
    graph.nodes.emplace("velocity", std::move(node));

    const auto &storedNode = *graph.nodes.at("velocity");
    assert(zeno::isAetherGeoVertexVelNode(storedNode));
    const auto plan = zeno::buildAetherExecutionPlan(graph, {"velocity"});
    assert(plan.allNodesCpuParallelSafe);

    std::map<std::string, zeno::AetherPreparedNodeWork> work;
    work.emplace("velocity", zeno::makeAetherGeoVertexVelPreparedWork());
    const auto outputs = zeno::executeAetherPreparedPlanTaskflow(graph, plan, work);
    const auto particles = std::dynamic_pointer_cast<zeno::ParticlesObject>(
        outputs.at("velocity").at("MeshVel"));
    assert(particles);
    assert(particles->pos.size() == 1 && particles->vel.size() == 1);
    assert(std::abs(particles->pos[0].x - 3.0f) < 1.0e-5f);
    assert(std::abs(particles->vel[0].x - 1.0f) < 1.0e-5f);
    assert(std::abs(particles->vel[0].y - 1.0f) < 1.0e-5f);
    assert(std::abs(particles->vel[0].z) < 1.0e-5f);
    assert(graph.nodes.at("velocity")->outputs.empty());

    auto invalid = zeno::AetherPreparedNodeInputs{};
    invalid.nodeId = "invalid";
    invalid.values["dt"] = std::make_shared<zeno::NumericObject>(0.0f);
    invalid.values["TargetMesh"] = target;
    invalid.values["OriginMesh"] = origin;
    bool rejectedZeroDt = false;
    try {
        (void)zeno::makeAetherGeoVertexVelPreparedWork()(invalid);
    } catch (std::invalid_argument const &) {
        rejectedZeroDt = true;
    }
    assert(rejectedZeroDt);

    invalid.values["dt"] = std::make_shared<zeno::NumericObject>(1.0f);
    auto shortOrigin = std::make_shared<zeno::MeshObject>();
    invalid.values["OriginMesh"] = shortOrigin;
    bool rejectedMismatchedVertexCounts = false;
    try {
        (void)zeno::makeAetherGeoVertexVelPreparedWork()(invalid);
    } catch (std::invalid_argument const &) {
        rejectedMismatchedVertexCounts = true;
    }
    assert(rejectedMismatchedVertexCounts);
    return 0;
}
