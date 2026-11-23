#include <zeno/AetherOldzenbasePreparedWork.h>
#include <zeno/MeshObject.h>
#include <zeno/NumericObject.h>
#include <zeno/core/Graph.h>
#include <zeno/core/Session.h>

#include <cassert>
#include <cmath>
#include <memory>
#include <stdexcept>

int main() {
    auto &session = zeno::getSession();
    const auto classIt = session.nodeClasses.find("TransformMesh");
    assert(classIt != session.nodeClasses.end());

    zeno::Graph graph;
    graph.session = &session;

    auto source = std::make_shared<zeno::MeshObject>();
    source->vertices.emplace_back(1.0f, 2.0f, 3.0f);
    source->normals.emplace_back(0.0f, 1.0f, 0.0f);

    auto node = classIt->second->new_instance();
    node->graph = &graph;
    node->nodeClass = classIt->second.get();
    node->myname = "transform";
    node->inputs["mesh"] = source;
    node->inputs["translate"] = std::make_shared<zeno::NumericObject>(
        zeno::vec3f{4.0f, 0.0f, 0.0f});
    graph.nodes.emplace("transform", std::move(node));

    const auto &storedNode = *graph.nodes.at("transform");
    assert(zeno::isAetherTransformMeshNode(storedNode));
    const auto plan = zeno::buildAetherExecutionPlan(graph, {"transform"});
    assert(plan.allNodesCpuParallelSafe);

    std::map<std::string, zeno::AetherPreparedNodeWork> work;
    work.emplace("transform", zeno::makeAetherTransformMeshPreparedWork());
    const auto outputs = zeno::executeAetherPreparedPlanTaskflow(graph, plan, work);
    const auto transformed = std::dynamic_pointer_cast<zeno::MeshObject>(
        outputs.at("transform").at("mesh"));
    assert(transformed);
    assert(std::abs(transformed->vertices.at(0).x - 5.0f) < 1.0e-5f);
    assert(std::abs(transformed->vertices.at(0).y - 2.0f) < 1.0e-5f);
    assert(std::abs(transformed->vertices.at(0).z - 3.0f) < 1.0e-5f);
    assert(std::abs(transformed->normals.at(0).y - 1.0f) < 1.0e-5f);

    // The prepared callback returns a new object and never mutates graph outputs
    // or the shared source mesh.
    assert(transformed.get() != source.get());
    assert(std::abs(source->vertices.at(0).x - 1.0f) < 1.0e-5f);
    assert(graph.nodes.at("transform")->outputs.empty());

    auto singularSource = std::make_shared<zeno::MeshObject>();
    singularSource->vertices.emplace_back(1.0f, 0.0f, 0.0f);
    singularSource->normals.emplace_back(0.0f, 1.0f, 0.0f);
    zeno::AetherPreparedNodeInputs singularInputs;
    singularInputs.nodeId = "singular";
    singularInputs.values["mesh"] = singularSource;
    singularInputs.values["scaling"] = std::make_shared<zeno::NumericObject>(
        zeno::vec3f{0.0f, 1.0f, 1.0f});
    bool rejectedSingularNormalTransform = false;
    try {
        (void)zeno::makeAetherTransformMeshPreparedWork()(singularInputs);
    } catch (std::invalid_argument const &) {
        rejectedSingularNormalTransform = true;
    }
    assert(rejectedSingularNormalTransform);
    return 0;
}
