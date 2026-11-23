#include <zeno/zeno.h>
#include <zeno/MeshObject.h>
#include <zeno/ParticlesObject.h>
#include <zeno/NumericObject.h>
#include <zeno/AetherOldzenbasePreparedWork.h>

#include <cmath>
#include <stdexcept>

namespace zeno {
namespace {

std::shared_ptr<ParticlesObject> buildVertexVelocity(
    std::map<std::string, zany> const &inputs) {
    const auto dtIt = inputs.find("dt");
    const auto targetIt = inputs.find("TargetMesh");
    const auto originIt = inputs.find("OriginMesh");
    if (dtIt == inputs.end() || !dtIt->second ||
        targetIt == inputs.end() || !targetIt->second ||
        originIt == inputs.end() || !originIt->second) {
        throw std::invalid_argument("GeoVertexVel requires dt, TargetMesh, and OriginMesh");
    }

    const auto dtObject = std::dynamic_pointer_cast<NumericObject>(dtIt->second);
    const auto target = std::dynamic_pointer_cast<MeshObject>(targetIt->second);
    const auto origin = std::dynamic_pointer_cast<MeshObject>(originIt->second);
    if (!dtObject || !target || !origin)
        throw std::invalid_argument("GeoVertexVel received an input with an invalid object type");

    const float dt = dtObject->get<float>();
    if (!std::isfinite(dt) || std::abs(dt) <= 1.0e-8f)
        throw std::invalid_argument("GeoVertexVel requires a finite, non-zero dt");
    if (target->vertices.size() != origin->vertices.size())
        throw std::invalid_argument("GeoVertexVel meshes must have equal vertex counts");

    auto result = std::make_shared<ParticlesObject>();
    result->pos.resize(target->vertices.size());
    result->vel.resize(target->vertices.size());
    for (std::size_t i = 0; i < target->vertices.size(); ++i) {
        const auto &targetPosition = target->vertices[i];
        const auto &originPosition = origin->vertices[i];
        const auto velocity = (targetPosition - originPosition) / dt;
        if (!std::isfinite(velocity.x) || !std::isfinite(velocity.y) ||
            !std::isfinite(velocity.z))
            throw std::invalid_argument("GeoVertexVel produced a non-finite velocity");
        result->pos[i] = targetPosition;
        result->vel[i] = velocity;
    }
    return result;
}

} // namespace

struct GeoVertexVel : zeno::INode {
    AetherNodeExecutionCapabilities getAetherExecutionCapabilities() const override {
        AetherNodeExecutionCapabilities capabilities;
        capabilities.cpuThreadSafe = true;
        capabilities.sideEffectFree = true;
        capabilities.deterministic = true;
        return capabilities;
    }

    void apply() override {
        std::map<std::string, zany> resolvedInputs;
        resolvedInputs.emplace("dt", get_input("dt"));
        resolvedInputs.emplace("TargetMesh", get_input("TargetMesh"));
        resolvedInputs.emplace("OriginMesh", get_input("OriginMesh"));
        set_output("MeshVel", buildVertexVelocity(resolvedInputs));
    }
};

bool isAetherGeoVertexVelNode(INode const &node) noexcept {
    return dynamic_cast<GeoVertexVel const *>(&node) != nullptr;
}

#ifdef ZENO_AETHER_ENABLE_TASKFLOW
AetherPreparedNodeWork makeAetherGeoVertexVelPreparedWork() {
    return [](AetherPreparedNodeInputs const &inputs) {
        return std::map<std::string, zany>{{"MeshVel", buildVertexVelocity(inputs.values)}};
    };
}
#endif

static int defGeoVertexVel = zeno::defNodeClass<GeoVertexVel>("GeoVertexVel",
    { /* inputs: */ {
        "dt", "TargetMesh", "OriginMesh",
    }, /* outputs: */ {
        "MeshVel",
    }, /* params: */ {
    }, /* category: */ {
        "deprecated",
    }});

} // namespace zeno
