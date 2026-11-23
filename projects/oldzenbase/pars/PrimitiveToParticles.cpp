#include <zeno/zeno.h>
#include <zeno/ParticlesObject.h>
#include <zeno/PrimitiveObject.h>
#include <zeno/AetherOldzenbasePreparedWork.h>

#include <stdexcept>

namespace zeno {
namespace {

std::shared_ptr<ParticlesObject> buildParticlesFromPrimitive(
    std::map<std::string, zany> const &inputs) {
    const auto primitiveIt = inputs.find("prim");
    if (primitiveIt == inputs.end() || !primitiveIt->second)
        throw std::invalid_argument("PrimitiveToParticles requires a non-null prim input");
    const auto primitive = std::dynamic_pointer_cast<PrimitiveObject>(primitiveIt->second);
    if (!primitive)
        throw std::invalid_argument("PrimitiveToParticles input 'prim' is not a PrimitiveObject");

    const auto count = primitive->size();
    const bool hasVelocity = primitive->has_attr("vel");
    if (hasVelocity && !primitive->attr_is<zeno::vec3f>("vel"))
        throw std::invalid_argument("PrimitiveToParticles 'vel' attribute must be vec3f");

    const std::vector<zeno::vec3f> *velocities = nullptr;
    if (hasVelocity) {
        velocities = &primitive->attr<zeno::vec3f>("vel");
        if (velocities->size() != count)
            throw std::invalid_argument("PrimitiveToParticles 'vel' attribute size must match positions");
    }

    const auto &positions = primitive->attr<zeno::vec3f>("pos");
    auto result = std::make_shared<ParticlesObject>();
    result->pos.resize(count);
    result->vel.resize(count, glm::vec3(0.0f));
    for (std::size_t i = 0; i < count; ++i) {
        result->pos[i] = glm::vec3(positions[i][0], positions[i][1], positions[i][2]);
        if (velocities) {
            const auto &velocity = (*velocities)[i];
            result->vel[i] = glm::vec3(velocity[0], velocity[1], velocity[2]);
        }
    }
    return result;
}

} // namespace

struct PrimitiveToParticles : zeno::INode {
    AetherNodeExecutionCapabilities getAetherExecutionCapabilities() const override {
        AetherNodeExecutionCapabilities capabilities;
        capabilities.cpuThreadSafe = true;
        capabilities.sideEffectFree = true;
        capabilities.deterministic = true;
        return capabilities;
    }

    void apply() override {
        std::map<std::string, zany> resolvedInputs{{"prim", get_input("prim")}};
        set_output("pars", buildParticlesFromPrimitive(resolvedInputs));
    }
};

bool isAetherPrimitiveToParticlesNode(INode const &node) noexcept {
    return dynamic_cast<PrimitiveToParticles const *>(&node) != nullptr;
}

#ifdef ZENO_AETHER_ENABLE_TASKFLOW
AetherPreparedNodeWork makeAetherPrimitiveToParticlesPreparedWork() {
    return [](AetherPreparedNodeInputs const &inputs) {
        return std::map<std::string, zany>{{"pars", buildParticlesFromPrimitive(inputs.values)}};
    };
}
#endif

static int defPrimitiveToParticles = zeno::defNodeClass<PrimitiveToParticles>("PrimitiveToParticles",
    { /* inputs: */ {
        "prim",
    }, /* outputs: */ {
        "pars",
    }, /* params: */ {
    }, /* category: */ {
        "deprecated",
    }});

} // namespace zeno
