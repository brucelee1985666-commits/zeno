#include <zeno/zeno.h>
#include <zeno/ParticlesObject.h>
#include <zeno/PrimitiveObject.h>
#include <zeno/AetherOldzenbasePreparedWork.h>

#include <stdexcept>

namespace zeno {
namespace {

std::shared_ptr<PrimitiveObject> buildPrimitiveFromParticles(
    std::map<std::string, zany> const &inputs) {
    const auto particlesIt = inputs.find("pars");
    if (particlesIt == inputs.end() || !particlesIt->second)
        throw std::invalid_argument("ParticlesToPrimitive requires a non-null pars input");
    const auto particles = std::dynamic_pointer_cast<ParticlesObject>(particlesIt->second);
    if (!particles)
        throw std::invalid_argument("ParticlesToPrimitive input 'pars' is not a ParticlesObject");
    if (particles->pos.size() != particles->vel.size())
        throw std::invalid_argument("ParticlesToPrimitive requires matching position and velocity counts");

    auto result = std::make_shared<PrimitiveObject>();
    result->add_attr<zeno::vec3f>("pos");
    result->add_attr<zeno::vec3f>("vel", zeno::vec3f(0.0f));
    result->resize(particles->pos.size());
    for (std::size_t i = 0; i < particles->pos.size(); ++i) {
        const auto &position = particles->pos[i];
        const auto &velocity = particles->vel[i];
        result->attr<zeno::vec3f>("pos")[i] =
            zeno::vec3f(position.x, position.y, position.z);
        result->attr<zeno::vec3f>("vel")[i] =
            zeno::vec3f(velocity.x, velocity.y, velocity.z);
    }
    return result;
}

} // namespace

struct ParticlesToPrimitive : zeno::INode {
    AetherNodeExecutionCapabilities getAetherExecutionCapabilities() const override {
        AetherNodeExecutionCapabilities capabilities;
        capabilities.cpuThreadSafe = true;
        capabilities.sideEffectFree = true;
        capabilities.deterministic = true;
        return capabilities;
    }

    void apply() override {
        std::map<std::string, zany> resolvedInputs{{"pars", get_input("pars")}};
        set_output("prim", buildPrimitiveFromParticles(resolvedInputs));
    }
};

bool isAetherParticlesToPrimitiveNode(INode const &node) noexcept {
    return dynamic_cast<ParticlesToPrimitive const *>(&node) != nullptr;
}

#ifdef ZENO_AETHER_ENABLE_TASKFLOW
AetherPreparedNodeWork makeAetherParticlesToPrimitivePreparedWork() {
    return [](AetherPreparedNodeInputs const &inputs) {
        return std::map<std::string, zany>{{"prim", buildPrimitiveFromParticles(inputs.values)}};
    };
}
#endif

static int defParticlesToPrimitive = zeno::defNodeClass<ParticlesToPrimitive>("ParticlesToPrimitive",
    { /* inputs: */ {
        "pars",
    }, /* outputs: */ {
        "prim",
    }, /* params: */ {
    }, /* category: */ {
        "deprecated",
    }});

} // namespace zeno
