#include <zeno/zeno.h>
#include <zeno/MeshObject.h>
#include <zeno/NumericObject.h>
#include <zeno/AetherOldzenbasePreparedWork.h>
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtx/transform.hpp>
#include <cmath>
#include <stdexcept>

namespace zeno {
namespace {

glm::vec3 readOptionalVector(
    std::map<std::string, zany> const &inputs,
    char const *name,
    glm::vec3 const &fallback) {
  const auto it = inputs.find(name);
  if (it == inputs.end() || !it->second)
    return fallback;
  const auto numeric = std::dynamic_pointer_cast<NumericObject>(it->second);
  if (!numeric)
    throw std::invalid_argument(std::string("TransformMesh input '") + name + "' is not NumericObject");
  const auto value = numeric->get<zeno::vec3f>();
  return glm::vec3(value[0], value[1], value[2]);
}

std::shared_ptr<MeshObject> transformMesh(
    std::map<std::string, zany> const &inputs) {
  const auto meshIt = inputs.find("mesh");
  if (meshIt == inputs.end() || !meshIt->second)
    throw std::invalid_argument("TransformMesh requires a non-null mesh input");
  const auto mesh = std::dynamic_pointer_cast<MeshObject>(meshIt->second);
  if (!mesh)
    throw std::invalid_argument("TransformMesh input 'mesh' is not a MeshObject");

  const auto translate = readOptionalVector(inputs, "translate", glm::vec3(0.0f));
  const auto rotate = readOptionalVector(inputs, "rotate", glm::vec3(0.0f));
  const auto scaling = readOptionalVector(inputs, "scaling", glm::vec3(1.0f));

  const glm::mat4 matTrans = glm::translate(translate);
  const glm::mat4 matRotx = glm::rotate(rotate.x, glm::vec3(1, 0, 0));
  const glm::mat4 matRoty = glm::rotate(rotate.y, glm::vec3(0, 1, 0));
  const glm::mat4 matRotz = glm::rotate(rotate.z, glm::vec3(0, 0, 1));
  const glm::mat4 matScal = glm::scale(scaling);
  const glm::mat4 matrix = matRotz * matRoty * matRotx * matScal * matTrans;

  const glm::mat3 linear(matrix);
  const float determinant = glm::determinant(linear);
  if (!mesh->normals.empty() && std::abs(determinant) <= 1.0e-8f)
    throw std::invalid_argument("TransformMesh cannot transform normals with singular scaling");
  const glm::mat3 normalMatrix = mesh->normals.empty()
      ? glm::mat3(1.0f)
      : glm::transpose(glm::inverse(linear));

  auto output = std::make_shared<MeshObject>();
  output->vertices.reserve(mesh->vertices.size());
  output->uvs = mesh->uvs;
  output->normals.reserve(mesh->normals.size());

  for (const auto &vertex : mesh->vertices) {
    const glm::vec4 transformed = matrix * glm::vec4(vertex, 1.0f);
    if (!std::isfinite(transformed.w) || std::abs(transformed.w) <= 1.0e-8f)
      throw std::runtime_error("TransformMesh produced an invalid homogeneous coordinate");
    const glm::vec3 position = glm::vec3(transformed) / transformed.w;
    if (!std::isfinite(position.x) || !std::isfinite(position.y) || !std::isfinite(position.z))
      throw std::runtime_error("TransformMesh produced a non-finite vertex");
    output->vertices.push_back(position);
  }

  for (const auto &normal : mesh->normals) {
    const glm::vec3 transformed = normalMatrix * normal;
    const float lengthSquared = glm::dot(transformed, transformed);
    if (lengthSquared <= 1.0e-16f)
      output->normals.emplace_back(0.0f);
    else
      output->normals.push_back(transformed / std::sqrt(lengthSquared));
  }
  return output;
}

} // namespace

struct TransformMesh : zeno::INode {
  AetherNodeExecutionCapabilities getAetherExecutionCapabilities() const override {
    AetherNodeExecutionCapabilities capabilities;
    capabilities.cpuThreadSafe = true;
    capabilities.sideEffectFree = true;
    capabilities.deterministic = true;
    return capabilities;
  }

  void apply() override {
    std::map<std::string, zany> resolvedInputs;
    resolvedInputs.emplace("mesh", get_input("mesh"));
    for (const char *socket : {"translate", "rotate", "scaling"}) {
      if (has_input(socket))
        resolvedInputs.emplace(socket, get_input(socket));
    }
    set_output("mesh", transformMesh(resolvedInputs));
  }
};

bool isAetherTransformMeshNode(INode const &node) noexcept {
  return dynamic_cast<TransformMesh const *>(&node) != nullptr;
}

#ifdef ZENO_AETHER_ENABLE_TASKFLOW
AetherPreparedNodeWork makeAetherTransformMeshPreparedWork() {
  return [](AetherPreparedNodeInputs const &inputs) {
    return std::map<std::string, zany>{{"mesh", transformMesh(inputs.values)}};
  };
}
#endif

static int defTransformMesh = zeno::defNodeClass<TransformMesh>("TransformMesh",
    { /* inputs: */ {
    "mesh",
    "translate",
    "rotate",
    "scaling",
    }, /* outputs: */ {
    "mesh",
    }, /* params: */ {
    }, /* category: */ {
    "deprecated",
    }});

} // namespace zeno
