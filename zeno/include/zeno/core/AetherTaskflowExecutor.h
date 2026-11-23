#pragma once
#include <zeno/utils/api.h>
#include <cstddef>
#include <functional>
#include <memory>
#include <string>
#include <utility>
#include <vector>
#ifdef ZENO_AETHER_ENABLE_TASKFLOW
namespace zeno {
// Explicit standalone task graph. Not an INode adapter: current Zeno nodes may
// recurse through inputs and mutate shared graph/session state.
struct AetherTaskflowTask { std::string name; std::function<void()> work; };
using AetherTaskflowDependency = std::pair<std::size_t, std::size_t>;
// Optional CPU backend powered by Taskflow 3.11.x (C++17). Callbacks may run
// concurrently; callers must provide thread-safe work and explicit dependencies.
class ZENO_API AetherTaskflowExecutor {
public:
 AetherTaskflowExecutor(); ~AetherTaskflowExecutor();
 AetherTaskflowExecutor(AetherTaskflowExecutor const &) = delete;
 AetherTaskflowExecutor &operator=(AetherTaskflowExecutor const &) = delete;
 AetherTaskflowExecutor(AetherTaskflowExecutor &&) noexcept;
 AetherTaskflowExecutor &operator=(AetherTaskflowExecutor &&) noexcept;
 void run(std::vector<AetherTaskflowTask> const &, std::vector<AetherTaskflowDependency> const &);
private: struct Impl; std::unique_ptr<Impl> impl_;
};
}
#endif // ZENO_AETHER_ENABLE_TASKFLOW
