#include <zeno/core/AetherTaskflowExecutor.h>
#include <taskflow/taskflow.hpp>
#include <algorithm>
#include <queue>
#include <set>
#include <stdexcept>
#include <thread>
#include <vector>
#if TF_VERSION < 301100 || TF_VERSION >= 400000
#error "Aether currently supports Taskflow 3.11.x (C++17); Taskflow 4.x requires C++20"
#endif
namespace zeno {
struct AetherTaskflowExecutor::Impl {
 Impl() : executor(std::max<std::size_t>(1, std::thread::hardware_concurrency())) {}
 tf::Executor executor;
};
AetherTaskflowExecutor::AetherTaskflowExecutor() : impl_(std::make_unique<Impl>()) {}
AetherTaskflowExecutor::~AetherTaskflowExecutor() = default;
AetherTaskflowExecutor::AetherTaskflowExecutor(AetherTaskflowExecutor &&) noexcept = default;
AetherTaskflowExecutor &AetherTaskflowExecutor::operator=(AetherTaskflowExecutor &&) noexcept = default;
void AetherTaskflowExecutor::run(std::vector<AetherTaskflowTask> const &tasks, std::vector<AetherTaskflowDependency> const &dependencies) {
 if (!impl_) throw std::logic_error("AetherTaskflowExecutor is in a moved-from state");
 const std::size_t count = tasks.size();
 for (const auto &task : tasks) if (!task.work) throw std::invalid_argument("Aether Taskflow task has no work callback: " + task.name);
 std::vector<std::vector<std::size_t>> successors(count);
 std::vector<std::size_t> indegree(count, 0);
 std::set<AetherTaskflowDependency> uniqueEdges;
 for (const auto &edge : dependencies) {
  const auto from=edge.first, to=edge.second;
  if (from>=count || to>=count) throw std::out_of_range("Aether Taskflow dependency index is outside the task list");
  if (from==to) throw std::invalid_argument("Aether Taskflow task cannot depend on itself");
  if (!uniqueEdges.insert(edge).second) throw std::invalid_argument("Aether Taskflow dependency is duplicated");
  successors[from].push_back(to); ++indegree[to];
 }
 std::queue<std::size_t> ready;
 for(std::size_t i=0;i<count;++i) if(indegree[i]==0) ready.push(i);
 std::size_t visited=0;
 while(!ready.empty()) { auto current=ready.front(); ready.pop(); ++visited; for(auto next:successors[current]) if(--indegree[next]==0) ready.push(next); }
 if(visited!=count) throw std::invalid_argument("Aether Taskflow dependency graph contains a cycle");
 tf::Taskflow flow; std::vector<tf::Task> handles; handles.reserve(count);
 for(const auto &task:tasks) handles.emplace_back(flow.emplace(task.work).name(task.name));
 for(const auto &edge:dependencies) handles[edge.first].precede(handles[edge.second]);
 impl_->executor.run(flow).get();
}
}
