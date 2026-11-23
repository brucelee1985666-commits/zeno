#include <zeno/core/AetherTaskflowExecutor.h>
#include <atomic>
#include <cassert>
#include <stdexcept>
#include <vector>
int main() {
 zeno::AetherTaskflowExecutor executor; std::atomic<int> state{0};
 std::vector<zeno::AetherTaskflowTask> tasks{{"seed",[&]{state.store(1);}}, {"left",[&]{assert(state.load()==1);state.fetch_add(2);}}, {"right",[&]{assert(state.load()==1);state.fetch_add(4);}}, {"join",[&]{assert(state.load()==7);state.store(8);}}};
 std::vector<zeno::AetherTaskflowDependency> edges{{0,1},{0,2},{1,3},{2,3}};
 executor.run(tasks,edges); assert(state.load()==8);
 bool rejected=false; try { executor.run({{"a",[]{}},{"b",[]{}}},{{0,1},{1,0}}); } catch(std::invalid_argument const &) { rejected=true; }
 assert(rejected); return 0;
}
