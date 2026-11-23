#include <zeno/core/AetherTaskflowExecutor.h>
#include <atomic>
#include <cassert>
#include <stdexcept>
#include <vector>

int main() {
    zeno::AetherTaskflowExecutor executor;
    std::atomic<int> state{0};
    std::vector<zeno::AetherTaskflowTask> tasks{
        {"seed", [&] { state.store(1, std::memory_order_release); }},
        {"left", [&] {
            assert((state.load(std::memory_order_acquire) & 1) != 0);
            state.fetch_or(2, std::memory_order_acq_rel);
        }},
        {"right", [&] {
            assert((state.load(std::memory_order_acquire) & 1) != 0);
            state.fetch_or(4, std::memory_order_acq_rel);
        }},
        {"join", [&] {
            assert((state.load(std::memory_order_acquire) & 6) == 6);
            state.store(8, std::memory_order_release);
        }}
    };
    std::vector<zeno::AetherTaskflowDependency> edges{{0, 1}, {0, 2}, {1, 3}, {2, 3}};
    executor.run(tasks, edges);
    assert(state.load(std::memory_order_acquire) == 8);

    bool rejected = false;
    try {
        executor.run({{"a", [] {}}, {"b", [] {}}}, {{0, 1}, {1, 0}});
    } catch (std::invalid_argument const &) {
        rejected = true;
    }
    assert(rejected);
    return 0;
}
