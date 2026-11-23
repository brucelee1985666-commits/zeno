#include <zeno/core/AetherTaskflowExecutor.h>

#include <atomic>
#include <cassert>
#include <stdexcept>
#include <vector>

int main() {
    zeno::AetherTaskflowExecutor executor;
    std::atomic<unsigned> state{0};
    std::atomic<bool> orderingViolation{false};

    // The DAG intentionally allows left/right to execute concurrently.
    // Check only the declared happens-before edges, never sibling ordering.
    std::vector<zeno::AetherTaskflowTask> tasks{
        {"seed", [&] {
            state.store(1u, std::memory_order_release);
        }},
        {"left", [&] {
            if ((state.load(std::memory_order_acquire) & 1u) == 0u)
                orderingViolation.store(true, std::memory_order_relaxed);
            state.fetch_or(2u, std::memory_order_acq_rel);
        }},
        {"right", [&] {
            if ((state.load(std::memory_order_acquire) & 1u) == 0u)
                orderingViolation.store(true, std::memory_order_relaxed);
            state.fetch_or(4u, std::memory_order_acq_rel);
        }},
        {"join", [&] {
            if ((state.load(std::memory_order_acquire) & 6u) != 6u)
                orderingViolation.store(true, std::memory_order_relaxed);
            state.store(8u, std::memory_order_release);
        }}
    };

    const std::vector<zeno::AetherTaskflowDependency> edges{
        {0, 1}, {0, 2}, {1, 3}, {2, 3}
    };
    executor.run(tasks, edges);

    assert(!orderingViolation.load(std::memory_order_relaxed));
    assert(state.load(std::memory_order_acquire) == 8u);

    bool rejectedCycle = false;
    try {
        executor.run({{"a", [] {}}, {"b", [] {}}}, {{0, 1}, {1, 0}});
    } catch (std::invalid_argument const &) {
        rejectedCycle = true;
    }
    assert(rejectedCycle);
    return 0;
}
