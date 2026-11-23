#include <taskflow/taskflow.hpp>

#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <vector>

namespace {
std::uint64_t work(std::uint64_t seed, std::uint64_t iterations) {
    std::uint64_t value = seed;
    for (std::uint64_t i = 0; i < iterations; ++i) {
        value ^= value << 13;
        value ^= value >> 7;
        value ^= value << 17;
        value += i + 0x9e3779b97f4a7c15ULL;
    }
    return value;
}
}

int main(int argc, char **argv) {
    const std::size_t taskCount = argc > 1
        ? static_cast<std::size_t>(std::strtoull(argv[1], nullptr, 10)) : 8;
    const std::uint64_t iterations = argc > 2
        ? std::strtoull(argv[2], nullptr, 10) : 2000000ULL;
    if (taskCount == 0 || iterations == 0) return 2;

    std::vector<std::uint64_t> serial(taskCount), parallel(taskCount);
    const auto serialStart = std::chrono::steady_clock::now();
    for (std::size_t i = 0; i < taskCount; ++i)
        serial[i] = work(i + 1, iterations);
    const auto serialEnd = std::chrono::steady_clock::now();

    tf::Executor executor;
    tf::Taskflow flow("aether-independent-cpu-tasks");
    for (std::size_t i = 0; i < taskCount; ++i)
        flow.emplace([&, i] { parallel[i] = work(i + 1, iterations); }).name("cpu-" + std::to_string(i));
    const auto parallelStart = std::chrono::steady_clock::now();
    executor.run(flow).get();
    const auto parallelEnd = std::chrono::steady_clock::now();

    if (serial != parallel) return 3;
    const auto serialMs = std::chrono::duration<double, std::milli>(serialEnd - serialStart).count();
    const auto parallelMs = std::chrono::duration<double, std::milli>(parallelEnd - parallelStart).count();
    std::cout << "tasks,iterations_per_task,serial_ms,taskflow_ms,speedup,checksum\n";
    std::cout << taskCount << ',' << iterations << ',' << serialMs << ','
              << parallelMs << ',' << (parallelMs > 0.0 ? serialMs / parallelMs : 0.0)
              << ',' << parallel.front() << '\n';
    return 0;
}
