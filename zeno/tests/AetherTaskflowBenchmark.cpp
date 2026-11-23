#include <taskflow/taskflow.hpp>

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <string>
#include <vector>

namespace {
using Clock = std::chrono::steady_clock;
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
double milliseconds(Clock::time_point start, Clock::time_point end) {
    return std::chrono::duration<double, std::milli>(end - start).count();
}
}

int main(int argc, char **argv) {
    const std::size_t taskCount = argc > 1
        ? static_cast<std::size_t>(std::strtoull(argv[1], nullptr, 10)) : 8;
    const std::uint64_t iterations = argc > 2
        ? std::strtoull(argv[2], nullptr, 10) : 2000000ULL;
    const std::size_t trials = argc > 3
        ? static_cast<std::size_t>(std::strtoull(argv[3], nullptr, 10)) : 5;
    if (taskCount == 0 || iterations == 0 || trials == 0) return 2;

    std::vector<std::uint64_t> serial(taskCount), parallel(taskCount);
    tf::Executor executor;
    tf::Taskflow flow("aether-independent-cpu-tasks");
    for (std::size_t i = 0; i < taskCount; ++i)
        flow.emplace([&, i] { parallel[i] = work(i + 1, iterations); }).name("cpu-" + std::to_string(i));

    // Warm-up both paths before collecting measurements.
    for (std::size_t i = 0; i < taskCount; ++i) serial[i] = work(i + 1, iterations);
    executor.run(flow).get();

    std::vector<double> serialSamples, taskflowSamples;
    serialSamples.reserve(trials);
    taskflowSamples.reserve(trials);
    for (std::size_t trial = 0; trial < trials; ++trial) {
        auto start = Clock::now();
        for (std::size_t i = 0; i < taskCount; ++i)
            serial[i] = work(i + 1, iterations);
        serialSamples.push_back(milliseconds(start, Clock::now()));

        start = Clock::now();
        executor.run(flow).get();
        taskflowSamples.push_back(milliseconds(start, Clock::now()));
        if (serial != parallel) return 3;
    }

    std::sort(serialSamples.begin(), serialSamples.end());
    std::sort(taskflowSamples.begin(), taskflowSamples.end());
    const double serialMedian = serialSamples[trials / 2];
    const double taskflowMedian = taskflowSamples[trials / 2];
    std::cout << "tasks,iterations_per_task,trials,serial_median_ms,taskflow_median_ms,median_speedup,checksum\n";
    std::cout << taskCount << ',' << iterations << ',' << trials << ','
              << serialMedian << ',' << taskflowMedian << ','
              << (taskflowMedian > 0.0 ? serialMedian / taskflowMedian : 0.0)
              << ',' << parallel.front() << '\n';
    return 0;
}
