#pragma once

#include <cuda.h>

#if defined(ZENO_AETHER_ENABLE_PROFILER)
#include <zeno/core/NodeExecutionProfiler.h>
#endif

namespace zeno {

// Measures only the work bracketed on the supplied stream. The caller should
// invoke collect() after its existing synchronization point; this class never
// introduces a device-wide or stream synchronization of its own.
class CudaKernelProfileScope {
public:
    CudaKernelProfileScope(const char *kernelName, CUstream launchStream) noexcept
#if defined(ZENO_AETHER_ENABLE_PROFILER)
        : name(kernelName), stream(launchStream) {
        if (cuEventCreate(&startEvent, CU_EVENT_DEFAULT) != CUDA_SUCCESS)
            return;
        if (cuEventRecord(startEvent, stream) != CUDA_SUCCESS) {
            cleanup();
            return;
        }
        active = true;
    }
#else
    {
        (void)kernelName;
        (void)launchStream;
    }
#endif

    CudaKernelProfileScope(CudaKernelProfileScope const &) = delete;
    CudaKernelProfileScope &operator=(CudaKernelProfileScope const &) = delete;

    ~CudaKernelProfileScope() { cleanup(); }

    void finish() noexcept {
#if defined(ZENO_AETHER_ENABLE_PROFILER)
        if (!active || finished)
            return;
        if (cuEventCreate(&stopEvent, CU_EVENT_DEFAULT) != CUDA_SUCCESS ||
            cuEventRecord(stopEvent, stream) != CUDA_SUCCESS) {
            cleanup();
            return;
        }
        finished = true;
#endif
    }

    void collect() noexcept {
#if defined(ZENO_AETHER_ENABLE_PROFILER)
        if (!active || !finished)
            return;
        float elapsedMilliseconds = 0.0f;
        if (cuEventElapsedTime(&elapsedMilliseconds, startEvent, stopEvent) == CUDA_SUCCESS)
            recordGpuKernelExecution(name, static_cast<double>(elapsedMilliseconds));
        cleanup();
#endif
    }

private:
    void cleanup() noexcept {
#if defined(ZENO_AETHER_ENABLE_PROFILER)
        if (startEvent) {
            cuEventDestroy(startEvent);
            startEvent = nullptr;
        }
        if (stopEvent) {
            cuEventDestroy(stopEvent);
            stopEvent = nullptr;
        }
        active = false;
        finished = false;
#endif
    }

#if defined(ZENO_AETHER_ENABLE_PROFILER)
    const char *name;
    CUstream stream;
    CUevent startEvent = nullptr;
    CUevent stopEvent = nullptr;
    bool active = false;
    bool finished = false;
#endif
};

} // namespace zeno
