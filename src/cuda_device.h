#pragma once

#include "macro.h"

#include <cuda_runtime.h>

#include <array>
#include <cassert>

namespace cugo
{

 /**
 * To aid in the easier passing of streams and events to functions that
 * call kernels.
 */
struct CudaDeviceInfo
{
    cudaStream_t stream;
    cudaEvent_t event;
};

class CudaDevice
{
public:
    static constexpr int MaxStreamCount = 8;
    static constexpr int MaxEventCount = 8;

    using StreamContainer = std::array<cudaStream_t, MaxStreamCount>;
    using EventContainer = std::array<cudaEvent_t, MaxStreamCount>;

    CudaDevice() { init(); }
    ~CudaDevice();

    /**
     * @brief Initialise the CUDA backend. This will check that the system
     * has the required compute capability and initialise the cuda streams
     *
     */
    void init();

    /**
    * @brief Destroy the cuda backend.
    */
    void destroy();

    /**
     * @brief Find the 'best' cuda device on the system. This is
     * determined, in the instance of multiple GPUs being detected,
     * by the GPU with the maximum GFLOPS.
     *
     */
    int findCudaDevice();

    inline StreamContainer& getStreams() noexcept { return streams_; }

    inline cudaStream_t getStream(uint8_t idx) const noexcept
    {
        assert(idx < streams_.size());
        return streams_[idx];
    }

    inline CudaDeviceInfo getStreamAndEvent(uint8_t idx) const noexcept
    {
        assert(idx < streams_.size());
        assert(idx < events_.size());
        return {streams_[idx], events_[idx]};
    }

    inline cudaEvent_t getEvent(uint8_t idx) const noexcept
    {
        assert(idx < events_.size());
        return events_[idx];
    }


private:
    int ConvertSMVer2Cores(int major, int minor);

    const char* ConvertSMVer2ArchName(int major, int minor);

    /**
     * @brief Detect the best GPU on the system via maximum GFLOPS
     */
    int gpuGetMaxGflopsDeviceId();

private:
    /// cuda streams
    StreamContainer streams_;

    // cuda events - created in advance to avoid overhead of creating
    // before kernel calls.
    EventContainer events_;

    /// the GPU device id that will be used
    int deviceId_;

    /// Properties of the GPU that will be used.
    cudaDeviceProp deviceProp_;
};

} // namespace cugo