#pragma once

#include "macro.h"

#include <array>
#include <cuda_runtime.h>

namespace cugo
{

class CudaDevice
{
public:

    static constexpr int MaxStreamCount = 8;
    using StreamContainer = std::array<cudaStream_t, MaxStreamCount>;

    CudaDevice() { init(); }

	/**
     * @brief Initialise the CUDA backend. This will check that the system
     * has the required compute capability and initialise the cuda streams
     *
     */
    void init();

	/**
	* @brief Find the 'best' cuda device on the system. This is 
	* determined, in the instance of multiple GPUs being detected,
	* by the GPU with the maximum GFLOPS.
	*
	*/ 
    int findCudaDevice();

    StreamContainer& getStreams() noexcept { return streams_; }

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

    /// the GPU device id that will be used
    int deviceId_;

    /// Properties of the GPU that will be used.
    cudaDeviceProp deviceProp_;
};

}