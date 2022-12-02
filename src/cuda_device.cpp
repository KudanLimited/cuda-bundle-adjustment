
#include "cuda_device.h"

#include <cuda_runtime.h>

#include <cstdint>
#include <cstdio>
#include <cstdlib>

namespace cugo
{

static const char* _cudaGetErrorEnum(cudaError_t error) { return cudaGetErrorName(error); }

void CudaDevice::init()
{
    deviceId_ = findCudaDevice();
    CUDA_CHECK(cudaGetDeviceProperties(&deviceProp_, deviceId_));

#ifndef USE_ZERO_COPY
    for (int i = 0; i < streams_.size(); ++i)
    {
        CUDA_CHECK(cudaStreamCreate(&streams_[i]));
    }
    for (int i = 0; i < events_.size(); ++i)
    {
        CUDA_CHECK(cudaEventCreate(&events_[i]));
    }
    CUDA_CHECK(cudaEventCreate(&timeStart));
    CUDA_CHECK(cudaEventCreate(&timeStop));
#else
    // Set flag to enable zero copy access
    cudaSetDeviceFlags(cudaDeviceMapHost);

    // use default stream if using zero copy as copying data to the device
    // async is no longer required.
    for (int i = 0; i < streams_.size(); ++i)
    {
        streams_[i] = 0;
    }
#endif
}

void CudaDevice::destroy()
{
    for (int i = 0; i < streams_.size(); ++i)
    {
        CUDA_CHECK(cudaStreamDestroy(streams_[i]));
    }
    for (int i = 0; i < events_.size(); ++i)
    {
        CUDA_CHECK(cudaEventDestroy(events_[i]));
    }
    CUDA_CHECK(cudaEventDestroy(timeStart));
    CUDA_CHECK(cudaEventDestroy(timeStop));
}

CudaDevice::~CudaDevice() { destroy(); }

int CudaDevice::ConvertSMVer2Cores(int major, int minor)
{
    // Defines for GPU Architecture types (using the SM version to determine
    // the # of cores per SM
    typedef struct
    {
        int SM; // 0xMm (hexidecimal notation), M = SM Major version,
        // and m = SM minor version
        int Cores;
    } sSMtoCores;

    sSMtoCores nGpuArchCoresPerSM[] = {
        {0x30, 192},
        {0x32, 192},
        {0x35, 192},
        {0x37, 192},
        {0x50, 128},
        {0x52, 128},
        {0x53, 128},
        {0x60, 64},
        {0x61, 128},
        {0x62, 128},
        {0x70, 64},
        {0x72, 64},
        {0x75, 64},
        {0x80, 64},
        {0x86, 128},
        {0x87, 128},
        {-1, -1}};

    int index = 0;

    while (nGpuArchCoresPerSM[index].SM != -1)
    {
        if (nGpuArchCoresPerSM[index].SM == ((major << 4) + minor))
        {
            return nGpuArchCoresPerSM[index].Cores;
        }

        index++;
    }

    // If we don't find the values, we default use the previous one
    // to run properly
    printf(
        "MapSMtoCores for SM %d.%d is undefined."
        "  Default to use %d Cores/SM\n",
        major,
        minor,
        nGpuArchCoresPerSM[index - 1].Cores);
    return nGpuArchCoresPerSM[index - 1].Cores;
}

const char* CudaDevice::ConvertSMVer2ArchName(int major, int minor)
{
    // Defines for GPU Architecture types (using the SM version to determine
    // the GPU Arch name)
    typedef struct
    {
        int SM; // 0xMm (hexidecimal notation), M = SM Major version,
        // and m = SM minor version
        const char* name;
    } sSMtoArchName;

    sSMtoArchName nGpuArchNameSM[] = {
        {0x30, "Kepler"},
        {0x32, "Kepler"},
        {0x35, "Kepler"},
        {0x37, "Kepler"},
        {0x50, "Maxwell"},
        {0x52, "Maxwell"},
        {0x53, "Maxwell"},
        {0x60, "Pascal"},
        {0x61, "Pascal"},
        {0x62, "Pascal"},
        {0x70, "Volta"},
        {0x72, "Xavier"},
        {0x75, "Turing"},
        {0x80, "Ampere"},
        {0x86, "Ampere"},
        {-1, "Graphics Device"}};

    int index = 0;

    while (nGpuArchNameSM[index].SM != -1)
    {
        if (nGpuArchNameSM[index].SM == ((major << 4) + minor))
        {
            return nGpuArchNameSM[index].name;
        }

        index++;
    }

    // If we don't find the values, we default use the previous one
    // to run properly
    printf(
        "MapSMtoArchName for SM %d.%d is undefined."
        "  Default to use %s\n",
        major,
        minor,
        nGpuArchNameSM[index - 1].name);
    return nGpuArchNameSM[index - 1].name;
}


int CudaDevice::gpuGetMaxGflopsDeviceId()
{
    int current_device = 0, sm_per_multiproc = 0;
    int max_perf_device = 0;
    int device_count = 0;
    int devices_prohibited = 0;

    uint64_t max_compute_perf = 0;
    CUDA_CHECK(cudaGetDeviceCount(&device_count));

    if (device_count == 0)
    {
        fprintf(
            stderr,
            "gpuGetMaxGflopsDeviceId() CUDA error:"
            " no devices supporting CUDA.\n");
        exit(EXIT_FAILURE);
    }

    // Find the best CUDA capable GPU device
    current_device = 0;

    while (current_device < device_count)
    {
        int computeMode = -1, major = 0, minor = 0;
        CUDA_CHECK(cudaDeviceGetAttribute(&computeMode, cudaDevAttrComputeMode, current_device));
        CUDA_CHECK(
            cudaDeviceGetAttribute(&major, cudaDevAttrComputeCapabilityMajor, current_device));
        CUDA_CHECK(
            cudaDeviceGetAttribute(&minor, cudaDevAttrComputeCapabilityMinor, current_device));

        // If this GPU is not running on Compute Mode prohibited,
        // then we can add it to the list
        if (computeMode != cudaComputeModeProhibited)
        {
            if (major == 9999 && minor == 9999)
            {
                sm_per_multiproc = 1;
            }
            else
            {
                sm_per_multiproc = ConvertSMVer2Cores(major, minor);
            }
            int multiProcessorCount = 0, clockRate = 0;
            CUDA_CHECK(cudaDeviceGetAttribute(
                &multiProcessorCount, cudaDevAttrMultiProcessorCount, current_device));
            cudaError_t result =
                cudaDeviceGetAttribute(&clockRate, cudaDevAttrClockRate, current_device);
            if (result != cudaSuccess)
            {
                // If cudaDevAttrClockRate attribute is not supported we
                // set clockRate as 1, to consider GPU with most SMs and CUDA Cores.
                if (result == cudaErrorInvalidValue)
                {
                    clockRate = 1;
                }
                else
                {
                    fprintf(
                        stderr,
                        "CUDA error at %s:%d code=%d(%s) \n",
                        __FILE__,
                        __LINE__,
                        static_cast<unsigned int>(result),
                        _cudaGetErrorEnum(result));
                    exit(EXIT_FAILURE);
                }
            }
            uint64_t compute_perf = (uint64_t)multiProcessorCount * sm_per_multiproc * clockRate;

            if (compute_perf > max_compute_perf)
            {
                max_compute_perf = compute_perf;
                max_perf_device = current_device;
            }
        }
        else
        {
            devices_prohibited++;
        }

        ++current_device;
    }

    if (devices_prohibited == device_count)
    {
        fprintf(
            stderr,
            "gpuGetMaxGflopsDeviceId() CUDA error:"
            " all devices have compute mode prohibited.\n");
        exit(EXIT_FAILURE);
    }

    return max_perf_device;
}


// Initialization code to find the best CUDA Device
inline int CudaDevice::findCudaDevice()
{
    int devID = 0;

    // pick the device with highest Gflops/s
    devID = gpuGetMaxGflopsDeviceId();
    CUDA_CHECK(cudaSetDevice(devID));
    int major = 0, minor = 0;
    CUDA_CHECK(cudaDeviceGetAttribute(&major, cudaDevAttrComputeCapabilityMajor, devID));
    CUDA_CHECK(cudaDeviceGetAttribute(&minor, cudaDevAttrComputeCapabilityMinor, devID));
    printf(
        "GPU Device %d : \"%s\" with compute capability %d.%d\n\n",
        devID,
        ConvertSMVer2ArchName(major, minor),
        major,
        minor);

    return devID;
}

void CudaDevice::startTimingEvent() const noexcept { CUDA_CHECK(cudaEventRecord(timeStart)); }
double CudaDevice::stopTimingEvent() const noexcept
{
    float timeTaken;
    CUDA_CHECK(cudaEventRecord(timeStop));
    CUDA_CHECK(cudaEventSynchronize(timeStop));
    CUDA_CHECK(cudaEventElapsedTime(&timeTaken, timeStart, timeStop));
    return static_cast<double>(timeTaken);
}

} // namespace cugo