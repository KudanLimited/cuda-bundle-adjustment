#pragma once

#include "scalar.h"
#include "device_matrix.h"
#include "cuda_device.h"

#include <cassert>

namespace cugo
{

enum class RobustKernelType
{
    None,
    Cauchy,
    Tukey
};

/**
 * @brief Robust kernel paramters.
 */
class RobustKernel
{
public:
    RobustKernel();
    ~RobustKernel();

    void create(const RobustKernelType type, const Scalar delta);
    void setDeviceInfo(const CudaDeviceInfo& deviceInfo);

private:
    CudaDeviceInfo deviceInfo_;

    GpuVec<Scalar> d_delta_;
};

} // namespace cugo