#pragma once

#include "scalar.h"
#include "device_matrix.h"

namespace cugo
{

enum class RobustKernelType
{
    None,
    Cauchy,
    Turkey
};

/**
 * @brief Robust kernel paramters.
 */
struct RobustKernel
{
    RobustKernel() : type(RobustKernelType::None), delta(1.0) {}
    RobustKernel(const RobustKernelType t, Scalar d) : type(t), delta(d) {}

    RobustKernelType type;
    Scalar delta;
    GpuVec<Scalar> d_delta;
};

} // namespace cugo