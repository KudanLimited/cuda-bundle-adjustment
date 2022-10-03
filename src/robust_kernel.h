#pragma once

#include "scalar.h"
#include "device_matrix.h"

namespace cugo
{

enum class RobustKernelType
{
    /// Use when a computeError/computeQuadratic doesn't support a
    /// a robust kernel implementation
    Uninitialised,
    /// Use when application of a robust kernel isn't required but
    /// is used by the kernel
    None,
    Cauchy,
    Turkey
};

/**
 * @brief Robust kernel paramters.
 */
struct RobustKernel
{
    RobustKernel() : type_(RobustKernelType::Uninitialised), delta_(1.0) {}
    
    RobustKernel(const RobustKernelType t, Scalar d) : type_(t), delta_(d) 
    {
        d_delta_.assign(1, &delta_);
    }

    void setKernel(const RobustKernelType type, Scalar delta) 
    { 
        type_ = type;
        delta_ = delta;
        d_delta_.assign(1, &delta_);
    }

    const RobustKernelType type() const noexcept { return type_; }
    Scalar const * data() const noexcept { return d_delta_.data(); }

private:

    RobustKernelType type_;
    Scalar delta_;
    GpuVec<Scalar> d_delta_;
};

} // namespace cugo