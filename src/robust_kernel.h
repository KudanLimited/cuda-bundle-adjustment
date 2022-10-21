#pragma once

#include "scalar.h"
#include "device_matrix.h"

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

private:

    GpuVec<Scalar> d_delta_;
    bool isInit_;
};

} // namespace cugo