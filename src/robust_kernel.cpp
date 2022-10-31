#include "robust_kernel.h"

#include "cuda/cuda_block_solver.h"

namespace cugo
{

RobustKernel::RobustKernel()  {}
RobustKernel::~RobustKernel()
{
}

void RobustKernel::create(
    const RobustKernelType type, const Scalar delta)
{
    d_delta_.assign(1, &delta);
    gpu::createRkFunction(type, d_delta_, deviceInfo_);
}

void RobustKernel::setDeviceInfo(const CudaDeviceInfo& deviceInfo) { deviceInfo_ = deviceInfo; }


}