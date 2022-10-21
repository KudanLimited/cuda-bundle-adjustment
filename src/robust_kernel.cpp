#include "robust_kernel.h"

#include "cuda/cuda_block_solver.h"

namespace cugo
{

RobustKernel::RobustKernel() : isInit_(false) {}
RobustKernel::~RobustKernel()
{
    if (isInit_)
    {
        gpu::deleteRkFunction();
    }
}

void RobustKernel::create(const RobustKernelType type, const Scalar delta)
{
    d_delta_.assign(1, &delta);

    if (isInit_)
    {
        // delete the old virtual function before creating a new one
        gpu::deleteRkFunction();
    }
    gpu::createRkFunction(type, d_delta_);
    isInit_ = true;
}


}