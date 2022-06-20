#pragma once

#include "macro.h"
#include "maths.h"
#include "scalar.h"

namespace cugo
{

class BaseRobustKernel
{
public:
    BaseRobustKernel(const Scalar delta) : delta_(delta) {};
    BaseRobustKernel() = default;

    virtual ~BaseRobustKernel() {};

    void setDelta(const Scalar delta) { delta_ = delta; }

    virtual void robustify(const Scalar chi, maths::Vec3<Scalar>& rhoOut) = 0;

protected:
    Scalar delta_;
};

class CUGO_API RobustKernelCauchy : public BaseRobustKernel
{
public:
    RobustKernelCauchy(const Scalar delta) : BaseRobustKernel(delta) {}
    RobustKernelCauchy() = default;

    ~RobustKernelCauchy() {}

    void robustify(const Scalar chi, maths::Vec3<Scalar>& rhoOut) override;
};

} // namespace cugo