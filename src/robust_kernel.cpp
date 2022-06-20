#include "robust_kernel.h"

namespace cugo
{

void RobustKernelCauchy::robustify(const Scalar e2, maths::Vec3& rho)
{
    Scalar dsqr = _delta * _delta;
    Scalar dsqrReci = 1.0 / dsqr;
    Scalar aux = dsqrReci * e2 + 1.0;
    rho[0] = dsqr * log(aux);
    rho[1] = 1.0 / aux;
    rho[2] = -dsqrReci * std::pow(rho[1], 2);
}

} // namespace cugo