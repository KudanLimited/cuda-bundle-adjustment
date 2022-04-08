#ifndef _MATHS_H
#define _MATHS_H

#include "scalar.h"

#include <Eigen/Core>
#include <Eigen/Geometry>

namespace cuba
{

namespace maths
{

template <class T, int N>
using Vec = Eigen::Matrix<T, N, 1>;

template <typename Scalar>
using Vec2 = Vec<Scalar, 2>;
template <typename Scalar>
using Vec3 = Vec<Scalar, 3>;
template <typename Scalar>
using Vec4 = Vec<Scalar, 4>;

using Vec2d = Vec2<double>;
using Vec3d = Vec3<double>;
using Vec4d = Vec4<double>;

using Vec2f = Vec2<float>;
using Vec3f = Vec3<float>;
using Vec4f = Vec4<float>;

template <typename Scalar>
struct Se3
{
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    
	Se3() = default;
    ~Se3() {}
	Se3(const Eigen::Quaternion<Scalar>& r, const Vec3<Scalar>& t): r(r), t(t) {}

	Eigen::Quaternion<Scalar> r;
	Vec3<Scalar> t;
};

using Se3F = Se3<float>;
using Se3D = Se3<double>;

} // namespace maths
} // namespace cuba

#endif // _MATHS_H