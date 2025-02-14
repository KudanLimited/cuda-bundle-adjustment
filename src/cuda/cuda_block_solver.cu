/*
Copyright 2020 Fixstars Corporation

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

http ://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
*/

#include "cuda_block_solver.h"

#include <cooperative_groups.h>
#include <cooperative_groups/reduce.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <thrust/device_ptr.h>
#include <thrust/gather.h>
#include <thrust/scan.h>
#include <thrust/sort.h>

#include <algorithm>

namespace cg = cooperative_groups;

namespace cugo
{
namespace gpu
{

////////////////////////////////////////////////////////////////////////////////////
// Type definitions
////////////////////////////////////////////////////////////////////////////////////
struct LessRowId
{
    __device__ bool operator()(const Vec3i& lhs, const Vec3i& rhs) const
    {
        if (lhs[0] == rhs[0])
        {
            return lhs[1] < rhs[1];
        }
        return lhs[0] < rhs[0];
    }
};

struct LessColId
{
    __device__ bool operator()(const Vec3i& lhs, const Vec3i& rhs) const
    {
        if (lhs[1] == rhs[1])
        {
            return lhs[0] < rhs[0];
        }
        return lhs[1] < rhs[1];
    }
};

template <typename T, int ROWS, int COLS>
struct MatView
{
    __device__ inline T& operator()(int i, int j) { return data[j * ROWS + i]; }
    __device__ inline MatView(T* data) : data(data) {}
    T* data;
};

template <typename T, int ROWS, int COLS>
struct ConstMatView
{
    __device__ inline T operator()(int i, int j) const { return data[j * ROWS + i]; }
    __device__ inline ConstMatView(const T* data) : data(data) {}
    const T* data;
};

template <typename T, int ROWS, int COLS>
struct Matx
{
    using View = MatView<T, ROWS, COLS>;
    using ConstView = ConstMatView<T, ROWS, COLS>;
    __device__ inline T& operator()(int i, int j) { return data[j * ROWS + i]; }
    __device__ inline T operator()(int i, int j) const { return data[j * ROWS + i]; }

    __device__ inline operator View() { return View(data); }
    __device__ inline operator ConstView() const { return ConstView(data); }
    T data[ROWS * COLS];
};

using MatView2x3d = MatView<Scalar, 2, 3>;
using MatView2x6d = MatView<Scalar, 2, 6>;
using MatView3x3d = MatView<Scalar, 3, 3>;
using MatView3x6d = MatView<Scalar, 3, 6>;
using ConstMatView3x3d = ConstMatView<Scalar, 3, 3>;

struct CameraParamView
{
    __device__ inline CameraParamView(const Scalar* data) : data(data) {}
    __device__ inline CameraParamView(const Vec5d& camera) : data(camera.data) {}
    __device__ inline Scalar fx() const { return data[0]; }
    __device__ inline Scalar fy() const { return data[1]; }
    __device__ inline Scalar cx() const { return data[2]; }
    __device__ inline Scalar cy() const { return data[3]; }
    __device__ inline Scalar bf() const { return data[4]; }

    const Scalar* data;
};

////////////////////////////////////////////////////////////////////////////////////
// Host functions
////////////////////////////////////////////////////////////////////////////////////
static int divUp(int total, int grain) { return (total + grain - 1) / grain; }

////////////////////////////////////////////////////////////////////////////////////
// Device functions (template matrix and verctor operation)
////////////////////////////////////////////////////////////////////////////////////

// assignment operations
using AssignOP = void (*)(Scalar*, Scalar);
__device__ inline void ASSIGN(Scalar* address, Scalar value) { *address = value; }
__device__ inline void ACCUM(Scalar* address, Scalar value) { *address += value; }
__device__ inline void DEACCUM(Scalar* address, Scalar value) { *address -= value; }
__device__ inline void ACCUM_ATOMIC(Scalar* address, Scalar value) { atomicAdd(address, value); }
__device__ inline void DEACCUM_ATOMIC(Scalar* address, Scalar value) { atomicAdd(address, -value); }

// recursive dot product for inline expansion
template <int N>
__device__ inline Scalar dot_(const Scalar* __restrict__ a, const Scalar* __restrict__ b)
{
    return dot_<N - 1>(a, b) + a[N - 1] * b[N - 1];
}

template <>
__device__ inline Scalar dot_<1>(const Scalar* __restrict__ a, const Scalar* __restrict__ b)
{
    return a[0] * b[0];
}

// recursive dot product for inline expansion (strided access pattern)
template <int N, int S1, int S2>
__device__ inline Scalar dot_stride_(const Scalar* __restrict__ a, const Scalar* __restrict__ b)
{
    return dot_stride_<N - 1, S1, S2>(a, b) + a[S1 * (N - 1)] * b[S2 * (N - 1)];
}

template <>
__device__ inline Scalar
dot_stride_<1, PDIM, 1>(const Scalar* __restrict__ a, const Scalar* __restrict__ b)
{
    return a[0] * b[0];
}
template <>
__device__ inline Scalar
dot_stride_<1, LDIM, 1>(const Scalar* __restrict__ a, const Scalar* __restrict__ b)
{
    return a[0] * b[0];
}
template <>
__device__ inline Scalar
dot_stride_<1, PDIM, PDIM>(const Scalar* __restrict__ a, const Scalar* __restrict__ b)
{
    return a[0] * b[0];
}
template <>
__device__ inline Scalar
dot_stride_<3, 1, 1>(const Scalar* __restrict__ a, const Scalar* __restrict__ b)
{
    return a[0] * b[0];
}

// matrix(tansposed)-vector product: b = AT*x
template <int M, int N, AssignOP OP = ASSIGN>
__device__ inline void MatTMulVec(const Scalar* __restrict__ A, const Scalar* __restrict__ x, Scalar* __restrict__ b, Scalar omega)
{
#pragma unroll
    for (int i = 0; i < M; i++)
    {
        OP(b + i, omega * dot_<N>(A + i * N, x));
    }
}

// matrix(tansposed)-matrix product: C = AT*B
template <int L, int M, int N, AssignOP OP = ASSIGN>
__device__ inline void MatTMulMat(
    const Scalar* __restrict__ A,
    const Scalar* __restrict__ B,
    Scalar* __restrict__ C,
    Scalar omega)
{
#pragma unroll
    for (int i = 0; i < N; i++)
    {
        MatTMulVec<L, M, OP>(A, B + i * M, C + i * L, omega);
    }
}

// matrix-vector product: b = A*x
template <int M, int N, int S = 1, AssignOP OP = ASSIGN>
__device__ inline void
MatMulVec(const Scalar* __restrict__ A, const Scalar* __restrict__ x, Scalar* __restrict__ b)
{
#pragma unroll
    for (int i = 0; i < M; i++)
    {
        OP(b + i, dot_stride_<N, M, S>(A + i, x));
    }
}

// matrix-matrix product: C = A*B
template <int L, int M, int N, AssignOP OP = ASSIGN>
__device__ inline void
MatMulMat(const Scalar* __restrict__ A, const Scalar* __restrict__ B, Scalar* __restrict__ C)
{
#pragma unroll
    for (int i = 0; i < N; i++)
    {
        MatMulVec<L, M, 1, OP>(A, B + i * M, C + i * L);
    }
}

// matrix-matrix(tansposed) product: C = A*BT
template <int L, int M, int N, AssignOP OP = ASSIGN>
__device__ inline void
MatMulMatT(const Scalar* __restrict__ A, const Scalar* __restrict__ B, Scalar* __restrict__ C)
{
#pragma unroll
    for (int i = 0; i < N; i++)
    {
        MatMulVec<L, M, N, OP>(A, B + i, C + i * L);
    }
}

// squared L2 norm
template <int N>
__device__ inline Scalar squaredNorm(const Scalar* x)
{
    return dot_<N>(x, x);
}
template <int N>
__device__ inline Scalar squaredNorm(const Vecxd<N>& x)
{
    return squaredNorm<N>(x.data);
}
__device__ inline Scalar squaredNorm(const QuatD& x) { return squaredNorm<4>(x.data); }

// L2 norm
template <int N>
__device__ inline Scalar norm(const Scalar* x)
{
    return sqrt(squaredNorm<N>(x));
}
template <int N>
__device__ inline Scalar norm(const Vecxd<N>& x)
{
    return norm<N>(x.data);
}

__device__ inline Scalar norm(const QuatD& x) { return norm<4>(x.data); }

__device__ inline Matx<Scalar, 1, 6>
Vec1x3MulMat3x6(const Scalar* __restrict__ mat, const Scalar* __restrict__ vec)
{
    Matx<Scalar, 1, 6> output;
    output(0, 0) = vec[0] * mat[0] + vec[1] * mat[6] + vec[2] * mat[12];
    output(0, 1) = vec[0] * mat[1] + vec[1] * mat[7] + vec[2] * mat[13];
    output(0, 2) = vec[0] * mat[2] + vec[1] * mat[8] + vec[2] * mat[14];
    output(0, 3) = vec[0] * mat[3] + vec[1] * mat[9] + vec[2] * mat[15];
    output(0, 4) = vec[0] * mat[4] + vec[1] * mat[10] + vec[2] * mat[16];
    output(0, 5) = vec[0] * mat[5] + vec[1] * mat[11] + vec[2] * mat[17];
    return output;
}

////////////////////////////////////////////////////////////////////////////////////
// Device functions
////////////////////////////////////////////////////////////////////////////////////

__device__ inline void cross(const Vec4d& a, const Vec3d& b, Vec3d& c)
{
    c[0] = a[1] * b[2] - a[2] * b[1];
    c[1] = a[2] * b[0] - a[0] * b[2];
    c[2] = a[0] * b[1] - a[1] * b[0];
}

__device__ inline void cross(const Vec3d& a, const Vec3d& b, Vec3d& c)
{
    c[0] = a[1] * b[2] - a[2] * b[1];
    c[1] = a[2] * b[0] - a[0] * b[2];
    c[2] = a[0] * b[1] - a[1] * b[0];
}

__device__ inline void cross(const QuatD& a, const Vec3d& b, Vec3d& c)
{
    c[0] = a[1] * b[2] - a[2] * b[1];
    c[1] = a[2] * b[0] - a[0] * b[2];
    c[2] = a[0] * b[1] - a[1] * b[0];
}

__device__ inline Scalar dot(const Vec3d& a, const Vec3d& b)
{
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
}

__device__ inline QuatD quatMulQuat(const QuatD& a, const QuatD& b)
{
    QuatD out;
    out[0] = a[3] * b[0] + a[0] * b[3] + a[1] * b[2] - a[2] * b[1];
    out[1] = a[3] * b[1] + a[1] * b[3] + a[2] * b[0] - a[0] * b[2];
    out[2] = a[3] * b[2] + a[2] * b[3] + a[0] * b[1] - a[1] * b[0];
    out[3] = a[3] * b[3] - a[0] * b[0] - a[1] * b[1] - a[2] * b[2];
    return out;
}

__device__ inline Vec3d quatMulVec(const QuatD& q, const Vec3d& v)
{
    Vec3d uv;
    cross(q, v, uv);
    uv[0] = uv[0] + uv[0];
    uv[1] = uv[1] + uv[1];
    uv[2] = uv[2] + uv[2];

    //  = vec + r.w * uv + cross(rVec, uv);
    Vec3d ut, out;
    cross(q, uv, ut);
    out[0] = v[0] + q[3] * uv[0] + ut[0];
    out[1] = v[1] + q[3] * uv[1] + ut[1];
    out[2] = v[2] + q[3] * uv[2] + ut[2];

    return out;
}

__device__ inline void se3MulVec(const Se3D& se3, const Vec3d& vec, Vec3d& out)
{
    Vec3d rv = quatMulVec(se3.r, vec);

    // = (r * v) + t
    out[0] = rv[0] + se3.t[0];
    out[1] = rv[1] + se3.t[1];
    out[2] = rv[2] + se3.t[2];
}

__device__ inline void se3MulSe3(const Se3D& se3a, const Se3D& se3b, Se3D& out)
{
    out.r = quatMulQuat(se3a.r, se3b.r);
    Vec3d q1t2 = quatMulVec(se3a.r, se3b.t);
    out.t[0] = q1t2[0] + se3a.t[0];
    out.t[1] = q1t2[1] + se3a.t[1];
    out.t[2] = q1t2[2] + se3a.t[2];
}

__device__ inline Scalar
distance(const Vec3d& point, const Vec3d& a, const Vec3d& b, const Scalar& length)
{
    // from the loam paper.
    Vec3d aP, bP;
    // point a = point - a;
    aP[0] = point[0] - a[0];
    aP[1] = point[1] - a[1];
    aP[2] = point[2] - a[2];
    bP[0] = point[0] - b[0];
    bP[1] = point[1] - b[1];
    bP[2] = point[2] - b[2];

    Vec3d aPbPx;
    cross(aP, bP, aPbPx);
    Scalar numerator = norm(aPbPx);
    return numerator / length;
}

__device__ inline Scalar
signedDistance(const Vec3d& point, const Vec3d& normal, Scalar originDistance)
{
    Scalar d = dot(normal, point) - originDistance;
    return d;
}

__device__ inline void rotate(const QuatD& q, const Vec3d& Xw, Vec3d& Xc)
{
    Vec3d tmp1, tmp2;

    cross(q, Xw, tmp1);

    tmp1[0] += tmp1[0];
    tmp1[1] += tmp1[1];
    tmp1[2] += tmp1[2];

    cross(q, tmp1, tmp2);

    Xc[0] = Xw[0] + q[3] * tmp1[0] + tmp2[0];
    Xc[1] = Xw[1] + q[3] * tmp1[1] + tmp2[1];
    Xc[2] = Xw[2] + q[3] * tmp1[2] + tmp2[2];
}

__device__ inline void projectW2C(const QuatD& q, const Vec3d& t, const Vec3d& Xw, Vec3d& Xc)
{
    rotate(q, Xw, Xc);
    Xc[0] += t[0];
    Xc[1] += t[1];
    Xc[2] += t[2];
}

template <int MDIM>
__device__ inline void projectC2I(const Vec3d& Xc, CameraParamView cam, Vecxd<MDIM>& p)
{
}

template <>
__device__ inline void projectC2I<2>(const Vec3d& Xc, CameraParamView cam, Vec2d& p)
{
    const Scalar invZ = 1.0 / Xc[2];
    p[0] = cam.fx() * invZ * Xc[0] + cam.cx();
    p[1] = cam.fy() * invZ * Xc[1] + cam.cy();
}

template <>
__device__ inline void projectC2I<3>(const Vec3d& Xc, CameraParamView cam, Vec3d& p)
{
    const Scalar invZ = 1.0 / Xc[2];
    p[0] = cam.fx() * invZ * Xc[0] + cam.cx();
    p[1] = cam.fy() * invZ * Xc[1] + cam.cy();
    p[2] = p[0] - cam.bf() * invZ;
}

__device__ inline void camProjectDepth(const Vec3d& Xc, CameraParamView cam, Vec3d& p)
{
    const Scalar invZ = 1.0 / Xc[2];
    p[0] = cam.fx() * invZ * Xc[0] + cam.cx();
    p[1] = cam.fy() * invZ * Xc[1] + cam.cy();
    p[2] = invZ;
}

__device__ inline Matx<Scalar, 3, 3> identityMat3x3()
{
    Matx<Scalar, 3, 3> M;
    M(0, 0) = 1.0;
    M(1, 0) = 0.0;
    M(2, 0) = 0.0;
    M(0, 1) = 0.0;
    M(1, 1) = 1.0;
    M(2, 1) = 0.0;
    M(0, 2) = 0.0;
    M(1, 2) = 0.0;
    M(2, 2) = 1.0;
    return M;
}

__device__ inline void quaternionToRotationMatrix(const QuatD& q, MatView3x3d R)
{
    const Scalar x = q[0];
    const Scalar y = q[1];
    const Scalar z = q[2];
    const Scalar w = q[3];

    const Scalar tx = 2 * x;
    const Scalar ty = 2 * y;
    const Scalar tz = 2 * z;
    const Scalar twx = tx * w;
    const Scalar twy = ty * w;
    const Scalar twz = tz * w;
    const Scalar txx = tx * x;
    const Scalar txy = ty * x;
    const Scalar txz = tz * x;
    const Scalar tyy = ty * y;
    const Scalar tyz = tz * y;
    const Scalar tzz = tz * z;

    R(0, 0) = 1 - (tyy + tzz);
    R(0, 1) = txy - twz;
    R(0, 2) = txz + twy;
    R(1, 0) = txy + twz;
    R(1, 1) = 1 - (txx + tzz);
    R(1, 2) = tyz - twx;
    R(2, 0) = txz - twy;
    R(2, 1) = tyz + twx;
    R(2, 2) = 1 - (txx + tyy);
}

template <int MDIM>
__device__ void computeJacobians(
    const Vec3d& Xc,
    const QuatD& q,
    CameraParamView cam,
    MatView<Scalar, MDIM, PDIM> JP,
    MatView<Scalar, MDIM, LDIM> JL)
{
}

template <>
__device__ void computeJacobians<2>(
    const Vec3d& Xc, const QuatD& q, CameraParamView cam, MatView2x6d JP, MatView2x3d JL)
{
    const Scalar X = Xc[0];
    const Scalar Y = Xc[1];
    const Scalar Z = Xc[2];
    const Scalar invZ = 1.0 / Z;
    const Scalar x = invZ * X;
    const Scalar y = invZ * Y;
    const Scalar fu = cam.fx();
    const Scalar fv = cam.fy();
    const Scalar fu_invZ = fu * invZ;
    const Scalar fv_invZ = fv * invZ;

    Matx<Scalar, 3, 3> R;
    quaternionToRotationMatrix(q, R);

    JL(0, 0) = -fu_invZ * (R(0, 0) - x * R(2, 0));
    JL(0, 1) = -fu_invZ * (R(0, 1) - x * R(2, 1));
    JL(0, 2) = -fu_invZ * (R(0, 2) - x * R(2, 2));
    JL(1, 0) = -fv_invZ * (R(1, 0) - y * R(2, 0));
    JL(1, 1) = -fv_invZ * (R(1, 1) - y * R(2, 1));
    JL(1, 2) = -fv_invZ * (R(1, 2) - y * R(2, 2));

    JP(0, 0) = +fu * x * y;
    JP(0, 1) = -fu * (1 + x * x);
    JP(0, 2) = +fu * y;
    JP(0, 3) = -fu_invZ;
    JP(0, 4) = 0;
    JP(0, 5) = +fu_invZ * x;

    JP(1, 0) = +fv * (1 + y * y);
    JP(1, 1) = -fv * x * y;
    JP(1, 2) = -fv * x;
    JP(1, 3) = 0;
    JP(1, 4) = -fv_invZ;
    JP(1, 5) = +fv_invZ * y;
}

template <>
__device__ void computeJacobians<3>(
    const Vec3d& Xc, const QuatD& q, CameraParamView cam, MatView3x6d JP, MatView3x3d JL)
{
    const Scalar X = Xc[0];
    const Scalar Y = Xc[1];
    const Scalar Z = Xc[2];
    const Scalar invZ = 1.0 / Z;
    const Scalar invZZ = invZ * invZ;
    const Scalar fu = cam.fx();
    const Scalar fv = cam.fy();
    const Scalar bf = cam.bf();

    Matx<Scalar, 3, 3> R;
    quaternionToRotationMatrix(q, R);

    JL(0, 0) = -fu * R(0, 0) * invZ + fu * X * R(2, 0) * invZZ;
    JL(0, 1) = -fu * R(0, 1) * invZ + fu * X * R(2, 1) * invZZ;
    JL(0, 2) = -fu * R(0, 2) * invZ + fu * X * R(2, 2) * invZZ;

    JL(1, 0) = -fv * R(1, 0) * invZ + fv * Y * R(2, 0) * invZZ;
    JL(1, 1) = -fv * R(1, 1) * invZ + fv * Y * R(2, 1) * invZZ;
    JL(1, 2) = -fv * R(1, 2) * invZ + fv * Y * R(2, 2) * invZZ;

    JL(2, 0) = JL(0, 0) - bf * R(2, 0) * invZZ;
    JL(2, 1) = JL(0, 1) - bf * R(2, 1) * invZZ;
    JL(2, 2) = JL(0, 2) - bf * R(2, 2) * invZZ;

    JP(0, 0) = X * Y * invZZ * fu;
    JP(0, 1) = -(1 + (X * X * invZZ)) * fu;
    JP(0, 2) = Y * invZ * fu;
    JP(0, 3) = -1 * invZ * fu;
    JP(0, 4) = 0;
    JP(0, 5) = X * invZZ * fu;

    JP(1, 0) = (1 + Y * Y * invZZ) * fv;
    JP(1, 1) = -X * Y * invZZ * fv;
    JP(1, 2) = -X * invZ * fv;
    JP(1, 3) = 0;
    JP(1, 4) = -1 * invZ * fv;
    JP(1, 5) = Y * invZZ * fv;

    JP(2, 0) = JP(0, 0) - bf * Y * invZZ;
    JP(2, 1) = JP(0, 1) + bf * X * invZZ;
    JP(2, 2) = JP(0, 2);
    JP(2, 3) = JP(0, 3);
    JP(2, 4) = 0;
    JP(2, 5) = JP(0, 5) - bf * invZZ;
}

__device__ inline QuatD conjugate(const QuatD& r)
{
    QuatD output;
    output[0] = -r[0];
    output[1] = -r[1];
    output[2] = -r[2];
    output[3] = r[3];
    return output;
}

__device__ inline Vec3d addVec3(const Vec3d& a, const Vec3d& b)
{
    Vec3d output;
    output[0] = a[0] + b[0];
    output[1] = a[1] + b[1];
    output[2] = a[2] + b[2];
    return output;
}

__device__ inline Vec3d addVec3Scalar(const Vec3d& a, const Scalar& b)
{
    Vec3d output;
    output[0] = a[0] + b;
    output[1] = a[1] + b;
    output[2] = a[2] + b;
    return output;
}

__device__ inline Se3D inverse(const Se3D& se3)
{
    const Scalar normSquaredR = squaredNorm(se3.r);
    const Scalar invNormSquaredR = 1.0 / normSquaredR;
    const QuatD conjR = conjugate(se3.r);
    QuatD invQuat;
    invQuat[0] = conjR[0] * invNormSquaredR;
    invQuat[1] = conjR[1] * invNormSquaredR;
    invQuat[2] = conjR[2] * invNormSquaredR;
    invQuat[3] = conjR[3] * invNormSquaredR;

    Se3D output;
    output.r = invQuat;

    Vec3d rvec, uv, ut, invTrans;
    rvec[0] = invQuat[0];
    rvec[1] = invQuat[1];
    rvec[2] = invQuat[2];

    cross(rvec, se3.t, uv);
    const Vec3d uv2 = addVec3(uv, uv);

    // inv = trans + rot.w * uv + cross(rvec, uv)
    const Vec3d tw = addVec3Scalar(se3.t, se3.r[3]);
    cross(rvec, uv, ut);
    output.t[0] = -(tw[0] * ut[0]);
    output.t[1] = -(tw[1] * ut[1]);
    output.t[2] = -(tw[2] * ut[2]);
    return output;
}

__device__ inline void Sym3x3Inv(ConstMatView3x3d A, MatView3x3d B)
{
    const Scalar A00 = A(0, 0);
    const Scalar A01 = A(0, 1);
    const Scalar A11 = A(1, 1);
    const Scalar A02 = A(2, 0);
    const Scalar A12 = A(1, 2);
    const Scalar A22 = A(2, 2);

    const Scalar det = A00 * A11 * A22 + A01 * A12 * A02 + A02 * A01 * A12 - A00 * A12 * A12 -
        A02 * A11 * A02 - A01 * A01 * A22;

    const Scalar invDet = 1 / det;

    const Scalar B00 = invDet * (A11 * A22 - A12 * A12);
    const Scalar B01 = invDet * (A02 * A12 - A01 * A22);
    const Scalar B11 = invDet * (A00 * A22 - A02 * A02);
    const Scalar B02 = invDet * (A01 * A12 - A02 * A11);
    const Scalar B12 = invDet * (A02 * A01 - A00 * A12);
    const Scalar B22 = invDet * (A00 * A11 - A01 * A01);

    B(0, 0) = B00;
    B(0, 1) = B01;
    B(0, 2) = B02;
    B(1, 0) = B01;
    B(1, 1) = B11;
    B(1, 2) = B12;
    B(2, 0) = B02;
    B(2, 1) = B12;
    B(2, 2) = B22;
}

__device__ inline void skew1(Scalar x, Scalar y, Scalar z, MatView3x3d M)
{
    M(0, 0) = +0;
    M(0, 1) = -z;
    M(0, 2) = +y;
    M(1, 0) = +z;
    M(1, 1) = +0;
    M(1, 2) = -x;
    M(2, 0) = -y;
    M(2, 1) = +x;
    M(2, 2) = +0;
}

__device__ inline void skew2(Scalar x, Scalar y, Scalar z, MatView3x3d M)
{
    const Scalar xx = x * x;
    const Scalar yy = y * y;
    const Scalar zz = z * z;

    const Scalar xy = x * y;
    const Scalar yz = y * z;
    const Scalar zx = z * x;

    M(0, 0) = -yy - zz;
    M(0, 1) = +xy;
    M(0, 2) = +zx;
    M(1, 0) = +xy;
    M(1, 1) = -zz - xx;
    M(1, 2) = +yz;
    M(2, 0) = +zx;
    M(2, 1) = +yz;
    M(2, 2) = -xx - yy;
}

__device__ inline void
addOmega(Scalar a1, ConstMatView3x3d O1, Scalar a2, ConstMatView3x3d O2, MatView3x3d R)
{
    R(0, 0) = 1 + a1 * O1(0, 0) + a2 * O2(0, 0);
    R(1, 0) = 0 + a1 * O1(1, 0) + a2 * O2(1, 0);
    R(2, 0) = 0 + a1 * O1(2, 0) + a2 * O2(2, 0);

    R(0, 1) = 0 + a1 * O1(0, 1) + a2 * O2(0, 1);
    R(1, 1) = 1 + a1 * O1(1, 1) + a2 * O2(1, 1);
    R(2, 1) = 0 + a1 * O1(2, 1) + a2 * O2(2, 1);

    R(0, 2) = 0 + a1 * O1(0, 2) + a2 * O2(0, 2);
    R(1, 2) = 0 + a1 * O1(1, 2) + a2 * O2(1, 2);
    R(2, 2) = 1 + a1 * O1(2, 2) + a2 * O2(2, 2);
}

__device__ inline void rotationMatrixToQuaternion(ConstMatView3x3d R, QuatD& q)
{
    Scalar t = R(0, 0) + R(1, 1) + R(2, 2);
    if (t > 0)
    {
        t = sqrt(t + 1);
        q[3] = Scalar(0.5) * t;
        t = Scalar(0.5) / t;
        q[0] = (R(2, 1) - R(1, 2)) * t;
        q[1] = (R(0, 2) - R(2, 0)) * t;
        q[2] = (R(1, 0) - R(0, 1)) * t;
    }
    else
    {
        int i = 0;
        if (R(1, 1) > R(0, 0))
        {
            i = 1;
        }
        if (R(2, 2) > R(i, i))
        {
            i = 2;
        }
        int j = (i + 1) % 3;
        int k = (j + 1) % 3;

        t = sqrt(R(i, i) - R(j, j) - R(k, k) + 1);
        q[i] = Scalar(0.5) * t;
        t = Scalar(0.5) / t;
        q[3] = (R(k, j) - R(j, k)) * t;
        q[j] = (R(j, i) + R(i, j)) * t;
        q[k] = (R(k, i) + R(i, k)) * t;
    }
}

__device__ inline void multiplyQuaternion(const QuatD& a, const QuatD& b, QuatD& c)
{
    c[3] = a[3] * b[3] - a[0] * b[0] - a[1] * b[1] - a[2] * b[2];
    c[0] = a[3] * b[0] + a[0] * b[3] + a[1] * b[2] - a[2] * b[1];
    c[1] = a[3] * b[1] + a[1] * b[3] + a[2] * b[0] - a[0] * b[2];
    c[2] = a[3] * b[2] + a[2] * b[3] + a[0] * b[1] - a[1] * b[0];
}

__device__ inline void normalizeQuaternion(const QuatD& a, QuatD& b)
{
    Scalar invn = 1 / norm(a);
    if (a[3] < 0)
    {
        invn = -invn;
    }
    for (int i = 0; i < 4; i++)
    {
        b[i] = invn * a[i];
    }
}

__device__ inline Scalar pow2(const Scalar x) { return x * x; }

__device__ inline Scalar pow3(const Scalar x) { return x * x * x; }

__device__ inline void updateExp(const Scalar* __restrict__ update, QuatD& rot, Vec3d& trans)
{
    Vec3d omega(update);
    Vec3d upsilon(update + 3);

    const Scalar theta = norm(omega);

    Matx<Scalar, 3, 3> O1, O2;
    skew1(omega[0], omega[1], omega[2], O1);
    skew2(omega[0], omega[1], omega[2], O2);

    Scalar R[9], V[9];
    if (theta < Scalar(0.00001))
    {
        addOmega(Scalar(1.0), O1, Scalar(0.5), O2, R);
        addOmega(Scalar(0.5), O1, Scalar(1) / 6, O2, V);
    }
    else
    {
        const Scalar a1 = sin(theta) / theta;
        const Scalar a2 = (1 - cos(theta)) / (theta * theta);
        const Scalar a3 = (theta - sin(theta)) / pow3(theta);
        addOmega(a1, O1, a2, O2, R);
        addOmega(a2, O1, a3, O2, V);
    }

    rotationMatrixToQuaternion(R, rot);
    MatMulVec<3, 3>(V, upsilon.data, trans.data);
}

__device__ inline void updatePose(const QuatD& q1, const Vec3d& t1, QuatD& q2, Vec3d& t2)
{
    Vec3d u;
    rotate(q1, t2, u);

    t2[0] = t1[0] + u[0];
    t2[1] = t1[1] + u[1];
    t2[2] = t1[2] + u[2];

    QuatD r;
    multiplyQuaternion(q1, q2, r);
    normalizeQuaternion(r, q2);
}

template <int N>
__device__ inline void copy(const Scalar* __restrict__ src, Scalar* __restrict__ dst)
{
    for (int i = 0; i < N; i++)
    {
        dst[i] = src[i];
    }
}

__device__ inline Vec3i makeVec3i(int i, int j, int k)
{
    Vec3i vec;
    vec[0] = i;
    vec[1] = j;
    vec[2] = k;
    return vec;
}

// ============================================================================================================
// grid-stride parallel reduction (old method)

__device__ void parallelReductionAndAdd(
    Scalar* __restrict__ cache,
    const int blockSize,
    const int tid,
    const Scalar value,
    Scalar* accumValue)
{
    cache[tid] = value;
    __syncthreads();

    for (int stride = blockSize / 2; stride > 0; stride >>= 1)
    {
        if (tid < stride)
        {
            cache[tid] += cache[tid + stride];
        }
        __syncthreads();
    }

    if (tid == 0)
    {
        atomicAdd(accumValue, cache[0]);
    }
}

// ======================================================================================
// co-operative groups and parallel reduction

template <typename KernelFunction, typename... KernelParameters>
inline void cooperativeLaunch(
    const KernelFunction& kernel_function,
    const cudaStream_t streamId,
    const int gridSize,
    const int blockSize,
    const int sharedBytes,
    KernelParameters... parameters)
{
    void* argumentsPtrs[sizeof...(KernelParameters)];
    auto argIdx = 0;

    detail::for_each_argument_address(
        [&](void* x) { argumentsPtrs[argIdx++] = x; }, parameters...);

    cudaLaunchCooperativeKernel<KernelFunction>(
        &kernel_function,
        gridSize,
        blockSize,
        argumentsPtrs,
        sharedBytes,
        streamId);
}

__device__ void reduceBlock(double* sdata, const cg::thread_block& cta)
{
    const int tid = cta.thread_rank();
    cg::thread_block_tile<32> tile32 = cg::tiled_partition<32>(cta);

    sdata[tid] = cg::reduce(tile32, sdata[tid], cg::plus<double>());
    cg::sync(cta);

    double accum = 0.0;
    if (cta.thread_rank() == 0)
    {
        for (int i = 0; i < blockDim.x; i += tile32.size())
        {
            accum += sdata[i];
        }
        sdata[0] = accum;
    }
    cg::sync(cta);
}

__device__ void parallelReductionAndAddCooperative(
    cg::thread_block block,
    Scalar* __restrict__ cache,
    const int blockSize,
    const int tid,
    const Scalar value,
    Scalar* accumValue)
{
    Scalar sum = value;
    cache[tid] = sum;
    cg::sync(block);

    if ((blockSize >= 512) && (tid < 256))
    {
        cache[tid] = sum = sum + cache[tid + 256];
    }
    cg::sync(block);

    if ((blockSize >= 256) && (tid < 128))
    {
        cache[tid] = sum = sum + cache[tid + 128];
    }
    cg::sync(block);

    if ((blockSize >= 128) && (tid < 64))
    {
        cache[tid] = sum = sum + cache[tid + 64];
    }
    cg::sync(block);

    cg::thread_block_tile<32> tile32 = cg::tiled_partition<32>(block);
    if (block.thread_rank() < 32)
    {
        if (blockSize >= 64)
        {
            sum += cache[tid + 32];
        }
        for (int offset = tile32.size() / 2; offset > 0; offset /= 2)
        {
            sum += tile32.shfl_down(sum, offset);
        }
    }

    // write result for this block to global mem
    if (block.thread_rank() == 0)
    {
        atomicAdd(accumValue, sum);
    }
}

////////////////////////////////////////////////////////////////////////////////////
// Robust Kernel functions
////////////////////////////////////////////////////////////////////////////////////

class RobustKernelFunc
{
public:
    __device__ inline RobustKernelFunc(const Scalar delta) : delta_(delta) {}
    __device__ virtual Scalar robustify(const Scalar x) const { return x; }
    __device__ virtual Scalar derivative(const Scalar x) const { return 1; }

protected:

    Scalar delta_;
};


class RobustKernelTurkey : public RobustKernelFunc
{
public:
    __device__ inline RobustKernelTurkey(const Scalar delta) : RobustKernelFunc(delta), deltaSq_(delta * delta) {}

    __device__ inline Scalar robustify(const Scalar x) const
    {
        const Scalar maxv = (Scalar(1) / 3) * deltaSq_;
        return x <= deltaSq_ ? maxv * (1 - pow3(1 - x / deltaSq_)) : maxv;
    }

    __device__ inline Scalar derivative(const Scalar x) const
    {
        return x <= deltaSq_ ? pow2(1 - x / deltaSq_) : 0;
    }

    Scalar deltaSq_;
};

class RobustKernelCauchy : public RobustKernelFunc
{
public:
    __device__ inline RobustKernelCauchy(const Scalar delta)
        : RobustKernelFunc(delta), deltaSq_(delta * delta)
    {
    }

    __device__ inline Scalar robustify(const Scalar x) const
    {
        Scalar dsqrReci = 1.0 / deltaSq_;
        Scalar aux = dsqrReci * x + 1.0;
        return deltaSq_ * log(aux);
    }

    __device__ inline Scalar derivative(const Scalar x) const
    {
        Scalar dsqrReci = 1.0 / deltaSq_;
        Scalar aux = dsqrReci * x + 1.0;
        return 1.0 / aux;
    }

    Scalar deltaSq_;
};

// buffer used for creating the robust kernel function
__managed__ __align__(16) char rkBuffer[128];
__managed__ RobustKernelFunc* robustKernelFunc = nullptr;

// required to create virtual functions on the device
__global__ void createRkFunctionKernel(const RobustKernelType type, const Scalar* delta)
{
    if (threadIdx.x == 0 && blockIdx.x == 0)
    {
        switch (type) 
        {
            case RobustKernelType::None:
                robustKernelFunc = new (rkBuffer) RobustKernelFunc(delta[0]);
                break;
            case RobustKernelType::Tukey:
                robustKernelFunc = new (rkBuffer) RobustKernelTurkey(delta[0]);
                break;
            case RobustKernelType::Cauchy:
                robustKernelFunc = new (rkBuffer) RobustKernelCauchy(delta[0]);
                break;
        }
    }
}

__global__ void deleteRkFunctionKernel() { delete robustKernelFunc; }

////////////////////////////////////////////////////////////////////////////////////
// Kernel functions
////////////////////////////////////////////////////////////////////////////////////

template <int MDIM>
__global__ void computeActiveErrorsKernel(
    int nedges,
    int nomegas,
    int ncameras,
    const Se3D* se3,
    const Vec3d* Xws,
    const Vecxd<MDIM>* measurements,
    const Scalar* omegas,
    const Vec2i* edge2PL,
    const Vec5d* cameras,
    const int* outliers,
    Vecxd<MDIM>* errors,
    Vec3d* Xcs,
    Scalar* chiValues)
{
    using Vecmd = Vecxd<MDIM>;

    int iE = blockIdx.x * blockDim.x + threadIdx.x;
    if (iE >= nedges || outliers[iE])
    {
        return;
    }
    const Scalar omega = (nomegas > 1) ? omegas[iE] : omegas[0];
    const int iP = edge2PL[iE][0];
    const int iL = edge2PL[iE][1];

    const QuatD& q = se3[iP].r;
    const Vec3d& t = se3[iP].t;
    const Vec3d& Xw = Xws[iL];
    const Vecmd& measurement = measurements[iE];
    const Vec5d& camera = (ncameras > 1) ? cameras[iE] : cameras[0];

    // project world to camera
    Vec3d Xc;
    projectW2C(q, t, Xw, Xc);

    // project camera to image
    Vecmd proj;
    projectC2I(Xc, camera, proj);

    // compute residual
    Vecmd error;
    for (int i = 0; i < MDIM; i++)
    {
        error[i] = proj[i] - measurement[i];
    }

    errors[iE] = error;
    Xcs[iE] = Xc;
    chiValues[iE] = robustKernelFunc->robustify(omega * squaredNorm(error));
}

__global__ void computeChiValueKernel(
    int nedges, int blockSize, const Scalar* chiValues, const int* outliers, Scalar* totalChi)
{
    const int tid = threadIdx.x;
    extern __shared__ Scalar cache[];

    int iE = blockIdx.x * blockDim.x + threadIdx.x;
    const int gridSize = gridDim.x * blockDim.x;

    Scalar sumchi = 0;

    while (iE < nedges)
    {
        if (!outliers[iE])
        {
            sumchi += chiValues[iE];
        }
        iE += gridSize;
    }

    parallelReductionAndAdd(cache, blockSize, tid, sumchi, totalChi);
}

__global__ void computeOutliersKernel(
    int nedges, const Scalar errorThreshold, const Scalar* chiValues, int* outliers)
{
    int iE = blockIdx.x * blockDim.x + threadIdx.x;
    if (iE >= nedges)
    {
        return;
    }

    const Scalar chi2 = chiValues[iE];
    if (chi2 > errorThreshold)
    {
        outliers[iE] = 1;
    }
}

    template <int MDIM>
__global__ void constructQuadraticFormKernel(
    int nedges,
    int nomegas,
    int ncameras,
    const Vec3d* Xcs,
    const Se3D* se3,
    const Vecxd<MDIM>* errors,
    const Scalar* omegas,
    const Vec2i* edge2PL,
    const int* edge2Hpl,
    const uint8_t* flags,
    const Vec5d* cameras,
    const int* outliers,
    PxPBlockPtr Hpp,
    Px1BlockPtr bp,
    LxLBlockPtr Hll,
    Lx1BlockPtr bl,
    PxLBlockPtr Hpl)
{
    using Vecmd = Vecxd<MDIM>;

    const int iE = blockIdx.x * blockDim.x + threadIdx.x;
    if (iE >= nedges || outliers[iE])
    {
        return;
    }

    const Scalar omega = (nomegas > 1) ? omegas[iE] : omegas[0];
    const int iP = edge2PL[iE][0];
    const int iL = edge2PL[iE][1];
    const int iPL = edge2Hpl[iE];
    const int flag = flags[iE];

    const QuatD& q = se3[iP].r;
    const Vec3d& Xc = Xcs[iE];
    const Vecmd& error = errors[iE];
    const Vec5d& camera = (ncameras > 1) ? cameras[iE] : cameras[0];

    const Scalar e = squaredNorm(error) * omega;
    const Scalar rho1 = robustKernelFunc->derivative(e);
    const Scalar rhoOmega = omega * rho1;

    // compute Jacobians
    Scalar JP[MDIM * PDIM];
    Scalar JL[MDIM * LDIM];
    computeJacobians<MDIM>(Xc, q, camera, JP, JL);

    if (!(flag & EDGE_FLAG_FIXED_P))
    {
        // Hpp += = JPT*Ω*JP
        MatTMulMat<PDIM, MDIM, PDIM, ACCUM_ATOMIC>(JP, JP, Hpp.at(iP), rhoOmega);

        // bp += = JPT*Ω*r
        MatTMulVec<PDIM, MDIM, ACCUM_ATOMIC>(JP, error.data, bp.at(iP), rhoOmega);
    }
    if (!(flag & EDGE_FLAG_FIXED_L))
    {
        // Hll += = JLT*Ω*JL
        MatTMulMat<LDIM, MDIM, LDIM, ACCUM_ATOMIC>(JL, JL, Hll.at(iL), rhoOmega);

        // bl += = JLT*Ω*r
        MatTMulVec<LDIM, MDIM, ACCUM_ATOMIC>(JL, error.data, bl.at(iL), rhoOmega);
    }
    if (!flag)
    {
        // Hpl += = JPT*Ω*JL
        MatTMulMat<PDIM, MDIM, LDIM, ASSIGN>(JP, JL, Hpl.at(iPL), rhoOmega);
    }
}

template <int DIM>
__global__ void maxDiagonalKernel(int size, int blockSize, const Scalar* __restrict__ D, Scalar* __restrict__ maxD)
{
    const int sharedIdx = threadIdx.x;
    extern __shared__ Scalar cache[];

    Scalar maxVal = 0;
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < size; i += gridDim.x * blockDim.x)
    {
        const int j = i / DIM;
        const int k = i % DIM;
        const Scalar* ptrBlock = D + j * DIM * DIM;
        maxVal = max(maxVal, ptrBlock[k * DIM + k]);
    }

    cache[sharedIdx] = maxVal;
    __syncthreads();

    for (int stride = blockSize / 2; stride > 0; stride >>= 1)
    {
        if (sharedIdx < stride)
        {
            cache[sharedIdx] = max(cache[sharedIdx], cache[sharedIdx + stride]);
        }
        __syncthreads();
    }

    if (sharedIdx == 0)
    {
        maxD[blockIdx.x] = cache[0];
    }
}

template <int DIM>
__global__ void
addLambdaKernel(int size, Scalar* __restrict__ D, Scalar lambda, Scalar* __restrict__ backup)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= size)
    {
        return;
    }
    const int j = i / DIM;
    const int k = i % DIM;
    Scalar* ptrBlock = D + j * DIM * DIM;
    backup[i] = ptrBlock[k * DIM + k];
    ptrBlock[k * DIM + k] += lambda;
}

template <int DIM>
__global__ void
restoreDiagonalKernel(int size, Scalar* __restrict__ D, const Scalar* __restrict__ backup)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= size)
    {
        return;
    }
    const int j = i / DIM;
    const int k = i % DIM;
    Scalar* ptrBlock = D + j * DIM * DIM;
    ptrBlock[k * DIM + k] = backup[i];
}

__global__ void computeBschureKernel(
    int cols,
    LxLBlockPtr Hll,
    LxLBlockPtr invHll,
    Lx1BlockPtr bl,
    PxLBlockPtr Hpl,
    const int* __restrict__ HplColPtr,
    const int* __restrict__ HplRowInd,
    Px1BlockPtr bsc,
    PxLBlockPtr Hpl_invHll)
{
    const int colId = blockIdx.x * blockDim.x + threadIdx.x;
    if (colId >= cols)
    {
        return;
    }
    Scalar iHll[LDIM * LDIM];
    Scalar Hpl_iHll[PDIM * LDIM];

    Sym3x3Inv(Hll.at(colId), iHll);
    copy<LDIM * LDIM>(iHll, invHll.at(colId));

    for (int i = HplColPtr[colId]; i < HplColPtr[colId + 1]; i++)
    {
        MatMulMat<6, 3, 3>(Hpl.at(i), iHll, Hpl_iHll);
        MatMulVec<6, 3, 1, DEACCUM_ATOMIC>(Hpl_iHll, bl.at(colId), bsc.at(HplRowInd[i]));
        copy<PDIM * LDIM>(Hpl_iHll, Hpl_invHll.at(i));
    }
}

__global__ void initializeHschurKernel(
    int rows, PxPBlockPtr Hpp, PxPBlockPtr Hsc, const int* HscRowPtr)
{
    const int rowId = blockIdx.x * blockDim.x + threadIdx.x;
    if (rowId >= rows)
    {
        return;
    }
    copy<PDIM * PDIM>(Hpp.at(rowId), Hsc.at(HscRowPtr[rowId]));
}

__global__ void computeHschureKernel(
    int size,
    const Vec3i* mulBlockIds,
    PxLBlockPtr Hpl_invHll,
    PxLBlockPtr Hpl,
    PxPBlockPtr Hschur)
{
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= size)
    {
        return;
    }
    const Vec3i index = mulBlockIds[tid];
    Scalar A[PDIM * LDIM];
    Scalar B[PDIM * LDIM];
    copy<PDIM * LDIM>(Hpl_invHll.at(index[0]), A);
    copy<PDIM * LDIM>(Hpl.at(index[1]), B);
    MatMulMatT<6, 3, 6, DEACCUM_ATOMIC>(A, B, Hschur.at(index[2]));
}

__global__ void findHschureMulBlockIndicesKernel(
    int cols,
    const int* __restrict__ HplColPtr,
    const int* __restrict__ HplRowInd,
    const int* __restrict__ HscRowPtr,
    const int* __restrict__ HscColInd,
    Vec3i*  mulBlockIds,
    int* nindices)
{
    const int colId = blockIdx.x * blockDim.x + threadIdx.x;
    if (colId >= cols)
    {
        return;
    }
    const int i0 = HplColPtr[colId];
    const int i1 = HplColPtr[colId + 1];
    for (int i = i0; i < i1; i++)
    {
        const int iP1 = HplRowInd[i];
        int k = HscRowPtr[iP1];
        for (int j = i; j < i1; j++)
        {
            const int iP2 = HplRowInd[j];
            while (HscColInd[k] < iP2)
            {
                k++;
            }
            const int pos = atomicAdd(nindices, 1);
            mulBlockIds[pos] = makeVec3i(i, j, k);
        }
    }
}

__global__ void permuteNnzPerRowKernel(
    int size,
    const int* __restrict__ srcRowPtr,
    const int* __restrict__ P,
    int*  nnzPerRow)
{
    const int rowId = blockIdx.x * blockDim.x + threadIdx.x;
    if (rowId >= size)
    {
        return;
    }
    nnzPerRow[P[rowId]] = srcRowPtr[rowId + 1] - srcRowPtr[rowId];
}

__global__ void permuteColIndKernel(
    int size,
    const int* __restrict__ srcRowPtr,
    const int* __restrict__ srcColInd,
    const int* __restrict__ P,
    int*  dstColInd,
    int*  dstMap,
    int*  nnzPerRow)
{
    const int rowId = blockIdx.x * blockDim.x + threadIdx.x;
    if (rowId >= size)
    {
        return;
    }
    const int i0 = srcRowPtr[rowId];
    const int i1 = srcRowPtr[rowId + 1];
    const int permRowId = P[rowId];
    for (int srck = i0; srck < i1; srck++)
    {
        const int dstk = nnzPerRow[permRowId]++;
        dstColInd[dstk] = P[srcColInd[srck]];
        dstMap[dstk] = srck;
    }
}

__global__ void schurComplementPostKernel(
    int cols,
    LxLBlockPtr invHll,
    Lx1BlockPtr bl,
    PxLBlockPtr Hpl,
    const int* __restrict__ HplColPtr,
    const int* __restrict__ HplRowInd,
    Px1BlockPtr xp,
    Lx1BlockPtr xl)
{
    const int colId = blockIdx.x * blockDim.x + threadIdx.x;
    if (colId >= cols)
    {
        return;
    }
    Scalar cl[LDIM];
    copy<LDIM>(bl.at(colId), cl);

    for (int i = HplColPtr[colId]; i < HplColPtr[colId + 1]; i++)
    {
        MatTMulVec<3, 6, DEACCUM>(Hpl.at(i), xp.at(HplRowInd[i]), cl, 1);
    }
    MatMulVec<3, 3>(invHll.at(colId), cl, xl.at(colId));
}

__global__ void updatePosesKernel(int size, Px1BlockPtr xp, Se3D* est)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= size)
    {
        return;
    }
    QuatD expq;
    Vec3d expt;
    updateExp(xp.at(i), expq, expt);
    updatePose(expq, expt, est[i].r, est[i].t);
}

__global__ void updateLandmarksKernel(int size, Lx1BlockPtr xl, Vec3d* Xws)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= size)
    {
        return;
    }
    const Scalar* dXw = xl.at(i);
    Vec3d& Xw = Xws[i];
    Xw[0] += dXw[0];
    Xw[1] += dXw[1];
    Xw[2] += dXw[2];
}

__global__ void computeScaleKernel(
    const Scalar* __restrict__ x,
    const Scalar* __restrict__ b,
    Scalar* __restrict__ scale,
    Scalar lambda,
    int size,
    int blockSize)
{
    cg::thread_block block = cg::this_thread_block();
    const int tid = threadIdx.x;
    extern __shared__ Scalar cache[];

    Scalar sum = 0;
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < size; i += gridDim.x * blockDim.x)
    {
        sum += x[i] * (lambda * x[i] + b[i]);
    }

    parallelReductionAndAdd(cache, blockSize, tid, sum, scale);
}

__global__ void convertBSRToCSRKernel(
    int size, const Scalar* __restrict__ src, Scalar* __restrict__ dst, const int* map)
{
    const int dstk = blockIdx.x * blockDim.x + threadIdx.x;
    if (dstk >= size)
    {
        return;
    }
    dst[dstk] = src[map[dstk]];
}

__global__ void
nnzPerColKernel(const Vec3i* __restrict__ blockpos, int nblocks, int* __restrict__ nnzPerCol)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= nblocks)
    {
        return;
    }
    const int colId = blockpos[i][1];
    atomicAdd(&nnzPerCol[colId], 1);
}

__global__ void setRowIndKernel(
    const Vec3i* __restrict__ blockpos,
    int nblocks,
    int* __restrict__ rowInd,
    int* __restrict__ indexPL)
{
    const int k = blockIdx.x * blockDim.x + threadIdx.x;
    if (k >= nblocks)
    {
        return;
    }
    const int rowId = blockpos[k][0];
    const int edgeId = blockpos[k][2];
    rowInd[k] = rowId;
    indexPL[edgeId] = k;
}

////////////////////////////////////////////////////////////////////////////////////
// Public functions
////////////////////////////////////////////////////////////////////////////////////

void waitForKernelCompletion() { CUDA_CHECK(cudaDeviceSynchronize()); }

void recordEvent(const CudaDeviceInfo& info)
{
    CUDA_CHECK(cudaEventRecord(info.event, info.stream));
}

void calculateOccupancy(int size, void* kernelFunc, int& outputBlockSize, int& outputGridSize)
{
    int minGridSize;
    cudaOccupancyMaxPotentialBlockSize(&minGridSize, &outputBlockSize, kernelFunc);
    outputGridSize = divUp(size, outputBlockSize);
    if (outputGridSize < minGridSize)
    {
        outputGridSize = minGridSize;
    }
}
       
void waitForEvent(const cudaEvent_t event) { CUDA_CHECK(cudaEventSynchronize(event)); }

void createRkFunction(
    RobustKernelType type, const GpuVec<Scalar>& d_delta, const CudaDeviceInfo& deviceInfo)
{
    createRkFunctionKernel<<<1, 1, 0, deviceInfo.stream>>>(type, d_delta.data());
    CUDA_CHECK(cudaEventRecord(deviceInfo.event, deviceInfo.stream));
    CUDA_CHECK(cudaEventSynchronize(deviceInfo.event));
}

void exclusiveScan(const int* src, int* dst, int size)
{
    auto ptrSrc = thrust::device_pointer_cast(src);
    auto ptrDst = thrust::device_pointer_cast(dst);
    thrust::exclusive_scan(ptrSrc, ptrSrc + size, ptrDst);
}

void buildHplStructure(
    GpuVec3i& blockpos,
    GpuHplBlockMat& Hpl,
    GpuVec1i& indexPL,
    GpuVec1i& nnzPerCol,
    const CudaDeviceInfo& deviceInfo1,
    const CudaDeviceInfo& deviceInfo2)
{
    const int nblocks = Hpl.nnz();
    int blockSize;
    int gridSize;
    calculateOccupancy(nblocks, (void*)nnzPerColKernel, blockSize, gridSize);

    int* colPtr = Hpl.outerIndices();
    int* rowInd = Hpl.innerIndices();

    auto ptrBlockPos = thrust::device_pointer_cast(blockpos.data());
    thrust::sort(ptrBlockPos, ptrBlockPos + nblocks, LessColId());

    CUDA_CHECK(cudaMemset(nnzPerCol, 0, sizeof(int) * (Hpl.cols() + 1)));
    nnzPerColKernel<<<gridSize, blockSize, 0, deviceInfo1.stream>>>(blockpos, nblocks, nnzPerCol);
    CUDA_CHECK(cudaEventRecord(deviceInfo1.event, deviceInfo1.stream));

    setRowIndKernel<<<gridSize, blockSize, 0 , deviceInfo2.stream>>>(blockpos, nblocks, rowInd, indexPL);
    CUDA_CHECK(cudaEventRecord(deviceInfo2.event, deviceInfo2.stream));

    CUDA_CHECK(cudaEventRecord(deviceInfo1.event, deviceInfo1.stream));
    CUDA_CHECK(cudaEventSynchronize(deviceInfo1.event));

    exclusiveScan(nnzPerCol, colPtr, Hpl.cols() + 1);

    CUDA_CHECK(cudaEventSynchronize(deviceInfo2.event));
    CUDA_CHECK(cudaGetLastError());
}

void findHschureMulBlockIndices(
    const GpuHplBlockMat& Hpl,
    const GpuHscBlockMat& Hsc,
    GpuVec3i& mulBlockIds,
    const CudaDeviceInfo& deviceInfo)
{
    int blockSize;
    int gridSize;
    calculateOccupancy(Hpl.cols(), (void*)nnzPerColKernel, blockSize, gridSize);

    DeviceBuffer<int> nindices(1);
    nindices.fillZero();

    findHschureMulBlockIndicesKernel<<<gridSize, blockSize, 0, deviceInfo.stream>>>(
        Hpl.cols(),
        Hpl.outerIndices(),
        Hpl.innerIndices(),
        Hsc.outerIndices(),
        Hsc.innerIndices(),
        mulBlockIds,
        nindices);
    
    auto ptrSrc = thrust::device_pointer_cast(mulBlockIds.data());
    thrust::sort(ptrSrc, ptrSrc + mulBlockIds.size(), LessRowId());

    CUDA_CHECK(cudaEventRecord(deviceInfo.event, deviceInfo.stream));
    CUDA_CHECK(cudaEventSynchronize(deviceInfo.event));
    CUDA_CHECK(cudaGetLastError());
}

void computeOutliers(
    int nedges,
    const Scalar errorThreshold,
    const Scalar* chiValues,
    int* outliers,
    const CudaDeviceInfo& deviceInfo)
{
    int gridSize;
    int blockSize;
    calculateOccupancy(nedges, (void*)computeOutliersKernel, blockSize, gridSize);
    computeOutliersKernel<<<gridSize, blockSize, 0, deviceInfo.stream>>>(
        nedges, errorThreshold, chiValues, outliers);

    CUDA_CHECK(cudaEventRecord(deviceInfo.event, deviceInfo.stream));
    CUDA_CHECK(cudaEventSynchronize(deviceInfo.event));
    CUDA_CHECK(cudaGetLastError());
}

template <int M>
Scalar computeActiveErrors_(
    const GpuVecSe3d& poseEstimate,
    const GpuVec3d& landmarkEstimate,
    const GpuVecxd<M>& measurements,
    const GpuVec1d& omegas,
    const GpuVec2i& edge2PL,
    const GpuVec5d& cameras,
    const RobustKernel& robustKernel,
    const GpuVec1i& outliers,
    GpuVecxd<M>& errors,
    GpuVec3d& Xcs,
    Scalar* chiValues,
    Scalar* chi,
    const CudaDeviceInfo& deviceInfo)
{
}

template <>
Scalar computeActiveErrors_<2>(
    const GpuVecSe3d& poseEstimate,
    const GpuVec3d& landmarkEstimate,
    const GpuVecxd<2>& measurements,
    const GpuVec1d& omegas,
    const GpuVec2i& edge2PL,
    const GpuVec5d& cameras,
    const RobustKernel& robustKernel,
    const GpuVec1i& outliers,
    GpuVecxd<2>& errors,
    GpuVec3d& Xcs,
    Scalar* chiValues,
    Scalar* chi,
    const CudaDeviceInfo& deviceInfo)
{
    const int nedges = measurements.ssize();
    const int nomegas = omegas.ssize();
    const int ncameras = cameras.ssize();

    int blockSize;
    int gridSize;
    calculateOccupancy(nedges, (void*)computeActiveErrorsKernel<2>, blockSize, gridSize);

    CUDA_CHECK(cudaMemset(chi, 0, sizeof(Scalar)));
    CUDA_CHECK(cudaMemset(chiValues, 0, sizeof(Scalar) * nedges));

    computeActiveErrorsKernel<2><<<gridSize, blockSize, 0, deviceInfo.stream>>>(
        nedges,
        nomegas,
        ncameras,
        poseEstimate,
        landmarkEstimate,
        measurements,
        omegas,
        edge2PL,
        cameras,
        outliers,
        errors,
        Xcs,
        chiValues);

    blockSize = 512;
    gridSize = divUp(nedges, blockSize);
    const int sharedBytes = blockSize * sizeof(Scalar);

    computeChiValueKernel<<<gridSize, blockSize, sharedBytes, deviceInfo.stream>>>(
        nedges, blockSize, chiValues, outliers, chi);

    hAsyncScalar h_chi;
    h_chi.download(chi, deviceInfo.stream);

    CUDA_CHECK(cudaEventRecord(deviceInfo.event, deviceInfo.stream));
    CUDA_CHECK(cudaEventSynchronize(deviceInfo.event));
    CUDA_CHECK(cudaGetLastError());

    return *h_chi;
}

template <>
Scalar computeActiveErrors_<3>(
    const GpuVecSe3d& poseEstimate,
    const GpuVec3d& landmarkEstimate,
    const GpuVecxd<3>& measurements,
    const GpuVec1d& omegas,
    const GpuVec2i& edge2PL,
    const GpuVec5d& cameras,
    const RobustKernel& robustKernel,
    const GpuVec1i& outliers,
    GpuVecxd<3>& errors,
    GpuVec3d& Xcs,
    Scalar* chiValues,
    Scalar* chi,
    const CudaDeviceInfo& deviceInfo)
{
    const int nedges = measurements.ssize();
    const int nomegas = omegas.ssize();
    const int ncameras = cameras.ssize();

    int blockSize;
    int gridSize;
    calculateOccupancy(nedges, (void*)computeActiveErrorsKernel<3>, blockSize, gridSize);

    CUDA_CHECK(cudaMemset(chi, 0, sizeof(Scalar)));
    CUDA_CHECK(cudaMemset(chiValues, 0, sizeof(Scalar) * nedges));

    computeActiveErrorsKernel<3><<<gridSize, blockSize, 0, deviceInfo.stream>>>(
        nedges,
        nomegas,
        ncameras,
        poseEstimate,
        landmarkEstimate,
        measurements,
        omegas,
        edge2PL,
        cameras,
        outliers,
        errors,
        Xcs,
        chiValues);

    blockSize = 512;
    gridSize = divUp(nedges, blockSize);
    const int sharedBytes = blockSize * sizeof(Scalar);

    computeChiValueKernel<<<gridSize, blockSize, sharedBytes, deviceInfo.stream>>>(
        nedges, blockSize, chiValues, outliers, chi);

    hAsyncScalar h_chi;
    h_chi.download(chi, deviceInfo.stream);

    CUDA_CHECK(cudaEventRecord(deviceInfo.event, deviceInfo.stream));
    CUDA_CHECK(cudaEventSynchronize(deviceInfo.event));
    CUDA_CHECK(cudaGetLastError());

    return *h_chi;
}

template <int M>
void constructQuadraticForm_(
    const GpuVec3d& Xcs,
    const GpuVecSe3d& se3,
    GpuVecxd<M>& errors,
    const GpuVec1d& omegas,
    const GpuVec2i& edge2PL,
    const GpuVec1i& edge2Hpl,
    const GpuVec1b& flags,
    const GpuVec5d& cameras,
    const RobustKernel& robustKernel,
    const GpuVec1i& outliers,
    GpuPxPBlockVec& Hpp,
    GpuPx1BlockVec& bp,
    GpuLxLBlockVec& Hll,
    GpuLx1BlockVec& bl,
    GpuHplBlockMat& Hpl,
    const CudaDeviceInfo& deviceInfo)
{
}

template <>
void constructQuadraticForm_<2>(
    const GpuVec3d& Xcs,
    const GpuVecSe3d& se3,
    GpuVec2d& errors,
    const GpuVec1d& omegas,
    const GpuVec2i& edge2PL,
    const GpuVec1i& edge2Hpl,
    const GpuVec1b& flags,
    const GpuVec5d& cameras,
    const RobustKernel& robustKernel,
    const GpuVec1i& outliers,
    GpuPxPBlockVec& Hpp,
    GpuPx1BlockVec& bp,
    GpuLxLBlockVec& Hll,
    GpuLx1BlockVec& bl,
    GpuHplBlockMat& Hpl,
    const CudaDeviceInfo& deviceInfo)
{
    const int nedges = errors.ssize();
    const int nomegas = omegas.ssize();
    const int ncameras = cameras.ssize();
    const int block = 512;
    const int grid = divUp(nedges, block);

    constructQuadraticFormKernel<2><<<grid, block, 0, deviceInfo.stream>>>(
        nedges,
        nomegas,
        ncameras,
        Xcs,
        se3,
        errors,
        omegas,
        edge2PL,
        edge2Hpl,
        flags,
        cameras,
        outliers,
        Hpp,
        bp,
        Hll,
        bl,
        Hpl);
    CUDA_CHECK(cudaGetLastError());
}

template <>
void constructQuadraticForm_<3>(
    const GpuVec3d& Xcs,
    const GpuVecSe3d& se3,
    GpuVec3d& errors,
    const GpuVec1d& omegas,
    const GpuVec2i& edge2PL,
    const GpuVec1i& edge2Hpl,
    const GpuVec1b& flags,
    const GpuVec5d& cameras,
    const RobustKernel& robustKernel,
    const GpuVec1i& outliers,
    GpuPxPBlockVec& Hpp,
    GpuPx1BlockVec& bp,
    GpuLxLBlockVec& Hll,
    GpuLx1BlockVec& bl,
    GpuHplBlockMat& Hpl,
    const CudaDeviceInfo& deviceInfo)
{
    const int nedges = errors.ssize();
    const int nomegas = omegas.ssize();
    const int ncameras = cameras.ssize();
    const int block = 512;
    const int grid = divUp(nedges, block);

    constructQuadraticFormKernel<3><<<grid, block, 0, deviceInfo.stream>>>(
        nedges,
        nomegas,
        ncameras,
        Xcs,
        se3,
        errors,
        omegas,
        edge2PL,
        edge2Hpl,
        flags,
        cameras,
        outliers,
        Hpp,
        bp,
        Hll,
        bl,
        Hpl);
    CUDA_CHECK(cudaGetLastError());
}

template <typename T, int DIM>
Scalar maxDiagonal_(const DeviceBlockVector<T, DIM, DIM>& D, Scalar* d_maxD, Scalar * h_tmpMax, const CudaDeviceInfo& deviceInfo)
{
    const int size = D.size() * DIM;
    int gridSize;
    int blockSize;
    calculateOccupancy(size, (void*)maxDiagonalKernel<DIM>, blockSize, gridSize);
    gridSize = 4;

    int sharedBytes = sizeof(Scalar) * blockSize;

    maxDiagonalKernel<DIM><<<gridSize, blockSize, sharedBytes, deviceInfo.stream>>>(size, blockSize, D.values(), d_maxD);
    
    CUDA_CHECK(cudaMemcpyAsync(h_tmpMax, d_maxD, sizeof(Scalar) * gridSize, cudaMemcpyDeviceToHost, deviceInfo.stream));

    CUDA_CHECK(cudaEventRecord(deviceInfo.event, deviceInfo.stream));
    CUDA_CHECK(cudaEventSynchronize(deviceInfo.event));
    
    Scalar maxv = 0;
    for (int i = 0; i < gridSize; i++)
    {
        maxv = std::max(maxv, h_tmpMax[i]);
    }
    return maxv;
}

Scalar maxDiagonal(
    const GpuPxPBlockVec& Hpp, Scalar* d_maxD, Scalar* h_tmpMax, const CudaDeviceInfo& deviceInfo)
{
    return maxDiagonal_(Hpp, d_maxD, h_tmpMax, deviceInfo);
}

Scalar maxDiagonal(
    const GpuLxLBlockVec& Hll, Scalar* d_maxD, Scalar* h_tmpMax, const CudaDeviceInfo& deviceInfo)
{
    return maxDiagonal_(Hll, d_maxD, h_tmpMax, deviceInfo);
}

template <typename T, int DIM>
void addLambda_(
    DeviceBlockVector<T, DIM, DIM>& D,
    Scalar lambda,
    DeviceBlockVector<T, DIM, 1>& backup,
    const CudaDeviceInfo& deviceInfo)
{
    const int size = D.size() * DIM;

    int gridSize;
    int blockSize;
    calculateOccupancy(size, (void*)addLambdaKernel<DIM>, blockSize, gridSize);
    
    addLambdaKernel<DIM><<<gridSize, blockSize, 0, deviceInfo.stream>>>(size, D.values(), lambda, backup.values());
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaEventRecord(deviceInfo.event, deviceInfo.stream));
    CUDA_CHECK(cudaEventSynchronize(deviceInfo.event));
}

void addLambda(
    GpuPxPBlockVec& Hpp, Scalar lambda, GpuPx1BlockVec& backup, const CudaDeviceInfo& deviceInfo)
{
    addLambda_(Hpp, lambda, backup, deviceInfo);
}

void addLambda(
    GpuLxLBlockVec& Hll, Scalar lambda, GpuLx1BlockVec& backup, const CudaDeviceInfo& deviceInfo)
{
    addLambda_(Hll, lambda, backup, deviceInfo);
}

template <typename T, int DIM>
void restoreDiagonal_(
    DeviceBlockVector<T, DIM, DIM>& D,
    const DeviceBlockVector<T, DIM, 1>& backup,
    const CudaDeviceInfo& deviceInfo)
{
    const int size = D.size() * DIM;
    int gridSize;
    int blockSize;
    calculateOccupancy(size, (void*)restoreDiagonalKernel<DIM>, blockSize, gridSize);

    restoreDiagonalKernel<DIM>
        <<<gridSize, blockSize, 0, deviceInfo.stream>>>(size, D.values(), backup.values());
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaEventRecord(deviceInfo.event, deviceInfo.stream));
    CUDA_CHECK(cudaEventSynchronize(deviceInfo.event));
}

void restoreDiagonal(
    GpuPxPBlockVec& Hpp, const GpuPx1BlockVec& backup, const CudaDeviceInfo& deviceInfo)
{
    restoreDiagonal_(Hpp, backup, deviceInfo);
}

void restoreDiagonal(
    GpuLxLBlockVec& Hll, const GpuLx1BlockVec& backup, const CudaDeviceInfo& deviceInfo)
{
    restoreDiagonal_(Hll, backup, deviceInfo);
}

void computeBschure(
    const GpuPx1BlockVec& bp,
    const GpuHplBlockMat& Hpl,
    const GpuLxLBlockVec& Hll,
    const GpuLx1BlockVec& bl,
    GpuPx1BlockVec& bsc,
    GpuLxLBlockVec& invHll,
    GpuPxLBlockVec& Hpl_invHll,
    const CudaDeviceInfo& deviceInfo)
{
    const int cols = Hll.size();
    int blockSize;
    int gridSize;
    calculateOccupancy(cols, (void*)computeBschureKernel, blockSize, gridSize);

    bp.copyTo(bsc);
    computeBschureKernel<<<gridSize, blockSize, 0, deviceInfo.stream>>>(
        cols, Hll, invHll, bl, Hpl, Hpl.outerIndices(), Hpl.innerIndices(), bsc, Hpl_invHll);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaEventRecord(deviceInfo.event, deviceInfo.stream));
    CUDA_CHECK(cudaEventSynchronize(deviceInfo.event));
}

void computeHschure(
    const GpuPxPBlockVec& Hpp,
    const GpuPxLBlockVec& Hpl_invHll,
    const GpuHplBlockMat& Hpl,
    const GpuVec3i& mulBlockIds,
    GpuHscBlockMat& Hsc,
    const CudaDeviceInfo& deviceInfo)
{
    const int nmulBlocks = mulBlockIds.ssize();

    int blockSize;
    int gridSize;
    calculateOccupancy(Hsc.rows(), (void*)initializeHschurKernel, blockSize, gridSize);
   
    Hsc.fillZero();
    initializeHschurKernel<<<gridSize, blockSize, 0, deviceInfo.stream>>>(Hsc.rows(), Hpp, Hsc, Hsc.outerIndices());

    CUDA_CHECK(cudaEventRecord(deviceInfo.event, deviceInfo.stream));
    CUDA_CHECK(cudaEventSynchronize(deviceInfo.event));

    calculateOccupancy(nmulBlocks, (void*)computeHschureKernel, blockSize, gridSize);

    computeHschureKernel<<<gridSize, blockSize, 0, deviceInfo.stream>>>(
        nmulBlocks, mulBlockIds, Hpl_invHll, Hpl, Hsc);

    CUDA_CHECK(cudaEventRecord(deviceInfo.event, deviceInfo.stream));
    CUDA_CHECK(cudaEventSynchronize(deviceInfo.event));
    CUDA_CHECK(cudaGetLastError());
}

void convertHschureBSRToCSR(
    const GpuHscBlockMat& HscBSR,
    const GpuVec1i& BSR2CSR,
    GpuVec1d& HscCSR,
    const CudaDeviceInfo& deviceInfo)
{
    const int size = HscCSR.ssize();
    int blockSize;
    int gridSize;
    calculateOccupancy(size, (void*)convertBSRToCSRKernel, blockSize, gridSize);

    convertBSRToCSRKernel<<<gridSize, blockSize, 0, deviceInfo.stream>>>(size, HscBSR.values(), HscCSR, BSR2CSR);

    CUDA_CHECK(cudaEventRecord(deviceInfo.event, deviceInfo.stream));
    CUDA_CHECK(cudaEventSynchronize(deviceInfo.event));
    CUDA_CHECK(cudaGetLastError());
}

void convertHppBSRToCSR(const GpuHppBlockMat& HppBSR, const GpuVec1i& BSR2CSR, GpuVec1d& HppCSR)
{
    const int size = HppCSR.ssize();
    const int block = 1024;
    const int grid = divUp(size, block);
    convertBSRToCSRKernel<<<grid, block>>>(size, HppBSR.values(), HppCSR, BSR2CSR);
}

void twistCSR(
    int size,
    int nnz,
    const int* srcRowPtr,
    const int* srcColInd,
    const int* P,
    int* dstRowPtr,
    int* dstColInd,
    int* dstMap,
    int* nnzPerRow,
    const CudaDeviceInfo& deviceInfo)
{
    int blockSize;
    int gridSize;
    calculateOccupancy(size, (void*)permuteNnzPerRowKernel, blockSize, gridSize);

    // permute rows
    permuteNnzPerRowKernel<<<gridSize, blockSize, 0, deviceInfo.stream>>>(size, srcRowPtr, P, nnzPerRow);
    CUDA_CHECK(cudaEventRecord(deviceInfo.event, deviceInfo.stream));
    CUDA_CHECK(cudaEventSynchronize(deviceInfo.event));

    exclusiveScan(nnzPerRow, dstRowPtr, size + 1);

    CUDA_CHECK(
        cudaMemcpy(nnzPerRow, dstRowPtr, sizeof(int) * (size + 1), cudaMemcpyDeviceToDevice));

    // permute cols
    calculateOccupancy(size, (void*)permuteColIndKernel, blockSize, gridSize);
    permuteColIndKernel<<<gridSize, blockSize, 0, deviceInfo.stream>>>(
        size, srcRowPtr, srcColInd, P, dstColInd, dstMap, nnzPerRow);
    
    CUDA_CHECK(cudaEventRecord(deviceInfo.event, deviceInfo.stream));
    CUDA_CHECK(cudaEventSynchronize(deviceInfo.event));
    CUDA_CHECK(cudaGetLastError());
}

void permute(int size, const Scalar* src, Scalar* dst, const int* P)
{
    auto ptrSrc = thrust::device_pointer_cast(src);
    auto ptrDst = thrust::device_pointer_cast(dst);
    auto ptrMap = thrust::device_pointer_cast(P);
    thrust::gather(ptrMap, ptrMap + size, ptrSrc, ptrDst);
}

void schurComplementPost(
    const GpuLxLBlockVec& invHll,
    const GpuLx1BlockVec& bl,
    const GpuHplBlockMat& Hpl,
    const GpuPx1BlockVec& xp,
    GpuLx1BlockVec& xl,
    const CudaDeviceInfo& deviceInfo)
{
    int blockSize;
    int gridSize;
    calculateOccupancy(Hpl.cols(), (void*)schurComplementPostKernel, blockSize, gridSize);

    schurComplementPostKernel<<<gridSize, blockSize, 0, deviceInfo.stream>>>(
        Hpl.cols(), invHll, bl, Hpl, Hpl.outerIndices(), Hpl.innerIndices(), xp, xl);

    CUDA_CHECK(cudaEventRecord(deviceInfo.event, deviceInfo.stream));
    CUDA_CHECK(cudaEventSynchronize(deviceInfo.event));
    CUDA_CHECK(cudaGetLastError());
}

void updatePoses(const GpuPx1BlockVec& xp, GpuVecSe3d& estimate, const CudaDeviceInfo& deviceInfo)
{
    int gridSize;
    int blockSize;
    calculateOccupancy(xp.size(), (void*)updatePosesKernel, blockSize, gridSize);
   
    updatePosesKernel<<<gridSize, blockSize, 0, deviceInfo.stream>>>(xp.size(), xp, estimate);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaEventRecord(deviceInfo.event, deviceInfo.stream));
    CUDA_CHECK(cudaEventSynchronize(deviceInfo.event));
}

void updateLandmarks(const GpuLx1BlockVec& xl, GpuVec3d& estimate, const CudaDeviceInfo& deviceInfo)
{
    int gridSize;
    int blockSize;
    calculateOccupancy(xl.size(), (void*)updateLandmarksKernel, blockSize, gridSize);

    updateLandmarksKernel<<<gridSize, blockSize, 0, deviceInfo.stream>>>(xl.size(), xl, estimate);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaEventRecord(deviceInfo.event, deviceInfo.stream));
    CUDA_CHECK(cudaEventSynchronize(deviceInfo.event));
}

void computeScale(
    const GpuVec1d& x,
    const GpuVec1d& b,
    Scalar* scale,
    Scalar lambda,
    const CudaDeviceInfo& deviceInfo)
{
    const int blockSize = 1024;
    const int gridSize = 4;
    const int sharedBytes = blockSize * sizeof(Scalar);

    CUDA_CHECK(cudaMemset(scale, 0, sizeof(Scalar)));

    computeScaleKernel<<<gridSize, blockSize, sharedBytes, deviceInfo.stream>>>(x, b, scale, lambda, x.ssize(), blockSize);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaEventRecord(deviceInfo.event, deviceInfo.stream));
    CUDA_CHECK(cudaEventSynchronize(deviceInfo.event));
}

////////////////////////////////////////////////////////////////////////////////////
// ICP custom edge - Jacobian functions
////////////////////////////////////////////////////////////////////////////////////

__device__ inline Matx<Scalar, 1, 6>
computeJacobians_Plane(const Se3D& est, const PointToPlaneMatch<double>& measurement)
{
    Vec3d Pw;
    se3MulVec(est, measurement.pointP, Pw);

    Matx<Scalar, 1, 3> Jdot;
    Jdot(0, 0) = measurement.normal[0];
    Jdot(0, 1) = measurement.normal[1];
    Jdot(0, 2) = measurement.normal[2];

    // LEFT side is negative skew symmetric of Y
    // where Y = SE(3) * point = Pw
    Matx<Scalar, 3, 3> Yx;
    skew1(Pw[0], Pw[1], Pw[2], Yx);

    Matx<Scalar, 3, 6> Jse3exp1;
    /* clang-format off */ 
    Jse3exp1(0, 0) = Yx(0, 0); 
    Jse3exp1(1, 0) = Yx(1, 0);  
    Jse3exp1(2, 0) = Yx(2, 0); 
    Jse3exp1(0, 1) = 1.0; 
    Jse3exp1(1, 1) = 0.0; 
    Jse3exp1(2, 1) = 0.0; 
    Jse3exp1(0, 2) = Yx(0, 1);
    Jse3exp1(1, 2) = Yx(1, 1);
    Jse3exp1(2, 2) = Yx(2, 1);
    Jse3exp1(0, 3) = 0.0; 
    Jse3exp1(1, 3) = 1.0; 
    Jse3exp1(2, 3) = 0.0; 
    Jse3exp1(0, 4) = Yx(0, 2); 
    Jse3exp1(1, 4) = Yx(1, 2);
    Jse3exp1(2, 4) = Yx(2, 2);
    Jse3exp1(0, 5) = 0.0;
    Jse3exp1(1, 5) = 0.0;
    Jse3exp1(2, 5) = 1.0;
    /* clang-format on */

    // Multiply them all together and we're done (maybe!)
    Matx<Scalar, 1, 6> J = Vec1x3MulMat3x6(Jse3exp1.data, Jdot.data);
    return J;
}

__device__ inline Matx<Scalar, 1, 6>
computeJacobians_Line(const Se3D& est, const PointToLineMatch<double>& measurement)
{
    Vec3d Pw;
    se3MulVec(est, measurement.pointP, Pw);

    Vec3d A = Pw; //-measurement.start();
    Vec3d B = Pw; //-measurement.end();

    Vec3d crossAB;
    cross(A, B, crossAB);
    Scalar normCrossAB = norm(crossAB);

    Matx<Scalar, 1, 3> Jnorm;
    if (normCrossAB > 1e-6f)
    {
        Scalar invNormCrossAB = 1.0 / normCrossAB;
        Jnorm(0, 0) = crossAB[0] * invNormCrossAB;
        Jnorm(0, 1) = crossAB[1] * invNormCrossAB;
        Jnorm(0, 2) = crossAB[2] * invNormCrossAB;
    }

    // LEFT side is negative skew symmetric of Y
    // where Y = SE(3) * point = Pw
    Matx<Scalar, 3, 3> Yx;
    skew1(Pw[0], Pw[1], Pw[2], Yx);

    Matx<Scalar, 3, 6> Jse3exp1;
    /* clang-format off */ 
    Jse3exp1(0, 0) = Yx(0, 0); 
    Jse3exp1(1, 0) = Yx(1, 0);  
    Jse3exp1(2, 0) = Yx(2, 0); 
    Jse3exp1(0, 1) = 1.0; 
    Jse3exp1(1, 1) = 0.0; 
    Jse3exp1(2, 1) = 0.0; 
    Jse3exp1(0, 2) = Yx(0, 1);
    Jse3exp1(1, 2) = Yx(1, 1);
    Jse3exp1(2, 2) = Yx(2, 1);
    Jse3exp1(0, 3) = 0.0; 
    Jse3exp1(1, 3) = 1.0; 
    Jse3exp1(2, 3) = 0.0; 
    Jse3exp1(0, 4) = Yx(0, 2); 
    Jse3exp1(1, 4) = Yx(1, 2);
    Jse3exp1(2, 4) = Yx(2, 2);
    Jse3exp1(0, 5) = 0.0;
    Jse3exp1(1, 5) = 0.0;
    Jse3exp1(2, 5) = 1.0;
    /* clang-format on */

    Matx<Scalar, 3, 3> Ax, Bx;
    skew1(A[0], A[1], A[2], Ax);
    skew1(B[0], B[1], B[2], Bx);

    Matx<Scalar, 3, 3> ABdiff; // = Ax - Bx;
    Matx<Scalar, 3, 6> Jcross;
    MatMulMat<3, 3, 6>(ABdiff.data, Jse3exp1.data, Jcross.data);

    // assemble everything
    Matx<Scalar, 1, 6> J = Vec1x3MulMat3x6(Jcross.data, Jnorm.data);
    return J;
    //_jacobianOplusXi = Jnorm * Jcross / measurement.start() - measurement.end();
}

////////////////////////////////////////////////////////////////////////////////////
// BA custom edge - Kernel functions
////////////////////////////////////////////////////////////////////////////////////

__global__ void computeActiveErrorsKernel_DepthBa(
    int nedges,
    int nomegas,
    int ncameras,
    const Se3D* se3,
    const Vec3d* Xws,
    const Vec3d* measurements,
    const Scalar* omegas,
    const Vec2i* edge2PL,
    const Vec5d* cameras,
    const Scalar errorThreshold,
    Vec3d* errors,
    int* outliers,
    Vec3d* Xcs,
    Scalar* chi)
{
    Scalar sumchi = 0;
    for (int iE = blockIdx.x * blockDim.x + threadIdx.x; iE < nedges; iE += gridDim.x * blockDim.x)
    {
        const Scalar omega = (nomegas > 1) ? omegas[iE] : omegas[0];
        const int iP = edge2PL[iE][0];
        const int iL = edge2PL[iE][1];

        const QuatD& q = se3[iP].r;
        const Vec3d& t = se3[iP].t;
        const Vec3d& Xw = Xws[iL];
        const Vec3d& measurement = measurements[iE];
        const Vec5d& camera = (ncameras > 1) ? cameras[iE] : cameras[0];

        // project world to camera
        Vec3d Xc;
        projectW2C(q, t, Xw, Xc);

        // project camera to image
        Vec3d proj;
        camProjectDepth(Xc, camera, proj);

        // compute residual
        Vec3d error;
        error[0] = measurement[0] - proj[0];
        error[1] = measurement[1] - proj[1];
        error[2] = measurement[2] - proj[2];
        errors[iE] = error;
        Xcs[iE] = Xc;

        const Scalar chi2 = robustKernelFunc->robustify(omega * squaredNorm(error));
        sumchi += chi2;
        if (errorThreshold > 0.0 && chi2 > errorThreshold)
        {
            outliers[iE] = 1;
        }
    }

    const int sharedIdx = threadIdx.x;
    __shared__ Scalar cache[BLOCK_ACTIVE_ERRORS];

    cache[sharedIdx] = sumchi;
    __syncthreads();

    for (int stride = BLOCK_ACTIVE_ERRORS / 2; stride > 0; stride >>= 1)
    {
        if (sharedIdx < stride)
        {
            cache[sharedIdx] += cache[sharedIdx + stride];
        }
        __syncthreads();
    }

    if (sharedIdx == 0)
    {
        atomicAdd(chi, cache[0]);
    }
}

////////////////////////////////////////////////////////////////////////////////////
// ICP custom edge - Kernel functions
////////////////////////////////////////////////////////////////////////////////////

__global__ void computeActiveErrorsKernel_Line(
    int nedges,
    int nomegas,
    const Se3D* poseEstimate,
    const PointToLineMatch<double>* measurements,
    const Scalar* omegas,
    const Vec2i* edge2PL,
    Scalar* errors,
    Vec3d* Xcs,
    Scalar* chi)
{
    const int iE = blockIdx.x * blockDim.x + threadIdx.x;
    if (iE >= nedges)
    {
        return;
    }

    const Vec2i index = edge2PL[iE];
    const int iP = index[0];

    const Se3D est = poseEstimate[iP];
    const PointToLineMatch<double> measurement = measurements[iE];

    Vec3d Pw;
    se3MulVec(est, measurement.pointP, Pw);

    // compute residual
    Scalar error = distance(Pw, measurement.a, measurement.b, measurement.length);
    errors[iE] = error;
    atomicAdd(chi, errors[iE]);
}

__global__ void computeActiveErrorsKernel_Plane(
    int nedges,
    int nomegas,
    int blockSize,
    const Se3D* poseEstimate,
    const PointToPlaneMatch<double>* measurements,
    const Scalar* omegas,
    const Vec2i* edge2PL,
    Scalar* errors,
    Vec3d* Xcs,
    Scalar* chi)
{
    Scalar sumchi = 0;
    
    extern __shared__ Scalar cache[];

    const int tid = threadIdx.x;
    const int gridSize = blockSize * gridDim.x;
    int iE = blockIdx.x * blockDim.x + threadIdx.x;

    while (iE < nedges)
    {
        const Scalar omega = (nomegas > 1) ? omegas[iE] : omegas[0];
        const int iP = edge2PL[iE][0];
        const Se3D est = poseEstimate[iP];
        const PointToPlaneMatch<double> measurement = measurements[iE];

        Vec3d Pw;
        se3MulVec(est, measurement.pointP, Pw);

        // compute residual
        Scalar error = signedDistance(Pw, measurement.normal, measurement.originDistance);
        errors[iE] = error;
        sumchi += (errors[iE] * errors[iE]) * omega;

        iE += gridSize;
    } 
   
    parallelReductionAndAdd(cache, blockSize, tid, sumchi, chi);
}

template <int MDIM>
__global__ void constructQuadraticFormKernel_Plane(
    int nedges,
    int nomegas,
    const Se3D* se3,
    const Scalar* errors,
    const PointToPlaneMatch<double>* measurements,
    const Scalar* omegas,
    const Vec2i* edge2PL,
    const int* edge2Hpl,
    const uint8_t* flags,
    PxPBlockPtr Hpp,
    Px1BlockPtr bp,
    LxLBlockPtr Hll,
    Lx1BlockPtr bl,
    PxLBlockPtr Hpl)
{
    const int iE = blockIdx.x * blockDim.x + threadIdx.x;
    if (iE >= nedges)
    {
        return;
    }
    const Scalar omega = (nomegas > 1) ? omegas[iE] : omegas[0];
    const int iP = edge2PL[iE][0];
    const int flag = flags[iE];
    const PointToPlaneMatch<double> measurement = measurements[iE];

    const Se3D rt = se3[iP];
    const Scalar error = errors[iE];

    // compute Jacobians
    Matx<Scalar, MDIM, PDIM> JP = computeJacobians_Plane(rt, measurement);

    if (!(flag & EDGE_FLAG_FIXED_P))
    {
        // Hpp += JPT*Ω*JP
        MatTMulMat<PDIM, MDIM, PDIM, ACCUM_ATOMIC>(JP.data, JP.data, Hpp.at(iP), omega);
        // bp -= JPT*Ω*r
        MatTMulVec<PDIM, MDIM, DEACCUM_ATOMIC>(JP.data, &error, bp.at(iP), omega);
    }
}

template <int MDIM>
__global__ void constructQuadraticFormKernel_Line(
    int nedges,
    int nomegas,
    const Se3D* se3,
    const Scalar* errors,
    const PointToLineMatch<double>* measurements,
    const Scalar* omegas,
    const Vec2i* edge2PL,
    const int* edge2Hpl,
    const uint8_t* flags,
    PxPBlockPtr Hpp,
    Px1BlockPtr bp,
    LxLBlockPtr Hll,
    Lx1BlockPtr bl,
    PxLBlockPtr Hpl)
{
    const int iE = blockIdx.x * blockDim.x + threadIdx.x;
    if (iE >= nedges)
    {
        return;
    }
    const Scalar omega = (nomegas > 1) ? omegas[iE] : omegas[0];
    const int iP = edge2PL[iE][0];
    const int flag = flags[iE];
    const PointToLineMatch<double> measurement = measurements[iE];

    const Se3D& rt = se3[iP];
    Scalar error = errors[iE];

    // compute Jacobians
    Matx<Scalar, 1, PDIM> JP = computeJacobians_Line(rt, measurement);

    if (!(flag & EDGE_FLAG_FIXED_P))
    {
        // Hpp += = JPT*Ω*JP
        MatTMulMat<PDIM, MDIM, PDIM, ACCUM_ATOMIC>(JP.data, JP.data, Hpp.at(iP), omega);

        // bp += = JPT*Ω*r
        MatTMulVec<PDIM, MDIM, DEACCUM_ATOMIC>(JP.data, &error, bp.at(iP), omega);
    }
}

////////////////////////////////////////////////////////////////////////////////////
// BA custom edge - Wrapper functions
////////////////////////////////////////////////////////////////////////////////////

Scalar computeActiveErrors_DepthBa(
    const GpuVecSe3d& poseEstimate,
    const GpuVec3d& landmarkEstimate,
    const GpuVec3d& measurements,
    const GpuVec1d& omegas,
    const GpuVec2i& edge2PL,
    const GpuVec5d& cameras,
    const Scalar errorThreshold,
    const RobustKernel& robustKernel,
    GpuVec3d& errors,
    GpuVec1i& outliers,
    GpuVec3d& Xcs,
    Scalar* chi,
    const CudaDeviceInfo& deviceInfo)
{
    const int nedges = measurements.ssize();
    const int nomegas = omegas.ssize();
    const int ncameras = cameras.ssize();
    const int block = BLOCK_ACTIVE_ERRORS;
    const int grid = 16;

    if (errorThreshold > 0.0)
    {
        CUDA_CHECK(cudaMemset(outliers.data(), 0, nedges * sizeof(int)));
    }
    CUDA_CHECK(cudaMemset(chi, 0, sizeof(Scalar)));
    
    computeActiveErrorsKernel_DepthBa<<<grid, block, 0, deviceInfo.stream>>>(
        nedges,
        nomegas,
        ncameras,
        poseEstimate,
        landmarkEstimate,
        measurements,
        omegas,
        edge2PL,
        cameras,
        errorThreshold,
        errors,
        outliers,
        Xcs,
        chi);
    CUDA_CHECK(cudaGetLastError());
    
    Scalar h_chi = 0;
    CUDA_CHECK(cudaMemcpy(&h_chi, chi, sizeof(Scalar), cudaMemcpyDeviceToHost));
  
    return h_chi;
}

////////////////////////////////////////////////////////////////////////////////////
// ICP custom edge - Wrapper functions
////////////////////////////////////////////////////////////////////////////////////
Scalar computeActiveErrors_Line(
    const GpuVecSe3d& poseEstimate,
    const GpuVec<PointToLineMatch<double>>& measurements,
    const GpuVec1d& omegas,
    const GpuVec2i& edge2PL,
    GpuVec1d& errors,
    GpuVec3d& Xcs,
    Scalar* chi)
{
    const int nedges = measurements.ssize();
    const int nomegas = omegas.ssize();
    const int block = BLOCK_ACTIVE_ERRORS;
    const int grid = divUp(nedges, block);

    CUDA_CHECK(cudaMemset(chi, 0, sizeof(Scalar)));
    computeActiveErrorsKernel_Line<<<grid, block>>>(
        nedges, nomegas, poseEstimate, measurements, omegas, edge2PL, errors, Xcs, chi);
    CUDA_CHECK(cudaGetLastError());

    Scalar h_chi = 0;
    CUDA_CHECK(cudaMemcpy(&h_chi, chi, sizeof(Scalar), cudaMemcpyDeviceToHost));

    return h_chi;
}

Scalar computeActiveErrors_Plane(
    const GpuVecSe3d& poseEstimate,
    const GpuVec<PointToPlaneMatch<double>>& measurements,
    const GpuVec1d& omegas,
    const GpuVec2i& edge2PL,
    GpuVec1d& errors,
    GpuVec3d& Xcs,
    Scalar* chi,
    const CudaDeviceInfo& deviceInfo)
{
    const int nedges = measurements.ssize();
    const int nomegas = omegas.ssize();
    
    // Using a fixed thread number here as otherwise if the occupancy is
    // calculated at runtime based on the system, this will lead to
    // inconsistent results from the kernel. Maybe due to shared buffer
    // issues at larger block size (more shared memory used / less iterations
    // of the kernel)
    const int blockSize = 512;
    const int gridSize = divUp(nedges, blockSize);
    const int sharedBytes = blockSize * sizeof(Scalar);

    CUDA_CHECK(cudaMemset(chi, 0, sizeof(Scalar)));
    computeActiveErrorsKernel_Plane<<<gridSize, blockSize, sharedBytes, deviceInfo.stream>>>(
        nedges, nomegas, blockSize, poseEstimate, measurements, omegas, edge2PL, errors, Xcs, chi);
  
    hAsyncScalar h_chi;
    h_chi.download(chi, deviceInfo.stream);

    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaEventRecord(deviceInfo.event, deviceInfo.stream));
    CUDA_CHECK(cudaEventSynchronize(deviceInfo.event));
    
    return *h_chi;
}

void constructQuadraticForm_Plane(
    const GpuVecSe3d& se3,
    GpuVec1d& errors,
    const GpuVec<PointToPlaneMatch<double>>& measurements,
    const GpuVec1d& omegas,
    const GpuVec2i& edge2PL,
    const GpuVec1i& edge2Hpl,
    const GpuVec1b& flags,
    GpuPxPBlockVec& Hpp,
    GpuPx1BlockVec& bp,
    GpuLxLBlockVec& Hll,
    GpuLx1BlockVec& bl,
    GpuHplBlockMat& Hpl,
    const CudaDeviceInfo& deviceInfo)
{
    const int nedges = errors.ssize();
    const int nomegas = omegas.ssize();

    int gridSize;
    int blockSize;
    calculateOccupancy(nedges, (void*)constructQuadraticFormKernel_Plane<1>, blockSize, gridSize);

    constructQuadraticFormKernel_Plane<1><<<gridSize, blockSize, 0, deviceInfo.stream>>>(
        nedges,
        nomegas,
        se3,
        errors,
        measurements,
        omegas,
        edge2PL,
        edge2Hpl,
        flags,
        Hpp,
        bp,
        Hll,
        bl,
        Hpl);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaEventRecord(deviceInfo.event, deviceInfo.stream));
    CUDA_CHECK(cudaEventSynchronize(deviceInfo.event));
}

void constructQuadraticForm_Line(
    const GpuVecSe3d& se3,
    GpuVec1d& errors,
    const GpuVec<PointToLineMatch<double>>& measurements,
    const GpuVec1d& omegas,
    const GpuVec2i& edge2PL,
    const GpuVec1i& edge2Hpl,
    const GpuVec1b& flags,
    GpuPxPBlockVec& Hpp,
    GpuPx1BlockVec& bp,
    GpuLxLBlockVec& Hll,
    GpuLx1BlockVec& bl,
    GpuHplBlockMat& Hpl)
{
    const int nedges = errors.ssize();
    const int nomegas = omegas.ssize();
    const int block = 512;
    const int grid = divUp(nedges, block);

    constructQuadraticFormKernel_Line<1><<<grid, block>>>(
        nedges,
        nomegas,
        se3,
        errors,
        measurements,
        omegas,
        edge2PL,
        edge2Hpl,
        flags,
        Hpp,
        bp,
        Hll,
        bl,
        Hpl);
    CUDA_CHECK(cudaGetLastError());
}

} // namespace gpu
} // namespace cugo
