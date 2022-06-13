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

#ifndef __FIXED_VECTOR_H__
#define __FIXED_VECTOR_H__

#include "maths.h"
#include "scalar.h"

#include <cuda_runtime.h>

#include <cassert>
#include <cstdint>

namespace cugo
{
#define HOST_DEVICE __host__ __device__ inline

template <typename T, int N>
struct Vec
{
    HOST_DEVICE Vec() {}
    HOST_DEVICE Vec(const T* values)
    {
        for (int i = 0; i < N; i++)
            data[i] = values[i];
    }

    HOST_DEVICE Vec(maths::Vec<T, N>& vec)
    {
        const T* values = vec.data();
        for (int i = 0; i < N; i++)
            data[i] = values[i];
    }

    template <typename U>
    HOST_DEVICE Vec(const U* values)
    {
        for (int i = 0; i < N; i++)
            data[i] = T(values[i]);
    }

    HOST_DEVICE T& operator[](int i) { return data[i]; }
    HOST_DEVICE const T& operator[](int i) const { return data[i]; }

    HOST_DEVICE Vec<T, N> operator-(const Vec<T, N>& other)
    {
        Vec<T, N> output;
        output[0] = this->data[0] - other[0];
        output[1] = this->data[1] - other[1];
        output[2] = this->data[2] - other[2];
        return output;
    }

    HOST_DEVICE void copyTo(T* rhs) const
    {
        for (int i = 0; i < N; i++)
            rhs[i] = data[i];
    }

    HOST_DEVICE void copyTo(maths::Vec<T, N>& rhs)
    {
        T* vec = rhs.data();
        for (int i = 0; i < N; i++)
        {
            vec[i] = data[i];
        }
    }

    template <typename U>
    HOST_DEVICE void copyTo(U* rhs) const
    {
        for (int i = 0; i < N; i++)
            rhs[i] = U(data[i]);
    }

    T data[N];
};

template <typename T, int N>
__device__ inline Vec<T, N> operator*(const T& f, const Vec<T, N>& m)
{
    Vec<T, N> result;
    for (int idx = 0; idx < N; ++idx)
    {
        result.data[idx] = f * m.data[idx];
    }
    return result;
}

using Vec2 = Vec<Scalar, 2>;
using Vec3 = Vec<Scalar, 3>;
using Vec4 = Vec<Scalar, 4>;
using Vec6 = Vec<Scalar, 6>;

using Vec1d = Vec<double, 1>;
using Vec2d = Vec<double, 2>;
using Vec3d = Vec<double, 3>;
using Vec4d = Vec<double, 4>;
using Vec6d = Vec<double, 6>;

template <int DIM>
using VecNd = Vec<double, DIM>;

using Vec1i = Vec<int, 1>;
using Vec2i = Vec<int, 2>;
using Vec3i = Vec<int, 3>;
using Vec4i = Vec<int, 4>;

template <int DIM>
using VecNi = Vec<int, DIM>;


template <typename T>
struct Quat
{
    HOST_DEVICE Quat() {}
    HOST_DEVICE Quat(const T* values)
    {
        for (int i = 0; i < 4; i++)
            data[i] = values[i];
    }

    template <typename U>
    HOST_DEVICE Quat(const U* values)
    {
        for (int i = 0; i < 4; i++)
            data[i] = T(values[i]);
    }

    HOST_DEVICE T& operator[](int i)
    {
        assert(i < 4);
        return data[i];
    }
    HOST_DEVICE const T& operator[](int i) const
    {
        assert(i < 4);
        return data[i];
    }

    HOST_DEVICE void copyTo(T* rhs) const
    {
        for (int i = 0; i < 4; i++)
            rhs[i] = data[i];
    }

    template <typename U>
    HOST_DEVICE void copyTo(U* rhs) const
    {
        for (int i = 0; i < 4; i++)
            rhs[i] = U(data[i]);
    }

    T data[4];
};

using QuatF = Quat<float>;
using QuatD = Quat<double>;

template <typename T>
struct Se3
{
    HOST_DEVICE Se3() {}
    HOST_DEVICE Se3(const T* rValues, const T* tValues)
    {
        for (int i = 0; i < 4; i++)
        {
            r.data[i] = rValues[i];
        }
        for (int i = 0; i < 3; i++)
        {
            t.data[i] = tValues[i];
        }
    }

    template <typename U>
    HOST_DEVICE Se3(const U* rValues, const U* tValues)
    {
        for (int i = 0; i < 4; i++)
        {
            r.data[i] = T(rValues[i]);
        }
        for (int i = 0; i < 3; i++)
        {
            t.data[i] = T(tValues[i]);
        }
    }

    HOST_DEVICE Se3(const maths::Se3<T>& se3)
    {
        const T* rValues = se3.r.coeffs().data();
        for (int i = 0; i < 4; i++)
        {
            r.data[i] = rValues[i];
        }

        const T* tValues = se3.t.data();
        for (int i = 0; i < 3; i++)
        {
            t.data[i] = tValues[i];
        }
    }

    HOST_DEVICE void copyTo(T* r_rhs, T* t_rhs) const
    {
        for (int i = 0; i < 4; i++)
        {
            r_rhs[i] = r.data[i];
        }
        for (int i = 0; i < 3; i++)
        {
            t_rhs[i] = t.data[i];
        }
    }

    HOST_DEVICE void copyTo(maths::Se3<T>& rhs) const
    {
        T* rhs_r = rhs.r.coeffs().data();
        for (int i = 0; i < 4; i++)
        {
            rhs_r[i] = r.data[i];
        }
        T* rhs_t = rhs.t.data();
        for (int i = 0; i < 3; i++)
        {
            rhs_t[i] = t.data[i];
        }
    }

    template <typename U>
    HOST_DEVICE void copyTo(U* r_rhs, U* t_rhs) const
    {
        for (int i = 0; i < 4; i++)
        {
            r_rhs[i] = U(r.data[i]);
        }
        for (int i = 0; i < 3; i++)
        {
            t_rhs[i] = U(t.data[i]);
        }
    }

    Quat<T> r;
    Vec<T, 3> t;
};

using Se3F = Se3<float>;
using Se3D = Se3<double>;

} // namespace cugo

#endif // !__FIXED_VECTOR_H__
