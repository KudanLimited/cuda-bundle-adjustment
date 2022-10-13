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

#pragma once

#include "scalar.h"

#include <cassert>
#include <cstdint>


namespace cugo
{
#define HOST_DEVICE __host__ __device__ inline

template <typename T, int N>
struct Vec
{
    HOST_DEVICE Vec() {}
    HOST_DEVICE Vec(const T* values) noexcept
    {
        #pragma unroll
        for (int i = 0; i < N; i++)
        {
            data[i] = values[i];
        }
    }

    template <typename U>
    HOST_DEVICE Vec(const U* values) noexcept 
    {
        #pragma unroll
        for (int i = 0; i < N; i++)
        {
            data[i] = T(values[i]);
        }
    }

    HOST_DEVICE Vec(const Vec<T, N>& vec) noexcept
    {
        #pragma unroll
        for (int i = 0; i < N; i++)
        {
            data[i] = vec[i];
        }
    }

    HOST_DEVICE Vec(const T& x, const T& y, const T& z) noexcept
    {
        data[0] = x;
        data[1] = y;
        data[2] = z;
    }

     HOST_DEVICE Vec(T x, T y, T z) noexcept
    {
        data[0] = x;
        data[1] = y;
        data[2] = z;
    }

    HOST_DEVICE T& operator[](int i) noexcept { return data[i]; }
    HOST_DEVICE const T& operator[](int i) const noexcept { return data[i]; }

    HOST_DEVICE Vec<T, N> operator-(const Vec<T, N>& other) noexcept
    {
        Vec<T, N> output;
        output[0] = this->data[0] - other[0];
        output[1] = this->data[1] - other[1];
        output[2] = this->data[2] - other[2];
        return output;
    }

    HOST_DEVICE void copyTo(T* rhs) const noexcept
    {
        #pragma unroll
        for (int i = 0; i < N; i++)
        {
            rhs[i] = data[i];
        }
    }

    template <typename U>
    HOST_DEVICE void copyTo(U* rhs) const noexcept
    {
        #pragma unroll
        for (int i = 0; i < N; i++)
        {
            rhs[i] = U(data[i]);
        }
    }

    T data[N];
};

template <typename T, int N>
__device__ inline Vec<T, N> operator*(const T& f, const Vec<T, N>& m) noexcept
{
    Vec<T, N> result;
    for (int idx = 0; idx < N; ++idx)
    {
        result.data[idx] = f * m.data[idx];
    }
    return result;
}

using Vec1 = Vec<Scalar, 2>;
using Vec2 = Vec<Scalar, 2>;
using Vec3 = Vec<Scalar, 3>;
using Vec4 = Vec<Scalar, 4>;
using Vec5 = Vec<Scalar, 5>;
using Vec6 = Vec<Scalar, 6>;

using Vec1d = Vec<double, 1>;
using Vec2d = Vec<double, 2>;
using Vec3d = Vec<double, 3>;
using Vec4d = Vec<double, 4>;
using Vec5d = Vec<double, 5>;
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
    HOST_DEVICE Quat(const T* values) noexcept
    {
        #pragma unroll
        for (int i = 0; i < 4; i++)
        {
            data[i] = values[i];
        }
    }

    template <typename U>
    HOST_DEVICE Quat(const U* values) noexcept
    {
        for (int i = 0; i < 4; i++)
            data[i] = T(values[i]);
    }

    HOST_DEVICE Quat(const Vec<T, 4>& vec) noexcept
    {
        #pragma unroll
        for (int i = 0; i < 4; i++)
        {
            data[i] = vec[i];
        }
    }

    HOST_DEVICE Quat(const Quat<T>& quat) noexcept
    {
        #pragma unroll
        for (int i = 0; i < 4; i++)
        {
            data[i] = quat[i];
        }
    }

    HOST_DEVICE Quat(const T& x, const T& y, const T& z, const T& w) noexcept
    {
        data[0] = x;
        data[1] = y;
        data[2] = z;
        data[3] = w;
    }

    HOST_DEVICE Quat(T x, T y, T z, T w) noexcept
    {
        data[0] = x;
        data[1] = y;
        data[2] = z;
        data[3] = w;
    }

    HOST_DEVICE T& operator[](int i) noexcept
    {
        assert(i < 4);
        return data[i];
    }
    HOST_DEVICE const T& operator[](int i) const noexcept
    {
        assert(i < 4);
        return data[i];
    }

    HOST_DEVICE void copyTo(T* rhs) const noexcept
    {
        #pragma unroll
        for (int i = 0; i < 4; i++)
        {
            rhs[i] = data[i];
        }
    }

    template <typename U>
    HOST_DEVICE void copyTo(U* rhs) const noexcept
    {
        #pragma unroll
        for (int i = 0; i < 4; i++)
        {
            rhs[i] = U(data[i]);
        }
    }

    T data[4];
};

using QuatF = Quat<float>;
using QuatD = Quat<double>;

template <typename T>
struct Se3
{
    HOST_DEVICE Se3() {}
    HOST_DEVICE Se3(const T* rValues, const T* tValues) noexcept
    {
        #pragma unroll
        for (int i = 0; i < 4; i++)
        {
            r.data[i] = rValues[i];
        }
        #pragma unroll
        for (int i = 0; i < 3; i++)
        {
            t.data[i] = tValues[i];
        }
    }

    template <typename U>
    HOST_DEVICE Se3(const U* rValues, const U* tValues) noexcept
    {
        #pragma unroll
        for (int i = 0; i < 4; i++)
        {
            r.data[i] = T(rValues[i]);
        }
        #pragma unroll
        for (int i = 0; i < 3; i++)
        {
            t.data[i] = T(tValues[i]);
        }
    }

    HOST_DEVICE void copyTo(T* r_rhs, T* t_rhs) const noexcept
    {
        #pragma unroll
        for (int i = 0; i < 4; i++)
        {
            r_rhs[i] = r.data[i];
        }
        #pragma unroll
        for (int i = 0; i < 3; i++)
        {
            t_rhs[i] = t.data[i];
        }
    }

    template <typename U>
    HOST_DEVICE void copyTo(U* r_rhs, U* t_rhs) const noexcept
    {
        #pragma unroll
        for (int i = 0; i < 4; i++)
        {
            r_rhs[i] = U(r.data[i]);
        }
        #pragma unroll
        for (int i = 0; i < 3; i++)
        {
            t_rhs[i] = U(t.data[i]);
        }
    }

    HOST_DEVICE Se3(const Quat<T>& quat, const Vec<T, 3>& vec) noexcept
    {
    #pragma unroll
        for (int i = 0; i < 4; i++)
        {
            r.data[i] = quat[i];
        }
    #pragma unroll
        for (int i = 0; i < 3; i++)
        {
            t.data[i] = vec[i];
        }
    }

    Quat<T> r;
    Vec<T, 3> t;
};

using Se3F = Se3<float>;
using Se3D = Se3<double>;

/**
 * @brief A simplified vector object that uses a pinned memory allocation
 * for async methods.
 * @tparam T The type of values to be stored.
 * @tparam N The dimension of the vector;
*/
template<typename T, int N>
struct HostAsyncVec
{
public:

    HostAsyncVec() noexcept
    { 
        CUDA_CHECK(cudaHostAlloc((void**)&values_, sizeof(T) * N, cudaHostAllocDefault));
    }

    ~HostAsyncVec() noexcept 
    { 
        CUDA_CHECK(cudaFreeHost(values_));
    }

    T operator[](int idx) noexcept { return values_[idx]; }
    T operator[](int idx) const noexcept { return values_[idx]; }
    T operator*() const noexcept { return *values_; }

    T* values() const noexcept { return values_;  }

    void download(T* d_src, const cudaStream_t stream = 0) 
    {
        CUDA_CHECK(cudaMemcpyAsync(values_, d_src, sizeof(T) * N, cudaMemcpyDeviceToHost, stream));
    }

private:

    T* values_;
};

using hAsyncScalar = HostAsyncVec<Scalar, 1>;
using hAsyncVec2 = HostAsyncVec<Scalar, 2>;
using hAsyncVec3 = HostAsyncVec<Scalar, 3>;
using hAsyncVec4 = HostAsyncVec<Scalar, 4>;


} // namespace cugo

