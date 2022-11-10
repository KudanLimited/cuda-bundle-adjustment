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
#include "macro.h"

#include <cassert>
#include <cstdint>


namespace cugo
{

template <typename T, int N>
struct Vec
{
    constexpr static size_t Size = N;

    HOST_DEVICE_INLINE Vec() {}
    HOST_DEVICE_INLINE Vec(const T* values) noexcept
    {
#pragma unroll
        for (int i = 0; i < Size; i++)
        {
            data[i] = values[i];
        }
    }

    template <typename U>
    HOST_DEVICE_INLINE Vec(const U* values) noexcept
    {
#pragma unroll
        for (int i = 0; i < Size; i++)
        {
            data[i] = T(values[i]);
        }
    }

    HOST_DEVICE_INLINE Vec(const Vec<T, N>& vec) noexcept
    {
#pragma unroll
        for (int i = 0; i < Size; i++)
        {
            data[i] = vec[i];
        }
    }

    HOST_DEVICE_INLINE T& operator[](int i) noexcept { return data[i]; }
    HOST_DEVICE_INLINE const T& operator[](int i) const noexcept { return data[i]; }

    HOST_DEVICE_INLINE void copyTo(T* rhs) const noexcept
    {
#pragma unroll
        for (int i = 0; i < Size; i++)
        {
            rhs[i] = data[i];
        }
    }

    template <typename U>
    HOST_DEVICE_INLINE void copyTo(U* rhs) const noexcept
    {
#pragma unroll
        for (int i = 0; i < Size; i++)
        {
            rhs[i] = U(data[i]);
        }
    }

    T data[Size];
};

template <typename T>
struct Vec<T, 2>
{
    static constexpr size_t Size = 2;
  
    HOST_DEVICE_INLINE Vec() {}
    HOST_DEVICE_INLINE Vec(const T* values) noexcept : x(values[0]), y(values[1])
    {
    }

    template <typename U>
    HOST_DEVICE_INLINE Vec(const U* values) noexcept : x(T(values[0])), y(T(values[1]))
    {
    }

    template <typename U>
    HOST_DEVICE_INLINE Vec(const U& x, const U& y) : x(T(x)), y(T(y))
    {
    }

    HOST_DEVICE_INLINE T& operator[](int i) noexcept 
    { 
        assert(i < Size);
        return data[i]; 
    }
    HOST_DEVICE_INLINE const T& operator[](int i) const noexcept 
    { 
        assert(i < Size);
        return data[i]; 
    }

    HOST_DEVICE_INLINE void copyTo(T* rhs) const noexcept
    {
        rhs[0] = data[0];
        rhs[1] = data[1];
    }

    template <typename U>
    HOST_DEVICE_INLINE void copyTo(T* rhs) const noexcept
    {
        rhs[0] = U(data[0]);
        rhs[1] = U(data[1]);
    }

    union
    {
        T data[Size];
        struct
        {
            T x, y;
        };
    };
};

template <typename T>
using Vec2 = Vec<T, 2>;

template <typename T>
__device__ inline Vec2<T> operator*(const T& f, const Vec2<T>& m) noexcept
{
    Vec2<T> result;
    result.data[0] = f * m.data[0];
    result.data[1] = f * m.data[1];
    return result;
}

template <typename T>
struct Vec<T, 3>
{
    static constexpr size_t Size = 3;

    HOST_DEVICE_INLINE Vec() {}
    HOST_DEVICE_INLINE Vec(const T* values) noexcept : x(values[0]), y(values[1]), z(values[2])
    {
    }

    template <typename U>
    HOST_DEVICE_INLINE Vec(const U& x, const U& y, const U& z) : x(T(x)), y(T(y)), z(T(z)) {}

    template <typename U>
    HOST_DEVICE_INLINE Vec(const U* values) noexcept
        : x(T(values[0])), y(T(values[1])), z(T(values[2]))
    {
    }

    HOST_DEVICE_INLINE T& operator[](int i) noexcept
    {
        assert(i < Size);
        return data[i];
    }
    HOST_DEVICE_INLINE const T& operator[](int i) const noexcept
    {
        assert(i < Size);
        return data[i];
    }

    HOST_DEVICE_INLINE void copyTo(T* rhs) const noexcept
    {
        rhs[0] = data[0];
        rhs[1] = data[1];
        rhs[2] = data[2];
    }

    template <typename U>
    HOST_DEVICE_INLINE void copyTo(T* rhs) const noexcept
    {
        rhs[0] = U(data[0]);
        rhs[1] = U(data[1]);
        rhs[2] = U(data[2]);
    }

    union
    {
        T data[Size];
        struct
        {
            T x, y, z;
        };
    };
    
};

template <typename T>
using Vec3 = Vec<T, 3>;

template <typename T>
__device__ inline Vec3<T> operator*(const T& f, const Vec3<T>& m) noexcept
{
    Vec3<T> result;
    result.data[0] = f * m.data[0];
    result.data[1] = f * m.data[1];
    result.data[2] = f * m.data[2];
    return result;
}

template <typename T>
struct Vec<T, 4>
{
    static constexpr size_t Size = 4;

    HOST_DEVICE_INLINE Vec() {}
    HOST_DEVICE_INLINE Vec(const T* values) noexcept
        : x(values[0]), y(values[1]), z(values[2]), w(values[3])
    {
    }

    template <typename U>
    HOST_DEVICE_INLINE Vec(const U* values) noexcept
        : x(T(values[0])), y(T(values[1])), z(T(values[2])), w(T(values[3]))
    {
    }

    template <typename U>
    HOST_DEVICE_INLINE Vec(const U& x, const U& y, const U& z, const U& w)
        : x(T(x)), y(T(y)), z(T(z)), w(T(w))
    {
    }

    HOST_DEVICE_INLINE T& operator[](int i) noexcept
    {
        assert(i < Size);
        return data[i];
    }
    HOST_DEVICE_INLINE const T& operator[](int i) const noexcept
    {
        assert(i < Size);
        return data[i];
    }

    HOST_DEVICE_INLINE void copyTo(T* rhs) const noexcept
    {
        rhs[0] = data[0];
        rhs[1] = data[1];
        rhs[2] = data[2];
        rhs[3] = data[3];
    }

    template <typename U>
    HOST_DEVICE_INLINE void copyTo(T* rhs) const noexcept
    {
        rhs[0] = U(data[0]);
        rhs[1] = U(data[1]);
        rhs[2] = U(data[2]);
        rhs[3] = U(data[3]);
    }

    union
    {
        T data[Size];
        struct
        {
            T x, y, z, w;
        };
    };
};

template <typename T>
using Vec4 = Vec<T, 4>;

template <typename T>
__device__ inline Vec4<T> operator*(const T& f, const Vec4<T>& m) noexcept
{
    Vec4<T> result;
    result.data[0] = f * m.data[0];
    result.data[1] = f * m.data[1];
    result.data[2] = f * m.data[2];
    result.data[3] = f * m.data[3];
    return result;
}


using Vec2i = Vec2<int>;
using Vec3i = Vec3<int>;
using Vec4i = Vec4<int>;
using Vec2f = Vec2<float>;
using Vec3f = Vec3<float>;
using Vec4f = Vec4<float>;
using Vec2d = Vec2<double>;
using Vec3d = Vec3<double>;
using Vec4d = Vec4<double>;



using Vec5f = Vec<float, 5>;
using Vec5d = Vec<double, 5>;
using Vec6f = Vec<float, 6>;
using Vec6d = Vec<double, 6>;


template <typename T>
struct Quat
{
    constexpr static size_t Size = 4;
     
    HOST_DEVICE_INLINE Quat() {}
    HOST_DEVICE_INLINE Quat(const T* values) noexcept
        : x(values[0]), y(values[1]), z(values[2]), w(values[3])
    {
    }

    template <typename U>
    HOST_DEVICE_INLINE Quat(const U* values) noexcept
        : x(T(values[0])), y(T(values[1])), z(T(values[2])), w(T(values[3]))
    {
    }

    template <typename U>
    HOST_DEVICE_INLINE Quat(const U& x, const U& y, const U& z, const U& w)
        : x(T(x)), y(T(y)), z(T(z)), w(T(w))
    {
    }

    HOST_DEVICE_INLINE Quat(const Vec4<T>& vec) noexcept
        : x(vec.x), y(vec.y), z(vec.z), w(vec.w)
    {
    }

    HOST_DEVICE_INLINE Quat(const Quat<T>& quat) noexcept 
        : x(quat.x), y(quat.y), z(quat.z), w(quat.w)
    {
    }

    HOST_DEVICE_INLINE T& operator[](int i) noexcept
    {
        assert(i < 4);
        return data[i];
    }
    HOST_DEVICE_INLINE const T& operator[](int i) const noexcept
    {
        assert(i < 4);
        return data[i];
    }

    HOST_DEVICE_INLINE void copyTo(T* rhs) const noexcept
    {
        #pragma unroll
        for (int i = 0; i < 4; i++)
        {
            rhs[i] = data[i];
        }
    }

    template <typename U>
    HOST_DEVICE_INLINE void copyTo(U* rhs) const noexcept
    {
        #pragma unroll
        for (int i = 0; i < 4; i++)
        {
            rhs[i] = U(data[i]);
        }
    }

    union
    {
        T data[Size];
        struct
        {
            T x, y, z, w;
        };
    };
};

using QuatF = Quat<float>;
using QuatD = Quat<double>;

template <typename T>
struct Se3
{
    HOST_DEVICE_INLINE Se3() {}
    HOST_DEVICE_INLINE Se3(const T* rValues, const T* tValues) noexcept
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
    HOST_DEVICE_INLINE Se3(const U* rValues, const U* tValues) noexcept
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

    HOST_DEVICE_INLINE void copyTo(T* r_rhs, T* t_rhs) const noexcept
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
    HOST_DEVICE_INLINE void copyTo(U* r_rhs, U* t_rhs) const noexcept
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

    HOST_DEVICE_INLINE Se3(const Quat<T>& quat, const Vec3<T>& vec) noexcept
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
    Vec3<T> t;
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

