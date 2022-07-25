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

#ifndef __DEVICE_BUFFER_H__
#define __DEVICE_BUFFER_H__

#include "macro.h"

#include <cuda_runtime.h>

#include <cassert>
#include <cstdint>

namespace cugo
{
template <typename T>
class DeviceBuffer
{
public:
    DeviceBuffer() : data_(nullptr), size_(0), capacity_(0), allocated_(false) {}
    DeviceBuffer(size_t size) : data_(nullptr), size_(0), capacity_(0), allocated_(false)
    {
        resize(size);
    }
    ~DeviceBuffer() { destroy(); }

    void allocate(size_t count)
    {
        if (data_ && capacity_ >= count)
        {
            return;
        }

        destroy();
        CUDA_CHECK(cudaMalloc(&data_, sizeof(T) * count));
        capacity_ = count;
        allocated_ = true;
    }

    void destroy()
    {
        if (allocated_ && data_)
        {
            CUDA_CHECK(cudaFree(data_));
        }
        data_ = nullptr;
        size_ = 0;
        allocated_ = false;
    }

    void resize(size_t size)
    {
        allocate(size);
        size_ = size;
    }

    void map(size_t size, void* data)
    {
        data_ = (T*)data;
        size_ = size;
        allocated_ = false;
    }

    template <typename U>
    void offset(DeviceBuffer<U>& buffer, size_t size, size_t offset)
    {
        void* offset_ptr = static_cast<uint8_t*>(buffer.data()) + offset;
        data_ = (T*)offset_ptr;
        size_ = size;
        // this is offset into a larger buffer so don't try and
        // free upon destruction.
        allocated_ = false;
    }

    void assign(size_t size, const void* h_data)
    {
#ifdef USE_ZERO_COPY
        CUDA_CHECK(cudaHostGetDevicePointer(&data_, (T*)h_data, 0));
#else
        resize(size);
        upload((T*)h_data);
#endif
    }

    void insert(size_t size, const void* h_data, size_t offset)
    {
        assert(offset < capacity_);
        assert(size < capacity_);
        T* data_ptr = data_ + offset;
        CUDA_CHECK(cudaMemcpy(data_ptr, h_data, sizeof(T) * size, cudaMemcpyHostToDevice));
    }

    void assignAsync(size_t size, const void* h_data, cudaStream_t stream = 0)
    {
#ifdef USE_ZERO_COPY
        CUDA_CHECK(cudaHostGetDevicePointer(&data_, (T*)h_data, 0));
        size_ = size;
#else
        resize(size);
        uploadAsync((T*)h_data, stream);
#endif
    }

    void upload(const T* h_data)
    {
        assert(h_data != nullptr);
        assert(size_ > 0);
        CUDA_CHECK(cudaMemcpy(data_, h_data, sizeof(T) * size_, cudaMemcpyHostToDevice));
    }

    void uploadAsync(const T* h_data, cudaStream_t stream = 0)
    {
        assert(h_data != nullptr);
        assert(size_ > 0);
        CUDA_CHECK(
            cudaMemcpyAsync(data_, h_data, sizeof(T) * size_, cudaMemcpyHostToDevice, stream));
    }

    void download(T* h_data) const
    {
#ifndef USE_ZERO_COPY
        CUDA_CHECK(cudaMemcpy(h_data, data_, sizeof(T) * size_, cudaMemcpyDeviceToHost));
#endif
    }

    void downloadAsync(T* h_data, int stream = 0) const
    {
#ifndef USE_ZERO_COPY
        CUDA_CHECK(
            cudaMemcpyAsync(h_data, data_, sizeof(T) * size_, cudaMemcpyDeviceToHost, stream));
#endif
    }

    void copyTo(T* rhs) const
    {
        CUDA_CHECK(cudaMemcpy(rhs, data_, sizeof(T) * size_, cudaMemcpyDeviceToDevice));
    }

    void copyToAsync(T* rhs, int stream = 0) const
    {
        CUDA_CHECK(
            cudaMemcpyAsync(rhs, data_, sizeof(T) * size_, cudaMemcpyDeviceToDevice, stream));
    }

    void fillZero() { CUDA_CHECK(cudaMemset(data_, 0, sizeof(T) * size_)); }

    T* data() { return data_; }
    const T* data() const { return data_; }

    size_t size() const { return size_; }
    int ssize() const { return static_cast<int>(size_); }

    operator T*() { return data_; }
    operator const T*() const { return data_; }

private:
    T* data_;
    size_t size_, capacity_;
    bool allocated_;
};

} // namespace cugo

#endif // !__DEVICE_BUFFER_H__
