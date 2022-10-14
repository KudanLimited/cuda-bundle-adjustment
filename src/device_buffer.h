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

#include "macro.h"

#include <cuda_runtime.h>

#include <cassert>
#include <cstdint>

namespace cugo
{

/**
 * @brief A wrapper class for device allocated memory.
 * @tparam T The type this buffer will hold
 */
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

    /**
     * @brief Allocate memory on the device.
     * Note: If the buffer has already been allocated and is the @p count is
     * less than the previously allocated size, then no allocation will occur.
     * @param count The number of type @p T slots to allocate
     */
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

    /**
     * @brief If a buffer has been allocated on the device, destroy the buffer.
     */
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

    /**
     * @brief Resize a buffer to the specified size. If this is the first call, then
     * the device memory is allocated. if a buffer has already been allocated, a resize
     * will only occur if the previous size is smaller than the newly required size.
     * @param size The size to resize the buffer to.
     * @param clear If true, sets the allocated memory to zeros.
     */
    void resize(size_t size, bool clear = false)
    {
        allocate(size);
        if (clear)
        {
            cudaMemset(data_, 0, sizeof(T) * size_);
        }
        size_ = size;
    }

    /**
     * @brief Map a buffer onto previously allocated device memory.
     * Note: This function does no memory allocations and therefore will not
     * be destroyed. Also, capacity will be set to zero to indicate this is
     * just a mapping.
     * @param size The size of the buffer to map.
     * @param data A pointer to a valid device memory space.
     */
    void map(size_t size, void* data)
    {
        data_ = reinterpret_cast<T*>(data);
        size_ = size;
        allocated_ = false;
    }

    /**
     * @brief Similiar to @p map, but allows for the buufer to map onto to be
     * passed to the function along with the offset.
     * @tparam U The buffer type used for mapping
     * @param buffer The @p DeviceBuffer to use for mapping
     * @param size The size of allocated space to use
     * @param offset An offset into the @p buffer which this buffer will map to.
     */
    template <typename U>
    void offset(DeviceBuffer<U>& buffer, size_t size, size_t offset)
    {
        void* offset_ptr = static_cast<uint8_t*>(buffer.data()) + offset;
        data_ = reinterpret_cast<T*>(offset_ptr);
        size_ = size;
        // this is offset into a larger buffer so don't try and
        // free upon destruction.
        allocated_ = false;
    }

    /**
     * @brief Resize/allocate memory space and upload data from host to device.
     *
     * @param size The size of the memory space required
     * @param h_data A pointer to the host data to upload.
     */
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

    /**
     * @brief Similiar to @p assign, but the upload is carried out async so will not
     * block the host thread.
     * Note: Memory for async operations must used pinned memory allocation
     *
     * @param size The size of the memory space required
     * @param h_data A pointer to the host data to upload.
     * @param stream A cuda stream which will be used for the upload. If not stated,
     * the default stream is used.
     */
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

    /**
     * @brief Upload data from host to device.
     *
     * @param h_data A pointer to the host data to upload.
     */
    void upload(const T* h_data)
    {
        assert(h_data != nullptr);
        assert(size_ > 0);
        CUDA_CHECK(cudaMemcpy(data_, h_data, sizeof(T) * size_, cudaMemcpyHostToDevice));
    }

    /**
     * @brief Similar to @p upload, but the upload occurs async so does not
     * block the host thread.
     *
     * @param h_data A pointer to the host data to upload.
     * @param stream A cuda stream which will be used for the upload. If not stated,
     * the default stream is used.
     */
    void uploadAsync(const T* h_data, cudaStream_t stream = 0)
    {
        assert(h_data != nullptr);
        assert(size_ > 0);
        CUDA_CHECK(
            cudaMemcpyAsync(data_, h_data, sizeof(T) * size_, cudaMemcpyHostToDevice, stream));
    }

    /**
     * @brief Download data from device to host.
     * @param h_data A pointer to memory where the data will be downloaded to.#
     * Note: This must be of adequate size to stage the downnloaded data (no check is carried out)
     */
    void download(T* h_data) const
    {
        CUDA_CHECK(cudaMemcpy(h_data, data_, sizeof(T) * size_, cudaMemcpyDeviceToHost));
    }

    /**
     * @brief Similiar to @p download, but the download is carried out async so the host
     * thread is not blocked.
     * @param h_data A pointer to memory where the data will be downloaded to.
     */
    void downloadAsync(T* h_data, const cudaStream_t stream = 0) const
    {
        CUDA_CHECK(
            cudaMemcpyAsync(h_data, data_, sizeof(T) * size_, cudaMemcpyDeviceToHost, stream));
    }

    /**
     * @brief Copy data from one device memory location to another.
     *
     * @param rhs A device memory pointer to copy the data to.
     */
    void copyTo(T* rhs) const
    {
        CUDA_CHECK(cudaMemcpy(rhs, data_, sizeof(T) * size_, cudaMemcpyDeviceToDevice));
    }

    /**
     * @brief Copy data from one device memory location to another. This is done async so
     * the host thread is not blocked.
     *
     * @param rhs A device memory pointer to copy the data to.
     */
    void copyToAsync(T* rhs, const cudaStream_t stream) const
    {
        CUDA_CHECK(
            cudaMemcpyAsync(rhs, data_, sizeof(T) * size_, cudaMemcpyDeviceToDevice, stream));
    }

    /**
     * @brief Fill the allocated device memory with zeros.
     *
     */
    void fillZero() { CUDA_CHECK(cudaMemset(data_, 0, sizeof(T) * size_)); }

    /**
     * @brief Get the pointer to device memory
     * @return A device memory pointer of this buffer.
     */
    T* data() { return data_; }

    /**
     * @brief Get the constant pointer to device memory
     * @return A device memory pointer of this buffer.
     */
    const T* data() const { return data_; }

    /**
     * @brief Get the size of this buffer.
     * @return The size of the allocated space.
     */
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
