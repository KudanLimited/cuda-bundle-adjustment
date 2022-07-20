#pragma once

#include "macro.h"

#include <cassert>

namespace cugo
{

template <typename T>
class async_vector
{
public:
    async_vector() : data_(nullptr), size_(0), capacity_(0), curr_data_(nullptr) {}
    async_vector(size_t size) : size_(size) { allocate(size); }

    ~async_vector()
    {
        if (data_)
        {
            destroy();
        }
    }

    void reserve(size_t size) noexcept
    {
        // only resize if we the new size is greater than the
        // already existing size
        if (!data_ || size > capacity_)
        {
            allocate(size);
        }
    }

    void push_back(const T& data) noexcept
    {
        assert(data_);
        assert(capacity_ > 0);
        memcpy(curr_data_, &data, sizeof(T));
        ++curr_data_;
        ++size_;
    }

    void push_back(const T&& data) noexcept
    {
        assert(data_);
        assert(capacity_ > 0);
        memcpy(curr_data_, &data, sizeof(T));
        ++curr_data_;
        ++size_;
    }

    void clear() noexcept
    {
        curr_data_ = data_;
        size_ = 0;
    }

    void* data() noexcept { return static_cast<void*>(data_); }

    T& operator[](int index) noexcept
    {
        assert(index < capacity_);
        return data_[index];
    }

    T& operator[](int index) const noexcept
    {
        assert(index < capacity_);
        return data_[index];
    }

    size_t size() const noexcept { return size_; }

private:
    void allocate(size_t size) noexcept
    {
        if (data_)
        {
            CUDA_CHECK(cudaFreeHost(data_));
            size_ = 0;
        }
        CUDA_CHECK(cudaHostAlloc(&data_, sizeof(T) * size, cudaHostAllocMapped));
        curr_data_ = data_;
        capacity_ = size;
    }

    void destroy() noexcept
    {
        CUDA_CHECK(cudaFreeHost(data_));
        data_ = nullptr;
        curr_data_ = nullptr;
        capacity_ = 0;
        size_ = 0;
    }

private:
    T* data_;
    size_t size_;
    size_t capacity_;
    T* curr_data_;
};

} // namespace cugo