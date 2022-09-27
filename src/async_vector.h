#pragma once

#include "macro.h"

#include <cuda_runtime.h>

#include <cassert>
#include <cstring>
#include <utility>

namespace cugo
{

/**
 * @brief A simple vector that uses pinned memory allocations for use with CUDA async mem copies
 * @tparam T The data type that will be stored in the vector.
 */
template <typename T>
class CUGO_API async_vector
{
public:
    async_vector() : data_(nullptr), size_(0), capacity_(0) {}
    async_vector(size_t size) : size_(0) { allocate(size); }

    ~async_vector()
    {
        if (data_)
        {
            destroy();
        }
    }

    /**
     * @brief Allocate pinned memory based on the number of elements specified (size * sizeof T will
     * be the allocated size)
     * @param size The number of elements to reserve.
     */
    void reserve(size_t size) noexcept
    {
        // only resize if we the new size is greater than the
        // already existing size
        if (!data_ || size >= capacity_)
        {
            allocate(size);
        }
        size_ = 0;
    }

    /**
     * @brief Similiar to @p reserve, but also the size of elements is set.
     * @param size The number of elements to resize the container to.
     */
    void resize(size_t size) noexcept
    {
        reserve(size);
        size_ = size;
    }

    /**
     * @brief Copy the specified element of the type @p T after the current element.
     * @param data The data to push into the vector
     */
    void push_back(const T& data) noexcept
    {
        assert(capacity_ > 0);
        assert(size < capacity_);
        data_[size_++] = data;
    }

    /**
     * @brief Move the specified element of the type @p T after the current element.
     * @param data The data to move into memory after the current element.
     */
    void push_back(const T&& data) noexcept
    {
        assert(capacity_ > 0);
        assert(size < capacity_);
        data_[size_++] = std::move(data);
    }

    /**
     * @brief Clears all elements from the vector. Resets the current data pointer to the beginning
     * of the allocated memory space.
     *
     */
    void clear() noexcept { size_ = 0; }

    /**
     * @brief Get the pointer to the allocated pinned memory space.
     * @return A pointer to the allocated pinned memory space.
     */
    T* data() noexcept { return data_; }
    const T* data() const noexcept { return data_; }

    T& operator[](int index) noexcept
    {
        assert(index < size_);
        return data_[index];
    }

    T& operator[](int index) const noexcept
    {
        assert(index < size_);
        return data_[index];
    }

    size_t size() const noexcept { return size_; }

    void zero() noexcept
    {
        if (data_ && capacity_ > 0)
        {
            memset(data_, 0, sizeof(T) * capacity_);
        }
    }

    async_vector<T>& operator=(const async_vector<T>& rhs) noexcept
    {
        destroy();
        if (this != &rhs)
        {
            size_ = rhs.size_;
            capacity_ = rhs.capacity_;
            CUDA_CHECK(cudaHostAlloc(&data_, sizeof(T) * capacity_, cudaHostAllocMapped));
            memcpy(data_, rhs.data_, sizeof(T) * size_);
        }
        return *this;
    }

    // iterators
    using value_type = T;
    using reference = T&;
    using const_reference = const T&;
    using iterator = T*;
    using const_iterator = const T*;
    using size_type = size_t;

    iterator begin() { return data_; }
    const_iterator begin() const { return data_; }
    iterator end() { return data_ + size_; }
    const_iterator end() const { return data_ + size_; }
    const_iterator cbegin() const { return data_; }
    const_iterator cend() const { return data_ + size_; }

private:
    /**
     * @brief Allocate the required pinned memory. If a chunk is already allocated, this will be
     * destroyed and the required chunk allocated.
     * @param The size of the memory chunk to allocate - the number of elements of type @p T
     */
    void allocate(size_t size) noexcept
    {
        if (data_)
        {
            CUDA_CHECK(cudaFreeHost(data_));
            size_ = 0;
            data_ = nullptr;
        }
        CUDA_CHECK(cudaHostAlloc(&data_, sizeof(T) * size, cudaHostAllocMapped));
        capacity_ = size;
    }

    /**
     * @brief Deallocates the current memory block and resets all variables.
     */
    void destroy() noexcept
    {
        if (data_)
        {
            CUDA_CHECK(cudaFreeHost(data_));
            data_ = nullptr;
            capacity_ = 0;
            size_ = 0;
        }
    }

private:
    T* data_;
    size_t size_;
    size_t capacity_;
};

} // namespace cugo