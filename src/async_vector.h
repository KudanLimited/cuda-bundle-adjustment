#pragma once

#include "macro.h"

#include <cassert>

namespace cugo
{

/**
 * @brief A simple vector that uses pinned memory allocations for use with CUDA async mem copies
 * @tparam T The data type that will be stored in the vector.
 */
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

    /**
     * @brief Allocate pinned memory based on the number of elements specified (size * sizeof T will
     * be the allocated size)
     * @param size The number of elements to reserve.
     */
    void reserve(size_t size) noexcept
    {
        // only resize if we the new size is greater than the
        // already existing size
        if (!data_ || size > capacity_)
        {
            allocate(size);
        }
    }

    /**
     * @brief Copy the specified element of the type @p T after the current element.
     * @param data The data to push into the vector
     */
    void push_back(const T& data) noexcept
    {
        assert(data_);
        assert(capacity_ > 0);
        memcpy(curr_data_, &data, sizeof(T));
        ++curr_data_;
        ++size_;
    }

    /**
     * @brief Move the specified element of the type @p T after the current element.
     * @param data The data to move into memory after the current element.
     */
    void push_back(const T&& data) noexcept
    {
        assert(data_);
        assert(capacity_ > 0);
        memcpy(curr_data_, &data, sizeof(T));
        ++curr_data_;
        ++size_;
    }

    /**
     * @brief Clears all elements from the vector. Resets the current data pointer to the beginning
     * of the allocated memory space.
     *
     */
    void clear() noexcept
    {
        curr_data_ = data_;
        size_ = 0;
    }

    /**
     * @brief Get the pointer to the allocated pinned memory space.
     * @return A pointer to the allocated pinned memory space.
     */
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
        }
        CUDA_CHECK(cudaHostAlloc(&data_, sizeof(T) * size, cudaHostAllocMapped));
        curr_data_ = data_;
        capacity_ = size;
    }

    /**
     * @brief Deallocates the current memory block and resets all variables.
     */
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