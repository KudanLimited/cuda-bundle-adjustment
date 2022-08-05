#pragma once

#include "macro.h"

#include <cuda_runtime.h>

#include <cassert>
#include <cstring>
#include <memory>

namespace cugo
{
class Arena;

/**
 * @brief A wrapper around an Arena derived offset that gives the ability to use standard C++
 * container functions.
 * @tparam T The type of data that is referred to by this chunk
 */
template <typename T>
class ArenaPtr
{
public:
    ArenaPtr() = default;

    ArenaPtr(size_t capacity, T* data, size_t offset)
        : size_(0), data_(data), capacity_(capacity), offset_(offset)
    {
    }
    ~ArenaPtr() {}

    /**
     * @brief Copies data to the memory slot after the current element. Must be within the allocated
     * chunk capacity.
     * @param data The data to copy to the memory chunk
     */
    void push_back(const T& data) noexcept
    {
        assert(size_ < capacity_);
        T* data_ptr = data_ + size_;
        memcpy(data_ptr, &data, sizeof(T));
        ++size_;
    }

    /**
     * @brief Moves data to the memory slot after the current element. Must be within the allocated
     * chunk capacity.
     * @param data The data to move to the memory chunk
     */
    void push_back(const T&& data) noexcept
    {
        assert(size_ < capacity_);
        T* data_ptr = data_ + size_;
        memcpy(data_ptr, &data, sizeof(T));
        ++size_;
    }

    /**
     * @brief The number of elements that have been added to this chunk.
     * @return The current element count.
     */
    size_t size() const noexcept { return size_; }

    /**
     * @brief A pointer to the memory chunk.
     * @return A void pointer to the allocated chunk.
     */
    void* data() noexcept { return static_cast<void*>(data_); }

    /**
     * @brief Clear all elements from this chunk.
     */
    void clear() noexcept { size_ = 0; }

    /**
     * @brief The offset of this chunk in relation to the main memory pool.
     * @return size_t The memory chunk offset.
     */
    size_t bufferOffset() const { return offset_; }

private:
    /// Pointer to the beginning of the memory pool chunk that this ArenaPtr points to.
    T* data_;
    /// The number of elements that have currently been added to the chunk.
    size_t size_;
    /// The total number of elements this chunk can store.
    size_t capacity_;
    /// The offset within the memory pool
    size_t offset_;
};

/**
 * @brief A memory pool class that allocates a chunk of memory and create smaller chunks that are
 * offsets into the] main pool wrapped in a @p ArenaPtr class. Note: This is quite simple at the
 * moment - there is no functionality for mamaging the tidying of chunks or deletion
 */
class Arena
{
public:
    Arena() : arena_(nullptr), currSize_(0), capacity_(0) {}

    Arena(size_t totalSize) : capacity_(totalSize), currSize_(0) { allocatePool(totalSize); }

    Arena(const Arena& rhs) = delete;
    Arena& operator=(const Arena& rhs) = delete;

    ~Arena() noexcept
    {
        if (arena_)
        {
            destroy();
        }
    }

    /**
     * @brief Allocates a chunk from the main memory pool aligned to 64bit address and wraps the
     * offset in a @p ArenaPtr
     * @tparam T The data type this chunk will use.
     * @param size The number of elements of size @p T that this chunk will hold.
     * @return A @p ArenaPtr that manages the chunk.
     */
    template <typename T>
    std::unique_ptr<ArenaPtr<T>> allocate(size_t size) noexcept
    {
        size_t bytesInsert = size * sizeof(T);
        // destroy the current memory chunk, and allocate a larger one
        // if we have exceeded the capacity
        if (bytesInsert > capacity_)
        {
            allocatePool(bytesInsert);
        }
        void* arena_ptr = static_cast<char*>(arena_) + currSize_;

        // check pointer aligned to 64bit - otherwise we will get errors in cuda
        std::size_t a = alignof(T);
        T* aligned_ptr = nullptr;
        if (std::align(a, sizeof(T), arena_ptr, capacity_))
        {
            aligned_ptr = reinterpret_cast<T*>(arena_ptr);
        }
        else
        {
            std::runtime_error("Alignment error whilst trying to allocate arena ptr.");
        }

        size_t offset = currSize_;
        currSize_ += bytesInsert;
        return std::make_unique<ArenaPtr<T>>(size, aligned_ptr, offset);
    }

    /**
     * @brief Resize the memory pool to the specified size. If the pool has been allocated,
     * and the current size is greater than the requested size, then no allocation will occur.
     * @param size The size of the pool to allocate in bytes.
     */
    void resize(size_t size) noexcept
    {
        clear();
        if (!arena_ || size > capacity_)
        {
            allocatePool(size);
        }
    }

    /**
     * @brief Clears all allocated chunks from the pool. This does not deallocate memory.
     */
    void clear() noexcept { currSize_ = 0; }

    /**
     * @brief Get a pointer to the memory pool allocated memory.
     * @return The address of the memory pool.
     */
    void* data() noexcept { return arena_; }

private:
    /**
     * @brief Allocate the memory pool. Memory is pinned for use with CUDA async mem copies. If a
     * pool already exists, this will be destroyed before creating the new pool.
     * @param size The size of memory to allocate in bytes.
     */
    void allocatePool(size_t size) noexcept
    {
        if (arena_)
        {
            CUDA_CHECK(cudaFreeHost(arena_));
            currSize_ = 0;
        }
        CUDA_CHECK(cudaMallocHost(&arena_, size));
        capacity_ = size;
    }

    /**
     * @brief Deallocate the memory used by the pool.
     */
    void destroy() noexcept
    {
        CUDA_CHECK(cudaFreeHost(arena_));
        arena_ = nullptr;
        capacity_ = 0;
        currSize_ = 0;
    }

private:
    void* arena_;
    size_t currSize_;
    size_t capacity_;
};

} // namespace cugo