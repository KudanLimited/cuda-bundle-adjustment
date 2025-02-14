#pragma once

#include "macro.h"

#include <cuda_runtime.h>

#include <cassert>
#include <cstring>
#include <memory>
#include <stdexcept>

namespace cugo
{
class Arena;

/**
 * @brief A wrapper around an Arena derived offset that gives the ability to use standard C++
 * container functions.
 * @tparam T The type of data that is referred to by this chunk
 */
template <typename T>
class ArenaPool
{
public:
    ArenaPool() = default;

    ArenaPool(size_t capacity, T* data, size_t offset)
        : size_(0), data_(data), capacity_(capacity), offset_(offset)
    {
    }
    virtual ~ArenaPool() {}

    /**
     * @brief Copies data to the memory slot after the current element. Must be within the allocated
     * chunk capacity.
     * @param data The data to copy to the memory chunk
     */
    void push_back(const T& data) noexcept
    {
        assert(size_ < capacity_);
        data_[size_++] = data;
    }

    /**
     * @brief Moves data to the memory slot after the current element. Must be within the allocated
     * chunk capacity.
     * @param data The data to move to the memory chunk
     */
    void push_back(const T&& data) noexcept
    {
        assert(size_ < capacity_);
        data_[size_++] = std::move(data);
    }

    /**
     * @brief Get the last item from the pool.
     * @return The last value of type @p T pushed to the pool
     */
    T& back() noexcept { return *(data_ + (size_ - 1)); }

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
    size_t bufferOffset() const noexcept { return offset_; }

private:
    /// Pointer to the beginning of the memory pool chunk that this ArenaPtr points to.
    T* data_;
    /// The number of elements that have currently been added to the chunk.
    size_t size_;
    /// The total number of elements this chunk can store.
    size_t capacity_;
    /// The offset within the memory pool in bytes
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
    std::unique_ptr<ArenaPool<T>> allocate(size_t size)
    {
        std::size_t totalSize = size * sizeof(T);
        // TODO: readjust the arena size if the capacity is reached - this will also
        // require re-adjusting the already allocated memory pools with the new mem
        // address (the tricky bit). Will probably need to add some sort of linked-list
        // method to the memory pools. Throws an exception for now.
        if (totalSize > capacity_)
        {
            throw std::runtime_error("Capacity reached for Arena. Increase the allocated size.");
        }

        T* alignedArenaPtr = reinterpret_cast<T*>(alignedPtr<T>(arena_ + currSize_));
        std::size_t alignedOffset = reinterpret_cast<uint8_t*>(alignedArenaPtr) - arena_;
        currSize_ += totalSize;
        return std::make_unique<ArenaPool<T>>(size, alignedArenaPtr, alignedOffset);
    }

    /**
     * @brief Resize the memory pool to the specified size. If the pool has been allocated,
     * and the current size is greater than the requested size, then no allocation will occur.
     * Note: this will destroy all elements in the arena.
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
     * @param retain If an arena is already allocated, if true, copies the old arena to the
     * newly allocated space.
     *
     */
    void allocatePool(size_t size, bool retain = false) noexcept
    {
        uint8_t* oldArena = arena_;
#ifndef USE_ZERO_COPY
        CUDA_CHECK(cudaHostAlloc(&arena_, size, cudaHostAllocDefault));
#else
        CUDA_CHECK(cudaHostAlloc(&arena_, size, cudaHostAllocMapped));
#endif

        // if stated, copy the old contents into the newly allocated space
        if (oldArena && retain)
        {
            memcpy(arena_, oldArena, capacity_);
        }

        capacity_ = size;

        if (oldArena)
        {
            CUDA_CHECK(cudaFreeHost(oldArena));
            currSize_ = 0;
        }
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

    /**
     * @brief Align a pointer based upon the memory specifications
     * @tparam T The type of the pointer
     * @param ptr The pointer to align
     * @return The aligned pointer as a void type
     */
    template <typename T>
    void* alignedPtr(void* ptr)
    {
        std::size_t alignment = alignof(T);
        std::uintptr_t uintPtr = reinterpret_cast<std::uintptr_t>(ptr);
        std::uintptr_t alignedPtr = (uintPtr + (alignment - 1)) & ~(alignment - 1);
        return reinterpret_cast<void*>(alignedPtr);
    }

private:
    uint8_t* arena_;
    size_t currSize_;
    size_t capacity_;
};

} // namespace cugo