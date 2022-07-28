#pragma once

#include "macro.h"

#include <cuda_runtime.h>

#include <cassert>
#include <cstring>
#include <memory>

namespace cugo
{
class Arena;

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

    void push_back(const T& data) noexcept
    {
        assert(size_ < capacity_);
        T* data_ptr = data_ + size_;
        memcpy(data_ptr, &data, sizeof(T));
        ++size_;
    }

    void push_back(const T&& data) noexcept
    {
        assert(size_ < capacity_);
        T* data_ptr = data_ + size_;
        memcpy(data_ptr, &data, sizeof(T));
        ++size_;
    }

    size_t size() const noexcept { return size_; }

    void* data() noexcept { return static_cast<void*>(data_); }

    void clear() noexcept { size_ = 0; }

    size_t bufferOffset() const { return offset_; }

private:
    T* data_;
    size_t size_;
    size_t capacity_;
    size_t offset_;
};

class Arena
{
public:
    Arena() : arena_(nullptr), currSize_(0), capacity_(0) {}

    Arena(size_t totalSize) : capacity_(totalSize), currSize_(0) { allocate(totalSize); }

    Arena(const Arena& rhs) = delete;
    Arena& operator=(const Arena& rhs) = delete;

    ~Arena() noexcept
    {
        if (arena_)
        {
            destroy();
        }
    }

    template <typename T>
    std::unique_ptr<ArenaPtr<T>> allocate(size_t size) noexcept
    {
        size_t bytesInsert = size * sizeof(T);
        // destroy the current memory chunk, and allocate a larger one
        // if we have exceeded the capacity
        if (bytesInsert > capacity_)
        {
            allocate(bytesInsert);
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

    void resize(size_t size) noexcept
    {
        clear();
        if (!arena_ || size > capacity_)
        {
            allocate(size);
        }
    }

    void clear() noexcept { currSize_ = 0; }

    void* data() noexcept { return arena_; }

private:
    void allocate(size_t size) noexcept
    {
        if (arena_)
        {
            CUDA_CHECK(cudaFreeHost(arena_));
            currSize_ = 0;
        }
        CUDA_CHECK(cudaMallocHost(&arena_, size));
        capacity_ = size;
    }

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