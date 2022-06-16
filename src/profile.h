#ifndef __PROFILE_H__
#define __PROFILE_H__

#include <chrono>

#include "cuda/cuda_block_solver.h"

namespace cugo
{

using time_point = decltype(std::chrono::steady_clock::now());

static inline time_point get_time_point()
{
    gpu::waitForKernelCompletion();
    return std::chrono::steady_clock::now();
}

static inline double get_duration(const time_point& from, const time_point& to)
{
    return std::chrono::duration_cast<std::chrono::duration<double>>(to - from).count();
}

}
#endif