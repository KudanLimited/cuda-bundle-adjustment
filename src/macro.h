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

#include <cstdio>
#include <cassert>

#define CUDA_CHECK(err)                                                                            \
    do                                                                                             \
    {                                                                                              \
        if (err != cudaSuccess)                                                                    \
        {                                                                                          \
            fprintf(stderr,                                                                        \
                "[CUDA Error] %s (code: %d) at %s:%d\n",                                           \
                cudaGetErrorString(err),                                                           \
                err,                                                                               \
                __FILE__,                                                                          \
                __LINE__);                                                                         \
            throw(err);                                                                               \
        }                                                                                          \
    } while (0)
    
    

#define CHECK_CUSPARSE(func)                                                                       \
    {                                                                                              \
        cusparseStatus_t status = (func);                                                          \
        if (status != CUSPARSE_STATUS_SUCCESS)                                                     \
        {                                                                                          \
            fprintf(stderr, "CUSPARSE API failed with error (%d) at line %d\n", status, __LINE__); \
            throw(status);                                                                      \
        }                                                                                          \
    }                                                                                              
   

#define CHECK_CUSOLVER(func)                                                                       \
    {                                                                                              \
        cusolverStatus_t status = (func);                                                          \
        if (status != CUSOLVER_STATUS_SUCCESS)                                                     \
        {                                                                                          \
            fprintf(stderr, "CUSOLVER API failed with error (%d) at line %d\n", status, __LINE__); \
            throw(status);                                                                     \
        }                                                                                          \
    }                                                                                              


#if __has_builtin(__builtin_expect)
#ifdef __cplusplus
#define CUGO_LIKELY(exp) (__builtin_expect(!!(exp), true))
#define CUGO_UNLIKELY(exp) (__builtin_expect(!!(exp), false))
#else
#define CUGO_LIKELY(exp) (__builtin_expect(!!(exp), 1))
#define CUGO_UNLIKELY(exp) (__builtin_expect(!!(exp), 0))
#endif
#else
#define CUGO_LIKELY(exp) (!!(exp))
#define CUGO_UNLIKELY(exp) (!!(exp))
#endif

#ifdef _WIN32
    #ifdef __GNUC__
        #define GUGO_API __attribute__((dllexport))
    #else
        #define CUGO_API __declspec(dllexport)
    #endif
#else
    #if __GNUC__>=4
        #define CUGO_API __attribute__((visibility("default")))
    #else
        #define CUGO_API
    #endif
#endif


