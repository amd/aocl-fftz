// Copyright Advanced Micro Devices, Inc.
// SPDX-License-Identifier: BSD-3-Clause

/** @file allocator.h
 *
 *  @brief Wrappers to basic memory allocation and management primitives
 *
 *  This file contains wrapper functions and macros for allocating, managing,
 *  and destroying the memory as needed by AOCL-FFTZ.
 *
 *  @author S. Biplab Raut
 */

#ifndef ALLOCATOR_H
#define ALLOCATOR_H

#include <stdlib.h>
#include <string.h>

#define MIN_ALIGNMENT 64

#define GET_PADDED_SIZE(x)                                                     \
    (                                                                          \
        (((FFTZ_UINTP)(x) + (FFTZ_UINTP)(MIN_ALIGNMENT) - 1u) \
         & ~((FFTZ_UINTP)(MIN_ALIGNMENT) - 1u)) \
    )

#ifdef _WINDOWS

#define ALLOC_ALIGN_UNINIT(ptr, type, num_bytes)                               \
{                                                                              \
    ptr = (type *)_aligned_malloc(num_bytes, MIN_ALIGNMENT);                   \
}

#define ALLOC_ALIGN_INIT(ptr, type, num_bytes)                                 \
{                                                                              \
    ptr = (type *)_aligned_malloc(num_bytes, MIN_ALIGNMENT);                   \
    if (ptr)                                                                   \
    {                                                                          \
        memset(ptr, 0, (num_bytes));                                           \
    }                                                                          \
}

#define FREE_ALIGN_ALLOCATED_MEM(mem_ptr)                                      \
{                                                                              \
    if (mem_ptr)                                                               \
    {                                                                          \
        _aligned_free(mem_ptr);                                                \
    }                                                                          \
    mem_ptr = NULL;                                                            \
}

#else

#define ALLOC_ALIGN_UNINIT(ptr, type, num_bytes)                               \
{                                                                              \
    if (posix_memalign((FFTZ_VOID **)(&ptr), MIN_ALIGNMENT, num_bytes)) \
    {                                                                          \
        ptr = NULL;                                                            \
    }                                                                          \
}

#define ALLOC_ALIGN_INIT(ptr, type, num_bytes)                                 \
{                                                                              \
    if (posix_memalign((FFTZ_VOID **)(&ptr), MIN_ALIGNMENT, num_bytes) == 0) \
    {                                                                          \
        memset(ptr, 0, (num_bytes));                                           \
    }                                                                          \
    else                                                                       \
    {                                                                          \
        ptr = NULL;                                                            \
    }                                                                          \
}

#define FREE_ALIGN_ALLOCATED_MEM(mem_ptr)                                      \
{                                                                              \
    if (mem_ptr)                                                               \
    {                                                                          \
        free(mem_ptr);                                                         \
    }                                                                          \
    mem_ptr = NULL;                                                            \
}
#endif

#define ALLOC_UNALIGN_UNINIT(ptr, type, num_bytes)                             \
{                                                                              \
    ptr = (type *)malloc(num_bytes);                                           \
}

#define ALLOC_UNALIGN_INIT(ptr, type, num_bytes)                               \
{                                                                              \
    ptr = (type *)malloc(num_bytes);                                           \
    if (ptr)                                                                   \
    {                                                                          \
        memset(ptr, 0, (num_bytes));                                           \
    }                                                                          \
}

#define FREE_UNALIGN_ALLOCATED_MEM(mem_ptr)                                    \
{                                                                              \
    if (mem_ptr)                                                               \
    {                                                                          \
        free(mem_ptr);                                                         \
    }                                                                          \
    mem_ptr = NULL;                                                            \
}

#define ALLOC_UNINIT(ptr, type, num_bytes, is_align)                           \
{                                                                              \
    if (is_align)                                                              \
    {                                                                          \
        ALLOC_ALIGN_UNINIT(ptr, type, num_bytes)                               \
    }                                                                          \
    else                                                                       \
    {                                                                          \
        ALLOC_UNALIGN_UNINIT(ptr, type, num_bytes)                             \
    }                                                                          \
}

#define ALLOC_INIT(ptr, type, num_bytes, is_align)                             \
{                                                                              \
    if (is_align != 0)                                                         \
    {                                                                          \
        ALLOC_ALIGN_INIT(ptr, type, num_bytes)                                 \
    }                                                                          \
    else                                                                       \
    {                                                                          \
        ALLOC_UNALIGN_INIT(ptr, type, num_bytes)                               \
    }                                                                          \
}

#define FREE_ALLOCATED_MEM(mem_ptr, is_align)                                  \
{                                                                              \
    if (is_align)                                                              \
    {                                                                          \
        FREE_ALIGN_ALLOCATED_MEM(mem_ptr)                                      \
    }                                                                          \
    else                                                                       \
    {                                                                          \
        FREE_UNALIGN_ALLOCATED_MEM(mem_ptr)                                    \
    }                                                                          \
}

#endif // ALLOCATOR_H
