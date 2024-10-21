/**
 * Copyright (C) 2023, Advanced Micro Devices. All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 * 1. Redistributions of source code must retain the above copyright notice,
 * this list of conditions and the following disclaimer.
 * 2. Redistributions in binary form must reproduce the above copyright notice,
 * this list of conditions and the following disclaimer in the documentation
 * and/or other materials provided with the distribution.
 * 3. Neither the name of the copyright holder nor the names of its
 * contributors may be used to endorse or promote products derived from this
 * software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
 * ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
 * LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
 * CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
 * SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
 * INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
 * CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
 * ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
 * POSSIBILITY OF SUCH DAMAGE.
 */

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

#define MIN_ALIGNMENT 16

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
    if (posix_memalign((VOID **)(&ptr), MIN_ALIGNMENT, num_bytes))             \
    {                                                                          \
        ptr = NULL;                                                            \
    }                                                                          \
}

#define ALLOC_ALIGN_INIT(ptr, type, num_bytes)                                 \
{                                                                              \
    if (posix_memalign((VOID **)(&ptr), MIN_ALIGNMENT, num_bytes) == 0)        \
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
    if (is_align)                                                              \
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
