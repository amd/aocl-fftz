/**
 * Copyright (C) 2025, Advanced Micro Devices. All rights reserved.
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

/** @file fftw_misc_wrapper.c
 *
 *  @brief Contains wrapper implementations of FFTW miscellaneous APIs.
 *
 *  This file contains implementations for the miscellaneous APIs provided
 *  by FFTW.
 */

#include "fftz_translator.h"

INT32 thread_num = 1;

// Allocate aligned memory for given number of bytes
VOID *fftw_malloc(size_t n)
{
    VOID *ptr = NULL;
    ALLOC_ALIGN_UNINIT(ptr, VOID, n);
    return ptr;
}

// Allocate aligned memory for given number of bytes
VOID *fftwf_malloc(size_t n)
{
    VOID *ptr = NULL;
    ALLOC_ALIGN_UNINIT(ptr, VOID, n);
    return ptr;
}

// Allocate aligned memory for complex double datatype for given size
fftw_complex *fftw_alloc_complex(size_t n)
{
    VOID *ptr = NULL;
    ALLOC_ALIGN_UNINIT(ptr, VOID, sizeof(fftw_complex) * n);
    return ptr;
}

// Allocate aligned memory for complex float datatype for given size
fftwf_complex *fftwf_alloc_complex(size_t n)
{
    VOID *ptr = NULL;
    ALLOC_ALIGN_UNINIT(ptr, VOID, sizeof(fftwf_complex) * n);
    return ptr;
}

double *fftw_alloc_real(size_t n)
{
    VOID *ptr = NULL;
    ALLOC_ALIGN_UNINIT(ptr, VOID, sizeof(double) * n);
    return ptr;
}

float *fftwf_alloc_real(size_t n)
{
    VOID *ptr = NULL;
    ALLOC_ALIGN_UNINIT(ptr, VOID, sizeof(float) * n);
    return ptr;
}

VOID fftw_free(VOID *mem_ptr)
{
    FREE_ALIGN_ALLOCATED_MEM(mem_ptr);
}

VOID fftwf_free(VOID *mem_ptr)
{
    FREE_ALIGN_ALLOCATED_MEM(mem_ptr);
}

// FFTW planner stores some persistant data other than plan, which can be
// destroyed using cleanup() but this has no requirement in fftz. Hence having
// it as an empty function.
VOID fftw_cleanup(VOID)
{
}

VOID fftwf_cleanup(VOID)
{
}

VOID fftw_plan_with_nthreads(INT32 nthreads)
{
    thread_num = nthreads;
}

VOID fftwf_plan_with_nthreads(INT32 nthreads)
{
    thread_num = nthreads;
}

INT32 fftw_planner_nthreads(VOID)
{
    return thread_num;
}

INT32 fftwf_planner_nthreads(VOID)
{
    return thread_num;
}

INT32 fftw_init_threads(VOID)
{
    thread_num = 1;
    return 1;
}

INT32 fftwf_init_threads(VOID)
{
    thread_num = 1;
    return 1;
}

VOID fftw_cleanup_threads(VOID)
{
   thread_num = 1; // reset to default
}

VOID fftwf_cleanup_threads(VOID)
{
    thread_num = 1; // reset to default
}

VOID fftw_print_plan(const fftw_plan p)
{
    return;
}

VOID fftwf_print_plan(const fftwf_plan p)
{
    return;
}

VOID fftw_flops(const fftw_plan p, double *add, double *mul, double *fmas)
{
    return;
}

VOID fftwf_flops(const fftwf_plan p, double *add, double *mul, double *fmas)
{
    return;
}

double fftw_estimate_cost(const fftw_plan p)
{
    return 0;
}

double fftwf_estimate_cost(const fftwf_plan p)
{
    return 0;
}

double fftw_cost(const fftw_plan p)
{
    return 0;
}

double fftwf_cost(const fftwf_plan p)
{
    return 0;
}

INT32 fftw_alignment_of(double *p)
{
    return (int)(((uintptr_t)p) % 16);
}

INT32 fftwf_alignment_of(float *p)
{
    return (int)(((uintptr_t)p) % 16);
}
