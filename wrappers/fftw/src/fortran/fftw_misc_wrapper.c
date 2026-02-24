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
 *  @brief Contains wrapper implementations of miscellaneous FFTW Fortran APIs.
 *
 *  This file contains implementations for the miscellaneous Fortran APIs
 *  provided by FFTW.
 */

#include "src/translator/fftz_translator.h"

void dfftw_cleanup_(void)
{
    fftw_cleanup();
}

void dfftwf_cleanup_(void)
{
    fftwf_cleanup();
}

VOID dfftw_plan_with_nthreads_(INT32 *nthreads)
{
    fftw_plan_with_nthreads(*nthreads);
}

VOID dfftwf_plan_with_nthreads_(INT32 *nthreads)
{
    fftwf_plan_with_nthreads(*nthreads);
}

VOID dfftw_init_threads_(INT32 *ret)
{
    *ret = fftw_init_threads();
}

VOID dfftwf_init_threads_(INT32 *ret)
{
    *ret = fftwf_init_threads();
}

void dfftw_cleanup_threads_(void)
{
    fftw_cleanup_threads();
}

void dfftwf_cleanup_threads_(void)
{
    fftwf_cleanup_threads();
}

void dfftw_flops_(fftw_plan *p, double *add, double *mul, double *fma)
{
    fftw_flops(*p, add, mul, fma);
}

void dfftwf_flops_(fftwf_plan *p, double *add, double *mul, double *fma)
{
    fftwf_flops(*p, add, mul, fma);
}

void dfftw_estimate_cost_(double *cost, fftw_plan *const p)
{
    *cost = fftw_estimate_cost(*p);
}

void dfftwf_estimate_cost_(double *cost, fftwf_plan *const p)
{
    *cost = fftwf_estimate_cost(*p);
}

void dfftw_cost_(double *cost, fftw_plan *const p)
{
    *cost = fftw_cost(*p);
}

void dfftwf_cost_(double *cost, fftwf_plan *const p)
{
    *cost = fftwf_cost(*p);
}
