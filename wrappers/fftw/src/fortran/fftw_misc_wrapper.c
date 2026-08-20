// Copyright Advanced Micro Devices, Inc.
// SPDX-License-Identifier: BSD-3-Clause

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

FFTZ_VOID dfftw_plan_with_nthreads_(FFTZ_INT32 *nthreads)
{
    fftw_plan_with_nthreads(*nthreads);
}

FFTZ_VOID dfftwf_plan_with_nthreads_(FFTZ_INT32 *nthreads)
{
    fftwf_plan_with_nthreads(*nthreads);
}

FFTZ_VOID dfftw_init_threads_(FFTZ_INT32 *ret)
{
    *ret = fftw_init_threads();
}

FFTZ_VOID dfftwf_init_threads_(FFTZ_INT32 *ret)
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
