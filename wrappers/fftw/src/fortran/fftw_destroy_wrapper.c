// Copyright Advanced Micro Devices, Inc.
// SPDX-License-Identifier: BSD-3-Clause

/** @file fftw_destroy_wrapper.c
 *
 *  @brief Contains wrapper implementations of FFTW Fortran destroy APIs.
 *
 *  This file contains implementations for the Fortran destroy APIs provided by
 *  FFTW.
 */

#include "src/translator/fftz_translator.h"

FFTZ_VOID dfftw_destroy_plan_(fftw_plan *p)
{
    fftw_destroy_plan(*p);
}

FFTZ_VOID dfftwf_destroy_plan_(fftwf_plan *p)
{
    fftwf_destroy_plan(*p);
}
