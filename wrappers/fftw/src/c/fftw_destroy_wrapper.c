// Copyright Advanced Micro Devices, Inc.
// SPDX-License-Identifier: BSD-3-Clause

/** @file fftw_destroy_wrapper.c
 *
 *  @brief Contains wrapper implementations of FFTW destroy APIs.
 *
 *  This file contains implementations for destroy APIs provided by FFTW.
 */

#include "src/translator/fftz_translator.h"

FFTZ_VOID fftw_destroy_plan(fftw_plan p)
{
    aoclfftz_destroy(p);
}

FFTZ_VOID fftwf_destroy_plan(fftwf_plan p)
{
    aoclfftz_destroy(p);
}
