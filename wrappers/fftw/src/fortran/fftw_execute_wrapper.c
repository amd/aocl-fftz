// Copyright Advanced Micro Devices, Inc.
// SPDX-License-Identifier: BSD-3-Clause

/** @file fftw_execute_wrapper.c
 *
 *  @brief Contains wrapper implementations of FFTW Fortran execute APIs
 *
 *  This file contains implementations for the Fortran execute APIs provided by
 * FFTW.
 */

#include "src/translator/fftz_translator.h"

FFTZ_VOID dfftw_execute_(fftw_plan *const p)
{
    fftw_execute(*p);
}

FFTZ_VOID dfftwf_execute_(fftwf_plan *const p)
{
    fftwf_execute(*p);
}

FFTZ_VOID dfftw_execute_dft_(fftw_plan *p, fftw_complex *in, fftw_complex *out)
{
    fftw_execute_dft(*p, in, out);
}

FFTZ_VOID dfftwf_execute_dft_(fftwf_plan *p, fftwf_complex *in,
                              fftwf_complex *out)
{
    fftwf_execute_dft(*p, in, out);
}

FFTZ_VOID dfftw_execute_dft_r2c_(fftw_plan *p, FFTZ_DOUBLE *in,
                                 fftw_complex *out)
{
    fftw_execute_dft_r2c(*p, in, out);
}

FFTZ_VOID dfftwf_execute_dft_r2c_(fftwf_plan *p, FFTZ_FLOAT *in,
                                  fftwf_complex *out)
{
    fftwf_execute_dft_r2c(*p, in, out);
}

FFTZ_VOID dfftw_execute_dft_c2r_(fftw_plan *p, fftw_complex *in,
                                 FFTZ_DOUBLE *out)
{
    fftw_execute_dft_c2r(*p, in, out);
}

FFTZ_VOID dfftwf_execute_dft_c2r_(fftwf_plan *p, fftwf_complex *in,
                                  FFTZ_FLOAT *out)
{
    fftwf_execute_dft_c2r(*p, in, out);
}
