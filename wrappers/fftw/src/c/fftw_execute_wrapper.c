// Copyright Advanced Micro Devices, Inc.
// SPDX-License-Identifier: BSD-3-Clause

/** @file fftw_execute_wrapper.c
 *
 *  @brief Contains wrapper implementations of FFTW execute APIs
 *
 *  This file contains implementations for the execute APIs provided by FFTW.
 */

#include "src/translator/fftz_translator.h"

FFTZ_VOID fftw_execute(const fftw_plan p)
{
    aoclfftz_execute((FFTZ_VOID *)p);
}

FFTZ_VOID fftwf_execute(const fftwf_plan p)
{
    aoclfftz_execute((FFTZ_VOID *)p);
}

FFTZ_VOID fftw_execute_dft(const fftw_plan p, fftw_complex *in,
                           fftw_complex *out)
{
    aoclfftz_execute_io((FFTZ_VOID *)p, (FFTZ_VOID *)in, (FFTZ_VOID *)out);
}

FFTZ_VOID fftwf_execute_dft(const fftwf_plan p, fftwf_complex *in,
                       fftwf_complex *out)
{
    aoclfftz_execute_io((FFTZ_VOID *)p, (FFTZ_VOID *)in, (FFTZ_VOID *)out);
}

FFTZ_VOID fftw_execute_dft_r2c(const fftw_plan p, FFTZ_DOUBLE *in,
                               fftw_complex *out)
{
    aoclfftz_execute_io((FFTZ_VOID *)p, (FFTZ_VOID *)in, (FFTZ_VOID *)out);
}

FFTZ_VOID fftwf_execute_dft_r2c(const fftwf_plan p, FFTZ_FLOAT *in,
                                fftwf_complex *out)
{
    aoclfftz_execute_io((FFTZ_VOID *)p, (FFTZ_VOID *)in, (FFTZ_VOID *)out);
}

FFTZ_VOID fftw_execute_dft_c2r(const fftw_plan p, fftw_complex *in,
                               FFTZ_DOUBLE *out)
{
    aoclfftz_execute_io((FFTZ_VOID *)p, (FFTZ_VOID *)in, (FFTZ_VOID *)out);
}

FFTZ_VOID fftwf_execute_dft_c2r(const fftwf_plan p, fftwf_complex *in,
                                FFTZ_FLOAT *out)
{
    aoclfftz_execute_io((FFTZ_VOID *)p, (FFTZ_VOID *)in, (FFTZ_VOID *)out);
}
