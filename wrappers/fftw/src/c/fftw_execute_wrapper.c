// Copyright Advanced Micro Devices, Inc.
// SPDX-License-Identifier: BSD-3-Clause

/** @file fftw_execute_wrapper.c
 *
 *  @brief Contains wrapper implementations of FFTW execute APIs
 *
 *  This file contains implementations for the execute APIs provided by FFTW.
 */

#include "src/translator/fftz_translator.h"

VOID fftw_execute(const fftw_plan p)
{
    aoclfftz_execute((VOID *)p);
}

VOID fftwf_execute(const fftwf_plan p)
{
    aoclfftz_execute((VOID *)p);
}

VOID fftw_execute_dft(const fftw_plan p, fftw_complex *in, fftw_complex *out)
{
    aoclfftz_execute_io((VOID *)p, (VOID *)in, (VOID *)out);
}

VOID fftwf_execute_dft(const fftwf_plan p, fftwf_complex *in,
                       fftwf_complex *out)
{
    aoclfftz_execute_io((VOID *)p, (VOID *)in, (VOID *)out);
}

VOID fftw_execute_dft_r2c(const fftw_plan p, DOUBLE *in, fftw_complex *out)
{
    aoclfftz_execute_io((VOID *)p, (VOID *)in, (VOID *)out);
}

VOID fftwf_execute_dft_r2c(const fftwf_plan p, FLOAT *in, fftwf_complex *out)
{
    aoclfftz_execute_io((VOID *)p, (VOID *)in, (VOID *)out);
}

VOID fftw_execute_dft_c2r(const fftw_plan p, fftw_complex *in, DOUBLE *out)
{
    aoclfftz_execute_io((VOID *)p, (VOID *)in, (VOID *)out);
}

VOID fftwf_execute_dft_c2r(const fftwf_plan p, fftwf_complex *in, FLOAT *out)
{
    aoclfftz_execute_io((VOID *)p, (VOID *)in, (VOID *)out);
}
