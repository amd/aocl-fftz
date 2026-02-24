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

/** @file fftw_execute_wrapper.c
 *
 *  @brief Contains wrapper implementations of FFTW Fortran execute APIs
 *
 *  This file contains implementations for the Fortran execute APIs provided by
 * FFTW.
 */

#include "src/translator/fftz_translator.h"

VOID dfftw_execute_(fftw_plan *const p)
{
    fftw_execute(*p);
}

VOID dfftwf_execute_(fftwf_plan *const p)
{
    fftwf_execute(*p);
}

VOID dfftw_execute_dft_(fftw_plan *p, fftw_complex *in, fftw_complex *out)
{
    fftw_execute_dft(*p, in, out);
}

VOID dfftwf_execute_dft_(fftwf_plan *p, fftwf_complex *in, fftwf_complex *out)
{
    fftwf_execute_dft(*p, in, out);
}

VOID dfftw_execute_dft_r2c_(fftw_plan *p, DOUBLE *in, fftw_complex *out)
{
    fftw_execute_dft_r2c(*p, in, out);
}

VOID dfftwf_execute_dft_r2c_(fftwf_plan *p, FLOAT *in, fftwf_complex *out)
{
    fftwf_execute_dft_r2c(*p, in, out);
}

VOID dfftw_execute_dft_c2r_(fftw_plan *p, fftw_complex *in, DOUBLE *out)
{
    fftw_execute_dft_c2r(*p, in, out);
}

VOID dfftwf_execute_dft_c2r_(fftwf_plan *p, fftwf_complex *in, FLOAT *out)
{
    fftwf_execute_dft_c2r(*p, in, out);
}
