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

/** @file fftw_fortran_wrapper.c
 *
 *  @brief Contains wrapper implementations of FFTW fortran APIs
 *
 *  This file contains implementations for the fortran APIs provided by FFTW.
 */

#include "src/fftw_wrapper.h"

VOID dfftw_init_threads_(INT32 *ret)
{
    *ret = fftw_init_threads();
}

VOID dfftwf_init_threads_(INT32 *ret)
{
    *ret = fftwf_init_threads();
}

VOID dfftw_plan_with_nthreads_(INT32 *nthreads)
{
    fftw_plan_with_nthreads(*nthreads);
}

VOID dfftwf_plan_with_nthreads_(INT32 *nthreads)
{
    fftwf_plan_with_nthreads(*nthreads);
}

VOID dfftw_plan_dft_3d_(fftw_plan *p, INT32 *nx, INT32 *ny, INT32 *nz,
                        fftw_complex *in, fftw_complex *out, INT32 *sign,
                        INT32 *flags)
{
    *p = fftw_plan_dft_3d(*nx, *ny, *nz, in, out, *sign, *flags);
}

VOID dfftwf_plan_dft_3d_(fftwf_plan *p, INT32 *nx, INT32 *ny, INT32 *nz,
                         fftwf_complex *in, fftwf_complex *out, INT32 *sign,
                         INT32 *flags)
{
    *p = fftwf_plan_dft_3d(*nx, *ny, *nz, in, out, *sign, *flags);
}

VOID dfftw_plan_dft_r2c_3d_(fftw_plan *p, INT32 *nx, INT32 *ny, INT32 *nz,
                            DOUBLE *in, fftw_complex *out, INT32 *flags)
{
    *p = fftw_plan_dft_r2c_3d(*nx, *ny, *nz, in, out, *flags);
}

VOID dfftwf_plan_dft_r2c_3d_(fftwf_plan *p, INT32 *nx, INT32 *ny, INT32 *nz,
                             FLOAT *in, fftwf_complex *out, INT32 *flags)
{
    *p = fftwf_plan_dft_r2c_3d(*nx, *ny, *nz, in, out, *flags);
}

VOID dfftw_plan_dft_c2r_3d_(fftw_plan *p, INT32 *nx, INT32 *ny, INT32 *nz,
                            fftw_complex *in, DOUBLE *out, INT32 *flags)
{
    *p = fftw_plan_dft_c2r_3d(*nx, *ny, *nz, in, out, *flags);
}

VOID dfftwf_plan_dft_c2r_3d_(fftwf_plan *p, INT32 *nx, INT32 *ny, INT32 *nz,
                             fftwf_complex *in, FLOAT *out, INT32 *flags)
{
    *p = fftwf_plan_dft_c2r_3d(*nx, *ny, *nz, in, out, *flags);
}

VOID dfftw_plan_many_dft_(fftw_plan *p, INT32 *rank, const INT32 *n,
                          INT32 *howmany, fftw_complex *in,
                          const INT32 *inembed, INT32 *istride, INT32 *idist,
                          fftw_complex *out, const INT32 *onembed,
                          INT32 *ostride, INT32 *odist, INT32 *sign,
                          INT32 *flags)
{
    *p = fftw_plan_many_dft(*rank, n, *howmany, in, inembed, *istride, *idist,
                            out, onembed, *ostride, *odist, *sign, *flags);
}

VOID dfftwf_plan_many_dft_(fftwf_plan *p, INT32 *rank, const INT32 *n,
                           INT32 *howmany, fftwf_complex *in,
                           const INT32 *inembed, INT32 *istride, INT32 *idist,
                           fftwf_complex *out, const INT32 *onembed,
                           INT32 *ostride, INT32 *odist, INT32 *sign,
                           INT32 *flags)
{
    *p = fftwf_plan_many_dft(*rank, n, *howmany, in, inembed, *istride, *idist,
                             out, onembed, *ostride, *odist, *sign, *flags);
}

VOID dfftw_plan_many_dft_r2c_(fftw_plan *p, INT32 *rank, const INT32 *n,
                              INT32 *howmany, DOUBLE *in, const INT32 *inembed,
                              INT32 *istride, INT32 *idist, fftw_complex *out,
                              const INT32 *onembed, INT32 *ostride,
                              INT32 *odist, INT32 *flags)
{
    *p = fftw_plan_many_dft_r2c(*rank, n, *howmany, in, inembed, *istride,
                                *idist, out, onembed, *ostride, *odist, *flags);
}

VOID dfftwf_plan_many_dft_r2c_(fftwf_plan *p, INT32 *rank, const INT32 *n,
                               INT32 *howmany, FLOAT *in, const INT32 *inembed,
                               INT32 *istride, INT32 *idist, fftwf_complex *out,
                               const INT32 *onembed, INT32 *ostride,
                               INT32 *odist, INT32 *flags)
{
    *p = fftwf_plan_many_dft_r2c(*rank, n, *howmany, in, inembed, *istride,
                                 *idist, out, onembed, *ostride, *odist,
                                 *flags);
}

VOID dfftw_plan_many_dft_c2r_(fftw_plan *p, INT32 *rank, const INT32 *n,
                              INT32 *howmany, fftw_complex *in,
                              const INT32 *inembed, INT32 *istride,
                              INT32 *idist, DOUBLE *out, const INT32 *onembed,
                              INT32 *ostride, INT32 *odist, INT32 *flags)
{
    *p = fftw_plan_many_dft_c2r(*rank, n, *howmany, in, inembed, *istride,
                                *idist, out, onembed, *ostride, *odist, *flags);
}

VOID dfftwf_plan_many_dft_c2r_(fftwf_plan *p, INT32 *rank, const INT32 *n,
                               INT32 *howmany, fftwf_complex *in,
                               const INT32 *inembed, INT32 *istride,
                               INT32 *idist, FLOAT *out, const INT32 *onembed,
                               INT32 *ostride, INT32 *odist, INT32 *flags)
{
    *p = fftwf_plan_many_dft_c2r(*rank, n, *howmany, in, inembed, *istride,
                                 *idist, out, onembed, *ostride, *odist,
                                 *flags);
}

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

VOID dfftw_destroy_plan_(fftw_plan *p)
{
    fftw_destroy_plan(*p);
}

VOID dfftwf_destroy_plan_(fftwf_plan *p)
{
    fftwf_destroy_plan(*p);
}
