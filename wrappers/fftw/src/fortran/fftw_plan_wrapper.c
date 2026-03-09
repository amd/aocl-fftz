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

/** @file fftw_plan_wrapper.c
 *
 *  @brief Contains wrapper implementations of FFTW planner Fortran APIs
 *
 *  This file contains implementations for the plan Fortran APIs provided by
 *  FFTW.
 */

#include "src/translator/fftz_translator.h"

VOID dfftw_plan_dft_(fftw_plan *p, INT32 *rank, const INT32 *n,
                     fftw_complex *in, fftw_complex *out, INT32 *sign,
                     INT32 *flags)
{
    INT32 *reversed_n = NULL;
    INT32 dim_rank = *rank;
    ALLOC_ALIGN_UNINIT(reversed_n, INT32, dim_rank * sizeof(INT32));
    if (reversed_n == NULL)
    {
        return;
    }
    // since the dim.n is passed in reverse order as compared to that of C APIs,
    // reverse it for Fortran API.
    reverse_array(n, reversed_n, dim_rank);
    *p = fftw_plan_dft(dim_rank, reversed_n, in, out, *sign, *flags);
    FREE_ALIGN_ALLOCATED_MEM(reversed_n);
}

VOID dfftwf_plan_dft_(fftwf_plan *p, INT32 *rank, const INT32 *n,
                      fftwf_complex *in, fftwf_complex *out, INT32 *sign,
                      INT32 *flags)
{
    INT32 *reversed_n = NULL;
    INT32 dim_rank = *rank;
    ALLOC_ALIGN_UNINIT(reversed_n, INT32, dim_rank * sizeof(INT32));
    if (reversed_n == NULL)
    {
        return;
    }
    // since the dim.n is passed in reverse order as compared to that of C APIs,
    // reverse it for Fortran API.
    reverse_array(n, reversed_n, dim_rank);
    *p = fftwf_plan_dft(dim_rank, reversed_n, in, out, *sign, *flags);
    FREE_ALIGN_ALLOCATED_MEM(reversed_n);
}

VOID dfftw_plan_dft_1d_(fftw_plan *p, INT32 *n, fftw_complex *in,
                        fftw_complex *out, INT32 *sign, INT32 *flags)
{
    *p = fftw_plan_dft_1d(*n, in, out, *sign, *flags);
}

VOID dfftwf_plan_dft_1d_(fftwf_plan *p, INT32 *n, fftwf_complex *in,
                         fftwf_complex *out, INT32 *sign, INT32 *flags)
{
    *p = fftwf_plan_dft_1d(*n, in, out, *sign, *flags);
}

VOID dfftw_plan_dft_2d_(fftw_plan *p, INT32 *nx, INT32 *ny, fftw_complex *in,
                        fftw_complex *out, INT32 *sign, INT32 *flags)
{
    *p = fftw_plan_dft_2d(*ny, *nx, in, out, *sign, *flags);
}

VOID dfftwf_plan_dft_2d_(fftwf_plan *p, INT32 *nx, INT32 *ny, fftwf_complex *in,
                         fftwf_complex *out, INT32 *sign, INT32 *flags)
{
    *p = fftwf_plan_dft_2d(*ny, *nx, in, out, *sign, *flags);
}

VOID dfftw_plan_dft_3d_(fftw_plan *p, INT32 *nx, INT32 *ny, INT32 *nz,
                        fftw_complex *in, fftw_complex *out, INT32 *sign,
                        INT32 *flags)
{
    *p = fftw_plan_dft_3d(*nz, *ny, *nx, in, out, *sign, *flags);
}

VOID dfftwf_plan_dft_3d_(fftwf_plan *p, INT32 *nx, INT32 *ny, INT32 *nz,
                         fftwf_complex *in, fftwf_complex *out, INT32 *sign,
                         INT32 *flags)
{
    *p = fftwf_plan_dft_3d(*nz, *ny, *nx, in, out, *sign, *flags);
}

VOID dfftw_plan_many_dft_(fftw_plan *p, INT32 *rank, const INT32 *n,
                          INT32 *howmany, fftw_complex *in,
                          const INT32 *inembed, INT32 *istride, INT32 *idist,
                          fftw_complex *out, const INT32 *onembed,
                          INT32 *ostride, INT32 *odist, INT32 *sign,
                          INT32 *flags)
{
    // since the dim.n is passed in reverse order as compared to that of C APIs,
    // reverse it for Fortran API.

    INT32 dim_rank = *rank;
    *p = NULL;

    INT32 *reversed_n = NULL;
    INT32 *reversed_inembed = NULL;
    INT32 *reversed_onembed = NULL;

    ALLOC_ALIGN_UNINIT(reversed_n, INT32, dim_rank * sizeof(INT32));

    if (reversed_n == NULL)
    {
        return;
    }

    reverse_array(n, reversed_n, dim_rank);

    if (inembed != NULL)
    {
        ALLOC_ALIGN_UNINIT(reversed_inembed, INT32, dim_rank * sizeof(INT32));
        if (reversed_inembed == NULL)
        {
            goto free_and_exit;
        }
        reverse_array(inembed, reversed_inembed, dim_rank);
    }

    if (onembed != NULL)
    {
        ALLOC_ALIGN_UNINIT(reversed_onembed, INT32, dim_rank * sizeof(INT32));
        if (reversed_onembed == NULL)
        {
            goto free_and_exit;
        }
        reverse_array(onembed, reversed_onembed, dim_rank);
    }

    *p = fftw_plan_many_dft(dim_rank, reversed_n, *howmany, in, reversed_inembed, *istride,
                            *idist, out, reversed_onembed, *ostride, *odist, *sign,
                            *flags);

free_and_exit:
    FREE_ALIGN_ALLOCATED_MEM(reversed_n);
    FREE_ALIGN_ALLOCATED_MEM(reversed_inembed);
    FREE_ALIGN_ALLOCATED_MEM(reversed_onembed);
}

VOID dfftwf_plan_many_dft_(fftwf_plan *p, INT32 *rank, const INT32 *n,
                           INT32 *howmany, fftwf_complex *in,
                           const INT32 *inembed, INT32 *istride, INT32 *idist,
                           fftwf_complex *out, const INT32 *onembed,
                           INT32 *ostride, INT32 *odist, INT32 *sign,
                           INT32 *flags)
{
    // since the dim.n is passed in reverse order as compared to that of C APIs,
    // reverse it for Fortran API.

    INT32 dim_rank = *rank;
    *p = NULL;

    INT32 *reversed_n = NULL;
    INT32 *reversed_inembed = NULL;
    INT32 *reversed_onembed = NULL;

    ALLOC_ALIGN_UNINIT(reversed_n, INT32, dim_rank * sizeof(INT32));

    if (reversed_n == NULL)
    {
        return;
    }

    reverse_array(n, reversed_n, dim_rank);

    if (inembed != NULL)
    {
        ALLOC_ALIGN_UNINIT(reversed_inembed, INT32, dim_rank * sizeof(INT32));
        if (reversed_inembed == NULL)
        {
            goto free_and_exit;
        }
        reverse_array(inembed, reversed_inembed, dim_rank);
    }

    if (onembed != NULL)
    {
        ALLOC_ALIGN_UNINIT(reversed_onembed, INT32, dim_rank * sizeof(INT32));
        if (reversed_onembed == NULL)
        {
            goto free_and_exit;
        }
        reverse_array(onembed, reversed_onembed, dim_rank);
    }

    *p = fftwf_plan_many_dft(dim_rank, reversed_n, *howmany, in, reversed_inembed, *istride,
                             *idist, out, reversed_onembed, *ostride, *odist, *sign,
                             *flags);

free_and_exit:
    FREE_ALIGN_ALLOCATED_MEM(reversed_n);
    FREE_ALIGN_ALLOCATED_MEM(reversed_inembed);
    FREE_ALIGN_ALLOCATED_MEM(reversed_onembed);
}

VOID dfftw_plan_guru_dft_(fftw_plan *p, INT32 *rank, const INT32 *n,
                          const INT32 *is, const INT32 *os, INT32 *howmany_rank,
                          const INT32 *h_n, const INT32 *h_is,
                          const INT32 *h_os, fftw_complex *in,
                          fftw_complex *out, INT32 *sign, INT32 *flags)
{
    dv_desc *p_dv_desc = get_fortran_guru_dv_desc(*rank, n, is, os,
                                                  *howmany_rank, h_n, h_is,
                                                  h_os);
    if (p_dv_desc == NULL)
    {
        *p = NULL;
        return;
    }
    *p = get_handle_d(p_dv_desc, *sign, in, out, COMPLEX);
}

VOID dfftwf_plan_guru_dft_(fftwf_plan *p, INT32 *rank, const INT32 *n,
                           const INT32 *is, const INT32 *os,
                           INT32 *howmany_rank, const INT32 *h_n,
                           const INT32 *h_is, const INT32 *h_os,
                           fftwf_complex *in, fftwf_complex *out, INT32 *sign,
                           INT32 *flags)
{
    dv_desc *p_dv_desc = get_fortran_guru_dv_desc(*rank, n, is, os,
                                                  *howmany_rank, h_n, h_is,
                                                  h_os);
    if (p_dv_desc == NULL)
    {
        *p = NULL;
        return;
    }
    *p = get_handle_f(p_dv_desc, *sign, in, out, COMPLEX);
}

VOID dfftw_plan_dft_r2c_(fftw_plan *p, INT32 *rank, const INT32 *n, DOUBLE *in,
                         fftw_complex *out, INT32 *flags)
{
    INT32 *reversed_n = NULL;
    INT32 dim_rank = *rank;
    ALLOC_ALIGN_UNINIT(reversed_n, INT32, dim_rank * sizeof(INT32));
    if (reversed_n == NULL)
    {
        return;
    }
    // since the dim.n is passed in reverse order as compared to that of C APIs,
    // reverse it for Fortran API.
    reverse_array(n, reversed_n, dim_rank);
    *p = fftw_plan_dft_r2c(dim_rank, reversed_n, in, out, *flags);
    FREE_ALIGN_ALLOCATED_MEM(reversed_n);
}

VOID dfftwf_plan_dft_r2c_(fftwf_plan *p, INT32 *rank, const INT32 *n, FLOAT *in,
                          fftwf_complex *out, INT32 *flags)
{
    INT32 *reversed_n = NULL;
    INT32 dim_rank = *rank;
    ALLOC_ALIGN_UNINIT(reversed_n, INT32, dim_rank * sizeof(INT32));
    if (reversed_n == NULL)
    {
        return;
    }
    // since the dim.n is passed in reverse order as compared to that of C APIs,
    // reverse it for Fortran API.
    reverse_array(n, reversed_n, dim_rank);
    *p = fftwf_plan_dft_r2c(dim_rank, reversed_n, in, out, *flags);
    FREE_ALIGN_ALLOCATED_MEM(reversed_n);
}

VOID dfftw_plan_dft_r2c_1d_(fftw_plan *p, INT32 *n, DOUBLE *in,
                            fftw_complex *out, INT32 *flags)
{
    *p = fftw_plan_dft_r2c_1d(*n, in, out, *flags);
}

VOID dfftwf_plan_dft_r2c_1d_(fftwf_plan *p, INT32 *n, FLOAT *in,
                             fftwf_complex *out, INT32 *flags)
{
    *p = fftwf_plan_dft_r2c_1d(*n, in, out, *flags);
}

VOID dfftw_plan_dft_r2c_2d_(fftw_plan *p, INT32 *nx, INT32 *ny, DOUBLE *in,
                            fftw_complex *out, INT32 *flags)
{
    *p = fftw_plan_dft_r2c_2d(*ny, *nx, in, out, *flags);
}

VOID dfftwf_plan_dft_r2c_2d_(fftwf_plan *p, INT32 *nx, INT32 *ny, FLOAT *in,
                             fftwf_complex *out, INT32 *flags)
{
    *p = fftwf_plan_dft_r2c_2d(*ny, *nx, in, out, *flags);
}

VOID dfftw_plan_dft_r2c_3d_(fftw_plan *p, INT32 *nx, INT32 *ny, INT32 *nz,
                            DOUBLE *in, fftw_complex *out, INT32 *flags)
{
    *p = fftw_plan_dft_r2c_3d(*nz, *ny, *nx, in, out, *flags);
}

VOID dfftwf_plan_dft_r2c_3d_(fftwf_plan *p, INT32 *nx, INT32 *ny, INT32 *nz,
                             FLOAT *in, fftwf_complex *out, INT32 *flags)
{
    *p = fftwf_plan_dft_r2c_3d(*nz, *ny, *nx, in, out, *flags);
}

VOID dfftw_plan_many_dft_r2c_(fftw_plan *p, INT32 *rank, const INT32 *n,
                              INT32 *howmany, DOUBLE *in, const INT32 *inembed,
                              INT32 *istride, INT32 *idist, fftw_complex *out,
                              const INT32 *onembed, INT32 *ostride,
                              INT32 *odist, INT32 *flags)
{
    // since the dim.n is passed in reverse order as compared to that of C APIs,
    // reverse it for Fortran API.

    INT32 dim_rank = *rank;
    *p = NULL;

    INT32 *reversed_n = NULL;
    INT32 *reversed_inembed = NULL;
    INT32 *reversed_onembed = NULL;

    ALLOC_ALIGN_UNINIT(reversed_n, INT32, dim_rank * sizeof(INT32));

    if (reversed_n == NULL)
    {
        return;
    }

    reverse_array(n, reversed_n, dim_rank);

    if (inembed != NULL)
    {
        ALLOC_ALIGN_UNINIT(reversed_inembed, INT32, dim_rank * sizeof(INT32));
        if (reversed_inembed == NULL)
        {
            goto free_and_exit;
        }
        reverse_array(inembed, reversed_inembed, dim_rank);
    }

    if (onembed != NULL)
    {
        ALLOC_ALIGN_UNINIT(reversed_onembed, INT32, dim_rank * sizeof(INT32));
        if (reversed_onembed == NULL)
        {
            goto free_and_exit;
        }
        reverse_array(onembed, reversed_onembed, dim_rank);
    }

    *p = fftw_plan_many_dft_r2c(dim_rank, reversed_n, *howmany, in, reversed_inembed,
                                *istride, *idist, out, reversed_onembed, *ostride,
                                *odist, *flags);

free_and_exit:
    FREE_ALIGN_ALLOCATED_MEM(reversed_n);
    FREE_ALIGN_ALLOCATED_MEM(reversed_inembed);
    FREE_ALIGN_ALLOCATED_MEM(reversed_onembed);
}

VOID dfftwf_plan_many_dft_r2c_(fftwf_plan *p, INT32 *rank, const INT32 *n,
                               INT32 *howmany, FLOAT *in, const INT32 *inembed,
                               INT32 *istride, INT32 *idist, fftwf_complex *out,
                               const INT32 *onembed, INT32 *ostride,
                               INT32 *odist, INT32 *flags)
{
    // since the dim.n is passed in reverse order as compared to that of C APIs,
    // reverse it for Fortran API.

    INT32 dim_rank = *rank;
    *p = NULL;

    INT32 *reversed_n = NULL;
    INT32 *reversed_inembed = NULL;
    INT32 *reversed_onembed = NULL;

    ALLOC_ALIGN_UNINIT(reversed_n, INT32, dim_rank * sizeof(INT32));

    if (reversed_n == NULL)
    {
        return;
    }

    reverse_array(n, reversed_n, dim_rank);

    if (inembed != NULL)
    {
        ALLOC_ALIGN_UNINIT(reversed_inembed, INT32, dim_rank * sizeof(INT32));
        if (reversed_inembed == NULL)
        {
            goto free_and_exit;
        }
        reverse_array(inembed, reversed_inembed, dim_rank);
    }

    if (onembed != NULL)
    {
        ALLOC_ALIGN_UNINIT(reversed_onembed, INT32, dim_rank * sizeof(INT32));
        if (reversed_onembed == NULL)
        {
            goto free_and_exit;
        }
        reverse_array(onembed, reversed_onembed, dim_rank);
    }

    *p = fftwf_plan_many_dft_r2c(dim_rank, reversed_n, *howmany, in, reversed_inembed,
                                 *istride, *idist, out, reversed_onembed, *ostride,
                                 *odist, *flags);

free_and_exit:
    FREE_ALIGN_ALLOCATED_MEM(reversed_n);
    FREE_ALIGN_ALLOCATED_MEM(reversed_inembed);
    FREE_ALIGN_ALLOCATED_MEM(reversed_onembed);
}

VOID dfftw_plan_guru_dft_r2c_(fftw_plan *p, INT32 *rank, const INT32 *n,
                              const INT32 *is, const INT32 *os,
                              INT32 *howmany_rank, const INT32 *h_n,
                              const INT32 *h_is, const INT32 *h_os, DOUBLE *in,
                              fftw_complex *out, INT32 *flags)
{
    dv_desc *p_dv_desc = get_fortran_guru_dv_desc(*rank, n, is, os,
                                                  *howmany_rank, h_n, h_is,
                                                  h_os);
    if (p_dv_desc == NULL)
    {
        *p = NULL;
        return;
    }
    *p = get_handle_d(p_dv_desc, FFTW_FORWARD, in, out, REAL);
}

VOID dfftwf_plan_guru_dft_r2c_(fftwf_plan *p, INT32 *rank, const INT32 *n,
                               const INT32 *is, const INT32 *os,
                               INT32 *howmany_rank, const INT32 *h_n,
                               const INT32 *h_is, const INT32 *h_os, FLOAT *in,
                               fftwf_complex *out, INT32 *flags)
{
    dv_desc *p_dv_desc = get_fortran_guru_dv_desc(*rank, n, is, os,
                                                  *howmany_rank, h_n, h_is,
                                                  h_os);
    if (p_dv_desc == NULL)
    {
        *p = NULL;
        return;
    }
    *p = get_handle_f(p_dv_desc, FFTW_FORWARD, in, out, REAL);
}

VOID dfftw_plan_dft_c2r_(fftw_plan *p, INT32 *rank, const INT32 *n,
                         fftw_complex *in, DOUBLE *out, INT32 *flags)
{
    INT32 *reversed_n = NULL;
    INT32 dim_rank = *rank;
    ALLOC_ALIGN_UNINIT(reversed_n, INT32, dim_rank * sizeof(INT32));
    if (reversed_n == NULL)
    {
        return;
    }
    // since the dim.n is passed in reverse order as compared to that of C APIs,
    // reverse it for Fortran API.
    reverse_array(n, reversed_n, dim_rank);
    *p = fftw_plan_dft_c2r(dim_rank, reversed_n, in, out, *flags);
    FREE_ALIGN_ALLOCATED_MEM(reversed_n);
}

VOID dfftwf_plan_dft_c2r_(fftwf_plan *p, INT32 *rank, const INT32 *n,
                          fftwf_complex *in, FLOAT *out, INT32 *flags)
{
    INT32 *reversed_n = NULL;
    INT32 dim_rank = *rank;
    ALLOC_ALIGN_UNINIT(reversed_n, INT32, dim_rank * sizeof(INT32));
    if (reversed_n == NULL)
    {
        return;
    }
    // since the dim.n is passed in reverse order as compared to that of C APIs,
    // reverse it for Fortran API.
    reverse_array(n, reversed_n, dim_rank);
    *p = fftwf_plan_dft_c2r(dim_rank, reversed_n, in, out, *flags);
    FREE_ALIGN_ALLOCATED_MEM(reversed_n);
}

VOID dfftw_plan_dft_c2r_1d_(fftw_plan *p, INT32 *n, fftw_complex *in,
                            DOUBLE *out, INT32 *flags)
{
    *p = fftw_plan_dft_c2r_1d(*n, in, out, *flags);
}

VOID dfftwf_plan_dft_c2r_1d_(fftwf_plan *p, INT32 *n, fftwf_complex *in,
                             FLOAT *out, INT32 *flags)
{
    *p = fftwf_plan_dft_c2r_1d(*n, in, out, *flags);
}

VOID dfftw_plan_dft_c2r_2d_(fftw_plan *p, INT32 *nx, INT32 *ny,
                            fftw_complex *in, DOUBLE *out, INT32 *flags)
{
    *p = fftw_plan_dft_c2r_2d(*ny, *nx, in, out, *flags);
}

VOID dfftwf_plan_dft_c2r_2d_(fftwf_plan *p, INT32 *nx, INT32 *ny,
                             fftwf_complex *in, FLOAT *out, INT32 *flags)
{
    *p = fftwf_plan_dft_c2r_2d(*ny, *nx, in, out, *flags);
}

VOID dfftw_plan_dft_c2r_3d_(fftw_plan *p, INT32 *nx, INT32 *ny, INT32 *nz,
                            fftw_complex *in, DOUBLE *out, INT32 *flags)
{
    *p = fftw_plan_dft_c2r_3d(*nz, *ny, *nx, in, out, *flags);
}

VOID dfftwf_plan_dft_c2r_3d_(fftwf_plan *p, INT32 *nx, INT32 *ny, INT32 *nz,
                             fftwf_complex *in, FLOAT *out, INT32 *flags)
{
    *p = fftwf_plan_dft_c2r_3d(*nz, *ny, *nx, in, out, *flags);
}

VOID dfftw_plan_many_dft_c2r_(fftw_plan *p, INT32 *rank, const INT32 *n,
                              INT32 *howmany, fftw_complex *in,
                              const INT32 *inembed, INT32 *istride,
                              INT32 *idist, DOUBLE *out, const INT32 *onembed,
                              INT32 *ostride, INT32 *odist, INT32 *flags)
{
    // since the dim.n is passed in reverse order as compared to that of C APIs,
    // reverse it for Fortran API.

    INT32 dim_rank = *rank;
    *p = NULL;

    INT32 *reversed_n = NULL;
    INT32 *reversed_inembed = NULL;
    INT32 *reversed_onembed = NULL;

    ALLOC_ALIGN_UNINIT(reversed_n, INT32, dim_rank * sizeof(INT32));

    if (reversed_n == NULL)
    {
        return;
    }

    reverse_array(n, reversed_n, dim_rank);

    if (inembed != NULL)
    {
        ALLOC_ALIGN_UNINIT(reversed_inembed, INT32, dim_rank * sizeof(INT32));
        if (reversed_inembed == NULL)
        {
            goto free_and_exit;
        }
        reverse_array(inembed, reversed_inembed, dim_rank);
    }

    if (onembed != NULL)
    {
        ALLOC_ALIGN_UNINIT(reversed_onembed, INT32, dim_rank * sizeof(INT32));
        if (reversed_onembed == NULL)
        {
            goto free_and_exit;
        }
        reverse_array(onembed, reversed_onembed, dim_rank);
    }

    *p = fftw_plan_many_dft_c2r(dim_rank, reversed_n, *howmany, in, reversed_inembed,
                                *istride, *idist, out, reversed_onembed, *ostride,
                                *odist, *flags);

free_and_exit:
    FREE_ALIGN_ALLOCATED_MEM(reversed_n);
    FREE_ALIGN_ALLOCATED_MEM(reversed_inembed);
    FREE_ALIGN_ALLOCATED_MEM(reversed_onembed);
}

VOID dfftwf_plan_many_dft_c2r_(fftwf_plan *p, INT32 *rank, const INT32 *n,
                               INT32 *howmany, fftwf_complex *in,
                               const INT32 *inembed, INT32 *istride,
                               INT32 *idist, FLOAT *out, const INT32 *onembed,
                               INT32 *ostride, INT32 *odist, INT32 *flags)
{
    // since the dim.n is passed in reverse order as compared to that of C APIs,
    // reverse it for Fortran API.

    INT32 dim_rank = *rank;
    *p = NULL;

    INT32 *reversed_n = NULL;
    INT32 *reversed_inembed = NULL;
    INT32 *reversed_onembed = NULL;

    ALLOC_ALIGN_UNINIT(reversed_n, INT32, dim_rank * sizeof(INT32));

    if (reversed_n == NULL)
    {
        return;
    }

    reverse_array(n, reversed_n, dim_rank);

    if (inembed != NULL)
    {
        ALLOC_ALIGN_UNINIT(reversed_inembed, INT32, dim_rank * sizeof(INT32));
        if (reversed_inembed == NULL)
        {
            goto free_and_exit;
        }
        reverse_array(inembed, reversed_inembed, dim_rank);
    }

    if (onembed != NULL)
    {
        ALLOC_ALIGN_UNINIT(reversed_onembed, INT32, dim_rank * sizeof(INT32));
        if (reversed_onembed == NULL)
        {
            goto free_and_exit;
        }
        reverse_array(onembed, reversed_onembed, dim_rank);
    }

    *p = fftwf_plan_many_dft_c2r(dim_rank, reversed_n, *howmany, in, reversed_inembed,
                                 *istride, *idist, out, reversed_onembed, *ostride,
                                 *odist, *flags);

free_and_exit:
    FREE_ALIGN_ALLOCATED_MEM(reversed_n);
    FREE_ALIGN_ALLOCATED_MEM(reversed_inembed);
    FREE_ALIGN_ALLOCATED_MEM(reversed_onembed);
}

VOID dfftw_plan_guru_dft_c2r_(fftw_plan *p, INT32 *rank, const INT32 *n,
                              const INT32 *is, const INT32 *os,
                              INT32 *howmany_rank, const INT32 *h_n,
                              const INT32 *h_is, const INT32 *h_os,
                              fftw_complex *in, DOUBLE *out, INT32 *flags)
{
    dv_desc *p_dv_desc = get_fortran_guru_dv_desc(*rank, n, is, os,
                                                  *howmany_rank, h_n, h_is,
                                                  h_os);
    if (p_dv_desc == NULL)
    {
        *p = NULL;
        return;
    }
    *p = get_handle_d(p_dv_desc, FFTW_BACKWARD, in, out, REAL);
}

VOID dfftwf_plan_guru_dft_c2r_(fftwf_plan *p, INT32 *rank, const INT32 *n,
                               const INT32 *is, const INT32 *os,
                               INT32 *howmany_rank, const INT32 *h_n,
                               const INT32 *h_is, const INT32 *h_os,
                               fftwf_complex *in, FLOAT *out, INT32 *flags)
{
    dv_desc *p_dv_desc = get_fortran_guru_dv_desc(*rank, n, is, os,
                                                  *howmany_rank, h_n, h_is,
                                                  h_os);
    if (p_dv_desc == NULL)
    {
        *p = NULL;
        return;
    }
    *p = get_handle_f(p_dv_desc, FFTW_BACKWARD, in, out, REAL);
}
