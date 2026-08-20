// Copyright Advanced Micro Devices, Inc.
// SPDX-License-Identifier: BSD-3-Clause

/** @file fftw_plan_wrapper.c
 *
 *  @brief Contains wrapper implementations of FFTW planner Fortran APIs
 *
 *  This file contains implementations for the plan Fortran APIs provided by
 *  FFTW.
 */

#include "src/translator/fftz_translator.h"

FFTZ_VOID dfftw_plan_dft_(fftw_plan *p, FFTZ_INT32 *rank, const FFTZ_INT32 *n,
                     fftw_complex *in, fftw_complex *out, FFTZ_INT32 *sign,
                     FFTZ_INT32 *flags)
{
    FFTZ_INT32 *reversed_n = NULL;
    FFTZ_INT32 dim_rank = *rank;
    ALLOC_ALIGN_UNINIT(reversed_n, FFTZ_INT32, dim_rank * sizeof(FFTZ_INT32));
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

FFTZ_VOID dfftwf_plan_dft_(fftwf_plan *p, FFTZ_INT32 *rank, const FFTZ_INT32 *n,
                      fftwf_complex *in, fftwf_complex *out, FFTZ_INT32 *sign,
                      FFTZ_INT32 *flags)
{
    FFTZ_INT32 *reversed_n = NULL;
    FFTZ_INT32 dim_rank = *rank;
    ALLOC_ALIGN_UNINIT(reversed_n, FFTZ_INT32, dim_rank * sizeof(FFTZ_INT32));
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

FFTZ_VOID dfftw_plan_dft_1d_(fftw_plan *p, FFTZ_INT32 *n, fftw_complex *in,
                        fftw_complex *out, FFTZ_INT32 *sign, FFTZ_INT32 *flags)
{
    *p = fftw_plan_dft_1d(*n, in, out, *sign, *flags);
}

FFTZ_VOID dfftwf_plan_dft_1d_(fftwf_plan *p, FFTZ_INT32 *n, fftwf_complex *in,
                              fftwf_complex *out, FFTZ_INT32 *sign,
                              FFTZ_INT32 *flags)
{
    *p = fftwf_plan_dft_1d(*n, in, out, *sign, *flags);
}

FFTZ_VOID dfftw_plan_dft_2d_(fftw_plan *p, FFTZ_INT32 *nx, FFTZ_INT32 *ny,
                             fftw_complex *in, fftw_complex *out,
                             FFTZ_INT32 *sign, FFTZ_INT32 *flags)
{
    *p = fftw_plan_dft_2d(*ny, *nx, in, out, *sign, *flags);
}

FFTZ_VOID dfftwf_plan_dft_2d_(fftwf_plan *p, FFTZ_INT32 *nx, FFTZ_INT32 *ny,
                              fftwf_complex *in, fftwf_complex *out,
                              FFTZ_INT32 *sign, FFTZ_INT32 *flags)
{
    *p = fftwf_plan_dft_2d(*ny, *nx, in, out, *sign, *flags);
}

FFTZ_VOID dfftw_plan_dft_3d_(fftw_plan *p, FFTZ_INT32 *nx, FFTZ_INT32 *ny,
                             FFTZ_INT32 *nz, fftw_complex *in,
                             fftw_complex *out, FFTZ_INT32 *sign,
                             FFTZ_INT32 *flags)
{
    *p = fftw_plan_dft_3d(*nz, *ny, *nx, in, out, *sign, *flags);
}

FFTZ_VOID dfftwf_plan_dft_3d_(fftwf_plan *p, FFTZ_INT32 *nx, FFTZ_INT32 *ny,
                              FFTZ_INT32 *nz, fftwf_complex *in,
                              fftwf_complex *out, FFTZ_INT32 *sign,
                              FFTZ_INT32 *flags)
{
    *p = fftwf_plan_dft_3d(*nz, *ny, *nx, in, out, *sign, *flags);
}

FFTZ_VOID dfftw_plan_many_dft_(fftw_plan *p, FFTZ_INT32 *rank,
                               const FFTZ_INT32 *n, FFTZ_INT32 *howmany,
                               fftw_complex *in, const FFTZ_INT32 *inembed,
                               FFTZ_INT32 *istride, FFTZ_INT32 *idist,
                               fftw_complex *out, const FFTZ_INT32 *onembed,
                               FFTZ_INT32 *ostride, FFTZ_INT32 *odist,
                               FFTZ_INT32 *sign, FFTZ_INT32 *flags)
{
    // since the dim.n is passed in reverse order as compared to that of C APIs,
    // reverse it for Fortran API.

    FFTZ_INT32 dim_rank = *rank;
    *p = NULL;

    FFTZ_INT32 *reversed_n = NULL;
    FFTZ_INT32 *reversed_inembed = NULL;
    FFTZ_INT32 *reversed_onembed = NULL;

    ALLOC_ALIGN_UNINIT(reversed_n, FFTZ_INT32, dim_rank * sizeof(FFTZ_INT32));

    if (reversed_n == NULL)
    {
        return;
    }

    reverse_array(n, reversed_n, dim_rank);

    if (inembed != NULL)
    {
        ALLOC_ALIGN_UNINIT(reversed_inembed, FFTZ_INT32,
                           dim_rank * sizeof(FFTZ_INT32));
        if (reversed_inembed == NULL)
        {
            goto free_and_exit;
        }
        reverse_array(inembed, reversed_inembed, dim_rank);
    }

    if (onembed != NULL)
    {
        ALLOC_ALIGN_UNINIT(reversed_onembed, FFTZ_INT32,
                           dim_rank * sizeof(FFTZ_INT32));
        if (reversed_onembed == NULL)
        {
            goto free_and_exit;
        }
        reverse_array(onembed, reversed_onembed, dim_rank);
    }

    *p = fftw_plan_many_dft(dim_rank, reversed_n, *howmany, in,
                            reversed_inembed, *istride, *idist, out,
                            reversed_onembed, *ostride, *odist, *sign, *flags);

free_and_exit:
    FREE_ALIGN_ALLOCATED_MEM(reversed_n);
    FREE_ALIGN_ALLOCATED_MEM(reversed_inembed);
    FREE_ALIGN_ALLOCATED_MEM(reversed_onembed);
}

FFTZ_VOID dfftwf_plan_many_dft_(fftwf_plan *p, FFTZ_INT32 *rank,
                                const FFTZ_INT32 *n, FFTZ_INT32 *howmany,
                                fftwf_complex *in, const FFTZ_INT32 *inembed,
                                FFTZ_INT32 *istride, FFTZ_INT32 *idist,
                                fftwf_complex *out, const FFTZ_INT32 *onembed,
                                FFTZ_INT32 *ostride, FFTZ_INT32 *odist,
                                FFTZ_INT32 *sign, FFTZ_INT32 *flags)
{
    // since the dim.n is passed in reverse order as compared to that of C APIs,
    // reverse it for Fortran API.

    FFTZ_INT32 dim_rank = *rank;
    *p = NULL;

    FFTZ_INT32 *reversed_n = NULL;
    FFTZ_INT32 *reversed_inembed = NULL;
    FFTZ_INT32 *reversed_onembed = NULL;

    ALLOC_ALIGN_UNINIT(reversed_n, FFTZ_INT32, dim_rank * sizeof(FFTZ_INT32));

    if (reversed_n == NULL)
    {
        return;
    }

    reverse_array(n, reversed_n, dim_rank);

    if (inembed != NULL)
    {
        ALLOC_ALIGN_UNINIT(reversed_inembed, FFTZ_INT32,
                           dim_rank * sizeof(FFTZ_INT32));
        if (reversed_inembed == NULL)
        {
            goto free_and_exit;
        }
        reverse_array(inembed, reversed_inembed, dim_rank);
    }

    if (onembed != NULL)
    {
        ALLOC_ALIGN_UNINIT(reversed_onembed, FFTZ_INT32,
                           dim_rank * sizeof(FFTZ_INT32));
        if (reversed_onembed == NULL)
        {
            goto free_and_exit;
        }
        reverse_array(onembed, reversed_onembed, dim_rank);
    }

    *p = fftwf_plan_many_dft(dim_rank, reversed_n, *howmany, in,
                             reversed_inembed, *istride, *idist, out,
                             reversed_onembed, *ostride, *odist, *sign, *flags);

free_and_exit:
    FREE_ALIGN_ALLOCATED_MEM(reversed_n);
    FREE_ALIGN_ALLOCATED_MEM(reversed_inembed);
    FREE_ALIGN_ALLOCATED_MEM(reversed_onembed);
}

FFTZ_VOID dfftw_plan_guru_dft_(fftw_plan *p, FFTZ_INT32 *rank,
                               const FFTZ_INT32 *n, const FFTZ_INT32 *is,
                               const FFTZ_INT32 *os, FFTZ_INT32 *howmany_rank,
                               const FFTZ_INT32 *h_n, const FFTZ_INT32 *h_is,
                               const FFTZ_INT32 *h_os, fftw_complex *in,
                               fftw_complex *out, FFTZ_INT32 *sign,
                               FFTZ_INT32 *flags)
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

FFTZ_VOID dfftwf_plan_guru_dft_(fftwf_plan *p, FFTZ_INT32 *rank,
                                const FFTZ_INT32 *n, const FFTZ_INT32 *is,
                                const FFTZ_INT32 *os, FFTZ_INT32 *howmany_rank,
                                const FFTZ_INT32 *h_n, const FFTZ_INT32 *h_is,
                                const FFTZ_INT32 *h_os, fftwf_complex *in,
                                fftwf_complex *out, FFTZ_INT32 *sign,
                                FFTZ_INT32 *flags)
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

FFTZ_VOID dfftw_plan_dft_r2c_(fftw_plan *p, FFTZ_INT32 *rank,
                              const FFTZ_INT32 *n, FFTZ_DOUBLE *in,
                              fftw_complex *out, FFTZ_INT32 *flags)
{
    FFTZ_INT32 *reversed_n = NULL;
    FFTZ_INT32 dim_rank = *rank;
    ALLOC_ALIGN_UNINIT(reversed_n, FFTZ_INT32, dim_rank * sizeof(FFTZ_INT32));
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

FFTZ_VOID dfftwf_plan_dft_r2c_(fftwf_plan *p, FFTZ_INT32 *rank,
                               const FFTZ_INT32 *n, FFTZ_FLOAT *in,
                               fftwf_complex *out, FFTZ_INT32 *flags)
{
    FFTZ_INT32 *reversed_n = NULL;
    FFTZ_INT32 dim_rank = *rank;
    ALLOC_ALIGN_UNINIT(reversed_n, FFTZ_INT32, dim_rank * sizeof(FFTZ_INT32));
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

FFTZ_VOID dfftw_plan_dft_r2c_1d_(fftw_plan *p, FFTZ_INT32 *n, FFTZ_DOUBLE *in,
                            fftw_complex *out, FFTZ_INT32 *flags)
{
    *p = fftw_plan_dft_r2c_1d(*n, in, out, *flags);
}

FFTZ_VOID dfftwf_plan_dft_r2c_1d_(fftwf_plan *p, FFTZ_INT32 *n, FFTZ_FLOAT *in,
                             fftwf_complex *out, FFTZ_INT32 *flags)
{
    *p = fftwf_plan_dft_r2c_1d(*n, in, out, *flags);
}

FFTZ_VOID dfftw_plan_dft_r2c_2d_(fftw_plan *p, FFTZ_INT32 *nx, FFTZ_INT32 *ny,
                                 FFTZ_DOUBLE *in, fftw_complex *out,
                                 FFTZ_INT32 *flags)
{
    *p = fftw_plan_dft_r2c_2d(*ny, *nx, in, out, *flags);
}

FFTZ_VOID dfftwf_plan_dft_r2c_2d_(fftwf_plan *p, FFTZ_INT32 *nx, FFTZ_INT32 *ny,
                                  FFTZ_FLOAT *in, fftwf_complex *out,
                                  FFTZ_INT32 *flags)
{
    *p = fftwf_plan_dft_r2c_2d(*ny, *nx, in, out, *flags);
}

FFTZ_VOID dfftw_plan_dft_r2c_3d_(fftw_plan *p, FFTZ_INT32 *nx, FFTZ_INT32 *ny,
                                 FFTZ_INT32 *nz, FFTZ_DOUBLE *in,
                                 fftw_complex *out, FFTZ_INT32 *flags)
{
    *p = fftw_plan_dft_r2c_3d(*nz, *ny, *nx, in, out, *flags);
}

FFTZ_VOID dfftwf_plan_dft_r2c_3d_(fftwf_plan *p, FFTZ_INT32 *nx, FFTZ_INT32 *ny,
                                  FFTZ_INT32 *nz, FFTZ_FLOAT *in,
                                  fftwf_complex *out, FFTZ_INT32 *flags)
{
    *p = fftwf_plan_dft_r2c_3d(*nz, *ny, *nx, in, out, *flags);
}

FFTZ_VOID dfftw_plan_many_dft_r2c_(fftw_plan *p, FFTZ_INT32 *rank,
                                   const FFTZ_INT32 *n, FFTZ_INT32 *howmany,
                                   FFTZ_DOUBLE *in, const FFTZ_INT32 *inembed,
                                   FFTZ_INT32 *istride, FFTZ_INT32 *idist,
                                   fftw_complex *out, const FFTZ_INT32 *onembed,
                                   FFTZ_INT32 *ostride, FFTZ_INT32 *odist,
                                   FFTZ_INT32 *flags)
{
    // since the dim.n is passed in reverse order as compared to that of C APIs,
    // reverse it for Fortran API.

    FFTZ_INT32 dim_rank = *rank;
    *p = NULL;

    FFTZ_INT32 *reversed_n = NULL;
    FFTZ_INT32 *reversed_inembed = NULL;
    FFTZ_INT32 *reversed_onembed = NULL;

    ALLOC_ALIGN_UNINIT(reversed_n, FFTZ_INT32, dim_rank * sizeof(FFTZ_INT32));

    if (reversed_n == NULL)
    {
        return;
    }

    reverse_array(n, reversed_n, dim_rank);

    if (inembed != NULL)
    {
        ALLOC_ALIGN_UNINIT(reversed_inembed, FFTZ_INT32,
                           dim_rank * sizeof(FFTZ_INT32));
        if (reversed_inembed == NULL)
        {
            goto free_and_exit;
        }
        reverse_array(inembed, reversed_inembed, dim_rank);
    }

    if (onembed != NULL)
    {
        ALLOC_ALIGN_UNINIT(reversed_onembed, FFTZ_INT32,
                           dim_rank * sizeof(FFTZ_INT32));
        if (reversed_onembed == NULL)
        {
            goto free_and_exit;
        }
        reverse_array(onembed, reversed_onembed, dim_rank);
    }

    *p = fftw_plan_many_dft_r2c(dim_rank, reversed_n, *howmany, in,
                                reversed_inembed, *istride, *idist, out,
                                reversed_onembed, *ostride, *odist, *flags);

free_and_exit:
    FREE_ALIGN_ALLOCATED_MEM(reversed_n);
    FREE_ALIGN_ALLOCATED_MEM(reversed_inembed);
    FREE_ALIGN_ALLOCATED_MEM(reversed_onembed);
}

FFTZ_VOID dfftwf_plan_many_dft_r2c_(fftwf_plan *p, FFTZ_INT32 *rank,
                                    const FFTZ_INT32 *n, FFTZ_INT32 *howmany,
                                    FFTZ_FLOAT *in, const FFTZ_INT32 *inembed,
                                    FFTZ_INT32 *istride, FFTZ_INT32 *idist,
                                    fftwf_complex *out,
                                    const FFTZ_INT32 *onembed,
                                    FFTZ_INT32 *ostride, FFTZ_INT32 *odist,
                                    FFTZ_INT32 *flags)
{
    // since the dim.n is passed in reverse order as compared to that of C APIs,
    // reverse it for Fortran API.

    FFTZ_INT32 dim_rank = *rank;
    *p = NULL;

    FFTZ_INT32 *reversed_n = NULL;
    FFTZ_INT32 *reversed_inembed = NULL;
    FFTZ_INT32 *reversed_onembed = NULL;

    ALLOC_ALIGN_UNINIT(reversed_n, FFTZ_INT32, dim_rank * sizeof(FFTZ_INT32));

    if (reversed_n == NULL)
    {
        return;
    }

    reverse_array(n, reversed_n, dim_rank);

    if (inembed != NULL)
    {
        ALLOC_ALIGN_UNINIT(reversed_inembed, FFTZ_INT32,
                           dim_rank * sizeof(FFTZ_INT32));
        if (reversed_inembed == NULL)
        {
            goto free_and_exit;
        }
        reverse_array(inembed, reversed_inembed, dim_rank);
    }

    if (onembed != NULL)
    {
        ALLOC_ALIGN_UNINIT(reversed_onembed, FFTZ_INT32,
                           dim_rank * sizeof(FFTZ_INT32));
        if (reversed_onembed == NULL)
        {
            goto free_and_exit;
        }
        reverse_array(onembed, reversed_onembed, dim_rank);
    }

    *p = fftwf_plan_many_dft_r2c(dim_rank, reversed_n, *howmany, in,
                                 reversed_inembed, *istride, *idist, out,
                                 reversed_onembed, *ostride, *odist, *flags);

free_and_exit:
    FREE_ALIGN_ALLOCATED_MEM(reversed_n);
    FREE_ALIGN_ALLOCATED_MEM(reversed_inembed);
    FREE_ALIGN_ALLOCATED_MEM(reversed_onembed);
}

FFTZ_VOID dfftw_plan_guru_dft_r2c_(fftw_plan *p, FFTZ_INT32 *rank,
                                   const FFTZ_INT32 *n, const FFTZ_INT32 *is,
                                   const FFTZ_INT32 *os,
                                   FFTZ_INT32 *howmany_rank,
                                   const FFTZ_INT32 *h_n,
                                   const FFTZ_INT32 *h_is,
                                   const FFTZ_INT32 *h_os, FFTZ_DOUBLE *in,
                                   fftw_complex *out, FFTZ_INT32 *flags)
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

FFTZ_VOID dfftwf_plan_guru_dft_r2c_(fftwf_plan *p, FFTZ_INT32 *rank,
                                    const FFTZ_INT32 *n, const FFTZ_INT32 *is,
                                    const FFTZ_INT32 *os,
                                    FFTZ_INT32 *howmany_rank,
                                    const FFTZ_INT32 *h_n,
                                    const FFTZ_INT32 *h_is,
                                    const FFTZ_INT32 *h_os, FFTZ_FLOAT *in,
                                    fftwf_complex *out, FFTZ_INT32 *flags)
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

FFTZ_VOID dfftw_plan_dft_c2r_(fftw_plan *p, FFTZ_INT32 *rank,
                              const FFTZ_INT32 *n, fftw_complex *in,
                              FFTZ_DOUBLE *out, FFTZ_INT32 *flags)
{
    FFTZ_INT32 *reversed_n = NULL;
    FFTZ_INT32 dim_rank = *rank;
    ALLOC_ALIGN_UNINIT(reversed_n, FFTZ_INT32, dim_rank * sizeof(FFTZ_INT32));
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

FFTZ_VOID dfftwf_plan_dft_c2r_(fftwf_plan *p, FFTZ_INT32 *rank,
                               const FFTZ_INT32 *n, fftwf_complex *in,
                               FFTZ_FLOAT *out, FFTZ_INT32 *flags)
{
    FFTZ_INT32 *reversed_n = NULL;
    FFTZ_INT32 dim_rank = *rank;
    ALLOC_ALIGN_UNINIT(reversed_n, FFTZ_INT32, dim_rank * sizeof(FFTZ_INT32));
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

FFTZ_VOID dfftw_plan_dft_c2r_1d_(fftw_plan *p, FFTZ_INT32 *n, fftw_complex *in,
                            FFTZ_DOUBLE *out, FFTZ_INT32 *flags)
{
    *p = fftw_plan_dft_c2r_1d(*n, in, out, *flags);
}

FFTZ_VOID dfftwf_plan_dft_c2r_1d_(fftwf_plan *p, FFTZ_INT32 *n,
                                  fftwf_complex *in, FFTZ_FLOAT *out,
                                  FFTZ_INT32 *flags)
{
    *p = fftwf_plan_dft_c2r_1d(*n, in, out, *flags);
}

FFTZ_VOID dfftw_plan_dft_c2r_2d_(fftw_plan *p, FFTZ_INT32 *nx, FFTZ_INT32 *ny,
                                 fftw_complex *in, FFTZ_DOUBLE *out,
                                 FFTZ_INT32 *flags)
{
    *p = fftw_plan_dft_c2r_2d(*ny, *nx, in, out, *flags);
}

FFTZ_VOID dfftwf_plan_dft_c2r_2d_(fftwf_plan *p, FFTZ_INT32 *nx, FFTZ_INT32 *ny,
                                  fftwf_complex *in, FFTZ_FLOAT *out,
                                  FFTZ_INT32 *flags)
{
    *p = fftwf_plan_dft_c2r_2d(*ny, *nx, in, out, *flags);
}

FFTZ_VOID dfftw_plan_dft_c2r_3d_(fftw_plan *p, FFTZ_INT32 *nx, FFTZ_INT32 *ny,
                                 FFTZ_INT32 *nz, fftw_complex *in,
                                 FFTZ_DOUBLE *out, FFTZ_INT32 *flags)
{
    *p = fftw_plan_dft_c2r_3d(*nz, *ny, *nx, in, out, *flags);
}

FFTZ_VOID dfftwf_plan_dft_c2r_3d_(fftwf_plan *p, FFTZ_INT32 *nx, FFTZ_INT32 *ny,
                                  FFTZ_INT32 *nz, fftwf_complex *in,
                                  FFTZ_FLOAT *out, FFTZ_INT32 *flags)
{
    *p = fftwf_plan_dft_c2r_3d(*nz, *ny, *nx, in, out, *flags);
}

FFTZ_VOID dfftw_plan_many_dft_c2r_(fftw_plan *p, FFTZ_INT32 *rank,
                                   const FFTZ_INT32 *n, FFTZ_INT32 *howmany,
                                   fftw_complex *in, const FFTZ_INT32 *inembed,
                                   FFTZ_INT32 *istride, FFTZ_INT32 *idist,
                                   FFTZ_DOUBLE *out, const FFTZ_INT32 *onembed,
                                   FFTZ_INT32 *ostride, FFTZ_INT32 *odist,
                                   FFTZ_INT32 *flags)
{
    // since the dim.n is passed in reverse order as compared to that of C APIs,
    // reverse it for Fortran API.

    FFTZ_INT32 dim_rank = *rank;
    *p = NULL;

    FFTZ_INT32 *reversed_n = NULL;
    FFTZ_INT32 *reversed_inembed = NULL;
    FFTZ_INT32 *reversed_onembed = NULL;

    ALLOC_ALIGN_UNINIT(reversed_n, FFTZ_INT32, dim_rank * sizeof(FFTZ_INT32));

    if (reversed_n == NULL)
    {
        return;
    }

    reverse_array(n, reversed_n, dim_rank);

    if (inembed != NULL)
    {
        ALLOC_ALIGN_UNINIT(reversed_inembed, FFTZ_INT32,
                           dim_rank * sizeof(FFTZ_INT32));
        if (reversed_inembed == NULL)
        {
            goto free_and_exit;
        }
        reverse_array(inembed, reversed_inembed, dim_rank);
    }

    if (onembed != NULL)
    {
        ALLOC_ALIGN_UNINIT(reversed_onembed, FFTZ_INT32,
                           dim_rank * sizeof(FFTZ_INT32));
        if (reversed_onembed == NULL)
        {
            goto free_and_exit;
        }
        reverse_array(onembed, reversed_onembed, dim_rank);
    }

    *p = fftw_plan_many_dft_c2r(dim_rank, reversed_n, *howmany, in,
                                reversed_inembed, *istride, *idist, out,
                                reversed_onembed, *ostride, *odist, *flags);

free_and_exit:
    FREE_ALIGN_ALLOCATED_MEM(reversed_n);
    FREE_ALIGN_ALLOCATED_MEM(reversed_inembed);
    FREE_ALIGN_ALLOCATED_MEM(reversed_onembed);
}

FFTZ_VOID dfftwf_plan_many_dft_c2r_(fftwf_plan *p, FFTZ_INT32 *rank,
                                    const FFTZ_INT32 *n, FFTZ_INT32 *howmany,
                                    fftwf_complex *in,
                                    const FFTZ_INT32 *inembed,
                                    FFTZ_INT32 *istride, FFTZ_INT32 *idist,
                                    FFTZ_FLOAT *out, const FFTZ_INT32 *onembed,
                                    FFTZ_INT32 *ostride, FFTZ_INT32 *odist,
                                    FFTZ_INT32 *flags)
{
    // since the dim.n is passed in reverse order as compared to that of C APIs,
    // reverse it for Fortran API.

    FFTZ_INT32 dim_rank = *rank;
    *p = NULL;

    FFTZ_INT32 *reversed_n = NULL;
    FFTZ_INT32 *reversed_inembed = NULL;
    FFTZ_INT32 *reversed_onembed = NULL;

    ALLOC_ALIGN_UNINIT(reversed_n, FFTZ_INT32, dim_rank * sizeof(FFTZ_INT32));

    if (reversed_n == NULL)
    {
        return;
    }

    reverse_array(n, reversed_n, dim_rank);

    if (inembed != NULL)
    {
        ALLOC_ALIGN_UNINIT(reversed_inembed, FFTZ_INT32,
                           dim_rank * sizeof(FFTZ_INT32));
        if (reversed_inembed == NULL)
        {
            goto free_and_exit;
        }
        reverse_array(inembed, reversed_inembed, dim_rank);
    }

    if (onembed != NULL)
    {
        ALLOC_ALIGN_UNINIT(reversed_onembed, FFTZ_INT32,
                           dim_rank * sizeof(FFTZ_INT32));
        if (reversed_onembed == NULL)
        {
            goto free_and_exit;
        }
        reverse_array(onembed, reversed_onembed, dim_rank);
    }

    *p = fftwf_plan_many_dft_c2r(dim_rank, reversed_n, *howmany, in,
                                 reversed_inembed, *istride, *idist, out,
                                 reversed_onembed, *ostride, *odist, *flags);

free_and_exit:
    FREE_ALIGN_ALLOCATED_MEM(reversed_n);
    FREE_ALIGN_ALLOCATED_MEM(reversed_inembed);
    FREE_ALIGN_ALLOCATED_MEM(reversed_onembed);
}

FFTZ_VOID dfftw_plan_guru_dft_c2r_(fftw_plan *p, FFTZ_INT32 *rank,
                                   const FFTZ_INT32 *n, const FFTZ_INT32 *is,
                                   const FFTZ_INT32 *os,
                                   FFTZ_INT32 *howmany_rank,
                                   const FFTZ_INT32 *h_n,
                                   const FFTZ_INT32 *h_is,
                                   const FFTZ_INT32 *h_os, fftw_complex *in,
                                   FFTZ_DOUBLE *out, FFTZ_INT32 *flags)
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

FFTZ_VOID dfftwf_plan_guru_dft_c2r_(fftwf_plan *p, FFTZ_INT32 *rank,
                                    const FFTZ_INT32 *n, const FFTZ_INT32 *is,
                                    const FFTZ_INT32 *os,
                                    FFTZ_INT32 *howmany_rank,
                                    const FFTZ_INT32 *h_n,
                                    const FFTZ_INT32 *h_is,
                                    const FFTZ_INT32 *h_os, fftwf_complex *in,
                                    FFTZ_FLOAT *out, FFTZ_INT32 *flags)
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
