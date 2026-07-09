// Copyright Advanced Micro Devices, Inc.
// SPDX-License-Identifier: BSD-3-Clause

/** @file fftw_plan_wrapper.c
 *
 *  @brief Contains wrapper implementations of FFTW planner APIs
 *
 *  This file contains implementations for the plan APIs provided by FFTW.
 */

#include "src/translator/fftz_translator.h"

fftw_plan fftw_plan_dft(FFTZ_INT32 rank, const FFTZ_INT32 *n, fftw_complex *in,
                        fftw_complex *out, FFTZ_INT32 sign, unsigned flags)
{
    dv_desc *p_dv_desc = get_dv_desc(rank, n);
    return get_handle_d(p_dv_desc, sign, in, out, COMPLEX);
}

fftwf_plan fftwf_plan_dft(FFTZ_INT32 rank, const FFTZ_INT32 *n,
                          fftwf_complex *in, fftwf_complex *out,
                          FFTZ_INT32 sign, unsigned flags)
{
    dv_desc *p_dv_desc = get_dv_desc(rank, n);
    return get_handle_f(p_dv_desc, sign, in, out, COMPLEX);
}

// fftw_plan_dft_1d is used for 1D problems, so setting vec_ranks and vecs
// values as 1 and setting dim rank as 1.
fftw_plan fftw_plan_dft_1d(FFTZ_INT32 n, fftw_complex *in, fftw_complex *out,
                           FFTZ_INT32 sign, unsigned flags)
{
    dv_desc *p_dv_desc = get_dv_desc(1, &n);
    return get_handle_d(p_dv_desc, sign, in, out, COMPLEX);
}

fftwf_plan fftwf_plan_dft_1d(FFTZ_INT32 n, fftwf_complex *in,
                             fftwf_complex *out, FFTZ_INT32 sign,
                             unsigned flags)
{
    dv_desc *p_dv_desc = get_dv_desc(1, &n);
    return get_handle_f(p_dv_desc, sign, in, out, COMPLEX);
}

// fftw_plan_dft_2d is used for 2D problems, so setting vec_ranks and vecs
// values as 1 and setting dim rank as 2.
fftw_plan fftw_plan_dft_2d(FFTZ_INT32 n0, FFTZ_INT32 n1, fftw_complex *in,
                           fftw_complex *out, FFTZ_INT32 sign, unsigned flags)
{
    const FFTZ_INT32 *dims_n = (FFTZ_INT32[]){n0, n1};
    dv_desc *p_dv_desc = get_dv_desc(2, dims_n);
    return get_handle_d(p_dv_desc, sign, in, out, COMPLEX);
}

fftwf_plan fftwf_plan_dft_2d(FFTZ_INT32 n0, FFTZ_INT32 n1, fftwf_complex *in,
                             fftwf_complex *out, FFTZ_INT32 sign,
                             unsigned flags)
{
    const FFTZ_INT32 *dims_n = (FFTZ_INT32[]){n0, n1};
    dv_desc *p_dv_desc = get_dv_desc(2, dims_n);
    return get_handle_f(p_dv_desc, sign, in, out, COMPLEX);
}

// fftw_plan_dft_3d is used for 3D problems, so setting vec_ranks and vecs
// values as 1 and setting dim rank as 3.
fftw_plan fftw_plan_dft_3d(FFTZ_INT32 n0, FFTZ_INT32 n1, FFTZ_INT32 n2,
                           fftw_complex *in, fftw_complex *out, FFTZ_INT32 sign,
                           unsigned flags)
{
    const FFTZ_INT32 *dims_n = (FFTZ_INT32[]){n0, n1, n2};
    dv_desc *p_dv_desc = get_dv_desc(3, dims_n);
    return get_handle_d(p_dv_desc, sign, in, out, COMPLEX);
}

fftwf_plan fftwf_plan_dft_3d(FFTZ_INT32 n0, FFTZ_INT32 n1, FFTZ_INT32 n2,
                             fftwf_complex *in, fftwf_complex *out,
                             FFTZ_INT32 sign, unsigned flags)
{
    const FFTZ_INT32 *dims_n = (FFTZ_INT32[]){n0, n1, n2};
    dv_desc *p_dv_desc = get_dv_desc(3, dims_n);
    return get_handle_f(p_dv_desc, sign, in, out, COMPLEX);
}

// fftw_plan_many_dft is used for ND problems, so setting vec_ranks and vecs
// values as 1.
fftw_plan fftw_plan_many_dft(FFTZ_INT32 rank, const FFTZ_INT32 *n,
                             FFTZ_INT32 howmany, fftw_complex *in,
                             const FFTZ_INT32 *inembed, FFTZ_INT32 istride,
                             FFTZ_INT32 idist, fftw_complex *out,
                             const FFTZ_INT32 *onembed, FFTZ_INT32 ostride,
                             FFTZ_INT32 odist, FFTZ_INT32 sign, unsigned flags)
{
    dv_desc *p_dv_desc = get_many_dv_desc(rank, n, howmany, inembed,
                                               istride, idist, onembed, ostride,
                                               odist);
    return get_handle_d(p_dv_desc, sign, in, out, COMPLEX);
}

fftwf_plan fftwf_plan_many_dft(FFTZ_INT32 rank, const FFTZ_INT32 *n,
                               FFTZ_INT32 howmany, fftwf_complex *in,
                               const FFTZ_INT32 *inembed, FFTZ_INT32 istride,
                               FFTZ_INT32 idist, fftwf_complex *out,
                               const FFTZ_INT32 *onembed, FFTZ_INT32 ostride,
                               FFTZ_INT32 odist, FFTZ_INT32 sign,
                               unsigned flags)
{
    dv_desc *p_dv_desc = get_many_dv_desc(rank, n, howmany, inembed,
                                               istride, idist, onembed, ostride,
                                               odist);
    return get_handle_f(p_dv_desc, sign, in, out, COMPLEX);
}

fftw_plan fftw_plan_guru_dft(FFTZ_INT32 rank, const fftw_iodim *dims,
                             FFTZ_INT32 howmany_rank,
                             const fftw_iodim *howmany_dims, fftw_complex *in,
                             fftw_complex *out, FFTZ_INT32 sign, unsigned flags)
{
    dv_desc *p_dv_desc = get_guru_dv_desc(rank, dims, howmany_rank,
                                         howmany_dims);
    return get_handle_d(p_dv_desc, sign, in, out, COMPLEX);
}

fftwf_plan fftwf_plan_guru_dft(FFTZ_INT32 rank, const fftwf_iodim *dims_f,
                               FFTZ_INT32 howmany_rank,
                               const fftwf_iodim *howmany_dims_f,
                               fftwf_complex *in, fftwf_complex *out,
                               FFTZ_INT32 sign, unsigned flags)
{
    const fftw_iodim *dims = (const fftw_iodim *)dims_f;
    const fftw_iodim *howmany_dims = (const fftw_iodim *)howmany_dims_f;
    dv_desc *p_dv_desc = get_guru_dv_desc(rank, dims, howmany_rank,
                                         howmany_dims);
    return get_handle_f(p_dv_desc, sign, in, out, COMPLEX);
}

fftw_plan fftw_plan_guru64_dft(FFTZ_INT32 rank, const fftw_iodim64 *dims,
                               FFTZ_INT32 howmany_rank,
                               const fftw_iodim64 *howmany_dims,
                               fftw_complex *in, fftw_complex *out,
                               FFTZ_INT32 sign, unsigned flags)
{
    dv_desc_64_ *p_dv_desc = get_guru_64_dv_desc(rank, dims,
                                                howmany_rank, howmany_dims);
    return get_handle_d_64_(p_dv_desc, sign, in, out, COMPLEX);
}

fftwf_plan fftwf_plan_guru64_dft(FFTZ_INT32 rank, const fftwf_iodim64 *dims_f,
                                 FFTZ_INT32 howmany_rank,
                                 const fftwf_iodim64 *howmany_dims_f,
                                 fftwf_complex *in, fftwf_complex *out,
                                 FFTZ_INT32 sign, unsigned flags)
{
    const fftw_iodim64 *dims = (const fftw_iodim64 *)dims_f;
    const fftw_iodim64 *howmany_dims = (const fftw_iodim64 *)howmany_dims_f;
    dv_desc_64_ *p_dv_desc = get_guru_64_dv_desc(rank, dims,
                                                howmany_rank, howmany_dims);
    return get_handle_f_64_(p_dv_desc, sign, in, out, COMPLEX);
}

fftw_plan fftw_plan_dft_r2c(FFTZ_INT32 rank, const FFTZ_INT32 *n,
                            FFTZ_DOUBLE *in, fftw_complex *out, unsigned flags)
{
    dv_desc *p_dv_desc = get_r2c_dv_desc(rank, n,
                                         (FFTZ_VOID *)in == (FFTZ_VOID *)out);
    return get_handle_d(p_dv_desc, FFTW_FORWARD, in, out, REAL);
}

fftwf_plan fftwf_plan_dft_r2c(FFTZ_INT32 rank, const FFTZ_INT32 *n,
                              FFTZ_FLOAT *in, fftwf_complex *out,
                              unsigned flags)
{
    dv_desc *p_dv_desc = get_r2c_dv_desc(rank, n,
                                         (FFTZ_VOID *)in == (FFTZ_VOID *)out);
    return get_handle_f(p_dv_desc, FFTW_FORWARD, in, out, REAL);
}

fftw_plan fftw_plan_dft_r2c_1d(FFTZ_INT32 n, FFTZ_DOUBLE *in, fftw_complex *out,
                               unsigned flags)
{
    dv_desc *p_dv_desc = get_r2c_dv_desc(1, &n,
                                         (FFTZ_VOID *)in == (FFTZ_VOID *)out);
    return get_handle_d(p_dv_desc, FFTW_FORWARD, in, out, REAL);
}

fftwf_plan fftwf_plan_dft_r2c_1d(FFTZ_INT32 n, FFTZ_FLOAT *in,
                                 fftwf_complex *out, unsigned flags)
{
    dv_desc *p_dv_desc = get_r2c_dv_desc(1, &n,
                                         (FFTZ_VOID *)in == (FFTZ_VOID *)out);
    return get_handle_f(p_dv_desc, FFTW_FORWARD, in, out, REAL);
}

fftw_plan fftw_plan_dft_r2c_2d(FFTZ_INT32 n0, FFTZ_INT32 n1, FFTZ_DOUBLE *in,
                               fftw_complex *out, unsigned flags)
{
    const FFTZ_INT32 *dims_n = (FFTZ_INT32[]){n0, n1};
    dv_desc *p_dv_desc = get_r2c_dv_desc(2, dims_n,
                                           (FFTZ_VOID *)in == (FFTZ_VOID *)out);
    return get_handle_d(p_dv_desc, FFTW_FORWARD, in, out, REAL);
}

fftwf_plan fftwf_plan_dft_r2c_2d(FFTZ_INT32 n0, FFTZ_INT32 n1, FFTZ_FLOAT *in,
                                 fftwf_complex *out, unsigned flags)
{
    const FFTZ_INT32 *dims_n = (FFTZ_INT32[]){n0, n1};
    dv_desc *p_dv_desc = get_r2c_dv_desc(2, dims_n,
                                           (FFTZ_VOID *)in == (FFTZ_VOID *)out);
    return get_handle_f(p_dv_desc, FFTW_FORWARD, in, out, REAL);
}

fftw_plan fftw_plan_dft_r2c_3d(FFTZ_INT32 n0, FFTZ_INT32 n1, FFTZ_INT32 n2,
                               FFTZ_DOUBLE *in, fftw_complex *out,
                               unsigned flags)
{
    const FFTZ_INT32 *dims_n = (FFTZ_INT32[]){n0, n1, n2};
    dv_desc *p_dv_desc = get_r2c_dv_desc(3, dims_n,
                                           (FFTZ_VOID *)in == (FFTZ_VOID *)out);
    return get_handle_d(p_dv_desc, FFTW_FORWARD, in, out, REAL);
}

fftwf_plan fftwf_plan_dft_r2c_3d(FFTZ_INT32 n0, FFTZ_INT32 n1, FFTZ_INT32 n2,
                                 FFTZ_FLOAT *in, fftwf_complex *out,
                                 unsigned flags)
{
    const FFTZ_INT32 *dims_n = (FFTZ_INT32[]){n0, n1, n2};
    dv_desc *p_dv_desc = get_r2c_dv_desc(3, dims_n,
                                           (FFTZ_VOID *)in == (FFTZ_VOID *)out);
    return get_handle_f(p_dv_desc, FFTW_FORWARD, in, out, REAL);
}

fftw_plan fftw_plan_many_dft_r2c(FFTZ_INT32 rank, const FFTZ_INT32 *n,
                                 FFTZ_INT32 howmany, FFTZ_DOUBLE *in,
                                 const FFTZ_INT32 *inembed, FFTZ_INT32 istride,
                                 FFTZ_INT32 idist, fftw_complex *out,
                                 const FFTZ_INT32 *onembed, FFTZ_INT32 ostride,
                                 FFTZ_INT32 odist, unsigned flags)
{
    dv_desc *p_dv_desc = get_many_r2c_dv_desc(rank, n, howmany, inembed,
        istride, idist, onembed, ostride, odist,
        (FFTZ_VOID *)in == (FFTZ_VOID *)out);
    return get_handle_d(p_dv_desc, FFTW_FORWARD, in, out, REAL);
}

fftwf_plan fftwf_plan_many_dft_r2c(FFTZ_INT32 rank, const FFTZ_INT32 *n,
                                   FFTZ_INT32 howmany, FFTZ_FLOAT *in,
                                   const FFTZ_INT32 *inembed,
                                   FFTZ_INT32 istride, FFTZ_INT32 idist,
                                   fftwf_complex *out,
                                   const FFTZ_INT32 *onembed,
                                   FFTZ_INT32 ostride, FFTZ_INT32 odist,
                                   unsigned flags)
{
    dv_desc *p_dv_desc = get_many_r2c_dv_desc(rank, n, howmany, inembed,
        istride, idist, onembed, ostride, odist,
        (FFTZ_VOID *)in == (FFTZ_VOID *)out);
    return get_handle_f(p_dv_desc, FFTW_FORWARD, in, out, REAL);
}

fftw_plan fftw_plan_guru_dft_r2c(FFTZ_INT32 rank, const fftw_iodim *dims,
                                 FFTZ_INT32 howmany_rank,
                                 const fftw_iodim *howmany_dims,
                                 FFTZ_DOUBLE *in, fftw_complex *out,
                                 unsigned flags)
{
    dv_desc *p_dv_desc = get_guru_dv_desc(rank, dims, howmany_rank,
                                         howmany_dims);
    return get_handle_d(p_dv_desc, FFTW_FORWARD, in, out, REAL);
}

fftwf_plan fftwf_plan_guru_dft_r2c(FFTZ_INT32 rank, const fftwf_iodim *dims_f,
                                   FFTZ_INT32 howmany_rank,
                                   const fftwf_iodim *howmany_dims_f,
                                   FFTZ_FLOAT *in, fftwf_complex *out,
                                   unsigned flags)
{
    const fftw_iodim *dims = (const fftw_iodim *)dims_f;
    const fftw_iodim *howmany_dims = (const fftw_iodim *)howmany_dims_f;
    dv_desc *p_dv_desc = get_guru_dv_desc(rank, dims, howmany_rank,
                                         howmany_dims);
    return get_handle_f(p_dv_desc, FFTW_FORWARD, in, out, REAL);
}

fftw_plan fftw_plan_guru64_dft_r2c(FFTZ_INT32 rank, const fftw_iodim64 *dims,
                                   FFTZ_INT32 howmany_rank,
                                   const fftw_iodim64 *howmany_dims,
                                   FFTZ_DOUBLE *in, fftw_complex *out,
                                   unsigned flags)
{
    dv_desc_64_ *p_dv_desc = get_guru_64_dv_desc(rank, dims, howmany_rank,
                                                howmany_dims);
    return get_handle_d_64_(p_dv_desc, FFTW_FORWARD, in, out, REAL);
}

fftwf_plan fftwf_plan_guru64_dft_r2c(FFTZ_INT32 rank,
                                     const fftwf_iodim64 *dims_f,
                                     FFTZ_INT32 howmany_rank,
                                     const fftwf_iodim64 *howmany_dims_f,
                                     FFTZ_FLOAT *in, fftwf_complex *out,
                                     unsigned flags)
{
    const fftw_iodim64 *dims = (const fftw_iodim64 *)dims_f;
    const fftw_iodim64 *howmany_dims = (const fftw_iodim64 *)howmany_dims_f;
    dv_desc_64_ *p_dv_desc = get_guru_64_dv_desc(rank, dims, howmany_rank,
                                                howmany_dims);
    return get_handle_f_64_(p_dv_desc, FFTW_FORWARD, in, out, REAL);
}


fftw_plan fftw_plan_dft_c2r(FFTZ_INT32 rank, const FFTZ_INT32 *n,
                            fftw_complex *in, FFTZ_DOUBLE *out, unsigned flags)
{
    dv_desc *p_dv_desc = get_c2r_dv_desc(rank, n,
                                         (FFTZ_VOID *)in == (FFTZ_VOID *)out);
    return get_handle_d(p_dv_desc, FFTW_BACKWARD, in, out, REAL);
}

fftwf_plan fftwf_plan_dft_c2r(FFTZ_INT32 rank, const FFTZ_INT32 *n,
                              fftwf_complex *in, FFTZ_FLOAT *out,
                              unsigned flags)
{
    dv_desc *p_dv_desc = get_c2r_dv_desc(rank, n,
                                         (FFTZ_VOID *)in == (FFTZ_VOID *)out);
    return get_handle_f(p_dv_desc, FFTW_BACKWARD, in, out, REAL);
}

fftw_plan fftw_plan_dft_c2r_1d(FFTZ_INT32 n, fftw_complex *in, FFTZ_DOUBLE *out,
                               unsigned flags)
{
    dv_desc *p_dv_desc = get_c2r_dv_desc(1, &n,
                                         (FFTZ_VOID *)in == (FFTZ_VOID *)out);
    return get_handle_d(p_dv_desc, FFTW_BACKWARD, in, out, REAL);
}

fftwf_plan fftwf_plan_dft_c2r_1d(FFTZ_INT32 n, fftwf_complex *in,
                                 FFTZ_FLOAT *out, unsigned flags)
{
    dv_desc *p_dv_desc = get_c2r_dv_desc(1, &n,
                                         (FFTZ_VOID *)in == (FFTZ_VOID *)out);
    return get_handle_f(p_dv_desc, FFTW_BACKWARD, in, out, REAL);
}

fftw_plan fftw_plan_dft_c2r_2d(FFTZ_INT32 n0, FFTZ_INT32 n1, fftw_complex *in,
                               FFTZ_DOUBLE *out, unsigned flags)
{
    const FFTZ_INT32 *dims_n = (FFTZ_INT32[]){n0, n1};
    dv_desc *p_dv_desc = get_c2r_dv_desc(2, dims_n,
                                           (FFTZ_VOID *)in == (FFTZ_VOID *)out);
    return get_handle_d(p_dv_desc, FFTW_BACKWARD, in, out, REAL);
}

fftwf_plan fftwf_plan_dft_c2r_2d(FFTZ_INT32 n0, FFTZ_INT32 n1,
                                 fftwf_complex *in, FFTZ_FLOAT *out,
                                 unsigned flags)
{
    const FFTZ_INT32 *dims_n = (FFTZ_INT32[]){n0, n1};
    dv_desc *p_dv_desc = get_c2r_dv_desc(2, dims_n,
                                           (FFTZ_VOID *)in == (FFTZ_VOID *)out);
    return get_handle_f(p_dv_desc, FFTW_BACKWARD, in, out, REAL);
}

fftw_plan fftw_plan_dft_c2r_3d(FFTZ_INT32 n0, FFTZ_INT32 n1, FFTZ_INT32 n2,
                               fftw_complex *in, FFTZ_DOUBLE *out,
                               unsigned flags)
{
    const FFTZ_INT32 *dims_n = (FFTZ_INT32[]){n0, n1, n2};
    dv_desc *p_dv_desc = get_c2r_dv_desc(3, dims_n,
                                           (FFTZ_VOID *)in == (FFTZ_VOID *)out);
    return get_handle_d(p_dv_desc, FFTW_BACKWARD, in, out, REAL);
}

fftwf_plan fftwf_plan_dft_c2r_3d(FFTZ_INT32 n0, FFTZ_INT32 n1, FFTZ_INT32 n2,
                                 fftwf_complex *in, FFTZ_FLOAT *out,
                                 unsigned flags)
{
    const FFTZ_INT32 *dims_n = (FFTZ_INT32[]){n0, n1, n2};
    dv_desc *p_dv_desc = get_c2r_dv_desc(3, dims_n,
                                           (FFTZ_VOID *)in == (FFTZ_VOID *)out);
    return get_handle_f(p_dv_desc, FFTW_BACKWARD, in, out, REAL);
}

fftw_plan fftw_plan_many_dft_c2r(FFTZ_INT32 rank, const FFTZ_INT32 *n,
                                 FFTZ_INT32 howmany, fftw_complex *in,
                                 const FFTZ_INT32 *inembed, FFTZ_INT32 istride,
                                 FFTZ_INT32 idist, FFTZ_DOUBLE *out,
                                 const FFTZ_INT32 *onembed, FFTZ_INT32 ostride,
                                 FFTZ_INT32 odist, unsigned flags)
{
    dv_desc *p_dv_desc = get_many_c2r_dv_desc(rank, n, howmany, inembed,
        istride, idist, onembed, ostride, odist,
        (FFTZ_VOID *)in == (FFTZ_VOID *)out);
    return get_handle_d(p_dv_desc, FFTW_BACKWARD, in, out, REAL);
}

fftwf_plan fftwf_plan_many_dft_c2r(FFTZ_INT32 rank, const FFTZ_INT32 *n,
                                   FFTZ_INT32 howmany, fftwf_complex *in,
                                   const FFTZ_INT32 *inembed,
                                   FFTZ_INT32 istride, FFTZ_INT32 idist,
                                   FFTZ_FLOAT *out, const FFTZ_INT32 *onembed,
                                   FFTZ_INT32 ostride, FFTZ_INT32 odist,
                                   unsigned flags)
{
    dv_desc *p_dv_desc = get_many_c2r_dv_desc(rank, n, howmany, inembed,
        istride, idist, onembed, ostride, odist,
        (FFTZ_VOID *)in == (FFTZ_VOID *)out);
    return get_handle_f(p_dv_desc, FFTW_BACKWARD, in, out, REAL);
}

fftw_plan fftw_plan_guru_dft_c2r(FFTZ_INT32 rank, const fftw_iodim *dims,
                                 FFTZ_INT32 howmany_rank,
                                 const fftw_iodim *howmany_dims,
                                 fftw_complex *in, FFTZ_DOUBLE *out,
                                 unsigned flags)
{
    dv_desc *p_dv_desc = get_guru_dv_desc(rank, dims, howmany_rank,
                                         howmany_dims);
    return get_handle_d(p_dv_desc, FFTW_BACKWARD, in, out, REAL);
}

fftwf_plan fftwf_plan_guru_dft_c2r(FFTZ_INT32 rank, const fftwf_iodim *dims_f,
                                   FFTZ_INT32 howmany_rank,
                                   const fftwf_iodim *howmany_dims_f,
                                   fftwf_complex *in, FFTZ_FLOAT *out,
                                   unsigned flags)
{
    const fftw_iodim *dims = (const fftw_iodim *)dims_f;
    const fftw_iodim *howmany_dims = (const fftw_iodim *)howmany_dims_f;
    dv_desc *p_dv_desc = get_guru_dv_desc(rank, dims, howmany_rank,
                                         howmany_dims);
    return get_handle_f(p_dv_desc, FFTW_BACKWARD, in, out, REAL);
}

fftw_plan fftw_plan_guru64_dft_c2r(FFTZ_INT32 rank, const fftw_iodim64 *dims,
                                   FFTZ_INT32 howmany_rank,
                                   const fftw_iodim64 *howmany_dims,
                                   fftw_complex *in, FFTZ_DOUBLE *out,
                                   unsigned flags)
{
    dv_desc_64_ *p_dv_desc = get_guru_64_dv_desc(rank, dims, howmany_rank,
                                                howmany_dims);
    return get_handle_d_64_(p_dv_desc, FFTW_BACKWARD, in, out, REAL);
}

fftwf_plan fftwf_plan_guru64_dft_c2r(FFTZ_INT32 rank,
                                     const fftwf_iodim64 *dims_f,
                                     FFTZ_INT32 howmany_rank,
                                     const fftwf_iodim64 *howmany_dims_f,
                                     fftwf_complex *in, FFTZ_FLOAT *out,
                                     unsigned flags)
{
    const fftw_iodim64 *dims = (const fftw_iodim64 *)dims_f;
    const fftw_iodim64 *howmany_dims = (const fftw_iodim64 *)howmany_dims_f;
    dv_desc_64_ *p_dv_desc = get_guru_64_dv_desc(rank, dims, howmany_rank,
                                                howmany_dims);
    return get_handle_f_64_(p_dv_desc, FFTW_BACKWARD, in, out, REAL);
}
