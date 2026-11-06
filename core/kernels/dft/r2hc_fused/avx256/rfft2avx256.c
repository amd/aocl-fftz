// Copyright Advanced Micro Devices, Inc.
// SPDX-License-Identifier: BSD-3-Clause

/** @file rfft2avx256.c
 *
 *  @brief Radix-2 r2hc_fused Real-FFT kernel with AVX-256 operations using x86
 *  SIMD intrinsics.
 *
 *  This file contains the DIT radix-2 real-to-halfcomplex fused of two
 *  different implementations (Standard DFT and Shifted DFT that differs in DFT
 *  weight matrix) using AVX256 SIMD operations for single-precision and
 *  double-precision inputs.
 *
 *  @author Jeya R
 */

#include "core/kernels/kernel.h"
#include "core/kernels/simd_includes/r2hc_simd_avx256.h"

static const ops_cycles_t ops_cnt[NUM_PRECISIONS][NUM_FFT_DIRS] =
                                                    {{{0, 0, 2, 56, 39, 8},
                                                      {0, 0, 4, 56, 42, 8}},
                                                     {{0, 0, 2, 28, 3, 8},
                                                      {0, 0, 4, 28, 3, 8}}};

ops_cycles_t get_ops_cnt_r2hcf_rfft2avx256(FFTZ_UINT8 precision,
                                           FFTZ_UINT8 direction)
{
     if (precision == DT_FLOAT)
    {
        if (direction == FORWARD_FFT_DIR)
        {
            return ops_cnt[0][0];
        }
        else
        {
            return ops_cnt[0][1];
        }
    }
    else
    {
        if (direction == FORWARD_FFT_DIR)
        {
            return ops_cnt[1][0];
        }
        else
        {
            return ops_cnt[1][1];
        }
    }
}

static FFTZ_VOID r2hcf_rfft2avx256_fp32_fwd(FFTZ_VOID *in_real,
                                            FFTZ_VOID *in_complex,
                                            FFTZ_VOID *out_real,
                                            FFTZ_VOID *out_complex,
                                            FFTZ_INTP n,
                                            aoclfftz_strides_t *strides,
                                            FFTZ_VOID *twd, FFTZ_UINT8 flag)
{
    FFTZ_FLOAT *in_r = (FFTZ_FLOAT *)in_real;
    FFTZ_FLOAT *out_r = (FFTZ_FLOAT *)out_real;
    FFTZ_FLOAT *out_cp = (FFTZ_FLOAT *)out_complex;
#ifdef VOLATILE_STRIDE_ARRAY
    volatile FFTZ_INTP *in_strides = strides->in_strides;
    volatile FFTZ_INTP *out_strides = strides->out_strides;
#else
    FFTZ_INTP *in_strides = strides->in_strides;
    FFTZ_INTP *out_strides = strides->out_strides;
#endif
    FFTZ_INTP v_in_stride = strides->v_in_stride;
    FFTZ_INTP v_out_stride = strides->v_out_stride;
    FFTZ_UINT8 is_contiguous_in = (v_in_stride == 1);
    FFTZ_INTP v_out_dc_nyq_stride = strides->v_out_sym_stride;
    FFTZ_UINT8 is_contiguous_out_dc_nyq = (v_out_dc_nyq_stride == 1);

    FFTZ_INTP cnt;
    FFTZ_FLOAT *curr_in, *curr_out;
    FFTZ_INTP N = n / NUM_SETS_REAL_256_S;
    FFTZ_INTP remaining_sets = n % NUM_SETS_REAL_256_S;

    for (cnt = 0; cnt < N; cnt++)
    {
        /* Standard DFT */
        __m256 av_in0, av_in1;
        __m256 v_out0, v_out1, v_out2;

        curr_in = in_r;
        curr_out = out_cp;
        // Input point 1: x(0)
        LDR_256_S(curr_in, v_in_stride, av_in0, is_contiguous_in);

        curr_in = in_r + in_strides[2];
        LDR_256_S(curr_in, v_in_stride, av_in1, is_contiguous_in);

        // Output point 1: x(0)
        v_out0 = _mm256_add_ps(av_in0, av_in1);
        curr_out = out_r;
        STR_256_S(curr_out, v_out_dc_nyq_stride, v_out0, is_contiguous_out_dc_nyq);
        curr_out = out_r + out_strides[3];
        // Output point 4: x(3)
        v_out1 = _mm256_sub_ps(av_in0, av_in1);
        STR_256_S(curr_out, v_out_dc_nyq_stride, v_out1, is_contiguous_out_dc_nyq);

        /* Shifted DFT */
        __m256 bv_in0, bv_in1;

        curr_in = in_r + in_strides[1];
        LDR_256_S(curr_in, v_in_stride, bv_in0, is_contiguous_in);

        curr_in = in_r + in_strides[3];
        LDR_256_S(curr_in, v_in_stride, bv_in1, is_contiguous_in);

        // Output point 2: x(1)  & Output point 3: x(2)
        curr_out = out_cp + out_strides[1];
        v_out2 = NEGATE_256_S(bv_in1);
        STRI_2x256_S(curr_out, v_out_stride, bv_in0, v_out2);

        in_r += v_in_stride * NUM_SETS_REAL_256_S;
        out_cp += v_out_stride * NUM_SETS_REAL_256_S;
        out_r += v_out_dc_nyq_stride * NUM_SETS_REAL_256_S;
    }

    // tail cases
    if (remaining_sets & NUM_SETS_REAL_128_S)
    {
        /* Standard DFT */
        __m128 av_in0, av_in1;
        __m128 v_out0, v_out1, v_out2;

        curr_in = in_r;
        curr_out = out_cp;

        // Input point 1: x(0)
        LDR_128_S(curr_in, v_in_stride, av_in0, is_contiguous_in);

        curr_in = in_r + in_strides[2];
        LDR_128_S(curr_in, v_in_stride, av_in1, is_contiguous_in);

        // Output point 1: x(0)
        v_out0 = _mm_add_ps(av_in0, av_in1);
        curr_out = out_r;
        STR_128_S(curr_out, v_out_dc_nyq_stride, v_out0, is_contiguous_out_dc_nyq);
        curr_out = out_r + out_strides[3];
        // Output point 4: x(3)
        v_out1 = _mm_sub_ps(av_in0, av_in1);
        STR_128_S(curr_out, v_out_dc_nyq_stride, v_out1, is_contiguous_out_dc_nyq);

        /* Shifted DFT */
        __m128 bv_in0, bv_in1;

        curr_in = in_r + in_strides[1];
        LDR_128_S(curr_in, v_in_stride, bv_in0, is_contiguous_in);

        curr_in = in_r + in_strides[3];
        LDR_128_S(curr_in, v_in_stride, bv_in1, is_contiguous_in);

        // Output point 2: x(1)  & Output point 3: x(2)
        curr_out = out_cp + out_strides[1];
        v_out2 = NEGATE_128_S(bv_in1);
        STRI_2x128_S(curr_out, v_out_stride, bv_in0, v_out2);

        in_r = in_r + (v_in_stride << 2);
        out_cp = out_cp + (v_out_stride << 2);
        out_r = out_r + (v_out_dc_nyq_stride << 2);
    }

    if (remaining_sets & 2)
    {
        /* Standard DFT */
        __m128 av_in0, av_in1;
        __m128 v_out0, v_out1, v_out2;

        curr_in = in_r;
        curr_out = out_cp;

        // Input point 1: x(0)
        LDHR_128_S(curr_in, v_in_stride, av_in0);

        curr_in = in_r + in_strides[2];
        LDHR_128_S(curr_in, v_in_stride, av_in1);

        // Output point 1: x(0)
        v_out0 = _mm_add_ps(av_in0, av_in1);
        curr_out = out_r;
        STHR_128_S(curr_out, v_out_dc_nyq_stride, v_out0);
        curr_out = out_r + out_strides[3];
        // Output point 4: x(3)
        v_out1 = _mm_sub_ps(av_in0, av_in1);
        STHR_128_S(curr_out, v_out_dc_nyq_stride, v_out1);

        /* Shifted DFT */
        __m128 bv_in0, bv_in1;

        curr_in = in_r + in_strides[1];
        LDHR_128_S(curr_in, v_in_stride, bv_in0);

        curr_in = in_r + in_strides[3];
        LDHR_128_S(curr_in, v_in_stride, bv_in1);

        // Output point 2: x(1)  & Output point 3: x(2)
        curr_out = out_cp + out_strides[1];
        v_out2 = NEGATE_128_S(bv_in1);
        STHRI_2x128_S(curr_out, v_out_stride, bv_in0, v_out2);

        in_r = in_r + (v_in_stride << 1);
        out_cp = out_cp + (v_out_stride << 1);
        out_r = out_r + (v_out_dc_nyq_stride << 1);
    }

    if (remaining_sets & 1)
    {
        /* Standard DFT */
        FFTZ_FLOAT av_in0, av_in1;
        av_in0 = *in_r;
        av_in1 = in_r[in_strides[2]];

        *out_r = av_in0 + av_in1;
        out_r[out_strides[3]] = av_in0 - av_in1;

        /* Shifted DFT */
        FFTZ_FLOAT bv_in0, bv_in1;
        bv_in0 = in_r[in_strides[1]];
        bv_in1 = in_r[in_strides[3]];

        out_cp[out_strides[1]] = bv_in0;
        out_cp[out_strides[2]] = -bv_in1;
    }
}

static FFTZ_VOID r2hcf_rfft2avx256_fp32_bwd(FFTZ_VOID *in_real,
                                            FFTZ_VOID *in_complex,
                                            FFTZ_VOID *out_real,
                                            FFTZ_VOID *out_complex,
                                            FFTZ_INTP n,
                                            aoclfftz_strides_t *strides,
                                            FFTZ_VOID *twd, FFTZ_UINT8 flag)
{
    FFTZ_FLOAT *in_r = (FFTZ_FLOAT *)in_real;
    FFTZ_FLOAT *in_cp = (FFTZ_FLOAT *)in_complex;
    FFTZ_FLOAT *out_r = (FFTZ_FLOAT *)out_real;
#ifdef VOLATILE_STRIDE_ARRAY
    volatile FFTZ_INTP *in_strides = strides->in_strides;
    volatile FFTZ_INTP *out_strides = strides->out_strides;
#else
    FFTZ_INTP *in_strides = strides->in_strides;
    FFTZ_INTP *out_strides = strides->out_strides;
#endif
    FFTZ_INTP v_in_stride = strides->v_in_stride;
    FFTZ_INTP v_out_stride = strides->v_out_stride;
    FFTZ_UINT8 is_contiguous_out = (v_out_stride == 1);
    FFTZ_INTP v_in_dc_nyq_stride = strides->v_in_sym_stride;
    FFTZ_UINT8 is_contiguous_in_dc_nyq = (v_in_dc_nyq_stride == 1);

    FFTZ_INTP cnt;
    FFTZ_FLOAT *curr_in, *curr_out;
    FFTZ_INTP N = n / NUM_SETS_REAL_256_S;
    FFTZ_INTP remaining_sets = n % NUM_SETS_REAL_256_S;

    for (cnt = 0; cnt < N; cnt++)
    {
        /* Standard DFT */
        __m256 av_in0, av_in1;
        __m256 v_out0, v_out1, v_out2, v_out3;

        curr_in = in_r;
        curr_out = out_r;

        // Input point 1: x(0)
        LDR_256_S(curr_in, v_in_dc_nyq_stride, av_in0, is_contiguous_in_dc_nyq);

        curr_in = in_r + in_strides[3];
        LDR_256_S(curr_in, v_in_dc_nyq_stride, av_in1, is_contiguous_in_dc_nyq);

        // Output point 1: X(0)
        v_out0 = _mm256_add_ps(av_in0, av_in1);
        STR_256_S(curr_out, v_out_stride, v_out0, is_contiguous_out);
        // Output point 3: X(2)
        curr_out = out_r + out_strides[2];
        v_out2 = _mm256_sub_ps(av_in0, av_in1);
        STR_256_S(curr_out, v_out_stride, v_out2, is_contiguous_out);

        /* Shifted DFT */
        __m256 bv_in0, bv_in1;

        // Input point 2: x(1) & Input point 3: x(2)
        curr_in = in_cp + in_strides[1];
        LDRI_2x256_S(curr_in, v_in_stride, bv_in0, bv_in1);

        // Output point 2: X(1)
        curr_out = out_r + out_strides[1];
        v_out1 = _mm256_add_ps(bv_in0, bv_in0);
        STR_256_S(curr_out, v_out_stride, v_out1, is_contiguous_out);

        curr_out = out_r + out_strides[3];
        v_out3 = NEGATE_256_S(_mm256_add_ps(bv_in1, bv_in1));
        STR_256_S(curr_out, v_out_stride, v_out3, is_contiguous_out);

        in_cp += v_in_stride * NUM_SETS_REAL_256_S;
        in_r += v_in_dc_nyq_stride * NUM_SETS_REAL_256_S;
        out_r += v_out_stride * NUM_SETS_REAL_256_S;
    }

    // tailcases
    if (remaining_sets & NUM_SETS_REAL_128_S)
    {
        /* Standard DFT */
        __m128 av_in0, av_in1;
        __m128 v_out0, v_out1, v_out2, v_out3;

        curr_in = in_r;
        curr_out = out_r;

        // Input point 1: x(0)
        LDR_128_S(curr_in, v_in_dc_nyq_stride, av_in0, is_contiguous_in_dc_nyq);

        curr_in = in_r + in_strides[3];
        LDR_128_S(curr_in, v_in_dc_nyq_stride, av_in1, is_contiguous_in_dc_nyq);

        // Output point 1: X(0)
        v_out0 = _mm_add_ps(av_in0, av_in1);
        STR_128_S(curr_out, v_out_stride, v_out0, is_contiguous_out);
        // Output point 3: X(2)
        curr_out = out_r + out_strides[2];
        v_out2 = _mm_sub_ps(av_in0, av_in1);
        STR_128_S(curr_out, v_out_stride, v_out2, is_contiguous_out);

        /* Shifted DFT */
        __m128 bv_in0, bv_in1;

        // Input point 2: x(1) & Input point 3: x(2)
        curr_in = in_cp + in_strides[1];
        LDRI_2x128_S(curr_in, v_in_stride, bv_in0, bv_in1);

        // Output point 2: X(1)
        curr_out = out_r + out_strides[1];
        v_out1 = _mm_add_ps(bv_in0, bv_in0);
        STR_128_S(curr_out, v_out_stride, v_out1, is_contiguous_out);

        curr_out = out_r + out_strides[3];
        v_out3 = NEGATE_128_S(_mm_add_ps(bv_in1, bv_in1));
        STR_128_S(curr_out, v_out_stride, v_out3, is_contiguous_out);

        in_cp = in_cp + (v_in_stride << 2);
        in_r = in_r + (v_in_dc_nyq_stride << 2);
        out_r = out_r + (v_out_stride << 2);
    }

    if (remaining_sets & 2)
    {
        /* Standard DFT */
        __m128 av_in0, av_in1;
        __m128 v_out0, v_out1, v_out2, v_out3;

        curr_in = in_r;
        curr_out = out_r;

        // Input point 1: x(0)
        LDHR_128_S(curr_in, v_in_dc_nyq_stride, av_in0);

        curr_in = in_r + in_strides[3];
        LDHR_128_S(curr_in, v_in_dc_nyq_stride, av_in1);

        // Output point 1: X(0)
        v_out0 = _mm_add_ps(av_in0, av_in1);
        STHR_128_S(curr_out, v_out_stride, v_out0);
        // Output point 3: X(2)
        curr_out = out_r + out_strides[2];
        v_out2 = _mm_sub_ps(av_in0, av_in1);
        STHR_128_S(curr_out, v_out_stride, v_out2);

        /* Shifted DFT */
        __m128 bv_in0, bv_in1;

        // Input point 2: x(1) & Input point 3: x(2)
        curr_in = in_cp + in_strides[1];
        LDHRI_2x128_S(curr_in, v_in_stride, bv_in0, bv_in1);

        // Output point 2: X(1)
        curr_out = out_r + out_strides[1];
        v_out1 = _mm_add_ps(bv_in0, bv_in0);
        STHR_128_S(curr_out, v_out_stride, v_out1);

        curr_out = out_r + out_strides[3];
        v_out3 = NEGATE_128_S(_mm_add_ps(bv_in1, bv_in1));
        STHR_128_S(curr_out, v_out_stride, v_out3);

        in_cp = in_cp + (v_in_stride << 1);
        in_r = in_r + (v_in_dc_nyq_stride << 1);
        out_r = out_r + (v_out_stride << 1);
    }

    if (remaining_sets & 1)
    {
        /* Standard DFT */
        FFTZ_FLOAT av_in0, av_in1;
        av_in0 = *in_r;
        av_in1 = in_r[in_strides[3]];

        *out_r = av_in0 + av_in1;
        out_r[out_strides[2]] = av_in0 - av_in1;

        /* Shifted DFT */
        FFTZ_FLOAT bv_in0, bv_in1;
        bv_in0 = in_cp[in_strides[1]];
        bv_in1 = in_cp[in_strides[2]];

        out_r[out_strides[1]] = bv_in0 + bv_in0;
        out_r[out_strides[3]] = -(bv_in1 + bv_in1);
    }
}

static FFTZ_VOID r2hcf_rfft2avx256_fp64_fwd(FFTZ_VOID *in_real,
                                            FFTZ_VOID *in_complex,
                                            FFTZ_VOID *out_real,
                                            FFTZ_VOID *out_complex,
                                            FFTZ_INTP n,
                                            aoclfftz_strides_t *strides,
                                            FFTZ_VOID *twd, FFTZ_UINT8 flag)
{
    FFTZ_DOUBLE *in_r = (FFTZ_DOUBLE *)in_real;
    FFTZ_DOUBLE *out_r = (FFTZ_DOUBLE *)out_real;
    FFTZ_DOUBLE *out_cp = (FFTZ_DOUBLE *)out_complex;
#ifdef VOLATILE_STRIDE_ARRAY
    volatile FFTZ_INTP *in_strides = strides->in_strides;
    volatile FFTZ_INTP *out_strides = strides->out_strides;
#else
    FFTZ_INTP *in_strides = strides->in_strides;
    FFTZ_INTP *out_strides = strides->out_strides;
#endif
    FFTZ_INTP v_in_stride = strides->v_in_stride;
    FFTZ_INTP v_out_stride = strides->v_out_stride;
    FFTZ_UINT8 is_contiguous_in = (v_in_stride == 1);
    FFTZ_INTP v_out_dc_nyq_stride = strides->v_out_sym_stride;
    FFTZ_UINT8 is_contiguous_out_dc_nyq = (v_out_dc_nyq_stride == 1);

    FFTZ_INTP cnt;
    FFTZ_DOUBLE *curr_in, *curr_out;
    FFTZ_INTP N = n / NUM_SETS_REAL_256_D;
    FFTZ_INTP remaining_sets = n % NUM_SETS_REAL_256_D;

    for (cnt = 0; cnt < N; cnt++)
    {
        /* Standard DFT */
        __m256d av_in0, av_in1;
        __m256d v_out0, v_out1, v_out2;

        curr_in = in_r;
        curr_out = out_cp;

        // Input point 1: x(0)
        LDR_256_D(curr_in, v_in_stride, av_in0, is_contiguous_in);

        curr_in = in_r + in_strides[2];
        LDR_256_D(curr_in, v_in_stride, av_in1, is_contiguous_in);

        // Output point 1: x(0)
        v_out0 = _mm256_add_pd(av_in0, av_in1);
        curr_out = out_r;
        STR_256_D(curr_out, v_out_dc_nyq_stride, v_out0, is_contiguous_out_dc_nyq);
        curr_out = out_r + out_strides[3];
        // Output point 4: x(3)
        v_out1 = _mm256_sub_pd(av_in0, av_in1);
        STR_256_D(curr_out, v_out_dc_nyq_stride, v_out1, is_contiguous_out_dc_nyq);

        /* Shifted DFT */
        __m256d bv_in0, bv_in1;

        curr_in = in_r + in_strides[1];
        LDR_256_D(curr_in, v_in_stride, bv_in0, is_contiguous_in);

        curr_in = in_r + in_strides[3];
        LDR_256_D(curr_in, v_in_stride, bv_in1, is_contiguous_in);

        // Output point 2: x(1)  & Output point 3: x(2)
        curr_out = out_cp + out_strides[1];
        v_out2 = NEGATE_256_D(bv_in1);
        STRI_2x256_D(curr_out, v_out_stride, bv_in0, v_out2);

        in_r += v_in_stride * NUM_SETS_REAL_256_D;
        out_cp += v_out_stride * NUM_SETS_REAL_256_D;
        out_r += v_out_dc_nyq_stride * NUM_SETS_REAL_256_D;
    }

    // tailcases
    if (remaining_sets & NUM_SETS_REAL_128_D)
    {
        /* Standard DFT */
        __m128d av_in0, av_in1;
        __m128d v_out0, v_out1, v_out2;

        curr_in = in_r;
        curr_out = out_cp;

        // Input point 1: x(0)
        LDR_128_D(curr_in, v_in_stride, av_in0, is_contiguous_in);

        curr_in = in_r + in_strides[2];
        LDR_128_D(curr_in, v_in_stride, av_in1, is_contiguous_in);

        // Output point 1: x(0)
        v_out0 = _mm_add_pd(av_in0, av_in1);
        curr_out = out_r;
        STR_128_D(curr_out, v_out_dc_nyq_stride, v_out0, is_contiguous_out_dc_nyq);
        curr_out = out_r + out_strides[3];
        // Output point 4: x(3)
        v_out1 = _mm_sub_pd(av_in0, av_in1);
        STR_128_D(curr_out, v_out_dc_nyq_stride, v_out1, is_contiguous_out_dc_nyq);

        /* Shifted DFT */
        __m128d bv_in0, bv_in1;

        curr_in = in_r + in_strides[1];
        LDR_128_D(curr_in, v_in_stride, bv_in0, is_contiguous_in);

        curr_in = in_r + in_strides[3];
        LDR_128_D(curr_in, v_in_stride, bv_in1, is_contiguous_in);

        // Output point 2: x(1)  & Output point 3: x(2)
        curr_out = out_cp + out_strides[1];
        v_out2 = NEGATE_128_D(bv_in1);
        STRI_2x128_D(curr_out, v_out_stride, bv_in0, v_out2);

        in_r = in_r + (v_in_stride << 1);
        out_cp = out_cp + (v_out_stride << 1);
        out_r = out_r + (v_out_dc_nyq_stride << 1);
    }
    // tail cases
    if (remaining_sets & 1)
    {
        /* Standard DFT */
        FFTZ_DOUBLE av_in0, av_in1;
        av_in0 = *in_r;
        av_in1 = in_r[in_strides[2]];

        *out_r = av_in0 + av_in1;
        out_r[out_strides[3]] = av_in0 - av_in1;

        /* Shifted DFT */
        FFTZ_DOUBLE bv_in0, bv_in1;
        bv_in0 = in_r[in_strides[1]];
        bv_in1 = in_r[in_strides[3]];

        out_cp[out_strides[1]] = bv_in0;
        out_cp[out_strides[2]] = -bv_in1;
    }
}

static FFTZ_VOID r2hcf_rfft2avx256_fp64_bwd(FFTZ_VOID *in_real,
                                            FFTZ_VOID *in_complex,
                                            FFTZ_VOID *out_real,
                                            FFTZ_VOID *out_complex,
                                            FFTZ_INTP n,
                                            aoclfftz_strides_t *strides,
                                            FFTZ_VOID *twd, FFTZ_UINT8 flag)
{
    FFTZ_DOUBLE *in_r = (FFTZ_DOUBLE *)in_real;
    FFTZ_DOUBLE *in_cp = (FFTZ_DOUBLE *)in_complex;
    FFTZ_DOUBLE *out_r = (FFTZ_DOUBLE *)out_real;
#ifdef VOLATILE_STRIDE_ARRAY
    volatile FFTZ_INTP *in_strides = strides->in_strides;
    volatile FFTZ_INTP *out_strides = strides->out_strides;
#else
    FFTZ_INTP *in_strides = strides->in_strides;
    FFTZ_INTP *out_strides = strides->out_strides;
#endif
    FFTZ_INTP v_in_stride = strides->v_in_stride;
    FFTZ_INTP v_out_stride = strides->v_out_stride;
    FFTZ_UINT8 is_contiguous_out = (v_out_stride == 1);
    FFTZ_INTP v_in_dc_nyq_stride = strides->v_in_sym_stride;
    FFTZ_UINT8 is_contiguous_in_dc_nyq = (v_in_dc_nyq_stride == 1);

    FFTZ_INTP cnt;
    FFTZ_DOUBLE *curr_in, *curr_out;
    FFTZ_INTP N = n / NUM_SETS_REAL_256_D;
    FFTZ_INTP remaining_sets = n % NUM_SETS_REAL_256_D;

    for (cnt = 0; cnt < N; cnt++)
    {
        /* Standard DFT */
        __m256d av_in0, av_in1;
        __m256d v_out0, v_out1, v_out2, v_out3;

        curr_in = in_r;
        curr_out = out_r;

        // Input point 1: x(0)
        LDR_256_D(curr_in, v_in_dc_nyq_stride, av_in0, is_contiguous_in_dc_nyq);

        curr_in = in_r + in_strides[3];
        LDR_256_D(curr_in, v_in_dc_nyq_stride, av_in1, is_contiguous_in_dc_nyq);

        // Output point 1: X(0)
        v_out0 = _mm256_add_pd(av_in0, av_in1);
        STR_256_D(curr_out, v_out_stride, v_out0, is_contiguous_out);
        // Output point 3: X(2)
        curr_out = out_r + out_strides[2];
        v_out2 = _mm256_sub_pd(av_in0, av_in1);
        STR_256_D(curr_out, v_out_stride, v_out2, is_contiguous_out);

        /* Shifted DFT */
        __m256d bv_in0, bv_in1;

        // Input point 2: x(1) & Input point 3: x(2)
        curr_in = in_cp + in_strides[1];
        LDRI_2x256_D(curr_in, v_in_stride, bv_in0, bv_in1);

        // Output point 2: X(1)
        curr_out = out_r + out_strides[1];
        v_out1 = _mm256_add_pd(bv_in0, bv_in0);
        STR_256_D(curr_out, v_out_stride, v_out1, is_contiguous_out);

        curr_out = out_r + out_strides[3];
        v_out3 = NEGATE_256_D(_mm256_add_pd(bv_in1, bv_in1));
        STR_256_D(curr_out, v_out_stride, v_out3, is_contiguous_out);

        in_cp += v_in_stride * NUM_SETS_REAL_256_D;
        in_r += v_in_dc_nyq_stride * NUM_SETS_REAL_256_D;
        out_r += v_out_stride * NUM_SETS_REAL_256_D;
    }

    // tailcases
    if (remaining_sets & NUM_SETS_REAL_128_D)
    {
        /* Standard DFT */
        __m128d av_in0, av_in1;
        __m128d v_out0, v_out1, v_out2, v_out3;

        curr_in = in_r;
        curr_out = out_r;

        // Input point 1: x(0)
        LDR_128_D(curr_in, v_in_dc_nyq_stride, av_in0, is_contiguous_in_dc_nyq);

        curr_in = in_r + in_strides[3];
        LDR_128_D(curr_in, v_in_dc_nyq_stride, av_in1, is_contiguous_in_dc_nyq);

        // Output point 1: X(0)
        v_out0 = _mm_add_pd(av_in0, av_in1);
        STR_128_D(curr_out, v_out_stride, v_out0, is_contiguous_out);
        // Output point 3: X(2)
        curr_out = out_r + out_strides[2];
        v_out2 = _mm_sub_pd(av_in0, av_in1);
        STR_128_D(curr_out, v_out_stride, v_out2, is_contiguous_out);

        /* Shifted DFT */
        __m128d bv_in0, bv_in1;

        // Input point 2: x(1) & Input point 3: x(2)
        curr_in = in_cp + in_strides[1];
        LDRI_2x128_D(curr_in, v_in_stride, bv_in0, bv_in1);

        // Output point 2: X(1)
        curr_out = out_r + out_strides[1];
        v_out1 = _mm_add_pd(bv_in0, bv_in0);
        STR_128_D(curr_out, v_out_stride, v_out1, is_contiguous_out);

        curr_out = out_r + out_strides[3];
        v_out3 = NEGATE_128_D(_mm_add_pd(bv_in1, bv_in1));
        STR_128_D(curr_out, v_out_stride, v_out3, is_contiguous_out);

        in_cp = in_cp + (v_in_stride << 1);
        in_r = in_r + (v_in_dc_nyq_stride << 1);
        out_r = out_r + (v_out_stride << 1);
    }
    // tail cases
    if (remaining_sets & 1)
    {
        /* Standard DFT */
        FFTZ_DOUBLE av_in0, av_in1;
        av_in0 = *in_r;
        av_in1 = in_r[in_strides[3]];

        *out_r = av_in0 + av_in1;
        out_r[out_strides[2]] = av_in0 - av_in1;

        /* Shifted DFT */
        FFTZ_DOUBLE bv_in0, bv_in1;
        bv_in0 = in_cp[in_strides[1]];
        bv_in1 = in_cp[in_strides[2]];

        out_r[out_strides[1]] = bv_in0 + bv_in0;
        out_r[out_strides[3]] = -(bv_in1 + bv_in1);
    }
}

kfft_ register_kernel_r2hcf_rfft2avx256(FFTZ_UINT8 precision,
                                        FFTZ_UINT8 direction)
{
    if (direction == FORWARD_FFT_DIR)
    {
        if (precision == DT_FLOAT)
        {
            return r2hcf_rfft2avx256_fp32_fwd;
        }
        else if (precision == DT_DOUBLE)
        {
            return r2hcf_rfft2avx256_fp64_fwd;
        }
        else
        {
            return NULL;
        }
    }
    else
    {
        if (precision == DT_FLOAT)
        {
            return r2hcf_rfft2avx256_fp32_bwd;
        }
        else if (precision == DT_DOUBLE)
        {
            return r2hcf_rfft2avx256_fp64_bwd;
        }
        else
        {
            return NULL;
        }
    }
}
