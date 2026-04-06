// Copyright Advanced Micro Devices, Inc.
// SPDX-License-Identifier: BSD-3-Clause

/** @file rfft2avx128.c
 *
 *  @brief Radix-2 r2hc_fused Real-FFT kernel with AVX-128 operations using x86
 *  SIMD intrinsics
 *
 *  This file contains the DIT radix-2 real-to-halfcomplex fused of two
 *  different implementations (Standard DFT and Shifted DFT that differs in DFT
 *  weight matrix) using x86 SIMD intrinsics for single-precision and
 *  double-precision inputs
 *
 *  @author Jeevanantham N
 */

#include "core/kernels/kernel.h"
#include "core/kernels/simd_includes/r2hc_simd_avx128.h"

static const ops_cycles_t ops_cnt[NUM_PRECISIONS][NUM_FFT_DIRS] =
                                                    {{{0, 0, 2, 28, 20, 1},
                                                      {0, 0, 4, 28, 21, 1}},
                                                     {{0, 0, 2, 14, 2, 1},
                                                      {0, 0, 4, 14, 2, 1}}};

ops_cycles_t get_ops_cnt_r2hcf_rfft2avx128(UINT8 precision, UINT8 direction)
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

static VOID r2hcf_rfft2avx128_fp32_fwd(VOID *in_real, VOID *in_imag,
                                       VOID *out_real, VOID *out_imag, INTP n,
                                       aoclfftz_strides_t *strides, VOID *twd, UINT8 flag)
{
    FLOAT *in = (FLOAT *)in_real;
    FLOAT *out = (FLOAT *)out_real;
#ifdef VOLATILE_STRIDE_ARRAY
    volatile INTP *in_strides = strides->in_strides;
    volatile INTP *out_strides = strides->out_strides;
#else
    INTP *in_strides = strides->in_strides;
    INTP *out_strides = strides->out_strides;
#endif
    INTP v_in_stride = strides->v_in_stride;
    INTP v_out_stride = strides->v_out_stride;

    INTP cnt;
    FLOAT *curr_in, *curr_out;
    INTP N = n / NUM_SETS_REAL_128_S;

    for (cnt = 0; cnt < N; cnt++)
    {
        /* Standard DFT */
        __m128 av_in0, av_in1;
        __m128 v_out0, v_out1, v_out2;

        curr_in = in;
        curr_out = out;

        // Input point 1: x(0)
        LDR_128_S(curr_in, v_in_stride, av_in0);
        // Input point 3: x(2)
        curr_in = in + in_strides[2];
        LDR_128_S(curr_in, v_in_stride, av_in1);

        // Output point 1: x(0)
        v_out0 = _mm_add_ps(av_in0, av_in1);
        STR_128_S(curr_out, v_out_stride, v_out0);
        // Output point 4: x(3)
        curr_out = out + out_strides[3];
        v_out1 = _mm_sub_ps(av_in0, av_in1);
        STR_128_S(curr_out, v_out_stride, v_out1);

        /* Shifted DFT */
        __m128 bv_in0, bv_in1;

        // Input point 2: x(1)
        curr_in = in + in_strides[1];
        LDR_128_S(curr_in, v_in_stride, bv_in0);
        // Input point 4: x(3)
        curr_in = in + in_strides[3];
        LDR_128_S(curr_in, v_in_stride, bv_in1);

        // Output point 2: x(1)  & Output point 3: x(2)
        curr_out = out + out_strides[1];
        v_out2 = NEGATE_128_S(bv_in1);
        STRI_2x128_S(curr_out, v_out_stride, bv_in0, v_out2);

        in  += v_in_stride * NUM_SETS_REAL_128_S;
        out += v_out_stride * NUM_SETS_REAL_128_S;
    }
    // tail cases
    if (n & 2)
    {
        /* Standard DFT */
       __m128 av_in0, av_in1;
       __m128 v_out0, v_out1, v_out2;

       curr_in = in;
       curr_out = out;

       // Input point 1: x(0)
       LDHR_128_S(curr_in, v_in_stride, av_in0);
       // Input point 3: x(2)
       curr_in = in + in_strides[2];
       LDHR_128_S(curr_in, v_in_stride, av_in1);

       // Output point 1: x(0)
       v_out0 = _mm_add_ps(av_in0, av_in1);
       STHR_128_S(curr_out, v_out_stride, v_out0);
       // Output point 4: x(3)
       curr_out = out + out_strides[3];
       v_out1 = _mm_sub_ps(av_in0, av_in1);
       STHR_128_S(curr_out, v_out_stride, v_out1);

        /* Shifted DFT */
       __m128 bv_in0, bv_in1;

       // Input point 2: x(1)
       curr_in = in + in_strides[1];
       LDHR_128_S(curr_in, v_in_stride, bv_in0);
       // Input point 4: x(3)
       curr_in = in + in_strides[3];
       LDHR_128_S(curr_in, v_in_stride, bv_in1);

       // Output point 2: x(1)  & Output point 3: x(2)
       curr_out = out + out_strides[1];
       v_out2 = NEGATE_128_S(bv_in1);
       STHRI_2x128_S(curr_out, v_out_stride, bv_in0, v_out2);

       in = in + (v_in_stride << 1);
       out = out + (v_out_stride << 1);
    }
    // tail cases
    if (n & 1)
    {
        /* Standard DFT */
        FLOAT av_in0, av_in1;
        av_in0 = *in;
        av_in1 = in[in_strides[2]];

        *out = av_in0 + av_in1;
        out[out_strides[3]] = av_in0 - av_in1;

        /* Shifted DFT */
        FLOAT bv_in0, bv_in1;
        bv_in0 = in[in_strides[1]];
        bv_in1 = in[in_strides[3]];

        out[out_strides[1]] = bv_in0;
        out[out_strides[2]] = -bv_in1;
    }
}

static VOID r2hcf_rfft2avx128_fp32_bwd(VOID *in_real, VOID *in_imag,
                                       VOID *out_real, VOID *out_imag, INTP n,
                                       aoclfftz_strides_t *strides, VOID *twd, UINT8 flag)
{
    FLOAT *in = (FLOAT *)in_real;
    FLOAT *out = (FLOAT *)out_real;
#ifdef VOLATILE_STRIDE_ARRAY
    volatile INTP *in_strides = strides->in_strides;
    volatile INTP *out_strides = strides->out_strides;
#else
    INTP *in_strides = strides->in_strides;
    INTP *out_strides = strides->out_strides;
#endif
    INTP v_in_stride = strides->v_in_stride;
    INTP v_out_stride = strides->v_out_stride;

    INTP cnt;
    FLOAT *curr_in, *curr_out;
    INTP N = n / NUM_SETS_REAL_128_S;

    for (cnt = 0; cnt < N; cnt++)
    {
        /* Standard DFT */
        __m128 av_in0, av_in1;
        __m128 v_out0, v_out1, v_out2, v_out3;

        curr_in = in;
        curr_out = out;

        // Input point 1: x(0)
        LDR_128_S(curr_in, v_in_stride, av_in0);
        // Input point 4: x(3)
        curr_in = in + in_strides[3];
        LDR_128_S(curr_in, v_in_stride, av_in1);

        // Output point 1: X(0)
        v_out0 = _mm_add_ps(av_in0, av_in1);
        STR_128_S(curr_out, v_out_stride, v_out0);
        // Output point 3: X(2)
        curr_out = out + out_strides[2];
        v_out2 = _mm_sub_ps(av_in0, av_in1);
        STR_128_S(curr_out, v_out_stride, v_out2);

        /* Shifted DFT */
        __m128 bv_in0, bv_in1;

        // Input point 2: x(1) & Input point 3: x(2)
        curr_in = in + in_strides[1];
        LDRI_2x128_S(curr_in, v_in_stride, bv_in0, bv_in1);

        // Output point 2: X(1)
        curr_out = out + out_strides[1];
        v_out1 = _mm_add_ps(bv_in0, bv_in0);
        STR_128_S(curr_out, v_out_stride, v_out1);
        // Output point 4: X(3)
        curr_out = out + out_strides[3];
        v_out3 = NEGATE_128_S(_mm_add_ps(bv_in1, bv_in1));
        STR_128_S(curr_out, v_out_stride, v_out3);

        in  += v_in_stride * NUM_SETS_REAL_128_S;
        out += v_out_stride * NUM_SETS_REAL_128_S;
    }
    // tail cases
    if (n & 2)
    {
        /* Standard DFT */
       __m128 av_in0, av_in1;
       __m128 v_out0, v_out1, v_out2, v_out3;

       curr_in = in;
       curr_out = out;

       // Input point 1: x(0)
       LDHR_128_S(curr_in, v_in_stride, av_in0);
       // Input point 4: x(3)
       curr_in = in + in_strides[3];
       LDHR_128_S(curr_in, v_in_stride, av_in1);

       // Output point 1: X(0)
       v_out0 = _mm_add_ps(av_in0, av_in1);
       STHR_128_S(curr_out, v_out_stride, v_out0);
       // Output point 3: X(2)
       curr_out = out + out_strides[2];
       v_out2 = _mm_sub_ps(av_in0, av_in1);
       STHR_128_S(curr_out, v_out_stride, v_out2);

        /* Shifted DFT */
       __m128 bv_in0, bv_in1;

       // Input point 2: x(1) & Input point 3: x(2)
       curr_in = in + in_strides[1];
       LDHRI_2x128_S(curr_in, v_in_stride, bv_in0, bv_in1);

       // Output point 2: X(1)
       curr_out = out + out_strides[1];
       v_out1 = _mm_add_ps(bv_in0, bv_in0);
       STHR_128_S(curr_out, v_out_stride, v_out1);
       // Output point 4: X(3)
       curr_out = out + out_strides[3];
       v_out3 = NEGATE_128_S(_mm_add_ps(bv_in1, bv_in1));
       STHR_128_S(curr_out, v_out_stride, v_out3);

       in = in + (v_in_stride << 1);
       out = out + (v_out_stride << 1);
    }
    // tail cases
    if (n & 1)
    {
        /* Standard DFT */
        FLOAT av_in0, av_in1;
        av_in0 = *in;
        av_in1 = in[in_strides[3]];

        *out = av_in0 + av_in1;
        out[out_strides[2]] = av_in0 - av_in1;

        /* Shifted DFT */
        FLOAT bv_in0, bv_in1;
        bv_in0 = in[in_strides[1]];
        bv_in1 = in[in_strides[2]];

        out[out_strides[1]] = bv_in0 + bv_in0;
        out[out_strides[3]] = -(bv_in1 + bv_in1);
    }
}

static VOID r2hcf_rfft2avx128_fp64_fwd(VOID *in_real, VOID *in_imag,
                                       VOID *out_real, VOID *out_imag, INTP n,
                                       aoclfftz_strides_t *strides, VOID *twd, UINT8 flag)
{
    DOUBLE *in = (DOUBLE *)in_real;
    DOUBLE *out = (DOUBLE *)out_real;
#ifdef VOLATILE_STRIDE_ARRAY
    volatile INTP *in_strides = strides->in_strides;
    volatile INTP *out_strides = strides->out_strides;
#else
    INTP *in_strides = strides->in_strides;
    INTP *out_strides = strides->out_strides;
#endif
    INTP v_in_stride = strides->v_in_stride;
    INTP v_out_stride = strides->v_out_stride;

    INTP cnt;
    DOUBLE *curr_in, *curr_out;
    INTP N = n / NUM_SETS_REAL_128_D;

    for (cnt = 0; cnt < N; cnt++)
    {
        /* Standard DFT */
        __m128d av_in0, av_in1;
        __m128d v_out0, v_out1, v_out2;

        curr_in = in;
        curr_out = out;

        // Input point 1: x(0)
        LDR_128_D(curr_in, v_in_stride, av_in0);
        // Input point 3: x(2)
        curr_in = in + in_strides[2];
        LDR_128_D(curr_in, v_in_stride, av_in1);

        // Output point 1: x(0)
        v_out0 = _mm_add_pd(av_in0, av_in1);
        STR_128_D(curr_out, v_out_stride, v_out0);
        // Output point 4: x(3)
        curr_out = out + out_strides[3];
        v_out1 = _mm_sub_pd(av_in0, av_in1);
        STR_128_D(curr_out, v_out_stride, v_out1);

        /* Shifted DFT */
        __m128d bv_in0, bv_in1;

        // Input point 2: x(1)
        curr_in = in + in_strides[1];
        LDR_128_D(curr_in, v_in_stride, bv_in0);
        // Input point 4: x(3)
        curr_in = in + in_strides[3];
        LDR_128_D(curr_in, v_in_stride, bv_in1);

        // Output point 2: x(1)  & Output point 3: x(2)
        curr_out = out + out_strides[1];
        v_out2 = NEGATE_128_D(bv_in1);
        STRI_2x128_D(curr_out, v_out_stride, bv_in0, v_out2);

        in += v_in_stride * NUM_SETS_REAL_128_D;
        out += v_out_stride * NUM_SETS_REAL_128_D;
    }
    // tail cases
    if (n & 1)
    {
        /* Standard DFT */
        DOUBLE av_in0, av_in1;
        av_in0 = *in;
        av_in1 = in[in_strides[2]];

        *out = av_in0 + av_in1;
        out[out_strides[3]] = av_in0 - av_in1;

        /* Shifted DFT */
        DOUBLE bv_in0, bv_in1;
        bv_in0 = in[in_strides[1]];
        bv_in1 = in[in_strides[3]];

        out[out_strides[1]] = bv_in0;
        out[out_strides[2]] = -bv_in1;
    }
}

static VOID r2hcf_rfft2avx128_fp64_bwd(VOID *in_real, VOID *in_imag,
                                       VOID *out_real, VOID *out_imag, INTP n,
                                       aoclfftz_strides_t *strides, VOID *twd, UINT8 flag)
{
    DOUBLE *in = (DOUBLE *)in_real;
    DOUBLE *out = (DOUBLE *)out_real;
#ifdef VOLATILE_STRIDE_ARRAY
    volatile INTP *in_strides = strides->in_strides;
    volatile INTP *out_strides = strides->out_strides;
#else
    INTP *in_strides = strides->in_strides;
    INTP *out_strides = strides->out_strides;
#endif
    INTP v_in_stride = strides->v_in_stride;
    INTP v_out_stride = strides->v_out_stride;

    INTP cnt;
    DOUBLE *curr_in, *curr_out;
    INTP N = n / NUM_SETS_REAL_128_D;

    for (cnt = 0; cnt < N; cnt++)
    {
        /* Standard DFT */
        __m128d av_in0, av_in1;
        __m128d v_out0, v_out1, v_out2, v_out3;

        curr_in = in;
        curr_out = out;

        // Input point 1: x(0)
        LDR_128_D(curr_in, v_in_stride, av_in0);
        // Input point 4: x(3)
        curr_in = in + in_strides[3];
        LDR_128_D(curr_in, v_in_stride, av_in1);

        // Output point 1: X(0)
        v_out0 = _mm_add_pd(av_in0, av_in1);
        STR_128_D(curr_out, v_out_stride, v_out0);
        // Output point 3: X(2)
        curr_out = out + out_strides[2];
        v_out2 = _mm_sub_pd(av_in0, av_in1);
        STR_128_D(curr_out, v_out_stride, v_out2);

        /* Shifted DFT */
        __m128d bv_in0, bv_in1;

        // Input point 2: x(1) & Input point 3: x(2)
        curr_in = in + in_strides[1];
        LDRI_2x128_D(curr_in, v_in_stride, bv_in0, bv_in1);

        // Output point 2: X(1)
        curr_out = out + out_strides[1];
        v_out1 = _mm_add_pd(bv_in0, bv_in0);
        STR_128_D(curr_out, v_out_stride, v_out1);
        // Output point 4: X(3)
        curr_out = out + out_strides[3];
        v_out3 = NEGATE_128_D(_mm_add_pd(bv_in1, bv_in1));
        STR_128_D(curr_out, v_out_stride, v_out3);

        in += v_in_stride * NUM_SETS_REAL_128_D;
        out += v_out_stride * NUM_SETS_REAL_128_D;
    }
    // tail cases
    if (n & 1)
    {
        /* Standard DFT */
        DOUBLE av_in0, av_in1;
        av_in0 = *in;
        av_in1 = in[in_strides[3]];

        *out = av_in0 + av_in1;
        out[out_strides[2]] = av_in0 - av_in1;

        /* Shifted DFT */
        DOUBLE bv_in0, bv_in1;
        bv_in0 = in[in_strides[1]];
        bv_in1 = in[in_strides[2]];

        out[out_strides[1]] = bv_in0 + bv_in0;
        out[out_strides[3]] = -(bv_in1 + bv_in1);
    }
}

kfft_ register_kernel_r2hcf_rfft2avx128(UINT8 precision, UINT8 direction)
{
    if (direction == FORWARD_FFT_DIR)
    {
        if (precision == DT_FLOAT)
        {
            return r2hcf_rfft2avx128_fp32_fwd;
        }
        else if (precision == DT_DOUBLE)
        {
            return r2hcf_rfft2avx128_fp64_fwd;
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
            return r2hcf_rfft2avx128_fp32_bwd;
        }
        else if (precision == DT_DOUBLE)
        {
            return r2hcf_rfft2avx128_fp64_bwd;
        }
        else
        {
            return NULL;
        }
    }
}
