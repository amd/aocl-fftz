// Copyright Advanced Micro Devices, Inc.
// SPDX-License-Identifier: BSD-3-Clause

/** @file rfft2avx128.c
 *
 *  @brief Radix-2 r2hc Real-FFT kernel with AVX-128 operations using x86 SIMD
 *  intrinsics
 *
 *  This file contains the DIT radix-2 real-to-halfcomplex implementations using
 *  AVX128 SIMD operations for single-precision and double-precision inputs.
 *
 *  @author Jeevanantham N
 */

#include "core/kernels/kernel.h"
#include "core/kernels/simd_includes/r2hc_simd_avx128.h"

static const ops_cycles_t ops_cnt[NUM_PRECISIONS] = {{0, 0, 2, 16, 12, 0},
                                                     {0, 0, 2, 8, 0, 0}};

ops_cycles_t get_ops_cnt_r2hc_rfft2avx128(FFTZ_UINT8 precision,
                                          FFTZ_UINT8 direction)
{
    if (precision == DT_FLOAT)
    {
        return ops_cnt[0];
    }
    else
    {
        return ops_cnt[1];
    }
}

static FFTZ_VOID r2hc_rfft2avx128_fp32(FFTZ_VOID *in_real,
                                       FFTZ_VOID *in_complex,
                                       FFTZ_VOID *out_real,
                                       FFTZ_VOID *out_complex, FFTZ_INTP n,
                                       aoclfftz_strides_t *strides,
                                       FFTZ_VOID *twd, FFTZ_UINT8 flag)
{
    AOCLFFTZ_LOG(DEBUG, global_logger_mode, "Enter");
    FFTZ_FLOAT *in_r = (FFTZ_FLOAT *)in_real;
    FFTZ_FLOAT *out_r = (FFTZ_FLOAT *)out_real;
#ifdef VOLATILE_STRIDE_ARRAY
    volatile FFTZ_INTP *in_strides = strides->in_strides;
    volatile FFTZ_INTP *out_strides = strides->out_strides;
#else
    FFTZ_INTP *in_strides = strides->in_strides;
    FFTZ_INTP *out_strides = strides->out_strides;
#endif
    // Used for endpoints: DC and Nyquist.
    FFTZ_INTP v_in_dc_nyq_stride = strides->v_in_sym_stride;
    FFTZ_UINT8 is_contiguous_in_dc_nyq = (v_in_dc_nyq_stride == 1);
    FFTZ_INTP v_out_dc_nyq_stride = strides->v_out_sym_stride;
    FFTZ_UINT8 is_contiguous_out_dc_nyq = (v_out_dc_nyq_stride == 1);

    FFTZ_INTP cnt;
    FFTZ_FLOAT *curr_in, *curr_out;
    FFTZ_INTP N = n / NUM_SETS_REAL_128_S;

    for (cnt = 0; cnt < N; cnt++)
    {
        __m128 v_in0, v_in1;
        __m128 v_out0, v_out1;

        curr_in = in_r;
        curr_out = out_r;

        // Input point 1: x(0)
        LDR_128_S(curr_in, v_in_dc_nyq_stride, v_in0, is_contiguous_in_dc_nyq);
        // Input point 2: x(1)
        curr_in = in_r + in_strides[1];
        LDR_128_S(curr_in, v_in_dc_nyq_stride, v_in1, is_contiguous_in_dc_nyq);

        // Output point 1: X(0)
        v_out0 = _mm_add_ps(v_in0, v_in1);

        // Output point 2: X(0)
        v_out1 = _mm_sub_ps(v_in0, v_in1);
        STR_128_S(curr_out, v_out_dc_nyq_stride, v_out0, is_contiguous_out_dc_nyq);
        curr_out = out_r + out_strides[1];
        STR_128_S(curr_out, v_out_dc_nyq_stride, v_out1, is_contiguous_out_dc_nyq);

        in_r = in_r + (v_in_dc_nyq_stride << 2);
        out_r = out_r + (v_out_dc_nyq_stride << 2);
    }
    // tail cases
    if (n & 2)
    {
        __m128 v_in0, v_in1;
        __m128 v_out0, v_out1;

        curr_in = in_r;
        curr_out = out_r;

        // Input point 1: x(0)
        LDHR_128_S(curr_in, v_in_dc_nyq_stride, v_in0);
        // Input point 2: x(1)
        curr_in = in_r + in_strides[1];
        LDHR_128_S(curr_in, v_in_dc_nyq_stride, v_in1);

        // Output point 1: X(0)
        v_out0 = _mm_add_ps(v_in0, v_in1);

        // Output point 2: X(0)
        v_out1 = _mm_sub_ps(v_in0, v_in1);

        STHR_128_S(curr_out, v_out_dc_nyq_stride, v_out0);
        curr_out = out_r + out_strides[1];
        STHR_128_S(curr_out, v_out_dc_nyq_stride, v_out1);

        in_r = in_r + (v_in_dc_nyq_stride << 1);
        out_r = out_r + (v_out_dc_nyq_stride << 1);
    }
    // tail cases
    if (n & 1)
    {
        FFTZ_FLOAT v_in0, v_in1;

        curr_in = in_r;
        curr_out = out_r;
        // Input point 1: x(0)
        v_in0 = *in_r;
        // Input point 2: x(1)
        v_in1 = in_r[in_strides[1]];

        // Output point 1: X(0)
        *out_r = v_in0 + v_in1;

        // Output point 2: X(0)
        out_r[out_strides[1]] = v_in0 - v_in1;
    }
    AOCLFFTZ_LOG(DEBUG, global_logger_mode, "Exit");
}

static FFTZ_VOID r2hc_rfft2avx128_fp64(FFTZ_VOID *in_real,
                                       FFTZ_VOID *in_complex,
                                       FFTZ_VOID *out_real,
                                       FFTZ_VOID *out_complex, FFTZ_INTP n,
                                       aoclfftz_strides_t *strides,
                                       FFTZ_VOID *twd, FFTZ_UINT8 flag)
{
    AOCLFFTZ_LOG(DEBUG, global_logger_mode, "Enter");
    FFTZ_DOUBLE *in_r = (FFTZ_DOUBLE *)in_real;
    FFTZ_DOUBLE *out_r = (FFTZ_DOUBLE *)out_real;
#ifdef VOLATILE_STRIDE_ARRAY
    volatile FFTZ_INTP *in_strides = strides->in_strides;
    volatile FFTZ_INTP *out_strides = strides->out_strides;
#else
    FFTZ_INTP *in_strides = strides->in_strides;
    FFTZ_INTP *out_strides = strides->out_strides;
#endif
    // Used for endpoints: DC and Nyquist.
    FFTZ_INTP v_in_dc_nyq_stride = strides->v_in_sym_stride;
    FFTZ_UINT8 is_contiguous_in_dc_nyq = (v_in_dc_nyq_stride == 1);
    FFTZ_INTP v_out_dc_nyq_stride = strides->v_out_sym_stride;
    FFTZ_UINT8 is_contiguous_out_dc_nyq = (v_out_dc_nyq_stride == 1);

    FFTZ_INTP cnt;
    FFTZ_DOUBLE *curr_in, *curr_out;
    FFTZ_INTP N = n / NUM_SETS_REAL_128_D;

    for (cnt = 0; cnt < N; cnt++)
    {
        __m128d v_in0, v_in1;
        __m128d v_out0, v_out1;

        curr_in = in_r;
        curr_out = out_r;

        // Input point 1: x(0)
        LDR_128_D(curr_in, v_in_dc_nyq_stride, v_in0, is_contiguous_in_dc_nyq);
        // Input point 2: x(1)
        curr_in = in_r + in_strides[1];
        LDR_128_D(curr_in, v_in_dc_nyq_stride, v_in1, is_contiguous_in_dc_nyq);

        // Output point 1: X(0)
        v_out0 = _mm_add_pd(v_in0, v_in1);

        // Output point 2: X(0)
        v_out1 = _mm_sub_pd(v_in0, v_in1);
        STR_128_D(curr_out, v_out_dc_nyq_stride, v_out0, is_contiguous_out_dc_nyq);
        curr_out = out_r + out_strides[1];
        STR_128_D(curr_out, v_out_dc_nyq_stride, v_out1, is_contiguous_out_dc_nyq);

        in_r = in_r + (v_in_dc_nyq_stride << 1);
        out_r = out_r + (v_out_dc_nyq_stride << 1);
    }
    // tail cases
    if (n & 1)
    {
        FFTZ_DOUBLE v_in0, v_in1;

        curr_in = in_r;
        curr_out = out_r;
        // Input point 1: x(0)
        v_in0 = *in_r;
        // Input point 2: x(1)
        v_in1 = in_r[in_strides[1]];

        // Output point 1: X(0)
        *out_r = v_in0 + v_in1;

        // Output point 2: X(0)
        out_r[out_strides[1]] = v_in0 - v_in1;
    }
    AOCLFFTZ_LOG(DEBUG, global_logger_mode, "Exit");
}

kfft_ register_kernel_r2hc_rfft2avx128(FFTZ_UINT8 precision,
                                       FFTZ_UINT8 direction)
{

    if (precision == DT_FLOAT)
    {
        return r2hc_rfft2avx128_fp32;
    }
    else if (precision == DT_DOUBLE)
    {
        return r2hc_rfft2avx128_fp64;
    }
    else
    {
        return NULL;
    }
}
