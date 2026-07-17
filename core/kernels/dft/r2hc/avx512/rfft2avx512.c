// Copyright Advanced Micro Devices, Inc.
// SPDX-License-Identifier: BSD-3-Clause

/** @file rfft2avx512.c
 *
 *  @brief Radix-2 r2hc Real-FFT kernel with AVX-512 operations using x86 SIMD
 *  intrinsics
 *
 *  This file contains the DIT radix-5 real-to-halfcomplex implementations using
 *  AVX512 SIMD operations for single-precision and double-precision inputs.
 *
 *  @author Jeevanantham N
 */

#include "core/kernels/kernel.h"
#include "core/kernels/simd_includes/r2hc_simd_avx512.h"

static const ops_cycles_t ops_cnt[NUM_PRECISIONS] = {{0, 0, 2, 64, 48, 12},
                                                     {0, 0, 2, 32, 0, 12}};

ops_cycles_t get_ops_cnt_r2hc_rfft2avx512(FFTZ_UINT8 precision,
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
static FFTZ_VOID r2hc_rfft2avx512_fp32(FFTZ_VOID *in_real, FFTZ_VOID *in_imag,
                                       FFTZ_VOID *out_real, FFTZ_VOID *out_imag,
                                       FFTZ_INTP n, aoclfftz_strides_t *strides,
                                       FFTZ_VOID *twd, FFTZ_UINT8 flag)
{
    AOCLFFTZ_LOG(DEBUG, global_logger_mode, "Enter");
    FFTZ_FLOAT *in = (FFTZ_FLOAT *)in_real;
    FFTZ_FLOAT *out = (FFTZ_FLOAT *)out_real;
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
    FFTZ_UINT8 is_contiguous_out = (v_out_stride == 1);

    FFTZ_INTP cnt;
    FFTZ_FLOAT *curr_in, *curr_out;
    FFTZ_INTP N = n / NUM_SETS_REAL_512_S;
    FFTZ_INTP remaining_sets = n % NUM_SETS_REAL_512_S;

    for (cnt = 0; cnt < N; cnt++)
    {
        __m512 v_in0, v_in1;
        __m512 v_out0, v_out1;

        curr_in = in;
        curr_out = out;

        // Input point 1: x(0)
        LDR_512_S(curr_in, v_in_stride, v_in0, is_contiguous_in);
        // Input point 2: x(1)
        curr_in = in + in_strides[1];
        LDR_512_S(curr_in, v_in_stride, v_in1, is_contiguous_in);

        // Output point 1: X(0)
        v_out0 = _mm512_add_ps(v_in0, v_in1);

        // Output point 2: X(1)
        v_out1 = _mm512_sub_ps(v_in0, v_in1);

        STR_512_S(curr_out, v_out_stride, v_out0, is_contiguous_out);
        curr_out = out + out_strides[1];
        STR_512_S(curr_out, v_out_stride, v_out1, is_contiguous_out);

        in  = in + (v_in_stride << 4);
        out = out + (v_out_stride << 4);
    }
    // tail cases
    if (remaining_sets & NUM_SETS_REAL_256_S)
    {
        __m256 v_in0, v_in1;
        __m256 v_out0, v_out1;

        curr_in = in;
        curr_out = out;

        // Input point 1: x(0)
        LDR_256_S(curr_in, v_in_stride, v_in0, is_contiguous_in);
        // Input point 2: x(1)
        curr_in = in + in_strides[1];
        LDR_256_S(curr_in, v_in_stride, v_in1, is_contiguous_in);

        // Output point 1: X(0)
        v_out0 = _mm256_add_ps(v_in0, v_in1);

        // Output point 2: X(1)
        v_out1 = _mm256_sub_ps(v_in0, v_in1);

        STR_256_S(curr_out, v_out_stride, v_out0, is_contiguous_out);
        curr_out = out + out_strides[1];
        STR_256_S(curr_out, v_out_stride, v_out1, is_contiguous_out);

        in  = in + (v_in_stride << 3);
        out = out + (v_out_stride << 3);
    }
    // tail cases
    if (remaining_sets & NUM_SETS_REAL_128_S)
    {
        __m128 v_in0, v_in1;
        __m128 v_out0, v_out1;

        curr_in = in;
        curr_out = out;

        // Input point 1: x(0)
        LDR_128_S(curr_in, v_in_stride, v_in0, is_contiguous_in);
        // Input point 2: x(1)
        curr_in = in + in_strides[1];
        LDR_128_S(curr_in, v_in_stride, v_in1, is_contiguous_in);

        // Output point 1: X(0)
        v_out0 = _mm_add_ps(v_in0, v_in1);

        // Output point 2: X(1)
        v_out1 = _mm_sub_ps(v_in0, v_in1);

        STR_128_S(curr_out, v_out_stride, v_out0, is_contiguous_out);
        curr_out = out + out_strides[1];
        STR_128_S(curr_out, v_out_stride, v_out1, is_contiguous_out);

        in  = in + (v_in_stride << 2);
        out = out + (v_out_stride << 2);
    }
    // tail cases
    if (remaining_sets & 2)
    {
        __m128 v_in0, v_in1;
        __m128 v_out0, v_out1;

        curr_in = in;
        curr_out = out;

        // Input point 1: x(0)
        LDHR_128_S(curr_in, v_in_stride, v_in0);
        // Input point 2: x(1)
        curr_in = in + in_strides[1];
        LDHR_128_S(curr_in, v_in_stride, v_in1);

        // Output point 1: X(0)
        v_out0 = _mm_add_ps(v_in0, v_in1);

        // Output point 2: X(1)
        v_out1 = _mm_sub_ps(v_in0, v_in1);

        STHR_128_S(curr_out, v_out_stride, v_out0);
        curr_out = out + out_strides[1];
        STHR_128_S(curr_out, v_out_stride, v_out1);

        in  = in + (v_in_stride << 1);
        out = out + (v_out_stride << 1);
    }
    // tail cases
    if (remaining_sets & 1)
    {
        FFTZ_FLOAT v_in0, v_in1;
        // Input point 1: x(0)
        v_in0 = *in;
        // Input point 2: x(1)
        v_in1 = in[in_strides[1]];

        // Output point 1: X(0)
        *out = v_in0 + v_in1;

        // Output point 2: X(1)
        out[out_strides[1]] = v_in0 - v_in1;
    }
    AOCLFFTZ_LOG(DEBUG, global_logger_mode, "Exit");
}

FFTZ_VOID r2hc_rfft2avx512_fp64(FFTZ_VOID *in_real, FFTZ_VOID *in_imag,
                                FFTZ_VOID *out_real, FFTZ_VOID *out_imag,
                                FFTZ_INTP n, aoclfftz_strides_t *strides,
                                FFTZ_VOID *twd, FFTZ_UINT8 flag)
{
    AOCLFFTZ_LOG(DEBUG, global_logger_mode, "Enter");
    FFTZ_DOUBLE *in = (FFTZ_DOUBLE *)in_real;
    FFTZ_DOUBLE *out = (FFTZ_DOUBLE *)out_real;
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
    FFTZ_UINT8 is_contiguous_out = (v_out_stride == 1);

    FFTZ_INTP cnt;
    FFTZ_DOUBLE *curr_in, *curr_out;
    FFTZ_INTP N = n / NUM_SETS_REAL_512_D;
    FFTZ_INTP remaining_sets = n % NUM_SETS_REAL_512_D;

    for (cnt = 0; cnt < N; cnt++)
    {
        __m512d v_in0, v_in1;
        __m512d v_out0, v_out1;

        curr_in = in;
        curr_out = out;

        // Input point 1: x(0)
        LDR_512_D(curr_in, v_in_stride, v_in0, is_contiguous_in);
        // Input point 2: x(1)
        curr_in = in + in_strides[1];
        LDR_512_D(curr_in, v_in_stride, v_in1, is_contiguous_in);

        // Output point 1: X(0)
        v_out0 = _mm512_add_pd(v_in0, v_in1);

        // Output point 2: X(1)
        v_out1 = _mm512_sub_pd(v_in0, v_in1);

        STR_512_D(curr_out, v_out_stride, v_out0, is_contiguous_out);
        curr_out = out + out_strides[1];
        STR_512_D(curr_out, v_out_stride, v_out1, is_contiguous_out);

        in  = in + (v_in_stride << 3);
        out = out + (v_out_stride << 3);
    }
    // tail cases
    if (remaining_sets & NUM_SETS_REAL_256_D)
    {
        __m256d v_in0, v_in1;
        __m256d v_out0, v_out1;

        curr_in = in;
        curr_out = out;

        // Input point 1: x(0)
        LDR_256_D(curr_in, v_in_stride, v_in0, is_contiguous_in);
        // Input point 2: x(1)
        curr_in = in + in_strides[1];
        LDR_256_D(curr_in, v_in_stride, v_in1, is_contiguous_in);

        // Output point 1: X(0)
        v_out0 = _mm256_add_pd(v_in0, v_in1);

        // Output point 2: X(1)
        v_out1 = _mm256_sub_pd(v_in0, v_in1);

        STR_256_D(curr_out, v_out_stride, v_out0, is_contiguous_out);
        curr_out = out + out_strides[1];
        STR_256_D(curr_out, v_out_stride, v_out1, is_contiguous_out);

        in  = in + (v_in_stride << 2);
        out = out + (v_out_stride << 2);
    }
    // tail cases
    if (remaining_sets & 2)
    {
        __m128d v_in0, v_in1;
        __m128d v_out0, v_out1;

        curr_in = in;
        curr_out = out;

        // Input point 1: x(0)
        LDR_128_D(curr_in, v_in_stride, v_in0, is_contiguous_in);
        // Input point 2: x(1)
        curr_in = in + in_strides[1];
        LDR_128_D(curr_in, v_in_stride, v_in1, is_contiguous_in);

        // Output point 1: X(0)
        v_out0 = _mm_add_pd(v_in0, v_in1);

        // Output point 2: X(1)
        v_out1 = _mm_sub_pd(v_in0, v_in1);

        STR_128_D(curr_out, v_out_stride, v_out0, is_contiguous_out);
        curr_out = out + out_strides[1];
        STR_128_D(curr_out, v_out_stride, v_out1, is_contiguous_out);

        in  = in + (v_in_stride << 1);
        out = out + (v_out_stride << 1);
    }
    // tail cases
    if (remaining_sets & 1)
    {
        FFTZ_DOUBLE v_in0, v_in1;
        // Input point 1: x(0)
        v_in0 = *in;
        // Input point 2: x(1)
        v_in1 = in[in_strides[1]];

        // Output point 1: X(0)
        *out = v_in0 + v_in1;

        // Output point 2: X(0)
        out[out_strides[1]] = v_in0 - v_in1;
    }
    AOCLFFTZ_LOG(DEBUG, global_logger_mode, "Exit");
}

kfft_ register_kernel_r2hc_rfft2avx512(FFTZ_UINT8 precision,
                                       FFTZ_UINT8 direction)
{

    if (precision == DT_FLOAT)
    {
        return r2hc_rfft2avx512_fp32;
    }
    else if (precision == DT_DOUBLE)
    {
        return r2hc_rfft2avx512_fp64;
    }
    else
    {
        return NULL;
    }
}
