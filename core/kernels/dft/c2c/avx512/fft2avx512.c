// Copyright Advanced Micro Devices, Inc.
// SPDX-License-Identifier: BSD-3-Clause

/** @file fft2avx512.c
 *
 *  @brief Radix-2 FFT kernel with AVX-512 operations using x86 SIMD intrinsics
 *
 *  This file contains the DIT radix-2 FFT implementations using AVX-512
 *  operations for single-precision and double-precision inputs.
 *
 *  @author Jeya R
 *
 */

#include "core/kernels/kernel.h"
#include "core/kernels/simd_includes/simd_common_avx512.h"

static const ops_cycles_t ops_cnt[NUM_PRECISIONS] = {{0, 0, 2, 32, 6, 12},
                                                     {0, 0, 2, 16, 0, 12}};

ops_cycles_t get_ops_cnt_fft2avx512(FFTZ_UINT8 precision, FFTZ_UINT8 direction)
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

static FFTZ_VOID fft2avx512fp32(FFTZ_VOID *in_real, FFTZ_VOID *in_imag,
                                FFTZ_VOID *out_real, FFTZ_VOID *out_imag,
                                FFTZ_INTP n, aoclfftz_strides_t *strides,
                                FFTZ_VOID *twd, FFTZ_UINT8 flag)
{
    AOCLFFTZ_LOG(DEBUG, global_logger_mode, "Enter");
    FFTZ_FLOAT *in_r = in_real;
    FFTZ_FLOAT *out_r = out_real;
    FFTZ_FLOAT *curr_in, *curr_out;
#ifdef VOLATILE_STRIDE_ARRAY
    volatile FFTZ_INTP *in_strides = strides->in_strides;
    volatile FFTZ_INTP *out_strides = strides->out_strides;
#else
    FFTZ_INTP *in_strides = strides->in_strides;
    FFTZ_INTP *out_strides = strides->out_strides;
#endif
    FFTZ_INTP v_in_stride = strides->v_in_stride;
    FFTZ_UINT8 is_contiguous_in = (v_in_stride == DATA_STRIDE);
    FFTZ_INTP v_out_stride = strides->v_out_stride;
    FFTZ_UINT8 is_contiguous_out = (v_out_stride == DATA_STRIDE);
    FFTZ_INTP N = n / NUM_SETS_512_S;
    FFTZ_INTP remaining_sets = n % NUM_SETS_512_S;
    FFTZ_INTP count;

    for (count = 0; count < N; count++)
    {
        __m512 v_in0, v_in1;
        __m512 v_out0, v_out1;

        curr_in = in_r;
        GATHER8_512_S(curr_in, v_in_stride, v_in0, is_contiguous_in);
        curr_in = in_r + in_strides[1];
        GATHER8_512_S(curr_in, v_in_stride, v_in1, is_contiguous_in);

        // Output point 1: X[0]
        v_out0 = _mm512_add_ps(v_in0, v_in1);
        // Output point 2: X[1]
        v_out1 = _mm512_sub_ps(v_in0, v_in1);

        curr_out = out_r;
        if (out_strides[1] == DATA_STRIDE)
        {
            TRANSPOSE_ST2_512_S(curr_out, v_out_stride, v_out0, v_out1);
        }
        else
        {
            SCATTER8_512_S(curr_out, v_out_stride, v_out0, is_contiguous_out);
            curr_out = out_r + out_strides[1];
            SCATTER8_512_S(curr_out, v_out_stride, v_out1, is_contiguous_out);
        }

        in_r += NUM_SETS_512_S * v_in_stride;
        out_r += NUM_SETS_512_S * v_out_stride;
    }
    // tail cases
    if (remaining_sets >= NUM_SETS_256_S)
    {
        __m256 v_in0, v_in1;
        __m256 v_out0, v_out1;
        curr_in = in_r;
        GATHER4_256_S(curr_in, v_in_stride, v_in0, is_contiguous_in);
        curr_in = in_r + in_strides[1];
        GATHER4_256_S(curr_in, v_in_stride, v_in1, is_contiguous_in);

        // Output point 1: X[0]
        v_out0 = _mm256_add_ps(v_in0, v_in1);
        // Output point 2: X[1]
        v_out1 = _mm256_sub_ps(v_in0, v_in1);

        curr_out = out_r;
        if (out_strides[1] == DATA_STRIDE)
        {
            TRANSPOSE_ST2_256_S(curr_out, v_out_stride, v_out0, v_out1);
        }
        else
        {
            SCATTER4_256_S(curr_out, v_out_stride, v_out0, is_contiguous_out);
            curr_out = out_r + out_strides[1];
            SCATTER4_256_S(curr_out, v_out_stride, v_out1, is_contiguous_out);
        }

        in_r += NUM_SETS_256_S * v_in_stride;
        out_r += NUM_SETS_256_S * v_out_stride;
        remaining_sets = remaining_sets - NUM_SETS_256_S;
    }
    // tail case
    if (remaining_sets >= NUM_SETS_128_S)
    {
        __m128 v_in0, v_in1;
        __m128 v_out0, v_out1;
        curr_in = in_r;
        GATHER2_128_S(curr_in, v_in_stride, v_in0, is_contiguous_in);
        curr_in = in_r + in_strides[1];
        GATHER2_128_S(curr_in, v_in_stride, v_in1, is_contiguous_in);

        // Output point 1: X[0]
        v_out0 = _mm_add_ps(v_in0, v_in1);
        // Output point 2: X[1]
        v_out1 = _mm_sub_ps(v_in0, v_in1);

        curr_out = out_r;
        if (out_strides[1] == DATA_STRIDE)
        {
            TRANSPOSE_ST2_128_S(curr_out, v_out_stride, v_out0, v_out1);
        }
        else
        {
            SCATTER2_128_S(curr_out, v_out_stride, v_out0, is_contiguous_out);
            curr_out = out_r + out_strides[1];
            SCATTER2_128_S(curr_out, v_out_stride, v_out1, is_contiguous_out);
        }

        in_r = in_r + (v_in_stride << 1);
        out_r = out_r + (v_out_stride << 1);
    }
    if (remaining_sets & 1)
    {
        __m128 v_in0, v_in1;
        __m128 v_out0, v_out1;

        curr_in = in_r;
        LD_LOW_128_S(curr_in, v_in0);
        curr_in = in_r + in_strides[1];
        LD_LOW_128_S(curr_in, v_in1);

        // Output point 1: X[0]
        v_out0 = _mm_add_ps(v_in0, v_in1);
        // Output point 2: X[1]
        v_out1 = _mm_sub_ps(v_in0, v_in1);

        curr_out = out_r;
        if (out_strides[1] == DATA_STRIDE)
        {
            ST_128_S(curr_out, v_out0, v_out1);
        }
        else
        {
            ST_LOW_128_S(curr_out, v_out0);
            curr_out = out_r + out_strides[1];
            ST_LOW_128_S(curr_out, v_out1);
        }
    }
    AOCLFFTZ_LOG(DEBUG, global_logger_mode, "Exit");
}

static FFTZ_VOID fft2avx512fp64(FFTZ_VOID *in_real, FFTZ_VOID *in_imag,
                                FFTZ_VOID *out_real, FFTZ_VOID *out_imag,
                                FFTZ_INTP n, aoclfftz_strides_t *strides,
                                FFTZ_VOID *twd, FFTZ_UINT8 flag)
{
    AOCLFFTZ_LOG(DEBUG, global_logger_mode, "Enter");
    FFTZ_DOUBLE *in_r = in_real;
    FFTZ_DOUBLE *out_r = out_real;
    FFTZ_DOUBLE *curr_in, *curr_out;
#ifdef VOLATILE_STRIDE_ARRAY
    volatile FFTZ_INTP *in_strides = strides->in_strides;
    volatile FFTZ_INTP *out_strides = strides->out_strides;
#else
    FFTZ_INTP *in_strides = strides->in_strides;
    FFTZ_INTP *out_strides = strides->out_strides;
#endif
    FFTZ_INTP v_in_stride = strides->v_in_stride;
    FFTZ_UINT8 is_contiguous_in = (v_in_stride == DATA_STRIDE);
    FFTZ_INTP v_out_stride = strides->v_out_stride;
    FFTZ_UINT8 is_contiguous_out = (v_out_stride == DATA_STRIDE);
    FFTZ_INTP N = n / NUM_SETS_512_D;
    FFTZ_INTP remaining_sets = n % NUM_SETS_512_D;
    FFTZ_INTP count;

    for (count = 0; count < N; count++)
    {
        __m512d v_in0, v_in1;
        __m512d v_out0, v_out1;

        curr_in = in_r;
        GATHER4_512_D(curr_in, v_in_stride, v_in0, is_contiguous_in);
        curr_in = in_r + in_strides[1];
        GATHER4_512_D(curr_in, v_in_stride, v_in1, is_contiguous_in);

        // Output point 1: X[0]
        v_out0 = _mm512_add_pd(v_in0, v_in1);
        // Output point 2: X[1]
        v_out1 = _mm512_sub_pd(v_in0, v_in1);

        curr_out = out_r;
        if (out_strides[1] == DATA_STRIDE)
        {
            TRANSPOSE_ST2_512_D(curr_out, v_out_stride, v_out0, v_out1);
        }
        else
        {
            SCATTER4_512_D(curr_out, v_out_stride, v_out0, is_contiguous_out);
            curr_out = out_r + out_strides[1];
            SCATTER4_512_D(curr_out, v_out_stride, v_out1, is_contiguous_out);
        }

        in_r += NUM_SETS_512_D * v_in_stride;
        out_r += NUM_SETS_512_D * v_out_stride;
    }
    // tail cases
    if (remaining_sets >= NUM_SETS_256_D)
    {
        __m256d v_in0, v_in1;
        __m256d v_out0, v_out1;
        curr_in = in_r;
        GATHER2_256_D(curr_in, v_in_stride, v_in0, is_contiguous_in);
        curr_in = in_r + in_strides[1];
        GATHER2_256_D(curr_in, v_in_stride, v_in1, is_contiguous_in);

        // Output point 1: X[0]
        v_out0 = _mm256_add_pd(v_in0, v_in1);
        // Output point 2: X[1]
        v_out1 = _mm256_sub_pd(v_in0, v_in1);

        curr_out = out_r;
        if (out_strides[1] == DATA_STRIDE)
        {
            TRANSPOSE_ST2_256_D(curr_out, v_out_stride, v_out0, v_out1);
        }
        else
        {
            SCATTER2_256_D(curr_out, v_out_stride, v_out0, is_contiguous_out);
            curr_out = out_r + out_strides[1];
            SCATTER2_256_D(curr_out, v_out_stride, v_out1, is_contiguous_out);
        }

        in_r += NUM_SETS_256_D * v_in_stride;
        out_r += NUM_SETS_256_D * v_out_stride;
    }
    // tail case
    if (remaining_sets & 1)
    {
        __m128d v_in0, v_in1;
        __m128d v_out0, v_out1;
        curr_in = in_r;
        LD_128_D(curr_in, v_in0);
        curr_in = in_r + in_strides[1];
        LD_128_D(curr_in, v_in1);

        // Output point 1: X[0]
        v_out0 = _mm_add_pd(v_in0, v_in1);
        // Output point 2: X[1]
        v_out1 = _mm_sub_pd(v_in0, v_in1);

        curr_out = out_r;
        ST_128_D(curr_out, v_out0);
        curr_out = out_r + out_strides[1];
        ST_128_D(curr_out, v_out1);
    }
    AOCLFFTZ_LOG(DEBUG, global_logger_mode, "Exit");
}

kfft_ register_kernel_fft2avx512(FFTZ_UINT8 precision,
                                 FFTZ_UINT8 direction /* unused */)
{
    if (precision == DT_FLOAT)
    {
        return fft2avx512fp32;
    }
    else if (precision == DT_DOUBLE)
    {
        return fft2avx512fp64;
    }
    else
    {
        return NULL;
    }
}
