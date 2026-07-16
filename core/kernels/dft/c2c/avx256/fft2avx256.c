// Copyright Advanced Micro Devices, Inc.
// SPDX-License-Identifier: BSD-3-Clause

/** @file fft2avx256.c
 *
 *  @brief Radix-2 FFT kernel with AVX-256 operations using x86 SIMD intrinsics
 *
 *  This file contains the DIT radix-2 FFT implementations using AVX-256
 *  operations for single-precision and double-precision inputs.
 *
 *  @author Jeya R
 *  @author S. Biplab Raut
 *
 */

#include "core/kernels/kernel.h"
#include "core/kernels/simd_includes/simd_common.h"

static const ops_cycles_t ops_cnt[NUM_PRECISIONS] = {{0, 0, 2, 16, 2, 8},
                                                     {0, 0, 2,  8, 0, 8}};

ops_cycles_t get_ops_cnt_fft2avx256(UINT8 precision, UINT8 direction)
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

static VOID fft2avx256fp32(VOID *in_real, VOID *in_imag, VOID *out_real,
                           VOID *out_imag, INTP n, aoclfftz_strides_t *strides,
                           VOID *twd, UINT8 flag)
{
    AOCLFFTZ_LOG(DEBUG, global_logger_mode, "Enter");
    FLOAT *in_r = (FLOAT *)in_real;
    FLOAT *out_r = (FLOAT *)out_real;
    FLOAT *curr_in, *curr_out;
#ifdef VOLATILE_STRIDE_ARRAY
    volatile INTP *in_strides = strides->in_strides;
    volatile INTP *out_strides = strides->out_strides;
#else
    INTP *in_strides = strides->in_strides;
    INTP *out_strides = strides->out_strides;
#endif
    INTP v_in_stride = strides->v_in_stride;
    UINT8 is_contiguous_in = (v_in_stride == DATA_STRIDE);
    INTP v_out_stride = strides->v_out_stride;
    UINT8 is_contiguous_out = (v_out_stride == DATA_STRIDE);
    INTP N = n / NUM_SETS_256_S;
    INTP remaining_sets = n % NUM_SETS_256_S;
    INTP count;

    __m256 _in0, _in1;
    __m256 _out0, _out1;

    for (count = 0; count < N; count++)
    {
        curr_in = in_r;
        GATHER4_256_S(curr_in, v_in_stride, _in0, is_contiguous_in);
        curr_in = in_r + in_strides[1];
        GATHER4_256_S(curr_in, v_in_stride, _in1, is_contiguous_in);

        // Output point 1: X[0]
        _out0 = _mm256_add_ps(_in0, _in1);
        // Output point 2: X[1]
        _out1 = _mm256_sub_ps(_in0, _in1);

        curr_out = out_r;
        if (out_strides[1] == DATA_STRIDE)
        {
            TRANSPOSE_ST2_256_S(curr_out, v_out_stride, _out0, _out1);
        }
        else
        {
            SCATTER4_256_S(curr_out, v_out_stride, _out0, is_contiguous_out);
            curr_out = out_r + out_strides[1];
            SCATTER4_256_S(curr_out, v_out_stride, _out1, is_contiguous_out);
        }
        in_r += NUM_SETS_256_S * v_in_stride;
        out_r += NUM_SETS_256_S * v_out_stride;
    }
    // tail cases
    if (remaining_sets >= 2)
    {
        __m128 _in0, _in1;
        __m128 _out0, _out1;
        curr_in = in_r;
        GATHER2_128_S(curr_in, v_in_stride, _in0, is_contiguous_in);
        curr_in = in_r + in_strides[1];
        GATHER2_128_S(curr_in, v_in_stride, _in1, is_contiguous_in);

        // Output point 1: X[0]
        _out0 = _mm_add_ps(_in0, _in1);
        // Output point 2: X[1]
        _out1 = _mm_sub_ps(_in0, _in1);

        curr_out = out_r;
        if (out_strides[1] == DATA_STRIDE)
        {
            TRANSPOSE_ST2_128_S(curr_out, v_out_stride, _out0, _out1);
        }
        else
        {
            SCATTER2_128_S(curr_out, v_out_stride, _out0, is_contiguous_out);
            curr_out = out_r + out_strides[1];
            SCATTER2_128_S(curr_out, v_out_stride, _out1, is_contiguous_out);
        }

        in_r = in_r + (v_in_stride << 1);
        out_r = out_r + (v_out_stride << 1);
    }
    if (remaining_sets & 1)
    {
        __m128 _in0, _in1;
        __m128 _out0, _out1;

        curr_in = in_r;
        LD_LOW_128_S(curr_in, _in0);
        curr_in = in_r + in_strides[1];
        LD_LOW_128_S(curr_in, _in1);

        // Output point 1: X[0]
        _out0 = _mm_add_ps(_in0, _in1);
        // Output point 2: X[1]
        _out1 = _mm_sub_ps(_in0, _in1);

        curr_out = out_r;
        if (out_strides[1] == DATA_STRIDE)
        {
            ST_128_S(curr_out, _out0, _out1);
        }
        else
        {
            ST_LOW_128_S(curr_out, _out0);
            curr_out = out_r + out_strides[1];
            ST_LOW_128_S(curr_out, _out1);
        }
    }
    AOCLFFTZ_LOG(DEBUG, global_logger_mode, "Exit");
}

static VOID fft2avx256fp64(VOID *in_real, VOID *in_imag, VOID *out_real,
                           VOID *out_imag, INTP n, aoclfftz_strides_t *strides,
                           VOID *twd, UINT8 flag)
{
    AOCLFFTZ_LOG(DEBUG, global_logger_mode, "Enter");
    DOUBLE *in_r = (DOUBLE *)in_real;
    DOUBLE *out_r = (DOUBLE *)out_real;
    DOUBLE *curr_in, *curr_out;
#ifdef VOLATILE_STRIDE_ARRAY
    volatile INTP *in_strides = strides->in_strides;
    volatile INTP *out_strides = strides->out_strides;
#else
    INTP *in_strides = strides->in_strides;
    INTP *out_strides = strides->out_strides;
#endif
    INTP v_in_stride = strides->v_in_stride;
    UINT8 is_contiguous_in = (v_in_stride == DATA_STRIDE);
    INTP v_out_stride = strides->v_out_stride;
    UINT8 is_contiguous_out = (v_out_stride == DATA_STRIDE);
    INTP N = n / NUM_SETS_256_D;
    INTP count;

    __m256d _in0, _in1;
    __m256d _out0, _out1;

    for (count = 0; count < N; count++)
    {
        curr_in = in_r;
        GATHER2_256_D(curr_in, v_in_stride, _in0, is_contiguous_in);
        curr_in = in_r + in_strides[1];
        GATHER2_256_D(curr_in, v_in_stride, _in1, is_contiguous_in);

        // Output point 1: X[0]
        _out0 = _mm256_add_pd(_in0, _in1);
        // Output point 2: X[1]
        _out1 = _mm256_sub_pd(_in0, _in1);

        curr_out = out_r;
        if (out_strides[1] == DATA_STRIDE)
        {
            TRANSPOSE_ST2_256_D(curr_out, v_out_stride, _out0, _out1);
        }
        else
        {
            SCATTER2_256_D(curr_out, v_out_stride, _out0, is_contiguous_out);
            curr_out = out_r + out_strides[1];
            SCATTER2_256_D(curr_out, v_out_stride, _out1, is_contiguous_out);
        }

        in_r += NUM_SETS_256_D * v_in_stride;
        out_r += NUM_SETS_256_D * v_out_stride;
    }
    // tail case
    if (n & 1)
    {
        __m128d _in0, _in1;
        __m128d _out0, _out1;
        curr_in = in_r;
        LD_128_D(curr_in, _in0);
        curr_in = in_r + in_strides[1];
        LD_128_D(curr_in, _in1);

        // Output point 1: X[0]
        _out0 = _mm_add_pd(_in0, _in1);
        // Output point 2: X[1]
        _out1 = _mm_sub_pd(_in0, _in1);

        curr_out = out_r;
        ST_128_D(curr_out, _out0);
        curr_out = out_r + out_strides[1];
        ST_128_D(curr_out, _out1);
    }
    AOCLFFTZ_LOG(DEBUG, global_logger_mode, "Exit");
}

kfft_ register_kernel_fft2avx256(UINT8 precision, UINT8 direction /* unused */)
{
    if (precision == DT_FLOAT)
    {
        return fft2avx256fp32;
    }
    else if (precision == DT_DOUBLE)
    {
        return fft2avx256fp64;
    }
    else
    {
        return NULL;
    }
}
