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

/** @file twid_fft2.h
 *
 *  @brief The ISA generic kernel template for the radix 2 twiddle kernel
 *
 *  This file contains the DIT twiddle radix-2 FFT implementations for
 *  single-precision and double-precision inputs.
 *
 *  @author Ashwin K. Godbole
 */

// This header has no include guards.
// This is intentional.
// The functions defined in this file are not usable by default.
// They are "instantiated" only when "included" in another file.

#include "core/kernels/simd_includes/generic_kernels_common.h"

static VOID TWID_KNAME_FP32(VOID *in_real, VOID *in_imag, VOID *out_real,
                            VOID *out_imag, INTP n, aoclfftz_strides_t *strides,
                            VOID *twd, UINT8 flag)
{
#ifdef AOCL_ENABLE_LOG
    AOCLFFTZ_LOG_UNFORMATTED(TRACE, TRACE, "Enter");
#endif
    FLOAT *in_r = in_real;
    FLOAT *out_r = out_real;
    FLOAT *curr_in, *curr_out;
#ifdef VOLATILE_STRIDE_ARRAY
    volatile INTP *in_strides = strides->in_strides;
    volatile INTP *out_strides = strides->out_strides;
#else
    INTP *in_strides = strides->in_strides;
    INTP *out_strides = strides->out_strides;
#endif
    INTP v_in_stride = strides->v_in_stride;
    INTP v_out_stride = strides->v_out_stride;

    aoclfftz_twiddle_t *tws = (aoclfftz_twiddle_t *)twd;
    FLOAT *tw = (FLOAT *)(tws->TW);
    UINTP cols = tws->cols;

    INTP N = n / NUM_SETS_S;
    INTP remaining_sets = n % NUM_SETS_S;

#if defined(KERNEL_USE_AVX512)
    INTP do_256_whole = (INTP)(remaining_sets >= NUM_SETS_256_S);
    INTP do_128_whole =
        (INTP)(remaining_sets % NUM_SETS_256_S >= NUM_SETS_128_S);
    INTP cnt_256 = N * NUM_SETS_512_S;
    INTP cnt_128 = N * NUM_SETS_512_S + do_256_whole * NUM_SETS_256_S;
    INTP cnt_128_low = N * NUM_SETS_512_S + do_256_whole * NUM_SETS_256_S +
                       do_128_whole * NUM_SETS_128_S;
#elif defined(KERNEL_USE_AVX256)
    INTP do_128_whole = (INTP)(remaining_sets >= NUM_SETS_128_S);
    INTP cnt_128 = N * NUM_SETS_256_S;
    INTP cnt_128_low = N * NUM_SETS_256_S + do_128_whole * NUM_SETS_128_S;
#elif defined(KERNEL_USE_AVX128)
    INTP cnt_128_low = N * NUM_SETS_128_S;
#endif

    INTP count;

    if (flag)
    {
        in_r = in_imag;
        out_r = out_imag;
    }

    for (count = 0; count < N; count++)
    {
        VREGTYPE_S v_in0, v_in1;
        VREGTYPE_S v_out0, v_out1;

        curr_in = in_r;
        curr_out = out_r;

        if (flag)
        {
            ITW_GATHER_S(curr_in, in_strides, 1, v_in_stride, v_in1, tw, cols,
                         (count * NUM_SETS_S));
        }
        else
        {
            TW_GATHER_S(curr_in, in_strides, 1, v_in_stride, v_in1, tw, cols,
                        (count * NUM_SETS_S));
        }

        GATHER_S(curr_in, v_in_stride, v_in0);

        // Output point 1: X[0]
        v_out0 = ADD_S(v_in0, v_in1);
        // Output point 2: X[1]
        v_out1 = SUB_S(v_in0, v_in1);

        SCATTER_S(curr_out, v_out_stride, v_out0);
        SCATTER_S(curr_out + out_strides[1], v_out_stride, v_out1);

        in_r += NUM_SETS_S * v_in_stride;
        out_r += NUM_SETS_S * v_out_stride;
    }

    // The following contains code that performs the FFT on the tail cases.
    // These tails are conditionally "instantiated" based on the ISA requested
    // by the "includee" file.

#if defined(KERNEL_USE_AVX512)
    if (remaining_sets >= NUM_SETS_256_S)
    {
        __m256 v_in0, v_in1;
        __m256 v_out0, v_out1;

        curr_in = in_r;
        curr_out = out_r;

        if (flag)
        {
            ITW_GATHER_256_S(curr_in, in_strides, 1, v_in_stride, v_in1, tw,
                             cols, cnt_256);
        }
        else
        {
            TW_GATHER_256_S(curr_in, in_strides, 1, v_in_stride, v_in1, tw,
                            cols, cnt_256);
        }

        GATHER4_256_S(curr_in, v_in_stride, v_in0);

        // Output point 1: X[0]
        v_out0 = _mm256_add_ps(v_in0, v_in1);
        // Output point 2: X[1]
        v_out1 = _mm256_sub_ps(v_in0, v_in1);

        SCATTER4_256_S(curr_out, v_out_stride, v_out0);
        SCATTER4_256_S(curr_out + out_strides[1], v_out_stride, v_out1);

        in_r += NUM_SETS_256_S * v_in_stride;
        out_r += NUM_SETS_256_S * v_out_stride;
        remaining_sets = remaining_sets - NUM_SETS_256_S;
    }
#endif

#if defined(KERNEL_USE_AVX512) || defined(KERNEL_USE_AVX256)
    if (remaining_sets >= NUM_SETS_128_S)
    {
        __m128 v_in0, v_in1;
        __m128 v_out0, v_out1;

        curr_in = in_r;
        curr_out = out_r;

        if (flag)
        {
            ITW_GATHER_128_S(curr_in, in_strides, 1, v_in_stride, v_in1, tw,
                             cols, cnt_128);
        }
        else
        {
            TW_GATHER_128_S(curr_in, in_strides, 1, v_in_stride, v_in1, tw,
                            cols, cnt_128);
        }

        GATHER2_128_S(curr_in, v_in_stride, v_in0);

        // Output point 1: X[0]
        v_out0 = _mm_add_ps(v_in0, v_in1);
        // Output point 2: X[1]
        v_out1 = _mm_sub_ps(v_in0, v_in1);

        SCATTER2_128_S(curr_out, v_out_stride, v_out0);
        SCATTER2_128_S(curr_out + out_strides[1], v_out_stride, v_out1);

        in_r = in_r + (v_in_stride << 1);
        out_r = out_r + (v_out_stride << 1);
    }
#endif

#if defined(KERNEL_USE_AVX512) || defined(KERNEL_USE_AVX256) || defined(KERNEL_USE_AVX128)
    if (remaining_sets & 1)
    {
        __m128 v_in0, v_in1;
        __m128 v_out0, v_out1;

        curr_in = in_r;
        curr_out = out_r;

        if (flag)
        {
            ITW_GATHER_LOW_128_S(curr_in, in_strides, 1, v_in1, tw, cols,
                                 cnt_128_low);
        }
        else
        {
            TW_GATHER_LOW_128_S(curr_in, in_strides, 1, v_in1, tw, cols,
                                cnt_128_low);
        }

        LD_LOW_128_S(curr_in, v_in0);

        // Output point 1: X[0]
        v_out0 = _mm_add_ps(v_in0, v_in1);
        // Output point 2: X[1]
        v_out1 = _mm_sub_ps(v_in0, v_in1);

        ST_LOW_128_S(curr_out, v_out0);
        ST_LOW_128_S(curr_out + out_strides[1], v_out1);
    }
#endif

#ifdef AOCL_ENABLE_LOG
    AOCLFFTZ_LOG_UNFORMATTED(TRACE, TRACE, "Exit");
#endif
}

static VOID TWID_KNAME_FP64(VOID *in_real, VOID *in_imag, VOID *out_real,
                            VOID *out_imag, INTP n, aoclfftz_strides_t *strides,
                            VOID *twd, UINT8 flag)
{
#ifdef AOCL_ENABLE_LOG
    AOCLFFTZ_LOG_UNFORMATTED(TRACE, TRACE, "Enter");
#endif
    DOUBLE *in_r = in_real;
    DOUBLE *out_r = out_real;
    DOUBLE *curr_in, *curr_out;
#ifdef VOLATILE_STRIDE_ARRAY
    volatile INTP *in_strides = strides->in_strides;
    volatile INTP *out_strides = strides->out_strides;
#else
    INTP *in_strides = strides->in_strides;
    INTP *out_strides = strides->out_strides;
#endif
    INTP v_in_stride = strides->v_in_stride;
    INTP v_out_stride = strides->v_out_stride;

    aoclfftz_twiddle_t *tws = (aoclfftz_twiddle_t *)twd;
    DOUBLE *tw = (DOUBLE *)(tws->TW);
    UINTP cols = tws->cols;

    INTP N = n / NUM_SETS_D;
    INTP count;

#if defined(KERNEL_USE_AVX512)
    INTP remaining_sets = n % NUM_SETS_D;
    INTP do_256_whole = (INTP)(remaining_sets >= NUM_SETS_256_D);
    INTP cnt_256 = N * NUM_SETS_512_D;
    INTP cnt_128 = N * NUM_SETS_512_D + do_256_whole * NUM_SETS_256_D;
#elif defined(KERNEL_USE_AVX256)
    INTP remaining_sets = n % NUM_SETS_D;
    INTP cnt_128 = N * NUM_SETS_256_D;
#elif defined(KERNEL_USE_AVX128)
    // nothing, since double doesn't have any tail cases to process
#endif

    if (flag)
    {
        in_r = in_imag;
        out_r = out_imag;
    }

    for (count = 0; count < N; count++)
    {
        VREGTYPE_D v_in0, v_in1;
        VREGTYPE_D v_out0, v_out1;

        curr_in = in_r;
        curr_out = out_r;

        if (flag)
        {
            ITW_GATHER_D(curr_in, in_strides, 1, v_in_stride, v_in1, tw, cols,
                         (count * NUM_SETS_D));
        }
        else
        {
            TW_GATHER_D(curr_in, in_strides, 1, v_in_stride, v_in1, tw, cols,
                        (count * NUM_SETS_D));
        }

        GATHER_D(curr_in, v_in_stride, v_in0);

        // Output point 1: X[0]
        v_out0 = ADD_D(v_in0, v_in1);
        // Output point 2: X[1]
        v_out1 = SUB_D(v_in0, v_in1);

        SCATTER_D(curr_out, v_out_stride, v_out0);
        SCATTER_D(curr_out + out_strides[1], v_out_stride, v_out1);

        in_r += NUM_SETS_D * v_in_stride;
        out_r += NUM_SETS_D * v_out_stride;
    }

    // The following contains code that performs the FFT on the tail cases.
    // These tails are conditionally "instantiated" based on the ISA requested
    // by the "includee" file.

#if defined(KERNEL_USE_AVX512)
    if (remaining_sets >= NUM_SETS_256_D)
    {
        __m256d v_in0, v_in1;
        __m256d v_out0, v_out1;

        curr_in = in_r;
        curr_out = out_r;

        if (flag)
        {
            ITW_GATHER_256_D(curr_in, in_strides, 1, v_in_stride, v_in1, tw,
                             cols, cnt_256);
        }
        else
        {
            TW_GATHER_256_D(curr_in, in_strides, 1, v_in_stride, v_in1, tw,
                            cols, cnt_256);
        }

        GATHER2_256_D(curr_in, v_in_stride, v_in0);

        // Output point 1: X[0]
        v_out0 = _mm256_add_pd(v_in0, v_in1);
        // Output point 2: X[1]
        v_out1 = _mm256_sub_pd(v_in0, v_in1);

        SCATTER2_256_D(curr_out, v_out_stride, v_out0);
        SCATTER2_256_D(curr_out + out_strides[1], v_out_stride, v_out1);

        in_r += NUM_SETS_256_D * v_in_stride;
        out_r += NUM_SETS_256_D * v_out_stride;
    }
#endif

#if defined(KERNEL_USE_AVX512) || defined(KERNEL_USE_AVX256)
    if (remaining_sets & 1)
    {
        __m128d v_in0, v_in1;
        __m128d v_out0, v_out1;

        curr_in = in_r;
        curr_out = out_r;

        if (flag)
        {
            ITW_GATHER_128_D(curr_in, in_strides, 1, /* unused */ 0, v_in1, tw,
                             cols, cnt_128);
        }
        else
        {
            TW_GATHER_128_D(curr_in, in_strides, 1, /* unused */ 0, v_in1, tw,
                            cols, cnt_128);
        }

        LD_128_D(curr_in, v_in0);

        // Output point 1: X[0]
        v_out0 = _mm_add_pd(v_in0, v_in1);
        // Output point 2: X[1]
        v_out1 = _mm_sub_pd(v_in0, v_in1);

        ST_128_D(curr_out, v_out0);
        ST_128_D(curr_out + out_strides[1], v_out1);
    }
#endif
#ifdef AOCL_ENABLE_LOG
    AOCLFFTZ_LOG_UNFORMATTED(TRACE, TRACE, "Exit");
#endif
}

kfft_ REGISTER_KERNEL(UINT8 precision, UINT8 direction)
{
    if (precision == DT_FLOAT)
    {
        return TWID_KNAME_FP32;
    }
    else if (precision == DT_DOUBLE)
    {
        return TWID_KNAME_FP64;
    }
    else
    {
        return NULL;
    }
}

ops_cycles_t GET_OPS_COUNT(UINT8 precision, UINT8 direction)
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
