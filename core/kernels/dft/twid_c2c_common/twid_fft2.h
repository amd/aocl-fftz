// Copyright Advanced Micro Devices, Inc.
// SPDX-License-Identifier: BSD-3-Clause

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

#define RADIX 2

static FFTZ_VOID TWID_KNAME_FP32(FFTZ_VOID *in_real, FFTZ_VOID *in_imag,
                                 FFTZ_VOID *out_real, FFTZ_VOID *out_imag,
                                 FFTZ_INTP n, aoclfftz_strides_t *strides,
                                 FFTZ_VOID *twd, FFTZ_UINT8 flag)
{
    AOCLFFTZ_LOG(DEBUG, global_logger_mode, "Enter");
    FFTZ_FLOAT *in_r = (FFTZ_FLOAT *)in_real;
    FFTZ_FLOAT *out_r = (FFTZ_FLOAT *)out_real;
#if defined(KERNEL_VARIANT_C2R)
    FFTZ_FLOAT *in_h2_r = in_r;
#elif defined(KERNEL_VARIANT_R2C)
    FFTZ_FLOAT *out_h2_r = out_r;
#endif

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
#if defined(KERNEL_VARIANT_C2R)
    FFTZ_INTP v_in_h2_stride = strides->v_in_h2_stride;
#elif defined(KERNEL_VARIANT_R2C)
    FFTZ_INTP v_out_h2_stride = strides->v_out_h2_stride;
#endif

    aoclfftz_twiddle_t *tws = (aoclfftz_twiddle_t *)twd;
    FFTZ_FLOAT *tw = (FFTZ_FLOAT *)tws->TW;
    FFTZ_UINTP load_multi_cols = tws->load_multi_cols;

    FFTZ_INTP N = n / NUM_SETS_S;
    FFTZ_INTP remaining_sets = n % NUM_SETS_S;

    FFTZ_FLOAT *tw_ptr = tw;

    FFTZ_INTP count;

    for (count = 0; count < N; count++)
    {
        VREGTYPE_S v_in0, v_in1;
        VREGTYPE_S v_out0, v_out1;

#if defined(KERNEL_VARIANT_C2R)
        LOAD_IN_S(in_h2_r, in_strides, 1, v_in_h2_stride, v_in1, tw_ptr,
                  load_multi_cols, 0);
        v_in1 = IN_H2_S(v_in1);
#else
        LOAD_IN_S(in_r, in_strides, 1, v_in_stride, v_in1, tw_ptr,
                  load_multi_cols, is_contiguous_in);
#endif

        GATHER_S(in_r, v_in_stride, v_in0, is_contiguous_in);

        // Output point 1: X[0]
        v_out0 = ADD_S(v_in0, v_in1);
        // Output point 2: X[1]
        v_out1 = SUB_S(v_in0, v_in1);

        SCATTER_S(out_r, v_out_stride, v_out0, is_contiguous_out);
#if defined(KERNEL_VARIANT_R2C)
        v_out1 = OUT_H2_S(v_out1);
        STORE_OUT_S(out_h2_r, out_strides, 1, v_out_h2_stride, v_out1, tw_ptr,
                    load_multi_cols, 0);
#else
        STORE_OUT_S(out_r, out_strides, 1, v_out_stride, v_out1, tw_ptr,
                    load_multi_cols, is_contiguous_out);
#endif

        tw_ptr += load_multi_cols * (RADIX - 1) * NUM_SETS_S * DATA_STRIDE;
        in_r += NUM_SETS_S * v_in_stride;
        out_r += NUM_SETS_S * v_out_stride;
#if defined(KERNEL_VARIANT_C2R)
        in_h2_r += NUM_SETS_S * v_in_h2_stride;
#elif defined(KERNEL_VARIANT_R2C)
        out_h2_r += NUM_SETS_S * v_out_h2_stride;
#endif
    }

    // The following contains code that performs the FFT on the tail cases.
    // These tails are conditionally "instantiated" based on the ISA requested
    // by the "includee" file.

#if defined(KERNEL_USE_AVX512)
    if (remaining_sets >= NUM_SETS_256_S)
    {
        __m256 v_in0, v_in1;
        __m256 v_out0, v_out1;

#if defined(KERNEL_VARIANT_C2R)
        LOAD_IN_256_S(in_h2_r, in_strides, 1, v_in_h2_stride, v_in1, tw_ptr,
                      load_multi_cols, 0);
        v_in1 = IN_H2_256_S(v_in1);
#else
        LOAD_IN_256_S(in_r, in_strides, 1, v_in_stride, v_in1, tw_ptr,
                      load_multi_cols, is_contiguous_in);
#endif

        GATHER4_256_S(in_r, v_in_stride, v_in0, is_contiguous_in);

        // Output point 1: X[0]
        v_out0 = _mm256_add_ps(v_in0, v_in1);
        // Output point 2: X[1]
        v_out1 = _mm256_sub_ps(v_in0, v_in1);

        SCATTER4_256_S(out_r, v_out_stride, v_out0, is_contiguous_out);
#if defined(KERNEL_VARIANT_R2C)
        v_out1 = OUT_H2_256_S(v_out1);
        STORE_OUT_256_S(out_h2_r, out_strides, 1, v_out_h2_stride, v_out1,
                        tw_ptr, load_multi_cols, 0);
#else
        STORE_OUT_256_S(out_r, out_strides, 1, v_out_stride, v_out1, tw_ptr,
                        load_multi_cols, is_contiguous_out);
#endif
        tw_ptr += load_multi_cols * (RADIX - 1) * NUM_SETS_256_S * DATA_STRIDE;

        in_r += NUM_SETS_256_S * v_in_stride;
        out_r += NUM_SETS_256_S * v_out_stride;
#if defined(KERNEL_VARIANT_C2R)
        in_h2_r += NUM_SETS_256_S * v_in_h2_stride;
#elif defined(KERNEL_VARIANT_R2C)
        out_h2_r += NUM_SETS_256_S * v_out_h2_stride;
#endif
        remaining_sets = remaining_sets - NUM_SETS_256_S;
    }
#endif

#if defined(KERNEL_USE_AVX512) || defined(KERNEL_USE_AVX256)
    if (remaining_sets >= NUM_SETS_128_S)
    {
        __m128 v_in0, v_in1;
        __m128 v_out0, v_out1;

#if defined(KERNEL_VARIANT_C2R)
        LOAD_IN_128_S(in_h2_r, in_strides, 1, v_in_h2_stride, v_in1, tw_ptr,
                      load_multi_cols, 0);
        v_in1 = IN_H2_128_S(v_in1);
#else
        LOAD_IN_128_S(in_r, in_strides, 1, v_in_stride, v_in1, tw_ptr,
                      load_multi_cols, is_contiguous_in);
#endif

        GATHER2_128_S(in_r, v_in_stride, v_in0, is_contiguous_in);

        // Output point 1: X[0]
        v_out0 = _mm_add_ps(v_in0, v_in1);
        // Output point 2: X[1]
        v_out1 = _mm_sub_ps(v_in0, v_in1);

        SCATTER2_128_S(out_r, v_out_stride, v_out0, is_contiguous_out);
#if defined(KERNEL_VARIANT_R2C)
        v_out1 = OUT_H2_128_S(v_out1);
        STORE_OUT_128_S(out_h2_r, out_strides, 1, v_out_h2_stride, v_out1,
                        tw_ptr, load_multi_cols, 0);
#else
        STORE_OUT_128_S(out_r, out_strides, 1, v_out_stride, v_out1, tw_ptr,
                        load_multi_cols, is_contiguous_out);
#endif
        tw_ptr += load_multi_cols * (RADIX - 1) * NUM_SETS_128_S * DATA_STRIDE;

        in_r += (v_in_stride << 1);
        out_r += (v_out_stride << 1);
#if defined(KERNEL_VARIANT_C2R)
        in_h2_r += (v_in_h2_stride << 1);
#elif defined(KERNEL_VARIANT_R2C)
        out_h2_r += (v_out_h2_stride << 1);
#endif
    }
#endif

#if defined(KERNEL_USE_AVX512) || defined(KERNEL_USE_AVX256) ||                \
    defined(KERNEL_USE_AVX128)
    if (remaining_sets & 1)
    {
        __m128 v_in0, v_in1;
        __m128 v_out0, v_out1;

#if defined(KERNEL_VARIANT_C2R)
        LOAD_IN_64_S(in_h2_r, in_strides, 1, v_in1, tw_ptr, load_multi_cols, 0);
        v_in1 = IN_H2_128_S(v_in1);
#else
        LOAD_IN_64_S(in_r, in_strides, 1, v_in1, tw_ptr, load_multi_cols,
                     is_contiguous_in);
#endif

        LD_LOW_128_S(in_r, v_in0);

        // Output point 1: X[0]
        v_out0 = _mm_add_ps(v_in0, v_in1);
        // Output point 2: X[1]
        v_out1 = _mm_sub_ps(v_in0, v_in1);

        ST_LOW_128_S(out_r, v_out0);
#if defined(KERNEL_VARIANT_R2C)
        v_out1 = OUT_H2_128_S(v_out1);
        STORE_OUT_64_S(out_h2_r, out_strides, 1, v_out1, tw_ptr,
                       load_multi_cols, 0);
#else
        STORE_OUT_64_S(out_r, out_strides, 1, v_out1, tw_ptr, load_multi_cols,
                       is_contiguous_out);
#endif
    }
#endif

    AOCLFFTZ_LOG(DEBUG, global_logger_mode, "Exit");
}

static FFTZ_VOID TWID_KNAME_FP64(FFTZ_VOID *in_real, FFTZ_VOID *in_imag,
                                 FFTZ_VOID *out_real, FFTZ_VOID *out_imag,
                                 FFTZ_INTP n, aoclfftz_strides_t *strides,
                                 FFTZ_VOID *twd, FFTZ_UINT8 flag)
{
    AOCLFFTZ_LOG(DEBUG, global_logger_mode, "Enter");
    FFTZ_DOUBLE *in_r = (FFTZ_DOUBLE *)in_real;
    FFTZ_DOUBLE *out_r = (FFTZ_DOUBLE *)out_real;
#if defined(KERNEL_VARIANT_C2R)
    FFTZ_DOUBLE *in_h2_r = in_r;
#elif defined(KERNEL_VARIANT_R2C)
    FFTZ_DOUBLE *out_h2_r = out_r;
#endif

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
#if defined(KERNEL_VARIANT_C2R)
    FFTZ_INTP v_in_h2_stride = strides->v_in_h2_stride;
#elif defined(KERNEL_VARIANT_R2C)
    FFTZ_INTP v_out_h2_stride = strides->v_out_h2_stride;
#endif

    aoclfftz_twiddle_t *tws = (aoclfftz_twiddle_t *)twd;
    FFTZ_DOUBLE *tw = (FFTZ_DOUBLE *)tws->TW;
    FFTZ_UINTP load_multi_cols = tws->load_multi_cols;

    FFTZ_INTP N = n / NUM_SETS_D;
    FFTZ_INTP count;

#if defined(KERNEL_USE_AVX512) || defined(KERNEL_USE_AVX256)
    FFTZ_INTP remaining_sets = n % NUM_SETS_D;
#endif

    FFTZ_DOUBLE *tw_ptr = tw;

    for (count = 0; count < N; count++)
    {
        VREGTYPE_D v_in0, v_in1;
        VREGTYPE_D v_out0, v_out1;

        GATHER_D(in_r, v_in_stride, v_in0, is_contiguous_in);

#if defined(KERNEL_VARIANT_C2R)
        LOAD_IN_D(in_h2_r, in_strides, 1, v_in_h2_stride, v_in1, tw_ptr,
                  load_multi_cols, 0);
        v_in1 = IN_H2_D(v_in1);
#else
        LOAD_IN_D(in_r, in_strides, 1, v_in_stride, v_in1, tw_ptr,
                  load_multi_cols, is_contiguous_in);
#endif

        // Output point 1: X[0]
        v_out0 = ADD_D(v_in0, v_in1);
        // Output point 2: X[1]
        v_out1 = SUB_D(v_in0, v_in1);

        SCATTER_D(out_r, v_out_stride, v_out0, is_contiguous_out);
#if defined(KERNEL_VARIANT_R2C)
        v_out1 = OUT_H2_D(v_out1);
        STORE_OUT_D(out_h2_r, out_strides, 1, v_out_h2_stride, v_out1, tw_ptr,
                    load_multi_cols, 0);
#else
        STORE_OUT_D(out_r, out_strides, 1, v_out_stride, v_out1, tw_ptr,
                    load_multi_cols, is_contiguous_out);
#endif

        tw_ptr += load_multi_cols * (RADIX - 1) * NUM_SETS_D * DATA_STRIDE;
        in_r += NUM_SETS_D * v_in_stride;
        out_r += NUM_SETS_D * v_out_stride;
#if defined(KERNEL_VARIANT_C2R)
        in_h2_r += NUM_SETS_D * v_in_h2_stride;
#elif defined(KERNEL_VARIANT_R2C)
        out_h2_r += NUM_SETS_D * v_out_h2_stride;
#endif
    }

    // The following contains code that performs the FFT on the tail cases.
    // These tails are conditionally "instantiated" based on the ISA requested
    // by the "includee" file.

#if defined(KERNEL_USE_AVX512)
    if (remaining_sets >= NUM_SETS_256_D)
    {
        __m256d v_in0, v_in1;
        __m256d v_out0, v_out1;

        GATHER2_256_D(in_r, v_in_stride, v_in0, is_contiguous_in);

#if defined(KERNEL_VARIANT_C2R)
        LOAD_IN_256_D(in_h2_r, in_strides, 1, v_in_h2_stride, v_in1, tw_ptr,
                      load_multi_cols, 0);
        v_in1 = IN_H2_256_D(v_in1);
#else
        LOAD_IN_256_D(in_r, in_strides, 1, v_in_stride, v_in1, tw_ptr,
                      load_multi_cols, is_contiguous_in);
#endif

        // Output point 1: X[0]
        v_out0 = _mm256_add_pd(v_in0, v_in1);
        // Output point 2: X[1]
        v_out1 = _mm256_sub_pd(v_in0, v_in1);

        SCATTER2_256_D(out_r, v_out_stride, v_out0, is_contiguous_out);
#if defined(KERNEL_VARIANT_R2C)
        v_out1 = OUT_H2_256_D(v_out1);
        STORE_OUT_256_D(out_h2_r, out_strides, 1, v_out_h2_stride, v_out1,
                        tw_ptr, load_multi_cols, 0);
#else
        STORE_OUT_256_D(out_r, out_strides, 1, v_out_stride, v_out1, tw_ptr,
                        load_multi_cols, is_contiguous_out);
#endif
        tw_ptr += load_multi_cols * (RADIX - 1) * NUM_SETS_256_D * DATA_STRIDE;

        in_r += NUM_SETS_256_D * v_in_stride;
        out_r += NUM_SETS_256_D * v_out_stride;
#if defined(KERNEL_VARIANT_C2R)
        in_h2_r += NUM_SETS_256_D * v_in_h2_stride;
#elif defined(KERNEL_VARIANT_R2C)
        out_h2_r += NUM_SETS_256_D * v_out_h2_stride;
#endif
    }
#endif

#if defined(KERNEL_USE_AVX512) || defined(KERNEL_USE_AVX256)
    if (remaining_sets & 1)
    {
        __m128d v_in0, v_in1;
        __m128d v_out0, v_out1;

        LD_128_D(in_r, v_in0);

#if defined(KERNEL_VARIANT_C2R)
        LOAD_IN_128_D(in_h2_r, in_strides, 1, 0, v_in1, tw_ptr, load_multi_cols,
                      0);
        v_in1 = IN_H2_128_D(v_in1);
#else
        LOAD_IN_128_D(in_r, in_strides, 1, 0, v_in1, tw_ptr, load_multi_cols,
                      is_contiguous_in);
#endif

        // Output point 1: X[0]
        v_out0 = _mm_add_pd(v_in0, v_in1);
        // Output point 2: X[1]
        v_out1 = _mm_sub_pd(v_in0, v_in1);

        ST_128_D(out_r, v_out0);
#if defined(KERNEL_VARIANT_R2C)
        v_out1 = OUT_H2_128_D(v_out1);
        STORE_OUT_128_D(out_h2_r, out_strides, 1, 0, v_out1, tw_ptr,
                        load_multi_cols, 0);
#else
        STORE_OUT_128_D(out_r, out_strides, 1, 0, v_out1, tw_ptr,
                        load_multi_cols, is_contiguous_out);
#endif
    }
#endif
    AOCLFFTZ_LOG(DEBUG, global_logger_mode, "Exit");
}

kfft_ REGISTER_KERNEL(FFTZ_UINT8 precision, FFTZ_UINT8 direction)
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

ops_cycles_t GET_OPS_COUNT(FFTZ_UINT8 precision, FFTZ_UINT8 direction)
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

#undef RADIX
