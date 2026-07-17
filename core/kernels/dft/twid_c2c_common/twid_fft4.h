// Copyright Advanced Micro Devices, Inc.
// SPDX-License-Identifier: BSD-3-Clause

/** @file twid_fft4.h
 *
 *  @brief The ISA generic kernel template for the radix 4 twiddle kernel
 *
 *  This file contains the DIT twiddle radix-4 FFT implementations for
 *  single-precision and double-precision inputs.
 *
 *  @author Ashwin K. Godbole
 */

// This header has no include guards.
// This is intentional.
// The functions defined in this file are not usable by default.
// They are "instantiated" only when "included" in another file.

#include "core/kernels/simd_includes/generic_kernels_common.h"

#define RADIX 4

static FFTZ_VOID TWID_KNAME_FP32(FFTZ_VOID *in_real, FFTZ_VOID *in_imag,
                                 FFTZ_VOID *out_real, FFTZ_VOID *out_imag,
                                 FFTZ_INTP n, aoclfftz_strides_t *strides,
                                 FFTZ_VOID *twd, FFTZ_UINT8 flag)
{
    AOCLFFTZ_LOG(DEBUG, global_logger_mode, "Enter");
    const FFTZ_FLOAT CRTM_4_1 = 1.0;

    FFTZ_FLOAT *in_r = (FFTZ_FLOAT *)in_real;
    FFTZ_FLOAT *out_r = (FFTZ_FLOAT *)out_real;
#if defined(KERNEL_VARIANT_C2R)
    FFTZ_FLOAT *in_h2_r = (FFTZ_FLOAT *)in_real;
#elif defined(KERNEL_VARIANT_R2C)
    FFTZ_FLOAT *out_h2_r = (FFTZ_FLOAT *)out_real;
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

    VREGTYPE_S v_C1 = BCAST_S(CRTM_4_1);

    FFTZ_INTP count;

#if defined(KERNEL_DIRECTION_BWD)
    v_C1 = NEG_S(v_C1, 1);
#endif
    FFTZ_FLOAT *tw_ptr = tw;

    // Single linear-repack loop (matches the other radices). The load/store
    // macros branch on load_multi_cols internally, so both the multi-column
    // (walked) and single-column (broadcast) twiddle layouts are handled here
    // without a second loop body.
    for (count = 0; count < N; count++)
    {
        VREGTYPE_S v_in0, v_in1, v_in2, v_in3;
        VREGTYPE_S v_av1, v_av2;
        VREGTYPE_S v_out0, v_out1, v_out2, v_out3;

        LOAD_IN_S(in_r, in_strides, 1, v_in_stride, v_in1, tw_ptr,
                  load_multi_cols, is_contiguous_in);
#if defined(KERNEL_VARIANT_C2R)
        LOAD_IN_H2_S(in_h2_r, in_strides, 2, v_in_h2_stride, v_in2, tw_ptr,
                  load_multi_cols, 0);
        v_in2 = IN_H2_S(v_in2);
        LOAD_IN_H2_S(in_h2_r, in_strides, 3, v_in_h2_stride, v_in3, tw_ptr,
                  load_multi_cols, 0);
        v_in3 = IN_H2_S(v_in3);
#else
        LOAD_IN_S(in_r, in_strides, 2, v_in_stride, v_in2, tw_ptr,
                  load_multi_cols, is_contiguous_in);
        LOAD_IN_S(in_r, in_strides, 3, v_in_stride, v_in3, tw_ptr,
                  load_multi_cols, is_contiguous_in);
#endif

        GATHER_S(in_r, v_in_stride, v_in0, is_contiguous_in);

        v_av1 = ADD_S(v_in0, v_in2);
        v_av2 = ADD_S(v_in1, v_in3);

        // Output point 1: X[0]
        v_out0 = ADD_S(v_av1, v_av2);
        // Output point 3: X[2]
        v_out2 = SUB_S(v_av1, v_av2);

        v_av1 = SUB_S(v_in3, v_in1);
        v_av1 = MUL_S(v_C1, v_av1);
        v_av1 = SWAP_RI_S(CONJ_S(v_av1));
        v_av2 = SUB_S(v_in0, v_in2);

        // Output point 2: X[1]
        v_out1 = ADD_S(v_av2, v_av1);
        // Output point 4: X[3]
        v_out3 = SUB_S(v_av2, v_av1);

        SCATTER_S(out_r, v_out_stride, v_out0, is_contiguous_out);
        STORE_OUT_S(out_r, out_strides, 1, v_out_stride, v_out1, tw_ptr,
                    load_multi_cols, is_contiguous_out);
#if defined(KERNEL_VARIANT_R2C)
        v_out2 = OUT_H2_S(v_out2);
        STORE_OUT_H2_S(out_h2_r, out_strides, 2, v_out_h2_stride, v_out2,
                       tw_ptr, load_multi_cols, 0);
        v_out3 = OUT_H2_S(v_out3);
        STORE_OUT_H2_S(out_h2_r, out_strides, 3, v_out_h2_stride, v_out3,
                       tw_ptr, load_multi_cols, 0);
#else
        STORE_OUT_S(out_r, out_strides, 2, v_out_stride, v_out2, tw_ptr,
                    load_multi_cols, is_contiguous_out);
        STORE_OUT_S(out_r, out_strides, 3, v_out_stride, v_out3, tw_ptr,
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
        __m256 v_in0, v_in1, v_in2, v_in3;
        __m256 v_av1, v_av2;
        __m256 v_out0, v_out1, v_out2, v_out3;

        __m256 K1 = CAST_512_TO_256_S(v_C1);

        LOAD_IN_256_S(in_r, in_strides, 1, v_in_stride, v_in1, tw_ptr,
                      load_multi_cols, is_contiguous_in);
#if defined(KERNEL_VARIANT_C2R)
        LOAD_IN_H2_256_S(in_h2_r, in_strides, 2, v_in_h2_stride, v_in2, tw_ptr,
                      load_multi_cols, 0);
        v_in2 = IN_H2_256_S(v_in2);
        LOAD_IN_H2_256_S(in_h2_r, in_strides, 3, v_in_h2_stride, v_in3, tw_ptr,
                      load_multi_cols, 0);
        v_in3 = IN_H2_256_S(v_in3);
#else
        LOAD_IN_256_S(in_r, in_strides, 2, v_in_stride, v_in2, tw_ptr,
                      load_multi_cols, is_contiguous_in);
        LOAD_IN_256_S(in_r, in_strides, 3, v_in_stride, v_in3, tw_ptr,
                      load_multi_cols, is_contiguous_in);
#endif

        GATHER4_256_S(in_r, v_in_stride, v_in0, is_contiguous_in);

        v_av1 = _mm256_add_ps(v_in0, v_in2);
        v_av2 = _mm256_add_ps(v_in1, v_in3);

        // Output point 1: X[0]
        v_out0 = _mm256_add_ps(v_av1, v_av2);
        // Output point 3: X[2]
        v_out2 = _mm256_sub_ps(v_av1, v_av2);

        v_av1 = _mm256_sub_ps(v_in3, v_in1);
        v_av1 = _mm256_mul_ps(K1, v_av1);
        v_av1 = SWAP_RI_256_S(CONJ_256_S(v_av1));
        v_av2 = _mm256_sub_ps(v_in0, v_in2);

        // Output point 2: X[1]
        v_out1 = _mm256_add_ps(v_av2, v_av1);
        // Output point 4: X[3]
        v_out3 = _mm256_sub_ps(v_av2, v_av1);

        SCATTER4_256_S(out_r, v_out_stride, v_out0, is_contiguous_out);
        STORE_OUT_256_S(out_r, out_strides, 1, v_out_stride, v_out1, tw_ptr,
                        load_multi_cols, is_contiguous_out);
#if defined(KERNEL_VARIANT_R2C)
        v_out2 = OUT_H2_256_S(v_out2);
        STORE_OUT_H2_256_S(out_h2_r, out_strides, 2, v_out_h2_stride, v_out2,
                        tw_ptr, load_multi_cols, 0);
        v_out3 = OUT_H2_256_S(v_out3);
        STORE_OUT_H2_256_S(out_h2_r, out_strides, 3, v_out_h2_stride, v_out3,
                        tw_ptr, load_multi_cols, 0);
#else
        STORE_OUT_256_S(out_r, out_strides, 2, v_out_stride, v_out2, tw_ptr,
                        load_multi_cols, is_contiguous_out);
        STORE_OUT_256_S(out_r, out_strides, 3, v_out_stride, v_out3, tw_ptr,
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
        __m128 v_in0, v_in1, v_in2, v_in3;
        __m128 v_av1, v_av2;
        __m128 v_out0, v_out1, v_out2, v_out3;

#if defined(KERNEL_USE_AVX512)
        __m128 K1 = CAST_512_TO_128_S(v_C1);
#elif defined(KERNEL_USE_AVX256)
        __m128 K1 = CAST_256_TO_128_S(v_C1);
#endif

        LOAD_IN_128_S(in_r, in_strides, 1, v_in_stride, v_in1, tw_ptr,
                      load_multi_cols, is_contiguous_in);
#if defined(KERNEL_VARIANT_C2R)
        LOAD_IN_H2_128_S(in_h2_r, in_strides, 2, v_in_h2_stride, v_in2, tw_ptr,
                      load_multi_cols, 0);
        v_in2 = IN_H2_128_S(v_in2);
        LOAD_IN_H2_128_S(in_h2_r, in_strides, 3, v_in_h2_stride, v_in3, tw_ptr,
                      load_multi_cols, 0);
        v_in3 = IN_H2_128_S(v_in3);
#else
        LOAD_IN_128_S(in_r, in_strides, 2, v_in_stride, v_in2, tw_ptr,
                      load_multi_cols, is_contiguous_in);
        LOAD_IN_128_S(in_r, in_strides, 3, v_in_stride, v_in3, tw_ptr,
                      load_multi_cols, is_contiguous_in);
#endif

        GATHER2_128_S(in_r, v_in_stride, v_in0, is_contiguous_in);

        v_av1 = _mm_add_ps(v_in0, v_in2);
        v_av2 = _mm_add_ps(v_in1, v_in3);

        // Output point 1: X[0]
        v_out0 = _mm_add_ps(v_av1, v_av2);
        // Output point 3: X[2]
        v_out2 = _mm_sub_ps(v_av1, v_av2);

        v_av1 = _mm_sub_ps(v_in3, v_in1);
        v_av1 = _mm_mul_ps(K1, v_av1);
        v_av1 = SWAP_RI_128_S(CONJ_128_S(v_av1));
        v_av2 = _mm_sub_ps(v_in0, v_in2);

        // Output point 2: X[1]
        v_out1 = _mm_add_ps(v_av2, v_av1);
        // Output point 4: X[3]
        v_out3 = _mm_sub_ps(v_av2, v_av1);

        SCATTER2_128_S(out_r, v_out_stride, v_out0, is_contiguous_out);
        STORE_OUT_128_S(out_r, out_strides, 1, v_out_stride, v_out1, tw_ptr,
                        load_multi_cols, is_contiguous_out);
#if defined(KERNEL_VARIANT_R2C)
        v_out2 = OUT_H2_128_S(v_out2);
        STORE_OUT_H2_128_S(out_h2_r, out_strides, 2, v_out_h2_stride, v_out2,
                        tw_ptr, load_multi_cols, 0);
        v_out3 = OUT_H2_128_S(v_out3);
        STORE_OUT_H2_128_S(out_h2_r, out_strides, 3, v_out_h2_stride, v_out3,
                        tw_ptr, load_multi_cols, 0);
#else
        STORE_OUT_128_S(out_r, out_strides, 2, v_out_stride, v_out2, tw_ptr,
                        load_multi_cols, is_contiguous_out);
        STORE_OUT_128_S(out_r, out_strides, 3, v_out_stride, v_out3, tw_ptr,
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
        remaining_sets = remaining_sets - NUM_SETS_128_S;
    }
#endif

#if defined(KERNEL_USE_AVX512) || defined(KERNEL_USE_AVX256) ||                \
    defined(KERNEL_USE_AVX128)
    if (remaining_sets & 1)
    {
        __m128 v_in0, v_in1, v_in2, v_in3;
        __m128 v_av1, v_av2;
        __m128 v_out0, v_out1, v_out2, v_out3;

#if defined(KERNEL_USE_AVX512)
        __m128 K1 = CAST_512_TO_128_S(v_C1);
#elif defined(KERNEL_USE_AVX256)
        __m128 K1 = CAST_256_TO_128_S(v_C1);
#elif defined(KERNEL_USE_AVX128)
        __m128 K1 = v_C1;
#endif

        LOAD_IN_64_S(in_r, in_strides, 1, v_in1, tw_ptr, load_multi_cols,
                     is_contiguous_in);
#if defined(KERNEL_VARIANT_C2R)
        LOAD_IN_H2_64_S(in_h2_r, in_strides, 2, v_in2, tw_ptr, load_multi_cols,
                        0);
        v_in2 = IN_H2_128_S(v_in2);
        LOAD_IN_H2_64_S(in_h2_r, in_strides, 3, v_in3, tw_ptr, load_multi_cols,
                        0);
        v_in3 = IN_H2_128_S(v_in3);
#else
        LOAD_IN_64_S(in_r, in_strides, 2, v_in2, tw_ptr, load_multi_cols,
                     is_contiguous_in);
        LOAD_IN_64_S(in_r, in_strides, 3, v_in3, tw_ptr, load_multi_cols,
                     is_contiguous_in);
#endif

        LD_LOW_128_S(in_r, v_in0);

        v_av1 = _mm_add_ps(v_in0, v_in2);
        v_av2 = _mm_add_ps(v_in1, v_in3);

        // Output point 1: X[0]
        v_out0 = _mm_add_ps(v_av1, v_av2);
        // Output point 3: X[2]
        v_out2 = _mm_sub_ps(v_av1, v_av2);

        v_av1 = _mm_sub_ps(v_in3, v_in1);
        v_av1 = _mm_mul_ps(K1, v_av1);
        v_av1 = SWAP_RI_128_S(CONJ_128_S(v_av1));
        v_av2 = _mm_sub_ps(v_in0, v_in2);

        // Output point 2: X[1]
        v_out1 = _mm_add_ps(v_av2, v_av1);
        // Output point 4: X[3]
        v_out3 = _mm_sub_ps(v_av2, v_av1);

        ST_LOW_128_S(out_r, v_out0);
        STORE_OUT_64_S(out_r, out_strides, 1, v_out1, tw_ptr, load_multi_cols,
                       is_contiguous_out);
#if defined(KERNEL_VARIANT_R2C)
        v_out2 = OUT_H2_128_S(v_out2);
        STORE_OUT_H2_64_S(out_h2_r, out_strides, 2, v_out2, tw_ptr,
                       load_multi_cols, 0);
        v_out3 = OUT_H2_128_S(v_out3);
        STORE_OUT_H2_64_S(out_h2_r, out_strides, 3, v_out3, tw_ptr,
                       load_multi_cols, 0);
#else
        STORE_OUT_64_S(out_r, out_strides, 2, v_out2, tw_ptr, load_multi_cols,
                       is_contiguous_out);
        STORE_OUT_64_S(out_r, out_strides, 3, v_out3, tw_ptr, load_multi_cols,
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
    const FFTZ_DOUBLE CRTM_4_1 = 1.0;

    FFTZ_DOUBLE *in_r = (FFTZ_DOUBLE *)in_real;
    FFTZ_DOUBLE *out_r = (FFTZ_DOUBLE *)out_real;
#if defined(KERNEL_VARIANT_C2R)
    FFTZ_DOUBLE *in_h2_r = (FFTZ_DOUBLE *)in_real;
#elif defined(KERNEL_VARIANT_R2C)
    FFTZ_DOUBLE *out_h2_r = (FFTZ_DOUBLE *)out_real;
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

    const FFTZ_INTP in_stride_1 = in_strides[1];
    const FFTZ_INTP in_stride_2 = in_strides[2];
    const FFTZ_INTP in_stride_3 = in_strides[3];
    const FFTZ_INTP out_stride_1 = out_strides[1];
    const FFTZ_INTP out_stride_2 = out_strides[2];
    const FFTZ_INTP out_stride_3 = out_strides[3];
    const FFTZ_INTP in_stride_inc = NUM_SETS_D * v_in_stride;
    const FFTZ_INTP out_stride_inc = NUM_SETS_D * v_out_stride;
#if defined(KERNEL_VARIANT_C2R)
    const FFTZ_INTP in_h2_stride_inc = NUM_SETS_D * v_in_h2_stride;
#elif defined(KERNEL_VARIANT_R2C)
    const FFTZ_INTP out_h2_stride_inc = NUM_SETS_D * v_out_h2_stride;
#endif
    // Linear / tile-packed twiddle layout (drops the trivial k = 0 row):
    // within a tile, point k lives at (k - 1) * NUM_SETS_D complex pairs;
    // consecutive tiles are (radix - 1) * NUM_SETS_D complex pairs apart.
    const FFTZ_UINTP tw_col_base_1 = 0;
    const FFTZ_UINTP tw_col_base_2 = DATA_STRIDE * NUM_SETS_D;
    const FFTZ_UINTP tw_col_base_3 = DATA_STRIDE * 2 * NUM_SETS_D;
    const FFTZ_UINTP tw_stride = DATA_STRIDE * 3 * NUM_SETS_D;

    VREGTYPE_D v_C1 = BCAST_D(CRTM_4_1);

#if defined(KERNEL_DIRECTION_BWD)
    v_C1 = NEG_D(v_C1, 1);
#endif

    FFTZ_DOUBLE *tw_ptr = tw;

    // Hoist loop-invariant load_multi_cols outside the loop.
    if (load_multi_cols)
    {
        for (count = 0; count < N; count++)
        {
            VREGTYPE_D v_in0, v_in1, v_in2, v_in3;
            VREGTYPE_D v_av1, v_av2;
            VREGTYPE_D v_out0, v_out1, v_out2, v_out3;
            VREGTYPE_D twv1, twv2, twv3;

            twv1 = LOADU_D(tw_ptr + tw_col_base_1);
            twv2 = LOADU_D(tw_ptr + tw_col_base_2);
            twv3 = LOADU_D(tw_ptr + tw_col_base_3);
            tw_ptr += tw_stride;

            TWID_PRELOADED_LOAD_D(in_r, in_stride_1, v_in_stride, v_in1,
                                  twv1, is_contiguous_in);
#if defined(KERNEL_VARIANT_C2R)
            TWID_PRELOADED_LOAD_D(in_h2_r, in_stride_2, v_in_h2_stride, v_in2,
                                  twv2, 0);
            v_in2 = IN_H2_D(v_in2);
            TWID_PRELOADED_LOAD_D(in_h2_r, in_stride_3, v_in_h2_stride, v_in3,
                                  twv3, 0);
            v_in3 = IN_H2_D(v_in3);
#else
            TWID_PRELOADED_LOAD_D(in_r, in_stride_2, v_in_stride, v_in2,
                                  twv2, is_contiguous_in);
            TWID_PRELOADED_LOAD_D(in_r, in_stride_3, v_in_stride, v_in3,
                                  twv3, is_contiguous_in);
#endif

            GATHER_D(in_r, v_in_stride, v_in0, is_contiguous_in);

            v_av1 = ADD_D(v_in0, v_in2);
            v_av2 = ADD_D(v_in1, v_in3);
            v_out0 = ADD_D(v_av1, v_av2);
            v_out2 = SUB_D(v_av1, v_av2);

            v_av1 = SUB_D(v_in3, v_in1);
            v_av1 = MUL_D(v_C1, v_av1);
            v_av1 = SWAP_RI_D(CONJ_D(v_av1));
            v_av2 = SUB_D(v_in0, v_in2);
            v_out1 = ADD_D(v_av2, v_av1);
            v_out3 = SUB_D(v_av2, v_av1);

            SCATTER_D(out_r, v_out_stride, v_out0, is_contiguous_out);
            TWID_PRELOADED_STORE_D(out_r, out_stride_1, v_out_stride, v_out1,
                                   twv1, is_contiguous_out);
#if defined(KERNEL_VARIANT_R2C)
            v_out2 = OUT_H2_D(v_out2);
            TWID_PRELOADED_STORE_D(out_h2_r, out_stride_2, v_out_h2_stride,
                                   v_out2, twv2, 0);
            v_out3 = OUT_H2_D(v_out3);
            TWID_PRELOADED_STORE_D(out_h2_r, out_stride_3, v_out_h2_stride,
                                   v_out3, twv3, 0);
#else
            TWID_PRELOADED_STORE_D(out_r, out_stride_2, v_out_stride, v_out2,
                                   twv2, is_contiguous_out);
            TWID_PRELOADED_STORE_D(out_r, out_stride_3, v_out_stride, v_out3,
                                   twv3, is_contiguous_out);
#endif

            in_r += in_stride_inc;
            out_r += out_stride_inc;
#if defined(KERNEL_VARIANT_C2R)
            in_h2_r += in_h2_stride_inc;
#elif defined(KERNEL_VARIANT_R2C)
            out_h2_r += out_h2_stride_inc;
#endif
        }
    }
    else
    {
        // Broadcast (single-column) path: twiddle for point k is the constant
        // at (k - 1) complex pairs into the buffer (k = 0 row dropped).
        VREGTYPE_D twv1 = BROADCAST_D(tw);
        VREGTYPE_D twv2 = BROADCAST_D(tw + DATA_STRIDE);
        VREGTYPE_D twv3 = BROADCAST_D(tw + 2 * DATA_STRIDE);

        for (count = 0; count < N; count++)
        {
            VREGTYPE_D v_in0, v_in1, v_in2, v_in3;
            VREGTYPE_D v_av1, v_av2;
            VREGTYPE_D v_out0, v_out1, v_out2, v_out3;

            TWID_PRELOADED_LOAD_D(in_r, in_stride_1, v_in_stride, v_in1,
                                  twv1, is_contiguous_in);
#if defined(KERNEL_VARIANT_C2R)
            TWID_PRELOADED_LOAD_D(in_h2_r, in_stride_2, v_in_h2_stride, v_in2,
                                  twv2, 0);
            v_in2 = IN_H2_D(v_in2);
            TWID_PRELOADED_LOAD_D(in_h2_r, in_stride_3, v_in_h2_stride, v_in3,
                                  twv3, 0);
            v_in3 = IN_H2_D(v_in3);
#else
            TWID_PRELOADED_LOAD_D(in_r, in_stride_2, v_in_stride, v_in2,
                                  twv2, is_contiguous_in);
            TWID_PRELOADED_LOAD_D(in_r, in_stride_3, v_in_stride, v_in3,
                                  twv3, is_contiguous_in);
#endif

            GATHER_D(in_r, v_in_stride, v_in0, is_contiguous_in);

            v_av1 = ADD_D(v_in0, v_in2);
            v_av2 = ADD_D(v_in1, v_in3);
            v_out0 = ADD_D(v_av1, v_av2);
            v_out2 = SUB_D(v_av1, v_av2);

            v_av1 = SUB_D(v_in3, v_in1);
            v_av1 = MUL_D(v_C1, v_av1);
            v_av1 = SWAP_RI_D(CONJ_D(v_av1));
            v_av2 = SUB_D(v_in0, v_in2);
            v_out1 = ADD_D(v_av2, v_av1);
            v_out3 = SUB_D(v_av2, v_av1);

            SCATTER_D(out_r, v_out_stride, v_out0, is_contiguous_out);
            TWID_PRELOADED_STORE_D(out_r, out_stride_1, v_out_stride, v_out1,
                                   twv1, is_contiguous_out);
#if defined(KERNEL_VARIANT_R2C)
            v_out2 = OUT_H2_D(v_out2);
            TWID_PRELOADED_STORE_D(out_h2_r, out_stride_2, v_out_h2_stride,
                                   v_out2, twv2, 0);
            v_out3 = OUT_H2_D(v_out3);
            TWID_PRELOADED_STORE_D(out_h2_r, out_stride_3, v_out_h2_stride,
                                   v_out3, twv3, 0);
#else
            TWID_PRELOADED_STORE_D(out_r, out_stride_2, v_out_stride, v_out2,
                                   twv2, is_contiguous_out);
            TWID_PRELOADED_STORE_D(out_r, out_stride_3, v_out_stride, v_out3,
                                   twv3, is_contiguous_out);
#endif

            in_r += in_stride_inc;
            out_r += out_stride_inc;
#if defined(KERNEL_VARIANT_C2R)
            in_h2_r += in_h2_stride_inc;
#elif defined(KERNEL_VARIANT_R2C)
            out_h2_r += out_h2_stride_inc;
#endif
        }
    }

    // The following contains code that performs the FFT on the tail cases.
    // These tails are conditionally "instantiated" based on the ISA requested
    // by the "includee" file.

#if defined(KERNEL_USE_AVX512)
    if (remaining_sets >= NUM_SETS_256_D)
    {
        __m256d v_in0, v_in1, v_in2, v_in3;
        __m256d v_av1, v_av2;
        __m256d v_out0, v_out1, v_out2, v_out3;
        __m256d twv1, twv2, twv3;

        __m256d K1 = CAST_512_TO_256_D(v_C1);

        // Linear tail tile: the walked tile pointer is already positioned at
        // this tail's tile; point k lives at (k - 1) * NUM_SETS_256_D within
        // it.
        if (load_multi_cols)
        {
            twv1 = _mm256_loadu_pd(tw_ptr + 0 * NUM_SETS_256_D * DATA_STRIDE);
            twv2 = _mm256_loadu_pd(tw_ptr + 1 * NUM_SETS_256_D * DATA_STRIDE);
            twv3 = _mm256_loadu_pd(tw_ptr + 2 * NUM_SETS_256_D * DATA_STRIDE);
            tw_ptr += 3 * NUM_SETS_256_D * DATA_STRIDE;
        }
        else
        {
            // Broadcast: point k is the constant at (k - 1) complex pairs.
            twv1 = _mm256_broadcast_pd((__m128d *)(tw));
            twv2 = _mm256_broadcast_pd((__m128d *)(tw + DATA_STRIDE));
            twv3 = _mm256_broadcast_pd((__m128d *)(tw + 2 * DATA_STRIDE));
        }

        TWID_PRELOADED_LOAD_256_D(in_r, in_stride_1, v_in_stride, v_in1,
                                  twv1, is_contiguous_in);
#if defined(KERNEL_VARIANT_C2R)
        TWID_PRELOADED_LOAD_256_D(in_h2_r, in_stride_2, v_in_h2_stride, v_in2,
                                  twv2, 0);
        v_in2 = IN_H2_256_D(v_in2);
        TWID_PRELOADED_LOAD_256_D(in_h2_r, in_stride_3, v_in_h2_stride, v_in3,
                                  twv3, 0);
        v_in3 = IN_H2_256_D(v_in3);
#else
        TWID_PRELOADED_LOAD_256_D(in_r, in_stride_2, v_in_stride, v_in2,
                                  twv2, is_contiguous_in);
        TWID_PRELOADED_LOAD_256_D(in_r, in_stride_3, v_in_stride, v_in3,
                                  twv3, is_contiguous_in);
#endif

        GATHER2_256_D(in_r, v_in_stride, v_in0, is_contiguous_in);

        v_av1 = _mm256_add_pd(v_in0, v_in2);
        v_av2 = _mm256_add_pd(v_in1, v_in3);
        v_out0 = _mm256_add_pd(v_av1, v_av2);
        v_out2 = _mm256_sub_pd(v_av1, v_av2);

        v_av1 = _mm256_sub_pd(v_in3, v_in1);
        v_av1 = _mm256_mul_pd(K1, v_av1);
        v_av1 = SWAP_RI_256_D(CONJ_256_D(v_av1));
        v_av2 = _mm256_sub_pd(v_in0, v_in2);
        v_out1 = _mm256_add_pd(v_av2, v_av1);
        v_out3 = _mm256_sub_pd(v_av2, v_av1);

        SCATTER2_256_D(out_r, v_out_stride, v_out0, is_contiguous_out);
        TWID_PRELOADED_STORE_256_D(out_r, out_stride_1, v_out_stride, v_out1,
                                   twv1, is_contiguous_out);
#if defined(KERNEL_VARIANT_R2C)
        v_out2 = OUT_H2_256_D(v_out2);
        TWID_PRELOADED_STORE_256_D(out_h2_r, out_stride_2, v_out_h2_stride,
                                   v_out2, twv2, 0);
        v_out3 = OUT_H2_256_D(v_out3);
        TWID_PRELOADED_STORE_256_D(out_h2_r, out_stride_3, v_out_h2_stride,
                                   v_out3, twv3, 0);
#else
        TWID_PRELOADED_STORE_256_D(out_r, out_stride_2, v_out_stride, v_out2,
                                   twv2, is_contiguous_out);
        TWID_PRELOADED_STORE_256_D(out_r, out_stride_3, v_out_stride, v_out3,
                                   twv3, is_contiguous_out);
#endif

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
        __m128d v_in0, v_in1, v_in2, v_in3;
        __m128d v_av1, v_av2;
        __m128d v_out0, v_out1, v_out2, v_out3;
        __m128d twv1, twv2, twv3;

#if defined(KERNEL_USE_AVX512)
        __m128d K1 = CAST_512_TO_128_D(v_C1);
#elif defined(KERNEL_USE_AVX256)
        __m128d K1 = CAST_256_TO_128_D(v_C1);
#endif

        // Linear tail tile (single double-complex set, NUM_SETS_128_D == 1):
        // the walked tile pointer is positioned at this tail's tile; point k is
        // at (k - 1) complex pairs in. On the broadcast path tw_ptr == tw, and
        // a 128-bit load of one complex pair is identical to a broadcast.
        twv1 = _mm_loadu_pd(tw_ptr + 0 * DATA_STRIDE);
        twv2 = _mm_loadu_pd(tw_ptr + 1 * DATA_STRIDE);
        twv3 = _mm_loadu_pd(tw_ptr + 2 * DATA_STRIDE);

        TWID_PRELOADED_LOAD_128_D(in_r, in_stride_1, 0, v_in1, twv1,
                                  is_contiguous_in);
#if defined(KERNEL_VARIANT_C2R)
        TWID_PRELOADED_LOAD_128_D(in_h2_r, in_stride_2, 0, v_in2, twv2, 0);
        v_in2 = IN_H2_128_D(v_in2);
        TWID_PRELOADED_LOAD_128_D(in_h2_r, in_stride_3, 0, v_in3, twv3, 0);
        v_in3 = IN_H2_128_D(v_in3);
#else
        TWID_PRELOADED_LOAD_128_D(in_r, in_stride_2, 0, v_in2, twv2,
                                  is_contiguous_in);
        TWID_PRELOADED_LOAD_128_D(in_r, in_stride_3, 0, v_in3, twv3,
                                  is_contiguous_in);
#endif

        LD_128_D(in_r, v_in0);

        v_av1 = _mm_add_pd(v_in0, v_in2);
        v_av2 = _mm_add_pd(v_in1, v_in3);
        v_out0 = _mm_add_pd(v_av1, v_av2);
        v_out2 = _mm_sub_pd(v_av1, v_av2);

        v_av1 = _mm_sub_pd(v_in3, v_in1);
        v_av1 = _mm_mul_pd(K1, v_av1);
        v_av1 = SWAP_RI_128_D(CONJ_128_D(v_av1));
        v_av2 = _mm_sub_pd(v_in0, v_in2);
        v_out1 = _mm_add_pd(v_av2, v_av1);
        v_out3 = _mm_sub_pd(v_av2, v_av1);

        ST_128_D(out_r, v_out0);
        TWID_PRELOADED_STORE_128_D(out_r, out_stride_1, 0, v_out1, twv1,
                                   is_contiguous_out);
#if defined(KERNEL_VARIANT_R2C)
        v_out2 = OUT_H2_128_D(v_out2);
        TWID_PRELOADED_STORE_128_D(out_h2_r, out_stride_2, 0, v_out2, twv2, 0);
        v_out3 = OUT_H2_128_D(v_out3);
        TWID_PRELOADED_STORE_128_D(out_h2_r, out_stride_3, 0, v_out3, twv3, 0);
#else
        TWID_PRELOADED_STORE_128_D(out_r, out_stride_2, 0, v_out2, twv2,
                                   is_contiguous_out);
        TWID_PRELOADED_STORE_128_D(out_r, out_stride_3, 0, v_out3, twv3,
                                   is_contiguous_out);
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
