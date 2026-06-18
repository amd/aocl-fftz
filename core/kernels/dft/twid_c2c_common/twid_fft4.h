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

static VOID TWID_KNAME_FP32(VOID *in_real, VOID *in_imag, VOID *out_real,
                            VOID *out_imag, INTP n, aoclfftz_strides_t *strides,
                            VOID *twd, UINT8 flag)
{
    AOCLFFTZ_LOG(DEBUG, global_logger_mode, "Enter");
    const FLOAT CRTM_4_1 = 1.0;

    FLOAT *in_r = (FLOAT *)in_real;
    FLOAT *out_r = (FLOAT *)out_real;
#if defined(KERNEL_VARIANT_C2R)
    FLOAT *in_h2_r = in_r;
#elif defined(KERNEL_VARIANT_R2C)
    FLOAT *out_h2_r = out_r;
#endif

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
#if defined(KERNEL_VARIANT_C2R)
    INTP v_in_h2_stride = strides->v_in_h2_stride;
#elif defined(KERNEL_VARIANT_R2C)
    INTP v_out_h2_stride = strides->v_out_h2_stride;
#endif

    aoclfftz_twiddle_t *tws = (aoclfftz_twiddle_t *)twd;
    FLOAT *tw = (FLOAT *)tws->TW;
    UINTP cols = tws->cols;
    UINTP load_multi_cols = tws->load_multi_cols;

    INTP N = n / NUM_SETS_S;
    INTP remaining_sets = n % NUM_SETS_S;

#if defined(KERNEL_USE_AVX512)
    INTP do_256_whole = (INTP)(remaining_sets >= NUM_SETS_256_S);
    INTP do_128_whole =
        (INTP)(remaining_sets % NUM_SETS_256_S >= NUM_SETS_128_S);
    INTP cnt_256 = load_multi_cols * (N * NUM_SETS_512_S);
    INTP cnt_128 =
        load_multi_cols * (N * NUM_SETS_512_S + do_256_whole * NUM_SETS_256_S);
    INTP cnt_128_low =
        load_multi_cols * (N * NUM_SETS_512_S + do_256_whole * NUM_SETS_256_S +
                           do_128_whole * NUM_SETS_128_S);
#elif defined(KERNEL_USE_AVX256)
    INTP do_128_whole = (INTP)(remaining_sets >= NUM_SETS_128_S);
    INTP cnt_128 = load_multi_cols * (N * NUM_SETS_256_S);
    INTP cnt_128_low =
        load_multi_cols * (N * NUM_SETS_256_S + do_128_whole * NUM_SETS_128_S);
#elif defined(KERNEL_USE_AVX128)
    INTP cnt_128_low = load_multi_cols * (N * NUM_SETS_128_S);
#endif

    VREGTYPE_S v_C1 = BCAST_S(CRTM_4_1);

    INTP count;

#if defined(KERNEL_DIRECTION_BWD)
    v_C1 = NEG_S(v_C1, 1);
#endif
    for (count = 0; count < N; count++)
    {
        VREGTYPE_S v_in0, v_in1, v_in2, v_in3;
        VREGTYPE_S v_av1, v_av2;
        VREGTYPE_S v_out0, v_out1, v_out2, v_out3;
        INTP col = count * load_multi_cols * NUM_SETS_S;

        LOAD_IN_S(in_r, in_strides, 1, v_in_stride, v_in1, tw, cols, col,
                  load_multi_cols, is_contiguous_in);
#if defined(KERNEL_VARIANT_C2R)
        LOAD_IN_S(in_h2_r, in_strides, 2, v_in_h2_stride, v_in2, tw, cols, col,
                  load_multi_cols, 0);
        v_in2 = IN_H2_S(v_in2);
        LOAD_IN_S(in_h2_r, in_strides, 3, v_in_h2_stride, v_in3, tw, cols, col,
                  load_multi_cols, 0);
        v_in3 = IN_H2_S(v_in3);
#else
        LOAD_IN_S(in_r, in_strides, 2, v_in_stride, v_in2, tw, cols, col,
                  load_multi_cols, is_contiguous_in);
        LOAD_IN_S(in_r, in_strides, 3, v_in_stride, v_in3, tw, cols, col,
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
        STORE_OUT_S(out_r, out_strides, 1, v_out_stride, v_out1, tw, cols, col,
                    load_multi_cols, is_contiguous_out);
#if defined(KERNEL_VARIANT_R2C)
        v_out2 = OUT_H2_S(v_out2);
        STORE_OUT_S(out_h2_r, out_strides, 2, v_out_h2_stride, v_out2, tw, cols,
                    col, load_multi_cols, 0);
        v_out3 = OUT_H2_S(v_out3);
        STORE_OUT_S(out_h2_r, out_strides, 3, v_out_h2_stride, v_out3, tw, cols,
                    col, load_multi_cols, 0);
#else
        STORE_OUT_S(out_r, out_strides, 2, v_out_stride, v_out2, tw, cols, col,
                    load_multi_cols, is_contiguous_out);
        STORE_OUT_S(out_r, out_strides, 3, v_out_stride, v_out3, tw, cols, col,
                    load_multi_cols, is_contiguous_out);
#endif

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

        LOAD_IN_256_S(in_r, in_strides, 1, v_in_stride, v_in1, tw, cols,
                      cnt_256, load_multi_cols, is_contiguous_in);
#if defined(KERNEL_VARIANT_C2R)
        LOAD_IN_256_S(in_h2_r, in_strides, 2, v_in_h2_stride, v_in2, tw, cols,
                      cnt_256, load_multi_cols, 0);
        v_in2 = IN_H2_256_S(v_in2);
        LOAD_IN_256_S(in_h2_r, in_strides, 3, v_in_h2_stride, v_in3, tw, cols,
                      cnt_256, load_multi_cols, 0);
        v_in3 = IN_H2_256_S(v_in3);
#else
        LOAD_IN_256_S(in_r, in_strides, 2, v_in_stride, v_in2, tw, cols,
                      cnt_256, load_multi_cols, is_contiguous_in);
        LOAD_IN_256_S(in_r, in_strides, 3, v_in_stride, v_in3, tw, cols,
                      cnt_256, load_multi_cols, is_contiguous_in);
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
        STORE_OUT_256_S(out_r, out_strides, 1, v_out_stride, v_out1, tw, cols,
                        cnt_256, load_multi_cols, is_contiguous_out);
#if defined(KERNEL_VARIANT_R2C)
        v_out2 = OUT_H2_256_S(v_out2);
        STORE_OUT_256_S(out_h2_r, out_strides, 2, v_out_h2_stride, v_out2, tw,
                        cols, cnt_256, load_multi_cols, 0);
        v_out3 = OUT_H2_256_S(v_out3);
        STORE_OUT_256_S(out_h2_r, out_strides, 3, v_out_h2_stride, v_out3, tw,
                        cols, cnt_256, load_multi_cols, 0);
#else
        STORE_OUT_256_S(out_r, out_strides, 2, v_out_stride, v_out2, tw, cols,
                        cnt_256, load_multi_cols, is_contiguous_out);
        STORE_OUT_256_S(out_r, out_strides, 3, v_out_stride, v_out3, tw, cols,
                        cnt_256, load_multi_cols, is_contiguous_out);
#endif

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

        LOAD_IN_128_S(in_r, in_strides, 1, v_in_stride, v_in1, tw, cols,
                      cnt_128, load_multi_cols, is_contiguous_in);
#if defined(KERNEL_VARIANT_C2R)
        LOAD_IN_128_S(in_h2_r, in_strides, 2, v_in_h2_stride, v_in2, tw, cols,
                      cnt_128, load_multi_cols, 0);
        v_in2 = IN_H2_128_S(v_in2);
        LOAD_IN_128_S(in_h2_r, in_strides, 3, v_in_h2_stride, v_in3, tw, cols,
                      cnt_128, load_multi_cols, 0);
        v_in3 = IN_H2_128_S(v_in3);
#else
        LOAD_IN_128_S(in_r, in_strides, 2, v_in_stride, v_in2, tw, cols,
                      cnt_128, load_multi_cols, is_contiguous_in);
        LOAD_IN_128_S(in_r, in_strides, 3, v_in_stride, v_in3, tw, cols,
                      cnt_128, load_multi_cols, is_contiguous_in);
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
        STORE_OUT_128_S(out_r, out_strides, 1, v_out_stride, v_out1, tw, cols,
                        cnt_128, load_multi_cols, is_contiguous_out);
#if defined(KERNEL_VARIANT_R2C)
        v_out2 = OUT_H2_128_S(v_out2);
        STORE_OUT_128_S(out_h2_r, out_strides, 2, v_out_h2_stride, v_out2, tw,
                        cols, cnt_128, load_multi_cols, 0);
        v_out3 = OUT_H2_128_S(v_out3);
        STORE_OUT_128_S(out_h2_r, out_strides, 3, v_out_h2_stride, v_out3, tw,
                        cols, cnt_128, load_multi_cols, 0);
#else
        STORE_OUT_128_S(out_r, out_strides, 2, v_out_stride, v_out2, tw, cols,
                        cnt_128, load_multi_cols, is_contiguous_out);
        STORE_OUT_128_S(out_r, out_strides, 3, v_out_stride, v_out3, tw, cols,
                        cnt_128, load_multi_cols, is_contiguous_out);
#endif

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

        LOAD_IN_64_S(in_r, in_strides, 1, v_in1, tw, cols, cnt_128_low,
                     load_multi_cols, is_contiguous_in);
#if defined(KERNEL_VARIANT_C2R)
        LOAD_IN_64_S(in_h2_r, in_strides, 2, v_in2, tw, cols, cnt_128_low,
                     load_multi_cols, 0);
        v_in2 = IN_H2_128_S(v_in2);
        LOAD_IN_64_S(in_h2_r, in_strides, 3, v_in3, tw, cols, cnt_128_low,
                     load_multi_cols, 0);
        v_in3 = IN_H2_128_S(v_in3);
#else
        LOAD_IN_64_S(in_r, in_strides, 2, v_in2, tw, cols, cnt_128_low,
                     load_multi_cols, is_contiguous_in);
        LOAD_IN_64_S(in_r, in_strides, 3, v_in3, tw, cols, cnt_128_low,
                     load_multi_cols, is_contiguous_in);
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
        STORE_OUT_64_S(out_r, out_strides, 1, v_out1, tw, cols, cnt_128_low,
                       load_multi_cols, is_contiguous_out);
#if defined(KERNEL_VARIANT_R2C)
        v_out2 = OUT_H2_128_S(v_out2);
        STORE_OUT_64_S(out_h2_r, out_strides, 2, v_out2, tw, cols, cnt_128_low,
                       load_multi_cols, 0);
        v_out3 = OUT_H2_128_S(v_out3);
        STORE_OUT_64_S(out_h2_r, out_strides, 3, v_out3, tw, cols, cnt_128_low,
                       load_multi_cols, 0);
#else
        STORE_OUT_64_S(out_r, out_strides, 2, v_out2, tw, cols, cnt_128_low,
                       load_multi_cols, is_contiguous_out);
        STORE_OUT_64_S(out_r, out_strides, 3, v_out3, tw, cols, cnt_128_low,
                       load_multi_cols, is_contiguous_out);
#endif
    }
#endif

    AOCLFFTZ_LOG(DEBUG, global_logger_mode, "Exit");
}

static VOID TWID_KNAME_FP64(VOID *in_real, VOID *in_imag, VOID *out_real,
                            VOID *out_imag, INTP n, aoclfftz_strides_t *strides,
                            VOID *twd, UINT8 flag)
{
    AOCLFFTZ_LOG(DEBUG, global_logger_mode, "Enter");
    const DOUBLE CRTM_4_1 = 1.0;

    DOUBLE *in_r = (DOUBLE *)in_real;
    DOUBLE *out_r = (DOUBLE *)out_real;
#if defined(KERNEL_VARIANT_C2R)
    DOUBLE *in_h2_r = in_r;
#elif defined(KERNEL_VARIANT_R2C)
    DOUBLE *out_h2_r = out_r;
#endif

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
#if defined(KERNEL_VARIANT_C2R)
    INTP v_in_h2_stride = strides->v_in_h2_stride;
#elif defined(KERNEL_VARIANT_R2C)
    INTP v_out_h2_stride = strides->v_out_h2_stride;
#endif

    aoclfftz_twiddle_t *tws = (aoclfftz_twiddle_t *)twd;
    DOUBLE *tw = (DOUBLE *)tws->TW;
    UINTP cols = tws->cols;
    UINTP load_multi_cols = tws->load_multi_cols;

    INTP N = n / NUM_SETS_D;
    INTP count;

#if defined(KERNEL_USE_AVX512)
    INTP remaining_sets = n % NUM_SETS_D;
    INTP do_256_whole = (INTP)(remaining_sets >= NUM_SETS_256_D);
    INTP cnt_256 = load_multi_cols * (N * NUM_SETS_512_D);
    INTP cnt_128 =
        load_multi_cols * (N * NUM_SETS_512_D + do_256_whole * NUM_SETS_256_D);
#elif defined(KERNEL_USE_AVX256)
    INTP remaining_sets = n % NUM_SETS_D;
    INTP cnt_128 = load_multi_cols * (N * NUM_SETS_256_D);
#elif defined(KERNEL_USE_AVX128)
    // nothing, since double doesn't have any tail cases to process for AVX128
#endif

    const INTP in_stride_1 = in_strides[1];
    const INTP in_stride_2 = in_strides[2];
    const INTP in_stride_3 = in_strides[3];
    const INTP out_stride_1 = out_strides[1];
    const INTP out_stride_2 = out_strides[2];
    const INTP out_stride_3 = out_strides[3];
    const INTP in_stride_inc = NUM_SETS_D * v_in_stride;
    const INTP out_stride_inc = NUM_SETS_D * v_out_stride;
#if defined(KERNEL_VARIANT_C2R)
    const INTP in_h2_stride_inc = NUM_SETS_D * v_in_h2_stride;
#elif defined(KERNEL_VARIANT_R2C)
    const INTP out_h2_stride_inc = NUM_SETS_D * v_out_h2_stride;
#endif
    const UINTP tw_col_base_1 = DATA_STRIDE * cols;
    const UINTP tw_col_base_2 = DATA_STRIDE * 2 * cols;
    const UINTP tw_col_base_3 = DATA_STRIDE * 3 * cols;
    const UINTP tw_stride = DATA_STRIDE * NUM_SETS_D;

    VREGTYPE_D v_C1 = BCAST_D(CRTM_4_1);

#if defined(KERNEL_DIRECTION_BWD)
    v_C1 = NEG_D(v_C1, 1);
#endif

    DOUBLE *tw_base_1 = tw + tw_col_base_1;
    DOUBLE *tw_base_2 = tw + tw_col_base_2;
    DOUBLE *tw_base_3 = tw + tw_col_base_3;

    if (load_multi_cols)
    {
        for (count = 0; count < N; count++)
        {
            VREGTYPE_D v_in0, v_in1, v_in2, v_in3;
            VREGTYPE_D v_av1, v_av2;
            VREGTYPE_D v_out0, v_out1, v_out2, v_out3;
            VREGTYPE_D twv1, twv2, twv3;

            DOUBLE *tw_addr_1 = tw_base_1 + count * tw_stride;
            DOUBLE *tw_addr_2 = tw_base_2 + count * tw_stride;
            DOUBLE *tw_addr_3 = tw_base_3 + count * tw_stride;
            twv1 = LOADU_D(tw_addr_1);
            twv2 = LOADU_D(tw_addr_2);
            twv3 = LOADU_D(tw_addr_3);

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
        VREGTYPE_D twv1 = BROADCAST_D(tw_base_1);
        VREGTYPE_D twv2 = BROADCAST_D(tw_base_2);
        VREGTYPE_D twv3 = BROADCAST_D(tw_base_3);

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

        const UINTP addr1 = DATA_STRIDE * (cols + cnt_256);
        const UINTP addr2 = DATA_STRIDE * (2 * cols + cnt_256);
        const UINTP addr3 = DATA_STRIDE * (3 * cols + cnt_256);
        if (load_multi_cols)
        {
            twv1 = _mm256_loadu_pd(tw + addr1);
            twv2 = _mm256_loadu_pd(tw + addr2);
            twv3 = _mm256_loadu_pd(tw + addr3);
        }
        else
        {
            twv1 = _mm256_broadcast_pd((__m128d *)(tw + addr1));
            twv2 = _mm256_broadcast_pd((__m128d *)(tw + addr2));
            twv3 = _mm256_broadcast_pd((__m128d *)(tw + addr3));
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

        const UINTP addr1 = DATA_STRIDE * (cols + cnt_128);
        const UINTP addr2 = DATA_STRIDE * (2 * cols + cnt_128);
        const UINTP addr3 = DATA_STRIDE * (3 * cols + cnt_128);
        twv1 = _mm_loadu_pd(tw + addr1);
        twv2 = _mm_loadu_pd(tw + addr2);
        twv3 = _mm_loadu_pd(tw + addr3);

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
