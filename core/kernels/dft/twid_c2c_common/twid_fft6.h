// Copyright Advanced Micro Devices, Inc.
// SPDX-License-Identifier: BSD-3-Clause

/** @file twid_fft6.h
 *
 *  @brief The ISA generic kernel template for the radix 6 twiddle kernel
 *
 *  This file contains the DIT twiddle radix-6 FFT implementations for
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
    const FLOAT CRTM_6[2] = {0.500000000000000000000000000000000000000000000,
                             0.866025403784438646763723170752936183471402627};

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

    VREGTYPE_S v_C1 = BCAST_S(CRTM_6[0]);
    VREGTYPE_S v_C2 = BCAST_S(CRTM_6[1]);

    INTP count;

#if defined(KERNEL_DIRECTION_BWD)
    v_C2 = NEG_S(v_C2, 1);
#endif
    for (count = 0; count < N; count++)
    {
        VREGTYPE_S v_in0, v_in1, v_in2, v_in3, v_in4, v_in5;
        VREGTYPE_S v_av1, v_av2, v_av3, v_av4, v_cv1, v_cv2, v_cv3;
        VREGTYPE_S v_out0, v_out1, v_out2, v_out3, v_out4, v_out5;
        INTP col = count * load_multi_cols * NUM_SETS_S;

        LOAD_IN_S(in_r, in_strides, 1, v_in_stride, v_in1, tw, cols, col,
                  load_multi_cols, is_contiguous_in);
        LOAD_IN_S(in_r, in_strides, 2, v_in_stride, v_in2, tw, cols, col,
                  load_multi_cols, is_contiguous_in);
#if defined(KERNEL_VARIANT_C2R)
        LOAD_IN_S(in_h2_r, in_strides, 3, v_in_h2_stride, v_in3, tw, cols, col,
                  load_multi_cols, 0);
        v_in3 = IN_H2_S(v_in3);
        LOAD_IN_S(in_h2_r, in_strides, 4, v_in_h2_stride, v_in4, tw, cols, col,
                  load_multi_cols, 0);
        v_in4 = IN_H2_S(v_in4);
        LOAD_IN_S(in_h2_r, in_strides, 5, v_in_h2_stride, v_in5, tw, cols, col,
                  load_multi_cols, 0);
        v_in5 = IN_H2_S(v_in5);
#else
        LOAD_IN_S(in_r, in_strides, 3, v_in_stride, v_in3, tw, cols, col,
                  load_multi_cols, is_contiguous_in);
        LOAD_IN_S(in_r, in_strides, 4, v_in_stride, v_in4, tw, cols, col,
                  load_multi_cols, is_contiguous_in);
        LOAD_IN_S(in_r, in_strides, 5, v_in_stride, v_in5, tw, cols, col,
                  load_multi_cols, is_contiguous_in);
#endif
        GATHER_S(in_r, v_in_stride, v_in0, is_contiguous_in);

        v_av1 = ADD_S(v_in0, v_in3);
        v_av2 = ADD_S(v_in2, v_in4);
        v_av3 = ADD_S(v_in1, v_in5);
        v_av4 = ADD_S(v_av2, v_av3);
        v_cv1 = SUB_S(v_av1, MUL_S(v_C1, v_av4));

        // Output point 1: X[0]
        v_out0 = ADD_S(v_av1, v_av4);

        v_av1 = SUB_S(v_av2, v_av3);
        v_av2 = SUB_S(v_in0, v_in3);
        v_cv2 = SUB_S(v_av2, MUL_S(v_C1, v_av1));

        // Output point 4: X[3]
        v_out3 = ADD_S(v_av2, v_av1);

        v_av1 = SUB_S(v_in1, v_in5);
        v_av2 = SUB_S(v_in2, v_in4);
        v_av3 = SUB_S(v_av1, v_av2);
        v_cv3 = MUL_S(v_C2, v_av3);
        v_cv3 = CONJ_S(SWAP_RI_S(v_cv3));

        // Output point 3: X[2]
        v_out2 = ADD_S(v_cv1, v_cv3);
        // Output point 5: X[4]
        v_out4 = SUB_S(v_cv1, v_cv3);

        v_av3 = ADD_S(v_av1, v_av2);
        v_cv3 = MUL_S(v_C2, v_av3);
        v_cv3 = CONJ_S(SWAP_RI_S(v_cv3));

        // Output point 2: X[1]
        v_out1 = ADD_S(v_cv2, v_cv3);
        // Output point 6: X[5]
        v_out5 = SUB_S(v_cv2, v_cv3);

        SCATTER_S(out_r, v_out_stride, v_out0, is_contiguous_out);
        STORE_OUT_S(out_r, out_strides, 1, v_out_stride, v_out1, tw, cols, col,
                    load_multi_cols, is_contiguous_out);
        STORE_OUT_S(out_r, out_strides, 2, v_out_stride, v_out2, tw, cols, col,
                    load_multi_cols, is_contiguous_out);
#if defined(KERNEL_VARIANT_R2C)
        v_out3 = OUT_H2_S(v_out3);
        STORE_OUT_S(out_h2_r, out_strides, 3, v_out_h2_stride, v_out3, tw, cols,
                    col, load_multi_cols, 0);
        v_out4 = OUT_H2_S(v_out4);
        STORE_OUT_S(out_h2_r, out_strides, 4, v_out_h2_stride, v_out4, tw, cols,
                    col, load_multi_cols, 0);
        v_out5 = OUT_H2_S(v_out5);
        STORE_OUT_S(out_h2_r, out_strides, 5, v_out_h2_stride, v_out5, tw, cols,
                    col, load_multi_cols, 0);
#else
        STORE_OUT_S(out_r, out_strides, 3, v_out_stride, v_out3, tw, cols, col,
                    load_multi_cols, is_contiguous_out);
        STORE_OUT_S(out_r, out_strides, 4, v_out_stride, v_out4, tw, cols, col,
                    load_multi_cols, is_contiguous_out);
        STORE_OUT_S(out_r, out_strides, 5, v_out_stride, v_out5, tw, cols, col,
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
        __m256 v_in0, v_in1, v_in2, v_in3, v_in4, v_in5;
        __m256 v_av1, v_av2, v_av3, v_av4, v_cv1, v_cv2, v_cv3;
        __m256 v_out0, v_out1, v_out2, v_out3, v_out4, v_out5;

        __m256 K1 = CAST_512_TO_256_S(v_C1);
        __m256 K2 = CAST_512_TO_256_S(v_C2);

        LOAD_IN_256_S(in_r, in_strides, 1, v_in_stride, v_in1, tw, cols,
                      cnt_256, load_multi_cols, is_contiguous_in);
        LOAD_IN_256_S(in_r, in_strides, 2, v_in_stride, v_in2, tw, cols,
                      cnt_256, load_multi_cols, is_contiguous_in);
#if defined(KERNEL_VARIANT_C2R)
        LOAD_IN_256_S(in_h2_r, in_strides, 3, v_in_h2_stride, v_in3, tw, cols,
                      cnt_256, load_multi_cols, 0);
        v_in3 = IN_H2_256_S(v_in3);
        LOAD_IN_256_S(in_h2_r, in_strides, 4, v_in_h2_stride, v_in4, tw, cols,
                      cnt_256, load_multi_cols, 0);
        v_in4 = IN_H2_256_S(v_in4);
        LOAD_IN_256_S(in_h2_r, in_strides, 5, v_in_h2_stride, v_in5, tw, cols,
                      cnt_256, load_multi_cols, 0);
        v_in5 = IN_H2_256_S(v_in5);
#else
        LOAD_IN_256_S(in_r, in_strides, 3, v_in_stride, v_in3, tw, cols,
                      cnt_256, load_multi_cols, is_contiguous_in);
        LOAD_IN_256_S(in_r, in_strides, 4, v_in_stride, v_in4, tw, cols,
                      cnt_256, load_multi_cols, is_contiguous_in);
        LOAD_IN_256_S(in_r, in_strides, 5, v_in_stride, v_in5, tw, cols,
                      cnt_256, load_multi_cols, is_contiguous_in);
#endif
        GATHER4_256_S(in_r, v_in_stride, v_in0, is_contiguous_in);

        v_av1 = _mm256_add_ps(v_in0, v_in3);
        v_av2 = _mm256_add_ps(v_in2, v_in4);
        v_av3 = _mm256_add_ps(v_in1, v_in5);
        v_av4 = _mm256_add_ps(v_av2, v_av3);
        v_cv1 = _mm256_sub_ps(v_av1, _mm256_mul_ps(K1, v_av4));

        // Output point 1: X[0]
        v_out0 = _mm256_add_ps(v_av1, v_av4);

        v_av1 = _mm256_sub_ps(v_av2, v_av3);
        v_av2 = _mm256_sub_ps(v_in0, v_in3);
        v_cv2 = _mm256_sub_ps(v_av2, _mm256_mul_ps(K1, v_av1));

        // Output point 4: X[3]
        v_out3 = _mm256_add_ps(v_av2, v_av1);

        v_av1 = _mm256_sub_ps(v_in1, v_in5);
        v_av2 = _mm256_sub_ps(v_in2, v_in4);
        v_av3 = _mm256_sub_ps(v_av1, v_av2);
        v_cv3 = _mm256_mul_ps(K2, v_av3);
        v_cv3 = CONJ_256_S(SWAP_RI_256_S(v_cv3));

        // Output point 3: X[2]
        v_out2 = _mm256_add_ps(v_cv1, v_cv3);
        // Output point 5: X[4]
        v_out4 = _mm256_sub_ps(v_cv1, v_cv3);

        v_av3 = _mm256_add_ps(v_av1, v_av2);
        v_cv3 = _mm256_mul_ps(K2, v_av3);
        v_cv3 = CONJ_256_S(SWAP_RI_256_S(v_cv3));

        // Output point 2: X[1]
        v_out1 = _mm256_add_ps(v_cv2, v_cv3);
        // Output point 6: X[5]
        v_out5 = _mm256_sub_ps(v_cv2, v_cv3);

        SCATTER4_256_S(out_r, v_out_stride, v_out0, is_contiguous_out);
        STORE_OUT_256_S(out_r, out_strides, 1, v_out_stride, v_out1, tw, cols,
                        cnt_256, load_multi_cols, is_contiguous_out);
        STORE_OUT_256_S(out_r, out_strides, 2, v_out_stride, v_out2, tw, cols,
                        cnt_256, load_multi_cols, is_contiguous_out);
#if defined(KERNEL_VARIANT_R2C)
        v_out3 = OUT_H2_256_S(v_out3);
        STORE_OUT_256_S(out_h2_r, out_strides, 3, v_out_h2_stride, v_out3, tw,
                        cols, cnt_256, load_multi_cols, 0);
        v_out4 = OUT_H2_256_S(v_out4);
        STORE_OUT_256_S(out_h2_r, out_strides, 4, v_out_h2_stride, v_out4, tw,
                        cols, cnt_256, load_multi_cols, 0);
        v_out5 = OUT_H2_256_S(v_out5);
        STORE_OUT_256_S(out_h2_r, out_strides, 5, v_out_h2_stride, v_out5, tw,
                        cols, cnt_256, load_multi_cols, 0);
#else
        STORE_OUT_256_S(out_r, out_strides, 3, v_out_stride, v_out3, tw, cols,
                        cnt_256, load_multi_cols, is_contiguous_out);
        STORE_OUT_256_S(out_r, out_strides, 4, v_out_stride, v_out4, tw, cols,
                        cnt_256, load_multi_cols, is_contiguous_out);
        STORE_OUT_256_S(out_r, out_strides, 5, v_out_stride, v_out5, tw, cols,
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
        __m128 v_in0, v_in1, v_in2, v_in3, v_in4, v_in5;
        __m128 v_av1, v_av2, v_av3, v_av4, v_cv1, v_cv2, v_cv3;
        __m128 v_out0, v_out1, v_out2, v_out3, v_out4, v_out5;

#if defined(KERNEL_USE_AVX512)
        __m128 K1 = CAST_512_TO_128_S(v_C1);
        __m128 K2 = CAST_512_TO_128_S(v_C2);
#elif defined(KERNEL_USE_AVX256)
        __m128 K1 = CAST_256_TO_128_S(v_C1);
        __m128 K2 = CAST_256_TO_128_S(v_C2);
#endif

        LOAD_IN_128_S(in_r, in_strides, 1, v_in_stride, v_in1, tw, cols,
                      cnt_128, load_multi_cols, is_contiguous_in);
        LOAD_IN_128_S(in_r, in_strides, 2, v_in_stride, v_in2, tw, cols,
                      cnt_128, load_multi_cols, is_contiguous_in);
#if defined(KERNEL_VARIANT_C2R)
        LOAD_IN_128_S(in_h2_r, in_strides, 3, v_in_h2_stride, v_in3, tw, cols,
                      cnt_128, load_multi_cols, 0);
        v_in3 = IN_H2_128_S(v_in3);
        LOAD_IN_128_S(in_h2_r, in_strides, 4, v_in_h2_stride, v_in4, tw, cols,
                      cnt_128, load_multi_cols, 0);
        v_in4 = IN_H2_128_S(v_in4);
        LOAD_IN_128_S(in_h2_r, in_strides, 5, v_in_h2_stride, v_in5, tw, cols,
                      cnt_128, load_multi_cols, 0);
        v_in5 = IN_H2_128_S(v_in5);
#else
        LOAD_IN_128_S(in_r, in_strides, 3, v_in_stride, v_in3, tw, cols,
                      cnt_128, load_multi_cols, is_contiguous_in);
        LOAD_IN_128_S(in_r, in_strides, 4, v_in_stride, v_in4, tw, cols,
                      cnt_128, load_multi_cols, is_contiguous_in);
        LOAD_IN_128_S(in_r, in_strides, 5, v_in_stride, v_in5, tw, cols,
                      cnt_128, load_multi_cols, is_contiguous_in);
#endif
        GATHER2_128_S(in_r, v_in_stride, v_in0, is_contiguous_in);

        v_av1 = _mm_add_ps(v_in0, v_in3);
        v_av2 = _mm_add_ps(v_in2, v_in4);
        v_av3 = _mm_add_ps(v_in1, v_in5);
        v_av4 = _mm_add_ps(v_av2, v_av3);
        v_cv1 = _mm_sub_ps(v_av1, _mm_mul_ps(K1, v_av4));

        // Output point 1: X[0]
        v_out0 = _mm_add_ps(v_av1, v_av4);

        v_av1 = _mm_sub_ps(v_av2, v_av3);
        v_av2 = _mm_sub_ps(v_in0, v_in3);
        v_cv2 = _mm_sub_ps(v_av2, _mm_mul_ps(K1, v_av1));

        // Output point 4: X[3]
        v_out3 = _mm_add_ps(v_av2, v_av1);

        v_av1 = _mm_sub_ps(v_in1, v_in5);
        v_av2 = _mm_sub_ps(v_in2, v_in4);
        v_av3 = _mm_sub_ps(v_av1, v_av2);
        v_cv3 = _mm_mul_ps(K2, v_av3);
        v_cv3 = CONJ_128_S(SWAP_RI_128_S(v_cv3));

        // Output point 3: X[2]
        v_out2 = _mm_add_ps(v_cv1, v_cv3);
        // Output point 5: X[4]
        v_out4 = _mm_sub_ps(v_cv1, v_cv3);

        v_av3 = _mm_add_ps(v_av1, v_av2);
        v_cv3 = _mm_mul_ps(K2, v_av3);
        v_cv3 = CONJ_128_S(SWAP_RI_128_S(v_cv3));

        // Output point 2: X[1]
        v_out1 = _mm_add_ps(v_cv2, v_cv3);
        // Output point 6: X[5]
        v_out5 = _mm_sub_ps(v_cv2, v_cv3);

        SCATTER2_128_S(out_r, v_out_stride, v_out0, is_contiguous_out);
        STORE_OUT_128_S(out_r, out_strides, 1, v_out_stride, v_out1, tw, cols,
                        cnt_128, load_multi_cols, is_contiguous_out);
        STORE_OUT_128_S(out_r, out_strides, 2, v_out_stride, v_out2, tw, cols,
                        cnt_128, load_multi_cols, is_contiguous_out);
#if defined(KERNEL_VARIANT_R2C)
        v_out3 = OUT_H2_128_S(v_out3);
        STORE_OUT_128_S(out_h2_r, out_strides, 3, v_out_h2_stride, v_out3, tw,
                        cols, cnt_128, load_multi_cols, 0);
        v_out4 = OUT_H2_128_S(v_out4);
        STORE_OUT_128_S(out_h2_r, out_strides, 4, v_out_h2_stride, v_out4, tw,
                        cols, cnt_128, load_multi_cols, 0);
        v_out5 = OUT_H2_128_S(v_out5);
        STORE_OUT_128_S(out_h2_r, out_strides, 5, v_out_h2_stride, v_out5, tw,
                        cols, cnt_128, load_multi_cols, 0);
#else
        STORE_OUT_128_S(out_r, out_strides, 3, v_out_stride, v_out3, tw, cols,
                        cnt_128, load_multi_cols, is_contiguous_out);
        STORE_OUT_128_S(out_r, out_strides, 4, v_out_stride, v_out4, tw, cols,
                        cnt_128, load_multi_cols, is_contiguous_out);
        STORE_OUT_128_S(out_r, out_strides, 5, v_out_stride, v_out5, tw, cols,
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
        __m128 v_in0, v_in1, v_in2, v_in3, v_in4, v_in5;
        __m128 v_av1, v_av2, v_av3, v_av4, v_cv1, v_cv2, v_cv3;
        __m128 v_out0, v_out1, v_out2, v_out3, v_out4, v_out5;

#if defined(KERNEL_USE_AVX512)
        __m128 K1 = CAST_512_TO_128_S(v_C1);
        __m128 K2 = CAST_512_TO_128_S(v_C2);
#elif defined(KERNEL_USE_AVX256)
        __m128 K1 = CAST_256_TO_128_S(v_C1);
        __m128 K2 = CAST_256_TO_128_S(v_C2);
#elif defined(KERNEL_USE_AVX128)
        __m128 K1 = v_C1;
        __m128 K2 = v_C2;
#endif

        LOAD_IN_64_S(in_r, in_strides, 1, v_in1, tw, cols, cnt_128_low,
                     load_multi_cols, is_contiguous_in);
        LOAD_IN_64_S(in_r, in_strides, 2, v_in2, tw, cols, cnt_128_low,
                     load_multi_cols, is_contiguous_in);
#if defined(KERNEL_VARIANT_C2R)
        LOAD_IN_64_S(in_h2_r, in_strides, 3, v_in3, tw, cols, cnt_128_low,
                     load_multi_cols, 0);
        v_in3 = IN_H2_128_S(v_in3);
        LOAD_IN_64_S(in_h2_r, in_strides, 4, v_in4, tw, cols, cnt_128_low,
                     load_multi_cols, 0);
        v_in4 = IN_H2_128_S(v_in4);
        LOAD_IN_64_S(in_h2_r, in_strides, 5, v_in5, tw, cols, cnt_128_low,
                     load_multi_cols, 0);
        v_in5 = IN_H2_128_S(v_in5);
#else
        LOAD_IN_64_S(in_r, in_strides, 3, v_in3, tw, cols, cnt_128_low,
                     load_multi_cols, is_contiguous_in);
        LOAD_IN_64_S(in_r, in_strides, 4, v_in4, tw, cols, cnt_128_low,
                     load_multi_cols, is_contiguous_in);
        LOAD_IN_64_S(in_r, in_strides, 5, v_in5, tw, cols, cnt_128_low,
                     load_multi_cols, is_contiguous_in);
#endif

        LD_LOW_128_S(in_r, v_in0);

        v_av1 = _mm_add_ps(v_in0, v_in3);
        v_av2 = _mm_add_ps(v_in2, v_in4);
        v_av3 = _mm_add_ps(v_in1, v_in5);
        v_av4 = _mm_add_ps(v_av2, v_av3);
        v_cv1 = _mm_sub_ps(v_av1, _mm_mul_ps(K1, v_av4));

        // Output point 1: X[0]
        v_out0 = _mm_add_ps(v_av1, v_av4);

        v_av1 = _mm_sub_ps(v_av2, v_av3);
        v_av2 = _mm_sub_ps(v_in0, v_in3);
        v_cv2 = _mm_sub_ps(v_av2, _mm_mul_ps(K1, v_av1));

        // Output point 4: X[3]
        v_out3 = _mm_add_ps(v_av2, v_av1);

        v_av1 = _mm_sub_ps(v_in1, v_in5);
        v_av2 = _mm_sub_ps(v_in2, v_in4);
        v_av3 = _mm_sub_ps(v_av1, v_av2);
        v_cv3 = _mm_mul_ps(K2, v_av3);
        v_cv3 = CONJ_128_S(SWAP_RI_128_S(v_cv3));

        // Output point 3: X[2]
        v_out2 = _mm_add_ps(v_cv1, v_cv3);
        // Output point 5: X[4]
        v_out4 = _mm_sub_ps(v_cv1, v_cv3);

        v_av3 = _mm_add_ps(v_av1, v_av2);
        v_cv3 = _mm_mul_ps(K2, v_av3);
        v_cv3 = CONJ_128_S(SWAP_RI_128_S(v_cv3));

        // Output point 2: X[1]
        v_out1 = _mm_add_ps(v_cv2, v_cv3);
        // Output point 6: X[5]
        v_out5 = _mm_sub_ps(v_cv2, v_cv3);

        ST_LOW_128_S(out_r, v_out0);
        STORE_OUT_64_S(out_r, out_strides, 1, v_out1, tw, cols, cnt_128_low,
                       load_multi_cols, is_contiguous_out);
        STORE_OUT_64_S(out_r, out_strides, 2, v_out2, tw, cols, cnt_128_low,
                       load_multi_cols, is_contiguous_out);
#if defined(KERNEL_VARIANT_R2C)
        v_out3 = OUT_H2_128_S(v_out3);
        STORE_OUT_64_S(out_h2_r, out_strides, 3, v_out3, tw, cols, cnt_128_low,
                       load_multi_cols, 0);
        v_out4 = OUT_H2_128_S(v_out4);
        STORE_OUT_64_S(out_h2_r, out_strides, 4, v_out4, tw, cols, cnt_128_low,
                       load_multi_cols, 0);
        v_out5 = OUT_H2_128_S(v_out5);
        STORE_OUT_64_S(out_h2_r, out_strides, 5, v_out5, tw, cols, cnt_128_low,
                       load_multi_cols, 0);
#else
        STORE_OUT_64_S(out_r, out_strides, 3, v_out3, tw, cols, cnt_128_low,
                       load_multi_cols, is_contiguous_out);
        STORE_OUT_64_S(out_r, out_strides, 4, v_out4, tw, cols, cnt_128_low,
                       load_multi_cols, is_contiguous_out);
        STORE_OUT_64_S(out_r, out_strides, 5, v_out5, tw, cols, cnt_128_low,
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
    const DOUBLE CRTM_6[2] = {0.500000000000000000000000000000000000000000000,
                              0.866025403784438646763723170752936183471402627};

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

    VREGTYPE_D v_C1 = BCAST_D(CRTM_6[0]);
    VREGTYPE_D v_C2 = BCAST_D(CRTM_6[1]);

#if defined(KERNEL_DIRECTION_BWD)
    v_C2 = NEG_D(v_C2, 1);
#endif
    for (count = 0; count < N; count++)
    {
        VREGTYPE_D v_in0, v_in1, v_in2, v_in3, v_in4, v_in5;
        VREGTYPE_D v_av1, v_av2, v_av3, v_av4, v_cv1, v_cv2, v_cv3;
        VREGTYPE_D v_out0, v_out1, v_out2, v_out3, v_out4, v_out5;
        INTP col = count * load_multi_cols * NUM_SETS_D;

        LOAD_IN_D(in_r, in_strides, 1, v_in_stride, v_in1, tw, cols, col,
                  load_multi_cols, is_contiguous_in);
        LOAD_IN_D(in_r, in_strides, 2, v_in_stride, v_in2, tw, cols, col,
                  load_multi_cols, is_contiguous_in);
#if defined(KERNEL_VARIANT_C2R)
        LOAD_IN_D(in_h2_r, in_strides, 3, v_in_h2_stride, v_in3, tw, cols, col,
                  load_multi_cols, 0);
        v_in3 = IN_H2_D(v_in3);
        LOAD_IN_D(in_h2_r, in_strides, 4, v_in_h2_stride, v_in4, tw, cols, col,
                  load_multi_cols, 0);
        v_in4 = IN_H2_D(v_in4);
        LOAD_IN_D(in_h2_r, in_strides, 5, v_in_h2_stride, v_in5, tw, cols, col,
                  load_multi_cols, 0);
        v_in5 = IN_H2_D(v_in5);
#else
        LOAD_IN_D(in_r, in_strides, 3, v_in_stride, v_in3, tw, cols, col,
                  load_multi_cols, is_contiguous_in);
        LOAD_IN_D(in_r, in_strides, 4, v_in_stride, v_in4, tw, cols, col,
                  load_multi_cols, is_contiguous_in);
        LOAD_IN_D(in_r, in_strides, 5, v_in_stride, v_in5, tw, cols, col,
                  load_multi_cols, is_contiguous_in);
#endif
        GATHER_D(in_r, v_in_stride, v_in0, is_contiguous_in);

        v_av1 = ADD_D(v_in0, v_in3);
        v_av2 = ADD_D(v_in2, v_in4);
        v_av3 = ADD_D(v_in1, v_in5);
        v_av4 = ADD_D(v_av2, v_av3);
        v_cv1 = SUB_D(v_av1, MUL_D(v_C1, v_av4));

        // Output point 1: X[0]
        v_out0 = ADD_D(v_av1, v_av4);

        v_av1 = SUB_D(v_av2, v_av3);
        v_av2 = SUB_D(v_in0, v_in3);
        v_cv2 = SUB_D(v_av2, MUL_D(v_C1, v_av1));

        // Output point 4: X[3]
        v_out3 = ADD_D(v_av2, v_av1);

        v_av1 = SUB_D(v_in1, v_in5);
        v_av2 = SUB_D(v_in2, v_in4);
        v_av3 = SUB_D(v_av1, v_av2);
        v_cv3 = MUL_D(v_C2, v_av3);
        v_cv3 = CONJ_D(SWAP_RI_D(v_cv3));

        // Output point 3: X[2]
        v_out2 = ADD_D(v_cv1, v_cv3);
        // Output point 5: X[4]
        v_out4 = SUB_D(v_cv1, v_cv3);

        v_av3 = ADD_D(v_av1, v_av2);
        v_cv3 = MUL_D(v_C2, v_av3);
        v_cv3 = CONJ_D(SWAP_RI_D(v_cv3));

        // Output point 2: X[1]
        v_out1 = ADD_D(v_cv2, v_cv3);
        // Output point 6: X[5]
        v_out5 = SUB_D(v_cv2, v_cv3);

        SCATTER_D(out_r, v_out_stride, v_out0, is_contiguous_out);
        STORE_OUT_D(out_r, out_strides, 1, v_out_stride, v_out1, tw, cols, col,
                    load_multi_cols, is_contiguous_out);
        STORE_OUT_D(out_r, out_strides, 2, v_out_stride, v_out2, tw, cols, col,
                    load_multi_cols, is_contiguous_out);
#if defined(KERNEL_VARIANT_R2C)
        v_out3 = OUT_H2_D(v_out3);
        STORE_OUT_D(out_h2_r, out_strides, 3, v_out_h2_stride, v_out3, tw, cols,
                    col, load_multi_cols, 0);
        v_out4 = OUT_H2_D(v_out4);
        STORE_OUT_D(out_h2_r, out_strides, 4, v_out_h2_stride, v_out4, tw, cols,
                    col, load_multi_cols, 0);
        v_out5 = OUT_H2_D(v_out5);
        STORE_OUT_D(out_h2_r, out_strides, 5, v_out_h2_stride, v_out5, tw, cols,
                    col, load_multi_cols, 0);
#else
        STORE_OUT_D(out_r, out_strides, 3, v_out_stride, v_out3, tw, cols, col,
                    load_multi_cols, is_contiguous_out);
        STORE_OUT_D(out_r, out_strides, 4, v_out_stride, v_out4, tw, cols, col,
                    load_multi_cols, is_contiguous_out);
        STORE_OUT_D(out_r, out_strides, 5, v_out_stride, v_out5, tw, cols, col,
                    load_multi_cols, is_contiguous_out);
#endif

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
        __m256d v_in0, v_in1, v_in2, v_in3, v_in4, v_in5;
        __m256d v_av1, v_av2, v_av3, v_av4, v_cv1, v_cv2, v_cv3;
        __m256d v_out0, v_out1, v_out2, v_out3, v_out4, v_out5;

        __m256d K1 = CAST_512_TO_256_D(v_C1);
        __m256d K2 = CAST_512_TO_256_D(v_C2);

        LOAD_IN_256_D(in_r, in_strides, 1, v_in_stride, v_in1, tw, cols,
                      cnt_256, load_multi_cols, is_contiguous_in);
        LOAD_IN_256_D(in_r, in_strides, 2, v_in_stride, v_in2, tw, cols,
                      cnt_256, load_multi_cols, is_contiguous_in);
#if defined(KERNEL_VARIANT_C2R)
        LOAD_IN_256_D(in_h2_r, in_strides, 3, v_in_h2_stride, v_in3, tw, cols,
                      cnt_256, load_multi_cols, 0);
        v_in3 = IN_H2_256_D(v_in3);
        LOAD_IN_256_D(in_h2_r, in_strides, 4, v_in_h2_stride, v_in4, tw, cols,
                      cnt_256, load_multi_cols, 0);
        v_in4 = IN_H2_256_D(v_in4);
        LOAD_IN_256_D(in_h2_r, in_strides, 5, v_in_h2_stride, v_in5, tw, cols,
                      cnt_256, load_multi_cols, 0);
        v_in5 = IN_H2_256_D(v_in5);
#else
        LOAD_IN_256_D(in_r, in_strides, 3, v_in_stride, v_in3, tw, cols,
                      cnt_256, load_multi_cols, is_contiguous_in);
        LOAD_IN_256_D(in_r, in_strides, 4, v_in_stride, v_in4, tw, cols,
                      cnt_256, load_multi_cols, is_contiguous_in);
        LOAD_IN_256_D(in_r, in_strides, 5, v_in_stride, v_in5, tw, cols,
                      cnt_256, load_multi_cols, is_contiguous_in);
#endif
        GATHER2_256_D(in_r, v_in_stride, v_in0, is_contiguous_in);

        v_av1 = _mm256_add_pd(v_in0, v_in3);
        v_av2 = _mm256_add_pd(v_in2, v_in4);
        v_av3 = _mm256_add_pd(v_in1, v_in5);
        v_av4 = _mm256_add_pd(v_av2, v_av3);
        v_cv1 = _mm256_sub_pd(v_av1, _mm256_mul_pd(K1, v_av4));

        // Output point 1: X[0]
        v_out0 = _mm256_add_pd(v_av1, v_av4);

        v_av1 = _mm256_sub_pd(v_av2, v_av3);
        v_av2 = _mm256_sub_pd(v_in0, v_in3);
        v_cv2 = _mm256_sub_pd(v_av2, _mm256_mul_pd(K1, v_av1));

        // Output point 4: X[3]
        v_out3 = _mm256_add_pd(v_av2, v_av1);

        v_av1 = _mm256_sub_pd(v_in1, v_in5);
        v_av2 = _mm256_sub_pd(v_in2, v_in4);
        v_av3 = _mm256_sub_pd(v_av1, v_av2);
        v_cv3 = _mm256_mul_pd(K2, v_av3);
        v_cv3 = CONJ_256_D(SWAP_RI_256_D(v_cv3));

        // Output point 3: X[2]
        v_out2 = _mm256_add_pd(v_cv1, v_cv3);
        // Output point 5: X[4]
        v_out4 = _mm256_sub_pd(v_cv1, v_cv3);

        v_av3 = _mm256_add_pd(v_av1, v_av2);
        v_cv3 = _mm256_mul_pd(K2, v_av3);
        v_cv3 = CONJ_256_D(SWAP_RI_256_D(v_cv3));

        // Output point 2: X[1]
        v_out1 = _mm256_add_pd(v_cv2, v_cv3);
        // Output point 6: X[5]
        v_out5 = _mm256_sub_pd(v_cv2, v_cv3);

        SCATTER2_256_D(out_r, v_out_stride, v_out0, is_contiguous_out);
        STORE_OUT_256_D(out_r, out_strides, 1, v_out_stride, v_out1, tw, cols,
                        cnt_256, load_multi_cols, is_contiguous_out);
        STORE_OUT_256_D(out_r, out_strides, 2, v_out_stride, v_out2, tw, cols,
                        cnt_256, load_multi_cols, is_contiguous_out);
#if defined(KERNEL_VARIANT_R2C)
        v_out3 = OUT_H2_256_D(v_out3);
        STORE_OUT_256_D(out_h2_r, out_strides, 3, v_out_h2_stride, v_out3, tw,
                        cols, cnt_256, load_multi_cols, 0);
        v_out4 = OUT_H2_256_D(v_out4);
        STORE_OUT_256_D(out_h2_r, out_strides, 4, v_out_h2_stride, v_out4, tw,
                        cols, cnt_256, load_multi_cols, 0);
        v_out5 = OUT_H2_256_D(v_out5);
        STORE_OUT_256_D(out_h2_r, out_strides, 5, v_out_h2_stride, v_out5, tw,
                        cols, cnt_256, load_multi_cols, 0);
#else
        STORE_OUT_256_D(out_r, out_strides, 3, v_out_stride, v_out3, tw, cols,
                        cnt_256, load_multi_cols, is_contiguous_out);
        STORE_OUT_256_D(out_r, out_strides, 4, v_out_stride, v_out4, tw, cols,
                        cnt_256, load_multi_cols, is_contiguous_out);
        STORE_OUT_256_D(out_r, out_strides, 5, v_out_stride, v_out5, tw, cols,
                        cnt_256, load_multi_cols, is_contiguous_out);
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
        __m128d v_in0, v_in1, v_in2, v_in3, v_in4, v_in5;
        __m128d v_av1, v_av2, v_av3, v_av4, v_cv1, v_cv2, v_cv3;
        __m128d v_out0, v_out1, v_out2, v_out3, v_out4, v_out5;

#if defined(KERNEL_USE_AVX512)
        __m128d K1 = CAST_512_TO_128_D(v_C1);
        __m128d K2 = CAST_512_TO_128_D(v_C2);
#elif defined(KERNEL_USE_AVX256)
        __m128d K1 = CAST_256_TO_128_D(v_C1);
        __m128d K2 = CAST_256_TO_128_D(v_C2);
#endif

        LOAD_IN_128_D(in_r, in_strides, 1, /* unused */ 0, v_in1, tw, cols,
                      cnt_128, load_multi_cols, is_contiguous_in);
        LOAD_IN_128_D(in_r, in_strides, 2, /* unused */ 0, v_in2, tw, cols,
                      cnt_128, load_multi_cols, is_contiguous_in);
#if defined(KERNEL_VARIANT_C2R)
        LOAD_IN_128_D(in_h2_r, in_strides, 3, /* unused */ 0, v_in3, tw, cols,
                      cnt_128, load_multi_cols, 0);
        v_in3 = IN_H2_128_D(v_in3);
        LOAD_IN_128_D(in_h2_r, in_strides, 4, /* unused */ 0, v_in4, tw, cols,
                      cnt_128, load_multi_cols, 0);
        v_in4 = IN_H2_128_D(v_in4);
        LOAD_IN_128_D(in_h2_r, in_strides, 5, /* unused */ 0, v_in5, tw, cols,
                      cnt_128, load_multi_cols, 0);
        v_in5 = IN_H2_128_D(v_in5);
#else
        LOAD_IN_128_D(in_r, in_strides, 3, /* unused */ 0, v_in3, tw, cols,
                      cnt_128, load_multi_cols, is_contiguous_in);
        LOAD_IN_128_D(in_r, in_strides, 4, /* unused */ 0, v_in4, tw, cols,
                      cnt_128, load_multi_cols, is_contiguous_in);
        LOAD_IN_128_D(in_r, in_strides, 5, /* unused */ 0, v_in5, tw, cols,
                      cnt_128, load_multi_cols, is_contiguous_in);
#endif

        LD_128_D(in_r, v_in0);

        v_av1 = _mm_add_pd(v_in0, v_in3);
        v_av2 = _mm_add_pd(v_in2, v_in4);
        v_av3 = _mm_add_pd(v_in1, v_in5);
        v_av4 = _mm_add_pd(v_av2, v_av3);
        v_cv1 = _mm_sub_pd(v_av1, _mm_mul_pd(K1, v_av4));

        // Output point 1: X[0]
        v_out0 = _mm_add_pd(v_av1, v_av4);

        v_av1 = _mm_sub_pd(v_av2, v_av3);
        v_av2 = _mm_sub_pd(v_in0, v_in3);
        v_cv2 = _mm_sub_pd(v_av2, _mm_mul_pd(K1, v_av1));

        // Output point 4: X[3]
        v_out3 = _mm_add_pd(v_av2, v_av1);

        v_av1 = _mm_sub_pd(v_in1, v_in5);
        v_av2 = _mm_sub_pd(v_in2, v_in4);
        v_av3 = _mm_sub_pd(v_av1, v_av2);
        v_cv3 = _mm_mul_pd(K2, v_av3);
        v_cv3 = CONJ_128_D(SWAP_RI_128_D(v_cv3));

        // Output point 3: X[2]
        v_out2 = _mm_add_pd(v_cv1, v_cv3);
        // Output point 5: X[4]
        v_out4 = _mm_sub_pd(v_cv1, v_cv3);

        v_av3 = _mm_add_pd(v_av1, v_av2);
        v_cv3 = _mm_mul_pd(K2, v_av3);
        v_cv3 = CONJ_128_D(SWAP_RI_128_D(v_cv3));

        // Output point 2: X[1]
        v_out1 = _mm_add_pd(v_cv2, v_cv3);
        // Output point 6: X[5]
        v_out5 = _mm_sub_pd(v_cv2, v_cv3);

        ST_128_D(out_r, v_out0);
        STORE_OUT_128_D(out_r, out_strides, 1, /* unused */ 0, v_out1, tw, cols,
                        cnt_128, load_multi_cols, is_contiguous_out);
        STORE_OUT_128_D(out_r, out_strides, 2, /* unused */ 0, v_out2, tw, cols,
                        cnt_128, load_multi_cols, is_contiguous_out);
#if defined(KERNEL_VARIANT_R2C)
        v_out3 = OUT_H2_128_D(v_out3);
        STORE_OUT_128_D(out_h2_r, out_strides, 3, /* unused */ 0, v_out3, tw,
                        cols, cnt_128, load_multi_cols, 0);
        v_out4 = OUT_H2_128_D(v_out4);
        STORE_OUT_128_D(out_h2_r, out_strides, 4, /* unused */ 0, v_out4, tw,
                        cols, cnt_128, load_multi_cols, 0);
        v_out5 = OUT_H2_128_D(v_out5);
        STORE_OUT_128_D(out_h2_r, out_strides, 5, /* unused */ 0, v_out5, tw,
                        cols, cnt_128, load_multi_cols, 0);
#else
        STORE_OUT_128_D(out_r, out_strides, 3, /* unused */ 0, v_out3, tw, cols,
                        cnt_128, load_multi_cols, is_contiguous_out);
        STORE_OUT_128_D(out_r, out_strides, 4, /* unused */ 0, v_out4, tw, cols,
                        cnt_128, load_multi_cols, is_contiguous_out);
        STORE_OUT_128_D(out_r, out_strides, 5, /* unused */ 0, v_out5, tw, cols,
                        cnt_128, load_multi_cols, is_contiguous_out);
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
