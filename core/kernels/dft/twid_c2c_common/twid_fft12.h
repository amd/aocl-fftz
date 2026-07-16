// Copyright Advanced Micro Devices, Inc.
// SPDX-License-Identifier: BSD-3-Clause

/** @file twid_fft12.h
 *
 *  @brief The ISA generic kernel template for the radix 12 twiddle kernel
 *
 *  This file contains the DIT twiddle radix-12 FFT implementations for
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

    const FLOAT CRTM_12[3] = {
        0.86602540378443864676372317075293618347140262700000f,
        0.50000000000000000000000000000000000000000000000000f,
        1.00000000000000000000000000000000000000000000000000f};

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

    VREGTYPE_S v_C1 = BCAST_S(CRTM_12[0]);
    VREGTYPE_S v_C2 = BCAST_S(CRTM_12[1]);
    VREGTYPE_S v_C3 = BCAST_S(CRTM_12[2]);
    VREGTYPE_S v_C4 = BCAST_S(CRTM_12[0]);
    VREGTYPE_S v_C5 = BCAST_S(CRTM_12[1]);

    INTP count;

#if defined(KERNEL_DIRECTION_BWD)
    v_C3 = NEG_S(v_C3, 1);
    v_C4 = NEG_S(v_C4, 1);
    v_C5 = NEG_S(v_C5, 1);
#endif
    for (count = 0; count < N; count++)
    {
        VREGTYPE_S v_in0, v_in1, v_in2, v_in3, v_in4, v_in5, v_in6, v_in7,
            v_in8, v_in9, v_in10, v_in11;
        VREGTYPE_S v_av1, v_av2, v_av3, v_av4, v_av5, v_av6, v_av7, v_av8,
            v_av9, v_av10, v_av11, v_av12;
        VREGTYPE_S v_cv1, v_cv2, v_cv3, v_cv4, v_cv5, v_cv6, v_cv7, v_cv8;
        VREGTYPE_S v_tv1, v_tv2, v_tv3, v_tv4, v_tv5;
        VREGTYPE_S v_out0, v_out1, v_out2, v_out3, v_out4, v_out5, v_out6,
            v_out7, v_out8, v_out9, v_out10, v_out11;
        INTP col = count * load_multi_cols * NUM_SETS_S;

        LOAD_IN_S(in_r, in_strides, 1, v_in_stride, v_in1, tw, cols, col,
                  load_multi_cols, is_contiguous_in);
        LOAD_IN_S(in_r, in_strides, 2, v_in_stride, v_in2, tw, cols, col,
                  load_multi_cols, is_contiguous_in);
        LOAD_IN_S(in_r, in_strides, 3, v_in_stride, v_in3, tw, cols, col,
                  load_multi_cols, is_contiguous_in);
        LOAD_IN_S(in_r, in_strides, 4, v_in_stride, v_in4, tw, cols, col,
                  load_multi_cols, is_contiguous_in);
        LOAD_IN_S(in_r, in_strides, 5, v_in_stride, v_in5, tw, cols, col,
                  load_multi_cols, is_contiguous_in);
#if defined(KERNEL_VARIANT_C2R)
        LOAD_IN_S(in_h2_r, in_strides, 6, v_in_h2_stride, v_in6, tw, cols, col,
                  load_multi_cols, 0);
        v_in6 = IN_H2_S(v_in6);
        LOAD_IN_S(in_h2_r, in_strides, 7, v_in_h2_stride, v_in7, tw, cols, col,
                  load_multi_cols, 0);
        v_in7 = IN_H2_S(v_in7);
        LOAD_IN_S(in_h2_r, in_strides, 8, v_in_h2_stride, v_in8, tw, cols, col,
                  load_multi_cols, 0);
        v_in8 = IN_H2_S(v_in8);
        LOAD_IN_S(in_h2_r, in_strides, 9, v_in_h2_stride, v_in9, tw, cols, col,
                  load_multi_cols, 0);
        v_in9 = IN_H2_S(v_in9);
        LOAD_IN_S(in_h2_r, in_strides, 10, v_in_h2_stride, v_in10, tw, cols,
                  col, load_multi_cols, 0);
        v_in10 = IN_H2_S(v_in10);
        LOAD_IN_S(in_h2_r, in_strides, 11, v_in_h2_stride, v_in11, tw, cols,
                  col, load_multi_cols, 0);
        v_in11 = IN_H2_S(v_in11);
#else
        LOAD_IN_S(in_r, in_strides, 6, v_in_stride, v_in6, tw, cols, col,
                  load_multi_cols, is_contiguous_in);
        LOAD_IN_S(in_r, in_strides, 7, v_in_stride, v_in7, tw, cols, col,
                  load_multi_cols, is_contiguous_in);
        LOAD_IN_S(in_r, in_strides, 8, v_in_stride, v_in8, tw, cols, col,
                  load_multi_cols, is_contiguous_in);
        LOAD_IN_S(in_r, in_strides, 9, v_in_stride, v_in9, tw, cols, col,
                  load_multi_cols, is_contiguous_in);
        LOAD_IN_S(in_r, in_strides, 10, v_in_stride, v_in10, tw, cols, col,
                  load_multi_cols, is_contiguous_in);
        LOAD_IN_S(in_r, in_strides, 11, v_in_stride, v_in11, tw, cols, col,
                  load_multi_cols, is_contiguous_in);
#endif
        GATHER_S(in_r, v_in_stride, v_in0, is_contiguous_in);

        // Common operations
        v_av1 = ADD_S(v_in0, v_in6);
        v_av2 = ADD_S(v_in2, v_in4);
        v_av3 = ADD_S(v_in8, v_in10);
        v_av4 = ADD_S(v_in1, v_in5);
        v_av5 = ADD_S(v_in7, v_in11);
        v_av6 = ADD_S(v_in3, v_in9);

        v_cv1 = ADD_S(v_av2, v_av3);
        v_cv2 = ADD_S(v_av4, v_av5);
        v_cv3 = ADD_S(v_av1, v_av6);
        v_cv4 = SUB_S(v_av1, v_av6);

        v_cv5 = ADD_S(v_cv1, v_cv2);
        v_cv6 = SUB_S(v_cv1, v_cv2);

        // output point 1 & 7
        v_out0 = ADD_S(v_cv3, v_cv5);
        v_out6 = ADD_S(v_cv4, v_cv6);

        // Common operations
        v_av7 = SUB_S(v_in0, v_in6);
        v_av8 = SUB_S(v_in2, v_in4);
        v_av9 = SUB_S(v_in8, v_in10);
        v_av10 = SUB_S(v_in1, v_in5);
        v_av11 = SUB_S(v_in7, v_in11);
        v_av12 = SUB_S(v_in3, v_in9);

        v_cv1 = SUB_S(v_av8, v_av9);
        v_cv2 = SUB_S(v_av4, v_av5);
        v_cv7 = SUB_S(v_cv2, v_av12);
        v_cv8 = SUB_S(v_av7, v_cv1);

        v_tv1 = MUL_S(v_C3, v_cv7);
        v_tv1 = SWAP_RI_S(CONJ_S(v_tv1));

        // output point 4 & 10
        v_out3 = SUB_S(v_cv8, v_tv1);
        v_out9 = ADD_S(v_cv8, v_tv1);

        v_tv1 = MUL_S(v_C3, v_av12);
        v_tv1 = CONJ_S(v_tv1);

        v_tv2 = MUL_S(v_C2, v_cv1);
        v_cv1 = ADD_S(v_av7, v_tv2);
        v_cv7 = SUB_S(v_av10, v_av11);

        v_tv3 = MUL_S(v_C1, v_cv7);
        v_cv8 = ADD_S(v_cv1, v_tv3);

        v_tv4 = MUL_S(v_C5, v_cv2);
        v_tv4 = CONJ_S(v_tv4);

        v_cv2 = SUB_S(v_av2, v_av3);
        v_tv5 = MUL_S(v_C4, v_cv2);
        v_tv5 = CONJ_S(v_tv5);

        v_cv2 = ADD_S(v_tv1, v_tv4);
        v_cv7 = ADD_S(v_cv2, v_tv5);
        v_cv7 = SWAP_RI_S(v_cv7);

        // output point 2 & 12
        v_out1 = SUB_S(v_cv8, v_cv7);
        v_out11 = ADD_S(v_cv8, v_cv7);

        v_cv7 = SUB_S(v_cv1, v_tv3);
        v_cv8 = SUB_S(v_cv2, v_tv5);
        v_cv8 = SWAP_RI_S(v_cv8);

        // output point 6 & 8
        v_out5 = SUB_S(v_cv7, v_cv8);
        v_out7 = ADD_S(v_cv7, v_cv8);

        v_tv1 = MUL_S(v_C2, v_cv6);
        v_cv1 = SUB_S(v_cv4, v_tv1);
        v_cv2 = ADD_S(v_av8, v_av9);
        v_cv4 = ADD_S(v_av10, v_av11);
        v_cv6 = ADD_S(v_cv4, v_cv2);
        v_tv2 = MUL_S(v_C4, v_cv6);
        v_tv2 = SWAP_RI_S(CONJ_S(v_tv2));

        // output point 3 & 11
        v_out2 = SUB_S(v_cv1, v_tv2);
        v_out10 = ADD_S(v_cv1, v_tv2);

        v_tv1 = MUL_S(v_C2, v_cv5);
        v_cv1 = SUB_S(v_cv3, v_tv1);
        v_cv6 = SUB_S(v_cv4, v_cv2);
        v_tv2 = MUL_S(v_C4, v_cv6);
        v_tv2 = SWAP_RI_S(CONJ_S(v_tv2));

        // output point 5 & 9
        v_out4 = SUB_S(v_cv1, v_tv2);
        v_out8 = ADD_S(v_cv1, v_tv2);

        SCATTER_S(out_r, v_out_stride, v_out0, is_contiguous_out);
        STORE_OUT_S(out_r, out_strides, 1, v_out_stride, v_out1, tw, cols, col,
                    load_multi_cols, is_contiguous_out);
        STORE_OUT_S(out_r, out_strides, 2, v_out_stride, v_out2, tw, cols, col,
                    load_multi_cols, is_contiguous_out);
        STORE_OUT_S(out_r, out_strides, 3, v_out_stride, v_out3, tw, cols, col,
                    load_multi_cols, is_contiguous_out);
        STORE_OUT_S(out_r, out_strides, 4, v_out_stride, v_out4, tw, cols, col,
                    load_multi_cols, is_contiguous_out);
        STORE_OUT_S(out_r, out_strides, 5, v_out_stride, v_out5, tw, cols, col,
                    load_multi_cols, is_contiguous_out);
#if defined(KERNEL_VARIANT_R2C)
        v_out6 = OUT_H2_S(v_out6);
        STORE_OUT_S(out_h2_r, out_strides, 6, v_out_h2_stride, v_out6, tw, cols,
                    col, load_multi_cols, 0);
        v_out7 = OUT_H2_S(v_out7);
        STORE_OUT_S(out_h2_r, out_strides, 7, v_out_h2_stride, v_out7, tw, cols,
                    col, load_multi_cols, 0);
        v_out8 = OUT_H2_S(v_out8);
        STORE_OUT_S(out_h2_r, out_strides, 8, v_out_h2_stride, v_out8, tw, cols,
                    col, load_multi_cols, 0);
        v_out9 = OUT_H2_S(v_out9);
        STORE_OUT_S(out_h2_r, out_strides, 9, v_out_h2_stride, v_out9, tw, cols,
                    col, load_multi_cols, 0);
        v_out10 = OUT_H2_S(v_out10);
        STORE_OUT_S(out_h2_r, out_strides, 10, v_out_h2_stride, v_out10, tw,
                    cols, col, load_multi_cols, 0);
        v_out11 = OUT_H2_S(v_out11);
        STORE_OUT_S(out_h2_r, out_strides, 11, v_out_h2_stride, v_out11, tw,
                    cols, col, load_multi_cols, 0);
#else
        STORE_OUT_S(out_r, out_strides, 6, v_out_stride, v_out6, tw, cols, col,
                    load_multi_cols, is_contiguous_out);
        STORE_OUT_S(out_r, out_strides, 7, v_out_stride, v_out7, tw, cols, col,
                    load_multi_cols, is_contiguous_out);
        STORE_OUT_S(out_r, out_strides, 8, v_out_stride, v_out8, tw, cols, col,
                    load_multi_cols, is_contiguous_out);
        STORE_OUT_S(out_r, out_strides, 9, v_out_stride, v_out9, tw, cols, col,
                    load_multi_cols, is_contiguous_out);
        STORE_OUT_S(out_r, out_strides, 10, v_out_stride, v_out10, tw, cols,
                    col, load_multi_cols, is_contiguous_out);
        STORE_OUT_S(out_r, out_strides, 11, v_out_stride, v_out11, tw, cols,
                    col, load_multi_cols, is_contiguous_out);
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
        __m256 v_in0, v_in1, v_in2, v_in3, v_in4, v_in5, v_in6, v_in7, v_in8,
            v_in9, v_in10, v_in11;
        __m256 v_av1, v_av2, v_av3, v_av4, v_av5, v_av6, v_av7, v_av8, v_av9,
            v_av10, v_av11, v_av12;
        __m256 v_cv1, v_cv2, v_cv3, v_cv4, v_cv5, v_cv6, v_cv7, v_cv8;
        __m256 v_tv1, v_tv2, v_tv3, v_tv4, v_tv5;
        __m256 v_out0, v_out1, v_out2, v_out3, v_out4, v_out5, v_out6, v_out7,
            v_out8, v_out9, v_out10, v_out11;

        __m256 v_K1 = CAST_512_TO_256_S(v_C1);
        __m256 v_K2 = CAST_512_TO_256_S(v_C2);
        __m256 v_K3 = CAST_512_TO_256_S(v_C3);
        __m256 v_K4 = CAST_512_TO_256_S(v_C4);
        __m256 v_K5 = CAST_512_TO_256_S(v_C5);

        LOAD_IN_256_S(in_r, in_strides, 1, v_in_stride, v_in1, tw, cols,
                      cnt_256, load_multi_cols, is_contiguous_in);
        LOAD_IN_256_S(in_r, in_strides, 2, v_in_stride, v_in2, tw, cols,
                      cnt_256, load_multi_cols, is_contiguous_in);
        LOAD_IN_256_S(in_r, in_strides, 3, v_in_stride, v_in3, tw, cols,
                      cnt_256, load_multi_cols, is_contiguous_in);
        LOAD_IN_256_S(in_r, in_strides, 4, v_in_stride, v_in4, tw, cols,
                      cnt_256, load_multi_cols, is_contiguous_in);
        LOAD_IN_256_S(in_r, in_strides, 5, v_in_stride, v_in5, tw, cols,
                      cnt_256, load_multi_cols, is_contiguous_in);
#if defined(KERNEL_VARIANT_C2R)
        LOAD_IN_256_S(in_h2_r, in_strides, 6, v_in_h2_stride, v_in6, tw, cols,
                      cnt_256, load_multi_cols, 0);
        v_in6 = IN_H2_256_S(v_in6);
        LOAD_IN_256_S(in_h2_r, in_strides, 7, v_in_h2_stride, v_in7, tw, cols,
                      cnt_256, load_multi_cols, 0);
        v_in7 = IN_H2_256_S(v_in7);
        LOAD_IN_256_S(in_h2_r, in_strides, 8, v_in_h2_stride, v_in8, tw, cols,
                      cnt_256, load_multi_cols, 0);
        v_in8 = IN_H2_256_S(v_in8);
        LOAD_IN_256_S(in_h2_r, in_strides, 9, v_in_h2_stride, v_in9, tw, cols,
                      cnt_256, load_multi_cols, 0);
        v_in9 = IN_H2_256_S(v_in9);
        LOAD_IN_256_S(in_h2_r, in_strides, 10, v_in_h2_stride, v_in10, tw, cols,
                      cnt_256, load_multi_cols, 0);
        v_in10 = IN_H2_256_S(v_in10);
        LOAD_IN_256_S(in_h2_r, in_strides, 11, v_in_h2_stride, v_in11, tw, cols,
                      cnt_256, load_multi_cols, 0);
        v_in11 = IN_H2_256_S(v_in11);
#else
        LOAD_IN_256_S(in_r, in_strides, 6, v_in_stride, v_in6, tw, cols,
                      cnt_256, load_multi_cols, is_contiguous_in);
        LOAD_IN_256_S(in_r, in_strides, 7, v_in_stride, v_in7, tw, cols,
                      cnt_256, load_multi_cols, is_contiguous_in);
        LOAD_IN_256_S(in_r, in_strides, 8, v_in_stride, v_in8, tw, cols,
                      cnt_256, load_multi_cols, is_contiguous_in);
        LOAD_IN_256_S(in_r, in_strides, 9, v_in_stride, v_in9, tw, cols,
                      cnt_256, load_multi_cols, is_contiguous_in);
        LOAD_IN_256_S(in_r, in_strides, 10, v_in_stride, v_in10, tw, cols,
                      cnt_256, load_multi_cols, is_contiguous_in);
        LOAD_IN_256_S(in_r, in_strides, 11, v_in_stride, v_in11, tw, cols,
                      cnt_256, load_multi_cols, is_contiguous_in);
#endif
        GATHER4_256_S(in_r, v_in_stride, v_in0, is_contiguous_in);

        // Common operations
        v_av1 = _mm256_add_ps(v_in0, v_in6);
        v_av2 = _mm256_add_ps(v_in2, v_in4);
        v_av3 = _mm256_add_ps(v_in8, v_in10);
        v_av4 = _mm256_add_ps(v_in1, v_in5);
        v_av5 = _mm256_add_ps(v_in7, v_in11);
        v_av6 = _mm256_add_ps(v_in3, v_in9);

        v_cv1 = _mm256_add_ps(v_av2, v_av3);
        v_cv2 = _mm256_add_ps(v_av4, v_av5);
        v_cv3 = _mm256_add_ps(v_av1, v_av6);
        v_cv4 = _mm256_sub_ps(v_av1, v_av6);

        v_cv5 = _mm256_add_ps(v_cv1, v_cv2);
        v_cv6 = _mm256_sub_ps(v_cv1, v_cv2);

        // output point 1 & 7
        v_out0 = _mm256_add_ps(v_cv3, v_cv5);
        v_out6 = _mm256_add_ps(v_cv4, v_cv6);

        // Common operations
        v_av7 = _mm256_sub_ps(v_in0, v_in6);
        v_av8 = _mm256_sub_ps(v_in2, v_in4);
        v_av9 = _mm256_sub_ps(v_in8, v_in10);
        v_av10 = _mm256_sub_ps(v_in1, v_in5);
        v_av11 = _mm256_sub_ps(v_in7, v_in11);
        v_av12 = _mm256_sub_ps(v_in3, v_in9);

        v_cv1 = _mm256_sub_ps(v_av8, v_av9);
        v_cv2 = _mm256_sub_ps(v_av4, v_av5);
        v_cv7 = _mm256_sub_ps(v_cv2, v_av12);
        v_cv8 = _mm256_sub_ps(v_av7, v_cv1);

        v_tv1 = _mm256_mul_ps(v_K3, v_cv7);
        v_tv1 = SWAP_RI_256_S(CONJ_256_S(v_tv1));

        // output point 4 & 10
        v_out3 = _mm256_sub_ps(v_cv8, v_tv1);
        v_out9 = _mm256_add_ps(v_cv8, v_tv1);

        v_tv1 = _mm256_mul_ps(v_K3, v_av12);
        v_tv1 = CONJ_256_S(v_tv1);

        v_tv2 = _mm256_mul_ps(v_K2, v_cv1);
        v_cv1 = _mm256_add_ps(v_av7, v_tv2);
        v_cv7 = _mm256_sub_ps(v_av10, v_av11);

        v_tv3 = _mm256_mul_ps(v_K1, v_cv7);
        v_cv8 = _mm256_add_ps(v_cv1, v_tv3);

        v_tv4 = _mm256_mul_ps(v_K5, v_cv2);
        v_tv4 = CONJ_256_S(v_tv4);

        v_cv2 = _mm256_sub_ps(v_av2, v_av3);
        v_tv5 = _mm256_mul_ps(v_K4, v_cv2);
        v_tv5 = CONJ_256_S(v_tv5);

        v_cv2 = _mm256_add_ps(v_tv1, v_tv4);
        v_cv7 = _mm256_add_ps(v_cv2, v_tv5);
        v_cv7 = SWAP_RI_256_S(v_cv7);

        // output point 2 & 12
        v_out1 = _mm256_sub_ps(v_cv8, v_cv7);
        v_out11 = _mm256_add_ps(v_cv8, v_cv7);

        v_cv7 = _mm256_sub_ps(v_cv1, v_tv3);
        v_cv8 = _mm256_sub_ps(v_cv2, v_tv5);
        v_cv8 = SWAP_RI_256_S(v_cv8);

        // output point 6 & 8
        v_out5 = _mm256_sub_ps(v_cv7, v_cv8);
        v_out7 = _mm256_add_ps(v_cv7, v_cv8);

        v_tv1 = _mm256_mul_ps(v_K2, v_cv6);
        v_cv1 = _mm256_sub_ps(v_cv4, v_tv1);
        v_cv2 = _mm256_add_ps(v_av8, v_av9);
        v_cv4 = _mm256_add_ps(v_av10, v_av11);
        v_cv6 = _mm256_add_ps(v_cv4, v_cv2);
        v_tv2 = _mm256_mul_ps(v_K4, v_cv6);
        v_tv2 = SWAP_RI_256_S(CONJ_256_S(v_tv2));

        // output point 3 & 11
        v_out2 = _mm256_sub_ps(v_cv1, v_tv2);
        v_out10 = _mm256_add_ps(v_cv1, v_tv2);

        v_tv1 = _mm256_mul_ps(v_K2, v_cv5);
        v_cv1 = _mm256_sub_ps(v_cv3, v_tv1);
        v_cv6 = _mm256_sub_ps(v_cv4, v_cv2);
        v_tv2 = _mm256_mul_ps(v_K4, v_cv6);
        v_tv2 = SWAP_RI_256_S(CONJ_256_S(v_tv2));

        // output point 5 & 9
        v_out4 = _mm256_sub_ps(v_cv1, v_tv2);
        v_out8 = _mm256_add_ps(v_cv1, v_tv2);

        SCATTER4_256_S(out_r, v_out_stride, v_out0, is_contiguous_out);
        STORE_OUT_256_S(out_r, out_strides, 1, v_out_stride, v_out1, tw, cols,
                        cnt_256, load_multi_cols, is_contiguous_out);
        STORE_OUT_256_S(out_r, out_strides, 2, v_out_stride, v_out2, tw, cols,
                        cnt_256, load_multi_cols, is_contiguous_out);
        STORE_OUT_256_S(out_r, out_strides, 3, v_out_stride, v_out3, tw, cols,
                        cnt_256, load_multi_cols, is_contiguous_out);
        STORE_OUT_256_S(out_r, out_strides, 4, v_out_stride, v_out4, tw, cols,
                        cnt_256, load_multi_cols, is_contiguous_out);
        STORE_OUT_256_S(out_r, out_strides, 5, v_out_stride, v_out5, tw, cols,
                        cnt_256, load_multi_cols, is_contiguous_out);
#if defined(KERNEL_VARIANT_R2C)
        v_out6 = OUT_H2_256_S(v_out6);
        STORE_OUT_256_S(out_h2_r, out_strides, 6, v_out_h2_stride, v_out6, tw,
                        cols, cnt_256, load_multi_cols, 0);
        v_out7 = OUT_H2_256_S(v_out7);
        STORE_OUT_256_S(out_h2_r, out_strides, 7, v_out_h2_stride, v_out7, tw,
                        cols, cnt_256, load_multi_cols, 0);
        v_out8 = OUT_H2_256_S(v_out8);
        STORE_OUT_256_S(out_h2_r, out_strides, 8, v_out_h2_stride, v_out8, tw,
                        cols, cnt_256, load_multi_cols, 0);
        v_out9 = OUT_H2_256_S(v_out9);
        STORE_OUT_256_S(out_h2_r, out_strides, 9, v_out_h2_stride, v_out9, tw,
                        cols, cnt_256, load_multi_cols, 0);
        v_out10 = OUT_H2_256_S(v_out10);
        STORE_OUT_256_S(out_h2_r, out_strides, 10, v_out_h2_stride, v_out10, tw,
                        cols, cnt_256, load_multi_cols, 0);
        v_out11 = OUT_H2_256_S(v_out11);
        STORE_OUT_256_S(out_h2_r, out_strides, 11, v_out_h2_stride, v_out11, tw,
                        cols, cnt_256, load_multi_cols, 0);
#else
        STORE_OUT_256_S(out_r, out_strides, 6, v_out_stride, v_out6, tw, cols,
                        cnt_256, load_multi_cols, is_contiguous_out);
        STORE_OUT_256_S(out_r, out_strides, 7, v_out_stride, v_out7, tw, cols,
                        cnt_256, load_multi_cols, is_contiguous_out);
        STORE_OUT_256_S(out_r, out_strides, 8, v_out_stride, v_out8, tw, cols,
                        cnt_256, load_multi_cols, is_contiguous_out);
        STORE_OUT_256_S(out_r, out_strides, 9, v_out_stride, v_out9, tw, cols,
                        cnt_256, load_multi_cols, is_contiguous_out);
        STORE_OUT_256_S(out_r, out_strides, 10, v_out_stride, v_out10, tw, cols,
                        cnt_256, load_multi_cols, is_contiguous_out);
        STORE_OUT_256_S(out_r, out_strides, 11, v_out_stride, v_out11, tw, cols,
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
        __m128 v_in0, v_in1, v_in2, v_in3, v_in4, v_in5, v_in6, v_in7, v_in8,
            v_in9, v_in10, v_in11;
        __m128 v_av1, v_av2, v_av3, v_av4, v_av5, v_av6, v_av7, v_av8, v_av9,
            v_av10, v_av11, v_av12;
        __m128 v_cv1, v_cv2, v_cv3, v_cv4, v_cv5, v_cv6, v_cv7, v_cv8;
        __m128 v_tv1, v_tv2, v_tv3, v_tv4, v_tv5;
        __m128 v_out0, v_out1, v_out2, v_out3, v_out4, v_out5, v_out6, v_out7,
            v_out8, v_out9, v_out10, v_out11;

#if defined(KERNEL_USE_AVX512)
        __m128 v_K1 = CAST_512_TO_128_S(v_C1);
        __m128 v_K2 = CAST_512_TO_128_S(v_C2);
        __m128 v_K3 = CAST_512_TO_128_S(v_C3);
        __m128 v_K4 = CAST_512_TO_128_S(v_C4);
        __m128 v_K5 = CAST_512_TO_128_S(v_C5);
#elif defined(KERNEL_USE_AVX256)
        __m128 v_K1 = CAST_256_TO_128_S(v_C1);
        __m128 v_K2 = CAST_256_TO_128_S(v_C2);
        __m128 v_K3 = CAST_256_TO_128_S(v_C3);
        __m128 v_K4 = CAST_256_TO_128_S(v_C4);
        __m128 v_K5 = CAST_256_TO_128_S(v_C5);
#endif

        LOAD_IN_128_S(in_r, in_strides, 1, v_in_stride, v_in1, tw, cols,
                      cnt_128, load_multi_cols, is_contiguous_in);
        LOAD_IN_128_S(in_r, in_strides, 2, v_in_stride, v_in2, tw, cols,
                      cnt_128, load_multi_cols, is_contiguous_in);
        LOAD_IN_128_S(in_r, in_strides, 3, v_in_stride, v_in3, tw, cols,
                      cnt_128, load_multi_cols, is_contiguous_in);
        LOAD_IN_128_S(in_r, in_strides, 4, v_in_stride, v_in4, tw, cols,
                      cnt_128, load_multi_cols, is_contiguous_in);
        LOAD_IN_128_S(in_r, in_strides, 5, v_in_stride, v_in5, tw, cols,
                      cnt_128, load_multi_cols, is_contiguous_in);
#if defined(KERNEL_VARIANT_C2R)
        LOAD_IN_128_S(in_h2_r, in_strides, 6, v_in_h2_stride, v_in6, tw, cols,
                      cnt_128, load_multi_cols, 0);
        v_in6 = IN_H2_128_S(v_in6);
        LOAD_IN_128_S(in_h2_r, in_strides, 7, v_in_h2_stride, v_in7, tw, cols,
                      cnt_128, load_multi_cols, 0);
        v_in7 = IN_H2_128_S(v_in7);
        LOAD_IN_128_S(in_h2_r, in_strides, 8, v_in_h2_stride, v_in8, tw, cols,
                      cnt_128, load_multi_cols, 0);
        v_in8 = IN_H2_128_S(v_in8);
        LOAD_IN_128_S(in_h2_r, in_strides, 9, v_in_h2_stride, v_in9, tw, cols,
                      cnt_128, load_multi_cols, 0);
        v_in9 = IN_H2_128_S(v_in9);
        LOAD_IN_128_S(in_h2_r, in_strides, 10, v_in_h2_stride, v_in10, tw, cols,
                      cnt_128, load_multi_cols, 0);
        v_in10 = IN_H2_128_S(v_in10);
        LOAD_IN_128_S(in_h2_r, in_strides, 11, v_in_h2_stride, v_in11, tw, cols,
                      cnt_128, load_multi_cols, 0);
        v_in11 = IN_H2_128_S(v_in11);
#else
        LOAD_IN_128_S(in_r, in_strides, 6, v_in_stride, v_in6, tw, cols,
                      cnt_128, load_multi_cols, is_contiguous_in);
        LOAD_IN_128_S(in_r, in_strides, 7, v_in_stride, v_in7, tw, cols,
                      cnt_128, load_multi_cols, is_contiguous_in);
        LOAD_IN_128_S(in_r, in_strides, 8, v_in_stride, v_in8, tw, cols,
                      cnt_128, load_multi_cols, is_contiguous_in);
        LOAD_IN_128_S(in_r, in_strides, 9, v_in_stride, v_in9, tw, cols,
                      cnt_128, load_multi_cols, is_contiguous_in);
        LOAD_IN_128_S(in_r, in_strides, 10, v_in_stride, v_in10, tw, cols,
                      cnt_128, load_multi_cols, is_contiguous_in);
        LOAD_IN_128_S(in_r, in_strides, 11, v_in_stride, v_in11, tw, cols,
                      cnt_128, load_multi_cols, is_contiguous_in);
#endif
        GATHER2_128_S(in_r, v_in_stride, v_in0, is_contiguous_in);

        // Common operations
        v_av1 = _mm_add_ps(v_in0, v_in6);
        v_av2 = _mm_add_ps(v_in2, v_in4);
        v_av3 = _mm_add_ps(v_in8, v_in10);
        v_av4 = _mm_add_ps(v_in1, v_in5);
        v_av5 = _mm_add_ps(v_in7, v_in11);
        v_av6 = _mm_add_ps(v_in3, v_in9);

        v_cv1 = _mm_add_ps(v_av2, v_av3);
        v_cv2 = _mm_add_ps(v_av4, v_av5);
        v_cv3 = _mm_add_ps(v_av1, v_av6);
        v_cv4 = _mm_sub_ps(v_av1, v_av6);

        v_cv5 = _mm_add_ps(v_cv1, v_cv2);
        v_cv6 = _mm_sub_ps(v_cv1, v_cv2);

        // output point 1 & 7
        v_out0 = _mm_add_ps(v_cv3, v_cv5);
        v_out6 = _mm_add_ps(v_cv4, v_cv6);

        // Common operations
        v_av7 = _mm_sub_ps(v_in0, v_in6);
        v_av8 = _mm_sub_ps(v_in2, v_in4);
        v_av9 = _mm_sub_ps(v_in8, v_in10);
        v_av10 = _mm_sub_ps(v_in1, v_in5);
        v_av11 = _mm_sub_ps(v_in7, v_in11);
        v_av12 = _mm_sub_ps(v_in3, v_in9);

        v_cv1 = _mm_sub_ps(v_av8, v_av9);
        v_cv2 = _mm_sub_ps(v_av4, v_av5);
        v_cv7 = _mm_sub_ps(v_cv2, v_av12);
        v_cv8 = _mm_sub_ps(v_av7, v_cv1);

        v_tv1 = _mm_mul_ps(v_K3, v_cv7);
        v_tv1 = SWAP_RI_128_S(CONJ_128_S(v_tv1));

        // output point 4 & 10
        v_out3 = _mm_sub_ps(v_cv8, v_tv1);
        v_out9 = _mm_add_ps(v_cv8, v_tv1);

        v_tv1 = _mm_mul_ps(v_K3, v_av12);
        v_tv1 = CONJ_128_S(v_tv1);

        v_tv2 = _mm_mul_ps(v_K2, v_cv1);
        v_cv1 = _mm_add_ps(v_av7, v_tv2);
        v_cv7 = _mm_sub_ps(v_av10, v_av11);

        v_tv3 = _mm_mul_ps(v_K1, v_cv7);
        v_cv8 = _mm_add_ps(v_cv1, v_tv3);

        v_tv4 = _mm_mul_ps(v_K5, v_cv2);
        v_tv4 = CONJ_128_S(v_tv4);

        v_cv2 = _mm_sub_ps(v_av2, v_av3);
        v_tv5 = _mm_mul_ps(v_K4, v_cv2);
        v_tv5 = CONJ_128_S(v_tv5);

        v_cv2 = _mm_add_ps(v_tv1, v_tv4);
        v_cv7 = _mm_add_ps(v_cv2, v_tv5);
        v_cv7 = SWAP_RI_128_S(v_cv7);

        // output point 2 & 12
        v_out1 = _mm_sub_ps(v_cv8, v_cv7);
        v_out11 = _mm_add_ps(v_cv8, v_cv7);

        v_cv7 = _mm_sub_ps(v_cv1, v_tv3);
        v_cv8 = _mm_sub_ps(v_cv2, v_tv5);
        v_cv8 = SWAP_RI_128_S(v_cv8);

        // output point 6 & 8
        v_out5 = _mm_sub_ps(v_cv7, v_cv8);
        v_out7 = _mm_add_ps(v_cv7, v_cv8);

        v_tv1 = _mm_mul_ps(v_K2, v_cv6);
        v_cv1 = _mm_sub_ps(v_cv4, v_tv1);
        v_cv2 = _mm_add_ps(v_av8, v_av9);
        v_cv4 = _mm_add_ps(v_av10, v_av11);
        v_cv6 = _mm_add_ps(v_cv4, v_cv2);
        v_tv2 = _mm_mul_ps(v_K4, v_cv6);
        v_tv2 = SWAP_RI_128_S(CONJ_128_S(v_tv2));

        // output point 3 & 11
        v_out2 = _mm_sub_ps(v_cv1, v_tv2);
        v_out10 = _mm_add_ps(v_cv1, v_tv2);

        v_tv1 = _mm_mul_ps(v_K2, v_cv5);
        v_cv1 = _mm_sub_ps(v_cv3, v_tv1);
        v_cv6 = _mm_sub_ps(v_cv4, v_cv2);
        v_tv2 = _mm_mul_ps(v_K4, v_cv6);
        v_tv2 = SWAP_RI_128_S(CONJ_128_S(v_tv2));

        // output point 5 & 9
        v_out4 = _mm_sub_ps(v_cv1, v_tv2);
        v_out8 = _mm_add_ps(v_cv1, v_tv2);

        SCATTER2_128_S(out_r, v_out_stride, v_out0, is_contiguous_out);
        STORE_OUT_128_S(out_r, out_strides, 1, v_out_stride, v_out1, tw, cols,
                        cnt_128, load_multi_cols, is_contiguous_out);
        STORE_OUT_128_S(out_r, out_strides, 2, v_out_stride, v_out2, tw, cols,
                        cnt_128, load_multi_cols, is_contiguous_out);
        STORE_OUT_128_S(out_r, out_strides, 3, v_out_stride, v_out3, tw, cols,
                        cnt_128, load_multi_cols, is_contiguous_out);
        STORE_OUT_128_S(out_r, out_strides, 4, v_out_stride, v_out4, tw, cols,
                        cnt_128, load_multi_cols, is_contiguous_out);
        STORE_OUT_128_S(out_r, out_strides, 5, v_out_stride, v_out5, tw, cols,
                        cnt_128, load_multi_cols, is_contiguous_out);
#if defined(KERNEL_VARIANT_R2C)
        v_out6 = OUT_H2_128_S(v_out6);
        STORE_OUT_128_S(out_h2_r, out_strides, 6, v_out_h2_stride, v_out6, tw,
                        cols, cnt_128, load_multi_cols, 0);
        v_out7 = OUT_H2_128_S(v_out7);
        STORE_OUT_128_S(out_h2_r, out_strides, 7, v_out_h2_stride, v_out7, tw,
                        cols, cnt_128, load_multi_cols, 0);
        v_out8 = OUT_H2_128_S(v_out8);
        STORE_OUT_128_S(out_h2_r, out_strides, 8, v_out_h2_stride, v_out8, tw,
                        cols, cnt_128, load_multi_cols, 0);
        v_out9 = OUT_H2_128_S(v_out9);
        STORE_OUT_128_S(out_h2_r, out_strides, 9, v_out_h2_stride, v_out9, tw,
                        cols, cnt_128, load_multi_cols, 0);
        v_out10 = OUT_H2_128_S(v_out10);
        STORE_OUT_128_S(out_h2_r, out_strides, 10, v_out_h2_stride, v_out10, tw,
                        cols, cnt_128, load_multi_cols, 0);
        v_out11 = OUT_H2_128_S(v_out11);
        STORE_OUT_128_S(out_h2_r, out_strides, 11, v_out_h2_stride, v_out11, tw,
                        cols, cnt_128, load_multi_cols, 0);
#else
        STORE_OUT_128_S(out_r, out_strides, 6, v_out_stride, v_out6, tw, cols,
                        cnt_128, load_multi_cols, is_contiguous_out);
        STORE_OUT_128_S(out_r, out_strides, 7, v_out_stride, v_out7, tw, cols,
                        cnt_128, load_multi_cols, is_contiguous_out);
        STORE_OUT_128_S(out_r, out_strides, 8, v_out_stride, v_out8, tw, cols,
                        cnt_128, load_multi_cols, is_contiguous_out);
        STORE_OUT_128_S(out_r, out_strides, 9, v_out_stride, v_out9, tw, cols,
                        cnt_128, load_multi_cols, is_contiguous_out);
        STORE_OUT_128_S(out_r, out_strides, 10, v_out_stride, v_out10, tw, cols,
                        cnt_128, load_multi_cols, is_contiguous_out);
        STORE_OUT_128_S(out_r, out_strides, 11, v_out_stride, v_out11, tw, cols,
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
        __m128 v_in0, v_in1, v_in2, v_in3, v_in4, v_in5, v_in6, v_in7, v_in8,
            v_in9, v_in10, v_in11;
        __m128 v_av1, v_av2, v_av3, v_av4, v_av5, v_av6, v_av7, v_av8, v_av9,
            v_av10, v_av11, v_av12;
        __m128 v_cv1, v_cv2, v_cv3, v_cv4, v_cv5, v_cv6, v_cv7, v_cv8;
        __m128 v_tv1, v_tv2, v_tv3, v_tv4, v_tv5;
        __m128 v_out0, v_out1, v_out2, v_out3, v_out4, v_out5, v_out6, v_out7,
            v_out8, v_out9, v_out10, v_out11;

#if defined(KERNEL_USE_AVX512)
        __m128 v_K1 = CAST_512_TO_128_S(v_C1);
        __m128 v_K2 = CAST_512_TO_128_S(v_C2);
        __m128 v_K3 = CAST_512_TO_128_S(v_C3);
        __m128 v_K4 = CAST_512_TO_128_S(v_C4);
        __m128 v_K5 = CAST_512_TO_128_S(v_C5);
#elif defined(KERNEL_USE_AVX256)
        __m128 v_K1 = CAST_256_TO_128_S(v_C1);
        __m128 v_K2 = CAST_256_TO_128_S(v_C2);
        __m128 v_K3 = CAST_256_TO_128_S(v_C3);
        __m128 v_K4 = CAST_256_TO_128_S(v_C4);
        __m128 v_K5 = CAST_256_TO_128_S(v_C5);
#elif defined(KERNEL_USE_AVX128)
        __m128 v_K1 = v_C1;
        __m128 v_K2 = v_C2;
        __m128 v_K3 = v_C3;
        __m128 v_K4 = v_C4;
        __m128 v_K5 = v_C5;
#endif

        LOAD_IN_64_S(in_r, in_strides, 1, v_in1, tw, cols, cnt_128_low,
                     load_multi_cols, is_contiguous_in);
        LOAD_IN_64_S(in_r, in_strides, 2, v_in2, tw, cols, cnt_128_low,
                     load_multi_cols, is_contiguous_in);
        LOAD_IN_64_S(in_r, in_strides, 3, v_in3, tw, cols, cnt_128_low,
                     load_multi_cols, is_contiguous_in);
        LOAD_IN_64_S(in_r, in_strides, 4, v_in4, tw, cols, cnt_128_low,
                     load_multi_cols, is_contiguous_in);
        LOAD_IN_64_S(in_r, in_strides, 5, v_in5, tw, cols, cnt_128_low,
                     load_multi_cols, is_contiguous_in);
#if defined(KERNEL_VARIANT_C2R)
        LOAD_IN_64_S(in_h2_r, in_strides, 6, v_in6, tw, cols, cnt_128_low,
                     load_multi_cols, 0);
        v_in6 = IN_H2_128_S(v_in6);
        LOAD_IN_64_S(in_h2_r, in_strides, 7, v_in7, tw, cols, cnt_128_low,
                     load_multi_cols, 0);
        v_in7 = IN_H2_128_S(v_in7);
        LOAD_IN_64_S(in_h2_r, in_strides, 8, v_in8, tw, cols, cnt_128_low,
                     load_multi_cols, 0);
        v_in8 = IN_H2_128_S(v_in8);
        LOAD_IN_64_S(in_h2_r, in_strides, 9, v_in9, tw, cols, cnt_128_low,
                     load_multi_cols, 0);
        v_in9 = IN_H2_128_S(v_in9);
        LOAD_IN_64_S(in_h2_r, in_strides, 10, v_in10, tw, cols, cnt_128_low,
                     load_multi_cols, 0);
        v_in10 = IN_H2_128_S(v_in10);
        LOAD_IN_64_S(in_h2_r, in_strides, 11, v_in11, tw, cols, cnt_128_low,
                     load_multi_cols, 0);
        v_in11 = IN_H2_128_S(v_in11);
#else
        LOAD_IN_64_S(in_r, in_strides, 6, v_in6, tw, cols, cnt_128_low,
                     load_multi_cols, is_contiguous_in);
        LOAD_IN_64_S(in_r, in_strides, 7, v_in7, tw, cols, cnt_128_low,
                     load_multi_cols, is_contiguous_in);
        LOAD_IN_64_S(in_r, in_strides, 8, v_in8, tw, cols, cnt_128_low,
                     load_multi_cols, is_contiguous_in);
        LOAD_IN_64_S(in_r, in_strides, 9, v_in9, tw, cols, cnt_128_low,
                     load_multi_cols, is_contiguous_in);
        LOAD_IN_64_S(in_r, in_strides, 10, v_in10, tw, cols, cnt_128_low,
                     load_multi_cols, is_contiguous_in);
        LOAD_IN_64_S(in_r, in_strides, 11, v_in11, tw, cols, cnt_128_low,
                     load_multi_cols, is_contiguous_in);
#endif

        LD_LOW_128_S(in_r, v_in0);

        // Common operations
        v_av1 = _mm_add_ps(v_in0, v_in6);
        v_av2 = _mm_add_ps(v_in2, v_in4);
        v_av3 = _mm_add_ps(v_in8, v_in10);
        v_av4 = _mm_add_ps(v_in1, v_in5);
        v_av5 = _mm_add_ps(v_in7, v_in11);
        v_av6 = _mm_add_ps(v_in3, v_in9);

        v_cv1 = _mm_add_ps(v_av2, v_av3);
        v_cv2 = _mm_add_ps(v_av4, v_av5);
        v_cv3 = _mm_add_ps(v_av1, v_av6);
        v_cv4 = _mm_sub_ps(v_av1, v_av6);

        v_cv5 = _mm_add_ps(v_cv1, v_cv2);
        v_cv6 = _mm_sub_ps(v_cv1, v_cv2);

        // output point 1 & 7
        v_out0 = _mm_add_ps(v_cv3, v_cv5);
        v_out6 = _mm_add_ps(v_cv4, v_cv6);

        // Common operations
        v_av7 = _mm_sub_ps(v_in0, v_in6);
        v_av8 = _mm_sub_ps(v_in2, v_in4);
        v_av9 = _mm_sub_ps(v_in8, v_in10);
        v_av10 = _mm_sub_ps(v_in1, v_in5);
        v_av11 = _mm_sub_ps(v_in7, v_in11);
        v_av12 = _mm_sub_ps(v_in3, v_in9);

        v_cv1 = _mm_sub_ps(v_av8, v_av9);
        v_cv2 = _mm_sub_ps(v_av4, v_av5);
        v_cv7 = _mm_sub_ps(v_cv2, v_av12);
        v_cv8 = _mm_sub_ps(v_av7, v_cv1);

        v_tv1 = _mm_mul_ps(v_K3, v_cv7);
        v_tv1 = SWAP_RI_128_S(CONJ_128_S(v_tv1));

        // output point 4 & 10
        v_out3 = _mm_sub_ps(v_cv8, v_tv1);
        v_out9 = _mm_add_ps(v_cv8, v_tv1);

        v_tv1 = _mm_mul_ps(v_K3, v_av12);
        v_tv1 = CONJ_128_S(v_tv1);

        v_tv2 = _mm_mul_ps(v_K2, v_cv1);
        v_cv1 = _mm_add_ps(v_av7, v_tv2);
        v_cv7 = _mm_sub_ps(v_av10, v_av11);

        v_tv3 = _mm_mul_ps(v_K1, v_cv7);
        v_cv8 = _mm_add_ps(v_cv1, v_tv3);

        v_tv4 = _mm_mul_ps(v_K5, v_cv2);
        v_tv4 = CONJ_128_S(v_tv4);

        v_cv2 = _mm_sub_ps(v_av2, v_av3);
        v_tv5 = _mm_mul_ps(v_K4, v_cv2);
        v_tv5 = CONJ_128_S(v_tv5);

        v_cv2 = _mm_add_ps(v_tv1, v_tv4);
        v_cv7 = _mm_add_ps(v_cv2, v_tv5);
        v_cv7 = SWAP_RI_128_S(v_cv7);

        // output point 2 & 12
        v_out1 = _mm_sub_ps(v_cv8, v_cv7);
        v_out11 = _mm_add_ps(v_cv8, v_cv7);

        v_cv7 = _mm_sub_ps(v_cv1, v_tv3);
        v_cv8 = _mm_sub_ps(v_cv2, v_tv5);
        v_cv8 = SWAP_RI_128_S(v_cv8);

        // output point 6 & 8
        v_out5 = _mm_sub_ps(v_cv7, v_cv8);
        v_out7 = _mm_add_ps(v_cv7, v_cv8);

        v_tv1 = _mm_mul_ps(v_K2, v_cv6);
        v_cv1 = _mm_sub_ps(v_cv4, v_tv1);
        v_cv2 = _mm_add_ps(v_av8, v_av9);
        v_cv4 = _mm_add_ps(v_av10, v_av11);
        v_cv6 = _mm_add_ps(v_cv4, v_cv2);
        v_tv2 = _mm_mul_ps(v_K4, v_cv6);
        v_tv2 = SWAP_RI_128_S(CONJ_128_S(v_tv2));

        // output point 3 & 11
        v_out2 = _mm_sub_ps(v_cv1, v_tv2);
        v_out10 = _mm_add_ps(v_cv1, v_tv2);

        v_tv1 = _mm_mul_ps(v_K2, v_cv5);
        v_cv1 = _mm_sub_ps(v_cv3, v_tv1);
        v_cv6 = _mm_sub_ps(v_cv4, v_cv2);
        v_tv2 = _mm_mul_ps(v_K4, v_cv6);
        v_tv2 = SWAP_RI_128_S(CONJ_128_S(v_tv2));

        // output point 5 & 9
        v_out4 = _mm_sub_ps(v_cv1, v_tv2);
        v_out8 = _mm_add_ps(v_cv1, v_tv2);

        ST_LOW_128_S(out_r, v_out0);
        STORE_OUT_64_S(out_r, out_strides, 1, v_out1, tw, cols, cnt_128_low,
                       load_multi_cols, is_contiguous_out);
        STORE_OUT_64_S(out_r, out_strides, 2, v_out2, tw, cols, cnt_128_low,
                       load_multi_cols, is_contiguous_out);
        STORE_OUT_64_S(out_r, out_strides, 3, v_out3, tw, cols, cnt_128_low,
                       load_multi_cols, is_contiguous_out);
        STORE_OUT_64_S(out_r, out_strides, 4, v_out4, tw, cols, cnt_128_low,
                       load_multi_cols, is_contiguous_out);
        STORE_OUT_64_S(out_r, out_strides, 5, v_out5, tw, cols, cnt_128_low,
                       load_multi_cols, is_contiguous_out);
#if defined(KERNEL_VARIANT_R2C)
        v_out6 = OUT_H2_128_S(v_out6);
        STORE_OUT_64_S(out_h2_r, out_strides, 6, v_out6, tw, cols, cnt_128_low,
                       load_multi_cols, 0);
        v_out7 = OUT_H2_128_S(v_out7);
        STORE_OUT_64_S(out_h2_r, out_strides, 7, v_out7, tw, cols, cnt_128_low,
                       load_multi_cols, 0);
        v_out8 = OUT_H2_128_S(v_out8);
        STORE_OUT_64_S(out_h2_r, out_strides, 8, v_out8, tw, cols, cnt_128_low,
                       load_multi_cols, 0);
        v_out9 = OUT_H2_128_S(v_out9);
        STORE_OUT_64_S(out_h2_r, out_strides, 9, v_out9, tw, cols, cnt_128_low,
                       load_multi_cols, 0);
        v_out10 = OUT_H2_128_S(v_out10);
        STORE_OUT_64_S(out_h2_r, out_strides, 10, v_out10, tw, cols,
                       cnt_128_low, load_multi_cols, 0);
        v_out11 = OUT_H2_128_S(v_out11);
        STORE_OUT_64_S(out_h2_r, out_strides, 11, v_out11, tw, cols,
                       cnt_128_low, load_multi_cols, 0);
#else
        STORE_OUT_64_S(out_r, out_strides, 6, v_out6, tw, cols, cnt_128_low,
                       load_multi_cols, is_contiguous_out);
        STORE_OUT_64_S(out_r, out_strides, 7, v_out7, tw, cols, cnt_128_low,
                       load_multi_cols, is_contiguous_out);
        STORE_OUT_64_S(out_r, out_strides, 8, v_out8, tw, cols, cnt_128_low,
                       load_multi_cols, is_contiguous_out);
        STORE_OUT_64_S(out_r, out_strides, 9, v_out9, tw, cols, cnt_128_low,
                       load_multi_cols, is_contiguous_out);
        STORE_OUT_64_S(out_r, out_strides, 10, v_out10, tw, cols, cnt_128_low,
                       load_multi_cols, is_contiguous_out);
        STORE_OUT_64_S(out_r, out_strides, 11, v_out11, tw, cols, cnt_128_low,
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

    const DOUBLE CRTM_12[3] = {
        0.86602540378443864676372317075293618347140262700000,
        0.50000000000000000000000000000000000000000000000000,
        1.00000000000000000000000000000000000000000000000000};

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

    VREGTYPE_D v_C1 = BCAST_D(CRTM_12[0]);
    VREGTYPE_D v_C2 = BCAST_D(CRTM_12[1]);
    VREGTYPE_D v_C3 = BCAST_D(CRTM_12[2]);
    VREGTYPE_D v_C4 = BCAST_D(CRTM_12[0]);
    VREGTYPE_D v_C5 = BCAST_D(CRTM_12[1]);

#if defined(KERNEL_DIRECTION_BWD)
    v_C3 = NEG_D(v_C3, 1);
    v_C4 = NEG_D(v_C4, 1);
    v_C5 = NEG_D(v_C5, 1);
#endif
    for (count = 0; count < N; count++)
    {
        VREGTYPE_D v_in0, v_in1, v_in2, v_in3, v_in4, v_in5, v_in6, v_in7,
            v_in8, v_in9, v_in10, v_in11;
        VREGTYPE_D v_av1, v_av2, v_av3, v_av4, v_av5, v_av6, v_av7, v_av8,
            v_av9, v_av10, v_av11, v_av12;
        VREGTYPE_D v_out0, v_out1, v_out2, v_out3, v_out4, v_out5, v_out6,
            v_out7, v_out8, v_out9, v_out10, v_out11;
        VREGTYPE_D v_cv1, v_cv2, v_cv3, v_cv4, v_cv5, v_cv6, v_cv7, v_cv8;
        VREGTYPE_D v_tv1, v_tv2, v_tv3, v_tv4, v_tv5;
        INTP col = count * load_multi_cols * NUM_SETS_D;

        LOAD_IN_D(in_r, in_strides, 1, v_in_stride, v_in1, tw, cols, col,
                  load_multi_cols, is_contiguous_in);
        LOAD_IN_D(in_r, in_strides, 2, v_in_stride, v_in2, tw, cols, col,
                  load_multi_cols, is_contiguous_in);
        LOAD_IN_D(in_r, in_strides, 3, v_in_stride, v_in3, tw, cols, col,
                  load_multi_cols, is_contiguous_in);
        LOAD_IN_D(in_r, in_strides, 4, v_in_stride, v_in4, tw, cols, col,
                  load_multi_cols, is_contiguous_in);
        LOAD_IN_D(in_r, in_strides, 5, v_in_stride, v_in5, tw, cols, col,
                  load_multi_cols, is_contiguous_in);
#if defined(KERNEL_VARIANT_C2R)
        LOAD_IN_D(in_h2_r, in_strides, 6, v_in_h2_stride, v_in6, tw, cols, col,
                  load_multi_cols, 0);
        v_in6 = IN_H2_D(v_in6);
        LOAD_IN_D(in_h2_r, in_strides, 7, v_in_h2_stride, v_in7, tw, cols, col,
                  load_multi_cols, 0);
        v_in7 = IN_H2_D(v_in7);
        LOAD_IN_D(in_h2_r, in_strides, 8, v_in_h2_stride, v_in8, tw, cols, col,
                  load_multi_cols, 0);
        v_in8 = IN_H2_D(v_in8);
        LOAD_IN_D(in_h2_r, in_strides, 9, v_in_h2_stride, v_in9, tw, cols, col,
                  load_multi_cols, 0);
        v_in9 = IN_H2_D(v_in9);
        LOAD_IN_D(in_h2_r, in_strides, 10, v_in_h2_stride, v_in10, tw, cols,
                  col, load_multi_cols, 0);
        v_in10 = IN_H2_D(v_in10);
        LOAD_IN_D(in_h2_r, in_strides, 11, v_in_h2_stride, v_in11, tw, cols,
                  col, load_multi_cols, 0);
        v_in11 = IN_H2_D(v_in11);
#else
        LOAD_IN_D(in_r, in_strides, 6, v_in_stride, v_in6, tw, cols, col,
                  load_multi_cols, is_contiguous_in);
        LOAD_IN_D(in_r, in_strides, 7, v_in_stride, v_in7, tw, cols, col,
                  load_multi_cols, is_contiguous_in);
        LOAD_IN_D(in_r, in_strides, 8, v_in_stride, v_in8, tw, cols, col,
                  load_multi_cols, is_contiguous_in);
        LOAD_IN_D(in_r, in_strides, 9, v_in_stride, v_in9, tw, cols, col,
                  load_multi_cols, is_contiguous_in);
        LOAD_IN_D(in_r, in_strides, 10, v_in_stride, v_in10, tw, cols, col,
                  load_multi_cols, is_contiguous_in);
        LOAD_IN_D(in_r, in_strides, 11, v_in_stride, v_in11, tw, cols, col,
                  load_multi_cols, is_contiguous_in);
#endif
        GATHER_D(in_r, v_in_stride, v_in0, is_contiguous_in);

        // Common operations
        v_av1 = ADD_D(v_in0, v_in6);
        v_av2 = ADD_D(v_in2, v_in4);
        v_av3 = ADD_D(v_in8, v_in10);
        v_av4 = ADD_D(v_in1, v_in5);
        v_av5 = ADD_D(v_in7, v_in11);
        v_av6 = ADD_D(v_in3, v_in9);

        v_cv1 = ADD_D(v_av2, v_av3);
        v_cv2 = ADD_D(v_av4, v_av5);
        v_cv3 = ADD_D(v_av1, v_av6);
        v_cv4 = SUB_D(v_av1, v_av6);

        v_cv5 = ADD_D(v_cv1, v_cv2);
        v_cv6 = SUB_D(v_cv1, v_cv2);

        // output point 1 & 7
        v_out0 = ADD_D(v_cv3, v_cv5);
        v_out6 = ADD_D(v_cv4, v_cv6);

        // Common operations
        v_av7 = SUB_D(v_in0, v_in6);
        v_av8 = SUB_D(v_in2, v_in4);
        v_av9 = SUB_D(v_in8, v_in10);
        v_av10 = SUB_D(v_in1, v_in5);
        v_av11 = SUB_D(v_in7, v_in11);
        v_av12 = SUB_D(v_in3, v_in9);

        v_cv1 = SUB_D(v_av8, v_av9);
        v_cv2 = SUB_D(v_av4, v_av5);
        v_cv7 = SUB_D(v_cv2, v_av12);
        v_cv8 = SUB_D(v_av7, v_cv1);

        v_tv1 = MUL_D(v_C3, v_cv7);
        v_tv1 = SWAP_RI_D(CONJ_D(v_tv1));

        // output point 4 & 10
        v_out3 = SUB_D(v_cv8, v_tv1);
        v_out9 = ADD_D(v_cv8, v_tv1);

        v_tv1 = MUL_D(v_C3, v_av12);
        v_tv1 = CONJ_D(v_tv1);

        v_tv2 = MUL_D(v_C2, v_cv1);
        v_cv1 = ADD_D(v_av7, v_tv2);
        v_cv7 = SUB_D(v_av10, v_av11);

        v_tv3 = MUL_D(v_C1, v_cv7);
        v_cv8 = ADD_D(v_cv1, v_tv3);

        v_tv4 = MUL_D(v_C5, v_cv2);
        v_tv4 = CONJ_D(v_tv4);

        v_cv2 = SUB_D(v_av2, v_av3);
        v_tv5 = MUL_D(v_C4, v_cv2);
        v_tv5 = CONJ_D(v_tv5);

        v_cv2 = ADD_D(v_tv1, v_tv4);
        v_cv7 = ADD_D(v_cv2, v_tv5);
        v_cv7 = SWAP_RI_D(v_cv7);

        // output point 2 & 12
        v_out1 = SUB_D(v_cv8, v_cv7);
        v_out11 = ADD_D(v_cv8, v_cv7);

        v_cv7 = SUB_D(v_cv1, v_tv3);
        v_cv8 = SUB_D(v_cv2, v_tv5);
        v_cv8 = SWAP_RI_D(v_cv8);

        // output point 6 & 8
        v_out5 = SUB_D(v_cv7, v_cv8);
        v_out7 = ADD_D(v_cv7, v_cv8);

        v_tv1 = MUL_D(v_C2, v_cv6);
        v_cv1 = SUB_D(v_cv4, v_tv1);
        v_cv2 = ADD_D(v_av8, v_av9);
        v_cv4 = ADD_D(v_av10, v_av11);
        v_cv6 = ADD_D(v_cv4, v_cv2);
        v_tv2 = MUL_D(v_C4, v_cv6);
        v_tv2 = SWAP_RI_D(CONJ_D(v_tv2));

        // output point 3 & 11
        v_out2 = SUB_D(v_cv1, v_tv2);
        v_out10 = ADD_D(v_cv1, v_tv2);

        v_tv1 = MUL_D(v_C2, v_cv5);
        v_cv1 = SUB_D(v_cv3, v_tv1);
        v_cv6 = SUB_D(v_cv4, v_cv2);
        v_tv2 = MUL_D(v_C4, v_cv6);
        v_tv2 = SWAP_RI_D(CONJ_D(v_tv2));

        // output point 5 & 9
        v_out4 = SUB_D(v_cv1, v_tv2);
        v_out8 = ADD_D(v_cv1, v_tv2);

        SCATTER_D(out_r, v_out_stride, v_out0, is_contiguous_out);
        STORE_OUT_D(out_r, out_strides, 1, v_out_stride, v_out1, tw, cols, col,
                    load_multi_cols, is_contiguous_out);
        STORE_OUT_D(out_r, out_strides, 2, v_out_stride, v_out2, tw, cols, col,
                    load_multi_cols, is_contiguous_out);
        STORE_OUT_D(out_r, out_strides, 3, v_out_stride, v_out3, tw, cols, col,
                    load_multi_cols, is_contiguous_out);
        STORE_OUT_D(out_r, out_strides, 4, v_out_stride, v_out4, tw, cols, col,
                    load_multi_cols, is_contiguous_out);
        STORE_OUT_D(out_r, out_strides, 5, v_out_stride, v_out5, tw, cols, col,
                    load_multi_cols, is_contiguous_out);
#if defined(KERNEL_VARIANT_R2C)
        v_out6 = OUT_H2_D(v_out6);
        STORE_OUT_D(out_h2_r, out_strides, 6, v_out_h2_stride, v_out6, tw, cols,
                    col, load_multi_cols, 0);
        v_out7 = OUT_H2_D(v_out7);
        STORE_OUT_D(out_h2_r, out_strides, 7, v_out_h2_stride, v_out7, tw, cols,
                    col, load_multi_cols, 0);
        v_out8 = OUT_H2_D(v_out8);
        STORE_OUT_D(out_h2_r, out_strides, 8, v_out_h2_stride, v_out8, tw, cols,
                    col, load_multi_cols, 0);
        v_out9 = OUT_H2_D(v_out9);
        STORE_OUT_D(out_h2_r, out_strides, 9, v_out_h2_stride, v_out9, tw, cols,
                    col, load_multi_cols, 0);
        v_out10 = OUT_H2_D(v_out10);
        STORE_OUT_D(out_h2_r, out_strides, 10, v_out_h2_stride, v_out10, tw,
                    cols, col, load_multi_cols, 0);
        v_out11 = OUT_H2_D(v_out11);
        STORE_OUT_D(out_h2_r, out_strides, 11, v_out_h2_stride, v_out11, tw,
                    cols, col, load_multi_cols, 0);
#else
        STORE_OUT_D(out_r, out_strides, 6, v_out_stride, v_out6, tw, cols, col,
                    load_multi_cols, is_contiguous_out);
        STORE_OUT_D(out_r, out_strides, 7, v_out_stride, v_out7, tw, cols, col,
                    load_multi_cols, is_contiguous_out);
        STORE_OUT_D(out_r, out_strides, 8, v_out_stride, v_out8, tw, cols, col,
                    load_multi_cols, is_contiguous_out);
        STORE_OUT_D(out_r, out_strides, 9, v_out_stride, v_out9, tw, cols, col,
                    load_multi_cols, is_contiguous_out);
        STORE_OUT_D(out_r, out_strides, 10, v_out_stride, v_out10, tw, cols,
                    col, load_multi_cols, is_contiguous_out);
        STORE_OUT_D(out_r, out_strides, 11, v_out_stride, v_out11, tw, cols,
                    col, load_multi_cols, is_contiguous_out);
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
        __m256d v_in0, v_in1, v_in2, v_in3, v_in4, v_in5, v_in6, v_in7, v_in8,
            v_in9, v_in10, v_in11;
        __m256d v_av1, v_av2, v_av3, v_av4, v_av5, v_av6, v_av7, v_av8, v_av9,
            v_av10, v_av11, v_av12;
        __m256d v_out0, v_out1, v_out2, v_out3, v_out4, v_out5, v_out6, v_out7,
            v_out8, v_out9, v_out10, v_out11;
        __m256d v_cv1, v_cv2, v_cv3, v_cv4, v_cv5, v_cv6, v_cv7, v_cv8;
        __m256d v_tv1, v_tv2, v_tv3, v_tv4, v_tv5;

        __m256d v_K1 = CAST_512_TO_256_D(v_C1);
        __m256d v_K2 = CAST_512_TO_256_D(v_C2);
        __m256d v_K3 = CAST_512_TO_256_D(v_C3);
        __m256d v_K4 = CAST_512_TO_256_D(v_C4);
        __m256d v_K5 = CAST_512_TO_256_D(v_C5);

        LOAD_IN_256_D(in_r, in_strides, 1, v_in_stride, v_in1, tw, cols,
                      cnt_256, load_multi_cols, is_contiguous_in);
        LOAD_IN_256_D(in_r, in_strides, 2, v_in_stride, v_in2, tw, cols,
                      cnt_256, load_multi_cols, is_contiguous_in);
        LOAD_IN_256_D(in_r, in_strides, 3, v_in_stride, v_in3, tw, cols,
                      cnt_256, load_multi_cols, is_contiguous_in);
        LOAD_IN_256_D(in_r, in_strides, 4, v_in_stride, v_in4, tw, cols,
                      cnt_256, load_multi_cols, is_contiguous_in);
        LOAD_IN_256_D(in_r, in_strides, 5, v_in_stride, v_in5, tw, cols,
                      cnt_256, load_multi_cols, is_contiguous_in);
#if defined(KERNEL_VARIANT_C2R)
        LOAD_IN_256_D(in_h2_r, in_strides, 6, v_in_h2_stride, v_in6, tw, cols,
                      cnt_256, load_multi_cols, 0);
        v_in6 = IN_H2_256_D(v_in6);
        LOAD_IN_256_D(in_h2_r, in_strides, 7, v_in_h2_stride, v_in7, tw, cols,
                      cnt_256, load_multi_cols, 0);
        v_in7 = IN_H2_256_D(v_in7);
        LOAD_IN_256_D(in_h2_r, in_strides, 8, v_in_h2_stride, v_in8, tw, cols,
                      cnt_256, load_multi_cols, 0);
        v_in8 = IN_H2_256_D(v_in8);
        LOAD_IN_256_D(in_h2_r, in_strides, 9, v_in_h2_stride, v_in9, tw, cols,
                      cnt_256, load_multi_cols, 0);
        v_in9 = IN_H2_256_D(v_in9);
        LOAD_IN_256_D(in_h2_r, in_strides, 10, v_in_h2_stride, v_in10, tw, cols,
                      cnt_256, load_multi_cols, 0);
        v_in10 = IN_H2_256_D(v_in10);
        LOAD_IN_256_D(in_h2_r, in_strides, 11, v_in_h2_stride, v_in11, tw, cols,
                      cnt_256, load_multi_cols, 0);
        v_in11 = IN_H2_256_D(v_in11);
#else
        LOAD_IN_256_D(in_r, in_strides, 6, v_in_stride, v_in6, tw, cols,
                      cnt_256, load_multi_cols, is_contiguous_in);
        LOAD_IN_256_D(in_r, in_strides, 7, v_in_stride, v_in7, tw, cols,
                      cnt_256, load_multi_cols, is_contiguous_in);
        LOAD_IN_256_D(in_r, in_strides, 8, v_in_stride, v_in8, tw, cols,
                      cnt_256, load_multi_cols, is_contiguous_in);
        LOAD_IN_256_D(in_r, in_strides, 9, v_in_stride, v_in9, tw, cols,
                      cnt_256, load_multi_cols, is_contiguous_in);
        LOAD_IN_256_D(in_r, in_strides, 10, v_in_stride, v_in10, tw, cols,
                      cnt_256, load_multi_cols, is_contiguous_in);
        LOAD_IN_256_D(in_r, in_strides, 11, v_in_stride, v_in11, tw, cols,
                      cnt_256, load_multi_cols, is_contiguous_in);
#endif
        GATHER2_256_D(in_r, v_in_stride, v_in0, is_contiguous_in);

        // Common operations
        v_av1 = _mm256_add_pd(v_in0, v_in6);
        v_av2 = _mm256_add_pd(v_in2, v_in4);
        v_av3 = _mm256_add_pd(v_in8, v_in10);
        v_av4 = _mm256_add_pd(v_in1, v_in5);
        v_av5 = _mm256_add_pd(v_in7, v_in11);
        v_av6 = _mm256_add_pd(v_in3, v_in9);

        v_cv1 = _mm256_add_pd(v_av2, v_av3);
        v_cv2 = _mm256_add_pd(v_av4, v_av5);
        v_cv3 = _mm256_add_pd(v_av1, v_av6);
        v_cv4 = _mm256_sub_pd(v_av1, v_av6);

        v_cv5 = _mm256_add_pd(v_cv1, v_cv2);
        v_cv6 = _mm256_sub_pd(v_cv1, v_cv2);

        // output point 1 & 7
        v_out0 = _mm256_add_pd(v_cv3, v_cv5);
        v_out6 = _mm256_add_pd(v_cv4, v_cv6);

        // Common operations
        v_av7 = _mm256_sub_pd(v_in0, v_in6);
        v_av8 = _mm256_sub_pd(v_in2, v_in4);
        v_av9 = _mm256_sub_pd(v_in8, v_in10);
        v_av10 = _mm256_sub_pd(v_in1, v_in5);
        v_av11 = _mm256_sub_pd(v_in7, v_in11);
        v_av12 = _mm256_sub_pd(v_in3, v_in9);

        v_cv1 = _mm256_sub_pd(v_av8, v_av9);
        v_cv2 = _mm256_sub_pd(v_av4, v_av5);
        v_cv7 = _mm256_sub_pd(v_cv2, v_av12);
        v_cv8 = _mm256_sub_pd(v_av7, v_cv1);

        v_tv1 = _mm256_mul_pd(v_K3, v_cv7);
        v_tv1 = SWAP_RI_256_D(CONJ_256_D(v_tv1));

        // output point 4 & 10
        v_out3 = _mm256_sub_pd(v_cv8, v_tv1);
        v_out9 = _mm256_add_pd(v_cv8, v_tv1);

        v_tv1 = _mm256_mul_pd(v_K3, v_av12);
        v_tv1 = CONJ_256_D(v_tv1);

        v_tv2 = _mm256_mul_pd(v_K2, v_cv1);
        v_cv1 = _mm256_add_pd(v_av7, v_tv2);
        v_cv7 = _mm256_sub_pd(v_av10, v_av11);

        v_tv3 = _mm256_mul_pd(v_K1, v_cv7);
        v_cv8 = _mm256_add_pd(v_cv1, v_tv3);

        v_tv4 = _mm256_mul_pd(v_K5, v_cv2);
        v_tv4 = CONJ_256_D(v_tv4);

        v_cv2 = _mm256_sub_pd(v_av2, v_av3);
        v_tv5 = _mm256_mul_pd(v_K4, v_cv2);
        v_tv5 = CONJ_256_D(v_tv5);

        v_cv2 = _mm256_add_pd(v_tv1, v_tv4);
        v_cv7 = _mm256_add_pd(v_cv2, v_tv5);
        v_cv7 = SWAP_RI_256_D(v_cv7);

        // output point 2 & 12
        v_out1 = _mm256_sub_pd(v_cv8, v_cv7);
        v_out11 = _mm256_add_pd(v_cv8, v_cv7);

        v_cv7 = _mm256_sub_pd(v_cv1, v_tv3);
        v_cv8 = _mm256_sub_pd(v_cv2, v_tv5);
        v_cv8 = SWAP_RI_256_D(v_cv8);

        // output point 6 & 8
        v_out5 = _mm256_sub_pd(v_cv7, v_cv8);
        v_out7 = _mm256_add_pd(v_cv7, v_cv8);

        v_tv1 = _mm256_mul_pd(v_K2, v_cv6);
        v_cv1 = _mm256_sub_pd(v_cv4, v_tv1);
        v_cv2 = _mm256_add_pd(v_av8, v_av9);
        v_cv4 = _mm256_add_pd(v_av10, v_av11);
        v_cv6 = _mm256_add_pd(v_cv4, v_cv2);
        v_tv2 = _mm256_mul_pd(v_K4, v_cv6);
        v_tv2 = SWAP_RI_256_D(CONJ_256_D(v_tv2));

        // output point 3 & 11
        v_out2 = _mm256_sub_pd(v_cv1, v_tv2);
        v_out10 = _mm256_add_pd(v_cv1, v_tv2);

        v_tv1 = _mm256_mul_pd(v_K2, v_cv5);
        v_cv1 = _mm256_sub_pd(v_cv3, v_tv1);
        v_cv6 = _mm256_sub_pd(v_cv4, v_cv2);
        v_tv2 = _mm256_mul_pd(v_K4, v_cv6);
        v_tv2 = SWAP_RI_256_D(CONJ_256_D(v_tv2));

        // output point 5 & 9
        v_out4 = _mm256_sub_pd(v_cv1, v_tv2);
        v_out8 = _mm256_add_pd(v_cv1, v_tv2);

        SCATTER2_256_D(out_r, v_out_stride, v_out0, is_contiguous_out);
        STORE_OUT_256_D(out_r, out_strides, 1, v_out_stride, v_out1, tw, cols,
                        cnt_256, load_multi_cols, is_contiguous_out);
        STORE_OUT_256_D(out_r, out_strides, 2, v_out_stride, v_out2, tw, cols,
                        cnt_256, load_multi_cols, is_contiguous_out);
        STORE_OUT_256_D(out_r, out_strides, 3, v_out_stride, v_out3, tw, cols,
                        cnt_256, load_multi_cols, is_contiguous_out);
        STORE_OUT_256_D(out_r, out_strides, 4, v_out_stride, v_out4, tw, cols,
                        cnt_256, load_multi_cols, is_contiguous_out);
        STORE_OUT_256_D(out_r, out_strides, 5, v_out_stride, v_out5, tw, cols,
                        cnt_256, load_multi_cols, is_contiguous_out);
#if defined(KERNEL_VARIANT_R2C)
        v_out6 = OUT_H2_256_D(v_out6);
        STORE_OUT_256_D(out_h2_r, out_strides, 6, v_out_h2_stride, v_out6, tw,
                        cols, cnt_256, load_multi_cols, 0);
        v_out7 = OUT_H2_256_D(v_out7);
        STORE_OUT_256_D(out_h2_r, out_strides, 7, v_out_h2_stride, v_out7, tw,
                        cols, cnt_256, load_multi_cols, 0);
        v_out8 = OUT_H2_256_D(v_out8);
        STORE_OUT_256_D(out_h2_r, out_strides, 8, v_out_h2_stride, v_out8, tw,
                        cols, cnt_256, load_multi_cols, 0);
        v_out9 = OUT_H2_256_D(v_out9);
        STORE_OUT_256_D(out_h2_r, out_strides, 9, v_out_h2_stride, v_out9, tw,
                        cols, cnt_256, load_multi_cols, 0);
        v_out10 = OUT_H2_256_D(v_out10);
        STORE_OUT_256_D(out_h2_r, out_strides, 10, v_out_h2_stride, v_out10, tw,
                        cols, cnt_256, load_multi_cols, 0);
        v_out11 = OUT_H2_256_D(v_out11);
        STORE_OUT_256_D(out_h2_r, out_strides, 11, v_out_h2_stride, v_out11, tw,
                        cols, cnt_256, load_multi_cols, 0);
#else
        STORE_OUT_256_D(out_r, out_strides, 6, v_out_stride, v_out6, tw, cols,
                        cnt_256, load_multi_cols, is_contiguous_out);
        STORE_OUT_256_D(out_r, out_strides, 7, v_out_stride, v_out7, tw, cols,
                        cnt_256, load_multi_cols, is_contiguous_out);
        STORE_OUT_256_D(out_r, out_strides, 8, v_out_stride, v_out8, tw, cols,
                        cnt_256, load_multi_cols, is_contiguous_out);
        STORE_OUT_256_D(out_r, out_strides, 9, v_out_stride, v_out9, tw, cols,
                        cnt_256, load_multi_cols, is_contiguous_out);
        STORE_OUT_256_D(out_r, out_strides, 10, v_out_stride, v_out10, tw, cols,
                        cnt_256, load_multi_cols, is_contiguous_out);
        STORE_OUT_256_D(out_r, out_strides, 11, v_out_stride, v_out11, tw, cols,
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
        __m128d v_in0, v_in1, v_in2, v_in3, v_in4, v_in5, v_in6, v_in7, v_in8,
            v_in9, v_in10, v_in11;
        __m128d v_av1, v_av2, v_av3, v_av4, v_av5, v_av6, v_av7, v_av8, v_av9,
            v_av10, v_av11, v_av12;
        __m128d v_out0, v_out1, v_out2, v_out3, v_out4, v_out5, v_out6, v_out7,
            v_out8, v_out9, v_out10, v_out11;
        __m128d v_cv1, v_cv2, v_cv3, v_cv4, v_cv5, v_cv6, v_cv7, v_cv8;
        __m128d v_tv1, v_tv2, v_tv3, v_tv4, v_tv5;

#if defined(KERNEL_USE_AVX512)
        __m128d v_K1 = CAST_512_TO_128_D(v_C1);
        __m128d v_K2 = CAST_512_TO_128_D(v_C2);
        __m128d v_K3 = CAST_512_TO_128_D(v_C3);
        __m128d v_K4 = CAST_512_TO_128_D(v_C4);
        __m128d v_K5 = CAST_512_TO_128_D(v_C5);
#elif defined(KERNEL_USE_AVX256)
        __m128d v_K1 = CAST_256_TO_128_D(v_C1);
        __m128d v_K2 = CAST_256_TO_128_D(v_C2);
        __m128d v_K3 = CAST_256_TO_128_D(v_C3);
        __m128d v_K4 = CAST_256_TO_128_D(v_C4);
        __m128d v_K5 = CAST_256_TO_128_D(v_C5);
#endif

        LOAD_IN_128_D(in_r, in_strides, 1, /* unused */ 0, v_in1, tw, cols,
                      cnt_128, load_multi_cols, is_contiguous_in);
        LOAD_IN_128_D(in_r, in_strides, 2, /* unused */ 0, v_in2, tw, cols,
                      cnt_128, load_multi_cols, is_contiguous_in);
        LOAD_IN_128_D(in_r, in_strides, 3, /* unused */ 0, v_in3, tw, cols,
                      cnt_128, load_multi_cols, is_contiguous_in);
        LOAD_IN_128_D(in_r, in_strides, 4, /* unused */ 0, v_in4, tw, cols,
                      cnt_128, load_multi_cols, is_contiguous_in);
        LOAD_IN_128_D(in_r, in_strides, 5, /* unused */ 0, v_in5, tw, cols,
                      cnt_128, load_multi_cols, is_contiguous_in);
#if defined(KERNEL_VARIANT_C2R)
        LOAD_IN_128_D(in_h2_r, in_strides, 6, /* unused */ 0, v_in6, tw, cols,
                      cnt_128, load_multi_cols, 0);
        v_in6 = IN_H2_128_D(v_in6);
        LOAD_IN_128_D(in_h2_r, in_strides, 7, /* unused */ 0, v_in7, tw, cols,
                      cnt_128, load_multi_cols, 0);
        v_in7 = IN_H2_128_D(v_in7);
        LOAD_IN_128_D(in_h2_r, in_strides, 8, /* unused */ 0, v_in8, tw, cols,
                      cnt_128, load_multi_cols, 0);
        v_in8 = IN_H2_128_D(v_in8);
        LOAD_IN_128_D(in_h2_r, in_strides, 9, /* unused */ 0, v_in9, tw, cols,
                      cnt_128, load_multi_cols, 0);
        v_in9 = IN_H2_128_D(v_in9);
        LOAD_IN_128_D(in_h2_r, in_strides, 10, /* unused */ 0, v_in10, tw, cols,
                      cnt_128, load_multi_cols, 0);
        v_in10 = IN_H2_128_D(v_in10);
        LOAD_IN_128_D(in_h2_r, in_strides, 11, /* unused */ 0, v_in11, tw, cols,
                      cnt_128, load_multi_cols, 0);
        v_in11 = IN_H2_128_D(v_in11);
#else
        LOAD_IN_128_D(in_r, in_strides, 6, /* unused */ 0, v_in6, tw, cols,
                      cnt_128, load_multi_cols, is_contiguous_in);
        LOAD_IN_128_D(in_r, in_strides, 7, /* unused */ 0, v_in7, tw, cols,
                      cnt_128, load_multi_cols, is_contiguous_in);
        LOAD_IN_128_D(in_r, in_strides, 8, /* unused */ 0, v_in8, tw, cols,
                      cnt_128, load_multi_cols, is_contiguous_in);
        LOAD_IN_128_D(in_r, in_strides, 9, /* unused */ 0, v_in9, tw, cols,
                      cnt_128, load_multi_cols, is_contiguous_in);
        LOAD_IN_128_D(in_r, in_strides, 10, /* unused */ 0, v_in10, tw, cols,
                      cnt_128, load_multi_cols, is_contiguous_in);
        LOAD_IN_128_D(in_r, in_strides, 11, /* unused */ 0, v_in11, tw, cols,
                      cnt_128, load_multi_cols, is_contiguous_in);
#endif

        LD_128_D(in_r, v_in0);

        // Common operations
        v_av1 = _mm_add_pd(v_in0, v_in6);
        v_av2 = _mm_add_pd(v_in2, v_in4);
        v_av3 = _mm_add_pd(v_in8, v_in10);
        v_av4 = _mm_add_pd(v_in1, v_in5);
        v_av5 = _mm_add_pd(v_in7, v_in11);
        v_av6 = _mm_add_pd(v_in3, v_in9);

        v_cv1 = _mm_add_pd(v_av2, v_av3);
        v_cv2 = _mm_add_pd(v_av4, v_av5);
        v_cv3 = _mm_add_pd(v_av1, v_av6);
        v_cv4 = _mm_sub_pd(v_av1, v_av6);

        v_cv5 = _mm_add_pd(v_cv1, v_cv2);
        v_cv6 = _mm_sub_pd(v_cv1, v_cv2);

        // output point 1 & 7
        v_out0 = _mm_add_pd(v_cv3, v_cv5);
        v_out6 = _mm_add_pd(v_cv4, v_cv6);

        // Common operations
        v_av7 = _mm_sub_pd(v_in0, v_in6);
        v_av8 = _mm_sub_pd(v_in2, v_in4);
        v_av9 = _mm_sub_pd(v_in8, v_in10);
        v_av10 = _mm_sub_pd(v_in1, v_in5);
        v_av11 = _mm_sub_pd(v_in7, v_in11);
        v_av12 = _mm_sub_pd(v_in3, v_in9);

        v_cv1 = _mm_sub_pd(v_av8, v_av9);
        v_cv2 = _mm_sub_pd(v_av4, v_av5);
        v_cv7 = _mm_sub_pd(v_cv2, v_av12);
        v_cv8 = _mm_sub_pd(v_av7, v_cv1);

        v_tv1 = _mm_mul_pd(v_K3, v_cv7);
        v_tv1 = SWAP_RI_128_D(CONJ_128_D(v_tv1));

        // output point 4 & 10
        v_out3 = _mm_sub_pd(v_cv8, v_tv1);
        v_out9 = _mm_add_pd(v_cv8, v_tv1);

        v_tv1 = _mm_mul_pd(v_K3, v_av12);
        v_tv1 = CONJ_128_D(v_tv1);

        v_tv2 = _mm_mul_pd(v_K2, v_cv1);
        v_cv1 = _mm_add_pd(v_av7, v_tv2);
        v_cv7 = _mm_sub_pd(v_av10, v_av11);

        v_tv3 = _mm_mul_pd(v_K1, v_cv7);
        v_cv8 = _mm_add_pd(v_cv1, v_tv3);

        v_tv4 = _mm_mul_pd(v_K5, v_cv2);
        v_tv4 = CONJ_128_D(v_tv4);

        v_cv2 = _mm_sub_pd(v_av2, v_av3);
        v_tv5 = _mm_mul_pd(v_K4, v_cv2);
        v_tv5 = CONJ_128_D(v_tv5);

        v_cv2 = _mm_add_pd(v_tv1, v_tv4);
        v_cv7 = _mm_add_pd(v_cv2, v_tv5);
        v_cv7 = SWAP_RI_128_D(v_cv7);

        // output point 2 & 12
        v_out1 = _mm_sub_pd(v_cv8, v_cv7);
        v_out11 = _mm_add_pd(v_cv8, v_cv7);

        v_cv7 = _mm_sub_pd(v_cv1, v_tv3);
        v_cv8 = _mm_sub_pd(v_cv2, v_tv5);
        v_cv8 = SWAP_RI_128_D(v_cv8);

        // output point 6 & 8
        v_out5 = _mm_sub_pd(v_cv7, v_cv8);
        v_out7 = _mm_add_pd(v_cv7, v_cv8);

        v_tv1 = _mm_mul_pd(v_K2, v_cv6);
        v_cv1 = _mm_sub_pd(v_cv4, v_tv1);
        v_cv2 = _mm_add_pd(v_av8, v_av9);
        v_cv4 = _mm_add_pd(v_av10, v_av11);
        v_cv6 = _mm_add_pd(v_cv4, v_cv2);
        v_tv2 = _mm_mul_pd(v_K4, v_cv6);
        v_tv2 = SWAP_RI_128_D(CONJ_128_D(v_tv2));

        // output point 3 & 11
        v_out2 = _mm_sub_pd(v_cv1, v_tv2);
        v_out10 = _mm_add_pd(v_cv1, v_tv2);

        v_tv1 = _mm_mul_pd(v_K2, v_cv5);
        v_cv1 = _mm_sub_pd(v_cv3, v_tv1);
        v_cv6 = _mm_sub_pd(v_cv4, v_cv2);
        v_tv2 = _mm_mul_pd(v_K4, v_cv6);
        v_tv2 = SWAP_RI_128_D(CONJ_128_D(v_tv2));

        // output point 5 & 9
        v_out4 = _mm_sub_pd(v_cv1, v_tv2);
        v_out8 = _mm_add_pd(v_cv1, v_tv2);

        ST_128_D(out_r, v_out0);
        STORE_OUT_128_D(out_r, out_strides, 1, /* unused */ 0, v_out1, tw, cols,
                        cnt_128, load_multi_cols, is_contiguous_out);
        STORE_OUT_128_D(out_r, out_strides, 2, /* unused */ 0, v_out2, tw, cols,
                        cnt_128, load_multi_cols, is_contiguous_out);
        STORE_OUT_128_D(out_r, out_strides, 3, /* unused */ 0, v_out3, tw, cols,
                        cnt_128, load_multi_cols, is_contiguous_out);
        STORE_OUT_128_D(out_r, out_strides, 4, /* unused */ 0, v_out4, tw, cols,
                        cnt_128, load_multi_cols, is_contiguous_out);
        STORE_OUT_128_D(out_r, out_strides, 5, /* unused */ 0, v_out5, tw, cols,
                        cnt_128, load_multi_cols, is_contiguous_out);
#if defined(KERNEL_VARIANT_R2C)
        v_out6 = OUT_H2_128_D(v_out6);
        STORE_OUT_128_D(out_h2_r, out_strides, 6, /* unused */ 0, v_out6, tw,
                        cols, cnt_128, load_multi_cols, 0);
        v_out7 = OUT_H2_128_D(v_out7);
        STORE_OUT_128_D(out_h2_r, out_strides, 7, /* unused */ 0, v_out7, tw,
                        cols, cnt_128, load_multi_cols, 0);
        v_out8 = OUT_H2_128_D(v_out8);
        STORE_OUT_128_D(out_h2_r, out_strides, 8, /* unused */ 0, v_out8, tw,
                        cols, cnt_128, load_multi_cols, 0);
        v_out9 = OUT_H2_128_D(v_out9);
        STORE_OUT_128_D(out_h2_r, out_strides, 9, /* unused */ 0, v_out9, tw,
                        cols, cnt_128, load_multi_cols, 0);
        v_out10 = OUT_H2_128_D(v_out10);
        STORE_OUT_128_D(out_h2_r, out_strides, 10, /* unused */ 0, v_out10, tw,
                        cols, cnt_128, load_multi_cols, 0);
        v_out11 = OUT_H2_128_D(v_out11);
        STORE_OUT_128_D(out_h2_r, out_strides, 11, /* unused */ 0, v_out11, tw,
                        cols, cnt_128, load_multi_cols, 0);
#else
        STORE_OUT_128_D(out_r, out_strides, 6, /* unused */ 0, v_out6, tw, cols,
                        cnt_128, load_multi_cols, is_contiguous_out);
        STORE_OUT_128_D(out_r, out_strides, 7, /* unused */ 0, v_out7, tw, cols,
                        cnt_128, load_multi_cols, is_contiguous_out);
        STORE_OUT_128_D(out_r, out_strides, 8, /* unused */ 0, v_out8, tw, cols,
                        cnt_128, load_multi_cols, is_contiguous_out);
        STORE_OUT_128_D(out_r, out_strides, 9, /* unused */ 0, v_out9, tw, cols,
                        cnt_128, load_multi_cols, is_contiguous_out);
        STORE_OUT_128_D(out_r, out_strides, 10, /* unused */ 0, v_out10, tw,
                        cols, cnt_128, load_multi_cols, is_contiguous_out);
        STORE_OUT_128_D(out_r, out_strides, 11, /* unused */ 0, v_out11, tw,
                        cols, cnt_128, load_multi_cols, is_contiguous_out);
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
