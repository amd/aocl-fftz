// Copyright Advanced Micro Devices, Inc.
// SPDX-License-Identifier: BSD-3-Clause

/** @file twid_fft16.h
 *
 *  @brief The ISA generic kernel template for the radix 16 twiddle kernel
 *
 *  This file contains the DIT twiddle radix-16 FFT implementations for
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

    const FLOAT CRTM_16[4] = {
        0.92387953251128675612818318939678828682241662586364f,
        0.38268343236508977172845998403039886676134456248563f,
        0.70710678118654752440084436210484903928483593768847f,
        1.00000000000000000000000000000000000000000000000000f};

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

    VREGTYPE_S v_C1 = BCAST_S(CRTM_16[0]);
    VREGTYPE_S v_C2 = BCAST_S(CRTM_16[1]);
    VREGTYPE_S v_C3 = BCAST_S(CRTM_16[2]);
    VREGTYPE_S v_C4 = BCAST_S(CRTM_16[2]);
    VREGTYPE_S v_C5 = BCAST_S(CRTM_16[1]);
    VREGTYPE_S v_C6 = BCAST_S(CRTM_16[0]);
    VREGTYPE_S v_C7 = BCAST_S(CRTM_16[3]);

    INTP count;

    v_C2 = NEG_S(v_C2, flag);
    v_C4 = NEG_S(v_C4, flag);
    v_C6 = NEG_S(v_C6, flag);
    v_C7 = NEG_S(v_C7, flag);

    for (count = 0; count < N; count++)
    {
        VREGTYPE_S v_in0, v_in1, v_in2, v_in3, v_in4, v_in5;
        VREGTYPE_S v_in6, v_in7, v_in8, v_in9, v_in10;
        VREGTYPE_S v_in11, v_in12, v_in13, v_in14, v_in15;
        VREGTYPE_S v_tv1, v_tv2, v_tv3, v_tv4, v_tv5, v_tv6;
        VREGTYPE_S v_tv7, v_tv9, v_tv11, v_tv12;
        VREGTYPE_S v_av1, v_av2, v_av3, v_av4, v_av5, v_av6;
        VREGTYPE_S v_av7, v_av8, v_av9, v_av10, v_av12, v_av14;
        VREGTYPE_S v_av15, v_av16, v_av17, v_av18, v_av19, v_av21;
        VREGTYPE_S v_av22, v_av23, v_av24, v_av25, v_av20, v_av26;
        VREGTYPE_S v_av27, v_av28, v_av29, v_av30, v_av31;
        VREGTYPE_S v_av32, v_av33, v_av34, v_av35, v_av36;
        VREGTYPE_S v_av39, v_av40, v_av41, v_av42, v_av43;
        VREGTYPE_S v_cv1, v_cv2, v_cv3, v_cv4, v_cv5, v_cv6, v_cv7;
        VREGTYPE_S v_cv8, v_cv9, v_cv10, v_cv11, v_cv12, v_cv13;
        VREGTYPE_S v_cv14, v_cv15, v_cv16;
        VREGTYPE_S v_out0, v_out1, v_out2, v_out3, v_out4, v_out5, v_out6;
        VREGTYPE_S v_out7, v_out8, v_out9, v_out10, v_out11, v_out12;
        VREGTYPE_S v_out13, v_out14, v_out15;

        curr_in = in_r;
        curr_out = out_r;

        if (flag)
        {
            ITW_GATHER_S(curr_in, in_strides, 1, v_in_stride, v_in1, tw, cols,
                         (count * load_multi_cols * NUM_SETS_S),
                         load_multi_cols);
            ITW_GATHER_S(curr_in, in_strides, 2, v_in_stride, v_in2, tw, cols,
                         (count * load_multi_cols * NUM_SETS_S),
                         load_multi_cols);
            ITW_GATHER_S(curr_in, in_strides, 3, v_in_stride, v_in3, tw, cols,
                         (count * load_multi_cols * NUM_SETS_S),
                         load_multi_cols);
            ITW_GATHER_S(curr_in, in_strides, 4, v_in_stride, v_in4, tw, cols,
                         (count * load_multi_cols * NUM_SETS_S),
                         load_multi_cols);
            ITW_GATHER_S(curr_in, in_strides, 5, v_in_stride, v_in5, tw, cols,
                         (count * load_multi_cols * NUM_SETS_S),
                         load_multi_cols);
            ITW_GATHER_S(curr_in, in_strides, 6, v_in_stride, v_in6, tw, cols,
                         (count * load_multi_cols * NUM_SETS_S),
                         load_multi_cols);
            ITW_GATHER_S(curr_in, in_strides, 7, v_in_stride, v_in7, tw, cols,
                         (count * load_multi_cols * NUM_SETS_S),
                         load_multi_cols);
            ITW_GATHER_S(curr_in, in_strides, 8, v_in_stride, v_in8, tw, cols,
                         (count * load_multi_cols * NUM_SETS_S),
                         load_multi_cols);
            ITW_GATHER_S(curr_in, in_strides, 9, v_in_stride, v_in9, tw, cols,
                         (count * load_multi_cols * NUM_SETS_S),
                         load_multi_cols);
            ITW_GATHER_S(curr_in, in_strides, 10, v_in_stride, v_in10, tw, cols,
                         (count * load_multi_cols * NUM_SETS_S),
                         load_multi_cols);
            ITW_GATHER_S(curr_in, in_strides, 11, v_in_stride, v_in11, tw, cols,
                         (count * load_multi_cols * NUM_SETS_S),
                         load_multi_cols);
            ITW_GATHER_S(curr_in, in_strides, 12, v_in_stride, v_in12, tw, cols,
                         (count * load_multi_cols * NUM_SETS_S),
                         load_multi_cols);
            ITW_GATHER_S(curr_in, in_strides, 13, v_in_stride, v_in13, tw, cols,
                         (count * load_multi_cols * NUM_SETS_S),
                         load_multi_cols);
            ITW_GATHER_S(curr_in, in_strides, 14, v_in_stride, v_in14, tw, cols,
                         (count * load_multi_cols * NUM_SETS_S),
                         load_multi_cols);
            ITW_GATHER_S(curr_in, in_strides, 15, v_in_stride, v_in15, tw, cols,
                         (count * load_multi_cols * NUM_SETS_S),
                         load_multi_cols);
        }
        else
        {
            TW_GATHER_S(curr_in, in_strides, 1, v_in_stride, v_in1, tw, cols,
                        (count * load_multi_cols * NUM_SETS_S),
                        load_multi_cols);
            TW_GATHER_S(curr_in, in_strides, 2, v_in_stride, v_in2, tw, cols,
                        (count * load_multi_cols * NUM_SETS_S),
                        load_multi_cols);
            TW_GATHER_S(curr_in, in_strides, 3, v_in_stride, v_in3, tw, cols,
                        (count * load_multi_cols * NUM_SETS_S),
                        load_multi_cols);
            TW_GATHER_S(curr_in, in_strides, 4, v_in_stride, v_in4, tw, cols,
                        (count * load_multi_cols * NUM_SETS_S),
                        load_multi_cols);
            TW_GATHER_S(curr_in, in_strides, 5, v_in_stride, v_in5, tw, cols,
                        (count * load_multi_cols * NUM_SETS_S),
                        load_multi_cols);
            TW_GATHER_S(curr_in, in_strides, 6, v_in_stride, v_in6, tw, cols,
                        (count * load_multi_cols * NUM_SETS_S),
                        load_multi_cols);
            TW_GATHER_S(curr_in, in_strides, 7, v_in_stride, v_in7, tw, cols,
                        (count * load_multi_cols * NUM_SETS_S),
                        load_multi_cols);
            TW_GATHER_S(curr_in, in_strides, 8, v_in_stride, v_in8, tw, cols,
                        (count * load_multi_cols * NUM_SETS_S),
                        load_multi_cols);
            TW_GATHER_S(curr_in, in_strides, 9, v_in_stride, v_in9, tw, cols,
                        (count * load_multi_cols * NUM_SETS_S),
                        load_multi_cols);
            TW_GATHER_S(curr_in, in_strides, 10, v_in_stride, v_in10, tw, cols,
                        (count * load_multi_cols * NUM_SETS_S),
                        load_multi_cols);
            TW_GATHER_S(curr_in, in_strides, 11, v_in_stride, v_in11, tw, cols,
                        (count * load_multi_cols * NUM_SETS_S),
                        load_multi_cols);
            TW_GATHER_S(curr_in, in_strides, 12, v_in_stride, v_in12, tw, cols,
                        (count * load_multi_cols * NUM_SETS_S),
                        load_multi_cols);
            TW_GATHER_S(curr_in, in_strides, 13, v_in_stride, v_in13, tw, cols,
                        (count * load_multi_cols * NUM_SETS_S),
                        load_multi_cols);
            TW_GATHER_S(curr_in, in_strides, 14, v_in_stride, v_in14, tw, cols,
                        (count * load_multi_cols * NUM_SETS_S),
                        load_multi_cols);
            TW_GATHER_S(curr_in, in_strides, 15, v_in_stride, v_in15, tw, cols,
                        (count * load_multi_cols * NUM_SETS_S),
                        load_multi_cols);
        }

        GATHER_S(curr_in, v_in_stride, v_in0);

        // common calculations
        v_cv1 = ADD_S(v_in0, v_in8);
        v_cv2 = ADD_S(v_in1, v_in15);
        v_cv3 = ADD_S(v_in2, v_in14);
        v_cv4 = ADD_S(v_in3, v_in13);
        v_cv5 = ADD_S(v_in4, v_in12);
        v_cv6 = ADD_S(v_in5, v_in11);
        v_cv7 = ADD_S(v_in6, v_in10);
        v_cv8 = ADD_S(v_in7, v_in9);
        v_cv9 = SUB_S(v_in0, v_in8);
        v_cv10 = SUB_S(v_in1, v_in15);
        v_cv11 = SUB_S(v_in2, v_in14);
        v_cv12 = SUB_S(v_in3, v_in13);
        v_cv13 = SUB_S(v_in4, v_in12);
        v_cv14 = SUB_S(v_in5, v_in11);
        v_cv15 = SUB_S(v_in6, v_in10);
        v_cv16 = SUB_S(v_in7, v_in9);

        v_av1 = SUB_S(v_cv8, v_cv2);
        v_tv1 = MUL_S(v_C1, v_av1);
        v_av2 = SUB_S(v_cv7, v_cv3);
        v_tv2 = MUL_S(v_C3, v_av2);
        v_av3 = SUB_S(v_cv6, v_cv4);
        v_tv3 = MUL_S(v_C5, v_av3);
        v_av4 = SUB_S(v_cv9, v_tv2);
        v_av5 = ADD_S(v_tv3, v_tv1);
        v_av15 = SUB_S(v_av4, v_av5);
        v_av6 = ADD_S(v_cv16, v_cv10);
        v_tv4 = MUL_S(v_C2, v_av6);
        v_av7 = ADD_S(v_cv11, v_cv15);
        v_tv5 = MUL_S(v_C4, v_av7);
        v_av8 = ADD_S(v_cv12, v_cv14);
        v_tv6 = MUL_S(v_C6, v_av8);
        v_tv7 = MUL_S(v_C7, v_cv13);
        v_av9 = ADD_S(v_tv5, v_tv7);
        v_av41 = ADD_S(v_tv4, v_tv6);
        v_av10 = ADD_S(v_av41, v_av9);
        v_av10 = SWAP_RI_S(CONJ_S(v_av10));

        // Output point 2 : X[1]
        v_out1 = SUB_S(v_av15, v_av10);
        // Output point 16 : X[15]
        v_out15 = ADD_S(v_av15, v_av10);

        v_av12 = ADD_S(v_av4, v_av5);
        v_av14 = SUB_S(v_av41, v_av9);
        v_av14 = SWAP_RI_S(CONJ_S(v_av14));

        // Output point 8 : X[7]
        v_out7 = SUB_S(v_av12, v_av14);
        // Output point 10 : X[9]
        v_out9 = ADD_S(v_av12, v_av14);

        v_av16 = SUB_S(v_cv1, v_cv5);
        v_av17 = ADD_S(v_cv2, v_cv8);
        v_av18 = ADD_S(v_cv4, v_cv6);
        v_av19 = SUB_S(v_av17, v_av18);
        v_av20 = MUL_S(v_C3, v_av19);
        v_av21 = ADD_S(v_av16, v_av20);
        v_av22 = SUB_S(v_cv14, v_cv12);
        v_av23 = SUB_S(v_cv16, v_cv10);
        v_av24 = MUL_S(v_C4, ADD_S(v_av22, v_av23));
        v_av25 = SUB_S(v_cv15, v_cv11);
        v_av26 = MUL_S(v_C7, v_av25);
        v_av27 = ADD_S(v_av24, v_av26);
        v_av27 = SWAP_RI_S(CONJ_S(v_av27));

        // Output point 3 : X[2]
        v_out2 = ADD_S(v_av21, v_av27);
        // Output point 15 : X[14]
        v_out14 = SUB_S(v_av21, v_av27);

        v_av21 = SUB_S(v_av16, v_av20);
        v_av27 = SUB_S(v_av24, v_av26);
        v_av27 = SWAP_RI_S(CONJ_S(v_av27));

        // Output point 7 : X[6]
        v_out6 = ADD_S(v_av21, v_av27);
        // Output point 11 : X[10]
        v_out10 = SUB_S(v_av21, v_av27);

        v_av28 = SUB_S(MUL_S(v_C1, v_av3), MUL_S(v_C5, v_av1));

        v_av29 = ADD_S(v_cv9, v_tv2);
        v_av30 = ADD_S(v_av29, v_av28);
        v_tv9 = MUL_S(v_C2, v_av8);
        v_tv11 = MUL_S(v_C6, v_av6);
        v_av31 = SUB_S(v_tv11, v_tv9);
        v_av32 = SUB_S(v_tv5, v_tv7);
        v_av33 = ADD_S(v_av31, v_av32);
        v_av33 = SWAP_RI_S(CONJ_S(v_av33));

        // Output point 4 : X[3]
        v_out3 = SUB_S(v_av30, v_av33);
        // Output point 14 : X[13]
        v_out13 = ADD_S(v_av30, v_av33);

        v_av42 = SUB_S(v_av29, v_av28);
        v_av43 = SUB_S(v_av31, v_av32);
        v_av43 = SWAP_RI_S(CONJ_S(v_av43));

        // Output point 6 : X[5]
        v_out5 = SUB_S(v_av42, v_av43);
        // Output point 12 : X[11]
        v_out11 = ADD_S(v_av42, v_av43);

        v_av34 = ADD_S(v_cv1, v_cv5);
        v_av35 = ADD_S(v_cv3, v_cv7);
        v_av36 = SUB_S(v_av34, v_av35);
        v_tv12 = MUL_S(v_C7, SUB_S(v_av22, v_av23));
        v_tv12 = SWAP_RI_S(CONJ_S(v_tv12));

        // Output point 5 : X[4]
        v_out4 = SUB_S(v_av36, v_tv12);
        // Output point 13 : X[12]
        v_out12 = ADD_S(v_av36, v_tv12);

        v_av39 = ADD_S(v_av34, v_av35);
        v_av40 = ADD_S(v_av17, v_av18);

        // Output point 1 : X[0]
        v_out0 = ADD_S(v_av39, v_av40);
        // Output point 9 : X[8]
        v_out8 = SUB_S(v_av39, v_av40);

        SCATTER_S(curr_out, v_out_stride, v_out0);
        SCATTER_S(curr_out + out_strides[1], v_out_stride, v_out1);
        SCATTER_S(curr_out + out_strides[2], v_out_stride, v_out2);
        SCATTER_S(curr_out + out_strides[3], v_out_stride, v_out3);
        SCATTER_S(curr_out + out_strides[4], v_out_stride, v_out4);
        SCATTER_S(curr_out + out_strides[5], v_out_stride, v_out5);
        SCATTER_S(curr_out + out_strides[6], v_out_stride, v_out6);
        SCATTER_S(curr_out + out_strides[7], v_out_stride, v_out7);
        SCATTER_S(curr_out + out_strides[8], v_out_stride, v_out8);
        SCATTER_S(curr_out + out_strides[9], v_out_stride, v_out9);
        SCATTER_S(curr_out + out_strides[10], v_out_stride, v_out10);
        SCATTER_S(curr_out + out_strides[11], v_out_stride, v_out11);
        SCATTER_S(curr_out + out_strides[12], v_out_stride, v_out12);
        SCATTER_S(curr_out + out_strides[13], v_out_stride, v_out13);
        SCATTER_S(curr_out + out_strides[14], v_out_stride, v_out14);
        SCATTER_S(curr_out + out_strides[15], v_out_stride, v_out15);

        in_r += NUM_SETS_S * v_in_stride;
        out_r += NUM_SETS_S * v_out_stride;
    }

    // The following contains code that performs the FFT on the tail cases.
    // These tails are conditionally "instantiated" based on the ISA requested
    // by the "includee" file.

#if defined(KERNEL_USE_AVX512)
    if (remaining_sets >= NUM_SETS_256_S)
    {
        __m256 v_in0, v_in1, v_in2, v_in3, v_in4, v_in5;
        __m256 v_in6, v_in7, v_in8, v_in9, v_in10;
        __m256 v_in11, v_in12, v_in13, v_in14, v_in15;
        __m256 v_tv1, v_tv2, v_tv3, v_tv4, v_tv5, v_tv6;
        __m256 v_tv7, v_tv9, v_tv11, v_tv12;
        __m256 v_av1, v_av2, v_av3, v_av4, v_av5, v_av6;
        __m256 v_av7, v_av8, v_av9, v_av10, v_av12, v_av14;
        __m256 v_av15, v_av16, v_av17, v_av18, v_av19, v_av21;
        __m256 v_av22, v_av23, v_av24, v_av25, v_av20, v_av26;
        __m256 v_av27, v_av28, v_av29, v_av30, v_av31;
        __m256 v_av32, v_av33, v_av34, v_av35, v_av36;
        __m256 v_av39, v_av40, v_av41, v_av42, v_av43;
        __m256 v_cv1, v_cv2, v_cv3, v_cv4, v_cv5, v_cv6, v_cv7;
        __m256 v_cv8, v_cv9, v_cv10, v_cv11, v_cv12, v_cv13;
        __m256 v_cv14, v_cv15, v_cv16;
        __m256 v_out0, v_out1, v_out2, v_out3, v_out4, v_out5, v_out6;
        __m256 v_out7, v_out8, v_out9, v_out10, v_out11, v_out12;
        __m256 v_out13, v_out14, v_out15;

        __m256 v_K1 = CAST_512_TO_256_S(v_C1);
        __m256 v_K2 = CAST_512_TO_256_S(v_C2);
        __m256 v_K3 = CAST_512_TO_256_S(v_C3);
        __m256 v_K4 = CAST_512_TO_256_S(v_C4);
        __m256 v_K5 = CAST_512_TO_256_S(v_C5);
        __m256 v_K6 = CAST_512_TO_256_S(v_C6);
        __m256 v_K7 = CAST_512_TO_256_S(v_C7);

        curr_in = in_r;
        curr_out = out_r;

        if (flag)
        {
            ITW_GATHER_256_S(curr_in, in_strides, 1, v_in_stride, v_in1, tw,
                             cols, cnt_256, load_multi_cols);
            ITW_GATHER_256_S(curr_in, in_strides, 2, v_in_stride, v_in2, tw,
                             cols, cnt_256, load_multi_cols);
            ITW_GATHER_256_S(curr_in, in_strides, 3, v_in_stride, v_in3, tw,
                             cols, cnt_256, load_multi_cols);
            ITW_GATHER_256_S(curr_in, in_strides, 4, v_in_stride, v_in4, tw,
                             cols, cnt_256, load_multi_cols);
            ITW_GATHER_256_S(curr_in, in_strides, 5, v_in_stride, v_in5, tw,
                             cols, cnt_256, load_multi_cols);
            ITW_GATHER_256_S(curr_in, in_strides, 6, v_in_stride, v_in6, tw,
                             cols, cnt_256, load_multi_cols);
            ITW_GATHER_256_S(curr_in, in_strides, 7, v_in_stride, v_in7, tw,
                             cols, cnt_256, load_multi_cols);
            ITW_GATHER_256_S(curr_in, in_strides, 8, v_in_stride, v_in8, tw,
                             cols, cnt_256, load_multi_cols);
            ITW_GATHER_256_S(curr_in, in_strides, 9, v_in_stride, v_in9, tw,
                             cols, cnt_256, load_multi_cols);
            ITW_GATHER_256_S(curr_in, in_strides, 10, v_in_stride, v_in10, tw,
                             cols, cnt_256, load_multi_cols);
            ITW_GATHER_256_S(curr_in, in_strides, 11, v_in_stride, v_in11, tw,
                             cols, cnt_256, load_multi_cols);
            ITW_GATHER_256_S(curr_in, in_strides, 12, v_in_stride, v_in12, tw,
                             cols, cnt_256, load_multi_cols);
            ITW_GATHER_256_S(curr_in, in_strides, 13, v_in_stride, v_in13, tw,
                             cols, cnt_256, load_multi_cols);
            ITW_GATHER_256_S(curr_in, in_strides, 14, v_in_stride, v_in14, tw,
                             cols, cnt_256, load_multi_cols);
            ITW_GATHER_256_S(curr_in, in_strides, 15, v_in_stride, v_in15, tw,
                             cols, cnt_256, load_multi_cols);
        }
        else
        {
            TW_GATHER_256_S(curr_in, in_strides, 1, v_in_stride, v_in1, tw,
                            cols, cnt_256, load_multi_cols);
            TW_GATHER_256_S(curr_in, in_strides, 2, v_in_stride, v_in2, tw,
                            cols, cnt_256, load_multi_cols);
            TW_GATHER_256_S(curr_in, in_strides, 3, v_in_stride, v_in3, tw,
                            cols, cnt_256, load_multi_cols);
            TW_GATHER_256_S(curr_in, in_strides, 4, v_in_stride, v_in4, tw,
                            cols, cnt_256, load_multi_cols);
            TW_GATHER_256_S(curr_in, in_strides, 5, v_in_stride, v_in5, tw,
                            cols, cnt_256, load_multi_cols);
            TW_GATHER_256_S(curr_in, in_strides, 6, v_in_stride, v_in6, tw,
                            cols, cnt_256, load_multi_cols);
            TW_GATHER_256_S(curr_in, in_strides, 7, v_in_stride, v_in7, tw,
                            cols, cnt_256, load_multi_cols);
            TW_GATHER_256_S(curr_in, in_strides, 8, v_in_stride, v_in8, tw,
                            cols, cnt_256, load_multi_cols);
            TW_GATHER_256_S(curr_in, in_strides, 9, v_in_stride, v_in9, tw,
                            cols, cnt_256, load_multi_cols);
            TW_GATHER_256_S(curr_in, in_strides, 10, v_in_stride, v_in10, tw,
                            cols, cnt_256, load_multi_cols);
            TW_GATHER_256_S(curr_in, in_strides, 11, v_in_stride, v_in11, tw,
                            cols, cnt_256, load_multi_cols);
            TW_GATHER_256_S(curr_in, in_strides, 12, v_in_stride, v_in12, tw,
                            cols, cnt_256, load_multi_cols);
            TW_GATHER_256_S(curr_in, in_strides, 13, v_in_stride, v_in13, tw,
                            cols, cnt_256, load_multi_cols);
            TW_GATHER_256_S(curr_in, in_strides, 14, v_in_stride, v_in14, tw,
                            cols, cnt_256, load_multi_cols);
            TW_GATHER_256_S(curr_in, in_strides, 15, v_in_stride, v_in15, tw,
                            cols, cnt_256, load_multi_cols);
        }

        GATHER4_256_S(curr_in, v_in_stride, v_in0);

        // common calculations
        v_cv1 = _mm256_add_ps(v_in0, v_in8);
        v_cv2 = _mm256_add_ps(v_in1, v_in15);
        v_cv3 = _mm256_add_ps(v_in2, v_in14);
        v_cv4 = _mm256_add_ps(v_in3, v_in13);
        v_cv5 = _mm256_add_ps(v_in4, v_in12);
        v_cv6 = _mm256_add_ps(v_in5, v_in11);
        v_cv7 = _mm256_add_ps(v_in6, v_in10);
        v_cv8 = _mm256_add_ps(v_in7, v_in9);
        v_cv9 = _mm256_sub_ps(v_in0, v_in8);
        v_cv10 = _mm256_sub_ps(v_in1, v_in15);
        v_cv11 = _mm256_sub_ps(v_in2, v_in14);
        v_cv12 = _mm256_sub_ps(v_in3, v_in13);
        v_cv13 = _mm256_sub_ps(v_in4, v_in12);
        v_cv14 = _mm256_sub_ps(v_in5, v_in11);
        v_cv15 = _mm256_sub_ps(v_in6, v_in10);
        v_cv16 = _mm256_sub_ps(v_in7, v_in9);

        v_av1 = _mm256_sub_ps(v_cv8, v_cv2);
        v_tv1 = _mm256_mul_ps(v_K1, v_av1);
        v_av2 = _mm256_sub_ps(v_cv7, v_cv3);
        v_tv2 = _mm256_mul_ps(v_K3, v_av2);
        v_av3 = _mm256_sub_ps(v_cv6, v_cv4);
        v_tv3 = _mm256_mul_ps(v_K5, v_av3);
        v_av4 = _mm256_sub_ps(v_cv9, v_tv2);
        v_av5 = _mm256_add_ps(v_tv3, v_tv1);
        v_av15 = _mm256_sub_ps(v_av4, v_av5);
        v_av6 = _mm256_add_ps(v_cv16, v_cv10);
        v_tv4 = _mm256_mul_ps(v_K2, v_av6);
        v_av7 = _mm256_add_ps(v_cv11, v_cv15);
        v_tv5 = _mm256_mul_ps(v_K4, v_av7);
        v_av8 = _mm256_add_ps(v_cv12, v_cv14);
        v_tv6 = _mm256_mul_ps(v_K6, v_av8);
        v_tv7 = _mm256_mul_ps(v_K7, v_cv13);
        v_av9 = _mm256_add_ps(v_tv5, v_tv7);
        v_av41 = _mm256_add_ps(v_tv4, v_tv6);
        v_av10 = _mm256_add_ps(v_av41, v_av9);
        v_av10 = SWAP_RI_256_S(CONJ_256_S(v_av10));

        // Output point 2 : X[1]
        v_out1 = _mm256_sub_ps(v_av15, v_av10);
        // Output point 16 : X[15]
        v_out15 = _mm256_add_ps(v_av15, v_av10);

        v_av12 = _mm256_add_ps(v_av4, v_av5);
        v_av14 = _mm256_sub_ps(v_av41, v_av9);
        v_av14 = SWAP_RI_256_S(CONJ_256_S(v_av14));

        // Output point 8 : X[7]
        v_out7 = _mm256_sub_ps(v_av12, v_av14);
        // Output point 10 : X[9]
        v_out9 = _mm256_add_ps(v_av12, v_av14);

        v_av16 = _mm256_sub_ps(v_cv1, v_cv5);
        v_av17 = _mm256_add_ps(v_cv2, v_cv8);
        v_av18 = _mm256_add_ps(v_cv4, v_cv6);
        v_av19 = _mm256_sub_ps(v_av17, v_av18);
        v_av20 = _mm256_mul_ps(v_K3, v_av19);
        v_av21 = _mm256_add_ps(v_av16, v_av20);
        v_av22 = _mm256_sub_ps(v_cv14, v_cv12);
        v_av23 = _mm256_sub_ps(v_cv16, v_cv10);
        v_av24 = _mm256_mul_ps(v_K4, _mm256_add_ps(v_av22, v_av23));
        v_av25 = _mm256_sub_ps(v_cv15, v_cv11);
        v_av26 = _mm256_mul_ps(v_K7, v_av25);
        v_av27 = _mm256_add_ps(v_av24, v_av26);
        v_av27 = SWAP_RI_256_S(CONJ_256_S(v_av27));

        // Output point 3 : X[2]
        v_out2 = _mm256_add_ps(v_av21, v_av27);
        // Output point 15 : X[14]
        v_out14 = _mm256_sub_ps(v_av21, v_av27);

        v_av21 = _mm256_sub_ps(v_av16, v_av20);
        v_av27 = _mm256_sub_ps(v_av24, v_av26);
        v_av27 = SWAP_RI_256_S(CONJ_256_S(v_av27));

        // Output point 7 : X[6]
        v_out6 = _mm256_add_ps(v_av21, v_av27);
        // Output point 11 : X[10]
        v_out10 = _mm256_sub_ps(v_av21, v_av27);

        v_av28 = _mm256_sub_ps(_mm256_mul_ps(v_K1, v_av3),
                               _mm256_mul_ps(v_K5, v_av1));

        v_av29 = _mm256_add_ps(v_cv9, v_tv2);
        v_av30 = _mm256_add_ps(v_av29, v_av28);
        v_tv9 = _mm256_mul_ps(v_K2, v_av8);
        v_tv11 = _mm256_mul_ps(v_K6, v_av6);
        v_av31 = _mm256_sub_ps(v_tv11, v_tv9);
        v_av32 = _mm256_sub_ps(v_tv5, v_tv7);
        v_av33 = _mm256_add_ps(v_av31, v_av32);
        v_av33 = SWAP_RI_256_S(CONJ_256_S(v_av33));

        // Output point 4 : X[3]
        v_out3 = _mm256_sub_ps(v_av30, v_av33);
        // Output point 14 : X[13]
        v_out13 = _mm256_add_ps(v_av30, v_av33);

        v_av42 = _mm256_sub_ps(v_av29, v_av28);
        v_av43 = _mm256_sub_ps(v_av31, v_av32);
        v_av43 = SWAP_RI_256_S(CONJ_256_S(v_av43));

        // Output point 6 : X[5]
        v_out5 = _mm256_sub_ps(v_av42, v_av43);
        // Output point 12 : X[11]
        v_out11 = _mm256_add_ps(v_av42, v_av43);

        v_av34 = _mm256_add_ps(v_cv1, v_cv5);
        v_av35 = _mm256_add_ps(v_cv3, v_cv7);
        v_av36 = _mm256_sub_ps(v_av34, v_av35);
        v_tv12 = _mm256_mul_ps(v_K7, _mm256_sub_ps(v_av22, v_av23));
        v_tv12 = SWAP_RI_256_S(CONJ_256_S(v_tv12));

        // Output point 5 : X[4]
        v_out4 = _mm256_sub_ps(v_av36, v_tv12);
        // Output point 13 : X[12]
        v_out12 = _mm256_add_ps(v_av36, v_tv12);

        v_av39 = _mm256_add_ps(v_av34, v_av35);
        v_av40 = _mm256_add_ps(v_av17, v_av18);

        // Output point 1 : X[0]
        v_out0 = _mm256_add_ps(v_av39, v_av40);
        // Output point 9 : X[8]
        v_out8 = _mm256_sub_ps(v_av39, v_av40);

        SCATTER4_256_S(curr_out, v_out_stride, v_out0);
        SCATTER4_256_S(curr_out + out_strides[1], v_out_stride, v_out1);
        SCATTER4_256_S(curr_out + out_strides[2], v_out_stride, v_out2);
        SCATTER4_256_S(curr_out + out_strides[3], v_out_stride, v_out3);
        SCATTER4_256_S(curr_out + out_strides[4], v_out_stride, v_out4);
        SCATTER4_256_S(curr_out + out_strides[5], v_out_stride, v_out5);
        SCATTER4_256_S(curr_out + out_strides[6], v_out_stride, v_out6);
        SCATTER4_256_S(curr_out + out_strides[7], v_out_stride, v_out7);
        SCATTER4_256_S(curr_out + out_strides[8], v_out_stride, v_out8);
        SCATTER4_256_S(curr_out + out_strides[9], v_out_stride, v_out9);
        SCATTER4_256_S(curr_out + out_strides[10], v_out_stride, v_out10);
        SCATTER4_256_S(curr_out + out_strides[11], v_out_stride, v_out11);
        SCATTER4_256_S(curr_out + out_strides[12], v_out_stride, v_out12);
        SCATTER4_256_S(curr_out + out_strides[13], v_out_stride, v_out13);
        SCATTER4_256_S(curr_out + out_strides[14], v_out_stride, v_out14);
        SCATTER4_256_S(curr_out + out_strides[15], v_out_stride, v_out15);

        in_r += NUM_SETS_256_S * v_in_stride;
        out_r += NUM_SETS_256_S * v_out_stride;
        remaining_sets = remaining_sets - NUM_SETS_256_S;
    }
#endif

#if defined(KERNEL_USE_AVX512) || defined(KERNEL_USE_AVX256)
    if (remaining_sets >= NUM_SETS_128_S)
    {
        __m128 v_in0, v_in1, v_in2, v_in3, v_in4, v_in5;
        __m128 v_in6, v_in7, v_in8, v_in9, v_in10;
        __m128 v_in11, v_in12, v_in13, v_in14, v_in15;
        __m128 v_tv1, v_tv2, v_tv3, v_tv4, v_tv5, v_tv6;
        __m128 v_tv7, v_tv9, v_tv11, v_tv12;
        __m128 v_av1, v_av2, v_av3, v_av4, v_av5, v_av6;
        __m128 v_av7, v_av8, v_av9, v_av10, v_av12, v_av14;
        __m128 v_av15, v_av16, v_av17, v_av18, v_av19, v_av21;
        __m128 v_av22, v_av23, v_av24, v_av25, v_av20, v_av26;
        __m128 v_av27, v_av28, v_av29, v_av30, v_av31;
        __m128 v_av32, v_av33, v_av34, v_av35, v_av36;
        __m128 v_av39, v_av40, v_av41, v_av42, v_av43;
        __m128 v_cv1, v_cv2, v_cv3, v_cv4, v_cv5, v_cv6, v_cv7;
        __m128 v_cv8, v_cv9, v_cv10, v_cv11, v_cv12, v_cv13;
        __m128 v_cv14, v_cv15, v_cv16;
        __m128 v_out0, v_out1, v_out2, v_out3, v_out4, v_out5, v_out6;
        __m128 v_out7, v_out8, v_out9, v_out10, v_out11, v_out12;
        __m128 v_out13, v_out14, v_out15;

#if defined(KERNEL_USE_AVX512)
        __m128 v_K1 = CAST_512_TO_128_S(v_C1);
        __m128 v_K2 = CAST_512_TO_128_S(v_C2);
        __m128 v_K3 = CAST_512_TO_128_S(v_C3);
        __m128 v_K4 = CAST_512_TO_128_S(v_C4);
        __m128 v_K5 = CAST_512_TO_128_S(v_C5);
        __m128 v_K6 = CAST_512_TO_128_S(v_C6);
        __m128 v_K7 = CAST_512_TO_128_S(v_C7);
#elif defined(KERNEL_USE_AVX256)
        __m128 v_K1 = CAST_256_TO_128_S(v_C1);
        __m128 v_K2 = CAST_256_TO_128_S(v_C2);
        __m128 v_K3 = CAST_256_TO_128_S(v_C3);
        __m128 v_K4 = CAST_256_TO_128_S(v_C4);
        __m128 v_K5 = CAST_256_TO_128_S(v_C5);
        __m128 v_K6 = CAST_256_TO_128_S(v_C6);
        __m128 v_K7 = CAST_256_TO_128_S(v_C7);
#endif

        curr_in = in_r;
        curr_out = out_r;

        if (flag)
        {
            ITW_GATHER_128_S(curr_in, in_strides, 1, v_in_stride, v_in1, tw,
                             cols, cnt_128, load_multi_cols);
            ITW_GATHER_128_S(curr_in, in_strides, 2, v_in_stride, v_in2, tw,
                             cols, cnt_128, load_multi_cols);
            ITW_GATHER_128_S(curr_in, in_strides, 3, v_in_stride, v_in3, tw,
                             cols, cnt_128, load_multi_cols);
            ITW_GATHER_128_S(curr_in, in_strides, 4, v_in_stride, v_in4, tw,
                             cols, cnt_128, load_multi_cols);
            ITW_GATHER_128_S(curr_in, in_strides, 5, v_in_stride, v_in5, tw,
                             cols, cnt_128, load_multi_cols);
            ITW_GATHER_128_S(curr_in, in_strides, 6, v_in_stride, v_in6, tw,
                             cols, cnt_128, load_multi_cols);
            ITW_GATHER_128_S(curr_in, in_strides, 7, v_in_stride, v_in7, tw,
                             cols, cnt_128, load_multi_cols);
            ITW_GATHER_128_S(curr_in, in_strides, 8, v_in_stride, v_in8, tw,
                             cols, cnt_128, load_multi_cols);
            ITW_GATHER_128_S(curr_in, in_strides, 9, v_in_stride, v_in9, tw,
                             cols, cnt_128, load_multi_cols);
            ITW_GATHER_128_S(curr_in, in_strides, 10, v_in_stride, v_in10, tw,
                             cols, cnt_128, load_multi_cols);
            ITW_GATHER_128_S(curr_in, in_strides, 11, v_in_stride, v_in11, tw,
                             cols, cnt_128, load_multi_cols);
            ITW_GATHER_128_S(curr_in, in_strides, 12, v_in_stride, v_in12, tw,
                             cols, cnt_128, load_multi_cols);
            ITW_GATHER_128_S(curr_in, in_strides, 13, v_in_stride, v_in13, tw,
                             cols, cnt_128, load_multi_cols);
            ITW_GATHER_128_S(curr_in, in_strides, 14, v_in_stride, v_in14, tw,
                             cols, cnt_128, load_multi_cols);
            ITW_GATHER_128_S(curr_in, in_strides, 15, v_in_stride, v_in15, tw,
                             cols, cnt_128, load_multi_cols);
        }
        else
        {
            TW_GATHER_128_S(curr_in, in_strides, 1, v_in_stride, v_in1, tw,
                            cols, cnt_128, load_multi_cols);
            TW_GATHER_128_S(curr_in, in_strides, 2, v_in_stride, v_in2, tw,
                            cols, cnt_128, load_multi_cols);
            TW_GATHER_128_S(curr_in, in_strides, 3, v_in_stride, v_in3, tw,
                            cols, cnt_128, load_multi_cols);
            TW_GATHER_128_S(curr_in, in_strides, 4, v_in_stride, v_in4, tw,
                            cols, cnt_128, load_multi_cols);
            TW_GATHER_128_S(curr_in, in_strides, 5, v_in_stride, v_in5, tw,
                            cols, cnt_128, load_multi_cols);
            TW_GATHER_128_S(curr_in, in_strides, 6, v_in_stride, v_in6, tw,
                            cols, cnt_128, load_multi_cols);
            TW_GATHER_128_S(curr_in, in_strides, 7, v_in_stride, v_in7, tw,
                            cols, cnt_128, load_multi_cols);
            TW_GATHER_128_S(curr_in, in_strides, 8, v_in_stride, v_in8, tw,
                            cols, cnt_128, load_multi_cols);
            TW_GATHER_128_S(curr_in, in_strides, 9, v_in_stride, v_in9, tw,
                            cols, cnt_128, load_multi_cols);
            TW_GATHER_128_S(curr_in, in_strides, 10, v_in_stride, v_in10, tw,
                            cols, cnt_128, load_multi_cols);
            TW_GATHER_128_S(curr_in, in_strides, 11, v_in_stride, v_in11, tw,
                            cols, cnt_128, load_multi_cols);
            TW_GATHER_128_S(curr_in, in_strides, 12, v_in_stride, v_in12, tw,
                            cols, cnt_128, load_multi_cols);
            TW_GATHER_128_S(curr_in, in_strides, 13, v_in_stride, v_in13, tw,
                            cols, cnt_128, load_multi_cols);
            TW_GATHER_128_S(curr_in, in_strides, 14, v_in_stride, v_in14, tw,
                            cols, cnt_128, load_multi_cols);
            TW_GATHER_128_S(curr_in, in_strides, 15, v_in_stride, v_in15, tw,
                            cols, cnt_128, load_multi_cols);
        }

        GATHER2_128_S(curr_in, v_in_stride, v_in0);

        // common calculations
        v_cv1 = _mm_add_ps(v_in0, v_in8);
        v_cv2 = _mm_add_ps(v_in1, v_in15);
        v_cv3 = _mm_add_ps(v_in2, v_in14);
        v_cv4 = _mm_add_ps(v_in3, v_in13);
        v_cv5 = _mm_add_ps(v_in4, v_in12);
        v_cv6 = _mm_add_ps(v_in5, v_in11);
        v_cv7 = _mm_add_ps(v_in6, v_in10);
        v_cv8 = _mm_add_ps(v_in7, v_in9);
        v_cv9 = _mm_sub_ps(v_in0, v_in8);
        v_cv10 = _mm_sub_ps(v_in1, v_in15);
        v_cv11 = _mm_sub_ps(v_in2, v_in14);
        v_cv12 = _mm_sub_ps(v_in3, v_in13);
        v_cv13 = _mm_sub_ps(v_in4, v_in12);
        v_cv14 = _mm_sub_ps(v_in5, v_in11);
        v_cv15 = _mm_sub_ps(v_in6, v_in10);
        v_cv16 = _mm_sub_ps(v_in7, v_in9);

        v_av1 = _mm_sub_ps(v_cv8, v_cv2);
        v_tv1 = _mm_mul_ps(v_K1, v_av1);
        v_av2 = _mm_sub_ps(v_cv7, v_cv3);
        v_tv2 = _mm_mul_ps(v_K3, v_av2);
        v_av3 = _mm_sub_ps(v_cv6, v_cv4);
        v_tv3 = _mm_mul_ps(v_K5, v_av3);
        v_av4 = _mm_sub_ps(v_cv9, v_tv2);
        v_av5 = _mm_add_ps(v_tv3, v_tv1);
        v_av15 = _mm_sub_ps(v_av4, v_av5);
        v_av6 = _mm_add_ps(v_cv16, v_cv10);
        v_tv4 = _mm_mul_ps(v_K2, v_av6);
        v_av7 = _mm_add_ps(v_cv11, v_cv15);
        v_tv5 = _mm_mul_ps(v_K4, v_av7);
        v_av8 = _mm_add_ps(v_cv12, v_cv14);
        v_tv6 = _mm_mul_ps(v_K6, v_av8);
        v_tv7 = _mm_mul_ps(v_K7, v_cv13);
        v_av9 = _mm_add_ps(v_tv5, v_tv7);
        v_av41 = _mm_add_ps(v_tv4, v_tv6);
        v_av10 = _mm_add_ps(v_av41, v_av9);
        v_av10 = SWAP_RI_128_S(CONJ_128_S(v_av10));

        // Output point 2 : X[1]
        v_out1 = _mm_sub_ps(v_av15, v_av10);
        // Output point 16 : X[15]
        v_out15 = _mm_add_ps(v_av15, v_av10);

        v_av12 = _mm_add_ps(v_av4, v_av5);
        v_av14 = _mm_sub_ps(v_av41, v_av9);
        v_av14 = SWAP_RI_128_S(CONJ_128_S(v_av14));

        // Output point 8 : X[7]
        v_out7 = _mm_sub_ps(v_av12, v_av14);
        // Output point 10 : X[9]
        v_out9 = _mm_add_ps(v_av12, v_av14);

        v_av16 = _mm_sub_ps(v_cv1, v_cv5);
        v_av17 = _mm_add_ps(v_cv2, v_cv8);
        v_av18 = _mm_add_ps(v_cv4, v_cv6);
        v_av19 = _mm_sub_ps(v_av17, v_av18);
        v_av20 = _mm_mul_ps(v_K3, v_av19);
        v_av21 = _mm_add_ps(v_av16, v_av20);
        v_av22 = _mm_sub_ps(v_cv14, v_cv12);
        v_av23 = _mm_sub_ps(v_cv16, v_cv10);
        v_av24 = _mm_mul_ps(v_K4, _mm_add_ps(v_av22, v_av23));
        v_av25 = _mm_sub_ps(v_cv15, v_cv11);
        v_av26 = _mm_mul_ps(v_K7, v_av25);
        v_av27 = _mm_add_ps(v_av24, v_av26);
        v_av27 = SWAP_RI_128_S(CONJ_128_S(v_av27));

        // Output point 3 : X[2]
        v_out2 = _mm_add_ps(v_av21, v_av27);
        // Output point 15 : X[14]
        v_out14 = _mm_sub_ps(v_av21, v_av27);

        v_av21 = _mm_sub_ps(v_av16, v_av20);
        v_av27 = _mm_sub_ps(v_av24, v_av26);
        v_av27 = SWAP_RI_128_S(CONJ_128_S(v_av27));

        // Output point 7 : X[6]
        v_out6 = _mm_add_ps(v_av21, v_av27);
        // Output point 11 : X[10]
        v_out10 = _mm_sub_ps(v_av21, v_av27);

        v_av28 = _mm_sub_ps(_mm_mul_ps(v_K1, v_av3), _mm_mul_ps(v_K5, v_av1));
        v_av29 = _mm_add_ps(v_cv9, v_tv2);
        v_av30 = _mm_add_ps(v_av29, v_av28);
        v_tv9 = _mm_mul_ps(v_K2, v_av8);
        v_tv11 = _mm_mul_ps(v_K6, v_av6);
        v_av31 = _mm_sub_ps(v_tv11, v_tv9);
        v_av32 = _mm_sub_ps(v_tv5, v_tv7);
        v_av33 = _mm_add_ps(v_av31, v_av32);
        v_av33 = SWAP_RI_128_S(CONJ_128_S(v_av33));

        // Output point 4 : X[3]
        v_out3 = _mm_sub_ps(v_av30, v_av33);
        // Output point 14 : X[13]

        v_out13 = _mm_add_ps(v_av30, v_av33);
        v_av42 = _mm_sub_ps(v_av29, v_av28);
        v_av43 = _mm_sub_ps(v_av31, v_av32);
        v_av43 = SWAP_RI_128_S(CONJ_128_S(v_av43));

        // Output point 6 : X[5]
        v_out5 = _mm_sub_ps(v_av42, v_av43);
        // Output point 12 : X[11]
        v_out11 = _mm_add_ps(v_av42, v_av43);

        v_av34 = _mm_add_ps(v_cv1, v_cv5);
        v_av35 = _mm_add_ps(v_cv3, v_cv7);
        v_av36 = _mm_sub_ps(v_av34, v_av35);
        v_tv12 = _mm_mul_ps(v_K7, _mm_sub_ps(v_av22, v_av23));
        v_tv12 = SWAP_RI_128_S(CONJ_128_S(v_tv12));

        // Output point 5 : X[4]
        v_out4 = _mm_sub_ps(v_av36, v_tv12);
        // Output point 13 : X[12]
        v_out12 = _mm_add_ps(v_av36, v_tv12);

        v_av39 = _mm_add_ps(v_av34, v_av35);
        v_av40 = _mm_add_ps(v_av17, v_av18);

        // Output point 1 : X[0]
        v_out0 = _mm_add_ps(v_av39, v_av40);
        // Output point 9 : X[8]
        v_out8 = _mm_sub_ps(v_av39, v_av40);

        SCATTER2_128_S(curr_out, v_out_stride, v_out0);
        SCATTER2_128_S(curr_out + out_strides[1], v_out_stride, v_out1);
        SCATTER2_128_S(curr_out + out_strides[2], v_out_stride, v_out2);
        SCATTER2_128_S(curr_out + out_strides[3], v_out_stride, v_out3);
        SCATTER2_128_S(curr_out + out_strides[4], v_out_stride, v_out4);
        SCATTER2_128_S(curr_out + out_strides[5], v_out_stride, v_out5);
        SCATTER2_128_S(curr_out + out_strides[6], v_out_stride, v_out6);
        SCATTER2_128_S(curr_out + out_strides[7], v_out_stride, v_out7);
        SCATTER2_128_S(curr_out + out_strides[8], v_out_stride, v_out8);
        SCATTER2_128_S(curr_out + out_strides[9], v_out_stride, v_out9);
        SCATTER2_128_S(curr_out + out_strides[10], v_out_stride, v_out10);
        SCATTER2_128_S(curr_out + out_strides[11], v_out_stride, v_out11);
        SCATTER2_128_S(curr_out + out_strides[12], v_out_stride, v_out12);
        SCATTER2_128_S(curr_out + out_strides[13], v_out_stride, v_out13);
        SCATTER2_128_S(curr_out + out_strides[14], v_out_stride, v_out14);
        SCATTER2_128_S(curr_out + out_strides[15], v_out_stride, v_out15);

        in_r = in_r + (v_in_stride << 1);
        out_r = out_r + (v_out_stride << 1);
        remaining_sets = remaining_sets - NUM_SETS_128_S;
    }
#endif

#if defined(KERNEL_USE_AVX512) || defined(KERNEL_USE_AVX256) ||                      \
    defined(KERNEL_USE_AVX128)
    if (remaining_sets & 1)
    {
        __m128 v_in0, v_in1, v_in2, v_in3, v_in4, v_in5;
        __m128 v_in6, v_in7, v_in8, v_in9, v_in10;
        __m128 v_in11, v_in12, v_in13, v_in14, v_in15;
        __m128 v_tv1, v_tv2, v_tv3, v_tv4, v_tv5, v_tv6;
        __m128 v_tv7, v_tv9, v_tv11, v_tv12;
        __m128 v_av1, v_av2, v_av3, v_av4, v_av5, v_av6;
        __m128 v_av7, v_av8, v_av9, v_av10, v_av12, v_av14;
        __m128 v_av15, v_av16, v_av17, v_av18, v_av19, v_av21;
        __m128 v_av22, v_av23, v_av24, v_av25, v_av20, v_av26;
        __m128 v_av27, v_av28, v_av29, v_av30, v_av31;
        __m128 v_av32, v_av33, v_av34, v_av35, v_av36;
        __m128 v_av39, v_av40, v_av41, v_av42, v_av43;
        __m128 v_cv1, v_cv2, v_cv3, v_cv4, v_cv5, v_cv6, v_cv7;
        __m128 v_cv8, v_cv9, v_cv10, v_cv11, v_cv12, v_cv13;
        __m128 v_cv14, v_cv15, v_cv16;
        __m128 v_out0, v_out1, v_out2, v_out3, v_out4, v_out5, v_out6;
        __m128 v_out7, v_out8, v_out9, v_out10, v_out11, v_out12;
        __m128 v_out13, v_out14, v_out15;

#if defined(KERNEL_USE_AVX512)
        __m128 v_K1 = CAST_512_TO_128_S(v_C1);
        __m128 v_K2 = CAST_512_TO_128_S(v_C2);
        __m128 v_K3 = CAST_512_TO_128_S(v_C3);
        __m128 v_K4 = CAST_512_TO_128_S(v_C4);
        __m128 v_K5 = CAST_512_TO_128_S(v_C5);
        __m128 v_K6 = CAST_512_TO_128_S(v_C6);
        __m128 v_K7 = CAST_512_TO_128_S(v_C7);
#elif defined(KERNEL_USE_AVX256)
        __m128 v_K1 = CAST_256_TO_128_S(v_C1);
        __m128 v_K2 = CAST_256_TO_128_S(v_C2);
        __m128 v_K3 = CAST_256_TO_128_S(v_C3);
        __m128 v_K4 = CAST_256_TO_128_S(v_C4);
        __m128 v_K5 = CAST_256_TO_128_S(v_C5);
        __m128 v_K6 = CAST_256_TO_128_S(v_C6);
        __m128 v_K7 = CAST_256_TO_128_S(v_C7);
#elif defined(KERNEL_USE_AVX128)
        __m128 v_K1 = v_C1;
        __m128 v_K2 = v_C2;
        __m128 v_K3 = v_C3;
        __m128 v_K4 = v_C4;
        __m128 v_K5 = v_C5;
        __m128 v_K6 = v_C6;
        __m128 v_K7 = v_C7;
#endif

        curr_in = in_r;
        curr_out = out_r;
        if (flag)
        {
            ITW_GATHER_LOW_128_S(curr_in, in_strides, 1, v_in1, tw, cols,
                                 cnt_128_low, load_multi_cols);
            ITW_GATHER_LOW_128_S(curr_in, in_strides, 2, v_in2, tw, cols,
                                 cnt_128_low, load_multi_cols);
            ITW_GATHER_LOW_128_S(curr_in, in_strides, 3, v_in3, tw, cols,
                                 cnt_128_low, load_multi_cols);
            ITW_GATHER_LOW_128_S(curr_in, in_strides, 4, v_in4, tw, cols,
                                 cnt_128_low, load_multi_cols);
            ITW_GATHER_LOW_128_S(curr_in, in_strides, 5, v_in5, tw, cols,
                                 cnt_128_low, load_multi_cols);
            ITW_GATHER_LOW_128_S(curr_in, in_strides, 6, v_in6, tw, cols,
                                 cnt_128_low, load_multi_cols);
            ITW_GATHER_LOW_128_S(curr_in, in_strides, 7, v_in7, tw, cols,
                                 cnt_128_low, load_multi_cols);
            ITW_GATHER_LOW_128_S(curr_in, in_strides, 8, v_in8, tw, cols,
                                 cnt_128_low, load_multi_cols);
            ITW_GATHER_LOW_128_S(curr_in, in_strides, 9, v_in9, tw, cols,
                                 cnt_128_low, load_multi_cols);
            ITW_GATHER_LOW_128_S(curr_in, in_strides, 10, v_in10, tw, cols,
                                 cnt_128_low, load_multi_cols);
            ITW_GATHER_LOW_128_S(curr_in, in_strides, 11, v_in11, tw, cols,
                                 cnt_128_low, load_multi_cols);
            ITW_GATHER_LOW_128_S(curr_in, in_strides, 12, v_in12, tw, cols,
                                 cnt_128_low, load_multi_cols);
            ITW_GATHER_LOW_128_S(curr_in, in_strides, 13, v_in13, tw, cols,
                                 cnt_128_low, load_multi_cols);
            ITW_GATHER_LOW_128_S(curr_in, in_strides, 14, v_in14, tw, cols,
                                 cnt_128_low, load_multi_cols);
            ITW_GATHER_LOW_128_S(curr_in, in_strides, 15, v_in15, tw, cols,
                                 cnt_128_low, load_multi_cols);
        }
        else
        {
            TW_GATHER_LOW_128_S(curr_in, in_strides, 1, v_in1, tw, cols,
                                cnt_128_low, load_multi_cols);
            TW_GATHER_LOW_128_S(curr_in, in_strides, 2, v_in2, tw, cols,
                                cnt_128_low, load_multi_cols);
            TW_GATHER_LOW_128_S(curr_in, in_strides, 3, v_in3, tw, cols,
                                cnt_128_low, load_multi_cols);
            TW_GATHER_LOW_128_S(curr_in, in_strides, 4, v_in4, tw, cols,
                                cnt_128_low, load_multi_cols);
            TW_GATHER_LOW_128_S(curr_in, in_strides, 5, v_in5, tw, cols,
                                cnt_128_low, load_multi_cols);
            TW_GATHER_LOW_128_S(curr_in, in_strides, 6, v_in6, tw, cols,
                                cnt_128_low, load_multi_cols);
            TW_GATHER_LOW_128_S(curr_in, in_strides, 7, v_in7, tw, cols,
                                cnt_128_low, load_multi_cols);
            TW_GATHER_LOW_128_S(curr_in, in_strides, 8, v_in8, tw, cols,
                                cnt_128_low, load_multi_cols);
            TW_GATHER_LOW_128_S(curr_in, in_strides, 9, v_in9, tw, cols,
                                cnt_128_low, load_multi_cols);
            TW_GATHER_LOW_128_S(curr_in, in_strides, 10, v_in10, tw, cols,
                                cnt_128_low, load_multi_cols);
            TW_GATHER_LOW_128_S(curr_in, in_strides, 11, v_in11, tw, cols,
                                cnt_128_low, load_multi_cols);
            TW_GATHER_LOW_128_S(curr_in, in_strides, 12, v_in12, tw, cols,
                                cnt_128_low, load_multi_cols);
            TW_GATHER_LOW_128_S(curr_in, in_strides, 13, v_in13, tw, cols,
                                cnt_128_low, load_multi_cols);
            TW_GATHER_LOW_128_S(curr_in, in_strides, 14, v_in14, tw, cols,
                                cnt_128_low, load_multi_cols);
            TW_GATHER_LOW_128_S(curr_in, in_strides, 15, v_in15, tw, cols,
                                cnt_128_low, load_multi_cols);
        }

        LD_LOW_128_S(curr_in, v_in0);

        // common calculations
        v_cv1 = _mm_add_ps(v_in0, v_in8);
        v_cv2 = _mm_add_ps(v_in1, v_in15);
        v_cv3 = _mm_add_ps(v_in2, v_in14);
        v_cv4 = _mm_add_ps(v_in3, v_in13);
        v_cv5 = _mm_add_ps(v_in4, v_in12);
        v_cv6 = _mm_add_ps(v_in5, v_in11);
        v_cv7 = _mm_add_ps(v_in6, v_in10);
        v_cv8 = _mm_add_ps(v_in7, v_in9);
        v_cv9 = _mm_sub_ps(v_in0, v_in8);
        v_cv10 = _mm_sub_ps(v_in1, v_in15);
        v_cv11 = _mm_sub_ps(v_in2, v_in14);
        v_cv12 = _mm_sub_ps(v_in3, v_in13);
        v_cv13 = _mm_sub_ps(v_in4, v_in12);
        v_cv14 = _mm_sub_ps(v_in5, v_in11);
        v_cv15 = _mm_sub_ps(v_in6, v_in10);
        v_cv16 = _mm_sub_ps(v_in7, v_in9);

        v_av1 = _mm_sub_ps(v_cv8, v_cv2);
        v_tv1 = _mm_mul_ps(v_K1, v_av1);
        v_av2 = _mm_sub_ps(v_cv7, v_cv3);
        v_tv2 = _mm_mul_ps(v_K3, v_av2);
        v_av3 = _mm_sub_ps(v_cv6, v_cv4);
        v_tv3 = _mm_mul_ps(v_K5, v_av3);
        v_av4 = _mm_sub_ps(v_cv9, v_tv2);
        v_av5 = _mm_add_ps(v_tv3, v_tv1);
        v_av15 = _mm_sub_ps(v_av4, v_av5);
        v_av6 = _mm_add_ps(v_cv16, v_cv10);
        v_tv4 = _mm_mul_ps(v_K2, v_av6);
        v_av7 = _mm_add_ps(v_cv11, v_cv15);
        v_tv5 = _mm_mul_ps(v_K4, v_av7);
        v_av8 = _mm_add_ps(v_cv12, v_cv14);
        v_tv6 = _mm_mul_ps(v_K6, v_av8);
        v_tv7 = _mm_mul_ps(v_K7, v_cv13);
        v_av9 = _mm_add_ps(v_tv5, v_tv7);
        v_av41 = _mm_add_ps(v_tv4, v_tv6);
        v_av10 = _mm_add_ps(v_av41, v_av9);
        v_av10 = SWAP_RI_128_S(CONJ_128_S(v_av10));

        // Output point 2 : X[1]
        v_out1 = _mm_sub_ps(v_av15, v_av10);
        // Output point 16 : X[15]
        v_out15 = _mm_add_ps(v_av15, v_av10);

        v_av12 = _mm_add_ps(v_av4, v_av5);
        v_av14 = _mm_sub_ps(v_av41, v_av9);
        v_av14 = SWAP_RI_128_S(CONJ_128_S(v_av14));

        // Output point 8 : X[7]
        v_out7 = _mm_sub_ps(v_av12, v_av14);
        // Output point 10 : X[9]
        v_out9 = _mm_add_ps(v_av12, v_av14);

        v_av16 = _mm_sub_ps(v_cv1, v_cv5);
        v_av17 = _mm_add_ps(v_cv2, v_cv8);
        v_av18 = _mm_add_ps(v_cv4, v_cv6);
        v_av19 = _mm_sub_ps(v_av17, v_av18);
        v_av20 = _mm_mul_ps(v_K3, v_av19);
        v_av21 = _mm_add_ps(v_av16, v_av20);
        v_av22 = _mm_sub_ps(v_cv14, v_cv12);
        v_av23 = _mm_sub_ps(v_cv16, v_cv10);
        v_av24 = _mm_mul_ps(v_K4, _mm_add_ps(v_av22, v_av23));
        v_av25 = _mm_sub_ps(v_cv15, v_cv11);
        v_av26 = _mm_mul_ps(v_K7, v_av25);
        v_av27 = _mm_add_ps(v_av24, v_av26);
        v_av27 = SWAP_RI_128_S(CONJ_128_S(v_av27));

        // Output point 3 : X[2]
        v_out2 = _mm_add_ps(v_av21, v_av27);
        // Output point 15 : X[14]
        v_out14 = _mm_sub_ps(v_av21, v_av27);

        v_av21 = _mm_sub_ps(v_av16, v_av20);
        v_av27 = _mm_sub_ps(v_av24, v_av26);
        v_av27 = SWAP_RI_128_S(CONJ_128_S(v_av27));

        // Output point 7 : X[6]
        v_out6 = _mm_add_ps(v_av21, v_av27);
        // Output point 11 : X[10]
        v_out10 = _mm_sub_ps(v_av21, v_av27);

        v_av28 = _mm_sub_ps(_mm_mul_ps(v_K1, v_av3), _mm_mul_ps(v_K5, v_av1));
        v_av29 = _mm_add_ps(v_cv9, v_tv2);
        v_av30 = _mm_add_ps(v_av29, v_av28);
        v_tv9 = _mm_mul_ps(v_K2, v_av8);
        v_tv11 = _mm_mul_ps(v_K6, v_av6);
        v_av31 = _mm_sub_ps(v_tv11, v_tv9);
        v_av32 = _mm_sub_ps(v_tv5, v_tv7);
        v_av33 = _mm_add_ps(v_av31, v_av32);
        v_av33 = SWAP_RI_128_S(CONJ_128_S(v_av33));

        // Output point 4 : X[3]
        v_out3 = _mm_sub_ps(v_av30, v_av33);
        // Output point 14 : X[13]

        v_out13 = _mm_add_ps(v_av30, v_av33);
        v_av42 = _mm_sub_ps(v_av29, v_av28);
        v_av43 = _mm_sub_ps(v_av31, v_av32);
        v_av43 = SWAP_RI_128_S(CONJ_128_S(v_av43));

        // Output point 6 : X[5]
        v_out5 = _mm_sub_ps(v_av42, v_av43);
        // Output point 12 : X[11]
        v_out11 = _mm_add_ps(v_av42, v_av43);

        v_av34 = _mm_add_ps(v_cv1, v_cv5);
        v_av35 = _mm_add_ps(v_cv3, v_cv7);
        v_av36 = _mm_sub_ps(v_av34, v_av35);
        v_tv12 = _mm_mul_ps(v_K7, _mm_sub_ps(v_av22, v_av23));
        v_tv12 = SWAP_RI_128_S(CONJ_128_S(v_tv12));

        // Output point 5 : X[4]
        v_out4 = _mm_sub_ps(v_av36, v_tv12);
        // Output point 13 : X[12]
        v_out12 = _mm_add_ps(v_av36, v_tv12);

        v_av39 = _mm_add_ps(v_av34, v_av35);
        v_av40 = _mm_add_ps(v_av17, v_av18);

        // Output point 1 : X[0]
        v_out0 = _mm_add_ps(v_av39, v_av40);
        // Output point 9 : X[8]
        v_out8 = _mm_sub_ps(v_av39, v_av40);

        ST_LOW_128_S(curr_out, v_out0);
        ST_LOW_128_S(curr_out + out_strides[1], v_out1);
        ST_LOW_128_S(curr_out + out_strides[2], v_out2);
        ST_LOW_128_S(curr_out + out_strides[3], v_out3);
        ST_LOW_128_S(curr_out + out_strides[4], v_out4);
        ST_LOW_128_S(curr_out + out_strides[5], v_out5);
        ST_LOW_128_S(curr_out + out_strides[6], v_out6);
        ST_LOW_128_S(curr_out + out_strides[7], v_out7);
        ST_LOW_128_S(curr_out + out_strides[8], v_out8);
        ST_LOW_128_S(curr_out + out_strides[9], v_out9);
        ST_LOW_128_S(curr_out + out_strides[10], v_out10);
        ST_LOW_128_S(curr_out + out_strides[11], v_out11);
        ST_LOW_128_S(curr_out + out_strides[12], v_out12);
        ST_LOW_128_S(curr_out + out_strides[13], v_out13);
        ST_LOW_128_S(curr_out + out_strides[14], v_out14);
        ST_LOW_128_S(curr_out + out_strides[15], v_out15);
    }
#endif

    AOCLFFTZ_LOG(DEBUG, global_logger_mode, "Exit");
}

static VOID TWID_KNAME_FP64(VOID *in_real, VOID *in_imag, VOID *out_real,
                            VOID *out_imag, INTP n, aoclfftz_strides_t *strides,
                            VOID *twd, UINT8 flag)
{
    AOCLFFTZ_LOG(DEBUG, global_logger_mode, "Enter");

    const DOUBLE CRTM_16[4] = {
        0.92387953251128675612818318939678828682241662586364,
        0.38268343236508977172845998403039886676134456248563,
        0.70710678118654752440084436210484903928483593768847,
        1.00000000000000000000000000000000000000000000000000};

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

    VREGTYPE_D v_C1 = BCAST_D(CRTM_16[0]);
    VREGTYPE_D v_C2 = BCAST_D(CRTM_16[1]);
    VREGTYPE_D v_C3 = BCAST_D(CRTM_16[2]);
    VREGTYPE_D v_C4 = BCAST_D(CRTM_16[2]);
    VREGTYPE_D v_C5 = BCAST_D(CRTM_16[1]);
    VREGTYPE_D v_C6 = BCAST_D(CRTM_16[0]);
    VREGTYPE_D v_C7 = BCAST_D(CRTM_16[3]);

    v_C2 = NEG_D(v_C2, flag);
    v_C4 = NEG_D(v_C4, flag);
    v_C6 = NEG_D(v_C6, flag);
    v_C7 = NEG_D(v_C7, flag);

    for (count = 0; count < N; count++)
    {
        VREGTYPE_D v_in0, v_in1, v_in2, v_in3, v_in4, v_in5;
        VREGTYPE_D v_in6, v_in7, v_in8, v_in9, v_in10;
        VREGTYPE_D v_in11, v_in12, v_in13, v_in14, v_in15;
        VREGTYPE_D v_tv1, v_tv2, v_tv3, v_tv4, v_tv5, v_tv6;
        VREGTYPE_D v_tv7, v_tv9, v_tv11, v_tv12;
        VREGTYPE_D v_av1, v_av2, v_av3, v_av4, v_av5, v_av6;
        VREGTYPE_D v_av7, v_av8, v_av9, v_av10, v_av12, v_av14;
        VREGTYPE_D v_av15, v_av16, v_av17, v_av18, v_av19, v_av21;
        VREGTYPE_D v_av22, v_av23, v_av24, v_av25, v_av20, v_av26;
        VREGTYPE_D v_av27, v_av28, v_av29, v_av30, v_av31;
        VREGTYPE_D v_av32, v_av33, v_av34, v_av35, v_av36;
        VREGTYPE_D v_av39, v_av40, v_av41, v_av42, v_av43;
        VREGTYPE_D v_cv1, v_cv2, v_cv3, v_cv4, v_cv5, v_cv6, v_cv7;
        VREGTYPE_D v_cv8, v_cv9, v_cv10, v_cv11, v_cv12, v_cv13;
        VREGTYPE_D v_cv14, v_cv15, v_cv16;
        VREGTYPE_D v_out0, v_out1, v_out2, v_out3, v_out4, v_out5, v_out6;
        VREGTYPE_D v_out7, v_out8, v_out9, v_out10, v_out11, v_out12;
        VREGTYPE_D v_out13, v_out14, v_out15;

        curr_in = in_r;
        curr_out = out_r;

        if (flag)
        {
            ITW_GATHER_D(curr_in, in_strides, 1, v_in_stride, v_in1, tw, cols,
                         (count * load_multi_cols * NUM_SETS_D),
                         load_multi_cols);
            ITW_GATHER_D(curr_in, in_strides, 2, v_in_stride, v_in2, tw, cols,
                         (count * load_multi_cols * NUM_SETS_D),
                         load_multi_cols);
            ITW_GATHER_D(curr_in, in_strides, 3, v_in_stride, v_in3, tw, cols,
                         (count * load_multi_cols * NUM_SETS_D),
                         load_multi_cols);
            ITW_GATHER_D(curr_in, in_strides, 4, v_in_stride, v_in4, tw, cols,
                         (count * load_multi_cols * NUM_SETS_D),
                         load_multi_cols);
            ITW_GATHER_D(curr_in, in_strides, 5, v_in_stride, v_in5, tw, cols,
                         (count * load_multi_cols * NUM_SETS_D),
                         load_multi_cols);
            ITW_GATHER_D(curr_in, in_strides, 6, v_in_stride, v_in6, tw, cols,
                         (count * load_multi_cols * NUM_SETS_D),
                         load_multi_cols);
            ITW_GATHER_D(curr_in, in_strides, 7, v_in_stride, v_in7, tw, cols,
                         (count * load_multi_cols * NUM_SETS_D),
                         load_multi_cols);
            ITW_GATHER_D(curr_in, in_strides, 8, v_in_stride, v_in8, tw, cols,
                         (count * load_multi_cols * NUM_SETS_D),
                         load_multi_cols);
            ITW_GATHER_D(curr_in, in_strides, 9, v_in_stride, v_in9, tw, cols,
                         (count * load_multi_cols * NUM_SETS_D),
                         load_multi_cols);
            ITW_GATHER_D(curr_in, in_strides, 10, v_in_stride, v_in10, tw, cols,
                         (count * load_multi_cols * NUM_SETS_D),
                         load_multi_cols);
            ITW_GATHER_D(curr_in, in_strides, 11, v_in_stride, v_in11, tw, cols,
                         (count * load_multi_cols * NUM_SETS_D),
                         load_multi_cols);
            ITW_GATHER_D(curr_in, in_strides, 12, v_in_stride, v_in12, tw, cols,
                         (count * load_multi_cols * NUM_SETS_D),
                         load_multi_cols);
            ITW_GATHER_D(curr_in, in_strides, 13, v_in_stride, v_in13, tw, cols,
                         (count * load_multi_cols * NUM_SETS_D),
                         load_multi_cols);
            ITW_GATHER_D(curr_in, in_strides, 14, v_in_stride, v_in14, tw, cols,
                         (count * load_multi_cols * NUM_SETS_D),
                         load_multi_cols);
            ITW_GATHER_D(curr_in, in_strides, 15, v_in_stride, v_in15, tw, cols,
                         (count * load_multi_cols * NUM_SETS_D),
                         load_multi_cols);
        }
        else
        {
            TW_GATHER_D(curr_in, in_strides, 1, v_in_stride, v_in1, tw, cols,
                        (count * load_multi_cols * NUM_SETS_D),
                        load_multi_cols);
            TW_GATHER_D(curr_in, in_strides, 2, v_in_stride, v_in2, tw, cols,
                        (count * load_multi_cols * NUM_SETS_D),
                        load_multi_cols);
            TW_GATHER_D(curr_in, in_strides, 3, v_in_stride, v_in3, tw, cols,
                        (count * load_multi_cols * NUM_SETS_D),
                        load_multi_cols);
            TW_GATHER_D(curr_in, in_strides, 4, v_in_stride, v_in4, tw, cols,
                        (count * load_multi_cols * NUM_SETS_D),
                        load_multi_cols);
            TW_GATHER_D(curr_in, in_strides, 5, v_in_stride, v_in5, tw, cols,
                        (count * load_multi_cols * NUM_SETS_D),
                        load_multi_cols);
            TW_GATHER_D(curr_in, in_strides, 6, v_in_stride, v_in6, tw, cols,
                        (count * load_multi_cols * NUM_SETS_D),
                        load_multi_cols);
            TW_GATHER_D(curr_in, in_strides, 7, v_in_stride, v_in7, tw, cols,
                        (count * load_multi_cols * NUM_SETS_D),
                        load_multi_cols);
            TW_GATHER_D(curr_in, in_strides, 8, v_in_stride, v_in8, tw, cols,
                        (count * load_multi_cols * NUM_SETS_D),
                        load_multi_cols);
            TW_GATHER_D(curr_in, in_strides, 9, v_in_stride, v_in9, tw, cols,
                        (count * load_multi_cols * NUM_SETS_D),
                        load_multi_cols);
            TW_GATHER_D(curr_in, in_strides, 10, v_in_stride, v_in10, tw, cols,
                        (count * load_multi_cols * NUM_SETS_D),
                        load_multi_cols);
            TW_GATHER_D(curr_in, in_strides, 11, v_in_stride, v_in11, tw, cols,
                        (count * load_multi_cols * NUM_SETS_D),
                        load_multi_cols);
            TW_GATHER_D(curr_in, in_strides, 12, v_in_stride, v_in12, tw, cols,
                        (count * load_multi_cols * NUM_SETS_D),
                        load_multi_cols);
            TW_GATHER_D(curr_in, in_strides, 13, v_in_stride, v_in13, tw, cols,
                        (count * load_multi_cols * NUM_SETS_D),
                        load_multi_cols);
            TW_GATHER_D(curr_in, in_strides, 14, v_in_stride, v_in14, tw, cols,
                        (count * load_multi_cols * NUM_SETS_D),
                        load_multi_cols);
            TW_GATHER_D(curr_in, in_strides, 15, v_in_stride, v_in15, tw, cols,
                        (count * load_multi_cols * NUM_SETS_D),
                        load_multi_cols);
        }

        GATHER_D(curr_in, v_in_stride, v_in0);

        // common calculations
        v_cv1 = ADD_D(v_in0, v_in8);
        v_cv2 = ADD_D(v_in1, v_in15);
        v_cv3 = ADD_D(v_in2, v_in14);
        v_cv4 = ADD_D(v_in3, v_in13);
        v_cv5 = ADD_D(v_in4, v_in12);
        v_cv6 = ADD_D(v_in5, v_in11);
        v_cv7 = ADD_D(v_in6, v_in10);
        v_cv8 = ADD_D(v_in7, v_in9);
        v_cv9 = SUB_D(v_in0, v_in8);
        v_cv10 = SUB_D(v_in1, v_in15);
        v_cv11 = SUB_D(v_in2, v_in14);
        v_cv12 = SUB_D(v_in3, v_in13);
        v_cv13 = SUB_D(v_in4, v_in12);
        v_cv14 = SUB_D(v_in5, v_in11);
        v_cv15 = SUB_D(v_in6, v_in10);
        v_cv16 = SUB_D(v_in7, v_in9);

        v_av1 = SUB_D(v_cv8, v_cv2);
        v_tv1 = MUL_D(v_C1, v_av1);
        v_av2 = SUB_D(v_cv7, v_cv3);
        v_tv2 = MUL_D(v_C3, v_av2);
        v_av3 = SUB_D(v_cv6, v_cv4);
        v_tv3 = MUL_D(v_C5, v_av3);
        v_av4 = SUB_D(v_cv9, v_tv2);
        v_av5 = ADD_D(v_tv3, v_tv1);
        v_av15 = SUB_D(v_av4, v_av5);
        v_av6 = ADD_D(v_cv16, v_cv10);
        v_tv4 = MUL_D(v_C2, v_av6);
        v_av7 = ADD_D(v_cv11, v_cv15);
        v_tv5 = MUL_D(v_C4, v_av7);
        v_av8 = ADD_D(v_cv12, v_cv14);
        v_tv6 = MUL_D(v_C6, v_av8);
        v_tv7 = MUL_D(v_C7, v_cv13);
        v_av9 = ADD_D(v_tv5, v_tv7);
        v_av41 = ADD_D(v_tv4, v_tv6);
        v_av10 = ADD_D(v_av41, v_av9);
        v_av10 = SWAP_RI_D(CONJ_D(v_av10));

        // Output point 2 : X[1]
        v_out1 = SUB_D(v_av15, v_av10);
        // Output point 16 : X[15]
        v_out15 = ADD_D(v_av15, v_av10);
        v_av12 = ADD_D(v_av4, v_av5);
        v_av14 = SUB_D(v_av41, v_av9);
        v_av14 = SWAP_RI_D(CONJ_D(v_av14));

        // Output point 8 : X[7]
        v_out7 = SUB_D(v_av12, v_av14);
        // Output point 10 : X[9]
        v_out9 = ADD_D(v_av12, v_av14);

        v_av16 = SUB_D(v_cv1, v_cv5);
        v_av17 = ADD_D(v_cv2, v_cv8);
        v_av18 = ADD_D(v_cv4, v_cv6);
        v_av19 = SUB_D(v_av17, v_av18);
        v_av20 = MUL_D(v_C3, v_av19);
        v_av21 = ADD_D(v_av16, v_av20);
        v_av22 = SUB_D(v_cv14, v_cv12);
        v_av23 = SUB_D(v_cv16, v_cv10);
        v_av24 = MUL_D(v_C4, ADD_D(v_av22, v_av23));
        v_av25 = SUB_D(v_cv15, v_cv11);
        v_av26 = MUL_D(v_C7, v_av25);
        v_av27 = ADD_D(v_av24, v_av26);
        v_av27 = SWAP_RI_D(CONJ_D(v_av27));

        // Output point 3 : X[2]
        v_out2 = ADD_D(v_av21, v_av27);
        // Output point 15 : X[14]
        v_out14 = SUB_D(v_av21, v_av27);

        v_av21 = SUB_D(v_av16, v_av20);
        v_av27 = SUB_D(v_av24, v_av26);
        v_av27 = SWAP_RI_D(CONJ_D(v_av27));

        // Output point 7 : X[6]
        v_out6 = ADD_D(v_av21, v_av27);
        // Output point 11 : X[10]
        v_out10 = SUB_D(v_av21, v_av27);

        v_av28 = SUB_D(MUL_D(v_C1, v_av3), MUL_D(v_C5, v_av1));
        v_av29 = ADD_D(v_cv9, v_tv2);
        v_av30 = ADD_D(v_av29, v_av28);
        v_tv9 = MUL_D(v_C2, v_av8);
        v_tv11 = MUL_D(v_C6, v_av6);
        v_av31 = SUB_D(v_tv11, v_tv9);
        v_av32 = SUB_D(v_tv5, v_tv7);
        v_av33 = ADD_D(v_av31, v_av32);
        v_av33 = SWAP_RI_D(CONJ_D(v_av33));

        // Output point 4 : X[3]
        v_out3 = SUB_D(v_av30, v_av33);
        // Output point 14 : X[13]
        v_out13 = ADD_D(v_av30, v_av33);

        v_av42 = SUB_D(v_av29, v_av28);
        v_av43 = SUB_D(v_av31, v_av32);
        v_av43 = SWAP_RI_D(CONJ_D(v_av43));

        // Output point 6 : X[5]
        v_out5 = SUB_D(v_av42, v_av43);
        // Output point 12 : X[11]
        v_out11 = ADD_D(v_av42, v_av43);

        v_av34 = ADD_D(v_cv1, v_cv5);
        v_av35 = ADD_D(v_cv3, v_cv7);
        v_av36 = SUB_D(v_av34, v_av35);
        v_tv12 = MUL_D(v_C7, SUB_D(v_av22, v_av23));

        v_tv12 = SWAP_RI_D(CONJ_D(v_tv12));

        // Output point 5 : X[4]
        v_out4 = SUB_D(v_av36, v_tv12);
        // Output point 13 : X[12]
        v_out12 = ADD_D(v_av36, v_tv12);

        v_av39 = ADD_D(v_av34, v_av35);
        v_av40 = ADD_D(v_av17, v_av18);

        // Output point 1 : X[0]
        v_out0 = ADD_D(v_av39, v_av40);
        // Output point 9 : X[8]
        v_out8 = SUB_D(v_av39, v_av40);

        SCATTER_D(curr_out, v_out_stride, v_out0);
        SCATTER_D(curr_out + out_strides[1], v_out_stride, v_out1);
        SCATTER_D(curr_out + out_strides[2], v_out_stride, v_out2);
        SCATTER_D(curr_out + out_strides[3], v_out_stride, v_out3);
        SCATTER_D(curr_out + out_strides[4], v_out_stride, v_out4);
        SCATTER_D(curr_out + out_strides[5], v_out_stride, v_out5);
        SCATTER_D(curr_out + out_strides[6], v_out_stride, v_out6);
        SCATTER_D(curr_out + out_strides[7], v_out_stride, v_out7);
        SCATTER_D(curr_out + out_strides[8], v_out_stride, v_out8);
        SCATTER_D(curr_out + out_strides[9], v_out_stride, v_out9);
        SCATTER_D(curr_out + out_strides[10], v_out_stride, v_out10);
        SCATTER_D(curr_out + out_strides[11], v_out_stride, v_out11);
        SCATTER_D(curr_out + out_strides[12], v_out_stride, v_out12);
        SCATTER_D(curr_out + out_strides[13], v_out_stride, v_out13);
        SCATTER_D(curr_out + out_strides[14], v_out_stride, v_out14);
        SCATTER_D(curr_out + out_strides[15], v_out_stride, v_out15);

        in_r += NUM_SETS_D * v_in_stride;
        out_r += NUM_SETS_D * v_out_stride;
    }

    // The following contains code that performs the FFT on the tail cases.
    // These tails are conditionally "instantiated" based on the ISA requested
    // by the "includee" file.

#if defined(KERNEL_USE_AVX512)
    if (remaining_sets >= NUM_SETS_256_D)
    {
        __m256d v_in0, v_in1, v_in2, v_in3, v_in4, v_in5;
        __m256d v_in6, v_in7, v_in8, v_in9, v_in10;
        __m256d v_in11, v_in12, v_in13, v_in14, v_in15;
        __m256d v_tv1, v_tv2, v_tv3, v_tv4, v_tv5, v_tv6;
        __m256d v_tv7, v_tv9, v_tv11, v_tv12;
        __m256d v_av1, v_av2, v_av3, v_av4, v_av5, v_av6;
        __m256d v_av7, v_av8, v_av9, v_av10, v_av12, v_av14;
        __m256d v_av15, v_av16, v_av17, v_av18, v_av19, v_av21;
        __m256d v_av22, v_av23, v_av24, v_av25, v_av20, v_av26;
        __m256d v_av27, v_av28, v_av29, v_av30, v_av31;
        __m256d v_av32, v_av33, v_av34, v_av35, v_av36;
        __m256d v_av39, v_av40, v_av41, v_av42, v_av43;
        __m256d v_cv1, v_cv2, v_cv3, v_cv4, v_cv5, v_cv6, v_cv7;
        __m256d v_cv8, v_cv9, v_cv10, v_cv11, v_cv12, v_cv13;
        __m256d v_cv14, v_cv15, v_cv16;
        __m256d v_out0, v_out1, v_out2, v_out3, v_out4, v_out5, v_out6;
        __m256d v_out7, v_out8, v_out9, v_out10, v_out11, v_out12;
        __m256d v_out13, v_out14, v_out15;

        __m256d v_K1 = CAST_512_TO_256_D(v_C1);
        __m256d v_K2 = CAST_512_TO_256_D(v_C2);
        __m256d v_K3 = CAST_512_TO_256_D(v_C3);
        __m256d v_K4 = CAST_512_TO_256_D(v_C4);
        __m256d v_K5 = CAST_512_TO_256_D(v_C5);
        __m256d v_K6 = CAST_512_TO_256_D(v_C6);
        __m256d v_K7 = CAST_512_TO_256_D(v_C7);

        curr_in = in_r;
        curr_out = out_r;

        if (flag)
        {
            ITW_GATHER_256_D(curr_in, in_strides, 1, v_in_stride, v_in1, tw,
                             cols, cnt_256, load_multi_cols);
            ITW_GATHER_256_D(curr_in, in_strides, 2, v_in_stride, v_in2, tw,
                             cols, cnt_256, load_multi_cols);
            ITW_GATHER_256_D(curr_in, in_strides, 3, v_in_stride, v_in3, tw,
                             cols, cnt_256, load_multi_cols);
            ITW_GATHER_256_D(curr_in, in_strides, 4, v_in_stride, v_in4, tw,
                             cols, cnt_256, load_multi_cols);
            ITW_GATHER_256_D(curr_in, in_strides, 5, v_in_stride, v_in5, tw,
                             cols, cnt_256, load_multi_cols);
            ITW_GATHER_256_D(curr_in, in_strides, 6, v_in_stride, v_in6, tw,
                             cols, cnt_256, load_multi_cols);
            ITW_GATHER_256_D(curr_in, in_strides, 7, v_in_stride, v_in7, tw,
                             cols, cnt_256, load_multi_cols);
            ITW_GATHER_256_D(curr_in, in_strides, 8, v_in_stride, v_in8, tw,
                             cols, cnt_256, load_multi_cols);
            ITW_GATHER_256_D(curr_in, in_strides, 9, v_in_stride, v_in9, tw,
                             cols, cnt_256, load_multi_cols);
            ITW_GATHER_256_D(curr_in, in_strides, 10, v_in_stride, v_in10, tw,
                             cols, cnt_256, load_multi_cols);
            ITW_GATHER_256_D(curr_in, in_strides, 11, v_in_stride, v_in11, tw,
                             cols, cnt_256, load_multi_cols);
            ITW_GATHER_256_D(curr_in, in_strides, 12, v_in_stride, v_in12, tw,
                             cols, cnt_256, load_multi_cols);
            ITW_GATHER_256_D(curr_in, in_strides, 13, v_in_stride, v_in13, tw,
                             cols, cnt_256, load_multi_cols);
            ITW_GATHER_256_D(curr_in, in_strides, 14, v_in_stride, v_in14, tw,
                             cols, cnt_256, load_multi_cols);
            ITW_GATHER_256_D(curr_in, in_strides, 15, v_in_stride, v_in15, tw,
                             cols, cnt_256, load_multi_cols);
        }
        else
        {
            TW_GATHER_256_D(curr_in, in_strides, 1, v_in_stride, v_in1, tw,
                            cols, cnt_256, load_multi_cols);
            TW_GATHER_256_D(curr_in, in_strides, 2, v_in_stride, v_in2, tw,
                            cols, cnt_256, load_multi_cols);
            TW_GATHER_256_D(curr_in, in_strides, 3, v_in_stride, v_in3, tw,
                            cols, cnt_256, load_multi_cols);
            TW_GATHER_256_D(curr_in, in_strides, 4, v_in_stride, v_in4, tw,
                            cols, cnt_256, load_multi_cols);
            TW_GATHER_256_D(curr_in, in_strides, 5, v_in_stride, v_in5, tw,
                            cols, cnt_256, load_multi_cols);
            TW_GATHER_256_D(curr_in, in_strides, 6, v_in_stride, v_in6, tw,
                            cols, cnt_256, load_multi_cols);
            TW_GATHER_256_D(curr_in, in_strides, 7, v_in_stride, v_in7, tw,
                            cols, cnt_256, load_multi_cols);
            TW_GATHER_256_D(curr_in, in_strides, 8, v_in_stride, v_in8, tw,
                            cols, cnt_256, load_multi_cols);
            TW_GATHER_256_D(curr_in, in_strides, 9, v_in_stride, v_in9, tw,
                            cols, cnt_256, load_multi_cols);
            TW_GATHER_256_D(curr_in, in_strides, 10, v_in_stride, v_in10, tw,
                            cols, cnt_256, load_multi_cols);
            TW_GATHER_256_D(curr_in, in_strides, 11, v_in_stride, v_in11, tw,
                            cols, cnt_256, load_multi_cols);
            TW_GATHER_256_D(curr_in, in_strides, 12, v_in_stride, v_in12, tw,
                            cols, cnt_256, load_multi_cols);
            TW_GATHER_256_D(curr_in, in_strides, 13, v_in_stride, v_in13, tw,
                            cols, cnt_256, load_multi_cols);
            TW_GATHER_256_D(curr_in, in_strides, 14, v_in_stride, v_in14, tw,
                            cols, cnt_256, load_multi_cols);
            TW_GATHER_256_D(curr_in, in_strides, 15, v_in_stride, v_in15, tw,
                            cols, cnt_256, load_multi_cols);
        }

        GATHER2_256_D(curr_in, v_in_stride, v_in0);

        // common calculations
        v_cv1 = _mm256_add_pd(v_in0, v_in8);
        v_cv2 = _mm256_add_pd(v_in1, v_in15);
        v_cv3 = _mm256_add_pd(v_in2, v_in14);
        v_cv4 = _mm256_add_pd(v_in3, v_in13);
        v_cv5 = _mm256_add_pd(v_in4, v_in12);
        v_cv6 = _mm256_add_pd(v_in5, v_in11);
        v_cv7 = _mm256_add_pd(v_in6, v_in10);
        v_cv8 = _mm256_add_pd(v_in7, v_in9);
        v_cv9 = _mm256_sub_pd(v_in0, v_in8);
        v_cv10 = _mm256_sub_pd(v_in1, v_in15);
        v_cv11 = _mm256_sub_pd(v_in2, v_in14);
        v_cv12 = _mm256_sub_pd(v_in3, v_in13);
        v_cv13 = _mm256_sub_pd(v_in4, v_in12);
        v_cv14 = _mm256_sub_pd(v_in5, v_in11);
        v_cv15 = _mm256_sub_pd(v_in6, v_in10);
        v_cv16 = _mm256_sub_pd(v_in7, v_in9);

        v_av1 = _mm256_sub_pd(v_cv8, v_cv2);
        v_tv1 = _mm256_mul_pd(v_K1, v_av1);
        v_av2 = _mm256_sub_pd(v_cv7, v_cv3);
        v_tv2 = _mm256_mul_pd(v_K3, v_av2);
        v_av3 = _mm256_sub_pd(v_cv6, v_cv4);
        v_tv3 = _mm256_mul_pd(v_K5, v_av3);
        v_av4 = _mm256_sub_pd(v_cv9, v_tv2);
        v_av5 = _mm256_add_pd(v_tv3, v_tv1);
        v_av15 = _mm256_sub_pd(v_av4, v_av5);
        v_av6 = _mm256_add_pd(v_cv16, v_cv10);
        v_tv4 = _mm256_mul_pd(v_K2, v_av6);
        v_av7 = _mm256_add_pd(v_cv11, v_cv15);
        v_tv5 = _mm256_mul_pd(v_K4, v_av7);
        v_av8 = _mm256_add_pd(v_cv12, v_cv14);
        v_tv6 = _mm256_mul_pd(v_K6, v_av8);
        v_tv7 = _mm256_mul_pd(v_K7, v_cv13);
        v_av9 = _mm256_add_pd(v_tv5, v_tv7);
        v_av41 = _mm256_add_pd(v_tv4, v_tv6);
        v_av10 = _mm256_add_pd(v_av41, v_av9);
        v_av10 = SWAP_RI_256_D(CONJ_256_D(v_av10));

        // Output point 2 : X[1]
        v_out1 = _mm256_sub_pd(v_av15, v_av10);
        // Output point 16 : X[15]
        v_out15 = _mm256_add_pd(v_av15, v_av10);
        v_av12 = _mm256_add_pd(v_av4, v_av5);
        v_av14 = _mm256_sub_pd(v_av41, v_av9);
        v_av14 = SWAP_RI_256_D(CONJ_256_D(v_av14));

        // Output point 8 : X[7]
        v_out7 = _mm256_sub_pd(v_av12, v_av14);
        // Output point 10 : X[9]
        v_out9 = _mm256_add_pd(v_av12, v_av14);

        v_av16 = _mm256_sub_pd(v_cv1, v_cv5);
        v_av17 = _mm256_add_pd(v_cv2, v_cv8);
        v_av18 = _mm256_add_pd(v_cv4, v_cv6);
        v_av19 = _mm256_sub_pd(v_av17, v_av18);
        v_av20 = _mm256_mul_pd(v_K3, v_av19);
        v_av21 = _mm256_add_pd(v_av16, v_av20);
        v_av22 = _mm256_sub_pd(v_cv14, v_cv12);
        v_av23 = _mm256_sub_pd(v_cv16, v_cv10);
        v_av24 = _mm256_mul_pd(v_K4, _mm256_add_pd(v_av22, v_av23));
        v_av25 = _mm256_sub_pd(v_cv15, v_cv11);
        v_av26 = _mm256_mul_pd(v_K7, v_av25);
        v_av27 = _mm256_add_pd(v_av24, v_av26);
        v_av27 = SWAP_RI_256_D(CONJ_256_D(v_av27));

        // Output point 3 : X[2]
        v_out2 = _mm256_add_pd(v_av21, v_av27);
        // Output point 15 : X[14]
        v_out14 = _mm256_sub_pd(v_av21, v_av27);

        v_av21 = _mm256_sub_pd(v_av16, v_av20);
        v_av27 = _mm256_sub_pd(v_av24, v_av26);
        v_av27 = SWAP_RI_256_D(CONJ_256_D(v_av27));

        // Output point 7 : X[6]
        v_out6 = _mm256_add_pd(v_av21, v_av27);
        // Output point 11 : X[10]
        v_out10 = _mm256_sub_pd(v_av21, v_av27);

        v_av28 = _mm256_sub_pd(_mm256_mul_pd(v_K1, v_av3),
                               _mm256_mul_pd(v_K5, v_av1));
        v_av29 = _mm256_add_pd(v_cv9, v_tv2);
        v_av30 = _mm256_add_pd(v_av29, v_av28);
        v_tv9 = _mm256_mul_pd(v_K2, v_av8);
        v_tv11 = _mm256_mul_pd(v_K6, v_av6);
        v_av31 = _mm256_sub_pd(v_tv11, v_tv9);
        v_av32 = _mm256_sub_pd(v_tv5, v_tv7);
        v_av33 = _mm256_add_pd(v_av31, v_av32);
        v_av33 = SWAP_RI_256_D(CONJ_256_D(v_av33));

        // Output point 4 : X[3]
        v_out3 = _mm256_sub_pd(v_av30, v_av33);
        // Output point 14 : X[13]
        v_out13 = _mm256_add_pd(v_av30, v_av33);

        v_av42 = _mm256_sub_pd(v_av29, v_av28);
        v_av43 = _mm256_sub_pd(v_av31, v_av32);
        v_av43 = SWAP_RI_256_D(CONJ_256_D(v_av43));

        // Output point 6 : X[5]
        v_out5 = _mm256_sub_pd(v_av42, v_av43);
        // Output point 12 : X[11]
        v_out11 = _mm256_add_pd(v_av42, v_av43);

        v_av34 = _mm256_add_pd(v_cv1, v_cv5);
        v_av35 = _mm256_add_pd(v_cv3, v_cv7);
        v_av36 = _mm256_sub_pd(v_av34, v_av35);
        v_tv12 = _mm256_mul_pd(v_K7, _mm256_sub_pd(v_av22, v_av23));

        v_tv12 = SWAP_RI_256_D(CONJ_256_D(v_tv12));

        // Output point 5 : X[4]
        v_out4 = _mm256_sub_pd(v_av36, v_tv12);
        // Output point 13 : X[12]
        v_out12 = _mm256_add_pd(v_av36, v_tv12);

        v_av39 = _mm256_add_pd(v_av34, v_av35);
        v_av40 = _mm256_add_pd(v_av17, v_av18);

        // Output point 1 : X[0]
        v_out0 = _mm256_add_pd(v_av39, v_av40);
        // Output point 9 : X[8]
        v_out8 = _mm256_sub_pd(v_av39, v_av40);

        SCATTER2_256_D(curr_out, v_out_stride, v_out0);
        SCATTER2_256_D(curr_out + out_strides[1], v_out_stride, v_out1);
        SCATTER2_256_D(curr_out + out_strides[2], v_out_stride, v_out2);
        SCATTER2_256_D(curr_out + out_strides[3], v_out_stride, v_out3);
        SCATTER2_256_D(curr_out + out_strides[4], v_out_stride, v_out4);
        SCATTER2_256_D(curr_out + out_strides[5], v_out_stride, v_out5);
        SCATTER2_256_D(curr_out + out_strides[6], v_out_stride, v_out6);
        SCATTER2_256_D(curr_out + out_strides[7], v_out_stride, v_out7);
        SCATTER2_256_D(curr_out + out_strides[8], v_out_stride, v_out8);
        SCATTER2_256_D(curr_out + out_strides[9], v_out_stride, v_out9);
        SCATTER2_256_D(curr_out + out_strides[10], v_out_stride, v_out10);
        SCATTER2_256_D(curr_out + out_strides[11], v_out_stride, v_out11);
        SCATTER2_256_D(curr_out + out_strides[12], v_out_stride, v_out12);
        SCATTER2_256_D(curr_out + out_strides[13], v_out_stride, v_out13);
        SCATTER2_256_D(curr_out + out_strides[14], v_out_stride, v_out14);
        SCATTER2_256_D(curr_out + out_strides[15], v_out_stride, v_out15);

        in_r += NUM_SETS_256_D * v_in_stride;
        out_r += NUM_SETS_256_D * v_out_stride;
    }
#endif

#if defined(KERNEL_USE_AVX512) || defined(KERNEL_USE_AVX256)
    if (remaining_sets & 1)
    {
        __m128d v_in0, v_in1, v_in2, v_in3, v_in4, v_in5;
        __m128d v_in6, v_in7, v_in8, v_in9, v_in10;
        __m128d v_in11, v_in12, v_in13, v_in14, v_in15;
        __m128d v_tv1, v_tv2, v_tv3, v_tv4, v_tv5, v_tv6;
        __m128d v_tv7, v_tv9, v_tv11, v_tv12;
        __m128d v_av1, v_av2, v_av3, v_av4, v_av5, v_av6;
        __m128d v_av7, v_av8, v_av9, v_av10, v_av12, v_av14;
        __m128d v_av15, v_av16, v_av17, v_av18, v_av19, v_av21;
        __m128d v_av22, v_av23, v_av24, v_av25, v_av20, v_av26;
        __m128d v_av27, v_av28, v_av29, v_av30, v_av31;
        __m128d v_av32, v_av33, v_av34, v_av35, v_av36;
        __m128d v_av39, v_av40, v_av41, v_av42, v_av43;
        __m128d v_cv1, v_cv2, v_cv3, v_cv4, v_cv5, v_cv6, v_cv7;
        __m128d v_cv8, v_cv9, v_cv10, v_cv11, v_cv12, v_cv13;
        __m128d v_cv14, v_cv15, v_cv16;
        __m128d v_out0, v_out1, v_out2, v_out3, v_out4, v_out5, v_out6;
        __m128d v_out7, v_out8, v_out9, v_out10, v_out11, v_out12;
        __m128d v_out13, v_out14, v_out15;

#if defined(KERNEL_USE_AVX512)
        __m128d v_K1 = CAST_512_TO_128_D(v_C1);
        __m128d v_K2 = CAST_512_TO_128_D(v_C2);
        __m128d v_K3 = CAST_512_TO_128_D(v_C3);
        __m128d v_K4 = CAST_512_TO_128_D(v_C4);
        __m128d v_K5 = CAST_512_TO_128_D(v_C5);
        __m128d v_K6 = CAST_512_TO_128_D(v_C6);
        __m128d v_K7 = CAST_512_TO_128_D(v_C7);
#elif defined(KERNEL_USE_AVX256)
        __m128d v_K1 = CAST_256_TO_128_D(v_C1);
        __m128d v_K2 = CAST_256_TO_128_D(v_C2);
        __m128d v_K3 = CAST_256_TO_128_D(v_C3);
        __m128d v_K4 = CAST_256_TO_128_D(v_C4);
        __m128d v_K5 = CAST_256_TO_128_D(v_C5);
        __m128d v_K6 = CAST_256_TO_128_D(v_C6);
        __m128d v_K7 = CAST_256_TO_128_D(v_C7);
#endif

        curr_in = in_r;
        curr_out = out_r;

        if (flag)
        {
            ITW_GATHER_128_D(curr_in, in_strides, 1, /* unused */ 0, v_in1, tw,
                             cols, cnt_128, load_multi_cols);
            ITW_GATHER_128_D(curr_in, in_strides, 2, /* unused */ 0, v_in2, tw,
                             cols, cnt_128, load_multi_cols);
            ITW_GATHER_128_D(curr_in, in_strides, 3, /* unused */ 0, v_in3, tw,
                             cols, cnt_128, load_multi_cols);
            ITW_GATHER_128_D(curr_in, in_strides, 4, /* unused */ 0, v_in4, tw,
                             cols, cnt_128, load_multi_cols);
            ITW_GATHER_128_D(curr_in, in_strides, 5, /* unused */ 0, v_in5, tw,
                             cols, cnt_128, load_multi_cols);
            ITW_GATHER_128_D(curr_in, in_strides, 6, /* unused */ 0, v_in6, tw,
                             cols, cnt_128, load_multi_cols);
            ITW_GATHER_128_D(curr_in, in_strides, 7, /* unused */ 0, v_in7, tw,
                             cols, cnt_128, load_multi_cols);
            ITW_GATHER_128_D(curr_in, in_strides, 8, /* unused */ 0, v_in8, tw,
                             cols, cnt_128, load_multi_cols);
            ITW_GATHER_128_D(curr_in, in_strides, 9, /* unused */ 0, v_in9, tw,
                             cols, cnt_128, load_multi_cols);
            ITW_GATHER_128_D(curr_in, in_strides, 10, /* unused */ 0, v_in10,
                             tw, cols, cnt_128, load_multi_cols);
            ITW_GATHER_128_D(curr_in, in_strides, 11, /* unused */ 0, v_in11,
                             tw, cols, cnt_128, load_multi_cols);
            ITW_GATHER_128_D(curr_in, in_strides, 12, /* unused */ 0, v_in12,
                             tw, cols, cnt_128, load_multi_cols);
            ITW_GATHER_128_D(curr_in, in_strides, 13, /* unused */ 0, v_in13,
                             tw, cols, cnt_128, load_multi_cols);
            ITW_GATHER_128_D(curr_in, in_strides, 14, /* unused */ 0, v_in14,
                             tw, cols, cnt_128, load_multi_cols);
            ITW_GATHER_128_D(curr_in, in_strides, 15, /* unused */ 0, v_in15,
                             tw, cols, cnt_128, load_multi_cols);
        }
        else
        {
            TW_GATHER_128_D(curr_in, in_strides, 1, /* unused */ 0, v_in1, tw,
                            cols, cnt_128, load_multi_cols);
            TW_GATHER_128_D(curr_in, in_strides, 2, /* unused */ 0, v_in2, tw,
                            cols, cnt_128, load_multi_cols);
            TW_GATHER_128_D(curr_in, in_strides, 3, /* unused */ 0, v_in3, tw,
                            cols, cnt_128, load_multi_cols);
            TW_GATHER_128_D(curr_in, in_strides, 4, /* unused */ 0, v_in4, tw,
                            cols, cnt_128, load_multi_cols);
            TW_GATHER_128_D(curr_in, in_strides, 5, /* unused */ 0, v_in5, tw,
                            cols, cnt_128, load_multi_cols);
            TW_GATHER_128_D(curr_in, in_strides, 6, /* unused */ 0, v_in6, tw,
                            cols, cnt_128, load_multi_cols);
            TW_GATHER_128_D(curr_in, in_strides, 7, /* unused */ 0, v_in7, tw,
                            cols, cnt_128, load_multi_cols);
            TW_GATHER_128_D(curr_in, in_strides, 8, /* unused */ 0, v_in8, tw,
                            cols, cnt_128, load_multi_cols);
            TW_GATHER_128_D(curr_in, in_strides, 9, /* unused */ 0, v_in9, tw,
                            cols, cnt_128, load_multi_cols);
            TW_GATHER_128_D(curr_in, in_strides, 10, /* unused */ 0, v_in10, tw,
                            cols, cnt_128, load_multi_cols);
            TW_GATHER_128_D(curr_in, in_strides, 11, /* unused */ 0, v_in11, tw,
                            cols, cnt_128, load_multi_cols);
            TW_GATHER_128_D(curr_in, in_strides, 12, /* unused */ 0, v_in12, tw,
                            cols, cnt_128, load_multi_cols);
            TW_GATHER_128_D(curr_in, in_strides, 13, /* unused */ 0, v_in13, tw,
                            cols, cnt_128, load_multi_cols);
            TW_GATHER_128_D(curr_in, in_strides, 14, /* unused */ 0, v_in14, tw,
                            cols, cnt_128, load_multi_cols);
            TW_GATHER_128_D(curr_in, in_strides, 15, /* unused */ 0, v_in15, tw,
                            cols, cnt_128, load_multi_cols);
        }

        LD_128_D(curr_in, v_in0);

        // common calculations
        v_cv1 = _mm_add_pd(v_in0, v_in8);
        v_cv2 = _mm_add_pd(v_in1, v_in15);
        v_cv3 = _mm_add_pd(v_in2, v_in14);
        v_cv4 = _mm_add_pd(v_in3, v_in13);
        v_cv5 = _mm_add_pd(v_in4, v_in12);
        v_cv6 = _mm_add_pd(v_in5, v_in11);
        v_cv7 = _mm_add_pd(v_in6, v_in10);
        v_cv8 = _mm_add_pd(v_in7, v_in9);
        v_cv9 = _mm_sub_pd(v_in0, v_in8);
        v_cv10 = _mm_sub_pd(v_in1, v_in15);
        v_cv11 = _mm_sub_pd(v_in2, v_in14);
        v_cv12 = _mm_sub_pd(v_in3, v_in13);
        v_cv13 = _mm_sub_pd(v_in4, v_in12);
        v_cv14 = _mm_sub_pd(v_in5, v_in11);
        v_cv15 = _mm_sub_pd(v_in6, v_in10);
        v_cv16 = _mm_sub_pd(v_in7, v_in9);

        v_av1 = _mm_sub_pd(v_cv8, v_cv2);
        v_tv1 = _mm_mul_pd(v_K1, v_av1);
        v_av2 = _mm_sub_pd(v_cv7, v_cv3);
        v_tv2 = _mm_mul_pd(v_K3, v_av2);
        v_av3 = _mm_sub_pd(v_cv6, v_cv4);
        v_tv3 = _mm_mul_pd(v_K5, v_av3);
        v_av4 = _mm_sub_pd(v_cv9, v_tv2);
        v_av5 = _mm_add_pd(v_tv3, v_tv1);
        v_av15 = _mm_sub_pd(v_av4, v_av5);
        v_av6 = _mm_add_pd(v_cv16, v_cv10);
        v_tv4 = _mm_mul_pd(v_K2, v_av6);
        v_av7 = _mm_add_pd(v_cv11, v_cv15);
        v_tv5 = _mm_mul_pd(v_K4, v_av7);
        v_av8 = _mm_add_pd(v_cv12, v_cv14);
        v_tv6 = _mm_mul_pd(v_K6, v_av8);
        v_tv7 = _mm_mul_pd(v_K7, v_cv13);
        v_av9 = _mm_add_pd(v_tv5, v_tv7);
        v_av41 = _mm_add_pd(v_tv4, v_tv6);
        v_av10 = _mm_add_pd(v_av41, v_av9);

        v_av10 = SWAP_RI_128_D(CONJ_128_D(v_av10));

        // Output point 2 : X[1]
        v_out1 = _mm_sub_pd(v_av15, v_av10);
        // Output point 16 : X[15]
        v_out15 = _mm_add_pd(v_av15, v_av10);

        v_av12 = _mm_add_pd(v_av4, v_av5);
        v_av14 = _mm_sub_pd(v_av41, v_av9);

        v_av14 = SWAP_RI_128_D(CONJ_128_D(v_av14));

        // Output point 8 : X[7]
        v_out7 = _mm_sub_pd(v_av12, v_av14);
        // Output point 10 : X[9]
        v_out9 = _mm_add_pd(v_av12, v_av14);

        v_av16 = _mm_sub_pd(v_cv1, v_cv5);
        v_av17 = _mm_add_pd(v_cv2, v_cv8);
        v_av18 = _mm_add_pd(v_cv4, v_cv6);
        v_av19 = _mm_sub_pd(v_av17, v_av18);
        v_av20 = _mm_mul_pd(v_K3, v_av19);
        v_av21 = _mm_add_pd(v_av16, v_av20);
        v_av22 = _mm_sub_pd(v_cv14, v_cv12);
        v_av23 = _mm_sub_pd(v_cv16, v_cv10);
        v_av24 = _mm_mul_pd(v_K4, _mm_add_pd(v_av22, v_av23));
        v_av25 = _mm_sub_pd(v_cv15, v_cv11);
        v_av26 = _mm_mul_pd(v_K7, v_av25);
        v_av27 = _mm_add_pd(v_av24, v_av26);
        v_av27 = SWAP_RI_128_D(CONJ_128_D(v_av27));

        // Output point 3 : X[2]
        v_out2 = _mm_add_pd(v_av21, v_av27);
        // Output point 15 : X[14]
        v_out14 = _mm_sub_pd(v_av21, v_av27);

        v_av21 = _mm_sub_pd(v_av16, v_av20);
        v_av27 = _mm_sub_pd(v_av24, v_av26);
        v_av27 = SWAP_RI_128_D(CONJ_128_D(v_av27));

        // Output point 7 : X[6]
        v_out6 = _mm_add_pd(v_av21, v_av27);
        // Output point 11 : X[10]
        v_out10 = _mm_sub_pd(v_av21, v_av27);

        v_av28 = _mm_sub_pd(_mm_mul_pd(v_K1, v_av3), _mm_mul_pd(v_K5, v_av1));
        v_av29 = _mm_add_pd(v_cv9, v_tv2);
        v_av30 = _mm_add_pd(v_av29, v_av28);
        v_tv9 = _mm_mul_pd(v_K2, v_av8);
        v_tv11 = _mm_mul_pd(v_K6, v_av6);
        v_av31 = _mm_sub_pd(v_tv11, v_tv9);
        v_av32 = _mm_sub_pd(v_tv5, v_tv7);
        v_av33 = _mm_add_pd(v_av31, v_av32);
        v_av33 = SWAP_RI_128_D(CONJ_128_D(v_av33));

        // Output point 4 : X[3]
        v_out3 = _mm_sub_pd(v_av30, v_av33);
        // Output point 14 : X[13]
        v_out13 = _mm_add_pd(v_av30, v_av33);

        v_av42 = _mm_sub_pd(v_av29, v_av28);
        v_av43 = _mm_sub_pd(v_av31, v_av32);
        v_av43 = SWAP_RI_128_D(CONJ_128_D(v_av43));

        // Output point 6 : X[5]
        v_out5 = _mm_sub_pd(v_av42, v_av43);
        // Output point 12 : X[11]
        v_out11 = _mm_add_pd(v_av42, v_av43);

        v_av34 = _mm_add_pd(v_cv1, v_cv5);
        v_av35 = _mm_add_pd(v_cv3, v_cv7);
        v_av36 = _mm_sub_pd(v_av34, v_av35);
        v_tv12 = _mm_mul_pd(v_K7, _mm_sub_pd(v_av22, v_av23));
        v_tv12 = SWAP_RI_128_D(CONJ_128_D(v_tv12));

        // Output point 5 : X[4]
        v_out4 = _mm_sub_pd(v_av36, v_tv12);
        // Output point 13 : X[12]
        v_out12 = _mm_add_pd(v_av36, v_tv12);

        v_av39 = _mm_add_pd(v_av34, v_av35);
        v_av40 = _mm_add_pd(v_av17, v_av18);

        // Output point 1 : X[0]
        v_out0 = _mm_add_pd(v_av39, v_av40);
        // Output point 9 : X[8]
        v_out8 = _mm_sub_pd(v_av39, v_av40);

        ST_128_D(curr_out, v_out0);
        ST_128_D(curr_out + out_strides[1], v_out1);
        ST_128_D(curr_out + out_strides[2], v_out2);
        ST_128_D(curr_out + out_strides[3], v_out3);
        ST_128_D(curr_out + out_strides[4], v_out4);
        ST_128_D(curr_out + out_strides[5], v_out5);
        ST_128_D(curr_out + out_strides[6], v_out6);
        ST_128_D(curr_out + out_strides[7], v_out7);
        ST_128_D(curr_out + out_strides[8], v_out8);
        ST_128_D(curr_out + out_strides[9], v_out9);
        ST_128_D(curr_out + out_strides[10], v_out10);
        ST_128_D(curr_out + out_strides[11], v_out11);
        ST_128_D(curr_out + out_strides[12], v_out12);
        ST_128_D(curr_out + out_strides[13], v_out13);
        ST_128_D(curr_out + out_strides[14], v_out14);
        ST_128_D(curr_out + out_strides[15], v_out15);
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
