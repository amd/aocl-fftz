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

#define RADIX 16

static FFTZ_VOID TWID_KNAME_FP32(FFTZ_VOID *in_real, FFTZ_VOID *in_imag,
                                 FFTZ_VOID *out_real, FFTZ_VOID *out_imag,
                                 FFTZ_INTP n, aoclfftz_strides_t *strides,
                                 FFTZ_VOID *twd, FFTZ_UINT8 flag)
{
    AOCLFFTZ_LOG(DEBUG, global_logger_mode, "Enter");

    const FFTZ_FLOAT CRTM_16[4] = {
        0.92387953251128675612818318939678828682241662586364f,
        0.38268343236508977172845998403039886676134456248563f,
        0.70710678118654752440084436210484903928483593768847f,
        1.00000000000000000000000000000000000000000000000000f};

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

    VREGTYPE_S v_C1 = BCAST_S(CRTM_16[0]);
    VREGTYPE_S v_C2 = BCAST_S(CRTM_16[1]);
    VREGTYPE_S v_C3 = BCAST_S(CRTM_16[2]);
    VREGTYPE_S v_C4 = BCAST_S(CRTM_16[2]);
    VREGTYPE_S v_C5 = BCAST_S(CRTM_16[1]);
    VREGTYPE_S v_C6 = BCAST_S(CRTM_16[0]);
    VREGTYPE_S v_C7 = BCAST_S(CRTM_16[3]);

    FFTZ_INTP count;

#if defined(KERNEL_DIRECTION_BWD)
    v_C2 = NEG_S(v_C2, 1);
    v_C4 = NEG_S(v_C4, 1);
    v_C6 = NEG_S(v_C6, 1);
    v_C7 = NEG_S(v_C7, 1);
#endif
    FFTZ_FLOAT *tw_ptr = tw;

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

        GATHER_S(in_r, v_in_stride, v_in0, is_contiguous_in);

        LOAD_IN_S(in_r, in_strides, 1, v_in_stride, v_in1, tw_ptr,
                  load_multi_cols, is_contiguous_in);
        LOAD_IN_S(in_r, in_strides, 2, v_in_stride, v_in2, tw_ptr,
                  load_multi_cols, is_contiguous_in);
        LOAD_IN_S(in_r, in_strides, 3, v_in_stride, v_in3, tw_ptr,
                  load_multi_cols, is_contiguous_in);
        LOAD_IN_S(in_r, in_strides, 4, v_in_stride, v_in4, tw_ptr,
                  load_multi_cols, is_contiguous_in);
        LOAD_IN_S(in_r, in_strides, 5, v_in_stride, v_in5, tw_ptr,
                  load_multi_cols, is_contiguous_in);
        LOAD_IN_S(in_r, in_strides, 6, v_in_stride, v_in6, tw_ptr,
                  load_multi_cols, is_contiguous_in);
        LOAD_IN_S(in_r, in_strides, 7, v_in_stride, v_in7, tw_ptr,
                  load_multi_cols, is_contiguous_in);
#if defined(KERNEL_VARIANT_C2R)
        LOAD_IN_H2_S(in_h2_r, in_strides, 8, v_in_h2_stride, v_in8, tw_ptr,
                  load_multi_cols, 0);
        v_in8 = IN_H2_S(v_in8);
        LOAD_IN_H2_S(in_h2_r, in_strides, 9, v_in_h2_stride, v_in9, tw_ptr,
                  load_multi_cols, 0);
        v_in9 = IN_H2_S(v_in9);
        LOAD_IN_H2_S(in_h2_r, in_strides, 10, v_in_h2_stride, v_in10, tw_ptr,
                  load_multi_cols, 0);
        v_in10 = IN_H2_S(v_in10);
        LOAD_IN_H2_S(in_h2_r, in_strides, 11, v_in_h2_stride, v_in11, tw_ptr,
                  load_multi_cols, 0);
        v_in11 = IN_H2_S(v_in11);
        LOAD_IN_H2_S(in_h2_r, in_strides, 12, v_in_h2_stride, v_in12, tw_ptr,
                  load_multi_cols, 0);
        v_in12 = IN_H2_S(v_in12);
        LOAD_IN_H2_S(in_h2_r, in_strides, 13, v_in_h2_stride, v_in13, tw_ptr,
                  load_multi_cols, 0);
        v_in13 = IN_H2_S(v_in13);
        LOAD_IN_H2_S(in_h2_r, in_strides, 14, v_in_h2_stride, v_in14, tw_ptr,
                  load_multi_cols, 0);
        v_in14 = IN_H2_S(v_in14);
        LOAD_IN_H2_S(in_h2_r, in_strides, 15, v_in_h2_stride, v_in15, tw_ptr,
                  load_multi_cols, 0);
        v_in15 = IN_H2_S(v_in15);
#else
        LOAD_IN_S(in_r, in_strides, 8, v_in_stride, v_in8, tw_ptr,
                  load_multi_cols, is_contiguous_in);
        LOAD_IN_S(in_r, in_strides, 9, v_in_stride, v_in9, tw_ptr,
                  load_multi_cols, is_contiguous_in);
        LOAD_IN_S(in_r, in_strides, 10, v_in_stride, v_in10, tw_ptr,
                  load_multi_cols, is_contiguous_in);
        LOAD_IN_S(in_r, in_strides, 11, v_in_stride, v_in11, tw_ptr,
                  load_multi_cols, is_contiguous_in);
        LOAD_IN_S(in_r, in_strides, 12, v_in_stride, v_in12, tw_ptr,
                  load_multi_cols, is_contiguous_in);
        LOAD_IN_S(in_r, in_strides, 13, v_in_stride, v_in13, tw_ptr,
                  load_multi_cols, is_contiguous_in);
        LOAD_IN_S(in_r, in_strides, 14, v_in_stride, v_in14, tw_ptr,
                  load_multi_cols, is_contiguous_in);
        LOAD_IN_S(in_r, in_strides, 15, v_in_stride, v_in15, tw_ptr,
                  load_multi_cols, is_contiguous_in);
#endif

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

        SCATTER_S(out_r, v_out_stride, v_out0, is_contiguous_out);
        STORE_OUT_S(out_r, out_strides, 1, v_out_stride, v_out1, tw_ptr,
                    load_multi_cols, is_contiguous_out);
        STORE_OUT_S(out_r, out_strides, 2, v_out_stride, v_out2, tw_ptr,
                    load_multi_cols, is_contiguous_out);
        STORE_OUT_S(out_r, out_strides, 3, v_out_stride, v_out3, tw_ptr,
                    load_multi_cols, is_contiguous_out);
        STORE_OUT_S(out_r, out_strides, 4, v_out_stride, v_out4, tw_ptr,
                    load_multi_cols, is_contiguous_out);
        STORE_OUT_S(out_r, out_strides, 5, v_out_stride, v_out5, tw_ptr,
                    load_multi_cols, is_contiguous_out);
        STORE_OUT_S(out_r, out_strides, 6, v_out_stride, v_out6, tw_ptr,
                    load_multi_cols, is_contiguous_out);
        STORE_OUT_S(out_r, out_strides, 7, v_out_stride, v_out7, tw_ptr,
                    load_multi_cols, is_contiguous_out);
#if defined(KERNEL_VARIANT_R2C)
        v_out8 = OUT_H2_S(v_out8);
        STORE_OUT_H2_S(out_h2_r, out_strides, 8, v_out_h2_stride, v_out8,
                       tw_ptr, load_multi_cols, 0);
        v_out9 = OUT_H2_S(v_out9);
        STORE_OUT_H2_S(out_h2_r, out_strides, 9, v_out_h2_stride, v_out9,
                       tw_ptr, load_multi_cols, 0);
        v_out10 = OUT_H2_S(v_out10);
        STORE_OUT_H2_S(out_h2_r, out_strides, 10, v_out_h2_stride, v_out10,
                       tw_ptr, load_multi_cols, 0);
        v_out11 = OUT_H2_S(v_out11);
        STORE_OUT_H2_S(out_h2_r, out_strides, 11, v_out_h2_stride, v_out11,
                       tw_ptr, load_multi_cols, 0);
        v_out12 = OUT_H2_S(v_out12);
        STORE_OUT_H2_S(out_h2_r, out_strides, 12, v_out_h2_stride, v_out12,
                       tw_ptr, load_multi_cols, 0);
        v_out13 = OUT_H2_S(v_out13);
        STORE_OUT_H2_S(out_h2_r, out_strides, 13, v_out_h2_stride, v_out13,
                       tw_ptr, load_multi_cols, 0);
        v_out14 = OUT_H2_S(v_out14);
        STORE_OUT_H2_S(out_h2_r, out_strides, 14, v_out_h2_stride, v_out14,
                       tw_ptr, load_multi_cols, 0);
        v_out15 = OUT_H2_S(v_out15);
        STORE_OUT_H2_S(out_h2_r, out_strides, 15, v_out_h2_stride, v_out15,
                       tw_ptr, load_multi_cols, 0);
#else
        STORE_OUT_S(out_r, out_strides, 8, v_out_stride, v_out8, tw_ptr,
                    load_multi_cols, is_contiguous_out);
        STORE_OUT_S(out_r, out_strides, 9, v_out_stride, v_out9, tw_ptr,
                    load_multi_cols, is_contiguous_out);
        STORE_OUT_S(out_r, out_strides, 10, v_out_stride, v_out10, tw_ptr,
                    load_multi_cols, is_contiguous_out);
        STORE_OUT_S(out_r, out_strides, 11, v_out_stride, v_out11, tw_ptr,
                    load_multi_cols, is_contiguous_out);
        STORE_OUT_S(out_r, out_strides, 12, v_out_stride, v_out12, tw_ptr,
                    load_multi_cols, is_contiguous_out);
        STORE_OUT_S(out_r, out_strides, 13, v_out_stride, v_out13, tw_ptr,
                    load_multi_cols, is_contiguous_out);
        STORE_OUT_S(out_r, out_strides, 14, v_out_stride, v_out14, tw_ptr,
                    load_multi_cols, is_contiguous_out);
        STORE_OUT_S(out_r, out_strides, 15, v_out_stride, v_out15, tw_ptr,
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

        GATHER4_256_S(in_r, v_in_stride, v_in0, is_contiguous_in);

        // Fused load+twiddle at column 0 off the walked tile pointer.
        LOAD_IN_256_S(in_r, in_strides, 1, v_in_stride, v_in1, tw_ptr,
                      load_multi_cols, is_contiguous_in);
        LOAD_IN_256_S(in_r, in_strides, 2, v_in_stride, v_in2, tw_ptr,
                      load_multi_cols, is_contiguous_in);
        LOAD_IN_256_S(in_r, in_strides, 3, v_in_stride, v_in3, tw_ptr,
                      load_multi_cols, is_contiguous_in);
        LOAD_IN_256_S(in_r, in_strides, 4, v_in_stride, v_in4, tw_ptr,
                      load_multi_cols, is_contiguous_in);
        LOAD_IN_256_S(in_r, in_strides, 5, v_in_stride, v_in5, tw_ptr,
                      load_multi_cols, is_contiguous_in);
        LOAD_IN_256_S(in_r, in_strides, 6, v_in_stride, v_in6, tw_ptr,
                      load_multi_cols, is_contiguous_in);
        LOAD_IN_256_S(in_r, in_strides, 7, v_in_stride, v_in7, tw_ptr,
                      load_multi_cols, is_contiguous_in);
#if defined(KERNEL_VARIANT_C2R)
        LOAD_IN_H2_256_S(in_h2_r, in_strides, 8, v_in_h2_stride, v_in8, tw_ptr,
                      load_multi_cols, 0);
        v_in8 = IN_H2_256_S(v_in8);
        LOAD_IN_H2_256_S(in_h2_r, in_strides, 9, v_in_h2_stride, v_in9, tw_ptr,
                      load_multi_cols, 0);
        v_in9 = IN_H2_256_S(v_in9);
        LOAD_IN_H2_256_S(in_h2_r, in_strides, 10, v_in_h2_stride, v_in10,
                         tw_ptr, load_multi_cols, 0);
        v_in10 = IN_H2_256_S(v_in10);
        LOAD_IN_H2_256_S(in_h2_r, in_strides, 11, v_in_h2_stride, v_in11,
                         tw_ptr, load_multi_cols, 0);
        v_in11 = IN_H2_256_S(v_in11);
        LOAD_IN_H2_256_S(in_h2_r, in_strides, 12, v_in_h2_stride, v_in12,
                         tw_ptr, load_multi_cols, 0);
        v_in12 = IN_H2_256_S(v_in12);
        LOAD_IN_H2_256_S(in_h2_r, in_strides, 13, v_in_h2_stride, v_in13,
                         tw_ptr, load_multi_cols, 0);
        v_in13 = IN_H2_256_S(v_in13);
        LOAD_IN_H2_256_S(in_h2_r, in_strides, 14, v_in_h2_stride, v_in14,
                         tw_ptr, load_multi_cols, 0);
        v_in14 = IN_H2_256_S(v_in14);
        LOAD_IN_H2_256_S(in_h2_r, in_strides, 15, v_in_h2_stride, v_in15,
                         tw_ptr, load_multi_cols, 0);
        v_in15 = IN_H2_256_S(v_in15);
#else
        LOAD_IN_256_S(in_r, in_strides, 8, v_in_stride, v_in8, tw_ptr,
                      load_multi_cols, is_contiguous_in);
        LOAD_IN_256_S(in_r, in_strides, 9, v_in_stride, v_in9, tw_ptr,
                      load_multi_cols, is_contiguous_in);
        LOAD_IN_256_S(in_r, in_strides, 10, v_in_stride, v_in10, tw_ptr,
                      load_multi_cols, is_contiguous_in);
        LOAD_IN_256_S(in_r, in_strides, 11, v_in_stride, v_in11, tw_ptr,
                      load_multi_cols, is_contiguous_in);
        LOAD_IN_256_S(in_r, in_strides, 12, v_in_stride, v_in12, tw_ptr,
                      load_multi_cols, is_contiguous_in);
        LOAD_IN_256_S(in_r, in_strides, 13, v_in_stride, v_in13, tw_ptr,
                      load_multi_cols, is_contiguous_in);
        LOAD_IN_256_S(in_r, in_strides, 14, v_in_stride, v_in14, tw_ptr,
                      load_multi_cols, is_contiguous_in);
        LOAD_IN_256_S(in_r, in_strides, 15, v_in_stride, v_in15, tw_ptr,
                      load_multi_cols, is_contiguous_in);
#endif

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

        SCATTER4_256_S(out_r, v_out_stride, v_out0, is_contiguous_out);
        STORE_OUT_256_S(out_r, out_strides, 1, v_out_stride, v_out1, tw_ptr,
                        load_multi_cols, is_contiguous_out);
        STORE_OUT_256_S(out_r, out_strides, 2, v_out_stride, v_out2, tw_ptr,
                        load_multi_cols, is_contiguous_out);
        STORE_OUT_256_S(out_r, out_strides, 3, v_out_stride, v_out3, tw_ptr,
                        load_multi_cols, is_contiguous_out);
        STORE_OUT_256_S(out_r, out_strides, 4, v_out_stride, v_out4, tw_ptr,
                        load_multi_cols, is_contiguous_out);
        STORE_OUT_256_S(out_r, out_strides, 5, v_out_stride, v_out5, tw_ptr,
                        load_multi_cols, is_contiguous_out);
        STORE_OUT_256_S(out_r, out_strides, 6, v_out_stride, v_out6, tw_ptr,
                        load_multi_cols, is_contiguous_out);
        STORE_OUT_256_S(out_r, out_strides, 7, v_out_stride, v_out7, tw_ptr,
                        load_multi_cols, is_contiguous_out);
#if defined(KERNEL_VARIANT_R2C)
        v_out8 = OUT_H2_256_S(v_out8);
        STORE_OUT_H2_256_S(out_h2_r, out_strides, 8, v_out_h2_stride, v_out8,
                        tw_ptr, load_multi_cols, 0);
        v_out9 = OUT_H2_256_S(v_out9);
        STORE_OUT_H2_256_S(out_h2_r, out_strides, 9, v_out_h2_stride, v_out9,
                        tw_ptr, load_multi_cols, 0);
        v_out10 = OUT_H2_256_S(v_out10);
        STORE_OUT_H2_256_S(out_h2_r, out_strides, 10, v_out_h2_stride, v_out10,
                        tw_ptr, load_multi_cols, 0);
        v_out11 = OUT_H2_256_S(v_out11);
        STORE_OUT_H2_256_S(out_h2_r, out_strides, 11, v_out_h2_stride, v_out11,
                        tw_ptr, load_multi_cols, 0);
        v_out12 = OUT_H2_256_S(v_out12);
        STORE_OUT_H2_256_S(out_h2_r, out_strides, 12, v_out_h2_stride, v_out12,
                        tw_ptr, load_multi_cols, 0);
        v_out13 = OUT_H2_256_S(v_out13);
        STORE_OUT_H2_256_S(out_h2_r, out_strides, 13, v_out_h2_stride, v_out13,
                        tw_ptr, load_multi_cols, 0);
        v_out14 = OUT_H2_256_S(v_out14);
        STORE_OUT_H2_256_S(out_h2_r, out_strides, 14, v_out_h2_stride, v_out14,
                        tw_ptr, load_multi_cols, 0);
        v_out15 = OUT_H2_256_S(v_out15);
        STORE_OUT_H2_256_S(out_h2_r, out_strides, 15, v_out_h2_stride, v_out15,
                        tw_ptr, load_multi_cols, 0);
#else
        STORE_OUT_256_S(out_r, out_strides, 8, v_out_stride, v_out8, tw_ptr,
                        load_multi_cols, is_contiguous_out);
        STORE_OUT_256_S(out_r, out_strides, 9, v_out_stride, v_out9, tw_ptr,
                        load_multi_cols, is_contiguous_out);
        STORE_OUT_256_S(out_r, out_strides, 10, v_out_stride, v_out10, tw_ptr,
                        load_multi_cols, is_contiguous_out);
        STORE_OUT_256_S(out_r, out_strides, 11, v_out_stride, v_out11, tw_ptr,
                        load_multi_cols, is_contiguous_out);
        STORE_OUT_256_S(out_r, out_strides, 12, v_out_stride, v_out12, tw_ptr,
                        load_multi_cols, is_contiguous_out);
        STORE_OUT_256_S(out_r, out_strides, 13, v_out_stride, v_out13, tw_ptr,
                        load_multi_cols, is_contiguous_out);
        STORE_OUT_256_S(out_r, out_strides, 14, v_out_stride, v_out14, tw_ptr,
                        load_multi_cols, is_contiguous_out);
        STORE_OUT_256_S(out_r, out_strides, 15, v_out_stride, v_out15, tw_ptr,
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

        LOAD_IN_128_S(in_r, in_strides, 1, v_in_stride, v_in1, tw_ptr,
                      load_multi_cols, is_contiguous_in);
        LOAD_IN_128_S(in_r, in_strides, 2, v_in_stride, v_in2, tw_ptr,
                      load_multi_cols, is_contiguous_in);
        LOAD_IN_128_S(in_r, in_strides, 3, v_in_stride, v_in3, tw_ptr,
                      load_multi_cols, is_contiguous_in);
        LOAD_IN_128_S(in_r, in_strides, 4, v_in_stride, v_in4, tw_ptr,
                      load_multi_cols, is_contiguous_in);
        LOAD_IN_128_S(in_r, in_strides, 5, v_in_stride, v_in5, tw_ptr,
                      load_multi_cols, is_contiguous_in);
        LOAD_IN_128_S(in_r, in_strides, 6, v_in_stride, v_in6, tw_ptr,
                      load_multi_cols, is_contiguous_in);
        LOAD_IN_128_S(in_r, in_strides, 7, v_in_stride, v_in7, tw_ptr,
                      load_multi_cols, is_contiguous_in);
#if defined(KERNEL_VARIANT_C2R)
        LOAD_IN_H2_128_S(in_h2_r, in_strides, 8, v_in_h2_stride, v_in8, tw_ptr,
                      load_multi_cols, 0);
        v_in8 = IN_H2_128_S(v_in8);
        LOAD_IN_H2_128_S(in_h2_r, in_strides, 9, v_in_h2_stride, v_in9, tw_ptr,
                      load_multi_cols, 0);
        v_in9 = IN_H2_128_S(v_in9);
        LOAD_IN_H2_128_S(in_h2_r, in_strides, 10, v_in_h2_stride, v_in10,
                         tw_ptr, load_multi_cols, 0);
        v_in10 = IN_H2_128_S(v_in10);
        LOAD_IN_H2_128_S(in_h2_r, in_strides, 11, v_in_h2_stride, v_in11,
                         tw_ptr, load_multi_cols, 0);
        v_in11 = IN_H2_128_S(v_in11);
        LOAD_IN_H2_128_S(in_h2_r, in_strides, 12, v_in_h2_stride, v_in12,
                         tw_ptr, load_multi_cols, 0);
        v_in12 = IN_H2_128_S(v_in12);
        LOAD_IN_H2_128_S(in_h2_r, in_strides, 13, v_in_h2_stride, v_in13,
                         tw_ptr, load_multi_cols, 0);
        v_in13 = IN_H2_128_S(v_in13);
        LOAD_IN_H2_128_S(in_h2_r, in_strides, 14, v_in_h2_stride, v_in14,
                         tw_ptr, load_multi_cols, 0);
        v_in14 = IN_H2_128_S(v_in14);
        LOAD_IN_H2_128_S(in_h2_r, in_strides, 15, v_in_h2_stride, v_in15,
                         tw_ptr, load_multi_cols, 0);
        v_in15 = IN_H2_128_S(v_in15);
#else
        LOAD_IN_128_S(in_r, in_strides, 8, v_in_stride, v_in8, tw_ptr,
                      load_multi_cols, is_contiguous_in);
        LOAD_IN_128_S(in_r, in_strides, 9, v_in_stride, v_in9, tw_ptr,
                      load_multi_cols, is_contiguous_in);
        LOAD_IN_128_S(in_r, in_strides, 10, v_in_stride, v_in10, tw_ptr,
                      load_multi_cols, is_contiguous_in);
        LOAD_IN_128_S(in_r, in_strides, 11, v_in_stride, v_in11, tw_ptr,
                      load_multi_cols, is_contiguous_in);
        LOAD_IN_128_S(in_r, in_strides, 12, v_in_stride, v_in12, tw_ptr,
                      load_multi_cols, is_contiguous_in);
        LOAD_IN_128_S(in_r, in_strides, 13, v_in_stride, v_in13, tw_ptr,
                      load_multi_cols, is_contiguous_in);
        LOAD_IN_128_S(in_r, in_strides, 14, v_in_stride, v_in14, tw_ptr,
                      load_multi_cols, is_contiguous_in);
        LOAD_IN_128_S(in_r, in_strides, 15, v_in_stride, v_in15, tw_ptr,
                      load_multi_cols, is_contiguous_in);
#endif

        GATHER2_128_S(in_r, v_in_stride, v_in0, is_contiguous_in);

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

        SCATTER2_128_S(out_r, v_out_stride, v_out0, is_contiguous_out);
        STORE_OUT_128_S(out_r, out_strides, 1, v_out_stride, v_out1, tw_ptr,
                        load_multi_cols, is_contiguous_out);
        STORE_OUT_128_S(out_r, out_strides, 2, v_out_stride, v_out2, tw_ptr,
                        load_multi_cols, is_contiguous_out);
        STORE_OUT_128_S(out_r, out_strides, 3, v_out_stride, v_out3, tw_ptr,
                        load_multi_cols, is_contiguous_out);
        STORE_OUT_128_S(out_r, out_strides, 4, v_out_stride, v_out4, tw_ptr,
                        load_multi_cols, is_contiguous_out);
        STORE_OUT_128_S(out_r, out_strides, 5, v_out_stride, v_out5, tw_ptr,
                        load_multi_cols, is_contiguous_out);
        STORE_OUT_128_S(out_r, out_strides, 6, v_out_stride, v_out6, tw_ptr,
                        load_multi_cols, is_contiguous_out);
        STORE_OUT_128_S(out_r, out_strides, 7, v_out_stride, v_out7, tw_ptr,
                        load_multi_cols, is_contiguous_out);
#if defined(KERNEL_VARIANT_R2C)
        v_out8 = OUT_H2_128_S(v_out8);
        STORE_OUT_H2_128_S(out_h2_r, out_strides, 8, v_out_h2_stride, v_out8,
                        tw_ptr, load_multi_cols, 0);
        v_out9 = OUT_H2_128_S(v_out9);
        STORE_OUT_H2_128_S(out_h2_r, out_strides, 9, v_out_h2_stride, v_out9,
                        tw_ptr, load_multi_cols, 0);
        v_out10 = OUT_H2_128_S(v_out10);
        STORE_OUT_H2_128_S(out_h2_r, out_strides, 10, v_out_h2_stride, v_out10,
                        tw_ptr, load_multi_cols, 0);
        v_out11 = OUT_H2_128_S(v_out11);
        STORE_OUT_H2_128_S(out_h2_r, out_strides, 11, v_out_h2_stride, v_out11,
                        tw_ptr, load_multi_cols, 0);
        v_out12 = OUT_H2_128_S(v_out12);
        STORE_OUT_H2_128_S(out_h2_r, out_strides, 12, v_out_h2_stride, v_out12,
                        tw_ptr, load_multi_cols, 0);
        v_out13 = OUT_H2_128_S(v_out13);
        STORE_OUT_H2_128_S(out_h2_r, out_strides, 13, v_out_h2_stride, v_out13,
                        tw_ptr, load_multi_cols, 0);
        v_out14 = OUT_H2_128_S(v_out14);
        STORE_OUT_H2_128_S(out_h2_r, out_strides, 14, v_out_h2_stride, v_out14,
                        tw_ptr, load_multi_cols, 0);
        v_out15 = OUT_H2_128_S(v_out15);
        STORE_OUT_H2_128_S(out_h2_r, out_strides, 15, v_out_h2_stride, v_out15,
                        tw_ptr, load_multi_cols, 0);
#else
        STORE_OUT_128_S(out_r, out_strides, 8, v_out_stride, v_out8, tw_ptr,
                        load_multi_cols, is_contiguous_out);
        STORE_OUT_128_S(out_r, out_strides, 9, v_out_stride, v_out9, tw_ptr,
                        load_multi_cols, is_contiguous_out);
        STORE_OUT_128_S(out_r, out_strides, 10, v_out_stride, v_out10, tw_ptr,
                        load_multi_cols, is_contiguous_out);
        STORE_OUT_128_S(out_r, out_strides, 11, v_out_stride, v_out11, tw_ptr,
                        load_multi_cols, is_contiguous_out);
        STORE_OUT_128_S(out_r, out_strides, 12, v_out_stride, v_out12, tw_ptr,
                        load_multi_cols, is_contiguous_out);
        STORE_OUT_128_S(out_r, out_strides, 13, v_out_stride, v_out13, tw_ptr,
                        load_multi_cols, is_contiguous_out);
        STORE_OUT_128_S(out_r, out_strides, 14, v_out_stride, v_out14, tw_ptr,
                        load_multi_cols, is_contiguous_out);
        STORE_OUT_128_S(out_r, out_strides, 15, v_out_stride, v_out15, tw_ptr,
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

        LOAD_IN_64_S(in_r, in_strides, 1, v_in1, tw_ptr, load_multi_cols,
                     is_contiguous_in);
        LOAD_IN_64_S(in_r, in_strides, 2, v_in2, tw_ptr, load_multi_cols,
                     is_contiguous_in);
        LOAD_IN_64_S(in_r, in_strides, 3, v_in3, tw_ptr, load_multi_cols,
                     is_contiguous_in);
        LOAD_IN_64_S(in_r, in_strides, 4, v_in4, tw_ptr, load_multi_cols,
                     is_contiguous_in);
        LOAD_IN_64_S(in_r, in_strides, 5, v_in5, tw_ptr, load_multi_cols,
                     is_contiguous_in);
        LOAD_IN_64_S(in_r, in_strides, 6, v_in6, tw_ptr, load_multi_cols,
                     is_contiguous_in);
        LOAD_IN_64_S(in_r, in_strides, 7, v_in7, tw_ptr, load_multi_cols,
                     is_contiguous_in);
#if defined(KERNEL_VARIANT_C2R)
        LOAD_IN_H2_64_S(in_h2_r, in_strides, 8, v_in8, tw_ptr, load_multi_cols,
                        0);
        v_in8 = IN_H2_128_S(v_in8);
        LOAD_IN_H2_64_S(in_h2_r, in_strides, 9, v_in9, tw_ptr, load_multi_cols,
                        0);
        v_in9 = IN_H2_128_S(v_in9);
        LOAD_IN_H2_64_S(in_h2_r, in_strides, 10, v_in10, tw_ptr,
                        load_multi_cols, 0);
        v_in10 = IN_H2_128_S(v_in10);
        LOAD_IN_H2_64_S(in_h2_r, in_strides, 11, v_in11, tw_ptr,
                        load_multi_cols, 0);
        v_in11 = IN_H2_128_S(v_in11);
        LOAD_IN_H2_64_S(in_h2_r, in_strides, 12, v_in12, tw_ptr,
                        load_multi_cols, 0);
        v_in12 = IN_H2_128_S(v_in12);
        LOAD_IN_H2_64_S(in_h2_r, in_strides, 13, v_in13, tw_ptr,
                        load_multi_cols, 0);
        v_in13 = IN_H2_128_S(v_in13);
        LOAD_IN_H2_64_S(in_h2_r, in_strides, 14, v_in14, tw_ptr,
                        load_multi_cols, 0);
        v_in14 = IN_H2_128_S(v_in14);
        LOAD_IN_H2_64_S(in_h2_r, in_strides, 15, v_in15, tw_ptr,
                        load_multi_cols, 0);
        v_in15 = IN_H2_128_S(v_in15);
#else
        LOAD_IN_64_S(in_r, in_strides, 8, v_in8, tw_ptr, load_multi_cols,
                     is_contiguous_in);
        LOAD_IN_64_S(in_r, in_strides, 9, v_in9, tw_ptr, load_multi_cols,
                     is_contiguous_in);
        LOAD_IN_64_S(in_r, in_strides, 10, v_in10, tw_ptr, load_multi_cols,
                     is_contiguous_in);
        LOAD_IN_64_S(in_r, in_strides, 11, v_in11, tw_ptr, load_multi_cols,
                     is_contiguous_in);
        LOAD_IN_64_S(in_r, in_strides, 12, v_in12, tw_ptr, load_multi_cols,
                     is_contiguous_in);
        LOAD_IN_64_S(in_r, in_strides, 13, v_in13, tw_ptr, load_multi_cols,
                     is_contiguous_in);
        LOAD_IN_64_S(in_r, in_strides, 14, v_in14, tw_ptr, load_multi_cols,
                     is_contiguous_in);
        LOAD_IN_64_S(in_r, in_strides, 15, v_in15, tw_ptr, load_multi_cols,
                     is_contiguous_in);
#endif

        LD_LOW_128_S(in_r, v_in0);

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

        ST_LOW_128_S(out_r, v_out0);
        STORE_OUT_64_S(out_r, out_strides, 1, v_out1, tw_ptr, load_multi_cols,
                       is_contiguous_out);
        STORE_OUT_64_S(out_r, out_strides, 2, v_out2, tw_ptr, load_multi_cols,
                       is_contiguous_out);
        STORE_OUT_64_S(out_r, out_strides, 3, v_out3, tw_ptr, load_multi_cols,
                       is_contiguous_out);
        STORE_OUT_64_S(out_r, out_strides, 4, v_out4, tw_ptr, load_multi_cols,
                       is_contiguous_out);
        STORE_OUT_64_S(out_r, out_strides, 5, v_out5, tw_ptr, load_multi_cols,
                       is_contiguous_out);
        STORE_OUT_64_S(out_r, out_strides, 6, v_out6, tw_ptr, load_multi_cols,
                       is_contiguous_out);
        STORE_OUT_64_S(out_r, out_strides, 7, v_out7, tw_ptr, load_multi_cols,
                       is_contiguous_out);
#if defined(KERNEL_VARIANT_R2C)
        v_out8 = OUT_H2_128_S(v_out8);
        STORE_OUT_H2_64_S(out_h2_r, out_strides, 8, v_out8, tw_ptr,
                       load_multi_cols, 0);
        v_out9 = OUT_H2_128_S(v_out9);
        STORE_OUT_H2_64_S(out_h2_r, out_strides, 9, v_out9, tw_ptr,
                       load_multi_cols, 0);
        v_out10 = OUT_H2_128_S(v_out10);
        STORE_OUT_H2_64_S(out_h2_r, out_strides, 10, v_out10, tw_ptr,
                       load_multi_cols, 0);
        v_out11 = OUT_H2_128_S(v_out11);
        STORE_OUT_H2_64_S(out_h2_r, out_strides, 11, v_out11, tw_ptr,
                       load_multi_cols, 0);
        v_out12 = OUT_H2_128_S(v_out12);
        STORE_OUT_H2_64_S(out_h2_r, out_strides, 12, v_out12, tw_ptr,
                       load_multi_cols, 0);
        v_out13 = OUT_H2_128_S(v_out13);
        STORE_OUT_H2_64_S(out_h2_r, out_strides, 13, v_out13, tw_ptr,
                       load_multi_cols, 0);
        v_out14 = OUT_H2_128_S(v_out14);
        STORE_OUT_H2_64_S(out_h2_r, out_strides, 14, v_out14, tw_ptr,
                       load_multi_cols, 0);
        v_out15 = OUT_H2_128_S(v_out15);
        STORE_OUT_H2_64_S(out_h2_r, out_strides, 15, v_out15, tw_ptr,
                       load_multi_cols, 0);
#else
        STORE_OUT_64_S(out_r, out_strides, 8, v_out8, tw_ptr, load_multi_cols,
                       is_contiguous_out);
        STORE_OUT_64_S(out_r, out_strides, 9, v_out9, tw_ptr, load_multi_cols,
                       is_contiguous_out);
        STORE_OUT_64_S(out_r, out_strides, 10, v_out10, tw_ptr, load_multi_cols,
                       is_contiguous_out);
        STORE_OUT_64_S(out_r, out_strides, 11, v_out11, tw_ptr, load_multi_cols,
                       is_contiguous_out);
        STORE_OUT_64_S(out_r, out_strides, 12, v_out12, tw_ptr, load_multi_cols,
                       is_contiguous_out);
        STORE_OUT_64_S(out_r, out_strides, 13, v_out13, tw_ptr, load_multi_cols,
                       is_contiguous_out);
        STORE_OUT_64_S(out_r, out_strides, 14, v_out14, tw_ptr, load_multi_cols,
                       is_contiguous_out);
        STORE_OUT_64_S(out_r, out_strides, 15, v_out15, tw_ptr, load_multi_cols,
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

    const FFTZ_DOUBLE CRTM_16[4] = {
        0.92387953251128675612818318939678828682241662586364,
        0.38268343236508977172845998403039886676134456248563,
        0.70710678118654752440084436210484903928483593768847,
        1.00000000000000000000000000000000000000000000000000};

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

    VREGTYPE_D v_C1 = BCAST_D(CRTM_16[0]);
    VREGTYPE_D v_C2 = BCAST_D(CRTM_16[1]);
    VREGTYPE_D v_C3 = BCAST_D(CRTM_16[2]);
    VREGTYPE_D v_C4 = BCAST_D(CRTM_16[2]);
    VREGTYPE_D v_C5 = BCAST_D(CRTM_16[1]);
    VREGTYPE_D v_C6 = BCAST_D(CRTM_16[0]);
    VREGTYPE_D v_C7 = BCAST_D(CRTM_16[3]);

#if defined(KERNEL_DIRECTION_BWD)
    v_C2 = NEG_D(v_C2, 1);
    v_C4 = NEG_D(v_C4, 1);
    v_C6 = NEG_D(v_C6, 1);
    v_C7 = NEG_D(v_C7, 1);
#endif
    FFTZ_DOUBLE *tw_ptr = tw;

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

        GATHER_D(in_r, v_in_stride, v_in0, is_contiguous_in);

        LOAD_IN_D(in_r, in_strides, 1, v_in_stride, v_in1, tw_ptr,
                  load_multi_cols, is_contiguous_in);
        LOAD_IN_D(in_r, in_strides, 2, v_in_stride, v_in2, tw_ptr,
                  load_multi_cols, is_contiguous_in);
        LOAD_IN_D(in_r, in_strides, 3, v_in_stride, v_in3, tw_ptr,
                  load_multi_cols, is_contiguous_in);
        LOAD_IN_D(in_r, in_strides, 4, v_in_stride, v_in4, tw_ptr,
                  load_multi_cols, is_contiguous_in);
        LOAD_IN_D(in_r, in_strides, 5, v_in_stride, v_in5, tw_ptr,
                  load_multi_cols, is_contiguous_in);
        LOAD_IN_D(in_r, in_strides, 6, v_in_stride, v_in6, tw_ptr,
                  load_multi_cols, is_contiguous_in);
        LOAD_IN_D(in_r, in_strides, 7, v_in_stride, v_in7, tw_ptr,
                  load_multi_cols, is_contiguous_in);
#if defined(KERNEL_VARIANT_C2R)
        LOAD_IN_H2_D(in_h2_r, in_strides, 8, v_in_h2_stride, v_in8, tw_ptr,
                  load_multi_cols, 0);
        v_in8 = IN_H2_D(v_in8);
        LOAD_IN_H2_D(in_h2_r, in_strides, 9, v_in_h2_stride, v_in9, tw_ptr,
                  load_multi_cols, 0);
        v_in9 = IN_H2_D(v_in9);
        LOAD_IN_H2_D(in_h2_r, in_strides, 10, v_in_h2_stride, v_in10, tw_ptr,
                  load_multi_cols, 0);
        v_in10 = IN_H2_D(v_in10);
        LOAD_IN_H2_D(in_h2_r, in_strides, 11, v_in_h2_stride, v_in11, tw_ptr,
                  load_multi_cols, 0);
        v_in11 = IN_H2_D(v_in11);
        LOAD_IN_H2_D(in_h2_r, in_strides, 12, v_in_h2_stride, v_in12, tw_ptr,
                  load_multi_cols, 0);
        v_in12 = IN_H2_D(v_in12);
        LOAD_IN_H2_D(in_h2_r, in_strides, 13, v_in_h2_stride, v_in13, tw_ptr,
                  load_multi_cols, 0);
        v_in13 = IN_H2_D(v_in13);
        LOAD_IN_H2_D(in_h2_r, in_strides, 14, v_in_h2_stride, v_in14, tw_ptr,
                  load_multi_cols, 0);
        v_in14 = IN_H2_D(v_in14);
        LOAD_IN_H2_D(in_h2_r, in_strides, 15, v_in_h2_stride, v_in15, tw_ptr,
                  load_multi_cols, 0);
        v_in15 = IN_H2_D(v_in15);
#else
        LOAD_IN_D(in_r, in_strides, 8, v_in_stride, v_in8, tw_ptr,
                  load_multi_cols, is_contiguous_in);
        LOAD_IN_D(in_r, in_strides, 9, v_in_stride, v_in9, tw_ptr,
                  load_multi_cols, is_contiguous_in);
        LOAD_IN_D(in_r, in_strides, 10, v_in_stride, v_in10, tw_ptr,
                  load_multi_cols, is_contiguous_in);
        LOAD_IN_D(in_r, in_strides, 11, v_in_stride, v_in11, tw_ptr,
                  load_multi_cols, is_contiguous_in);
        LOAD_IN_D(in_r, in_strides, 12, v_in_stride, v_in12, tw_ptr,
                  load_multi_cols, is_contiguous_in);
        LOAD_IN_D(in_r, in_strides, 13, v_in_stride, v_in13, tw_ptr,
                  load_multi_cols, is_contiguous_in);
        LOAD_IN_D(in_r, in_strides, 14, v_in_stride, v_in14, tw_ptr,
                  load_multi_cols, is_contiguous_in);
        LOAD_IN_D(in_r, in_strides, 15, v_in_stride, v_in15, tw_ptr,
                  load_multi_cols, is_contiguous_in);
#endif

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

        SCATTER_D(out_r, v_out_stride, v_out0, is_contiguous_out);
        STORE_OUT_D(out_r, out_strides, 1, v_out_stride, v_out1, tw_ptr,
                    load_multi_cols, is_contiguous_out);
        STORE_OUT_D(out_r, out_strides, 2, v_out_stride, v_out2, tw_ptr,
                    load_multi_cols, is_contiguous_out);
        STORE_OUT_D(out_r, out_strides, 3, v_out_stride, v_out3, tw_ptr,
                    load_multi_cols, is_contiguous_out);
        STORE_OUT_D(out_r, out_strides, 4, v_out_stride, v_out4, tw_ptr,
                    load_multi_cols, is_contiguous_out);
        STORE_OUT_D(out_r, out_strides, 5, v_out_stride, v_out5, tw_ptr,
                    load_multi_cols, is_contiguous_out);
        STORE_OUT_D(out_r, out_strides, 6, v_out_stride, v_out6, tw_ptr,
                    load_multi_cols, is_contiguous_out);
        STORE_OUT_D(out_r, out_strides, 7, v_out_stride, v_out7, tw_ptr,
                    load_multi_cols, is_contiguous_out);
#if defined(KERNEL_VARIANT_R2C)
        v_out8 = OUT_H2_D(v_out8);
        STORE_OUT_H2_D(out_h2_r, out_strides, 8, v_out_h2_stride, v_out8,
                       tw_ptr, load_multi_cols, 0);
        v_out9 = OUT_H2_D(v_out9);
        STORE_OUT_H2_D(out_h2_r, out_strides, 9, v_out_h2_stride, v_out9,
                       tw_ptr, load_multi_cols, 0);
        v_out10 = OUT_H2_D(v_out10);
        STORE_OUT_H2_D(out_h2_r, out_strides, 10, v_out_h2_stride, v_out10,
                       tw_ptr, load_multi_cols, 0);
        v_out11 = OUT_H2_D(v_out11);
        STORE_OUT_H2_D(out_h2_r, out_strides, 11, v_out_h2_stride, v_out11,
                       tw_ptr, load_multi_cols, 0);
        v_out12 = OUT_H2_D(v_out12);
        STORE_OUT_H2_D(out_h2_r, out_strides, 12, v_out_h2_stride, v_out12,
                       tw_ptr, load_multi_cols, 0);
        v_out13 = OUT_H2_D(v_out13);
        STORE_OUT_H2_D(out_h2_r, out_strides, 13, v_out_h2_stride, v_out13,
                       tw_ptr, load_multi_cols, 0);
        v_out14 = OUT_H2_D(v_out14);
        STORE_OUT_H2_D(out_h2_r, out_strides, 14, v_out_h2_stride, v_out14,
                       tw_ptr, load_multi_cols, 0);
        v_out15 = OUT_H2_D(v_out15);
        STORE_OUT_H2_D(out_h2_r, out_strides, 15, v_out_h2_stride, v_out15,
                       tw_ptr, load_multi_cols, 0);
#else
        STORE_OUT_D(out_r, out_strides, 8, v_out_stride, v_out8, tw_ptr,
                    load_multi_cols, is_contiguous_out);
        STORE_OUT_D(out_r, out_strides, 9, v_out_stride, v_out9, tw_ptr,
                    load_multi_cols, is_contiguous_out);
        STORE_OUT_D(out_r, out_strides, 10, v_out_stride, v_out10, tw_ptr,
                    load_multi_cols, is_contiguous_out);
        STORE_OUT_D(out_r, out_strides, 11, v_out_stride, v_out11, tw_ptr,
                    load_multi_cols, is_contiguous_out);
        STORE_OUT_D(out_r, out_strides, 12, v_out_stride, v_out12, tw_ptr,
                    load_multi_cols, is_contiguous_out);
        STORE_OUT_D(out_r, out_strides, 13, v_out_stride, v_out13, tw_ptr,
                    load_multi_cols, is_contiguous_out);
        STORE_OUT_D(out_r, out_strides, 14, v_out_stride, v_out14, tw_ptr,
                    load_multi_cols, is_contiguous_out);
        STORE_OUT_D(out_r, out_strides, 15, v_out_stride, v_out15, tw_ptr,
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

        GATHER2_256_D(in_r, v_in_stride, v_in0, is_contiguous_in);

        // Fused load+twiddle-multiply at column 0 relative to the walked
        // tile pointer (mirrors the main loop's load_multi_cols arm).
        LOAD_IN_256_D(in_r, in_strides, 1, v_in_stride, v_in1, tw_ptr,
                      load_multi_cols, is_contiguous_in);
        LOAD_IN_256_D(in_r, in_strides, 2, v_in_stride, v_in2, tw_ptr,
                      load_multi_cols, is_contiguous_in);
        LOAD_IN_256_D(in_r, in_strides, 3, v_in_stride, v_in3, tw_ptr,
                      load_multi_cols, is_contiguous_in);
        LOAD_IN_256_D(in_r, in_strides, 4, v_in_stride, v_in4, tw_ptr,
                      load_multi_cols, is_contiguous_in);
        LOAD_IN_256_D(in_r, in_strides, 5, v_in_stride, v_in5, tw_ptr,
                      load_multi_cols, is_contiguous_in);
        LOAD_IN_256_D(in_r, in_strides, 6, v_in_stride, v_in6, tw_ptr,
                      load_multi_cols, is_contiguous_in);
        LOAD_IN_256_D(in_r, in_strides, 7, v_in_stride, v_in7, tw_ptr,
                      load_multi_cols, is_contiguous_in);
#if defined(KERNEL_VARIANT_C2R)
        LOAD_IN_H2_256_D(in_h2_r, in_strides, 8, v_in_h2_stride, v_in8, tw_ptr,
                      load_multi_cols, 0);
        v_in8 = IN_H2_256_D(v_in8);
        LOAD_IN_H2_256_D(in_h2_r, in_strides, 9, v_in_h2_stride, v_in9, tw_ptr,
                      load_multi_cols, 0);
        v_in9 = IN_H2_256_D(v_in9);
        LOAD_IN_H2_256_D(in_h2_r, in_strides, 10, v_in_h2_stride, v_in10,
                         tw_ptr, load_multi_cols, 0);
        v_in10 = IN_H2_256_D(v_in10);
        LOAD_IN_H2_256_D(in_h2_r, in_strides, 11, v_in_h2_stride, v_in11,
                         tw_ptr, load_multi_cols, 0);
        v_in11 = IN_H2_256_D(v_in11);
        LOAD_IN_H2_256_D(in_h2_r, in_strides, 12, v_in_h2_stride, v_in12,
                         tw_ptr, load_multi_cols, 0);
        v_in12 = IN_H2_256_D(v_in12);
        LOAD_IN_H2_256_D(in_h2_r, in_strides, 13, v_in_h2_stride, v_in13,
                         tw_ptr, load_multi_cols, 0);
        v_in13 = IN_H2_256_D(v_in13);
        LOAD_IN_H2_256_D(in_h2_r, in_strides, 14, v_in_h2_stride, v_in14,
                         tw_ptr, load_multi_cols, 0);
        v_in14 = IN_H2_256_D(v_in14);
        LOAD_IN_H2_256_D(in_h2_r, in_strides, 15, v_in_h2_stride, v_in15,
                         tw_ptr, load_multi_cols, 0);
        v_in15 = IN_H2_256_D(v_in15);
#else
        LOAD_IN_256_D(in_r, in_strides, 8, v_in_stride, v_in8, tw_ptr,
                      load_multi_cols, is_contiguous_in);
        LOAD_IN_256_D(in_r, in_strides, 9, v_in_stride, v_in9, tw_ptr,
                      load_multi_cols, is_contiguous_in);
        LOAD_IN_256_D(in_r, in_strides, 10, v_in_stride, v_in10, tw_ptr,
                      load_multi_cols, is_contiguous_in);
        LOAD_IN_256_D(in_r, in_strides, 11, v_in_stride, v_in11, tw_ptr,
                      load_multi_cols, is_contiguous_in);
        LOAD_IN_256_D(in_r, in_strides, 12, v_in_stride, v_in12, tw_ptr,
                      load_multi_cols, is_contiguous_in);
        LOAD_IN_256_D(in_r, in_strides, 13, v_in_stride, v_in13, tw_ptr,
                      load_multi_cols, is_contiguous_in);
        LOAD_IN_256_D(in_r, in_strides, 14, v_in_stride, v_in14, tw_ptr,
                      load_multi_cols, is_contiguous_in);
        LOAD_IN_256_D(in_r, in_strides, 15, v_in_stride, v_in15, tw_ptr,
                      load_multi_cols, is_contiguous_in);
#endif

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

        SCATTER2_256_D(out_r, v_out_stride, v_out0, is_contiguous_out);
        // Fused twiddle store at column 0 relative to the walked tile
        // pointer (mirrors the main loop's load_multi_cols arm).
        STORE_OUT_256_D(out_r, out_strides, 1, v_out_stride, v_out1, tw_ptr,
                        load_multi_cols, is_contiguous_out);
        STORE_OUT_256_D(out_r, out_strides, 2, v_out_stride, v_out2, tw_ptr,
                        load_multi_cols, is_contiguous_out);
        STORE_OUT_256_D(out_r, out_strides, 3, v_out_stride, v_out3, tw_ptr,
                        load_multi_cols, is_contiguous_out);
        STORE_OUT_256_D(out_r, out_strides, 4, v_out_stride, v_out4, tw_ptr,
                        load_multi_cols, is_contiguous_out);
        STORE_OUT_256_D(out_r, out_strides, 5, v_out_stride, v_out5, tw_ptr,
                        load_multi_cols, is_contiguous_out);
        STORE_OUT_256_D(out_r, out_strides, 6, v_out_stride, v_out6, tw_ptr,
                        load_multi_cols, is_contiguous_out);
        STORE_OUT_256_D(out_r, out_strides, 7, v_out_stride, v_out7, tw_ptr,
                        load_multi_cols, is_contiguous_out);
#if defined(KERNEL_VARIANT_R2C)
        v_out8 = OUT_H2_256_D(v_out8);
        STORE_OUT_H2_256_D(out_h2_r, out_strides, 8, v_out_h2_stride, v_out8,
                        tw_ptr, load_multi_cols, 0);
        v_out9 = OUT_H2_256_D(v_out9);
        STORE_OUT_H2_256_D(out_h2_r, out_strides, 9, v_out_h2_stride, v_out9,
                        tw_ptr, load_multi_cols, 0);
        v_out10 = OUT_H2_256_D(v_out10);
        STORE_OUT_H2_256_D(out_h2_r, out_strides, 10, v_out_h2_stride, v_out10,
                        tw_ptr, load_multi_cols, 0);
        v_out11 = OUT_H2_256_D(v_out11);
        STORE_OUT_H2_256_D(out_h2_r, out_strides, 11, v_out_h2_stride, v_out11,
                        tw_ptr, load_multi_cols, 0);
        v_out12 = OUT_H2_256_D(v_out12);
        STORE_OUT_H2_256_D(out_h2_r, out_strides, 12, v_out_h2_stride, v_out12,
                        tw_ptr, load_multi_cols, 0);
        v_out13 = OUT_H2_256_D(v_out13);
        STORE_OUT_H2_256_D(out_h2_r, out_strides, 13, v_out_h2_stride, v_out13,
                        tw_ptr, load_multi_cols, 0);
        v_out14 = OUT_H2_256_D(v_out14);
        STORE_OUT_H2_256_D(out_h2_r, out_strides, 14, v_out_h2_stride, v_out14,
                        tw_ptr, load_multi_cols, 0);
        v_out15 = OUT_H2_256_D(v_out15);
        STORE_OUT_H2_256_D(out_h2_r, out_strides, 15, v_out_h2_stride, v_out15,
                        tw_ptr, load_multi_cols, 0);
#else
        STORE_OUT_256_D(out_r, out_strides, 8, v_out_stride, v_out8, tw_ptr,
                        load_multi_cols, is_contiguous_out);
        STORE_OUT_256_D(out_r, out_strides, 9, v_out_stride, v_out9, tw_ptr,
                        load_multi_cols, is_contiguous_out);
        STORE_OUT_256_D(out_r, out_strides, 10, v_out_stride, v_out10, tw_ptr,
                        load_multi_cols, is_contiguous_out);
        STORE_OUT_256_D(out_r, out_strides, 11, v_out_stride, v_out11, tw_ptr,
                        load_multi_cols, is_contiguous_out);
        STORE_OUT_256_D(out_r, out_strides, 12, v_out_stride, v_out12, tw_ptr,
                        load_multi_cols, is_contiguous_out);
        STORE_OUT_256_D(out_r, out_strides, 13, v_out_stride, v_out13, tw_ptr,
                        load_multi_cols, is_contiguous_out);
        STORE_OUT_256_D(out_r, out_strides, 14, v_out_stride, v_out14, tw_ptr,
                        load_multi_cols, is_contiguous_out);
        STORE_OUT_256_D(out_r, out_strides, 15, v_out_stride, v_out15, tw_ptr,
                        load_multi_cols, is_contiguous_out);
#endif
        // Walk to the next tile for the following (128-bit) tail.
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

        LD_128_D(in_r, v_in0);

        // Fused load+twiddle-multiply at column 0 relative to the walked
        // tile pointer (mirrors the main loop's load_multi_cols arm).
        LOAD_IN_128_D(in_r, in_strides, 1, 0, v_in1, tw_ptr, load_multi_cols,
                      is_contiguous_in);
        LOAD_IN_128_D(in_r, in_strides, 2, 0, v_in2, tw_ptr, load_multi_cols,
                      is_contiguous_in);
        LOAD_IN_128_D(in_r, in_strides, 3, 0, v_in3, tw_ptr, load_multi_cols,
                      is_contiguous_in);
        LOAD_IN_128_D(in_r, in_strides, 4, 0, v_in4, tw_ptr, load_multi_cols,
                      is_contiguous_in);
        LOAD_IN_128_D(in_r, in_strides, 5, 0, v_in5, tw_ptr, load_multi_cols,
                      is_contiguous_in);
        LOAD_IN_128_D(in_r, in_strides, 6, 0, v_in6, tw_ptr, load_multi_cols,
                      is_contiguous_in);
        LOAD_IN_128_D(in_r, in_strides, 7, 0, v_in7, tw_ptr, load_multi_cols,
                      is_contiguous_in);
#if defined(KERNEL_VARIANT_C2R)
        LOAD_IN_H2_128_D(in_h2_r, in_strides, 8, 0, v_in8, tw_ptr,
                         load_multi_cols, 0);
        v_in8 = IN_H2_128_D(v_in8);
        LOAD_IN_H2_128_D(in_h2_r, in_strides, 9, 0, v_in9, tw_ptr,
                         load_multi_cols, 0);
        v_in9 = IN_H2_128_D(v_in9);
        LOAD_IN_H2_128_D(in_h2_r, in_strides, 10, 0, v_in10, tw_ptr,
                      load_multi_cols, 0);
        v_in10 = IN_H2_128_D(v_in10);
        LOAD_IN_H2_128_D(in_h2_r, in_strides, 11, 0, v_in11, tw_ptr,
                      load_multi_cols, 0);
        v_in11 = IN_H2_128_D(v_in11);
        LOAD_IN_H2_128_D(in_h2_r, in_strides, 12, 0, v_in12, tw_ptr,
                      load_multi_cols, 0);
        v_in12 = IN_H2_128_D(v_in12);
        LOAD_IN_H2_128_D(in_h2_r, in_strides, 13, 0, v_in13, tw_ptr,
                      load_multi_cols, 0);
        v_in13 = IN_H2_128_D(v_in13);
        LOAD_IN_H2_128_D(in_h2_r, in_strides, 14, 0, v_in14, tw_ptr,
                      load_multi_cols, 0);
        v_in14 = IN_H2_128_D(v_in14);
        LOAD_IN_H2_128_D(in_h2_r, in_strides, 15, 0, v_in15, tw_ptr,
                      load_multi_cols, 0);
        v_in15 = IN_H2_128_D(v_in15);
#else
        LOAD_IN_128_D(in_r, in_strides, 8, 0, v_in8, tw_ptr, load_multi_cols,
                      is_contiguous_in);
        LOAD_IN_128_D(in_r, in_strides, 9, 0, v_in9, tw_ptr, load_multi_cols,
                      is_contiguous_in);
        LOAD_IN_128_D(in_r, in_strides, 10, 0, v_in10, tw_ptr, load_multi_cols,
                      is_contiguous_in);
        LOAD_IN_128_D(in_r, in_strides, 11, 0, v_in11, tw_ptr, load_multi_cols,
                      is_contiguous_in);
        LOAD_IN_128_D(in_r, in_strides, 12, 0, v_in12, tw_ptr, load_multi_cols,
                      is_contiguous_in);
        LOAD_IN_128_D(in_r, in_strides, 13, 0, v_in13, tw_ptr, load_multi_cols,
                      is_contiguous_in);
        LOAD_IN_128_D(in_r, in_strides, 14, 0, v_in14, tw_ptr, load_multi_cols,
                      is_contiguous_in);
        LOAD_IN_128_D(in_r, in_strides, 15, 0, v_in15, tw_ptr, load_multi_cols,
                      is_contiguous_in);
#endif

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

        ST_128_D(out_r, v_out0);
        // Fused twiddle store at column 0 relative to the walked tile
        // pointer (mirrors the main loop's load_multi_cols arm).
        STORE_OUT_128_D(out_r, out_strides, 1, 0, v_out1, tw_ptr,
                        load_multi_cols, is_contiguous_out);
        STORE_OUT_128_D(out_r, out_strides, 2, 0, v_out2, tw_ptr,
                        load_multi_cols, is_contiguous_out);
        STORE_OUT_128_D(out_r, out_strides, 3, 0, v_out3, tw_ptr,
                        load_multi_cols, is_contiguous_out);
        STORE_OUT_128_D(out_r, out_strides, 4, 0, v_out4, tw_ptr,
                        load_multi_cols, is_contiguous_out);
        STORE_OUT_128_D(out_r, out_strides, 5, 0, v_out5, tw_ptr,
                        load_multi_cols, is_contiguous_out);
        STORE_OUT_128_D(out_r, out_strides, 6, 0, v_out6, tw_ptr,
                        load_multi_cols, is_contiguous_out);
        STORE_OUT_128_D(out_r, out_strides, 7, 0, v_out7, tw_ptr,
                        load_multi_cols, is_contiguous_out);
#if defined(KERNEL_VARIANT_R2C)
        v_out8 = OUT_H2_128_D(v_out8);
        STORE_OUT_H2_128_D(out_h2_r, out_strides, 8, 0, v_out8, tw_ptr,
                        load_multi_cols, 0);
        v_out9 = OUT_H2_128_D(v_out9);
        STORE_OUT_H2_128_D(out_h2_r, out_strides, 9, 0, v_out9, tw_ptr,
                        load_multi_cols, 0);
        v_out10 = OUT_H2_128_D(v_out10);
        STORE_OUT_H2_128_D(out_h2_r, out_strides, 10, 0, v_out10, tw_ptr,
                        load_multi_cols, 0);
        v_out11 = OUT_H2_128_D(v_out11);
        STORE_OUT_H2_128_D(out_h2_r, out_strides, 11, 0, v_out11, tw_ptr,
                        load_multi_cols, 0);
        v_out12 = OUT_H2_128_D(v_out12);
        STORE_OUT_H2_128_D(out_h2_r, out_strides, 12, 0, v_out12, tw_ptr,
                        load_multi_cols, 0);
        v_out13 = OUT_H2_128_D(v_out13);
        STORE_OUT_H2_128_D(out_h2_r, out_strides, 13, 0, v_out13, tw_ptr,
                        load_multi_cols, 0);
        v_out14 = OUT_H2_128_D(v_out14);
        STORE_OUT_H2_128_D(out_h2_r, out_strides, 14, 0, v_out14, tw_ptr,
                        load_multi_cols, 0);
        v_out15 = OUT_H2_128_D(v_out15);
        STORE_OUT_H2_128_D(out_h2_r, out_strides, 15, 0, v_out15, tw_ptr,
                        load_multi_cols, 0);
#else
        STORE_OUT_128_D(out_r, out_strides, 8, 0, v_out8, tw_ptr,
                        load_multi_cols, is_contiguous_out);
        STORE_OUT_128_D(out_r, out_strides, 9, 0, v_out9, tw_ptr,
                        load_multi_cols, is_contiguous_out);
        STORE_OUT_128_D(out_r, out_strides, 10, 0, v_out10, tw_ptr,
                        load_multi_cols, is_contiguous_out);
        STORE_OUT_128_D(out_r, out_strides, 11, 0, v_out11, tw_ptr,
                        load_multi_cols, is_contiguous_out);
        STORE_OUT_128_D(out_r, out_strides, 12, 0, v_out12, tw_ptr,
                        load_multi_cols, is_contiguous_out);
        STORE_OUT_128_D(out_r, out_strides, 13, 0, v_out13, tw_ptr,
                        load_multi_cols, is_contiguous_out);
        STORE_OUT_128_D(out_r, out_strides, 14, 0, v_out14, tw_ptr,
                        load_multi_cols, is_contiguous_out);
        STORE_OUT_128_D(out_r, out_strides, 15, 0, v_out15, tw_ptr,
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
