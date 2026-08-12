// Copyright Advanced Micro Devices, Inc.
// SPDX-License-Identifier: BSD-3-Clause

/** @file twid_fft9.h
 *
 *  @brief The ISA generic kernel template for the radix 9 twiddle kernel
 *
 *  This file contains the DIT twiddle radix-9 FFT implementations for
 *  single-precision and double-precision inputs.
 *
 *  @author Ashwin K. Godbole
 */

// This header has no include guards.
// This is intentional.
// The functions defined in this file are not usable by default.
// They are "instantiated" only when "included" in another file.

#include "core/kernels/simd_includes/generic_kernels_common.h"

static FFTZ_VOID TWID_KNAME_FP32(FFTZ_VOID *in_real, FFTZ_VOID *in_imag,
                                 FFTZ_VOID *out_real, FFTZ_VOID *out_imag,
                                 FFTZ_INTP n, aoclfftz_strides_t *strides,
                                 FFTZ_VOID *twd, FFTZ_UINT8 flag)
{
    AOCLFFTZ_LOG(DEBUG, global_logger_mode, "Enter");

    const FFTZ_FLOAT CRTM_9[8] = {
        -0.939692620785908384054109277324731469936208134f,
        0.342020143325668733044099614682259580763083368f,
        0.984807753012208059366743024589523013670643252f,
        0.173648177666930348851716626769314796000375677f,
        0.642787609686539326322643409907263432907559884f,
        0.766044443118978035202392650555416673935832457f,
        0.500000000000000000000000000000000000000000000f,
        0.866025403784438646763723170752936183471402627f};

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
    FFTZ_UINTP cols = tws->cols;
    FFTZ_UINTP load_multi_cols = tws->load_multi_cols;

    FFTZ_INTP N = n / NUM_SETS_S;
    FFTZ_INTP remaining_sets = n % NUM_SETS_S;

#if defined(KERNEL_USE_AVX512)
    FFTZ_INTP do_256_whole = (FFTZ_INTP)(remaining_sets >= NUM_SETS_256_S);
    FFTZ_INTP do_128_whole =
        (FFTZ_INTP)(remaining_sets % NUM_SETS_256_S >= NUM_SETS_128_S);
    FFTZ_INTP cnt_256 = load_multi_cols * (N * NUM_SETS_512_S);
    FFTZ_INTP cnt_128 =
        load_multi_cols * (N * NUM_SETS_512_S + do_256_whole * NUM_SETS_256_S);
    FFTZ_INTP cnt_128_low =
        load_multi_cols * (N * NUM_SETS_512_S + do_256_whole * NUM_SETS_256_S +
                           do_128_whole * NUM_SETS_128_S);
#elif defined(KERNEL_USE_AVX256)
    FFTZ_INTP do_128_whole = (FFTZ_INTP)(remaining_sets >= NUM_SETS_128_S);
    FFTZ_INTP cnt_128 = load_multi_cols * (N * NUM_SETS_256_S);
    FFTZ_INTP cnt_128_low =
        load_multi_cols * (N * NUM_SETS_256_S + do_128_whole * NUM_SETS_128_S);
#elif defined(KERNEL_USE_AVX128)
    FFTZ_INTP cnt_128_low = load_multi_cols * (N * NUM_SETS_128_S);
#endif

    VREGTYPE_S v_C1 = BCAST_S(CRTM_9[0]);
    VREGTYPE_S v_C2 = BCAST_S(CRTM_9[1]);
    VREGTYPE_S v_C3 = BCAST_S(CRTM_9[2]);
    VREGTYPE_S v_C4 = BCAST_S(CRTM_9[3]);
    VREGTYPE_S v_C5 = BCAST_S(CRTM_9[4]);
    VREGTYPE_S v_C6 = BCAST_S(CRTM_9[5]);
    VREGTYPE_S v_C7 = BCAST_S(CRTM_9[6]);
    VREGTYPE_S v_C8 = BCAST_S(CRTM_9[7]);

    FFTZ_INTP count;

#if defined(KERNEL_DIRECTION_BWD)
    v_C8 = NEG_S(v_C8, 1);
    v_C5 = NEG_S(v_C5, 1);
    v_C3 = NEG_S(v_C3, 1);
    v_C2 = NEG_S(v_C2, 1);
#endif
    for (count = 0; count < N; count++)
    {
        VREGTYPE_S v_in0, v_in1, v_in2, v_in3, v_in4, v_in5, v_in6, v_in7,
            v_in8;
        VREGTYPE_S v_out0, v_out1, v_out2, v_out3, v_out4, v_out5, v_out6,
            v_out7, v_out8;
        VREGTYPE_S v_tv1, v_tv2, v_tv3, v_tv4, v_av1;
        VREGTYPE_S v_cv1, v_cv2, v_cv3, v_cv4, v_cv5, v_cv6, v_cv7, v_cv8,
            v_cv9;
        FFTZ_INTP col = count * load_multi_cols * NUM_SETS_S;

        LOAD_IN_S(in_r, in_strides, 1, v_in_stride, v_in1, tw, cols, col,
                  load_multi_cols, is_contiguous_in);
        LOAD_IN_S(in_r, in_strides, 2, v_in_stride, v_in2, tw, cols, col,
                  load_multi_cols, is_contiguous_in);
        LOAD_IN_S(in_r, in_strides, 3, v_in_stride, v_in3, tw, cols, col,
                  load_multi_cols, is_contiguous_in);
        LOAD_IN_S(in_r, in_strides, 4, v_in_stride, v_in4, tw, cols, col,
                  load_multi_cols, is_contiguous_in);
#if defined(KERNEL_VARIANT_C2R)
        LOAD_IN_S(in_h2_r, in_strides, 5, v_in_h2_stride, v_in5, tw, cols, col,
                  load_multi_cols, 0);
        v_in5 = IN_H2_S(v_in5);
        LOAD_IN_S(in_h2_r, in_strides, 6, v_in_h2_stride, v_in6, tw, cols, col,
                  load_multi_cols, 0);
        v_in6 = IN_H2_S(v_in6);
        LOAD_IN_S(in_h2_r, in_strides, 7, v_in_h2_stride, v_in7, tw, cols, col,
                  load_multi_cols, 0);
        v_in7 = IN_H2_S(v_in7);
        LOAD_IN_S(in_h2_r, in_strides, 8, v_in_h2_stride, v_in8, tw, cols, col,
                  load_multi_cols, 0);
        v_in8 = IN_H2_S(v_in8);
#else
        LOAD_IN_S(in_r, in_strides, 5, v_in_stride, v_in5, tw, cols, col,
                  load_multi_cols, is_contiguous_in);
        LOAD_IN_S(in_r, in_strides, 6, v_in_stride, v_in6, tw, cols, col,
                  load_multi_cols, is_contiguous_in);
        LOAD_IN_S(in_r, in_strides, 7, v_in_stride, v_in7, tw, cols, col,
                  load_multi_cols, is_contiguous_in);
        LOAD_IN_S(in_r, in_strides, 8, v_in_stride, v_in8, tw, cols, col,
                  load_multi_cols, is_contiguous_in);
#endif
        GATHER_S(in_r, v_in_stride, v_in0, is_contiguous_in);

        // common operations
        v_av1 = ADD_S(v_in3, v_in6);
        v_cv1 = ADD_S(v_in0, v_av1);
        v_tv1 = SUB_S(v_in0, MUL_S(v_C7, v_av1));
        v_tv2 = MUL_S(v_C8, SUB_S(v_in3, v_in6));
        v_tv2 = CONJ_S(SWAP_RI_S(v_tv2));
        v_cv2 = ADD_S(v_tv1, v_tv2);
        v_cv3 = SUB_S(v_tv1, v_tv2);

        v_av1 = ADD_S(v_in4, v_in7);
        v_cv4 = ADD_S(v_in1, v_av1);
        v_tv1 = SUB_S(v_in1, MUL_S(v_C7, v_av1));
        v_tv2 = MUL_S(v_C8, SUB_S(v_in4, v_in7));
        v_tv2 = CONJ_S(SWAP_RI_S(v_tv2));
        v_cv5 = ADD_S(v_tv1, v_tv2);
        v_cv6 = SUB_S(v_tv1, v_tv2);

        v_av1 = ADD_S(v_in5, v_in8);
        v_cv7 = ADD_S(v_in2, v_av1);
        v_tv1 = SUB_S(v_in2, MUL_S(v_C7, v_av1));
        v_tv2 = MUL_S(v_C8, SUB_S(v_in5, v_in8));
        v_tv2 = CONJ_S(SWAP_RI_S(v_tv2));
        v_cv8 = ADD_S(v_tv1, v_tv2);
        v_cv9 = SUB_S(v_tv1, v_tv2);

        // Output point 1: X[0]
        v_av1 = ADD_S(v_cv4, v_cv7);
        v_out0 = ADD_S(v_cv1, v_av1);

        // Output point 4: X[3]
        v_tv1 = SUB_S(v_cv1, MUL_S(v_C7, v_av1));
        v_tv2 = MUL_S(v_C8, SUB_S(v_cv4, v_cv7));
        v_tv2 = CONJ_S(SWAP_RI_S(v_tv2));
        v_out3 = ADD_S(v_tv1, v_tv2);
        // Output point 7: X[6]
        v_out6 = SUB_S(v_tv1, v_tv2);

        v_tv3 = MUL_S(v_cv5, v_C6);
        v_tv4 = MUL_S(v_cv5, v_C5);
        SUBADD_SWAPA_S(v_tv3, v_tv4, v_cv5);
        v_tv3 = MUL_S(v_cv8, v_C4);
        v_tv4 = MUL_S(v_cv8, v_C3);
        SUBADD_SWAPA_S(v_tv3, v_tv4, v_cv8);
        // Output point 2: X[1]
        v_av1 = ADD_S(v_cv5, v_cv8);
        v_out1 = ADD_S(v_cv2, v_av1);

        // Output point 5: X[4]
        v_tv1 = SUB_S(v_cv2, MUL_S(v_C7, v_av1));
        v_tv2 = MUL_S(v_C8, SUB_S(v_cv5, v_cv8));
        v_tv2 = CONJ_S(SWAP_RI_S(v_tv2));
        v_out4 = ADD_S(v_tv1, v_tv2);
        // Output point 8: X[7]
        v_out7 = SUB_S(v_tv1, v_tv2);

        v_tv3 = MUL_S(v_cv6, v_C4);
        v_tv4 = MUL_S(v_cv6, v_C3);
        SUBADD_SWAPA_S(v_tv3, v_tv4, v_cv6);
        v_tv3 = MUL_S(v_cv9, v_C1);
        v_tv4 = MUL_S(v_cv9, v_C2);
        SUBADD_SWAPA_S(v_tv3, v_tv4, v_cv9);
        // Output point 3: X[2]
        v_av1 = ADD_S(v_cv6, v_cv9);
        v_out2 = ADD_S(v_cv3, v_av1);

        // Output point 6: X[5]
        v_tv1 = SUB_S(v_cv3, MUL_S(v_C7, v_av1));
        v_tv2 = MUL_S(v_C8, SUB_S(v_cv6, v_cv9));
        v_tv2 = CONJ_S(SWAP_RI_S(v_tv2));
        v_out5 = ADD_S(v_tv1, v_tv2);
        // Output point 9: X[8]
        v_out8 = SUB_S(v_tv1, v_tv2);

        SCATTER_S(out_r, v_out_stride, v_out0, is_contiguous_out);
        STORE_OUT_S(out_r, out_strides, 1, v_out_stride, v_out1, tw, cols, col,
                    load_multi_cols, is_contiguous_out);
        STORE_OUT_S(out_r, out_strides, 2, v_out_stride, v_out2, tw, cols, col,
                    load_multi_cols, is_contiguous_out);
        STORE_OUT_S(out_r, out_strides, 3, v_out_stride, v_out3, tw, cols, col,
                    load_multi_cols, is_contiguous_out);
        STORE_OUT_S(out_r, out_strides, 4, v_out_stride, v_out4, tw, cols, col,
                    load_multi_cols, is_contiguous_out);
#if defined(KERNEL_VARIANT_R2C)
        v_out5 = OUT_H2_S(v_out5);
        STORE_OUT_S(out_h2_r, out_strides, 5, v_out_h2_stride, v_out5, tw, cols,
                    col, load_multi_cols, 0);
        v_out6 = OUT_H2_S(v_out6);
        STORE_OUT_S(out_h2_r, out_strides, 6, v_out_h2_stride, v_out6, tw, cols,
                    col, load_multi_cols, 0);
        v_out7 = OUT_H2_S(v_out7);
        STORE_OUT_S(out_h2_r, out_strides, 7, v_out_h2_stride, v_out7, tw, cols,
                    col, load_multi_cols, 0);
        v_out8 = OUT_H2_S(v_out8);
        STORE_OUT_S(out_h2_r, out_strides, 8, v_out_h2_stride, v_out8, tw, cols,
                    col, load_multi_cols, 0);
#else
        STORE_OUT_S(out_r, out_strides, 5, v_out_stride, v_out5, tw, cols, col,
                    load_multi_cols, is_contiguous_out);
        STORE_OUT_S(out_r, out_strides, 6, v_out_stride, v_out6, tw, cols, col,
                    load_multi_cols, is_contiguous_out);
        STORE_OUT_S(out_r, out_strides, 7, v_out_stride, v_out7, tw, cols, col,
                    load_multi_cols, is_contiguous_out);
        STORE_OUT_S(out_r, out_strides, 8, v_out_stride, v_out8, tw, cols, col,
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
        __m256 v_in0, v_in1, v_in2, v_in3, v_in4, v_in5, v_in6, v_in7, v_in8;
        __m256 v_out0, v_out1, v_out2, v_out3, v_out4, v_out5, v_out6, v_out7,
            v_out8;
        __m256 v_tv1, v_tv2, v_tv3, v_tv4, v_av1;
        __m256 v_cv1, v_cv2, v_cv3, v_cv4, v_cv5, v_cv6, v_cv7, v_cv8, v_cv9;

        __m256 v_K1 = CAST_512_TO_256_S(v_C1);
        __m256 v_K2 = CAST_512_TO_256_S(v_C2);
        __m256 v_K3 = CAST_512_TO_256_S(v_C3);
        __m256 v_K4 = CAST_512_TO_256_S(v_C4);
        __m256 v_K5 = CAST_512_TO_256_S(v_C5);
        __m256 v_K6 = CAST_512_TO_256_S(v_C6);
        __m256 v_K7 = CAST_512_TO_256_S(v_C7);
        __m256 v_K8 = CAST_512_TO_256_S(v_C8);

        LOAD_IN_256_S(in_r, in_strides, 1, v_in_stride, v_in1, tw, cols,
                      cnt_256, load_multi_cols, is_contiguous_in);
        LOAD_IN_256_S(in_r, in_strides, 2, v_in_stride, v_in2, tw, cols,
                      cnt_256, load_multi_cols, is_contiguous_in);
        LOAD_IN_256_S(in_r, in_strides, 3, v_in_stride, v_in3, tw, cols,
                      cnt_256, load_multi_cols, is_contiguous_in);
        LOAD_IN_256_S(in_r, in_strides, 4, v_in_stride, v_in4, tw, cols,
                      cnt_256, load_multi_cols, is_contiguous_in);
#if defined(KERNEL_VARIANT_C2R)
        LOAD_IN_256_S(in_h2_r, in_strides, 5, v_in_h2_stride, v_in5, tw, cols,
                      cnt_256, load_multi_cols, 0);
        v_in5 = IN_H2_256_S(v_in5);
        LOAD_IN_256_S(in_h2_r, in_strides, 6, v_in_h2_stride, v_in6, tw, cols,
                      cnt_256, load_multi_cols, 0);
        v_in6 = IN_H2_256_S(v_in6);
        LOAD_IN_256_S(in_h2_r, in_strides, 7, v_in_h2_stride, v_in7, tw, cols,
                      cnt_256, load_multi_cols, 0);
        v_in7 = IN_H2_256_S(v_in7);
        LOAD_IN_256_S(in_h2_r, in_strides, 8, v_in_h2_stride, v_in8, tw, cols,
                      cnt_256, load_multi_cols, 0);
        v_in8 = IN_H2_256_S(v_in8);
#else
        LOAD_IN_256_S(in_r, in_strides, 5, v_in_stride, v_in5, tw, cols,
                      cnt_256, load_multi_cols, is_contiguous_in);
        LOAD_IN_256_S(in_r, in_strides, 6, v_in_stride, v_in6, tw, cols,
                      cnt_256, load_multi_cols, is_contiguous_in);
        LOAD_IN_256_S(in_r, in_strides, 7, v_in_stride, v_in7, tw, cols,
                      cnt_256, load_multi_cols, is_contiguous_in);
        LOAD_IN_256_S(in_r, in_strides, 8, v_in_stride, v_in8, tw, cols,
                      cnt_256, load_multi_cols, is_contiguous_in);
#endif
        GATHER4_256_S(in_r, v_in_stride, v_in0, is_contiguous_in);

        // common operations
        v_av1 = _mm256_add_ps(v_in3, v_in6);
        v_cv1 = _mm256_add_ps(v_in0, v_av1);
        v_tv1 = _mm256_sub_ps(v_in0, _mm256_mul_ps(v_K7, v_av1));
        v_tv2 = _mm256_mul_ps(v_K8, _mm256_sub_ps(v_in3, v_in6));
        v_tv2 = CONJ_256_S(SWAP_RI_256_S(v_tv2));
        v_cv2 = _mm256_add_ps(v_tv1, v_tv2);
        v_cv3 = _mm256_sub_ps(v_tv1, v_tv2);

        v_av1 = _mm256_add_ps(v_in4, v_in7);
        v_cv4 = _mm256_add_ps(v_in1, v_av1);
        v_tv1 = _mm256_sub_ps(v_in1, _mm256_mul_ps(v_K7, v_av1));
        v_tv2 = _mm256_mul_ps(v_K8, _mm256_sub_ps(v_in4, v_in7));
        v_tv2 = CONJ_256_S(SWAP_RI_256_S(v_tv2));
        v_cv5 = _mm256_add_ps(v_tv1, v_tv2);
        v_cv6 = _mm256_sub_ps(v_tv1, v_tv2);

        v_av1 = _mm256_add_ps(v_in5, v_in8);
        v_cv7 = _mm256_add_ps(v_in2, v_av1);
        v_tv1 = _mm256_sub_ps(v_in2, _mm256_mul_ps(v_K7, v_av1));
        v_tv2 = _mm256_mul_ps(v_K8, _mm256_sub_ps(v_in5, v_in8));
        v_tv2 = CONJ_256_S(SWAP_RI_256_S(v_tv2));
        v_cv8 = _mm256_add_ps(v_tv1, v_tv2);
        v_cv9 = _mm256_sub_ps(v_tv1, v_tv2);

        // Output point 1: X[0]
        v_av1 = _mm256_add_ps(v_cv4, v_cv7);
        v_out0 = _mm256_add_ps(v_cv1, v_av1);

        // Output point 4: X[3]
        v_tv1 = _mm256_sub_ps(v_cv1, _mm256_mul_ps(v_K7, v_av1));
        v_tv2 = _mm256_mul_ps(v_K8, _mm256_sub_ps(v_cv4, v_cv7));
        v_tv2 = CONJ_256_S(SWAP_RI_256_S(v_tv2));
        v_out3 = _mm256_add_ps(v_tv1, v_tv2);
        // Output point 7: X[6]
        v_out6 = _mm256_sub_ps(v_tv1, v_tv2);

        v_tv3 = _mm256_mul_ps(v_cv5, v_K6);
        v_tv4 = _mm256_mul_ps(v_cv5, v_K5);
        SUBADD_SWAPA_256_S(v_tv3, v_tv4, v_cv5);
        v_tv3 = _mm256_mul_ps(v_cv8, v_K4);
        v_tv4 = _mm256_mul_ps(v_cv8, v_K3);
        SUBADD_SWAPA_256_S(v_tv3, v_tv4, v_cv8);
        // Output point 2: X[1]
        v_av1 = _mm256_add_ps(v_cv5, v_cv8);
        v_out1 = _mm256_add_ps(v_cv2, v_av1);

        // Output point 5: X[4]
        v_tv1 = _mm256_sub_ps(v_cv2, _mm256_mul_ps(v_K7, v_av1));
        v_tv2 = _mm256_mul_ps(v_K8, _mm256_sub_ps(v_cv5, v_cv8));
        v_tv2 = CONJ_256_S(SWAP_RI_256_S(v_tv2));
        v_out4 = _mm256_add_ps(v_tv1, v_tv2);
        // Output point 8: X[7]
        v_out7 = _mm256_sub_ps(v_tv1, v_tv2);

        v_tv3 = _mm256_mul_ps(v_cv6, v_K4);
        v_tv4 = _mm256_mul_ps(v_cv6, v_K3);
        SUBADD_SWAPA_256_S(v_tv3, v_tv4, v_cv6);
        v_tv3 = _mm256_mul_ps(v_cv9, v_K1);
        v_tv4 = _mm256_mul_ps(v_cv9, v_K2);
        SUBADD_SWAPA_256_S(v_tv3, v_tv4, v_cv9);
        // Output point 3: X[2]
        v_av1 = _mm256_add_ps(v_cv6, v_cv9);
        v_out2 = _mm256_add_ps(v_cv3, v_av1);

        // Output point 6: X[5]
        v_tv1 = _mm256_sub_ps(v_cv3, _mm256_mul_ps(v_K7, v_av1));
        v_tv2 = _mm256_mul_ps(v_K8, _mm256_sub_ps(v_cv6, v_cv9));
        v_tv2 = CONJ_256_S(SWAP_RI_256_S(v_tv2));
        v_out5 = _mm256_add_ps(v_tv1, v_tv2);
        // Output point 9: X[8]
        v_out8 = _mm256_sub_ps(v_tv1, v_tv2);

        SCATTER4_256_S(out_r, v_out_stride, v_out0, is_contiguous_out);
        STORE_OUT_256_S(out_r, out_strides, 1, v_out_stride, v_out1, tw, cols,
                        cnt_256, load_multi_cols, is_contiguous_out);
        STORE_OUT_256_S(out_r, out_strides, 2, v_out_stride, v_out2, tw, cols,
                        cnt_256, load_multi_cols, is_contiguous_out);
        STORE_OUT_256_S(out_r, out_strides, 3, v_out_stride, v_out3, tw, cols,
                        cnt_256, load_multi_cols, is_contiguous_out);
        STORE_OUT_256_S(out_r, out_strides, 4, v_out_stride, v_out4, tw, cols,
                        cnt_256, load_multi_cols, is_contiguous_out);
#if defined(KERNEL_VARIANT_R2C)
        v_out5 = OUT_H2_256_S(v_out5);
        STORE_OUT_256_S(out_h2_r, out_strides, 5, v_out_h2_stride, v_out5, tw,
                        cols, cnt_256, load_multi_cols, 0);
        v_out6 = OUT_H2_256_S(v_out6);
        STORE_OUT_256_S(out_h2_r, out_strides, 6, v_out_h2_stride, v_out6, tw,
                        cols, cnt_256, load_multi_cols, 0);
        v_out7 = OUT_H2_256_S(v_out7);
        STORE_OUT_256_S(out_h2_r, out_strides, 7, v_out_h2_stride, v_out7, tw,
                        cols, cnt_256, load_multi_cols, 0);
        v_out8 = OUT_H2_256_S(v_out8);
        STORE_OUT_256_S(out_h2_r, out_strides, 8, v_out_h2_stride, v_out8, tw,
                        cols, cnt_256, load_multi_cols, 0);
#else
        STORE_OUT_256_S(out_r, out_strides, 5, v_out_stride, v_out5, tw, cols,
                        cnt_256, load_multi_cols, is_contiguous_out);
        STORE_OUT_256_S(out_r, out_strides, 6, v_out_stride, v_out6, tw, cols,
                        cnt_256, load_multi_cols, is_contiguous_out);
        STORE_OUT_256_S(out_r, out_strides, 7, v_out_stride, v_out7, tw, cols,
                        cnt_256, load_multi_cols, is_contiguous_out);
        STORE_OUT_256_S(out_r, out_strides, 8, v_out_stride, v_out8, tw, cols,
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
        __m128 v_in0, v_in1, v_in2, v_in3, v_in4, v_in5, v_in6, v_in7, v_in8;
        __m128 v_out0, v_out1, v_out2, v_out3, v_out4, v_out5, v_out6, v_out7,
            v_out8;
        __m128 v_tv1, v_tv2, v_tv3, v_tv4, v_av1;
        __m128 v_cv1, v_cv2, v_cv3, v_cv4, v_cv5, v_cv6, v_cv7, v_cv8, v_cv9;

#if defined(KERNEL_USE_AVX512)
        __m128 v_K1 = CAST_512_TO_128_S(v_C1);
        __m128 v_K2 = CAST_512_TO_128_S(v_C2);
        __m128 v_K3 = CAST_512_TO_128_S(v_C3);
        __m128 v_K4 = CAST_512_TO_128_S(v_C4);
        __m128 v_K5 = CAST_512_TO_128_S(v_C5);
        __m128 v_K6 = CAST_512_TO_128_S(v_C6);
        __m128 v_K7 = CAST_512_TO_128_S(v_C7);
        __m128 v_K8 = CAST_512_TO_128_S(v_C8);
#elif defined(KERNEL_USE_AVX256)
        __m128 v_K1 = CAST_256_TO_128_S(v_C1);
        __m128 v_K2 = CAST_256_TO_128_S(v_C2);
        __m128 v_K3 = CAST_256_TO_128_S(v_C3);
        __m128 v_K4 = CAST_256_TO_128_S(v_C4);
        __m128 v_K5 = CAST_256_TO_128_S(v_C5);
        __m128 v_K6 = CAST_256_TO_128_S(v_C6);
        __m128 v_K7 = CAST_256_TO_128_S(v_C7);
        __m128 v_K8 = CAST_256_TO_128_S(v_C8);
#endif

        LOAD_IN_128_S(in_r, in_strides, 1, v_in_stride, v_in1, tw, cols,
                      cnt_128, load_multi_cols, is_contiguous_in);
        LOAD_IN_128_S(in_r, in_strides, 2, v_in_stride, v_in2, tw, cols,
                      cnt_128, load_multi_cols, is_contiguous_in);
        LOAD_IN_128_S(in_r, in_strides, 3, v_in_stride, v_in3, tw, cols,
                      cnt_128, load_multi_cols, is_contiguous_in);
        LOAD_IN_128_S(in_r, in_strides, 4, v_in_stride, v_in4, tw, cols,
                      cnt_128, load_multi_cols, is_contiguous_in);
#if defined(KERNEL_VARIANT_C2R)
        LOAD_IN_128_S(in_h2_r, in_strides, 5, v_in_h2_stride, v_in5, tw, cols,
                      cnt_128, load_multi_cols, 0);
        v_in5 = IN_H2_128_S(v_in5);
        LOAD_IN_128_S(in_h2_r, in_strides, 6, v_in_h2_stride, v_in6, tw, cols,
                      cnt_128, load_multi_cols, 0);
        v_in6 = IN_H2_128_S(v_in6);
        LOAD_IN_128_S(in_h2_r, in_strides, 7, v_in_h2_stride, v_in7, tw, cols,
                      cnt_128, load_multi_cols, 0);
        v_in7 = IN_H2_128_S(v_in7);
        LOAD_IN_128_S(in_h2_r, in_strides, 8, v_in_h2_stride, v_in8, tw, cols,
                      cnt_128, load_multi_cols, 0);
        v_in8 = IN_H2_128_S(v_in8);
#else
        LOAD_IN_128_S(in_r, in_strides, 5, v_in_stride, v_in5, tw, cols,
                      cnt_128, load_multi_cols, is_contiguous_in);
        LOAD_IN_128_S(in_r, in_strides, 6, v_in_stride, v_in6, tw, cols,
                      cnt_128, load_multi_cols, is_contiguous_in);
        LOAD_IN_128_S(in_r, in_strides, 7, v_in_stride, v_in7, tw, cols,
                      cnt_128, load_multi_cols, is_contiguous_in);
        LOAD_IN_128_S(in_r, in_strides, 8, v_in_stride, v_in8, tw, cols,
                      cnt_128, load_multi_cols, is_contiguous_in);
#endif
        GATHER2_128_S(in_r, v_in_stride, v_in0, is_contiguous_in);

        // common operations
        v_av1 = _mm_add_ps(v_in3, v_in6);
        v_cv1 = _mm_add_ps(v_in0, v_av1);
        v_tv1 = _mm_sub_ps(v_in0, _mm_mul_ps(v_K7, v_av1));
        v_tv2 = _mm_mul_ps(v_K8, _mm_sub_ps(v_in3, v_in6));
        v_tv2 = CONJ_128_S(SWAP_RI_128_S(v_tv2));
        v_cv2 = _mm_add_ps(v_tv1, v_tv2);
        v_cv3 = _mm_sub_ps(v_tv1, v_tv2);

        v_av1 = _mm_add_ps(v_in4, v_in7);
        v_cv4 = _mm_add_ps(v_in1, v_av1);
        v_tv1 = _mm_sub_ps(v_in1, _mm_mul_ps(v_K7, v_av1));
        v_tv2 = _mm_mul_ps(v_K8, _mm_sub_ps(v_in4, v_in7));
        v_tv2 = CONJ_128_S(SWAP_RI_128_S(v_tv2));
        v_cv5 = _mm_add_ps(v_tv1, v_tv2);
        v_cv6 = _mm_sub_ps(v_tv1, v_tv2);

        v_av1 = _mm_add_ps(v_in5, v_in8);
        v_cv7 = _mm_add_ps(v_in2, v_av1);
        v_tv1 = _mm_sub_ps(v_in2, _mm_mul_ps(v_K7, v_av1));
        v_tv2 = _mm_mul_ps(v_K8, _mm_sub_ps(v_in5, v_in8));
        v_tv2 = CONJ_128_S(SWAP_RI_128_S(v_tv2));
        v_cv8 = _mm_add_ps(v_tv1, v_tv2);
        v_cv9 = _mm_sub_ps(v_tv1, v_tv2);

        v_av1 = _mm_add_ps(v_cv4, v_cv7);

        // Output point 1: X[0]
        v_out0 = _mm_add_ps(v_cv1, v_av1);

        v_tv1 = _mm_sub_ps(v_cv1, _mm_mul_ps(v_K7, v_av1));
        v_tv2 = _mm_mul_ps(v_K8, _mm_sub_ps(v_cv4, v_cv7));
        v_tv2 = CONJ_128_S(SWAP_RI_128_S(v_tv2));

        // Output point 4: X[3]
        v_out3 = _mm_add_ps(v_tv1, v_tv2);
        // Output point 7: X[6]
        v_out6 = _mm_sub_ps(v_tv1, v_tv2);

        v_tv3 = _mm_mul_ps(v_cv5, v_K6);
        v_tv4 = _mm_mul_ps(v_cv5, v_K5);
        SUBADD_SWAPA_128_S(v_tv3, v_tv4, v_cv5);
        v_tv3 = _mm_mul_ps(v_cv8, v_K4);
        v_tv4 = _mm_mul_ps(v_cv8, v_K3);
        SUBADD_SWAPA_128_S(v_tv3, v_tv4, v_cv8);
        v_av1 = _mm_add_ps(v_cv5, v_cv8);

        // Output point 2: X[1]
        v_out1 = _mm_add_ps(v_cv2, v_av1);

        v_tv1 = _mm_sub_ps(v_cv2, _mm_mul_ps(v_K7, v_av1));
        v_tv2 = _mm_mul_ps(v_K8, _mm_sub_ps(v_cv5, v_cv8));
        v_tv2 = CONJ_128_S(SWAP_RI_128_S(v_tv2));

        // Output point 5: X[4]
        v_out4 = _mm_add_ps(v_tv1, v_tv2);
        // Output point 8: X[7]
        v_out7 = _mm_sub_ps(v_tv1, v_tv2);

        v_tv3 = _mm_mul_ps(v_cv6, v_K4);
        v_tv4 = _mm_mul_ps(v_cv6, v_K3);
        SUBADD_SWAPA_128_S(v_tv3, v_tv4, v_cv6);
        v_tv3 = _mm_mul_ps(v_cv9, v_K1);
        v_tv4 = _mm_mul_ps(v_cv9, v_K2);
        SUBADD_SWAPA_128_S(v_tv3, v_tv4, v_cv9);
        v_av1 = _mm_add_ps(v_cv6, v_cv9);

        // Output point 3: X[2]
        v_out2 = _mm_add_ps(v_cv3, v_av1);

        v_tv1 = _mm_sub_ps(v_cv3, _mm_mul_ps(v_K7, v_av1));
        v_tv2 = _mm_mul_ps(v_K8, _mm_sub_ps(v_cv6, v_cv9));
        v_tv2 = CONJ_128_S(SWAP_RI_128_S(v_tv2));

        // Output point 6: X[5]
        v_out5 = _mm_add_ps(v_tv1, v_tv2);
        // Output point 9: X[8]
        v_out8 = _mm_sub_ps(v_tv1, v_tv2);

        SCATTER2_128_S(out_r, v_out_stride, v_out0, is_contiguous_out);
        STORE_OUT_128_S(out_r, out_strides, 1, v_out_stride, v_out1, tw, cols,
                        cnt_128, load_multi_cols, is_contiguous_out);
        STORE_OUT_128_S(out_r, out_strides, 2, v_out_stride, v_out2, tw, cols,
                        cnt_128, load_multi_cols, is_contiguous_out);
        STORE_OUT_128_S(out_r, out_strides, 3, v_out_stride, v_out3, tw, cols,
                        cnt_128, load_multi_cols, is_contiguous_out);
        STORE_OUT_128_S(out_r, out_strides, 4, v_out_stride, v_out4, tw, cols,
                        cnt_128, load_multi_cols, is_contiguous_out);
#if defined(KERNEL_VARIANT_R2C)
        v_out5 = OUT_H2_128_S(v_out5);
        STORE_OUT_128_S(out_h2_r, out_strides, 5, v_out_h2_stride, v_out5, tw,
                        cols, cnt_128, load_multi_cols, 0);
        v_out6 = OUT_H2_128_S(v_out6);
        STORE_OUT_128_S(out_h2_r, out_strides, 6, v_out_h2_stride, v_out6, tw,
                        cols, cnt_128, load_multi_cols, 0);
        v_out7 = OUT_H2_128_S(v_out7);
        STORE_OUT_128_S(out_h2_r, out_strides, 7, v_out_h2_stride, v_out7, tw,
                        cols, cnt_128, load_multi_cols, 0);
        v_out8 = OUT_H2_128_S(v_out8);
        STORE_OUT_128_S(out_h2_r, out_strides, 8, v_out_h2_stride, v_out8, tw,
                        cols, cnt_128, load_multi_cols, 0);
#else
        STORE_OUT_128_S(out_r, out_strides, 5, v_out_stride, v_out5, tw, cols,
                        cnt_128, load_multi_cols, is_contiguous_out);
        STORE_OUT_128_S(out_r, out_strides, 6, v_out_stride, v_out6, tw, cols,
                        cnt_128, load_multi_cols, is_contiguous_out);
        STORE_OUT_128_S(out_r, out_strides, 7, v_out_stride, v_out7, tw, cols,
                        cnt_128, load_multi_cols, is_contiguous_out);
        STORE_OUT_128_S(out_r, out_strides, 8, v_out_stride, v_out8, tw, cols,
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
        __m128 v_in0, v_in1, v_in2, v_in3, v_in4, v_in5, v_in6, v_in7, v_in8;
        __m128 v_out0, v_out1, v_out2, v_out3, v_out4, v_out5, v_out6, v_out7,
            v_out8;
        __m128 v_tv1, v_tv2, v_tv3, v_tv4, v_av1;
        __m128 v_cv1, v_cv2, v_cv3, v_cv4, v_cv5, v_cv6, v_cv7, v_cv8, v_cv9;

#if defined(KERNEL_USE_AVX512)
        __m128 v_K1 = CAST_512_TO_128_S(v_C1);
        __m128 v_K2 = CAST_512_TO_128_S(v_C2);
        __m128 v_K3 = CAST_512_TO_128_S(v_C3);
        __m128 v_K4 = CAST_512_TO_128_S(v_C4);
        __m128 v_K5 = CAST_512_TO_128_S(v_C5);
        __m128 v_K6 = CAST_512_TO_128_S(v_C6);
        __m128 v_K7 = CAST_512_TO_128_S(v_C7);
        __m128 v_K8 = CAST_512_TO_128_S(v_C8);
#elif defined(KERNEL_USE_AVX256)
        __m128 v_K1 = CAST_256_TO_128_S(v_C1);
        __m128 v_K2 = CAST_256_TO_128_S(v_C2);
        __m128 v_K3 = CAST_256_TO_128_S(v_C3);
        __m128 v_K4 = CAST_256_TO_128_S(v_C4);
        __m128 v_K5 = CAST_256_TO_128_S(v_C5);
        __m128 v_K6 = CAST_256_TO_128_S(v_C6);
        __m128 v_K7 = CAST_256_TO_128_S(v_C7);
        __m128 v_K8 = CAST_256_TO_128_S(v_C8);
#elif defined(KERNEL_USE_AVX128)
        __m128 v_K1 = v_C1;
        __m128 v_K2 = v_C2;
        __m128 v_K3 = v_C3;
        __m128 v_K4 = v_C4;
        __m128 v_K5 = v_C5;
        __m128 v_K6 = v_C6;
        __m128 v_K7 = v_C7;
        __m128 v_K8 = v_C8;
#endif

        LOAD_IN_64_S(in_r, in_strides, 1, v_in1, tw, cols, cnt_128_low,
                     load_multi_cols, is_contiguous_in);
        LOAD_IN_64_S(in_r, in_strides, 2, v_in2, tw, cols, cnt_128_low,
                     load_multi_cols, is_contiguous_in);
        LOAD_IN_64_S(in_r, in_strides, 3, v_in3, tw, cols, cnt_128_low,
                     load_multi_cols, is_contiguous_in);
        LOAD_IN_64_S(in_r, in_strides, 4, v_in4, tw, cols, cnt_128_low,
                     load_multi_cols, is_contiguous_in);
#if defined(KERNEL_VARIANT_C2R)
        LOAD_IN_64_S(in_h2_r, in_strides, 5, v_in5, tw, cols, cnt_128_low,
                     load_multi_cols, 0);
        v_in5 = IN_H2_128_S(v_in5);
        LOAD_IN_64_S(in_h2_r, in_strides, 6, v_in6, tw, cols, cnt_128_low,
                     load_multi_cols, 0);
        v_in6 = IN_H2_128_S(v_in6);
        LOAD_IN_64_S(in_h2_r, in_strides, 7, v_in7, tw, cols, cnt_128_low,
                     load_multi_cols, 0);
        v_in7 = IN_H2_128_S(v_in7);
        LOAD_IN_64_S(in_h2_r, in_strides, 8, v_in8, tw, cols, cnt_128_low,
                     load_multi_cols, 0);
        v_in8 = IN_H2_128_S(v_in8);
#else
        LOAD_IN_64_S(in_r, in_strides, 5, v_in5, tw, cols, cnt_128_low,
                     load_multi_cols, is_contiguous_in);
        LOAD_IN_64_S(in_r, in_strides, 6, v_in6, tw, cols, cnt_128_low,
                     load_multi_cols, is_contiguous_in);
        LOAD_IN_64_S(in_r, in_strides, 7, v_in7, tw, cols, cnt_128_low,
                     load_multi_cols, is_contiguous_in);
        LOAD_IN_64_S(in_r, in_strides, 8, v_in8, tw, cols, cnt_128_low,
                     load_multi_cols, is_contiguous_in);
#endif

        LD_LOW_128_S(in_r, v_in0);

        // common operations
        v_av1 = _mm_add_ps(v_in3, v_in6);
        v_cv1 = _mm_add_ps(v_in0, v_av1);
        v_tv1 = _mm_sub_ps(v_in0, _mm_mul_ps(v_K7, v_av1));
        v_tv2 = _mm_mul_ps(v_K8, _mm_sub_ps(v_in3, v_in6));
        v_tv2 = CONJ_128_S(SWAP_RI_128_S(v_tv2));
        v_cv2 = _mm_add_ps(v_tv1, v_tv2);
        v_cv3 = _mm_sub_ps(v_tv1, v_tv2);

        v_av1 = _mm_add_ps(v_in4, v_in7);
        v_cv4 = _mm_add_ps(v_in1, v_av1);
        v_tv1 = _mm_sub_ps(v_in1, _mm_mul_ps(v_K7, v_av1));
        v_tv2 = _mm_mul_ps(v_K8, _mm_sub_ps(v_in4, v_in7));
        v_tv2 = CONJ_128_S(SWAP_RI_128_S(v_tv2));
        v_cv5 = _mm_add_ps(v_tv1, v_tv2);
        v_cv6 = _mm_sub_ps(v_tv1, v_tv2);

        v_av1 = _mm_add_ps(v_in5, v_in8);
        v_cv7 = _mm_add_ps(v_in2, v_av1);
        v_tv1 = _mm_sub_ps(v_in2, _mm_mul_ps(v_K7, v_av1));
        v_tv2 = _mm_mul_ps(v_K8, _mm_sub_ps(v_in5, v_in8));
        v_tv2 = CONJ_128_S(SWAP_RI_128_S(v_tv2));
        v_cv8 = _mm_add_ps(v_tv1, v_tv2);
        v_cv9 = _mm_sub_ps(v_tv1, v_tv2);

        v_av1 = _mm_add_ps(v_cv4, v_cv7);

        // Output point 1: X[0]
        v_out0 = _mm_add_ps(v_cv1, v_av1);

        v_tv1 = _mm_sub_ps(v_cv1, _mm_mul_ps(v_K7, v_av1));
        v_tv2 = _mm_mul_ps(v_K8, _mm_sub_ps(v_cv4, v_cv7));
        v_tv2 = CONJ_128_S(SWAP_RI_128_S(v_tv2));

        // Output point 4: X[3]
        v_out3 = _mm_add_ps(v_tv1, v_tv2);
        // Output point 7: X[6]
        v_out6 = _mm_sub_ps(v_tv1, v_tv2);

        v_tv3 = _mm_mul_ps(v_cv5, v_K6);
        v_tv4 = _mm_mul_ps(v_cv5, v_K5);
        v_tv3 = SWAP_RI_128_S(v_tv3);
        v_cv5 = _mm_addsub_ps(v_tv3, v_tv4);
        v_cv5 = SWAP_RI_128_S(v_cv5);
        v_tv3 = _mm_mul_ps(v_cv8, v_K4);
        v_tv4 = _mm_mul_ps(v_cv8, v_K3);
        v_tv3 = SWAP_RI_128_S(v_tv3);
        v_cv8 = _mm_addsub_ps(v_tv3, v_tv4);
        v_cv8 = SWAP_RI_128_S(v_cv8);
        v_av1 = _mm_add_ps(v_cv5, v_cv8);

        // Output point 2: X[1]
        v_out1 = _mm_add_ps(v_cv2, v_av1);

        v_tv1 = _mm_sub_ps(v_cv2, _mm_mul_ps(v_K7, v_av1));
        v_tv2 = _mm_mul_ps(v_K8, _mm_sub_ps(v_cv5, v_cv8));
        v_tv2 = CONJ_128_S(SWAP_RI_128_S(v_tv2));

        // Output point 5: X[4]
        v_out4 = _mm_add_ps(v_tv1, v_tv2);
        // Output point 8: X[7]
        v_out7 = _mm_sub_ps(v_tv1, v_tv2);

        v_tv3 = _mm_mul_ps(v_cv6, v_K4);
        v_tv4 = _mm_mul_ps(v_cv6, v_K3);
        v_tv3 = SWAP_RI_128_S(v_tv3);
        v_cv6 = _mm_addsub_ps(v_tv3, v_tv4);
        v_cv6 = SWAP_RI_128_S(v_cv6);
        v_tv3 = _mm_mul_ps(v_cv9, v_K1);
        v_tv4 = _mm_mul_ps(v_cv9, v_K2);
        v_tv3 = SWAP_RI_128_S(v_tv3);
        v_cv9 = _mm_addsub_ps(v_tv3, v_tv4);
        v_cv9 = SWAP_RI_128_S(v_cv9);
        v_av1 = _mm_add_ps(v_cv6, v_cv9);

        // Output point 3: X[2]
        v_out2 = _mm_add_ps(v_cv3, v_av1);

        v_tv1 = _mm_sub_ps(v_cv3, _mm_mul_ps(v_K7, v_av1));
        v_tv2 = _mm_mul_ps(v_K8, _mm_sub_ps(v_cv6, v_cv9));
        v_tv2 = CONJ_128_S(SWAP_RI_128_S(v_tv2));

        // Output point 6: X[5]
        v_out5 = _mm_add_ps(v_tv1, v_tv2);
        // Output point 9: X[8]
        v_out8 = _mm_sub_ps(v_tv1, v_tv2);

        ST_LOW_128_S(out_r, v_out0);
        STORE_OUT_64_S(out_r, out_strides, 1, v_out1, tw, cols, cnt_128_low,
                       load_multi_cols, is_contiguous_out);
        STORE_OUT_64_S(out_r, out_strides, 2, v_out2, tw, cols, cnt_128_low,
                       load_multi_cols, is_contiguous_out);
        STORE_OUT_64_S(out_r, out_strides, 3, v_out3, tw, cols, cnt_128_low,
                       load_multi_cols, is_contiguous_out);
        STORE_OUT_64_S(out_r, out_strides, 4, v_out4, tw, cols, cnt_128_low,
                       load_multi_cols, is_contiguous_out);
#if defined(KERNEL_VARIANT_R2C)
        v_out5 = OUT_H2_128_S(v_out5);
        STORE_OUT_64_S(out_h2_r, out_strides, 5, v_out5, tw, cols, cnt_128_low,
                       load_multi_cols, 0);
        v_out6 = OUT_H2_128_S(v_out6);
        STORE_OUT_64_S(out_h2_r, out_strides, 6, v_out6, tw, cols, cnt_128_low,
                       load_multi_cols, 0);
        v_out7 = OUT_H2_128_S(v_out7);
        STORE_OUT_64_S(out_h2_r, out_strides, 7, v_out7, tw, cols, cnt_128_low,
                       load_multi_cols, 0);
        v_out8 = OUT_H2_128_S(v_out8);
        STORE_OUT_64_S(out_h2_r, out_strides, 8, v_out8, tw, cols, cnt_128_low,
                       load_multi_cols, 0);
#else
        STORE_OUT_64_S(out_r, out_strides, 5, v_out5, tw, cols, cnt_128_low,
                       load_multi_cols, is_contiguous_out);
        STORE_OUT_64_S(out_r, out_strides, 6, v_out6, tw, cols, cnt_128_low,
                       load_multi_cols, is_contiguous_out);
        STORE_OUT_64_S(out_r, out_strides, 7, v_out7, tw, cols, cnt_128_low,
                       load_multi_cols, is_contiguous_out);
        STORE_OUT_64_S(out_r, out_strides, 8, v_out8, tw, cols, cnt_128_low,
                       load_multi_cols, is_contiguous_out);
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

    const FFTZ_DOUBLE CRTM_9[8] = {
        -0.939692620785908384054109277324731469936208134,
        0.342020143325668733044099614682259580763083368,
        0.984807753012208059366743024589523013670643252,
        0.173648177666930348851716626769314796000375677,
        0.642787609686539326322643409907263432907559884,
        0.766044443118978035202392650555416673935832457,
        0.500000000000000000000000000000000000000000000,
        0.866025403784438646763723170752936183471402627};

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
    FFTZ_UINTP cols = tws->cols;
    FFTZ_UINTP load_multi_cols = tws->load_multi_cols;

    FFTZ_INTP N = n / NUM_SETS_D;
    FFTZ_INTP count;

#if defined(KERNEL_USE_AVX512)
    FFTZ_INTP remaining_sets = n % NUM_SETS_D;
    FFTZ_INTP do_256_whole = (FFTZ_INTP)(remaining_sets >= NUM_SETS_256_D);
    FFTZ_INTP cnt_256 = load_multi_cols * (N * NUM_SETS_512_D);
    FFTZ_INTP cnt_128 =
        load_multi_cols * (N * NUM_SETS_512_D + do_256_whole * NUM_SETS_256_D);
#elif defined(KERNEL_USE_AVX256)
    FFTZ_INTP remaining_sets = n % NUM_SETS_D;
    FFTZ_INTP cnt_128 = load_multi_cols * (N * NUM_SETS_D);
#elif defined(KERNEL_USE_AVX128)
    // nothing, since double doesn't have any tail cases to process for AVX128
#endif

    VREGTYPE_D v_C1 = BCAST_D(CRTM_9[0]);
    VREGTYPE_D v_C2 = BCAST_D(CRTM_9[1]);
    VREGTYPE_D v_C3 = BCAST_D(CRTM_9[2]);
    VREGTYPE_D v_C4 = BCAST_D(CRTM_9[3]);
    VREGTYPE_D v_C5 = BCAST_D(CRTM_9[4]);
    VREGTYPE_D v_C6 = BCAST_D(CRTM_9[5]);
    VREGTYPE_D v_C7 = BCAST_D(CRTM_9[6]);
    VREGTYPE_D v_C8 = BCAST_D(CRTM_9[7]);

#if defined(KERNEL_DIRECTION_BWD)
    v_C8 = NEG_D(v_C8, 1);
    v_C5 = NEG_D(v_C5, 1);
    v_C3 = NEG_D(v_C3, 1);
    v_C2 = NEG_D(v_C2, 1);
#endif
    for (count = 0; count < N; count++)
    {
        VREGTYPE_D v_in0, v_in1, v_in2, v_in3, v_in4, v_in5, v_in6, v_in7,
            v_in8;
        VREGTYPE_D v_out0, v_out1, v_out2, v_out3, v_out4, v_out5, v_out6,
            v_out7, v_out8;
        VREGTYPE_D v_tv1, v_tv2, v_tv3, v_tv4, v_av1;
        VREGTYPE_D v_cv1, v_cv2, v_cv3, v_cv4, v_cv5, v_cv6, v_cv7, v_cv8,
            v_cv9;
        FFTZ_INTP col = count * load_multi_cols * NUM_SETS_D;

        LOAD_IN_D(in_r, in_strides, 1, v_in_stride, v_in1, tw, cols, col,
                  load_multi_cols, is_contiguous_in);
        LOAD_IN_D(in_r, in_strides, 2, v_in_stride, v_in2, tw, cols, col,
                  load_multi_cols, is_contiguous_in);
        LOAD_IN_D(in_r, in_strides, 3, v_in_stride, v_in3, tw, cols, col,
                  load_multi_cols, is_contiguous_in);
        LOAD_IN_D(in_r, in_strides, 4, v_in_stride, v_in4, tw, cols, col,
                  load_multi_cols, is_contiguous_in);
#if defined(KERNEL_VARIANT_C2R)
        LOAD_IN_D(in_h2_r, in_strides, 5, v_in_h2_stride, v_in5, tw, cols, col,
                  load_multi_cols, 0);
        v_in5 = IN_H2_D(v_in5);
        LOAD_IN_D(in_h2_r, in_strides, 6, v_in_h2_stride, v_in6, tw, cols, col,
                  load_multi_cols, 0);
        v_in6 = IN_H2_D(v_in6);
        LOAD_IN_D(in_h2_r, in_strides, 7, v_in_h2_stride, v_in7, tw, cols, col,
                  load_multi_cols, 0);
        v_in7 = IN_H2_D(v_in7);
        LOAD_IN_D(in_h2_r, in_strides, 8, v_in_h2_stride, v_in8, tw, cols, col,
                  load_multi_cols, 0);
        v_in8 = IN_H2_D(v_in8);
#else
        LOAD_IN_D(in_r, in_strides, 5, v_in_stride, v_in5, tw, cols, col,
                  load_multi_cols, is_contiguous_in);
        LOAD_IN_D(in_r, in_strides, 6, v_in_stride, v_in6, tw, cols, col,
                  load_multi_cols, is_contiguous_in);
        LOAD_IN_D(in_r, in_strides, 7, v_in_stride, v_in7, tw, cols, col,
                  load_multi_cols, is_contiguous_in);
        LOAD_IN_D(in_r, in_strides, 8, v_in_stride, v_in8, tw, cols, col,
                  load_multi_cols, is_contiguous_in);
#endif
        GATHER_D(in_r, v_in_stride, v_in0, is_contiguous_in);

        // common operations
        v_av1 = ADD_D(v_in3, v_in6);
        v_cv1 = ADD_D(v_in0, v_av1);
        v_tv1 = SUB_D(v_in0, MUL_D(v_C7, v_av1));
        v_tv2 = MUL_D(v_C8, SUB_D(v_in3, v_in6));
        v_tv2 = CONJ_D(SWAP_RI_D(v_tv2));
        v_cv2 = ADD_D(v_tv1, v_tv2);
        v_cv3 = SUB_D(v_tv1, v_tv2);

        v_av1 = ADD_D(v_in4, v_in7);
        v_cv4 = ADD_D(v_in1, v_av1);
        v_tv1 = SUB_D(v_in1, MUL_D(v_C7, v_av1));
        v_tv2 = MUL_D(v_C8, SUB_D(v_in4, v_in7));
        v_tv2 = CONJ_D(SWAP_RI_D(v_tv2));
        v_cv5 = ADD_D(v_tv1, v_tv2);
        v_cv6 = SUB_D(v_tv1, v_tv2);

        v_av1 = ADD_D(v_in5, v_in8);
        v_cv7 = ADD_D(v_in2, v_av1);
        v_tv1 = SUB_D(v_in2, MUL_D(v_C7, v_av1));
        v_tv2 = MUL_D(v_C8, SUB_D(v_in5, v_in8));
        v_tv2 = CONJ_D(SWAP_RI_D(v_tv2));
        v_cv8 = ADD_D(v_tv1, v_tv2);
        v_cv9 = SUB_D(v_tv1, v_tv2);

        // Output point 1: X[0]
        v_av1 = ADD_D(v_cv4, v_cv7);
        v_out0 = ADD_D(v_cv1, v_av1);

        v_tv1 = SUB_D(v_cv1, MUL_D(v_C7, v_av1));
        v_tv2 = MUL_D(v_C8, SUB_D(v_cv4, v_cv7));
        v_tv2 = CONJ_D(SWAP_RI_D(v_tv2));

        // Output point 4: X[3]
        v_out3 = ADD_D(v_tv1, v_tv2);
        // Output point 7: X[6]
        v_out6 = SUB_D(v_tv1, v_tv2);

        v_tv3 = MUL_D(v_cv5, v_C6);
        v_tv4 = MUL_D(v_cv5, v_C5);
        SUBADD_SWAPA_D(v_tv3, v_tv4, v_cv5);
        v_tv3 = MUL_D(v_cv8, v_C4);
        v_tv4 = MUL_D(v_cv8, v_C3);
        SUBADD_SWAPA_D(v_tv3, v_tv4, v_cv8);
        v_av1 = ADD_D(v_cv5, v_cv8);

        // Output point 2: X[1]
        v_out1 = ADD_D(v_cv2, v_av1);

        v_tv1 = SUB_D(v_cv2, MUL_D(v_C7, v_av1));
        v_tv2 = MUL_D(v_C8, SUB_D(v_cv5, v_cv8));
        v_tv2 = CONJ_D(SWAP_RI_D(v_tv2));

        // Output point 5: X[4]
        v_out4 = ADD_D(v_tv1, v_tv2);
        // Output point 8: X[7]
        v_out7 = SUB_D(v_tv1, v_tv2);

        v_tv3 = MUL_D(v_cv6, v_C4);
        v_tv4 = MUL_D(v_cv6, v_C3);
        SUBADD_SWAPA_D(v_tv3, v_tv4, v_cv6);
        v_tv3 = MUL_D(v_cv9, v_C1);
        v_tv4 = MUL_D(v_cv9, v_C2);
        SUBADD_SWAPA_D(v_tv3, v_tv4, v_cv9);
        v_av1 = ADD_D(v_cv6, v_cv9);

        // Output point 3: X[2]
        v_out2 = ADD_D(v_cv3, v_av1);

        v_tv1 = SUB_D(v_cv3, MUL_D(v_C7, v_av1));
        v_tv2 = MUL_D(v_C8, SUB_D(v_cv6, v_cv9));
        v_tv2 = CONJ_D(SWAP_RI_D(v_tv2));

        // Output point 6: X[5]
        v_out5 = ADD_D(v_tv1, v_tv2);
        // Output point 9: X[8]
        v_out8 = SUB_D(v_tv1, v_tv2);

        SCATTER_D(out_r, v_out_stride, v_out0, is_contiguous_out);
        STORE_OUT_D(out_r, out_strides, 1, v_out_stride, v_out1, tw, cols, col,
                    load_multi_cols, is_contiguous_out);
        STORE_OUT_D(out_r, out_strides, 2, v_out_stride, v_out2, tw, cols, col,
                    load_multi_cols, is_contiguous_out);
        STORE_OUT_D(out_r, out_strides, 3, v_out_stride, v_out3, tw, cols, col,
                    load_multi_cols, is_contiguous_out);
        STORE_OUT_D(out_r, out_strides, 4, v_out_stride, v_out4, tw, cols, col,
                    load_multi_cols, is_contiguous_out);
#if defined(KERNEL_VARIANT_R2C)
        v_out5 = OUT_H2_D(v_out5);
        STORE_OUT_D(out_h2_r, out_strides, 5, v_out_h2_stride, v_out5, tw, cols,
                    col, load_multi_cols, 0);
        v_out6 = OUT_H2_D(v_out6);
        STORE_OUT_D(out_h2_r, out_strides, 6, v_out_h2_stride, v_out6, tw, cols,
                    col, load_multi_cols, 0);
        v_out7 = OUT_H2_D(v_out7);
        STORE_OUT_D(out_h2_r, out_strides, 7, v_out_h2_stride, v_out7, tw, cols,
                    col, load_multi_cols, 0);
        v_out8 = OUT_H2_D(v_out8);
        STORE_OUT_D(out_h2_r, out_strides, 8, v_out_h2_stride, v_out8, tw, cols,
                    col, load_multi_cols, 0);
#else
        STORE_OUT_D(out_r, out_strides, 5, v_out_stride, v_out5, tw, cols, col,
                    load_multi_cols, is_contiguous_out);
        STORE_OUT_D(out_r, out_strides, 6, v_out_stride, v_out6, tw, cols, col,
                    load_multi_cols, is_contiguous_out);
        STORE_OUT_D(out_r, out_strides, 7, v_out_stride, v_out7, tw, cols, col,
                    load_multi_cols, is_contiguous_out);
        STORE_OUT_D(out_r, out_strides, 8, v_out_stride, v_out8, tw, cols, col,
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
        __m256d v_in0, v_in1, v_in2, v_in3, v_in4, v_in5, v_in6, v_in7, v_in8;
        __m256d v_out0, v_out1, v_out2, v_out3, v_out4, v_out5, v_out6, v_out7,
            v_out8;
        __m256d v_tv1, v_tv2, v_tv3, v_tv4, v_av1;
        __m256d v_cv1, v_cv2, v_cv3, v_cv4, v_cv5, v_cv6, v_cv7, v_cv8, v_cv9;

        __m256d v_K1 = CAST_512_TO_256_D(v_C1);
        __m256d v_K2 = CAST_512_TO_256_D(v_C2);
        __m256d v_K3 = CAST_512_TO_256_D(v_C3);
        __m256d v_K4 = CAST_512_TO_256_D(v_C4);
        __m256d v_K5 = CAST_512_TO_256_D(v_C5);
        __m256d v_K6 = CAST_512_TO_256_D(v_C6);
        __m256d v_K7 = CAST_512_TO_256_D(v_C7);
        __m256d v_K8 = CAST_512_TO_256_D(v_C8);

        LOAD_IN_256_D(in_r, in_strides, 1, v_in_stride, v_in1, tw, cols,
                      cnt_256, load_multi_cols, is_contiguous_in);
        LOAD_IN_256_D(in_r, in_strides, 2, v_in_stride, v_in2, tw, cols,
                      cnt_256, load_multi_cols, is_contiguous_in);
        LOAD_IN_256_D(in_r, in_strides, 3, v_in_stride, v_in3, tw, cols,
                      cnt_256, load_multi_cols, is_contiguous_in);
        LOAD_IN_256_D(in_r, in_strides, 4, v_in_stride, v_in4, tw, cols,
                      cnt_256, load_multi_cols, is_contiguous_in);
#if defined(KERNEL_VARIANT_C2R)
        LOAD_IN_256_D(in_h2_r, in_strides, 5, v_in_h2_stride, v_in5, tw, cols,
                      cnt_256, load_multi_cols, 0);
        v_in5 = IN_H2_256_D(v_in5);
        LOAD_IN_256_D(in_h2_r, in_strides, 6, v_in_h2_stride, v_in6, tw, cols,
                      cnt_256, load_multi_cols, 0);
        v_in6 = IN_H2_256_D(v_in6);
        LOAD_IN_256_D(in_h2_r, in_strides, 7, v_in_h2_stride, v_in7, tw, cols,
                      cnt_256, load_multi_cols, 0);
        v_in7 = IN_H2_256_D(v_in7);
        LOAD_IN_256_D(in_h2_r, in_strides, 8, v_in_h2_stride, v_in8, tw, cols,
                      cnt_256, load_multi_cols, 0);
        v_in8 = IN_H2_256_D(v_in8);
#else
        LOAD_IN_256_D(in_r, in_strides, 5, v_in_stride, v_in5, tw, cols,
                      cnt_256, load_multi_cols, is_contiguous_in);
        LOAD_IN_256_D(in_r, in_strides, 6, v_in_stride, v_in6, tw, cols,
                      cnt_256, load_multi_cols, is_contiguous_in);
        LOAD_IN_256_D(in_r, in_strides, 7, v_in_stride, v_in7, tw, cols,
                      cnt_256, load_multi_cols, is_contiguous_in);
        LOAD_IN_256_D(in_r, in_strides, 8, v_in_stride, v_in8, tw, cols,
                      cnt_256, load_multi_cols, is_contiguous_in);
#endif
        GATHER2_256_D(in_r, v_in_stride, v_in0, is_contiguous_in);

        // common operations
        v_av1 = _mm256_add_pd(v_in3, v_in6);
        v_cv1 = _mm256_add_pd(v_in0, v_av1);
        v_tv1 = _mm256_sub_pd(v_in0, _mm256_mul_pd(v_K7, v_av1));
        v_tv2 = _mm256_mul_pd(v_K8, _mm256_sub_pd(v_in3, v_in6));
        v_tv2 = CONJ_256_D(SWAP_RI_256_D(v_tv2));
        v_cv2 = _mm256_add_pd(v_tv1, v_tv2);
        v_cv3 = _mm256_sub_pd(v_tv1, v_tv2);

        v_av1 = _mm256_add_pd(v_in4, v_in7);
        v_cv4 = _mm256_add_pd(v_in1, v_av1);
        v_tv1 = _mm256_sub_pd(v_in1, _mm256_mul_pd(v_K7, v_av1));
        v_tv2 = _mm256_mul_pd(v_K8, _mm256_sub_pd(v_in4, v_in7));
        v_tv2 = CONJ_256_D(SWAP_RI_256_D(v_tv2));
        v_cv5 = _mm256_add_pd(v_tv1, v_tv2);
        v_cv6 = _mm256_sub_pd(v_tv1, v_tv2);

        v_av1 = _mm256_add_pd(v_in5, v_in8);
        v_cv7 = _mm256_add_pd(v_in2, v_av1);
        v_tv1 = _mm256_sub_pd(v_in2, _mm256_mul_pd(v_K7, v_av1));
        v_tv2 = _mm256_mul_pd(v_K8, _mm256_sub_pd(v_in5, v_in8));
        v_tv2 = CONJ_256_D(SWAP_RI_256_D(v_tv2));
        v_cv8 = _mm256_add_pd(v_tv1, v_tv2);
        v_cv9 = _mm256_sub_pd(v_tv1, v_tv2);

        // Output point 1: X[0]
        v_av1 = _mm256_add_pd(v_cv4, v_cv7);
        v_out0 = _mm256_add_pd(v_cv1, v_av1);

        v_tv1 = _mm256_sub_pd(v_cv1, _mm256_mul_pd(v_K7, v_av1));
        v_tv2 = _mm256_mul_pd(v_K8, _mm256_sub_pd(v_cv4, v_cv7));
        v_tv2 = CONJ_256_D(SWAP_RI_256_D(v_tv2));

        // Output point 4: X[3]
        v_out3 = _mm256_add_pd(v_tv1, v_tv2);
        // Output point 7: X[6]
        v_out6 = _mm256_sub_pd(v_tv1, v_tv2);

        v_tv3 = _mm256_mul_pd(v_cv5, v_K6);
        v_tv4 = _mm256_mul_pd(v_cv5, v_K5);
        SUBADD_SWAPA_256_D(v_tv3, v_tv4, v_cv5);
        v_tv3 = _mm256_mul_pd(v_cv8, v_K4);
        v_tv4 = _mm256_mul_pd(v_cv8, v_K3);
        SUBADD_SWAPA_256_D(v_tv3, v_tv4, v_cv8);
        v_av1 = _mm256_add_pd(v_cv5, v_cv8);

        // Output point 2: X[1]
        v_out1 = _mm256_add_pd(v_cv2, v_av1);

        v_tv1 = _mm256_sub_pd(v_cv2, _mm256_mul_pd(v_K7, v_av1));
        v_tv2 = _mm256_mul_pd(v_K8, _mm256_sub_pd(v_cv5, v_cv8));
        v_tv2 = CONJ_256_D(SWAP_RI_256_D(v_tv2));

        // Output point 5: X[4]
        v_out4 = _mm256_add_pd(v_tv1, v_tv2);
        // Output point 8: X[7]
        v_out7 = _mm256_sub_pd(v_tv1, v_tv2);

        v_tv3 = _mm256_mul_pd(v_cv6, v_K4);
        v_tv4 = _mm256_mul_pd(v_cv6, v_K3);
        SUBADD_SWAPA_256_D(v_tv3, v_tv4, v_cv6);
        v_tv3 = _mm256_mul_pd(v_cv9, v_K1);
        v_tv4 = _mm256_mul_pd(v_cv9, v_K2);
        SUBADD_SWAPA_256_D(v_tv3, v_tv4, v_cv9);
        v_av1 = _mm256_add_pd(v_cv6, v_cv9);

        // Output point 3: X[2]
        v_out2 = _mm256_add_pd(v_cv3, v_av1);

        v_tv1 = _mm256_sub_pd(v_cv3, _mm256_mul_pd(v_K7, v_av1));
        v_tv2 = _mm256_mul_pd(v_K8, _mm256_sub_pd(v_cv6, v_cv9));
        v_tv2 = CONJ_256_D(SWAP_RI_256_D(v_tv2));

        // Output point 6: X[5]
        v_out5 = _mm256_add_pd(v_tv1, v_tv2);
        // Output point 9: X[8]
        v_out8 = _mm256_sub_pd(v_tv1, v_tv2);

        SCATTER2_256_D(out_r, v_out_stride, v_out0, is_contiguous_out);
        STORE_OUT_256_D(out_r, out_strides, 1, v_out_stride, v_out1, tw, cols,
                        cnt_256, load_multi_cols, is_contiguous_out);
        STORE_OUT_256_D(out_r, out_strides, 2, v_out_stride, v_out2, tw, cols,
                        cnt_256, load_multi_cols, is_contiguous_out);
        STORE_OUT_256_D(out_r, out_strides, 3, v_out_stride, v_out3, tw, cols,
                        cnt_256, load_multi_cols, is_contiguous_out);
        STORE_OUT_256_D(out_r, out_strides, 4, v_out_stride, v_out4, tw, cols,
                        cnt_256, load_multi_cols, is_contiguous_out);
#if defined(KERNEL_VARIANT_R2C)
        v_out5 = OUT_H2_256_D(v_out5);
        STORE_OUT_256_D(out_h2_r, out_strides, 5, v_out_h2_stride, v_out5, tw,
                        cols, cnt_256, load_multi_cols, 0);
        v_out6 = OUT_H2_256_D(v_out6);
        STORE_OUT_256_D(out_h2_r, out_strides, 6, v_out_h2_stride, v_out6, tw,
                        cols, cnt_256, load_multi_cols, 0);
        v_out7 = OUT_H2_256_D(v_out7);
        STORE_OUT_256_D(out_h2_r, out_strides, 7, v_out_h2_stride, v_out7, tw,
                        cols, cnt_256, load_multi_cols, 0);
        v_out8 = OUT_H2_256_D(v_out8);
        STORE_OUT_256_D(out_h2_r, out_strides, 8, v_out_h2_stride, v_out8, tw,
                        cols, cnt_256, load_multi_cols, 0);
#else
        STORE_OUT_256_D(out_r, out_strides, 5, v_out_stride, v_out5, tw, cols,
                        cnt_256, load_multi_cols, is_contiguous_out);
        STORE_OUT_256_D(out_r, out_strides, 6, v_out_stride, v_out6, tw, cols,
                        cnt_256, load_multi_cols, is_contiguous_out);
        STORE_OUT_256_D(out_r, out_strides, 7, v_out_stride, v_out7, tw, cols,
                        cnt_256, load_multi_cols, is_contiguous_out);
        STORE_OUT_256_D(out_r, out_strides, 8, v_out_stride, v_out8, tw, cols,
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
        __m128d v_in0, v_in1, v_in2, v_in3, v_in4, v_in5, v_in6, v_in7, v_in8;
        __m128d v_out0, v_out1, v_out2, v_out3, v_out4, v_out5, v_out6, v_out7,
            v_out8;
        __m128d v_tv1, v_tv2, v_tv3, v_tv4, v_av1;
        __m128d v_cv1, v_cv2, v_cv3, v_cv4, v_cv5, v_cv6, v_cv7, v_cv8, v_cv9;

#if defined(KERNEL_USE_AVX512)
        __m128d v_K1 = CAST_512_TO_128_D(v_C1);
        __m128d v_K2 = CAST_512_TO_128_D(v_C2);
        __m128d v_K3 = CAST_512_TO_128_D(v_C3);
        __m128d v_K4 = CAST_512_TO_128_D(v_C4);
        __m128d v_K5 = CAST_512_TO_128_D(v_C5);
        __m128d v_K6 = CAST_512_TO_128_D(v_C6);
        __m128d v_K7 = CAST_512_TO_128_D(v_C7);
        __m128d v_K8 = CAST_512_TO_128_D(v_C8);
#elif defined(KERNEL_USE_AVX256)
        __m128d v_K1 = CAST_256_TO_128_D(v_C1);
        __m128d v_K2 = CAST_256_TO_128_D(v_C2);
        __m128d v_K3 = CAST_256_TO_128_D(v_C3);
        __m128d v_K4 = CAST_256_TO_128_D(v_C4);
        __m128d v_K5 = CAST_256_TO_128_D(v_C5);
        __m128d v_K6 = CAST_256_TO_128_D(v_C6);
        __m128d v_K7 = CAST_256_TO_128_D(v_C7);
        __m128d v_K8 = CAST_256_TO_128_D(v_C8);
#endif

        LOAD_IN_128_D(in_r, in_strides, 1, /* unused */ 0, v_in1, tw, cols,
                      cnt_128, load_multi_cols, is_contiguous_in);
        LOAD_IN_128_D(in_r, in_strides, 2, /* unused */ 0, v_in2, tw, cols,
                      cnt_128, load_multi_cols, is_contiguous_in);
        LOAD_IN_128_D(in_r, in_strides, 3, /* unused */ 0, v_in3, tw, cols,
                      cnt_128, load_multi_cols, is_contiguous_in);
        LOAD_IN_128_D(in_r, in_strides, 4, /* unused */ 0, v_in4, tw, cols,
                      cnt_128, load_multi_cols, is_contiguous_in);
#if defined(KERNEL_VARIANT_C2R)
        LOAD_IN_128_D(in_h2_r, in_strides, 5, /* unused */ 0, v_in5, tw, cols,
                      cnt_128, load_multi_cols, 0);
        v_in5 = IN_H2_128_D(v_in5);
        LOAD_IN_128_D(in_h2_r, in_strides, 6, /* unused */ 0, v_in6, tw, cols,
                      cnt_128, load_multi_cols, 0);
        v_in6 = IN_H2_128_D(v_in6);
        LOAD_IN_128_D(in_h2_r, in_strides, 7, /* unused */ 0, v_in7, tw, cols,
                      cnt_128, load_multi_cols, 0);
        v_in7 = IN_H2_128_D(v_in7);
        LOAD_IN_128_D(in_h2_r, in_strides, 8, /* unused */ 0, v_in8, tw, cols,
                      cnt_128, load_multi_cols, 0);
        v_in8 = IN_H2_128_D(v_in8);
#else
        LOAD_IN_128_D(in_r, in_strides, 5, /* unused */ 0, v_in5, tw, cols,
                      cnt_128, load_multi_cols, is_contiguous_in);
        LOAD_IN_128_D(in_r, in_strides, 6, /* unused */ 0, v_in6, tw, cols,
                      cnt_128, load_multi_cols, is_contiguous_in);
        LOAD_IN_128_D(in_r, in_strides, 7, /* unused */ 0, v_in7, tw, cols,
                      cnt_128, load_multi_cols, is_contiguous_in);
        LOAD_IN_128_D(in_r, in_strides, 8, /* unused */ 0, v_in8, tw, cols,
                      cnt_128, load_multi_cols, is_contiguous_in);
#endif

        LD_128_D(in_r, v_in0);

        // common operations
        v_av1 = _mm_add_pd(v_in3, v_in6);
        v_cv1 = _mm_add_pd(v_in0, v_av1);
        v_tv1 = _mm_sub_pd(v_in0, _mm_mul_pd(v_K7, v_av1));
        v_tv2 = _mm_mul_pd(v_K8, _mm_sub_pd(v_in3, v_in6));
        v_tv2 = CONJ_128_D(SWAP_RI_128_D(v_tv2));
        v_cv2 = _mm_add_pd(v_tv1, v_tv2);
        v_cv3 = _mm_sub_pd(v_tv1, v_tv2);

        v_av1 = _mm_add_pd(v_in4, v_in7);
        v_cv4 = _mm_add_pd(v_in1, v_av1);
        v_tv1 = _mm_sub_pd(v_in1, _mm_mul_pd(v_K7, v_av1));
        v_tv2 = _mm_mul_pd(v_K8, _mm_sub_pd(v_in4, v_in7));
        v_tv2 = CONJ_128_D(SWAP_RI_128_D(v_tv2));
        v_cv5 = _mm_add_pd(v_tv1, v_tv2);
        v_cv6 = _mm_sub_pd(v_tv1, v_tv2);

        v_av1 = _mm_add_pd(v_in5, v_in8);
        v_cv7 = _mm_add_pd(v_in2, v_av1);
        v_tv1 = _mm_sub_pd(v_in2, _mm_mul_pd(v_K7, v_av1));
        v_tv2 = _mm_mul_pd(v_K8, _mm_sub_pd(v_in5, v_in8));
        v_tv2 = CONJ_128_D(SWAP_RI_128_D(v_tv2));
        v_cv8 = _mm_add_pd(v_tv1, v_tv2);
        v_cv9 = _mm_sub_pd(v_tv1, v_tv2);

        v_av1 = _mm_add_pd(v_cv4, v_cv7);

        // Output point 1: X[0]
        v_out0 = _mm_add_pd(v_cv1, v_av1);

        v_tv1 = _mm_sub_pd(v_cv1, _mm_mul_pd(v_K7, v_av1));
        v_tv2 = _mm_mul_pd(v_K8, _mm_sub_pd(v_cv4, v_cv7));
        v_tv2 = CONJ_128_D(SWAP_RI_128_D(v_tv2));

        // Output point 4: X[3]
        v_out3 = _mm_add_pd(v_tv1, v_tv2);
        // Output point 7: X[6]
        v_out6 = _mm_sub_pd(v_tv1, v_tv2);

        v_tv3 = _mm_mul_pd(v_cv5, v_K6);
        v_tv4 = _mm_mul_pd(v_cv5, v_K5);
        v_tv3 = SWAP_RI_128_D(v_tv3);
        v_cv5 = _mm_addsub_pd(v_tv3, v_tv4);
        v_cv5 = SWAP_RI_128_D(v_cv5);
        v_tv3 = _mm_mul_pd(v_cv8, v_K4);
        v_tv4 = _mm_mul_pd(v_cv8, v_K3);
        v_tv3 = SWAP_RI_128_D(v_tv3);
        v_cv8 = _mm_addsub_pd(v_tv3, v_tv4);
        v_cv8 = SWAP_RI_128_D(v_cv8);
        v_av1 = _mm_add_pd(v_cv5, v_cv8);

        // Output point 2: X[1]
        v_out1 = _mm_add_pd(v_cv2, v_av1);

        v_tv1 = _mm_sub_pd(v_cv2, _mm_mul_pd(v_K7, v_av1));
        v_tv2 = _mm_mul_pd(v_K8, _mm_sub_pd(v_cv5, v_cv8));
        v_tv2 = CONJ_128_D(SWAP_RI_128_D(v_tv2));

        // Output point 5: X[4]
        v_out4 = _mm_add_pd(v_tv1, v_tv2);
        // Output point 8: X[7]
        v_out7 = _mm_sub_pd(v_tv1, v_tv2);

        v_tv3 = _mm_mul_pd(v_cv6, v_K4);
        v_tv4 = _mm_mul_pd(v_cv6, v_K3);
        v_tv3 = SWAP_RI_128_D(v_tv3);
        v_cv6 = _mm_addsub_pd(v_tv3, v_tv4);
        v_cv6 = SWAP_RI_128_D(v_cv6);
        v_tv3 = _mm_mul_pd(v_cv9, v_K1);
        v_tv4 = _mm_mul_pd(v_cv9, v_K2);
        v_tv3 = SWAP_RI_128_D(v_tv3);
        v_cv9 = _mm_addsub_pd(v_tv3, v_tv4);
        v_cv9 = SWAP_RI_128_D(v_cv9);
        v_av1 = _mm_add_pd(v_cv6, v_cv9);

        // Output point 3: X[2]
        v_out2 = _mm_add_pd(v_cv3, v_av1);

        v_tv1 = _mm_sub_pd(v_cv3, _mm_mul_pd(v_K7, v_av1));
        v_tv2 = _mm_mul_pd(v_K8, _mm_sub_pd(v_cv6, v_cv9));
        v_tv2 = CONJ_128_D(SWAP_RI_128_D(v_tv2));

        // Output point 6: X[5]
        v_out5 = _mm_add_pd(v_tv1, v_tv2);
        // Output point 9: X[8]
        v_out8 = _mm_sub_pd(v_tv1, v_tv2);

        ST_128_D(out_r, v_out0);
        STORE_OUT_128_D(out_r, out_strides, 1, /* unused */ 0, v_out1, tw, cols,
                        cnt_128, load_multi_cols, is_contiguous_out);
        STORE_OUT_128_D(out_r, out_strides, 2, /* unused */ 0, v_out2, tw, cols,
                        cnt_128, load_multi_cols, is_contiguous_out);
        STORE_OUT_128_D(out_r, out_strides, 3, /* unused */ 0, v_out3, tw, cols,
                        cnt_128, load_multi_cols, is_contiguous_out);
        STORE_OUT_128_D(out_r, out_strides, 4, /* unused */ 0, v_out4, tw, cols,
                        cnt_128, load_multi_cols, is_contiguous_out);
#if defined(KERNEL_VARIANT_R2C)
        v_out5 = OUT_H2_128_D(v_out5);
        STORE_OUT_128_D(out_h2_r, out_strides, 5, /* unused */ 0, v_out5, tw,
                        cols, cnt_128, load_multi_cols, 0);
        v_out6 = OUT_H2_128_D(v_out6);
        STORE_OUT_128_D(out_h2_r, out_strides, 6, /* unused */ 0, v_out6, tw,
                        cols, cnt_128, load_multi_cols, 0);
        v_out7 = OUT_H2_128_D(v_out7);
        STORE_OUT_128_D(out_h2_r, out_strides, 7, /* unused */ 0, v_out7, tw,
                        cols, cnt_128, load_multi_cols, 0);
        v_out8 = OUT_H2_128_D(v_out8);
        STORE_OUT_128_D(out_h2_r, out_strides, 8, /* unused */ 0, v_out8, tw,
                        cols, cnt_128, load_multi_cols, 0);
#else
        STORE_OUT_128_D(out_r, out_strides, 5, /* unused */ 0, v_out5, tw, cols,
                        cnt_128, load_multi_cols, is_contiguous_out);
        STORE_OUT_128_D(out_r, out_strides, 6, /* unused */ 0, v_out6, tw, cols,
                        cnt_128, load_multi_cols, is_contiguous_out);
        STORE_OUT_128_D(out_r, out_strides, 7, /* unused */ 0, v_out7, tw, cols,
                        cnt_128, load_multi_cols, is_contiguous_out);
        STORE_OUT_128_D(out_r, out_strides, 8, /* unused */ 0, v_out8, tw, cols,
                        cnt_128, load_multi_cols, is_contiguous_out);
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
