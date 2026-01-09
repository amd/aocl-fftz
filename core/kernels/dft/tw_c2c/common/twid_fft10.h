// Copyright Advanced Micro Devices, Inc.
// SPDX-License-Identifier: BSD-3-Clause

/** @file twid_fft10.h
 *
 *  @brief The ISA generic kernel template for the radix 10 twiddle kernel
 *
 *  This file contains the DIT twiddle radix-10 FFT implementations for
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

    const FLOAT CRTM_10[4] = {
        0.55901699437494742410229341718281905886015458990288f,
        0.58778525229247315738615484497912915412138427663885f,
        0.25000000000000000000000000000000000000000000000000f,
        0.95105651629515357211643933337938214340569863400000f};

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

    VREGTYPE_S v_C1 = BCAST_S(CRTM_10[0]);
    VREGTYPE_S v_C2 = BCAST_S(CRTM_10[1]);
    VREGTYPE_S v_C3 = BCAST_S(CRTM_10[2]);
    VREGTYPE_S v_C4 = BCAST_S(CRTM_10[3]);

    INTP count;

    v_C2 = NEG_S(v_C2, flag);
    v_C4 = NEG_S(v_C4, flag);

    for (count = 0; count < N; count++)
    {
        VREGTYPE_S v_in0, v_in1, v_in2, v_in3, v_in4, v_in5, v_in6, v_in7,
            v_in8, v_in9;
        VREGTYPE_S v_out0, v_out1, v_out2, v_out3, v_out4, v_out5, v_out6,
            v_out7, v_out8;
        VREGTYPE_S v_out9, v_cv1, v_cv2, v_cv3, v_cv4, v_tv1, v_tv2;
        VREGTYPE_S v_av1, v_av2, v_av3, v_av4, v_av5, v_av6, v_av7, v_av8,
            v_av9;

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
        }

        GATHER_S(curr_in, v_in_stride, v_in0);

        // common operations
        v_av1 = ADD_S(v_in1, v_in9);
        v_av2 = ADD_S(v_in2, v_in8);
        v_av3 = ADD_S(v_in3, v_in7);
        v_av4 = ADD_S(v_in4, v_in6);
        v_av5 = SUB_S(v_in9, v_in1);
        v_av6 = SUB_S(v_in8, v_in2);
        v_av7 = SUB_S(v_in7, v_in3);
        v_av8 = SUB_S(v_in6, v_in4);

        // Output point 6:X[5]
        v_av9 = SUB_S(v_in0, v_in5);
        v_cv1 = SUB_S(v_av2, v_av3);
        v_cv2 = SUB_S(v_av1, v_av4);
        v_cv3 = SUB_S(v_cv1, v_cv2);
        v_out5 = ADD_S(v_av9, v_cv3);

        // Output point 2:X[1]
        v_cv4 = ADD_S(v_cv1, v_cv2);
        v_cv1 = SUB_S(v_av9, MUL_S(v_C3, v_cv3));
        v_tv1 = MUL_S(v_C1, v_cv4);
        v_cv2 = ADD_S(v_cv1, v_tv1);
        v_cv3 = ADD_S(v_av6, v_av7);
        v_cv4 = ADD_S(v_av5, v_av8);
        v_tv2 = ADD_S(MUL_S(v_C4, v_cv3), MUL_S(v_C2, v_cv4));
        v_tv2 = SWAP_RI_S(CONJ_S(v_tv2));
        v_out1 = ADD_S(v_cv2, v_tv2);
        // Output point 10:X[9]
        v_out9 = SUB_S(v_cv2, v_tv2);

        // Output point 4:X[3]
        v_cv2 = SUB_S(v_cv1, v_tv1);
        v_tv2 = SUB_S(MUL_S(v_C4, v_cv4), MUL_S(v_C2, v_cv3));
        v_tv2 = SWAP_RI_S(CONJ_S(v_tv2));
        v_out3 = ADD_S(v_cv2, v_tv2);
        // Output point 8:X[7]
        v_out7 = SUB_S(v_cv2, v_tv2);

        // Output point 1:X[0]
        v_av9 = ADD_S(v_in0, v_in5);
        v_cv1 = ADD_S(v_av2, v_av3);
        v_cv2 = ADD_S(v_av1, v_av4);
        v_cv3 = ADD_S(v_cv2, v_cv1);
        v_out0 = ADD_S(v_av9, v_cv3);

        // Output point 3:X[2]
        v_cv4 = SUB_S(v_cv2, v_cv1);
        v_cv1 = SUB_S(v_av9, MUL_S(v_C3, v_cv3));
        v_tv1 = MUL_S(v_C1, v_cv4);
        v_cv2 = ADD_S(v_cv1, v_tv1);

        v_cv3 = SUB_S(v_av6, v_av7);
        v_cv4 = SUB_S(v_av5, v_av8);
        v_tv2 = ADD_S(MUL_S(v_C4, v_cv4), MUL_S(v_C2, v_cv3));
        v_tv2 = SWAP_RI_S(CONJ_S(v_tv2));
        v_out2 = ADD_S(v_cv2, v_tv2);
        // Output point 9:X[8]
        v_out8 = SUB_S(v_cv2, v_tv2);

        // Output point 7:X[6]
        v_cv2 = SUB_S(v_cv1, v_tv1);
        v_tv2 = SUB_S(MUL_S(v_C4, v_cv3), MUL_S(v_C2, v_cv4));
        v_tv2 = SWAP_RI_S(CONJ_S(v_tv2));
        v_out6 = ADD_S(v_cv2, v_tv2);
        // Output point 5:X[4]
        v_out4 = SUB_S(v_cv2, v_tv2);

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

        in_r += NUM_SETS_S * v_in_stride;
        out_r += NUM_SETS_S * v_out_stride;
    }

    // The following contains code that performs the FFT on the tail cases.
    // These tails are conditionally "instantiated" based on the ISA requested
    // by the "includee" file.

#if defined(KERNEL_USE_AVX512)
    if (remaining_sets >= NUM_SETS_256_S)
    {
        __m256 v_in0, v_in1, v_in2, v_in3, v_in4, v_in5, v_in6, v_in7, v_in8,
            v_in9;
        __m256 v_out0, v_out1, v_out2, v_out3, v_out4, v_out5, v_out6, v_out7,
            v_out8;
        __m256 v_out9, v_cv1, v_cv2, v_cv3, v_cv4, v_tv1, v_tv2;
        __m256 v_av1, v_av2, v_av3, v_av4, v_av5, v_av6, v_av7, v_av8, v_av9;

        __m256 v_K1 = CAST_512_TO_256_S(v_C1);
        __m256 v_K2 = CAST_512_TO_256_S(v_C2);
        __m256 v_K3 = CAST_512_TO_256_S(v_C3);
        __m256 v_K4 = CAST_512_TO_256_S(v_C4);

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
        }

        GATHER4_256_S(curr_in, v_in_stride, v_in0);

        // common operations
        v_av1 = _mm256_add_ps(v_in1, v_in9);
        v_av2 = _mm256_add_ps(v_in2, v_in8);
        v_av3 = _mm256_add_ps(v_in3, v_in7);
        v_av4 = _mm256_add_ps(v_in4, v_in6);
        v_av5 = _mm256_sub_ps(v_in9, v_in1);
        v_av6 = _mm256_sub_ps(v_in8, v_in2);
        v_av7 = _mm256_sub_ps(v_in7, v_in3);
        v_av8 = _mm256_sub_ps(v_in6, v_in4);

        // Output point 6:X[5]
        v_av9 = _mm256_sub_ps(v_in0, v_in5);
        v_cv1 = _mm256_sub_ps(v_av2, v_av3);
        v_cv2 = _mm256_sub_ps(v_av1, v_av4);
        v_cv3 = _mm256_sub_ps(v_cv1, v_cv2);
        v_out5 = _mm256_add_ps(v_av9, v_cv3);

        // Output point 2:X[1]
        v_cv4 = _mm256_add_ps(v_cv1, v_cv2);
        v_cv1 = _mm256_sub_ps(v_av9, _mm256_mul_ps(v_K3, v_cv3));
        v_tv1 = _mm256_mul_ps(v_K1, v_cv4);
        v_cv2 = _mm256_add_ps(v_cv1, v_tv1);
        v_cv3 = _mm256_add_ps(v_av6, v_av7);
        v_cv4 = _mm256_add_ps(v_av5, v_av8);
        v_tv2 = _mm256_add_ps(_mm256_mul_ps(v_K4, v_cv3),
                              _mm256_mul_ps(v_K2, v_cv4));
        v_tv2 = SWAP_RI_256_S(CONJ_256_S(v_tv2));
        v_out1 = _mm256_add_ps(v_cv2, v_tv2);
        // Output point 10:X[9]
        v_out9 = _mm256_sub_ps(v_cv2, v_tv2);

        // Output point 4:X[3]
        v_cv2 = _mm256_sub_ps(v_cv1, v_tv1);
        v_tv2 = _mm256_sub_ps(_mm256_mul_ps(v_K4, v_cv4),
                              _mm256_mul_ps(v_K2, v_cv3));
        v_tv2 = SWAP_RI_256_S(CONJ_256_S(v_tv2));
        v_out3 = _mm256_add_ps(v_cv2, v_tv2);
        // Output point 8:X[7]
        v_out7 = _mm256_sub_ps(v_cv2, v_tv2);

        // Output point 1:X[0]
        v_av9 = _mm256_add_ps(v_in0, v_in5);
        v_cv1 = _mm256_add_ps(v_av2, v_av3);
        v_cv2 = _mm256_add_ps(v_av1, v_av4);
        v_cv3 = _mm256_add_ps(v_cv2, v_cv1);
        v_out0 = _mm256_add_ps(v_av9, v_cv3);

        // Output point 3:X[2]
        v_cv4 = _mm256_sub_ps(v_cv2, v_cv1);
        v_cv1 = _mm256_sub_ps(v_av9, _mm256_mul_ps(v_K3, v_cv3));
        v_tv1 = _mm256_mul_ps(v_K1, v_cv4);
        v_cv2 = _mm256_add_ps(v_cv1, v_tv1);

        v_cv3 = _mm256_sub_ps(v_av6, v_av7);
        v_cv4 = _mm256_sub_ps(v_av5, v_av8);
        v_tv2 = _mm256_add_ps(_mm256_mul_ps(v_K4, v_cv4),
                              _mm256_mul_ps(v_K2, v_cv3));
        v_tv2 = SWAP_RI_256_S(CONJ_256_S(v_tv2));
        v_out2 = _mm256_add_ps(v_cv2, v_tv2);
        // Output point 9:X[8]
        v_out8 = _mm256_sub_ps(v_cv2, v_tv2);

        // Output point 7:X[6]
        v_cv2 = _mm256_sub_ps(v_cv1, v_tv1);
        v_tv2 = _mm256_sub_ps(_mm256_mul_ps(v_K4, v_cv3),
                              _mm256_mul_ps(v_K2, v_cv4));
        v_tv2 = SWAP_RI_256_S(CONJ_256_S(v_tv2));
        v_out6 = _mm256_add_ps(v_cv2, v_tv2);
        // Output point 5:X[4]
        v_out4 = _mm256_sub_ps(v_cv2, v_tv2);

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

        in_r += NUM_SETS_256_S * v_in_stride;
        out_r += NUM_SETS_256_S * v_out_stride;
        remaining_sets = remaining_sets - NUM_SETS_256_S;
    }
#endif

#if defined(KERNEL_USE_AVX512) || defined(KERNEL_USE_AVX256)
    if (remaining_sets >= NUM_SETS_128_S)
    {
        __m128 v_in0, v_in1, v_in2, v_in3, v_in4, v_in5, v_in6, v_in7, v_in8,
            v_in9;
        __m128 v_out0, v_out1, v_out2, v_out3, v_out4, v_out5, v_out6, v_out7,
            v_out8;
        __m128 v_out9, v_cv1, v_cv2, v_cv3, v_cv4, v_tv1, v_tv2;
        __m128 v_av1, v_av2, v_av3, v_av4, v_av5, v_av6, v_av7, v_av8, v_av9;

#if defined(KERNEL_USE_AVX512)
        __m128 v_K1 = CAST_512_TO_128_S(v_C1);
        __m128 v_K2 = CAST_512_TO_128_S(v_C2);
        __m128 v_K3 = CAST_512_TO_128_S(v_C3);
        __m128 v_K4 = CAST_512_TO_128_S(v_C4);
#elif defined(KERNEL_USE_AVX256)
        __m128 v_K1 = CAST_256_TO_128_S(v_C1);
        __m128 v_K2 = CAST_256_TO_128_S(v_C2);
        __m128 v_K3 = CAST_256_TO_128_S(v_C3);
        __m128 v_K4 = CAST_256_TO_128_S(v_C4);
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
        }

        GATHER2_128_S(curr_in, v_in_stride, v_in0);

        // common operations
        v_av1 = _mm_add_ps(v_in1, v_in9);
        v_av2 = _mm_add_ps(v_in2, v_in8);
        v_av3 = _mm_add_ps(v_in3, v_in7);
        v_av4 = _mm_add_ps(v_in4, v_in6);
        v_av5 = _mm_sub_ps(v_in9, v_in1);
        v_av6 = _mm_sub_ps(v_in8, v_in2);
        v_av7 = _mm_sub_ps(v_in7, v_in3);
        v_av8 = _mm_sub_ps(v_in6, v_in4);
        // Output point 6:X[5]
        v_av9 = _mm_sub_ps(v_in0, v_in5);
        v_cv1 = _mm_sub_ps(v_av2, v_av3);
        v_cv2 = _mm_sub_ps(v_av1, v_av4);
        v_cv3 = _mm_sub_ps(v_cv1, v_cv2);
        v_out5 = _mm_add_ps(v_av9, v_cv3);

        // Output point 2:X[1]
        v_cv4 = _mm_add_ps(v_cv1, v_cv2);
        v_cv1 = _mm_sub_ps(v_av9, _mm_mul_ps(v_K3, v_cv3));
        v_tv1 = _mm_mul_ps(v_K1, v_cv4);
        v_cv2 = _mm_add_ps(v_cv1, v_tv1);
        v_cv3 = _mm_add_ps(v_av6, v_av7);
        v_cv4 = _mm_add_ps(v_av5, v_av8);
        v_tv2 = _mm_add_ps(_mm_mul_ps(v_K4, v_cv3), _mm_mul_ps(v_K2, v_cv4));
        v_tv2 = SWAP_RI_128_S(CONJ_128_S(v_tv2));
        v_out1 = _mm_add_ps(v_cv2, v_tv2);
        // Output point 10:X[9]
        v_out9 = _mm_sub_ps(v_cv2, v_tv2);

        // Output point 4:X[3]
        v_cv2 = _mm_sub_ps(v_cv1, v_tv1);
        v_tv2 = _mm_sub_ps(_mm_mul_ps(v_K4, v_cv4), _mm_mul_ps(v_K2, v_cv3));
        v_tv2 = SWAP_RI_128_S(CONJ_128_S(v_tv2));
        v_out3 = _mm_add_ps(v_cv2, v_tv2);
        // Output point 8:X[7]
        v_out7 = _mm_sub_ps(v_cv2, v_tv2);

        // Output point 1:X[0]
        v_av9 = _mm_add_ps(v_in0, v_in5);
        v_cv1 = _mm_add_ps(v_av2, v_av3);
        v_cv2 = _mm_add_ps(v_av1, v_av4);
        v_cv3 = _mm_add_ps(v_cv2, v_cv1);
        v_out0 = _mm_add_ps(v_av9, v_cv3);

        // Output point 3:X[2]
        v_cv4 = _mm_sub_ps(v_cv2, v_cv1);
        v_cv1 = _mm_sub_ps(v_av9, _mm_mul_ps(v_K3, v_cv3));
        v_tv1 = _mm_mul_ps(v_K1, v_cv4);
        v_cv2 = _mm_add_ps(v_cv1, v_tv1);

        v_cv3 = _mm_sub_ps(v_av6, v_av7);
        v_cv4 = _mm_sub_ps(v_av5, v_av8);
        v_tv2 = _mm_add_ps(_mm_mul_ps(v_K4, v_cv4), _mm_mul_ps(v_K2, v_cv3));
        v_tv2 = SWAP_RI_128_S(CONJ_128_S(v_tv2));
        v_out2 = _mm_add_ps(v_cv2, v_tv2);
        // Output point 9:X[8]
        v_out8 = _mm_sub_ps(v_cv2, v_tv2);

        // Output point 7:X[6]
        v_cv2 = _mm_sub_ps(v_cv1, v_tv1);
        v_tv2 = _mm_sub_ps(_mm_mul_ps(v_K4, v_cv3), _mm_mul_ps(v_K2, v_cv4));
        v_tv2 = SWAP_RI_128_S(CONJ_128_S(v_tv2));
        v_out6 = _mm_add_ps(v_cv2, v_tv2);
        // Output point 5:X[4]
        v_out4 = _mm_sub_ps(v_cv2, v_tv2);

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

        in_r = in_r + (v_in_stride << 1);
        out_r = out_r + (v_out_stride << 1);
        remaining_sets = remaining_sets - NUM_SETS_128_S;
    }
#endif

#if defined(KERNEL_USE_AVX512) || defined(KERNEL_USE_AVX256) || defined(KERNEL_USE_AVX128)
    if (remaining_sets & 1)
    {
        __m128 v_in0, v_in1, v_in2, v_in3, v_in4, v_in5, v_in6, v_in7, v_in8,
            v_in9;
        __m128 v_out0, v_out1, v_out2, v_out3, v_out4, v_out5, v_out6, v_out7,
            v_out8;
        __m128 v_out9, v_cv1, v_cv2, v_cv3, v_cv4, v_tv1, v_tv2;
        __m128 v_av1, v_av2, v_av3, v_av4, v_av5, v_av6, v_av7, v_av8, v_av9;

#if defined(KERNEL_USE_AVX512)
        __m128 v_K1 = CAST_512_TO_128_S(v_C1);
        __m128 v_K2 = CAST_512_TO_128_S(v_C2);
        __m128 v_K3 = CAST_512_TO_128_S(v_C3);
        __m128 v_K4 = CAST_512_TO_128_S(v_C4);
#elif defined(KERNEL_USE_AVX256)
        __m128 v_K1 = CAST_256_TO_128_S(v_C1);
        __m128 v_K2 = CAST_256_TO_128_S(v_C2);
        __m128 v_K3 = CAST_256_TO_128_S(v_C3);
        __m128 v_K4 = CAST_256_TO_128_S(v_C4);
#elif defined(KERNEL_USE_AVX128)
        __m128 v_K1 = v_C1;
        __m128 v_K2 = v_C2;
        __m128 v_K3 = v_C3;
        __m128 v_K4 = v_C4;
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
        }

        LD_LOW_128_S(curr_in, v_in0);

        // common operations
        v_av1 = _mm_add_ps(v_in1, v_in9);
        v_av2 = _mm_add_ps(v_in2, v_in8);
        v_av3 = _mm_add_ps(v_in3, v_in7);
        v_av4 = _mm_add_ps(v_in4, v_in6);
        v_av5 = _mm_sub_ps(v_in9, v_in1);
        v_av6 = _mm_sub_ps(v_in8, v_in2);
        v_av7 = _mm_sub_ps(v_in7, v_in3);
        v_av8 = _mm_sub_ps(v_in6, v_in4);
        // Output point 6:X[5]
        v_av9 = _mm_sub_ps(v_in0, v_in5);
        v_cv1 = _mm_sub_ps(v_av2, v_av3);
        v_cv2 = _mm_sub_ps(v_av1, v_av4);
        v_cv3 = _mm_sub_ps(v_cv1, v_cv2);
        v_out5 = _mm_add_ps(v_av9, v_cv3);

        // Output point 2:X[1]
        v_cv4 = _mm_add_ps(v_cv1, v_cv2);
        v_cv1 = _mm_sub_ps(v_av9, _mm_mul_ps(v_K3, v_cv3));
        v_tv1 = _mm_mul_ps(v_K1, v_cv4);
        v_cv2 = _mm_add_ps(v_cv1, v_tv1);
        v_cv3 = _mm_add_ps(v_av6, v_av7);
        v_cv4 = _mm_add_ps(v_av5, v_av8);
        v_tv2 = _mm_add_ps(_mm_mul_ps(v_K4, v_cv3), _mm_mul_ps(v_K2, v_cv4));
        v_tv2 = SWAP_RI_128_S(CONJ_128_S(v_tv2));
        v_out1 = _mm_add_ps(v_cv2, v_tv2);
        // Output point 10:X[9]
        v_out9 = _mm_sub_ps(v_cv2, v_tv2);

        // Output point 4:X[3]
        v_cv2 = _mm_sub_ps(v_cv1, v_tv1);
        v_tv2 = _mm_sub_ps(_mm_mul_ps(v_K4, v_cv4), _mm_mul_ps(v_K2, v_cv3));
        v_tv2 = SWAP_RI_128_S(CONJ_128_S(v_tv2));
        v_out3 = _mm_add_ps(v_cv2, v_tv2);
        // Output point 8:X[7]
        v_out7 = _mm_sub_ps(v_cv2, v_tv2);

        // Output point 1:X[0]
        v_av9 = _mm_add_ps(v_in0, v_in5);
        v_cv1 = _mm_add_ps(v_av2, v_av3);
        v_cv2 = _mm_add_ps(v_av1, v_av4);
        v_cv3 = _mm_add_ps(v_cv2, v_cv1);
        v_out0 = _mm_add_ps(v_av9, v_cv3);

        // Output point 3:X[2]
        v_cv4 = _mm_sub_ps(v_cv2, v_cv1);
        v_cv1 = _mm_sub_ps(v_av9, _mm_mul_ps(v_K3, v_cv3));
        v_tv1 = _mm_mul_ps(v_K1, v_cv4);
        v_cv2 = _mm_add_ps(v_cv1, v_tv1);

        v_cv3 = _mm_sub_ps(v_av6, v_av7);
        v_cv4 = _mm_sub_ps(v_av5, v_av8);
        v_tv2 = _mm_add_ps(_mm_mul_ps(v_K4, v_cv4), _mm_mul_ps(v_K2, v_cv3));
        v_tv2 = SWAP_RI_128_S(CONJ_128_S(v_tv2));
        v_out2 = _mm_add_ps(v_cv2, v_tv2);
        // Output point 9:X[8]
        v_out8 = _mm_sub_ps(v_cv2, v_tv2);

        // Output point 7:X[6]
        v_cv2 = _mm_sub_ps(v_cv1, v_tv1);
        v_tv2 = _mm_sub_ps(_mm_mul_ps(v_K4, v_cv3), _mm_mul_ps(v_K2, v_cv4));
        v_tv2 = SWAP_RI_128_S(CONJ_128_S(v_tv2));
        v_out6 = _mm_add_ps(v_cv2, v_tv2);
        // Output point 5:X[4]
        v_out4 = _mm_sub_ps(v_cv2, v_tv2);

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
    }
#endif

    AOCLFFTZ_LOG(DEBUG, global_logger_mode, "Exit");
}

static VOID TWID_KNAME_FP64(VOID *in_real, VOID *in_imag, VOID *out_real,
                            VOID *out_imag, INTP n, aoclfftz_strides_t *strides,
                            VOID *twd, UINT8 flag)
{
    AOCLFFTZ_LOG(DEBUG, global_logger_mode, "Enter");

    const DOUBLE CRTM_10[4] = {
        0.55901699437494742410229341718281905886015458990288,
        0.58778525229247315738615484497912915412138427663885,
        0.25000000000000000000000000000000000000000000000000,
        0.95105651629515357211643933337938214340569863400000};

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

    VREGTYPE_D v_C1 = BCAST_D(CRTM_10[0]);
    VREGTYPE_D v_C2 = BCAST_D(CRTM_10[1]);
    VREGTYPE_D v_C3 = BCAST_D(CRTM_10[2]);
    VREGTYPE_D v_C4 = BCAST_D(CRTM_10[3]);

    v_C2 = NEG_D(v_C2, flag);
    v_C4 = NEG_D(v_C4, flag);

    for (count = 0; count < N; count++)
    {
        VREGTYPE_D v_in0, v_in1, v_in2, v_in3, v_in4, v_in5, v_in6, v_in7,
            v_in8, v_in9;
        VREGTYPE_D v_out0, v_out1, v_out2, v_out3, v_out4, v_out5, v_out6,
            v_out7, v_out8;
        VREGTYPE_D v_out9, v_cv1, v_cv2, v_cv3, v_cv4, v_tv1, v_tv2;
        VREGTYPE_D v_av1, v_av2, v_av3, v_av4, v_av5, v_av6, v_av7, v_av8,
            v_av9;

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
        }

        GATHER_D(curr_in, v_in_stride, v_in0);

        // common operations
        v_av1 = ADD_D(v_in1, v_in9);
        v_av2 = ADD_D(v_in2, v_in8);
        v_av3 = ADD_D(v_in3, v_in7);
        v_av4 = ADD_D(v_in4, v_in6);
        v_av5 = SUB_D(v_in9, v_in1);
        v_av6 = SUB_D(v_in8, v_in2);
        v_av7 = SUB_D(v_in7, v_in3);
        v_av8 = SUB_D(v_in6, v_in4);

        // Output point 6:X[5]
        v_av9 = SUB_D(v_in0, v_in5);
        v_cv1 = SUB_D(v_av2, v_av3);
        v_cv2 = SUB_D(v_av1, v_av4);
        v_cv3 = SUB_D(v_cv1, v_cv2);
        v_out5 = ADD_D(v_av9, v_cv3);

        // Output point 2:X[1]
        v_cv4 = ADD_D(v_cv1, v_cv2);
        v_cv1 = SUB_D(v_av9, MUL_D(v_C3, v_cv3));
        v_tv1 = MUL_D(v_C1, v_cv4);
        v_cv2 = ADD_D(v_cv1, v_tv1);
        v_cv3 = ADD_D(v_av6, v_av7);
        v_cv4 = ADD_D(v_av5, v_av8);
        v_tv2 = ADD_D(MUL_D(v_C4, v_cv3), MUL_D(v_C2, v_cv4));
        v_tv2 = SWAP_RI_D(CONJ_D(v_tv2));
        v_out1 = ADD_D(v_cv2, v_tv2);
        // Output point 10:X[9]
        v_out9 = SUB_D(v_cv2, v_tv2);

        // Output point 4:X[3]
        v_cv2 = SUB_D(v_cv1, v_tv1);
        v_tv2 = SUB_D(MUL_D(v_C4, v_cv4), MUL_D(v_C2, v_cv3));
        v_tv2 = SWAP_RI_D(CONJ_D(v_tv2));
        v_out3 = ADD_D(v_cv2, v_tv2);
        // Output point 8:X[7]
        v_out7 = SUB_D(v_cv2, v_tv2);

        // Output point 1:X[0]
        v_av9 = ADD_D(v_in0, v_in5);
        v_cv1 = ADD_D(v_av2, v_av3);
        v_cv2 = ADD_D(v_av1, v_av4);
        v_cv3 = ADD_D(v_cv2, v_cv1);
        v_out0 = ADD_D(v_av9, v_cv3);

        // Output point 3:X[2]
        v_cv4 = SUB_D(v_cv2, v_cv1);
        v_cv1 = SUB_D(v_av9, MUL_D(v_C3, v_cv3));
        v_tv1 = MUL_D(v_C1, v_cv4);
        v_cv2 = ADD_D(v_cv1, v_tv1);

        v_cv3 = SUB_D(v_av6, v_av7);
        v_cv4 = SUB_D(v_av5, v_av8);
        v_tv2 = ADD_D(MUL_D(v_C4, v_cv4), MUL_D(v_C2, v_cv3));
        v_tv2 = SWAP_RI_D(CONJ_D(v_tv2));
        v_out2 = ADD_D(v_cv2, v_tv2);
        // Output point 9:X[8]
        v_out8 = SUB_D(v_cv2, v_tv2);

        // Output point 7:X[6]
        v_cv2 = SUB_D(v_cv1, v_tv1);
        v_tv2 = SUB_D(MUL_D(v_C4, v_cv3), MUL_D(v_C2, v_cv4));
        v_tv2 = SWAP_RI_D(CONJ_D(v_tv2));
        v_out6 = ADD_D(v_cv2, v_tv2);
        // Output point 5:X[4]
        v_out4 = SUB_D(v_cv2, v_tv2);

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

        in_r += NUM_SETS_D * v_in_stride;
        out_r += NUM_SETS_D * v_out_stride;
    }

    // The following contains code that performs the FFT on the tail cases.
    // These tails are conditionally "instantiated" based on the ISA requested
    // by the "includee" file.

#if defined(KERNEL_USE_AVX512)
    if (remaining_sets >= NUM_SETS_256_D)
    {
        __m256d v_in0, v_in1, v_in2, v_in3, v_in4, v_in5, v_in6, v_in7, v_in8,
            v_in9;
        __m256d v_out0, v_out1, v_out2, v_out3, v_out4, v_out5, v_out6, v_out7,
            v_out8;
        __m256d v_out9, v_cv1, v_cv2, v_cv3, v_cv4, v_tv1, v_tv2;
        __m256d v_av1, v_av2, v_av3, v_av4, v_av5, v_av6, v_av7, v_av8, v_av9;

        __m256d v_K1 = CAST_512_TO_256_D(v_C1);
        __m256d v_K2 = CAST_512_TO_256_D(v_C2);
        __m256d v_K3 = CAST_512_TO_256_D(v_C3);
        __m256d v_K4 = CAST_512_TO_256_D(v_C4);

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
        }

        GATHER2_256_D(curr_in, v_in_stride, v_in0);

        // common operations
        v_av1 = _mm256_add_pd(v_in1, v_in9);
        v_av2 = _mm256_add_pd(v_in2, v_in8);
        v_av3 = _mm256_add_pd(v_in3, v_in7);
        v_av4 = _mm256_add_pd(v_in4, v_in6);
        v_av5 = _mm256_sub_pd(v_in9, v_in1);
        v_av6 = _mm256_sub_pd(v_in8, v_in2);
        v_av7 = _mm256_sub_pd(v_in7, v_in3);
        v_av8 = _mm256_sub_pd(v_in6, v_in4);

        // Output point 6:X[5]
        v_av9 = _mm256_sub_pd(v_in0, v_in5);
        v_cv1 = _mm256_sub_pd(v_av2, v_av3);
        v_cv2 = _mm256_sub_pd(v_av1, v_av4);
        v_cv3 = _mm256_sub_pd(v_cv1, v_cv2);
        v_out5 = _mm256_add_pd(v_av9, v_cv3);

        // Output point 2:X[1]
        v_cv4 = _mm256_add_pd(v_cv1, v_cv2);
        v_cv1 = _mm256_sub_pd(v_av9, _mm256_mul_pd(v_K3, v_cv3));
        v_tv1 = _mm256_mul_pd(v_K1, v_cv4);
        v_cv2 = _mm256_add_pd(v_cv1, v_tv1);
        v_cv3 = _mm256_add_pd(v_av6, v_av7);
        v_cv4 = _mm256_add_pd(v_av5, v_av8);
        v_tv2 = _mm256_add_pd(_mm256_mul_pd(v_K4, v_cv3),
                              _mm256_mul_pd(v_K2, v_cv4));
        v_tv2 = SWAP_RI_256_D(CONJ_256_D(v_tv2));
        v_out1 = _mm256_add_pd(v_cv2, v_tv2);
        // Output point 10:X[9]
        v_out9 = _mm256_sub_pd(v_cv2, v_tv2);

        // Output point 4:X[3]
        v_cv2 = _mm256_sub_pd(v_cv1, v_tv1);
        v_tv2 = _mm256_sub_pd(_mm256_mul_pd(v_K4, v_cv4),
                              _mm256_mul_pd(v_K2, v_cv3));
        v_tv2 = SWAP_RI_256_D(CONJ_256_D(v_tv2));
        v_out3 = _mm256_add_pd(v_cv2, v_tv2);
        // Output point 8:X[7]
        v_out7 = _mm256_sub_pd(v_cv2, v_tv2);

        // Output point 1:X[0]
        v_av9 = _mm256_add_pd(v_in0, v_in5);
        v_cv1 = _mm256_add_pd(v_av2, v_av3);
        v_cv2 = _mm256_add_pd(v_av1, v_av4);
        v_cv3 = _mm256_add_pd(v_cv2, v_cv1);
        v_out0 = _mm256_add_pd(v_av9, v_cv3);

        // Output point 3:X[2]
        v_cv4 = _mm256_sub_pd(v_cv2, v_cv1);
        v_cv1 = _mm256_sub_pd(v_av9, _mm256_mul_pd(v_K3, v_cv3));
        v_tv1 = _mm256_mul_pd(v_K1, v_cv4);
        v_cv2 = _mm256_add_pd(v_cv1, v_tv1);

        v_cv3 = _mm256_sub_pd(v_av6, v_av7);
        v_cv4 = _mm256_sub_pd(v_av5, v_av8);
        v_tv2 = _mm256_add_pd(_mm256_mul_pd(v_K4, v_cv4),
                              _mm256_mul_pd(v_K2, v_cv3));
        v_tv2 = SWAP_RI_256_D(CONJ_256_D(v_tv2));
        v_out2 = _mm256_add_pd(v_cv2, v_tv2);
        // Output point 9:X[8]
        v_out8 = _mm256_sub_pd(v_cv2, v_tv2);

        // Output point 7:X[6]
        v_cv2 = _mm256_sub_pd(v_cv1, v_tv1);
        v_tv2 = _mm256_sub_pd(_mm256_mul_pd(v_K4, v_cv3),
                              _mm256_mul_pd(v_K2, v_cv4));
        v_tv2 = SWAP_RI_256_D(CONJ_256_D(v_tv2));
        v_out6 = _mm256_add_pd(v_cv2, v_tv2);
        // Output point 5:X[4]
        v_out4 = _mm256_sub_pd(v_cv2, v_tv2);

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

        in_r += NUM_SETS_256_D * v_in_stride;
        out_r += NUM_SETS_256_D * v_out_stride;
    }
#endif

#if defined(KERNEL_USE_AVX512) || defined(KERNEL_USE_AVX256)
    if (remaining_sets & 1)
    {
        __m128d v_in0, v_in1, v_in2, v_in3, v_in4, v_in5, v_in6, v_in7, v_in8,
            v_in9;
        __m128d v_out0, v_out1, v_out2, v_out3, v_out4, v_out5, v_out6, v_out7,
            v_out8;
        __m128d v_out9, v_cv1, v_cv2, v_cv3, v_cv4, v_tv1, v_tv2;
        __m128d v_av1, v_av2, v_av3, v_av4, v_av5, v_av6, v_av7, v_av8, v_av9;

        curr_in = in_r;
        curr_out = out_r;

#if defined(KERNEL_USE_AVX512)
        __m128d v_K1 = CAST_512_TO_128_D(v_C1);
        __m128d v_K2 = CAST_512_TO_128_D(v_C2);
        __m128d v_K3 = CAST_512_TO_128_D(v_C3);
        __m128d v_K4 = CAST_512_TO_128_D(v_C4);
#elif defined(KERNEL_USE_AVX256)
        __m128d v_K1 = CAST_256_TO_128_D(v_C1);
        __m128d v_K2 = CAST_256_TO_128_D(v_C2);
        __m128d v_K3 = CAST_256_TO_128_D(v_C3);
        __m128d v_K4 = CAST_256_TO_128_D(v_C4);
#endif

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
        }

        LD_128_D(curr_in, v_in0);

        // common operations
        v_av1 = _mm_add_pd(v_in1, v_in9);
        v_av2 = _mm_add_pd(v_in2, v_in8);
        v_av3 = _mm_add_pd(v_in3, v_in7);
        v_av4 = _mm_add_pd(v_in4, v_in6);
        v_av5 = _mm_sub_pd(v_in9, v_in1);
        v_av6 = _mm_sub_pd(v_in8, v_in2);
        v_av7 = _mm_sub_pd(v_in7, v_in3);
        v_av8 = _mm_sub_pd(v_in6, v_in4);
        // Output point 6:X[5]
        v_av9 = _mm_sub_pd(v_in0, v_in5);
        v_cv1 = _mm_sub_pd(v_av2, v_av3);
        v_cv2 = _mm_sub_pd(v_av1, v_av4);
        v_cv3 = _mm_sub_pd(v_cv1, v_cv2);
        v_out5 = _mm_add_pd(v_av9, v_cv3);

        // Output point 2:X[1]
        v_cv4 = _mm_add_pd(v_cv1, v_cv2);
        v_cv1 = _mm_sub_pd(v_av9, _mm_mul_pd(v_K3, v_cv3));
        v_tv1 = _mm_mul_pd(v_K1, v_cv4);
        v_cv2 = _mm_add_pd(v_cv1, v_tv1);
        v_cv3 = _mm_add_pd(v_av6, v_av7);
        v_cv4 = _mm_add_pd(v_av5, v_av8);
        v_tv2 = _mm_add_pd(_mm_mul_pd(v_K4, v_cv3), _mm_mul_pd(v_K2, v_cv4));
        v_tv2 = SWAP_RI_128_D(CONJ_128_D(v_tv2));
        v_out1 = _mm_add_pd(v_cv2, v_tv2);
        // Output point 10:X[9]
        v_out9 = _mm_sub_pd(v_cv2, v_tv2);

        // Output point 4:X[3]
        v_cv2 = _mm_sub_pd(v_cv1, v_tv1);
        v_tv2 = _mm_sub_pd(_mm_mul_pd(v_K4, v_cv4), _mm_mul_pd(v_K2, v_cv3));
        v_tv2 = SWAP_RI_128_D(CONJ_128_D(v_tv2));
        v_out3 = _mm_add_pd(v_cv2, v_tv2);
        // Output point 8:X[7]
        v_out7 = _mm_sub_pd(v_cv2, v_tv2);

        // Output point 1:X[0]
        v_av9 = _mm_add_pd(v_in0, v_in5);
        v_cv1 = _mm_add_pd(v_av2, v_av3);
        v_cv2 = _mm_add_pd(v_av1, v_av4);
        v_cv3 = _mm_add_pd(v_cv2, v_cv1);
        v_out0 = _mm_add_pd(v_av9, v_cv3);

        // Output point 3:X[2]
        v_cv4 = _mm_sub_pd(v_cv2, v_cv1);
        v_cv1 = _mm_sub_pd(v_av9, _mm_mul_pd(v_K3, v_cv3));
        v_tv1 = _mm_mul_pd(v_K1, v_cv4);
        v_cv2 = _mm_add_pd(v_cv1, v_tv1);

        v_cv3 = _mm_sub_pd(v_av6, v_av7);
        v_cv4 = _mm_sub_pd(v_av5, v_av8);
        v_tv2 = _mm_add_pd(_mm_mul_pd(v_K4, v_cv4), _mm_mul_pd(v_K2, v_cv3));
        v_tv2 = SWAP_RI_128_D(CONJ_128_D(v_tv2));
        v_out2 = _mm_add_pd(v_cv2, v_tv2);
        // Output point 9:X[8]
        v_out8 = _mm_sub_pd(v_cv2, v_tv2);

        // Output point 7:X[6]
        v_cv2 = _mm_sub_pd(v_cv1, v_tv1);
        v_tv2 = _mm_sub_pd(_mm_mul_pd(v_K4, v_cv3), _mm_mul_pd(v_K2, v_cv4));
        v_tv2 = SWAP_RI_128_D(CONJ_128_D(v_tv2));
        v_out6 = _mm_add_pd(v_cv2, v_tv2);
        // Output point 5:X[4]
        v_out4 = _mm_sub_pd(v_cv2, v_tv2);

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
