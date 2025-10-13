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

/** @file twid_fft7.h
 *
 *  @brief The ISA generic kernel template for the radix 7 twiddle kernel
 *
 *  This file contains the DIT twiddle radix-7 FFT implementations for
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
    const FLOAT CRTM_7[6] = {+0.222520933956314404288902564496794759466355569,
                             +0.900968867902419126236102319507445051165919162,
                             +0.623489801858733530525004884004239810632274731,
                             +0.433883739117558120475768332848358754609990728,
                             +0.781831482468029808708444526674057750232334519,
                             +0.974927912181823607018131682993931217232785801};

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

    VREGTYPE_S v_C1 = BCAST_S(CRTM_7[0]);
    VREGTYPE_S v_C2 = BCAST_S(CRTM_7[1]);
    VREGTYPE_S v_C3 = BCAST_S(CRTM_7[2]);
    VREGTYPE_S v_C4 = BCAST_S(CRTM_7[3]);
    VREGTYPE_S v_C5 = BCAST_S(CRTM_7[4]);
    VREGTYPE_S v_C6 = BCAST_S(CRTM_7[5]);

    INTP count;

    v_C4 = NEG_S(v_C4, flag);
    v_C5 = NEG_S(v_C5, flag);
    v_C6 = NEG_S(v_C6, flag);

    for (count = 0; count < N; count++)
    {
        VREGTYPE_S v_in0, v_in1, v_in2, v_in3, v_in4, v_in5, v_in6;
        VREGTYPE_S v_av1, v_av2, v_av3, v_av4, v_av5, v_av6, v_av7;
        VREGTYPE_S v_tv1, v_tv2, v_tv3, v_tv4, v_tv5, v_tv6, v_tv7;
        VREGTYPE_S v_tv8, v_tv9, v_tv10, v_tv11, v_tv12, v_tv13, v_tv14;
        VREGTYPE_S v_tv16, v_tv17, v_tv18, v_tv19;
        VREGTYPE_S v_cv1, v_cv2, v_cv3, v_cv4, v_cv5, v_cv6;
        VREGTYPE_S v_out0, v_out1, v_out2, v_out3, v_out4, v_out5, v_out6;

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
        }

        GATHER_S(curr_in, v_in_stride, v_in0);

        // common calculations
        v_av1 = ADD_S(v_in2, v_in5);
        v_av2 = SUB_S(v_in2, v_in5);
        v_av3 = ADD_S(v_in3, v_in4);
        v_av4 = SUB_S(v_in3, v_in4);
        v_av5 = ADD_S(v_in1, v_in6);
        v_av6 = SUB_S(v_in1, v_in6);
        v_av7 = ADD_S(ADD_S(v_av1, v_av3), v_av5);

        v_tv1 = MUL_S(v_av1, v_C1);
        v_tv2 = MUL_S(v_av3, v_C1);
        v_tv3 = MUL_S(v_av5, v_C1);
        v_tv4 = MUL_S(v_av1, v_C2);
        v_tv5 = MUL_S(v_av3, v_C2);
        v_tv6 = MUL_S(v_av5, v_C2);
        v_tv7 = MUL_S(v_av1, v_C3);
        v_tv8 = MUL_S(v_av3, v_C3);
        v_tv9 = MUL_S(v_av5, v_C3);
        v_tv10 = MUL_S(v_av2, v_C4);
        v_tv11 = MUL_S(v_av4, v_C4);
        v_tv12 = MUL_S(v_av6, v_C4);
        v_tv13 = MUL_S(v_av2, v_C5);
        v_tv14 = MUL_S(v_av4, v_C5);
        v_tv16 = MUL_S(v_av6, v_C5);
        v_tv17 = MUL_S(v_av2, v_C6);
        v_tv18 = MUL_S(v_av4, v_C6);
        v_tv19 = MUL_S(v_av6, v_C6);

        v_cv1 = SUB_S(ADD_S(v_tv1, v_tv5), v_tv9);
        v_cv2 = ADD_S(ADD_S(v_tv11, v_tv16), v_tv17);
        v_cv3 = SUB_S(ADD_S(v_tv3, v_tv4), v_tv8);
        v_cv4 = SUB_S(ADD_S(v_tv10, v_tv14), v_tv19);
        v_cv5 = SUB_S(ADD_S(v_tv2, v_tv6), v_tv7);
        v_cv6 = ADD_S(SUB_S(v_tv12, v_tv13), v_tv18);

        v_cv2 = SWAP_RI_S(CONJ_S(v_cv2));
        v_cv4 = SWAP_RI_S(CONJ_S(v_cv4));
        v_cv6 = SWAP_RI_S(CONJ_S(v_cv6));

        // Output point 1: X[0]
        v_out0 = ADD_S(v_in0, v_av7);
        // Output point 7: X[6]
        v_out6 = ADD_S(SUB_S(v_in0, v_cv1), v_cv2);
        // Output point 2: X[1]
        v_out1 = SUB_S(SUB_S(v_in0, v_cv1), v_cv2);
        // Output point 3: X[2]
        v_out2 = ADD_S(SUB_S(v_in0, v_cv3), v_cv4);
        // Output point 6: X[5]
        v_out5 = SUB_S(SUB_S(v_in0, v_cv3), v_cv4);
        // Output point 5: X[4]
        v_out4 = ADD_S(SUB_S(v_in0, v_cv5), v_cv6);
        // Output point 4: X[3]
        v_out3 = SUB_S(SUB_S(v_in0, v_cv5), v_cv6);

        SCATTER_S(curr_out, v_out_stride, v_out0);
        SCATTER_S(curr_out + out_strides[1], v_out_stride, v_out1);
        SCATTER_S(curr_out + out_strides[2], v_out_stride, v_out2);
        SCATTER_S(curr_out + out_strides[3], v_out_stride, v_out3);
        SCATTER_S(curr_out + out_strides[4], v_out_stride, v_out4);
        SCATTER_S(curr_out + out_strides[5], v_out_stride, v_out5);
        SCATTER_S(curr_out + out_strides[6], v_out_stride, v_out6);

        in_r += NUM_SETS_S * v_in_stride;
        out_r += NUM_SETS_S * v_out_stride;
    }

    // The following contains code that performs the FFT on the tail cases.
    // These tails are conditionally "instantiated" based on the ISA requested
    // by the "includee" file.

#if defined(KERNEL_USE_AVX512)
    if (remaining_sets >= NUM_SETS_256_S)
    {
        __m256 v_in0, v_in1, v_in2, v_in3, v_in4, v_in5, v_in6;
        __m256 v_av1, v_av2, v_av3, v_av4, v_av5, v_av6, v_av7;
        __m256 v_tv1, v_tv2, v_tv3, v_tv4, v_tv5, v_tv6, v_tv7;
        __m256 v_tv8, v_tv9, v_tv10, v_tv11, v_tv12, v_tv13, v_tv14;
        __m256 v_tv16, v_tv17, v_tv18, v_tv19;
        __m256 v_cv1, v_cv2, v_cv3, v_cv4, v_cv5, v_cv6;
        __m256 v_out0, v_out1, v_out2, v_out3, v_out4, v_out5, v_out6;

        __m256 K1 = CAST_512_TO_256_S(v_C1);
        __m256 K2 = CAST_512_TO_256_S(v_C2);
        __m256 K3 = CAST_512_TO_256_S(v_C3);
        __m256 K4 = CAST_512_TO_256_S(v_C4);
        __m256 K5 = CAST_512_TO_256_S(v_C5);
        __m256 K6 = CAST_512_TO_256_S(v_C6);

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
        }

        GATHER4_256_S(curr_in, v_in_stride, v_in0);

        // common calculations
        v_av1 = _mm256_add_ps(v_in2, v_in5);
        v_av2 = _mm256_sub_ps(v_in2, v_in5);
        v_av3 = _mm256_add_ps(v_in3, v_in4);
        v_av4 = _mm256_sub_ps(v_in3, v_in4);
        v_av5 = _mm256_add_ps(v_in1, v_in6);
        v_av6 = _mm256_sub_ps(v_in1, v_in6);
        v_av7 = _mm256_add_ps(_mm256_add_ps(v_av1, v_av3), v_av5);

        v_tv1 = _mm256_mul_ps(v_av1, K1);
        v_tv2 = _mm256_mul_ps(v_av3, K1);
        v_tv3 = _mm256_mul_ps(v_av5, K1);
        v_tv4 = _mm256_mul_ps(v_av1, K2);
        v_tv5 = _mm256_mul_ps(v_av3, K2);
        v_tv6 = _mm256_mul_ps(v_av5, K2);
        v_tv7 = _mm256_mul_ps(v_av1, K3);
        v_tv8 = _mm256_mul_ps(v_av3, K3);
        v_tv9 = _mm256_mul_ps(v_av5, K3);
        v_tv10 = _mm256_mul_ps(v_av2, K4);
        v_tv11 = _mm256_mul_ps(v_av4, K4);
        v_tv12 = _mm256_mul_ps(v_av6, K4);
        v_tv13 = _mm256_mul_ps(v_av2, K5);
        v_tv14 = _mm256_mul_ps(v_av4, K5);
        v_tv16 = _mm256_mul_ps(v_av6, K5);
        v_tv17 = _mm256_mul_ps(v_av2, K6);
        v_tv18 = _mm256_mul_ps(v_av4, K6);
        v_tv19 = _mm256_mul_ps(v_av6, K6);

        v_cv1 = _mm256_sub_ps(_mm256_add_ps(v_tv1, v_tv5), v_tv9);
        v_cv2 = _mm256_add_ps(_mm256_add_ps(v_tv11, v_tv16), v_tv17);
        v_cv3 = _mm256_sub_ps(_mm256_add_ps(v_tv3, v_tv4), v_tv8);
        v_cv4 = _mm256_sub_ps(_mm256_add_ps(v_tv10, v_tv14), v_tv19);
        v_cv5 = _mm256_sub_ps(_mm256_add_ps(v_tv2, v_tv6), v_tv7);
        v_cv6 = _mm256_add_ps(_mm256_sub_ps(v_tv12, v_tv13), v_tv18);

        v_cv2 = SWAP_RI_256_S(CONJ_256_S(v_cv2));
        v_cv4 = SWAP_RI_256_S(CONJ_256_S(v_cv4));
        v_cv6 = SWAP_RI_256_S(CONJ_256_S(v_cv6));

        // Output point 1: X[0]
        v_out0 = _mm256_add_ps(v_in0, v_av7);
        // Output point 7: X[6]
        v_out6 = _mm256_add_ps(_mm256_sub_ps(v_in0, v_cv1), v_cv2);
        // Output point 2: X[1]
        v_out1 = _mm256_sub_ps(_mm256_sub_ps(v_in0, v_cv1), v_cv2);
        // Output point 3: X[2]
        v_out2 = _mm256_add_ps(_mm256_sub_ps(v_in0, v_cv3), v_cv4);
        // Output point 6: X[5]
        v_out5 = _mm256_sub_ps(_mm256_sub_ps(v_in0, v_cv3), v_cv4);
        // Output point 5: X[4]
        v_out4 = _mm256_add_ps(_mm256_sub_ps(v_in0, v_cv5), v_cv6);
        // Output point 4: X[3]
        v_out3 = _mm256_sub_ps(_mm256_sub_ps(v_in0, v_cv5), v_cv6);

        SCATTER4_256_S(curr_out, v_out_stride, v_out0);
        SCATTER4_256_S(curr_out + out_strides[1], v_out_stride, v_out1);
        SCATTER4_256_S(curr_out + out_strides[2], v_out_stride, v_out2);
        SCATTER4_256_S(curr_out + out_strides[3], v_out_stride, v_out3);
        SCATTER4_256_S(curr_out + out_strides[4], v_out_stride, v_out4);
        SCATTER4_256_S(curr_out + out_strides[5], v_out_stride, v_out5);
        SCATTER4_256_S(curr_out + out_strides[6], v_out_stride, v_out6);

        in_r += NUM_SETS_256_S * v_in_stride;
        out_r += NUM_SETS_256_S * v_out_stride;
        remaining_sets = remaining_sets - NUM_SETS_256_S;
    }
#endif

#if defined(KERNEL_USE_AVX512) || defined(KERNEL_USE_AVX256)
    if (remaining_sets >= NUM_SETS_128_S)
    {
        __m128 v_in0, v_in1, v_in2, v_in3, v_in4, v_in5, v_in6;
        __m128 v_av1, v_av2, v_av3, v_av4, v_av5, v_av6, v_av7;
        __m128 v_tv1, v_tv2, v_tv3, v_tv4, v_tv5, v_tv6, v_tv7;
        __m128 v_tv8, v_tv9, v_tv10, v_tv11, v_tv12, v_tv13, v_tv14;
        __m128 v_tv16, v_tv17, v_tv18, v_tv19;
        __m128 v_cv1, v_cv2, v_cv3, v_cv4, v_cv5, v_cv6;
        __m128 v_out0, v_out1, v_out2, v_out3, v_out4, v_out5, v_out6;

#if defined(KERNEL_USE_AVX512)
        __m128 K1 = CAST_512_TO_128_S(v_C1);
        __m128 K2 = CAST_512_TO_128_S(v_C2);
        __m128 K3 = CAST_512_TO_128_S(v_C3);
        __m128 K4 = CAST_512_TO_128_S(v_C4);
        __m128 K5 = CAST_512_TO_128_S(v_C5);
        __m128 K6 = CAST_512_TO_128_S(v_C6);
#elif defined(KERNEL_USE_AVX256)
        __m128 K1 = CAST_256_TO_128_S(v_C1);
        __m128 K2 = CAST_256_TO_128_S(v_C2);
        __m128 K3 = CAST_256_TO_128_S(v_C3);
        __m128 K4 = CAST_256_TO_128_S(v_C4);
        __m128 K5 = CAST_256_TO_128_S(v_C5);
        __m128 K6 = CAST_256_TO_128_S(v_C6);
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
        }

        GATHER2_128_S(curr_in, v_in_stride, v_in0);

        // common calculations
        v_av1 = _mm_add_ps(v_in2, v_in5);
        v_av2 = _mm_sub_ps(v_in2, v_in5);
        v_av3 = _mm_add_ps(v_in3, v_in4);
        v_av4 = _mm_sub_ps(v_in3, v_in4);
        v_av5 = _mm_add_ps(v_in1, v_in6);
        v_av6 = _mm_sub_ps(v_in1, v_in6);
        v_av7 = _mm_add_ps(_mm_add_ps(v_av1, v_av3), v_av5);

        v_tv1 = _mm_mul_ps(v_av1, K1);
        v_tv2 = _mm_mul_ps(v_av3, K1);
        v_tv3 = _mm_mul_ps(v_av5, K1);
        v_tv4 = _mm_mul_ps(v_av1, K2);
        v_tv5 = _mm_mul_ps(v_av3, K2);
        v_tv6 = _mm_mul_ps(v_av5, K2);
        v_tv7 = _mm_mul_ps(v_av1, K3);
        v_tv8 = _mm_mul_ps(v_av3, K3);
        v_tv9 = _mm_mul_ps(v_av5, K3);
        v_tv10 = _mm_mul_ps(v_av2, K4);
        v_tv11 = _mm_mul_ps(v_av4, K4);
        v_tv12 = _mm_mul_ps(v_av6, K4);
        v_tv13 = _mm_mul_ps(v_av2, K5);
        v_tv14 = _mm_mul_ps(v_av4, K5);
        v_tv16 = _mm_mul_ps(v_av6, K5);
        v_tv17 = _mm_mul_ps(v_av2, K6);
        v_tv18 = _mm_mul_ps(v_av4, K6);
        v_tv19 = _mm_mul_ps(v_av6, K6);

        v_cv1 = _mm_sub_ps(_mm_add_ps(v_tv1, v_tv5), v_tv9);
        v_cv2 = _mm_add_ps(_mm_add_ps(v_tv11, v_tv16), v_tv17);
        v_cv3 = _mm_sub_ps(_mm_add_ps(v_tv3, v_tv4), v_tv8);
        v_cv4 = _mm_sub_ps(_mm_add_ps(v_tv10, v_tv14), v_tv19);
        v_cv5 = _mm_sub_ps(_mm_add_ps(v_tv2, v_tv6), v_tv7);
        v_cv6 = _mm_add_ps(_mm_sub_ps(v_tv12, v_tv13), v_tv18);

        v_cv2 = SWAP_RI_128_S(CONJ_128_S(v_cv2));
        v_cv4 = SWAP_RI_128_S(CONJ_128_S(v_cv4));
        v_cv6 = SWAP_RI_128_S(CONJ_128_S(v_cv6));

        // Output point 1: X[0]
        v_out0 = _mm_add_ps(v_in0, v_av7);
        // Output point 7: X[6]
        v_out6 = _mm_add_ps(_mm_sub_ps(v_in0, v_cv1), v_cv2);
        // Output point 2: X[1]
        v_out1 = _mm_sub_ps(_mm_sub_ps(v_in0, v_cv1), v_cv2);
        // Output point 3: X[2]
        v_out2 = _mm_add_ps(_mm_sub_ps(v_in0, v_cv3), v_cv4);
        // Output point 6: X[5]
        v_out5 = _mm_sub_ps(_mm_sub_ps(v_in0, v_cv3), v_cv4);
        // Output point 5: X[4]
        v_out4 = _mm_add_ps(_mm_sub_ps(v_in0, v_cv5), v_cv6);
        // Output point 4: X[3]
        v_out3 = _mm_sub_ps(_mm_sub_ps(v_in0, v_cv5), v_cv6);

        SCATTER2_128_S(curr_out, v_out_stride, v_out0);
        SCATTER2_128_S(curr_out + out_strides[1], v_out_stride, v_out1);
        SCATTER2_128_S(curr_out + out_strides[2], v_out_stride, v_out2);
        SCATTER2_128_S(curr_out + out_strides[3], v_out_stride, v_out3);
        SCATTER2_128_S(curr_out + out_strides[4], v_out_stride, v_out4);
        SCATTER2_128_S(curr_out + out_strides[5], v_out_stride, v_out5);
        SCATTER2_128_S(curr_out + out_strides[6], v_out_stride, v_out6);

        in_r = in_r + (v_in_stride << 1);
        out_r = out_r + (v_out_stride << 1);
        remaining_sets = remaining_sets - NUM_SETS_128_S;
    }
#endif

#if defined(KERNEL_USE_AVX512) || defined(KERNEL_USE_AVX256) || defined(KERNEL_USE_AVX128)
    if (remaining_sets & 1)
    {
        __m128 v_in0, v_in1, v_in2, v_in3, v_in4, v_in5, v_in6;
        __m128 v_av1, v_av2, v_av3, v_av4, v_av5, v_av6, v_av7;
        __m128 v_tv1, v_tv2, v_tv3, v_tv4, v_tv5, v_tv6, v_tv7;
        __m128 v_tv8, v_tv9, v_tv10, v_tv11, v_tv12, v_tv13, v_tv14;
        __m128 v_tv16, v_tv17, v_tv18, v_tv19;
        __m128 v_cv1, v_cv2, v_cv3, v_cv4, v_cv5, v_cv6;
        __m128 v_out0, v_out1, v_out2, v_out3, v_out4, v_out5, v_out6;

#if defined(KERNEL_USE_AVX512)
        __m128 K1 = CAST_512_TO_128_S(v_C1);
        __m128 K2 = CAST_512_TO_128_S(v_C2);
        __m128 K3 = CAST_512_TO_128_S(v_C3);
        __m128 K4 = CAST_512_TO_128_S(v_C4);
        __m128 K5 = CAST_512_TO_128_S(v_C5);
        __m128 K6 = CAST_512_TO_128_S(v_C6);
#elif defined(KERNEL_USE_AVX256)
        __m128 K1 = CAST_256_TO_128_S(v_C1);
        __m128 K2 = CAST_256_TO_128_S(v_C2);
        __m128 K3 = CAST_256_TO_128_S(v_C3);
        __m128 K4 = CAST_256_TO_128_S(v_C4);
        __m128 K5 = CAST_256_TO_128_S(v_C5);
        __m128 K6 = CAST_256_TO_128_S(v_C6);
#elif defined(KERNEL_USE_AVX128)
        __m128 K1 = v_C1;
        __m128 K2 = v_C2;
        __m128 K3 = v_C3;
        __m128 K4 = v_C4;
        __m128 K5 = v_C5;
        __m128 K6 = v_C6;
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
        }

        LD_LOW_128_S(curr_in, v_in0);

        // common calculations
        v_av1 = _mm_add_ps(v_in2, v_in5);
        v_av2 = _mm_sub_ps(v_in2, v_in5);
        v_av3 = _mm_add_ps(v_in3, v_in4);
        v_av4 = _mm_sub_ps(v_in3, v_in4);
        v_av5 = _mm_add_ps(v_in1, v_in6);
        v_av6 = _mm_sub_ps(v_in1, v_in6);
        v_av7 = _mm_add_ps(_mm_add_ps(v_av1, v_av3), v_av5);

        v_tv1 = _mm_mul_ps(v_av1, K1);
        v_tv2 = _mm_mul_ps(v_av3, K1);
        v_tv3 = _mm_mul_ps(v_av5, K1);
        v_tv4 = _mm_mul_ps(v_av1, K2);
        v_tv5 = _mm_mul_ps(v_av3, K2);
        v_tv6 = _mm_mul_ps(v_av5, K2);
        v_tv7 = _mm_mul_ps(v_av1, K3);
        v_tv8 = _mm_mul_ps(v_av3, K3);
        v_tv9 = _mm_mul_ps(v_av5, K3);
        v_tv10 = _mm_mul_ps(v_av2, K4);
        v_tv11 = _mm_mul_ps(v_av4, K4);
        v_tv12 = _mm_mul_ps(v_av6, K4);
        v_tv13 = _mm_mul_ps(v_av2, K5);
        v_tv14 = _mm_mul_ps(v_av4, K5);
        v_tv16 = _mm_mul_ps(v_av6, K5);
        v_tv17 = _mm_mul_ps(v_av2, K6);
        v_tv18 = _mm_mul_ps(v_av4, K6);
        v_tv19 = _mm_mul_ps(v_av6, K6);

        v_cv1 = _mm_sub_ps(_mm_add_ps(v_tv1, v_tv5), v_tv9);
        v_cv2 = _mm_add_ps(_mm_add_ps(v_tv11, v_tv16), v_tv17);
        v_cv3 = _mm_sub_ps(_mm_add_ps(v_tv3, v_tv4), v_tv8);
        v_cv4 = _mm_sub_ps(_mm_add_ps(v_tv10, v_tv14), v_tv19);
        v_cv5 = _mm_sub_ps(_mm_add_ps(v_tv2, v_tv6), v_tv7);
        v_cv6 = _mm_add_ps(_mm_sub_ps(v_tv12, v_tv13), v_tv18);

        v_cv2 = SWAP_RI_128_S(CONJ_128_S(v_cv2));
        v_cv4 = SWAP_RI_128_S(CONJ_128_S(v_cv4));
        v_cv6 = SWAP_RI_128_S(CONJ_128_S(v_cv6));

        // Output point 1: X[0]
        v_out0 = _mm_add_ps(v_in0, v_av7);
        // Output point 7: X[6]
        v_out6 = _mm_add_ps(_mm_sub_ps(v_in0, v_cv1), v_cv2);
        // Output point 2: X[1]
        v_out1 = _mm_sub_ps(_mm_sub_ps(v_in0, v_cv1), v_cv2);
        // Output point 3: X[2]
        v_out2 = _mm_add_ps(_mm_sub_ps(v_in0, v_cv3), v_cv4);
        // Output point 6: X[5]
        v_out5 = _mm_sub_ps(_mm_sub_ps(v_in0, v_cv3), v_cv4);
        // Output point 5: X[4]
        v_out4 = _mm_add_ps(_mm_sub_ps(v_in0, v_cv5), v_cv6);
        // Output point 4: X[3]
        v_out3 = _mm_sub_ps(_mm_sub_ps(v_in0, v_cv5), v_cv6);

        ST_LOW_128_S(curr_out, v_out0);
        ST_LOW_128_S(curr_out + out_strides[1], v_out1);
        ST_LOW_128_S(curr_out + out_strides[2], v_out2);
        ST_LOW_128_S(curr_out + out_strides[3], v_out3);
        ST_LOW_128_S(curr_out + out_strides[4], v_out4);
        ST_LOW_128_S(curr_out + out_strides[5], v_out5);
        ST_LOW_128_S(curr_out + out_strides[6], v_out6);
    }
#endif

    AOCLFFTZ_LOG(DEBUG, global_logger_mode, "Exit");
}

static VOID TWID_KNAME_FP64(VOID *in_real, VOID *in_imag, VOID *out_real,
                            VOID *out_imag, INTP n, aoclfftz_strides_t *strides,
                            VOID *twd, UINT8 flag)
{
    AOCLFFTZ_LOG(DEBUG, global_logger_mode, "Enter");
    const DOUBLE CRTM_7[6] = {+0.222520933956314404288902564496794759466355569,
                              +0.900968867902419126236102319507445051165919162,
                              +0.623489801858733530525004884004239810632274731,
                              +0.433883739117558120475768332848358754609990728,
                              +0.781831482468029808708444526674057750232334519,
                              +0.974927912181823607018131682993931217232785801};

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

    VREGTYPE_D v_C1 = BCAST_D(CRTM_7[0]);
    VREGTYPE_D v_C2 = BCAST_D(CRTM_7[1]);
    VREGTYPE_D v_C3 = BCAST_D(CRTM_7[2]);
    VREGTYPE_D v_C4 = BCAST_D(CRTM_7[3]);
    VREGTYPE_D v_C5 = BCAST_D(CRTM_7[4]);
    VREGTYPE_D v_C6 = BCAST_D(CRTM_7[5]);

    v_C4 = NEG_D(v_C4, flag);
    v_C5 = NEG_D(v_C5, flag);
    v_C6 = NEG_D(v_C6, flag);

    for (count = 0; count < N; count++)
    {
        VREGTYPE_D v_in0, v_in1, v_in2, v_in3, v_in4, v_in5, v_in6;
        VREGTYPE_D v_av1, v_av2, v_av3, v_av4, v_av5, v_av6, v_av7;
        VREGTYPE_D v_tv1, v_tv2, v_tv3, v_tv4, v_tv5, v_tv6, v_tv7;
        VREGTYPE_D v_tv8, v_tv9, v_tv10, v_tv11, v_tv12, v_tv13, v_tv14;
        VREGTYPE_D v_tv16, v_tv17, v_tv18, v_tv19;
        VREGTYPE_D v_cv1, v_cv2, v_cv3, v_cv4, v_cv5, v_cv6;
        VREGTYPE_D v_out0, v_out1, v_out2, v_out3, v_out4, v_out5, v_out6;

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
        }

        GATHER_D(curr_in, v_in_stride, v_in0);

        // common calculations
        v_av1 = ADD_D(v_in2, v_in5);
        v_av2 = SUB_D(v_in2, v_in5);
        v_av3 = ADD_D(v_in3, v_in4);
        v_av4 = SUB_D(v_in3, v_in4);
        v_av5 = ADD_D(v_in1, v_in6);
        v_av6 = SUB_D(v_in1, v_in6);
        v_av7 = ADD_D(ADD_D(v_av1, v_av3), v_av5);

        v_tv1 = MUL_D(v_av1, v_C1);
        v_tv2 = MUL_D(v_av3, v_C1);
        v_tv3 = MUL_D(v_av5, v_C1);
        v_tv4 = MUL_D(v_av1, v_C2);
        v_tv5 = MUL_D(v_av3, v_C2);
        v_tv6 = MUL_D(v_av5, v_C2);
        v_tv7 = MUL_D(v_av1, v_C3);
        v_tv8 = MUL_D(v_av3, v_C3);
        v_tv9 = MUL_D(v_av5, v_C3);
        v_tv10 = MUL_D(v_av2, v_C4);
        v_tv11 = MUL_D(v_av4, v_C4);
        v_tv12 = MUL_D(v_av6, v_C4);
        v_tv13 = MUL_D(v_av2, v_C5);
        v_tv14 = MUL_D(v_av4, v_C5);
        v_tv16 = MUL_D(v_av6, v_C5);
        v_tv17 = MUL_D(v_av2, v_C6);
        v_tv18 = MUL_D(v_av4, v_C6);
        v_tv19 = MUL_D(v_av6, v_C6);

        v_cv1 = SUB_D(ADD_D(v_tv1, v_tv5), v_tv9);
        v_cv2 = ADD_D(ADD_D(v_tv11, v_tv16), v_tv17);
        v_cv3 = SUB_D(ADD_D(v_tv3, v_tv4), v_tv8);
        v_cv4 = SUB_D(ADD_D(v_tv10, v_tv14), v_tv19);
        v_cv5 = SUB_D(ADD_D(v_tv2, v_tv6), v_tv7);
        v_cv6 = ADD_D(SUB_D(v_tv12, v_tv13), v_tv18);

        v_cv2 = SWAP_RI_D(CONJ_D(v_cv2));
        v_cv4 = SWAP_RI_D(CONJ_D(v_cv4));
        v_cv6 = SWAP_RI_D(CONJ_D(v_cv6));

        // Output point 1: X[0]
        v_out0 = ADD_D(v_in0, v_av7);
        // Output point 7: X[6]
        v_out6 = ADD_D(SUB_D(v_in0, v_cv1), v_cv2);
        // Output point 2: X[1]
        v_out1 = SUB_D(SUB_D(v_in0, v_cv1), v_cv2);
        // Output point 3: X[2]
        v_out2 = ADD_D(SUB_D(v_in0, v_cv3), v_cv4);
        // Output point 6: X[5]
        v_out5 = SUB_D(SUB_D(v_in0, v_cv3), v_cv4);
        // Output point 5: X[4]
        v_out4 = ADD_D(SUB_D(v_in0, v_cv5), v_cv6);
        // Output point 4: X[3]
        v_out3 = SUB_D(SUB_D(v_in0, v_cv5), v_cv6);

        SCATTER_D(curr_out, v_out_stride, v_out0);
        SCATTER_D(curr_out + out_strides[1], v_out_stride, v_out1);
        SCATTER_D(curr_out + out_strides[2], v_out_stride, v_out2);
        SCATTER_D(curr_out + out_strides[3], v_out_stride, v_out3);
        SCATTER_D(curr_out + out_strides[4], v_out_stride, v_out4);
        SCATTER_D(curr_out + out_strides[5], v_out_stride, v_out5);
        SCATTER_D(curr_out + out_strides[6], v_out_stride, v_out6);

        in_r += NUM_SETS_D * v_in_stride;
        out_r += NUM_SETS_D * v_out_stride;
    }

    // The following contains code that performs the FFT on the tail cases.
    // These tails are conditionally "instantiated" based on the ISA requested
    // by the "includee" file.

#if defined(KERNEL_USE_AVX512)
    if (remaining_sets >= NUM_SETS_256_D)
    {
        __m256d v_in0, v_in1, v_in2, v_in3, v_in4, v_in5, v_in6;
        __m256d v_av1, v_av2, v_av3, v_av4, v_av5, v_av6, v_av7;
        __m256d v_tv1, v_tv2, v_tv3, v_tv4, v_tv5, v_tv6, v_tv7;
        __m256d v_tv8, v_tv9, v_tv10, v_tv11, v_tv12, v_tv13, v_tv14;
        __m256d v_tv16, v_tv17, v_tv18, v_tv19;
        __m256d v_cv1, v_cv2, v_cv3, v_cv4, v_cv5, v_cv6;
        __m256d v_out0, v_out1, v_out2, v_out3, v_out4, v_out5, v_out6;

        __m256d K1 = CAST_512_TO_256_D(v_C1);
        __m256d K2 = CAST_512_TO_256_D(v_C2);
        __m256d K3 = CAST_512_TO_256_D(v_C3);
        __m256d K4 = CAST_512_TO_256_D(v_C4);
        __m256d K5 = CAST_512_TO_256_D(v_C5);
        __m256d K6 = CAST_512_TO_256_D(v_C6);

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
        }

        GATHER2_256_D(curr_in, v_in_stride, v_in0);

        // common calculations
        v_av1 = _mm256_add_pd(v_in2, v_in5);
        v_av2 = _mm256_sub_pd(v_in2, v_in5);
        v_av3 = _mm256_add_pd(v_in3, v_in4);
        v_av4 = _mm256_sub_pd(v_in3, v_in4);
        v_av5 = _mm256_add_pd(v_in1, v_in6);
        v_av6 = _mm256_sub_pd(v_in1, v_in6);
        v_av7 = _mm256_add_pd(_mm256_add_pd(v_av1, v_av3), v_av5);

        v_tv1 = _mm256_mul_pd(v_av1, K1);
        v_tv2 = _mm256_mul_pd(v_av3, K1);
        v_tv3 = _mm256_mul_pd(v_av5, K1);
        v_tv4 = _mm256_mul_pd(v_av1, K2);
        v_tv5 = _mm256_mul_pd(v_av3, K2);
        v_tv6 = _mm256_mul_pd(v_av5, K2);
        v_tv7 = _mm256_mul_pd(v_av1, K3);
        v_tv8 = _mm256_mul_pd(v_av3, K3);
        v_tv9 = _mm256_mul_pd(v_av5, K3);
        v_tv10 = _mm256_mul_pd(v_av2, K4);
        v_tv11 = _mm256_mul_pd(v_av4, K4);
        v_tv12 = _mm256_mul_pd(v_av6, K4);
        v_tv13 = _mm256_mul_pd(v_av2, K5);
        v_tv14 = _mm256_mul_pd(v_av4, K5);
        v_tv16 = _mm256_mul_pd(v_av6, K5);
        v_tv17 = _mm256_mul_pd(v_av2, K6);
        v_tv18 = _mm256_mul_pd(v_av4, K6);
        v_tv19 = _mm256_mul_pd(v_av6, K6);

        v_cv1 = _mm256_sub_pd(_mm256_add_pd(v_tv1, v_tv5), v_tv9);
        v_cv2 = _mm256_add_pd(_mm256_add_pd(v_tv11, v_tv16), v_tv17);
        v_cv3 = _mm256_sub_pd(_mm256_add_pd(v_tv3, v_tv4), v_tv8);
        v_cv4 = _mm256_sub_pd(_mm256_add_pd(v_tv10, v_tv14), v_tv19);
        v_cv5 = _mm256_sub_pd(_mm256_add_pd(v_tv2, v_tv6), v_tv7);
        v_cv6 = _mm256_add_pd(_mm256_sub_pd(v_tv12, v_tv13), v_tv18);

        v_cv2 = SWAP_RI_256_D(CONJ_256_D(v_cv2));
        v_cv4 = SWAP_RI_256_D(CONJ_256_D(v_cv4));
        v_cv6 = SWAP_RI_256_D(CONJ_256_D(v_cv6));

        // Output point 1: X[0]
        v_out0 = _mm256_add_pd(v_in0, v_av7);
        // Output point 7: X[6]
        v_out6 = _mm256_add_pd(_mm256_sub_pd(v_in0, v_cv1), v_cv2);
        // Output point 2: X[1]
        v_out1 = _mm256_sub_pd(_mm256_sub_pd(v_in0, v_cv1), v_cv2);
        // Output point 3: X[2]
        v_out2 = _mm256_add_pd(_mm256_sub_pd(v_in0, v_cv3), v_cv4);
        // Output point 6: X[5]
        v_out5 = _mm256_sub_pd(_mm256_sub_pd(v_in0, v_cv3), v_cv4);
        // Output point 5: X[4]
        v_out4 = _mm256_add_pd(_mm256_sub_pd(v_in0, v_cv5), v_cv6);
        // Output point 4: X[3]
        v_out3 = _mm256_sub_pd(_mm256_sub_pd(v_in0, v_cv5), v_cv6);

        SCATTER2_256_D(curr_out, v_out_stride, v_out0);
        SCATTER2_256_D(curr_out + out_strides[1], v_out_stride, v_out1);
        SCATTER2_256_D(curr_out + out_strides[2], v_out_stride, v_out2);
        SCATTER2_256_D(curr_out + out_strides[3], v_out_stride, v_out3);
        SCATTER2_256_D(curr_out + out_strides[4], v_out_stride, v_out4);
        SCATTER2_256_D(curr_out + out_strides[5], v_out_stride, v_out5);
        SCATTER2_256_D(curr_out + out_strides[6], v_out_stride, v_out6);

        in_r += NUM_SETS_256_D * v_in_stride;
        out_r += NUM_SETS_256_D * v_out_stride;
    }
#endif

#if defined(KERNEL_USE_AVX512) || defined(KERNEL_USE_AVX256)
    if (remaining_sets & 1)
    {
        __m128d v_in0, v_in1, v_in2, v_in3, v_in4, v_in5, v_in6;
        __m128d v_av1, v_av2, v_av3, v_av4, v_av5, v_av6, v_av7;
        __m128d v_tv1, v_tv2, v_tv3, v_tv4, v_tv5, v_tv6, v_tv7;
        __m128d v_tv8, v_tv9, v_tv10, v_tv11, v_tv12, v_tv13, v_tv14;
        __m128d v_tv16, v_tv17, v_tv18, v_tv19;
        __m128d v_cv1, v_cv2, v_cv3, v_cv4, v_cv5, v_cv6;
        __m128d v_out0, v_out1, v_out2, v_out3, v_out4, v_out5, v_out6;

#if defined(KERNEL_USE_AVX512)
        __m128d K1 = CAST_512_TO_128_D(v_C1);
        __m128d K2 = CAST_512_TO_128_D(v_C2);
        __m128d K3 = CAST_512_TO_128_D(v_C3);
        __m128d K4 = CAST_512_TO_128_D(v_C4);
        __m128d K5 = CAST_512_TO_128_D(v_C5);
        __m128d K6 = CAST_512_TO_128_D(v_C6);
#elif defined(KERNEL_USE_AVX256)
        __m128d K1 = CAST_256_TO_128_D(v_C1);
        __m128d K2 = CAST_256_TO_128_D(v_C2);
        __m128d K3 = CAST_256_TO_128_D(v_C3);
        __m128d K4 = CAST_256_TO_128_D(v_C4);
        __m128d K5 = CAST_256_TO_128_D(v_C5);
        __m128d K6 = CAST_256_TO_128_D(v_C6);
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
        }

        LD_128_D(curr_in, v_in0);

        // common calculations
        v_av1 = _mm_add_pd(v_in2, v_in5);
        v_av2 = _mm_sub_pd(v_in2, v_in5);
        v_av3 = _mm_add_pd(v_in3, v_in4);
        v_av4 = _mm_sub_pd(v_in3, v_in4);
        v_av5 = _mm_add_pd(v_in1, v_in6);
        v_av6 = _mm_sub_pd(v_in1, v_in6);
        v_av7 = _mm_add_pd(_mm_add_pd(v_av1, v_av3), v_av5);

        v_tv1 = _mm_mul_pd(v_av1, K1);
        v_tv2 = _mm_mul_pd(v_av3, K1);
        v_tv3 = _mm_mul_pd(v_av5, K1);
        v_tv4 = _mm_mul_pd(v_av1, K2);
        v_tv5 = _mm_mul_pd(v_av3, K2);
        v_tv6 = _mm_mul_pd(v_av5, K2);
        v_tv7 = _mm_mul_pd(v_av1, K3);
        v_tv8 = _mm_mul_pd(v_av3, K3);
        v_tv9 = _mm_mul_pd(v_av5, K3);
        v_tv10 = _mm_mul_pd(v_av2, K4);
        v_tv11 = _mm_mul_pd(v_av4, K4);
        v_tv12 = _mm_mul_pd(v_av6, K4);
        v_tv13 = _mm_mul_pd(v_av2, K5);
        v_tv14 = _mm_mul_pd(v_av4, K5);
        v_tv16 = _mm_mul_pd(v_av6, K5);
        v_tv17 = _mm_mul_pd(v_av2, K6);
        v_tv18 = _mm_mul_pd(v_av4, K6);
        v_tv19 = _mm_mul_pd(v_av6, K6);

        v_cv1 = _mm_sub_pd(_mm_add_pd(v_tv1, v_tv5), v_tv9);
        v_cv2 = _mm_add_pd(_mm_add_pd(v_tv11, v_tv16), v_tv17);
        v_cv3 = _mm_sub_pd(_mm_add_pd(v_tv3, v_tv4), v_tv8);
        v_cv4 = _mm_sub_pd(_mm_add_pd(v_tv10, v_tv14), v_tv19);
        v_cv5 = _mm_sub_pd(_mm_add_pd(v_tv2, v_tv6), v_tv7);
        v_cv6 = _mm_add_pd(_mm_sub_pd(v_tv12, v_tv13), v_tv18);

        v_cv2 = SWAP_RI_128_D(CONJ_128_D(v_cv2));
        v_cv4 = SWAP_RI_128_D(CONJ_128_D(v_cv4));
        v_cv6 = SWAP_RI_128_D(CONJ_128_D(v_cv6));

        // Output point 1: X[0]
        v_out0 = _mm_add_pd(v_in0, v_av7);
        // Output point 7: X[6]
        v_out6 = _mm_add_pd(_mm_sub_pd(v_in0, v_cv1), v_cv2);
        // Output point 2: X[1]
        v_out1 = _mm_sub_pd(_mm_sub_pd(v_in0, v_cv1), v_cv2);
        // Output point 3: X[2]
        v_out2 = _mm_add_pd(_mm_sub_pd(v_in0, v_cv3), v_cv4);
        // Output point 6: X[5]
        v_out5 = _mm_sub_pd(_mm_sub_pd(v_in0, v_cv3), v_cv4);
        // Output point 5: X[4]
        v_out4 = _mm_add_pd(_mm_sub_pd(v_in0, v_cv5), v_cv6);
        // Output point 4: X[3]
        v_out3 = _mm_sub_pd(_mm_sub_pd(v_in0, v_cv5), v_cv6);

        ST_128_D(curr_out, v_out0);
        ST_128_D(curr_out + out_strides[1], v_out1);
        ST_128_D(curr_out + out_strides[2], v_out2);
        ST_128_D(curr_out + out_strides[3], v_out3);
        ST_128_D(curr_out + out_strides[4], v_out4);
        ST_128_D(curr_out + out_strides[5], v_out5);
        ST_128_D(curr_out + out_strides[6], v_out6);
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
