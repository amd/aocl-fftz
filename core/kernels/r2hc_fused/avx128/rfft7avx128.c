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

/** @file rfft7avx128.c
 *
 *  @brief Radix-7 r2hc_fused Real-FFT kernel with AVX-128 operations using x86
 *  SIMD intrinsics
 *
 *  This file contains the DIT radix-7 real-to-halfcomplex fused of two
 *  different implementations (Standard DFT and Shifted DFT that differs in DFT
 *  weight matrix) using x86 SIMD intrinsics for single-precision and
 *  double-precision inputs
 *
 *  @author Jeya R
 */
#include "core/kernels/kernel.h"
#include "core/kernels/simd_includes/r2hc_simd_avx128.h"

static const ops_cycles_t ops_cnt[NUM_PRECISIONS][NUM_FFT_DIRS] =
                                                    {{{0, 36, 48, 88, 61, 0},
                                                      {0, 38, 48, 88, 70, 0}},
                                                     {{0, 36, 48, 44, 13, 0},
                                                      {0, 38, 48, 44, 16, 0}}};

ops_cycles_t get_ops_cnt_r2hcf_rfft7avx128(UINT8 precision, UINT8 direction)
 {
     if (precision == DT_FLOAT)
     {
         if (direction == FORWARD_FFT_DIR)
         {
             return ops_cnt[0][0];
         }
         else
         {
             return ops_cnt[0][1];
         }
     }
     else
     {
         if (direction == FORWARD_FFT_DIR)
         {
             return ops_cnt[1][0];
         }
         else
         {
             return ops_cnt[1][1];
         }
     }
 }

static VOID r2hcf_rfft7avx128_fp32_fwd(VOID *in_real, VOID *in_imag,
                                       VOID *out_real, VOID *out_imag, INTP n,
                                       aoclfftz_strides_t *strides, UINT8 flag)
{
    const FLOAT CRTM_7_1 = 0.900968867902419126236102319507445051165919162f;
    const FLOAT CRTM_7_2 = 0.433883739117558120475768332848358754609990728f;
    const FLOAT CRTM_7_3 = 0.623489801858733530525004884004239810632274731f;
    const FLOAT CRTM_7_4 = 0.781831482468029808708444526674057750232334519f;
    const FLOAT CRTM_7_5 = 0.222520933956314404288902564496794759466355569f;
    const FLOAT CRTM_7_6 = 0.974927912181823607018131682993931217232785801f;

    FLOAT *in = (FLOAT *)in_real;
    FLOAT *out = (FLOAT *)out_real;
    #ifdef VOLATILE_STRIDE_ARRAY
    volatile INTP *in_strides = strides->in_strides;
    volatile INTP *out_strides = strides->out_strides;
    #else
    INTP *in_strides = strides->in_strides;
    INTP *out_strides = strides->out_strides;
    #endif
    INTP v_in_stride = (strides->v_in_stride);
    INTP v_out_stride = (strides->v_out_stride);

    INTP cnt;
    INTP N = n >> 2;
    FLOAT *curr_in, *curr_out;

    __m128 v_CRTM_7_1 = _mm_set1_ps(CRTM_7_1);
    __m128 v_CRTM_7_2 = _mm_set1_ps(CRTM_7_2);
    __m128 v_CRTM_7_3 = _mm_set1_ps(CRTM_7_3);
    __m128 v_CRTM_7_4 = _mm_set1_ps(CRTM_7_4);
    __m128 v_CRTM_7_5 = _mm_set1_ps(CRTM_7_5);
    __m128 v_CRTM_7_6 = _mm_set1_ps(CRTM_7_6);

    for (cnt = 0; cnt < N; cnt++)
    {
        __m128 av_in0, av_in1, av_in2, av_in3, av_in4, av_in5, av_in6;
        __m128 av_sv1, av_sv2, av_sv3, av_sv4, av_sv5, av_sv6, av_sv7, av_sv8,
               av_tv1, av_tv2, av_tv3, av_sv9, av_sv10, av_tv4, av_tv5, av_tv6,
               av_sv11, av_tv7, av_tv8, av_tv9, av_sv12, av_sv13, av_tv10,
               av_tv11, av_tv12, av_sv14, av_tv13, av_tv14, av_tv15, av_sv15,
               av_sv16, av_tv16, av_tv17, av_tv18, av_sv17;
        __m128 v_out0, v_out1, v_out2, v_out3, v_out4, v_out5, v_out6, v_out7,
               v_out8, v_out9, v_out10, v_out11, v_out12, v_out13;

        curr_in = in;
        curr_out = out;

        // Input point 1: x(0)
        LDR_128_S(curr_in, v_in_stride, av_in0);
        // Input point 3: x(2)
        curr_in = in + in_strides[2];
        LDR_128_S(curr_in, v_in_stride, av_in1);
        // Input point 5: x(4)
        curr_in = in + in_strides[4];
        LDR_128_S(curr_in, v_in_stride, av_in2);
        // Input point 7: x(6)
        curr_in = in + in_strides[6];
        LDR_128_S(curr_in, v_in_stride, av_in3);
        // Input point 9: x(8)
        curr_in = in + in_strides[8];
        LDR_128_S(curr_in, v_in_stride, av_in4);
        // Input point 11: x(10)
        curr_in = in + in_strides[10];
        LDR_128_S(curr_in, v_in_stride, av_in5);
        // Input point 13: x(12)
        curr_in = in + in_strides[12];
        LDR_128_S(curr_in, v_in_stride, av_in6);

        av_sv1 = _mm_add_ps(av_in6, av_in1);
        av_sv2 = _mm_sub_ps(av_in6, av_in1);
        av_sv3 = _mm_add_ps(av_in5, av_in2);
        av_sv4 = _mm_sub_ps(av_in5, av_in2);
        av_sv5 = _mm_add_ps(av_in4, av_in3);
        av_sv6 = _mm_sub_ps(av_in4, av_in3);

        av_sv7 = _mm_add_ps(av_in0, av_sv1);
        av_sv8 = _mm_add_ps(av_sv3, av_sv5);

        av_tv1 = _mm_mul_ps(v_CRTM_7_1, av_sv5);
        av_tv2 = _mm_mul_ps(v_CRTM_7_3, av_sv1);
        av_tv3 = _mm_mul_ps(v_CRTM_7_5, av_sv3);
        av_sv9 = _mm_sub_ps(av_in0, av_tv1);

        av_sv10 = _mm_sub_ps(av_tv2, av_tv3);
        av_tv4 = _mm_mul_ps(v_CRTM_7_2, av_sv6);
        av_tv5 = _mm_mul_ps(v_CRTM_7_4, av_sv2);

        av_tv6 = _mm_mul_ps(v_CRTM_7_6, av_sv4);
        av_sv11 = _mm_add_ps(av_tv4, av_tv5);

        av_tv7 = _mm_mul_ps(v_CRTM_7_1, av_sv3);
        av_tv8 = _mm_mul_ps(v_CRTM_7_3, av_sv5);
        av_tv9 = _mm_mul_ps(v_CRTM_7_5, av_sv1);

        av_sv12 = _mm_sub_ps(av_in0, av_tv7);
        av_sv13 = _mm_sub_ps(av_tv8, av_tv9);
        av_tv10 = _mm_mul_ps(v_CRTM_7_2, av_sv4);
        av_tv11 = _mm_mul_ps(v_CRTM_7_4, av_sv6);

        av_tv12 = _mm_mul_ps(v_CRTM_7_6, av_sv2);
        av_sv14 = _mm_add_ps(av_tv10, av_tv11);
        av_tv13 = _mm_mul_ps(v_CRTM_7_1, av_sv1);
        av_tv14 = _mm_mul_ps(v_CRTM_7_3, av_sv3);
        av_tv15 = _mm_mul_ps(v_CRTM_7_5, av_sv5);

        av_sv15 = _mm_sub_ps(av_in0, av_tv13);
        av_sv16 = _mm_sub_ps(av_tv14, av_tv15);
        av_tv16 = _mm_mul_ps(v_CRTM_7_2, av_sv2);
        av_tv17 = _mm_mul_ps(v_CRTM_7_4, av_sv4);
        av_tv18 = _mm_mul_ps(v_CRTM_7_6, av_sv6);
        av_sv17 = _mm_sub_ps(av_tv16, av_tv17);

        // Output pt 1: X(0)
        v_out0 = _mm_add_ps(av_sv7, av_sv8);
        STR_128_S(curr_out, v_out_stride, v_out0);
        // Output pt 4: X(3) & Output pt 5: X(4)
        curr_out = out + out_strides[3];
        v_out3 = _mm_add_ps(av_sv9, av_sv10);
        v_out4 = _mm_add_ps(av_tv6, av_sv11);
        STRI_2x128_S(curr_out, v_out_stride, v_out3, v_out4);
        // Output pt 8: X(7) & Output pt 9: X(8)
        curr_out = out + out_strides[7];
        v_out7 = _mm_add_ps(av_sv12, av_sv13);
        v_out8 = _mm_sub_ps(av_tv12, av_sv14);
        STRI_2x128_S(curr_out, v_out_stride, v_out7, v_out8);
        // Output pt 12: X(11) & Output pt 13: X(12)
        curr_out = out + out_strides[11];
        v_out11 = _mm_add_ps(av_sv15, av_sv16);
        v_out12 = _mm_add_ps(av_sv17, av_tv18);
        STRI_2x128_S(curr_out, v_out_stride, v_out11, v_out12);

        __m128 bv_in0, bv_in1, bv_in2, bv_in3, bv_in4, bv_in5, bv_in6;
        __m128 bv_sv1, bv_sv2, bv_sv3, bv_sv4, bv_sv5, bv_sv6, bv_tv1, bv_tv2,
               bv_tv3, bv_sv7, bv_sv8, bv_tv4, bv_tv5, bv_tv6, bv_sv9, bv_tv7,
               bv_tv8, bv_tv9, bv_sv10, bv_sv11, bv_tv10, bv_tv11, bv_tv12,
               bv_sv12, bv_tv13, bv_tv14, bv_tv15, bv_sv13, bv_sv14, bv_tv16,
               bv_tv17, bv_tv18, bv_sv15, bv_sv16, bv_sv17;

        // Input point 2: x(1)
        curr_in = in + in_strides[1];
        LDR_128_S(curr_in, v_in_stride, bv_in0);
        // Input point 4: x(3)
        curr_in = in + in_strides[3];
        LDR_128_S(curr_in, v_in_stride, bv_in1);
        // Input point 6: x(5)
        curr_in = in + in_strides[5];
        LDR_128_S(curr_in, v_in_stride, bv_in2);
        // Input point 8: x(7)
        curr_in = in + in_strides[7];
        LDR_128_S(curr_in, v_in_stride, bv_in3);
        // Input point 10: x(9)
        curr_in = in + in_strides[9];
        LDR_128_S(curr_in, v_in_stride, bv_in4);
        // Input point 12: x(11)
        curr_in = in + in_strides[11];
        LDR_128_S(curr_in, v_in_stride, bv_in5);
        // Input point 14: x(13)
        curr_in = in + in_strides[13];
        LDR_128_S(curr_in, v_in_stride, bv_in6);

        bv_sv1 = _mm_add_ps(bv_in6, bv_in1);
        bv_sv2 = _mm_sub_ps(bv_in6, bv_in1);
        bv_sv3 = _mm_add_ps(bv_in5, bv_in2);
        bv_sv4 = _mm_sub_ps(bv_in5, bv_in2);
        bv_sv5 = _mm_add_ps(bv_in4, bv_in3);
        bv_sv6 = _mm_sub_ps(bv_in4, bv_in3);

        bv_tv1 = _mm_mul_ps(v_CRTM_7_1, bv_sv2);
        bv_tv2 = _mm_mul_ps(v_CRTM_7_3, bv_sv4);

        bv_tv3 = _mm_mul_ps(v_CRTM_7_5, bv_sv6);
        bv_sv7 = _mm_sub_ps(bv_in0, bv_tv1);
        bv_sv8 = _mm_add_ps(bv_tv2, bv_tv3);
        bv_tv4 = _mm_mul_ps(v_CRTM_7_2, bv_sv1);

        bv_tv5 = _mm_mul_ps(v_CRTM_7_4, bv_sv3);
        bv_tv6 = _mm_mul_ps(v_CRTM_7_6, bv_sv5);
        bv_sv9 = NEGATE_128_S(_mm_add_ps(bv_tv4, bv_tv5));

        bv_tv7 = _mm_mul_ps(v_CRTM_7_1, bv_sv4);
        bv_tv8 = _mm_mul_ps(v_CRTM_7_3, bv_sv6);
        bv_tv9 = _mm_mul_ps(v_CRTM_7_5, bv_sv2);
        bv_sv10 = _mm_add_ps(bv_in0, bv_tv7);
        bv_sv11 = _mm_sub_ps(bv_tv8, bv_tv9);

        bv_tv10 = _mm_mul_ps(v_CRTM_7_2, bv_sv3);
        bv_tv11 = _mm_mul_ps(v_CRTM_7_4, bv_sv5);

        bv_tv12 = _mm_mul_ps(v_CRTM_7_6, bv_sv1);
        bv_sv12 = _mm_sub_ps(bv_tv11,bv_tv10);

        bv_tv13 = _mm_mul_ps(v_CRTM_7_1, bv_sv6);
        bv_tv14 = _mm_mul_ps(v_CRTM_7_3, bv_sv2);
        bv_tv15 = _mm_mul_ps(v_CRTM_7_5, bv_sv4);
        bv_sv13 = _mm_sub_ps(bv_in0, bv_tv13);
        bv_sv14 = _mm_add_ps(bv_tv14, bv_tv15);

        bv_tv16 = _mm_mul_ps(v_CRTM_7_2, bv_sv5);
        bv_tv17 = _mm_mul_ps(v_CRTM_7_4, bv_sv1);
        bv_tv18 = _mm_mul_ps(v_CRTM_7_6, bv_sv3);
        bv_sv15 = _mm_add_ps(bv_tv16, bv_tv17);
        bv_sv16 = _mm_add_ps(bv_in0, bv_sv2);
        bv_sv17 = _mm_sub_ps(bv_sv6, bv_sv4);

        // Output pt 2: X(1) & Output pt 3: X(2)
        curr_out = out + out_strides[1];
        v_out1 = _mm_sub_ps(bv_sv7, bv_sv8);
        v_out2 = _mm_sub_ps(bv_sv9, bv_tv6);
        STRI_2x128_S(curr_out, v_out_stride, v_out1, v_out2);
        // Output pt 6: X(5) & Output pt 7: X(6)
        curr_out = out + out_strides[5];
        v_out5 = _mm_add_ps(bv_sv10, bv_sv11);
        v_out6 = _mm_sub_ps(bv_sv12, bv_tv12);
        STRI_2x128_S(curr_out, v_out_stride, v_out5, v_out6);
        // Output pt 10: X(9) & Output pt 11: X(10)
        curr_out = out + out_strides[9];
        v_out9 = _mm_add_ps(bv_sv13, bv_sv14);
        v_out10 = _mm_sub_ps(bv_tv18, bv_sv15);
        STRI_2x128_S(curr_out, v_out_stride, v_out9, v_out10);
        // Output pt 14: X(13)
        curr_out = out + out_strides[13];
        v_out13 = _mm_add_ps(bv_sv16, bv_sv17);
        STR_128_S(curr_out, v_out_stride, v_out13);

        in = in + (v_in_stride << 2);
        out = out + (v_out_stride << 2);
    }
    // tail cases
    if (n & 2)
    {
        __m128 av_in0, av_in1, av_in2, av_in3, av_in4, av_in5, av_in6;
        __m128 av_sv1, av_sv2, av_sv3, av_sv4, av_sv5, av_sv6, av_sv7, av_sv8,
               av_tv1, av_tv2, av_tv3, av_sv9, av_sv10, av_tv4, av_tv5, av_tv6,
               av_sv11, av_tv7, av_tv8, av_tv9, av_sv12, av_sv13, av_tv10,
               av_tv11, av_tv12, av_sv14, av_tv13, av_tv14, av_tv15, av_sv15,
               av_sv16, av_tv16, av_tv17, av_tv18, av_sv17;
        __m128 v_out0, v_out1, v_out2, v_out3, v_out4, v_out5, v_out6, v_out7,
               v_out8, v_out9, v_out10, v_out11, v_out12, v_out13;

        curr_in = in;
        curr_out = out;

        // Input point 1: x(0)
        LDHR_128_S(curr_in, v_in_stride, av_in0);
        // Input point 3: x(2)
        curr_in = in + in_strides[2];
        LDHR_128_S(curr_in, v_in_stride, av_in1);
        // Input point 5: x(4)
        curr_in = in + in_strides[4];
        LDHR_128_S(curr_in, v_in_stride, av_in2);
        // Input point 7: x(6)
        curr_in = in + in_strides[6];
        LDHR_128_S(curr_in, v_in_stride, av_in3);
        // Input point 9: x(8)
        curr_in = in + in_strides[8];
        LDHR_128_S(curr_in, v_in_stride, av_in4);
        // Input point 11: x(10)
        curr_in = in + in_strides[10];
        LDHR_128_S(curr_in, v_in_stride, av_in5);
        // Input point 13: x(12)
        curr_in = in + in_strides[12];
        LDHR_128_S(curr_in, v_in_stride, av_in6);

        av_sv1 = _mm_add_ps(av_in6, av_in1);
        av_sv2 = _mm_sub_ps(av_in6, av_in1);
        av_sv3 = _mm_add_ps(av_in5, av_in2);
        av_sv4 = _mm_sub_ps(av_in5, av_in2);
        av_sv5 = _mm_add_ps(av_in4, av_in3);
        av_sv6 = _mm_sub_ps(av_in4, av_in3);

        av_sv7 = _mm_add_ps(av_in0, av_sv1);
        av_sv8 = _mm_add_ps(av_sv3, av_sv5);

        av_tv1 = _mm_mul_ps(v_CRTM_7_1, av_sv5);
        av_tv2 = _mm_mul_ps(v_CRTM_7_3, av_sv1);
        av_tv3 = _mm_mul_ps(v_CRTM_7_5, av_sv3);
        av_sv9 = _mm_sub_ps(av_in0, av_tv1);

        av_sv10 = _mm_sub_ps(av_tv2, av_tv3);
        av_tv4 = _mm_mul_ps(v_CRTM_7_2, av_sv6);
        av_tv5 = _mm_mul_ps(v_CRTM_7_4, av_sv2);

        av_tv6 = _mm_mul_ps(v_CRTM_7_6, av_sv4);
        av_sv11 = _mm_add_ps(av_tv4, av_tv5);

        av_tv7 = _mm_mul_ps(v_CRTM_7_1, av_sv3);
        av_tv8 = _mm_mul_ps(v_CRTM_7_3, av_sv5);
        av_tv9 = _mm_mul_ps(v_CRTM_7_5, av_sv1);

        av_sv12 = _mm_sub_ps(av_in0, av_tv7);
        av_sv13 = _mm_sub_ps(av_tv8, av_tv9);
        av_tv10 = _mm_mul_ps(v_CRTM_7_2, av_sv4);
        av_tv11 = _mm_mul_ps(v_CRTM_7_4, av_sv6);

        av_tv12 = _mm_mul_ps(v_CRTM_7_6, av_sv2);
        av_sv14 = _mm_add_ps(av_tv10, av_tv11);
        av_tv13 = _mm_mul_ps(v_CRTM_7_1, av_sv1);
        av_tv14 = _mm_mul_ps(v_CRTM_7_3, av_sv3);
        av_tv15 = _mm_mul_ps(v_CRTM_7_5, av_sv5);

        av_sv15 = _mm_sub_ps(av_in0, av_tv13);
        av_sv16 = _mm_sub_ps(av_tv14, av_tv15);
        av_tv16 = _mm_mul_ps(v_CRTM_7_2, av_sv2);
        av_tv17 = _mm_mul_ps(v_CRTM_7_4, av_sv4);
        av_tv18 = _mm_mul_ps(v_CRTM_7_6, av_sv6);
        av_sv17 = _mm_sub_ps(av_tv16, av_tv17);

        // Output pt 1: X(0)
        v_out0 = _mm_add_ps(av_sv7, av_sv8);
        STHR_128_S(curr_out, v_out_stride, v_out0);
        // Output pt 4: X(3) & Output pt 5: X(4)
        curr_out = out + out_strides[3];
        v_out3 = _mm_add_ps(av_sv9, av_sv10);
        v_out4 = _mm_add_ps(av_tv6, av_sv11);
        STHRI_2x128_S(curr_out, v_out_stride, v_out3, v_out4);
        // Output pt 8: X(7) & Output pt 9: X(8)
        curr_out = out + out_strides[7];
        v_out7 = _mm_add_ps(av_sv12, av_sv13);
        v_out8 = _mm_sub_ps(av_tv12, av_sv14);
        STHRI_2x128_S(curr_out, v_out_stride, v_out7, v_out8);
        // Output pt 12: X(11) & Output pt 13: X(12)
        curr_out = out + out_strides[11];
        v_out11 = _mm_add_ps(av_sv15, av_sv16);
        v_out12 = _mm_add_ps(av_sv17, av_tv18);
        STHRI_2x128_S(curr_out, v_out_stride, v_out11, v_out12);

        __m128 bv_in0, bv_in1, bv_in2, bv_in3, bv_in4, bv_in5, bv_in6;
        __m128 bv_sv1, bv_sv2, bv_sv3, bv_sv4, bv_sv5, bv_sv6, bv_tv1, bv_tv2,
               bv_tv3, bv_sv7, bv_sv8, bv_tv4, bv_tv5, bv_tv6, bv_sv9, bv_tv7,
               bv_tv8, bv_tv9, bv_sv10, bv_sv11, bv_tv10, bv_tv11, bv_tv12,
               bv_sv12, bv_tv13, bv_tv14, bv_tv15, bv_sv13, bv_sv14, bv_tv16,
               bv_tv17, bv_tv18, bv_sv15, bv_sv16, bv_sv17;

        // Input point 2: x(1)
        curr_in = in + in_strides[1];
        LDHR_128_S(curr_in, v_in_stride, bv_in0);
        // Input point 4: x(3)
        curr_in = in + in_strides[3];
        LDHR_128_S(curr_in, v_in_stride, bv_in1);
        // Input point 6: x(5)
        curr_in = in + in_strides[5];
        LDHR_128_S(curr_in, v_in_stride, bv_in2);
        // Input point 8: x(7)
        curr_in = in + in_strides[7];
        LDHR_128_S(curr_in, v_in_stride, bv_in3);
        // Input point 10: x(9)
        curr_in = in + in_strides[9];
        LDHR_128_S(curr_in, v_in_stride, bv_in4);
        // Input point 12: x(11)
        curr_in = in + in_strides[11];
        LDHR_128_S(curr_in, v_in_stride, bv_in5);
        // Input point 14: x(13)
        curr_in = in + in_strides[13];
        LDHR_128_S(curr_in, v_in_stride, bv_in6);

        bv_sv1 = _mm_add_ps(bv_in6, bv_in1);
        bv_sv2 = _mm_sub_ps(bv_in6, bv_in1);
        bv_sv3 = _mm_add_ps(bv_in5, bv_in2);
        bv_sv4 = _mm_sub_ps(bv_in5, bv_in2);
        bv_sv5 = _mm_add_ps(bv_in4, bv_in3);
        bv_sv6 = _mm_sub_ps(bv_in4, bv_in3);

        bv_tv1 = _mm_mul_ps(v_CRTM_7_1, bv_sv2);
        bv_tv2 = _mm_mul_ps(v_CRTM_7_3, bv_sv4);

        bv_tv3 = _mm_mul_ps(v_CRTM_7_5, bv_sv6);
        bv_sv7 = _mm_sub_ps(bv_in0, bv_tv1);
        bv_sv8 = _mm_add_ps(bv_tv2, bv_tv3);
        bv_tv4 = _mm_mul_ps(v_CRTM_7_2, bv_sv1);

        bv_tv5 = _mm_mul_ps(v_CRTM_7_4, bv_sv3);
        bv_tv6 = _mm_mul_ps(v_CRTM_7_6, bv_sv5);
        bv_sv9 = NEGATE_128_S(_mm_add_ps(bv_tv4, bv_tv5));

        bv_tv7 = _mm_mul_ps(v_CRTM_7_1, bv_sv4);
        bv_tv8 = _mm_mul_ps(v_CRTM_7_3, bv_sv6);
        bv_tv9 = _mm_mul_ps(v_CRTM_7_5, bv_sv2);
        bv_sv10 = _mm_add_ps(bv_in0, bv_tv7);
        bv_sv11 = _mm_sub_ps(bv_tv8, bv_tv9);

        bv_tv10 = _mm_mul_ps(v_CRTM_7_2, bv_sv3);
        bv_tv11 = _mm_mul_ps(v_CRTM_7_4, bv_sv5);

        bv_tv12 = _mm_mul_ps(v_CRTM_7_6, bv_sv1);
        bv_sv12 = _mm_sub_ps(bv_tv11,bv_tv10);

        bv_tv13 = _mm_mul_ps(v_CRTM_7_1, bv_sv6);
        bv_tv14 = _mm_mul_ps(v_CRTM_7_3, bv_sv2);
        bv_tv15 = _mm_mul_ps(v_CRTM_7_5, bv_sv4);
        bv_sv13 = _mm_sub_ps(bv_in0, bv_tv13);
        bv_sv14 = _mm_add_ps(bv_tv14, bv_tv15);

        bv_tv16 = _mm_mul_ps(v_CRTM_7_2, bv_sv5);
        bv_tv17 = _mm_mul_ps(v_CRTM_7_4, bv_sv1);
        bv_tv18 = _mm_mul_ps(v_CRTM_7_6, bv_sv3);
        bv_sv15 = _mm_add_ps(bv_tv16, bv_tv17);
        bv_sv16 = _mm_add_ps(bv_in0, bv_sv2);
        bv_sv17 = _mm_sub_ps(bv_sv6, bv_sv4);

        // Output pt 2: X(1) & Output pt 3: X(2)
        curr_out = out + out_strides[1];
        v_out1 = _mm_sub_ps(bv_sv7, bv_sv8);
        v_out2 = _mm_sub_ps(bv_sv9, bv_tv6);
        STHRI_2x128_S(curr_out, v_out_stride, v_out1, v_out2);
        // Output pt 6: X(5) & Output pt 7: X(6)
        curr_out = out + out_strides[5];
        v_out5 = _mm_add_ps(bv_sv10, bv_sv11);
        v_out6 = _mm_sub_ps(bv_sv12, bv_tv12);
        STHRI_2x128_S(curr_out, v_out_stride, v_out5, v_out6);
        // Output pt 10: X(9) & Output pt 11: X(10)
        curr_out = out + out_strides[9];
        v_out9 = _mm_add_ps(bv_sv13, bv_sv14);
        v_out10 = _mm_sub_ps(bv_tv18, bv_sv15);
        STHRI_2x128_S(curr_out, v_out_stride, v_out9, v_out10);
        // Output pt 14: X(13)
        curr_out = out + out_strides[13];
        v_out13 = _mm_add_ps(bv_sv16, bv_sv17);
        STHR_128_S(curr_out, v_out_stride, v_out13);

        in = in + (v_in_stride << 1);
        out = out + (v_out_stride << 1);
    }
    // tail cases
    if (n & 1)
    {
        /* Standard DFT */
        FLOAT av0, av1, av2, av3, av4, av5, av6;
        FLOAT at0, at1, at2, at3, at4, at5, at6, at7, at8, at9,
              at10, at11, at12, at13, at14, at15, at16, at17, at18, at19,
              at20, at21, at22, at23, at24, at25, at26, at27, at28, at29,
              at30, at31, at32, at33, at34;
        av0 = *in;                  // Input point 1: x(0)
        av1 = in[in_strides[2]];    // Input point 3: x(2)
        av2 = in[in_strides[4]];    // Input point 5: x(4)
        av3 = in[in_strides[6]];    // Input point 7: x(6)
        av4 = in[in_strides[8]];    // Input point 9: x(8)
        av5 = in[in_strides[10]];   // Input point 11: x(10)
        av6 = in[in_strides[12]];   // Input point 13: x(12)

        at0 = av6 + av1;
        at1 = av6 - av1;
        at2 = av5 + av2;
        at3 = av5 - av2;
        at4 = av4 + av3;
        at5 = av4 - av3;
        at6 = av0 + at0;
        at7 = at2 + at4;

        at8 = CRTM_7_1 * at4;
        at9 = CRTM_7_3 * at0;
        at10 = CRTM_7_5 * at2;
        at11 = CRTM_7_2 * at5;
        at12 = CRTM_7_4 * at1;
        at13 = av0 - at8;
        at14 = at9 - at10;

        at15 = CRTM_7_6 * at3;
        at16 = at11 + at12;
        at17 = CRTM_7_1 * at2;
        at18 = CRTM_7_3 * at4;
        at19 = CRTM_7_5 * at0;

        at20 = av0 - at17;
        at21 = at18 - at19;
        at22 = CRTM_7_2 * at3;
        at23 = CRTM_7_4 * at5;

        at24 = CRTM_7_6 * at1;
        at25 = CRTM_7_1 * at0;
        at26 = CRTM_7_3 * at2;
        at27 = CRTM_7_5 * at4;
        at28 = at22 + at23;

        at29 = av0 - at25;
        at30 = at26 - at27;
        at31 = CRTM_7_2 * at1;
        at32 = CRTM_7_4 * at3;
        at33 = CRTM_7_6 * at5;
        at34 = at31 - at32;

        *out = at6 + at7;                       // Output pt 1: X(0)
        out[out_strides[3]]  = at13 + at14;     // Output pt 4: X(3)
        out[out_strides[4]]  = at15 + at16;     // Output pt 5: X(4)
        out[out_strides[7]]  = at20 + at21;     // Output pt 8: X(7)
        out[out_strides[8]]  = at24 - at28;     // Output pt 9: X(8)
        out[out_strides[11]] = at29 + at30;     // Output pt 12: X(11)
        out[out_strides[12]] = at34 + at33;     // Output pt 13: X(12)

        /* Shifted DFT */
        FLOAT bv0, bv1, bv2, bv3, bv4, bv5, bv6;
        FLOAT bt0, bt1, bt2, bt3, bt4, bt5, bt6, bt7, bt8, bt9,
              bt10, bt11, bt12, bt13, bt14, bt15, bt16, bt17, bt18, bt19,
              bt20, bt21, bt22, bt23, bt24, bt25, bt26, bt27, bt28, bt29,
              bt30, bt31, bt32, bt33, bt34;

        bv0 = in[in_strides[1]];    // Input point 2: x(1)
        bv1 = in[in_strides[3]];    // Input point 4: x(3)
        bv2 = in[in_strides[5]];    // Input point 6: x(5)
        bv3 = in[in_strides[7]];    // Input point 8: x(7)
        bv4 = in[in_strides[9]];    // Input point 10: x(9)
        bv5 = in[in_strides[11]];   // Input point 12: x(11)
        bv6 = in[in_strides[13]];   // Input point 14: x(13)

        bt0 = bv6 + bv1;
        bt1 = bv6 - bv1;
        bt2 = bv5 + bv2;
        bt3 = bv5 - bv2;
        bt4 = bv4 + bv3;
        bt5 = bv4 - bv3;

        bt6 = CRTM_7_1 * bt1;
        bt7 = CRTM_7_3 * bt3;

        bt8  = CRTM_7_5 * bt5;
        bt9  = bv0 - bt6;
        bt10 = bt7 + bt8;
        bt11 = CRTM_7_2 * bt0;
        bt12 = CRTM_7_4 * bt2;
        bt13 = CRTM_7_6 * bt4;
        bt14 = -bt11 - bt12;

        bt15 = CRTM_7_1 * bt3;
        bt16 = CRTM_7_3 * bt5;

        bt17 = CRTM_7_5 * bt1;
        bt18 = bv0 + bt15;
        bt20 = CRTM_7_2 * bt2;
        bt19 = bt16 - bt17;
        bt21 = CRTM_7_4 * bt4;
        bt22 = CRTM_7_6 * bt0;
        bt23 = bt21 - bt20;

        bt24 = CRTM_7_1 * bt5;
        bt25 = CRTM_7_3 * bt1;
        bt26 = CRTM_7_5 * bt3;
        bt27 = bv0 - bt24;
        bt28 = bt25 + bt26;

        bt29 = CRTM_7_2 * bt4;
        bt30 = CRTM_7_4 * bt0;
        bt31 = CRTM_7_6 * bt2;
        bt32 = bt29 + bt30;
        bt33 = bv0 + bt1;
        bt34 = bt5 - bt3;

        out[out_strides[1]]  = bt9 - bt10;     // Output pt 2: X(1)
        out[out_strides[2]]  = bt14 - bt13;    // Output pt 3: X(2)
        out[out_strides[5]]  = bt18 + bt19;    // Output pt 6: X(5)
        out[out_strides[6]]  = bt23 - bt22;    // Output pt 7: X(6)
        out[out_strides[9]]  = bt27 + bt28;    // Output pt 10: X(9)
        out[out_strides[10]] = bt31 - bt32;    // Output pt 11: X(10)
        out[out_strides[13]] = bt33 + bt34;    // Output pt 14: X(13)
    }
}

static VOID r2hcf_rfft7avx128_fp32_bwd(VOID *in_real, VOID *in_imag,
                                       VOID *out_real, VOID *out_imag, INTP n,
                                       aoclfftz_strides_t *strides, UINT8 flag)
{
    const FLOAT CRTM_7_1 = 1.801937735804838252472204639014890102331838324f;
    const FLOAT CRTM_7_2 = 0.867767478235116240951536665696717509219981456f;
    const FLOAT CRTM_7_3 = 1.246979603717467061050009768008479621264549462f;
    const FLOAT CRTM_7_4 = 1.563662964936059617416889053348115500464669038f;
    const FLOAT CRTM_7_5 = 0.445041867912628808577805128993589518932711138f;
    const FLOAT CRTM_7_6 = 1.949855824363647214036263365987862434465571602f;
    const FLOAT CRTM_7_7 = 2.000000000000000000000000000000000000000000000f;

    FLOAT *in = (FLOAT *)in_real;
    FLOAT *out = (FLOAT *)out_real;
    #ifdef VOLATILE_STRIDE_ARRAY
    volatile INTP *in_strides = strides->in_strides;
    volatile INTP *out_strides = strides->out_strides;
    #else
    INTP *in_strides = strides->in_strides;
    INTP *out_strides = strides->out_strides;
    #endif
    INTP v_in_stride = (strides->v_in_stride);
    INTP v_out_stride = (strides->v_out_stride);

    INTP cnt;
    INTP N = n >> 2;
    FLOAT *curr_in, *curr_out;

    __m128 v_CRTM_7_1 = _mm_set1_ps(CRTM_7_1);
    __m128 v_CRTM_7_2 = _mm_set1_ps(CRTM_7_2);
    __m128 v_CRTM_7_3 = _mm_set1_ps(CRTM_7_3);
    __m128 v_CRTM_7_4 = _mm_set1_ps(CRTM_7_4);
    __m128 v_CRTM_7_5 = _mm_set1_ps(CRTM_7_5);
    __m128 v_CRTM_7_6 = _mm_set1_ps(CRTM_7_6);
    __m128 v_CRTM_7_7 = _mm_set1_ps(CRTM_7_7);

    for (cnt = 0; cnt < N; cnt++)
    {
        __m128 av_in0, av_in1, av_in2, av_in3, av_in4, av_in5, av_in6;
        __m128 av_tv1, av_tv2, av_tv3, av_sv1, av_sv2, av_sv3, av_tv4, av_tv5,
               av_tv6, av_sv4, av_sv5, av_tv7, av_tv8, av_tv9, av_sv6, av_sv7,
               av_sv8, av_tv10, av_tv11, av_tv12, av_sv9, av_sv10, av_tv13,
               av_tv14, av_tv15, av_sv11, av_sv12, av_sv13, av_tv16, av_tv17,
               av_tv18, av_sv14, av_sv15, av_sv16, av_sv17, av_tv19;
        __m128 v_out0, v_out1, v_out2, v_out3, v_out4, v_out5, v_out6, v_out7,
               v_out8, v_out9, v_out10, v_out11, v_out12, v_out13;

        curr_in = in;
        curr_out = out;

        // Input point 1: x(0)
        LDR_128_S(curr_in, v_in_stride, av_in0);
        // Input point 4: x(3) & Input point 5: x(4)
        curr_in = in + in_strides[3];
        LDRI_2x128_S(curr_in, v_in_stride, av_in1, av_in2);
        // Input point 8: x(7) & Input point 9: x(8)
        curr_in = in + in_strides[7];
        LDRI_2x128_S(curr_in, v_in_stride, av_in3, av_in4);
        // Input point 12: x(11) & Input point 13: x(12)
        curr_in = in + in_strides[11];
        LDRI_2x128_S(curr_in, v_in_stride, av_in5, av_in6);

        av_tv1 = _mm_mul_ps(v_CRTM_7_1, av_in5);
        av_tv2 = _mm_mul_ps(v_CRTM_7_3, av_in1);
        av_tv3 = _mm_mul_ps(v_CRTM_7_5, av_in3);
        av_sv1 = _mm_sub_ps(av_in0, av_tv1);
        av_sv2 = _mm_sub_ps(av_tv2, av_tv3);
        av_sv3 = _mm_add_ps(av_sv1, av_sv2);

        av_tv4 = _mm_mul_ps(v_CRTM_7_2, av_in6);
        av_tv5 = _mm_mul_ps(v_CRTM_7_4, av_in2);
        av_tv6 = _mm_mul_ps(v_CRTM_7_6, av_in4);
        av_sv4 = _mm_add_ps(av_tv4, av_tv5);
        av_sv5 = _mm_add_ps(av_sv4, av_tv6);

        av_tv7 = _mm_mul_ps(v_CRTM_7_1, av_in3);
        av_tv8 = _mm_mul_ps(v_CRTM_7_3, av_in5);
        av_tv9 = _mm_mul_ps(v_CRTM_7_5, av_in1);
        av_sv6 = _mm_sub_ps(av_in0, av_tv7);
        av_sv7 = _mm_sub_ps(av_tv8, av_tv9);
        av_sv8 = _mm_add_ps(av_sv6, av_sv7);

        av_tv10 = _mm_mul_ps(v_CRTM_7_2, av_in4);
        av_tv11 = _mm_mul_ps(v_CRTM_7_4, av_in6);
        av_tv12 = _mm_mul_ps(v_CRTM_7_6, av_in2);
        av_sv9 = _mm_add_ps(av_tv10, av_tv11);
        av_sv10 = _mm_sub_ps(av_tv12, av_sv9);

        av_tv13 = _mm_mul_ps(v_CRTM_7_1, av_in1);
        av_tv14 = _mm_mul_ps(v_CRTM_7_3, av_in3);
        av_tv15 = _mm_mul_ps(v_CRTM_7_5, av_in5);
        av_sv11 = _mm_sub_ps(av_in0, av_tv13);
        av_sv12 = _mm_sub_ps(av_tv14, av_tv15);
        av_sv13 = _mm_add_ps(av_sv11, av_sv12);

        av_tv16 = _mm_mul_ps(v_CRTM_7_2, av_in2);
        av_tv17 = _mm_mul_ps(v_CRTM_7_4, av_in4);
        av_tv18 = _mm_mul_ps(v_CRTM_7_6, av_in6);
        av_sv14 = _mm_sub_ps(av_tv16, av_tv17);
        av_sv15 = _mm_add_ps(av_sv14, av_tv18);

        av_sv16 = _mm_add_ps(av_in1, av_in3);
        av_sv17 = _mm_add_ps(av_sv16, av_in5);
        av_tv19 = _mm_mul_ps(v_CRTM_7_7, av_sv17);

        // Output pt 1: X(0)
        v_out0 = _mm_add_ps(av_in0, av_tv19);
        STR_128_S(curr_out, v_out_stride, v_out0);
        // Output pt 3: X(2)
        curr_out = out + out_strides[2];
        v_out2 = _mm_sub_ps(av_sv3, av_sv5);
        STR_128_S(curr_out, v_out_stride, v_out2);
        // Output pt 5: X(4)
        curr_out = out + out_strides[4];
        v_out4 = _mm_sub_ps(av_sv8, av_sv10);
        STR_128_S(curr_out, v_out_stride, v_out4);
        // Output pt 7: X(6)
        curr_out = out + out_strides[6];
        v_out6 = _mm_sub_ps(av_sv13, av_sv15);
        STR_128_S(curr_out, v_out_stride, v_out6);
        // Output pt 9: X(8)
        curr_out = out + out_strides[8];
        v_out8 = _mm_add_ps(av_sv13, av_sv15);
        STR_128_S(curr_out, v_out_stride, v_out8);
        // Output pt 11: X(10)
        curr_out = out + out_strides[10];
        v_out10 = _mm_add_ps(av_sv8, av_sv10);
        STR_128_S(curr_out, v_out_stride, v_out10);
        // Output pt 13: X(12)
        curr_out = out + out_strides[12];
        v_out12 = _mm_add_ps(av_sv3, av_sv5);
        STR_128_S(curr_out, v_out_stride, v_out12);

        __m128 bv_in0, bv_in1, bv_in2, bv_in3, bv_in4, bv_in5, bv_in6;
        __m128 bv_tv1, bv_tv2, bv_tv3, bv_sv1, bv_sv2, bv_sv3, bv_tv4, bv_tv5,
               bv_tv6, bv_sv4, bv_sv5, bv_tv7, bv_tv8, bv_tv9, bv_sv6, bv_sv7,
               bv_sv8, bv_tv10, bv_tv11, bv_tv12, bv_sv9, bv_sv10, bv_tv13,
               bv_tv14, bv_tv15, bv_sv11, bv_sv12, bv_sv13, bv_tv16, bv_tv17,
               bv_tv18, bv_sv14, bv_sv15, bv_sv16, bv_sv17, bv_tv19;

        // Input point 2: x(1) & Input point 3: x(2)
        curr_in = in + in_strides[1];
        LDRI_2x128_S(curr_in, v_in_stride, bv_in0, bv_in1);
        // Input point 6: x(5) & Input point 7: x(6)
        curr_in = in + in_strides[5];
        LDRI_2x128_S(curr_in, v_in_stride, bv_in2, bv_in3);
        // Input point 10: x(9) & Input point 11: x(10)
        curr_in = in + in_strides[9];
        LDRI_2x128_S(curr_in, v_in_stride, bv_in4, bv_in5);
        // Input point 14: x(13)
        curr_in = in + in_strides[13];
        LDR_128_S(curr_in, v_in_stride, bv_in6);

        bv_tv1 = _mm_mul_ps(v_CRTM_7_1, bv_in0);
        bv_tv2 = _mm_mul_ps(v_CRTM_7_3, bv_in4);
        bv_tv3 = _mm_mul_ps(v_CRTM_7_5, bv_in2);
        bv_sv1 = _mm_sub_ps(bv_in6, bv_tv1);
        bv_sv2 = _mm_sub_ps(bv_tv2, bv_tv3);
        bv_sv3 = _mm_add_ps(bv_sv1, bv_sv2);

        bv_tv4 = _mm_mul_ps(v_CRTM_7_2, bv_in1);
        bv_tv5 = _mm_mul_ps(v_CRTM_7_4, bv_in5);
        bv_tv6 = _mm_mul_ps(v_CRTM_7_6, bv_in3);
        bv_sv4 = _mm_add_ps(bv_tv4, bv_tv5);
        bv_sv5 = _mm_add_ps(bv_sv4, bv_tv6);

        bv_tv7 = _mm_mul_ps(v_CRTM_7_1, bv_in2);
        bv_tv8 = _mm_mul_ps(v_CRTM_7_3, bv_in0);
        bv_tv9 = _mm_mul_ps(v_CRTM_7_5, bv_in4);
        bv_sv6 = _mm_sub_ps(bv_in6, bv_tv7);
        bv_sv7 = _mm_sub_ps(bv_tv8, bv_tv9);
        bv_sv8 = _mm_add_ps(bv_sv6, bv_sv7);

        bv_tv10 = _mm_mul_ps(v_CRTM_7_2, bv_in3);
        bv_tv11 = _mm_mul_ps(v_CRTM_7_4, bv_in1);
        bv_tv12 = _mm_mul_ps(v_CRTM_7_6, bv_in5);
        bv_sv9 = _mm_add_ps(bv_tv10, bv_tv11);
        bv_sv10 = _mm_sub_ps(bv_tv12, bv_sv9);

        bv_tv13 = _mm_mul_ps(v_CRTM_7_1, bv_in4);
        bv_tv14 = _mm_mul_ps(v_CRTM_7_3, bv_in2);
        bv_tv15 = _mm_mul_ps(v_CRTM_7_5, bv_in0);
        bv_sv11 = _mm_sub_ps(bv_in6, bv_tv13);
        bv_sv12 = _mm_sub_ps(bv_tv14, bv_tv15);
        bv_sv13 = _mm_add_ps(bv_sv11, bv_sv12);

        bv_tv16 = _mm_mul_ps(v_CRTM_7_2, bv_in5);
        bv_tv17 = _mm_mul_ps(v_CRTM_7_4, bv_in3);
        bv_tv18 = _mm_mul_ps(v_CRTM_7_6, bv_in1);
        bv_sv14 = _mm_sub_ps(bv_tv16, bv_tv17);
        bv_sv15 = _mm_add_ps(bv_sv14, bv_tv18);

        bv_sv16 = _mm_add_ps(bv_in0, bv_in2);
        bv_sv17 = _mm_add_ps(bv_sv16, bv_in4);
        bv_tv19 = _mm_mul_ps(v_CRTM_7_7, bv_sv17);
        // Output pt 2: X(1)
        curr_out = out + out_strides[1];
        v_out1 = _mm_add_ps(bv_in6, bv_tv19);
        STR_128_S(curr_out, v_out_stride, v_out1);
        // Output pt 4: X(3)
        curr_out = out + out_strides[3];
        v_out3 = NEGATE_128_S(_mm_add_ps(bv_sv3, bv_sv5));
        STR_128_S(curr_out, v_out_stride, v_out3);
        // Output pt 6: X(5)
        curr_out = out + out_strides[5];
        v_out5 = _mm_add_ps(bv_sv8, bv_sv10);
        STR_128_S(curr_out, v_out_stride, v_out5);
        // Output pt 8: X(7)
        curr_out = out + out_strides[7];
        v_out7 = NEGATE_128_S(_mm_add_ps(bv_sv13, bv_sv15));
        STR_128_S(curr_out, v_out_stride, v_out7);
        // Output pt 10: X(9)
        curr_out = out + out_strides[9];
        v_out9 = _mm_sub_ps(bv_sv13, bv_sv15);
        STR_128_S(curr_out, v_out_stride, v_out9);
        // Output pt 12: X(11)
        curr_out = out + out_strides[11];
        v_out11 = _mm_sub_ps(bv_sv10, bv_sv8);
        STR_128_S(curr_out, v_out_stride, v_out11);
        // Output pt 14: X(13)
        curr_out = out + out_strides[13];
        v_out13 = _mm_sub_ps(bv_sv3, bv_sv5);
        STR_128_S(curr_out, v_out_stride, v_out13);

        in = in + (v_in_stride << 2);
        out = out + (v_out_stride << 2);
    }
    // tail cases
    if (n & 2)
    {
        __m128 av_in0, av_in1, av_in2, av_in3, av_in4, av_in5, av_in6;
        __m128 av_tv1, av_tv2, av_tv3, av_sv1, av_sv2, av_sv3, av_tv4, av_tv5,
               av_tv6, av_sv4, av_sv5, av_tv7, av_tv8, av_tv9, av_sv6, av_sv7,
               av_sv8, av_tv10, av_tv11, av_tv12, av_sv9, av_sv10, av_tv13,
               av_tv14, av_tv15, av_sv11, av_sv12, av_sv13, av_tv16, av_tv17,
               av_tv18, av_sv14, av_sv15, av_sv16, av_sv17, av_tv19;
        __m128 v_out0, v_out1, v_out2, v_out3, v_out4, v_out5, v_out6, v_out7,
               v_out8, v_out9, v_out10, v_out11, v_out12, v_out13;

        curr_in = in;
        curr_out = out;

        // Input point 1: x(0)
        LDHR_128_S(curr_in, v_in_stride, av_in0);
        // Input point 4: x(3) & Input point 5: x(4)
        curr_in = in + in_strides[3];
        LDHRI_2x128_S(curr_in, v_in_stride, av_in1, av_in2);
        // Input point 8: x(7) & Input point 9: x(8)
        curr_in = in + in_strides[7];
        LDHRI_2x128_S(curr_in, v_in_stride, av_in3, av_in4);
        // Input point 12: x(11) & Input point 13: x(12)
        curr_in = in + in_strides[11];
        LDHRI_2x128_S(curr_in, v_in_stride, av_in5, av_in6);

        av_tv1 = _mm_mul_ps(v_CRTM_7_1, av_in5);
        av_tv2 = _mm_mul_ps(v_CRTM_7_3, av_in1);
        av_tv3 = _mm_mul_ps(v_CRTM_7_5, av_in3);
        av_sv1 = _mm_sub_ps(av_in0, av_tv1);
        av_sv2 = _mm_sub_ps(av_tv2, av_tv3);
        av_sv3 = _mm_add_ps(av_sv1, av_sv2);

        av_tv4 = _mm_mul_ps(v_CRTM_7_2, av_in6);
        av_tv5 = _mm_mul_ps(v_CRTM_7_4, av_in2);
        av_tv6 = _mm_mul_ps(v_CRTM_7_6, av_in4);
        av_sv4 = _mm_add_ps(av_tv4, av_tv5);
        av_sv5 = _mm_add_ps(av_sv4, av_tv6);

        av_tv7 = _mm_mul_ps(v_CRTM_7_1, av_in3);
        av_tv8 = _mm_mul_ps(v_CRTM_7_3, av_in5);
        av_tv9 = _mm_mul_ps(v_CRTM_7_5, av_in1);
        av_sv6 = _mm_sub_ps(av_in0, av_tv7);
        av_sv7 = _mm_sub_ps(av_tv8, av_tv9);
        av_sv8 = _mm_add_ps(av_sv6, av_sv7);

        av_tv10 = _mm_mul_ps(v_CRTM_7_2, av_in4);
        av_tv11 = _mm_mul_ps(v_CRTM_7_4, av_in6);
        av_tv12 = _mm_mul_ps(v_CRTM_7_6, av_in2);
        av_sv9 = _mm_add_ps(av_tv10, av_tv11);
        av_sv10 = _mm_sub_ps(av_tv12, av_sv9);

        av_tv13 = _mm_mul_ps(v_CRTM_7_1, av_in1);
        av_tv14 = _mm_mul_ps(v_CRTM_7_3, av_in3);
        av_tv15 = _mm_mul_ps(v_CRTM_7_5, av_in5);
        av_sv11 = _mm_sub_ps(av_in0, av_tv13);
        av_sv12 = _mm_sub_ps(av_tv14, av_tv15);
        av_sv13 = _mm_add_ps(av_sv11, av_sv12);

        av_tv16 = _mm_mul_ps(v_CRTM_7_2, av_in2);
        av_tv17 = _mm_mul_ps(v_CRTM_7_4, av_in4);
        av_tv18 = _mm_mul_ps(v_CRTM_7_6, av_in6);
        av_sv14 = _mm_sub_ps(av_tv16, av_tv17);
        av_sv15 = _mm_add_ps(av_sv14, av_tv18);

        av_sv16 = _mm_add_ps(av_in1, av_in3);
        av_sv17 = _mm_add_ps(av_sv16, av_in5);
        av_tv19 = _mm_mul_ps(v_CRTM_7_7, av_sv17);

        // Output pt 1: X(0)
        v_out0 = _mm_add_ps(av_in0, av_tv19);
        STHR_128_S(curr_out, v_out_stride, v_out0);
        // Output pt 3: X(2)
        curr_out = out + out_strides[2];
        v_out2 = _mm_sub_ps(av_sv3, av_sv5);
        STHR_128_S(curr_out, v_out_stride, v_out2);
        // Output pt 5: X(4)
        curr_out = out + out_strides[4];
        v_out4 = _mm_sub_ps(av_sv8, av_sv10);
        STHR_128_S(curr_out, v_out_stride, v_out4);
        // Output pt 7: X(6)
        curr_out = out + out_strides[6];
        v_out6 = _mm_sub_ps(av_sv13, av_sv15);
        STHR_128_S(curr_out, v_out_stride, v_out6);
        // Output pt 9: X(8)
        curr_out = out + out_strides[8];
        v_out8 = _mm_add_ps(av_sv13, av_sv15);
        STHR_128_S(curr_out, v_out_stride, v_out8);
        // Output pt 11: X(10)
        curr_out = out + out_strides[10];
        v_out10 = _mm_add_ps(av_sv8, av_sv10);
        STHR_128_S(curr_out, v_out_stride, v_out10);
        // Output pt 13: X(12)
        curr_out = out + out_strides[12];
        v_out12 = _mm_add_ps(av_sv3, av_sv5);
        STHR_128_S(curr_out, v_out_stride, v_out12);

        __m128 bv_in0, bv_in1, bv_in2, bv_in3, bv_in4, bv_in5, bv_in6;
        __m128 bv_tv1, bv_tv2, bv_tv3, bv_sv1, bv_sv2, bv_sv3, bv_tv4, bv_tv5,
               bv_tv6, bv_sv4, bv_sv5, bv_tv7, bv_tv8, bv_tv9, bv_sv6, bv_sv7,
               bv_sv8, bv_tv10, bv_tv11, bv_tv12, bv_sv9, bv_sv10, bv_tv13,
               bv_tv14, bv_tv15, bv_sv11, bv_sv12, bv_sv13, bv_tv16, bv_tv17,
               bv_tv18, bv_sv14, bv_sv15, bv_sv16, bv_sv17, bv_tv19;

        // Input point 2: x(1) & Input point 3: x(2)
        curr_in = in + in_strides[1];
        LDHRI_2x128_S(curr_in, v_in_stride, bv_in0, bv_in1);
        // Input point 6: x(5) & Input point 7: x(6)
        curr_in = in + in_strides[5];
        LDHRI_2x128_S(curr_in, v_in_stride, bv_in2, bv_in3);
        // Input point 10: x(9) & Input point 11: x(10)
        curr_in = in + in_strides[9];
        LDHRI_2x128_S(curr_in, v_in_stride, bv_in4, bv_in5);
        // Input point 14: x(13)
        curr_in = in + in_strides[13];
        LDHR_128_S(curr_in, v_in_stride, bv_in6);

        bv_tv1 = _mm_mul_ps(v_CRTM_7_1, bv_in0);
        bv_tv2 = _mm_mul_ps(v_CRTM_7_3, bv_in4);
        bv_tv3 = _mm_mul_ps(v_CRTM_7_5, bv_in2);
        bv_sv1 = _mm_sub_ps(bv_in6, bv_tv1);
        bv_sv2 = _mm_sub_ps(bv_tv2, bv_tv3);
        bv_sv3 = _mm_add_ps(bv_sv1, bv_sv2);

        bv_tv4 = _mm_mul_ps(v_CRTM_7_2, bv_in1);
        bv_tv5 = _mm_mul_ps(v_CRTM_7_4, bv_in5);
        bv_tv6 = _mm_mul_ps(v_CRTM_7_6, bv_in3);
        bv_sv4 = _mm_add_ps(bv_tv4, bv_tv5);
        bv_sv5 = _mm_add_ps(bv_sv4, bv_tv6);

        bv_tv7 = _mm_mul_ps(v_CRTM_7_1, bv_in2);
        bv_tv8 = _mm_mul_ps(v_CRTM_7_3, bv_in0);
        bv_tv9 = _mm_mul_ps(v_CRTM_7_5, bv_in4);
        bv_sv6 = _mm_sub_ps(bv_in6, bv_tv7);
        bv_sv7 = _mm_sub_ps(bv_tv8, bv_tv9);
        bv_sv8 = _mm_add_ps(bv_sv6, bv_sv7);

        bv_tv10 = _mm_mul_ps(v_CRTM_7_2, bv_in3);
        bv_tv11 = _mm_mul_ps(v_CRTM_7_4, bv_in1);
        bv_tv12 = _mm_mul_ps(v_CRTM_7_6, bv_in5);
        bv_sv9 = _mm_add_ps(bv_tv10, bv_tv11);
        bv_sv10 = _mm_sub_ps(bv_tv12, bv_sv9);

        bv_tv13 = _mm_mul_ps(v_CRTM_7_1, bv_in4);
        bv_tv14 = _mm_mul_ps(v_CRTM_7_3, bv_in2);
        bv_tv15 = _mm_mul_ps(v_CRTM_7_5, bv_in0);
        bv_sv11 = _mm_sub_ps(bv_in6, bv_tv13);
        bv_sv12 = _mm_sub_ps(bv_tv14, bv_tv15);
        bv_sv13 = _mm_add_ps(bv_sv11, bv_sv12);

        bv_tv16 = _mm_mul_ps(v_CRTM_7_2, bv_in5);
        bv_tv17 = _mm_mul_ps(v_CRTM_7_4, bv_in3);
        bv_tv18 = _mm_mul_ps(v_CRTM_7_6, bv_in1);
        bv_sv14 = _mm_sub_ps(bv_tv16, bv_tv17);
        bv_sv15 = _mm_add_ps(bv_sv14, bv_tv18);

        bv_sv16 = _mm_add_ps(bv_in0, bv_in2);
        bv_sv17 = _mm_add_ps(bv_sv16, bv_in4);
        bv_tv19 = _mm_mul_ps(v_CRTM_7_7, bv_sv17);

        // Output pt 2: X(1)
        curr_out = out + out_strides[1];
        v_out1 = _mm_add_ps(bv_in6, bv_tv19);
        STHR_128_S(curr_out, v_out_stride, v_out1);
        // Output pt 4: X(3)
        curr_out = out + out_strides[3];
        v_out3 = NEGATE_128_S(_mm_add_ps(bv_sv3, bv_sv5));
        STHR_128_S(curr_out, v_out_stride, v_out3);
        // Output pt 6: X(5)
        curr_out = out + out_strides[5];
        v_out5 = _mm_add_ps(bv_sv8, bv_sv10);
        STHR_128_S(curr_out, v_out_stride, v_out5);
        // Output pt 8: X(7)
        curr_out = out + out_strides[7];
        v_out7 = NEGATE_128_S(_mm_add_ps(bv_sv13, bv_sv15));
        STHR_128_S(curr_out, v_out_stride, v_out7);
        // Output pt 10: X(9)
        curr_out = out + out_strides[9];
        v_out9 = _mm_sub_ps(bv_sv13, bv_sv15);
        STHR_128_S(curr_out, v_out_stride, v_out9);
        // Output pt 12: X(11)
        curr_out = out + out_strides[11];
        v_out11 = _mm_sub_ps(bv_sv10, bv_sv8);
        STHR_128_S(curr_out, v_out_stride, v_out11);
        // Output pt 14: X(13)
        curr_out = out + out_strides[13];
        v_out13 = _mm_sub_ps(bv_sv3, bv_sv5);
        STHR_128_S(curr_out, v_out_stride, v_out13);

        in = in + (v_in_stride << 1);
        out = out + (v_out_stride << 1);
    }
    // tail cases
    if (n & 1)
    {
        /* Standard DFT */
        FLOAT av0, av1, av2, av3, av4, av5, av6;
        FLOAT at0, at1, at2, at3, at4, at5, at6, at7, at8, at9,
              at10, at11, at12, at13, at14, at15, at16, at17, at18, at19,
              at20, at21, at22, at23, at24, at25, at26, at27, at28, at29,
              at30, at31, at32, at33, at34, at35;

        av0 = *in;                  // Input point 1: x(0)
        av1 = in[in_strides[3]];    // Input point 4: x(3)
        av2 = in[in_strides[4]];    // Input point 5: x(4)
        av3 = in[in_strides[7]];    // Input point 8: x(7)
        av4 = in[in_strides[8]];    // Input point 9: x(8)
        av5 = in[in_strides[11]];   // Input point 12: x(11)
        av6 = in[in_strides[12]];   // Input point 13: x(12)

        at0 = CRTM_7_1 * av5;
        at1 = CRTM_7_3 * av1;
        at2 = CRTM_7_5 * av3;
        at3 = av0 - at0;
        at4 = at1 - at2;

        at5 = CRTM_7_2 * av6;
        at6 = CRTM_7_4 * av2;
        at7 = CRTM_7_6 * av4;
        at8 = at5 + at6;
        at9 = at3 + at4;
        at10 = at8 + at7;

        at11 = CRTM_7_1 * av3;
        at12 = CRTM_7_3 * av5;
        at13 = CRTM_7_5 * av1;
        at14 = av0 - at11;
        at15 = at12 - at13;

        at16 = CRTM_7_2 * av4;
        at17 = CRTM_7_4 * av6;
        at18 = CRTM_7_6 * av2;
        at19 = at16 + at17;

        at20 = at14 + at15;
        at21 = at18 - at19;

        at22 = CRTM_7_1 * av1;
        at23 = CRTM_7_3 * av3;
        at24 = CRTM_7_5 * av5;
        at25 = av0 - at22;
        at26 = at23 - at24;

        at27 = CRTM_7_2 * av2;
        at28 = CRTM_7_4 * av4;
        at29 = CRTM_7_6 * av6;
        at30 = at27 - at28;

        at31 = av1 + av3;
        at32 = at25 + at26;
        at33 = at31 + av5;
        at34 = at30 + at29;
        at35 = CRTM_7_7 * at33;

        *out = av0 + at35;                     // Output pt 1: X(0)
        out[out_strides[2]]  = at9 - at10;     // Output pt 3: X(2)
        out[out_strides[4]]  = at20 - at21;    // Output pt 5: X(4)
        out[out_strides[6]]  = at32 - at34;    // Output pt 7: X(6)
        out[out_strides[8]]  = at32 + at34;    // Output pt 9: X(8)
        out[out_strides[10]] = at20 + at21;    // Output pt 11: X(10)
        out[out_strides[12]] = at9 + at10;     // Output pt 13: X(12)

        /* Shifted DFT */
        FLOAT bv0, bv1, bv2, bv3, bv4, bv5, bv6;
        FLOAT bt0, bt1, bt2, bt3, bt4, bt5, bt6, bt7, bt8, bt9,
              bt10, bt11, bt12, bt13, bt14, bt15, bt16, bt17, bt18, bt19,
              bt20, bt21, bt22, bt23, bt24, bt25, bt26, bt27, bt28, bt29,
              bt30, bt31, bt32, bt33, bt34, bt35;

        bv0 = in[in_strides[1]];    // Input point 2: x(1)
        bv1 = in[in_strides[2]];    // Input point 3: x(2)
        bv2 = in[in_strides[5]];    // Input point 6: x(5)
        bv3 = in[in_strides[6]];    // Input point 7: x(6)
        bv4 = in[in_strides[9]];    // Input point 10: x(9)
        bv5 = in[in_strides[10]];   // Input point 11: x(10)
        bv6 = in[in_strides[13]];   // Input point 14: x(13)

        bt0 = CRTM_7_1 * bv0;
        bt1 = CRTM_7_3 * bv4;
        bt2 = CRTM_7_5 * bv2;
        bt3 = bv6 - bt0;
        bt4 = bt1 - bt2;
        bt5 = bt3 + bt4;

        bt6 = CRTM_7_2 * bv1;
        bt7 = CRTM_7_4 * bv5;
        bt8 = CRTM_7_6 * bv3;
        bt9 = bt6 + bt7;

        bt10 = bt9 + bt8;
        bt11 = CRTM_7_1 * bv2;
        bt12 = CRTM_7_3 * bv0;
        bt13 = CRTM_7_5 * bv4;
        bt14 = bv6 - bt11;
        bt15 = bt12 - bt13;

        bt16 = CRTM_7_2 * bv3;
        bt17 = CRTM_7_4 * bv1;
        bt18 = CRTM_7_6 * bv5;
        bt19 = bt16 + bt17;

        bt20 = bt14 + bt15;
        bt21 = bt18 - bt19;

        bt22 = CRTM_7_1 * bv4;
        bt23 = CRTM_7_3 * bv2;
        bt24 = CRTM_7_5 * bv0;
        bt25 = bv6 - bt22;
        bt26 = bt23 - bt24;

        bt27 = CRTM_7_2 * bv5;
        bt28 = CRTM_7_4 * bv3;
        bt29 = CRTM_7_6 * bv1;
        bt30 = bt27 - bt28;

        bt31 = bv0 + bv2;
        bt32 = bt25 + bt26;
        bt33 = bt31 + bv4;
        bt34 = bt30 + bt29;
        bt35 = bt33 * CRTM_7_7;

        out[out_strides[1]]  = bv6 + bt35;     // Output pt 2: X(1)
        out[out_strides[3]]  = -bt5 - bt10;    // Output pt 4: X(3)
        out[out_strides[5]]  = bt20 + bt21;    // Output pt 6: X(5)
        out[out_strides[7]]  = -bt32 - bt34;   // Output pt 8: X(7)
        out[out_strides[9]]  = bt32 - bt34;    // Output pt 10: X(9)
        out[out_strides[11]] = bt21 - bt20;    // Output pt 12: X(11)
        out[out_strides[13]] = bt5 - bt10;     // Output pt 14: X(13)
   }
}
static VOID r2hcf_rfft7avx128_fp64_fwd(VOID *in_real, VOID *in_imag,
                                       VOID *out_real, VOID *out_imag, INTP n,
                                       aoclfftz_strides_t *strides, UINT8 flag)
{
    const DOUBLE CRTM_7_1 = 0.900968867902419126236102319507445051165919162;
    const DOUBLE CRTM_7_2 = 0.433883739117558120475768332848358754609990728;
    const DOUBLE CRTM_7_3 = 0.623489801858733530525004884004239810632274731;
    const DOUBLE CRTM_7_4 = 0.781831482468029808708444526674057750232334519;
    const DOUBLE CRTM_7_5 = 0.222520933956314404288902564496794759466355569;
    const DOUBLE CRTM_7_6 = 0.974927912181823607018131682993931217232785801;

    DOUBLE *in = (DOUBLE *)in_real;
    DOUBLE *out = (DOUBLE *)out_real;
    #ifdef VOLATILE_STRIDE_ARRAY
    volatile INTP *in_strides = strides->in_strides;
    volatile INTP *out_strides = strides->out_strides;
    #else
    INTP *in_strides = strides->in_strides;
    INTP *out_strides = strides->out_strides;
    #endif
    INTP v_in_stride = (strides->v_in_stride);
    INTP v_out_stride = (strides->v_out_stride);

    INTP cnt;
    INTP N = n >> 1;
    DOUBLE *curr_in, *curr_out;

    __m128d v_CRTM_7_1 = _mm_set1_pd(CRTM_7_1);
    __m128d v_CRTM_7_2 = _mm_set1_pd(CRTM_7_2);
    __m128d v_CRTM_7_3 = _mm_set1_pd(CRTM_7_3);
    __m128d v_CRTM_7_4 = _mm_set1_pd(CRTM_7_4);
    __m128d v_CRTM_7_5 = _mm_set1_pd(CRTM_7_5);
    __m128d v_CRTM_7_6 = _mm_set1_pd(CRTM_7_6);

    for (cnt = 0; cnt < N; cnt++)
    {
        __m128d av_in0, av_in1, av_in2, av_in3, av_in4, av_in5, av_in6;
        __m128d av_sv1, av_sv2, av_sv3, av_sv4, av_sv5, av_sv6, av_sv7, av_sv8,
                av_tv1, av_tv2, av_tv3, av_sv9, av_sv10, av_tv4, av_tv5, av_tv6,
                av_sv11, av_tv7, av_tv8, av_tv9, av_sv12, av_sv13, av_tv10,
                av_tv11, av_tv12, av_sv14, av_tv13, av_tv14, av_tv15, av_sv15,
                av_sv16, av_tv16, av_tv17, av_tv18, av_sv17;
        __m128d v_out0, v_out1, v_out2, v_out3, v_out4, v_out5, v_out6, v_out7,
                v_out8, v_out9, v_out10, v_out11, v_out12, v_out13;

        curr_in = in;
        curr_out = out;

        // Input point 1: x(0)
        LDR_128_D(curr_in, v_in_stride, av_in0);
        // Input point 3: x(2)
        curr_in = in + in_strides[2];
        LDR_128_D(curr_in, v_in_stride, av_in1);
        // Input point 5: x(4)
        curr_in = in + in_strides[4];
        LDR_128_D(curr_in, v_in_stride, av_in2);
        // Input point 7: x(6)
        curr_in = in + in_strides[6];
        LDR_128_D(curr_in, v_in_stride, av_in3);
        // Input point 9: x(8)
        curr_in = in + in_strides[8];
        LDR_128_D(curr_in, v_in_stride, av_in4);
        // Input point 11: x(10)
        curr_in = in + in_strides[10];
        LDR_128_D(curr_in, v_in_stride, av_in5);
        // Input point 13: x(12)
        curr_in = in + in_strides[12];
        LDR_128_D(curr_in, v_in_stride, av_in6);

        av_sv1 = _mm_add_pd(av_in6, av_in1);
        av_sv2 = _mm_sub_pd(av_in6, av_in1);
        av_sv3 = _mm_add_pd(av_in5, av_in2);
        av_sv4 = _mm_sub_pd(av_in5, av_in2);
        av_sv5 = _mm_add_pd(av_in4, av_in3);
        av_sv6 = _mm_sub_pd(av_in4, av_in3);

        av_sv7 = _mm_add_pd(av_in0, av_sv1);
        av_sv8 = _mm_add_pd(av_sv3, av_sv5);

        av_tv1 = _mm_mul_pd(v_CRTM_7_1, av_sv5);
        av_tv2 = _mm_mul_pd(v_CRTM_7_3, av_sv1);
        av_tv3 = _mm_mul_pd(v_CRTM_7_5, av_sv3);
        av_sv9 = _mm_sub_pd(av_in0, av_tv1);

        av_sv10 = _mm_sub_pd(av_tv2, av_tv3);
        av_tv4 = _mm_mul_pd(v_CRTM_7_2, av_sv6);
        av_tv5 = _mm_mul_pd(v_CRTM_7_4, av_sv2);

        av_tv6 = _mm_mul_pd(v_CRTM_7_6, av_sv4);
        av_sv11 = _mm_add_pd(av_tv4, av_tv5);

        av_tv7 = _mm_mul_pd(v_CRTM_7_1, av_sv3);
        av_tv8 = _mm_mul_pd(v_CRTM_7_3, av_sv5);
        av_tv9 = _mm_mul_pd(v_CRTM_7_5, av_sv1);

        av_sv12 = _mm_sub_pd(av_in0, av_tv7);
        av_sv13 = _mm_sub_pd(av_tv8, av_tv9);
        av_tv10 = _mm_mul_pd(v_CRTM_7_2, av_sv4);
        av_tv11 = _mm_mul_pd(v_CRTM_7_4, av_sv6);

        av_tv12 = _mm_mul_pd(v_CRTM_7_6, av_sv2);
        av_sv14 = _mm_add_pd(av_tv10, av_tv11);
        av_tv13 = _mm_mul_pd(v_CRTM_7_1, av_sv1);
        av_tv14 = _mm_mul_pd(v_CRTM_7_3, av_sv3);
        av_tv15 = _mm_mul_pd(v_CRTM_7_5, av_sv5);

        av_sv15 = _mm_sub_pd(av_in0, av_tv13);
        av_sv16 = _mm_sub_pd(av_tv14, av_tv15);
        av_tv16 = _mm_mul_pd(v_CRTM_7_2, av_sv2);
        av_tv17 = _mm_mul_pd(v_CRTM_7_4, av_sv4);
        av_tv18 = _mm_mul_pd(v_CRTM_7_6, av_sv6);
        av_sv17 = _mm_sub_pd(av_tv16, av_tv17);

        // Output pt 1: X(0)
        v_out0 = _mm_add_pd(av_sv7, av_sv8);
        STR_128_D(curr_out, v_out_stride, v_out0);
        // Output pt 4: X(3) & Output pt 5: X(4)
        curr_out = out + out_strides[3];
        v_out3 = _mm_add_pd(av_sv9, av_sv10);
        v_out4 = _mm_add_pd(av_tv6, av_sv11);
        STRI_2x128_D(curr_out, v_out_stride, v_out3, v_out4);
        // Output pt 8: X(7) & Output pt 9: X(8)
        curr_out = out + out_strides[7];
        v_out7 = _mm_add_pd(av_sv12, av_sv13);
        v_out8 = _mm_sub_pd(av_tv12, av_sv14);
        STRI_2x128_D(curr_out, v_out_stride, v_out7, v_out8);
        // Output pt 12: X(11) & Output pt 13: X(12)
        curr_out = out + out_strides[11];
        v_out11 = _mm_add_pd(av_sv15, av_sv16);
        v_out12 = _mm_add_pd(av_sv17, av_tv18);
        STRI_2x128_D(curr_out, v_out_stride, v_out11, v_out12);

        __m128d bv_in0, bv_in1, bv_in2, bv_in3, bv_in4, bv_in5, bv_in6;
        __m128d bv_sv1, bv_sv2, bv_sv3, bv_sv4, bv_sv5, bv_sv6, bv_tv1, bv_tv2,
                bv_tv3, bv_sv7, bv_sv8, bv_tv4, bv_tv5, bv_tv6, bv_sv9, bv_tv7,
                bv_tv8, bv_tv9, bv_sv10, bv_sv11, bv_tv10, bv_tv11, bv_tv12,
                bv_sv12, bv_tv13, bv_tv14, bv_tv15, bv_sv13, bv_sv14, bv_tv16,
                bv_tv17, bv_tv18, bv_sv15, bv_sv16, bv_sv17;

        // Input point 2: x(1)
        curr_in = in + in_strides[1];
        LDR_128_D(curr_in, v_in_stride, bv_in0);
        // Input point 4: x(3)
        curr_in = in + in_strides[3];
        LDR_128_D(curr_in, v_in_stride, bv_in1);
        // Input point 6: x(5)
        curr_in = in + in_strides[5];
        LDR_128_D(curr_in, v_in_stride, bv_in2);
        // Input point 8: x(7)
        curr_in = in + in_strides[7];
        LDR_128_D(curr_in, v_in_stride, bv_in3);
        // Input point 10: x(9)
        curr_in = in + in_strides[9];
        LDR_128_D(curr_in, v_in_stride, bv_in4);
        // Input point 12: x(11)
        curr_in = in + in_strides[11];
        LDR_128_D(curr_in, v_in_stride, bv_in5);
        // Input point 14: x(13)
        curr_in = in + in_strides[13];
        LDR_128_D(curr_in, v_in_stride, bv_in6);

        bv_sv1 = _mm_add_pd(bv_in6, bv_in1);
        bv_sv2 = _mm_sub_pd(bv_in6, bv_in1);
        bv_sv3 = _mm_add_pd(bv_in5, bv_in2);
        bv_sv4 = _mm_sub_pd(bv_in5, bv_in2);
        bv_sv5 = _mm_add_pd(bv_in4, bv_in3);
        bv_sv6 = _mm_sub_pd(bv_in4, bv_in3);

        bv_tv1 = _mm_mul_pd(v_CRTM_7_1, bv_sv2);
        bv_tv2 = _mm_mul_pd(v_CRTM_7_3, bv_sv4);

        bv_tv3 = _mm_mul_pd(v_CRTM_7_5, bv_sv6);
        bv_sv7 = _mm_sub_pd(bv_in0, bv_tv1);
        bv_sv8 = _mm_add_pd(bv_tv2, bv_tv3);
        bv_tv4 = _mm_mul_pd(v_CRTM_7_2, bv_sv1);

        bv_tv5 = _mm_mul_pd(v_CRTM_7_4, bv_sv3);
        bv_tv6 = _mm_mul_pd(v_CRTM_7_6, bv_sv5);
        bv_sv9 = NEGATE_128_D(_mm_add_pd(bv_tv4, bv_tv5));

        bv_tv7 = _mm_mul_pd(v_CRTM_7_1, bv_sv4);
        bv_tv8 = _mm_mul_pd(v_CRTM_7_3, bv_sv6);
        bv_tv9 = _mm_mul_pd(v_CRTM_7_5, bv_sv2);
        bv_sv10 = _mm_add_pd(bv_in0, bv_tv7);
        bv_sv11 = _mm_sub_pd(bv_tv8, bv_tv9);

        bv_tv10 = _mm_mul_pd(v_CRTM_7_2, bv_sv3);
        bv_tv11 = _mm_mul_pd(v_CRTM_7_4, bv_sv5);

        bv_tv12 = _mm_mul_pd(v_CRTM_7_6, bv_sv1);
        bv_sv12 = _mm_sub_pd(bv_tv11,bv_tv10);

        bv_tv13 = _mm_mul_pd(v_CRTM_7_1, bv_sv6);
        bv_tv14 = _mm_mul_pd(v_CRTM_7_3, bv_sv2);
        bv_tv15 = _mm_mul_pd(v_CRTM_7_5, bv_sv4);
        bv_sv13 = _mm_sub_pd(bv_in0, bv_tv13);
        bv_sv14 = _mm_add_pd(bv_tv14, bv_tv15);

        bv_tv16 = _mm_mul_pd(v_CRTM_7_2, bv_sv5);
        bv_tv17 = _mm_mul_pd(v_CRTM_7_4, bv_sv1);
        bv_tv18 = _mm_mul_pd(v_CRTM_7_6, bv_sv3);
        bv_sv15 = _mm_add_pd(bv_tv16, bv_tv17);
        bv_sv16 = _mm_add_pd(bv_in0, bv_sv2);
        bv_sv17 = _mm_sub_pd(bv_sv6, bv_sv4);

        // Output pt 2: X(1) & Output pt 3: X(2)
        curr_out = out + out_strides[1];
        v_out1 = _mm_sub_pd(bv_sv7, bv_sv8);
        v_out2 = _mm_sub_pd(bv_sv9, bv_tv6);
        STRI_2x128_D(curr_out, v_out_stride, v_out1, v_out2);
        // Output pt 6: X(5) & Output pt 7: X(6)
        curr_out = out + out_strides[5];
        v_out5 = _mm_add_pd(bv_sv10, bv_sv11);
        v_out6 = _mm_sub_pd(bv_sv12, bv_tv12);
        STRI_2x128_D(curr_out, v_out_stride, v_out5, v_out6);
        // Output pt 10: X(9) & Output pt 11: X(10)
        curr_out = out + out_strides[9];
        v_out9 = _mm_add_pd(bv_sv13, bv_sv14);
        v_out10 = _mm_sub_pd(bv_tv18, bv_sv15);
        STRI_2x128_D(curr_out, v_out_stride, v_out9, v_out10);
        // Output pt 14: X(13)
        curr_out = out + out_strides[13];
        v_out13 = _mm_add_pd(bv_sv16, bv_sv17);
        STR_128_D(curr_out, v_out_stride, v_out13);

        in = in + (v_in_stride << 1);
        out = out + (v_out_stride << 1);
    }
    // tail cases
    if (n & 1)
    {
        /* Standard DFT */
        DOUBLE av0, av1, av2, av3, av4, av5, av6;
        DOUBLE at0, at1, at2, at3, at4, at5, at6, at7, at8, at9,
              at10, at11, at12, at13, at14, at15, at16, at17, at18, at19,
              at20, at21, at22, at23, at24, at25, at26, at27, at28, at29,
              at30, at31, at32, at33, at34;

        av0 = *in;                  // Input point 1: x(0)
        av1 = in[in_strides[2]];    // Input point 3: x(2)
        av2 = in[in_strides[4]];    // Input point 5: x(4)
        av3 = in[in_strides[6]];    // Input point 7: x(6)
        av4 = in[in_strides[8]];    // Input point 9: x(8)
        av5 = in[in_strides[10]];   // Input point 11: x(10)
        av6 = in[in_strides[12]];   // Input point 13: x(12)

        at0 = av6 + av1;
        at1 = av6 - av1;
        at2 = av5 + av2;
        at3 = av5 - av2;
        at4 = av4 + av3;
        at5 = av4 - av3;
        at6 = av0 + at0;
        at7 = at2 + at4;

        at8 = CRTM_7_1 * at4;
        at9 = CRTM_7_3 * at0;
        at10 = CRTM_7_5 * at2;
        at11 = CRTM_7_2 * at5;
        at12 = CRTM_7_4 * at1;
        at13 = av0 - at8;
        at14 = at9 - at10;

        at15 = CRTM_7_6 * at3;
        at16 = at11 + at12;
        at17 = CRTM_7_1 * at2;
        at18 = CRTM_7_3 * at4;
        at19 = CRTM_7_5 * at0;

        at20 = av0 - at17;
        at21 = at18 - at19;
        at22 = CRTM_7_2 * at3;
        at23 = CRTM_7_4 * at5;

        at24 = CRTM_7_6 * at1;
        at25 = at22 + at23;
        at26 = CRTM_7_1 * at0;
        at27 = CRTM_7_3 * at2;
        at28 = CRTM_7_5 * at4;

        at29 = av0 - at26;
        at30 = at27 - at28;
        at31 = CRTM_7_2 * at1;
        at32 = CRTM_7_4 * at3;
        at33 = CRTM_7_6 * at5;
        at34 = at31 - at32;

        *out = at6 + at7;                       // Output pt 1: X(0)
        out[out_strides[3]]  = at13 + at14;     // Output pt 4: X(3)
        out[out_strides[4]]  = at15 + at16;     // Output pt 5: X(4)
        out[out_strides[7]]  = at20 + at21;     // Output pt 8: X(7)
        out[out_strides[8]]  = at24 - at25;     // Output pt 9: X(8)
        out[out_strides[11]] = at29 + at30;     // Output pt 12: X(11)
        out[out_strides[12]] = at34 + at33;     // Output pt 13: X(12)

        /* Shifted DFT */
        DOUBLE bv0, bv1, bv2, bv3, bv4, bv5, bv6;
        DOUBLE bt0, bt1, bt2, bt3, bt4, bt5, bt6, bt7, bt8, bt9,
              bt10, bt11, bt12, bt13, bt14, bt15, bt16, bt17, bt18, bt19,
              bt20, bt21, bt22, bt23, bt24, bt25, bt26, bt27, bt28, bt29,
              bt30, bt31, bt32, bt33, bt34;

        bv0 = in[in_strides[1]];    // Input point 2: x(1)
        bv1 = in[in_strides[3]];    // Input point 4: x(3)
        bv2 = in[in_strides[5]];    // Input point 6: x(5)
        bv3 = in[in_strides[7]];    // Input point 8: x(7)
        bv4 = in[in_strides[9]];    // Input point 10: x(9)
        bv5 = in[in_strides[11]];   // Input point 12: x(11)
        bv6 = in[in_strides[13]];   // Input point 14: x(13)

        bt0 = bv6 + bv1;
        bt1 = bv6 - bv1;
        bt2 = bv5 + bv2;
        bt3 = bv5 - bv2;
        bt4 = bv4 + bv3;
        bt5 = bv4 - bv3;

        bt6 = CRTM_7_1 * bt1;
        bt7 = CRTM_7_3 * bt3;

        bt8  = CRTM_7_5 * bt5;
        bt9  = bv0 - bt6;
        bt10 = bt7 + bt8;
        bt11 = CRTM_7_2 * bt0;
        bt12 = CRTM_7_4 * bt2;
        bt13 = CRTM_7_6 * bt4;
        bt14 = -bt11 - bt12;

        bt15 = CRTM_7_1 * bt3;
        bt16 = CRTM_7_3 * bt5;
        bt17 = CRTM_7_5 * bt1;
        bt18 = bv0 + bt15;
        bt19 = bt16 - bt17;
        bt20 = CRTM_7_2 * bt2;
        bt21 = CRTM_7_4 * bt4;
        bt22 = CRTM_7_6 * bt0;
        bt23 = bt21 - bt20;

        bt24 = CRTM_7_1 * bt5;
        bt25 = CRTM_7_3 * bt1;
        bt26 = CRTM_7_5 * bt3;
        bt27 = bv0 - bt24;
        bt28 = bt25 + bt26;

        bt29 = CRTM_7_2 * bt4;
        bt30 = CRTM_7_4 * bt0;
        bt31 = CRTM_7_6 * bt2;
        bt32 = bt29 + bt30;
        bt33 = bv0 + bt1;
        bt34 = bt5 - bt3;

        out[out_strides[1]]  = bt9 - bt10;     // Output pt 2: X(1)
        out[out_strides[2]]  = bt14 - bt13;    // Output pt 3: X(2)
        out[out_strides[5]]  = bt18 + bt19;    // Output pt 6: X(5)
        out[out_strides[6]]  = bt23 - bt22;    // Output pt 7: X(6)
        out[out_strides[9]]  = bt27 + bt28;    // Output pt 10: X(9)
        out[out_strides[10]] = bt31 - bt32;    // Output pt 11: X(10)
        out[out_strides[13]] = bt33 + bt34;    // Output pt 14: X(13)
    }
}

static VOID r2hcf_rfft7avx128_fp64_bwd(VOID *in_real, VOID *in_imag,
                                       VOID *out_real, VOID *out_imag, INTP n,
                                       aoclfftz_strides_t *strides, UINT8 flag)
{
    const DOUBLE CRTM_7_1 = 1.801937735804838252472204639014890102331838324;
    const DOUBLE CRTM_7_2 = 0.867767478235116240951536665696717509219981456;
    const DOUBLE CRTM_7_3 = 1.246979603717467061050009768008479621264549462;
    const DOUBLE CRTM_7_4 = 1.563662964936059617416889053348115500464669038;
    const DOUBLE CRTM_7_5 = 0.445041867912628808577805128993589518932711138;
    const DOUBLE CRTM_7_6 = 1.949855824363647214036263365987862434465571602;
    const DOUBLE CRTM_7_7 = 2.000000000000000000000000000000000000000000000;

    DOUBLE *in = (DOUBLE *)in_real;
    DOUBLE *out = (DOUBLE *)out_real;
#ifdef VOLATILE_STRIDE_ARRAY
    volatile INTP *in_strides = strides->in_strides;
    volatile INTP *out_strides = strides->out_strides;
#else
    INTP *in_strides = strides->in_strides;
    INTP *out_strides = strides->out_strides;
#endif
    INTP v_in_stride = (strides->v_in_stride);
    INTP v_out_stride = (strides->v_out_stride);

    INTP cnt;
    INTP N = n >> 1;
    DOUBLE *curr_in, *curr_out;

    __m128d v_CRTM_7_1 = _mm_set1_pd(CRTM_7_1);
    __m128d v_CRTM_7_2 = _mm_set1_pd(CRTM_7_2);
    __m128d v_CRTM_7_3 = _mm_set1_pd(CRTM_7_3);
    __m128d v_CRTM_7_4 = _mm_set1_pd(CRTM_7_4);
    __m128d v_CRTM_7_5 = _mm_set1_pd(CRTM_7_5);
    __m128d v_CRTM_7_6 = _mm_set1_pd(CRTM_7_6);
    __m128d v_CRTM_7_7 = _mm_set1_pd(CRTM_7_7);

    for (cnt = 0; cnt < N; cnt++)
    {
        __m128d av_in0, av_in1, av_in2, av_in3, av_in4, av_in5, av_in6;
        __m128d av_tv1, av_tv2, av_tv3, av_sv1, av_sv2, av_sv3, av_tv4, av_tv5,
               av_tv6, av_sv4, av_sv5, av_tv7, av_tv8, av_tv9, av_sv6, av_sv7,
               av_sv8, av_tv10, av_tv11, av_tv12, av_sv9, av_sv10, av_tv13,
               av_tv14, av_tv15, av_sv11, av_sv12, av_sv13, av_tv16, av_tv17,
               av_tv18, av_sv14, av_sv15, av_sv16, av_sv17, av_tv19;
        __m128d v_out0, v_out1, v_out2, v_out3, v_out4, v_out5, v_out6, v_out7,
               v_out8, v_out9, v_out10, v_out11, v_out12, v_out13;

        curr_in = in;
        curr_out = out;

        // Input point 1: x(0)
        LDR_128_D(curr_in, v_in_stride, av_in0);
        // Input point 4: x(3) & Input point 5: x(4)
        curr_in = in + in_strides[3];
        LDRI_2x128_D(curr_in, v_in_stride, av_in1, av_in2);
        // Input point 8: x(7) & Input point 9: x(8)
        curr_in = in + in_strides[7];
        LDRI_2x128_D(curr_in, v_in_stride, av_in3, av_in4);
        // Input point 12: x(11) & Input point 13: x(12)
        curr_in = in + in_strides[11];
        LDRI_2x128_D(curr_in, v_in_stride, av_in5, av_in6);

        av_tv1 = _mm_mul_pd(v_CRTM_7_1, av_in5);
        av_tv2 = _mm_mul_pd(v_CRTM_7_3, av_in1);
        av_tv3 = _mm_mul_pd(v_CRTM_7_5, av_in3);
        av_sv1 = _mm_sub_pd(av_in0, av_tv1);
        av_sv2 = _mm_sub_pd(av_tv2, av_tv3);
        av_sv3 = _mm_add_pd(av_sv1, av_sv2);

        av_tv4 = _mm_mul_pd(v_CRTM_7_2, av_in6);
        av_tv5 = _mm_mul_pd(v_CRTM_7_4, av_in2);
        av_tv6 = _mm_mul_pd(v_CRTM_7_6, av_in4);
        av_sv4 = _mm_add_pd(av_tv4, av_tv5);
        av_sv5 = _mm_add_pd(av_sv4, av_tv6);

        av_tv7 = _mm_mul_pd(v_CRTM_7_1, av_in3);
        av_tv8 = _mm_mul_pd(v_CRTM_7_3, av_in5);
        av_tv9 = _mm_mul_pd(v_CRTM_7_5, av_in1);
        av_sv6 = _mm_sub_pd(av_in0, av_tv7);
        av_sv7 = _mm_sub_pd(av_tv8, av_tv9);
        av_sv8 = _mm_add_pd(av_sv6, av_sv7);

        av_tv10 = _mm_mul_pd(v_CRTM_7_2, av_in4);
        av_tv11 = _mm_mul_pd(v_CRTM_7_4, av_in6);
        av_tv12 = _mm_mul_pd(v_CRTM_7_6, av_in2);
        av_sv9 = _mm_add_pd(av_tv10, av_tv11);
        av_sv10 = _mm_sub_pd(av_tv12, av_sv9);

        av_tv13 = _mm_mul_pd(v_CRTM_7_1, av_in1);
        av_tv14 = _mm_mul_pd(v_CRTM_7_3, av_in3);
        av_tv15 = _mm_mul_pd(v_CRTM_7_5, av_in5);
        av_sv11 = _mm_sub_pd(av_in0, av_tv13);
        av_sv12 = _mm_sub_pd(av_tv14, av_tv15);
        av_sv13 = _mm_add_pd(av_sv11, av_sv12);

        av_tv16 = _mm_mul_pd(v_CRTM_7_2, av_in2);
        av_tv17 = _mm_mul_pd(v_CRTM_7_4, av_in4);
        av_tv18 = _mm_mul_pd(v_CRTM_7_6, av_in6);
        av_sv14 = _mm_sub_pd(av_tv16, av_tv17);
        av_sv15 = _mm_add_pd(av_sv14, av_tv18);

        av_sv16 = _mm_add_pd(av_in1, av_in3);
        av_sv17 = _mm_add_pd(av_sv16, av_in5);
        av_tv19 = _mm_mul_pd(v_CRTM_7_7, av_sv17);

        // Output pt 1: X(0)
        v_out0 = _mm_add_pd(av_in0, av_tv19);
        STR_128_D(curr_out, v_out_stride, v_out0);
        // Output pt 3: X(2)
        curr_out = out + out_strides[2];
        v_out2 = _mm_sub_pd(av_sv3, av_sv5);
        STR_128_D(curr_out, v_out_stride, v_out2);
        // Output pt 5: X(4)
        curr_out = out + out_strides[4];
        v_out4 = _mm_sub_pd(av_sv8, av_sv10);
        STR_128_D(curr_out, v_out_stride, v_out4);
        // Output pt 7: X(6)
        curr_out = out + out_strides[6];
        v_out6 = _mm_sub_pd(av_sv13, av_sv15);
        STR_128_D(curr_out, v_out_stride, v_out6);
        // Output pt 9: X(8)
        curr_out = out + out_strides[8];
        v_out8 = _mm_add_pd(av_sv13, av_sv15);
        STR_128_D(curr_out, v_out_stride, v_out8);
        // Output pt 11: X(10)
        curr_out = out + out_strides[10];
        v_out10 = _mm_add_pd(av_sv8, av_sv10);
        STR_128_D(curr_out, v_out_stride, v_out10);
        // Output pt 13: X(12)
        curr_out = out + out_strides[12];
        v_out12 = _mm_add_pd(av_sv3, av_sv5);
        STR_128_D(curr_out, v_out_stride, v_out12);

        __m128d bv_in0, bv_in1, bv_in2, bv_in3, bv_in4, bv_in5, bv_in6;
        __m128d bv_tv1, bv_tv2, bv_tv3, bv_sv1, bv_sv2, bv_sv3, bv_tv4, bv_tv5,
               bv_tv6, bv_sv4, bv_sv5, bv_tv7, bv_tv8, bv_tv9, bv_sv6, bv_sv7,
               bv_sv8, bv_tv10, bv_tv11, bv_tv12, bv_sv9, bv_sv10, bv_tv13,
               bv_tv14, bv_tv15, bv_sv11, bv_sv12, bv_sv13, bv_tv16, bv_tv17,
               bv_tv18, bv_sv14, bv_sv15, bv_sv16, bv_sv17, bv_tv19;

        // Input point 2: x(1) & Input point 3: x(2)
        curr_in = in + in_strides[1];
        LDRI_2x128_D(curr_in, v_in_stride, bv_in0, bv_in1);
        // Input point 6: x(5) & Input point 7: x(6)
        curr_in = in + in_strides[5];
        LDRI_2x128_D(curr_in, v_in_stride, bv_in2, bv_in3);
        // Input point 10: x(9) & Input point 11: x(10)
        curr_in = in + in_strides[9];
        LDRI_2x128_D(curr_in, v_in_stride, bv_in4, bv_in5);
        // Input point 14: x(13)
        curr_in = in + in_strides[13];
        LDR_128_D(curr_in, v_in_stride, bv_in6);

        bv_tv1 = _mm_mul_pd(v_CRTM_7_1, bv_in0);
        bv_tv2 = _mm_mul_pd(v_CRTM_7_3, bv_in4);
        bv_tv3 = _mm_mul_pd(v_CRTM_7_5, bv_in2);
        bv_sv1 = _mm_sub_pd(bv_in6, bv_tv1);
        bv_sv2 = _mm_sub_pd(bv_tv2, bv_tv3);
        bv_sv3 = _mm_add_pd(bv_sv1, bv_sv2);

        bv_tv4 = _mm_mul_pd(v_CRTM_7_2, bv_in1);
        bv_tv5 = _mm_mul_pd(v_CRTM_7_4, bv_in5);
        bv_tv6 = _mm_mul_pd(v_CRTM_7_6, bv_in3);
        bv_sv4 = _mm_add_pd(bv_tv4, bv_tv5);
        bv_sv5 = _mm_add_pd(bv_sv4, bv_tv6);

        bv_tv7 = _mm_mul_pd(v_CRTM_7_1, bv_in2);
        bv_tv8 = _mm_mul_pd(v_CRTM_7_3, bv_in0);
        bv_tv9 = _mm_mul_pd(v_CRTM_7_5, bv_in4);
        bv_sv6 = _mm_sub_pd(bv_in6, bv_tv7);
        bv_sv7 = _mm_sub_pd(bv_tv8, bv_tv9);
        bv_sv8 = _mm_add_pd(bv_sv6, bv_sv7);

        bv_tv10 = _mm_mul_pd(v_CRTM_7_2, bv_in3);
        bv_tv11 = _mm_mul_pd(v_CRTM_7_4, bv_in1);
        bv_tv12 = _mm_mul_pd(v_CRTM_7_6, bv_in5);
        bv_sv9 = _mm_add_pd(bv_tv10, bv_tv11);
        bv_sv10 = _mm_sub_pd(bv_tv12, bv_sv9);

        bv_tv13 = _mm_mul_pd(v_CRTM_7_1, bv_in4);
        bv_tv14 = _mm_mul_pd(v_CRTM_7_3, bv_in2);
        bv_tv15 = _mm_mul_pd(v_CRTM_7_5, bv_in0);
        bv_sv11 = _mm_sub_pd(bv_in6, bv_tv13);
        bv_sv12 = _mm_sub_pd(bv_tv14, bv_tv15);
        bv_sv13 = _mm_add_pd(bv_sv11, bv_sv12);

        bv_tv16 = _mm_mul_pd(v_CRTM_7_2, bv_in5);
        bv_tv17 = _mm_mul_pd(v_CRTM_7_4, bv_in3);
        bv_tv18 = _mm_mul_pd(v_CRTM_7_6, bv_in1);
        bv_sv14 = _mm_sub_pd(bv_tv16, bv_tv17);
        bv_sv15 = _mm_add_pd(bv_sv14, bv_tv18);

        bv_sv16 = _mm_add_pd(bv_in0, bv_in2);
        bv_sv17 = _mm_add_pd(bv_sv16, bv_in4);
        bv_tv19 = _mm_mul_pd(v_CRTM_7_7, bv_sv17);

        // Output pt 2: X(1)
        curr_out = out + out_strides[1];
        v_out1 = _mm_add_pd(bv_in6, bv_tv19);
        STR_128_D(curr_out, v_out_stride, v_out1);
        // Output pt 4: X(3)
        curr_out = out + out_strides[3];
        v_out3 = NEGATE_128_D(_mm_add_pd(bv_sv3, bv_sv5));
        STR_128_D(curr_out, v_out_stride, v_out3);
        // Output pt 6: X(5)
        curr_out = out + out_strides[5];
        v_out5 = _mm_add_pd(bv_sv8, bv_sv10);
        STR_128_D(curr_out, v_out_stride, v_out5);
        // Output pt 8: X(7)
        curr_out = out + out_strides[7];
        v_out7 = NEGATE_128_D(_mm_add_pd(bv_sv13, bv_sv15));
        STR_128_D(curr_out, v_out_stride, v_out7);
        // Output pt 10: X(9)
        curr_out = out + out_strides[9];
        v_out9 = _mm_sub_pd(bv_sv13, bv_sv15);
        STR_128_D(curr_out, v_out_stride, v_out9);
        // Output pt 12: X(11)
        curr_out = out + out_strides[11];
        v_out11 = _mm_sub_pd(bv_sv10, bv_sv8);
        STR_128_D(curr_out, v_out_stride, v_out11);
        // Output pt 14: X(13)
        curr_out = out + out_strides[13];
        v_out13 = _mm_sub_pd(bv_sv3, bv_sv5);
        STR_128_D(curr_out, v_out_stride, v_out13);

        in = in + (v_in_stride << 1);
        out = out + (v_out_stride << 1);
    }
    // tail cases
    if (n & 1)
    {
        DOUBLE av0, av1, av2, av3, av4, av5, av6;
        DOUBLE at0, at1, at2, at3, at4, at5, at6, at7, at8, at9,
              at10, at11, at12, at13, at14, at15, at16, at17, at18, at19,
              at20, at21, at22, at23, at24, at25, at26, at27, at28, at29,
              at30, at31, at32, at33, at34, at35;

        av0 = *in;                  // Input point 1: x(0)
        av1 = in[in_strides[3]];    // Input point 4: x(3)
        av2 = in[in_strides[4]];    // Input point 5: x(4)
        av3 = in[in_strides[7]];    // Input point 8: x(7)
        av4 = in[in_strides[8]];    // Input point 9: x(8)
        av5 = in[in_strides[11]];   // Input point 12: x(11)
        av6 = in[in_strides[12]];   // Input point 13: x(12)

        at0 = CRTM_7_1 * av5;
        at1 = CRTM_7_3 * av1;
        at2 = CRTM_7_5 * av3;
        at3 = av0 - at0;
        at4 = at1 - at2;

        at5 = CRTM_7_2 * av6;
        at6 = CRTM_7_4 * av2;
        at7 = CRTM_7_6 * av4;
        at8 = at5 + at6;
        at9 = at3 + at4;
        at10 = at8 + at7;

        at11 = CRTM_7_1 * av3;
        at12 = CRTM_7_3 * av5;
        at13 = CRTM_7_5 * av1;
        at14 = av0 - at11;
        at15 = at12 - at13;

        at16 = CRTM_7_2 * av4;
        at17 = CRTM_7_4 * av6;
        at18 = CRTM_7_6 * av2;
        at19 = at16 + at17;

        at20 = at14 + at15;
        at21 = at18 - at19;

        at22 = CRTM_7_1 * av1;
        at23 = CRTM_7_3 * av3;
        at24 = CRTM_7_5 * av5;
        at25 = av0 - at22;
        at26 = at23 - at24;

        at27 = CRTM_7_2 * av2;
        at28 = CRTM_7_4 * av4;
        at29 = CRTM_7_6 * av6;
        at30 = at27 - at28;

        at31 = av1 + av3;
        at32 = at25 + at26;
        at33 = at31 + av5;
        at34 = at30 + at29;
        at35 = CRTM_7_7 * at33;

        *out = av0 + at35;                     // Output pt 1: X(0)
        out[out_strides[2]]  = at9 - at10;     // Output pt 3: X(2)
        out[out_strides[4]]  = at20 - at21;    // Output pt 5: X(4)
        out[out_strides[6]]  = at32 - at34;    // Output pt 7: X(6)
        out[out_strides[8]]  = at32 + at34;    // Output pt 9: X(8)
        out[out_strides[10]] = at20 + at21;    // Output pt 11: X(10)
        out[out_strides[12]] = at9 + at10;     // Output pt 13: X(12)

        /* Shifted DFT */
        DOUBLE bv0, bv1, bv2, bv3, bv4, bv5, bv6;
        DOUBLE bt0, bt1, bt2, bt3, bt4, bt5, bt6, bt7, bt8, bt9,
              bt10, bt11, bt12, bt13, bt14, bt15, bt16, bt17, bt18, bt19,
              bt20, bt21, bt22, bt23, bt24, bt25, bt26, bt27, bt28, bt29,
              bt30, bt31, bt32, bt33, bt34, bt35;

        bv0 = in[in_strides[1]];    // Input point 2: x(1)
        bv1 = in[in_strides[2]];    // Input point 3: x(2)
        bv2 = in[in_strides[5]];    // Input point 6: x(5)
        bv3 = in[in_strides[6]];    // Input point 7: x(6)
        bv4 = in[in_strides[9]];    // Input point 10: x(9)
        bv5 = in[in_strides[10]];   // Input point 11: x(10)
        bv6 = in[in_strides[13]];   // Input point 14: x(13)

        bt0 = CRTM_7_1 * bv0;
        bt1 = CRTM_7_3 * bv4;
        bt2 = CRTM_7_5 * bv2;
        bt3 = bv6 - bt0;
        bt4 = bt1 - bt2;
        bt5 = bt3 + bt4;

        bt6 = CRTM_7_2 * bv1;
        bt7 = CRTM_7_4 * bv5;
        bt8 = CRTM_7_6 * bv3;
        bt9 = bt6 + bt7;

        bt10 = bt9 + bt8;
        bt11 = CRTM_7_1 * bv2;
        bt12 = CRTM_7_3 * bv0;
        bt13 = CRTM_7_5 * bv4;
        bt14 = bv6 - bt11;
        bt15 = bt12 - bt13;

        bt16 = CRTM_7_2 * bv3;
        bt17 = CRTM_7_4 * bv1;
        bt18 = CRTM_7_6 * bv5;
        bt19 = bt16 + bt17;

        bt20 = bt14 + bt15;
        bt21 = bt18 - bt19;

        bt22 = CRTM_7_1 * bv4;
        bt23 = CRTM_7_3 * bv2;
        bt24 = CRTM_7_5 * bv0;
        bt25 = bv6 - bt22;
        bt26 = bt23 - bt24;

        bt27 = CRTM_7_2 * bv5;
        bt28 = CRTM_7_4 * bv3;
        bt29 = CRTM_7_6 * bv1;
        bt30 = bt27 - bt28;

        bt31 = bv0 + bv2;
        bt32 = bt25 + bt26;
        bt33 = bt31 + bv4;
        bt34 = bt30 + bt29;
        bt35 = bt33 * CRTM_7_7;

        out[out_strides[1]]  = bv6 + bt35;     // Output pt 2: X(1)
        out[out_strides[3]]  = -bt5 - bt10;    // Output pt 4: X(3)
        out[out_strides[5]]  = bt20 + bt21;    // Output pt 6: X(5)
        out[out_strides[7]]  = -bt32 - bt34;   // Output pt 8: X(7)
        out[out_strides[9]]  = bt32 - bt34;    // Output pt 10: X(9)
        out[out_strides[11]] = bt21 - bt20;    // Output pt 12: X(11)
        out[out_strides[13]] = bt5 - bt10;     // Output pt 14: X(13)
    }
}

kfft_ register_kernel_r2hcf_rfft7avx128(UINT8 precision, UINT8 direction)
{
    if (direction == FORWARD_FFT_DIR)
    {
        if (precision == DT_FLOAT)
        {
            return r2hcf_rfft7avx128_fp32_fwd;
        }
        else if (precision == DT_DOUBLE)
        {
            return r2hcf_rfft7avx128_fp64_fwd;
        }
        else
        {
            return NULL;
        }
    }
    else
    {
        if (precision == DT_FLOAT)
        {
            return r2hcf_rfft7avx128_fp32_bwd;
        }
        else if (precision == DT_DOUBLE)
        {
            return r2hcf_rfft7avx128_fp64_bwd;
        }
        else
        {
            return NULL;
        }
    }
}
