// Copyright Advanced Micro Devices, Inc.
// SPDX-License-Identifier: BSD-3-Clause

/** @file rfft13avx256.c
 *
 *  @brief Radix-13 r2hc Real-FFT kernel with AVX-256 operations using x86 SIMD
 *  intrinsics
 *
 *  This file contains the DIT radix-13 real-to-halfcomplex implementations
 *  using AVX256 SIMD operations for single-precision and double-precision
 *  inputs.
 *
 *  @author Amrin Fathima
 */

 #include "core/kernels/kernel.h"
 #include "core/kernels/simd_includes/r2hc_simd_avx256.h"
 
 static const ops_cycles_t ops_cnt[NUM_PRECISIONS][NUM_FFT_DIRS] =
                                                 {{{0, 34, 81, 26, 0, 0},
                                                   {0, 35, 76, 26, 0, 0}},
                                                  {{0, 34, 81, 26, 0, 0},
                                                   {0, 35, 76, 26, 0, 0}}};

 ops_cycles_t get_ops_cnt_r2hc_rfft13avx256(FFTZ_UINT8 precision,
                                            FFTZ_UINT8 direction)
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

 static FFTZ_VOID r2hc_rfft13avx256_fp32_fwd(FFTZ_VOID *in_real,
                                             FFTZ_VOID *in_imag,
                                             FFTZ_VOID *out_real,
                                             FFTZ_VOID *out_imag, FFTZ_INTP n,
                                             aoclfftz_strides_t *strides,
                                             FFTZ_VOID *twd, FFTZ_UINT8 flag)
 {
     AOCLFFTZ_LOG(DEBUG, global_logger_mode, "Enter");
     const FFTZ_FLOAT CRTM_13_1 =
         1.73205080756887719317660412343684583902359008789060f;
     const FFTZ_FLOAT CRTM_13_2 =
         0.38739058546761714712838513245787843298017865366345f;
     const FFTZ_FLOAT CRTM_13_3 =
         0.13298312460741867056777367279491668244720145803175f;
     const FFTZ_FLOAT CRTM_13_4 =
         0.11385447905579067614456922127652058891458054833493f;
     const FFTZ_FLOAT CRTM_13_5 =
         0.25176851643188325619942540855595308576560112726050f;
     const FFTZ_FLOAT CRTM_13_6 =
         0.86602540378443864676372317075293618347140262700000f;
     const FFTZ_FLOAT CRTM_13_7 =
         0.50000000000000000000000000000000000000000000000000f;
     const FFTZ_FLOAT CRTM_13_8 =
         2.00000000000000000000000000000000000000000000000000f;
     const FFTZ_FLOAT R13_DGC_1 =
         0.08333333333333341693177688718872429763576971755296f;
     const FFTZ_FLOAT R13_DGC_2 =
         0.25624767158293663769405689230781736686616059890980f;
     const FFTZ_FLOAT R13_DGC_3 =
         0.15689139105158457352993666120753768234094124525375f;
     const FFTZ_FLOAT R13_DGC_4 =
         0.25826039031174479484752691274654956353060547073611f;
     const FFTZ_FLOAT R13_DGC_5 =
         0.26596624921483734113554734558983336489440291606351f;
     const FFTZ_FLOAT R13_DGC_6 =
         0.57514072947400312136838554745545338846100160800000f;
     const FFTZ_FLOAT R13_DGC_7 =
         0.17413860115213590500566079492926474261696467600000f;
     const FFTZ_FLOAT R13_DGC_8 =
         0.07590298603719379320004091411468632874055721777845f;
     const FFTZ_FLOAT R13_DGC_9 =
         0.50353703286376651239885081711190617153120225452101f;
     const FFTZ_FLOAT R13_DGC_10 =
         0.30023863596633260097945594679913483560085296630859f;
     const FFTZ_FLOAT R13_DGC_11 =
         0.01159910560576819966993600274918208015151321887970f;
     const FFTZ_FLOAT R13_DGC_12 =
         0.30046260628866572339745744102401658892631530761719f;

     FFTZ_FLOAT *in = (FFTZ_FLOAT *)in_real;
     FFTZ_FLOAT *out = (FFTZ_FLOAT *)out_real;
 #ifdef VOLATILE_STRIDE_ARRAY
     volatile FFTZ_INTP *in_strides = strides->in_strides;
     volatile FFTZ_INTP *out_strides = strides->out_strides;
 #else
     FFTZ_INTP *in_strides = strides->in_strides;
     FFTZ_INTP *out_strides = strides->out_strides;
 #endif
     FFTZ_INTP v_in_stride = strides->v_in_stride;
     FFTZ_INTP v_out_stride = strides->v_out_stride;
 
     FFTZ_INTP cnt;
     FFTZ_FLOAT *curr_in, *curr_out;
     FFTZ_INTP N = n / NUM_SETS_REAL_256_S;
     FFTZ_INTP remaining_sets = n % NUM_SETS_REAL_256_S;
 
     __m256 v_CRTM_13_1 = _mm256_broadcast_ss(&CRTM_13_1);
     __m256 v_CRTM_13_2 = _mm256_broadcast_ss(&CRTM_13_2);
     __m256 v_CRTM_13_3 = _mm256_broadcast_ss(&CRTM_13_3);
     __m256 v_CRTM_13_4 = _mm256_broadcast_ss(&CRTM_13_4);
     __m256 v_CRTM_13_5 = _mm256_broadcast_ss(&CRTM_13_5);
     __m256 v_CRTM_13_6 = _mm256_broadcast_ss(&CRTM_13_6);
     __m256 v_CRTM_13_7 = _mm256_broadcast_ss(&CRTM_13_7);
     __m256 v_CRTM_13_8 = _mm256_broadcast_ss(&CRTM_13_8);
     __m256 v_R13_DGC_1 = _mm256_broadcast_ss(&R13_DGC_1);
     __m256 v_R13_DGC_2 = _mm256_broadcast_ss(&R13_DGC_2);
     __m256 v_R13_DGC_3 = _mm256_broadcast_ss(&R13_DGC_3);
     __m256 v_R13_DGC_4 = _mm256_broadcast_ss(&R13_DGC_4);
     __m256 v_R13_DGC_5 = _mm256_broadcast_ss(&R13_DGC_5);
     __m256 v_R13_DGC_6 = _mm256_broadcast_ss(&R13_DGC_6);
     __m256 v_R13_DGC_7 = _mm256_broadcast_ss(&R13_DGC_7);
     __m256 v_R13_DGC_8 = _mm256_broadcast_ss(&R13_DGC_8);
     __m256 v_R13_DGC_9 = _mm256_broadcast_ss(&R13_DGC_9);
     __m256 v_R13_DGC_10 = _mm256_broadcast_ss(&R13_DGC_10);
     __m256 v_R13_DGC_11 = _mm256_broadcast_ss(&R13_DGC_11);
     __m256 v_R13_DGC_12 = _mm256_broadcast_ss(&R13_DGC_12);
 
     for (cnt = 0; cnt < N; cnt++)
     {
         __m256 v_in0, v_in1, v_in2, v_in3, v_in4, v_in5, v_in6, v_in7, v_in8,
                v_in9, v_in10, v_in11, v_in12;
         __m256 v_s0, v_s1, v_s2, v_s3, v_s4, v_s5, v_s6, v_s7, v_s8, v_s9,
                v_s10, v_s11, v_s12, v_s13, v_s14, v_s15, v_s16, v_s17, v_s18,
                v_s19, v_s20, v_s21, v_s22, v_s23, v_s24, v_s25, v_s26, v_s27,
                v_s28, v_s29, v_s30, v_s31, v_s32, v_s33, v_s34, v_s35, v_s36,
                v_s37, v_s38, v_s39, v_s40, v_s41, v_s42, v_s43, v_s44, v_s45,
                v_s46, v_s47, v_s48, v_s49, v_s50, v_s51, v_s52, v_s53, v_s54,
                v_s55, v_s56, v_s57, v_s58, v_s59, v_s60, v_s61, v_s62, v_s63,
                v_s64;
         __m256 v_m0, v_m1, v_m2, v_m3, v_m4, v_m5, v_m6, v_m7, v_m8, v_m9,
                v_m10, v_m11, v_m12, v_m13, v_m14, v_m15, v_m16, v_m17, v_m18,
                v_m19, v_m20, v_m21, v_m22, v_m23, v_m24, v_m25, v_m26, v_m27,
                v_m28, v_m29, v_m30, v_m31, v_m32, v_m33;
         __m256 v_out0, v_out1, v_out2, v_out3, v_out4, v_out5, v_out6, v_out7,
                v_out8, v_out9, v_out10, v_out11, v_out12;
 
         curr_in = in;
         curr_out = out;
 
         curr_in = in;
         // Input point 1: x(0)
         LDR_256_S(curr_in, v_in_stride, v_in0);
         // Input point 2: x(1)
         curr_in = in + in_strides[1];
         LDR_256_S(curr_in, v_in_stride, v_in1);
         // Input point 3: x(2)
         curr_in = in + in_strides[2];
         LDR_256_S(curr_in, v_in_stride, v_in2);
         // Input point 4: x(3)
         curr_in = in + in_strides[3];
         LDR_256_S(curr_in, v_in_stride, v_in3);
         // Input point 5: x(4)
         curr_in = in + in_strides[4];
         LDR_256_S(curr_in, v_in_stride, v_in4);
         // Input point 6: x(5)
         curr_in = in + in_strides[5];
         LDR_256_S(curr_in, v_in_stride, v_in5);
         // Input point 7: x(6)
         curr_in = in + in_strides[6];
         LDR_256_S(curr_in, v_in_stride, v_in6);
         // Input point 8: x(7)
         curr_in = in + in_strides[7];
         LDR_256_S(curr_in, v_in_stride, v_in7);
         // Input point 9: x(8)
         curr_in = in + in_strides[8];
         LDR_256_S(curr_in, v_in_stride, v_in8);
         // Input point 10: x(9)
         curr_in = in + in_strides[9];
         LDR_256_S(curr_in, v_in_stride, v_in9);
         // Input point 11: x(10)
         curr_in = in + in_strides[10];
         LDR_256_S(curr_in, v_in_stride, v_in10);
         // Input point 12: x(11)
         curr_in = in + in_strides[11];
         LDR_256_S(curr_in, v_in_stride, v_in11);
         // Input point 13: x(12)
         curr_in = in + in_strides[12];
         LDR_256_S(curr_in, v_in_stride, v_in12);
 
         v_s0 = _mm256_add_ps(v_in2, v_in7);
         v_s1 = _mm256_sub_ps(v_in7, v_in2);
         v_s2 = _mm256_add_ps(v_in6, v_in11);
         v_s3 = _mm256_sub_ps(v_in11, v_in6);
         v_s4 = _mm256_add_ps(v_s0, v_s2);
         v_s5 = _mm256_sub_ps(v_s0, v_s2);
         v_m0 = _mm256_mul_ps(v_CRTM_13_6, v_s5);
         v_s6 = _mm256_add_ps(v_s1, v_s3);
         v_s7 = _mm256_sub_ps(v_s1, v_s3);
         v_s8 = _mm256_add_ps(v_in4, v_in10);
         v_s9 = _mm256_sub_ps(v_in10, v_in4);
         v_s10 = _mm256_add_ps(v_in3, v_in9);
         v_s11 = _mm256_sub_ps(v_in9, v_in3);
         v_s12 = _mm256_add_ps(v_s8, v_s10);
         v_s13 = _mm256_sub_ps(v_s8, v_s10);
         v_s14 = _mm256_sub_ps(v_s9, v_s11);
         v_s15 = _mm256_add_ps(v_s9, v_s11);
         v_m1 = _mm256_mul_ps(v_CRTM_13_6, v_s15);
         v_m2 = _mm256_mul_ps(v_CRTM_13_7, v_s13);

         v_s16 = _mm256_add_ps(v_s4, v_s12);
         v_s17 = _mm256_sub_ps(v_s4, v_s12);
         v_s18 = _mm256_add_ps(v_in8, v_in5);
         v_s19 = _mm256_sub_ps(v_in5, v_in8);
         v_m3 = _mm256_mul_ps(v_CRTM_13_7, v_s6);
         v_s42 = _mm256_add_ps(v_m3, v_s19);
         v_s20 = _mm256_sub_ps(v_in1, v_in12);
         v_s21 = _mm256_add_ps(v_in1, v_in12);
         v_s36 = _mm256_add_ps(v_s20, v_m2);
         v_s22 = _mm256_sub_ps(v_s20, v_s13);
         v_s23 = _mm256_sub_ps(v_s6, v_s19);
         v_m4 = _mm256_mul_ps(v_R13_DGC_6, v_s22);
         v_m5 = _mm256_mul_ps(v_R13_DGC_7, v_s23);
         v_s43 = _mm256_add_ps(v_m4, v_m5);
         v_m6 = _mm256_mul_ps(v_R13_DGC_6, v_s23);
         v_m7 = _mm256_mul_ps(v_R13_DGC_7, v_s22);
         v_s45 = _mm256_sub_ps(v_m6, v_m7);
         v_s47 = _mm256_add_ps(v_s21, v_s18);
         v_s48 = _mm256_sub_ps(v_s21, v_s18);
         v_m8 = _mm256_mul_ps(v_CRTM_13_7, v_s16);
         v_s32 = _mm256_sub_ps(v_s47, v_m8);
         v_s28 = _mm256_add_ps(v_s47, v_s16);
         v_m9 = _mm256_mul_ps(v_CRTM_13_7, v_s17);

         v_s33 = _mm256_add_ps(v_s48, v_m9);
         v_s39 = _mm256_sub_ps(v_s48, v_s17);
         v_m10 = _mm256_mul_ps(v_s39, v_R13_DGC_12);
         // Output point 1: X(0)
         v_out0 = _mm256_add_ps(v_s28, v_in0);
         STR_256_S(curr_out, v_out_stride, v_out0);
         v_m11 = NEGATE_256_S(_mm256_mul_ps(v_s28, v_R13_DGC_1));

         v_s63 = _mm256_add_ps(v_m11, v_in0);
         v_s24 = _mm256_add_ps(v_s63, v_m10);
         v_s25 = _mm256_sub_ps(v_s63, v_m10);
         v_s61 = _mm256_add_ps(v_s36, v_m0);
         v_s62 = _mm256_sub_ps(v_s36, v_m0);
         v_s46 = _mm256_add_ps(v_s42, v_m1);
         v_s29 = _mm256_sub_ps(v_s42, v_m1);
         v_m12 = _mm256_mul_ps(v_R13_DGC_2, v_s61);
         v_m13 = _mm256_mul_ps(v_R13_DGC_3, v_s46);
         v_s40 = NEGATE_256_S(_mm256_add_ps(v_m12, v_m13));

         v_m14 = _mm256_mul_ps(v_R13_DGC_2, v_s46);
         v_m15 = _mm256_mul_ps(v_R13_DGC_3, v_s61);
         v_s41 = _mm256_sub_ps(v_m14, v_m15);
         v_m16 = _mm256_mul_ps(v_R13_DGC_10, v_s29);
         v_m17 = _mm256_mul_ps(v_R13_DGC_11, v_s62);
         v_s34 = _mm256_sub_ps(v_m17, v_m16);
         v_m18 = _mm256_mul_ps(v_R13_DGC_10, v_s62);
         v_m19 = _mm256_mul_ps(v_R13_DGC_11, v_s29);
         v_s35 = _mm256_add_ps(v_m18, v_m19);
         v_s26 = _mm256_add_ps(v_s41, v_s34);
         v_s44 = _mm256_sub_ps(v_s41, v_s34);
         v_m20 = _mm256_mul_ps(v_CRTM_13_1, v_s44);
         v_s27 = _mm256_add_ps(v_s40, v_s35);
         v_s64 = _mm256_sub_ps(v_s40, v_s35);
         v_m21 = _mm256_mul_ps(v_CRTM_13_1, v_s64);
         v_s30 = _mm256_add_ps(v_s7, v_s14);
         v_m22 = _mm256_mul_ps(v_R13_DGC_4, v_s33);
         v_m23 = _mm256_mul_ps(v_CRTM_13_3, v_s30);
         v_s49 = _mm256_sub_ps(v_m22, v_m23);
         v_m24 = _mm256_mul_ps(v_CRTM_13_2, v_s30);
         v_m25 = _mm256_mul_ps(v_R13_DGC_5, v_s33);
         v_s37 = NEGATE_256_S(_mm256_add_ps(v_m24, v_m25));

         v_s31 = _mm256_sub_ps(v_s7, v_s14);
         v_m26 = _mm256_mul_ps(v_R13_DGC_8, v_s32);
         v_m27 = _mm256_mul_ps(v_CRTM_13_5, v_s31);
         v_s50 = _mm256_sub_ps(v_m26, v_m27);
         v_m28 = _mm256_mul_ps(v_CRTM_13_4, v_s31);
         v_m29 = _mm256_mul_ps(v_R13_DGC_9, v_s32);
         v_s38 = NEGATE_256_S(_mm256_add_ps(v_m28, v_m29));

         v_s51 = _mm256_add_ps(v_s37, v_s38);
         v_s59 = _mm256_sub_ps(v_s37, v_s38);
         v_s52 = _mm256_add_ps(v_s49, v_s50);
         v_s53 = _mm256_sub_ps(v_s49, v_s50);
         v_m30 = _mm256_mul_ps(v_CRTM_13_8, v_s52);
         // Output point 2: X(1)
         v_out1 = _mm256_add_ps(v_s24, v_m30);
         curr_out = out + out_strides[1];
         STR_256_S(curr_out, v_out_stride, v_out1);

         v_m31 = _mm256_mul_ps(v_CRTM_13_8, v_s26);
         // Output point 3: X(2)
         v_out2 = _mm256_add_ps(v_s45, v_m31);
         curr_out = out + out_strides[2];
         STR_256_S(curr_out, v_out_stride, v_out2);

         v_s54 = _mm256_sub_ps(v_s24, v_s52);
         v_s55 = _mm256_sub_ps(v_s45, v_s26);
         // Output point 6: X(5)
         v_out5 = _mm256_add_ps(v_s54, v_s59);
         curr_out = out + out_strides[5];
         STR_256_S(curr_out, v_out_stride, v_out5);

         // Output point 7: X(6)
         v_out6 = _mm256_add_ps(v_s55, v_m21);
         curr_out = out + out_strides[6];
         STR_256_S(curr_out, v_out_stride, v_out6);

         // Output point 8: X(7)
         v_out7 = _mm256_sub_ps(v_s54, v_s59);
         curr_out = out + out_strides[7];
         STR_256_S(curr_out, v_out_stride, v_out7);

         // Output point 9: X(8)
         v_out8 = _mm256_sub_ps(v_m21, v_s55);
         curr_out = out + out_strides[8];
         STR_256_S(curr_out, v_out_stride, v_out8);

         v_s56 = _mm256_sub_ps(v_s27, v_s43);
         v_m32 = _mm256_mul_ps(v_CRTM_13_8, v_s27);
         v_s57 = NEGATE_256_S(_mm256_add_ps(v_m32, v_s43));
         v_m33 = _mm256_mul_ps(v_CRTM_13_8, v_s53);
         v_s58 = _mm256_sub_ps(v_s25, v_m33);
         v_s60 = _mm256_add_ps(v_s25, v_s53);
         // Output point 4: X(3)
         v_out3 = _mm256_sub_ps(v_s60, v_s51);
         curr_out = out + out_strides[3];
         STR_256_S(curr_out, v_out_stride, v_out3);

         // Output point 12: X(11)
         v_out11 = _mm256_add_ps(v_s60, v_s51);
         curr_out = out + out_strides[11];
         STR_256_S(curr_out, v_out_stride, v_out11);

         // Output point 10: X(9)
         v_out9 = v_s58;
         curr_out = out + out_strides[9];
         STR_256_S(curr_out, v_out_stride, v_out9);

         // Output point 11: X(10)
         v_out10 = v_s57;
         curr_out = out + out_strides[10];
         STR_256_S(curr_out, v_out_stride, v_out10);

         // Output point 5: X(4)
         v_out4 = _mm256_add_ps(v_m20, v_s56);
         curr_out = out + out_strides[4];
         STR_256_S(curr_out, v_out_stride, v_out4);

         // Output point 13: X(12)
         v_out12 = _mm256_sub_ps(v_s56, v_m20);
         curr_out = out + out_strides[12];
         STR_256_S(curr_out, v_out_stride, v_out12);

         in += v_in_stride * NUM_SETS_REAL_256_S;
         out += v_out_stride * NUM_SETS_REAL_256_S;
     }
     // tail cases
     if (remaining_sets & NUM_SETS_REAL_128_S)
     {
         __m128 v_in0, v_in1, v_in2, v_in3, v_in4, v_in5, v_in6, v_in7, v_in8,
                v_in9, v_in10, v_in11, v_in12;
         __m128 v_s0, v_s1, v_s2, v_s3, v_s4, v_s5, v_s6, v_s7, v_s8, v_s9,
                v_s10, v_s11, v_s12, v_s13, v_s14, v_s15, v_s16, v_s17, v_s18,
                v_s19, v_s20, v_s21, v_s22, v_s23, v_s24, v_s25, v_s26, v_s27,
                v_s28, v_s29, v_s30, v_s31, v_s32, v_s33, v_s34, v_s35, v_s36,
                v_s37, v_s38, v_s39, v_s40, v_s41, v_s42, v_s43, v_s44, v_s45,
                v_s46, v_s47, v_s48, v_s49, v_s50, v_s51, v_s52, v_s53, v_s54,
                v_s55, v_s56, v_s57, v_s58, v_s59, v_s60, v_s61, v_s62, v_s63,
                v_s64;
         __m128 v_m0, v_m1, v_m2, v_m3, v_m4, v_m5, v_m6, v_m7, v_m8, v_m9,
                v_m10, v_m11, v_m12, v_m13, v_m14, v_m15, v_m16, v_m17, v_m18,
                v_m19, v_m20, v_m21, v_m22, v_m23, v_m24, v_m25, v_m26, v_m27,
                v_m28, v_m29, v_m30, v_m31, v_m32, v_m33;
         __m128 v_out0, v_out1, v_out2, v_out3, v_out4, v_out5, v_out6, v_out7,
                v_out8, v_out9, v_out10, v_out11, v_out12;
 
         curr_in = in;
         curr_out = out;
 
         __m128 v128_CRTM_13_1 = _mm256_castps256_ps128(v_CRTM_13_1);
         __m128 v128_CRTM_13_2 = _mm256_castps256_ps128(v_CRTM_13_2);
         __m128 v128_CRTM_13_3 = _mm256_castps256_ps128(v_CRTM_13_3);
         __m128 v128_CRTM_13_4 = _mm256_castps256_ps128(v_CRTM_13_4);
         __m128 v128_CRTM_13_5 = _mm256_castps256_ps128(v_CRTM_13_5);
         __m128 v128_CRTM_13_6 = _mm256_castps256_ps128(v_CRTM_13_6);
         __m128 v128_CRTM_13_7 = _mm256_castps256_ps128(v_CRTM_13_7);
         __m128 v128_CRTM_13_8 = _mm256_castps256_ps128(v_CRTM_13_8);
         __m128 v128_R13_DGC_1 = _mm256_castps256_ps128(v_R13_DGC_1);
         __m128 v128_R13_DGC_2 = _mm256_castps256_ps128(v_R13_DGC_2);
         __m128 v128_R13_DGC_3 = _mm256_castps256_ps128(v_R13_DGC_3);
         __m128 v128_R13_DGC_4 = _mm256_castps256_ps128(v_R13_DGC_4);
         __m128 v128_R13_DGC_5 = _mm256_castps256_ps128(v_R13_DGC_5);
         __m128 v128_R13_DGC_6 = _mm256_castps256_ps128(v_R13_DGC_6);
         __m128 v128_R13_DGC_7 = _mm256_castps256_ps128(v_R13_DGC_7);
         __m128 v128_R13_DGC_8 = _mm256_castps256_ps128(v_R13_DGC_8);
         __m128 v128_R13_DGC_9 = _mm256_castps256_ps128(v_R13_DGC_9);
         __m128 v128_R13_DGC_10 = _mm256_castps256_ps128(v_R13_DGC_10);
         __m128 v128_R13_DGC_11 = _mm256_castps256_ps128(v_R13_DGC_11);
         __m128 v128_R13_DGC_12 = _mm256_castps256_ps128(v_R13_DGC_12);
 
         // Input point 1: x(0)
         LDR_128_S(curr_in, v_in_stride, v_in0);
         // Input point 2: x(1)
         curr_in = in + in_strides[1];
         LDR_128_S(curr_in, v_in_stride, v_in1);
         // Input point 3: x(2)
         curr_in = in + in_strides[2];
         LDR_128_S(curr_in, v_in_stride, v_in2);
         // Input point 4: x(3)
         curr_in = in + in_strides[3];
         LDR_128_S(curr_in, v_in_stride, v_in3);
         // Input point 5: x(4)
         curr_in = in + in_strides[4];
         LDR_128_S(curr_in, v_in_stride, v_in4);
         // Input point 6: x(5)
         curr_in = in + in_strides[5];
         LDR_128_S(curr_in, v_in_stride, v_in5);
         // Input point 7: x(6)
         curr_in = in + in_strides[6];
         LDR_128_S(curr_in, v_in_stride, v_in6);
         // Input point 8: x(7)
         curr_in = in + in_strides[7];
         LDR_128_S(curr_in, v_in_stride, v_in7);
         // Input point 9: x(8)
         curr_in = in + in_strides[8];
         LDR_128_S(curr_in, v_in_stride, v_in8);
         // Input point 10: x(9)
         curr_in = in + in_strides[9];
         LDR_128_S(curr_in, v_in_stride, v_in9);
         // Input point 11: x(10)
         curr_in = in + in_strides[10];
         LDR_128_S(curr_in, v_in_stride, v_in10);
         // Input point 12: x(11)
         curr_in = in + in_strides[11];
         LDR_128_S(curr_in, v_in_stride, v_in11);
         // Input point 13: x(12)
         curr_in = in + in_strides[12];
         LDR_128_S(curr_in, v_in_stride, v_in12);
 
         v_s0 = _mm_add_ps(v_in2, v_in7);
         v_s1 = _mm_sub_ps(v_in7, v_in2);
         v_s2 = _mm_add_ps(v_in6, v_in11);
         v_s3 = _mm_sub_ps(v_in11, v_in6);
         v_s4 = _mm_add_ps(v_s0, v_s2);
         v_s5 = _mm_sub_ps(v_s0, v_s2);
         v_m0 = _mm_mul_ps(v128_CRTM_13_6, v_s5);
         v_s6 = _mm_add_ps(v_s1, v_s3);
         v_s7 = _mm_sub_ps(v_s1, v_s3);
         v_s8 = _mm_add_ps(v_in4, v_in10);
         v_s9 = _mm_sub_ps(v_in10, v_in4);
         v_s10 = _mm_add_ps(v_in3, v_in9);
         v_s11 = _mm_sub_ps(v_in9, v_in3);
         v_s12 = _mm_add_ps(v_s8, v_s10);
         v_s13 = _mm_sub_ps(v_s8, v_s10);
         v_s14 = _mm_sub_ps(v_s9, v_s11);
         v_s15 = _mm_add_ps(v_s9, v_s11);
         v_m1 = _mm_mul_ps(v128_CRTM_13_6, v_s15);
         v_m2 = _mm_mul_ps(v128_CRTM_13_7, v_s13);

         v_s16 = _mm_add_ps(v_s4, v_s12);
         v_s17 = _mm_sub_ps(v_s4, v_s12);
         v_s18 = _mm_add_ps(v_in8, v_in5);
         v_s19 = _mm_sub_ps(v_in5, v_in8);
         v_m3 = _mm_mul_ps(v128_CRTM_13_7, v_s6);
         v_s42 = _mm_add_ps(v_m3, v_s19);
         v_s20 = _mm_sub_ps(v_in1, v_in12);
         v_s21 = _mm_add_ps(v_in1, v_in12);
         v_s36 = _mm_add_ps(v_s20, v_m2);
         v_s22 = _mm_sub_ps(v_s20, v_s13);
         v_s23 = _mm_sub_ps(v_s6, v_s19);
         v_m4 = _mm_mul_ps(v128_R13_DGC_6, v_s22);
         v_m5 = _mm_mul_ps(v128_R13_DGC_7, v_s23);
         v_s43 = _mm_add_ps(v_m4, v_m5);
         v_m6 = _mm_mul_ps(v128_R13_DGC_6, v_s23);
         v_m7 = _mm_mul_ps(v128_R13_DGC_7, v_s22);
         v_s45 = _mm_sub_ps(v_m6, v_m7);
         v_s47 = _mm_add_ps(v_s21, v_s18);
         v_s48 = _mm_sub_ps(v_s21, v_s18);
         v_m8 = _mm_mul_ps(v128_CRTM_13_7, v_s16);
         v_s32 = _mm_sub_ps(v_s47, v_m8);
         v_s28 = _mm_add_ps(v_s47, v_s16);
         v_m9 = _mm_mul_ps(v128_CRTM_13_7, v_s17);

         v_s33 = _mm_add_ps(v_s48, v_m9);
         v_s39 = _mm_sub_ps(v_s48, v_s17);
         v_m10 = _mm_mul_ps(v_s39, v128_R13_DGC_12);
         // Output point 1: X(0)
         v_out0 = _mm_add_ps(v_s28, v_in0);
         STR_128_S(curr_out, v_out_stride, v_out0);
         v_m11 = NEGATE_128_S(_mm_mul_ps(v_s28, v128_R13_DGC_1));

         v_s63 = _mm_add_ps(v_m11, v_in0);
         v_s24 = _mm_add_ps(v_s63, v_m10);
         v_s25 = _mm_sub_ps(v_s63, v_m10);
         v_s61 = _mm_add_ps(v_s36, v_m0);
         v_s62 = _mm_sub_ps(v_s36, v_m0);
         v_s46 = _mm_add_ps(v_s42, v_m1);
         v_s29 = _mm_sub_ps(v_s42, v_m1);
         v_m12 = _mm_mul_ps(v128_R13_DGC_2, v_s61);
         v_m13 = _mm_mul_ps(v128_R13_DGC_3, v_s46);
         v_s40 = NEGATE_128_S(_mm_add_ps(v_m12, v_m13));

         v_m14 = _mm_mul_ps(v128_R13_DGC_2, v_s46);
         v_m15 = _mm_mul_ps(v128_R13_DGC_3, v_s61);
         v_s41 = _mm_sub_ps(v_m14, v_m15);
         v_m16 = _mm_mul_ps(v128_R13_DGC_10, v_s29);
         v_m17 = _mm_mul_ps(v128_R13_DGC_11, v_s62);
         v_s34 = _mm_sub_ps(v_m17, v_m16);
         v_m18 = _mm_mul_ps(v128_R13_DGC_10, v_s62);
         v_m19 = _mm_mul_ps(v128_R13_DGC_11, v_s29);
         v_s35 = _mm_add_ps(v_m18, v_m19);
         v_s26 = _mm_add_ps(v_s41, v_s34);
         v_s44 = _mm_sub_ps(v_s41, v_s34);
         v_m20 = _mm_mul_ps(v128_CRTM_13_1, v_s44);
         v_s27 = _mm_add_ps(v_s40, v_s35);
         v_s64 = _mm_sub_ps(v_s40, v_s35);
         v_m21 = _mm_mul_ps(v128_CRTM_13_1, v_s64);
         v_s30 = _mm_add_ps(v_s7, v_s14);
         v_m22 = _mm_mul_ps(v128_R13_DGC_4, v_s33);
         v_m23 = _mm_mul_ps(v128_CRTM_13_3, v_s30);
         v_s49 = _mm_sub_ps(v_m22, v_m23);
         v_m24 = _mm_mul_ps(v128_CRTM_13_2, v_s30);
         v_m25 = _mm_mul_ps(v128_R13_DGC_5, v_s33);
         v_s37 = NEGATE_128_S(_mm_add_ps(v_m24, v_m25));

         v_s31 = _mm_sub_ps(v_s7, v_s14);
         v_m26 = _mm_mul_ps(v128_R13_DGC_8, v_s32);
         v_m27 = _mm_mul_ps(v128_CRTM_13_5, v_s31);
         v_s50 = _mm_sub_ps(v_m26, v_m27);
         v_m28 = _mm_mul_ps(v128_CRTM_13_4, v_s31);
         v_m29 = _mm_mul_ps(v128_R13_DGC_9, v_s32);
         v_s38 = NEGATE_128_S(_mm_add_ps(v_m28, v_m29));

         v_s51 = _mm_add_ps(v_s37, v_s38);
         v_s59 = _mm_sub_ps(v_s37, v_s38);
         v_s52 = _mm_add_ps(v_s49, v_s50);
         v_s53 = _mm_sub_ps(v_s49, v_s50);
         v_m30 = _mm_mul_ps(v128_CRTM_13_8, v_s52);
         // Output point 2: X(1)
         v_out1 = _mm_add_ps(v_s24, v_m30);
         curr_out = out + out_strides[1];
         STR_128_S(curr_out, v_out_stride, v_out1);

         v_m31 = _mm_mul_ps(v128_CRTM_13_8, v_s26);
         // Output point 3: X(2)
         v_out2 = _mm_add_ps(v_s45, v_m31);
         curr_out = out + out_strides[2];
         STR_128_S(curr_out, v_out_stride, v_out2);

         v_s54 = _mm_sub_ps(v_s24, v_s52);
         v_s55 = _mm_sub_ps(v_s45, v_s26);
         // Output point 6: X(5)
         v_out5 = _mm_add_ps(v_s54, v_s59);
         curr_out = out + out_strides[5];
         STR_128_S(curr_out, v_out_stride, v_out5);

         // Output point 7: X(6)
         v_out6 = _mm_add_ps(v_s55, v_m21);
         curr_out = out + out_strides[6];
         STR_128_S(curr_out, v_out_stride, v_out6);

         // Output point 8: X(7)
         v_out7 = _mm_sub_ps(v_s54, v_s59);
         curr_out = out + out_strides[7];
         STR_128_S(curr_out, v_out_stride, v_out7);

         // Output point 9: X(8)
         v_out8 = _mm_sub_ps(v_m21, v_s55);
         curr_out = out + out_strides[8];
         STR_128_S(curr_out, v_out_stride, v_out8);

         v_s56 = _mm_sub_ps(v_s27, v_s43);
         v_m32 = _mm_mul_ps(v128_CRTM_13_8, v_s27);
         v_s57 = NEGATE_128_S(_mm_add_ps(v_m32, v_s43));
         v_m33 = _mm_mul_ps(v128_CRTM_13_8, v_s53);
         v_s58 = _mm_sub_ps(v_s25, v_m33);
         v_s60 = _mm_add_ps(v_s25, v_s53);
         // Output point 4: X(3)
         v_out3 = _mm_sub_ps(v_s60, v_s51);
         curr_out = out + out_strides[3];
         STR_128_S(curr_out, v_out_stride, v_out3);

         // Output point 12: X(11)
         v_out11 = _mm_add_ps(v_s60, v_s51);
         curr_out = out + out_strides[11];
         STR_128_S(curr_out, v_out_stride, v_out11);

         // Output point 10: X(9)
         v_out9 = v_s58;
         curr_out = out + out_strides[9];
         STR_128_S(curr_out, v_out_stride, v_out9);

         // Output point 11: X(10)
         v_out10 = v_s57;
         curr_out = out + out_strides[10];
         STR_128_S(curr_out, v_out_stride, v_out10);

         // Output point 5: X(4)
         v_out4 = _mm_add_ps(v_m20, v_s56);
         curr_out = out + out_strides[4];
         STR_128_S(curr_out, v_out_stride, v_out4);

         // Output point 13: X(12)
         v_out12 = _mm_sub_ps(v_s56, v_m20);
         curr_out = out + out_strides[12];
         STR_128_S(curr_out, v_out_stride, v_out12);

         in += v_in_stride * NUM_SETS_REAL_128_S;
         out += v_out_stride * NUM_SETS_REAL_128_S;
     }

     if (remaining_sets & 2)
     {
         __m128 v_in0, v_in1, v_in2, v_in3, v_in4, v_in5, v_in6, v_in7, v_in8,
                v_in9, v_in10, v_in11, v_in12;
         __m128 v_s0, v_s1, v_s2, v_s3, v_s4, v_s5, v_s6, v_s7, v_s8, v_s9,
                v_s10, v_s11, v_s12, v_s13, v_s14, v_s15, v_s16, v_s17, v_s18,
                v_s19, v_s20, v_s21, v_s22, v_s23, v_s24, v_s25, v_s26, v_s27,
                v_s28, v_s29, v_s30, v_s31, v_s32, v_s33, v_s34, v_s35, v_s36,
                v_s37, v_s38, v_s39, v_s40, v_s41, v_s42, v_s43, v_s44, v_s45,
                v_s46, v_s47, v_s48, v_s49, v_s50, v_s51, v_s52, v_s53, v_s54,
                v_s55, v_s56, v_s57, v_s58, v_s59, v_s60, v_s61, v_s62, v_s63,
                v_s64;
         __m128 v_m0, v_m1, v_m2, v_m3, v_m4, v_m5, v_m6, v_m7, v_m8, v_m9,
                v_m10, v_m11, v_m12, v_m13, v_m14, v_m15, v_m16, v_m17, v_m18,
                v_m19, v_m20, v_m21, v_m22, v_m23, v_m24, v_m25, v_m26, v_m27,
                v_m28, v_m29, v_m30, v_m31, v_m32, v_m33;
         __m128 v_out0, v_out1, v_out2, v_out3, v_out4, v_out5, v_out6, v_out7,
                v_out8, v_out9, v_out10, v_out11, v_out12;
 
         curr_in = in;
         curr_out = out;
 
         __m128 v128_CRTM_13_1 = _mm256_castps256_ps128(v_CRTM_13_1);
         __m128 v128_CRTM_13_2 = _mm256_castps256_ps128(v_CRTM_13_2);
         __m128 v128_CRTM_13_3 = _mm256_castps256_ps128(v_CRTM_13_3);
         __m128 v128_CRTM_13_4 = _mm256_castps256_ps128(v_CRTM_13_4);
         __m128 v128_CRTM_13_5 = _mm256_castps256_ps128(v_CRTM_13_5);
         __m128 v128_CRTM_13_6 = _mm256_castps256_ps128(v_CRTM_13_6);
         __m128 v128_CRTM_13_7 = _mm256_castps256_ps128(v_CRTM_13_7);
         __m128 v128_CRTM_13_8 = _mm256_castps256_ps128(v_CRTM_13_8);
         __m128 v128_R13_DGC_1 = _mm256_castps256_ps128(v_R13_DGC_1);
         __m128 v128_R13_DGC_2 = _mm256_castps256_ps128(v_R13_DGC_2);
         __m128 v128_R13_DGC_3 = _mm256_castps256_ps128(v_R13_DGC_3);
         __m128 v128_R13_DGC_4 = _mm256_castps256_ps128(v_R13_DGC_4);
         __m128 v128_R13_DGC_5 = _mm256_castps256_ps128(v_R13_DGC_5);
         __m128 v128_R13_DGC_6 = _mm256_castps256_ps128(v_R13_DGC_6);
         __m128 v128_R13_DGC_7 = _mm256_castps256_ps128(v_R13_DGC_7);
         __m128 v128_R13_DGC_8 = _mm256_castps256_ps128(v_R13_DGC_8);
         __m128 v128_R13_DGC_9 = _mm256_castps256_ps128(v_R13_DGC_9);
         __m128 v128_R13_DGC_10 = _mm256_castps256_ps128(v_R13_DGC_10);
         __m128 v128_R13_DGC_11 = _mm256_castps256_ps128(v_R13_DGC_11);
         __m128 v128_R13_DGC_12 = _mm256_castps256_ps128(v_R13_DGC_12);
 
         // Input point 1: x(0)
         LDHR_128_S(curr_in, v_in_stride, v_in0);
         // Input point 2: x(1)
         curr_in = in + in_strides[1];
         LDHR_128_S(curr_in, v_in_stride, v_in1);
         // Input point 3: x(2)
         curr_in = in + in_strides[2];
         LDHR_128_S(curr_in, v_in_stride, v_in2);
         // Input point 4: x(3)
         curr_in = in + in_strides[3];
         LDHR_128_S(curr_in, v_in_stride, v_in3);
         // Input point 5: x(4)
         curr_in = in + in_strides[4];
         LDHR_128_S(curr_in, v_in_stride, v_in4);
         // Input point 6: x(5)
         curr_in = in + in_strides[5];
         LDHR_128_S(curr_in, v_in_stride, v_in5);
         // Input point 7: x(6)
         curr_in = in + in_strides[6];
         LDHR_128_S(curr_in, v_in_stride, v_in6);
         // Input point 8: x(7)
         curr_in = in + in_strides[7];
         LDHR_128_S(curr_in, v_in_stride, v_in7);
         // Input point 9: x(8)
         curr_in = in + in_strides[8];
         LDHR_128_S(curr_in, v_in_stride, v_in8);
         // Input point 10: x(9)
         curr_in = in + in_strides[9];
         LDHR_128_S(curr_in, v_in_stride, v_in9);
         // Input point 11: x(10)
         curr_in = in + in_strides[10];
         LDHR_128_S(curr_in, v_in_stride, v_in10);
         // Input point 12: x(11)
         curr_in = in + in_strides[11];
         LDHR_128_S(curr_in, v_in_stride, v_in11);
         // Input point 13: x(12)
         curr_in = in + in_strides[12];
         LDHR_128_S(curr_in, v_in_stride, v_in12);
 
         v_s0 = _mm_add_ps(v_in2, v_in7);
         v_s1 = _mm_sub_ps(v_in7, v_in2);
         v_s2 = _mm_add_ps(v_in6, v_in11);
         v_s3 = _mm_sub_ps(v_in11, v_in6);
         v_s4 = _mm_add_ps(v_s0, v_s2);
         v_s5 = _mm_sub_ps(v_s0, v_s2);
         v_m0 = _mm_mul_ps(v128_CRTM_13_6, v_s5);
         v_s6 = _mm_add_ps(v_s1, v_s3);
         v_s7 = _mm_sub_ps(v_s1, v_s3);
         v_s8 = _mm_add_ps(v_in4, v_in10);
         v_s9 = _mm_sub_ps(v_in10, v_in4);
         v_s10 = _mm_add_ps(v_in3, v_in9);
         v_s11 = _mm_sub_ps(v_in9, v_in3);
         v_s12 = _mm_add_ps(v_s8, v_s10);
         v_s13 = _mm_sub_ps(v_s8, v_s10);
         v_s14 = _mm_sub_ps(v_s9, v_s11);
         v_s15 = _mm_add_ps(v_s9, v_s11);
         v_m1 = _mm_mul_ps(v128_CRTM_13_6, v_s15);
         v_m2 = _mm_mul_ps(v128_CRTM_13_7, v_s13);

         v_s16 = _mm_add_ps(v_s4, v_s12);
         v_s17 = _mm_sub_ps(v_s4, v_s12);
         v_s18 = _mm_add_ps(v_in8, v_in5);
         v_s19 = _mm_sub_ps(v_in5, v_in8);
         v_m3 = _mm_mul_ps(v128_CRTM_13_7, v_s6);
         v_s42 = _mm_add_ps(v_m3, v_s19);
         v_s20 = _mm_sub_ps(v_in1, v_in12);
         v_s21 = _mm_add_ps(v_in1, v_in12);
         v_s36 = _mm_add_ps(v_s20, v_m2);
         v_s22 = _mm_sub_ps(v_s20, v_s13);
         v_s23 = _mm_sub_ps(v_s6, v_s19);
         v_m4 = _mm_mul_ps(v128_R13_DGC_6, v_s22);
         v_m5 = _mm_mul_ps(v128_R13_DGC_7, v_s23);
         v_s43 = _mm_add_ps(v_m4, v_m5);
         v_m6 = _mm_mul_ps(v128_R13_DGC_6, v_s23);
         v_m7 = _mm_mul_ps(v128_R13_DGC_7, v_s22);
         v_s45 = _mm_sub_ps(v_m6, v_m7);
         v_s47 = _mm_add_ps(v_s21, v_s18);
         v_s48 = _mm_sub_ps(v_s21, v_s18);
         v_m8 = _mm_mul_ps(v128_CRTM_13_7, v_s16);
         v_s32 = _mm_sub_ps(v_s47, v_m8);
         v_s28 = _mm_add_ps(v_s47, v_s16);
         v_m9 = _mm_mul_ps(v128_CRTM_13_7, v_s17);

         v_s33 = _mm_add_ps(v_s48, v_m9);
         v_s39 = _mm_sub_ps(v_s48, v_s17);
         v_m10 = _mm_mul_ps(v_s39, v128_R13_DGC_12);
         // Output point 1: X(0)
         v_out0 = _mm_add_ps(v_s28, v_in0);
         STHR_128_S(curr_out, v_out_stride, v_out0);
         v_m11 = NEGATE_128_S(_mm_mul_ps(v_s28, v128_R13_DGC_1));

         v_s63 = _mm_add_ps(v_m11, v_in0);
         v_s24 = _mm_add_ps(v_s63, v_m10);
         v_s25 = _mm_sub_ps(v_s63, v_m10);
         v_s61 = _mm_add_ps(v_s36, v_m0);
         v_s62 = _mm_sub_ps(v_s36, v_m0);
         v_s46 = _mm_add_ps(v_s42, v_m1);
         v_s29 = _mm_sub_ps(v_s42, v_m1);
         v_m12 = _mm_mul_ps(v128_R13_DGC_2, v_s61);
         v_m13 = _mm_mul_ps(v128_R13_DGC_3, v_s46);
         v_s40 = NEGATE_128_S(_mm_add_ps(v_m12, v_m13));

         v_m14 = _mm_mul_ps(v128_R13_DGC_2, v_s46);
         v_m15 = _mm_mul_ps(v128_R13_DGC_3, v_s61);
         v_s41 = _mm_sub_ps(v_m14, v_m15);
         v_m16 = _mm_mul_ps(v128_R13_DGC_10, v_s29);
         v_m17 = _mm_mul_ps(v128_R13_DGC_11, v_s62);
         v_s34 = _mm_sub_ps(v_m17, v_m16);
         v_m18 = _mm_mul_ps(v128_R13_DGC_10, v_s62);
         v_m19 = _mm_mul_ps(v128_R13_DGC_11, v_s29);
         v_s35 = _mm_add_ps(v_m18, v_m19);
         v_s26 = _mm_add_ps(v_s41, v_s34);
         v_s44 = _mm_sub_ps(v_s41, v_s34);
         v_m20 = _mm_mul_ps(v128_CRTM_13_1, v_s44);
         v_s27 = _mm_add_ps(v_s40, v_s35);
         v_s64 = _mm_sub_ps(v_s40, v_s35);
         v_m21 = _mm_mul_ps(v128_CRTM_13_1, v_s64);
         v_s30 = _mm_add_ps(v_s7, v_s14);
         v_m22 = _mm_mul_ps(v128_R13_DGC_4, v_s33);
         v_m23 = _mm_mul_ps(v128_CRTM_13_3, v_s30);
         v_s49 = _mm_sub_ps(v_m22, v_m23);
         v_m24 = _mm_mul_ps(v128_CRTM_13_2, v_s30);
         v_m25 = _mm_mul_ps(v128_R13_DGC_5, v_s33);
         v_s37 = NEGATE_128_S(_mm_add_ps(v_m24, v_m25));

         v_s31 = _mm_sub_ps(v_s7, v_s14);
         v_m26 = _mm_mul_ps(v128_R13_DGC_8, v_s32);
         v_m27 = _mm_mul_ps(v128_CRTM_13_5, v_s31);
         v_s50 = _mm_sub_ps(v_m26, v_m27);
         v_m28 = _mm_mul_ps(v128_CRTM_13_4, v_s31);
         v_m29 = _mm_mul_ps(v128_R13_DGC_9, v_s32);
         v_s38 = NEGATE_128_S(_mm_add_ps(v_m28, v_m29));

         v_s51 = _mm_add_ps(v_s37, v_s38);
         v_s59 = _mm_sub_ps(v_s37, v_s38);
         v_s52 = _mm_add_ps(v_s49, v_s50);
         v_s53 = _mm_sub_ps(v_s49, v_s50);
         v_m30 = _mm_mul_ps(v128_CRTM_13_8, v_s52);
         // Output point 2: X(1)
         v_out1 = _mm_add_ps(v_s24, v_m30);
         curr_out = out + out_strides[1];
         STHR_128_S(curr_out, v_out_stride, v_out1);

         v_m31 = _mm_mul_ps(v128_CRTM_13_8, v_s26);
         // Output point 3: X(2)
         v_out2 = _mm_add_ps(v_s45, v_m31);
         curr_out = out + out_strides[2];
         STHR_128_S(curr_out, v_out_stride, v_out2);

         v_s54 = _mm_sub_ps(v_s24, v_s52);
         v_s55 = _mm_sub_ps(v_s45, v_s26);
         // Output point 6: X(5)
         v_out5 = _mm_add_ps(v_s54, v_s59);
         curr_out = out + out_strides[5];
         STHR_128_S(curr_out, v_out_stride, v_out5);

         // Output point 7: X(6)
         v_out6 = _mm_add_ps(v_s55, v_m21);
         curr_out = out + out_strides[6];
         STHR_128_S(curr_out, v_out_stride, v_out6);

         // Output point 8: X(7)
         v_out7 = _mm_sub_ps(v_s54, v_s59);
         curr_out = out + out_strides[7];
         STHR_128_S(curr_out, v_out_stride, v_out7);

         // Output point 9: X(8)
         v_out8 = _mm_sub_ps(v_m21, v_s55);
         curr_out = out + out_strides[8];
         STHR_128_S(curr_out, v_out_stride, v_out8);

         v_s56 = _mm_sub_ps(v_s27, v_s43);
         v_m32 = _mm_mul_ps(v128_CRTM_13_8, v_s27);
         v_s57 = NEGATE_128_S(_mm_add_ps(v_m32, v_s43));

         v_m33 = _mm_mul_ps(v128_CRTM_13_8, v_s53);
         v_s58 = _mm_sub_ps(v_s25, v_m33);
         v_s60 = _mm_add_ps(v_s25, v_s53);
         // Output point 4: X(3)
         v_out3 = _mm_sub_ps(v_s60, v_s51);
         curr_out = out + out_strides[3];
         STHR_128_S(curr_out, v_out_stride, v_out3);
         
         // Output point 12: X(11)
         v_out11 = _mm_add_ps(v_s60, v_s51);
         curr_out = out + out_strides[11];
         STHR_128_S(curr_out, v_out_stride, v_out11);

         // Output point 10: X(9)
         v_out9 = v_s58;
         curr_out = out + out_strides[9];
         STHR_128_S(curr_out, v_out_stride, v_out9);

         // Output point 11: X(10)
         v_out10 = v_s57;
         curr_out = out + out_strides[10];
         STHR_128_S(curr_out, v_out_stride, v_out10);

         // Output point 5: X(4)
         v_out4 = _mm_add_ps(v_m20, v_s56);
         curr_out = out + out_strides[4];
         STHR_128_S(curr_out, v_out_stride, v_out4);

         // Output point 13: X(12)
         v_out12 = _mm_sub_ps(v_s56, v_m20);
         curr_out = out + out_strides[12];
         STHR_128_S(curr_out, v_out_stride, v_out12);

         in = in + (v_in_stride << 1);
         out = out + (v_out_stride << 1);
     }
 
     if (remaining_sets & 1)
     {
         FFTZ_FLOAT in0, in1, in2, in3, in4, in5, in6, in7, in8, in9, in10,
                    in11, in12;
         FFTZ_FLOAT s0, s1, s2, s3, s4, s5, s6, s7, s8, s9, s10, s11, s12, s13,
                    s14, s15, s16, s17, s18, s19, s20, s21, s22, s23, s24, s25,
                    s26, s27, s28, s29, s30, s31, s32, s33, s34, s35, s36, s37,
                    s38, s39, s40, s41, s42, s43, s44, s45, s46, s47, s48, s49,
                    s50, s51, s52, s53, s54, s55, s56, s57,
                    s58, s59, s60, s61, s62, s63, s64;
         FFTZ_FLOAT m0, m1, m2, m3, m4, m5, m6, m7, m8, m9, m10, m11, m12, m13,
                    m14, m15, m16, m17, m18, m19, m20, m21, m22, m23, m24, m25,
                    m26, m27, m28, m29, m30, m31, m32, m33;

         // Input point 1: x(0)
         in0 = *in;
         // Input point 2: x(1)
         in1 = in[in_strides[1]];
         // Input point 3: x(2)
         in2 = in[in_strides[2]];
         // Input point 4: x(3)
         in3 = in[in_strides[3]];
         // Input point 5: x(4)
         in4 = in[in_strides[4]];
         // Input point 6: x(5)
         in5 = in[in_strides[5]];
         // Input point 7: x(6)
         in6 = in[in_strides[6]];
         // Input point 8: x(7)
         in7 = in[in_strides[7]];
         // Input point 9: x(8)
         in8 = in[in_strides[8]];
         // Input point 10: x(9)
         in9 = in[in_strides[9]];
         // Input point 11: x(10)
         in10 = in[in_strides[10]];
         // Input point 12: x(11)
         in11 = in[in_strides[11]];
         // Input point 13: x(12)
         in12 = in[in_strides[12]];
 
         s0 = in2 + in7;
         s1 = in7 - in2;
         s2 = in6 + in11;
         s3 = in11 - in6;
         s4 = s0 + s2;
         s5 = s0 - s2;
         m0 = CRTM_13_6 * s5;
         s6 = s1 + s3;
         s7 = s1 - s3;
         s8 = in4 + in10;
         s9 = in10 - in4;
         s10 = in3 + in9;
         s11 = in9 - in3;
         s12 = s8 + s10;
         s13 = s8 - s10;
         s14 = s9 - s11;
         s15 = s9 + s11;
         m1 = CRTM_13_6 * s15;
         m2 = CRTM_13_7 * s13;

         s16 = s4 + s12;
         s17 = s4 - s12;
         s18 = in8 + in5;
         s19 = in5 - in8;
         m3 = CRTM_13_7 * s6;
         s42 = m3 + s19;
         s20 = in1 - in12;
         s21 = in1 + in12;
         s36 = s20 + m2;
         s22 = s20 - s13;
         s23 = s6 - s19;
         m4 = R13_DGC_6 * s22;
         m5 = R13_DGC_7 * s23;
         s43 = m4 + m5;
         m6 = R13_DGC_6 * s23;
         m7 = R13_DGC_7 * s22;
         s45 = m6 - m7;
         s47 = s21 + s18;
         s48 = s21 - s18;
         m8 = CRTM_13_7 * s16;
         s32 = s47 - m8;
         s28 = s47 + s16;
         m9 = CRTM_13_7 * s17;
         
         s33 = s48 + m9;
         s39 = s48 - s17;
         m10 = s39 * R13_DGC_12;
         // Output point 1: X(0)
         *out = s28 + in0;

         m11 = -(s28 * R13_DGC_1);
         s63 = m11 + in0;
         s24 = s63 + m10;
         s25 = s63 - m10;
         s61 = s36 + m0;
         s62 = s36 - m0;
         s46 = s42 + m1;
         s29 = s42 - m1;
         m12 = R13_DGC_2 * s61;
         m13 = R13_DGC_3 * s46;
         s40 = -(m12 + m13);

         m14 = R13_DGC_2 * s46;
         m15 = R13_DGC_3 * s61;
         s41 = m14 - m15;
         m16 = R13_DGC_10 * s29;
         m17 = R13_DGC_11 * s62;
         s34 = m17 - m16;
         m18 = R13_DGC_10 * s62;
         m19 = R13_DGC_11 * s29;
         s35 = m18 + m19;
         s26 = s41 + s34;
         s44 = s41 - s34;
         m20 = CRTM_13_1 * s44;
         s27 = s40 + s35;
         s64 = s40 - s35;
         m21 = CRTM_13_1 * s64;
         s30 = s7 + s14;
         m22 = R13_DGC_4 * s33;
         m23 = CRTM_13_3 * s30;
         s49 = m22 - m23;
         m24 = CRTM_13_2 * s30;
         m25 = R13_DGC_5 * s33;
         s37 = -(m24 + m25);

         s31 = s7 - s14;
         m26 = R13_DGC_8 * s32;
         m27 = CRTM_13_5 * s31;
         s50 = m26 - m27;
         m28 = CRTM_13_4 * s31;
         m29 = R13_DGC_9 * s32;
         s38 = -(m28 + m29);

         s51 = s37 + s38;
         s59 = s37 - s38;
         s52 = s49 + s50;
         s53 = s49 - s50;
         m30 = CRTM_13_8 * s52;
         // Output point 2: X(1)
         out[out_strides[1]] = s24 + m30;

         m31 = CRTM_13_8 * s26;
         // Output point 3: X(2)
         out[out_strides[2]] = s45 + m31;

         s54 = s24 - s52;
         s55 = s45 - s26;
         // Output point 6: X(5)
         out[out_strides[5]] = s54 + s59;
         // Output point 7: X(6)
         out[out_strides[6]] = s55 + m21;

         // Output point 8: X(7)
         out[out_strides[7]] = s54 - s59;
         // Output point 9: X(8)
         out[out_strides[8]] = m21 - s55;

         s56 = s27 - s43;
         m32 = CRTM_13_8 * s27;
         s57 = -(m32 + s43);
         m33 = CRTM_13_8 * s53;
         s58 = s25 - m33;
         s60 = s25 + s53;
         // Output point 4: X(3)
         out[out_strides[3]] = s60 - s51;
         // Output point 12: X(11)
         out[out_strides[11]] = s60 + s51;

         // Output point 10: X(9)
         out[out_strides[9]] = s58;
         // Output point 11: X(10)
         out[out_strides[10]] = s57;

         // Output point 5: X(4)
         out[out_strides[4]] = m20 + s56;
         // Output point 13: X(12)
         out[out_strides[12]] = s56 - m20;
     }
     AOCLFFTZ_LOG(DEBUG, global_logger_mode, "Exit");
 }

 static FFTZ_VOID r2hc_rfft13avx256_fp32_bwd(FFTZ_VOID *in_real,
                                             FFTZ_VOID *in_imag,
                                             FFTZ_VOID *out_real,
                                             FFTZ_VOID *out_imag, FFTZ_INTP n,
                                             aoclfftz_strides_t *strides,
                                             FFTZ_VOID *twd, FFTZ_UINT8 flag)
 {
     AOCLFFTZ_LOG(DEBUG, global_logger_mode, "Enter");
     const FFTZ_FLOAT CRTM_13_1 =
         1.732050807568877293527446341505872366942805254f;
     const FFTZ_FLOAT CRTM_13_2 =
         1.007074065727533254493747707736933954186697125f;
     const FFTZ_FLOAT CRTM_13_3 =
         0.531932498429674575175042127684371897596660533f;
     const FFTZ_FLOAT CRTM_13_4 =
         1.150281458948006242736771094910906776922003215f;
     const FFTZ_FLOAT CRTM_13_5 =
         0.348277202304271810011321589858529485233929352f;
     const FFTZ_FLOAT CRTM_13_6 =
         0.500000000000000000000000000000000000000000000f;
     const FFTZ_FLOAT CRTM_13_7 =
         2.000000000000000000000000000000000000000000000f;
     const FFTZ_FLOAT R13_DGC_1 =
         0.166666666666666666666666666666666666666666667f;
     const FFTZ_FLOAT R13_DGC_2 =
         0.256247671582936600958684654061725059144125175f;
     const FFTZ_FLOAT R13_DGC_3 =
         0.156891391051584611046832726756003269660212636f;
     const FFTZ_FLOAT R13_DGC_4 =
         0.774781170935234584261351932853525703557550433f;
     const FFTZ_FLOAT R13_DGC_5 =
         0.265966249214837287587521063842185948798330267f;
     const FFTZ_FLOAT R13_DGC_6 =
         0.600925212577331548853203544578415991041882762f;
     const FFTZ_FLOAT R13_DGC_7 =
         0.151805972074387731966205794490207080712856746f;
     const FFTZ_FLOAT R13_DGC_8 =
         0.227708958111581597949308691735310621069285120f;
     const FFTZ_FLOAT R13_DGC_9 =
         0.503537032863766627246873853868466977093348562f;
     const FFTZ_FLOAT R13_DGC_10 =
         0.300238635966332641462884626667381504676006424f;
     const FFTZ_FLOAT R13_DGC_11 =
         0.011599105605768290721655456654083252189827041f;
     const FFTZ_FLOAT R13_DGC_12 =
         0.516520780623489722840901288569017135705033622f;

     FFTZ_FLOAT *in = (FFTZ_FLOAT *)in_real;
     FFTZ_FLOAT *out = (FFTZ_FLOAT *)out_real;
#ifdef VOLATILE_STRIDE_ARRAY
     volatile FFTZ_INTP *in_strides = strides->in_strides;
     volatile FFTZ_INTP *out_strides = strides->out_strides;
#else
     INTP *in_strides = strides->in_strides;
     FFTZ_INTP *out_strides = strides->out_strides;
#endif
     FFTZ_INTP v_in_stride = strides->v_in_stride;
     FFTZ_INTP v_out_stride = strides->v_out_stride;

     FFTZ_INTP cnt;
     FFTZ_FLOAT *curr_in, *curr_out;
     FFTZ_INTP N = n / NUM_SETS_REAL_256_S;
     FFTZ_INTP remaining_sets = n % NUM_SETS_REAL_256_S;

     __m256 v_CRTM_13_1 = _mm256_broadcast_ss(&CRTM_13_1);
     __m256 v_CRTM_13_2 = _mm256_broadcast_ss(&CRTM_13_2);
     __m256 v_CRTM_13_3 = _mm256_broadcast_ss(&CRTM_13_3);
     __m256 v_CRTM_13_4 = _mm256_broadcast_ss(&CRTM_13_4);
     __m256 v_CRTM_13_5 = _mm256_broadcast_ss(&CRTM_13_5);
     __m256 v_CRTM_13_6 = _mm256_broadcast_ss(&CRTM_13_6);
     __m256 v_CRTM_13_7 = _mm256_broadcast_ss(&CRTM_13_7);
     __m256 v_R13_DGC_1 = _mm256_broadcast_ss(&R13_DGC_1);
     __m256 v_R13_DGC_2 = _mm256_broadcast_ss(&R13_DGC_2);
     __m256 v_R13_DGC_3 = _mm256_broadcast_ss(&R13_DGC_3);
     __m256 v_R13_DGC_4 = _mm256_broadcast_ss(&R13_DGC_4);
     __m256 v_R13_DGC_5 = _mm256_broadcast_ss(&R13_DGC_5);
     __m256 v_R13_DGC_6 = _mm256_broadcast_ss(&R13_DGC_6);
     __m256 v_R13_DGC_7 = _mm256_broadcast_ss(&R13_DGC_7);
     __m256 v_R13_DGC_8 = _mm256_broadcast_ss(&R13_DGC_8);
     __m256 v_R13_DGC_9 = _mm256_broadcast_ss(&R13_DGC_9);
     __m256 v_R13_DGC_10 = _mm256_broadcast_ss(&R13_DGC_10);
     __m256 v_R13_DGC_11 = _mm256_broadcast_ss(&R13_DGC_11);
     __m256 v_R13_DGC_12 = _mm256_broadcast_ss(&R13_DGC_12);

     for (cnt = 0; cnt < N; cnt++)
     {
         __m256 v_in0, v_in1, v_in2, v_in3, v_in4, v_in5, v_in6, v_in7, v_in8,
                v_in9, v_in10, v_in11, v_in12;
         __m256 v_s0, v_s1, v_s2, v_s3, v_s4, v_s5, v_s6, v_s7, v_s8, v_s9,
                v_s10, v_s11, v_s12, v_s13, v_s14, v_s15, v_s16, v_s17, v_s18,
                v_s19, v_s20, v_s21, v_s22, v_s23, v_s24, v_s25, v_s26, v_s27,
                v_s28, v_s29, v_s30, v_s31, v_s32, v_s33, v_s34, v_s35, v_s36,
                v_s37, v_s38, v_s39, v_s40, v_s41, v_s42, v_s43, v_s44, v_s45,
                v_s46, v_s47, v_s48, v_s49, v_s50, v_s51, v_s52, v_s53, v_s54,
                v_s55, v_s56, v_s57, v_s58, v_s59, v_s60, v_s61, v_s62;
         __m256 v_m0, v_m1, v_m2, v_m3, v_m4, v_m5, v_m6, v_m7, v_m8, v_m9,
                v_m10, v_m11, v_m12, v_m13, v_m14, v_m15, v_m16, v_m17, v_m18,
                v_m19, v_m20, v_m21, v_m22, v_m23, v_m24, v_m25, v_m26, v_m27,
                v_m28, v_m29, v_m30, v_m31, v_m32, v_m33, v_m34;
         __m256 v_out0, v_out1, v_out2, v_out3, v_out4, v_out5, v_out6, v_out7,
                v_out8, v_out9, v_out10, v_out11, v_out12;
 
         curr_in = in;
         curr_out = out;

         // Input point 1: X(0)
         LDR_256_S(curr_in, v_in_stride, v_in0);
         // Input point 2: X(1) & Input point 3: X(2)
         curr_in = in + in_strides[1];
         LDRI_2x256_S(curr_in, v_in_stride, v_in1, v_in2);
         // Input point 4: X(3) & Input point 5: X(4)
         curr_in = in + in_strides[3];
         LDRI_2x256_S(curr_in, v_in_stride, v_in3, v_in4);
         // Input point 6: X(5) & Input point 7: X(6)
         curr_in = in + in_strides[5];
         LDRI_2x256_S(curr_in, v_in_stride, v_in5, v_in6);
         // Input point 8: X(7) & Input point 9: X(8)
         curr_in = in + in_strides[7];
         LDRI_2x256_S(curr_in, v_in_stride, v_in7, v_in8);
         // Input point 10: X(9) & Input point 11: X(10)
         curr_in = in + in_strides[9];
         LDRI_2x256_S(curr_in, v_in_stride, v_in9, v_in10);
         // Input point 12: X(11) & Input point 13: X(12)
         curr_in = in + in_strides[11];
         LDRI_2x256_S(curr_in, v_in_stride, v_in11, v_in12);
 
         v_s0 = _mm256_sub_ps(v_in6, v_in8);
         v_m0 = _mm256_mul_ps(v_CRTM_13_7, v_in2);
         v_s1 = _mm256_sub_ps(v_m0, v_s0);
         v_s59 = _mm256_add_ps(v_in6, v_in8);
         v_m1 = _mm256_mul_ps(v_CRTM_13_1, v_s59);
         v_s2 = _mm256_add_ps(v_in12, v_in4);
         v_s60 = _mm256_sub_ps(v_in12, v_in4);
         v_m2 = _mm256_mul_ps(v_CRTM_13_1, v_s60);
         v_m3 = _mm256_mul_ps(v_CRTM_13_7, v_in10);
         v_s3 = _mm256_sub_ps(v_s2, v_m3);
         v_s4 = _mm256_add_ps(v_s1, v_m2);
         v_s5 = _mm256_sub_ps(v_s3, v_m1);
         v_m4 = _mm256_mul_ps(v_R13_DGC_11, v_s4);
         v_m5 = _mm256_mul_ps(v_R13_DGC_10, v_s5);
         v_s6 = _mm256_add_ps(v_m4, v_m5);
         v_m6 = _mm256_mul_ps(v_R13_DGC_10, v_s4);
         v_m7 = _mm256_mul_ps(v_R13_DGC_11, v_s5);
         v_s7 = _mm256_sub_ps(v_m6, v_m7);
         v_s8 = _mm256_add_ps(v_in2, v_s0);
         v_s9 = _mm256_add_ps(v_s2, v_in10);
         v_m8 = _mm256_mul_ps(v_CRTM_13_4, v_s8);
         v_m9 = _mm256_mul_ps(v_CRTM_13_5, v_s9);
         
         v_s10 = _mm256_sub_ps(v_m8, v_m9);
         v_m10 = _mm256_mul_ps(v_CRTM_13_5, v_s8);
         v_m11 = _mm256_mul_ps(v_CRTM_13_4, v_s9);
         v_s11 = _mm256_add_ps(v_m10, v_m11);
         v_s12 = _mm256_sub_ps(v_s1, v_m2);
         v_s13 = _mm256_add_ps(v_m1, v_s3);
         v_m12 = _mm256_mul_ps(v_R13_DGC_3, v_s12);
         v_m13 = _mm256_mul_ps(v_R13_DGC_2, v_s13);
         v_s14 = _mm256_add_ps(v_m12, v_m13);
         v_m14 = _mm256_mul_ps(v_R13_DGC_3, v_s13);
         v_m15 = _mm256_mul_ps(v_R13_DGC_2, v_s12);
         v_s15 = _mm256_sub_ps(v_m14, v_m15);
         v_s16 = _mm256_add_ps(v_in3, v_in11);
         v_s17 = _mm256_add_ps(v_in9, v_s16);
         v_m16 = _mm256_mul_ps(v_CRTM_13_6, v_s16);
         v_s18 = _mm256_sub_ps(v_in9, v_m16);
         v_s19 = _mm256_sub_ps(v_in3, v_in11);
         v_s20 = _mm256_add_ps(v_in5, v_in7);
         v_s21 = _mm256_add_ps(v_in1, v_s20);
         v_m17 = _mm256_mul_ps(v_CRTM_13_6, v_s20);
         v_s22 = _mm256_sub_ps(v_in1, v_m17);
         v_s23 = _mm256_sub_ps(v_in5, v_in7);
         v_s24 = _mm256_sub_ps(v_s21, v_s17);
         v_m18 = _mm256_mul_ps(v_R13_DGC_6, v_s24);
         v_s25 = _mm256_add_ps(v_s21, v_s17);
         v_m33 = _mm256_mul_ps(v_CRTM_13_7, v_s25);
         // Output point 1: x(0)
         v_out0 = _mm256_add_ps(v_m33, v_in0);
         STR_256_S(curr_out, v_out_stride, v_out0);

         v_m34 = _mm256_mul_ps(v_R13_DGC_1, v_s25);
         v_s26 = _mm256_sub_ps(v_in0, v_m34);
         v_s27 = _mm256_add_ps(v_s23, v_s19);
         v_s28 = _mm256_add_ps(v_s22, v_s18);
         v_m19 = _mm256_mul_ps(v_R13_DGC_9, v_s27);
         v_m20 = _mm256_mul_ps(v_R13_DGC_7, v_s28);
         v_s29 = _mm256_add_ps(v_m19, v_m20);
         v_s30 = _mm256_sub_ps(v_s22, v_s18);
         v_s31 = _mm256_sub_ps(v_s23, v_s19);
         v_m21 = _mm256_mul_ps(v_R13_DGC_12, v_s30);
         v_m22 = _mm256_mul_ps(v_R13_DGC_5, v_s31);
         v_s32 = _mm256_sub_ps(v_m21, v_m22);
         v_s61 = _mm256_add_ps(v_s6, v_s14);
         v_m23 = _mm256_mul_ps(v_CRTM_13_1, v_s61);
         v_s62 = _mm256_sub_ps(v_s7, v_s15);
         v_m24 = _mm256_mul_ps(v_CRTM_13_1, v_s62);
         v_s33 = _mm256_add_ps(v_s7, v_s15);
         v_s34 = _mm256_sub_ps(v_s10, v_s33);
         v_m25 = _mm256_mul_ps(v_CRTM_13_7, v_s33);
         v_s35 = _mm256_add_ps(v_m25, v_s10);
         v_s36 = _mm256_sub_ps(v_s6, v_s14);
         v_m26 = _mm256_mul_ps(v_CRTM_13_7, v_s36);
         v_s37 = _mm256_sub_ps(v_m26, v_s11);
         v_s38 = _mm256_add_ps(v_s36, v_s11);
         v_m27 = _mm256_mul_ps(v_R13_DGC_4, v_s31);
         v_m28 = _mm256_mul_ps(v_CRTM_13_3, v_s30);
         v_s39 = _mm256_add_ps(v_m27, v_m28);
         v_m29 = _mm256_mul_ps(v_R13_DGC_8, v_s27);
         v_m30 = _mm256_mul_ps(v_CRTM_13_2, v_s28);
         v_s40 = _mm256_sub_ps(v_m29, v_m30);

         v_s41 = _mm256_sub_ps(v_s39, v_s40);
         v_s42 = _mm256_add_ps(v_s39, v_s40);
         v_s43 = _mm256_sub_ps(v_s26, v_s29);
         v_s44 = _mm256_sub_ps(v_m18, v_s32);
         v_s45 = _mm256_sub_ps(v_s43, v_s44);
         v_s46 = _mm256_add_ps(v_s44, v_s43);
         v_m31 = _mm256_mul_ps(v_CRTM_13_7, v_s29);
         v_s47 = _mm256_add_ps(v_m31, v_s26);
         v_m32 = _mm256_mul_ps(v_CRTM_13_7, v_s32);
         v_s48 = _mm256_add_ps(v_m32, v_m18);
         v_s49 = _mm256_sub_ps(v_s47, v_s48);
         // Output point 9: x(8)
         v_out8 = _mm256_add_ps(v_s49, v_s35);
         curr_out = out + out_strides[8];
         STR_256_S(curr_out, v_out_stride, v_out8);

         // Output point 6: x(5)
         v_out5 = _mm256_sub_ps(v_s49, v_s35);
         curr_out = out + out_strides[5];
         STR_256_S(curr_out, v_out_stride, v_out5);

         v_s50 = _mm256_add_ps(v_s48, v_s47);
         // Output point 13: x(12)
         v_out12 = _mm256_sub_ps(v_s50, v_s37);
         curr_out = out + out_strides[12];
         STR_256_S(curr_out, v_out_stride, v_out12);

         // Output point 2: x(1)
         v_out1 = _mm256_add_ps(v_s50, v_s37);
         curr_out = out + out_strides[1];
         STR_256_S(curr_out, v_out_stride, v_out1);

         v_s51 = _mm256_sub_ps(v_s45, v_m23);
         v_s52 = _mm256_sub_ps(v_s41, v_s34);
         // Output point 3: x(2)
         v_out2 = _mm256_add_ps(v_s51, v_s52);
         curr_out = out + out_strides[2];
         STR_256_S(curr_out, v_out_stride, v_out2);
         
         // Output point 8: x(7)
         v_out7 = _mm256_sub_ps(v_s51, v_s52);
         curr_out = out + out_strides[7];
         STR_256_S(curr_out, v_out_stride, v_out7);

         v_s53 = _mm256_sub_ps(v_s46, v_s38);
         v_s54 = _mm256_add_ps(v_s42, v_m24);
         // Output point 10: x(9)
         v_out9 = _mm256_add_ps(v_s54, v_s53);
         curr_out = out + out_strides[9];
         STR_256_S(curr_out, v_out_stride, v_out9);

         // Output point 4: x(3)
         v_out3 = _mm256_sub_ps(v_s53, v_s54);
         curr_out = out + out_strides[3];
         STR_256_S(curr_out, v_out_stride, v_out3);

         v_s55 = _mm256_sub_ps(v_s42, v_m24);
         v_s56 = _mm256_add_ps(v_s46, v_s38);
         // Output point 5: x(4)
         v_out4 = _mm256_add_ps(v_s55, v_s56);
         curr_out = out + out_strides[4];
         STR_256_S(curr_out, v_out_stride, v_out4);

         // Output point 11: x(10)
         v_out10 = _mm256_sub_ps(v_s56, v_s55);
         curr_out = out + out_strides[10];
         STR_256_S(curr_out, v_out_stride, v_out10);

         v_s57 = _mm256_add_ps(v_s45, v_m23);
         v_s58 = _mm256_add_ps(v_s41, v_s34);
         // Output point 7: x(6)
         v_out6 = _mm256_sub_ps(v_s57, v_s58);
         curr_out = out + out_strides[6];
         STR_256_S(curr_out, v_out_stride, v_out6);

         // Output point 12: x(11)
         v_out11 = _mm256_add_ps(v_s57, v_s58);
         curr_out = out + out_strides[11];
         STR_256_S(curr_out, v_out_stride, v_out11);

         in += v_in_stride * NUM_SETS_REAL_256_S;
         out += v_out_stride * NUM_SETS_REAL_256_S;
     }
     // tail cases
     if (remaining_sets & NUM_SETS_REAL_128_S)
     {
         __m128 v_in0, v_in1, v_in2, v_in3, v_in4, v_in5, v_in6, v_in7, v_in8,
                v_in9, v_in10, v_in11, v_in12;
         __m128 v_s0, v_s1, v_s2, v_s3, v_s4, v_s5, v_s6, v_s7, v_s8, v_s9,
                v_s10, v_s11, v_s12, v_s13, v_s14, v_s15, v_s16, v_s17, v_s18,
                v_s19, v_s20, v_s21, v_s22, v_s23, v_s24, v_s25, v_s26, v_s27,
                v_s28, v_s29, v_s30, v_s31, v_s32, v_s33, v_s34, v_s35, v_s36,
                v_s37, v_s38, v_s39, v_s40, v_s41, v_s42, v_s43, v_s44, v_s45,
                v_s46, v_s47, v_s48, v_s49, v_s50, v_s51, v_s52, v_s53, v_s54,
                v_s55, v_s56, v_s57, v_s58, v_s59, v_s60, v_s61, v_s62;
         __m128 v_m0, v_m1, v_m2, v_m3, v_m4, v_m5, v_m6, v_m7, v_m8, v_m9,
                v_m10, v_m11, v_m12, v_m13, v_m14, v_m15, v_m16, v_m17, v_m18,
                v_m19, v_m20, v_m21, v_m22, v_m23, v_m24, v_m25, v_m26, v_m27,
                v_m28, v_m29, v_m30, v_m31, v_m32, v_m33, v_m34;
         __m128 v_out0, v_out1, v_out2, v_out3, v_out4, v_out5, v_out6, v_out7,
                v_out8, v_out9, v_out10, v_out11, v_out12;
 
         curr_in = in;
         curr_out = out;
 
         __m128 v128_CRTM_13_1 = _mm256_castps256_ps128(v_CRTM_13_1);
         __m128 v128_CRTM_13_2 = _mm256_castps256_ps128(v_CRTM_13_2);
         __m128 v128_CRTM_13_3 = _mm256_castps256_ps128(v_CRTM_13_3);
         __m128 v128_CRTM_13_4 = _mm256_castps256_ps128(v_CRTM_13_4);
         __m128 v128_CRTM_13_5 = _mm256_castps256_ps128(v_CRTM_13_5);
         __m128 v128_CRTM_13_6 = _mm256_castps256_ps128(v_CRTM_13_6);
         __m128 v128_CRTM_13_7 = _mm256_castps256_ps128(v_CRTM_13_7);
         __m128 v128_R13_DGC_1 = _mm256_castps256_ps128(v_R13_DGC_1);
         __m128 v128_R13_DGC_2 = _mm256_castps256_ps128(v_R13_DGC_2);
         __m128 v128_R13_DGC_3 = _mm256_castps256_ps128(v_R13_DGC_3);
         __m128 v128_R13_DGC_4 = _mm256_castps256_ps128(v_R13_DGC_4);
         __m128 v128_R13_DGC_5 = _mm256_castps256_ps128(v_R13_DGC_5);
         __m128 v128_R13_DGC_6 = _mm256_castps256_ps128(v_R13_DGC_6);
         __m128 v128_R13_DGC_7 = _mm256_castps256_ps128(v_R13_DGC_7);
         __m128 v128_R13_DGC_8 = _mm256_castps256_ps128(v_R13_DGC_8);
         __m128 v128_R13_DGC_9 = _mm256_castps256_ps128(v_R13_DGC_9);
         __m128 v128_R13_DGC_10 = _mm256_castps256_ps128(v_R13_DGC_10);
         __m128 v128_R13_DGC_11 = _mm256_castps256_ps128(v_R13_DGC_11);
         __m128 v128_R13_DGC_12 = _mm256_castps256_ps128(v_R13_DGC_12);
 
         // Input point 1: X(0)
         LDR_128_S(curr_in, v_in_stride, v_in0);
         // Input point 2: X(1) & Input point 3: X(2)
         curr_in = in + in_strides[1];
         LDRI_2x128_S(curr_in, v_in_stride, v_in1, v_in2);
         // Input point 4: X(3) & Input point 5: X(4)
         curr_in = in + in_strides[3];
         LDRI_2x128_S(curr_in, v_in_stride, v_in3, v_in4);
         // Input point 6: X(5) & Input point 7: X(6)
         curr_in = in + in_strides[5];
         LDRI_2x128_S(curr_in, v_in_stride, v_in5, v_in6);
         // Input point 8: X(7) & Input point 9: X(8)
         curr_in = in + in_strides[7];
         LDRI_2x128_S(curr_in, v_in_stride, v_in7, v_in8);
         // Input point 10: X(9) & Input point 11: X(10)
         curr_in = in + in_strides[9];
         LDRI_2x128_S(curr_in, v_in_stride, v_in9, v_in10);
         // Input point 12: X(11) & Input point 13: X(12)
         curr_in = in + in_strides[11];
         LDRI_2x128_S(curr_in, v_in_stride, v_in11, v_in12);
 
         v_s0 = _mm_sub_ps(v_in6, v_in8);
         v_m0 = _mm_mul_ps(v128_CRTM_13_7, v_in2);
         v_s1 = _mm_sub_ps(v_m0, v_s0);
         v_s59 = _mm_add_ps(v_in6, v_in8);
         v_m1 = _mm_mul_ps(v128_CRTM_13_1, v_s59);
         v_s2 = _mm_add_ps(v_in12, v_in4);
         v_s60 = _mm_sub_ps(v_in12, v_in4);
         v_m2 = _mm_mul_ps(v128_CRTM_13_1, v_s60);
         v_m3 = _mm_mul_ps(v128_CRTM_13_7, v_in10);
         v_s3 = _mm_sub_ps(v_s2, v_m3);
         v_s4 = _mm_add_ps(v_s1, v_m2);
         v_s5 = _mm_sub_ps(v_s3, v_m1);
         v_m4 = _mm_mul_ps(v128_R13_DGC_11, v_s4);
         v_m5 = _mm_mul_ps(v128_R13_DGC_10, v_s5);
         v_s6 = _mm_add_ps(v_m4, v_m5);
         v_m6 = _mm_mul_ps(v128_R13_DGC_10, v_s4);
         v_m7 = _mm_mul_ps(v128_R13_DGC_11, v_s5);
         v_s7 = _mm_sub_ps(v_m6, v_m7);
         v_s8 = _mm_add_ps(v_in2, v_s0);
         v_s9 = _mm_add_ps(v_s2, v_in10);
         v_m8 = _mm_mul_ps(v128_CRTM_13_4, v_s8);
         v_m9 = _mm_mul_ps(v128_CRTM_13_5, v_s9);

         v_s10 = _mm_sub_ps(v_m8, v_m9);
         v_m10 = _mm_mul_ps(v128_CRTM_13_5, v_s8);
         v_m11 = _mm_mul_ps(v128_CRTM_13_4, v_s9);
         v_s11 = _mm_add_ps(v_m10, v_m11);
         v_s12 = _mm_sub_ps(v_s1, v_m2);
         v_s13 = _mm_add_ps(v_m1, v_s3);
         v_m12 = _mm_mul_ps(v128_R13_DGC_3, v_s12);
         v_m13 = _mm_mul_ps(v128_R13_DGC_2, v_s13);
         v_s14 = _mm_add_ps(v_m12, v_m13);
         v_m14 = _mm_mul_ps(v128_R13_DGC_3, v_s13);
         v_m15 = _mm_mul_ps(v128_R13_DGC_2, v_s12);
         v_s15 = _mm_sub_ps(v_m14, v_m15);
         v_s16 = _mm_add_ps(v_in3, v_in11);
         v_s17 = _mm_add_ps(v_in9, v_s16);
         v_m16 = _mm_mul_ps(v128_CRTM_13_6, v_s16);
         v_s18 = _mm_sub_ps(v_in9, v_m16);
         v_s19 = _mm_sub_ps(v_in3, v_in11);
         v_s20 = _mm_add_ps(v_in5, v_in7);
         v_s21 = _mm_add_ps(v_in1, v_s20);
         v_m17 = _mm_mul_ps(v128_CRTM_13_6, v_s20);
         v_s22 = _mm_sub_ps(v_in1, v_m17);
         v_s23 = _mm_sub_ps(v_in5, v_in7);
         v_s24 = _mm_sub_ps(v_s21, v_s17);
         v_m18 = _mm_mul_ps(v128_R13_DGC_6, v_s24);
         v_s25 = _mm_add_ps(v_s21, v_s17);
         v_m33 = _mm_mul_ps(v128_CRTM_13_7, v_s25);
         // Output point 1: x(0)
         v_out0 = _mm_add_ps(v_m33, v_in0);
         STR_128_S(curr_out, v_out_stride, v_out0);

         v_m34 = _mm_mul_ps(v128_R13_DGC_1, v_s25);
         v_s26 = _mm_sub_ps(v_in0, v_m34);
         v_s27 = _mm_add_ps(v_s23, v_s19);
         v_s28 = _mm_add_ps(v_s22, v_s18);
         v_m19 = _mm_mul_ps(v128_R13_DGC_9, v_s27);
         v_m20 = _mm_mul_ps(v128_R13_DGC_7, v_s28);
         v_s29 = _mm_add_ps(v_m19, v_m20);
         v_s30 = _mm_sub_ps(v_s22, v_s18);
         v_s31 = _mm_sub_ps(v_s23, v_s19);
         v_m21 = _mm_mul_ps(v128_R13_DGC_12, v_s30);
         v_m22 = _mm_mul_ps(v128_R13_DGC_5, v_s31);
         v_s32 = _mm_sub_ps(v_m21, v_m22);
         v_s61 = _mm_add_ps(v_s6, v_s14);
         v_m23 = _mm_mul_ps(v128_CRTM_13_1, v_s61);
         v_s62 = _mm_sub_ps(v_s7, v_s15);
         v_m24 = _mm_mul_ps(v128_CRTM_13_1, v_s62);
         v_s33 = _mm_add_ps(v_s7, v_s15);
         v_s34 = _mm_sub_ps(v_s10, v_s33);
         v_m25 = _mm_mul_ps(v128_CRTM_13_7, v_s33);
         v_s35 = _mm_add_ps(v_m25, v_s10);
         v_s36 = _mm_sub_ps(v_s6, v_s14);
         v_m26 = _mm_mul_ps(v128_CRTM_13_7, v_s36);
         v_s37 = _mm_sub_ps(v_m26, v_s11);
         v_s38 = _mm_add_ps(v_s36, v_s11);
         v_m27 = _mm_mul_ps(v128_R13_DGC_4, v_s31);
         v_m28 = _mm_mul_ps(v128_CRTM_13_3, v_s30);
         v_s39 = _mm_add_ps(v_m27, v_m28);
         v_m29 = _mm_mul_ps(v128_R13_DGC_8, v_s27);
         v_m30 = _mm_mul_ps(v128_CRTM_13_2, v_s28);
         v_s40 = _mm_sub_ps(v_m29, v_m30);

         v_s41 = _mm_sub_ps(v_s39, v_s40);
         v_s42 = _mm_add_ps(v_s39, v_s40);
         v_s43 = _mm_sub_ps(v_s26, v_s29);
         v_s44 = _mm_sub_ps(v_m18, v_s32);
         v_s45 = _mm_sub_ps(v_s43, v_s44);
         v_s46 = _mm_add_ps(v_s44, v_s43);
         v_m31 = _mm_mul_ps(v128_CRTM_13_7, v_s29);
         v_s47 = _mm_add_ps(v_m31, v_s26);
         v_m32 = _mm_mul_ps(v128_CRTM_13_7, v_s32);
         v_s48 = _mm_add_ps(v_m32, v_m18);
         v_s49 = _mm_sub_ps(v_s47, v_s48);
         // Output point 9: x(8)
         v_out8 = _mm_add_ps(v_s49, v_s35);
         curr_out = out + out_strides[8];
         STR_128_S(curr_out, v_out_stride, v_out8);

         // Output point 6: x(5)
         v_out5 = _mm_sub_ps(v_s49, v_s35);
         curr_out = out + out_strides[5];
         STR_128_S(curr_out, v_out_stride, v_out5);

         v_s50 = _mm_add_ps(v_s48, v_s47);
         // Output point 13: x(12)
         v_out12 = _mm_sub_ps(v_s50, v_s37);
         curr_out = out + out_strides[12];
         STR_128_S(curr_out, v_out_stride, v_out12);

         // Output point 2: x(1)
         v_out1 = _mm_add_ps(v_s50, v_s37);
         curr_out = out + out_strides[1];
         STR_128_S(curr_out, v_out_stride, v_out1);

         v_s51 = _mm_sub_ps(v_s45, v_m23);
         v_s52 = _mm_sub_ps(v_s41, v_s34);
         // Output point 3: x(2)
         v_out2 = _mm_add_ps(v_s51, v_s52);
         curr_out = out + out_strides[2];
         STR_128_S(curr_out, v_out_stride, v_out2);

         // Output point 8: x(7)
         v_out7 = _mm_sub_ps(v_s51, v_s52);
         curr_out = out + out_strides[7];
         STR_128_S(curr_out, v_out_stride, v_out7);

         v_s53 = _mm_sub_ps(v_s46, v_s38);
         v_s54 = _mm_add_ps(v_s42, v_m24);
         // Output point 10: x(9)
         v_out9 = _mm_add_ps(v_s54, v_s53);
         curr_out = out + out_strides[9];
         STR_128_S(curr_out, v_out_stride, v_out9);

         // Output point 4: x(3)
         v_out3 = _mm_sub_ps(v_s53, v_s54);
         curr_out = out + out_strides[3];
         STR_128_S(curr_out, v_out_stride, v_out3);

         v_s55 = _mm_sub_ps(v_s42, v_m24);
         v_s56 = _mm_add_ps(v_s46, v_s38);
         // Output point 5: x(4)
         v_out4 = _mm_add_ps(v_s55, v_s56);
         curr_out = out + out_strides[4];
         STR_128_S(curr_out, v_out_stride, v_out4);

         // Output point 11: x(10)
         v_out10 = _mm_sub_ps(v_s56, v_s55);
         curr_out = out + out_strides[10];
         STR_128_S(curr_out, v_out_stride, v_out10);

         v_s57 = _mm_add_ps(v_s45, v_m23);
         v_s58 = _mm_add_ps(v_s41, v_s34);
         // Output point 7: x(6)
         v_out6 = _mm_sub_ps(v_s57, v_s58);
         curr_out = out + out_strides[6];
         STR_128_S(curr_out, v_out_stride, v_out6);

         // Output point 12: x(11)
         v_out11 = _mm_add_ps(v_s57, v_s58);
         curr_out = out + out_strides[11];
         STR_128_S(curr_out, v_out_stride, v_out11);

         in += v_in_stride * NUM_SETS_REAL_128_S;
         out += v_out_stride * NUM_SETS_REAL_128_S;
     }

     if (remaining_sets & 2)
     {
         __m128 v_in0, v_in1, v_in2, v_in3, v_in4, v_in5, v_in6, v_in7, v_in8,
                v_in9, v_in10, v_in11, v_in12;
         __m128 v_s0, v_s1, v_s2, v_s3, v_s4, v_s5, v_s6, v_s7, v_s8, v_s9,
                v_s10, v_s11, v_s12, v_s13, v_s14, v_s15, v_s16, v_s17, v_s18,
                v_s19, v_s20, v_s21, v_s22, v_s23, v_s24, v_s25, v_s26, v_s27,
                v_s28, v_s29, v_s30, v_s31, v_s32, v_s33, v_s34, v_s35, v_s36,
                v_s37, v_s38, v_s39, v_s40, v_s41, v_s42, v_s43, v_s44, v_s45,
                v_s46, v_s47, v_s48, v_s49, v_s50, v_s51, v_s52, v_s53, v_s54,
                v_s55, v_s56, v_s57, v_s58, v_s59, v_s60, v_s61, v_s62;
         __m128 v_m0, v_m1, v_m2, v_m3, v_m4, v_m5, v_m6, v_m7, v_m8, v_m9,
                v_m10, v_m11, v_m12, v_m13, v_m14, v_m15, v_m16, v_m17, v_m18,
                v_m19, v_m20, v_m21, v_m22, v_m23, v_m24, v_m25, v_m26, v_m27,
                v_m28, v_m29, v_m30, v_m31, v_m32, v_m33, v_m34;
         __m128 v_out0, v_out1, v_out2, v_out3, v_out4, v_out5, v_out6, v_out7,
                v_out8, v_out9, v_out10, v_out11, v_out12;
 
         curr_in = in;
         curr_out = out;
 
         __m128 v128_CRTM_13_1 = _mm256_castps256_ps128(v_CRTM_13_1);
         __m128 v128_CRTM_13_2 = _mm256_castps256_ps128(v_CRTM_13_2);
         __m128 v128_CRTM_13_3 = _mm256_castps256_ps128(v_CRTM_13_3);
         __m128 v128_CRTM_13_4 = _mm256_castps256_ps128(v_CRTM_13_4);
         __m128 v128_CRTM_13_5 = _mm256_castps256_ps128(v_CRTM_13_5);
         __m128 v128_CRTM_13_6 = _mm256_castps256_ps128(v_CRTM_13_6);
         __m128 v128_CRTM_13_7 = _mm256_castps256_ps128(v_CRTM_13_7);
         __m128 v128_R13_DGC_1 = _mm256_castps256_ps128(v_R13_DGC_1);
         __m128 v128_R13_DGC_2 = _mm256_castps256_ps128(v_R13_DGC_2);
         __m128 v128_R13_DGC_3 = _mm256_castps256_ps128(v_R13_DGC_3);
         __m128 v128_R13_DGC_4 = _mm256_castps256_ps128(v_R13_DGC_4);
         __m128 v128_R13_DGC_5 = _mm256_castps256_ps128(v_R13_DGC_5);
         __m128 v128_R13_DGC_6 = _mm256_castps256_ps128(v_R13_DGC_6);
         __m128 v128_R13_DGC_7 = _mm256_castps256_ps128(v_R13_DGC_7);
         __m128 v128_R13_DGC_8 = _mm256_castps256_ps128(v_R13_DGC_8);
         __m128 v128_R13_DGC_9 = _mm256_castps256_ps128(v_R13_DGC_9);
         __m128 v128_R13_DGC_10 = _mm256_castps256_ps128(v_R13_DGC_10);
         __m128 v128_R13_DGC_11 = _mm256_castps256_ps128(v_R13_DGC_11);
         __m128 v128_R13_DGC_12 = _mm256_castps256_ps128(v_R13_DGC_12);
 
         // Input point 1: X(0)
         LDHR_128_S(curr_in, v_in_stride, v_in0);
         // Input point 2: X(1) & Input point 3: X(2)
         curr_in = in + in_strides[1];
         LDHRI_2x128_S(curr_in, v_in_stride, v_in1, v_in2);
         // Input point 4: X(3) & Input point 5: X(4)
         curr_in = in + in_strides[3];
         LDHRI_2x128_S(curr_in, v_in_stride, v_in3, v_in4);
         // Input point 6: X(5) & Input point 7: X(6)
         curr_in = in + in_strides[5];
         LDHRI_2x128_S(curr_in, v_in_stride, v_in5, v_in6);
         // Input point 8: X(7) & Input point 9: X(8)
         curr_in = in + in_strides[7];
         LDHRI_2x128_S(curr_in, v_in_stride, v_in7, v_in8);
         // Input point 10: X(9) & Input point 11: X(10)
         curr_in = in + in_strides[9];
         LDHRI_2x128_S(curr_in, v_in_stride, v_in9, v_in10);
         // Input point 12: X(11) & Input point 13: X(12)
         curr_in = in + in_strides[11];
         LDHRI_2x128_S(curr_in, v_in_stride, v_in11, v_in12);
 
         v_s0 = _mm_sub_ps(v_in6, v_in8);
         v_m0 = _mm_mul_ps(v128_CRTM_13_7, v_in2);
         v_s1 = _mm_sub_ps(v_m0, v_s0);
         v_s59 = _mm_add_ps(v_in6, v_in8);
         v_m1 = _mm_mul_ps(v128_CRTM_13_1, v_s59);
         v_s2 = _mm_add_ps(v_in12, v_in4);
         v_s60 = _mm_sub_ps(v_in12, v_in4);
         v_m2 = _mm_mul_ps(v128_CRTM_13_1, v_s60);
         v_m3 = _mm_mul_ps(v128_CRTM_13_7, v_in10);
         v_s3 = _mm_sub_ps(v_s2, v_m3);
         v_s4 = _mm_add_ps(v_s1, v_m2);
         v_s5 = _mm_sub_ps(v_s3, v_m1);
         v_m4 = _mm_mul_ps(v128_R13_DGC_11, v_s4);
         v_m5 = _mm_mul_ps(v128_R13_DGC_10, v_s5);
         v_s6 = _mm_add_ps(v_m4, v_m5);
         v_m6 = _mm_mul_ps(v128_R13_DGC_10, v_s4);
         v_m7 = _mm_mul_ps(v128_R13_DGC_11, v_s5);
         v_s7 = _mm_sub_ps(v_m6, v_m7);
         v_s8 = _mm_add_ps(v_in2, v_s0);
         v_s9 = _mm_add_ps(v_s2, v_in10);
         v_m8 = _mm_mul_ps(v128_CRTM_13_4, v_s8);
         v_m9 = _mm_mul_ps(v128_CRTM_13_5, v_s9);

         v_s10 = _mm_sub_ps(v_m8, v_m9);
         v_m10 = _mm_mul_ps(v128_CRTM_13_5, v_s8);
         v_m11 = _mm_mul_ps(v128_CRTM_13_4, v_s9);
         v_s11 = _mm_add_ps(v_m10, v_m11);
         v_s12 = _mm_sub_ps(v_s1, v_m2);
         v_s13 = _mm_add_ps(v_m1, v_s3);
         v_m12 = _mm_mul_ps(v128_R13_DGC_3, v_s12);
         v_m13 = _mm_mul_ps(v128_R13_DGC_2, v_s13);
         v_s14 = _mm_add_ps(v_m12, v_m13);
         v_m14 = _mm_mul_ps(v128_R13_DGC_3, v_s13);
         v_m15 = _mm_mul_ps(v128_R13_DGC_2, v_s12);
         v_s15 = _mm_sub_ps(v_m14, v_m15);
         v_s16 = _mm_add_ps(v_in3, v_in11);
         v_s17 = _mm_add_ps(v_in9, v_s16);
         v_m16 = _mm_mul_ps(v128_CRTM_13_6, v_s16);
         v_s18 = _mm_sub_ps(v_in9, v_m16);
         v_s19 = _mm_sub_ps(v_in3, v_in11);
         v_s20 = _mm_add_ps(v_in5, v_in7);
         v_s21 = _mm_add_ps(v_in1, v_s20);
         v_m17 = _mm_mul_ps(v128_CRTM_13_6, v_s20);
         v_s22 = _mm_sub_ps(v_in1, v_m17);
         v_s23 = _mm_sub_ps(v_in5, v_in7);
         v_s24 = _mm_sub_ps(v_s21, v_s17);
         v_m18 = _mm_mul_ps(v128_R13_DGC_6, v_s24);
         v_s25 = _mm_add_ps(v_s21, v_s17);
         v_m33 = _mm_mul_ps(v128_CRTM_13_7, v_s25);
         // Output point 1: x(0)
         v_out0 = _mm_add_ps(v_m33, v_in0);
         STHR_128_S(curr_out, v_out_stride, v_out0);

         v_m34 = _mm_mul_ps(v128_R13_DGC_1, v_s25);
         v_s26 = _mm_sub_ps(v_in0, v_m34);
         v_s27 = _mm_add_ps(v_s23, v_s19);
         v_s28 = _mm_add_ps(v_s22, v_s18);
         v_m19 = _mm_mul_ps(v128_R13_DGC_9, v_s27);
         v_m20 = _mm_mul_ps(v128_R13_DGC_7, v_s28);
         v_s29 = _mm_add_ps(v_m19, v_m20);
         v_s30 = _mm_sub_ps(v_s22, v_s18);
         v_s31 = _mm_sub_ps(v_s23, v_s19);
         v_m21 = _mm_mul_ps(v128_R13_DGC_12, v_s30);
         v_m22 = _mm_mul_ps(v128_R13_DGC_5, v_s31);
         v_s32 = _mm_sub_ps(v_m21, v_m22);
         v_s61 = _mm_add_ps(v_s6, v_s14);
         v_m23 = _mm_mul_ps(v128_CRTM_13_1, v_s61);
         v_s62 = _mm_sub_ps(v_s7, v_s15);
         v_m24 = _mm_mul_ps(v128_CRTM_13_1, v_s62);
         v_s33 = _mm_add_ps(v_s7, v_s15);
         v_s34 = _mm_sub_ps(v_s10, v_s33);
         v_m25 = _mm_mul_ps(v128_CRTM_13_7, v_s33);
         v_s35 = _mm_add_ps(v_m25, v_s10);
         v_s36 = _mm_sub_ps(v_s6, v_s14);
         v_m26 = _mm_mul_ps(v128_CRTM_13_7, v_s36);
         v_s37 = _mm_sub_ps(v_m26, v_s11);
         v_s38 = _mm_add_ps(v_s36, v_s11);
         v_m27 = _mm_mul_ps(v128_R13_DGC_4, v_s31);
         v_m28 = _mm_mul_ps(v128_CRTM_13_3, v_s30);
         v_s39 = _mm_add_ps(v_m27, v_m28);
         v_m29 = _mm_mul_ps(v128_R13_DGC_8, v_s27);
         v_m30 = _mm_mul_ps(v128_CRTM_13_2, v_s28);
         v_s40 = _mm_sub_ps(v_m29, v_m30);
         v_s41 = _mm_sub_ps(v_s39, v_s40);
         v_s42 = _mm_add_ps(v_s39, v_s40);
         v_s43 = _mm_sub_ps(v_s26, v_s29);
         v_s44 = _mm_sub_ps(v_m18, v_s32);
         v_s45 = _mm_sub_ps(v_s43, v_s44);
         v_s46 = _mm_add_ps(v_s44, v_s43);
         v_m31 = _mm_mul_ps(v128_CRTM_13_7, v_s29);
         v_s47 = _mm_add_ps(v_m31, v_s26);
         v_m32 = _mm_mul_ps(v128_CRTM_13_7, v_s32);
         v_s48 = _mm_add_ps(v_m32, v_m18);
         v_s49 = _mm_sub_ps(v_s47, v_s48);
         // Output point 9: x(8)
         v_out8 = _mm_add_ps(v_s49, v_s35);
         curr_out = out + out_strides[8];
         STHR_128_S(curr_out, v_out_stride, v_out8);
         
         // Output point 6: x(5)
         v_out5 = _mm_sub_ps(v_s49, v_s35);
         curr_out = out + out_strides[5];
         STHR_128_S(curr_out, v_out_stride, v_out5);

         v_s50 = _mm_add_ps(v_s48, v_s47);
         // Output point 13: x(12)
         v_out12 = _mm_sub_ps(v_s50, v_s37);
         curr_out = out + out_strides[12];
         STHR_128_S(curr_out, v_out_stride, v_out12);

         // Output point 2: x(1)
         v_out1 = _mm_add_ps(v_s50, v_s37);
         curr_out = out + out_strides[1];
         STHR_128_S(curr_out, v_out_stride, v_out1);

         v_s51 = _mm_sub_ps(v_s45, v_m23);
         v_s52 = _mm_sub_ps(v_s41, v_s34);
         // Output point 3: x(2)
         v_out2 = _mm_add_ps(v_s51, v_s52);
         curr_out = out + out_strides[2];
         STHR_128_S(curr_out, v_out_stride, v_out2);

         // Output point 8: x(7)
         v_out7 = _mm_sub_ps(v_s51, v_s52);
         curr_out = out + out_strides[7];
         STHR_128_S(curr_out, v_out_stride, v_out7);

         v_s53 = _mm_sub_ps(v_s46, v_s38);
         v_s54 = _mm_add_ps(v_s42, v_m24);
         // Output point 10: x(9)
         v_out9 = _mm_add_ps(v_s54, v_s53);
         curr_out = out + out_strides[9];
         STHR_128_S(curr_out, v_out_stride, v_out9);

         // Output point 4: x(3)
         v_out3 = _mm_sub_ps(v_s53, v_s54);
         curr_out = out + out_strides[3];
         STHR_128_S(curr_out, v_out_stride, v_out3);

         v_s55 = _mm_sub_ps(v_s42, v_m24);
         v_s56 = _mm_add_ps(v_s46, v_s38);
         // Output point 5: x(4)
         v_out4 = _mm_add_ps(v_s55, v_s56);
         curr_out = out + out_strides[4];
         STHR_128_S(curr_out, v_out_stride, v_out4);

         // Output point 11: x(10)
         v_out10 = _mm_sub_ps(v_s56, v_s55);
         curr_out = out + out_strides[10];
         STHR_128_S(curr_out, v_out_stride, v_out10);

         v_s57 = _mm_add_ps(v_s45, v_m23);
         v_s58 = _mm_add_ps(v_s41, v_s34);
         // Output point 7: x(6)
         v_out6 = _mm_sub_ps(v_s57, v_s58);
         curr_out = out + out_strides[6];
         STHR_128_S(curr_out, v_out_stride, v_out6);

         // Output point 12: x(11)
         v_out11 = _mm_add_ps(v_s57, v_s58);
         curr_out = out + out_strides[11];
         STHR_128_S(curr_out, v_out_stride, v_out11);

         in = in + (v_in_stride << 1);
         out = out + (v_out_stride << 1);
     }

     if (remaining_sets & 1)
     {
         FFTZ_FLOAT in0, in1, in2, in3, in4, in5, in6, in7, in8, in9, in10,
                    in11, in12;
         FFTZ_FLOAT s0, s1, s2, s3, s4, s5, s6, s7, s8, s9, s10, s11, s12, s13,
                    s14, s15, s16, s17, s18, s19, s20, s21, s22, s23, s24, s25,
                    s26, s27, s28, s29, s30, s31, s32, s33, s34, s35, s36, s37,
                    s38, s39, s40, s41, s42, s43, s44, s45, s46, s47, s48, s49,
                    s50, s51, s52, s53, s54, s55, s56, s57, s58, s59, s60, s61,
                    s62;
         FFTZ_FLOAT m0, m1, m2, m3, m4, m5, m6, m7, m8, m9, m10, m11, m12, m13,
                    m14, m15, m16, m17, m18, m19, m20, m21, m22, m23, m24, m25,
                    m26, m27, m28, m29, m30, m31, m32, m33, m34;

         // Input point 1: X(0)
         in0 = *in;
         // Input point 2: X(1)
         in1 = in[in_strides[1]];
         // Input point 3: X(2)
         in2 = in[in_strides[2]];
         // Input point 4: X(3)
         in3 = in[in_strides[3]];
         // Input point 5: X(4)
         in4 = in[in_strides[4]];
         // Input point 6: X(5)
         in5 = in[in_strides[5]];
         // Input point 7: X(6)
         in6 = in[in_strides[6]];
         // Input point 8: X(7)
         in7 = in[in_strides[7]];
         // Input point 9: X(8)
         in8 = in[in_strides[8]];
         // Input point 10: X(9)
         in9 = in[in_strides[9]];
         // Input point 11: X(10)
         in10 = in[in_strides[10]];
         // Input point 12: X(11)
         in11 = in[in_strides[11]];
         // Input point 13: X(12)
         in12 = in[in_strides[12]];
 
         s0 = in6 - in8;
         m0 = CRTM_13_7 * in2;
         s1 = m0 - s0;
         s59 = in6 + in8;
         m1 = CRTM_13_1 * s59;
         s2 = in12 + in4;
         s60 = in12 - in4;
         m2 = CRTM_13_1 * s60;
         m3 = CRTM_13_7 * in10;
         s3 = s2 - m3;
         s4 = s1 + m2;
         s5 = s3 - m1;
         m4 = R13_DGC_11 * s4;
         m5 = R13_DGC_10 * s5;
         s6 = m4 + m5;
         m6 = R13_DGC_10 * s4;
         m7 = R13_DGC_11 * s5;
         s7 = m6 - m7;
         s8 = in2 + s0;
         s9 = s2 + in10;
         m8 = CRTM_13_4 * s8;
         m9 = CRTM_13_5 * s9;

         s10 = m8 - m9;
         m10 = CRTM_13_5 * s8;
         m11 = CRTM_13_4 * s9;
         s11 = m10 + m11;
         s12 = s1 - m2;
         s13 = m1 + s3;
         m12 = R13_DGC_3 * s12;
         m13 = R13_DGC_2 * s13;
         s14 = m12 + m13;
         m14 = R13_DGC_3 * s13;
         m15 = R13_DGC_2 * s12;
         s15 = m14 - m15;
         s16 = in3 + in11;
         s17 = in9 + s16;
         m16 = CRTM_13_6 * s16;
         s18 = in9 - m16;
         s19 = in3 - in11;
         s20 = in5 + in7;
         s21 = in1 + s20;
         m17 = CRTM_13_6 * s20;
         s22 = in1 - m17;
         s23 = in5 - in7;
         s24 = s21 - s17;
         m18 = R13_DGC_6 * s24;
         s25 = s21 + s17;
         m33 = CRTM_13_7 * s25;
         // Output point 1: x(0)
         out[out_strides[0]] = m33 + in0;

         m34 = R13_DGC_1 * s25;
         s26 = in0 - m34;
         s27 = s23 + s19;
         s28 = s22 + s18;
         m19 = R13_DGC_9 * s27;
         m20 = R13_DGC_7 * s28;
         s29 = m19 + m20;
         s30 = s22 - s18;
         s31 = s23 - s19;
         m21 = R13_DGC_12 * s30;
         m22 = R13_DGC_5 * s31;
         s32 = m21 - m22;
         s61 = s6 + s14;
         m23 = CRTM_13_1 * s61;
         s62 = s7 - s15;
         m24 = CRTM_13_1 * s62;
         s33 = s7 + s15;
         s34 = s10 - s33;
         m25 = CRTM_13_7 * s33;
         s35 = m25 + s10;
         s36 = s6 - s14;
         m26 = CRTM_13_7 * s36;
         s37 = m26 - s11;
         s38 = s36 + s11;
         m27 = R13_DGC_4 * s31;
         m28 = CRTM_13_3 * s30;
         s39 = m27 + m28;
         m29 = R13_DGC_8 * s27;
         m30 = CRTM_13_2 * s28;
         s40 = m29 - m30;

         s41 = s39 - s40;
         s42 = s39 + s40;
         s43 = s26 - s29;
         s44 = m18 - s32;
         s45 = s43 - s44;
         s46 = s44 + s43;
         m31 = CRTM_13_7 * s29;
         s47 = m31 + s26;
         m32 = CRTM_13_7 * s32;
         s48 = m32 + m18;
         s49 = s47 - s48;
         // Output point 9: x(8)
         out[out_strides[8]] = s49 + s35;
         // Output point 6: x(5)
         out[out_strides[5]] = s49 - s35;

         s50 = s48 + s47;
         // Output point 13: x(12)
         out[out_strides[12]] = s50 - s37;
         // Output point 2: x(1)
         out[out_strides[1]] = s50 + s37;

         s51 = s45 - m23;
         s52 = s41 - s34;
         // Output point 3: x(2)
         out[out_strides[2]] = s51 + s52;
         // Output point 8: x(7)
         out[out_strides[7]] = s51 - s52;

         s53 = s46 - s38;
         s54 = s42 + m24;
         // Output point 10: x(9)
         out[out_strides[9]] = s54 + s53;
         // Output point 4: x(3)
         out[out_strides[3]] = s53 - s54;

         s55 = s42 - m24;
         s56 = s46 + s38;
         // Output point 5: x(4)
         out[out_strides[4]] = s55 + s56;
         // Output point 11: x(10)
         out[out_strides[10]] = s56 - s55;

         s57 = s45 + m23;
         s58 = s41 + s34;
         // Output point 7: x(6)
         out[out_strides[6]] = s57 - s58;
         // Output point 12: x(11)
         out[out_strides[11]] = s57 + s58;
     }
     AOCLFFTZ_LOG(DEBUG, global_logger_mode, "Exit");
 }

 static FFTZ_VOID r2hc_rfft13avx256_fp64_fwd(FFTZ_VOID *in_real,
                                             FFTZ_VOID *in_imag,
                                             FFTZ_VOID *out_real,
                                             FFTZ_VOID *out_imag, FFTZ_INTP n,
                                             aoclfftz_strides_t *strides,
                                             FFTZ_VOID *twd, FFTZ_UINT8 flag)
 {
     AOCLFFTZ_LOG(DEBUG, global_logger_mode, "Enter");
     const FFTZ_DOUBLE CRTM_13_1 =
         1.73205080756887719317660412343684583902359008789060;
     const FFTZ_DOUBLE CRTM_13_2 =
         0.38739058546761714712838513245787843298017865366345;
     const FFTZ_DOUBLE CRTM_13_3 =
         0.13298312460741867056777367279491668244720145803175;
     const FFTZ_DOUBLE CRTM_13_4 =
         0.11385447905579067614456922127652058891458054833493;
     const FFTZ_DOUBLE CRTM_13_5 =
         0.25176851643188325619942540855595308576560112726050;
     const FFTZ_DOUBLE CRTM_13_6 =
         0.86602540378443864676372317075293618347140262700000;
     const FFTZ_DOUBLE CRTM_13_7 =
         0.50000000000000000000000000000000000000000000000000;
     const FFTZ_DOUBLE CRTM_13_8 =
         2.00000000000000000000000000000000000000000000000000;
     const FFTZ_DOUBLE R13_DGC_1 =
         0.08333333333333341693177688718872429763576971755296;
     const FFTZ_DOUBLE R13_DGC_2 =
         0.25624767158293663769405689230781736686616059890980;
     const FFTZ_DOUBLE R13_DGC_3 =
         0.15689139105158457352993666120753768234094124525375;
     const FFTZ_DOUBLE R13_DGC_4 =
         0.25826039031174479484752691274654956353060547073611;
     const FFTZ_DOUBLE R13_DGC_5 =
         0.26596624921483734113554734558983336489440291606351;
     const FFTZ_DOUBLE R13_DGC_6 =
         0.57514072947400312136838554745545338846100160800000;
     const FFTZ_DOUBLE R13_DGC_7 =
         0.17413860115213590500566079492926474261696467600000;
     const FFTZ_DOUBLE R13_DGC_8 =
         0.07590298603719379320004091411468632874055721777845;
     const FFTZ_DOUBLE R13_DGC_9 =
         0.50353703286376651239885081711190617153120225452101;
     const FFTZ_DOUBLE R13_DGC_10 =
         0.30023863596633260097945594679913483560085296630859;
     const FFTZ_DOUBLE R13_DGC_11 =
         0.01159910560576819966993600274918208015151321887970;
     const FFTZ_DOUBLE R13_DGC_12 =
         0.30046260628866572339745744102401658892631530761719;

     FFTZ_DOUBLE *in = (FFTZ_DOUBLE *)in_real;
     FFTZ_DOUBLE *out = (FFTZ_DOUBLE *)out_real;
#ifdef VOLATILE_STRIDE_ARRAY
     volatile FFTZ_INTP *in_strides = strides->in_strides;
     volatile FFTZ_INTP *out_strides = strides->out_strides;
#else
     FFTZ_INTP *in_strides = strides->in_strides;
     FFTZ_INTP *out_strides = strides->out_strides;
#endif
     FFTZ_INTP v_in_stride = strides->v_in_stride;
     FFTZ_INTP v_out_stride = strides->v_out_stride;

     FFTZ_INTP cnt;
     FFTZ_DOUBLE *curr_in, *curr_out;
     FFTZ_INTP N = n / NUM_SETS_REAL_256_D;
     FFTZ_INTP remaining_sets = n % NUM_SETS_REAL_256_D;

     __m256d v_CRTM_13_1 = _mm256_broadcast_sd(&CRTM_13_1);
     __m256d v_CRTM_13_2 = _mm256_broadcast_sd(&CRTM_13_2);
     __m256d v_CRTM_13_3 = _mm256_broadcast_sd(&CRTM_13_3);
     __m256d v_CRTM_13_4 = _mm256_broadcast_sd(&CRTM_13_4);
     __m256d v_CRTM_13_5 = _mm256_broadcast_sd(&CRTM_13_5);
     __m256d v_CRTM_13_6 = _mm256_broadcast_sd(&CRTM_13_6);
     __m256d v_CRTM_13_7 = _mm256_broadcast_sd(&CRTM_13_7);
     __m256d v_CRTM_13_8 = _mm256_broadcast_sd(&CRTM_13_8);
     __m256d v_R13_DGC_1 = _mm256_broadcast_sd(&R13_DGC_1);
     __m256d v_R13_DGC_2 = _mm256_broadcast_sd(&R13_DGC_2);
     __m256d v_R13_DGC_3 = _mm256_broadcast_sd(&R13_DGC_3);
     __m256d v_R13_DGC_4 = _mm256_broadcast_sd(&R13_DGC_4);
     __m256d v_R13_DGC_5 = _mm256_broadcast_sd(&R13_DGC_5);
     __m256d v_R13_DGC_6 = _mm256_broadcast_sd(&R13_DGC_6);
     __m256d v_R13_DGC_7 = _mm256_broadcast_sd(&R13_DGC_7);
     __m256d v_R13_DGC_8 = _mm256_broadcast_sd(&R13_DGC_8);
     __m256d v_R13_DGC_9 = _mm256_broadcast_sd(&R13_DGC_9);
     __m256d v_R13_DGC_10 = _mm256_broadcast_sd(&R13_DGC_10);
     __m256d v_R13_DGC_11 = _mm256_broadcast_sd(&R13_DGC_11);
     __m256d v_R13_DGC_12 = _mm256_broadcast_sd(&R13_DGC_12);

     for (cnt = 0; cnt < N; cnt++)
     {
         __m256d v_in0, v_in1, v_in2, v_in3, v_in4, v_in5, v_in6, v_in7, v_in8,
                 v_in9, v_in10, v_in11, v_in12;
         __m256d v_s0, v_s1, v_s2, v_s3, v_s4, v_s5, v_s6, v_s7, v_s8, v_s9,
                 v_s10, v_s11, v_s12, v_s13, v_s14, v_s15, v_s16, v_s17, v_s18,
                 v_s19, v_s20, v_s21, v_s22, v_s23, v_s24, v_s25, v_s26, v_s27,
                 v_s28, v_s29, v_s30, v_s31, v_s32, v_s33, v_s34, v_s35, v_s36,
                 v_s37, v_s38, v_s39, v_s40, v_s41, v_s42, v_s43, v_s44, v_s45,
                 v_s46, v_s47, v_s48, v_s49, v_s50, v_s51, v_s52, v_s53, v_s54,
                 v_s55, v_s56, v_s57, v_s58, v_s59, v_s60, v_s61, v_s62, v_s63,
                 v_s64;
         __m256d v_m0, v_m1, v_m2, v_m3, v_m4, v_m5, v_m6, v_m7, v_m8, v_m9,
                 v_m10, v_m11, v_m12, v_m13, v_m14, v_m15, v_m16, v_m17, v_m18,
                 v_m19, v_m20, v_m21, v_m22, v_m23, v_m24, v_m25, v_m26, v_m27,
                 v_m28, v_m29, v_m30, v_m31, v_m32, v_m33;
         __m256d v_out0, v_out1, v_out2, v_out3, v_out4, v_out5, v_out6, v_out7,
                 v_out8, v_out9, v_out10, v_out11, v_out12;
 
         curr_in = in;
         curr_out = out;
 
         curr_in = in;
         // Input point 1: x(0)
         LDR_256_D(curr_in, v_in_stride, v_in0);
         // Input point 2: x(1)
         curr_in = in + in_strides[1];
         LDR_256_D(curr_in, v_in_stride, v_in1);
         // Input point 3: x(2)
         curr_in = in + in_strides[2];
         LDR_256_D(curr_in, v_in_stride, v_in2);
         // Input point 4: x(3)
         curr_in = in + in_strides[3];
         LDR_256_D(curr_in, v_in_stride, v_in3);
         // Input point 5: x(4)
         curr_in = in + in_strides[4];
         LDR_256_D(curr_in, v_in_stride, v_in4);
         // Input point 6: x(5)
         curr_in = in + in_strides[5];
         LDR_256_D(curr_in, v_in_stride, v_in5);
         // Input point 7: x(6)
         curr_in = in + in_strides[6];
         LDR_256_D(curr_in, v_in_stride, v_in6);
         // Input point 8: x(7)
         curr_in = in + in_strides[7];
         LDR_256_D(curr_in, v_in_stride, v_in7);
         // Input point 9: x(8)
         curr_in = in + in_strides[8];
         LDR_256_D(curr_in, v_in_stride, v_in8);
         // Input point 10: x(9)
         curr_in = in + in_strides[9];
         LDR_256_D(curr_in, v_in_stride, v_in9);
         // Input point 11: x(10)
         curr_in = in + in_strides[10];
         LDR_256_D(curr_in, v_in_stride, v_in10);
         // Input point 12: x(11)
         curr_in = in + in_strides[11];
         LDR_256_D(curr_in, v_in_stride, v_in11);
         // Input point 13: x(12)
         curr_in = in + in_strides[12];
         LDR_256_D(curr_in, v_in_stride, v_in12);
 
         v_s0 = _mm256_add_pd(v_in2, v_in7);
         v_s1 = _mm256_sub_pd(v_in7, v_in2);
         v_s2 = _mm256_add_pd(v_in6, v_in11);
         v_s3 = _mm256_sub_pd(v_in11, v_in6);
         v_s4 = _mm256_add_pd(v_s0, v_s2);
         v_s5 = _mm256_sub_pd(v_s0, v_s2);
         v_m0 = _mm256_mul_pd(v_CRTM_13_6, v_s5);
         v_s6 = _mm256_add_pd(v_s1, v_s3);
         v_s7 = _mm256_sub_pd(v_s1, v_s3);
         v_s8 = _mm256_add_pd(v_in4, v_in10);
         v_s9 = _mm256_sub_pd(v_in10, v_in4);
         v_s10 = _mm256_add_pd(v_in3, v_in9);
         v_s11 = _mm256_sub_pd(v_in9, v_in3);
         v_s12 = _mm256_add_pd(v_s8, v_s10);
         v_s13 = _mm256_sub_pd(v_s8, v_s10);
         v_s14 = _mm256_sub_pd(v_s9, v_s11);
         v_s15 = _mm256_add_pd(v_s9, v_s11);
         v_m1 = _mm256_mul_pd(v_CRTM_13_6, v_s15);
         v_m2 = _mm256_mul_pd(v_CRTM_13_7, v_s13);

         v_s16 = _mm256_add_pd(v_s4, v_s12);
         v_s17 = _mm256_sub_pd(v_s4, v_s12);
         v_s18 = _mm256_add_pd(v_in8, v_in5);
         v_s19 = _mm256_sub_pd(v_in5, v_in8);
         v_m3 = _mm256_mul_pd(v_CRTM_13_7, v_s6);
         v_s42 = _mm256_add_pd(v_m3, v_s19);
         v_s20 = _mm256_sub_pd(v_in1, v_in12);
         v_s21 = _mm256_add_pd(v_in1, v_in12);
         v_s36 = _mm256_add_pd(v_s20, v_m2);
         v_s22 = _mm256_sub_pd(v_s20, v_s13);
         v_s23 = _mm256_sub_pd(v_s6, v_s19);
         v_m4 = _mm256_mul_pd(v_R13_DGC_6, v_s22);
         v_m5 = _mm256_mul_pd(v_R13_DGC_7, v_s23);
         v_s43 = _mm256_add_pd(v_m4, v_m5);
         v_m6 = _mm256_mul_pd(v_R13_DGC_6, v_s23);
         v_m7 = _mm256_mul_pd(v_R13_DGC_7, v_s22);
         v_s45 = _mm256_sub_pd(v_m6, v_m7);
         v_s47 = _mm256_add_pd(v_s21, v_s18);
         v_s48 = _mm256_sub_pd(v_s21, v_s18);
         v_m8 = _mm256_mul_pd(v_CRTM_13_7, v_s16);
         v_s32 = _mm256_sub_pd(v_s47, v_m8);
         v_s28 = _mm256_add_pd(v_s47, v_s16);
         v_m9 = _mm256_mul_pd(v_CRTM_13_7, v_s17);
         v_s33 = _mm256_add_pd(v_s48, v_m9);
         v_s39 = _mm256_sub_pd(v_s48, v_s17);
         v_m10 = _mm256_mul_pd(v_s39, v_R13_DGC_12);
         // Output point 1: X(0)
         v_out0 = _mm256_add_pd(v_s28, v_in0);
         STR_256_D(curr_out, v_out_stride, v_out0);

         v_m11 = NEGATE_256_D(_mm256_mul_pd(v_s28, v_R13_DGC_1));
         v_s63 = _mm256_add_pd(v_m11, v_in0);
         v_s24 = _mm256_add_pd(v_s63, v_m10);
         v_s25 = _mm256_sub_pd(v_s63, v_m10);
         v_s61 = _mm256_add_pd(v_s36, v_m0);
         v_s62 = _mm256_sub_pd(v_s36, v_m0);
         v_s46 = _mm256_add_pd(v_s42, v_m1);
         v_s29 = _mm256_sub_pd(v_s42, v_m1);
         v_m12 = _mm256_mul_pd(v_R13_DGC_2, v_s61);
         v_m13 = _mm256_mul_pd(v_R13_DGC_3, v_s46);
         v_s40 = NEGATE_256_D(_mm256_add_pd(v_m12, v_m13));

         v_m14 = _mm256_mul_pd(v_R13_DGC_2, v_s46);
         v_m15 = _mm256_mul_pd(v_R13_DGC_3, v_s61);
         v_s41 = _mm256_sub_pd(v_m14, v_m15);
         v_m16 = _mm256_mul_pd(v_R13_DGC_10, v_s29);
         v_m17 = _mm256_mul_pd(v_R13_DGC_11, v_s62);
         v_s34 = _mm256_sub_pd(v_m17, v_m16);
         v_m18 = _mm256_mul_pd(v_R13_DGC_10, v_s62);
         v_m19 = _mm256_mul_pd(v_R13_DGC_11, v_s29);
         v_s35 = _mm256_add_pd(v_m18, v_m19);
         v_s26 = _mm256_add_pd(v_s41, v_s34);
         v_s44 = _mm256_sub_pd(v_s41, v_s34);
         v_m20 = _mm256_mul_pd(v_CRTM_13_1, v_s44);
         v_s27 = _mm256_add_pd(v_s40, v_s35);
         v_s64 = _mm256_sub_pd(v_s40, v_s35);
         v_m21 = _mm256_mul_pd(v_CRTM_13_1, v_s64);
         v_s30 = _mm256_add_pd(v_s7, v_s14);
         v_m22 = _mm256_mul_pd(v_R13_DGC_4, v_s33);
         v_m23 = _mm256_mul_pd(v_CRTM_13_3, v_s30);
         v_s49 = _mm256_sub_pd(v_m22, v_m23);
         v_m24 = _mm256_mul_pd(v_CRTM_13_2, v_s30);
         v_m25 = _mm256_mul_pd(v_R13_DGC_5, v_s33);
         v_s37 = NEGATE_256_D(_mm256_add_pd(v_m24, v_m25));

         v_s31 = _mm256_sub_pd(v_s7, v_s14);
         v_m26 = _mm256_mul_pd(v_R13_DGC_8, v_s32);
         v_m27 = _mm256_mul_pd(v_CRTM_13_5, v_s31);
         v_s50 = _mm256_sub_pd(v_m26, v_m27);
         v_m28 = _mm256_mul_pd(v_CRTM_13_4, v_s31);
         v_m29 = _mm256_mul_pd(v_R13_DGC_9, v_s32);
         v_s38 = NEGATE_256_D(_mm256_add_pd(v_m28, v_m29));

         v_s51 = _mm256_add_pd(v_s37, v_s38);
         v_s59 = _mm256_sub_pd(v_s37, v_s38);
         v_s52 = _mm256_add_pd(v_s49, v_s50);
         v_s53 = _mm256_sub_pd(v_s49, v_s50);
         v_m30 = _mm256_mul_pd(v_CRTM_13_8, v_s52);
         // Output point 2: X(1)
         v_out1 = _mm256_add_pd(v_s24, v_m30);
         curr_out = out + out_strides[1];
         STR_256_D(curr_out, v_out_stride, v_out1);

         v_m31 = _mm256_mul_pd(v_CRTM_13_8, v_s26);
         // Output point 3: X(2)
         v_out2 = _mm256_add_pd(v_s45, v_m31);
         curr_out = out + out_strides[2];
         STR_256_D(curr_out, v_out_stride, v_out2);

         v_s54 = _mm256_sub_pd(v_s24, v_s52);
         v_s55 = _mm256_sub_pd(v_s45, v_s26);
         // Output point 6: X(5)
         v_out5 = _mm256_add_pd(v_s54, v_s59);
         curr_out = out + out_strides[5];
         STR_256_D(curr_out, v_out_stride, v_out5);

         // Output point 7: X(6)
         v_out6 = _mm256_add_pd(v_s55, v_m21);
         curr_out = out + out_strides[6];
         STR_256_D(curr_out, v_out_stride, v_out6);

         // Output point 8: X(7)
         v_out7 = _mm256_sub_pd(v_s54, v_s59);
         curr_out = out + out_strides[7];
         STR_256_D(curr_out, v_out_stride, v_out7);

         // Output point 9: X(8)
         v_out8 = _mm256_sub_pd(v_m21, v_s55);
         curr_out = out + out_strides[8];
         STR_256_D(curr_out, v_out_stride, v_out8);

         v_s56 = _mm256_sub_pd(v_s27, v_s43);
         v_m32 = _mm256_mul_pd(v_CRTM_13_8, v_s27);
         v_s57 = NEGATE_256_D(_mm256_add_pd(v_m32, v_s43));
         v_m33 = _mm256_mul_pd(v_CRTM_13_8, v_s53);
         v_s58 = _mm256_sub_pd(v_s25, v_m33);
         v_s60 = _mm256_add_pd(v_s25, v_s53);
         // Output point 4: X(3)
         v_out3 = _mm256_sub_pd(v_s60, v_s51);
         curr_out = out + out_strides[3];
         STR_256_D(curr_out, v_out_stride, v_out3);

         // Output point 12: X(11)
         v_out11 = _mm256_add_pd(v_s60, v_s51);
         curr_out = out + out_strides[11];
         STR_256_D(curr_out, v_out_stride, v_out11);

         // Output point 10: X(9)
         v_out9 = v_s58;
         curr_out = out + out_strides[9];
         STR_256_D(curr_out, v_out_stride, v_out9);

         // Output point 11: X(10)
         v_out10 = v_s57;
         curr_out = out + out_strides[10];
         STR_256_D(curr_out, v_out_stride, v_out10);

         // Output point 5: X(4)
         v_out4 = _mm256_add_pd(v_m20, v_s56);
         curr_out = out + out_strides[4];
         STR_256_D(curr_out, v_out_stride, v_out4);

         // Output point 13: X(12)
         v_out12 = _mm256_sub_pd(v_s56, v_m20);
         curr_out = out + out_strides[12];
         STR_256_D(curr_out, v_out_stride, v_out12);

         in += v_in_stride * NUM_SETS_REAL_256_D;
         out += v_out_stride * NUM_SETS_REAL_256_D;
     }
     // tail cases
     if (remaining_sets & NUM_SETS_REAL_128_D)
     {
         __m128d v_in0, v_in1, v_in2, v_in3, v_in4, v_in5, v_in6, v_in7, v_in8,
                 v_in9, v_in10, v_in11, v_in12;
         __m128d v_s0, v_s1, v_s2, v_s3, v_s4, v_s5, v_s6, v_s7, v_s8, v_s9,
                 v_s10, v_s11, v_s12, v_s13, v_s14, v_s15, v_s16, v_s17, v_s18,
                 v_s19, v_s20, v_s21, v_s22, v_s23, v_s24, v_s25, v_s26, v_s27,
                 v_s28, v_s29, v_s30, v_s31, v_s32, v_s33, v_s34, v_s35, v_s36,
                 v_s37, v_s38, v_s39, v_s40, v_s41, v_s42, v_s43, v_s44, v_s45,
                 v_s46, v_s47, v_s48, v_s49, v_s50, v_s51, v_s52, v_s53, v_s54,
                 v_s55, v_s56, v_s57, v_s58, v_s59, v_s60, v_s61, v_s62, v_s63,
                 v_s64;
         __m128d v_m0, v_m1, v_m2, v_m3, v_m4, v_m5, v_m6, v_m7, v_m8, v_m9,
                 v_m10, v_m11, v_m12, v_m13, v_m14, v_m15, v_m16, v_m17, v_m18,
                 v_m19, v_m20, v_m21, v_m22, v_m23, v_m24, v_m25, v_m26, v_m27,
                 v_m28, v_m29, v_m30, v_m31, v_m32, v_m33;
         __m128d v_out0, v_out1, v_out2, v_out3, v_out4, v_out5, v_out6,
                 v_out7, v_out8, v_out9, v_out10, v_out11, v_out12;
 
         curr_in = in;
         curr_out = out;
 
         __m128d v128d_CRTM_13_1 = _mm256_castpd256_pd128(v_CRTM_13_1);
         __m128d v128d_CRTM_13_2 = _mm256_castpd256_pd128(v_CRTM_13_2);
         __m128d v128d_CRTM_13_3 = _mm256_castpd256_pd128(v_CRTM_13_3);
         __m128d v128d_CRTM_13_4 = _mm256_castpd256_pd128(v_CRTM_13_4);
         __m128d v128d_CRTM_13_5 = _mm256_castpd256_pd128(v_CRTM_13_5);
         __m128d v128d_CRTM_13_6 = _mm256_castpd256_pd128(v_CRTM_13_6);
         __m128d v128d_CRTM_13_7 = _mm256_castpd256_pd128(v_CRTM_13_7);
         __m128d v128d_CRTM_13_8 = _mm256_castpd256_pd128(v_CRTM_13_8);
         __m128d v128d_R13_DGC_1 = _mm256_castpd256_pd128(v_R13_DGC_1);
         __m128d v128d_R13_DGC_2 = _mm256_castpd256_pd128(v_R13_DGC_2);
         __m128d v128d_R13_DGC_3 = _mm256_castpd256_pd128(v_R13_DGC_3);
         __m128d v128d_R13_DGC_4 = _mm256_castpd256_pd128(v_R13_DGC_4);
         __m128d v128d_R13_DGC_5 = _mm256_castpd256_pd128(v_R13_DGC_5);
         __m128d v128d_R13_DGC_6 = _mm256_castpd256_pd128(v_R13_DGC_6);
         __m128d v128d_R13_DGC_7 = _mm256_castpd256_pd128(v_R13_DGC_7);
         __m128d v128d_R13_DGC_8 = _mm256_castpd256_pd128(v_R13_DGC_8);
         __m128d v128d_R13_DGC_9 = _mm256_castpd256_pd128(v_R13_DGC_9);
         __m128d v128d_R13_DGC_10 = _mm256_castpd256_pd128(v_R13_DGC_10);
         __m128d v128d_R13_DGC_11 = _mm256_castpd256_pd128(v_R13_DGC_11);
         __m128d v128d_R13_DGC_12 = _mm256_castpd256_pd128(v_R13_DGC_12);
 
         // Input point 1: x(0)
         LDR_128_D(curr_in, v_in_stride, v_in0);
         // Input point 2: x(1)
         curr_in = in + in_strides[1];
         LDR_128_D(curr_in, v_in_stride, v_in1);
         // Input point 3: x(2)
         curr_in = in + in_strides[2];
         LDR_128_D(curr_in, v_in_stride, v_in2);
         // Input point 4: x(3)
         curr_in = in + in_strides[3];
         LDR_128_D(curr_in, v_in_stride, v_in3);
         // Input point 5: x(4)
         curr_in = in + in_strides[4];
         LDR_128_D(curr_in, v_in_stride, v_in4);
         // Input point 6: x(5)
         curr_in = in + in_strides[5];
         LDR_128_D(curr_in, v_in_stride, v_in5);
         // Input point 7: x(6)
         curr_in = in + in_strides[6];
         LDR_128_D(curr_in, v_in_stride, v_in6);
         // Input point 8: x(7)
         curr_in = in + in_strides[7];
         LDR_128_D(curr_in, v_in_stride, v_in7);
         // Input point 9: x(8)
         curr_in = in + in_strides[8];
         LDR_128_D(curr_in, v_in_stride, v_in8);
         // Input point 10: x(9)
         curr_in = in + in_strides[9];
         LDR_128_D(curr_in, v_in_stride, v_in9);
         // Input point 11: x(10)
         curr_in = in + in_strides[10];
         LDR_128_D(curr_in, v_in_stride, v_in10);
         // Input point 12: x(11)
         curr_in = in + in_strides[11];
         LDR_128_D(curr_in, v_in_stride, v_in11);
         // Input point 13: x(12)
         curr_in = in + in_strides[12];
         LDR_128_D(curr_in, v_in_stride, v_in12);
 
         v_s0 = _mm_add_pd(v_in2, v_in7);
         v_s1 = _mm_sub_pd(v_in7, v_in2);
         v_s2 = _mm_add_pd(v_in6, v_in11);
         v_s3 = _mm_sub_pd(v_in11, v_in6);
         v_s4 = _mm_add_pd(v_s0, v_s2);
         v_s5 = _mm_sub_pd(v_s0, v_s2);
         v_m0 = _mm_mul_pd(v128d_CRTM_13_6, v_s5);
         v_s6 = _mm_add_pd(v_s1, v_s3);
         v_s7 = _mm_sub_pd(v_s1, v_s3);
         v_s8 = _mm_add_pd(v_in4, v_in10);
         v_s9 = _mm_sub_pd(v_in10, v_in4);
         v_s10 = _mm_add_pd(v_in3, v_in9);
         v_s11 = _mm_sub_pd(v_in9, v_in3);
         v_s12 = _mm_add_pd(v_s8, v_s10);
         v_s13 = _mm_sub_pd(v_s8, v_s10);
         v_s14 = _mm_sub_pd(v_s9, v_s11);
         v_s15 = _mm_add_pd(v_s9, v_s11);
         v_m1 = _mm_mul_pd(v128d_CRTM_13_6, v_s15);
         v_m2 = _mm_mul_pd(v128d_CRTM_13_7, v_s13);

         v_s16 = _mm_add_pd(v_s4, v_s12);
         v_s17 = _mm_sub_pd(v_s4, v_s12);
         v_s18 = _mm_add_pd(v_in8, v_in5);
         v_s19 = _mm_sub_pd(v_in5, v_in8);
         v_m3 = _mm_mul_pd(v128d_CRTM_13_7, v_s6);
         v_s42 = _mm_add_pd(v_m3, v_s19);
         v_s20 = _mm_sub_pd(v_in1, v_in12);
         v_s21 = _mm_add_pd(v_in1, v_in12);
         v_s36 = _mm_add_pd(v_s20, v_m2);
         v_s22 = _mm_sub_pd(v_s20, v_s13);
         v_s23 = _mm_sub_pd(v_s6, v_s19);
         v_m4 = _mm_mul_pd(v128d_R13_DGC_6, v_s22);
         v_m5 = _mm_mul_pd(v128d_R13_DGC_7, v_s23);
         v_s43 = _mm_add_pd(v_m4, v_m5);
         v_m6 = _mm_mul_pd(v128d_R13_DGC_6, v_s23);
         v_m7 = _mm_mul_pd(v128d_R13_DGC_7, v_s22);
         v_s45 = _mm_sub_pd(v_m6, v_m7);
         v_s47 = _mm_add_pd(v_s21, v_s18);
         v_s48 = _mm_sub_pd(v_s21, v_s18);
         v_m8 = _mm_mul_pd(v128d_CRTM_13_7, v_s16);
         v_s32 = _mm_sub_pd(v_s47, v_m8);
         v_s28 = _mm_add_pd(v_s47, v_s16);
         v_m9 = _mm_mul_pd(v128d_CRTM_13_7, v_s17);

         v_s33 = _mm_add_pd(v_s48, v_m9);
         v_s39 = _mm_sub_pd(v_s48, v_s17);
         v_m10 = _mm_mul_pd(v_s39, v128d_R13_DGC_12);
         // Output point 1: X(0)
         v_out0 = _mm_add_pd(v_s28, v_in0);
         STR_128_D(curr_out, v_out_stride, v_out0);
         v_m11 = NEGATE_128_D(_mm_mul_pd(v_s28, v128d_R13_DGC_1));

         v_s63 = _mm_add_pd(v_m11, v_in0);
         v_s24 = _mm_add_pd(v_s63, v_m10);
         v_s25 = _mm_sub_pd(v_s63, v_m10);
         v_s61 = _mm_add_pd(v_s36, v_m0);
         v_s62 = _mm_sub_pd(v_s36, v_m0);
         v_s46 = _mm_add_pd(v_s42, v_m1);
         v_s29 = _mm_sub_pd(v_s42, v_m1);
         v_m12 = _mm_mul_pd(v128d_R13_DGC_2, v_s61);
         v_m13 = _mm_mul_pd(v128d_R13_DGC_3, v_s46);
         v_s40 = NEGATE_128_D(_mm_add_pd(v_m12, v_m13));

         v_m14 = _mm_mul_pd(v128d_R13_DGC_2, v_s46);
         v_m15 = _mm_mul_pd(v128d_R13_DGC_3, v_s61);
         v_s41 = _mm_sub_pd(v_m14, v_m15);
         v_m16 = _mm_mul_pd(v128d_R13_DGC_10, v_s29);
         v_m17 = _mm_mul_pd(v128d_R13_DGC_11, v_s62);
         v_s34 = _mm_sub_pd(v_m17, v_m16);
         v_m18 = _mm_mul_pd(v128d_R13_DGC_10, v_s62);
         v_m19 = _mm_mul_pd(v128d_R13_DGC_11, v_s29);
         v_s35 = _mm_add_pd(v_m18, v_m19);
         v_s26 = _mm_add_pd(v_s41, v_s34);
         v_s44 = _mm_sub_pd(v_s41, v_s34);
         v_m20 = _mm_mul_pd(v128d_CRTM_13_1, v_s44);
         v_s27 = _mm_add_pd(v_s40, v_s35);
         v_s64 = _mm_sub_pd(v_s40, v_s35);
         v_m21 = _mm_mul_pd(v128d_CRTM_13_1, v_s64);
         v_s30 = _mm_add_pd(v_s7, v_s14);
         v_m22 = _mm_mul_pd(v128d_R13_DGC_4, v_s33);
         v_m23 = _mm_mul_pd(v128d_CRTM_13_3, v_s30);
         v_s49 = _mm_sub_pd(v_m22, v_m23);
         v_m24 = _mm_mul_pd(v128d_CRTM_13_2, v_s30);
         v_m25 = _mm_mul_pd(v128d_R13_DGC_5, v_s33);
         v_s37 = NEGATE_128_D(_mm_add_pd(v_m24, v_m25));

         v_s31 = _mm_sub_pd(v_s7, v_s14);
         v_m26 = _mm_mul_pd(v128d_R13_DGC_8, v_s32);
         v_m27 = _mm_mul_pd(v128d_CRTM_13_5, v_s31);
         v_s50 = _mm_sub_pd(v_m26, v_m27);
         v_m28 = _mm_mul_pd(v128d_CRTM_13_4, v_s31);
         v_m29 = _mm_mul_pd(v128d_R13_DGC_9, v_s32);
         v_s38 = NEGATE_128_D(_mm_add_pd(v_m28, v_m29));

         v_s51 = _mm_add_pd(v_s37, v_s38);
         v_s59 = _mm_sub_pd(v_s37, v_s38);
         v_s52 = _mm_add_pd(v_s49, v_s50);
         v_s53 = _mm_sub_pd(v_s49, v_s50);
         v_m30 = _mm_mul_pd(v128d_CRTM_13_8, v_s52);
         // Output point 2: X(1)
         v_out1 = _mm_add_pd(v_s24, v_m30);
         curr_out = out + out_strides[1];
         STR_128_D(curr_out, v_out_stride, v_out1);

         v_m31 = _mm_mul_pd(v128d_CRTM_13_8, v_s26);
         // Output point 3: X(2)
         v_out2 = _mm_add_pd(v_s45, v_m31);
         curr_out = out + out_strides[2];
         STR_128_D(curr_out, v_out_stride, v_out2);

         v_s54 = _mm_sub_pd(v_s24, v_s52);
         v_s55 = _mm_sub_pd(v_s45, v_s26);
         // Output point 6: X(5)
         v_out5 = _mm_add_pd(v_s54, v_s59);
         curr_out = out + out_strides[5];
         STR_128_D(curr_out, v_out_stride, v_out5);

         // Output point 7: X(6)
         v_out6 = _mm_add_pd(v_s55, v_m21);
         curr_out = out + out_strides[6];
         STR_128_D(curr_out, v_out_stride, v_out6);

         // Output point 8: X(7)
         v_out7 = _mm_sub_pd(v_s54, v_s59);
         curr_out = out + out_strides[7];
         STR_128_D(curr_out, v_out_stride, v_out7);

         // Output point 9: X(8)
         v_out8 = _mm_sub_pd(v_m21, v_s55);
         curr_out = out + out_strides[8];
         STR_128_D(curr_out, v_out_stride, v_out8);

         v_s56 = _mm_sub_pd(v_s27, v_s43);
         v_m32 = _mm_mul_pd(v128d_CRTM_13_8, v_s27);
         v_s57 = NEGATE_128_D(_mm_add_pd(v_m32, v_s43));
         v_m33 = _mm_mul_pd(v128d_CRTM_13_8, v_s53);
         v_s58 = _mm_sub_pd(v_s25, v_m33);
         v_s60 = _mm_add_pd(v_s25, v_s53);
         // Output point 4: X(3)
         v_out3 = _mm_sub_pd(v_s60, v_s51);
         curr_out = out + out_strides[3];
         STR_128_D(curr_out, v_out_stride, v_out3);

         // Output point 12: X(11)
         v_out11 = _mm_add_pd(v_s60, v_s51);
         curr_out = out + out_strides[11];
         STR_128_D(curr_out, v_out_stride, v_out11);

         // Output point 10: X(9)
         v_out9 = v_s58;
         curr_out = out + out_strides[9];
         STR_128_D(curr_out, v_out_stride, v_out9);

         // Output point 11: X(10)
         v_out10 = v_s57;
         curr_out = out + out_strides[10];
         STR_128_D(curr_out, v_out_stride, v_out10);

         // Output point 5: X(4)
         v_out4 = _mm_add_pd(v_m20, v_s56);
         curr_out = out + out_strides[4];
         STR_128_D(curr_out, v_out_stride, v_out4);

         // Output point 13: X(12)
         v_out12 = _mm_sub_pd(v_s56, v_m20);
         curr_out = out + out_strides[12];
         STR_128_D(curr_out, v_out_stride, v_out12);
         
         in += v_in_stride * NUM_SETS_REAL_128_D;
         out += v_out_stride * NUM_SETS_REAL_128_D;
     }

     if (remaining_sets & 1)
     {
           FFTZ_DOUBLE in0, in1, in2, in3, in4, in5, in6, in7, in8, in9, in10,
                       in11, in12;
           FFTZ_DOUBLE s0, s1, s2, s3, s4, s5, s6, s7, s8, s9, s10, s11, s12,
                       s13, s14, s15, s16, s17, s18, s19, s20, s21, s22, s23,
                       s24, s25, s26, s27, s28, s29, s30, s31, s32, s33, s34,
                       s35, s36, s37, s38, s39, s40, s41, s42, s43, s44, s45,
                       s46, s47, s48, s49, s50, s51, s52, s53, s54, s55, s56,
                       s57, s58, s59, s60, s61, s62, s63, s64;
           FFTZ_DOUBLE m0, m1, m2, m3, m4, m5, m6, m7, m8, m9, m10, m11, m12,
                       m13, m14, m15, m16, m17, m18, m19, m20, m21, m22, m23,
                       m24, m25, m26, m27, m28, m29, m30, m31, m32, m33;

         // Input point 1: x(0)
         in0 = *in;
         // Input point 2: x(1)
         in1 = in[in_strides[1]];
         // Input point 3: x(2)
         in2 = in[in_strides[2]];
         // Input point 4: x(3)
         in3 = in[in_strides[3]];
         // Input point 5: x(4)
         in4 = in[in_strides[4]];
         // Input point 6: x(5)
         in5 = in[in_strides[5]];
         // Input point 7: x(6)
         in6 = in[in_strides[6]];
         // Input point 8: x(7)
         in7 = in[in_strides[7]];
         // Input point 9: x(8)
         in8 = in[in_strides[8]];
         // Input point 10: x(9)
         in9 = in[in_strides[9]];
         // Input point 11: x(10)
         in10 = in[in_strides[10]];
         // Input point 12: x(11)
         in11 = in[in_strides[11]];
         // Input point 13: x(12)
         in12 = in[in_strides[12]];
 
         s0 = in2 + in7;
         s1 = in7 - in2;
         s2 = in6 + in11;
         s3 = in11 - in6;
         s4 = s0 + s2;
         s5 = s0 - s2;
         m0 = CRTM_13_6 * s5;
         s6 = s1 + s3;
         s7 = s1 - s3;
         s8 = in4 + in10;
         s9 = in10 - in4;
         s10 = in3 + in9;
         s11 = in9 - in3;
         s12 = s8 + s10;
         s13 = s8 - s10;
         s14 = s9 - s11;
         s15 = s9 + s11;
         m1 = CRTM_13_6 * s15;
         m2 = CRTM_13_7 * s13;

         s16 = s4 + s12;
         s17 = s4 - s12;
         s18 = in8 + in5;
         s19 = in5 - in8;
         m3 = CRTM_13_7 * s6;
         s42 = m3 + s19;
         s20 = in1 - in12;
         s21 = in1 + in12;
         s36 = s20 + m2;
         s22 = s20 - s13;
         s23 = s6 - s19;
         m4 = R13_DGC_6 * s22;
         m5 = R13_DGC_7 * s23;
         s43 = m4 + m5;
         m6 = R13_DGC_6 * s23;
         m7 = R13_DGC_7 * s22;
         s45 = m6 - m7;
         s47 = s21 + s18;
         s48 = s21 - s18;
         m8 = CRTM_13_7 * s16;
         s32 = s47 - m8;
         s28 = s47 + s16;
         m9 = CRTM_13_7 * s17;

         s33 = s48 + m9;
         s39 = s48 - s17;
         m10 = s39 * R13_DGC_12;
         // Output point 1: X(0)
         *out = s28 + in0;
         m11 = -(s28 * R13_DGC_1);

         s63 = m11 + in0;
         s24 = s63 + m10;
         s25 = s63 - m10;
         s61 = s36 + m0;
         s62 = s36 - m0;
         s46 = s42 + m1;
         s29 = s42 - m1;
         m12 = R13_DGC_2 * s61;
         m13 = R13_DGC_3 * s46;
         s40 = -(m12 + m13);

         m14 = R13_DGC_2 * s46;
         m15 = R13_DGC_3 * s61;
         s41 = m14 - m15;
         m16 = R13_DGC_10 * s29;
         m17 = R13_DGC_11 * s62;
         s34 = m17 - m16;
         m18 = R13_DGC_10 * s62;
         m19 = R13_DGC_11 * s29;
         s35 = m18 + m19;
         s26 = s41 + s34;
         s44 = s41 - s34;
         m20 = CRTM_13_1 * s44;
         s27 = s40 + s35;
         s64 = s40 - s35;
         m21 = CRTM_13_1 * s64;
         s30 = s7 + s14;
         m22 = R13_DGC_4 * s33;
         m23 = CRTM_13_3 * s30;
         s49 = m22 - m23;
         m24 = CRTM_13_2 * s30;
         m25 = R13_DGC_5 * s33;
         s37 = -(m24 + m25);

         s31 = s7 - s14;
         m26 = R13_DGC_8 * s32;
         m27 = CRTM_13_5 * s31;
         s50 = m26 - m27;
         m28 = CRTM_13_4 * s31;
         m29 = R13_DGC_9 * s32;
         s38 = -(m28 + m29);

         s51 = s37 + s38;
         s59 = s37 - s38;
         s52 = s49 + s50;
         s53 = s49 - s50;
         m30 = CRTM_13_8 * s52;
         // Output point 2: X(1)
         out[out_strides[1]] = s24 + m30;

         m31 = CRTM_13_8 * s26;
         // Output point 3: X(2)
         out[out_strides[2]] = s45 + m31;

         s54 = s24 - s52;
         s55 = s45 - s26;
         // Output point 6: X(5)
         out[out_strides[5]] = s54 + s59;
         // Output point 7: X(6)
         out[out_strides[6]] = s55 + m21;

         // Output point 8: X(7)
         out[out_strides[7]] = s54 - s59;
         // Output point 9: X(8)
         out[out_strides[8]] = m21 - s55;

         s56 = s27 - s43;
         m32 = CRTM_13_8 * s27;
         s57 = -(m32 + s43);
         m33 = CRTM_13_8 * s53;
         s58 = s25 - m33;
         s60 = s25 + s53;
         // Output point 4: X(3)
         out[out_strides[3]] = s60 - s51;
         // Output point 12: X(11)
         out[out_strides[11]] = s60 + s51;

         // Output point 10: X(9)
         out[out_strides[9]] = s58;
         // Output point 11: X(10)
         out[out_strides[10]] = s57;

         // Output point 5: X(4)
         out[out_strides[4]] = m20 + s56;
         // Output point 13: X(12)
         out[out_strides[12]] = s56 - m20;
     }
     AOCLFFTZ_LOG(DEBUG, global_logger_mode, "Exit");
 }

 static FFTZ_VOID r2hc_rfft13avx256_fp64_bwd(FFTZ_VOID *in_real,
                                             FFTZ_VOID *in_imag,
                                             FFTZ_VOID *out_real,
                                             FFTZ_VOID *out_imag, FFTZ_INTP n,
                                             aoclfftz_strides_t *strides,
                                             FFTZ_VOID *twd, FFTZ_UINT8 flag)
 {
     AOCLFFTZ_LOG(DEBUG, global_logger_mode, "Enter");
     const FFTZ_DOUBLE CRTM_13_1 =
         1.732050807568877293527446341505872366942805254;
     const FFTZ_DOUBLE CRTM_13_2 =
         1.007074065727533254493747707736933954186697125;
     const FFTZ_DOUBLE CRTM_13_3 =
         0.531932498429674575175042127684371897596660533;
     const FFTZ_DOUBLE CRTM_13_4 =
         1.150281458948006242736771094910906776922003215;
     const FFTZ_DOUBLE CRTM_13_5 =
         0.348277202304271810011321589858529485233929352;
     const FFTZ_DOUBLE CRTM_13_6 =
         0.500000000000000000000000000000000000000000000;
     const FFTZ_DOUBLE CRTM_13_7 =
         2.000000000000000000000000000000000000000000000;
     const FFTZ_DOUBLE R13_DGC_1 =
         0.166666666666666666666666666666666666666666667;
     const FFTZ_DOUBLE R13_DGC_2 =
         0.256247671582936600958684654061725059144125175;
     const FFTZ_DOUBLE R13_DGC_3 =
         0.156891391051584611046832726756003269660212636;
     const FFTZ_DOUBLE R13_DGC_4 =
         0.774781170935234584261351932853525703557550433;
     const FFTZ_DOUBLE R13_DGC_5 =
         0.265966249214837287587521063842185948798330267;
     const FFTZ_DOUBLE R13_DGC_6 =
         0.600925212577331548853203544578415991041882762;
     const FFTZ_DOUBLE R13_DGC_7 =
         0.151805972074387731966205794490207080712856746;
     const FFTZ_DOUBLE R13_DGC_8 =
         0.227708958111581597949308691735310621069285120;
     const FFTZ_DOUBLE R13_DGC_9 =
         0.503537032863766627246873853868466977093348562;
     const FFTZ_DOUBLE R13_DGC_10 =
         0.300238635966332641462884626667381504676006424;
     const FFTZ_DOUBLE R13_DGC_11 =
         0.011599105605768290721655456654083252189827041;
     const FFTZ_DOUBLE R13_DGC_12 =
         0.516520780623489722840901288569017135705033622;

     FFTZ_DOUBLE *in = (FFTZ_DOUBLE *)in_real;
     FFTZ_DOUBLE *out = (FFTZ_DOUBLE *)out_real;
#ifdef VOLATILE_STRIDE_ARRAY
     volatile FFTZ_INTP *in_strides = strides->in_strides;
     volatile FFTZ_INTP *out_strides = strides->out_strides;
#else
     INTP *in_strides = strides->in_strides;
     FFTZ_INTP *out_strides = strides->out_strides;
#endif
     FFTZ_INTP v_in_stride = strides->v_in_stride;
     FFTZ_INTP v_out_stride = strides->v_out_stride;

     FFTZ_INTP cnt;
     FFTZ_DOUBLE *curr_in, *curr_out;
     FFTZ_INTP N = n / NUM_SETS_REAL_256_D;
     FFTZ_INTP remaining_sets = n % NUM_SETS_REAL_256_D;

     __m256d v_CRTM_13_1 = _mm256_broadcast_sd(&CRTM_13_1);
     __m256d v_CRTM_13_2 = _mm256_broadcast_sd(&CRTM_13_2);
     __m256d v_CRTM_13_3 = _mm256_broadcast_sd(&CRTM_13_3);
     __m256d v_CRTM_13_4 = _mm256_broadcast_sd(&CRTM_13_4);
     __m256d v_CRTM_13_5 = _mm256_broadcast_sd(&CRTM_13_5);
     __m256d v_CRTM_13_6 = _mm256_broadcast_sd(&CRTM_13_6);
     __m256d v_CRTM_13_7 = _mm256_broadcast_sd(&CRTM_13_7);
     __m256d v_R13_DGC_1 = _mm256_broadcast_sd(&R13_DGC_1);
     __m256d v_R13_DGC_2 = _mm256_broadcast_sd(&R13_DGC_2);
     __m256d v_R13_DGC_3 = _mm256_broadcast_sd(&R13_DGC_3);
     __m256d v_R13_DGC_4 = _mm256_broadcast_sd(&R13_DGC_4);
     __m256d v_R13_DGC_5 = _mm256_broadcast_sd(&R13_DGC_5);
     __m256d v_R13_DGC_6 = _mm256_broadcast_sd(&R13_DGC_6);
     __m256d v_R13_DGC_7 = _mm256_broadcast_sd(&R13_DGC_7);
     __m256d v_R13_DGC_8 = _mm256_broadcast_sd(&R13_DGC_8);
     __m256d v_R13_DGC_9 = _mm256_broadcast_sd(&R13_DGC_9);
     __m256d v_R13_DGC_10 = _mm256_broadcast_sd(&R13_DGC_10);
     __m256d v_R13_DGC_11 = _mm256_broadcast_sd(&R13_DGC_11);
     __m256d v_R13_DGC_12 = _mm256_broadcast_sd(&R13_DGC_12);

     for (cnt = 0; cnt < N; cnt++)
     {
         __m256d v_in0, v_in1, v_in2, v_in3, v_in4, v_in5, v_in6, v_in7, v_in8,
                 v_in9, v_in10, v_in11, v_in12;
         __m256d v_s0, v_s1, v_s2, v_s3, v_s4, v_s5, v_s6, v_s7, v_s8, v_s9,
                 v_s10, v_s11, v_s12, v_s13, v_s14, v_s15, v_s16, v_s17, v_s18,
                 v_s19, v_s20, v_s21, v_s22, v_s23, v_s24, v_s25, v_s26, v_s27,
                 v_s28, v_s29, v_s30, v_s31, v_s32, v_s33, v_s34, v_s35, v_s36,
                 v_s37, v_s38, v_s39, v_s40, v_s41, v_s42, v_s43, v_s44, v_s45,
                 v_s46, v_s47, v_s48, v_s49, v_s50, v_s51, v_s52, v_s53, v_s54,
                 v_s55, v_s56, v_s57, v_s58, v_s59, v_s60, v_s61, v_s62;
         __m256d v_m0, v_m1, v_m2, v_m3, v_m4, v_m5, v_m6, v_m7, v_m8, v_m9,
                 v_m10, v_m11, v_m12, v_m13, v_m14, v_m15, v_m16, v_m17, v_m18,
                 v_m19, v_m20, v_m21, v_m22, v_m23, v_m24, v_m25, v_m26, v_m27,
                 v_m28, v_m29, v_m30, v_m31, v_m32, v_m33, v_m34;
         __m256d v_out0, v_out1, v_out2, v_out3, v_out4, v_out5, v_out6, v_out7,
                 v_out8, v_out9, v_out10, v_out11, v_out12;

         curr_in = in;
         curr_out = out;

         // Input point 1: X(0)
         LDR_256_D(curr_in, v_in_stride, v_in0);
         // Input point 2: X(1) & Input point 3: X(2)
         curr_in = in + in_strides[1];
         LDRI_2x256_D(curr_in, v_in_stride, v_in1, v_in2);
         // Input point 4: X(3) & Input point 5: X(4)
         curr_in = in + in_strides[3];
         LDRI_2x256_D(curr_in, v_in_stride, v_in3, v_in4);
         // Input point 6: X(5) & Input point 7: X(6)
         curr_in = in + in_strides[5];
         LDRI_2x256_D(curr_in, v_in_stride, v_in5, v_in6);
         // Input point 8: X(7) & Input point 9: X(8)
         curr_in = in + in_strides[7];
         LDRI_2x256_D(curr_in, v_in_stride, v_in7, v_in8);
         // Input point 10: X(9) & Input point 11: X(10)
         curr_in = in + in_strides[9];
         LDRI_2x256_D(curr_in, v_in_stride, v_in9, v_in10);
         // Input point 12: X(11) & Input point 13: X(12)
         curr_in = in + in_strides[11];
         LDRI_2x256_D(curr_in, v_in_stride, v_in11, v_in12);
 
         v_s0 = _mm256_sub_pd(v_in6, v_in8);
         v_m0 = _mm256_mul_pd(v_CRTM_13_7, v_in2);
         v_s1 = _mm256_sub_pd(v_m0, v_s0);
         v_s59 = _mm256_add_pd(v_in6, v_in8);
         v_m1 = _mm256_mul_pd(v_CRTM_13_1, v_s59);
         v_s2 = _mm256_add_pd(v_in12, v_in4);
         v_s60 = _mm256_sub_pd(v_in12, v_in4);
         v_m2 = _mm256_mul_pd(v_CRTM_13_1, v_s60);
         v_m3 = _mm256_mul_pd(v_CRTM_13_7, v_in10);
         v_s3 = _mm256_sub_pd(v_s2, v_m3);
         v_s4 = _mm256_add_pd(v_s1, v_m2);
         v_s5 = _mm256_sub_pd(v_s3, v_m1);
         v_m4 = _mm256_mul_pd(v_R13_DGC_11, v_s4);
         v_m5 = _mm256_mul_pd(v_R13_DGC_10, v_s5);
         v_s6 = _mm256_add_pd(v_m4, v_m5);
         v_m6 = _mm256_mul_pd(v_R13_DGC_10, v_s4);
         v_m7 = _mm256_mul_pd(v_R13_DGC_11, v_s5);
         v_s7 = _mm256_sub_pd(v_m6, v_m7);
         v_s8 = _mm256_add_pd(v_in2, v_s0);
         v_s9 = _mm256_add_pd(v_s2, v_in10);
         v_m8 = _mm256_mul_pd(v_CRTM_13_4, v_s8);
         v_m9 = _mm256_mul_pd(v_CRTM_13_5, v_s9);
         
         v_s10 = _mm256_sub_pd(v_m8, v_m9);
         v_m10 = _mm256_mul_pd(v_CRTM_13_5, v_s8);
         v_m11 = _mm256_mul_pd(v_CRTM_13_4, v_s9);
         v_s11 = _mm256_add_pd(v_m10, v_m11);
         v_s12 = _mm256_sub_pd(v_s1, v_m2);
         v_s13 = _mm256_add_pd(v_m1, v_s3);
         v_m12 = _mm256_mul_pd(v_R13_DGC_3, v_s12);
         v_m13 = _mm256_mul_pd(v_R13_DGC_2, v_s13);
         v_s14 = _mm256_add_pd(v_m12, v_m13);
         v_m14 = _mm256_mul_pd(v_R13_DGC_3, v_s13);
         v_m15 = _mm256_mul_pd(v_R13_DGC_2, v_s12);
         v_s15 = _mm256_sub_pd(v_m14, v_m15);
         v_s16 = _mm256_add_pd(v_in3, v_in11);
         v_s17 = _mm256_add_pd(v_in9, v_s16);
         v_m16 = _mm256_mul_pd(v_CRTM_13_6, v_s16);
         v_s18 = _mm256_sub_pd(v_in9, v_m16);
         v_s19 = _mm256_sub_pd(v_in3, v_in11);
         v_s20 = _mm256_add_pd(v_in5, v_in7);
         v_s21 = _mm256_add_pd(v_in1, v_s20);
         v_m17 = _mm256_mul_pd(v_CRTM_13_6, v_s20);
         v_s22 = _mm256_sub_pd(v_in1, v_m17);
         v_s23 = _mm256_sub_pd(v_in5, v_in7);
         v_s24 = _mm256_sub_pd(v_s21, v_s17);
         v_m18 = _mm256_mul_pd(v_R13_DGC_6, v_s24);
         v_s25 = _mm256_add_pd(v_s21, v_s17);
         v_m33 = _mm256_mul_pd(v_CRTM_13_7, v_s25);
         // Output point 1: x(0)
         v_out0 = _mm256_add_pd(v_m33, v_in0);
         STR_256_D(curr_out, v_out_stride, v_out0);

         v_m34 = _mm256_mul_pd(v_R13_DGC_1, v_s25);
         v_s26 = _mm256_sub_pd(v_in0, v_m34);
         v_s27 = _mm256_add_pd(v_s23, v_s19);
         v_s28 = _mm256_add_pd(v_s22, v_s18);
         v_m19 = _mm256_mul_pd(v_R13_DGC_9, v_s27);
         v_m20 = _mm256_mul_pd(v_R13_DGC_7, v_s28);
         v_s29 = _mm256_add_pd(v_m19, v_m20);
         v_s30 = _mm256_sub_pd(v_s22, v_s18);
         v_s31 = _mm256_sub_pd(v_s23, v_s19);
         v_m21 = _mm256_mul_pd(v_R13_DGC_12, v_s30);
         v_m22 = _mm256_mul_pd(v_R13_DGC_5, v_s31);
         v_s32 = _mm256_sub_pd(v_m21, v_m22);
         v_s61 = _mm256_add_pd(v_s6, v_s14);
         v_m23 = _mm256_mul_pd(v_CRTM_13_1, v_s61);
         v_s62 = _mm256_sub_pd(v_s7, v_s15);
         v_m24 = _mm256_mul_pd(v_CRTM_13_1, v_s62);
         v_s33 = _mm256_add_pd(v_s7, v_s15);
         v_s34 = _mm256_sub_pd(v_s10, v_s33);
         v_m25 = _mm256_mul_pd(v_CRTM_13_7, v_s33);
         v_s35 = _mm256_add_pd(v_m25, v_s10);
         v_s36 = _mm256_sub_pd(v_s6, v_s14);
         v_m26 = _mm256_mul_pd(v_CRTM_13_7, v_s36);
         v_s37 = _mm256_sub_pd(v_m26, v_s11);
         v_s38 = _mm256_add_pd(v_s36, v_s11);
         v_m27 = _mm256_mul_pd(v_R13_DGC_4, v_s31);
         v_m28 = _mm256_mul_pd(v_CRTM_13_3, v_s30);
         v_s39 = _mm256_add_pd(v_m27, v_m28);
         v_m29 = _mm256_mul_pd(v_R13_DGC_8, v_s27);
         v_m30 = _mm256_mul_pd(v_CRTM_13_2, v_s28);
         v_s40 = _mm256_sub_pd(v_m29, v_m30);
         v_s41 = _mm256_sub_pd(v_s39, v_s40);
         v_s42 = _mm256_add_pd(v_s39, v_s40);
         v_s43 = _mm256_sub_pd(v_s26, v_s29);
         v_s44 = _mm256_sub_pd(v_m18, v_s32);
         v_s45 = _mm256_sub_pd(v_s43, v_s44);
         v_s46 = _mm256_add_pd(v_s44, v_s43);
         v_m31 = _mm256_mul_pd(v_CRTM_13_7, v_s29);
         v_s47 = _mm256_add_pd(v_m31, v_s26);
         v_m32 = _mm256_mul_pd(v_CRTM_13_7, v_s32);
         v_s48 = _mm256_add_pd(v_m32, v_m18);
         v_s49 = _mm256_sub_pd(v_s47, v_s48);
         // Output point 9: x(8)
         v_out8 = _mm256_add_pd(v_s49, v_s35);
         curr_out = out + out_strides[8];
         STR_256_D(curr_out, v_out_stride, v_out8);

         // Output point 6: x(5)
         v_out5 = _mm256_sub_pd(v_s49, v_s35);
         curr_out = out + out_strides[5];
         STR_256_D(curr_out, v_out_stride, v_out5);

         v_s50 = _mm256_add_pd(v_s48, v_s47);
         // Output point 13: x(12)
         v_out12 = _mm256_sub_pd(v_s50, v_s37);
         curr_out = out + out_strides[12];
         STR_256_D(curr_out, v_out_stride, v_out12);

         // Output point 2: x(1)
         v_out1 = _mm256_add_pd(v_s50, v_s37);
         curr_out = out + out_strides[1];
         STR_256_D(curr_out, v_out_stride, v_out1);

         v_s51 = _mm256_sub_pd(v_s45, v_m23);
         v_s52 = _mm256_sub_pd(v_s41, v_s34);
         // Output point 3: x(2)
         v_out2 = _mm256_add_pd(v_s51, v_s52);
         curr_out = out + out_strides[2];
         STR_256_D(curr_out, v_out_stride, v_out2);

         // Output point 8: x(7)
         v_out7 = _mm256_sub_pd(v_s51, v_s52);
         curr_out = out + out_strides[7];
         STR_256_D(curr_out, v_out_stride, v_out7);

         v_s53 = _mm256_sub_pd(v_s46, v_s38);
         v_s54 = _mm256_add_pd(v_s42, v_m24);
         // Output point 4: x(3)
         v_out3 = _mm256_sub_pd(v_s53, v_s54);
         curr_out = out + out_strides[3];
         STR_256_D(curr_out, v_out_stride, v_out3);

         // Output point 10: x(9)
         v_out9 = _mm256_add_pd(v_s54, v_s53);
         curr_out = out + out_strides[9];
         STR_256_D(curr_out, v_out_stride, v_out9);

         v_s55 = _mm256_sub_pd(v_s42, v_m24);
         v_s56 = _mm256_add_pd(v_s46, v_s38);
         // Output point 5: x(4)
         v_out4 = _mm256_add_pd(v_s55, v_s56);
         curr_out = out + out_strides[4];
         STR_256_D(curr_out, v_out_stride, v_out4);

         // Output point 11: x(10)
         v_out10 = _mm256_sub_pd(v_s56, v_s55);
         curr_out = out + out_strides[10];
         STR_256_D(curr_out, v_out_stride, v_out10);

         v_s57 = _mm256_add_pd(v_s45, v_m23);
         v_s58 = _mm256_add_pd(v_s41, v_s34);
         // Output point 7: x(6)
         v_out6 = _mm256_sub_pd(v_s57, v_s58);
         curr_out = out + out_strides[6];
         STR_256_D(curr_out, v_out_stride, v_out6);

         // Output point 12: x(11)
         v_out11 = _mm256_add_pd(v_s57, v_s58);
         curr_out = out + out_strides[11];
         STR_256_D(curr_out, v_out_stride, v_out11);

         in += v_in_stride * NUM_SETS_REAL_256_D;
         out += v_out_stride * NUM_SETS_REAL_256_D;
     }
     // tail cases
     if (remaining_sets & NUM_SETS_REAL_128_D)
     {
         __m128d v_in0, v_in1, v_in2, v_in3, v_in4, v_in5, v_in6, v_in7, v_in8,
                 v_in9, v_in10, v_in11, v_in12;
         __m128d v_s0, v_s1, v_s2, v_s3, v_s4, v_s5, v_s6, v_s7, v_s8, v_s9,
                 v_s10, v_s11, v_s12, v_s13, v_s14, v_s15, v_s16, v_s17, v_s18,
                 v_s19, v_s20, v_s21, v_s22, v_s23, v_s24, v_s25, v_s26, v_s27,
                 v_s28, v_s29, v_s30, v_s31, v_s32, v_s33, v_s34, v_s35, v_s36,
                 v_s37, v_s38, v_s39, v_s40, v_s41, v_s42, v_s43, v_s44, v_s45,
                 v_s46, v_s47, v_s48, v_s49, v_s50, v_s51, v_s52, v_s53, v_s54,
                 v_s55, v_s56, v_s57, v_s58, v_s59, v_s60, v_s61, v_s62;
         __m128d v_m0, v_m1, v_m2, v_m3, v_m4, v_m5, v_m6, v_m7, v_m8, v_m9,
                 v_m10, v_m11, v_m12, v_m13, v_m14, v_m15, v_m16, v_m17, v_m18,
                 v_m19, v_m20, v_m21, v_m22, v_m23, v_m24, v_m25, v_m26, v_m27,
                 v_m28, v_m29, v_m30, v_m31, v_m32, v_m33, v_m34;
         __m128d v_out0, v_out1, v_out2, v_out3, v_out4, v_out5, v_out6, v_out7,
                 v_out8, v_out9, v_out10, v_out11, v_out12;
 
         curr_in = in;
         curr_out = out;
 
         __m128d v128d_CRTM_13_1 = _mm256_castpd256_pd128(v_CRTM_13_1);
         __m128d v128d_CRTM_13_2 = _mm256_castpd256_pd128(v_CRTM_13_2);
         __m128d v128d_CRTM_13_3 = _mm256_castpd256_pd128(v_CRTM_13_3);
         __m128d v128d_CRTM_13_4 = _mm256_castpd256_pd128(v_CRTM_13_4);
         __m128d v128d_CRTM_13_5 = _mm256_castpd256_pd128(v_CRTM_13_5);
         __m128d v128d_CRTM_13_6 = _mm256_castpd256_pd128(v_CRTM_13_6);
         __m128d v128d_CRTM_13_7 = _mm256_castpd256_pd128(v_CRTM_13_7);
         __m128d v128d_R13_DGC_1 = _mm256_castpd256_pd128(v_R13_DGC_1);
         __m128d v128d_R13_DGC_2 = _mm256_castpd256_pd128(v_R13_DGC_2);
         __m128d v128d_R13_DGC_3 = _mm256_castpd256_pd128(v_R13_DGC_3);
         __m128d v128d_R13_DGC_4 = _mm256_castpd256_pd128(v_R13_DGC_4);
         __m128d v128d_R13_DGC_5 = _mm256_castpd256_pd128(v_R13_DGC_5);
         __m128d v128d_R13_DGC_6 = _mm256_castpd256_pd128(v_R13_DGC_6);
         __m128d v128d_R13_DGC_7 = _mm256_castpd256_pd128(v_R13_DGC_7);
         __m128d v128d_R13_DGC_8 = _mm256_castpd256_pd128(v_R13_DGC_8);
         __m128d v128d_R13_DGC_9 = _mm256_castpd256_pd128(v_R13_DGC_9);
         __m128d v128d_R13_DGC_10 = _mm256_castpd256_pd128(v_R13_DGC_10);
         __m128d v128d_R13_DGC_11 = _mm256_castpd256_pd128(v_R13_DGC_11);
         __m128d v128d_R13_DGC_12 = _mm256_castpd256_pd128(v_R13_DGC_12);
 
         // Input point 1: X(0)
         LDR_128_D(curr_in, v_in_stride, v_in0);
         // Input point 2: X(1) & Input point 3: X(2)
         curr_in = in + in_strides[1];
         LDRI_2x128_D(curr_in, v_in_stride, v_in1, v_in2);
         // Input point 4: X(3) & Input point 5: X(4)
         curr_in = in + in_strides[3];
         LDRI_2x128_D(curr_in, v_in_stride, v_in3, v_in4);
         // Input point 6: X(5) & Input point 7: X(6)
         curr_in = in + in_strides[5];
         LDRI_2x128_D(curr_in, v_in_stride, v_in5, v_in6);
         // Input point 8: X(7) & Input point 9: X(8)
         curr_in = in + in_strides[7];
         LDRI_2x128_D(curr_in, v_in_stride, v_in7, v_in8);
         // Input point 10: X(9) & Input point 11: X(10)
         curr_in = in + in_strides[9];
         LDRI_2x128_D(curr_in, v_in_stride, v_in9, v_in10);
         // Input point 12: X(11) & Input point 13: X(12)
         curr_in = in + in_strides[11];
         LDRI_2x128_D(curr_in, v_in_stride, v_in11, v_in12);
 
         v_s0 = _mm_sub_pd(v_in6, v_in8);
         v_m0 = _mm_mul_pd(v128d_CRTM_13_7, v_in2);
         v_s1 = _mm_sub_pd(v_m0, v_s0);
         v_s59 = _mm_add_pd(v_in6, v_in8);
         v_m1 = _mm_mul_pd(v128d_CRTM_13_1, v_s59);
         v_s2 = _mm_add_pd(v_in12, v_in4);
         v_s60 = _mm_sub_pd(v_in12, v_in4);
         v_m2 = _mm_mul_pd(v128d_CRTM_13_1, v_s60);
         v_m3 = _mm_mul_pd(v128d_CRTM_13_7, v_in10);
         v_s3 = _mm_sub_pd(v_s2, v_m3);
         v_s4 = _mm_add_pd(v_s1, v_m2);
         v_s5 = _mm_sub_pd(v_s3, v_m1);
         v_m4 = _mm_mul_pd(v128d_R13_DGC_11, v_s4);
         v_m5 = _mm_mul_pd(v128d_R13_DGC_10, v_s5);
         v_s6 = _mm_add_pd(v_m4, v_m5);
         v_m6 = _mm_mul_pd(v128d_R13_DGC_10, v_s4);
         v_m7 = _mm_mul_pd(v128d_R13_DGC_11, v_s5);
         v_s7 = _mm_sub_pd(v_m6, v_m7);
         v_s8 = _mm_add_pd(v_in2, v_s0);
         v_s9 = _mm_add_pd(v_s2, v_in10);
         v_m8 = _mm_mul_pd(v128d_CRTM_13_4, v_s8);
         v_m9 = _mm_mul_pd(v128d_CRTM_13_5, v_s9);

         v_s10 = _mm_sub_pd(v_m8, v_m9);
         v_m10 = _mm_mul_pd(v128d_CRTM_13_5, v_s8);
         v_m11 = _mm_mul_pd(v128d_CRTM_13_4, v_s9);
         v_s11 = _mm_add_pd(v_m10, v_m11);
         v_s12 = _mm_sub_pd(v_s1, v_m2);
         v_s13 = _mm_add_pd(v_m1, v_s3);
         v_m12 = _mm_mul_pd(v128d_R13_DGC_3, v_s12);
         v_m13 = _mm_mul_pd(v128d_R13_DGC_2, v_s13);
         v_s14 = _mm_add_pd(v_m12, v_m13);
         v_m14 = _mm_mul_pd(v128d_R13_DGC_3, v_s13);
         v_m15 = _mm_mul_pd(v128d_R13_DGC_2, v_s12);
         v_s15 = _mm_sub_pd(v_m14, v_m15);
         v_s16 = _mm_add_pd(v_in3, v_in11);
         v_s17 = _mm_add_pd(v_in9, v_s16);
         v_m16 = _mm_mul_pd(v128d_CRTM_13_6, v_s16);
         v_s18 = _mm_sub_pd(v_in9, v_m16);
         v_s19 = _mm_sub_pd(v_in3, v_in11);
         v_s20 = _mm_add_pd(v_in5, v_in7);
         v_s21 = _mm_add_pd(v_in1, v_s20);
         v_m17 = _mm_mul_pd(v128d_CRTM_13_6, v_s20);
         v_s22 = _mm_sub_pd(v_in1, v_m17);
         v_s23 = _mm_sub_pd(v_in5, v_in7);
         v_s24 = _mm_sub_pd(v_s21, v_s17);
         v_m18 = _mm_mul_pd(v128d_R13_DGC_6, v_s24);
         v_s25 = _mm_add_pd(v_s21, v_s17);
         v_m33 = _mm_mul_pd(v128d_CRTM_13_7, v_s25);
         // Output point 1: x(0)
         v_out0 = _mm_add_pd(v_m33, v_in0);
         STR_128_D(curr_out, v_out_stride, v_out0);

         v_m34 = _mm_mul_pd(v128d_R13_DGC_1, v_s25);
         v_s26 = _mm_sub_pd(v_in0, v_m34);
         v_s27 = _mm_add_pd(v_s23, v_s19);
         v_s28 = _mm_add_pd(v_s22, v_s18);
         v_m19 = _mm_mul_pd(v128d_R13_DGC_9, v_s27);
         v_m20 = _mm_mul_pd(v128d_R13_DGC_7, v_s28);
         v_s29 = _mm_add_pd(v_m19, v_m20);
         v_s30 = _mm_sub_pd(v_s22, v_s18);
         v_s31 = _mm_sub_pd(v_s23, v_s19);
         v_m21 = _mm_mul_pd(v128d_R13_DGC_12, v_s30);
         v_m22 = _mm_mul_pd(v128d_R13_DGC_5, v_s31);
         v_s32 = _mm_sub_pd(v_m21, v_m22);
         v_s61 = _mm_add_pd(v_s6, v_s14);
         v_m23 = _mm_mul_pd(v128d_CRTM_13_1, v_s61);
         v_s62 = _mm_sub_pd(v_s7, v_s15);
         v_m24 = _mm_mul_pd(v128d_CRTM_13_1, v_s62);
         v_s33 = _mm_add_pd(v_s7, v_s15);
         v_s34 = _mm_sub_pd(v_s10, v_s33);
         v_m25 = _mm_mul_pd(v128d_CRTM_13_7, v_s33);
         v_s35 = _mm_add_pd(v_m25, v_s10);
         v_s36 = _mm_sub_pd(v_s6, v_s14);
         v_m26 = _mm_mul_pd(v128d_CRTM_13_7, v_s36);
         v_s37 = _mm_sub_pd(v_m26, v_s11);
         v_s38 = _mm_add_pd(v_s36, v_s11);
         v_m27 = _mm_mul_pd(v128d_R13_DGC_4, v_s31);
         v_m28 = _mm_mul_pd(v128d_CRTM_13_3, v_s30);

         v_s39 = _mm_add_pd(v_m27, v_m28);
         v_m29 = _mm_mul_pd(v128d_R13_DGC_8, v_s27);
         v_m30 = _mm_mul_pd(v128d_CRTM_13_2, v_s28);
         v_s40 = _mm_sub_pd(v_m29, v_m30);
         v_s41 = _mm_sub_pd(v_s39, v_s40);
         v_s42 = _mm_add_pd(v_s39, v_s40);
         v_s43 = _mm_sub_pd(v_s26, v_s29);
         v_s44 = _mm_sub_pd(v_m18, v_s32);
         v_s45 = _mm_sub_pd(v_s43, v_s44);
         v_s46 = _mm_add_pd(v_s44, v_s43);
         v_m31 = _mm_mul_pd(v128d_CRTM_13_7, v_s29);
         v_s47 = _mm_add_pd(v_m31, v_s26);
         v_m32 = _mm_mul_pd(v128d_CRTM_13_7, v_s32);
         v_s48 = _mm_add_pd(v_m32, v_m18);
         v_s49 = _mm_sub_pd(v_s47, v_s48);
         // Output point 9: x(8)
         v_out8 = _mm_add_pd(v_s49, v_s35);
         curr_out = out + out_strides[8];
         STR_128_D(curr_out, v_out_stride, v_out8);

         // Output point 6: x(5)
         v_out5 = _mm_sub_pd(v_s49, v_s35);
         curr_out = out + out_strides[5];
         STR_128_D(curr_out, v_out_stride, v_out5);

         v_s50 = _mm_add_pd(v_s48, v_s47);
         // Output point 13: x(12)
         v_out12 = _mm_sub_pd(v_s50, v_s37);
         curr_out = out + out_strides[12];
         STR_128_D(curr_out, v_out_stride, v_out12);

         // Output point 2: x(1)
         v_out1 = _mm_add_pd(v_s50, v_s37);
         curr_out = out + out_strides[1];
         STR_128_D(curr_out, v_out_stride, v_out1);

         v_s51 = _mm_sub_pd(v_s45, v_m23);
         v_s52 = _mm_sub_pd(v_s41, v_s34);
         // Output point 3: x(2)
         v_out2 = _mm_add_pd(v_s51, v_s52);
         curr_out = out + out_strides[2];
         STR_128_D(curr_out, v_out_stride, v_out2);

         // Output point 8: x(7)
         v_out7 = _mm_sub_pd(v_s51, v_s52);
         curr_out = out + out_strides[7];
         STR_128_D(curr_out, v_out_stride, v_out7);

         v_s53 = _mm_sub_pd(v_s46, v_s38);
         v_s54 = _mm_add_pd(v_s42, v_m24);
         // Output point 4: x(3)
         v_out3 = _mm_sub_pd(v_s53, v_s54);
         curr_out = out + out_strides[3];
         STR_128_D(curr_out, v_out_stride, v_out3);

         // Output point 10: x(9)
         v_out9 = _mm_add_pd(v_s54, v_s53);
         curr_out = out + out_strides[9];
         STR_128_D(curr_out, v_out_stride, v_out9);

         v_s55 = _mm_sub_pd(v_s42, v_m24);
         v_s56 = _mm_add_pd(v_s46, v_s38);
         // Output point 5: x(4)
         v_out4 = _mm_add_pd(v_s55, v_s56);
         curr_out = out + out_strides[4];
         STR_128_D(curr_out, v_out_stride, v_out4);

         // Output point 11: x(10)
         v_out10 = _mm_sub_pd(v_s56, v_s55);
         curr_out = out + out_strides[10];
         STR_128_D(curr_out, v_out_stride, v_out10);
         
         v_s57 = _mm_add_pd(v_s45, v_m23);
         v_s58 = _mm_add_pd(v_s41, v_s34);
         // Output point 7: x(6)
         v_out6 = _mm_sub_pd(v_s57, v_s58);
         curr_out = out + out_strides[6];
         STR_128_D(curr_out, v_out_stride, v_out6);

         // Output point 12: x(11)
         v_out11 = _mm_add_pd(v_s57, v_s58);
         curr_out = out + out_strides[11];
         STR_128_D(curr_out, v_out_stride, v_out11);

         in += v_in_stride * NUM_SETS_REAL_128_D;
         out += v_out_stride * NUM_SETS_REAL_128_D;
     }

     if (remaining_sets & 1)
     {
         FFTZ_DOUBLE in0, in1, in2, in3, in4, in5, in6, in7, in8, in9, in10,
                     in11, in12;
         FFTZ_DOUBLE s0, s1, s2, s3, s4, s5, s6, s7, s8, s9, s10, s11, s12,
                     s13, s14, s15, s16, s17, s18, s19, s20, s21, s22, s23,
                     s24, s25, s26, s27, s28, s29, s30, s31, s32, s33, s34,
                     s35, s36, s37, s38, s39, s40, s41, s42, s43, s44, s45,
                     s46, s47, s48, s49, s50, s51, s52, s53, s54, s55, s56,
                     s57, s58, s59, s60, s61, s62;
         FFTZ_DOUBLE m0, m1, m2, m3, m4, m5, m6, m7, m8, m9, m10, m11, m12,
                     m13, m14, m15, m16, m17, m18, m19, m20, m21, m22, m23,
                     m24, m25, m26, m27, m28, m29, m30, m31, m32, m33, m34;

         // Input point 1: X(0)
         in0 = *in;
         // Input point 2: X(1)
         in1 = in[in_strides[1]];
         // Input point 3: X(2)
         in2 = in[in_strides[2]];
         // Input point 4: X(3)
         in3 = in[in_strides[3]];
         // Input point 5: X(4)
         in4 = in[in_strides[4]];
         // Input point 6: X(5)
         in5 = in[in_strides[5]];
         // Input point 7: X(6)
         in6 = in[in_strides[6]];
         // Input point 8: X(7)
         in7 = in[in_strides[7]];
         // Input point 9: X(8)
         in8 = in[in_strides[8]];
         // Input point 10: X(9)
         in9 = in[in_strides[9]];
         // Input point 11: X(10)
         in10 = in[in_strides[10]];
         // Input point 12: X(11)
         in11 = in[in_strides[11]];
         // Input point 13: X(12)
         in12 = in[in_strides[12]];
 
         s0 = in6 - in8;
         m0 = CRTM_13_7 * in2;
         s1 = m0 - s0;
         s59 = in6 + in8;
         m1 = CRTM_13_1 * s59;
         s2 = in12 + in4;
         s60 = in12 - in4;
         m2 = CRTM_13_1 * s60;
         m3 = CRTM_13_7 * in10;
         s3 = s2 - m3;
         s4 = s1 + m2;
         s5 = s3 - m1;
         m4 = R13_DGC_11 * s4;
         m5 = R13_DGC_10 * s5;
         s6 = m4 + m5;
         m6 = R13_DGC_10 * s4;
         m7 = R13_DGC_11 * s5;
         s7 = m6 - m7;
         s8 = in2 + s0;
         s9 = s2 + in10;
         m8 = CRTM_13_4 * s8;
         m9 = CRTM_13_5 * s9;

         s10 = m8 - m9;
         m10 = CRTM_13_5 * s8;
         m11 = CRTM_13_4 * s9;
         s11 = m10 + m11;
         s12 = s1 - m2;
         s13 = m1 + s3;
         m12 = R13_DGC_3 * s12;
         m13 = R13_DGC_2 * s13;
         s14 = m12 + m13;
         m14 = R13_DGC_3 * s13;
         m15 = R13_DGC_2 * s12;
         s15 = m14 - m15;
         s16 = in3 + in11;
         s17 = in9 + s16;
         m16 = CRTM_13_6 * s16;
         s18 = in9 - m16;
         s19 = in3 - in11;
         s20 = in5 + in7;
         s21 = in1 + s20;
         m17 = CRTM_13_6 * s20;
         s22 = in1 - m17;
         s23 = in5 - in7;
         s24 = s21 - s17;
         m18 = R13_DGC_6 * s24;
         s25 = s21 + s17;
         m33 = CRTM_13_7 * s25;
         // Output point 1: x(0)
         out[out_strides[0]] = m33 + in0;

         m34 = R13_DGC_1 * s25;
         s26 = in0 - m34;
         s27 = s23 + s19;
         s28 = s22 + s18;
         m19 = R13_DGC_9 * s27;
         m20 = R13_DGC_7 * s28;
         s29 = m19 + m20;
         s30 = s22 - s18;
         s31 = s23 - s19;
         m21 = R13_DGC_12 * s30;
         m22 = R13_DGC_5 * s31;
         s32 = m21 - m22;
         s61 = s6 + s14;
         m23 = CRTM_13_1 * s61;
         s62 = s7 - s15;
         m24 = CRTM_13_1 * s62;
         s33 = s7 + s15;
         s34 = s10 - s33;
         m25 = CRTM_13_7 * s33;
         s35 = m25 + s10;
         s36 = s6 - s14;
         m26 = CRTM_13_7 * s36;
         s37 = m26 - s11;
         s38 = s36 + s11;
         m27 = R13_DGC_4 * s31;
         m28 = CRTM_13_3 * s30;
         s39 = m27 + m28;
         m29 = R13_DGC_8 * s27;
         m30 = CRTM_13_2 * s28;
         s40 = m29 - m30;
         s41 = s39 - s40;
         s42 = s39 + s40;
         s43 = s26 - s29;
         s44 = m18 - s32;
         s45 = s43 - s44;
         s46 = s44 + s43;
         m31 = CRTM_13_7 * s29;
         s47 = m31 + s26;
         m32 = CRTM_13_7 * s32;
         s48 = m32 + m18;
         s49 = s47 - s48;
         // Output point 9: x(8)
         out[out_strides[8]] = s49 + s35;
         // Output point 6: x(5)
         out[out_strides[5]] = s49 - s35;

         s50 = s48 + s47;
         // Output point 13: x(12)
         out[out_strides[12]] = s50 - s37;
         // Output point 2: x(1)
         out[out_strides[1]] = s50 + s37;

         s51 = s45 - m23;
         s52 = s41 - s34;
         // Output point 3: x(2)
         out[out_strides[2]] = s51 + s52;
         // Output point 8: x(7)
         out[out_strides[7]] = s51 - s52;

         s53 = s46 - s38;
         s54 = s42 + m24;
         // Output point 4: x(3)
         out[out_strides[3]] = s53 - s54;
         // Output point 10: x(9)
         out[out_strides[9]] = s54 + s53;

         s55 = s42 - m24;
         s56 = s46 + s38;
         // Output point 5: x(4)
         out[out_strides[4]] = s55 + s56;
         // Output point 11: x(10)
         out[out_strides[10]] = s56 - s55;

         s57 = s45 + m23;
         s58 = s41 + s34;
         // Output point 7: x(6)
         out[out_strides[6]] = s57 - s58;
         // Output point 12: x(11)
         out[out_strides[11]] = s57 + s58;
     }
     AOCLFFTZ_LOG(DEBUG, global_logger_mode, "Exit");
 }

 kfft_ register_kernel_r2hc_rfft13avx256(FFTZ_UINT8 precision,
                                         FFTZ_UINT8 direction)
 {
     if (direction == FORWARD_FFT_DIR)
     {
         if (precision == DT_FLOAT)
         {
             return r2hc_rfft13avx256_fp32_fwd;
         }
         else if (precision == DT_DOUBLE)
         {
             return r2hc_rfft13avx256_fp64_fwd;
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
             return r2hc_rfft13avx256_fp32_bwd;
         }
         else if (precision == DT_DOUBLE)
         {
             return r2hc_rfft13avx256_fp64_bwd;
         }
         else
         {
             return NULL;
         }
     }
 }
