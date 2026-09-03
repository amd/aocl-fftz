// Copyright Advanced Micro Devices, Inc.
// SPDX-License-Identifier: BSD-3-Clause

/** @file rfft10avx512.c
 *
 *  @brief Radix-10 r2hc Real-FFT kernel with AVX-512 operations using x86 SIMD
 *  intrinsics
 *
 *  This file contains the DIT radix-10 real-to-halfcomplex implementations
 *  using AVX512 SIMD operations for single-precision and double-precision
 *  inputs.
 *
 *  @author Jeevanantham N
 */

#include "core/kernels/kernel.h"
#include "core/kernels/simd_includes/r2hc_simd_avx512.h"

static const ops_cycles_t ops_cnt[NUM_PRECISIONS][NUM_FFT_DIRS] =
                                                {{{0, 12, 34, 256, 152, 61},
                                                  {0, 14, 34, 256, 184, 60}},
                                                 {{0, 12, 34, 128, 8,   61},
                                                  {0, 14, 34, 128, 8,   60}}};

ops_cycles_t get_ops_cnt_r2hc_rfft10avx512(FFTZ_UINT8 precision,
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

static FFTZ_VOID r2hc_rfft10avx512_fp32_fwd(FFTZ_VOID *in_real,
                                            FFTZ_VOID *in_complex,
                                            FFTZ_VOID *out_real,
                                            FFTZ_VOID *out_complex,
                                            FFTZ_INTP n,
                                            aoclfftz_strides_t *strides,
                                            FFTZ_VOID *twd, FFTZ_UINT8 flag)
{
    AOCLFFTZ_LOG(DEBUG, global_logger_mode, "Enter");
    const FFTZ_FLOAT CRTM_10_1 =
        0.55901699437494742410229341718281905886015458990288f;
    const FFTZ_FLOAT CRTM_10_2 =
        0.25000000000000000000000000000000000000000000000000f;
    const FFTZ_FLOAT CRTM_10_3 =
        0.58778525229247315738615484497912915412138427663885f;
    const FFTZ_FLOAT CRTM_10_4 =
        0.95105651629515357211643933337938214340569863400000f;
    FFTZ_FLOAT *in_r = (FFTZ_FLOAT *)in_real;
    FFTZ_FLOAT *out_r = (FFTZ_FLOAT *)out_real;
    FFTZ_FLOAT *out_cp = (FFTZ_FLOAT *)out_complex;
#ifdef VOLATILE_STRIDE_ARRAY
    volatile FFTZ_INTP *in_strides = strides->in_strides;
    volatile FFTZ_INTP *out_strides = strides->out_strides;
#else
    FFTZ_INTP *in_strides  = strides->in_strides;
    FFTZ_INTP *out_strides = strides->out_strides;
#endif
    FFTZ_INTP v_in_stride  = strides->v_in_stride;
    FFTZ_INTP v_out_stride = strides->v_out_stride;
    FFTZ_UINT8 is_contiguous_in = (v_in_stride == 1);
    // Used for endpoints: DC and Nyquist.
    FFTZ_INTP v_out_dc_nyq_stride = strides->v_out_sym_stride;
    FFTZ_UINT8 is_contiguous_out_dc_nyq = (v_out_dc_nyq_stride == 1);

    FFTZ_INTP cnt;
    FFTZ_FLOAT *curr_in, *curr_out;
    FFTZ_INTP N = n / NUM_SETS_REAL_512_S;
    FFTZ_INTP remaining_sets = n % NUM_SETS_REAL_512_S;

    __m512 v_CRTM_10_1 = _mm512_set1_ps(CRTM_10_1);
    __m512 v_CRTM_10_2 = _mm512_set1_ps(CRTM_10_2);
    __m512 v_CRTM_10_3 = _mm512_set1_ps(CRTM_10_3);
    __m512 v_CRTM_10_4 = _mm512_set1_ps(CRTM_10_4);

    for (cnt = 0; cnt < N; cnt++)
    {
        __m512 v_in0, v_in1, v_in2, v_in3, v_in4, v_in5, v_in6, v_in7, v_in8,
               v_in9;
        __m512 v_s0, v_s1, v_s2, v_s3, v_s4, v_s5, v_s6, v_s7, v_s8, v_s9,
               v_s10, v_s11, v_s12, v_s13, v_s14, v_s15, v_s16, v_s17, v_s18,
               v_s19, v_s20, v_s21, v_s22, v_s23;
        __m512 v_t0, v_t1, v_t2, v_t3, v_t4, v_t5, v_t6, v_t7, v_t8, v_t9,
               v_t10, v_t11;
        __m512 v_out0, v_out1, v_out2, v_out3, v_out4, v_out5, v_out6, v_out7,
               v_out8, v_out9;

        curr_in = in_r;
        curr_out = out_r;

        // Input point 1: x(0)
        LDR_512_S(curr_in, v_in_stride, v_in0, is_contiguous_in);
        // Input point 2: x(1)
        curr_in = in_r + in_strides[1];
        LDR_512_S(curr_in, v_in_stride, v_in1, is_contiguous_in);
        // Input point 3: x(2)
        curr_in = in_r + in_strides[2];
        LDR_512_S(curr_in, v_in_stride, v_in2, is_contiguous_in);
        // Input point 4: x(3)
        curr_in = in_r + in_strides[3];
        LDR_512_S(curr_in, v_in_stride, v_in3, is_contiguous_in);
        // Input point 5: x(4)
        curr_in = in_r + in_strides[4];
        LDR_512_S(curr_in, v_in_stride, v_in4, is_contiguous_in);
        // Input point 6: x(5)
        curr_in = in_r + in_strides[5];
        LDR_512_S(curr_in, v_in_stride, v_in5, is_contiguous_in);
        // Input point 7: x(6)
        curr_in = in_r + in_strides[6];
        LDR_512_S(curr_in, v_in_stride, v_in6, is_contiguous_in);
        // Input point 8: x(7)
        curr_in = in_r + in_strides[7];
        LDR_512_S(curr_in, v_in_stride, v_in7, is_contiguous_in);
        // Input point 9: x(8)
        curr_in = in_r + in_strides[8];
        LDR_512_S(curr_in, v_in_stride, v_in8, is_contiguous_in);
        // Input point 10: x(9)
        curr_in = in_r + in_strides[9];
        LDR_512_S(curr_in, v_in_stride, v_in9, is_contiguous_in);

        v_s0 = _mm512_add_ps(v_in0, v_in5);
        v_s1 = _mm512_sub_ps(v_in0, v_in5);
        v_s2 = _mm512_add_ps(v_in1, v_in9);
        v_s3 = _mm512_sub_ps(v_in1, v_in9);
        v_s4 = _mm512_add_ps(v_in2, v_in3);
        v_s5 = _mm512_sub_ps(v_in2, v_in3);
        v_s6 = _mm512_add_ps(v_in4, v_in6);
        v_s7 = _mm512_sub_ps(v_in4, v_in6);
        v_s8 = _mm512_add_ps(v_in7, v_in8);
        v_s9 = _mm512_sub_ps(v_in7, v_in8);

        v_s10 = _mm512_add_ps(v_s2, v_s6);
        v_s14 = _mm512_add_ps(v_s4, v_s8);
        v_s18 = _mm512_add_ps(v_s10, v_s14);
        // Output point 1: X(0)
        v_out0 = _mm512_add_ps(v_s0, v_s18);

        v_s11 = _mm512_sub_ps(v_s2, v_s6);
        v_s17 = _mm512_sub_ps(v_s5, v_s9);
        v_s20 = _mm512_add_ps(v_s11, v_s17);
        v_s21 = _mm512_sub_ps(v_s11, v_s17);
        // Output point 10: X(9)
        v_out9 = _mm512_sub_ps(v_s1, v_s21);

        v_t0 = _mm512_mul_ps(v_CRTM_10_2, v_s21);
        v_t1 = _mm512_mul_ps(v_CRTM_10_1, v_s20);
        v_s22 = _mm512_add_ps(v_t0, v_s1);
        // Output point 2: X(1)
        v_out1 = _mm512_add_ps(v_s22, v_t1);
        // Output point 6: X(5)
        v_out5 = _mm512_sub_ps(v_s22, v_t1);

        v_s12 = _mm512_add_ps(v_s3, v_s7);
        v_s15 = _mm512_sub_ps(v_s4, v_s8);
        v_t4 = _mm512_mul_ps(v_CRTM_10_3, v_s12);
        v_t10 = _mm512_mul_ps(v_CRTM_10_4, v_s15);
        // Output point 3: X(2)
        v_out2 = NEGATE_512_S(_mm512_add_ps(v_t4, v_t10));

        v_t6 = _mm512_mul_ps(v_CRTM_10_3, v_s15);
        v_t8 = _mm512_mul_ps(v_CRTM_10_4, v_s12);
        // Output point 7: X(6)
        v_out6 = _mm512_sub_ps(v_t6, v_t8);

        v_s13 = _mm512_sub_ps(v_s7, v_s3);
        v_s16 = _mm512_add_ps(v_s5, v_s9);
        v_t5 = _mm512_mul_ps(v_CRTM_10_3, v_s13);
        v_t11 = _mm512_mul_ps(v_CRTM_10_4, v_s16);
        // Output point 9: X(8)
        v_out8 = _mm512_add_ps(v_t11, v_t5);

        v_t7 = _mm512_mul_ps(v_CRTM_10_3, v_s16);
        v_t9 = _mm512_mul_ps(v_CRTM_10_4, v_s13);
        // Output point 5: X(4)
        v_out4 = _mm512_sub_ps(v_t9, v_t7);

        v_s19 = _mm512_sub_ps(v_s10, v_s14);
        v_t2 = _mm512_mul_ps(v_CRTM_10_2, v_s18);
        v_t3 = _mm512_mul_ps(v_CRTM_10_1, v_s19);
        v_s23 = _mm512_sub_ps(v_s0, v_t2);
        // Output point 4: X(3)
        v_out3 = _mm512_add_ps(v_s23, v_t3);
        // Output point 8: X(7)
        v_out7 = _mm512_sub_ps(v_s23, v_t3);

        STR_512_S(curr_out, v_out_dc_nyq_stride, v_out0, is_contiguous_out_dc_nyq);
        curr_out = out_cp + out_strides[1];
        STRI_2x512_S(curr_out, v_out_stride, v_out1, v_out2);
        curr_out = out_cp + out_strides[3];
        STRI_2x512_S(curr_out, v_out_stride, v_out3, v_out4);
        curr_out = out_cp + out_strides[5];
        STRI_2x512_S(curr_out, v_out_stride, v_out5, v_out6);
        curr_out = out_cp + out_strides[7];
        STRI_2x512_S(curr_out, v_out_stride, v_out7, v_out8);
        curr_out = out_r + out_strides[9];
        STR_512_S(curr_out, v_out_dc_nyq_stride, v_out9, is_contiguous_out_dc_nyq);

        in_r  = in_r + (v_in_stride << 4);
        out_cp = out_cp + (v_out_stride << 4);
        out_r = out_r + (v_out_dc_nyq_stride << 4);
    }
    // tail cases
    if (remaining_sets & NUM_SETS_REAL_256_S)
    {
        __m256 v_in0, v_in1, v_in2, v_in3, v_in4, v_in5, v_in6, v_in7, v_in8,
               v_in9;
        __m256 v_s0, v_s1, v_s2, v_s3, v_s4, v_s5, v_s6, v_s7, v_s8, v_s9,
               v_s10, v_s11, v_s12, v_s13, v_s14, v_s15, v_s16, v_s17, v_s18,
               v_s19, v_s20, v_s21, v_s22, v_s23;
        __m256 v_t0, v_t1, v_t2, v_t3, v_t4, v_t5, v_t6, v_t7, v_t8, v_t9,
               v_t10, v_t11;
        __m256 v_out0, v_out1, v_out2, v_out3, v_out4, v_out5, v_out6, v_out7,
               v_out8, v_out9;

        __m256 v256_CRTM_10_1 = _mm512_castps512_ps256(v_CRTM_10_1);
        __m256 v256_CRTM_10_2 = _mm512_castps512_ps256(v_CRTM_10_2);
        __m256 v256_CRTM_10_3 = _mm512_castps512_ps256(v_CRTM_10_3);
        __m256 v256_CRTM_10_4 = _mm512_castps512_ps256(v_CRTM_10_4);

        curr_in = in_r;
        curr_out = out_r;

        // Input point 1: x(0)
        LDR_256_S(curr_in, v_in_stride, v_in0, is_contiguous_in);
        // Input point 2: x(1)
        curr_in = in_r + in_strides[1];
        LDR_256_S(curr_in, v_in_stride, v_in1, is_contiguous_in);
        // Input point 3: x(2)
        curr_in = in_r + in_strides[2];
        LDR_256_S(curr_in, v_in_stride, v_in2, is_contiguous_in);
        // Input point 4: x(3)
        curr_in = in_r + in_strides[3];
        LDR_256_S(curr_in, v_in_stride, v_in3, is_contiguous_in);
        // Input point 5: x(4)
        curr_in = in_r + in_strides[4];
        LDR_256_S(curr_in, v_in_stride, v_in4, is_contiguous_in);
        // Input point 6: x(5)
        curr_in = in_r + in_strides[5];
        LDR_256_S(curr_in, v_in_stride, v_in5, is_contiguous_in);
        // Input point 7: x(6)
        curr_in = in_r + in_strides[6];
        LDR_256_S(curr_in, v_in_stride, v_in6, is_contiguous_in);
        // Input point 8: x(7)
        curr_in = in_r + in_strides[7];
        LDR_256_S(curr_in, v_in_stride, v_in7, is_contiguous_in);
        // Input point 9: x(8)
        curr_in = in_r + in_strides[8];
        LDR_256_S(curr_in, v_in_stride, v_in8, is_contiguous_in);
        // Input point 10: x(9)
        curr_in = in_r + in_strides[9];
        LDR_256_S(curr_in, v_in_stride, v_in9, is_contiguous_in);

        v_s0 = _mm256_add_ps(v_in0, v_in5);
        v_s1 = _mm256_sub_ps(v_in0, v_in5);
        v_s2 = _mm256_add_ps(v_in1, v_in9);
        v_s3 = _mm256_sub_ps(v_in1, v_in9);
        v_s4 = _mm256_add_ps(v_in2, v_in3);
        v_s5 = _mm256_sub_ps(v_in2, v_in3);
        v_s6 = _mm256_add_ps(v_in4, v_in6);
        v_s7 = _mm256_sub_ps(v_in4, v_in6);
        v_s8 = _mm256_add_ps(v_in7, v_in8);
        v_s9 = _mm256_sub_ps(v_in7, v_in8);

        v_s10 = _mm256_add_ps(v_s2, v_s6);
        v_s14 = _mm256_add_ps(v_s4, v_s8);
        v_s18 = _mm256_add_ps(v_s10, v_s14);
        // Output point 1: X(0)
        v_out0 = _mm256_add_ps(v_s0, v_s18);

        v_s11 = _mm256_sub_ps(v_s2, v_s6);
        v_s17 = _mm256_sub_ps(v_s5, v_s9);
        v_s20 = _mm256_add_ps(v_s11, v_s17);
        v_s21 = _mm256_sub_ps(v_s11, v_s17);
        // Output point 10: X(9)
        v_out9 = _mm256_sub_ps(v_s1, v_s21);

        v_t0 = _mm256_mul_ps(v256_CRTM_10_2, v_s21);
        v_t1 = _mm256_mul_ps(v256_CRTM_10_1, v_s20);
        v_s22 = _mm256_add_ps(v_t0, v_s1);
        // Output point 2: X(1)
        v_out1 = _mm256_add_ps(v_s22, v_t1);
        // Output point 6: X(5)
        v_out5 = _mm256_sub_ps(v_s22, v_t1);

        v_s12 = _mm256_add_ps(v_s3, v_s7);
        v_s15 = _mm256_sub_ps(v_s4, v_s8);
        v_t4 = _mm256_mul_ps(v256_CRTM_10_3, v_s12);
        v_t10 = _mm256_mul_ps(v256_CRTM_10_4, v_s15);
        // Output point 3: X(2)
        v_out2 = NEGATE_256_S(_mm256_add_ps(v_t4, v_t10));

        v_t6 = _mm256_mul_ps(v256_CRTM_10_3, v_s15);
        v_t8 = _mm256_mul_ps(v256_CRTM_10_4, v_s12);
        // Output point 7: X(6)
        v_out6 = _mm256_sub_ps(v_t6, v_t8);

        v_s13 = _mm256_sub_ps(v_s7, v_s3);
        v_s16 = _mm256_add_ps(v_s5, v_s9);
        v_t5 = _mm256_mul_ps(v256_CRTM_10_3, v_s13);
        v_t11 = _mm256_mul_ps(v256_CRTM_10_4, v_s16);
        // Output point 9: X(8)
        v_out8 = _mm256_add_ps(v_t11, v_t5);

        v_t7 = _mm256_mul_ps(v256_CRTM_10_3, v_s16);
        v_t9 = _mm256_mul_ps(v256_CRTM_10_4, v_s13);
        // Output point 5: X(4)
        v_out4 = _mm256_sub_ps(v_t9, v_t7);

        v_s19 = _mm256_sub_ps(v_s10, v_s14);
        v_t2 = _mm256_mul_ps(v256_CRTM_10_2, v_s18);
        v_t3 = _mm256_mul_ps(v256_CRTM_10_1, v_s19);
        v_s23 = _mm256_sub_ps(v_s0, v_t2);
        // Output point 4: X(3)
        v_out3 = _mm256_add_ps(v_s23, v_t3);
        // Output point 8: X(7)
        v_out7 = _mm256_sub_ps(v_s23, v_t3);

        STR_256_S(curr_out, v_out_dc_nyq_stride, v_out0, is_contiguous_out_dc_nyq);
        curr_out = out_cp + out_strides[1];
        STRI_2x256_S(curr_out, v_out_stride, v_out1, v_out2);
        curr_out = out_cp + out_strides[3];
        STRI_2x256_S(curr_out, v_out_stride, v_out3, v_out4);
        curr_out = out_cp + out_strides[5];
        STRI_2x256_S(curr_out, v_out_stride, v_out5, v_out6);
        curr_out = out_cp + out_strides[7];
        STRI_2x256_S(curr_out, v_out_stride, v_out7, v_out8);
        curr_out = out_r + out_strides[9];
        STR_256_S(curr_out, v_out_dc_nyq_stride, v_out9, is_contiguous_out_dc_nyq);

        in_r  = in_r + (v_in_stride << 3);
        out_cp = out_cp + (v_out_stride << 3);
        out_r = out_r + (v_out_dc_nyq_stride << 3);
    }
    // tail cases
    if (remaining_sets & NUM_SETS_REAL_128_S)
    {
        __m128 v_in0, v_in1, v_in2, v_in3, v_in4, v_in5, v_in6, v_in7, v_in8,
               v_in9;
        __m128 v_s0, v_s1, v_s2, v_s3, v_s4, v_s5, v_s6, v_s7, v_s8, v_s9,
               v_s10, v_s11, v_s12, v_s13, v_s14, v_s15, v_s16, v_s17, v_s18,
               v_s19, v_s20, v_s21, v_s22, v_s23;
        __m128 v_t0, v_t1, v_t2, v_t3, v_t4, v_t5, v_t6, v_t7, v_t8, v_t9,
               v_t10, v_t11;
        __m128 v_out0, v_out1, v_out2, v_out3, v_out4, v_out5, v_out6, v_out7,
               v_out8, v_out9;

        __m128 v128_CRTM_10_1 = _mm512_castps512_ps128(v_CRTM_10_1);
        __m128 v128_CRTM_10_2 = _mm512_castps512_ps128(v_CRTM_10_2);
        __m128 v128_CRTM_10_3 = _mm512_castps512_ps128(v_CRTM_10_3);
        __m128 v128_CRTM_10_4 = _mm512_castps512_ps128(v_CRTM_10_4);

        curr_in = in_r;
        curr_out = out_r;

        // Input point 1: x(0)
        LDR_128_S(curr_in, v_in_stride, v_in0, is_contiguous_in);
        // Input point 2: x(1)
        curr_in = in_r + in_strides[1];
        LDR_128_S(curr_in, v_in_stride, v_in1, is_contiguous_in);
        // Input point 3: x(2)
        curr_in = in_r + in_strides[2];
        LDR_128_S(curr_in, v_in_stride, v_in2, is_contiguous_in);
        // Input point 4: x(3)
        curr_in = in_r + in_strides[3];
        LDR_128_S(curr_in, v_in_stride, v_in3, is_contiguous_in);
        // Input point 5: x(4)
        curr_in = in_r + in_strides[4];
        LDR_128_S(curr_in, v_in_stride, v_in4, is_contiguous_in);
        // Input point 6: x(5)
        curr_in = in_r + in_strides[5];
        LDR_128_S(curr_in, v_in_stride, v_in5, is_contiguous_in);
        // Input point 7: x(6)
        curr_in = in_r + in_strides[6];
        LDR_128_S(curr_in, v_in_stride, v_in6, is_contiguous_in);
        // Input point 8: x(7)
        curr_in = in_r + in_strides[7];
        LDR_128_S(curr_in, v_in_stride, v_in7, is_contiguous_in);
        // Input point 9: x(8)
        curr_in = in_r + in_strides[8];
        LDR_128_S(curr_in, v_in_stride, v_in8, is_contiguous_in);
        // Input point 10: x(9)
        curr_in = in_r + in_strides[9];
        LDR_128_S(curr_in, v_in_stride, v_in9, is_contiguous_in);

        v_s0 = _mm_add_ps(v_in0, v_in5);
        v_s1 = _mm_sub_ps(v_in0, v_in5);
        v_s2 = _mm_add_ps(v_in1, v_in9);
        v_s3 = _mm_sub_ps(v_in1, v_in9);
        v_s4 = _mm_add_ps(v_in2, v_in3);
        v_s5 = _mm_sub_ps(v_in2, v_in3);
        v_s6 = _mm_add_ps(v_in4, v_in6);
        v_s7 = _mm_sub_ps(v_in4, v_in6);
        v_s8 = _mm_add_ps(v_in7, v_in8);
        v_s9 = _mm_sub_ps(v_in7, v_in8);

        v_s10 = _mm_add_ps(v_s2, v_s6);
        v_s14 = _mm_add_ps(v_s4, v_s8);
        v_s18 = _mm_add_ps(v_s10, v_s14);
        // Output point 1: X(0)
        v_out0 = _mm_add_ps(v_s0, v_s18);

        v_s11 = _mm_sub_ps(v_s2, v_s6);
        v_s17 = _mm_sub_ps(v_s5, v_s9);
        v_s20 = _mm_add_ps(v_s11, v_s17);
        v_s21 = _mm_sub_ps(v_s11, v_s17);
        // Output point 10: X(9)
        v_out9 = _mm_sub_ps(v_s1, v_s21);

        v_t0 = _mm_mul_ps(v128_CRTM_10_2, v_s21);
        v_t1 = _mm_mul_ps(v128_CRTM_10_1, v_s20);
        v_s22 = _mm_add_ps(v_t0, v_s1);
        // Output point 2: X(1)
        v_out1 = _mm_add_ps(v_s22, v_t1);
        // Output point 6: X(5)
        v_out5 = _mm_sub_ps(v_s22, v_t1);

        v_s12 = _mm_add_ps(v_s3, v_s7);
        v_s15 = _mm_sub_ps(v_s4, v_s8);
        v_t4 = _mm_mul_ps(v128_CRTM_10_3, v_s12);
        v_t10 = _mm_mul_ps(v128_CRTM_10_4, v_s15);
        // Output point 3: X(2)
        v_out2 = NEGATE_128_S(_mm_add_ps(v_t4, v_t10));

        v_t6 = _mm_mul_ps(v128_CRTM_10_3, v_s15);
        v_t8 = _mm_mul_ps(v128_CRTM_10_4, v_s12);
        // Output point 7: X(6)
        v_out6 = _mm_sub_ps(v_t6, v_t8);

        v_s13 = _mm_sub_ps(v_s7, v_s3);
        v_s16 = _mm_add_ps(v_s5, v_s9);
        v_t5 = _mm_mul_ps(v128_CRTM_10_3, v_s13);
        v_t11 = _mm_mul_ps(v128_CRTM_10_4, v_s16);
        // Output point 9: X(8)
        v_out8 = _mm_add_ps(v_t11, v_t5);

        v_t7 = _mm_mul_ps(v128_CRTM_10_3, v_s16);
        v_t9 = _mm_mul_ps(v128_CRTM_10_4, v_s13);
        // Output point 5: X(4)
        v_out4 = _mm_sub_ps(v_t9, v_t7);

        v_s19 = _mm_sub_ps(v_s10, v_s14);
        v_t2 = _mm_mul_ps(v128_CRTM_10_2, v_s18);
        v_t3 = _mm_mul_ps(v128_CRTM_10_1, v_s19);
        v_s23 = _mm_sub_ps(v_s0, v_t2);
        // Output point 4: X(3)
        v_out3 = _mm_add_ps(v_s23, v_t3);
        // Output point 8: X(7)
        v_out7 = _mm_sub_ps(v_s23, v_t3);

        STR_128_S(curr_out, v_out_dc_nyq_stride, v_out0, is_contiguous_out_dc_nyq);
        curr_out = out_cp + out_strides[1];
        STRI_2x128_S(curr_out, v_out_stride, v_out1, v_out2);
        curr_out = out_cp + out_strides[3];
        STRI_2x128_S(curr_out, v_out_stride, v_out3, v_out4);
        curr_out = out_cp + out_strides[5];
        STRI_2x128_S(curr_out, v_out_stride, v_out5, v_out6);
        curr_out = out_cp + out_strides[7];
        STRI_2x128_S(curr_out, v_out_stride, v_out7, v_out8);
        curr_out = out_r + out_strides[9];
        STR_128_S(curr_out, v_out_dc_nyq_stride, v_out9, is_contiguous_out_dc_nyq);

        in_r  = in_r + (v_in_stride << 2);
        out_cp = out_cp + (v_out_stride << 2);
        out_r = out_r + (v_out_dc_nyq_stride << 2);
    }
    // tail cases
    if (remaining_sets & 2)
    {
        __m128 v_in0, v_in1, v_in2, v_in3, v_in4, v_in5, v_in6, v_in7, v_in8,
               v_in9;
        __m128 v_s0, v_s1, v_s2, v_s3, v_s4, v_s5, v_s6, v_s7, v_s8, v_s9,
               v_s10, v_s11, v_s12, v_s13, v_s14, v_s15, v_s16, v_s17, v_s18,
               v_s19, v_s20, v_s21, v_s22, v_s23;
        __m128 v_t0, v_t1, v_t2, v_t3, v_t4, v_t5, v_t6, v_t7, v_t8, v_t9,
               v_t10, v_t11;
        __m128 v_out0, v_out1, v_out2, v_out3, v_out4, v_out5, v_out6, v_out7,
               v_out8, v_out9;

        __m128 v128_CRTM_10_1 = _mm512_castps512_ps128(v_CRTM_10_1);
        __m128 v128_CRTM_10_2 = _mm512_castps512_ps128(v_CRTM_10_2);
        __m128 v128_CRTM_10_3 = _mm512_castps512_ps128(v_CRTM_10_3);
        __m128 v128_CRTM_10_4 = _mm512_castps512_ps128(v_CRTM_10_4);

        curr_in = in_r;
        curr_out = out_r;

        // Input point 1: x(0)
        LDHR_128_S(curr_in, v_in_stride, v_in0);
        // Input point 2: x(1)
        curr_in = in_r + in_strides[1];
        LDHR_128_S(curr_in, v_in_stride, v_in1);
        // Input point 3: x(2)
        curr_in = in_r + in_strides[2];
        LDHR_128_S(curr_in, v_in_stride, v_in2);
        // Input point 4: x(3)
        curr_in = in_r + in_strides[3];
        LDHR_128_S(curr_in, v_in_stride, v_in3);
        // Input point 5: x(4)
        curr_in = in_r + in_strides[4];
        LDHR_128_S(curr_in, v_in_stride, v_in4);
        // Input point 6: x(5)
        curr_in = in_r + in_strides[5];
        LDHR_128_S(curr_in, v_in_stride, v_in5);
        // Input point 7: x(6)
        curr_in = in_r + in_strides[6];
        LDHR_128_S(curr_in, v_in_stride, v_in6);
        // Input point 8: x(7)
        curr_in = in_r + in_strides[7];
        LDHR_128_S(curr_in, v_in_stride, v_in7);
        // Input point 9: x(8)
        curr_in = in_r + in_strides[8];
        LDHR_128_S(curr_in, v_in_stride, v_in8);
        // Input point 10: x(9)
        curr_in = in_r + in_strides[9];
        LDHR_128_S(curr_in, v_in_stride, v_in9);

        v_s0 = _mm_add_ps(v_in0, v_in5);
        v_s1 = _mm_sub_ps(v_in0, v_in5);
        v_s2 = _mm_add_ps(v_in1, v_in9);
        v_s3 = _mm_sub_ps(v_in1, v_in9);
        v_s4 = _mm_add_ps(v_in2, v_in3);
        v_s5 = _mm_sub_ps(v_in2, v_in3);
        v_s6 = _mm_add_ps(v_in4, v_in6);
        v_s7 = _mm_sub_ps(v_in4, v_in6);
        v_s8 = _mm_add_ps(v_in7, v_in8);
        v_s9 = _mm_sub_ps(v_in7, v_in8);

        v_s10 = _mm_add_ps(v_s2, v_s6);
        v_s14 = _mm_add_ps(v_s4, v_s8);
        v_s18 = _mm_add_ps(v_s10, v_s14);
        // Output point 1: X(0)
        v_out0 = _mm_add_ps(v_s0, v_s18);

        v_s11 = _mm_sub_ps(v_s2, v_s6);
        v_s17 = _mm_sub_ps(v_s5, v_s9);
        v_s20 = _mm_add_ps(v_s11, v_s17);
        v_s21 = _mm_sub_ps(v_s11, v_s17);
        // Output point 10: X(9)
        v_out9 = _mm_sub_ps(v_s1, v_s21);

        v_t0 = _mm_mul_ps(v128_CRTM_10_2, v_s21);
        v_t1 = _mm_mul_ps(v128_CRTM_10_1, v_s20);
        v_s22 = _mm_add_ps(v_t0, v_s1);
        // Output point 2: X(1)
        v_out1 = _mm_add_ps(v_s22, v_t1);
        // Output point 6: X(5)
        v_out5 = _mm_sub_ps(v_s22, v_t1);

        v_s12 = _mm_add_ps(v_s3, v_s7);
        v_s15 = _mm_sub_ps(v_s4, v_s8);
        v_t4 = _mm_mul_ps(v128_CRTM_10_3, v_s12);
        v_t10 = _mm_mul_ps(v128_CRTM_10_4, v_s15);
        // Output point 3: X(2)
        v_out2 = NEGATE_128_S(_mm_add_ps(v_t4, v_t10));

        v_t6 = _mm_mul_ps(v128_CRTM_10_3, v_s15);
        v_t8 = _mm_mul_ps(v128_CRTM_10_4, v_s12);
        // Output point 7: X(6)
        v_out6 = _mm_sub_ps(v_t6, v_t8);

        v_s13 = _mm_sub_ps(v_s7, v_s3);
        v_s16 = _mm_add_ps(v_s5, v_s9);
        v_t5 = _mm_mul_ps(v128_CRTM_10_3, v_s13);
        v_t11 = _mm_mul_ps(v128_CRTM_10_4, v_s16);
        // Output point 9: X(8)
        v_out8 = _mm_add_ps(v_t11, v_t5);

        v_t7 = _mm_mul_ps(v128_CRTM_10_3, v_s16);
        v_t9 = _mm_mul_ps(v128_CRTM_10_4, v_s13);
        // Output point 5: X(4)
        v_out4 = _mm_sub_ps(v_t9, v_t7);

        v_s19 = _mm_sub_ps(v_s10, v_s14);
        v_t2 = _mm_mul_ps(v128_CRTM_10_2, v_s18);
        v_t3 = _mm_mul_ps(v128_CRTM_10_1, v_s19);
        v_s23 = _mm_sub_ps(v_s0, v_t2);
        // Output point 4: X(3)
        v_out3 = _mm_add_ps(v_s23, v_t3);
        // Output point 8: X(7)
        v_out7 = _mm_sub_ps(v_s23, v_t3);

        STHR_128_S(curr_out, v_out_dc_nyq_stride, v_out0);
        curr_out = out_cp + out_strides[1];
        STHRI_2x128_S(curr_out, v_out_stride, v_out1, v_out2);
        curr_out = out_cp + out_strides[3];
        STHRI_2x128_S(curr_out, v_out_stride, v_out3, v_out4);
        curr_out = out_cp + out_strides[5];
        STHRI_2x128_S(curr_out, v_out_stride, v_out5, v_out6);
        curr_out = out_cp + out_strides[7];
        STHRI_2x128_S(curr_out, v_out_stride, v_out7, v_out8);
        curr_out = out_r + out_strides[9];
        STHR_128_S(curr_out, v_out_dc_nyq_stride, v_out9);

        in_r  = in_r + (v_in_stride << 1);
        out_cp = out_cp + (v_out_stride << 1);
        out_r = out_r + (v_out_dc_nyq_stride << 1);
    }
    // tail cases
    if (remaining_sets & 1)
    {
        FFTZ_FLOAT v_in0, v_in1, v_in2, v_in3, v_in4, v_in5, v_in6, v_in7,
            v_in8, v_in9;
        FFTZ_FLOAT v_s0, v_s1, v_s2, v_s3, v_s4, v_s5, v_s6, v_s7, v_s8, v_s9,
              v_s10, v_s11, v_s12, v_s13, v_s14, v_s15, v_s16, v_s17, v_s18,
              v_s19, v_s20, v_s21, v_s22, v_s23;
        FFTZ_FLOAT v_t0, v_t1, v_t2, v_t3, v_t4, v_t5, v_t6, v_t7, v_t8, v_t9,
            v_t10, v_t11;

        curr_in = in_r;
        curr_out = out_r;
        // Input point 1: x(0)
        v_in0 = *in_r;
        // Input point 2: x(1)
        v_in1 = in_r[in_strides[1]];
        // Input point 3: x(2)
        v_in2 = in_r[in_strides[2]];
        // Input point 4: x(3)
        v_in3 = in_r[in_strides[3]];
        // Input point 5: x(4)
        v_in4 = in_r[in_strides[4]];
        // Input point 6: x(5)
        v_in5 = in_r[in_strides[5]];
        // Input point 7: x(6)
        v_in6 = in_r[in_strides[6]];
        // Input point 8: x(7)
        v_in7 = in_r[in_strides[7]];
        // Input point 9: x(8)
        v_in8 = in_r[in_strides[8]];
        // Input point 10: x(9)
        v_in9 = in_r[in_strides[9]];

        v_s0 = v_in0 + v_in5;
        v_s1 = v_in0 - v_in5;
        v_s2 = v_in1 + v_in9;
        v_s3 = v_in1 - v_in9;
        v_s4 = v_in2 + v_in3;
        v_s5 = v_in2 - v_in3;
        v_s6 = v_in4 + v_in6;
        v_s7 = v_in4 - v_in6;
        v_s8 = v_in7 + v_in8;
        v_s9 = v_in7 - v_in8;

        v_s10 = v_s2 + v_s6;
        v_s14 = v_s4 + v_s8;
        v_s18 = v_s10 + v_s14;
        // Output point 1: X(0)
        *out_r = v_s0 + v_s18;

        v_s11 = v_s2 - v_s6;
        v_s17 = v_s5 - v_s9;
        v_s20 = v_s11 + v_s17;
        v_s21 = v_s11 - v_s17;
        // Output point 10: X(9)
        out_r[out_strides[9]] = v_s1 - v_s21;

        v_t0 = CRTM_10_2 * v_s21;
        v_t1 = CRTM_10_1 * v_s20;
        v_s22 = v_t0 + v_s1;
        // Output point 2: X(1)
        out_cp[out_strides[1]] = v_s22 + v_t1;
        // Output point 6: X(5)
        out_cp[out_strides[5]] = v_s22 - v_t1;

        v_s12 = v_s3 + v_s7;
        v_s15 = v_s4 - v_s8;
        v_t4 = CRTM_10_3 * v_s12;
        v_t10 = CRTM_10_4 * v_s15;
        // Output point 3: X(2)
        out_cp[out_strides[2]] = -(v_t4 + v_t10);
        v_t6 = CRTM_10_3 * v_s15;
        v_t8 = CRTM_10_4 * v_s12;
        // Output point 7: X(6)
        out_cp[out_strides[6]] = v_t6 - v_t8;

        v_s13 = v_s7 - v_s3;
        v_s16 = v_s5 + v_s9;
        v_t5 = CRTM_10_3 * v_s13;
        v_t11 = CRTM_10_4 * v_s16;
        // Output point 9: X(8)
        out_cp[out_strides[8]] = v_t11 + v_t5;
        v_t7 = CRTM_10_3 * v_s16;
        v_t9 = CRTM_10_4 * v_s13;
        // Output point 5: X(4)
        out_cp[out_strides[4]] = v_t9 - v_t7;

        v_s19 = v_s10 - v_s14;
        v_t2 = CRTM_10_2 * v_s18;
        v_t3 = CRTM_10_1 * v_s19;
        v_s23 = v_s0 - v_t2;
        // Output point 4: X(3)
        out_cp[out_strides[3]] = v_s23 + v_t3;
        // Output point 8: X(7)
        out_cp[out_strides[7]] = v_s23 - v_t3;
    }
    AOCLFFTZ_LOG(DEBUG, global_logger_mode, "Exit");
}

static FFTZ_VOID r2hc_rfft10avx512_fp32_bwd(FFTZ_VOID *in_real,
                                            FFTZ_VOID *in_complex,
                                            FFTZ_VOID *out_real,
                                            FFTZ_VOID *out_complex,
                                            FFTZ_INTP n,
                                            aoclfftz_strides_t *strides,
                                            FFTZ_VOID *twd, FFTZ_UINT8 flag)
{
    AOCLFFTZ_LOG(DEBUG, global_logger_mode, "Enter");
    const FFTZ_FLOAT CRTM_10_1 =
        1.118033988749894848204586834365638117720309180f;
    const FFTZ_FLOAT CRTM_10_2 =
        0.500000000000000000000000000000000000000000000f;
    const FFTZ_FLOAT CRTM_10_3 =
        2.000000000000000000000000000000000000000000000f;
    const FFTZ_FLOAT CRTM_10_4 =
        1.175570504584946258337411909278145537195304875f;
    const FFTZ_FLOAT CRTM_10_5 =
        1.902113032590307144232878666758764286811397268f;
    FFTZ_FLOAT *in_r = (FFTZ_FLOAT *)in_real;
    FFTZ_FLOAT *in_cp = (FFTZ_FLOAT *)in_complex;
    FFTZ_FLOAT *out_r = (FFTZ_FLOAT *)out_real;
#ifdef VOLATILE_STRIDE_ARRAY
    volatile FFTZ_INTP *in_strides = strides->in_strides;
    volatile FFTZ_INTP *out_strides = strides->out_strides;
#else
    FFTZ_INTP *in_strides  = strides->in_strides;
    FFTZ_INTP *out_strides = strides->out_strides;
#endif
    FFTZ_INTP v_in_stride  = strides->v_in_stride;
    FFTZ_INTP v_out_stride = strides->v_out_stride;
    FFTZ_UINT8 is_contiguous_out = (v_out_stride == 1);
    // Used for endpoints: DC and Nyquist.
    FFTZ_INTP v_in_dc_nyq_stride = strides->v_in_sym_stride;
    FFTZ_UINT8 is_contiguous_in_dc_nyq = (v_in_dc_nyq_stride == 1);

    FFTZ_INTP cnt;
    FFTZ_FLOAT *curr_in, *curr_out;
    FFTZ_INTP N = n / NUM_SETS_REAL_512_S;
    FFTZ_INTP remaining_sets = n % NUM_SETS_REAL_512_S;

    __m512 v_CRTM_10_1 = _mm512_set1_ps(CRTM_10_1);
    __m512 v_CRTM_10_2 = _mm512_set1_ps(CRTM_10_2);
    __m512 v_CRTM_10_3 = _mm512_set1_ps(CRTM_10_3);
    __m512 v_CRTM_10_4 = _mm512_set1_ps(CRTM_10_4);
    __m512 v_CRTM_10_5 = _mm512_set1_ps(CRTM_10_5);

    for (cnt = 0; cnt < N; cnt++)
    {
        __m512 v_in0, v_in1, v_in2, v_in3, v_in4, v_in5, v_in6, v_in7, v_in8,
               v_in9;
        __m512 v_s0, v_s1, v_s2, v_s3, v_s4, v_s5, v_s6, v_s7, v_s8, v_s9,
               v_s10, v_s11, v_s12, v_s13, v_s14, v_s15, v_s16, v_s17, v_s18,
               v_s19, v_s20, v_s21, v_s22, v_s23;
        __m512 v_t0, v_t1, v_t2, v_t3, v_t4, v_t5, v_t6, v_t7, v_t8,
               v_t9, v_t10, v_t11, v_t12, v_t13;
        __m512 v_out0, v_out1, v_out2, v_out3, v_out4, v_out5, v_out6, v_out7,
               v_out8, v_out9;

        curr_in = in_r;
        curr_out = out_r;

        // Input point 1: x(0)
        LDR_512_S(curr_in, v_in_dc_nyq_stride, v_in0, is_contiguous_in_dc_nyq);
        // Input point 2: x(1) & Input point 3: x(2)
        curr_in = in_cp + in_strides[1];
        LDRI_2x512_S(curr_in, v_in_stride, v_in1, v_in2);
        // Input point 4: x(3) & Input point 5: x(4)
        curr_in = in_cp + in_strides[3];
        LDRI_2x512_S(curr_in, v_in_stride, v_in3, v_in4);
        // Input point 6: x(5) & Input point 7: x(6)
        curr_in = in_cp + in_strides[5];
        LDRI_2x512_S(curr_in, v_in_stride, v_in5, v_in6);
        // Input point 8: x(7) & Input point 9: x(8)
        curr_in = in_cp + in_strides[7];
        LDRI_2x512_S(curr_in, v_in_stride, v_in7, v_in8);
        // Input point 10: x(9) & Input point 11: x(10)
        curr_in = in_r + in_strides[9];
        LDR_512_S(curr_in, v_in_dc_nyq_stride, v_in9, is_contiguous_in_dc_nyq);

        v_s0 = _mm512_add_ps(v_in0, v_in9);
        v_s1 = _mm512_sub_ps(v_in0, v_in9);
        v_s2 = _mm512_add_ps(v_in1, v_in7);
        v_s3 = _mm512_sub_ps(v_in1, v_in7);
        v_s4 = _mm512_add_ps(v_in2, v_in8);
        v_s5 = _mm512_sub_ps(v_in2, v_in8);
        v_t10 = _mm512_mul_ps(v_CRTM_10_4, v_s5);
        v_s8 = _mm512_add_ps(v_in4, v_in6);

        v_t8 = _mm512_mul_ps(v_CRTM_10_4, v_s8);
        v_t9 = _mm512_mul_ps(v_CRTM_10_5, v_s4);
        v_s22 = _mm512_sub_ps(v_t8, v_t9);
        v_s9 = _mm512_sub_ps(v_in4, v_in6);
        v_t11 = _mm512_mul_ps(v_CRTM_10_5, v_s9);
        v_s23 = _mm512_sub_ps(v_t11, v_t10);

        v_s6 = _mm512_add_ps(v_in3, v_in5);
        v_s7 = _mm512_sub_ps(v_in3, v_in5);
        v_s13 = _mm512_sub_ps(v_s3, v_s7);
        v_t13 = _mm512_mul_ps(v_CRTM_10_3, v_s13);
        // Output point 6: X(5)
        v_out5 = _mm512_sub_ps(v_s1, v_t13);
        curr_out = out_r + out_strides[5];
        STR_512_S(curr_out, v_out_stride, v_out5, is_contiguous_out);

        v_s10 = _mm512_add_ps(v_s2, v_s6);
        v_t12 = _mm512_mul_ps(v_CRTM_10_3, v_s10);
        // Output point 1: X(0)
        v_out0 = _mm512_add_ps(v_s0, v_t12);
        curr_out = out_r;
        STR_512_S(curr_out, v_out_stride, v_out0, is_contiguous_out);

        v_t0 = _mm512_mul_ps(v_CRTM_10_2, v_s10);
        v_s15 = _mm512_sub_ps(v_s0, v_t0);
        v_s11 = _mm512_sub_ps(v_s2, v_s6);
        v_t2 = _mm512_mul_ps(v_CRTM_10_1, v_s11);
        v_s18 = _mm512_add_ps(v_s15, v_t2);
        v_s19 = _mm512_sub_ps(v_s15, v_t2);
        // Output point 5: X(4)
        v_out4 = _mm512_add_ps(v_s19, v_s23);
        curr_out = out_r + out_strides[4];
        STR_512_S(curr_out, v_out_stride, v_out4, is_contiguous_out);
        // Output point 7: X(6)
        v_out6 = _mm512_sub_ps(v_s19, v_s23);
        curr_out = out_r + out_strides[6];
        STR_512_S(curr_out, v_out_stride, v_out6, is_contiguous_out);

        v_s12 = _mm512_add_ps(v_s3, v_s7);
        v_t3 = _mm512_mul_ps(v_CRTM_10_1, v_s12);
        v_t1 = _mm512_mul_ps(v_CRTM_10_2, v_s13);
        v_s14 = _mm512_add_ps(v_s1, v_t1);
        v_s16 = _mm512_add_ps(v_s14, v_t3);
        v_s17 = _mm512_sub_ps(v_s14, v_t3);
        // Output point 4: X(3)
        v_out3 = _mm512_add_ps(v_s17, v_s22);
        curr_out = out_r + out_strides[3];
        STR_512_S(curr_out, v_out_stride, v_out3, is_contiguous_out);
        // Output point 8: X(7)
        v_out7 = _mm512_sub_ps(v_s17, v_s22);
        curr_out = out_r + out_strides[7];
        STR_512_S(curr_out, v_out_stride, v_out7, is_contiguous_out);

        v_t4 = _mm512_mul_ps(v_CRTM_10_4, v_s4);
        v_t5 = _mm512_mul_ps(v_CRTM_10_5, v_s8);
        v_s20 = _mm512_add_ps(v_t4, v_t5);
        // Output point 2: X(1)
        v_out1 = _mm512_sub_ps(v_s16, v_s20);
        curr_out = out_r + out_strides[1];
        STR_512_S(curr_out, v_out_stride, v_out1, is_contiguous_out);
        // Output point 10: X(9)
        v_out9 = _mm512_add_ps(v_s16, v_s20);
        curr_out = out_r + out_strides[9];
        STR_512_S(curr_out, v_out_stride, v_out9, is_contiguous_out);

        v_t6 = _mm512_mul_ps(v_CRTM_10_4, v_s9);
        v_t7 = _mm512_mul_ps(v_CRTM_10_5, v_s5);
        v_s21 = _mm512_add_ps(v_t6, v_t7);
        // Output point 3: X(2)
        v_out2 = _mm512_sub_ps(v_s18, v_s21);
        curr_out = out_r + out_strides[2];
        STR_512_S(curr_out, v_out_stride, v_out2, is_contiguous_out);
        // Output point 9: X(8)
        v_out8 = _mm512_add_ps(v_s18, v_s21);
        curr_out = out_r + out_strides[8];
        STR_512_S(curr_out, v_out_stride, v_out8, is_contiguous_out);

        in_r = in_r + (v_in_dc_nyq_stride << 4);
        in_cp  = in_cp + (v_in_stride << 4);
        out_r = out_r + (v_out_stride << 4);
    }
    // tail cases
    if (remaining_sets & NUM_SETS_REAL_256_S)
    {
        __m256 v_in0, v_in1, v_in2, v_in3, v_in4, v_in5, v_in6, v_in7, v_in8,
               v_in9;
        __m256 v_s0, v_s1, v_s2, v_s3, v_s4, v_s5, v_s6, v_s7, v_s8, v_s9,
               v_s10, v_s11, v_s12, v_s13, v_s14, v_s15, v_s16, v_s17, v_s18,
               v_s19, v_s20, v_s21, v_s22, v_s23;
        __m256 v_t0, v_t1, v_t2, v_t3, v_t4, v_t5, v_t6, v_t7, v_t8,
               v_t9, v_t10, v_t11, v_t12, v_t13;
        __m256 v_out0, v_out1, v_out2, v_out3, v_out4, v_out5, v_out6, v_out7,
               v_out8, v_out9;

        __m256 v256_CRTM_10_1 = _mm512_castps512_ps256(v_CRTM_10_1);
        __m256 v256_CRTM_10_2 = _mm512_castps512_ps256(v_CRTM_10_2);
        __m256 v256_CRTM_10_3 = _mm512_castps512_ps256(v_CRTM_10_3);
        __m256 v256_CRTM_10_4 = _mm512_castps512_ps256(v_CRTM_10_4);
        __m256 v256_CRTM_10_5 = _mm512_castps512_ps256(v_CRTM_10_5);

        curr_in = in_r;
        curr_out = out_r;

        // Input point 1: x(0)
        LDR_256_S(curr_in, v_in_dc_nyq_stride, v_in0, is_contiguous_in_dc_nyq);
        // Input point 2: x(1) & Input point 3: x(2)
        curr_in = in_cp + in_strides[1];
        LDRI_2x256_S(curr_in, v_in_stride, v_in1, v_in2);
        // Input point 4: x(3) & Input point 5: x(4)
        curr_in = in_cp + in_strides[3];
        LDRI_2x256_S(curr_in, v_in_stride, v_in3, v_in4);
        // Input point 6: x(5) & Input point 7: x(6)
        curr_in = in_cp + in_strides[5];
        LDRI_2x256_S(curr_in, v_in_stride, v_in5, v_in6);
        // Input point 8: x(7) & Input point 9: x(8)
        curr_in = in_cp + in_strides[7];
        LDRI_2x256_S(curr_in, v_in_stride, v_in7, v_in8);
        // Input point 10: x(9) & Input point 11: x(10)
        curr_in = in_r + in_strides[9];
        LDR_256_S(curr_in, v_in_dc_nyq_stride, v_in9, is_contiguous_in_dc_nyq);

        v_s0 = _mm256_add_ps(v_in0, v_in9);
        v_s1 = _mm256_sub_ps(v_in0, v_in9);
        v_s2 = _mm256_add_ps(v_in1, v_in7);
        v_s3 = _mm256_sub_ps(v_in1, v_in7);
        v_s4 = _mm256_add_ps(v_in2, v_in8);
        v_s5 = _mm256_sub_ps(v_in2, v_in8);
        v_t10 = _mm256_mul_ps(v256_CRTM_10_4, v_s5);
        v_s8 = _mm256_add_ps(v_in4, v_in6);

        v_t8 = _mm256_mul_ps(v256_CRTM_10_4, v_s8);
        v_t9 = _mm256_mul_ps(v256_CRTM_10_5, v_s4);
        v_s22 = _mm256_sub_ps(v_t8, v_t9);
        v_s9 = _mm256_sub_ps(v_in4, v_in6);
        v_t11 = _mm256_mul_ps(v256_CRTM_10_5, v_s9);
        v_s23 = _mm256_sub_ps(v_t11, v_t10);

        v_s6 = _mm256_add_ps(v_in3, v_in5);
        v_s7 = _mm256_sub_ps(v_in3, v_in5);
        v_s13 = _mm256_sub_ps(v_s3, v_s7);
        v_t13 = _mm256_mul_ps(v256_CRTM_10_3, v_s13);
        // Output point 6: X(5)
        v_out5 = _mm256_sub_ps(v_s1, v_t13);
        curr_out = out_r + out_strides[5];
        STR_256_S(curr_out, v_out_stride, v_out5, is_contiguous_out);

        v_s10 = _mm256_add_ps(v_s2, v_s6);
        v_t12 = _mm256_mul_ps(v256_CRTM_10_3, v_s10);
        // Output point 1: X(0)
        v_out0 = _mm256_add_ps(v_s0, v_t12);
        curr_out = out_r;
        STR_256_S(curr_out, v_out_stride, v_out0, is_contiguous_out);

        v_t0 = _mm256_mul_ps(v256_CRTM_10_2, v_s10);
        v_s15 = _mm256_sub_ps(v_s0, v_t0);
        v_s11 = _mm256_sub_ps(v_s2, v_s6);
        v_t2 = _mm256_mul_ps(v256_CRTM_10_1, v_s11);
        v_s18 = _mm256_add_ps(v_s15, v_t2);
        v_s19 = _mm256_sub_ps(v_s15, v_t2);
        // Output point 5: X(4)
        v_out4 = _mm256_add_ps(v_s19, v_s23);
        curr_out = out_r + out_strides[4];
        STR_256_S(curr_out, v_out_stride, v_out4, is_contiguous_out);
        // Output point 7: X(6)
        v_out6 = _mm256_sub_ps(v_s19, v_s23);
        curr_out = out_r + out_strides[6];
        STR_256_S(curr_out, v_out_stride, v_out6, is_contiguous_out);

        v_s12 = _mm256_add_ps(v_s3, v_s7);
        v_t3 = _mm256_mul_ps(v256_CRTM_10_1, v_s12);
        v_t1 = _mm256_mul_ps(v256_CRTM_10_2, v_s13);
        v_s14 = _mm256_add_ps(v_s1, v_t1);
        v_s16 = _mm256_add_ps(v_s14, v_t3);
        v_s17 = _mm256_sub_ps(v_s14, v_t3);
        // Output point 4: X(3)
        v_out3 = _mm256_add_ps(v_s17, v_s22);
        curr_out = out_r + out_strides[3];
        STR_256_S(curr_out, v_out_stride, v_out3, is_contiguous_out);
        // Output point 8: X(7)
        v_out7 = _mm256_sub_ps(v_s17, v_s22);
        curr_out = out_r + out_strides[7];
        STR_256_S(curr_out, v_out_stride, v_out7, is_contiguous_out);

        v_t4 = _mm256_mul_ps(v256_CRTM_10_4, v_s4);
        v_t5 = _mm256_mul_ps(v256_CRTM_10_5, v_s8);
        v_s20 = _mm256_add_ps(v_t4, v_t5);
        // Output point 2: X(1)
        v_out1 = _mm256_sub_ps(v_s16, v_s20);
        curr_out = out_r + out_strides[1];
        STR_256_S(curr_out, v_out_stride, v_out1, is_contiguous_out);
        // Output point 10: X(9)
        v_out9 = _mm256_add_ps(v_s16, v_s20);
        curr_out = out_r + out_strides[9];
        STR_256_S(curr_out, v_out_stride, v_out9, is_contiguous_out);

        v_t6 = _mm256_mul_ps(v256_CRTM_10_4, v_s9);
        v_t7 = _mm256_mul_ps(v256_CRTM_10_5, v_s5);
        v_s21 = _mm256_add_ps(v_t6, v_t7);
        // Output point 3: X(2)
        v_out2 = _mm256_sub_ps(v_s18, v_s21);
        curr_out = out_r + out_strides[2];
        STR_256_S(curr_out, v_out_stride, v_out2, is_contiguous_out);
        // Output point 9: X(8)
        v_out8 = _mm256_add_ps(v_s18, v_s21);
        curr_out = out_r + out_strides[8];
        STR_256_S(curr_out, v_out_stride, v_out8, is_contiguous_out);

        in_r = in_r + (v_in_dc_nyq_stride << 3);
        in_cp  = in_cp + (v_in_stride << 3);
        out_r = out_r + (v_out_stride << 3);
    }
    // tail cases
    if (remaining_sets & NUM_SETS_REAL_128_S)
    {
        __m128 v_in0, v_in1, v_in2, v_in3, v_in4, v_in5, v_in6, v_in7, v_in8,
               v_in9;
        __m128 v_s0, v_s1, v_s2, v_s3, v_s4, v_s5, v_s6, v_s7, v_s8, v_s9,
               v_s10, v_s11, v_s12, v_s13, v_s14, v_s15, v_s16, v_s17, v_s18,
               v_s19, v_s20, v_s21, v_s22, v_s23;
        __m128 v_t0, v_t1, v_t2, v_t3, v_t4, v_t5, v_t6, v_t7, v_t8,
               v_t9, v_t10, v_t11, v_t12, v_t13;
        __m128 v_out0, v_out1, v_out2, v_out3, v_out4, v_out5, v_out6, v_out7,
               v_out8, v_out9;

        __m128 v128_CRTM_10_1 = _mm512_castps512_ps128(v_CRTM_10_1);
        __m128 v128_CRTM_10_2 = _mm512_castps512_ps128(v_CRTM_10_2);
        __m128 v128_CRTM_10_3 = _mm512_castps512_ps128(v_CRTM_10_3);
        __m128 v128_CRTM_10_4 = _mm512_castps512_ps128(v_CRTM_10_4);
        __m128 v128_CRTM_10_5 = _mm512_castps512_ps128(v_CRTM_10_5);

        curr_in = in_r;
        curr_out = out_r;

        // Input point 1: x(0)
        LDR_128_S(curr_in, v_in_dc_nyq_stride, v_in0, is_contiguous_in_dc_nyq);
        // Input point 2: x(1) & Input point 3: x(2)
        curr_in = in_cp + in_strides[1];
        LDRI_2x128_S(curr_in, v_in_stride, v_in1, v_in2);
        // Input point 4: x(3) & Input point 5: x(4)
        curr_in = in_cp + in_strides[3];
        LDRI_2x128_S(curr_in, v_in_stride, v_in3, v_in4);
        // Input point 6: x(5) & Input point 7: x(6)
        curr_in = in_cp + in_strides[5];
        LDRI_2x128_S(curr_in, v_in_stride, v_in5, v_in6);
        // Input point 8: x(7) & Input point 9: x(8)
        curr_in = in_cp + in_strides[7];
        LDRI_2x128_S(curr_in, v_in_stride, v_in7, v_in8);
        // Input point 10: x(9) & Input point 11: x(10)
        curr_in = in_r + in_strides[9];
        LDR_128_S(curr_in, v_in_dc_nyq_stride, v_in9, is_contiguous_in_dc_nyq);

        v_s0 = _mm_add_ps(v_in0, v_in9);
        v_s1 = _mm_sub_ps(v_in0, v_in9);
        v_s2 = _mm_add_ps(v_in1, v_in7);
        v_s3 = _mm_sub_ps(v_in1, v_in7);
        v_s4 = _mm_add_ps(v_in2, v_in8);
        v_s5 = _mm_sub_ps(v_in2, v_in8);
        v_t10 = _mm_mul_ps(v128_CRTM_10_4, v_s5);
        v_s8 = _mm_add_ps(v_in4, v_in6);

        v_t8 = _mm_mul_ps(v128_CRTM_10_4, v_s8);
        v_t9 = _mm_mul_ps(v128_CRTM_10_5, v_s4);
        v_s22 = _mm_sub_ps(v_t8, v_t9);
        v_s9 = _mm_sub_ps(v_in4, v_in6);
        v_t11 = _mm_mul_ps(v128_CRTM_10_5, v_s9);
        v_s23 = _mm_sub_ps(v_t11, v_t10);

        v_s6 = _mm_add_ps(v_in3, v_in5);
        v_s7 = _mm_sub_ps(v_in3, v_in5);
        v_s13 = _mm_sub_ps(v_s3, v_s7);
        v_t13 = _mm_mul_ps(v128_CRTM_10_3, v_s13);
        // Output point 6: X(5)
        v_out5 = _mm_sub_ps(v_s1, v_t13);
        curr_out = out_r + out_strides[5];
        STR_128_S(curr_out, v_out_stride, v_out5, is_contiguous_out);

        v_s10 = _mm_add_ps(v_s2, v_s6);
        v_t12 = _mm_mul_ps(v128_CRTM_10_3, v_s10);
        // Output point 1: X(0)
        v_out0 = _mm_add_ps(v_s0, v_t12);
        curr_out = out_r;
        STR_128_S(curr_out, v_out_stride, v_out0, is_contiguous_out);

        v_t0 = _mm_mul_ps(v128_CRTM_10_2, v_s10);
        v_s15 = _mm_sub_ps(v_s0, v_t0);
        v_s11 = _mm_sub_ps(v_s2, v_s6);
        v_t2 = _mm_mul_ps(v128_CRTM_10_1, v_s11);
        v_s18 = _mm_add_ps(v_s15, v_t2);
        v_s19 = _mm_sub_ps(v_s15, v_t2);
        // Output point 5: X(4)
        v_out4 = _mm_add_ps(v_s19, v_s23);
        curr_out = out_r + out_strides[4];
        STR_128_S(curr_out, v_out_stride, v_out4, is_contiguous_out);
        // Output point 7: X(6)
        v_out6 = _mm_sub_ps(v_s19, v_s23);
        curr_out = out_r + out_strides[6];
        STR_128_S(curr_out, v_out_stride, v_out6, is_contiguous_out);

        v_s12 = _mm_add_ps(v_s3, v_s7);
        v_t3 = _mm_mul_ps(v128_CRTM_10_1, v_s12);
        v_t1 = _mm_mul_ps(v128_CRTM_10_2, v_s13);
        v_s14 = _mm_add_ps(v_s1, v_t1);
        v_s16 = _mm_add_ps(v_s14, v_t3);
        v_s17 = _mm_sub_ps(v_s14, v_t3);
        // Output point 4: X(3)
        v_out3 = _mm_add_ps(v_s17, v_s22);
        curr_out = out_r + out_strides[3];
        STR_128_S(curr_out, v_out_stride, v_out3, is_contiguous_out);
        // Output point 8: X(7)
        v_out7 = _mm_sub_ps(v_s17, v_s22);
        curr_out = out_r + out_strides[7];
        STR_128_S(curr_out, v_out_stride, v_out7, is_contiguous_out);

        v_t4 = _mm_mul_ps(v128_CRTM_10_4, v_s4);
        v_t5 = _mm_mul_ps(v128_CRTM_10_5, v_s8);
        v_s20 = _mm_add_ps(v_t4, v_t5);
        // Output point 2: X(1)
        v_out1 = _mm_sub_ps(v_s16, v_s20);
        curr_out = out_r + out_strides[1];
        STR_128_S(curr_out, v_out_stride, v_out1, is_contiguous_out);
        // Output point 10: X(9)
        v_out9 = _mm_add_ps(v_s16, v_s20);
        curr_out = out_r + out_strides[9];
        STR_128_S(curr_out, v_out_stride, v_out9, is_contiguous_out);

        v_t6 = _mm_mul_ps(v128_CRTM_10_4, v_s9);
        v_t7 = _mm_mul_ps(v128_CRTM_10_5, v_s5);
        v_s21 = _mm_add_ps(v_t6, v_t7);
        // Output point 3: X(2)
        v_out2 = _mm_sub_ps(v_s18, v_s21);
        curr_out = out_r + out_strides[2];
        STR_128_S(curr_out, v_out_stride, v_out2, is_contiguous_out);
        // Output point 9: X(8)
        v_out8 = _mm_add_ps(v_s18, v_s21);
        curr_out = out_r + out_strides[8];
        STR_128_S(curr_out, v_out_stride, v_out8, is_contiguous_out);

        in_r = in_r + (v_in_dc_nyq_stride << 2);
        in_cp  = in_cp + (v_in_stride << 2);
        out_r = out_r + (v_out_stride << 2);
    }
    // tail cases
    if (remaining_sets & 2)
    {
        __m128 v_in0, v_in1, v_in2, v_in3, v_in4, v_in5, v_in6, v_in7, v_in8,
               v_in9;
        __m128 v_s0, v_s1, v_s2, v_s3, v_s4, v_s5, v_s6, v_s7, v_s8, v_s9,
               v_s10, v_s11, v_s12, v_s13, v_s14, v_s15, v_s16, v_s17, v_s18,
               v_s19, v_s20, v_s21, v_s22, v_s23;
        __m128 v_t0, v_t1, v_t2, v_t3, v_t4, v_t5, v_t6, v_t7, v_t8,
               v_t9, v_t10, v_t11, v_t12, v_t13;
        __m128 v_out0, v_out1, v_out2, v_out3, v_out4, v_out5, v_out6, v_out7,
               v_out8, v_out9;

        __m128 v128_CRTM_10_1 = _mm512_castps512_ps128(v_CRTM_10_1);
        __m128 v128_CRTM_10_2 = _mm512_castps512_ps128(v_CRTM_10_2);
        __m128 v128_CRTM_10_3 = _mm512_castps512_ps128(v_CRTM_10_3);
        __m128 v128_CRTM_10_4 = _mm512_castps512_ps128(v_CRTM_10_4);
        __m128 v128_CRTM_10_5 = _mm512_castps512_ps128(v_CRTM_10_5);

        curr_in = in_r;
        curr_out = out_r;

        // Input point 1: x(0)
        LDHR_128_S(curr_in, v_in_dc_nyq_stride, v_in0);
        // Input point 2: x(1) & Input point 3: x(2)
        curr_in = in_cp + in_strides[1];
        LDHRI_2x128_S(curr_in, v_in_stride, v_in1, v_in2);
        // Input point 4: x(3) & Input point 5: x(4)
        curr_in = in_cp + in_strides[3];
        LDHRI_2x128_S(curr_in, v_in_stride, v_in3, v_in4);
        // Input point 6: x(5) & Input point 7: x(6)
        curr_in = in_cp + in_strides[5];
        LDHRI_2x128_S(curr_in, v_in_stride, v_in5, v_in6);
        // Input point 8: x(7) & Input point 9: x(8)
        curr_in = in_cp + in_strides[7];
        LDHRI_2x128_S(curr_in, v_in_stride, v_in7, v_in8);
        // Input point 10: x(9) & Input point 11: x(10)
        curr_in = in_r + in_strides[9];
        LDHR_128_S(curr_in, v_in_dc_nyq_stride, v_in9);

        v_s0 = _mm_add_ps(v_in0, v_in9);
        v_s1 = _mm_sub_ps(v_in0, v_in9);
        v_s2 = _mm_add_ps(v_in1, v_in7);
        v_s3 = _mm_sub_ps(v_in1, v_in7);
        v_s4 = _mm_add_ps(v_in2, v_in8);
        v_s5 = _mm_sub_ps(v_in2, v_in8);
        v_t10 = _mm_mul_ps(v128_CRTM_10_4, v_s5);
        v_s8 = _mm_add_ps(v_in4, v_in6);

        v_t8 = _mm_mul_ps(v128_CRTM_10_4, v_s8);
        v_t9 = _mm_mul_ps(v128_CRTM_10_5, v_s4);
        v_s22 = _mm_sub_ps(v_t8, v_t9);
        v_s9 = _mm_sub_ps(v_in4, v_in6);
        v_t11 = _mm_mul_ps(v128_CRTM_10_5, v_s9);
        v_s23 = _mm_sub_ps(v_t11, v_t10);

        v_s6 = _mm_add_ps(v_in3, v_in5);
        v_s7 = _mm_sub_ps(v_in3, v_in5);
        v_s13 = _mm_sub_ps(v_s3, v_s7);
        v_t13 = _mm_mul_ps(v128_CRTM_10_3, v_s13);
        // Output point 6: X(5)
        v_out5 = _mm_sub_ps(v_s1, v_t13);
        curr_out = out_r + out_strides[5];
        STHR_128_S(curr_out, v_out_stride, v_out5);

        v_s10 = _mm_add_ps(v_s2, v_s6);
        v_t12 = _mm_mul_ps(v128_CRTM_10_3, v_s10);
        // Output point 1: X(0)
        v_out0 = _mm_add_ps(v_s0, v_t12);
        curr_out = out_r;
        STHR_128_S(curr_out, v_out_stride, v_out0);

        v_t0 = _mm_mul_ps(v128_CRTM_10_2, v_s10);
        v_s15 = _mm_sub_ps(v_s0, v_t0);
        v_s11 = _mm_sub_ps(v_s2, v_s6);
        v_t2 = _mm_mul_ps(v128_CRTM_10_1, v_s11);
        v_s18 = _mm_add_ps(v_s15, v_t2);
        v_s19 = _mm_sub_ps(v_s15, v_t2);
        // Output point 5: X(4)
        v_out4 = _mm_add_ps(v_s19, v_s23);
        curr_out = out_r + out_strides[4];
        STHR_128_S(curr_out, v_out_stride, v_out4);
        // Output point 7: X(6)
        v_out6 = _mm_sub_ps(v_s19, v_s23);
        curr_out = out_r + out_strides[6];
        STHR_128_S(curr_out, v_out_stride, v_out6);

        v_s12 = _mm_add_ps(v_s3, v_s7);
        v_t3 = _mm_mul_ps(v128_CRTM_10_1, v_s12);
        v_t1 = _mm_mul_ps(v128_CRTM_10_2, v_s13);
        v_s14 = _mm_add_ps(v_s1, v_t1);
        v_s16 = _mm_add_ps(v_s14, v_t3);
        v_s17 = _mm_sub_ps(v_s14, v_t3);
        // Output point 4: X(3)
        v_out3 = _mm_add_ps(v_s17, v_s22);
        curr_out = out_r + out_strides[3];
        STHR_128_S(curr_out, v_out_stride, v_out3);
        // Output point 8: X(7)
        v_out7 = _mm_sub_ps(v_s17, v_s22);
        curr_out = out_r + out_strides[7];
        STHR_128_S(curr_out, v_out_stride, v_out7);

        v_t4 = _mm_mul_ps(v128_CRTM_10_4, v_s4);
        v_t5 = _mm_mul_ps(v128_CRTM_10_5, v_s8);
        v_s20 = _mm_add_ps(v_t4, v_t5);
        // Output point 2: X(1)
        v_out1 = _mm_sub_ps(v_s16, v_s20);
        curr_out = out_r + out_strides[1];
        STHR_128_S(curr_out, v_out_stride, v_out1);
        // Output point 10: X(9)
        v_out9 = _mm_add_ps(v_s16, v_s20);
        curr_out = out_r + out_strides[9];
        STHR_128_S(curr_out, v_out_stride, v_out9);

        v_t6 = _mm_mul_ps(v128_CRTM_10_4, v_s9);
        v_t7 = _mm_mul_ps(v128_CRTM_10_5, v_s5);
        v_s21 = _mm_add_ps(v_t6, v_t7);
        // Output point 3: X(2)
        v_out2 = _mm_sub_ps(v_s18, v_s21);
        curr_out = out_r + out_strides[2];
        STHR_128_S(curr_out, v_out_stride, v_out2);
        // Output point 9: X(8)
        v_out8 = _mm_add_ps(v_s18, v_s21);
        curr_out = out_r + out_strides[8];
        STHR_128_S(curr_out, v_out_stride, v_out8);

        in_r = in_r + (v_in_dc_nyq_stride << 1);
        in_cp  = in_cp + (v_in_stride << 1);
        out_r = out_r + (v_out_stride << 1);
    }
    // tail cases
    if (remaining_sets & 1)
    {
        FFTZ_FLOAT v_in0, v_in1, v_in2, v_in3, v_in4, v_in5, v_in6, v_in7,
            v_in8, v_in9;
        FFTZ_FLOAT v_s0, v_s1, v_s2, v_s3, v_s4, v_s5, v_s6, v_s7, v_s8, v_s9,
              v_s10, v_s11, v_s12, v_s13, v_s14, v_s15, v_s16, v_s17, v_s18,
              v_s19, v_s20, v_s21, v_s22, v_s23;
        FFTZ_FLOAT v_t0, v_t1, v_t2, v_t3, v_t4, v_t5, v_t6, v_t7, v_t8, v_t9,
              v_t10, v_t11, v_t12, v_t13;

        curr_in = in_r;
        curr_out = out_r;
        // Input point 1: x(0)
        v_in0 = *in_r;
        // Input point 2: x(1)
        v_in1 = in_cp[in_strides[1]];
        // Input point 3: x(2)
        v_in2 = in_cp[in_strides[2]];
        // Input point 4: x(3)
        v_in3 = in_cp[in_strides[3]];
        // Input point 5: x(4)
        v_in4 = in_cp[in_strides[4]];
        // Input point 6: x(5)
        v_in5 = in_cp[in_strides[5]];
        // Input point 7: x(6)
        v_in6 = in_cp[in_strides[6]];
        // Input point 8: x(7)
        v_in7 = in_cp[in_strides[7]];
        // Input point 9: x(8)
        v_in8 = in_cp[in_strides[8]];
        // Input point 10: x(9)
        v_in9 = in_r[in_strides[9]];

        v_s0 = v_in0 + v_in9;
        v_s1 = v_in0 - v_in9;
        v_s2 = v_in1 + v_in7;
        v_s3 = v_in1 - v_in7;
        v_s4 = v_in2 + v_in8;
        v_s5 = v_in2 - v_in8;
        v_t10 = CRTM_10_4 * v_s5;
        v_s8 = v_in4 + v_in6;

        v_t8 = CRTM_10_4 * v_s8;
        v_t9 = CRTM_10_5 * v_s4;
        v_s22 = v_t8 - v_t9;
        v_s9 = v_in4 - v_in6;
        v_t11 = CRTM_10_5 * v_s9;
        v_s23 = v_t11 - v_t10;

        v_s6 = v_in3 + v_in5;
        v_s7 = v_in3 - v_in5;
        v_s13 = v_s3 - v_s7;
        v_t13 = CRTM_10_3 * v_s13;
        // Output point 6: X(5)
        out_r[out_strides[5]] = v_s1 - v_t13;

        v_s10 = v_s2 + v_s6;
        v_t12 = CRTM_10_3 * v_s10;
        // Output point 1: X(0)
        *out_r = v_s0 + v_t12;

        v_t0 = CRTM_10_2 * v_s10;
        v_s15 = v_s0 - v_t0;
        v_s11 = v_s2 - v_s6;
        v_t2 = CRTM_10_1 * v_s11;
        v_s18 = v_s15 + v_t2;
        v_s19 = v_s15 - v_t2;
        // Output point 5: X(4)
        out_r[out_strides[4]] = v_s19 + v_s23;
        // Output point 7: X(6)
        out_r[out_strides[6]] = v_s19 - v_s23;

        v_s12 = v_s3 + v_s7;
        v_t3 = CRTM_10_1 * v_s12;
        v_t1 = CRTM_10_2 * v_s13;
        v_s14 = v_s1 + v_t1;
        v_s16 = v_s14 + v_t3;
        v_s17 = v_s14 - v_t3;
        // Output point 4: X(3)
        out_r[out_strides[3]] = v_s17 + v_s22;
        // Output point 8: X(7)
        out_r[out_strides[7]] = v_s17 - v_s22;

        v_t4 = CRTM_10_4 * v_s4;
        v_t5 = CRTM_10_5 * v_s8;
        v_s20 = v_t4 + v_t5;
        // Output point 2: X(1)
        out_r[out_strides[1]] = v_s16 - v_s20;
        // Output point 10: X(9)
        out_r[out_strides[9]] = v_s16 + v_s20;

        v_t6 = CRTM_10_4 * v_s9;
        v_t7 = CRTM_10_5 * v_s5;
        v_s21 = v_t6 + v_t7;
        // Output point 3: X(2)
        out_r[out_strides[2]] = v_s18 - v_s21;
        // Output point 9: X(8)
        out_r[out_strides[8]] = v_s18 + v_s21;
    }
    AOCLFFTZ_LOG(DEBUG, global_logger_mode, "Exit");
}

static FFTZ_VOID r2hc_rfft10avx512_fp64_fwd(FFTZ_VOID *in_real,
                                            FFTZ_VOID *in_complex,
                                            FFTZ_VOID *out_real,
                                            FFTZ_VOID *out_complex,
                                            FFTZ_INTP n,
                                            aoclfftz_strides_t *strides,
                                            FFTZ_VOID *twd, FFTZ_UINT8 flag)
{
    AOCLFFTZ_LOG(DEBUG, global_logger_mode, "Enter");
    const FFTZ_DOUBLE CRTM_10_1 =
        0.55901699437494742410229341718281905886015458990288;
    const FFTZ_DOUBLE CRTM_10_2 =
        0.25000000000000000000000000000000000000000000000000;
    const FFTZ_DOUBLE CRTM_10_3 =
        0.58778525229247315738615484497912915412138427663885;
    const FFTZ_DOUBLE CRTM_10_4 =
        0.95105651629515357211643933337938214340569863400000;
    FFTZ_DOUBLE *in_r = (FFTZ_DOUBLE *)in_real;
    FFTZ_DOUBLE *out_r = (FFTZ_DOUBLE *)out_real;
    FFTZ_DOUBLE *out_cp = (FFTZ_DOUBLE *)out_complex;
#ifdef VOLATILE_STRIDE_ARRAY
    volatile FFTZ_INTP *in_strides = strides->in_strides;
    volatile FFTZ_INTP *out_strides = strides->out_strides;
#else
    FFTZ_INTP *in_strides  = strides->in_strides;
    FFTZ_INTP *out_strides = strides->out_strides;
#endif
    FFTZ_INTP v_in_stride  = strides->v_in_stride;
    FFTZ_INTP v_out_stride = strides->v_out_stride;
    FFTZ_UINT8 is_contiguous_in = (v_in_stride == 1);
    // Used for endpoints: DC and Nyquist.
    FFTZ_INTP v_out_dc_nyq_stride = strides->v_out_sym_stride;
    FFTZ_UINT8 is_contiguous_out_dc_nyq = (v_out_dc_nyq_stride == 1);

    FFTZ_INTP cnt;
    FFTZ_DOUBLE *curr_in, *curr_out;
    FFTZ_INTP N = n / NUM_SETS_REAL_512_D;
    FFTZ_INTP remaining_sets = n % NUM_SETS_REAL_512_D;

    __m512d v_CRTM_10_1 = _mm512_set1_pd(CRTM_10_1);
    __m512d v_CRTM_10_2 = _mm512_set1_pd(CRTM_10_2);
    __m512d v_CRTM_10_3 = _mm512_set1_pd(CRTM_10_3);
    __m512d v_CRTM_10_4 = _mm512_set1_pd(CRTM_10_4);

    for (cnt = 0; cnt < N; cnt++)
    {
        __m512d v_in0, v_in1, v_in2, v_in3, v_in4, v_in5, v_in6, v_in7, v_in8,
                v_in9;
        __m512d v_s0, v_s1, v_s2, v_s3, v_s4, v_s5, v_s6, v_s7, v_s8, v_s9,
                v_s10, v_s11, v_s12, v_s13, v_s14, v_s15, v_s16, v_s17, v_s18,
                v_s19, v_s20, v_s21, v_s22, v_s23;
        __m512d v_t0, v_t1, v_t2, v_t3, v_t4, v_t5, v_t6, v_t7, v_t8, v_t9,
                v_t10, v_t11;
        __m512d v_out0, v_out1, v_out2, v_out3, v_out4, v_out5, v_out6, v_out7,
                v_out8, v_out9;

        curr_in = in_r;
        curr_out = out_r;

        // Input point 1: x(0)
        LDR_512_D(curr_in, v_in_stride, v_in0, is_contiguous_in);
        // Input point 2: x(1)
        curr_in = in_r + in_strides[1];
        LDR_512_D(curr_in, v_in_stride, v_in1, is_contiguous_in);
        // Input point 3: x(2)
        curr_in = in_r + in_strides[2];
        LDR_512_D(curr_in, v_in_stride, v_in2, is_contiguous_in);
        // Input point 4: x(3)
        curr_in = in_r + in_strides[3];
        LDR_512_D(curr_in, v_in_stride, v_in3, is_contiguous_in);
        // Input point 5: x(4)
        curr_in = in_r + in_strides[4];
        LDR_512_D(curr_in, v_in_stride, v_in4, is_contiguous_in);
        // Input point 6: x(5)
        curr_in = in_r + in_strides[5];
        LDR_512_D(curr_in, v_in_stride, v_in5, is_contiguous_in);
        // Input point 7: x(6)
        curr_in = in_r + in_strides[6];
        LDR_512_D(curr_in, v_in_stride, v_in6, is_contiguous_in);
        // Input point 8: x(7)
        curr_in = in_r + in_strides[7];
        LDR_512_D(curr_in, v_in_stride, v_in7, is_contiguous_in);
        // Input point 9: x(8)
        curr_in = in_r + in_strides[8];
        LDR_512_D(curr_in, v_in_stride, v_in8, is_contiguous_in);
        // Input point 10: x(9)
        curr_in = in_r + in_strides[9];
        LDR_512_D(curr_in, v_in_stride, v_in9, is_contiguous_in);

        v_s0 = _mm512_add_pd(v_in0, v_in5);
        v_s1 = _mm512_sub_pd(v_in0, v_in5);
        v_s2 = _mm512_add_pd(v_in1, v_in9);
        v_s3 = _mm512_sub_pd(v_in1, v_in9);
        v_s4 = _mm512_add_pd(v_in2, v_in3);
        v_s5 = _mm512_sub_pd(v_in2, v_in3);
        v_s6 = _mm512_add_pd(v_in4, v_in6);
        v_s7 = _mm512_sub_pd(v_in4, v_in6);
        v_s8 = _mm512_add_pd(v_in7, v_in8);
        v_s9 = _mm512_sub_pd(v_in7, v_in8);

        v_s10 = _mm512_add_pd(v_s2, v_s6);
        v_s14 = _mm512_add_pd(v_s4, v_s8);
        v_s18 = _mm512_add_pd(v_s10, v_s14);
        // Output point 1: X(0)
        v_out0 = _mm512_add_pd(v_s0, v_s18);

        v_s11 = _mm512_sub_pd(v_s2, v_s6);
        v_s17 = _mm512_sub_pd(v_s5, v_s9);
        v_s20 = _mm512_add_pd(v_s11, v_s17);
        v_s21 = _mm512_sub_pd(v_s11, v_s17);
        // Output point 10: X(9)
        v_out9 = _mm512_sub_pd(v_s1, v_s21);

        v_t0 = _mm512_mul_pd(v_CRTM_10_2, v_s21);
        v_t1 = _mm512_mul_pd(v_CRTM_10_1, v_s20);
        v_s22 = _mm512_add_pd(v_t0, v_s1);
        // Output point 2: X(1)
        v_out1 = _mm512_add_pd(v_s22, v_t1);
        // Output point 6: X(5)
        v_out5 = _mm512_sub_pd(v_s22, v_t1);

        v_s12 = _mm512_add_pd(v_s3, v_s7);
        v_s15 = _mm512_sub_pd(v_s4, v_s8);
        v_t4 = _mm512_mul_pd(v_CRTM_10_3, v_s12);
        v_t10 = _mm512_mul_pd(v_CRTM_10_4, v_s15);
        // Output point 3: X(2)
        v_out2 = NEGATE_512_D(_mm512_add_pd(v_t4, v_t10));

        v_t6 = _mm512_mul_pd(v_CRTM_10_3, v_s15);
        v_t8 = _mm512_mul_pd(v_CRTM_10_4, v_s12);
        // Output point 7: X(6)
        v_out6 = _mm512_sub_pd(v_t6, v_t8);

        v_s13 = _mm512_sub_pd(v_s7, v_s3);
        v_s16 = _mm512_add_pd(v_s5, v_s9);
        v_t5 = _mm512_mul_pd(v_CRTM_10_3, v_s13);
        v_t11 = _mm512_mul_pd(v_CRTM_10_4, v_s16);
        // Output point 9: X(8)
        v_out8 = _mm512_add_pd(v_t11, v_t5);

        v_t7 = _mm512_mul_pd(v_CRTM_10_3, v_s16);
        v_t9 = _mm512_mul_pd(v_CRTM_10_4, v_s13);
        // Output point 5: X(4)
        v_out4 = _mm512_sub_pd(v_t9, v_t7);

        v_s19 = _mm512_sub_pd(v_s10, v_s14);
        v_t2 = _mm512_mul_pd(v_CRTM_10_2, v_s18);
        v_t3 = _mm512_mul_pd(v_CRTM_10_1, v_s19);
        v_s23 = _mm512_sub_pd(v_s0, v_t2);
        // Output point 4: X(3)
        v_out3 = _mm512_add_pd(v_s23, v_t3);
        // Output point 8: X(7)
        v_out7 = _mm512_sub_pd(v_s23, v_t3);

        STR_512_D(curr_out, v_out_dc_nyq_stride, v_out0, is_contiguous_out_dc_nyq);
        curr_out = out_cp + out_strides[1];
        STRI_2x512_D(curr_out, v_out_stride, v_out1, v_out2);
        curr_out = out_cp + out_strides[3];
        STRI_2x512_D(curr_out, v_out_stride, v_out3, v_out4);
        curr_out = out_cp + out_strides[5];
        STRI_2x512_D(curr_out, v_out_stride, v_out5, v_out6);
        curr_out = out_cp + out_strides[7];
        STRI_2x512_D(curr_out, v_out_stride, v_out7, v_out8);
        curr_out = out_r + out_strides[9];
        STR_512_D(curr_out, v_out_dc_nyq_stride, v_out9, is_contiguous_out_dc_nyq);

        in_r  = in_r + (v_in_stride << 3);
        out_cp = out_cp + (v_out_stride << 3);
        out_r = out_r + (v_out_dc_nyq_stride << 3);
    }
    // tail cases
    if (remaining_sets & NUM_SETS_REAL_256_D)
    {
        __m256d v_in0, v_in1, v_in2, v_in3, v_in4, v_in5, v_in6, v_in7, v_in8,
                v_in9;
        __m256d v_s0, v_s1, v_s2, v_s3, v_s4, v_s5, v_s6, v_s7, v_s8, v_s9,
                v_s10, v_s11, v_s12, v_s13, v_s14, v_s15, v_s16, v_s17, v_s18,
                v_s19, v_s20, v_s21, v_s22, v_s23;
        __m256d v_t0, v_t1, v_t2, v_t3, v_t4, v_t5, v_t6, v_t7, v_t8, v_t9,
                v_t10, v_t11;
        __m256d v_out0, v_out1, v_out2, v_out3, v_out4, v_out5, v_out6, v_out7,
                v_out8, v_out9;

        __m256d v256_CRTM_10_1 = _mm512_castpd512_pd256(v_CRTM_10_1);
        __m256d v256_CRTM_10_2 = _mm512_castpd512_pd256(v_CRTM_10_2);
        __m256d v256_CRTM_10_3 = _mm512_castpd512_pd256(v_CRTM_10_3);
        __m256d v256_CRTM_10_4 = _mm512_castpd512_pd256(v_CRTM_10_4);

        curr_in = in_r;
        curr_out = out_r;

        // Input point 1: x(0)
        LDR_256_D(curr_in, v_in_stride, v_in0, is_contiguous_in);
        // Input point 2: x(1)
        curr_in = in_r + in_strides[1];
        LDR_256_D(curr_in, v_in_stride, v_in1, is_contiguous_in);
        // Input point 3: x(2)
        curr_in = in_r + in_strides[2];
        LDR_256_D(curr_in, v_in_stride, v_in2, is_contiguous_in);
        // Input point 4: x(3)
        curr_in = in_r + in_strides[3];
        LDR_256_D(curr_in, v_in_stride, v_in3, is_contiguous_in);
        // Input point 5: x(4)
        curr_in = in_r + in_strides[4];
        LDR_256_D(curr_in, v_in_stride, v_in4, is_contiguous_in);
        // Input point 6: x(5)
        curr_in = in_r + in_strides[5];
        LDR_256_D(curr_in, v_in_stride, v_in5, is_contiguous_in);
        // Input point 7: x(6)
        curr_in = in_r + in_strides[6];
        LDR_256_D(curr_in, v_in_stride, v_in6, is_contiguous_in);
        // Input point 8: x(7)
        curr_in = in_r + in_strides[7];
        LDR_256_D(curr_in, v_in_stride, v_in7, is_contiguous_in);
        // Input point 9: x(8)
        curr_in = in_r + in_strides[8];
        LDR_256_D(curr_in, v_in_stride, v_in8, is_contiguous_in);
        // Input point 10: x(9)
        curr_in = in_r + in_strides[9];
        LDR_256_D(curr_in, v_in_stride, v_in9, is_contiguous_in);

        v_s0 = _mm256_add_pd(v_in0, v_in5);
        v_s1 = _mm256_sub_pd(v_in0, v_in5);
        v_s2 = _mm256_add_pd(v_in1, v_in9);
        v_s3 = _mm256_sub_pd(v_in1, v_in9);
        v_s4 = _mm256_add_pd(v_in2, v_in3);
        v_s5 = _mm256_sub_pd(v_in2, v_in3);
        v_s6 = _mm256_add_pd(v_in4, v_in6);
        v_s7 = _mm256_sub_pd(v_in4, v_in6);
        v_s8 = _mm256_add_pd(v_in7, v_in8);
        v_s9 = _mm256_sub_pd(v_in7, v_in8);

        v_s10 = _mm256_add_pd(v_s2, v_s6);
        v_s14 = _mm256_add_pd(v_s4, v_s8);
        v_s18 = _mm256_add_pd(v_s10, v_s14);
        // Output point 1: X(0)
        v_out0 = _mm256_add_pd(v_s0, v_s18);

        v_s11 = _mm256_sub_pd(v_s2, v_s6);
        v_s17 = _mm256_sub_pd(v_s5, v_s9);
        v_s20 = _mm256_add_pd(v_s11, v_s17);
        v_s21 = _mm256_sub_pd(v_s11, v_s17);
        // Output point 10: X(9)
        v_out9 = _mm256_sub_pd(v_s1, v_s21);

        v_t0 = _mm256_mul_pd(v256_CRTM_10_2, v_s21);
        v_t1 = _mm256_mul_pd(v256_CRTM_10_1, v_s20);
        v_s22 = _mm256_add_pd(v_t0, v_s1);
        // Output point 2: X(1)
        v_out1 = _mm256_add_pd(v_s22, v_t1);
        // Output point 6: X(5)
        v_out5 = _mm256_sub_pd(v_s22, v_t1);

        v_s12 = _mm256_add_pd(v_s3, v_s7);
        v_s15 = _mm256_sub_pd(v_s4, v_s8);
        v_t4 = _mm256_mul_pd(v256_CRTM_10_3, v_s12);
        v_t10 = _mm256_mul_pd(v256_CRTM_10_4, v_s15);
        // Output point 3: X(2)
        v_out2 = NEGATE_256_D(_mm256_add_pd(v_t4, v_t10));

        v_t6 = _mm256_mul_pd(v256_CRTM_10_3, v_s15);
        v_t8 = _mm256_mul_pd(v256_CRTM_10_4, v_s12);
        // Output point 7: X(6)
        v_out6 = _mm256_sub_pd(v_t6, v_t8);

        v_s13 = _mm256_sub_pd(v_s7, v_s3);
        v_s16 = _mm256_add_pd(v_s5, v_s9);
        v_t5 = _mm256_mul_pd(v256_CRTM_10_3, v_s13);
        v_t11 = _mm256_mul_pd(v256_CRTM_10_4, v_s16);
        // Output point 9: X(8)
        v_out8 = _mm256_add_pd(v_t11, v_t5);

        v_t7 = _mm256_mul_pd(v256_CRTM_10_3, v_s16);
        v_t9 = _mm256_mul_pd(v256_CRTM_10_4, v_s13);
        // Output point 5: X(4)
        v_out4 = _mm256_sub_pd(v_t9, v_t7);

        v_s19 = _mm256_sub_pd(v_s10, v_s14);
        v_t2 = _mm256_mul_pd(v256_CRTM_10_2, v_s18);
        v_t3 = _mm256_mul_pd(v256_CRTM_10_1, v_s19);
        v_s23 = _mm256_sub_pd(v_s0, v_t2);
        // Output point 4: X(3)
        v_out3 = _mm256_add_pd(v_s23, v_t3);
        // Output point 8: X(7)
        v_out7 = _mm256_sub_pd(v_s23, v_t3);

        STR_256_D(curr_out, v_out_dc_nyq_stride, v_out0, is_contiguous_out_dc_nyq);
        curr_out = out_cp + out_strides[1];
        STRI_2x256_D(curr_out, v_out_stride, v_out1, v_out2);
        curr_out = out_cp + out_strides[3];
        STRI_2x256_D(curr_out, v_out_stride, v_out3, v_out4);
        curr_out = out_cp + out_strides[5];
        STRI_2x256_D(curr_out, v_out_stride, v_out5, v_out6);
        curr_out = out_cp + out_strides[7];
        STRI_2x256_D(curr_out, v_out_stride, v_out7, v_out8);
        curr_out = out_r + out_strides[9];
        STR_256_D(curr_out, v_out_dc_nyq_stride, v_out9, is_contiguous_out_dc_nyq);

        in_r  = in_r + (v_in_stride << 2);
        out_cp = out_cp + (v_out_stride << 2);
        out_r = out_r + (v_out_dc_nyq_stride << 2);
    }
    // tail cases
    if (remaining_sets & NUM_SETS_REAL_128_D)
    {
        __m128d v_in0, v_in1, v_in2, v_in3, v_in4, v_in5, v_in6, v_in7, v_in8,
                v_in9;
        __m128d v_s0, v_s1, v_s2, v_s3, v_s4, v_s5, v_s6, v_s7, v_s8, v_s9,
                v_s10, v_s11, v_s12, v_s13, v_s14, v_s15, v_s16, v_s17, v_s18,
                v_s19, v_s20, v_s21, v_s22, v_s23;
        __m128d v_t0, v_t1, v_t2, v_t3, v_t4, v_t5, v_t6, v_t7, v_t8, v_t9,
                v_t10, v_t11;
        __m128d v_out0, v_out1, v_out2, v_out3, v_out4, v_out5, v_out6, v_out7,
                v_out8, v_out9;

        __m128d v128_CRTM_10_1 = _mm512_castpd512_pd128(v_CRTM_10_1);
        __m128d v128_CRTM_10_2 = _mm512_castpd512_pd128(v_CRTM_10_2);
        __m128d v128_CRTM_10_3 = _mm512_castpd512_pd128(v_CRTM_10_3);
        __m128d v128_CRTM_10_4 = _mm512_castpd512_pd128(v_CRTM_10_4);

        curr_in = in_r;
        curr_out = out_r;

        // Input point 1: x(0)
        LDR_128_D(curr_in, v_in_stride, v_in0, is_contiguous_in);
        // Input point 2: x(1)
        curr_in = in_r + in_strides[1];
        LDR_128_D(curr_in, v_in_stride, v_in1, is_contiguous_in);
        // Input point 3: x(2)
        curr_in = in_r + in_strides[2];
        LDR_128_D(curr_in, v_in_stride, v_in2, is_contiguous_in);
        // Input point 4: x(3)
        curr_in = in_r + in_strides[3];
        LDR_128_D(curr_in, v_in_stride, v_in3, is_contiguous_in);
        // Input point 5: x(4)
        curr_in = in_r + in_strides[4];
        LDR_128_D(curr_in, v_in_stride, v_in4, is_contiguous_in);
        // Input point 6: x(5)
        curr_in = in_r + in_strides[5];
        LDR_128_D(curr_in, v_in_stride, v_in5, is_contiguous_in);
        // Input point 7: x(6)
        curr_in = in_r + in_strides[6];
        LDR_128_D(curr_in, v_in_stride, v_in6, is_contiguous_in);
        // Input point 8: x(7)
        curr_in = in_r + in_strides[7];
        LDR_128_D(curr_in, v_in_stride, v_in7, is_contiguous_in);
        // Input point 9: x(8)
        curr_in = in_r + in_strides[8];
        LDR_128_D(curr_in, v_in_stride, v_in8, is_contiguous_in);
        // Input point 10: x(9)
        curr_in = in_r + in_strides[9];
        LDR_128_D(curr_in, v_in_stride, v_in9, is_contiguous_in);

        v_s0 = _mm_add_pd(v_in0, v_in5);
        v_s1 = _mm_sub_pd(v_in0, v_in5);
        v_s2 = _mm_add_pd(v_in1, v_in9);
        v_s3 = _mm_sub_pd(v_in1, v_in9);
        v_s4 = _mm_add_pd(v_in2, v_in3);
        v_s5 = _mm_sub_pd(v_in2, v_in3);
        v_s6 = _mm_add_pd(v_in4, v_in6);
        v_s7 = _mm_sub_pd(v_in4, v_in6);
        v_s8 = _mm_add_pd(v_in7, v_in8);
        v_s9 = _mm_sub_pd(v_in7, v_in8);

        v_s10 = _mm_add_pd(v_s2, v_s6);
        v_s14 = _mm_add_pd(v_s4, v_s8);
        v_s18 = _mm_add_pd(v_s10, v_s14);
        // Output point 1: X(0)
        v_out0 = _mm_add_pd(v_s0, v_s18);

        v_s11 = _mm_sub_pd(v_s2, v_s6);
        v_s17 = _mm_sub_pd(v_s5, v_s9);
        v_s20 = _mm_add_pd(v_s11, v_s17);
        v_s21 = _mm_sub_pd(v_s11, v_s17);
        // Output point 10: X(9)
        v_out9 = _mm_sub_pd(v_s1, v_s21);

        v_t0 = _mm_mul_pd(v128_CRTM_10_2, v_s21);
        v_t1 = _mm_mul_pd(v128_CRTM_10_1, v_s20);
        v_s22 = _mm_add_pd(v_t0, v_s1);
        // Output point 2: X(1)
        v_out1 = _mm_add_pd(v_s22, v_t1);
        // Output point 6: X(5)
        v_out5 = _mm_sub_pd(v_s22, v_t1);

        v_s12 = _mm_add_pd(v_s3, v_s7);
        v_s15 = _mm_sub_pd(v_s4, v_s8);
        v_t4 = _mm_mul_pd(v128_CRTM_10_3, v_s12);
        v_t10 = _mm_mul_pd(v128_CRTM_10_4, v_s15);
        // Output point 3: X(2)
        v_out2 = NEGATE_128_D(_mm_add_pd(v_t4, v_t10));

        v_t6 = _mm_mul_pd(v128_CRTM_10_3, v_s15);
        v_t8 = _mm_mul_pd(v128_CRTM_10_4, v_s12);
        // Output point 7: X(6)
        v_out6 = _mm_sub_pd(v_t6, v_t8);

        v_s13 = _mm_sub_pd(v_s7, v_s3);
        v_s16 = _mm_add_pd(v_s5, v_s9);
        v_t5 = _mm_mul_pd(v128_CRTM_10_3, v_s13);
        v_t11 = _mm_mul_pd(v128_CRTM_10_4, v_s16);
        // Output point 9: X(8)
        v_out8 = _mm_add_pd(v_t11, v_t5);

        v_t7 = _mm_mul_pd(v128_CRTM_10_3, v_s16);
        v_t9 = _mm_mul_pd(v128_CRTM_10_4, v_s13);
        // Output point 5: X(4)
        v_out4 = _mm_sub_pd(v_t9, v_t7);

        v_s19 = _mm_sub_pd(v_s10, v_s14);
        v_t2 = _mm_mul_pd(v128_CRTM_10_2, v_s18);
        v_t3 = _mm_mul_pd(v128_CRTM_10_1, v_s19);
        v_s23 = _mm_sub_pd(v_s0, v_t2);
        // Output point 4: X(3)
        v_out3 = _mm_add_pd(v_s23, v_t3);
        // Output point 8: X(7)
        v_out7 = _mm_sub_pd(v_s23, v_t3);

        STR_128_D(curr_out, v_out_dc_nyq_stride, v_out0, is_contiguous_out_dc_nyq);
        curr_out = out_cp + out_strides[1];
        STRI_2x128_D(curr_out, v_out_stride, v_out1, v_out2);
        curr_out = out_cp + out_strides[3];
        STRI_2x128_D(curr_out, v_out_stride, v_out3, v_out4);
        curr_out = out_cp + out_strides[5];
        STRI_2x128_D(curr_out, v_out_stride, v_out5, v_out6);
        curr_out = out_cp + out_strides[7];
        STRI_2x128_D(curr_out, v_out_stride, v_out7, v_out8);
        curr_out = out_r + out_strides[9];
        STR_128_D(curr_out, v_out_dc_nyq_stride, v_out9, is_contiguous_out_dc_nyq);

        in_r  = in_r + (v_in_stride << 1);
        out_cp = out_cp + (v_out_stride << 1);
        out_r = out_r + (v_out_dc_nyq_stride << 1);
    }
    // tail cases
    if (remaining_sets & 1)
    {
        FFTZ_DOUBLE v_in0, v_in1, v_in2, v_in3, v_in4, v_in5, v_in6, v_in7,
            v_in8, v_in9;
        FFTZ_DOUBLE v_s0, v_s1, v_s2, v_s3, v_s4, v_s5, v_s6, v_s7, v_s8, v_s9,
               v_s10, v_s11, v_s12, v_s13, v_s14, v_s15, v_s16, v_s17, v_s18,
               v_s19, v_s20, v_s21, v_s22, v_s23;
        FFTZ_DOUBLE v_t0, v_t1, v_t2, v_t3, v_t4, v_t5, v_t6, v_t7, v_t8, v_t9,
               v_t10, v_t11;

        curr_in = in_r;
        curr_out = out_r;
        // Input point 1: x(0)
        v_in0 = *in_r;
        // Input point 2: x(1)
        v_in1 = in_r[in_strides[1]];
        // Input point 3: x(2)
        v_in2 = in_r[in_strides[2]];
        // Input point 4: x(3)
        v_in3 = in_r[in_strides[3]];
        // Input point 5: x(4)
        v_in4 = in_r[in_strides[4]];
        // Input point 6: x(5)
        v_in5 = in_r[in_strides[5]];
        // Input point 7: x(6)
        v_in6 = in_r[in_strides[6]];
        // Input point 8: x(7)
        v_in7 = in_r[in_strides[7]];
        // Input point 9: x(8)
        v_in8 = in_r[in_strides[8]];
        // Input point 10: x(9)
        v_in9 = in_r[in_strides[9]];

        v_s0 = v_in0 + v_in5;
        v_s1 = v_in0 - v_in5;
        v_s2 = v_in1 + v_in9;
        v_s3 = v_in1 - v_in9;
        v_s4 = v_in2 + v_in3;
        v_s5 = v_in2 - v_in3;
        v_s6 = v_in4 + v_in6;
        v_s7 = v_in4 - v_in6;
        v_s8 = v_in7 + v_in8;
        v_s9 = v_in7 - v_in8;

        v_s10 = v_s2 + v_s6;
        v_s14 = v_s4 + v_s8;
        v_s18 = v_s10 + v_s14;
        // Output point 1: X(0)
        *out_r = v_s0 + v_s18;

        v_s11 = v_s2 - v_s6;
        v_s17 = v_s5 - v_s9;
        v_s20 = v_s11 + v_s17;
        v_s21 = v_s11 - v_s17;
        // Output point 10: X(9)
        out_r[out_strides[9]] = v_s1 - v_s21;

        v_t0 = CRTM_10_2 * v_s21;
        v_t1 = CRTM_10_1 * v_s20;
        v_s22 = v_t0 + v_s1;
        // Output point 2: X(1)
        out_cp[out_strides[1]] = v_s22 + v_t1;
        // Output point 6: X(5)
        out_cp[out_strides[5]] = v_s22 - v_t1;

        v_s12 = v_s3 + v_s7;
        v_s15 = v_s4 - v_s8;
        v_t4 = CRTM_10_3 * v_s12;
        v_t10 = CRTM_10_4 * v_s15;
        // Output point 3: X(2)
        out_cp[out_strides[2]] = -(v_t4 + v_t10);
        v_t6 = CRTM_10_3 * v_s15;
        v_t8 = CRTM_10_4 * v_s12;
        // Output point 7: X(6)
        out_cp[out_strides[6]] = v_t6 - v_t8;

        v_s13 = v_s7 - v_s3;
        v_s16 = v_s5 + v_s9;
        v_t5 = CRTM_10_3 * v_s13;
        v_t11 = CRTM_10_4 * v_s16;
        // Output point 9: X(8)
        out_cp[out_strides[8]] = v_t11 + v_t5;
        v_t7 = CRTM_10_3 * v_s16;
        v_t9 = CRTM_10_4 * v_s13;
        // Output point 5: X(4)
        out_cp[out_strides[4]] = v_t9 - v_t7;

        v_s19 = v_s10 - v_s14;
        v_t2 = CRTM_10_2 * v_s18;
        v_t3 = CRTM_10_1 * v_s19;
        v_s23 = v_s0 - v_t2;
        // Output point 4: X(3)
        out_cp[out_strides[3]] = v_s23 + v_t3;
        // Output point 8: X(7)
        out_cp[out_strides[7]] = v_s23 - v_t3;
    }
    AOCLFFTZ_LOG(DEBUG, global_logger_mode, "Exit");
}

static FFTZ_VOID r2hc_rfft10avx512_fp64_bwd(FFTZ_VOID *in_real,
                                            FFTZ_VOID *in_complex,
                                            FFTZ_VOID *out_real,
                                            FFTZ_VOID *out_complex,
                                            FFTZ_INTP n,
                                            aoclfftz_strides_t *strides,
                                            FFTZ_VOID *twd, FFTZ_UINT8 flag)
{
    AOCLFFTZ_LOG(DEBUG, global_logger_mode, "Enter");
    const FFTZ_DOUBLE CRTM_10_1 =
        1.118033988749894848204586834365638117720309180;
    const FFTZ_DOUBLE CRTM_10_2 =
        0.500000000000000000000000000000000000000000000;
    const FFTZ_DOUBLE CRTM_10_3 =
        2.000000000000000000000000000000000000000000000;
    const FFTZ_DOUBLE CRTM_10_4 =
        1.175570504584946258337411909278145537195304875;
    const FFTZ_DOUBLE CRTM_10_5 =
        1.902113032590307144232878666758764286811397268;
    FFTZ_DOUBLE *in_r = (FFTZ_DOUBLE *)in_real;
    FFTZ_DOUBLE *in_cp = (FFTZ_DOUBLE *)in_complex;
    FFTZ_DOUBLE *out_r = (FFTZ_DOUBLE *)out_real;
#ifdef VOLATILE_STRIDE_ARRAY
    volatile FFTZ_INTP *in_strides = strides->in_strides;
    volatile FFTZ_INTP *out_strides = strides->out_strides;
#else
    FFTZ_INTP *in_strides  = strides->in_strides;
    FFTZ_INTP *out_strides = strides->out_strides;
#endif
    FFTZ_INTP v_in_stride  = strides->v_in_stride;
    FFTZ_INTP v_out_stride = strides->v_out_stride;
    FFTZ_UINT8 is_contiguous_out = (v_out_stride == 1);
    // Used for endpoints: DC and Nyquist.
    FFTZ_INTP v_in_dc_nyq_stride = strides->v_in_sym_stride;
    FFTZ_UINT8 is_contiguous_in_dc_nyq = (v_in_dc_nyq_stride == 1);

    FFTZ_INTP cnt;
    FFTZ_DOUBLE *curr_in, *curr_out;
    FFTZ_INTP N = n / NUM_SETS_REAL_512_D;
    FFTZ_INTP remaining_sets = n % NUM_SETS_REAL_512_D;

    __m512d v_CRTM_10_1 = _mm512_set1_pd(CRTM_10_1);
    __m512d v_CRTM_10_2 = _mm512_set1_pd(CRTM_10_2);
    __m512d v_CRTM_10_3 = _mm512_set1_pd(CRTM_10_3);
    __m512d v_CRTM_10_4 = _mm512_set1_pd(CRTM_10_4);
    __m512d v_CRTM_10_5 = _mm512_set1_pd(CRTM_10_5);

    for (cnt = 0; cnt < N; cnt++)
    {
        __m512d v_in0, v_in1, v_in2, v_in3, v_in4, v_in5, v_in6, v_in7, v_in8,
                v_in9;
        __m512d v_s0, v_s1, v_s2, v_s3, v_s4, v_s5, v_s6, v_s7, v_s8, v_s9,
                v_s10, v_s11, v_s12, v_s13, v_s14, v_s15, v_s16, v_s17, v_s18,
                v_s19, v_s20, v_s21, v_s22, v_s23;
        __m512d v_t0, v_t1, v_t2, v_t3, v_t4, v_t5, v_t6, v_t7, v_t8,
                v_t9, v_t10, v_t11, v_t12, v_t13;
        __m512d v_out0, v_out1, v_out2, v_out3, v_out4, v_out5, v_out6, v_out7,
                v_out8, v_out9;

        curr_in = in_r;
        curr_out = out_r;

        // Input point 1: x(0)
        LDR_512_D(curr_in, v_in_dc_nyq_stride, v_in0, is_contiguous_in_dc_nyq);
        // Input point 2: x(1) & Input point 3: x(2)
        curr_in = in_cp + in_strides[1];
        LDRI_2x512_D(curr_in, v_in_stride, v_in1, v_in2);
        // Input point 4: x(3) & Input point 5: x(4)
        curr_in = in_cp + in_strides[3];
        LDRI_2x512_D(curr_in, v_in_stride, v_in3, v_in4);
        // Input point 6: x(5) & Input point 7: x(6)
        curr_in = in_cp + in_strides[5];
        LDRI_2x512_D(curr_in, v_in_stride, v_in5, v_in6);
        // Input point 8: x(7) & Input point 9: x(8)
        curr_in = in_cp + in_strides[7];
        LDRI_2x512_D(curr_in, v_in_stride, v_in7, v_in8);
        // Input point 10: x(9) & Input point 11: x(10)
        curr_in = in_r + in_strides[9];
        LDR_512_D(curr_in, v_in_dc_nyq_stride, v_in9, is_contiguous_in_dc_nyq);

        v_s0 = _mm512_add_pd(v_in0, v_in9);
        v_s1 = _mm512_sub_pd(v_in0, v_in9);
        v_s2 = _mm512_add_pd(v_in1, v_in7);
        v_s3 = _mm512_sub_pd(v_in1, v_in7);
        v_s4 = _mm512_add_pd(v_in2, v_in8);
        v_s5 = _mm512_sub_pd(v_in2, v_in8);
        v_t10 = _mm512_mul_pd(v_CRTM_10_4, v_s5);
        v_s8 = _mm512_add_pd(v_in4, v_in6);

        v_t8 = _mm512_mul_pd(v_CRTM_10_4, v_s8);
        v_t9 = _mm512_mul_pd(v_CRTM_10_5, v_s4);
        v_s22 = _mm512_sub_pd(v_t8, v_t9);
        v_s9 = _mm512_sub_pd(v_in4, v_in6);
        v_t11 = _mm512_mul_pd(v_CRTM_10_5, v_s9);
        v_s23 = _mm512_sub_pd(v_t11, v_t10);

        v_s6 = _mm512_add_pd(v_in3, v_in5);
        v_s7 = _mm512_sub_pd(v_in3, v_in5);
        v_s13 = _mm512_sub_pd(v_s3, v_s7);
        v_t13 = _mm512_mul_pd(v_CRTM_10_3, v_s13);
        // Output point 6: X(5)
        v_out5 = _mm512_sub_pd(v_s1, v_t13);
        curr_out = out_r + out_strides[5];
        STR_512_D(curr_out, v_out_stride, v_out5, is_contiguous_out);

        v_s10 = _mm512_add_pd(v_s2, v_s6);
        v_t12 = _mm512_mul_pd(v_CRTM_10_3, v_s10);
        // Output point 1: X(0)
        v_out0 = _mm512_add_pd(v_s0, v_t12);
        curr_out = out_r;
        STR_512_D(curr_out, v_out_stride, v_out0, is_contiguous_out);

        v_t0 = _mm512_mul_pd(v_CRTM_10_2, v_s10);
        v_s15 = _mm512_sub_pd(v_s0, v_t0);
        v_s11 = _mm512_sub_pd(v_s2, v_s6);
        v_t2 = _mm512_mul_pd(v_CRTM_10_1, v_s11);
        v_s18 = _mm512_add_pd(v_s15, v_t2);
        v_s19 = _mm512_sub_pd(v_s15, v_t2);
        // Output point 5: X(4)
        v_out4 = _mm512_add_pd(v_s19, v_s23);
        curr_out = out_r + out_strides[4];
        STR_512_D(curr_out, v_out_stride, v_out4, is_contiguous_out);
        // Output point 7: X(6)
        v_out6 = _mm512_sub_pd(v_s19, v_s23);
        curr_out = out_r + out_strides[6];
        STR_512_D(curr_out, v_out_stride, v_out6, is_contiguous_out);

        v_s12 = _mm512_add_pd(v_s3, v_s7);
        v_t3 = _mm512_mul_pd(v_CRTM_10_1, v_s12);
        v_t1 = _mm512_mul_pd(v_CRTM_10_2, v_s13);
        v_s14 = _mm512_add_pd(v_s1, v_t1);
        v_s16 = _mm512_add_pd(v_s14, v_t3);
        v_s17 = _mm512_sub_pd(v_s14, v_t3);
        // Output point 4: X(3)
        v_out3 = _mm512_add_pd(v_s17, v_s22);
        curr_out = out_r + out_strides[3];
        STR_512_D(curr_out, v_out_stride, v_out3, is_contiguous_out);
        // Output point 8: X(7)
        v_out7 = _mm512_sub_pd(v_s17, v_s22);
        curr_out = out_r + out_strides[7];
        STR_512_D(curr_out, v_out_stride, v_out7, is_contiguous_out);

        v_t4 = _mm512_mul_pd(v_CRTM_10_4, v_s4);
        v_t5 = _mm512_mul_pd(v_CRTM_10_5, v_s8);
        v_s20 = _mm512_add_pd(v_t4, v_t5);
        // Output point 2: X(1)
        v_out1 = _mm512_sub_pd(v_s16, v_s20);
        curr_out = out_r + out_strides[1];
        STR_512_D(curr_out, v_out_stride, v_out1, is_contiguous_out);
        // Output point 10: X(9)
        v_out9 = _mm512_add_pd(v_s16, v_s20);
        curr_out = out_r + out_strides[9];
        STR_512_D(curr_out, v_out_stride, v_out9, is_contiguous_out);

        v_t6 = _mm512_mul_pd(v_CRTM_10_4, v_s9);
        v_t7 = _mm512_mul_pd(v_CRTM_10_5, v_s5);
        v_s21 = _mm512_add_pd(v_t6, v_t7);
        // Output point 3: X(2)
        v_out2 = _mm512_sub_pd(v_s18, v_s21);
        curr_out = out_r + out_strides[2];
        STR_512_D(curr_out, v_out_stride, v_out2, is_contiguous_out);
        // Output point 9: X(8)
        v_out8 = _mm512_add_pd(v_s18, v_s21);
        curr_out = out_r + out_strides[8];
        STR_512_D(curr_out, v_out_stride, v_out8, is_contiguous_out);

        in_r = in_r + (v_in_dc_nyq_stride << 3);
        in_cp  = in_cp + (v_in_stride << 3);
        out_r = out_r + (v_out_stride << 3);
    }
    // tail cases
    if (remaining_sets & NUM_SETS_REAL_256_D)
    {
        __m256d v_in0, v_in1, v_in2, v_in3, v_in4, v_in5, v_in6, v_in7, v_in8,
                v_in9;
        __m256d v_s0, v_s1, v_s2, v_s3, v_s4, v_s5, v_s6, v_s7, v_s8, v_s9,
                v_s10, v_s11, v_s12, v_s13, v_s14, v_s15, v_s16, v_s17, v_s18,
                v_s19, v_s20, v_s21, v_s22, v_s23;
        __m256d v_t0, v_t1, v_t2, v_t3, v_t4, v_t5, v_t6, v_t7, v_t8,
                v_t9, v_t10, v_t11, v_t12, v_t13;
        __m256d v_out0, v_out1, v_out2, v_out3, v_out4, v_out5, v_out6, v_out7,
                v_out8, v_out9;

        __m256d v256_CRTM_10_1 = _mm512_castpd512_pd256(v_CRTM_10_1);
        __m256d v256_CRTM_10_2 = _mm512_castpd512_pd256(v_CRTM_10_2);
        __m256d v256_CRTM_10_3 = _mm512_castpd512_pd256(v_CRTM_10_3);
        __m256d v256_CRTM_10_4 = _mm512_castpd512_pd256(v_CRTM_10_4);
        __m256d v256_CRTM_10_5 = _mm512_castpd512_pd256(v_CRTM_10_5);

        curr_in = in_r;
        curr_out = out_r;

        // Input point 1: x(0)
        LDR_256_D(curr_in, v_in_dc_nyq_stride, v_in0, is_contiguous_in_dc_nyq);
        // Input point 2: x(1) & Input point 3: x(2)
        curr_in = in_cp + in_strides[1];
        LDRI_2x256_D(curr_in, v_in_stride, v_in1, v_in2);
        // Input point 4: x(3) & Input point 5: x(4)
        curr_in = in_cp + in_strides[3];
        LDRI_2x256_D(curr_in, v_in_stride, v_in3, v_in4);
        // Input point 6: x(5) & Input point 7: x(6)
        curr_in = in_cp + in_strides[5];
        LDRI_2x256_D(curr_in, v_in_stride, v_in5, v_in6);
        // Input point 8: x(7) & Input point 9: x(8)
        curr_in = in_cp + in_strides[7];
        LDRI_2x256_D(curr_in, v_in_stride, v_in7, v_in8);
        // Input point 10: x(9) & Input point 11: x(10)
        curr_in = in_r + in_strides[9];
        LDR_256_D(curr_in, v_in_dc_nyq_stride, v_in9, is_contiguous_in_dc_nyq);

        v_s0 = _mm256_add_pd(v_in0, v_in9);
        v_s1 = _mm256_sub_pd(v_in0, v_in9);
        v_s2 = _mm256_add_pd(v_in1, v_in7);
        v_s3 = _mm256_sub_pd(v_in1, v_in7);
        v_s4 = _mm256_add_pd(v_in2, v_in8);
        v_s5 = _mm256_sub_pd(v_in2, v_in8);
        v_t10 = _mm256_mul_pd(v256_CRTM_10_4, v_s5);
        v_s8 = _mm256_add_pd(v_in4, v_in6);

        v_t8 = _mm256_mul_pd(v256_CRTM_10_4, v_s8);
        v_t9 = _mm256_mul_pd(v256_CRTM_10_5, v_s4);
        v_s22 = _mm256_sub_pd(v_t8, v_t9);
        v_s9 = _mm256_sub_pd(v_in4, v_in6);
        v_t11 = _mm256_mul_pd(v256_CRTM_10_5, v_s9);
        v_s23 = _mm256_sub_pd(v_t11, v_t10);

        v_s6 = _mm256_add_pd(v_in3, v_in5);
        v_s7 = _mm256_sub_pd(v_in3, v_in5);
        v_s13 = _mm256_sub_pd(v_s3, v_s7);
        v_t13 = _mm256_mul_pd(v256_CRTM_10_3, v_s13);
        // Output point 6: X(5)
        v_out5 = _mm256_sub_pd(v_s1, v_t13);
        curr_out = out_r + out_strides[5];
        STR_256_D(curr_out, v_out_stride, v_out5, is_contiguous_out);

        v_s10 = _mm256_add_pd(v_s2, v_s6);
        v_t12 = _mm256_mul_pd(v256_CRTM_10_3, v_s10);
        // Output point 1: X(0)
        v_out0 = _mm256_add_pd(v_s0, v_t12);
        curr_out = out_r;
        STR_256_D(curr_out, v_out_stride, v_out0, is_contiguous_out);

        v_t0 = _mm256_mul_pd(v256_CRTM_10_2, v_s10);
        v_s15 = _mm256_sub_pd(v_s0, v_t0);
        v_s11 = _mm256_sub_pd(v_s2, v_s6);
        v_t2 = _mm256_mul_pd(v256_CRTM_10_1, v_s11);
        v_s18 = _mm256_add_pd(v_s15, v_t2);
        v_s19 = _mm256_sub_pd(v_s15, v_t2);
        // Output point 5: X(4)
        v_out4 = _mm256_add_pd(v_s19, v_s23);
        curr_out = out_r + out_strides[4];
        STR_256_D(curr_out, v_out_stride, v_out4, is_contiguous_out);
        // Output point 7: X(6)
        v_out6 = _mm256_sub_pd(v_s19, v_s23);
        curr_out = out_r + out_strides[6];
        STR_256_D(curr_out, v_out_stride, v_out6, is_contiguous_out);

        v_s12 = _mm256_add_pd(v_s3, v_s7);
        v_t3 = _mm256_mul_pd(v256_CRTM_10_1, v_s12);
        v_t1 = _mm256_mul_pd(v256_CRTM_10_2, v_s13);
        v_s14 = _mm256_add_pd(v_s1, v_t1);
        v_s16 = _mm256_add_pd(v_s14, v_t3);
        v_s17 = _mm256_sub_pd(v_s14, v_t3);
        // Output point 4: X(3)
        v_out3 = _mm256_add_pd(v_s17, v_s22);
        curr_out = out_r + out_strides[3];
        STR_256_D(curr_out, v_out_stride, v_out3, is_contiguous_out);
        // Output point 8: X(7)
        v_out7 = _mm256_sub_pd(v_s17, v_s22);
        curr_out = out_r + out_strides[7];
        STR_256_D(curr_out, v_out_stride, v_out7, is_contiguous_out);

        v_t4 = _mm256_mul_pd(v256_CRTM_10_4, v_s4);
        v_t5 = _mm256_mul_pd(v256_CRTM_10_5, v_s8);
        v_s20 = _mm256_add_pd(v_t4, v_t5);
        // Output point 2: X(1)
        v_out1 = _mm256_sub_pd(v_s16, v_s20);
        curr_out = out_r + out_strides[1];
        STR_256_D(curr_out, v_out_stride, v_out1, is_contiguous_out);
        // Output point 10: X(9)
        v_out9 = _mm256_add_pd(v_s16, v_s20);
        curr_out = out_r + out_strides[9];
        STR_256_D(curr_out, v_out_stride, v_out9, is_contiguous_out);

        v_t6 = _mm256_mul_pd(v256_CRTM_10_4, v_s9);
        v_t7 = _mm256_mul_pd(v256_CRTM_10_5, v_s5);
        v_s21 = _mm256_add_pd(v_t6, v_t7);
        // Output point 3: X(2)
        v_out2 = _mm256_sub_pd(v_s18, v_s21);
        curr_out = out_r + out_strides[2];
        STR_256_D(curr_out, v_out_stride, v_out2, is_contiguous_out);
        // Output point 9: X(8)
        v_out8 = _mm256_add_pd(v_s18, v_s21);
        curr_out = out_r + out_strides[8];
        STR_256_D(curr_out, v_out_stride, v_out8, is_contiguous_out);

        in_r = in_r + (v_in_dc_nyq_stride << 2);
        in_cp  = in_cp + (v_in_stride << 2);
        out_r = out_r + (v_out_stride << 2);
    }
    // tail cases
    if (remaining_sets & NUM_SETS_REAL_128_D)
    {
        __m128d v_in0, v_in1, v_in2, v_in3, v_in4, v_in5, v_in6, v_in7, v_in8,
                v_in9;
        __m128d v_s0, v_s1, v_s2, v_s3, v_s4, v_s5, v_s6, v_s7, v_s8, v_s9,
                v_s10, v_s11, v_s12, v_s13, v_s14, v_s15, v_s16, v_s17, v_s18,
                v_s19, v_s20, v_s21, v_s22, v_s23;
        __m128d v_t0, v_t1, v_t2, v_t3, v_t4, v_t5, v_t6, v_t7, v_t8,
                v_t9, v_t10, v_t11, v_t12, v_t13;
        __m128d v_out0, v_out1, v_out2, v_out3, v_out4, v_out5, v_out6, v_out7,
                v_out8, v_out9;

        __m128d v128_CRTM_10_1 = _mm512_castpd512_pd128(v_CRTM_10_1);
        __m128d v128_CRTM_10_2 = _mm512_castpd512_pd128(v_CRTM_10_2);
        __m128d v128_CRTM_10_3 = _mm512_castpd512_pd128(v_CRTM_10_3);
        __m128d v128_CRTM_10_4 = _mm512_castpd512_pd128(v_CRTM_10_4);
        __m128d v128_CRTM_10_5 = _mm512_castpd512_pd128(v_CRTM_10_5);

        curr_in = in_r;
        curr_out = out_r;

        // Input point 1: x(0)
        LDR_128_D(curr_in, v_in_dc_nyq_stride, v_in0, is_contiguous_in_dc_nyq);
        // Input point 2: x(1) & Input point 3: x(2)
        curr_in = in_cp + in_strides[1];
        LDRI_2x128_D(curr_in, v_in_stride, v_in1, v_in2);
        // Input point 4: x(3) & Input point 5: x(4)
        curr_in = in_cp + in_strides[3];
        LDRI_2x128_D(curr_in, v_in_stride, v_in3, v_in4);
        // Input point 6: x(5) & Input point 7: x(6)
        curr_in = in_cp + in_strides[5];
        LDRI_2x128_D(curr_in, v_in_stride, v_in5, v_in6);
        // Input point 8: x(7) & Input point 9: x(8)
        curr_in = in_cp + in_strides[7];
        LDRI_2x128_D(curr_in, v_in_stride, v_in7, v_in8);
        // Input point 10: x(9) & Input point 11: x(10)
        curr_in = in_r + in_strides[9];
        LDR_128_D(curr_in, v_in_dc_nyq_stride, v_in9, is_contiguous_in_dc_nyq);

        v_s0 = _mm_add_pd(v_in0, v_in9);
        v_s1 = _mm_sub_pd(v_in0, v_in9);
        v_s2 = _mm_add_pd(v_in1, v_in7);
        v_s3 = _mm_sub_pd(v_in1, v_in7);
        v_s4 = _mm_add_pd(v_in2, v_in8);
        v_s5 = _mm_sub_pd(v_in2, v_in8);
        v_t10 = _mm_mul_pd(v128_CRTM_10_4, v_s5);
        v_s8 = _mm_add_pd(v_in4, v_in6);

        v_t8 = _mm_mul_pd(v128_CRTM_10_4, v_s8);
        v_t9 = _mm_mul_pd(v128_CRTM_10_5, v_s4);
        v_s22 = _mm_sub_pd(v_t8, v_t9);
        v_s9 = _mm_sub_pd(v_in4, v_in6);
        v_t11 = _mm_mul_pd(v128_CRTM_10_5, v_s9);
        v_s23 = _mm_sub_pd(v_t11, v_t10);

        v_s6 = _mm_add_pd(v_in3, v_in5);
        v_s7 = _mm_sub_pd(v_in3, v_in5);
        v_s13 = _mm_sub_pd(v_s3, v_s7);
        v_t13 = _mm_mul_pd(v128_CRTM_10_3, v_s13);
        // Output point 6: X(5)
        v_out5 = _mm_sub_pd(v_s1, v_t13);
        curr_out = out_r + out_strides[5];
        STR_128_D(curr_out, v_out_stride, v_out5, is_contiguous_out);

        v_s10 = _mm_add_pd(v_s2, v_s6);
        v_t12 = _mm_mul_pd(v128_CRTM_10_3, v_s10);
        // Output point 1: X(0)
        v_out0 = _mm_add_pd(v_s0, v_t12);
        curr_out = out_r;
        STR_128_D(curr_out, v_out_stride, v_out0, is_contiguous_out);

        v_t0 = _mm_mul_pd(v128_CRTM_10_2, v_s10);
        v_s15 = _mm_sub_pd(v_s0, v_t0);
        v_s11 = _mm_sub_pd(v_s2, v_s6);
        v_t2 = _mm_mul_pd(v128_CRTM_10_1, v_s11);
        v_s18 = _mm_add_pd(v_s15, v_t2);
        v_s19 = _mm_sub_pd(v_s15, v_t2);
        // Output point 5: X(4)
        v_out4 = _mm_add_pd(v_s19, v_s23);
        curr_out = out_r + out_strides[4];
        STR_128_D(curr_out, v_out_stride, v_out4, is_contiguous_out);
        // Output point 7: X(6)
        v_out6 = _mm_sub_pd(v_s19, v_s23);
        curr_out = out_r + out_strides[6];
        STR_128_D(curr_out, v_out_stride, v_out6, is_contiguous_out);

        v_s12 = _mm_add_pd(v_s3, v_s7);
        v_t3 = _mm_mul_pd(v128_CRTM_10_1, v_s12);
        v_t1 = _mm_mul_pd(v128_CRTM_10_2, v_s13);
        v_s14 = _mm_add_pd(v_s1, v_t1);
        v_s16 = _mm_add_pd(v_s14, v_t3);
        v_s17 = _mm_sub_pd(v_s14, v_t3);
        // Output point 4: X(3)
        v_out3 = _mm_add_pd(v_s17, v_s22);
        curr_out = out_r + out_strides[3];
        STR_128_D(curr_out, v_out_stride, v_out3, is_contiguous_out);
        // Output point 8: X(7)
        v_out7 = _mm_sub_pd(v_s17, v_s22);
        curr_out = out_r + out_strides[7];
        STR_128_D(curr_out, v_out_stride, v_out7, is_contiguous_out);

        v_t4 = _mm_mul_pd(v128_CRTM_10_4, v_s4);
        v_t5 = _mm_mul_pd(v128_CRTM_10_5, v_s8);
        v_s20 = _mm_add_pd(v_t4, v_t5);
        // Output point 2: X(1)
        v_out1 = _mm_sub_pd(v_s16, v_s20);
        curr_out = out_r + out_strides[1];
        STR_128_D(curr_out, v_out_stride, v_out1, is_contiguous_out);
        // Output point 10: X(9)
        v_out9 = _mm_add_pd(v_s16, v_s20);
        curr_out = out_r + out_strides[9];
        STR_128_D(curr_out, v_out_stride, v_out9, is_contiguous_out);

        v_t6 = _mm_mul_pd(v128_CRTM_10_4, v_s9);
        v_t7 = _mm_mul_pd(v128_CRTM_10_5, v_s5);
        v_s21 = _mm_add_pd(v_t6, v_t7);
        // Output point 3: X(2)
        v_out2 = _mm_sub_pd(v_s18, v_s21);
        curr_out = out_r + out_strides[2];
        STR_128_D(curr_out, v_out_stride, v_out2, is_contiguous_out);
        // Output point 9: X(8)
        v_out8 = _mm_add_pd(v_s18, v_s21);
        curr_out = out_r + out_strides[8];
        STR_128_D(curr_out, v_out_stride, v_out8, is_contiguous_out);

        in_r = in_r + (v_in_dc_nyq_stride << 1);
        in_cp  = in_cp + (v_in_stride << 1);
        out_r = out_r + (v_out_stride << 1);
    }
    // tail cases
    if (remaining_sets & 1)
    {
        FFTZ_DOUBLE v_in0, v_in1, v_in2, v_in3, v_in4, v_in5, v_in6, v_in7,
            v_in8, v_in9;
        FFTZ_DOUBLE v_s0, v_s1, v_s2, v_s3, v_s4, v_s5, v_s6, v_s7, v_s8, v_s9,
               v_s10, v_s11, v_s12, v_s13, v_s14, v_s15, v_s16, v_s17, v_s18,
               v_s19, v_s20, v_s21, v_s22, v_s23;
        FFTZ_DOUBLE v_t0, v_t1, v_t2, v_t3, v_t4, v_t5, v_t6, v_t7, v_t8, v_t9,
               v_t10, v_t11, v_t12, v_t13;

        curr_in = in_r;
        curr_out = out_r;
        // Input point 1: x(0)
        v_in0 = *in_r;
        // Input point 2: x(1)
        v_in1 = in_cp[in_strides[1]];
        // Input point 3: x(2)
        v_in2 = in_cp[in_strides[2]];
        // Input point 4: x(3)
        v_in3 = in_cp[in_strides[3]];
        // Input point 5: x(4)
        v_in4 = in_cp[in_strides[4]];
        // Input point 6: x(5)
        v_in5 = in_cp[in_strides[5]];
        // Input point 7: x(6)
        v_in6 = in_cp[in_strides[6]];
        // Input point 8: x(7)
        v_in7 = in_cp[in_strides[7]];
        // Input point 9: x(8)
        v_in8 = in_cp[in_strides[8]];
        // Input point 10: x(9)
        v_in9 = in_r[in_strides[9]];

        v_s0 = v_in0 + v_in9;
        v_s1 = v_in0 - v_in9;
        v_s2 = v_in1 + v_in7;
        v_s3 = v_in1 - v_in7;
        v_s4 = v_in2 + v_in8;
        v_s5 = v_in2 - v_in8;
        v_t10 = CRTM_10_4 * v_s5;
        v_s8 = v_in4 + v_in6;

        v_t8 = CRTM_10_4 * v_s8;
        v_t9 = CRTM_10_5 * v_s4;
        v_s22 = v_t8 - v_t9;
        v_s9 = v_in4 - v_in6;
        v_t11 = CRTM_10_5 * v_s9;
        v_s23 = v_t11 - v_t10;

        v_s6 = v_in3 + v_in5;
        v_s7 = v_in3 - v_in5;
        v_s13 = v_s3 - v_s7;
        v_t13 = CRTM_10_3 * v_s13;
        // Output point 6: X(5)
        out_r[out_strides[5]] = v_s1 - v_t13;

        v_s10 = v_s2 + v_s6;
        v_t12 = CRTM_10_3 * v_s10;
        // Output point 1: X(0)
        *out_r = v_s0 + v_t12;

        v_t0 = CRTM_10_2 * v_s10;
        v_s15 = v_s0 - v_t0;
        v_s11 = v_s2 - v_s6;
        v_t2 = CRTM_10_1 * v_s11;
        v_s18 = v_s15 + v_t2;
        v_s19 = v_s15 - v_t2;
        // Output point 5: X(4)
        out_r[out_strides[4]] = v_s19 + v_s23;
        // Output point 7: X(6)
        out_r[out_strides[6]] = v_s19 - v_s23;

        v_s12 = v_s3 + v_s7;
        v_t3 = CRTM_10_1 * v_s12;
        v_t1 = CRTM_10_2 * v_s13;
        v_s14 = v_s1 + v_t1;
        v_s16 = v_s14 + v_t3;
        v_s17 = v_s14 - v_t3;
        // Output point 4: X(3)
        out_r[out_strides[3]] = v_s17 + v_s22;
        // Output point 8: X(7)
        out_r[out_strides[7]] = v_s17 - v_s22;

        v_t4 = CRTM_10_4 * v_s4;
        v_t5 = CRTM_10_5 * v_s8;
        v_s20 = v_t4 + v_t5;
        // Output point 2: X(1)
        out_r[out_strides[1]] = v_s16 - v_s20;
        // Output point 10: X(9)
        out_r[out_strides[9]] = v_s16 + v_s20;

        v_t6 = CRTM_10_4 * v_s9;
        v_t7 = CRTM_10_5 * v_s5;
        v_s21 = v_t6 + v_t7;
        // Output point 3: X(2)
        out_r[out_strides[2]] = v_s18 - v_s21;
        // Output point 9: X(8)
        out_r[out_strides[8]] = v_s18 + v_s21;
    }
    AOCLFFTZ_LOG(DEBUG, global_logger_mode, "Exit");
}

kfft_ register_kernel_r2hc_rfft10avx512(FFTZ_UINT8 precision,
                                        FFTZ_UINT8 direction)
{
    if (direction == FORWARD_FFT_DIR)
    {
        if (precision == DT_FLOAT)
        {
            return r2hc_rfft10avx512_fp32_fwd;
        }
        else if (precision == DT_DOUBLE)
        {
            return r2hc_rfft10avx512_fp64_fwd;
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
            return r2hc_rfft10avx512_fp32_bwd;
        }
        else if (precision == DT_DOUBLE)
        {
            return r2hc_rfft10avx512_fp64_bwd;
        }
        else
        {
            return NULL;
        }
    }
}
