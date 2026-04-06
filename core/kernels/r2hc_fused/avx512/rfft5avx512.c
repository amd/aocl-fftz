// Copyright Advanced Micro Devices, Inc.
// SPDX-License-Identifier: BSD-3-Clause

/** @file rfft5avx512.c
 *
 *  @brief Radix-5 r2hc_fused Real-FFT kernel with with AVX-512 operations using
 *  x86 SIMD intrinsics.
 *
 *  This file contains the DIT radix-5 real-to-halfcomplex fused of two
 *  different implementations (Standard DFT and Shifted DFT that differs in DFT
 *  weight matrix) using x86 SIMD intrinsics for single-precision and
 *  double-precision inputs
 *
 *  @author Jeevanantham N
 */

#include "core/kernels/kernel.h"
#include "core/kernels/simd_includes/r2hc_simd_avx512.h"

static const ops_cycles_t ops_cnt[NUM_PRECISIONS][NUM_FFT_DIRS] =
                                                {{{0, 12, 24, 256, 152, 61},
                                                  {0, 14, 24, 256, 184, 61}},
                                                 {{0, 12, 24, 128, 8, 61},
                                                  {0, 14, 24, 128, 8, 61}}};

ops_cycles_t get_ops_cnt_r2hcf_rfft5avx512(UINT8 precision, UINT8 direction)
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

static VOID r2hcf_rfft5avx512_fp32_fwd(VOID *in_real, VOID *in_imag,
                                       VOID *out_real, VOID *out_imag, INTP n,
                                       aoclfftz_strides_t *strides, VOID *twd, UINT8 flag)
{
    AOCLFFTZ_LOG(DEBUG, global_logger_mode, "Enter");
    const FLOAT CRTM_5_1 = 0.559016994374947424102293417182819058860154590f;
    const FLOAT CRTM_5_2 = 0.951056516295153572116439333379382143405698632f;
    const FLOAT CRTM_5_3 = 0.587785252292473129168705954639072768597652438f;
    const FLOAT CRTM_5_4 = 0.250000000000000000000000000000000000000000000f;

    FLOAT *in = (FLOAT *)in_real;
    FLOAT *out = (FLOAT *)out_real;
#ifdef VOLATILE_STRIDE_ARRAY
    volatile INTP *in_strides = strides->in_strides;
    volatile INTP *out_strides = strides->out_strides;
#else
    INTP *in_strides = strides->in_strides;
    INTP *out_strides = strides->out_strides;
#endif
    INTP v_in_stride = strides->v_in_stride;
    INTP v_out_stride = strides->v_out_stride;

    INTP cnt;
    FLOAT *curr_in, *curr_out;
    INTP N = n / NUM_SETS_REAL_512_S;
    INTP remaining_sets = n % NUM_SETS_REAL_512_S;

    __m512 v_CRTM_5_1 = _mm512_set1_ps(CRTM_5_1);
    __m512 v_CRTM_5_2 = _mm512_set1_ps(CRTM_5_2);
    __m512 v_CRTM_5_3 = _mm512_set1_ps(CRTM_5_3);
    __m512 v_CRTM_5_4 = _mm512_set1_ps(CRTM_5_4);

    for (cnt = 0; cnt < N; cnt++)
    {
        /* Standard DFT */
        __m512 av_in0, av_in1, av_in2, av_in3, av_in4;
        __m512 av_s1, av_s2, av_s3, av_s4, av_s5, av_s6, av_s7;
        __m512 av_t1, av_t2, av_t3, av_t4, av_t5, av_t6;
        __m512 v_out0, v_out1, v_out2, v_out3, v_out4, v_out5, v_out6, v_out7,
               v_out8, v_out9;

        curr_in = in;
        curr_out = out;

        // Input point 1: x(0)
        LDR_512_S(curr_in, v_in_stride, av_in0);
        // Input point 3: x(2)
        curr_in = in + in_strides[2];
        LDR_512_S(curr_in, v_in_stride, av_in1);
        // Input point 5: x(4)
        curr_in = in + in_strides[4];
        LDR_512_S(curr_in, v_in_stride, av_in2);
        // Input point 7: x(6)
        curr_in = in + in_strides[6];
        LDR_512_S(curr_in, v_in_stride, av_in3);
        // Input point 9: x(8)
        curr_in = in + in_strides[8];
        LDR_512_S(curr_in, v_in_stride, av_in4);

        av_s1 = _mm512_add_ps(av_in1, av_in4);
        av_s2 = _mm512_sub_ps(av_in4, av_in1);
        av_s3 = _mm512_add_ps(av_in2, av_in3);
        av_s4 = _mm512_sub_ps(av_in2, av_in3);
        av_s5 = _mm512_add_ps(av_s1, av_s3);

        av_t1 = _mm512_mul_ps(v_CRTM_5_4, av_s5);
        av_s6 = _mm512_sub_ps(av_s1, av_s3);

        av_s7 = _mm512_sub_ps(av_in0, av_t1);
        av_t2 = _mm512_mul_ps(v_CRTM_5_1, av_s6);

        av_t3 = _mm512_mul_ps(v_CRTM_5_3, av_s4);
        av_t4 = _mm512_mul_ps(v_CRTM_5_2, av_s2);
        av_t5 = _mm512_mul_ps(v_CRTM_5_2, av_s4);
        av_t6 = _mm512_mul_ps(v_CRTM_5_3, av_s2);

        // Output point 1: X(0)
        v_out0 = _mm512_add_ps(av_in0, av_s5);
        STR_512_S(curr_out, v_out_stride, v_out0);
        // Output point 4: X(3) & Output point 5: X(4)
        curr_out = out + out_strides[3];
        v_out3 = _mm512_add_ps(av_s7, av_t2);
        v_out4 = _mm512_sub_ps(av_t4, av_t3);
        STRI_2x512_S(curr_out, v_out_stride, v_out3, v_out4);
        // Output point 8: X(7) & Output point 9: X(8)
        curr_out = out + out_strides[7];
        v_out7 = _mm512_sub_ps(av_s7, av_t2);
        v_out8 = _mm512_add_ps(av_t5, av_t6);
        STRI_2x512_S(curr_out, v_out_stride, v_out7, v_out8);

        /* Shifted DFT */
        __m512 bv_in0, bv_in1, bv_in2, bv_in3, bv_in4;
        __m512 bv_s1, bv_s2, bv_s3, bv_s4, bv_s5, bv_s6, bv_s7;
        __m512 bv_t1, bv_t2, bv_t3, bv_t4, bv_t5, bv_t6;

        // Input point 2: x(1)
        curr_in = in + in_strides[1];
        LDR_512_S(curr_in, v_in_stride, bv_in0);
        // Input point 4: x(3)
        curr_in = in + in_strides[3];
        LDR_512_S(curr_in, v_in_stride, bv_in1);
        // Input point 6: x(5)
        curr_in = in + in_strides[5];
        LDR_512_S(curr_in, v_in_stride, bv_in2);
        // Input point 8: x(7)
        curr_in = in + in_strides[7];
        LDR_512_S(curr_in, v_in_stride, bv_in3);
        // Input point 10: x(9)
        curr_in = in + in_strides[9];
        LDR_512_S(curr_in, v_in_stride, bv_in4);

        bv_s1 = _mm512_add_ps(bv_in1, bv_in4);
        bv_s2 = _mm512_sub_ps(bv_in1, bv_in4);
        bv_s3 = _mm512_add_ps(bv_in2, bv_in3);
        bv_s4 = _mm512_sub_ps(bv_in2, bv_in3);
        bv_s5 = _mm512_sub_ps(bv_s2, bv_s4);

        bv_s6 = _mm512_add_ps(bv_s2, bv_s4);
        bv_t1 = _mm512_mul_ps(v_CRTM_5_4, bv_s5);
        bv_s7 = _mm512_add_ps(bv_in0, bv_t1);
        bv_t2 = _mm512_mul_ps(v_CRTM_5_1, bv_s6);

        bv_t3 = _mm512_mul_ps(v_CRTM_5_2, bv_s3);
        bv_t4 = _mm512_mul_ps(v_CRTM_5_3, bv_s1);
        bv_t5 = _mm512_mul_ps(v_CRTM_5_3, bv_s3);
        bv_t6 = _mm512_mul_ps(v_CRTM_5_2, bv_s1);

        // Output point 2: X(1) & Output point 3: X(2)
        curr_out = out + out_strides[1];
        v_out1 = _mm512_add_ps(bv_s7, bv_t2);
        v_out2 = _mm512_sub_ps(NEGATE_512_S(bv_t4), bv_t3);
        STRI_2x512_S(curr_out, v_out_stride, v_out1, v_out2);
        // Output point 6: X(5) & Output point 7: X(6)
        curr_out = out + out_strides[5];
        v_out5 = _mm512_sub_ps(bv_s7, bv_t2);
        v_out6 = _mm512_sub_ps(bv_t5, bv_t6);
        STRI_2x512_S(curr_out, v_out_stride, v_out5, v_out6);
        // Output point 10: X(9)
        curr_out = out + out_strides[9];
        v_out9 = _mm512_sub_ps(bv_in0, bv_s5);
        STR_512_S(curr_out, v_out_stride, v_out9);

        in = in + (v_in_stride << 4);
        out = out + (v_out_stride << 4);
    }
    // tail cases
    if (remaining_sets & NUM_SETS_REAL_256_S)
    {
        /* Standard DFT */
        __m256 av_in0, av_in1, av_in2, av_in3, av_in4;
        __m256 av_s1, av_s2, av_s3, av_s4, av_s5, av_s6, av_s7;
        __m256 av_t1, av_t2, av_t3, av_t4, av_t5, av_t6;
        __m256 v_out0, v_out1, v_out2, v_out3, v_out4, v_out5, v_out6, v_out7,
               v_out8, v_out9;

        curr_in = in;
        curr_out = out;

        __m256 v256_CRTM_5_1 = _mm512_castps512_ps256(v_CRTM_5_1);
        __m256 v256_CRTM_5_2 = _mm512_castps512_ps256(v_CRTM_5_2);
        __m256 v256_CRTM_5_3 = _mm512_castps512_ps256(v_CRTM_5_3);
        __m256 v256_CRTM_5_4 = _mm512_castps512_ps256(v_CRTM_5_4);

        // Input point 1: x(0)
        LDR_256_S(curr_in, v_in_stride, av_in0);
        // Input point 3: x(2)
        curr_in = in + in_strides[2];
        LDR_256_S(curr_in, v_in_stride, av_in1);
        // Input point 5: x(4)
        curr_in = in + in_strides[4];
        LDR_256_S(curr_in, v_in_stride, av_in2);
        // Input point 7: x(6)
        curr_in = in + in_strides[6];
        LDR_256_S(curr_in, v_in_stride, av_in3);
        // Input point 9: x(8)
        curr_in = in + in_strides[8];
        LDR_256_S(curr_in, v_in_stride, av_in4);

        av_s1 = _mm256_add_ps(av_in1, av_in4);
        av_s2 = _mm256_sub_ps(av_in4, av_in1);
        av_s3 = _mm256_add_ps(av_in2, av_in3);
        av_s4 = _mm256_sub_ps(av_in2, av_in3);
        av_s5 = _mm256_add_ps(av_s1, av_s3);

        av_t1 = _mm256_mul_ps(v256_CRTM_5_4, av_s5);
        av_s6 = _mm256_sub_ps(av_s1, av_s3);

        av_s7 = _mm256_sub_ps(av_in0, av_t1);
        av_t2 = _mm256_mul_ps(v256_CRTM_5_1, av_s6);

        av_t3 = _mm256_mul_ps(v256_CRTM_5_3, av_s4);
        av_t4 = _mm256_mul_ps(v256_CRTM_5_2, av_s2);
        av_t5 = _mm256_mul_ps(v256_CRTM_5_2, av_s4);
        av_t6 = _mm256_mul_ps(v256_CRTM_5_3, av_s2);

        // Output point 1: X(0)
        v_out0 = _mm256_add_ps(av_in0, av_s5);
        STR_256_S(curr_out, v_out_stride, v_out0);
        // Output point 4: X(3) & Output point 5: X(4)
        curr_out = out + out_strides[3];
        v_out3 = _mm256_add_ps(av_s7, av_t2);
        v_out4 = _mm256_sub_ps(av_t4, av_t3);
        STRI_2x256_S(curr_out, v_out_stride, v_out3, v_out4);
        // Output point 8: X(7) & Output point 9: X(8)
        curr_out = out + out_strides[7];
        v_out7 = _mm256_sub_ps(av_s7, av_t2);
        v_out8 = _mm256_add_ps(av_t5, av_t6);
        STRI_2x256_S(curr_out, v_out_stride, v_out7, v_out8);

        /* Shifted DFT */
        __m256 bv_in0, bv_in1, bv_in2, bv_in3, bv_in4;
        __m256 bv_s1, bv_s2, bv_s3, bv_s4, bv_s5, bv_s6, bv_s7;
        __m256 bv_t1, bv_t2, bv_t3, bv_t4, bv_t5, bv_t6;

        // Input point 2: x(1)
        curr_in = in + in_strides[1];
        LDR_256_S(curr_in, v_in_stride, bv_in0);
        // Input point 4: x(3)
        curr_in = in + in_strides[3];
        LDR_256_S(curr_in, v_in_stride, bv_in1);
        // Input point 6: x(5)
        curr_in = in + in_strides[5];
        LDR_256_S(curr_in, v_in_stride, bv_in2);
        // Input point 8: x(7)
        curr_in = in + in_strides[7];
        LDR_256_S(curr_in, v_in_stride, bv_in3);
        // Input point 10: x(9)
        curr_in = in + in_strides[9];
        LDR_256_S(curr_in, v_in_stride, bv_in4);

        bv_s1 = _mm256_add_ps(bv_in1, bv_in4);
        bv_s2 = _mm256_sub_ps(bv_in1, bv_in4);
        bv_s3 = _mm256_add_ps(bv_in2, bv_in3);
        bv_s4 = _mm256_sub_ps(bv_in2, bv_in3);
        bv_s5 = _mm256_sub_ps(bv_s2, bv_s4);

        bv_s6 = _mm256_add_ps(bv_s2, bv_s4);
        bv_t1 = _mm256_mul_ps(v256_CRTM_5_4, bv_s5);
        bv_s7 = _mm256_add_ps(bv_in0, bv_t1);
        bv_t2 = _mm256_mul_ps(v256_CRTM_5_1, bv_s6);

        bv_t3 = _mm256_mul_ps(v256_CRTM_5_2, bv_s3);
        bv_t4 = _mm256_mul_ps(v256_CRTM_5_3, bv_s1);
        bv_t5 = _mm256_mul_ps(v256_CRTM_5_3, bv_s3);
        bv_t6 = _mm256_mul_ps(v256_CRTM_5_2, bv_s1);

        // Output point 2: X(1) & Output point 3: X(2)
        curr_out = out + out_strides[1];
        v_out1 = _mm256_add_ps(bv_s7, bv_t2);
        v_out2 = _mm256_sub_ps(NEGATE_256_S(bv_t4), bv_t3);
        STRI_2x256_S(curr_out, v_out_stride, v_out1, v_out2);
        // Output point 6: X(5) & Output point 7: X(6)
        curr_out = out + out_strides[5];
        v_out5 = _mm256_sub_ps(bv_s7, bv_t2);
        v_out6 = _mm256_sub_ps(bv_t5, bv_t6);
        STRI_2x256_S(curr_out, v_out_stride, v_out5, v_out6);
        // Output point 10: X(9)
        curr_out = out + out_strides[9];
        v_out9 = _mm256_sub_ps(bv_in0, bv_s5);
        STR_256_S(curr_out, v_out_stride, v_out9);

        in = in + (v_in_stride << 3);
        out = out + (v_out_stride << 3);
    }
    // tail cases
    if (remaining_sets & NUM_SETS_REAL_128_S)
    {
        /* Standard DFT */
        __m128 av_in0, av_in1, av_in2, av_in3, av_in4;
        __m128 av_s1, av_s2, av_s3, av_s4, av_s5, av_s6, av_s7;
        __m128 av_t1, av_t2, av_t3, av_t4, av_t5, av_t6;
        __m128 v_out0, v_out1, v_out2, v_out3, v_out4, v_out5, v_out6, v_out7,
               v_out8, v_out9;

        curr_in = in;
        curr_out = out;

        __m128 v128_CRTM_5_1 = _mm512_castps512_ps128(v_CRTM_5_1);
        __m128 v128_CRTM_5_2 = _mm512_castps512_ps128(v_CRTM_5_2);
        __m128 v128_CRTM_5_3 = _mm512_castps512_ps128(v_CRTM_5_3);
        __m128 v128_CRTM_5_4 = _mm512_castps512_ps128(v_CRTM_5_4);

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

        av_s1 = _mm_add_ps(av_in1, av_in4);
        av_s2 = _mm_sub_ps(av_in4, av_in1);
        av_s3 = _mm_add_ps(av_in2, av_in3);
        av_s4 = _mm_sub_ps(av_in2, av_in3);
        av_s5 = _mm_add_ps(av_s1, av_s3);

        av_t1 = _mm_mul_ps(v128_CRTM_5_4, av_s5);
        av_s6 = _mm_sub_ps(av_s1, av_s3);

        av_s7 = _mm_sub_ps(av_in0, av_t1);
        av_t2 = _mm_mul_ps(v128_CRTM_5_1, av_s6);

        av_t3 = _mm_mul_ps(v128_CRTM_5_3, av_s4);
        av_t4 = _mm_mul_ps(v128_CRTM_5_2, av_s2);
        av_t5 = _mm_mul_ps(v128_CRTM_5_2, av_s4);
        av_t6 = _mm_mul_ps(v128_CRTM_5_3, av_s2);

        // Output point 1: X(0)
        v_out0 = _mm_add_ps(av_in0, av_s5);
        STR_128_S(curr_out, v_out_stride, v_out0);
        // Output point 4: X(3) & Output point 5: X(4)
        curr_out = out + out_strides[3];
        v_out3 = _mm_add_ps(av_s7, av_t2);
        v_out4 = _mm_sub_ps(av_t4, av_t3);
        STRI_2x128_S(curr_out, v_out_stride, v_out3, v_out4);
        // Output point 8: X(7) & Output point 9: X(8)
        curr_out = out + out_strides[7];
        v_out7 = _mm_sub_ps(av_s7, av_t2);
        v_out8 = _mm_add_ps(av_t5, av_t6);
        STRI_2x128_S(curr_out, v_out_stride, v_out7, v_out8);

        /* Shifted DFT */
        __m128 bv_in0, bv_in1, bv_in2, bv_in3, bv_in4;
        __m128 bv_s1, bv_s2, bv_s3, bv_s4, bv_s5, bv_s6, bv_s7;
        __m128 bv_t1, bv_t2, bv_t3, bv_t4, bv_t5, bv_t6;

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

        bv_s1 = _mm_add_ps(bv_in1, bv_in4);
        bv_s2 = _mm_sub_ps(bv_in1, bv_in4);
        bv_s3 = _mm_add_ps(bv_in2, bv_in3);
        bv_s4 = _mm_sub_ps(bv_in2, bv_in3);
        bv_s5 = _mm_sub_ps(bv_s2, bv_s4);

        bv_s6 = _mm_add_ps(bv_s2, bv_s4);
        bv_t1 = _mm_mul_ps(v128_CRTM_5_4, bv_s5);
        bv_s7 = _mm_add_ps(bv_in0, bv_t1);
        bv_t2 = _mm_mul_ps(v128_CRTM_5_1, bv_s6);

        bv_t3 = _mm_mul_ps(v128_CRTM_5_2, bv_s3);
        bv_t4 = _mm_mul_ps(v128_CRTM_5_3, bv_s1);
        bv_t5 = _mm_mul_ps(v128_CRTM_5_3, bv_s3);
        bv_t6 = _mm_mul_ps(v128_CRTM_5_2, bv_s1);

        // Output point 2: X(1) & Output point 3: X(2)
        curr_out = out + out_strides[1];
        v_out1 = _mm_add_ps(bv_s7, bv_t2);
        v_out2 = _mm_sub_ps(NEGATE_128_S(bv_t4), bv_t3);
        STRI_2x128_S(curr_out, v_out_stride, v_out1, v_out2);
        // Output point 6: X(5) & Output point 7: X(6)
        curr_out = out + out_strides[5];
        v_out5 = _mm_sub_ps(bv_s7, bv_t2);
        v_out6 = _mm_sub_ps(bv_t5, bv_t6);
        STRI_2x128_S(curr_out, v_out_stride, v_out5, v_out6);
        // Output point 10: X(9)
        curr_out = out + out_strides[9];
        v_out9 = _mm_sub_ps(bv_in0, bv_s5);
        STR_128_S(curr_out, v_out_stride, v_out9);

        in = in + (v_in_stride << 2);
        out = out + (v_out_stride << 2);
    }
    // tail cases
    if (remaining_sets & 2)
    {
        /* Standard DFT */
        __m128 av_in0, av_in1, av_in2, av_in3, av_in4;
        __m128 av_s1, av_s2, av_s3, av_s4, av_s5, av_s6, av_s7;
        __m128 av_t1, av_t2, av_t3, av_t4, av_t5, av_t6;
        __m128 v_out0, v_out1, v_out2, v_out3, v_out4, v_out5, v_out6, v_out7,
               v_out8, v_out9;

        curr_in = in;
        curr_out = out;

        __m128 v128_CRTM_5_1 = _mm512_castps512_ps128(v_CRTM_5_1);
        __m128 v128_CRTM_5_2 = _mm512_castps512_ps128(v_CRTM_5_2);
        __m128 v128_CRTM_5_3 = _mm512_castps512_ps128(v_CRTM_5_3);
        __m128 v128_CRTM_5_4 = _mm512_castps512_ps128(v_CRTM_5_4);

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

        av_s1 = _mm_add_ps(av_in1, av_in4);
        av_s2 = _mm_sub_ps(av_in4, av_in1);
        av_s3 = _mm_add_ps(av_in2, av_in3);
        av_s4 = _mm_sub_ps(av_in2, av_in3);
        av_s5 = _mm_add_ps(av_s1, av_s3);

        av_t1 = _mm_mul_ps(v128_CRTM_5_4, av_s5);
        av_s6 = _mm_sub_ps(av_s1, av_s3);

        av_s7 = _mm_sub_ps(av_in0, av_t1);
        av_t2 = _mm_mul_ps(v128_CRTM_5_1, av_s6);

        av_t3 = _mm_mul_ps(v128_CRTM_5_3, av_s4);
        av_t4 = _mm_mul_ps(v128_CRTM_5_2, av_s2);
        av_t5 = _mm_mul_ps(v128_CRTM_5_2, av_s4);
        av_t6 = _mm_mul_ps(v128_CRTM_5_3, av_s2);

        // Output point 1: X(0)
        v_out0 = _mm_add_ps(av_in0, av_s5);
        STHR_128_S(curr_out, v_out_stride, v_out0);
        // Output point 4: X(3) & Output point 5: X(4)
        curr_out = out + out_strides[3];
        v_out3 = _mm_add_ps(av_s7, av_t2);
        v_out4 = _mm_sub_ps(av_t4, av_t3);
        STHRI_2x128_S(curr_out, v_out_stride, v_out3, v_out4);
        // Output point 8: X(7) & Output point 9: X(8)
        curr_out = out + out_strides[7];
        v_out7 = _mm_sub_ps(av_s7, av_t2);
        v_out8 = _mm_add_ps(av_t5, av_t6);
        STHRI_2x128_S(curr_out, v_out_stride, v_out7, v_out8);

        /* Shifted DFT */
        __m128 bv_in0, bv_in1, bv_in2, bv_in3, bv_in4;
        __m128 bv_s1, bv_s2, bv_s3, bv_s4, bv_s5, bv_s6, bv_s7;
        __m128 bv_t1, bv_t2, bv_t3, bv_t4, bv_t5, bv_t6;

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

        bv_s1 = _mm_add_ps(bv_in1, bv_in4);
        bv_s2 = _mm_sub_ps(bv_in1, bv_in4);
        bv_s3 = _mm_add_ps(bv_in2, bv_in3);
        bv_s4 = _mm_sub_ps(bv_in2, bv_in3);
        bv_s5 = _mm_sub_ps(bv_s2, bv_s4);

        bv_s6 = _mm_add_ps(bv_s2, bv_s4);
        bv_t1 = _mm_mul_ps(v128_CRTM_5_4, bv_s5);
        bv_s7 = _mm_add_ps(bv_in0, bv_t1);
        bv_t2 = _mm_mul_ps(v128_CRTM_5_1, bv_s6);

        bv_t3 = _mm_mul_ps(v128_CRTM_5_2, bv_s3);
        bv_t4 = _mm_mul_ps(v128_CRTM_5_3, bv_s1);
        bv_t5 = _mm_mul_ps(v128_CRTM_5_3, bv_s3);
        bv_t6 = _mm_mul_ps(v128_CRTM_5_2, bv_s1);

        // Output point 2: X(1) & Output point 3: X(2)
        curr_out = out + out_strides[1];
        v_out1 = _mm_add_ps(bv_s7, bv_t2);
        v_out2 = _mm_sub_ps(NEGATE_128_S(bv_t4), bv_t3);
        STHRI_2x128_S(curr_out, v_out_stride, v_out1, v_out2);
        // Output point 6: X(5) & Output point 7: X(6)
        curr_out = out + out_strides[5];
        v_out5 = _mm_sub_ps(bv_s7, bv_t2);
        v_out6 = _mm_sub_ps(bv_t5, bv_t6);
        STHRI_2x128_S(curr_out, v_out_stride, v_out5, v_out6);
        // Output point 10: X(9)
        curr_out = out + out_strides[9];
        v_out9 = _mm_sub_ps(bv_in0, bv_s5);
        STHR_128_S(curr_out, v_out_stride, v_out9);

        in = in + (v_in_stride << 1);
        out = out + (v_out_stride << 1);
    }
    // tail cases
    if (remaining_sets & 1)
    {
        /* Standard DFT */
        FLOAT a_in0, a_in1, a_in2, a_in3, a_in4;
        FLOAT a_s1, a_s2, a_s3, a_s4, a_s5, a_s6, a_s7;
        FLOAT a_t1, a_t2, a_t3, a_t4, a_t5, a_t6;

        a_in0 = *in;               // Input point 1: x(0)
        a_in1 = in[in_strides[2]]; // Input point 3: x(2)
        a_in2 = in[in_strides[4]]; // Input point 5: x(4)
        a_in3 = in[in_strides[6]]; // Input point 7: x(6)
        a_in4 = in[in_strides[8]]; // Input point 9: x(8)

        a_s1 = a_in1 + a_in4;
        a_s2 = a_in4 - a_in1;
        a_s3 = a_in2 + a_in3;
        a_s4 = a_in2 - a_in3;
        a_s5 = a_s1 + a_s3;

        a_t1 = CRTM_5_4 * a_s5;
        a_s6 = a_s1 - a_s3;

        a_s7 = a_in0 - a_t1;
        a_t2 = CRTM_5_1 * a_s6;

        a_t3 = CRTM_5_3 * a_s4;
        a_t4 = CRTM_5_2 * a_s2;
        a_t5 = CRTM_5_2 * a_s4;
        a_t6 = CRTM_5_3 * a_s2;

        *out = a_in0 + a_s5;               // Output point 1: X(0)
        out[out_strides[3]] = a_s7 + a_t2; // Output point 4: X(3)
        out[out_strides[4]] = a_t4 - a_t3; // Output point 5: X(4)
        out[out_strides[7]] = a_s7 - a_t2; // Output point 8: X(7)
        out[out_strides[8]] = a_t5 + a_t6; // Output point 9: X(8)

        /* Shifted DFT */
        FLOAT b_in0, b_in1, b_in2, b_in3, b_in4;
        FLOAT b_s1, b_s2, b_s3, b_s4, b_s5, b_s6, b_s7;
        FLOAT b_t1, b_t2, b_t3, b_t4, b_t5, b_t6;

        b_in0 = in[in_strides[1]]; // Input point 2: x(1)
        b_in1 = in[in_strides[3]]; // Input point 4: x(3)
        b_in2 = in[in_strides[5]]; // Input point 6: x(5)
        b_in3 = in[in_strides[7]]; // Input point 8: x(7)
        b_in4 = in[in_strides[9]]; // Input point 10: x(9)

        b_s1 = b_in1 + b_in4;
        b_s2 = b_in1 - b_in4;
        b_s3 = b_in2 + b_in3;
        b_s4 = b_in2 - b_in3;
        b_s5 = b_s2 - b_s4;

        b_s6 = b_s2 + b_s4;
        b_t1 = CRTM_5_4 * b_s5;
        b_s7 = b_in0 + b_t1;
        b_t2 = CRTM_5_1 * b_s6;

        b_t3 = CRTM_5_2 * b_s3;
        b_t4 = CRTM_5_3 * b_s1;
        b_t5 = CRTM_5_3 * b_s3;
        b_t6 = CRTM_5_2 * b_s1;

        out[out_strides[1]] = b_s7 + b_t2;  // Output point 2: X(1)
        out[out_strides[2]] = -b_t3 - b_t4; // Output point 3: X(2)
        out[out_strides[5]] = b_s7 - b_t2;  // Output point 6: X(5)
        out[out_strides[6]] = b_t5 - b_t6;  // Output point 7: X(6)
        out[out_strides[9]] = b_in0 - b_s5; // Output point 10: X(9)
    }
    AOCLFFTZ_LOG(DEBUG, global_logger_mode, "Exit");
}

static VOID r2hcf_rfft5avx512_fp32_bwd(VOID *in_real, VOID *in_imag,
                                       VOID *out_real, VOID *out_imag, INTP n,
                                       aoclfftz_strides_t *strides, VOID *twd, UINT8 flag)
{
    AOCLFFTZ_LOG(DEBUG, global_logger_mode, "Enter");
    const FLOAT CRTM_5_1 = 1.11803398874989484820458683436563811772030918f;
    const FLOAT CRTM_5_2 = 1.90211303259030714423287866675876428681139726f;
    const FLOAT CRTM_5_3 = 1.17557050458494625833741190927814553719530488f;
    const FLOAT CRTM_5_4 = 0.50000000000000000000000000000000000000000000f;
    const FLOAT CRTM_5_5 = 2.00000000000000000000000000000000000000000000f;

    FLOAT *in = (FLOAT *)in_real;
    FLOAT *out = (FLOAT *)out_real;
#ifdef VOLATILE_STRIDE_ARRAY
    volatile INTP *in_strides = strides->in_strides;
    volatile INTP *out_strides = strides->out_strides;
#else
    INTP *in_strides = strides->in_strides;
    INTP *out_strides = strides->out_strides;
#endif
    INTP v_in_stride = strides->v_in_stride;
    INTP v_out_stride = strides->v_out_stride;

    INTP cnt;
    FLOAT *curr_in, *curr_out;
    INTP N = n / NUM_SETS_REAL_512_S;
    INTP remaining_sets = n % NUM_SETS_REAL_512_S;

    __m512 v_CRTM_5_1 = _mm512_set1_ps(CRTM_5_1);
    __m512 v_CRTM_5_2 = _mm512_set1_ps(CRTM_5_2);
    __m512 v_CRTM_5_3 = _mm512_set1_ps(CRTM_5_3);
    __m512 v_CRTM_5_4 = _mm512_set1_ps(CRTM_5_4);
    __m512 v_CRTM_5_5 = _mm512_set1_ps(CRTM_5_5);

    for (cnt = 0; cnt < N; cnt++)
    {
        /* Standard DFT */
        __m512 av_in0, av_in1, av_in2, av_in3, av_in4;
        __m512 av_s1, av_s2, av_s3, av_s4, av_s5, av_s6, av_s7;
        __m512 av_t1, av_t2, av_t3, av_t4, av_t5, av_t6, av_t7;
        __m512 v_out0, v_out1, v_out2, v_out3, v_out4, v_out5, v_out6, v_out7,
               v_out8, v_out9;

        curr_in = in;
        curr_out = out;

        // Input point 1: x(0)
        LDR_512_S(curr_in, v_in_stride, av_in0);
        // Input point 4: x(3) & Input point 5: x(4)
        curr_in = in + in_strides[3];
        LDRI_2x512_S(curr_in, v_in_stride, av_in1, av_in2);
        // Input point 8: x(7) & Input point 9: x(8)
        curr_in = in + in_strides[7];
        LDRI_2x512_S(curr_in, v_in_stride, av_in3, av_in4);

        av_s1 = _mm512_add_ps(av_in1, av_in3);
        av_s2 = _mm512_sub_ps(av_in1, av_in3);
        av_t1 = _mm512_mul_ps(v_CRTM_5_4, av_s1);
        av_t2 = _mm512_mul_ps(v_CRTM_5_1, av_s2);
        av_s3 = _mm512_sub_ps(av_in0, av_t1);
        av_s4 = _mm512_add_ps(av_s3, av_t2);
        av_s5 = _mm512_sub_ps(av_s3, av_t2);

        av_t3 = _mm512_mul_ps(v_CRTM_5_2, av_in4);
        av_t4 = _mm512_mul_ps(v_CRTM_5_3, av_in2);
        av_t5 = _mm512_mul_ps(v_CRTM_5_3, av_in4);
        av_t6 = _mm512_mul_ps(v_CRTM_5_2, av_in2);
        av_t7 = _mm512_mul_ps(v_CRTM_5_5, av_s1);

        av_s6 = _mm512_add_ps(av_t6, av_t5);
        av_s7 = _mm512_sub_ps(av_t3, av_t4);

        // Output pt 1: X(0)
        v_out0 = _mm512_add_ps(av_in0, av_t7);
        STR_512_S(curr_out, v_out_stride, v_out0);
        // Output pt 3: X(2)
        curr_out = out + out_strides[2];
        v_out2 = _mm512_sub_ps(av_s4, av_s6);
        STR_512_S(curr_out, v_out_stride, v_out2);
        // Output pt 5: X(4)
        curr_out = out + out_strides[4];
        v_out4 = _mm512_add_ps(av_s5, av_s7);
        STR_512_S(curr_out, v_out_stride, v_out4);
        // Output pt 7: X(6)
        curr_out = out + out_strides[6];
        v_out6 = _mm512_sub_ps(av_s5, av_s7);
        STR_512_S(curr_out, v_out_stride, v_out6);
        // Output pt 9: X(8)
        curr_out = out + out_strides[8];
        v_out8 = _mm512_add_ps(av_s4, av_s6);
        STR_512_S(curr_out, v_out_stride, v_out8);

        /* Shifted DFT */
        __m512 bv_in0, bv_in1, bv_in2, bv_in3, bv_in4;
        __m512 bv_s1, bv_s2, bv_s3, bv_s4, bv_s5, bv_s6, bv_s7;
        __m512 bv_t1, bv_t2, bv_t3, bv_t4, bv_t5, bv_t6, bv_t7;

        // Input point 2: x(1) & Input point 3: x(2)
        curr_in = in + in_strides[1];
        LDRI_2x512_S(curr_in, v_in_stride, bv_in0, bv_in1);
        // Input point 6: x(5) & Input point 7: x(6)
        curr_in = in + in_strides[5];
        LDRI_2x512_S(curr_in, v_in_stride, bv_in2, bv_in3);
        // Input point 10: x(9)
        curr_in = in + in_strides[9];
        LDR_512_S(curr_in, v_in_stride, bv_in4);

        bv_s1 = _mm512_add_ps(bv_in0, bv_in2);
        bv_s2 = _mm512_sub_ps(bv_in2, bv_in0);
        bv_t1 = _mm512_mul_ps(v_CRTM_5_1, bv_s2);
        bv_t2 = _mm512_mul_ps(v_CRTM_5_4, bv_s1);
        bv_s3 = _mm512_sub_ps(bv_in4, bv_t2);
        bv_s4 = _mm512_add_ps(bv_s3, bv_t1);
        bv_s5 = _mm512_sub_ps(bv_s3, bv_t1);

        bv_t3 = _mm512_mul_ps(v_CRTM_5_3, bv_in1);
        bv_t4 = _mm512_mul_ps(v_CRTM_5_2, bv_in3);
        bv_t5 = _mm512_mul_ps(v_CRTM_5_2, bv_in1);
        bv_t6 = _mm512_mul_ps(v_CRTM_5_3, bv_in3);
        bv_s6 = _mm512_add_ps(bv_t3, bv_t4);

        bv_s7 = _mm512_sub_ps(bv_t6, bv_t5);
        bv_t7 = _mm512_mul_ps(v_CRTM_5_5, bv_s1);

        // Output pt 2: X(1)
        curr_out = out + out_strides[1];
        v_out1 = _mm512_add_ps(bv_in4, bv_t7);
        STR_512_S(curr_out, v_out_stride, v_out1);
        // Output pt 4: X(3)
        curr_out = out + out_strides[3];
        v_out3 = _mm512_sub_ps(NEGATE_512_S(bv_s6), bv_s4);
        STR_512_S(curr_out, v_out_stride, v_out3);
        // Output pt 6: X(5)
        curr_out = out + out_strides[5];
        v_out5 = _mm512_add_ps(bv_s5, bv_s7);
        STR_512_S(curr_out, v_out_stride, v_out5);
        // Output pt 8: X(7)
        curr_out = out + out_strides[7];
        v_out7 = _mm512_sub_ps(bv_s7, bv_s5);
        STR_512_S(curr_out, v_out_stride, v_out7);
        // Output pt 10: X(9)
        curr_out = out + out_strides[9];
        v_out9 = _mm512_sub_ps(bv_s4, bv_s6);
        STR_512_S(curr_out, v_out_stride, v_out9);

        in = in + (v_in_stride << 4);
        out = out + (v_out_stride << 4);
    }
    // tail cases
    if (remaining_sets & NUM_SETS_REAL_256_S)
    {
        /* Standard DFT */
        __m256 av_in0, av_in1, av_in2, av_in3, av_in4;
        __m256 av_s1, av_s2, av_s3, av_s4, av_s5, av_s6, av_s7;
        __m256 av_t1, av_t2, av_t3, av_t4, av_t5, av_t6, av_t7;
        __m256 v_out0, v_out1, v_out2, v_out3, v_out4, v_out5, v_out6, v_out7,
               v_out8, v_out9;

        curr_in = in;
        curr_out = out;

        __m256 v256_CRTM_5_1 = _mm512_castps512_ps256(v_CRTM_5_1);
        __m256 v256_CRTM_5_2 = _mm512_castps512_ps256(v_CRTM_5_2);
        __m256 v256_CRTM_5_3 = _mm512_castps512_ps256(v_CRTM_5_3);
        __m256 v256_CRTM_5_4 = _mm512_castps512_ps256(v_CRTM_5_4);
        __m256 v256_CRTM_5_5 = _mm512_castps512_ps256(v_CRTM_5_5);

        // Input point 1: x(0)
        LDR_256_S(curr_in, v_in_stride, av_in0);
        // Input point 4: x(3) & Input point 5: x(4)
        curr_in = in + in_strides[3];
        LDRI_2x256_S(curr_in, v_in_stride, av_in1, av_in2);
        // Input point 8: x(7) & Input point 9: x(8)
        curr_in = in + in_strides[7];
        LDRI_2x256_S(curr_in, v_in_stride, av_in3, av_in4);

        av_s1 = _mm256_add_ps(av_in1, av_in3);
        av_s2 = _mm256_sub_ps(av_in1, av_in3);
        av_t1 = _mm256_mul_ps(v256_CRTM_5_4, av_s1);
        av_t2 = _mm256_mul_ps(v256_CRTM_5_1, av_s2);
        av_s3 = _mm256_sub_ps(av_in0, av_t1);
        av_s4 = _mm256_add_ps(av_s3, av_t2);
        av_s5 = _mm256_sub_ps(av_s3, av_t2);

        av_t3 = _mm256_mul_ps(v256_CRTM_5_2, av_in4);
        av_t4 = _mm256_mul_ps(v256_CRTM_5_3, av_in2);
        av_t5 = _mm256_mul_ps(v256_CRTM_5_3, av_in4);
        av_t6 = _mm256_mul_ps(v256_CRTM_5_2, av_in2);
        av_t7 = _mm256_mul_ps(v256_CRTM_5_5, av_s1);

        av_s6 = _mm256_add_ps(av_t6, av_t5);
        av_s7 = _mm256_sub_ps(av_t3, av_t4);

        // Output pt 1: X(0)
        v_out0 = _mm256_add_ps(av_in0, av_t7);
        STR_256_S(curr_out, v_out_stride, v_out0);
        // Output pt 3: X(2)
        curr_out = out + out_strides[2];
        v_out2 = _mm256_sub_ps(av_s4, av_s6);
        STR_256_S(curr_out, v_out_stride, v_out2);
        // Output pt 5: X(4)
        curr_out = out + out_strides[4];
        v_out4 = _mm256_add_ps(av_s5, av_s7);
        STR_256_S(curr_out, v_out_stride, v_out4);
        // Output pt 7: X(6)
        curr_out = out + out_strides[6];
        v_out6 = _mm256_sub_ps(av_s5, av_s7);
        STR_256_S(curr_out, v_out_stride, v_out6);
        // Output pt 9: X(8)
        curr_out = out + out_strides[8];
        v_out8 = _mm256_add_ps(av_s4, av_s6);
        STR_256_S(curr_out, v_out_stride, v_out8);

        /* Shifted DFT */
        __m256 bv_in0, bv_in1, bv_in2, bv_in3, bv_in4;
        __m256 bv_s1, bv_s2, bv_s3, bv_s4, bv_s5, bv_s6, bv_s7;
        __m256 bv_t1, bv_t2, bv_t3, bv_t4, bv_t5, bv_t6, bv_t7;

        // Input point 2: x(1) & Input point 3: x(2)
        curr_in = in + in_strides[1];
        LDRI_2x256_S(curr_in, v_in_stride, bv_in0, bv_in1);
        // Input point 6: x(5) & Input point 7: x(6)
        curr_in = in + in_strides[5];
        LDRI_2x256_S(curr_in, v_in_stride, bv_in2, bv_in3);
        // Input point 10: x(9)
        curr_in = in + in_strides[9];
        LDR_256_S(curr_in, v_in_stride, bv_in4);

        bv_s1 = _mm256_add_ps(bv_in0, bv_in2);
        bv_s2 = _mm256_sub_ps(bv_in2, bv_in0);
        bv_t1 = _mm256_mul_ps(v256_CRTM_5_1, bv_s2);
        bv_t2 = _mm256_mul_ps(v256_CRTM_5_4, bv_s1);
        bv_s3 = _mm256_sub_ps(bv_in4, bv_t2);
        bv_s4 = _mm256_add_ps(bv_s3, bv_t1);
        bv_s5 = _mm256_sub_ps(bv_s3, bv_t1);

        bv_t3 = _mm256_mul_ps(v256_CRTM_5_3, bv_in1);
        bv_t4 = _mm256_mul_ps(v256_CRTM_5_2, bv_in3);
        bv_t5 = _mm256_mul_ps(v256_CRTM_5_2, bv_in1);
        bv_t6 = _mm256_mul_ps(v256_CRTM_5_3, bv_in3);
        bv_s6 = _mm256_add_ps(bv_t3, bv_t4);

        bv_s7 = _mm256_sub_ps(bv_t6, bv_t5);
        bv_t7 = _mm256_mul_ps(v256_CRTM_5_5, bv_s1);

        // Output pt 2: X(1)
        curr_out = out + out_strides[1];
        v_out1 = _mm256_add_ps(bv_in4, bv_t7);
        STR_256_S(curr_out, v_out_stride, v_out1);
        // Output pt 4: X(3)
        curr_out = out + out_strides[3];
        v_out3 = _mm256_sub_ps(NEGATE_256_S(bv_s6), bv_s4);
        STR_256_S(curr_out, v_out_stride, v_out3);
        // Output pt 6: X(5)
        curr_out = out + out_strides[5];
        v_out5 = _mm256_add_ps(bv_s5, bv_s7);
        STR_256_S(curr_out, v_out_stride, v_out5);
        // Output pt 8: X(7)
        curr_out = out + out_strides[7];
        v_out7 = _mm256_sub_ps(bv_s7, bv_s5);
        STR_256_S(curr_out, v_out_stride, v_out7);
        // Output pt 10: X(9)
        curr_out = out + out_strides[9];
        v_out9 = _mm256_sub_ps(bv_s4, bv_s6);
        STR_256_S(curr_out, v_out_stride, v_out9);

        in = in + (v_in_stride << 3);
        out = out + (v_out_stride << 3);
    }
    // tail cases
    if (remaining_sets & NUM_SETS_REAL_128_S)
    {
        /* Standard DFT */
        __m128 av_in0, av_in1, av_in2, av_in3, av_in4;
        __m128 av_s1, av_s2, av_s3, av_s4, av_s5, av_s6, av_s7;
        __m128 av_t1, av_t2, av_t3, av_t4, av_t5, av_t6, av_t7;
        __m128 v_out0, v_out1, v_out2, v_out3, v_out4, v_out5, v_out6, v_out7,
               v_out8, v_out9;

        curr_in = in;
        curr_out = out;

        __m128 v128_CRTM_5_1 = _mm512_castps512_ps128(v_CRTM_5_1);
        __m128 v128_CRTM_5_2 = _mm512_castps512_ps128(v_CRTM_5_2);
        __m128 v128_CRTM_5_3 = _mm512_castps512_ps128(v_CRTM_5_3);
        __m128 v128_CRTM_5_4 = _mm512_castps512_ps128(v_CRTM_5_4);
        __m128 v128_CRTM_5_5 = _mm512_castps512_ps128(v_CRTM_5_5);

        // Input point 1: x(0)
        LDR_128_S(curr_in, v_in_stride, av_in0);
        // Input point 4: x(3) & Input point 5: x(4)
        curr_in = in + in_strides[3];
        LDRI_2x128_S(curr_in, v_in_stride, av_in1, av_in2);
        // Input point 8: x(7) & Input point 9: x(8)
        curr_in = in + in_strides[7];
        LDRI_2x128_S(curr_in, v_in_stride, av_in3, av_in4);

        av_s1 = _mm_add_ps(av_in1, av_in3);
        av_s2 = _mm_sub_ps(av_in1, av_in3);
        av_t1 = _mm_mul_ps(v128_CRTM_5_4, av_s1);
        av_t2 = _mm_mul_ps(v128_CRTM_5_1, av_s2);
        av_s3 = _mm_sub_ps(av_in0, av_t1);
        av_s4 = _mm_add_ps(av_s3, av_t2);
        av_s5 = _mm_sub_ps(av_s3, av_t2);

        av_t3 = _mm_mul_ps(v128_CRTM_5_2, av_in4);
        av_t4 = _mm_mul_ps(v128_CRTM_5_3, av_in2);
        av_t5 = _mm_mul_ps(v128_CRTM_5_3, av_in4);
        av_t6 = _mm_mul_ps(v128_CRTM_5_2, av_in2);
        av_t7 = _mm_mul_ps(v128_CRTM_5_5, av_s1);

        av_s6 = _mm_add_ps(av_t6, av_t5);
        av_s7 = _mm_sub_ps(av_t3, av_t4);

        // Output pt 1: X(0)
        v_out0 = _mm_add_ps(av_in0, av_t7);
        STR_128_S(curr_out, v_out_stride, v_out0);
        // Output pt 3: X(2)
        curr_out = out + out_strides[2];
        v_out2 = _mm_sub_ps(av_s4, av_s6);
        STR_128_S(curr_out, v_out_stride, v_out2);
        // Output pt 5: X(4)
        curr_out = out + out_strides[4];
        v_out4 = _mm_add_ps(av_s5, av_s7);
        STR_128_S(curr_out, v_out_stride, v_out4);
        // Output pt 7: X(6)
        curr_out = out + out_strides[6];
        v_out6 = _mm_sub_ps(av_s5, av_s7);
        STR_128_S(curr_out, v_out_stride, v_out6);
        // Output pt 9: X(8)
        curr_out = out + out_strides[8];
        v_out8 = _mm_add_ps(av_s4, av_s6);
        STR_128_S(curr_out, v_out_stride, v_out8);

        /* Shifted DFT */
        __m128 bv_in0, bv_in1, bv_in2, bv_in3, bv_in4;
        __m128 bv_s1, bv_s2, bv_s3, bv_s4, bv_s5, bv_s6, bv_s7;
        __m128 bv_t1, bv_t2, bv_t3, bv_t4, bv_t5, bv_t6, bv_t7;

        // Input point 2: x(1) & Input point 3: x(2)
        curr_in = in + in_strides[1];
        LDRI_2x128_S(curr_in, v_in_stride, bv_in0, bv_in1);
        // Input point 6: x(5) & Input point 7: x(6)
        curr_in = in + in_strides[5];
        LDRI_2x128_S(curr_in, v_in_stride, bv_in2, bv_in3);
        // Input point 10: x(9)
        curr_in = in + in_strides[9];
        LDR_128_S(curr_in, v_in_stride, bv_in4);

        bv_s1 = _mm_add_ps(bv_in0, bv_in2);
        bv_s2 = _mm_sub_ps(bv_in2, bv_in0);
        bv_t1 = _mm_mul_ps(v128_CRTM_5_1, bv_s2);
        bv_t2 = _mm_mul_ps(v128_CRTM_5_4, bv_s1);
        bv_s3 = _mm_sub_ps(bv_in4, bv_t2);
        bv_s4 = _mm_add_ps(bv_s3, bv_t1);
        bv_s5 = _mm_sub_ps(bv_s3, bv_t1);

        bv_t3 = _mm_mul_ps(v128_CRTM_5_3, bv_in1);
        bv_t4 = _mm_mul_ps(v128_CRTM_5_2, bv_in3);
        bv_t5 = _mm_mul_ps(v128_CRTM_5_2, bv_in1);
        bv_t6 = _mm_mul_ps(v128_CRTM_5_3, bv_in3);
        bv_s6 = _mm_add_ps(bv_t3, bv_t4);

        bv_s7 = _mm_sub_ps(bv_t6, bv_t5);
        bv_t7 = _mm_mul_ps(v128_CRTM_5_5, bv_s1);

        // Output pt 2: X(1)
        curr_out = out + out_strides[1];
        v_out1 = _mm_add_ps(bv_in4, bv_t7);
        STR_128_S(curr_out, v_out_stride, v_out1);
        // Output pt 4: X(3)
        curr_out = out + out_strides[3];
        v_out3 = _mm_sub_ps(NEGATE_128_S(bv_s6), bv_s4);
        STR_128_S(curr_out, v_out_stride, v_out3);
        // Output pt 6: X(5)
        curr_out = out + out_strides[5];
        v_out5 = _mm_add_ps(bv_s5, bv_s7);
        STR_128_S(curr_out, v_out_stride, v_out5);
        // Output pt 8: X(7)
        curr_out = out + out_strides[7];
        v_out7 = _mm_sub_ps(bv_s7, bv_s5);
        STR_128_S(curr_out, v_out_stride, v_out7);
        // Output pt 10: X(9)
        curr_out = out + out_strides[9];
        v_out9 = _mm_sub_ps(bv_s4, bv_s6);
        STR_128_S(curr_out, v_out_stride, v_out9);

        in = in + (v_in_stride << 2);
        out = out + (v_out_stride << 2);
    }
    // tail cases
    if (remaining_sets & 2)
    {
        /* Standard DFT */
        __m128 av_in0, av_in1, av_in2, av_in3, av_in4;
        __m128 av_s1, av_s2, av_s3, av_s4, av_s5, av_s6, av_s7;
        __m128 av_t1, av_t2, av_t3, av_t4, av_t5, av_t6, av_t7;
        __m128 v_out0, v_out1, v_out2, v_out3, v_out4, v_out5, v_out6, v_out7,
               v_out8, v_out9;

        curr_in = in;
        curr_out = out;

        __m128 v128_CRTM_5_1 = _mm512_castps512_ps128(v_CRTM_5_1);
        __m128 v128_CRTM_5_2 = _mm512_castps512_ps128(v_CRTM_5_2);
        __m128 v128_CRTM_5_3 = _mm512_castps512_ps128(v_CRTM_5_3);
        __m128 v128_CRTM_5_4 = _mm512_castps512_ps128(v_CRTM_5_4);
        __m128 v128_CRTM_5_5 = _mm512_castps512_ps128(v_CRTM_5_5);

        // Input point 1: x(0)
        LDHR_128_S(curr_in, v_in_stride, av_in0);
        // Input point 4: x(3) & Input point 5: x(4)
        curr_in = in + in_strides[3];
        LDHRI_2x128_S(curr_in, v_in_stride, av_in1, av_in2);
        // Input point 8: x(7) & Input point 9: x(8)
        curr_in = in + in_strides[7];
        LDHRI_2x128_S(curr_in, v_in_stride, av_in3, av_in4);

        av_s1 = _mm_add_ps(av_in1, av_in3);
        av_s2 = _mm_sub_ps(av_in1, av_in3);
        av_t1 = _mm_mul_ps(v128_CRTM_5_4, av_s1);
        av_t2 = _mm_mul_ps(v128_CRTM_5_1, av_s2);
        av_s3 = _mm_sub_ps(av_in0, av_t1);
        av_s4 = _mm_add_ps(av_s3, av_t2);
        av_s5 = _mm_sub_ps(av_s3, av_t2);

        av_t3 = _mm_mul_ps(v128_CRTM_5_2, av_in4);
        av_t4 = _mm_mul_ps(v128_CRTM_5_3, av_in2);
        av_t5 = _mm_mul_ps(v128_CRTM_5_3, av_in4);
        av_t6 = _mm_mul_ps(v128_CRTM_5_2, av_in2);
        av_t7 = _mm_mul_ps(v128_CRTM_5_5, av_s1);

        av_s6 = _mm_add_ps(av_t6, av_t5);
        av_s7 = _mm_sub_ps(av_t3, av_t4);

        // Output pt 1: X(0)
        v_out0 = _mm_add_ps(av_in0, av_t7);
        STHR_128_S(curr_out, v_out_stride, v_out0);
        // Output pt 3: X(2)
        curr_out = out + out_strides[2];
        v_out2 = _mm_sub_ps(av_s4, av_s6);
        STHR_128_S(curr_out, v_out_stride, v_out2);
        // Output pt 5: X(4)
        curr_out = out + out_strides[4];
        v_out4 = _mm_add_ps(av_s5, av_s7);
        STHR_128_S(curr_out, v_out_stride, v_out4);
        // Output pt 7: X(6)
        curr_out = out + out_strides[6];
        v_out6 = _mm_sub_ps(av_s5, av_s7);
        STHR_128_S(curr_out, v_out_stride, v_out6);
        // Output pt 9: X(8)
        curr_out = out + out_strides[8];
        v_out8 = _mm_add_ps(av_s4, av_s6);
        STHR_128_S(curr_out, v_out_stride, v_out8);

        /* Shifted DFT */
        __m128 bv_in0, bv_in1, bv_in2, bv_in3, bv_in4;
        __m128 bv_s1, bv_s2, bv_s3, bv_s4, bv_s5, bv_s6, bv_s7;
        __m128 bv_t1, bv_t2, bv_t3, bv_t4, bv_t5, bv_t6, bv_t7;

        // Input point 2: x(1) & Input point 3: x(2)
        curr_in = in + in_strides[1];
        LDHRI_2x128_S(curr_in, v_in_stride, bv_in0, bv_in1);
        // Input point 6: x(5) & Input point 7: x(6)
        curr_in = in + in_strides[5];
        LDHRI_2x128_S(curr_in, v_in_stride, bv_in2, bv_in3);
        // Input point 10: x(9)
        curr_in = in + in_strides[9];
        LDHR_128_S(curr_in, v_in_stride, bv_in4);

        bv_s1 = _mm_add_ps(bv_in0, bv_in2);
        bv_s2 = _mm_sub_ps(bv_in2, bv_in0);
        bv_t1 = _mm_mul_ps(v128_CRTM_5_1, bv_s2);
        bv_t2 = _mm_mul_ps(v128_CRTM_5_4, bv_s1);
        bv_s3 = _mm_sub_ps(bv_in4, bv_t2);
        bv_s4 = _mm_add_ps(bv_s3, bv_t1);
        bv_s5 = _mm_sub_ps(bv_s3, bv_t1);

        bv_t3 = _mm_mul_ps(v128_CRTM_5_3, bv_in1);
        bv_t4 = _mm_mul_ps(v128_CRTM_5_2, bv_in3);
        bv_t5 = _mm_mul_ps(v128_CRTM_5_2, bv_in1);
        bv_t6 = _mm_mul_ps(v128_CRTM_5_3, bv_in3);
        bv_s6 = _mm_add_ps(bv_t3, bv_t4);

        bv_s7 = _mm_sub_ps(bv_t6, bv_t5);
        bv_t7 = _mm_mul_ps(v128_CRTM_5_5, bv_s1);

        // Output pt 2: X(1)
        curr_out = out + out_strides[1];
        v_out1 = _mm_add_ps(bv_in4, bv_t7);
        STHR_128_S(curr_out, v_out_stride, v_out1);
        // Output pt 4: X(3)
        curr_out = out + out_strides[3];
        v_out3 = _mm_sub_ps(NEGATE_128_S(bv_s6), bv_s4);
        STHR_128_S(curr_out, v_out_stride, v_out3);
        // Output pt 6: X(5)
        curr_out = out + out_strides[5];
        v_out5 = _mm_add_ps(bv_s5, bv_s7);
        STHR_128_S(curr_out, v_out_stride, v_out5);
        // Output pt 8: X(7)
        curr_out = out + out_strides[7];
        v_out7 = _mm_sub_ps(bv_s7, bv_s5);
        STHR_128_S(curr_out, v_out_stride, v_out7);
        // Output pt 10: X(9)
        curr_out = out + out_strides[9];
        v_out9 = _mm_sub_ps(bv_s4, bv_s6);
        STHR_128_S(curr_out, v_out_stride, v_out9);

        in = in + (v_in_stride << 1);
        out = out + (v_out_stride << 1);
    }
    // tail cases
    if (remaining_sets & 1)
    {
        /* Standard DFT */
        FLOAT a_in0, a_in1, a_in2, a_in3, a_in4;
        FLOAT a_s1, a_s2, a_s3, a_s4, a_s5, a_s6, a_s7;
        FLOAT a_t1, a_t2, a_t3, a_t4, a_t5, a_t6, a_t7;

        a_in0 = *in;               // Input point 1: x(0)
        a_in1 = in[in_strides[3]]; // Input point 4: x(3)
        a_in2 = in[in_strides[4]]; // Input point 5: x(4)
        a_in3 = in[in_strides[7]]; // Input point 8: x(7)
        a_in4 = in[in_strides[8]]; // Input point 9: x(8)

        a_s1 = a_in1 + a_in3;
        a_s2 = a_in1 - a_in3;
        a_t1 = CRTM_5_4 * a_s1;
        a_t2 = CRTM_5_1 * a_s2;
        a_s3 = a_in0 - a_t1;
        a_s4 = a_s3 + a_t2;
        a_s5 = a_s3 - a_t2;

        a_t3 = CRTM_5_2 * a_in4;
        a_t4 = CRTM_5_3 * a_in2;
        a_t5 = CRTM_5_3 * a_in4;
        a_t6 = CRTM_5_2 * a_in2;
        a_t7 = CRTM_5_5 * a_s1;

        a_s6 = a_t6 + a_t5;
        a_s7 = a_t3 - a_t4;

        *out = a_in0 + a_t7;               // Output point 1: X(0)
        out[out_strides[2]] = a_s4 - a_s6; // Output point 3: X(2)
        out[out_strides[4]] = a_s5 + a_s7; // Output point 5: X(4)
        out[out_strides[6]] = a_s5 - a_s7; // Output point 7: X(6)
        out[out_strides[8]] = a_s4 + a_s6; // Output point 9: X(8)

        /* Shifted DFT */
        FLOAT b_in0, b_in1, b_in2, b_in3, b_in4;
        FLOAT b_s1, b_s2, b_s3, b_s4, b_s5, b_s6, b_s7;
        FLOAT b_t1, b_t2, b_t3, b_t4, b_t5, b_t6, b_t7;

        b_in0 = in[in_strides[1]]; // Input point 2: x(1)
        b_in1 = in[in_strides[2]]; // Input point 3: x(2)
        b_in2 = in[in_strides[5]]; // Input point 6: x(5)
        b_in3 = in[in_strides[6]]; // Input point 7: x(6)
        b_in4 = in[in_strides[9]]; // Input point 10: x(9)

        b_s1 = b_in0 + b_in2;
        b_s2 = b_in2 - b_in0;
        b_t1 = CRTM_5_1 * b_s2;
        b_t2 = CRTM_5_4 * b_s1;
        b_s3 = b_in4 - b_t2;
        b_s4 = b_s3 + b_t1;
        b_s5 = b_s3 - b_t1;

        b_t3 = CRTM_5_3 * b_in1;
        b_t4 = CRTM_5_2 * b_in3;
        b_t5 = CRTM_5_2 * b_in1;
        b_t6 = CRTM_5_3 * b_in3;
        b_t7 = b_t4 + b_t3;

        b_s6 = b_t6 - b_t5;
        b_s7 = CRTM_5_5 * b_s1;

        out[out_strides[1]] = b_in4 + b_s7; // Output point 2: X(1)
        out[out_strides[3]] = -b_t7 - b_s4; // Output point 4: X(3)
        out[out_strides[5]] = b_s5 + b_s6;  // Output point 6: X(5)
        out[out_strides[7]] = b_s6 - b_s5;  // Output point 8: X(7)
        out[out_strides[9]] = b_s4 - b_t7;  // Output point 10: X(9)
    }
    AOCLFFTZ_LOG(DEBUG, global_logger_mode, "Exit");
}

static VOID r2hcf_rfft5avx512_fp64_fwd(VOID *in_real, VOID *in_imag,
                                       VOID *out_real, VOID *out_imag, INTP n,
                                       aoclfftz_strides_t *strides, VOID *twd, UINT8 flag)
{
    AOCLFFTZ_LOG(DEBUG, global_logger_mode, "Enter");
    const DOUBLE CRTM_5_1 = 0.559016994374947424102293417182819058860154590;
    const DOUBLE CRTM_5_2 = 0.951056516295153572116439333379382143405698632;
    const DOUBLE CRTM_5_3 = 0.587785252292473129168705954639072768597652438;
    const DOUBLE CRTM_5_4 = 0.250000000000000000000000000000000000000000000;

    DOUBLE *in = (DOUBLE *)in_real;
    DOUBLE *out = (DOUBLE *)out_real;
#ifdef VOLATILE_STRIDE_ARRAY
    volatile INTP *in_strides = strides->in_strides;
    volatile INTP *out_strides = strides->out_strides;
#else
    INTP *in_strides = strides->in_strides;
    INTP *out_strides = strides->out_strides;
#endif
    INTP v_in_stride = strides->v_in_stride;
    INTP v_out_stride = strides->v_out_stride;

    INTP cnt;
    DOUBLE *curr_in, *curr_out;
    INTP N = n / NUM_SETS_REAL_256_S;
    INTP remaining_sets = n % NUM_SETS_REAL_256_S;

    __m512d v_CRTM_5_1 = _mm512_set1_pd(CRTM_5_1);
    __m512d v_CRTM_5_2 = _mm512_set1_pd(CRTM_5_2);
    __m512d v_CRTM_5_3 = _mm512_set1_pd(CRTM_5_3);
    __m512d v_CRTM_5_4 = _mm512_set1_pd(CRTM_5_4);

    for (cnt = 0; cnt < N; cnt++)
    {
        /* Standard DFT */
        __m512d av_in0, av_in1, av_in2, av_in3, av_in4;
        __m512d av_s1, av_s2, av_s3, av_s4, av_s5, av_s6, av_s7;
        __m512d av_t1, av_t2, av_t3, av_t4, av_t5, av_t6;
        __m512d v_out0, v_out1, v_out2, v_out3, v_out4, v_out5, v_out6, v_out7,
                v_out8, v_out9;

        curr_in = in;
        curr_out = out;

        // Input point 1: x(0)
        LDR_512_D(curr_in, v_in_stride, av_in0);
        // Input point 3: x(2)
        curr_in = in + in_strides[2];
        LDR_512_D(curr_in, v_in_stride, av_in1);
        // Input point 5: x(4)
        curr_in = in + in_strides[4];
        LDR_512_D(curr_in, v_in_stride, av_in2);
        // Input point 7: x(6)
        curr_in = in + in_strides[6];
        LDR_512_D(curr_in, v_in_stride, av_in3);
        // Input point 9: x(8)
        curr_in = in + in_strides[8];
        LDR_512_D(curr_in, v_in_stride, av_in4);

        av_s1 = _mm512_add_pd(av_in1, av_in4);
        av_s2 = _mm512_sub_pd(av_in4, av_in1);
        av_s3 = _mm512_add_pd(av_in2, av_in3);
        av_s4 = _mm512_sub_pd(av_in2, av_in3);
        av_s5 = _mm512_add_pd(av_s1, av_s3);

        av_t1 = _mm512_mul_pd(v_CRTM_5_4, av_s5);
        av_s6 = _mm512_sub_pd(av_s1, av_s3);

        av_s7 = _mm512_sub_pd(av_in0, av_t1);
        av_t2 = _mm512_mul_pd(v_CRTM_5_1, av_s6);

        av_t3 = _mm512_mul_pd(v_CRTM_5_3, av_s4);
        av_t4 = _mm512_mul_pd(v_CRTM_5_2, av_s2);
        av_t5 = _mm512_mul_pd(v_CRTM_5_2, av_s4);
        av_t6 = _mm512_mul_pd(v_CRTM_5_3, av_s2);

        // Output point 1: X(0)
        v_out0 = _mm512_add_pd(av_in0, av_s5);
        STR_512_D(curr_out, v_out_stride, v_out0);
        // Output point 4: X(3) & Output point 5: X(4)
        curr_out = out + out_strides[3];
        v_out3 = _mm512_add_pd(av_s7, av_t2);
        v_out4 = _mm512_sub_pd(av_t4, av_t3);
        STRI_2x512_D(curr_out, v_out_stride, v_out3, v_out4);
        // Output point 8: X(7) & Output point 9: X(8)
        curr_out = out + out_strides[7];
        v_out7 = _mm512_sub_pd(av_s7, av_t2);
        v_out8 = _mm512_add_pd(av_t5, av_t6);
        STRI_2x512_D(curr_out, v_out_stride, v_out7, v_out8);

        /* Shifted DFT */
        __m512d bv_in0, bv_in1, bv_in2, bv_in3, bv_in4;
        __m512d bv_s1, bv_s2, bv_s3, bv_s4, bv_s5, bv_s6, bv_s7;
        __m512d bv_t1, bv_t2, bv_t3, bv_t4, bv_t5, bv_t6;

        // Input point 2: x(1)
        curr_in = in + in_strides[1];
        LDR_512_D(curr_in, v_in_stride, bv_in0);
        // Input point 4: x(3)
        curr_in = in + in_strides[3];
        LDR_512_D(curr_in, v_in_stride, bv_in1);
        // Input point 6: x(5)
        curr_in = in + in_strides[5];
        LDR_512_D(curr_in, v_in_stride, bv_in2);
        // Input point 8: x(7)
        curr_in = in + in_strides[7];
        LDR_512_D(curr_in, v_in_stride, bv_in3);
        // Input point 10: x(9)
        curr_in = in + in_strides[9];
        LDR_512_D(curr_in, v_in_stride, bv_in4);

        bv_s1 = _mm512_add_pd(bv_in1, bv_in4);
        bv_s2 = _mm512_sub_pd(bv_in1, bv_in4);
        bv_s3 = _mm512_add_pd(bv_in2, bv_in3);
        bv_s4 = _mm512_sub_pd(bv_in2, bv_in3);
        bv_s5 = _mm512_sub_pd(bv_s2, bv_s4);

        bv_s6 = _mm512_add_pd(bv_s2, bv_s4);
        bv_t1 = _mm512_mul_pd(v_CRTM_5_4, bv_s5);
        bv_s7 = _mm512_add_pd(bv_in0, bv_t1);
        bv_t2 = _mm512_mul_pd(v_CRTM_5_1, bv_s6);

        bv_t3 = _mm512_mul_pd(v_CRTM_5_2, bv_s3);
        bv_t4 = _mm512_mul_pd(v_CRTM_5_3, bv_s1);
        bv_t5 = _mm512_mul_pd(v_CRTM_5_3, bv_s3);
        bv_t6 = _mm512_mul_pd(v_CRTM_5_2, bv_s1);

        // Output point 2: X(1) & Output point 3: X(2)
        curr_out = out + out_strides[1];
        v_out1 = _mm512_add_pd(bv_s7, bv_t2);
        v_out2 = _mm512_sub_pd(NEGATE_512_D(bv_t4), bv_t3);
        STRI_2x512_D(curr_out, v_out_stride, v_out1, v_out2);
        // Output point 6: X(5) & Output point 7: X(6)
        curr_out = out + out_strides[5];
        v_out5 = _mm512_sub_pd(bv_s7, bv_t2);
        v_out6 = _mm512_sub_pd(bv_t5, bv_t6);
        STRI_2x512_D(curr_out, v_out_stride, v_out5, v_out6);
        // Output point 10: X(9)
        curr_out = out + out_strides[9];
        v_out9 = _mm512_sub_pd(bv_in0, bv_s5);
        STR_512_D(curr_out, v_out_stride, v_out9);

        in = in + (v_in_stride << 3);
        out = out + (v_out_stride << 3);
    }
    // tail cases
    if (remaining_sets & NUM_SETS_REAL_256_D)
    {
        /* Standard DFT */
        __m256d av_in0, av_in1, av_in2, av_in3, av_in4;
        __m256d av_s1, av_s2, av_s3, av_s4, av_s5, av_s6, av_s7;
        __m256d av_t1, av_t2, av_t3, av_t4, av_t5, av_t6;
        __m256d v_out0, v_out1, v_out2, v_out3, v_out4, v_out5, v_out6, v_out7,
                v_out8, v_out9;

        curr_in = in;
        curr_out = out;

        __m256d v256_CRTM_5_1 = _mm512_castpd512_pd256(v_CRTM_5_1);
        __m256d v256_CRTM_5_2 = _mm512_castpd512_pd256(v_CRTM_5_2);
        __m256d v256_CRTM_5_3 = _mm512_castpd512_pd256(v_CRTM_5_3);
        __m256d v256_CRTM_5_4 = _mm512_castpd512_pd256(v_CRTM_5_4);

        // Input point 1: x(0)
        LDR_256_D(curr_in, v_in_stride, av_in0);
        // Input point 3: x(2)
        curr_in = in + in_strides[2];
        LDR_256_D(curr_in, v_in_stride, av_in1);
        // Input point 5: x(4)
        curr_in = in + in_strides[4];
        LDR_256_D(curr_in, v_in_stride, av_in2);
        // Input point 7: x(6)
        curr_in = in + in_strides[6];
        LDR_256_D(curr_in, v_in_stride, av_in3);
        // Input point 9: x(8)
        curr_in = in + in_strides[8];
        LDR_256_D(curr_in, v_in_stride, av_in4);

        av_s1 = _mm256_add_pd(av_in1, av_in4);
        av_s2 = _mm256_sub_pd(av_in4, av_in1);
        av_s3 = _mm256_add_pd(av_in2, av_in3);
        av_s4 = _mm256_sub_pd(av_in2, av_in3);
        av_s5 = _mm256_add_pd(av_s1, av_s3);

        av_t1 = _mm256_mul_pd(v256_CRTM_5_4, av_s5);
        av_s6 = _mm256_sub_pd(av_s1, av_s3);

        av_s7 = _mm256_sub_pd(av_in0, av_t1);
        av_t2 = _mm256_mul_pd(v256_CRTM_5_1, av_s6);

        av_t3 = _mm256_mul_pd(v256_CRTM_5_3, av_s4);
        av_t4 = _mm256_mul_pd(v256_CRTM_5_2, av_s2);
        av_t5 = _mm256_mul_pd(v256_CRTM_5_2, av_s4);
        av_t6 = _mm256_mul_pd(v256_CRTM_5_3, av_s2);

        // Output point 1: X(0)
        v_out0 = _mm256_add_pd(av_in0, av_s5);
        STR_256_D(curr_out, v_out_stride, v_out0);
        // Output point 4: X(3) & Output point 5: X(4)
        curr_out = out + out_strides[3];
        v_out3 = _mm256_add_pd(av_s7, av_t2);
        v_out4 = _mm256_sub_pd(av_t4, av_t3);
        STRI_2x256_D(curr_out, v_out_stride, v_out3, v_out4);
        // Output point 8: X(7) & Output point 9: X(8)
        curr_out = out + out_strides[7];
        v_out7 = _mm256_sub_pd(av_s7, av_t2);
        v_out8 = _mm256_add_pd(av_t5, av_t6);
        STRI_2x256_D(curr_out, v_out_stride, v_out7, v_out8);

        /* Shifted DFT */
        __m256d bv_in0, bv_in1, bv_in2, bv_in3, bv_in4;
        __m256d bv_s1, bv_s2, bv_s3, bv_s4, bv_s5, bv_s6, bv_s7;
        __m256d bv_t1, bv_t2, bv_t3, bv_t4, bv_t5, bv_t6;

        // Input point 2: x(1)
        curr_in = in + in_strides[1];
        LDR_256_D(curr_in, v_in_stride, bv_in0);
        // Input point 4: x(3)
        curr_in = in + in_strides[3];
        LDR_256_D(curr_in, v_in_stride, bv_in1);
        // Input point 6: x(5)
        curr_in = in + in_strides[5];
        LDR_256_D(curr_in, v_in_stride, bv_in2);
        // Input point 8: x(7)
        curr_in = in + in_strides[7];
        LDR_256_D(curr_in, v_in_stride, bv_in3);
        // Input point 10: x(9)
        curr_in = in + in_strides[9];
        LDR_256_D(curr_in, v_in_stride, bv_in4);

        bv_s1 = _mm256_add_pd(bv_in1, bv_in4);
        bv_s2 = _mm256_sub_pd(bv_in1, bv_in4);
        bv_s3 = _mm256_add_pd(bv_in2, bv_in3);
        bv_s4 = _mm256_sub_pd(bv_in2, bv_in3);
        bv_s5 = _mm256_sub_pd(bv_s2, bv_s4);

        bv_s6 = _mm256_add_pd(bv_s2, bv_s4);
        bv_t1 = _mm256_mul_pd(v256_CRTM_5_4, bv_s5);
        bv_s7 = _mm256_add_pd(bv_in0, bv_t1);
        bv_t2 = _mm256_mul_pd(v256_CRTM_5_1, bv_s6);

        bv_t3 = _mm256_mul_pd(v256_CRTM_5_2, bv_s3);
        bv_t4 = _mm256_mul_pd(v256_CRTM_5_3, bv_s1);
        bv_t5 = _mm256_mul_pd(v256_CRTM_5_3, bv_s3);
        bv_t6 = _mm256_mul_pd(v256_CRTM_5_2, bv_s1);

        // Output point 2: X(1) & Output point 3: X(2)
        curr_out = out + out_strides[1];
        v_out1 = _mm256_add_pd(bv_s7, bv_t2);
        v_out2 = _mm256_sub_pd(NEGATE_256_D(bv_t4), bv_t3);
        STRI_2x256_D(curr_out, v_out_stride, v_out1, v_out2);
        // Output point 6: X(5) & Output point 7: X(6)
        curr_out = out + out_strides[5];
        v_out5 = _mm256_sub_pd(bv_s7, bv_t2);
        v_out6 = _mm256_sub_pd(bv_t5, bv_t6);
        STRI_2x256_D(curr_out, v_out_stride, v_out5, v_out6);
        // Output point 10: X(9)
        curr_out = out + out_strides[9];
        v_out9 = _mm256_sub_pd(bv_in0, bv_s5);
        STR_256_D(curr_out, v_out_stride, v_out9);

        in = in + (v_in_stride << 2);
        out = out + (v_out_stride << 2);
    }
    // tail cases
    if (remaining_sets & NUM_SETS_REAL_128_D)
    {
        /* Standard DFT */
        __m128d av_in0, av_in1, av_in2, av_in3, av_in4;
        __m128d av_s1, av_s2, av_s3, av_s4, av_s5, av_s6, av_s7;
        __m128d av_t1, av_t2, av_t3, av_t4, av_t5, av_t6;
        __m128d v_out0, v_out1, v_out2, v_out3, v_out4, v_out5, v_out6, v_out7,
                v_out8, v_out9;

        curr_in = in;
        curr_out = out;

        __m128d v128_CRTM_5_1 = _mm512_castpd512_pd128(v_CRTM_5_1);
        __m128d v128_CRTM_5_2 = _mm512_castpd512_pd128(v_CRTM_5_2);
        __m128d v128_CRTM_5_3 = _mm512_castpd512_pd128(v_CRTM_5_3);
        __m128d v128_CRTM_5_4 = _mm512_castpd512_pd128(v_CRTM_5_4);

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

        av_s1 = _mm_add_pd(av_in1, av_in4);
        av_s2 = _mm_sub_pd(av_in4, av_in1);
        av_s3 = _mm_add_pd(av_in2, av_in3);
        av_s4 = _mm_sub_pd(av_in2, av_in3);
        av_s5 = _mm_add_pd(av_s1, av_s3);

        av_t1 = _mm_mul_pd(v128_CRTM_5_4, av_s5);
        av_s6 = _mm_sub_pd(av_s1, av_s3);

        av_s7 = _mm_sub_pd(av_in0, av_t1);
        av_t2 = _mm_mul_pd(v128_CRTM_5_1, av_s6);

        av_t3 = _mm_mul_pd(v128_CRTM_5_3, av_s4);
        av_t4 = _mm_mul_pd(v128_CRTM_5_2, av_s2);
        av_t5 = _mm_mul_pd(v128_CRTM_5_2, av_s4);
        av_t6 = _mm_mul_pd(v128_CRTM_5_3, av_s2);

        // Output point 1: X(0)
        v_out0 = _mm_add_pd(av_in0, av_s5);
        STR_128_D(curr_out, v_out_stride, v_out0);
        // Output point 4: X(3) & Output point 5: X(4)
        curr_out = out + out_strides[3];
        v_out3 = _mm_add_pd(av_s7, av_t2);
        v_out4 = _mm_sub_pd(av_t4, av_t3);
        STRI_2x128_D(curr_out, v_out_stride, v_out3, v_out4);
        // Output point 8: X(7) & Output point 9: X(8)
        curr_out = out + out_strides[7];
        v_out7 = _mm_sub_pd(av_s7, av_t2);
        v_out8 = _mm_add_pd(av_t5, av_t6);
        STRI_2x128_D(curr_out, v_out_stride, v_out7, v_out8);

        /* Shifted DFT */
        __m128d bv_in0, bv_in1, bv_in2, bv_in3, bv_in4;
        __m128d bv_s1, bv_s2, bv_s3, bv_s4, bv_s5, bv_s6, bv_s7;
        __m128d bv_t1, bv_t2, bv_t3, bv_t4, bv_t5, bv_t6;

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

        bv_s1 = _mm_add_pd(bv_in1, bv_in4);
        bv_s2 = _mm_sub_pd(bv_in1, bv_in4);
        bv_s3 = _mm_add_pd(bv_in2, bv_in3);
        bv_s4 = _mm_sub_pd(bv_in2, bv_in3);
        bv_s5 = _mm_sub_pd(bv_s2, bv_s4);

        bv_s6 = _mm_add_pd(bv_s2, bv_s4);
        bv_t1 = _mm_mul_pd(v128_CRTM_5_4, bv_s5);
        bv_s7 = _mm_add_pd(bv_in0, bv_t1);
        bv_t2 = _mm_mul_pd(v128_CRTM_5_1, bv_s6);

        bv_t3 = _mm_mul_pd(v128_CRTM_5_2, bv_s3);
        bv_t4 = _mm_mul_pd(v128_CRTM_5_3, bv_s1);
        bv_t5 = _mm_mul_pd(v128_CRTM_5_3, bv_s3);
        bv_t6 = _mm_mul_pd(v128_CRTM_5_2, bv_s1);

        // Output point 2: X(1) & Output point 3: X(2)
        curr_out = out + out_strides[1];
        v_out1 = _mm_add_pd(bv_s7, bv_t2);
        v_out2 = _mm_sub_pd(NEGATE_128_D(bv_t4), bv_t3);
        STRI_2x128_D(curr_out, v_out_stride, v_out1, v_out2);
        // Output point 6: X(5) & Output point 7: X(6)
        curr_out = out + out_strides[5];
        v_out5 = _mm_sub_pd(bv_s7, bv_t2);
        v_out6 = _mm_sub_pd(bv_t5, bv_t6);
        STRI_2x128_D(curr_out, v_out_stride, v_out5, v_out6);
        // Output point 10: X(9)
        curr_out = out + out_strides[9];
        v_out9 = _mm_sub_pd(bv_in0, bv_s5);
        STR_128_D(curr_out, v_out_stride, v_out9);

        in = in + (v_in_stride << 1);
        out = out + (v_out_stride << 1);
    }
    // tail cases
    if (remaining_sets & 1)
    {
        /* Standard DFT */
        DOUBLE a_in0, a_in1, a_in2, a_in3, a_in4;
        DOUBLE a_s1, a_s2, a_s3, a_s4, a_s5, a_s6, a_s7;
        DOUBLE a_t1, a_t2, a_t3, a_t4, a_t5, a_t6;

        a_in0 = *in;               // Input point 1: x(0)
        a_in1 = in[in_strides[2]]; // Input point 3: x(2)
        a_in2 = in[in_strides[4]]; // Input point 5: x(4)
        a_in3 = in[in_strides[6]]; // Input point 7: x(6)
        a_in4 = in[in_strides[8]]; // Input point 9: x(8)

        a_s1 = a_in1 + a_in4;
        a_s2 = a_in4 - a_in1;
        a_s3 = a_in2 + a_in3;
        a_s4 = a_in2 - a_in3;
        a_s5 = a_s1 + a_s3;

        a_t1 = CRTM_5_4 * a_s5;
        a_s6 = a_s1 - a_s3;

        a_s7 = a_in0 - a_t1;
        a_t2 = CRTM_5_1 * a_s6;

        a_t3 = CRTM_5_3 * a_s4;
        a_t4 = CRTM_5_2 * a_s2;
        a_t5 = CRTM_5_2 * a_s4;
        a_t6 = CRTM_5_3 * a_s2;

        *out = a_in0 + a_s5;               // Output point 1: X(0)
        out[out_strides[3]] = a_s7 + a_t2; // Output point 4: X(3)
        out[out_strides[4]] = a_t4 - a_t3; // Output point 5: X(4)
        out[out_strides[7]] = a_s7 - a_t2; // Output point 8: X(7)
        out[out_strides[8]] = a_t5 + a_t6; // Output point 9: X(8)

        /* Shifted DFT */
        DOUBLE b_in0, b_in1, b_in2, b_in3, b_in4;
        DOUBLE b_s1, b_s2, b_s3, b_s4, b_s5, b_s6, b_s7;
        DOUBLE b_t1, b_t2, b_t3, b_t4, b_t5, b_t6;

        b_in0 = in[in_strides[1]]; // Input point 2: x(1)
        b_in1 = in[in_strides[3]]; // Input point 4: x(3)
        b_in2 = in[in_strides[5]]; // Input point 6: x(5)
        b_in3 = in[in_strides[7]]; // Input point 8: x(7)
        b_in4 = in[in_strides[9]]; // Input point 10: x(9)

        b_s1 = b_in1 + b_in4;
        b_s2 = b_in1 - b_in4;
        b_s3 = b_in2 + b_in3;
        b_s4 = b_in2 - b_in3;
        b_s5 = b_s2 - b_s4;

        b_s6 = b_s2 + b_s4;
        b_t1 = CRTM_5_4 * b_s5;
        b_s7 = b_in0 + b_t1;
        b_t2 = CRTM_5_1 * b_s6;

        b_t3 = CRTM_5_2 * b_s3;
        b_t4 = CRTM_5_3 * b_s1;
        b_t5 = CRTM_5_3 * b_s3;
        b_t6 = CRTM_5_2 * b_s1;

        out[out_strides[1]] = b_s7 + b_t2;  // Output point 2: X(1)
        out[out_strides[2]] = -b_t3 - b_t4; // Output point 3: X(2)
        out[out_strides[5]] = b_s7 - b_t2;  // Output point 6: X(5)
        out[out_strides[6]] = b_t5 - b_t6;  // Output point 7: X(6)
        out[out_strides[9]] = b_in0 - b_s5; // Output point 10: X(9)
    }
    AOCLFFTZ_LOG(DEBUG, global_logger_mode, "Exit");
}

static VOID r2hcf_rfft5avx512_fp64_bwd(VOID *in_real, VOID *in_imag,
                                       VOID *out_real, VOID *out_imag, INTP n,
                                       aoclfftz_strides_t *strides, VOID *twd, UINT8 flag)
{
    AOCLFFTZ_LOG(DEBUG, global_logger_mode, "Enter");
    const DOUBLE CRTM_5_1 = 1.11803398874989484820458683436563811772030918;
    const DOUBLE CRTM_5_2 = 1.90211303259030714423287866675876428681139726;
    const DOUBLE CRTM_5_3 = 1.17557050458494625833741190927814553719530488;
    const DOUBLE CRTM_5_4 = 0.50000000000000000000000000000000000000000000;
    const DOUBLE CRTM_5_5 = 2.000000000000000000000000000000000000000000000;

    DOUBLE *in = (DOUBLE *)in_real;
    DOUBLE *out = (DOUBLE *)out_real;
#ifdef VOLATILE_STRIDE_ARRAY
    volatile INTP *in_strides = strides->in_strides;
    volatile INTP *out_strides = strides->out_strides;
#else
    INTP *in_strides = strides->in_strides;
    INTP *out_strides = strides->out_strides;
#endif
    INTP v_in_stride = strides->v_in_stride;
    INTP v_out_stride = strides->v_out_stride;

    INTP cnt;
    DOUBLE *curr_in, *curr_out;
    INTP N = n / NUM_SETS_REAL_256_S;
    INTP remaining_sets = n % NUM_SETS_REAL_256_S;

    __m512d v_CRTM_5_1 = _mm512_set1_pd(CRTM_5_1);
    __m512d v_CRTM_5_2 = _mm512_set1_pd(CRTM_5_2);
    __m512d v_CRTM_5_3 = _mm512_set1_pd(CRTM_5_3);
    __m512d v_CRTM_5_4 = _mm512_set1_pd(CRTM_5_4);
    __m512d v_CRTM_5_5 = _mm512_set1_pd(CRTM_5_5);

    for (cnt = 0; cnt < N; cnt++)
    {
        /* Standard DFT */
        __m512d av_in0, av_in1, av_in2, av_in3, av_in4;
        __m512d av_s1, av_s2, av_s3, av_s4, av_s5, av_s6, av_s7;
        __m512d av_t1, av_t2, av_t3, av_t4, av_t5, av_t6, av_t7;
        __m512d v_out0, v_out1, v_out2, v_out3, v_out4, v_out5, v_out6, v_out7,
                v_out8, v_out9;

        curr_in = in;
        curr_out = out;

        // Input point 1: x(0)
        LDR_512_D(curr_in, v_in_stride, av_in0);
        // Input point 4: x(3) & Input point 5: x(4)
        curr_in = in + in_strides[3];
        LDRI_2x512_D(curr_in, v_in_stride, av_in1, av_in2);
        // Input point 8: x(7) & Input point 9: x(8)
        curr_in = in + in_strides[7];
        LDRI_2x512_D(curr_in, v_in_stride, av_in3, av_in4);

        av_s1 = _mm512_add_pd(av_in1, av_in3);
        av_s2 = _mm512_sub_pd(av_in1, av_in3);
        av_t1 = _mm512_mul_pd(v_CRTM_5_4, av_s1);
        av_t2 = _mm512_mul_pd(v_CRTM_5_1, av_s2);
        av_s3 = _mm512_sub_pd(av_in0, av_t1);
        av_s4 = _mm512_add_pd(av_s3, av_t2);
        av_s5 = _mm512_sub_pd(av_s3, av_t2);

        av_t3 = _mm512_mul_pd(v_CRTM_5_2, av_in4);
        av_t4 = _mm512_mul_pd(v_CRTM_5_3, av_in2);
        av_t5 = _mm512_mul_pd(v_CRTM_5_3, av_in4);
        av_t6 = _mm512_mul_pd(v_CRTM_5_2, av_in2);
        av_t7 = _mm512_mul_pd(v_CRTM_5_5, av_s1);

        av_s6 = _mm512_add_pd(av_t6, av_t5);
        av_s7 = _mm512_sub_pd(av_t3, av_t4);

        // Output pt 1: X(0)
        v_out0 = _mm512_add_pd(av_in0, av_t7);
        STR_512_D(curr_out, v_out_stride, v_out0);
        // Output pt 3: X(2)
        curr_out = out + out_strides[2];
        v_out2 = _mm512_sub_pd(av_s4, av_s6);
        STR_512_D(curr_out, v_out_stride, v_out2);
        // Output pt 5: X(4)
        curr_out = out + out_strides[4];
        v_out4 = _mm512_add_pd(av_s5, av_s7);
        STR_512_D(curr_out, v_out_stride, v_out4);
        // Output pt 7: X(6)
        curr_out = out + out_strides[6];
        v_out6 = _mm512_sub_pd(av_s5, av_s7);
        STR_512_D(curr_out, v_out_stride, v_out6);
        // Output pt 9: X(8)
        curr_out = out + out_strides[8];
        v_out8 = _mm512_add_pd(av_s4, av_s6);
        STR_512_D(curr_out, v_out_stride, v_out8);

        /* Shifted DFT */
        __m512d bv_in0, bv_in1, bv_in2, bv_in3, bv_in4;
        __m512d bv_s1, bv_s2, bv_s3, bv_s4, bv_s5, bv_s6, bv_s7;
        __m512d bv_t1, bv_t2, bv_t3, bv_t4, bv_t5, bv_t6, bv_t7;

        // Input point 2: x(1) & Input point 3: x(2)
        curr_in = in + in_strides[1];
        LDRI_2x512_D(curr_in, v_in_stride, bv_in0, bv_in1);
        // Input point 6: x(5) & Input point 7: x(6)
        curr_in = in + in_strides[5];
        LDRI_2x512_D(curr_in, v_in_stride, bv_in2, bv_in3);
        // Input point 10: x(9)
        curr_in = in + in_strides[9];
        LDR_512_D(curr_in, v_in_stride, bv_in4);

        bv_s1 = _mm512_add_pd(bv_in0, bv_in2);
        bv_s2 = _mm512_sub_pd(bv_in2, bv_in0);
        bv_t1 = _mm512_mul_pd(v_CRTM_5_1, bv_s2);
        bv_t2 = _mm512_mul_pd(v_CRTM_5_4, bv_s1);
        bv_s3 = _mm512_sub_pd(bv_in4, bv_t2);
        bv_s4 = _mm512_add_pd(bv_s3, bv_t1);
        bv_s5 = _mm512_sub_pd(bv_s3, bv_t1);

        bv_t3 = _mm512_mul_pd(v_CRTM_5_3, bv_in1);
        bv_t4 = _mm512_mul_pd(v_CRTM_5_2, bv_in3);
        bv_t5 = _mm512_mul_pd(v_CRTM_5_2, bv_in1);
        bv_t6 = _mm512_mul_pd(v_CRTM_5_3, bv_in3);
        bv_s6 = _mm512_add_pd(bv_t3, bv_t4);

        bv_s7 = _mm512_sub_pd(bv_t6, bv_t5);
        bv_t7 = _mm512_mul_pd(v_CRTM_5_5, bv_s1);

        // Output pt 2: X(1)
        curr_out = out + out_strides[1];
        v_out1 = _mm512_add_pd(bv_in4, bv_t7);
        STR_512_D(curr_out, v_out_stride, v_out1);
        // Output pt 4: X(3)
        curr_out = out + out_strides[3];
        v_out3 = _mm512_sub_pd(NEGATE_512_D(bv_s6), bv_s4);
        STR_512_D(curr_out, v_out_stride, v_out3);
        // Output pt 6: X(5)
        curr_out = out + out_strides[5];
        v_out5 = _mm512_add_pd(bv_s5, bv_s7);
        STR_512_D(curr_out, v_out_stride, v_out5);
        // Output pt 8: X(7)
        curr_out = out + out_strides[7];
        v_out7 = _mm512_sub_pd(bv_s7, bv_s5);
        STR_512_D(curr_out, v_out_stride, v_out7);
        // Output pt 10: X(9)
        curr_out = out + out_strides[9];
        v_out9 = _mm512_sub_pd(bv_s4, bv_s6);
        STR_512_D(curr_out, v_out_stride, v_out9);

        in = in + (v_in_stride << 3);
        out = out + (v_out_stride << 3);
    }
    // tail cases
    if (remaining_sets & NUM_SETS_REAL_256_D)
    {
        /* Standard DFT */
        __m256d av_in0, av_in1, av_in2, av_in3, av_in4;
        __m256d av_s1, av_s2, av_s3, av_s4, av_s5, av_s6, av_s7;
        __m256d av_t1, av_t2, av_t3, av_t4, av_t5, av_t6, av_t7;
        __m256d v_out0, v_out1, v_out2, v_out3, v_out4, v_out5, v_out6, v_out7,
                v_out8, v_out9;

        curr_in = in;
        curr_out = out;

        __m256d v256_CRTM_5_1 = _mm512_castpd512_pd256(v_CRTM_5_1);
        __m256d v256_CRTM_5_2 = _mm512_castpd512_pd256(v_CRTM_5_2);
        __m256d v256_CRTM_5_3 = _mm512_castpd512_pd256(v_CRTM_5_3);
        __m256d v256_CRTM_5_4 = _mm512_castpd512_pd256(v_CRTM_5_4);
        __m256d v256_CRTM_5_5 = _mm512_castpd512_pd256(v_CRTM_5_5);

        // Input point 1: x(0)
        LDR_256_D(curr_in, v_in_stride, av_in0);
        // Input point 4: x(3) & Input point 5: x(4)
        curr_in = in + in_strides[3];
        LDRI_2x256_D(curr_in, v_in_stride, av_in1, av_in2);
        // Input point 8: x(7) & Input point 9: x(8)
        curr_in = in + in_strides[7];
        LDRI_2x256_D(curr_in, v_in_stride, av_in3, av_in4);

        av_s1 = _mm256_add_pd(av_in1, av_in3);
        av_s2 = _mm256_sub_pd(av_in1, av_in3);
        av_t1 = _mm256_mul_pd(v256_CRTM_5_4, av_s1);
        av_t2 = _mm256_mul_pd(v256_CRTM_5_1, av_s2);
        av_s3 = _mm256_sub_pd(av_in0, av_t1);
        av_s4 = _mm256_add_pd(av_s3, av_t2);
        av_s5 = _mm256_sub_pd(av_s3, av_t2);

        av_t3 = _mm256_mul_pd(v256_CRTM_5_2, av_in4);
        av_t4 = _mm256_mul_pd(v256_CRTM_5_3, av_in2);
        av_t5 = _mm256_mul_pd(v256_CRTM_5_3, av_in4);
        av_t6 = _mm256_mul_pd(v256_CRTM_5_2, av_in2);
        av_t7 = _mm256_mul_pd(v256_CRTM_5_5, av_s1);

        av_s6 = _mm256_add_pd(av_t6, av_t5);
        av_s7 = _mm256_sub_pd(av_t3, av_t4);

        // Output pt 1: X(0)
        v_out0 = _mm256_add_pd(av_in0, av_t7);
        STR_256_D(curr_out, v_out_stride, v_out0);
        // Output pt 3: X(2)
        curr_out = out + out_strides[2];
        v_out2 = _mm256_sub_pd(av_s4, av_s6);
        STR_256_D(curr_out, v_out_stride, v_out2);
        // Output pt 5: X(4)
        curr_out = out + out_strides[4];
        v_out4 = _mm256_add_pd(av_s5, av_s7);
        STR_256_D(curr_out, v_out_stride, v_out4);
        // Output pt 7: X(6)
        curr_out = out + out_strides[6];
        v_out6 = _mm256_sub_pd(av_s5, av_s7);
        STR_256_D(curr_out, v_out_stride, v_out6);
        // Output pt 9: X(8)
        curr_out = out + out_strides[8];
        v_out8 = _mm256_add_pd(av_s4, av_s6);
        STR_256_D(curr_out, v_out_stride, v_out8);

        /* Shifted DFT */
        __m256d bv_in0, bv_in1, bv_in2, bv_in3, bv_in4;
        __m256d bv_s1, bv_s2, bv_s3, bv_s4, bv_s5, bv_s6, bv_s7;
        __m256d bv_t1, bv_t2, bv_t3, bv_t4, bv_t5, bv_t6, bv_t7;

        // Input point 2: x(1) & Input point 3: x(2)
        curr_in = in + in_strides[1];
        LDRI_2x256_D(curr_in, v_in_stride, bv_in0, bv_in1);
        // Input point 6: x(5) & Input point 7: x(6)
        curr_in = in + in_strides[5];
        LDRI_2x256_D(curr_in, v_in_stride, bv_in2, bv_in3);
        // Input point 10: x(9)
        curr_in = in + in_strides[9];
        LDR_256_D(curr_in, v_in_stride, bv_in4);

        bv_s1 = _mm256_add_pd(bv_in0, bv_in2);
        bv_s2 = _mm256_sub_pd(bv_in2, bv_in0);
        bv_t1 = _mm256_mul_pd(v256_CRTM_5_1, bv_s2);
        bv_t2 = _mm256_mul_pd(v256_CRTM_5_4, bv_s1);
        bv_s3 = _mm256_sub_pd(bv_in4, bv_t2);
        bv_s4 = _mm256_add_pd(bv_s3, bv_t1);
        bv_s5 = _mm256_sub_pd(bv_s3, bv_t1);

        bv_t3 = _mm256_mul_pd(v256_CRTM_5_3, bv_in1);
        bv_t4 = _mm256_mul_pd(v256_CRTM_5_2, bv_in3);
        bv_t5 = _mm256_mul_pd(v256_CRTM_5_2, bv_in1);
        bv_t6 = _mm256_mul_pd(v256_CRTM_5_3, bv_in3);
        bv_s6 = _mm256_add_pd(bv_t3, bv_t4);

        bv_s7 = _mm256_sub_pd(bv_t6, bv_t5);
        bv_t7 = _mm256_mul_pd(v256_CRTM_5_5, bv_s1);

        // Output pt 2: X(1)
        curr_out = out + out_strides[1];
        v_out1 = _mm256_add_pd(bv_in4, bv_t7);
        STR_256_D(curr_out, v_out_stride, v_out1);
        // Output pt 4: X(3)
        curr_out = out + out_strides[3];
        v_out3 = _mm256_sub_pd(NEGATE_256_D(bv_s6), bv_s4);
        STR_256_D(curr_out, v_out_stride, v_out3);
        // Output pt 6: X(5)
        curr_out = out + out_strides[5];
        v_out5 = _mm256_add_pd(bv_s5, bv_s7);
        STR_256_D(curr_out, v_out_stride, v_out5);
        // Output pt 8: X(7)
        curr_out = out + out_strides[7];
        v_out7 = _mm256_sub_pd(bv_s7, bv_s5);
        STR_256_D(curr_out, v_out_stride, v_out7);
        // Output pt 10: X(9)
        curr_out = out + out_strides[9];
        v_out9 = _mm256_sub_pd(bv_s4, bv_s6);
        STR_256_D(curr_out, v_out_stride, v_out9);

        in = in + (v_in_stride << 2);
        out = out + (v_out_stride << 2);
    }
    // tail cases
    if (remaining_sets & NUM_SETS_REAL_128_D)
    {
        /* Standard DFT */
        __m128d av_in0, av_in1, av_in2, av_in3, av_in4;
        __m128d av_s1, av_s2, av_s3, av_s4, av_s5, av_s6, av_s7;
        __m128d av_t1, av_t2, av_t3, av_t4, av_t5, av_t6, av_t7;
        __m128d v_out0, v_out1, v_out2, v_out3, v_out4, v_out5, v_out6, v_out7,
                v_out8, v_out9;

        curr_in = in;
        curr_out = out;

        __m128d v128_CRTM_5_1 = _mm512_castpd512_pd128(v_CRTM_5_1);
        __m128d v128_CRTM_5_2 = _mm512_castpd512_pd128(v_CRTM_5_2);
        __m128d v128_CRTM_5_3 = _mm512_castpd512_pd128(v_CRTM_5_3);
        __m128d v128_CRTM_5_4 = _mm512_castpd512_pd128(v_CRTM_5_4);
        __m128d v128_CRTM_5_5 = _mm512_castpd512_pd128(v_CRTM_5_5);

        // Input point 1: x(0)
        LDR_128_D(curr_in, v_in_stride, av_in0);
        // Input point 4: x(3) & Input point 5: x(4)
        curr_in = in + in_strides[3];
        LDRI_2x128_D(curr_in, v_in_stride, av_in1, av_in2);
        // Input point 8: x(7) & Input point 9: x(8)
        curr_in = in + in_strides[7];
        LDRI_2x128_D(curr_in, v_in_stride, av_in3, av_in4);

        av_s1 = _mm_add_pd(av_in1, av_in3);
        av_s2 = _mm_sub_pd(av_in1, av_in3);
        av_t1 = _mm_mul_pd(v128_CRTM_5_4, av_s1);
        av_t2 = _mm_mul_pd(v128_CRTM_5_1, av_s2);
        av_s3 = _mm_sub_pd(av_in0, av_t1);
        av_s4 = _mm_add_pd(av_s3, av_t2);
        av_s5 = _mm_sub_pd(av_s3, av_t2);

        av_t3 = _mm_mul_pd(v128_CRTM_5_2, av_in4);
        av_t4 = _mm_mul_pd(v128_CRTM_5_3, av_in2);
        av_t5 = _mm_mul_pd(v128_CRTM_5_3, av_in4);
        av_t6 = _mm_mul_pd(v128_CRTM_5_2, av_in2);
        av_t7 = _mm_mul_pd(v128_CRTM_5_5, av_s1);

        av_s6 = _mm_add_pd(av_t6, av_t5);
        av_s7 = _mm_sub_pd(av_t3, av_t4);

        // Output pt 1: X(0)
        v_out0 = _mm_add_pd(av_in0, av_t7);
        STR_128_D(curr_out, v_out_stride, v_out0);
        // Output pt 3: X(2)
        curr_out = out + out_strides[2];
        v_out2 = _mm_sub_pd(av_s4, av_s6);
        STR_128_D(curr_out, v_out_stride, v_out2);
        // Output pt 5: X(4)
        curr_out = out + out_strides[4];
        v_out4 = _mm_add_pd(av_s5, av_s7);
        STR_128_D(curr_out, v_out_stride, v_out4);
        // Output pt 7: X(6)
        curr_out = out + out_strides[6];
        v_out6 = _mm_sub_pd(av_s5, av_s7);
        STR_128_D(curr_out, v_out_stride, v_out6);
        // Output pt 9: X(8)
        curr_out = out + out_strides[8];
        v_out8 = _mm_add_pd(av_s4, av_s6);
        STR_128_D(curr_out, v_out_stride, v_out8);

        /* Shifted DFT */
        __m128d bv_in0, bv_in1, bv_in2, bv_in3, bv_in4;
        __m128d bv_s1, bv_s2, bv_s3, bv_s4, bv_s5, bv_s6, bv_s7;
        __m128d bv_t1, bv_t2, bv_t3, bv_t4, bv_t5, bv_t6, bv_t7;

        // Input point 2: x(1) & Input point 3: x(2)
        curr_in = in + in_strides[1];
        LDRI_2x128_D(curr_in, v_in_stride, bv_in0, bv_in1);
        // Input point 6: x(5) & Input point 7: x(6)
        curr_in = in + in_strides[5];
        LDRI_2x128_D(curr_in, v_in_stride, bv_in2, bv_in3);
        // Input point 10: x(9)
        curr_in = in + in_strides[9];
        LDR_128_D(curr_in, v_in_stride, bv_in4);

        bv_s1 = _mm_add_pd(bv_in0, bv_in2);
        bv_s2 = _mm_sub_pd(bv_in2, bv_in0);
        bv_t1 = _mm_mul_pd(v128_CRTM_5_1, bv_s2);
        bv_t2 = _mm_mul_pd(v128_CRTM_5_4, bv_s1);
        bv_s3 = _mm_sub_pd(bv_in4, bv_t2);
        bv_s4 = _mm_add_pd(bv_s3, bv_t1);
        bv_s5 = _mm_sub_pd(bv_s3, bv_t1);

        bv_t3 = _mm_mul_pd(v128_CRTM_5_3, bv_in1);
        bv_t4 = _mm_mul_pd(v128_CRTM_5_2, bv_in3);
        bv_t5 = _mm_mul_pd(v128_CRTM_5_2, bv_in1);
        bv_t6 = _mm_mul_pd(v128_CRTM_5_3, bv_in3);
        bv_s6 = _mm_add_pd(bv_t3, bv_t4);

        bv_s7 = _mm_sub_pd(bv_t6, bv_t5);
        bv_t7 = _mm_mul_pd(v128_CRTM_5_5, bv_s1);

        // Output pt 2: X(1)
        curr_out = out + out_strides[1];
        v_out1 = _mm_add_pd(bv_in4, bv_t7);
        STR_128_D(curr_out, v_out_stride, v_out1);
        // Output pt 4: X(3)
        curr_out = out + out_strides[3];
        v_out3 = _mm_sub_pd(NEGATE_128_D(bv_s6), bv_s4);
        STR_128_D(curr_out, v_out_stride, v_out3);
        // Output pt 6: X(5)
        curr_out = out + out_strides[5];
        v_out5 = _mm_add_pd(bv_s5, bv_s7);
        STR_128_D(curr_out, v_out_stride, v_out5);
        // Output pt 8: X(7)
        curr_out = out + out_strides[7];
        v_out7 = _mm_sub_pd(bv_s7, bv_s5);
        STR_128_D(curr_out, v_out_stride, v_out7);
        // Output pt 10: X(9)
        curr_out = out + out_strides[9];
        v_out9 = _mm_sub_pd(bv_s4, bv_s6);
        STR_128_D(curr_out, v_out_stride, v_out9);

        in = in + (v_in_stride << 1);
        out = out + (v_out_stride << 1);
    }
    // tail cases
    if (remaining_sets & 1)
    {
        /* Standard DFT */
        DOUBLE a_in0, a_in1, a_in2, a_in3, a_in4;
        DOUBLE a_s1, a_s2, a_s3, a_s4, a_s5, a_s6, a_s7;
        DOUBLE a_t1, a_t2, a_t3, a_t4, a_t5, a_t6, a_t7;

        a_in0 = *in;               // Input point 1: x(0)
        a_in1 = in[in_strides[3]]; // Input point 4: x(3)
        a_in2 = in[in_strides[4]]; // Input point 5: x(4)
        a_in3 = in[in_strides[7]]; // Input point 8: x(7)
        a_in4 = in[in_strides[8]]; // Input point 9: x(8)

        a_s1 = a_in1 + a_in3;
        a_s2 = a_in1 - a_in3;
        a_t1 = CRTM_5_4 * a_s1;
        a_t2 = CRTM_5_1 * a_s2;
        a_s3 = a_in0 - a_t1;
        a_s4 = a_s3 + a_t2;
        a_s5 = a_s3 - a_t2;

        a_t3 = CRTM_5_2 * a_in4;
        a_t4 = CRTM_5_3 * a_in2;
        a_t5 = CRTM_5_3 * a_in4;
        a_t6 = CRTM_5_2 * a_in2;
        a_t7 = CRTM_5_5 * a_s1;

        a_s6 = a_t6 + a_t5;
        a_s7 = a_t3 - a_t4;

        *out = a_in0 + a_t7;               // Output point 1: X(0)
        out[out_strides[2]] = a_s4 - a_s6; // Output point 3: X(2)
        out[out_strides[4]] = a_s5 + a_s7; // Output point 5: X(4)
        out[out_strides[6]] = a_s5 - a_s7; // Output point 7: X(6)
        out[out_strides[8]] = a_s4 + a_s6; // Output point 9: X(8)

        /* Shifted DFT */
        DOUBLE b_in0, b_in1, b_in2, b_in3, b_in4;
        DOUBLE b_s1, b_s2, b_s3, b_s4, b_s5, b_s6, b_s7;
        DOUBLE b_t1, b_t2, b_t3, b_t4, b_t5, b_t6, b_t7;

        b_in0 = in[in_strides[1]]; // Input point 2: x(1)
        b_in1 = in[in_strides[2]]; // Input point 3: x(2)
        b_in2 = in[in_strides[5]]; // Input point 6: x(5)
        b_in3 = in[in_strides[6]]; // Input point 7: x(6)
        b_in4 = in[in_strides[9]]; // Input point 10: x(9)

        b_s1 = b_in0 + b_in2;
        b_s2 = b_in2 - b_in0;
        b_t1 = CRTM_5_1 * b_s2;
        b_t2 = CRTM_5_4 * b_s1;
        b_s3 = b_in4 - b_t2;
        b_s4 = b_s3 + b_t1;
        b_s5 = b_s3 - b_t1;

        b_t3 = CRTM_5_3 * b_in1;
        b_t4 = CRTM_5_2 * b_in3;
        b_t5 = CRTM_5_2 * b_in1;
        b_t6 = CRTM_5_3 * b_in3;
        b_t7 = b_t4 + b_t3;

        b_s6 = b_t6 - b_t5;
        b_s7 = CRTM_5_5 * b_s1;

        out[out_strides[1]] = b_in4 + b_s7; // Output point 2: X(1)
        out[out_strides[3]] = -b_t7 - b_s4; // Output point 4: X(3)
        out[out_strides[5]] = b_s5 + b_s6;  // Output point 6: X(5)
        out[out_strides[7]] = b_s6 - b_s5;  // Output point 8: X(7)
        out[out_strides[9]] = b_s4 - b_t7;  // Output point 10: X(9)
    }
    AOCLFFTZ_LOG(DEBUG, global_logger_mode, "Exit");
}

kfft_ register_kernel_r2hcf_rfft5avx512(UINT8 precision, UINT8 direction)
{
    if (direction == FORWARD_FFT_DIR)
    {
        if (precision == DT_FLOAT)
        {
            return r2hcf_rfft5avx512_fp32_fwd;
        }
        else if (precision == DT_DOUBLE)
        {
            return r2hcf_rfft5avx512_fp64_fwd;
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
            return r2hcf_rfft5avx512_fp32_bwd;
        }
        else if (precision == DT_DOUBLE)
        {
            return r2hcf_rfft5avx512_fp64_bwd;
        }
        else
        {
            return NULL;
        }
    }
}
