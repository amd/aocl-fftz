// Copyright Advanced Micro Devices, Inc.
// SPDX-License-Identifier: BSD-3-Clause

/** @file rfft8avx256.c
 *
 *  @brief Radix-8 r2hc_fused Real-FFT kernel with with AVX-256 operations using
 *  x86 SIMD intrinsics.
 *
 *  This file contains the DIT radix-8 real-to-halfcomplex fused of two
 *  different implementations (Standard DFT and Shifted DFT that differs in DFT
 *  weight matrix) using x86 SIMD intrinsics for single-precision and
 *  double-precision inputs
 *
 *  @author Jeevanantham N
 */

#include "core/kernels/kernel.h"
#include "core/kernels/simd_includes/r2hc_simd_avx256.h"

static const ops_cycles_t ops_cnt[NUM_PRECISIONS][NUM_FFT_DIRS] =
                                                {{{0, 12, 42, 200, 122, 33},
                                                  {0, 16, 44, 200, 143, 33}},
                                                 {{0, 12, 42, 100,  14, 33},
                                                  {0, 16, 44, 100,  14, 33}}};

ops_cycles_t get_ops_cnt_r2hcf_rfft8avx256(FFTZ_UINT8 precision,
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

static FFTZ_VOID r2hcf_rfft8avx256_fp32_fwd(FFTZ_VOID *in_real,
                                            FFTZ_VOID *in_imag,
                                            FFTZ_VOID *out_real,
                                            FFTZ_VOID *out_imag, FFTZ_INTP n,
                                            aoclfftz_strides_t *strides,
                                            FFTZ_VOID *twd, FFTZ_UINT8 flag)
{
    AOCLFFTZ_LOG(DEBUG, global_logger_mode, "Enter");
    const FFTZ_FLOAT CRTM_8_1 =
        0.7071067811865475244008443621048490392848359377f;
    const FFTZ_FLOAT CRTM_8_2 =
        0.9238795325112867561281831893967882868224166259f;
    const FFTZ_FLOAT CRTM_8_3 =
        0.3826834323650897717284599840303988667613445625f;

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
    FFTZ_UINT8 is_contiguous_in = (v_in_stride == 1);
    FFTZ_UINT8 is_contiguous_out = (v_out_stride == 1);

    FFTZ_INTP cnt;
    FFTZ_FLOAT *curr_in, *curr_out;
    FFTZ_INTP N = n / NUM_SETS_REAL_256_S;
    FFTZ_INTP remaining_sets = n % NUM_SETS_REAL_256_S;

    __m256 v_CRTM_8_1 = _mm256_broadcast_ss(&CRTM_8_1);
    __m256 v_CRTM_8_2 = _mm256_broadcast_ss(&CRTM_8_2);
    __m256 v_CRTM_8_3 = _mm256_broadcast_ss(&CRTM_8_3);

    for (cnt = 0; cnt < N; cnt++)
    {
        // Standard DFT
        __m256 av_in0, av_in1, av_in2, av_in3, av_in4, av_in5, av_in6, av_in7;
        __m256 av_s0, av_s1, av_s2, av_s3, av_s4, av_s5, av_s6, av_s7, av_s8,
               av_s9, av_s10, av_s11;
        __m256 av_t0, av_t1;
        __m256 v_out0, v_out1, v_out2, v_out3, v_out4, v_out5, v_out6, v_out7,
               v_out8, v_out9, v_out10, v_out11, v_out12, v_out13, v_out14,
               v_out15;

        curr_in = in;
        curr_out = out;

        // Input point 1: x(0)
        LDR_256_S(curr_in, v_in_stride, av_in0, is_contiguous_in);
        // Input point 3: x(2)
        curr_in = in + in_strides[2];
        LDR_256_S(curr_in, v_in_stride, av_in1, is_contiguous_in);
        // Input point 5: x(4)
        curr_in = in + in_strides[4];
        LDR_256_S(curr_in, v_in_stride, av_in2, is_contiguous_in);
        // Input point 7: x(6)
        curr_in = in + in_strides[6];
        LDR_256_S(curr_in, v_in_stride, av_in3, is_contiguous_in);
        // Input point 9: x(8)
        curr_in = in + in_strides[8];
        LDR_256_S(curr_in, v_in_stride, av_in4, is_contiguous_in);
        // Input point 11: x(10)
        curr_in = in + in_strides[10];
        LDR_256_S(curr_in, v_in_stride, av_in5, is_contiguous_in);
        // Input point 13: x(12)
        curr_in = in + in_strides[12];
        LDR_256_S(curr_in, v_in_stride, av_in6, is_contiguous_in);
        // Input point 15: x(14)
        curr_in = in + in_strides[14];
        LDR_256_S(curr_in, v_in_stride, av_in7, is_contiguous_in);

        av_s0 = _mm256_add_ps(av_in7, av_in5);
        av_s1 = _mm256_sub_ps(av_in7, av_in5);
        av_s2 = _mm256_add_ps(av_in6, av_in2);
        av_s3 = _mm256_sub_ps(av_in6, av_in2);
        av_s4 = _mm256_add_ps(av_in4, av_in0);
        av_s5 = _mm256_sub_ps(av_in0, av_in4);
        av_s6 = _mm256_add_ps(av_in3, av_in1);
        av_s7 = _mm256_sub_ps(av_in3, av_in1);

        av_s8 = _mm256_add_ps(av_s6, av_s0);
        av_s9 = _mm256_add_ps(av_s4, av_s2);
        av_s10 = _mm256_sub_ps(av_s7, av_s1);
        av_s11 = _mm256_sub_ps(av_s0, av_s6);

        av_t0 = _mm256_mul_ps(v_CRTM_8_1, av_s10);
        av_t1 = _mm256_mul_ps(v_CRTM_8_1, av_s11);

        // Output pt 1: X(0)
        v_out0 = _mm256_add_ps(av_s9, av_s8);
        STR_256_S(curr_out, v_out_stride, v_out0, is_contiguous_out);
        // Output pt 4: X(3) & Output pt 5: X(4)
        curr_out = out + out_strides[3];
        v_out3 = _mm256_sub_ps(av_s5, av_t0);
        v_out4 = _mm256_add_ps(av_s3, av_t1);
        STRI_2x256_S(curr_out, v_out_stride, v_out3, v_out4);
        // Output pt 8: X(7) & Output pt 9: X(8)
        curr_out = out + out_strides[7];
        v_out7 = _mm256_sub_ps(av_s4, av_s2);
        v_out8 = _mm256_add_ps(av_s7, av_s1);
        STRI_2x256_S(curr_out, v_out_stride, v_out7, v_out8);
        // Output pt 12: X(11) & Output pt 13: X(12)
        curr_out = out + out_strides[11];
        v_out11 = _mm256_add_ps(av_t0, av_s5);
        v_out12 = _mm256_sub_ps(av_t1, av_s3);
        STRI_2x256_S(curr_out, v_out_stride, v_out11, v_out12);
        // Output pt 16: X(15)
        curr_out = out + out_strides[15];
        v_out15 = _mm256_sub_ps(av_s9, av_s8);
        STR_256_S(curr_out, v_out_stride, v_out15, is_contiguous_out);

        // Shifted DFT
        __m256 bv_in0, bv_in1, bv_in2, bv_in3, bv_in4, bv_in5, bv_in6, bv_in7;
        __m256 bv_s0, bv_s1, bv_s2, bv_s3, bv_s4, bv_s5, bv_s6, bv_s7, bv_s8,
               bv_s9, bv_s10, bv_s11, bv_s12, bv_s13;
        __m256 bv_t0, bv_t1, bv_t2, bv_t3, bv_t4, bv_t5, bv_t6, bv_t7, bv_t8,
               bv_t9;

        // Input point 2: x(1)
        curr_in = in + in_strides[1];
        LDR_256_S(curr_in, v_in_stride, bv_in0, is_contiguous_in);
        // Input point 4: x(3)
        curr_in = in + in_strides[3];
        LDR_256_S(curr_in, v_in_stride, bv_in1, is_contiguous_in);
        // Input point 6: x(5)
        curr_in = in + in_strides[5];
        LDR_256_S(curr_in, v_in_stride, bv_in2, is_contiguous_in);
        // Input point 8: x(7)
        curr_in = in + in_strides[7];
        LDR_256_S(curr_in, v_in_stride, bv_in3, is_contiguous_in);
        // Input point 10: x(9)
        curr_in = in + in_strides[9];
        LDR_256_S(curr_in, v_in_stride, bv_in4, is_contiguous_in);
        // Input point 12: x(11)
        curr_in = in + in_strides[11];
        LDR_256_S(curr_in, v_in_stride, bv_in5, is_contiguous_in);
        // Input point 14: x(13)
        curr_in = in + in_strides[13];
        LDR_256_S(curr_in, v_in_stride, bv_in6, is_contiguous_in);
        // Input point 16: x(15)
        curr_in = in + in_strides[15];
        LDR_256_S(curr_in, v_in_stride, bv_in7, is_contiguous_in);

        bv_s0 = _mm256_add_ps(bv_in6, bv_in2);
        bv_s1 = _mm256_sub_ps(bv_in6, bv_in2);
        bv_t0 = _mm256_mul_ps(v_CRTM_8_1, bv_s0);
        bv_t1 = _mm256_mul_ps(v_CRTM_8_1, bv_s1);
        bv_s2 = _mm256_add_ps(bv_in7, bv_in1);
        bv_s3 = _mm256_sub_ps(bv_in7, bv_in1);
        bv_s4 = _mm256_add_ps(bv_in5, bv_in3);
        bv_s5 = _mm256_sub_ps(bv_in5, bv_in3);

        bv_t2 = _mm256_mul_ps(v_CRTM_8_2, bv_s2);
        bv_t3 = _mm256_mul_ps(v_CRTM_8_3, bv_s2);
        bv_t4 = _mm256_mul_ps(v_CRTM_8_2, bv_s3);
        bv_t5 = _mm256_mul_ps(v_CRTM_8_3, bv_s3);

        bv_t6 = _mm256_mul_ps(v_CRTM_8_2, bv_s4);
        bv_t7 = _mm256_mul_ps(v_CRTM_8_3, bv_s4);

        bv_t8 = _mm256_mul_ps(v_CRTM_8_2, bv_s5);
        bv_t9 = _mm256_mul_ps(v_CRTM_8_3, bv_s5);

        bv_s6 = _mm256_sub_ps(bv_in0, bv_t1);
        bv_s7 = _mm256_add_ps(bv_t1, bv_in0);
        bv_s8 = _mm256_sub_ps(bv_in4, bv_t0);
        bv_s9 = _mm256_add_ps(bv_t0, bv_in4);
        bv_s10 = _mm256_add_ps(bv_t4, bv_t9);
        bv_s11 = _mm256_add_ps(bv_t6, bv_t3);
        bv_s12 = _mm256_sub_ps(bv_t8, bv_t5);
        bv_s13 = _mm256_sub_ps(bv_t7, bv_t2);

        // Output pt 2: X(1) & Output pt 3: X(2)
        curr_out = out + out_strides[1];
        v_out1 = _mm256_sub_ps(bv_s6, bv_s10);
        v_out2 = NEGATE_256_S(_mm256_add_ps(bv_s9, bv_s11));
        STRI_2x256_S(curr_out, v_out_stride, v_out1, v_out2);
        // Output pt 6: X(5) & Output pt 7: X(6)
        curr_out = out + out_strides[5];
        v_out5 = _mm256_add_ps(bv_s7, bv_s12);
        v_out6 = _mm256_add_ps(bv_s8, bv_s13);
        STRI_2x256_S(curr_out, v_out_stride, v_out5, v_out6);
        // Output pt 10: X(9) & Output pt 11: X(10)
        curr_out = out + out_strides[9];
        v_out9 = _mm256_sub_ps(bv_s7, bv_s12);
        v_out10 = _mm256_sub_ps(bv_s13, bv_s8);
        STRI_2x256_S(curr_out, v_out_stride, v_out9, v_out10);
        // Output pt 14: X(13)& Output pt 15: X(14)
        curr_out = out + out_strides[13];
        v_out13 = _mm256_add_ps(bv_s6, bv_s10);
        v_out14 = _mm256_sub_ps(bv_s9, bv_s11);
        STRI_2x256_S(curr_out, v_out_stride, v_out13, v_out14);

        in = in + (v_in_stride << 3);
        out = out + (v_out_stride << 3);
    }
    // tail cases
    if (remaining_sets & NUM_SETS_REAL_128_S)
    {
        // Standard DFT
        __m128 av_in0, av_in1, av_in2, av_in3, av_in4, av_in5, av_in6, av_in7;
        __m128 av_s0, av_s1, av_s2, av_s3, av_s4, av_s5, av_s6, av_s7, av_s8,
               av_s9, av_s10, av_s11;
        __m128 av_t0, av_t1;
        __m128 v_out0, v_out1, v_out2, v_out3, v_out4, v_out5, v_out6, v_out7,
               v_out8, v_out9, v_out10, v_out11, v_out12, v_out13, v_out14,
               v_out15;

        curr_in = in;
        curr_out = out;

        __m128 v128_CRTM_8_1 = _mm256_castps256_ps128(v_CRTM_8_1);
        __m128 v128_CRTM_8_2 = _mm256_castps256_ps128(v_CRTM_8_2);
        __m128 v128_CRTM_8_3 = _mm256_castps256_ps128(v_CRTM_8_3);

        // Input point 1: x(0)
        LDR_128_S(curr_in, v_in_stride, av_in0, is_contiguous_in);
        // Input point 3: x(2)
        curr_in = in + in_strides[2];
        LDR_128_S(curr_in, v_in_stride, av_in1, is_contiguous_in);
        // Input point 5: x(4)
        curr_in = in + in_strides[4];
        LDR_128_S(curr_in, v_in_stride, av_in2, is_contiguous_in);
        // Input point 7: x(6)
        curr_in = in + in_strides[6];
        LDR_128_S(curr_in, v_in_stride, av_in3, is_contiguous_in);
        // Input point 9: x(8)
        curr_in = in + in_strides[8];
        LDR_128_S(curr_in, v_in_stride, av_in4, is_contiguous_in);
        // Input point 11: x(10)
        curr_in = in + in_strides[10];
        LDR_128_S(curr_in, v_in_stride, av_in5, is_contiguous_in);
        // Input point 13: x(12)
        curr_in = in + in_strides[12];
        LDR_128_S(curr_in, v_in_stride, av_in6, is_contiguous_in);
        // Input point 15: x(14)
        curr_in = in + in_strides[14];
        LDR_128_S(curr_in, v_in_stride, av_in7, is_contiguous_in);

        av_s0 = _mm_add_ps(av_in7, av_in5);
        av_s1 = _mm_sub_ps(av_in7, av_in5);
        av_s2 = _mm_add_ps(av_in6, av_in2);
        av_s3 = _mm_sub_ps(av_in6, av_in2);
        av_s4 = _mm_add_ps(av_in4, av_in0);
        av_s5 = _mm_sub_ps(av_in0, av_in4);
        av_s6 = _mm_add_ps(av_in3, av_in1);
        av_s7 = _mm_sub_ps(av_in3, av_in1);

        av_s8 = _mm_add_ps(av_s6, av_s0);
        av_s9 = _mm_add_ps(av_s4, av_s2);
        av_s10 = _mm_sub_ps(av_s7, av_s1);
        av_s11 = _mm_sub_ps(av_s0, av_s6);

        av_t0 = _mm_mul_ps(v128_CRTM_8_1, av_s10);
        av_t1 = _mm_mul_ps(v128_CRTM_8_1, av_s11);

        // Output pt 1: X(0)
        v_out0 = _mm_add_ps(av_s9, av_s8);
        STR_128_S(curr_out, v_out_stride, v_out0, is_contiguous_out);
        // Output pt 4: X(3) & Output pt 5: X(4)
        curr_out = out + out_strides[3];
        v_out3 = _mm_sub_ps(av_s5, av_t0);
        v_out4 = _mm_add_ps(av_s3, av_t1);
        STRI_2x128_S(curr_out, v_out_stride, v_out3, v_out4);
        // Output pt 8: X(7) & Output pt 9: X(8)
        curr_out = out + out_strides[7];
        v_out7 = _mm_sub_ps(av_s4, av_s2);
        v_out8 = _mm_add_ps(av_s7, av_s1);
        STRI_2x128_S(curr_out, v_out_stride, v_out7, v_out8);
        // Output pt 12: X(11) & Output pt 13: X(12)
        curr_out = out + out_strides[11];
        v_out11 = _mm_add_ps(av_t0, av_s5);
        v_out12 = _mm_sub_ps(av_t1, av_s3);
        STRI_2x128_S(curr_out, v_out_stride, v_out11, v_out12);
        // Output pt 16: X(15)
        curr_out = out + out_strides[15];
        v_out15 = _mm_sub_ps(av_s9, av_s8);
        STR_128_S(curr_out, v_out_stride, v_out15, is_contiguous_out);

        // Shifted DFT
        __m128 bv_in0, bv_in1, bv_in2, bv_in3, bv_in4, bv_in5, bv_in6, bv_in7;
        __m128 bv_s0, bv_s1, bv_s2, bv_s3, bv_s4, bv_s5, bv_s6, bv_s7, bv_s8,
               bv_s9, bv_s10, bv_s11, bv_s12, bv_s13;
        __m128 bv_t0, bv_t1, bv_t2, bv_t3, bv_t4, bv_t5, bv_t6, bv_t7, bv_t8,
               bv_t9;

        // Input point 2: x(1)
        curr_in = in + in_strides[1];
        LDR_128_S(curr_in, v_in_stride, bv_in0, is_contiguous_in);
        // Input point 4: x(3)
        curr_in = in + in_strides[3];
        LDR_128_S(curr_in, v_in_stride, bv_in1, is_contiguous_in);
        // Input point 6: x(5)
        curr_in = in + in_strides[5];
        LDR_128_S(curr_in, v_in_stride, bv_in2, is_contiguous_in);
        // Input point 8: x(7)
        curr_in = in + in_strides[7];
        LDR_128_S(curr_in, v_in_stride, bv_in3, is_contiguous_in);
        // Input point 10: x(9)
        curr_in = in + in_strides[9];
        LDR_128_S(curr_in, v_in_stride, bv_in4, is_contiguous_in);
        // Input point 12: x(11)
        curr_in = in + in_strides[11];
        LDR_128_S(curr_in, v_in_stride, bv_in5, is_contiguous_in);
        // Input point 14: x(13)
        curr_in = in + in_strides[13];
        LDR_128_S(curr_in, v_in_stride, bv_in6, is_contiguous_in);
        // Input point 16: x(15)
        curr_in = in + in_strides[15];
        LDR_128_S(curr_in, v_in_stride, bv_in7, is_contiguous_in);

        bv_s0 = _mm_add_ps(bv_in6, bv_in2);
        bv_s1 = _mm_sub_ps(bv_in6, bv_in2);
        bv_t0 = _mm_mul_ps(v128_CRTM_8_1, bv_s0);
        bv_t1 = _mm_mul_ps(v128_CRTM_8_1, bv_s1);
        bv_s2 = _mm_add_ps(bv_in7, bv_in1);
        bv_s3 = _mm_sub_ps(bv_in7, bv_in1);
        bv_s4 = _mm_add_ps(bv_in5, bv_in3);
        bv_s5 = _mm_sub_ps(bv_in5, bv_in3);

        bv_t2 = _mm_mul_ps(v128_CRTM_8_2, bv_s2);
        bv_t3 = _mm_mul_ps(v128_CRTM_8_3, bv_s2);
        bv_t4 = _mm_mul_ps(v128_CRTM_8_2, bv_s3);
        bv_t5 = _mm_mul_ps(v128_CRTM_8_3, bv_s3);

        bv_t6 = _mm_mul_ps(v128_CRTM_8_2, bv_s4);
        bv_t7 = _mm_mul_ps(v128_CRTM_8_3, bv_s4);

        bv_t8 = _mm_mul_ps(v128_CRTM_8_2, bv_s5);
        bv_t9 = _mm_mul_ps(v128_CRTM_8_3, bv_s5);

        bv_s6 = _mm_sub_ps(bv_in0, bv_t1);
        bv_s7 = _mm_add_ps(bv_t1, bv_in0);
        bv_s8 = _mm_sub_ps(bv_in4, bv_t0);
        bv_s9 = _mm_add_ps(bv_t0, bv_in4);
        bv_s10 = _mm_add_ps(bv_t4, bv_t9);
        bv_s11 = _mm_add_ps(bv_t6, bv_t3);
        bv_s12 = _mm_sub_ps(bv_t8, bv_t5);
        bv_s13 = _mm_sub_ps(bv_t7, bv_t2);

        // Output pt 2: X(1) & Output pt 3: X(2)
        curr_out = out + out_strides[1];
        v_out1 = _mm_sub_ps(bv_s6, bv_s10);
        v_out2 = NEGATE_128_S(_mm_add_ps(bv_s9, bv_s11));
        STRI_2x128_S(curr_out, v_out_stride, v_out1, v_out2);
        // Output pt 6: X(5) & Output pt 7: X(6)
        curr_out = out + out_strides[5];
        v_out5 = _mm_add_ps(bv_s7, bv_s12);
        v_out6 = _mm_add_ps(bv_s8, bv_s13);
        STRI_2x128_S(curr_out, v_out_stride, v_out5, v_out6);
        // Output pt 10: X(9) & Output pt 11: X(10)
        curr_out = out + out_strides[9];
        v_out9 = _mm_sub_ps(bv_s7, bv_s12);
        v_out10 = _mm_sub_ps(bv_s13, bv_s8);
        STRI_2x128_S(curr_out, v_out_stride, v_out9, v_out10);
        // Output pt 14: X(13)& Output pt 15: X(14)
        curr_out = out + out_strides[13];
        v_out13 = _mm_add_ps(bv_s6, bv_s10);
        v_out14 = _mm_sub_ps(bv_s9, bv_s11);
        STRI_2x128_S(curr_out, v_out_stride, v_out13, v_out14);

        in = in + (v_in_stride << 2);
        out = out + (v_out_stride << 2);
    }
    // tail cases
    if (remaining_sets & 2)
    {
        // Standard DFT
        __m128 av_in0, av_in1, av_in2, av_in3, av_in4, av_in5, av_in6, av_in7;
        __m128 av_s0, av_s1, av_s2, av_s3, av_s4, av_s5, av_s6, av_s7, av_s8,
               av_s9, av_s10, av_s11;
        __m128 av_t0, av_t1;
        __m128 v_out0, v_out1, v_out2, v_out3, v_out4, v_out5, v_out6, v_out7,
               v_out8, v_out9, v_out10, v_out11, v_out12, v_out13, v_out14,
               v_out15;

        curr_in = in;
        curr_out = out;

        __m128 v128_CRTM_8_1 = _mm256_castps256_ps128(v_CRTM_8_1);
        __m128 v128_CRTM_8_2 = _mm256_castps256_ps128(v_CRTM_8_2);
        __m128 v128_CRTM_8_3 = _mm256_castps256_ps128(v_CRTM_8_3);

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
        // Input point 15: x(14)
        curr_in = in + in_strides[14];
        LDHR_128_S(curr_in, v_in_stride, av_in7);

        av_s0 = _mm_add_ps(av_in7, av_in5);
        av_s1 = _mm_sub_ps(av_in7, av_in5);
        av_s2 = _mm_add_ps(av_in6, av_in2);
        av_s3 = _mm_sub_ps(av_in6, av_in2);
        av_s4 = _mm_add_ps(av_in4, av_in0);
        av_s5 = _mm_sub_ps(av_in0, av_in4);
        av_s6 = _mm_add_ps(av_in3, av_in1);
        av_s7 = _mm_sub_ps(av_in3, av_in1);

        av_s8 = _mm_add_ps(av_s6, av_s0);
        av_s9 = _mm_add_ps(av_s4, av_s2);
        av_s10 = _mm_sub_ps(av_s7, av_s1);
        av_s11 = _mm_sub_ps(av_s0, av_s6);

        av_t0 = _mm_mul_ps(v128_CRTM_8_1, av_s10);
        av_t1 = _mm_mul_ps(v128_CRTM_8_1, av_s11);

        // Output pt 1: X(0)
        v_out0 = _mm_add_ps(av_s9, av_s8);
        STHR_128_S(curr_out, v_out_stride, v_out0);
        // Output pt 4: X(3) & Output pt 5: X(4)
        curr_out = out + out_strides[3];
        v_out3 = _mm_sub_ps(av_s5, av_t0);
        v_out4 = _mm_add_ps(av_s3, av_t1);
        STHRI_2x128_S(curr_out, v_out_stride, v_out3, v_out4);
        // Output pt 8: X(7) & Output pt 9: X(8)
        curr_out = out + out_strides[7];
        v_out7 = _mm_sub_ps(av_s4, av_s2);
        v_out8 = _mm_add_ps(av_s7, av_s1);
        STHRI_2x128_S(curr_out, v_out_stride, v_out7, v_out8);
        // Output pt 12: X(11) & Output pt 13: X(12)
        curr_out = out + out_strides[11];
        v_out11 = _mm_add_ps(av_t0, av_s5);
        v_out12 = _mm_sub_ps(av_t1, av_s3);
        STHRI_2x128_S(curr_out, v_out_stride, v_out11, v_out12);
        // Output pt 16: X(15)
        curr_out = out + out_strides[15];
        v_out15 = _mm_sub_ps(av_s9, av_s8);
        STHR_128_S(curr_out, v_out_stride, v_out15);

        // Shifted DFT
        __m128 bv_in0, bv_in1, bv_in2, bv_in3, bv_in4, bv_in5, bv_in6, bv_in7;
        __m128 bv_s0, bv_s1, bv_s2, bv_s3, bv_s4, bv_s5, bv_s6, bv_s7, bv_s8,
               bv_s9, bv_s10, bv_s11, bv_s12, bv_s13;
        __m128 bv_t0, bv_t1, bv_t2, bv_t3, bv_t4, bv_t5, bv_t6, bv_t7, bv_t8,
               bv_t9;

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
        // Input point 16: x(15)
        curr_in = in + in_strides[15];
        LDHR_128_S(curr_in, v_in_stride, bv_in7);

        bv_s0 = _mm_add_ps(bv_in6, bv_in2);
        bv_s1 = _mm_sub_ps(bv_in6, bv_in2);
        bv_t0 = _mm_mul_ps(v128_CRTM_8_1, bv_s0);
        bv_t1 = _mm_mul_ps(v128_CRTM_8_1, bv_s1);
        bv_s2 = _mm_add_ps(bv_in7, bv_in1);
        bv_s3 = _mm_sub_ps(bv_in7, bv_in1);
        bv_s4 = _mm_add_ps(bv_in5, bv_in3);
        bv_s5 = _mm_sub_ps(bv_in5, bv_in3);

        bv_t2 = _mm_mul_ps(v128_CRTM_8_2, bv_s2);
        bv_t3 = _mm_mul_ps(v128_CRTM_8_3, bv_s2);
        bv_t4 = _mm_mul_ps(v128_CRTM_8_2, bv_s3);
        bv_t5 = _mm_mul_ps(v128_CRTM_8_3, bv_s3);

        bv_t6 = _mm_mul_ps(v128_CRTM_8_2, bv_s4);
        bv_t7 = _mm_mul_ps(v128_CRTM_8_3, bv_s4);

        bv_t8 = _mm_mul_ps(v128_CRTM_8_2, bv_s5);
        bv_t9 = _mm_mul_ps(v128_CRTM_8_3, bv_s5);

        bv_s6 = _mm_sub_ps(bv_in0, bv_t1);
        bv_s7 = _mm_add_ps(bv_t1, bv_in0);
        bv_s8 = _mm_sub_ps(bv_in4, bv_t0);
        bv_s9 = _mm_add_ps(bv_t0, bv_in4);
        bv_s10 = _mm_add_ps(bv_t4, bv_t9);
        bv_s11 = _mm_add_ps(bv_t6, bv_t3);
        bv_s12 = _mm_sub_ps(bv_t8, bv_t5);
        bv_s13 = _mm_sub_ps(bv_t7, bv_t2);

        // Output pt 2: X(1) & Output pt 3: X(2)
        curr_out = out + out_strides[1];
        v_out1 = _mm_sub_ps(bv_s6, bv_s10);
        v_out2 = NEGATE_128_S(_mm_add_ps(bv_s9, bv_s11));
        STHRI_2x128_S(curr_out, v_out_stride, v_out1, v_out2);
        // Output pt 6: X(5) & Output pt 7: X(6)
        curr_out = out + out_strides[5];
        v_out5 = _mm_add_ps(bv_s7, bv_s12);
        v_out6 = _mm_add_ps(bv_s8, bv_s13);
        STHRI_2x128_S(curr_out, v_out_stride, v_out5, v_out6);
        // Output pt 10: X(9) & Output pt 11: X(10)
        curr_out = out + out_strides[9];
        v_out9 = _mm_sub_ps(bv_s7, bv_s12);
        v_out10 = _mm_sub_ps(bv_s13, bv_s8);
        STHRI_2x128_S(curr_out, v_out_stride, v_out9, v_out10);
        // Output pt 14: X(13)& Output pt 15: X(14)
        curr_out = out + out_strides[13];
        v_out13 = _mm_add_ps(bv_s6, bv_s10);
        v_out14 = _mm_sub_ps(bv_s9, bv_s11);
        STHRI_2x128_S(curr_out, v_out_stride, v_out13, v_out14);

        in = in + (v_in_stride << 1);
        out = out + (v_out_stride << 1);
    }
    // tail cases
    if (remaining_sets & 1)
    {
        // Standard DFT
        FFTZ_FLOAT a_in0, a_in1, a_in2, a_in3, a_in4, a_in5, a_in6, a_in7;
        FFTZ_FLOAT a_s0, a_s1, a_s2, a_s3, a_s4, a_s5, a_s6, a_s7, a_s8, a_s9,
            a_s10, a_s11;
        FFTZ_FLOAT a_t0, a_t1;

        a_in0 = *in;                // Input point 1: x(0)
        a_in1 = in[in_strides[2]];  // Input point 3: x(2)
        a_in2 = in[in_strides[4]];  // Input point 5: x(4)
        a_in3 = in[in_strides[6]];  // Input point 7: x(6)
        a_in4 = in[in_strides[8]];  // Input point 9: x(8)
        a_in5 = in[in_strides[10]]; // Input point 11: x(10)
        a_in6 = in[in_strides[12]]; // Input point 13: x(12)
        a_in7 = in[in_strides[14]]; // Input point 13: x(12)

        a_s0 = a_in7 + a_in5;
        a_s1 = a_in7 - a_in5;
        a_s2 = a_in6 + a_in2;
        a_s3 = a_in6 - a_in2;
        a_s4 = a_in0 + a_in4;
        a_s5 = a_in0 - a_in4;
        a_s6 = a_in3 + a_in1;
        a_s7 = a_in3 - a_in1;

        a_s8 = a_s7 - a_s1;
        a_s9 = a_s0 - a_s6;
        a_s10 = a_s0 + a_s6;
        a_s11 = a_s4 + a_s2;
        a_t0 = CRTM_8_1 * a_s8;
        a_t1 = CRTM_8_1 * a_s9;

        *out = a_s11 + a_s10;                 // Output pt 1: X(0)
        out[out_strides[3]] = a_s5 - a_t0;    // Output pt 4: X(3)
        out[out_strides[4]] = a_s3 + a_t1;    // Output pt 5: X(4)
        out[out_strides[7]] = a_s4 - a_s2;    // Output pt 8: X(7)
        out[out_strides[8]] = a_s7 + a_s1;    // Output pt 9: X(8)
        out[out_strides[11]] = a_t0 + a_s5;   // Output pt 12: X(11)
        out[out_strides[12]] = a_t1 - a_s3;   // Output pt 13: X(12)
        out[out_strides[15]] = a_s11 - a_s10; // Output pt 16: X(15)

        // Shifted DFT
        FFTZ_FLOAT b_in0, b_in1, b_in2, b_in3, b_in4, b_in5, b_in6, b_in7;
        FFTZ_FLOAT b_s0, b_s1, b_s2, b_s3, b_s4, b_s5, b_s6, b_s7, b_s8, b_s9,
            b_s10, b_s11, b_s12, b_s13;
        FFTZ_FLOAT b_t0, b_t1, b_t2, b_t3, b_t4, b_t5, b_t6, b_t7, b_t8, b_t9;

        b_in0 = in[in_strides[1]];  // Input point 2: x(1)
        b_in1 = in[in_strides[3]];  // Input point 4: x(3)
        b_in2 = in[in_strides[5]];  // Input point 6: x(5)
        b_in3 = in[in_strides[7]];  // Input point 8: x(7)
        b_in4 = in[in_strides[9]];  // Input point 10: x(9)
        b_in5 = in[in_strides[11]]; // Input point 12: x(11)
        b_in6 = in[in_strides[13]]; // Input point 14: x(13)
        b_in7 = in[in_strides[15]]; // Input point 14: x(13)

        b_s0 = b_in6 + b_in2;
        b_s1 = b_in6 - b_in2;
        b_s2 = b_in7 + b_in1;
        b_s3 = b_in7 - b_in1;
        b_s4 = b_in5 + b_in3;
        b_s5 = b_in5 - b_in3;

        b_t0 = CRTM_8_1 * b_s0;
        b_t1 = CRTM_8_1 * b_s1;
        b_t2 = CRTM_8_2 * b_s2;
        b_t3 = CRTM_8_3 * b_s2;
        b_t4 = CRTM_8_2 * b_s3;
        b_t5 = CRTM_8_3 * b_s3;
        b_t6 = CRTM_8_2 * b_s4;
        b_t7 = CRTM_8_3 * b_s4;
        b_t8 = CRTM_8_2 * b_s5;
        b_t9 = CRTM_8_3 * b_s5;

        b_s6 = b_in0 - b_t1;
        b_s7 = b_in0 + b_t1;
        b_s8 = b_in4 - b_t0;
        b_s9 = b_t0 + b_in4;
        b_s10 = b_t4 + b_t9;
        b_s11 = b_t6 + b_t3;
        b_s12 = b_t8 - b_t5;
        b_s13 = b_t7 - b_t2;

        out[out_strides[1]] = b_s6 - b_s10;  // Output pt 2: X(1)
        out[out_strides[2]] = -b_s9 - b_s11; // Output pt 3: X(2)
        out[out_strides[5]] = b_s7 + b_s12;  // Output pt 6: X(5)
        out[out_strides[6]] = b_s8 + b_s13;  // Output pt 7: X(6)
        out[out_strides[9]] = b_s7 - b_s12;  // Output pt 10: X(9)
        out[out_strides[10]] = b_s13 - b_s8; // Output pt 11: X(10)
        out[out_strides[13]] = b_s6 + b_s10; // Output pt 14: X(12)
        out[out_strides[14]] = b_s9 - b_s11; // Output pt 15: X(14)
    }
    AOCLFFTZ_LOG(DEBUG, global_logger_mode, "Exit");
}

static FFTZ_VOID r2hcf_rfft8avx256_fp32_bwd(FFTZ_VOID *in_real,
                                            FFTZ_VOID *in_imag,
                                            FFTZ_VOID *out_real,
                                            FFTZ_VOID *out_imag, FFTZ_INTP n,
                                            aoclfftz_strides_t *strides,
                                            FFTZ_VOID *twd, FFTZ_UINT8 flag)
{
    AOCLFFTZ_LOG(DEBUG, global_logger_mode, "Enter");
    const FFTZ_FLOAT CRTM_8_1 =
        1.414213562373095048801688724209698078569671875f;
    const FFTZ_FLOAT CRTM_8_2 =
        1.847759065022573256256366378793576573644833252f;
    const FFTZ_FLOAT CRTM_8_3 =
        0.765366864730179543456919968060797733522689125f;
    const FFTZ_FLOAT CRTM_8_4 =
        2.000000000000000000000000000000000000000000000f;

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
    FFTZ_UINT8 is_contiguous_in = (v_in_stride == 1);
    FFTZ_UINT8 is_contiguous_out = (v_out_stride == 1);

    FFTZ_INTP cnt;
    FFTZ_FLOAT *curr_in, *curr_out;
    FFTZ_INTP N = n / NUM_SETS_REAL_256_S;
    FFTZ_INTP remaining_sets = n % NUM_SETS_REAL_256_S;

    __m256 v_CRTM_8_1 = _mm256_broadcast_ss(&CRTM_8_1);
    __m256 v_CRTM_8_2 = _mm256_broadcast_ss(&CRTM_8_2);
    __m256 v_CRTM_8_3 = _mm256_broadcast_ss(&CRTM_8_3);
    __m256 v_CRTM_8_4 = _mm256_broadcast_ss(&CRTM_8_4);

    for (cnt = 0; cnt < N; cnt++)
    {
        // Standard DFT
        __m256 av_in0, av_in1, av_in2, av_in3, av_in4, av_in5, av_in6, av_in7;
        __m256 av_s0, av_s1, av_s2, av_s3, av_s4, av_s5, av_s6, av_s7, av_s8,
               av_s9, av_s10, av_s11, av_s12, av_s13;
        __m256 av_t0, av_t1, av_t2, av_t3;
        __m256 v_out0, v_out1, v_out2, v_out3, v_out4, v_out5, v_out6, v_out7,
               v_out8, v_out9, v_out10, v_out11, v_out12, v_out13, v_out14,
               v_out15;

        curr_in = in;
        curr_out = out;

        // Input point 1: x(0)
        LDR_256_S(curr_in, v_in_stride, av_in0, is_contiguous_in);
        // Input point 4: x(3) & Input point 5: x(4)
        curr_in = in + in_strides[3];
        LDRI_2x256_S(curr_in, v_in_stride, av_in1, av_in2);
        // Input point 8: x(7) & Input point 9: x(8)
        curr_in = in + in_strides[7];
        LDRI_2x256_S(curr_in, v_in_stride, av_in3, av_in4);
        // Input point 12: x(11) & Input point 13: x(12)
        curr_in = in + in_strides[11];
        LDRI_2x256_S(curr_in, v_in_stride, av_in5, av_in6);
        // Input point 16: x(15) & Input point 17: x(16)
        curr_in = in + in_strides[15];
        LDR_256_S(curr_in, v_in_stride, av_in7, is_contiguous_in);

        av_s0 = _mm256_add_ps(av_in7, av_in0);
        av_s1 = _mm256_sub_ps(av_in0, av_in7);
        av_s2 = _mm256_add_ps(av_in6, av_in2);
        av_s3 = _mm256_sub_ps(av_in6, av_in2);
        av_s4 = _mm256_add_ps(av_in5, av_in1);
        av_t0 = _mm256_mul_ps(v_CRTM_8_4, av_s3);
        av_t1 = _mm256_mul_ps(v_CRTM_8_4, av_s4);
        av_s5 = _mm256_sub_ps(av_in5, av_in1);
        av_s6 = _mm256_add_ps(av_in4, av_in4);
        av_s7 = _mm256_add_ps(av_in3, av_in3);

        av_s8 = _mm256_add_ps(av_s5, av_s2);
        av_s9 = _mm256_sub_ps(av_s5, av_s2);
        av_t2 = _mm256_mul_ps(v_CRTM_8_1, av_s8);
        av_t3 = _mm256_mul_ps(v_CRTM_8_1, av_s9);
        av_s10 = _mm256_add_ps(av_s7, av_s0);
        av_s11 = _mm256_sub_ps(av_s0, av_s7);
        av_s12 = _mm256_sub_ps(av_s1, av_s6);
        av_s13 = _mm256_add_ps(av_s6, av_s1);

        // Output pt 1: X(0)
        v_out0 = _mm256_add_ps(av_s10, av_t1);
        STR_256_S(curr_out, v_out_stride, v_out0, is_contiguous_out);
        // Output point 3: X(2)
        curr_out = out + out_strides[2];
        v_out2 = _mm256_sub_ps(av_s12, av_t2);
        STR_256_S(curr_out, v_out_stride, v_out2, is_contiguous_out);
        // Output pt 5: X(4)
        curr_out = out + out_strides[4];
        v_out4 = _mm256_add_ps(av_t0, av_s11);
        STR_256_S(curr_out, v_out_stride, v_out4, is_contiguous_out);
        // Output pt 7: X(6)
        curr_out = out + out_strides[6];
        v_out6 = _mm256_add_ps(av_s13, av_t3);
        STR_256_S(curr_out, v_out_stride, v_out6, is_contiguous_out);
        // Output pt 9: X(8)
        curr_out = out + out_strides[8];
        v_out8 = _mm256_sub_ps(av_s10, av_t1);
        STR_256_S(curr_out, v_out_stride, v_out8, is_contiguous_out);
        // Output pt 11: X(10)
        curr_out = out + out_strides[10];
        v_out10 = _mm256_add_ps(av_t2, av_s12);
        STR_256_S(curr_out, v_out_stride, v_out10, is_contiguous_out);
        // Output pt 13: X(12)
        curr_out = out + out_strides[12];
        v_out12 = _mm256_sub_ps(av_s11, av_t0);
        STR_256_S(curr_out, v_out_stride, v_out12, is_contiguous_out);
        // Output pt 15: X(14)
        curr_out = out + out_strides[14];
        v_out14 = _mm256_sub_ps(av_s13, av_t3);
        STR_256_S(curr_out, v_out_stride, v_out14, is_contiguous_out);

        // Shifted DFT
        __m256 bv_in0, bv_in1, bv_in2, bv_in3, bv_in4, bv_in5, bv_in6, bv_in7;
        __m256 bv_s0, bv_s1, bv_s2, bv_s3, bv_s4, bv_s5, bv_s6, bv_s7, bv_s8,
               bv_s9, bv_s10, bv_s11, bv_s12, bv_s13, bv_s14, bv_s15;
        __m256 bv_t0, bv_t1, bv_t2, bv_t3, bv_t4, bv_t5, bv_t6, bv_t7, bv_t8,
               bv_t9;

        // Input point 2: x(1) & Input point 3: x(2)
        curr_in = in + in_strides[1];
        LDRI_2x256_S(curr_in, v_in_stride, bv_in0, bv_in1);
        // Input point 6: x(5) & Input point 7: x(6)
        curr_in = in + in_strides[5];
        LDRI_2x256_S(curr_in, v_in_stride, bv_in2, bv_in3);
        // Input point 10: x(9) & Input point 11: x(10)
        curr_in = in + in_strides[9];
        LDRI_2x256_S(curr_in, v_in_stride, bv_in4, bv_in5);
        // Input point 14: x(13) & Input point 15: x(14)
        curr_in = in + in_strides[13];
        LDRI_2x256_S(curr_in, v_in_stride, bv_in6, bv_in7);

        bv_s0 = _mm256_add_ps(bv_in6, bv_in0);
        bv_s1 = _mm256_sub_ps(bv_in6, bv_in0);
        bv_s2 = _mm256_add_ps(bv_in4, bv_in2);
        bv_s3 = _mm256_sub_ps(bv_in4, bv_in2);
        bv_s4 = _mm256_add_ps(bv_in7, bv_in1);
        bv_s5 = _mm256_sub_ps(bv_in7, bv_in1);
        bv_s6 = _mm256_add_ps(bv_in5, bv_in3);
        bv_s7 = _mm256_sub_ps(bv_in5, bv_in3);
        bv_s8 = _mm256_add_ps(bv_s7, bv_s5);
        bv_s9 = _mm256_sub_ps(bv_s0, bv_s2);

        bv_t0 = _mm256_mul_ps(v_CRTM_8_1, bv_s8);
        bv_t1 = _mm256_mul_ps(v_CRTM_8_1, bv_s9);
        bv_s10 = _mm256_sub_ps(bv_s6, bv_s1);
        bv_s11 = _mm256_sub_ps(bv_s3, bv_s4);
        bv_s12 = _mm256_add_ps(bv_s4, bv_s3);
        bv_s13 = _mm256_add_ps(bv_s6, bv_s1);
        bv_s14 = _mm256_add_ps(bv_s0, bv_s2);
        bv_t2 = _mm256_mul_ps(v_CRTM_8_2, bv_s13);

        bv_t3 = _mm256_mul_ps(v_CRTM_8_3, bv_s12);
        bv_t4 = _mm256_mul_ps(v_CRTM_8_2, bv_s11);
        bv_t5 = _mm256_mul_ps(v_CRTM_8_3, bv_s10);
        bv_s15 = _mm256_sub_ps(bv_s5, bv_s7);
        bv_t6 = _mm256_mul_ps(v_CRTM_8_2, bv_s12);
        bv_t7 = _mm256_mul_ps(v_CRTM_8_3, bv_s13);
        bv_t8 = _mm256_mul_ps(v_CRTM_8_2, bv_s10);
        bv_t9 = _mm256_mul_ps(v_CRTM_8_3, bv_s11);

        // Output pt 2: X(1)
        curr_out = out + out_strides[1];
        v_out1 = _mm256_mul_ps(v_CRTM_8_4, bv_s14);
        STR_256_S(curr_out, v_out_stride, v_out1, is_contiguous_out);
        // Output pt 4: X(3)
        curr_out = out + out_strides[3];
        v_out3 = NEGATE_256_S(_mm256_add_ps(bv_t2, bv_t3));
        STR_256_S(curr_out, v_out_stride, v_out3, is_contiguous_out);
        // Output pt 6: X(5)
        curr_out = out + out_strides[5];
        v_out5 = _mm256_add_ps(bv_t0, bv_t1);
        STR_256_S(curr_out, v_out_stride, v_out5, is_contiguous_out);
        // Output pt 8: X(7)
        curr_out = out + out_strides[7];
        v_out7 = _mm256_add_ps(bv_t5, bv_t4);
        STR_256_S(curr_out, v_out_stride, v_out7, is_contiguous_out);
        // Output pt 10: X(9)
        curr_out = out + out_strides[9];
        v_out9 = _mm256_mul_ps(v_CRTM_8_4, bv_s15);
        STR_256_S(curr_out, v_out_stride, v_out9, is_contiguous_out);
        // Output pt 12: X(11)
        curr_out = out + out_strides[11];
        v_out11 = _mm256_sub_ps(bv_t7, bv_t6);
        STR_256_S(curr_out, v_out_stride, v_out11, is_contiguous_out);
        // Output pt 14: X(13)
        curr_out = out + out_strides[13];
        v_out13 = _mm256_sub_ps(bv_t0, bv_t1);
        STR_256_S(curr_out, v_out_stride, v_out13, is_contiguous_out);
        // Output pt 16: X(15)
        curr_out = out + out_strides[15];
        v_out15 = _mm256_sub_ps(bv_t9, bv_t8);
        STR_256_S(curr_out, v_out_stride, v_out15, is_contiguous_out);

        in = in + (v_in_stride << 3);
        out = out + (v_out_stride << 3);
    }
    // tail cases
    if (remaining_sets & NUM_SETS_REAL_128_S)
    {
        // Standard DFT
        __m128 av_in0, av_in1, av_in2, av_in3, av_in4, av_in5, av_in6, av_in7;
        __m128 av_s0, av_s1, av_s2, av_s3, av_s4, av_s5, av_s6, av_s7, av_s8,
               av_s9, av_s10, av_s11, av_s12, av_s13;
        __m128 av_t0, av_t1, av_t2, av_t3;
        __m128 v_out0, v_out1, v_out2, v_out3, v_out4, v_out5, v_out6, v_out7,
               v_out8, v_out9, v_out10, v_out11, v_out12, v_out13, v_out14,
               v_out15;

        curr_in = in;
        curr_out = out;

        __m128 v128_CRTM_8_1 = _mm256_castps256_ps128(v_CRTM_8_1);
        __m128 v128_CRTM_8_2 = _mm256_castps256_ps128(v_CRTM_8_2);
        __m128 v128_CRTM_8_3 = _mm256_castps256_ps128(v_CRTM_8_3);
        __m128 v128_CRTM_8_4 = _mm256_castps256_ps128(v_CRTM_8_4);

        // Input point 1: x(0)
        LDR_128_S(curr_in, v_in_stride, av_in0, is_contiguous_in);
        // Input point 4: x(3) & Input point 5: x(4)
        curr_in = in + in_strides[3];
        LDRI_2x128_S(curr_in, v_in_stride, av_in1, av_in2);
        // Input point 8: x(7) & Input point 9: x(8)
        curr_in = in + in_strides[7];
        LDRI_2x128_S(curr_in, v_in_stride, av_in3, av_in4);
        // Input point 12: x(11) & Input point 13: x(12)
        curr_in = in + in_strides[11];
        LDRI_2x128_S(curr_in, v_in_stride, av_in5, av_in6);
        // Input point 16: x(15) & Input point 17: x(16)
        curr_in = in + in_strides[15];
        LDR_128_S(curr_in, v_in_stride, av_in7, is_contiguous_in);

        av_s0 = _mm_add_ps(av_in7, av_in0);
        av_s1 = _mm_sub_ps(av_in0, av_in7);
        av_s2 = _mm_add_ps(av_in6, av_in2);
        av_s3 = _mm_sub_ps(av_in6, av_in2);
        av_s4 = _mm_add_ps(av_in5, av_in1);
        av_t0 = _mm_mul_ps(v128_CRTM_8_4, av_s3);
        av_t1 = _mm_mul_ps(v128_CRTM_8_4, av_s4);
        av_s5 = _mm_sub_ps(av_in5, av_in1);
        av_s6 = _mm_add_ps(av_in4, av_in4);
        av_s7 = _mm_add_ps(av_in3, av_in3);

        av_s8 = _mm_add_ps(av_s5, av_s2);
        av_s9 = _mm_sub_ps(av_s5, av_s2);
        av_t2 = _mm_mul_ps(v128_CRTM_8_1, av_s8);
        av_t3 = _mm_mul_ps(v128_CRTM_8_1, av_s9);
        av_s10 = _mm_add_ps(av_s7, av_s0);
        av_s11 = _mm_sub_ps(av_s0, av_s7);
        av_s12 = _mm_sub_ps(av_s1, av_s6);
        av_s13 = _mm_add_ps(av_s6, av_s1);

        // Output pt 1: X(0)
        v_out0 = _mm_add_ps(av_s10, av_t1);
        STR_128_S(curr_out, v_out_stride, v_out0, is_contiguous_out);
        // Output point 3: X(2)
        curr_out = out + out_strides[2];
        v_out2 = _mm_sub_ps(av_s12, av_t2);
        STR_128_S(curr_out, v_out_stride, v_out2, is_contiguous_out);
        // Output pt 5: X(4)
        curr_out = out + out_strides[4];
        v_out4 = _mm_add_ps(av_t0, av_s11);
        STR_128_S(curr_out, v_out_stride, v_out4, is_contiguous_out);
        // Output pt 7: X(6)
        curr_out = out + out_strides[6];
        v_out6 = _mm_add_ps(av_s13, av_t3);
        STR_128_S(curr_out, v_out_stride, v_out6, is_contiguous_out);
        // Output pt 9: X(8)
        curr_out = out + out_strides[8];
        v_out8 = _mm_sub_ps(av_s10, av_t1);
        STR_128_S(curr_out, v_out_stride, v_out8, is_contiguous_out);
        // Output pt 11: X(10)
        curr_out = out + out_strides[10];
        v_out10 = _mm_add_ps(av_t2, av_s12);
        STR_128_S(curr_out, v_out_stride, v_out10, is_contiguous_out);
        // Output pt 13: X(12)
        curr_out = out + out_strides[12];
        v_out12 = _mm_sub_ps(av_s11, av_t0);
        STR_128_S(curr_out, v_out_stride, v_out12, is_contiguous_out);
        // Output pt 15: X(14)
        curr_out = out + out_strides[14];
        v_out14 = _mm_sub_ps(av_s13, av_t3);
        STR_128_S(curr_out, v_out_stride, v_out14, is_contiguous_out);

        // Shifted DFT
        __m128 bv_in0, bv_in1, bv_in2, bv_in3, bv_in4, bv_in5, bv_in6, bv_in7;
        __m128 bv_s0, bv_s1, bv_s2, bv_s3, bv_s4, bv_s5, bv_s6, bv_s7, bv_s8,
               bv_s9, bv_s10, bv_s11, bv_s12, bv_s13, bv_s14, bv_s15;
        __m128 bv_t0, bv_t1, bv_t2, bv_t3, bv_t4, bv_t5, bv_t6, bv_t7, bv_t8,
               bv_t9;

        // Input point 2: x(1) & Input point 3: x(2)
        curr_in = in + in_strides[1];
        LDRI_2x128_S(curr_in, v_in_stride, bv_in0, bv_in1);
        // Input point 6: x(5) & Input point 7: x(6)
        curr_in = in + in_strides[5];
        LDRI_2x128_S(curr_in, v_in_stride, bv_in2, bv_in3);
        // Input point 10: x(9) & Input point 11: x(10)
        curr_in = in + in_strides[9];
        LDRI_2x128_S(curr_in, v_in_stride, bv_in4, bv_in5);
        // Input point 14: x(13) & Input point 15: x(14)
        curr_in = in + in_strides[13];
        LDRI_2x128_S(curr_in, v_in_stride, bv_in6, bv_in7);

        bv_s0 = _mm_add_ps(bv_in6, bv_in0);
        bv_s1 = _mm_sub_ps(bv_in6, bv_in0);
        bv_s2 = _mm_add_ps(bv_in4, bv_in2);
        bv_s3 = _mm_sub_ps(bv_in4, bv_in2);
        bv_s4 = _mm_add_ps(bv_in7, bv_in1);
        bv_s5 = _mm_sub_ps(bv_in7, bv_in1);
        bv_s6 = _mm_add_ps(bv_in5, bv_in3);
        bv_s7 = _mm_sub_ps(bv_in5, bv_in3);
        bv_s8 = _mm_add_ps(bv_s7, bv_s5);
        bv_s9 = _mm_sub_ps(bv_s0, bv_s2);

        bv_t0 = _mm_mul_ps(v128_CRTM_8_1, bv_s8);
        bv_t1 = _mm_mul_ps(v128_CRTM_8_1, bv_s9);
        bv_s10 = _mm_sub_ps(bv_s6, bv_s1);
        bv_s11 = _mm_sub_ps(bv_s3, bv_s4);
        bv_s12 = _mm_add_ps(bv_s4, bv_s3);
        bv_s13 = _mm_add_ps(bv_s6, bv_s1);
        bv_s14 = _mm_add_ps(bv_s0, bv_s2);
        bv_t2 = _mm_mul_ps(v128_CRTM_8_2, bv_s13);

        bv_t3 = _mm_mul_ps(v128_CRTM_8_3, bv_s12);
        bv_t4 = _mm_mul_ps(v128_CRTM_8_2, bv_s11);
        bv_t5 = _mm_mul_ps(v128_CRTM_8_3, bv_s10);
        bv_s15 = _mm_sub_ps(bv_s5, bv_s7);
        bv_t6 = _mm_mul_ps(v128_CRTM_8_2, bv_s12);
        bv_t7 = _mm_mul_ps(v128_CRTM_8_3, bv_s13);
        bv_t8 = _mm_mul_ps(v128_CRTM_8_2, bv_s10);
        bv_t9 = _mm_mul_ps(v128_CRTM_8_3, bv_s11);

        // Output pt 2: X(1)
        curr_out = out + out_strides[1];
        v_out1 = _mm_mul_ps(v128_CRTM_8_4, bv_s14);
        STR_128_S(curr_out, v_out_stride, v_out1, is_contiguous_out);
        // Output pt 4: X(3)
        curr_out = out + out_strides[3];
        v_out3 = NEGATE_128_S(_mm_add_ps(bv_t2, bv_t3));
        STR_128_S(curr_out, v_out_stride, v_out3, is_contiguous_out);
        // Output pt 6: X(5)
        curr_out = out + out_strides[5];
        v_out5 = _mm_add_ps(bv_t0, bv_t1);
        STR_128_S(curr_out, v_out_stride, v_out5, is_contiguous_out);
        // Output pt 8: X(7)
        curr_out = out + out_strides[7];
        v_out7 = _mm_add_ps(bv_t5, bv_t4);
        STR_128_S(curr_out, v_out_stride, v_out7, is_contiguous_out);
        // Output pt 10: X(9)
        curr_out = out + out_strides[9];
        v_out9 = _mm_mul_ps(v128_CRTM_8_4, bv_s15);
        STR_128_S(curr_out, v_out_stride, v_out9, is_contiguous_out);
        // Output pt 12: X(11)
        curr_out = out + out_strides[11];
        v_out11 = _mm_sub_ps(bv_t7, bv_t6);
        STR_128_S(curr_out, v_out_stride, v_out11, is_contiguous_out);
        // Output pt 14: X(13)
        curr_out = out + out_strides[13];
        v_out13 = _mm_sub_ps(bv_t0, bv_t1);
        STR_128_S(curr_out, v_out_stride, v_out13, is_contiguous_out);
        // Output pt 16: X(15)
        curr_out = out + out_strides[15];
        v_out15 = _mm_sub_ps(bv_t9, bv_t8);
        STR_128_S(curr_out, v_out_stride, v_out15, is_contiguous_out);

        in = in + (v_in_stride << 2);
        out = out + (v_out_stride << 2);
    }
    // tail cases
    if (remaining_sets & 2)
    {
        // Standard DFT
        __m128 av_in0, av_in1, av_in2, av_in3, av_in4, av_in5, av_in6, av_in7;
        __m128 av_s0, av_s1, av_s2, av_s3, av_s4, av_s5, av_s6, av_s7, av_s8,
               av_s9, av_s10, av_s11, av_s12, av_s13;
        __m128 av_t0, av_t1, av_t2, av_t3;
        __m128 v_out0, v_out1, v_out2, v_out3, v_out4, v_out5, v_out6, v_out7,
               v_out8, v_out9, v_out10, v_out11, v_out12, v_out13, v_out14,
               v_out15;

        curr_in = in;
        curr_out = out;

        __m128 v128_CRTM_8_1 = _mm256_castps256_ps128(v_CRTM_8_1);
        __m128 v128_CRTM_8_2 = _mm256_castps256_ps128(v_CRTM_8_2);
        __m128 v128_CRTM_8_3 = _mm256_castps256_ps128(v_CRTM_8_3);
        __m128 v128_CRTM_8_4 = _mm256_castps256_ps128(v_CRTM_8_4);

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
        // Input point 16: x(15) & Input point 17: x(16)
        curr_in = in + in_strides[15];
        LDHR_128_S(curr_in, v_in_stride, av_in7);

        av_s0 = _mm_add_ps(av_in7, av_in0);
        av_s1 = _mm_sub_ps(av_in0, av_in7);
        av_s2 = _mm_add_ps(av_in6, av_in2);
        av_s3 = _mm_sub_ps(av_in6, av_in2);
        av_s4 = _mm_add_ps(av_in5, av_in1);
        av_t0 = _mm_mul_ps(v128_CRTM_8_4, av_s3);
        av_t1 = _mm_mul_ps(v128_CRTM_8_4, av_s4);
        av_s5 = _mm_sub_ps(av_in5, av_in1);
        av_s6 = _mm_add_ps(av_in4, av_in4);
        av_s7 = _mm_add_ps(av_in3, av_in3);

        av_s8 = _mm_add_ps(av_s5, av_s2);
        av_s9 = _mm_sub_ps(av_s5, av_s2);
        av_t2 = _mm_mul_ps(v128_CRTM_8_1, av_s8);
        av_t3 = _mm_mul_ps(v128_CRTM_8_1, av_s9);
        av_s10 = _mm_add_ps(av_s7, av_s0);
        av_s11 = _mm_sub_ps(av_s0, av_s7);
        av_s12 = _mm_sub_ps(av_s1, av_s6);
        av_s13 = _mm_add_ps(av_s6, av_s1);

        // Output pt 1: X(0)
        v_out0 = _mm_add_ps(av_s10, av_t1);
        STHR_128_S(curr_out, v_out_stride, v_out0);
        // Output point 3: X(2)
        curr_out = out + out_strides[2];
        v_out2 = _mm_sub_ps(av_s12, av_t2);
        STHR_128_S(curr_out, v_out_stride, v_out2);
        // Output pt 5: X(4)
        curr_out = out + out_strides[4];
        v_out4 = _mm_add_ps(av_t0, av_s11);
        STHR_128_S(curr_out, v_out_stride, v_out4);
        // Output pt 7: X(6)
        curr_out = out + out_strides[6];
        v_out6 = _mm_add_ps(av_s13, av_t3);
        STHR_128_S(curr_out, v_out_stride, v_out6);
        // Output pt 9: X(8)
        curr_out = out + out_strides[8];
        v_out8 = _mm_sub_ps(av_s10, av_t1);
        STHR_128_S(curr_out, v_out_stride, v_out8);
        // Output pt 11: X(10)
        curr_out = out + out_strides[10];
        v_out10 = _mm_add_ps(av_t2, av_s12);
        STHR_128_S(curr_out, v_out_stride, v_out10);
        // Output pt 13: X(12)
        curr_out = out + out_strides[12];
        v_out12 = _mm_sub_ps(av_s11, av_t0);
        STHR_128_S(curr_out, v_out_stride, v_out12);
        // Output pt 15: X(14)
        curr_out = out + out_strides[14];
        v_out14 = _mm_sub_ps(av_s13, av_t3);
        STHR_128_S(curr_out, v_out_stride, v_out14);

        // Shifted DFT
        __m128 bv_in0, bv_in1, bv_in2, bv_in3, bv_in4, bv_in5, bv_in6, bv_in7;
        __m128 bv_s0, bv_s1, bv_s2, bv_s3, bv_s4, bv_s5, bv_s6, bv_s7, bv_s8,
               bv_s9, bv_s10, bv_s11, bv_s12, bv_s13, bv_s14, bv_s15;
        __m128 bv_t0, bv_t1, bv_t2, bv_t3, bv_t4, bv_t5, bv_t6, bv_t7, bv_t8,
               bv_t9;

        // Input point 2: x(1) & Input point 3: x(2)
        curr_in = in + in_strides[1];
        LDHRI_2x128_S(curr_in, v_in_stride, bv_in0, bv_in1);
        // Input point 6: x(5) & Input point 7: x(6)
        curr_in = in + in_strides[5];
        LDHRI_2x128_S(curr_in, v_in_stride, bv_in2, bv_in3);
        // Input point 10: x(9) & Input point 11: x(10)
        curr_in = in + in_strides[9];
        LDHRI_2x128_S(curr_in, v_in_stride, bv_in4, bv_in5);
        // Input point 14: x(13) & Input point 15: x(14)
        curr_in = in + in_strides[13];
        LDHRI_2x128_S(curr_in, v_in_stride, bv_in6, bv_in7);

        bv_s0 = _mm_add_ps(bv_in6, bv_in0);
        bv_s1 = _mm_sub_ps(bv_in6, bv_in0);
        bv_s2 = _mm_add_ps(bv_in4, bv_in2);
        bv_s3 = _mm_sub_ps(bv_in4, bv_in2);
        bv_s4 = _mm_add_ps(bv_in7, bv_in1);
        bv_s5 = _mm_sub_ps(bv_in7, bv_in1);
        bv_s6 = _mm_add_ps(bv_in5, bv_in3);
        bv_s7 = _mm_sub_ps(bv_in5, bv_in3);
        bv_s8 = _mm_add_ps(bv_s7, bv_s5);
        bv_s9 = _mm_sub_ps(bv_s0, bv_s2);

        bv_t0 = _mm_mul_ps(v128_CRTM_8_1, bv_s8);
        bv_t1 = _mm_mul_ps(v128_CRTM_8_1, bv_s9);
        bv_s10 = _mm_sub_ps(bv_s6, bv_s1);
        bv_s11 = _mm_sub_ps(bv_s3, bv_s4);
        bv_s12 = _mm_add_ps(bv_s4, bv_s3);
        bv_s13 = _mm_add_ps(bv_s6, bv_s1);
        bv_s14 = _mm_add_ps(bv_s0, bv_s2);
        bv_t2 = _mm_mul_ps(v128_CRTM_8_2, bv_s13);

        bv_t3 = _mm_mul_ps(v128_CRTM_8_3, bv_s12);
        bv_t4 = _mm_mul_ps(v128_CRTM_8_2, bv_s11);
        bv_t5 = _mm_mul_ps(v128_CRTM_8_3, bv_s10);
        bv_s15 = _mm_sub_ps(bv_s5, bv_s7);
        bv_t6 = _mm_mul_ps(v128_CRTM_8_2, bv_s12);
        bv_t7 = _mm_mul_ps(v128_CRTM_8_3, bv_s13);
        bv_t8 = _mm_mul_ps(v128_CRTM_8_2, bv_s10);
        bv_t9 = _mm_mul_ps(v128_CRTM_8_3, bv_s11);

        // Output pt 2: X(1)
        curr_out = out + out_strides[1];
        v_out1 = _mm_mul_ps(v128_CRTM_8_4, bv_s14);
        STHR_128_S(curr_out, v_out_stride, v_out1);
        // Output pt 4: X(3)
        curr_out = out + out_strides[3];
        v_out3 = NEGATE_128_S(_mm_add_ps(bv_t2, bv_t3));
        STHR_128_S(curr_out, v_out_stride, v_out3);
        // Output pt 6: X(5)
        curr_out = out + out_strides[5];
        v_out5 = _mm_add_ps(bv_t0, bv_t1);
        STHR_128_S(curr_out, v_out_stride, v_out5);
        // Output pt 8: X(7)
        curr_out = out + out_strides[7];
        v_out7 = _mm_add_ps(bv_t5, bv_t4);
        STHR_128_S(curr_out, v_out_stride, v_out7);
        // Output pt 10: X(9)
        curr_out = out + out_strides[9];
        v_out9 = _mm_mul_ps(v128_CRTM_8_4, bv_s15);
        STHR_128_S(curr_out, v_out_stride, v_out9);
        // Output pt 12: X(11)
        curr_out = out + out_strides[11];
        v_out11 = _mm_sub_ps(bv_t7, bv_t6);
        STHR_128_S(curr_out, v_out_stride, v_out11);
        // Output pt 14: X(13)
        curr_out = out + out_strides[13];
        v_out13 = _mm_sub_ps(bv_t0, bv_t1);
        STHR_128_S(curr_out, v_out_stride, v_out13);
        // Output pt 16: X(15)
        curr_out = out + out_strides[15];
        v_out15 = _mm_sub_ps(bv_t9, bv_t8);
        STHR_128_S(curr_out, v_out_stride, v_out15);

        in = in + (v_in_stride << 1);
        out = out + (v_out_stride << 1);
    }
    // tail cases
    if (remaining_sets & 1)
    {
        FFTZ_FLOAT a_in0, a_in1, a_in2, a_in3, a_in4, a_in5, a_in6, a_in7;
        FFTZ_FLOAT a_s0, a_s1, a_s2, a_s3, a_s4, a_s5, a_s6, a_s7, a_s8, a_s9,
            a_s10, a_s11;
        FFTZ_FLOAT a_t0, a_t1, a_t2, a_t3, a_t4, a_t5;

        // Standard DFT
        a_in0 = *in;                // Input point 1: x(0)
        a_in1 = in[in_strides[3]];  // Input point 4: x(3)
        a_in2 = in[in_strides[4]];  // Input point 5: x(4)
        a_in3 = in[in_strides[7]];  // Input point 8: x(7)
        a_in4 = in[in_strides[8]];  // Input point 9: x(8)
        a_in5 = in[in_strides[11]]; // Input point 12: x(11)
        a_in6 = in[in_strides[12]]; // Input point 13: x(12)
        a_in7 = in[in_strides[15]]; // Input point 14: x(13)

        a_s0 = a_in7 + a_in0;
        a_s1 = a_in0 - a_in7;
        a_s2 = a_in6 + a_in2;
        a_s3 = a_in6 - a_in2;
        a_s4 = a_in5 + a_in1;
        a_s5 = a_in5 - a_in1;
        a_t0 = CRTM_8_4 * a_s3;
        a_t1 = CRTM_8_4 * a_s4;
        a_t2 = CRTM_8_4 * a_in3;
        a_t3 = CRTM_8_4 * a_in4;

        a_s6 = a_s5 + a_s2;
        a_s7 = a_s5 - a_s2;
        a_t4 = CRTM_8_1 * a_s6;
        a_t5 = CRTM_8_1 * a_s7;
        a_s8 = a_s0 + a_t2;
        a_s9 = a_s0 - a_t2;
        a_s10 = a_s1 - a_t3;
        a_s11 = a_t3 + a_s1;

        *out = a_s8 + a_t1;                  // Output pt 1: X(0)
        out[out_strides[2]] = a_s10 - a_t4; // Output pt 3: X(2)
        out[out_strides[4]] = a_s9 + a_t0;   // Output pt 5: X(4)
        out[out_strides[6]] = a_s11 + a_t5;  // Output pt 7: X(6)
        out[out_strides[8]] = a_s8 - a_t1;   // Output pt 9: X(8)
        out[out_strides[10]] = a_t4 + a_s10; // Output pt 11: X(10)
        out[out_strides[12]] = a_s9 - a_t0;  // Output pt 13: X(12)
        out[out_strides[14]] = a_s11 - a_t5; // Output pt 15: X(14)

        // Shifted DFT
        FFTZ_FLOAT b_in0, b_in1, b_in2, b_in3, b_in4, b_in5, b_in6, b_in7;
        FFTZ_FLOAT b_s0, b_s1, b_s2, b_s3, b_s4, b_s5, b_s6, b_s7, b_s8, b_s9,
              b_s10, b_s11, b_s12, b_s13, b_s14, b_s15;
        FFTZ_FLOAT b_t0, b_t1, b_t2, b_t3, b_t4, b_t5, b_t6, b_t7, b_t8, b_t9;

        b_in0 = in[in_strides[1]];  // Input point 2: x(1)
        b_in1 = in[in_strides[2]];  // Input point 3: x(2)
        b_in2 = in[in_strides[5]];  // Input point 6: x(5)
        b_in3 = in[in_strides[6]];  // Input point 7: x(6)
        b_in4 = in[in_strides[9]];  // Input point 10: x(9)
        b_in5 = in[in_strides[10]]; // Input point 11: x(10)
        b_in6 = in[in_strides[13]]; // Input point 14: x(13)
        b_in7 = in[in_strides[14]]; // Input point 13: x(12)

        b_s0 = b_in6 + b_in0;
        b_s1 = b_in6 - b_in0;
        b_s2 = b_in4 + b_in2;
        b_s3 = b_in4 - b_in2;
        b_s4 = b_in7 + b_in1;
        b_s5 = b_in7 - b_in1;
        b_s6 = b_in5 + b_in3;
        b_s7 = b_in5 - b_in3;
        b_s8 = b_s5 + b_s7;
        b_s9 = b_s5 - b_s7;
        b_s10 = b_s0 - b_s2;
        b_s11 = b_s0 + b_s2;

        b_t0 = CRTM_8_1 * b_s8;
        b_t1 = CRTM_8_1 * b_s10;
        b_s12 = b_s6 + b_s1;
        b_s13 = b_s6 - b_s1;
        b_s14 = b_s3 + b_s4;
        b_s15 = b_s3 - b_s4;
        b_t2 = CRTM_8_2 * b_s12;
        b_t3 = CRTM_8_3 * b_s13;
        b_t4 = CRTM_8_3 * b_s14;
        b_t5 = CRTM_8_2 * b_s15;
        b_t6 = CRTM_8_3 * b_s12;
        b_t7 = CRTM_8_2 * b_s13;
        b_t8 = CRTM_8_2 * b_s14;
        b_t9 = CRTM_8_3 * b_s15;

        out[out_strides[1]] = CRTM_8_4 * b_s11; // Output pt 2: X(1)
        out[out_strides[3]] = -b_t2 - b_t4;     // Output pt 4: X(3)
        out[out_strides[5]] = b_t0 + b_t1;      // Output pt 6: X(5)
        out[out_strides[7]] = b_t3 + b_t5;      // Output pt 8: X(7)
        out[out_strides[9]] = CRTM_8_4 * b_s9;  // Output pt 10: X(9)
        out[out_strides[11]] = b_t6 - b_t8;     // Output pt 12: X(11)
        out[out_strides[13]] = b_t0 - b_t1;     // Output pt 14: X(12)
        out[out_strides[15]] = b_t9 - b_t7;     // Output pt 16: X(15)
    }
    AOCLFFTZ_LOG(DEBUG, global_logger_mode, "Exit");
}

static FFTZ_VOID r2hcf_rfft8avx256_fp64_fwd(FFTZ_VOID *in_real,
                                            FFTZ_VOID *in_imag,
                                            FFTZ_VOID *out_real,
                                            FFTZ_VOID *out_imag, FFTZ_INTP n,
                                            aoclfftz_strides_t *strides,
                                            FFTZ_VOID *twd, FFTZ_UINT8 flag)
{
    AOCLFFTZ_LOG(DEBUG, global_logger_mode, "Enter");
    const FFTZ_DOUBLE CRTM_8_1 =
        0.7071067811865475244008443621048490392848359377;
    const FFTZ_DOUBLE CRTM_8_2 =
        0.9238795325112867561281831893967882868224166259;
    const FFTZ_DOUBLE CRTM_8_3 =
        0.3826834323650897717284599840303988667613445625;

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
    FFTZ_UINT8 is_contiguous_in = (v_in_stride == 1);
    FFTZ_UINT8 is_contiguous_out = (v_out_stride == 1);

    FFTZ_INTP cnt;
    FFTZ_DOUBLE *curr_in, *curr_out;
    FFTZ_INTP N = n / NUM_SETS_REAL_256_D;
    FFTZ_INTP remaining_sets = n % NUM_SETS_REAL_256_D;

    __m256d v_CRTM_8_1 = _mm256_broadcast_sd(&CRTM_8_1);
    __m256d v_CRTM_8_2 = _mm256_broadcast_sd(&CRTM_8_2);
    __m256d v_CRTM_8_3 = _mm256_broadcast_sd(&CRTM_8_3);

    for (cnt = 0; cnt < N; cnt++)
    {
        // Standard DFT
        __m256d av_in0, av_in1, av_in2, av_in3, av_in4, av_in5, av_in6, av_in7;
        __m256d av_s0, av_s1, av_s2, av_s3, av_s4, av_s5, av_s6, av_s7, av_s8,
                av_s9, av_s10, av_s11;
        __m256d av_t0, av_t1;
        __m256d v_out0, v_out1, v_out2, v_out3, v_out4, v_out5, v_out6, v_out7,
                v_out8, v_out9, v_out10, v_out11, v_out12, v_out13, v_out14,
                v_out15;

        curr_in = in;
        curr_out = out;

        // Input point 1: x(0)
        LDR_256_D(curr_in, v_in_stride, av_in0, is_contiguous_in);
        // Input point 3: x(2)
        curr_in = in + in_strides[2];
        LDR_256_D(curr_in, v_in_stride, av_in1, is_contiguous_in);
        // Input point 5: x(4)
        curr_in = in + in_strides[4];
        LDR_256_D(curr_in, v_in_stride, av_in2, is_contiguous_in);
        // Input point 7: x(6)
        curr_in = in + in_strides[6];
        LDR_256_D(curr_in, v_in_stride, av_in3, is_contiguous_in);
        // Input point 9: x(8)
        curr_in = in + in_strides[8];
        LDR_256_D(curr_in, v_in_stride, av_in4, is_contiguous_in);
        // Input point 11: x(10)
        curr_in = in + in_strides[10];
        LDR_256_D(curr_in, v_in_stride, av_in5, is_contiguous_in);
        // Input point 13: x(12)
        curr_in = in + in_strides[12];
        LDR_256_D(curr_in, v_in_stride, av_in6, is_contiguous_in);
        // Input point 15: x(14)
        curr_in = in + in_strides[14];
        LDR_256_D(curr_in, v_in_stride, av_in7, is_contiguous_in);

        av_s0 = _mm256_add_pd(av_in7, av_in5);
        av_s1 = _mm256_sub_pd(av_in7, av_in5);
        av_s2 = _mm256_add_pd(av_in6, av_in2);
        av_s3 = _mm256_sub_pd(av_in6, av_in2);
        av_s4 = _mm256_add_pd(av_in4, av_in0);
        av_s5 = _mm256_sub_pd(av_in0, av_in4);
        av_s6 = _mm256_add_pd(av_in3, av_in1);
        av_s7 = _mm256_sub_pd(av_in3, av_in1);

        av_s8 = _mm256_add_pd(av_s6, av_s0);
        av_s9 = _mm256_add_pd(av_s4, av_s2);
        av_s10 = _mm256_sub_pd(av_s7, av_s1);
        av_s11 = _mm256_sub_pd(av_s0, av_s6);

        av_t0 = _mm256_mul_pd(v_CRTM_8_1, av_s10);
        av_t1 = _mm256_mul_pd(v_CRTM_8_1, av_s11);

        // Output pt 1: X(0)
        v_out0 = _mm256_add_pd(av_s9, av_s8);
        STR_256_D(curr_out, v_out_stride, v_out0, is_contiguous_out);
        // Output pt 4: X(3) & Output pt 5: X(4)
        curr_out = out + out_strides[3];
        v_out3 = _mm256_sub_pd(av_s5, av_t0);
        v_out4 = _mm256_add_pd(av_s3, av_t1);
        STRI_2x256_D(curr_out, v_out_stride, v_out3, v_out4);
        // Output pt 8: X(7) & Output pt 9: X(8)
        curr_out = out + out_strides[7];
        v_out7 = _mm256_sub_pd(av_s4, av_s2);
        v_out8 = _mm256_add_pd(av_s7, av_s1);
        STRI_2x256_D(curr_out, v_out_stride, v_out7, v_out8);
        // Output pt 12: X(11) & Output pt 13: X(12)
        curr_out = out + out_strides[11];
        v_out11 = _mm256_add_pd(av_t0, av_s5);
        v_out12 = _mm256_sub_pd(av_t1, av_s3);
        STRI_2x256_D(curr_out, v_out_stride, v_out11, v_out12);
        // Output pt 16: X(15)
        curr_out = out + out_strides[15];
        v_out15 = _mm256_sub_pd(av_s9, av_s8);
        STR_256_D(curr_out, v_out_stride, v_out15, is_contiguous_out);

        // Shifted DFT
        __m256d bv_in0, bv_in1, bv_in2, bv_in3, bv_in4, bv_in5, bv_in6, bv_in7;
        __m256d bv_s0, bv_s1, bv_s2, bv_s3, bv_s4, bv_s5, bv_s6, bv_s7, bv_s8,
                bv_s9, bv_s10, bv_s11, bv_s12, bv_s13;
        __m256d bv_t0, bv_t1, bv_t2, bv_t3, bv_t4, bv_t5, bv_t6, bv_t7, bv_t8,
                bv_t9;

        // Input point 2: x(1)
        curr_in = in + in_strides[1];
        LDR_256_D(curr_in, v_in_stride, bv_in0, is_contiguous_in);
        // Input point 4: x(3)
        curr_in = in + in_strides[3];
        LDR_256_D(curr_in, v_in_stride, bv_in1, is_contiguous_in);
        // Input point 6: x(5)
        curr_in = in + in_strides[5];
        LDR_256_D(curr_in, v_in_stride, bv_in2, is_contiguous_in);
        // Input point 8: x(7)
        curr_in = in + in_strides[7];
        LDR_256_D(curr_in, v_in_stride, bv_in3, is_contiguous_in);
        // Input point 10: x(9)
        curr_in = in + in_strides[9];
        LDR_256_D(curr_in, v_in_stride, bv_in4, is_contiguous_in);
        // Input point 12: x(11)
        curr_in = in + in_strides[11];
        LDR_256_D(curr_in, v_in_stride, bv_in5, is_contiguous_in);
        // Input point 14: x(13)
        curr_in = in + in_strides[13];
        LDR_256_D(curr_in, v_in_stride, bv_in6, is_contiguous_in);
        // Input point 16: x(15)
        curr_in = in + in_strides[15];
        LDR_256_D(curr_in, v_in_stride, bv_in7, is_contiguous_in);

        bv_s0 = _mm256_add_pd(bv_in6, bv_in2);
        bv_s1 = _mm256_sub_pd(bv_in6, bv_in2);
        bv_t0 = _mm256_mul_pd(v_CRTM_8_1, bv_s0);
        bv_t1 = _mm256_mul_pd(v_CRTM_8_1, bv_s1);
        bv_s2 = _mm256_add_pd(bv_in7, bv_in1);
        bv_s3 = _mm256_sub_pd(bv_in7, bv_in1);
        bv_s4 = _mm256_add_pd(bv_in5, bv_in3);
        bv_s5 = _mm256_sub_pd(bv_in5, bv_in3);

        bv_t2 = _mm256_mul_pd(v_CRTM_8_2, bv_s2);
        bv_t3 = _mm256_mul_pd(v_CRTM_8_3, bv_s2);
        bv_t4 = _mm256_mul_pd(v_CRTM_8_2, bv_s3);
        bv_t5 = _mm256_mul_pd(v_CRTM_8_3, bv_s3);

        bv_t6 = _mm256_mul_pd(v_CRTM_8_2, bv_s4);
        bv_t7 = _mm256_mul_pd(v_CRTM_8_3, bv_s4);

        bv_t8 = _mm256_mul_pd(v_CRTM_8_2, bv_s5);
        bv_t9 = _mm256_mul_pd(v_CRTM_8_3, bv_s5);

        bv_s6 = _mm256_sub_pd(bv_in0, bv_t1);
        bv_s7 = _mm256_add_pd(bv_t1, bv_in0);
        bv_s8 = _mm256_sub_pd(bv_in4, bv_t0);
        bv_s9 = _mm256_add_pd(bv_t0, bv_in4);
        bv_s10 = _mm256_add_pd(bv_t4, bv_t9);
        bv_s11 = _mm256_add_pd(bv_t6, bv_t3);
        bv_s12 = _mm256_sub_pd(bv_t8, bv_t5);
        bv_s13 = _mm256_sub_pd(bv_t7, bv_t2);

        // Output pt 2: X(1) & Output pt 3: X(2)
        curr_out = out + out_strides[1];
        v_out1 = _mm256_sub_pd(bv_s6, bv_s10);
        v_out2 = NEGATE_256_D(_mm256_add_pd(bv_s9, bv_s11));
        STRI_2x256_D(curr_out, v_out_stride, v_out1, v_out2);
        // Output pt 6: X(5) & Output pt 7: X(6)
        curr_out = out + out_strides[5];
        v_out5 = _mm256_add_pd(bv_s7, bv_s12);
        v_out6 = _mm256_add_pd(bv_s8, bv_s13);
        STRI_2x256_D(curr_out, v_out_stride, v_out5, v_out6);
        // Output pt 10: X(9) & Output pt 11: X(10)
        curr_out = out + out_strides[9];
        v_out9 = _mm256_sub_pd(bv_s7, bv_s12);
        v_out10 = _mm256_sub_pd(bv_s13, bv_s8);
        STRI_2x256_D(curr_out, v_out_stride, v_out9, v_out10);
        // Output pt 14: X(13)& Output pt 15: X(14)
        curr_out = out + out_strides[13];
        v_out13 = _mm256_add_pd(bv_s6, bv_s10);
        v_out14 = _mm256_sub_pd(bv_s9, bv_s11);
        STRI_2x256_D(curr_out, v_out_stride, v_out13, v_out14);

        in = in + (v_in_stride << 2);
        out = out + (v_out_stride << 2);
    }
    // tail cases
    if (remaining_sets & NUM_SETS_REAL_128_D)
    {
        // Standard DFT
        __m128d av_in0, av_in1, av_in2, av_in3, av_in4, av_in5, av_in6, av_in7;
        __m128d av_s0, av_s1, av_s2, av_s3, av_s4, av_s5, av_s6, av_s7, av_s8,
                av_s9, av_s10, av_s11;
        __m128d av_t0, av_t1;
        __m128d v_out0, v_out1, v_out2, v_out3, v_out4, v_out5, v_out6, v_out7,
                v_out8, v_out9, v_out10, v_out11, v_out12, v_out13, v_out14,
                v_out15;

        curr_in = in;
        curr_out = out;

        __m128d v128_CRTM_8_1 = _mm256_castpd256_pd128(v_CRTM_8_1);
        __m128d v128_CRTM_8_2 = _mm256_castpd256_pd128(v_CRTM_8_2);
        __m128d v128_CRTM_8_3 = _mm256_castpd256_pd128(v_CRTM_8_3);

        // Input point 1: x(0)
        LDR_128_D(curr_in, v_in_stride, av_in0, is_contiguous_in);
        // Input point 3: x(2)
        curr_in = in + in_strides[2];
        LDR_128_D(curr_in, v_in_stride, av_in1, is_contiguous_in);
        // Input point 5: x(4)
        curr_in = in + in_strides[4];
        LDR_128_D(curr_in, v_in_stride, av_in2, is_contiguous_in);
        // Input point 7: x(6)
        curr_in = in + in_strides[6];
        LDR_128_D(curr_in, v_in_stride, av_in3, is_contiguous_in);
        // Input point 9: x(8)
        curr_in = in + in_strides[8];
        LDR_128_D(curr_in, v_in_stride, av_in4, is_contiguous_in);
        // Input point 11: x(10)
        curr_in = in + in_strides[10];
        LDR_128_D(curr_in, v_in_stride, av_in5, is_contiguous_in);
        // Input point 13: x(12)
        curr_in = in + in_strides[12];
        LDR_128_D(curr_in, v_in_stride, av_in6, is_contiguous_in);
        // Input point 15: x(14)
        curr_in = in + in_strides[14];
        LDR_128_D(curr_in, v_in_stride, av_in7, is_contiguous_in);

        av_s0 = _mm_add_pd(av_in7, av_in5);
        av_s1 = _mm_sub_pd(av_in7, av_in5);
        av_s2 = _mm_add_pd(av_in6, av_in2);
        av_s3 = _mm_sub_pd(av_in6, av_in2);
        av_s4 = _mm_add_pd(av_in4, av_in0);
        av_s5 = _mm_sub_pd(av_in0, av_in4);
        av_s6 = _mm_add_pd(av_in3, av_in1);
        av_s7 = _mm_sub_pd(av_in3, av_in1);

        av_s8 = _mm_add_pd(av_s6, av_s0);
        av_s9 = _mm_add_pd(av_s4, av_s2);
        av_s10 = _mm_sub_pd(av_s7, av_s1);
        av_s11 = _mm_sub_pd(av_s0, av_s6);

        av_t0 = _mm_mul_pd(v128_CRTM_8_1, av_s10);
        av_t1 = _mm_mul_pd(v128_CRTM_8_1, av_s11);

        // Output pt 1: X(0)
        v_out0 = _mm_add_pd(av_s9, av_s8);
        STR_128_D(curr_out, v_out_stride, v_out0, is_contiguous_out);
        // Output pt 4: X(3) & Output pt 5: X(4)
        curr_out = out + out_strides[3];
        v_out3 = _mm_sub_pd(av_s5, av_t0);
        v_out4 = _mm_add_pd(av_s3, av_t1);
        STRI_2x128_D(curr_out, v_out_stride, v_out3, v_out4);
        // Output pt 8: X(7) & Output pt 9: X(8)
        curr_out = out + out_strides[7];
        v_out7 = _mm_sub_pd(av_s4, av_s2);
        v_out8 = _mm_add_pd(av_s7, av_s1);
        STRI_2x128_D(curr_out, v_out_stride, v_out7, v_out8);
        // Output pt 12: X(11) & Output pt 13: X(12)
        curr_out = out + out_strides[11];
        v_out11 = _mm_add_pd(av_t0, av_s5);
        v_out12 = _mm_sub_pd(av_t1, av_s3);
        STRI_2x128_D(curr_out, v_out_stride, v_out11, v_out12);
        // Output pt 16: X(15)
        curr_out = out + out_strides[15];
        v_out15 = _mm_sub_pd(av_s9, av_s8);
        STR_128_D(curr_out, v_out_stride, v_out15, is_contiguous_out);

        // Shifted DFT
        __m128d bv_in0, bv_in1, bv_in2, bv_in3, bv_in4, bv_in5, bv_in6, bv_in7;
        __m128d bv_s0, bv_s1, bv_s2, bv_s3, bv_s4, bv_s5, bv_s6, bv_s7, bv_s8,
                bv_s9, bv_s10, bv_s11, bv_s12, bv_s13;
        __m128d bv_t0, bv_t1, bv_t2, bv_t3, bv_t4, bv_t5, bv_t6, bv_t7, bv_t8,
                bv_t9;

        // Input point 2: x(1)
        curr_in = in + in_strides[1];
        LDR_128_D(curr_in, v_in_stride, bv_in0, is_contiguous_in);
        // Input point 4: x(3)
        curr_in = in + in_strides[3];
        LDR_128_D(curr_in, v_in_stride, bv_in1, is_contiguous_in);
        // Input point 6: x(5)
        curr_in = in + in_strides[5];
        LDR_128_D(curr_in, v_in_stride, bv_in2, is_contiguous_in);
        // Input point 8: x(7)
        curr_in = in + in_strides[7];
        LDR_128_D(curr_in, v_in_stride, bv_in3, is_contiguous_in);
        // Input point 10: x(9)
        curr_in = in + in_strides[9];
        LDR_128_D(curr_in, v_in_stride, bv_in4, is_contiguous_in);
        // Input point 12: x(11)
        curr_in = in + in_strides[11];
        LDR_128_D(curr_in, v_in_stride, bv_in5, is_contiguous_in);
        // Input point 14: x(13)
        curr_in = in + in_strides[13];
        LDR_128_D(curr_in, v_in_stride, bv_in6, is_contiguous_in);
        // Input point 16: x(15)
        curr_in = in + in_strides[15];
        LDR_128_D(curr_in, v_in_stride, bv_in7, is_contiguous_in);

        bv_s0 = _mm_add_pd(bv_in6, bv_in2);
        bv_s1 = _mm_sub_pd(bv_in6, bv_in2);
        bv_t0 = _mm_mul_pd(v128_CRTM_8_1, bv_s0);
        bv_t1 = _mm_mul_pd(v128_CRTM_8_1, bv_s1);
        bv_s2 = _mm_add_pd(bv_in7, bv_in1);
        bv_s3 = _mm_sub_pd(bv_in7, bv_in1);
        bv_s4 = _mm_add_pd(bv_in5, bv_in3);
        bv_s5 = _mm_sub_pd(bv_in5, bv_in3);

        bv_t2 = _mm_mul_pd(v128_CRTM_8_2, bv_s2);
        bv_t3 = _mm_mul_pd(v128_CRTM_8_3, bv_s2);
        bv_t4 = _mm_mul_pd(v128_CRTM_8_2, bv_s3);
        bv_t5 = _mm_mul_pd(v128_CRTM_8_3, bv_s3);

        bv_t6 = _mm_mul_pd(v128_CRTM_8_2, bv_s4);
        bv_t7 = _mm_mul_pd(v128_CRTM_8_3, bv_s4);

        bv_t8 = _mm_mul_pd(v128_CRTM_8_2, bv_s5);
        bv_t9 = _mm_mul_pd(v128_CRTM_8_3, bv_s5);

        bv_s6 = _mm_sub_pd(bv_in0, bv_t1);
        bv_s7 = _mm_add_pd(bv_t1, bv_in0);
        bv_s8 = _mm_sub_pd(bv_in4, bv_t0);
        bv_s9 = _mm_add_pd(bv_t0, bv_in4);
        bv_s10 = _mm_add_pd(bv_t4, bv_t9);
        bv_s11 = _mm_add_pd(bv_t6, bv_t3);
        bv_s12 = _mm_sub_pd(bv_t8, bv_t5);
        bv_s13 = _mm_sub_pd(bv_t7, bv_t2);

        // Output pt 2: X(1) & Output pt 3: X(2)
        curr_out = out + out_strides[1];
        v_out1 = _mm_sub_pd(bv_s6, bv_s10);
        v_out2 = NEGATE_128_D(_mm_add_pd(bv_s9, bv_s11));
        STRI_2x128_D(curr_out, v_out_stride, v_out1, v_out2);
        // Output pt 6: X(5) & Output pt 7: X(6)
        curr_out = out + out_strides[5];
        v_out5 = _mm_add_pd(bv_s7, bv_s12);
        v_out6 = _mm_add_pd(bv_s8, bv_s13);
        STRI_2x128_D(curr_out, v_out_stride, v_out5, v_out6);
        // Output pt 10: X(9) & Output pt 11: X(10)
        curr_out = out + out_strides[9];
        v_out9 = _mm_sub_pd(bv_s7, bv_s12);
        v_out10 = _mm_sub_pd(bv_s13, bv_s8);
        STRI_2x128_D(curr_out, v_out_stride, v_out9, v_out10);
        // Output pt 14: X(13)& Output pt 15: X(14)
        curr_out = out + out_strides[13];
        v_out13 = _mm_add_pd(bv_s6, bv_s10);
        v_out14 = _mm_sub_pd(bv_s9, bv_s11);
        STRI_2x128_D(curr_out, v_out_stride, v_out13, v_out14);

        in = in + (v_in_stride << 1);
        out = out + (v_out_stride << 1);
    }
    // tail cases
    if (remaining_sets & 1)
    {
        // Standard DFT
        FFTZ_DOUBLE a_in0, a_in1, a_in2, a_in3, a_in4, a_in5, a_in6, a_in7;
        FFTZ_DOUBLE a_s0, a_s1, a_s2, a_s3, a_s4, a_s5, a_s6, a_s7, a_s8, a_s9,
               a_s10, a_s11;
        FFTZ_DOUBLE a_t0, a_t1;

        a_in0 = *in;                // Input point 1: x(0)
        a_in1 = in[in_strides[2]];  // Input point 3: x(2)
        a_in2 = in[in_strides[4]];  // Input point 5: x(4)
        a_in3 = in[in_strides[6]];  // Input point 7: x(6)
        a_in4 = in[in_strides[8]];  // Input point 9: x(8)
        a_in5 = in[in_strides[10]]; // Input point 11: x(10)
        a_in6 = in[in_strides[12]]; // Input point 13: x(12)
        a_in7 = in[in_strides[14]]; // Input point 13: x(12)

        a_s0 = a_in7 + a_in5;
        a_s1 = a_in7 - a_in5;
        a_s2 = a_in6 + a_in2;
        a_s3 = a_in6 - a_in2;
        a_s4 = a_in0 + a_in4;
        a_s5 = a_in0 - a_in4;
        a_s6 = a_in3 + a_in1;
        a_s7 = a_in3 - a_in1;

        a_s8 = a_s7 - a_s1;
        a_s9 = a_s0 - a_s6;
        a_s10 = a_s0 + a_s6;
        a_s11 = a_s4 + a_s2;
        a_t0 = CRTM_8_1 * a_s8;
        a_t1 = CRTM_8_1 * a_s9;

        *out = a_s11 + a_s10;                 // Output pt 1: X(0)
        out[out_strides[3]] = a_s5 - a_t0;    // Output pt 4: X(3)
        out[out_strides[4]] = a_s3 + a_t1;    // Output pt 5: X(4)
        out[out_strides[7]] = a_s4 - a_s2;    // Output pt 8: X(7)
        out[out_strides[8]] = a_s7 + a_s1;    // Output pt 9: X(8)
        out[out_strides[11]] = a_t0 + a_s5;   // Output pt 12: X(11)
        out[out_strides[12]] = a_t1 - a_s3;   // Output pt 13: X(12)
        out[out_strides[15]] = a_s11 - a_s10; // Output pt 16: X(15)

        // Shifted DFT
        FFTZ_DOUBLE b_in0, b_in1, b_in2, b_in3, b_in4, b_in5, b_in6, b_in7;
        FFTZ_DOUBLE b_s0, b_s1, b_s2, b_s3, b_s4, b_s5, b_s6, b_s7, b_s8, b_s9,
               b_s10, b_s11, b_s12, b_s13;
        FFTZ_DOUBLE b_t0, b_t1, b_t2, b_t3, b_t4, b_t5, b_t6, b_t7, b_t8, b_t9;

        b_in0 = in[in_strides[1]];  // Input point 2: x(1)
        b_in1 = in[in_strides[3]];  // Input point 4: x(3)
        b_in2 = in[in_strides[5]];  // Input point 6: x(5)
        b_in3 = in[in_strides[7]];  // Input point 8: x(7)
        b_in4 = in[in_strides[9]];  // Input point 10: x(9)
        b_in5 = in[in_strides[11]]; // Input point 12: x(11)
        b_in6 = in[in_strides[13]]; // Input point 14: x(13)
        b_in7 = in[in_strides[15]]; // Input point 14: x(13)

        b_s0 = b_in6 + b_in2;
        b_s1 = b_in6 - b_in2;
        b_s2 = b_in7 + b_in1;
        b_s3 = b_in7 - b_in1;
        b_s4 = b_in5 + b_in3;
        b_s5 = b_in5 - b_in3;

        b_t0 = CRTM_8_1 * b_s0;
        b_t1 = CRTM_8_1 * b_s1;
        b_t2 = CRTM_8_2 * b_s2;
        b_t3 = CRTM_8_3 * b_s2;
        b_t4 = CRTM_8_2 * b_s3;
        b_t5 = CRTM_8_3 * b_s3;
        b_t6 = CRTM_8_2 * b_s4;
        b_t7 = CRTM_8_3 * b_s4;
        b_t8 = CRTM_8_2 * b_s5;
        b_t9 = CRTM_8_3 * b_s5;

        b_s6 = b_in0 - b_t1;
        b_s7 = b_in0 + b_t1;
        b_s8 = b_in4 - b_t0;
        b_s9 = b_t0 + b_in4;
        b_s10 = b_t4 + b_t9;
        b_s11 = b_t6 + b_t3;
        b_s12 = b_t8 - b_t5;
        b_s13 = b_t7 - b_t2;

        out[out_strides[1]] = b_s6 - b_s10;  // Output pt 2: X(1)
        out[out_strides[2]] = -b_s9 - b_s11; // Output pt 3: X(2)
        out[out_strides[5]] = b_s7 + b_s12;  // Output pt 6: X(5)
        out[out_strides[6]] = b_s8 + b_s13;  // Output pt 7: X(6)
        out[out_strides[9]] = b_s7 - b_s12;  // Output pt 10: X(9)
        out[out_strides[10]] = b_s13 - b_s8; // Output pt 11: X(10)
        out[out_strides[13]] = b_s6 + b_s10; // Output pt 14: X(12)
        out[out_strides[14]] = b_s9 - b_s11; // Output pt 15: X(14)
    }
    AOCLFFTZ_LOG(DEBUG, global_logger_mode, "Exit");
}

static FFTZ_VOID r2hcf_rfft8avx256_fp64_bwd(FFTZ_VOID *in_real,
                                            FFTZ_VOID *in_imag,
                                            FFTZ_VOID *out_real,
                                            FFTZ_VOID *out_imag, FFTZ_INTP n,
                                            aoclfftz_strides_t *strides,
                                            FFTZ_VOID *twd, FFTZ_UINT8 flag)
{
    AOCLFFTZ_LOG(DEBUG, global_logger_mode, "Enter");
    const FFTZ_DOUBLE CRTM_8_1 =
        1.414213562373095048801688724209698078569671875;
    const FFTZ_DOUBLE CRTM_8_2 =
        1.847759065022573256256366378793576573644833252;
    const FFTZ_DOUBLE CRTM_8_3 =
        0.765366864730179543456919968060797733522689125;
    const FFTZ_DOUBLE CRTM_8_4 =
        2.000000000000000000000000000000000000000000000;

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
    FFTZ_UINT8 is_contiguous_in = (v_in_stride == 1);
    FFTZ_UINT8 is_contiguous_out = (v_out_stride == 1);

    FFTZ_INTP cnt;
    FFTZ_DOUBLE *curr_in, *curr_out;
    FFTZ_INTP N = n / NUM_SETS_REAL_256_D;
    FFTZ_INTP remaining_sets = n % NUM_SETS_REAL_256_D;

    __m256d v_CRTM_8_1 = _mm256_broadcast_sd(&CRTM_8_1);
    __m256d v_CRTM_8_2 = _mm256_broadcast_sd(&CRTM_8_2);
    __m256d v_CRTM_8_3 = _mm256_broadcast_sd(&CRTM_8_3);
    __m256d v_CRTM_8_4 = _mm256_broadcast_sd(&CRTM_8_4);

    for (cnt = 0; cnt < N; cnt++)
    {
        // Standard DFT
        __m256d av_in0, av_in1, av_in2, av_in3, av_in4, av_in5, av_in6, av_in7;
        __m256d av_s0, av_s1, av_s2, av_s3, av_s4, av_s5, av_s6, av_s7, av_s8,
                av_s9, av_s10, av_s11, av_s12, av_s13;
        __m256d av_t0, av_t1, av_t2, av_t3;
        __m256d v_out0, v_out1, v_out2, v_out3, v_out4, v_out5, v_out6, v_out7,
                v_out8, v_out9, v_out10, v_out11, v_out12, v_out13, v_out14,
                v_out15;

        curr_in = in;
        curr_out = out;

        // Input point 1: x(0)
        LDR_256_D(curr_in, v_in_stride, av_in0, is_contiguous_in);
        // Input point 4: x(3) & Input point 5: x(4)
        curr_in = in + in_strides[3];
        LDRI_2x256_D(curr_in, v_in_stride, av_in1, av_in2);
        // Input point 8: x(7) & Input point 9: x(8)
        curr_in = in + in_strides[7];
        LDRI_2x256_D(curr_in, v_in_stride, av_in3, av_in4);
        // Input point 12: x(11) & Input point 13: x(12)
        curr_in = in + in_strides[11];
        LDRI_2x256_D(curr_in, v_in_stride, av_in5, av_in6);
        // Input point 16: x(15) & Input point 17: x(16)
        curr_in = in + in_strides[15];
        LDR_256_D(curr_in, v_in_stride, av_in7, is_contiguous_in);

        av_s0 = _mm256_add_pd(av_in7, av_in0);
        av_s1 = _mm256_sub_pd(av_in0, av_in7);
        av_s2 = _mm256_add_pd(av_in6, av_in2);
        av_s3 = _mm256_sub_pd(av_in6, av_in2);
        av_s4 = _mm256_add_pd(av_in5, av_in1);
        av_t0 = _mm256_mul_pd(v_CRTM_8_4, av_s3);
        av_t1 = _mm256_mul_pd(v_CRTM_8_4, av_s4);
        av_s5 = _mm256_sub_pd(av_in5, av_in1);
        av_s6 = _mm256_add_pd(av_in4, av_in4);
        av_s7 = _mm256_add_pd(av_in3, av_in3);

        av_s8 = _mm256_add_pd(av_s5, av_s2);
        av_s9 = _mm256_sub_pd(av_s5, av_s2);
        av_t2 = _mm256_mul_pd(v_CRTM_8_1, av_s8);
        av_t3 = _mm256_mul_pd(v_CRTM_8_1, av_s9);
        av_s10 = _mm256_add_pd(av_s7, av_s0);
        av_s11 = _mm256_sub_pd(av_s0, av_s7);
        av_s12 = _mm256_sub_pd(av_s1, av_s6);
        av_s13 = _mm256_add_pd(av_s6, av_s1);

        // Output pt 1: X(0)
        v_out0 = _mm256_add_pd(av_s10, av_t1);
        STR_256_D(curr_out, v_out_stride, v_out0, is_contiguous_out);
        // Output point 3: X(2)
        curr_out = out + out_strides[2];
        v_out2 = _mm256_sub_pd(av_s12, av_t2);
        STR_256_D(curr_out, v_out_stride, v_out2, is_contiguous_out);
        // Output pt 5: X(4)
        curr_out = out + out_strides[4];
        v_out4 = _mm256_add_pd(av_t0, av_s11);
        STR_256_D(curr_out, v_out_stride, v_out4, is_contiguous_out);
        // Output pt 7: X(6)
        curr_out = out + out_strides[6];
        v_out6 = _mm256_add_pd(av_s13, av_t3);
        STR_256_D(curr_out, v_out_stride, v_out6, is_contiguous_out);
        // Output pt 9: X(8)
        curr_out = out + out_strides[8];
        v_out8 = _mm256_sub_pd(av_s10, av_t1);
        STR_256_D(curr_out, v_out_stride, v_out8, is_contiguous_out);
        // Output pt 11: X(10)
        curr_out = out + out_strides[10];
        v_out10 = _mm256_add_pd(av_t2, av_s12);
        STR_256_D(curr_out, v_out_stride, v_out10, is_contiguous_out);
        // Output pt 13: X(12)
        curr_out = out + out_strides[12];
        v_out12 = _mm256_sub_pd(av_s11, av_t0);
        STR_256_D(curr_out, v_out_stride, v_out12, is_contiguous_out);
        // Output pt 15: X(14)
        curr_out = out + out_strides[14];
        v_out14 = _mm256_sub_pd(av_s13, av_t3);
        STR_256_D(curr_out, v_out_stride, v_out14, is_contiguous_out);

        // Shifted DFT
        __m256d bv_in0, bv_in1, bv_in2, bv_in3, bv_in4, bv_in5, bv_in6, bv_in7;
        __m256d bv_s0, bv_s1, bv_s2, bv_s3, bv_s4, bv_s5, bv_s6, bv_s7, bv_s8,
                bv_s9, bv_s10, bv_s11, bv_s12, bv_s13, bv_s14, bv_s15;
        __m256d bv_t0, bv_t1, bv_t2, bv_t3, bv_t4, bv_t5, bv_t6, bv_t7, bv_t8,
                bv_t9;

        // Input point 2: x(1) & Input point 3: x(2)
        curr_in = in + in_strides[1];
        LDRI_2x256_D(curr_in, v_in_stride, bv_in0, bv_in1);
        // Input point 6: x(5) & Input point 7: x(6)
        curr_in = in + in_strides[5];
        LDRI_2x256_D(curr_in, v_in_stride, bv_in2, bv_in3);
        // Input point 10: x(9) & Input point 11: x(10)
        curr_in = in + in_strides[9];
        LDRI_2x256_D(curr_in, v_in_stride, bv_in4, bv_in5);
        // Input point 14: x(13) & Input point 15: x(14)
        curr_in = in + in_strides[13];
        LDRI_2x256_D(curr_in, v_in_stride, bv_in6, bv_in7);

        bv_s0 = _mm256_add_pd(bv_in6, bv_in0);
        bv_s1 = _mm256_sub_pd(bv_in6, bv_in0);
        bv_s2 = _mm256_add_pd(bv_in4, bv_in2);
        bv_s3 = _mm256_sub_pd(bv_in4, bv_in2);
        bv_s4 = _mm256_add_pd(bv_in7, bv_in1);
        bv_s5 = _mm256_sub_pd(bv_in7, bv_in1);
        bv_s6 = _mm256_add_pd(bv_in5, bv_in3);
        bv_s7 = _mm256_sub_pd(bv_in5, bv_in3);
        bv_s8 = _mm256_add_pd(bv_s7, bv_s5);
        bv_s9 = _mm256_sub_pd(bv_s0, bv_s2);

        bv_t0 = _mm256_mul_pd(v_CRTM_8_1, bv_s8);
        bv_t1 = _mm256_mul_pd(v_CRTM_8_1, bv_s9);
        bv_s10 = _mm256_sub_pd(bv_s6, bv_s1);
        bv_s11 = _mm256_sub_pd(bv_s3, bv_s4);
        bv_s12 = _mm256_add_pd(bv_s4, bv_s3);
        bv_s13 = _mm256_add_pd(bv_s6, bv_s1);
        bv_s14 = _mm256_add_pd(bv_s0, bv_s2);
        bv_t2 = _mm256_mul_pd(v_CRTM_8_2, bv_s13);

        bv_t3 = _mm256_mul_pd(v_CRTM_8_3, bv_s12);
        bv_t4 = _mm256_mul_pd(v_CRTM_8_2, bv_s11);
        bv_t5 = _mm256_mul_pd(v_CRTM_8_3, bv_s10);
        bv_s15 = _mm256_sub_pd(bv_s5, bv_s7);
        bv_t6 = _mm256_mul_pd(v_CRTM_8_2, bv_s12);
        bv_t7 = _mm256_mul_pd(v_CRTM_8_3, bv_s13);
        bv_t8 = _mm256_mul_pd(v_CRTM_8_2, bv_s10);
        bv_t9 = _mm256_mul_pd(v_CRTM_8_3, bv_s11);

        // Output pt 2: X(1)
        curr_out = out + out_strides[1];
        v_out1 = _mm256_mul_pd(v_CRTM_8_4, bv_s14);
        STR_256_D(curr_out, v_out_stride, v_out1, is_contiguous_out);
        // Output pt 4: X(3)
        curr_out = out + out_strides[3];
        v_out3 = NEGATE_256_D(_mm256_add_pd(bv_t2, bv_t3));
        STR_256_D(curr_out, v_out_stride, v_out3, is_contiguous_out);
        // Output pt 6: X(5)
        curr_out = out + out_strides[5];
        v_out5 = _mm256_add_pd(bv_t0, bv_t1);
        STR_256_D(curr_out, v_out_stride, v_out5, is_contiguous_out);
        // Output pt 8: X(7)
        curr_out = out + out_strides[7];
        v_out7 = _mm256_add_pd(bv_t5, bv_t4);
        STR_256_D(curr_out, v_out_stride, v_out7, is_contiguous_out);
        // Output pt 10: X(9)
        curr_out = out + out_strides[9];
        v_out9 = _mm256_mul_pd(v_CRTM_8_4, bv_s15);
        STR_256_D(curr_out, v_out_stride, v_out9, is_contiguous_out);
        // Output pt 12: X(11)
        curr_out = out + out_strides[11];
        v_out11 = _mm256_sub_pd(bv_t7, bv_t6);
        STR_256_D(curr_out, v_out_stride, v_out11, is_contiguous_out);
        // Output pt 14: X(13)
        curr_out = out + out_strides[13];
        v_out13 = _mm256_sub_pd(bv_t0, bv_t1);
        STR_256_D(curr_out, v_out_stride, v_out13, is_contiguous_out);
        // Output pt 16: X(15)
        curr_out = out + out_strides[15];
        v_out15 = _mm256_sub_pd(bv_t9, bv_t8);
        STR_256_D(curr_out, v_out_stride, v_out15, is_contiguous_out);

        in = in + (v_in_stride << 2);
        out = out + (v_out_stride << 2);
    }
    // tail cases
    if (remaining_sets & NUM_SETS_REAL_128_D)
    {
        // Standard DFT
        __m128d av_in0, av_in1, av_in2, av_in3, av_in4, av_in5, av_in6, av_in7;
        __m128d av_s0, av_s1, av_s2, av_s3, av_s4, av_s5, av_s6, av_s7, av_s8,
                av_s9, av_s10, av_s11, av_s12, av_s13;
        __m128d av_t0, av_t1, av_t2, av_t3;
        __m128d v_out0, v_out1, v_out2, v_out3, v_out4, v_out5, v_out6, v_out7,
                v_out8, v_out9, v_out10, v_out11, v_out12, v_out13, v_out14,
                v_out15;

        curr_in = in;
        curr_out = out;

        __m128d v128_CRTM_8_1 = _mm256_castpd256_pd128(v_CRTM_8_1);
        __m128d v128_CRTM_8_2 = _mm256_castpd256_pd128(v_CRTM_8_2);
        __m128d v128_CRTM_8_3 = _mm256_castpd256_pd128(v_CRTM_8_3);
        __m128d v128_CRTM_8_4 = _mm256_castpd256_pd128(v_CRTM_8_4);

        // Input point 1: x(0)
        LDR_128_D(curr_in, v_in_stride, av_in0, is_contiguous_in);
        // Input point 4: x(3) & Input point 5: x(4)
        curr_in = in + in_strides[3];
        LDRI_2x128_D(curr_in, v_in_stride, av_in1, av_in2);
        // Input point 8: x(7) & Input point 9: x(8)
        curr_in = in + in_strides[7];
        LDRI_2x128_D(curr_in, v_in_stride, av_in3, av_in4);
        // Input point 12: x(11) & Input point 13: x(12)
        curr_in = in + in_strides[11];
        LDRI_2x128_D(curr_in, v_in_stride, av_in5, av_in6);
        // Input point 16: x(15) & Input point 17: x(16)
        curr_in = in + in_strides[15];
        LDR_128_D(curr_in, v_in_stride, av_in7, is_contiguous_in);

        av_s0 = _mm_add_pd(av_in7, av_in0);
        av_s1 = _mm_sub_pd(av_in0, av_in7);
        av_s2 = _mm_add_pd(av_in6, av_in2);
        av_s3 = _mm_sub_pd(av_in6, av_in2);
        av_s4 = _mm_add_pd(av_in5, av_in1);
        av_t0 = _mm_mul_pd(v128_CRTM_8_4, av_s3);
        av_t1 = _mm_mul_pd(v128_CRTM_8_4, av_s4);
        av_s5 = _mm_sub_pd(av_in5, av_in1);
        av_s6 = _mm_add_pd(av_in4, av_in4);
        av_s7 = _mm_add_pd(av_in3, av_in3);

        av_s8 = _mm_add_pd(av_s5, av_s2);
        av_s9 = _mm_sub_pd(av_s5, av_s2);
        av_t2 = _mm_mul_pd(v128_CRTM_8_1, av_s8);
        av_t3 = _mm_mul_pd(v128_CRTM_8_1, av_s9);
        av_s10 = _mm_add_pd(av_s7, av_s0);
        av_s11 = _mm_sub_pd(av_s0, av_s7);
        av_s12 = _mm_sub_pd(av_s1, av_s6);
        av_s13 = _mm_add_pd(av_s6, av_s1);

        // Output pt 1: X(0)
        v_out0 = _mm_add_pd(av_s10, av_t1);
        STR_128_D(curr_out, v_out_stride, v_out0, is_contiguous_out);
        // Output point 3: X(2)
        curr_out = out + out_strides[2];
        v_out2 = _mm_sub_pd(av_s12, av_t2);
        STR_128_D(curr_out, v_out_stride, v_out2, is_contiguous_out);
        // Output pt 5: X(4)
        curr_out = out + out_strides[4];
        v_out4 = _mm_add_pd(av_t0, av_s11);
        STR_128_D(curr_out, v_out_stride, v_out4, is_contiguous_out);
        // Output pt 7: X(6)
        curr_out = out + out_strides[6];
        v_out6 = _mm_add_pd(av_s13, av_t3);
        STR_128_D(curr_out, v_out_stride, v_out6, is_contiguous_out);
        // Output pt 9: X(8)
        curr_out = out + out_strides[8];
        v_out8 = _mm_sub_pd(av_s10, av_t1);
        STR_128_D(curr_out, v_out_stride, v_out8, is_contiguous_out);
        // Output pt 11: X(10)
        curr_out = out + out_strides[10];
        v_out10 = _mm_add_pd(av_t2, av_s12);
        STR_128_D(curr_out, v_out_stride, v_out10, is_contiguous_out);
        // Output pt 13: X(12)
        curr_out = out + out_strides[12];
        v_out12 = _mm_sub_pd(av_s11, av_t0);
        STR_128_D(curr_out, v_out_stride, v_out12, is_contiguous_out);
        // Output pt 15: X(14)
        curr_out = out + out_strides[14];
        v_out14 = _mm_sub_pd(av_s13, av_t3);
        STR_128_D(curr_out, v_out_stride, v_out14, is_contiguous_out);

        // Shifted DFT
        __m128d bv_in0, bv_in1, bv_in2, bv_in3, bv_in4, bv_in5, bv_in6, bv_in7;
        __m128d bv_s0, bv_s1, bv_s2, bv_s3, bv_s4, bv_s5, bv_s6, bv_s7, bv_s8,
                bv_s9, bv_s10, bv_s11, bv_s12, bv_s13, bv_s14, bv_s15;
        __m128d bv_t0, bv_t1, bv_t2, bv_t3, bv_t4, bv_t5, bv_t6, bv_t7, bv_t8,
                bv_t9;

        // Input point 2: x(1) & Input point 3: x(2)
        curr_in = in + in_strides[1];
        LDRI_2x128_D(curr_in, v_in_stride, bv_in0, bv_in1);
        // Input point 6: x(5) & Input point 7: x(6)
        curr_in = in + in_strides[5];
        LDRI_2x128_D(curr_in, v_in_stride, bv_in2, bv_in3);
        // Input point 10: x(9) & Input point 11: x(10)
        curr_in = in + in_strides[9];
        LDRI_2x128_D(curr_in, v_in_stride, bv_in4, bv_in5);
        // Input point 14: x(13) & Input point 15: x(14)
        curr_in = in + in_strides[13];
        LDRI_2x128_D(curr_in, v_in_stride, bv_in6, bv_in7);

        bv_s0 = _mm_add_pd(bv_in6, bv_in0);
        bv_s1 = _mm_sub_pd(bv_in6, bv_in0);
        bv_s2 = _mm_add_pd(bv_in4, bv_in2);
        bv_s3 = _mm_sub_pd(bv_in4, bv_in2);
        bv_s4 = _mm_add_pd(bv_in7, bv_in1);
        bv_s5 = _mm_sub_pd(bv_in7, bv_in1);
        bv_s6 = _mm_add_pd(bv_in5, bv_in3);
        bv_s7 = _mm_sub_pd(bv_in5, bv_in3);
        bv_s8 = _mm_add_pd(bv_s7, bv_s5);
        bv_s9 = _mm_sub_pd(bv_s0, bv_s2);

        bv_t0 = _mm_mul_pd(v128_CRTM_8_1, bv_s8);
        bv_t1 = _mm_mul_pd(v128_CRTM_8_1, bv_s9);
        bv_s10 = _mm_sub_pd(bv_s6, bv_s1);
        bv_s11 = _mm_sub_pd(bv_s3, bv_s4);
        bv_s12 = _mm_add_pd(bv_s4, bv_s3);
        bv_s13 = _mm_add_pd(bv_s6, bv_s1);
        bv_s14 = _mm_add_pd(bv_s0, bv_s2);
        bv_t2 = _mm_mul_pd(v128_CRTM_8_2, bv_s13);

        bv_t3 = _mm_mul_pd(v128_CRTM_8_3, bv_s12);
        bv_t4 = _mm_mul_pd(v128_CRTM_8_2, bv_s11);
        bv_t5 = _mm_mul_pd(v128_CRTM_8_3, bv_s10);
        bv_s15 = _mm_sub_pd(bv_s5, bv_s7);
        bv_t6 = _mm_mul_pd(v128_CRTM_8_2, bv_s12);
        bv_t7 = _mm_mul_pd(v128_CRTM_8_3, bv_s13);
        bv_t8 = _mm_mul_pd(v128_CRTM_8_2, bv_s10);
        bv_t9 = _mm_mul_pd(v128_CRTM_8_3, bv_s11);

        // Output pt 2: X(1)
        curr_out = out + out_strides[1];
        v_out1 = _mm_mul_pd(v128_CRTM_8_4, bv_s14);
        STR_128_D(curr_out, v_out_stride, v_out1, is_contiguous_out);
        // Output pt 4: X(3)
        curr_out = out + out_strides[3];
        v_out3 = NEGATE_128_D(_mm_add_pd(bv_t2, bv_t3));
        STR_128_D(curr_out, v_out_stride, v_out3, is_contiguous_out);
        // Output pt 6: X(5)
        curr_out = out + out_strides[5];
        v_out5 = _mm_add_pd(bv_t0, bv_t1);
        STR_128_D(curr_out, v_out_stride, v_out5, is_contiguous_out);
        // Output pt 8: X(7)
        curr_out = out + out_strides[7];
        v_out7 = _mm_add_pd(bv_t5, bv_t4);
        STR_128_D(curr_out, v_out_stride, v_out7, is_contiguous_out);
        // Output pt 10: X(9)
        curr_out = out + out_strides[9];
        v_out9 = _mm_mul_pd(v128_CRTM_8_4, bv_s15);
        STR_128_D(curr_out, v_out_stride, v_out9, is_contiguous_out);
        // Output pt 12: X(11)
        curr_out = out + out_strides[11];
        v_out11 = _mm_sub_pd(bv_t7, bv_t6);
        STR_128_D(curr_out, v_out_stride, v_out11, is_contiguous_out);
        // Output pt 14: X(13)
        curr_out = out + out_strides[13];
        v_out13 = _mm_sub_pd(bv_t0, bv_t1);
        STR_128_D(curr_out, v_out_stride, v_out13, is_contiguous_out);
        // Output pt 16: X(15)
        curr_out = out + out_strides[15];
        v_out15 = _mm_sub_pd(bv_t9, bv_t8);
        STR_128_D(curr_out, v_out_stride, v_out15, is_contiguous_out);

        in = in + (v_in_stride << 1);
        out = out + (v_out_stride << 1);
    }
    // tail cases
    if (remaining_sets & 1)
    {
        FFTZ_DOUBLE a_in0, a_in1, a_in2, a_in3, a_in4, a_in5, a_in6, a_in7;
        FFTZ_DOUBLE a_s0, a_s1, a_s2, a_s3, a_s4, a_s5, a_s6, a_s7, a_s8, a_s9,
               a_s10, a_s11;
        FFTZ_DOUBLE a_t0, a_t1, a_t2, a_t3, a_t4, a_t5;

        // Standard DFT
        a_in0 = *in;                // Input point 1: x(0)
        a_in1 = in[in_strides[3]];  // Input point 4: x(3)
        a_in2 = in[in_strides[4]];  // Input point 5: x(4)
        a_in3 = in[in_strides[7]];  // Input point 8: x(7)
        a_in4 = in[in_strides[8]];  // Input point 9: x(8)
        a_in5 = in[in_strides[11]]; // Input point 12: x(11)
        a_in6 = in[in_strides[12]]; // Input point 13: x(12)
        a_in7 = in[in_strides[15]]; // Input point 14: x(13)

        a_s0 = a_in7 + a_in0;
        a_s1 = a_in0 - a_in7;
        a_s2 = a_in6 + a_in2;
        a_s3 = a_in6 - a_in2;
        a_s4 = a_in5 + a_in1;
        a_s5 = a_in5 - a_in1;
        a_t0 = CRTM_8_4 * a_s3;
        a_t1 = CRTM_8_4 * a_s4;
        a_t2 = CRTM_8_4 * a_in3;
        a_t3 = CRTM_8_4 * a_in4;

        a_s6 = a_s5 + a_s2;
        a_s7 = a_s5 - a_s2;
        a_t4 = CRTM_8_1 * a_s6;
        a_t5 = CRTM_8_1 * a_s7;
        a_s8 = a_s0 + a_t2;
        a_s9 = a_s0 - a_t2;
        a_s10 = a_s1 - a_t3;
        a_s11 = a_t3 + a_s1;

        *out = a_s8 + a_t1;                  // Output pt 1: X(0)
        out[out_strides[2]] = a_s10 - a_t4;  // Output pt 3: X(2)
        out[out_strides[4]] = a_s9 + a_t0;   // Output pt 5: X(4)
        out[out_strides[6]] = a_s11 + a_t5;  // Output pt 7: X(6)
        out[out_strides[8]] = a_s8 - a_t1;   // Output pt 9: X(8)
        out[out_strides[10]] = a_t4 + a_s10; // Output pt 11: X(10)
        out[out_strides[12]] = a_s9 - a_t0;  // Output pt 13: X(12)
        out[out_strides[14]] = a_s11 - a_t5; // Output pt 15: X(14)

        // Shifted DFT
        FFTZ_DOUBLE b_in0, b_in1, b_in2, b_in3, b_in4, b_in5, b_in6, b_in7;
        FFTZ_DOUBLE b_s0, b_s1, b_s2, b_s3, b_s4, b_s5, b_s6, b_s7, b_s8, b_s9,
               b_s10, b_s11, b_s12, b_s13, b_s14, b_s15;
        FFTZ_DOUBLE b_t0, b_t1, b_t2, b_t3, b_t4, b_t5, b_t6, b_t7, b_t8, b_t9;

        b_in0 = in[in_strides[1]];  // Input point 2: x(1)
        b_in1 = in[in_strides[2]];  // Input point 3: x(2)
        b_in2 = in[in_strides[5]];  // Input point 6: x(5)
        b_in3 = in[in_strides[6]];  // Input point 7: x(6)
        b_in4 = in[in_strides[9]];  // Input point 10: x(9)
        b_in5 = in[in_strides[10]]; // Input point 11: x(10)
        b_in6 = in[in_strides[13]]; // Input point 14: x(13)
        b_in7 = in[in_strides[14]]; // Input point 13: x(12)

        b_s0 = b_in6 + b_in0;
        b_s1 = b_in6 - b_in0;
        b_s2 = b_in4 + b_in2;
        b_s3 = b_in4 - b_in2;
        b_s4 = b_in7 + b_in1;
        b_s5 = b_in7 - b_in1;
        b_s6 = b_in5 + b_in3;
        b_s7 = b_in5 - b_in3;
        b_s8 = b_s5 + b_s7;
        b_s9 = b_s5 - b_s7;
        b_s10 = b_s0 - b_s2;
        b_s11 = b_s0 + b_s2;

        b_t0 = CRTM_8_1 * b_s8;
        b_t1 = CRTM_8_1 * b_s10;
        b_s12 = b_s6 + b_s1;
        b_s13 = b_s6 - b_s1;
        b_s14 = b_s3 + b_s4;
        b_s15 = b_s3 - b_s4;
        b_t2 = CRTM_8_2 * b_s12;
        b_t3 = CRTM_8_3 * b_s13;
        b_t4 = CRTM_8_3 * b_s14;
        b_t5 = CRTM_8_2 * b_s15;
        b_t6 = CRTM_8_3 * b_s12;
        b_t7 = CRTM_8_2 * b_s13;
        b_t8 = CRTM_8_2 * b_s14;
        b_t9 = CRTM_8_3 * b_s15;

        out[out_strides[1]] = CRTM_8_4 * b_s11; // Output pt 2: X(1)
        out[out_strides[3]] = -b_t2 - b_t4;     // Output pt 4: X(3)
        out[out_strides[5]] = b_t0 + b_t1;      // Output pt 6: X(5)
        out[out_strides[7]] = b_t3 + b_t5;      // Output pt 8: X(7)
        out[out_strides[9]] = CRTM_8_4 * b_s9;  // Output pt 10: X(9)
        out[out_strides[11]] = b_t6 - b_t8;     // Output pt 12: X(11)
        out[out_strides[13]] = b_t0 - b_t1;     // Output pt 14: X(12)
        out[out_strides[15]] = b_t9 - b_t7;     // Output pt 16: X(15)
    }
    AOCLFFTZ_LOG(DEBUG, global_logger_mode, "Exit");
}

kfft_ register_kernel_r2hcf_rfft8avx256(FFTZ_UINT8 precision,
                                        FFTZ_UINT8 direction)
{
    if (direction == FORWARD_FFT_DIR)
    {
        if (precision == DT_FLOAT)
        {
            return r2hcf_rfft8avx256_fp32_fwd;
        }
        else if (precision == DT_DOUBLE)
        {
            return r2hcf_rfft8avx256_fp64_fwd;
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
            return r2hcf_rfft8avx256_fp32_bwd;
        }
        else if (precision == DT_DOUBLE)
        {
            return r2hcf_rfft8avx256_fp64_bwd;
        }
        else
        {
            return NULL;
        }
    }
}
