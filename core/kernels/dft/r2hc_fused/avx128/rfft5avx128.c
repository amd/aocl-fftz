// Copyright Advanced Micro Devices, Inc.
// SPDX-License-Identifier: BSD-3-Clause

/** @file rfft5avx128.c
 *
 *  @brief Radix-5 r2hc_fused Real-FFT kernel with with AVX-128 operations using
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
#include "core/kernels/simd_includes/r2hc_simd_avx128.h"

static const ops_cycles_t ops_cnt[NUM_PRECISIONS][NUM_FFT_DIRS] =
                                                {{{0, 12, 24, 64, 48, 1},
                                                  {0, 14, 24, 64, 48, 1}},
                                                 {{0, 12, 24, 32, 8, 1},
                                                  {0, 14, 24, 32, 8, 1}}};

ops_cycles_t get_ops_cnt_r2hcf_rfft5avx128(FFTZ_UINT8 precision,
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

static FFTZ_VOID r2hcf_rfft5avx128_fp32_fwd(FFTZ_VOID *in_real,
                                            FFTZ_VOID *in_complex,
                                            FFTZ_VOID *out_real,
                                            FFTZ_VOID *out_complex,
                                            FFTZ_INTP n,
                                            aoclfftz_strides_t *strides,
                                            FFTZ_VOID *twd, FFTZ_UINT8 flag)
{
    AOCLFFTZ_LOG(DEBUG, global_logger_mode, "Enter");
    const FFTZ_FLOAT CRTM_5_1 =
        0.559016994374947424102293417182819058860154590f;
    const FFTZ_FLOAT CRTM_5_2 =
        0.951056516295153572116439333379382143405698632f;
    const FFTZ_FLOAT CRTM_5_3 =
        0.587785252292473129168705954639072768597652438f;
    const FFTZ_FLOAT CRTM_5_4 =
        0.250000000000000000000000000000000000000000000f;

    FFTZ_FLOAT *in_r = (FFTZ_FLOAT *)in_real;
    FFTZ_FLOAT *out_r = (FFTZ_FLOAT *)out_real;
    FFTZ_FLOAT *out_cp = (FFTZ_FLOAT *)out_complex;

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
    FFTZ_INTP v_out_dc_nyq_stride = strides->v_out_sym_stride;
    FFTZ_UINT8 is_contiguous_out_dc_nyq = (v_out_dc_nyq_stride == 1);

    FFTZ_INTP cnt;
    FFTZ_FLOAT *curr_in, *curr_out;
    FFTZ_INTP N = n / NUM_SETS_REAL_128_S;

    __m128 v_CRTM_5_1 = _mm_set1_ps(CRTM_5_1);
    __m128 v_CRTM_5_2 = _mm_set1_ps(CRTM_5_2);
    __m128 v_CRTM_5_3 = _mm_set1_ps(CRTM_5_3);
    __m128 v_CRTM_5_4 = _mm_set1_ps(CRTM_5_4);

    for (cnt = 0; cnt < N; cnt++)
    {
        /* Standard DFT */
        __m128 av_in0, av_in1, av_in2, av_in3, av_in4;
        __m128 av_s1, av_s2, av_s3, av_s4, av_s5, av_s6, av_s7;
        __m128 av_t1, av_t2, av_t3, av_t4, av_t5, av_t6;
        __m128 v_out0, v_out1, v_out2, v_out3, v_out4, v_out5, v_out6, v_out7,
               v_out8, v_out9;

        curr_in = in_r;
        curr_out = out_cp;

        // Input point 1: x(0)
        LDR_128_S(curr_in, v_in_stride, av_in0, is_contiguous_in);
        // Input point 3: x(2)
        curr_in = in_r + in_strides[2];
        LDR_128_S(curr_in, v_in_stride, av_in1, is_contiguous_in);
        // Input point 5: x(4)
        curr_in = in_r + in_strides[4];
        LDR_128_S(curr_in, v_in_stride, av_in2, is_contiguous_in);
        // Input point 7: x(6)
        curr_in = in_r + in_strides[6];
        LDR_128_S(curr_in, v_in_stride, av_in3, is_contiguous_in);
        // Input point 9: x(8)
        curr_in = in_r + in_strides[8];
        LDR_128_S(curr_in, v_in_stride, av_in4, is_contiguous_in);

        av_s1 = _mm_add_ps(av_in1, av_in4);
        av_s2 = _mm_sub_ps(av_in4, av_in1);
        av_s3 = _mm_add_ps(av_in2, av_in3);
        av_s4 = _mm_sub_ps(av_in2, av_in3);
        av_s5 = _mm_add_ps(av_s1, av_s3);

        av_t1 = _mm_mul_ps(v_CRTM_5_4, av_s5);
        av_s6 = _mm_sub_ps(av_s1, av_s3);

        av_s7 = _mm_sub_ps(av_in0, av_t1);
        av_t2 = _mm_mul_ps(v_CRTM_5_1, av_s6);

        av_t3 = _mm_mul_ps(v_CRTM_5_3, av_s4);
        av_t4 = _mm_mul_ps(v_CRTM_5_2, av_s2);
        av_t5 = _mm_mul_ps(v_CRTM_5_2, av_s4);
        av_t6 = _mm_mul_ps(v_CRTM_5_3, av_s2);

        // Output point 1: X(0)
        v_out0 = _mm_add_ps(av_in0, av_s5);
        curr_out = out_r;
        STR_128_S(curr_out, v_out_dc_nyq_stride, v_out0, is_contiguous_out_dc_nyq);
        // Output point 4: X(3) & Output point 5: X(4)
        curr_out = out_cp + out_strides[3];
        v_out3 = _mm_add_ps(av_s7, av_t2);
        v_out4 = _mm_sub_ps(av_t4, av_t3);
        STRI_2x128_S(curr_out, v_out_stride, v_out3, v_out4);
        // Output point 8: X(7) & Output point 9: X(8)
        curr_out = out_cp + out_strides[7];
        v_out7 = _mm_sub_ps(av_s7, av_t2);
        v_out8 = _mm_add_ps(av_t5, av_t6);
        STRI_2x128_S(curr_out, v_out_stride, v_out7, v_out8);

        /* Shifted DFT */
        __m128 bv_in0, bv_in1, bv_in2, bv_in3, bv_in4;
        __m128 bv_s1, bv_s2, bv_s3, bv_s4, bv_s5, bv_s6, bv_s7;
        __m128 bv_t1, bv_t2, bv_t3, bv_t4, bv_t5, bv_t6;

        // Input point 2: x(1)
        curr_in = in_r + in_strides[1];
        LDR_128_S(curr_in, v_in_stride, bv_in0, is_contiguous_in);
        // Input point 4: x(3)
        curr_in = in_r + in_strides[3];
        LDR_128_S(curr_in, v_in_stride, bv_in1, is_contiguous_in);
        // Input point 6: x(5)
        curr_in = in_r + in_strides[5];
        LDR_128_S(curr_in, v_in_stride, bv_in2, is_contiguous_in);
        // Input point 8: x(7)
        curr_in = in_r + in_strides[7];
        LDR_128_S(curr_in, v_in_stride, bv_in3, is_contiguous_in);
        // Input point 10: x(9)
        curr_in = in_r + in_strides[9];
        LDR_128_S(curr_in, v_in_stride, bv_in4, is_contiguous_in);

        bv_s1 = _mm_add_ps(bv_in1, bv_in4);
        bv_s2 = _mm_sub_ps(bv_in1, bv_in4);
        bv_s3 = _mm_add_ps(bv_in2, bv_in3);
        bv_s4 = _mm_sub_ps(bv_in2, bv_in3);
        bv_s5 = _mm_sub_ps(bv_s2, bv_s4);

        bv_s6 = _mm_add_ps(bv_s2, bv_s4);
        bv_t1 = _mm_mul_ps(v_CRTM_5_4, bv_s5);
        bv_s7 = _mm_add_ps(bv_in0, bv_t1);
        bv_t2 = _mm_mul_ps(v_CRTM_5_1, bv_s6);

        bv_t3 = _mm_mul_ps(v_CRTM_5_2, bv_s3);
        bv_t4 = _mm_mul_ps(v_CRTM_5_3, bv_s1);
        bv_t5 = _mm_mul_ps(v_CRTM_5_3, bv_s3);
        bv_t6 = _mm_mul_ps(v_CRTM_5_2, bv_s1);

        // Output point 2: X(1) & Output point 3: X(2)
        curr_out = out_cp + out_strides[1];
        v_out1 = _mm_add_ps(bv_s7, bv_t2);
        v_out2 = _mm_sub_ps(NEGATE_128_S(bv_t4), bv_t3);
        STRI_2x128_S(curr_out, v_out_stride, v_out1, v_out2);
        // Output point 6: X(5) & Output point 7: X(6)
        curr_out = out_cp + out_strides[5];
        v_out5 = _mm_sub_ps(bv_s7, bv_t2);
        v_out6 = _mm_sub_ps(bv_t5, bv_t6);
        STRI_2x128_S(curr_out, v_out_stride, v_out5, v_out6);
        // Output point 10: X(9)
        curr_out = out_r + out_strides[9];
        v_out9 = _mm_sub_ps(bv_in0, bv_s5);
        STR_128_S(curr_out, v_out_dc_nyq_stride, v_out9, is_contiguous_out_dc_nyq);

        in_r += v_in_stride * NUM_SETS_REAL_128_S;
        out_cp += v_out_stride * NUM_SETS_REAL_128_S;
        out_r += v_out_dc_nyq_stride * NUM_SETS_REAL_128_S;
    }
    // tail cases
    if (n & 2)
    {
        /* Standard DFT */
        __m128 av_in0, av_in1, av_in2, av_in3, av_in4;
        __m128 av_s1, av_s2, av_s3, av_s4, av_s5, av_s6, av_s7;
        __m128 av_t1, av_t2, av_t3, av_t4, av_t5, av_t6;
        __m128 v_out0, v_out1, v_out2, v_out3, v_out4, v_out5, v_out6, v_out7,
               v_out8, v_out9;

        curr_in = in_r;
        curr_out = out_cp;

        // Input point 1: x(0)
        LDHR_128_S(curr_in, v_in_stride, av_in0);
        // Input point 3: x(2)
        curr_in = in_r + in_strides[2];
        LDHR_128_S(curr_in, v_in_stride, av_in1);
        // Input point 5: x(4)
        curr_in = in_r + in_strides[4];
        LDHR_128_S(curr_in, v_in_stride, av_in2);
        // Input point 7: x(6)
        curr_in = in_r + in_strides[6];
        LDHR_128_S(curr_in, v_in_stride, av_in3);
        // Input point 9: x(8)
        curr_in = in_r + in_strides[8];
        LDHR_128_S(curr_in, v_in_stride, av_in4);

        av_s1 = _mm_add_ps(av_in1, av_in4);
        av_s2 = _mm_sub_ps(av_in4, av_in1);
        av_s3 = _mm_add_ps(av_in2, av_in3);
        av_s4 = _mm_sub_ps(av_in2, av_in3);
        av_s5 = _mm_add_ps(av_s1, av_s3);

        av_t1 = _mm_mul_ps(v_CRTM_5_4, av_s5);
        av_s6 = _mm_sub_ps(av_s1, av_s3);

        av_s7 = _mm_sub_ps(av_in0, av_t1);
        av_t2 = _mm_mul_ps(v_CRTM_5_1, av_s6);

        av_t3 = _mm_mul_ps(v_CRTM_5_3, av_s4);
        av_t4 = _mm_mul_ps(v_CRTM_5_2, av_s2);
        av_t5 = _mm_mul_ps(v_CRTM_5_2, av_s4);
        av_t6 = _mm_mul_ps(v_CRTM_5_3, av_s2);

        // Output point 1: X(0)
        v_out0 = _mm_add_ps(av_in0, av_s5);
        curr_out = out_r;
        STHR_128_S(curr_out, v_out_dc_nyq_stride, v_out0);
        // Output point 4: X(3) & Output point 5: X(4)
        curr_out = out_cp + out_strides[3];
        v_out3 = _mm_add_ps(av_s7, av_t2);
        v_out4 = _mm_sub_ps(av_t4, av_t3);
        STHRI_2x128_S(curr_out, v_out_stride, v_out3, v_out4);
        // Output point 8: X(7) & Output point 9: X(8)
        curr_out = out_cp + out_strides[7];
        v_out7 = _mm_sub_ps(av_s7, av_t2);
        v_out8 = _mm_add_ps(av_t5, av_t6);
        STHRI_2x128_S(curr_out, v_out_stride, v_out7, v_out8);

        /* Shifted DFT */
        __m128 bv_in0, bv_in1, bv_in2, bv_in3, bv_in4;
        __m128 bv_s1, bv_s2, bv_s3, bv_s4, bv_s5, bv_s6, bv_s7;
        __m128 bv_t1, bv_t2, bv_t3, bv_t4, bv_t5, bv_t6;

        // Input point 2: x(1)
        curr_in = in_r + in_strides[1];
        LDHR_128_S(curr_in, v_in_stride, bv_in0);
        // Input point 4: x(3)
        curr_in = in_r + in_strides[3];
        LDHR_128_S(curr_in, v_in_stride, bv_in1);
        // Input point 6: x(5)
        curr_in = in_r + in_strides[5];
        LDHR_128_S(curr_in, v_in_stride, bv_in2);
        // Input point 8: x(7)
        curr_in = in_r + in_strides[7];
        LDHR_128_S(curr_in, v_in_stride, bv_in3);
        // Input point 10: x(9)
        curr_in = in_r + in_strides[9];
        LDHR_128_S(curr_in, v_in_stride, bv_in4);

        bv_s1 = _mm_add_ps(bv_in1, bv_in4);
        bv_s2 = _mm_sub_ps(bv_in1, bv_in4);
        bv_s3 = _mm_add_ps(bv_in2, bv_in3);
        bv_s4 = _mm_sub_ps(bv_in2, bv_in3);
        bv_s5 = _mm_sub_ps(bv_s2, bv_s4);

        bv_s6 = _mm_add_ps(bv_s2, bv_s4);
        bv_t1 = _mm_mul_ps(v_CRTM_5_4, bv_s5);
        bv_s7 = _mm_add_ps(bv_in0, bv_t1);
        bv_t2 = _mm_mul_ps(v_CRTM_5_1, bv_s6);

        bv_t3 = _mm_mul_ps(v_CRTM_5_2, bv_s3);
        bv_t4 = _mm_mul_ps(v_CRTM_5_3, bv_s1);
        bv_t5 = _mm_mul_ps(v_CRTM_5_3, bv_s3);
        bv_t6 = _mm_mul_ps(v_CRTM_5_2, bv_s1);

        // Output point 2: X(1) & Output point 3: X(2)
        curr_out = out_cp + out_strides[1];
        v_out1 = _mm_add_ps(bv_s7, bv_t2);
        v_out2 = _mm_sub_ps(NEGATE_128_S(bv_t4), bv_t3);
        STHRI_2x128_S(curr_out, v_out_stride, v_out1, v_out2);
        // Output point 6: X(5) & Output point 7: X(6)
        curr_out = out_cp + out_strides[5];
        v_out5 = _mm_sub_ps(bv_s7, bv_t2);
        v_out6 = _mm_sub_ps(bv_t5, bv_t6);
        STHRI_2x128_S(curr_out, v_out_stride, v_out5, v_out6);
        // Output point 10: X(9)
        curr_out = out_r + out_strides[9];
        v_out9 = _mm_sub_ps(bv_in0, bv_s5);
        STHR_128_S(curr_out, v_out_dc_nyq_stride, v_out9);

        in_r = in_r + (v_in_stride << 1);
        out_cp = out_cp + (v_out_stride << 1);
        out_r = out_r + (v_out_dc_nyq_stride << 1);
    }
    // tail cases
    if (n & 1)
    {
        /* Standard DFT */
        FFTZ_FLOAT a_in0, a_in1, a_in2, a_in3, a_in4;
        FFTZ_FLOAT a_s1, a_s2, a_s3, a_s4, a_s5, a_s6, a_s7;
        FFTZ_FLOAT a_t1, a_t2, a_t3, a_t4, a_t5, a_t6;

        a_in0 = *in_r;               // Input point 1: x(0)
        a_in1 = in_r[in_strides[2]]; // Input point 3: x(2)
        a_in2 = in_r[in_strides[4]]; // Input point 5: x(4)
        a_in3 = in_r[in_strides[6]]; // Input point 7: x(6)
        a_in4 = in_r[in_strides[8]]; // Input point 9: x(8)

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

        *out_r = a_in0 + a_s5;               // Output point 1: X(0)
        out_cp[out_strides[3]] = a_s7 + a_t2; // Output point 4: X(3)
        out_cp[out_strides[4]] = a_t4 - a_t3; // Output point 5: X(4)
        out_cp[out_strides[7]] = a_s7 - a_t2; // Output point 8: X(7)
        out_cp[out_strides[8]] = a_t5 + a_t6; // Output point 9: X(8)

        /* Shifted DFT */
        FFTZ_FLOAT b_in0, b_in1, b_in2, b_in3, b_in4;
        FFTZ_FLOAT b_s1, b_s2, b_s3, b_s4, b_s5, b_s6, b_s7;
        FFTZ_FLOAT b_t1, b_t2, b_t3, b_t4, b_t5, b_t6;

        b_in0 = in_r[in_strides[1]]; // Input point 2: x(1)
        b_in1 = in_r[in_strides[3]]; // Input point 4: x(3)
        b_in2 = in_r[in_strides[5]]; // Input point 6: x(5)
        b_in3 = in_r[in_strides[7]]; // Input point 8: x(7)
        b_in4 = in_r[in_strides[9]]; // Input point 10: x(9)

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

        out_cp[out_strides[1]] = b_s7 + b_t2;  // Output point 2: X(1)
        out_cp[out_strides[2]] = -b_t3 - b_t4; // Output point 3: X(2)
        out_cp[out_strides[5]] = b_s7 - b_t2;  // Output point 6: X(5)
        out_cp[out_strides[6]] = b_t5 - b_t6;  // Output point 7: X(6)
        out_r[out_strides[9]] = b_in0 - b_s5; // Output point 10: X(9)
    }
    AOCLFFTZ_LOG(DEBUG, global_logger_mode, "Exit");
}

static FFTZ_VOID r2hcf_rfft5avx128_fp32_bwd(FFTZ_VOID *in_real,
                                            FFTZ_VOID *in_complex,
                                            FFTZ_VOID *out_real,
                                            FFTZ_VOID *out_complex,
                                            FFTZ_INTP n,
                                            aoclfftz_strides_t *strides,
                                            FFTZ_VOID *twd, FFTZ_UINT8 flag)
{
    AOCLFFTZ_LOG(DEBUG, global_logger_mode, "Enter");
    const FFTZ_FLOAT CRTM_5_1 = 1.11803398874989484820458683436563811772030918f;
    const FFTZ_FLOAT CRTM_5_2 = 1.90211303259030714423287866675876428681139726f;
    const FFTZ_FLOAT CRTM_5_3 = 1.17557050458494625833741190927814553719530488f;
    const FFTZ_FLOAT CRTM_5_4 = 0.50000000000000000000000000000000000000000000f;
    const FFTZ_FLOAT CRTM_5_5 = 2.00000000000000000000000000000000000000000000f;

    FFTZ_FLOAT *in_r = (FFTZ_FLOAT *)in_real;
    FFTZ_FLOAT *in_cp = (FFTZ_FLOAT *)in_complex;
    FFTZ_FLOAT *out_r = (FFTZ_FLOAT *)out_real;
#ifdef VOLATILE_STRIDE_ARRAY
    volatile FFTZ_INTP *in_strides = strides->in_strides;
    volatile FFTZ_INTP *out_strides = strides->out_strides;
#else
    FFTZ_INTP *in_strides = strides->in_strides;
    FFTZ_INTP *out_strides = strides->out_strides;
#endif
    FFTZ_INTP v_in_stride = strides->v_in_stride;
    FFTZ_INTP v_out_stride = strides->v_out_stride;
    FFTZ_UINT8 is_contiguous_out = (v_out_stride == 1);
    FFTZ_INTP v_in_dc_nyq_stride = strides->v_in_sym_stride;
    FFTZ_UINT8 is_contiguous_in_dc_nyq = (v_in_dc_nyq_stride == 1);

    FFTZ_INTP cnt;
    FFTZ_FLOAT *curr_in, *curr_out;
    FFTZ_INTP N = n / NUM_SETS_REAL_128_S;

    __m128 v_CRTM_5_1 = _mm_set1_ps(CRTM_5_1);
    __m128 v_CRTM_5_2 = _mm_set1_ps(CRTM_5_2);
    __m128 v_CRTM_5_3 = _mm_set1_ps(CRTM_5_3);
    __m128 v_CRTM_5_4 = _mm_set1_ps(CRTM_5_4);
    __m128 v_CRTM_5_5 = _mm_set1_ps(CRTM_5_5);

    for (cnt = 0; cnt < N; cnt++)
    {
        /* Standard DFT */
        __m128 av_in0, av_in1, av_in2, av_in3, av_in4;
        __m128 av_s1, av_s2, av_s3, av_s4, av_s5, av_s6, av_s7;
        __m128 av_t1, av_t2, av_t3, av_t4, av_t5, av_t6, av_t7;
        __m128 v_out0, v_out1, v_out2, v_out3, v_out4, v_out5, v_out6, v_out7,
               v_out8, v_out9;

        curr_in = in_r;
        curr_out = out_r;

        // Input point 1: x(0)
        LDR_128_S(curr_in, v_in_dc_nyq_stride, av_in0, is_contiguous_in_dc_nyq);
        // Input point 4: x(3) & Input point 5: x(4)
        curr_in = in_cp + in_strides[3];
        LDRI_2x128_S(curr_in, v_in_stride, av_in1, av_in2);
        // Input point 8: x(7) & Input point 9: x(8)
        curr_in = in_cp + in_strides[7];
        LDRI_2x128_S(curr_in, v_in_stride, av_in3, av_in4);

        av_s1 = _mm_add_ps(av_in1, av_in3);
        av_s2 = _mm_sub_ps(av_in1, av_in3);
        av_t1 = _mm_mul_ps(v_CRTM_5_4, av_s1);
        av_t2 = _mm_mul_ps(v_CRTM_5_1, av_s2);
        av_s3 = _mm_sub_ps(av_in0, av_t1);
        av_s4 = _mm_add_ps(av_s3, av_t2);
        av_s5 = _mm_sub_ps(av_s3, av_t2);

        av_t3 = _mm_mul_ps(v_CRTM_5_2, av_in4);
        av_t4 = _mm_mul_ps(v_CRTM_5_3, av_in2);
        av_t5 = _mm_mul_ps(v_CRTM_5_3, av_in4);
        av_t6 = _mm_mul_ps(v_CRTM_5_2, av_in2);
        av_t7 = _mm_mul_ps(v_CRTM_5_5, av_s1);

        av_s6 = _mm_add_ps(av_t6, av_t5);
        av_s7 = _mm_sub_ps(av_t3, av_t4);

        // Output pt 1: X(0)
        v_out0 = _mm_add_ps(av_in0, av_t7);
        STR_128_S(curr_out, v_out_stride, v_out0, is_contiguous_out);
        // Output pt 3: X(2)
        curr_out = out_r + out_strides[2];
        v_out2 = _mm_sub_ps(av_s4, av_s6);
        STR_128_S(curr_out, v_out_stride, v_out2, is_contiguous_out);
        // Output pt 5: X(4)
        curr_out = out_r + out_strides[4];
        v_out4 = _mm_add_ps(av_s5, av_s7);
        STR_128_S(curr_out, v_out_stride, v_out4, is_contiguous_out);
        // Output pt 7: X(6)
        curr_out = out_r + out_strides[6];
        v_out6 = _mm_sub_ps(av_s5, av_s7);
        STR_128_S(curr_out, v_out_stride, v_out6, is_contiguous_out);
        // Output pt 9: X(8)
        curr_out = out_r + out_strides[8];
        v_out8 = _mm_add_ps(av_s4, av_s6);
        STR_128_S(curr_out, v_out_stride, v_out8, is_contiguous_out);

        /* Shifted DFT */
        __m128 bv_in0, bv_in1, bv_in2, bv_in3, bv_in4;
        __m128 bv_s1, bv_s2, bv_s3, bv_s4, bv_s5, bv_s6, bv_s7;
        __m128 bv_t1, bv_t2, bv_t3, bv_t4, bv_t5, bv_t6, bv_t7;

        // Input point 2: x(1) & Input point 3: x(2)
        curr_in = in_cp + in_strides[1];
        LDRI_2x128_S(curr_in, v_in_stride, bv_in0, bv_in1);
        // Input point 6: x(5) & Input point 7: x(6)
        curr_in = in_cp + in_strides[5];
        LDRI_2x128_S(curr_in, v_in_stride, bv_in2, bv_in3);
        // Input point 10: x(9)
        curr_in = in_r + in_strides[9];
        LDR_128_S(curr_in, v_in_dc_nyq_stride, bv_in4, is_contiguous_in_dc_nyq);

        bv_s1 = _mm_add_ps(bv_in0, bv_in2);
        bv_s2 = _mm_sub_ps(bv_in2, bv_in0);
        bv_t1 = _mm_mul_ps(v_CRTM_5_1, bv_s2);
        bv_t2 = _mm_mul_ps(v_CRTM_5_4, bv_s1);
        bv_s3 = _mm_sub_ps(bv_in4, bv_t2);
        bv_s4 = _mm_add_ps(bv_s3, bv_t1);
        bv_s5 = _mm_sub_ps(bv_s3, bv_t1);

        bv_t3 = _mm_mul_ps(v_CRTM_5_3, bv_in1);
        bv_t4 = _mm_mul_ps(v_CRTM_5_2, bv_in3);
        bv_t5 = _mm_mul_ps(v_CRTM_5_2, bv_in1);
        bv_t6 = _mm_mul_ps(v_CRTM_5_3, bv_in3);
        bv_s6 = _mm_add_ps(bv_t3, bv_t4);

        bv_s7 = _mm_sub_ps(bv_t6, bv_t5);
        bv_t7 = _mm_mul_ps(v_CRTM_5_5, bv_s1);

        // Output pt 2: X(1)
        curr_out = out_r + out_strides[1];
        v_out1 = _mm_add_ps(bv_in4, bv_t7);
        STR_128_S(curr_out, v_out_stride, v_out1, is_contiguous_out);
        // Output pt 4: X(3)
        curr_out = out_r + out_strides[3];
        v_out3 = _mm_sub_ps(NEGATE_128_S(bv_s6), bv_s4);
        STR_128_S(curr_out, v_out_stride, v_out3, is_contiguous_out);
        // Output pt 6: X(5)
        curr_out = out_r + out_strides[5];
        v_out5 = _mm_add_ps(bv_s5, bv_s7);
        STR_128_S(curr_out, v_out_stride, v_out5, is_contiguous_out);
        // Output pt 8: X(7)
        curr_out = out_r + out_strides[7];
        v_out7 = _mm_sub_ps(bv_s7, bv_s5);
        STR_128_S(curr_out, v_out_stride, v_out7, is_contiguous_out);
        // Output pt 10: X(9)
        curr_out = out_r + out_strides[9];
        v_out9 = _mm_sub_ps(bv_s4, bv_s6);
        STR_128_S(curr_out, v_out_stride, v_out9, is_contiguous_out);

        in_cp += v_in_stride * NUM_SETS_REAL_128_S;
        in_r += v_in_dc_nyq_stride * NUM_SETS_REAL_128_S;
        out_r += v_out_stride * NUM_SETS_REAL_128_S;
    }
    // tail cases
    if (n & 2)
    {
        /* Standard DFT */
        __m128 av_in0, av_in1, av_in2, av_in3, av_in4;
        __m128 av_s1, av_s2, av_s3, av_s4, av_s5, av_s6, av_s7;
        __m128 av_t1, av_t2, av_t3, av_t4, av_t5, av_t6, av_t7;
        __m128 v_out0, v_out1, v_out2, v_out3, v_out4, v_out5, v_out6, v_out7,
               v_out8, v_out9;

        curr_in = in_r;
        curr_out = out_r;

        // Input point 1: x(0)
        LDHR_128_S(curr_in, v_in_dc_nyq_stride, av_in0);
        // Input point 4: x(3) & Input point 5: x(4)
        curr_in = in_cp + in_strides[3];
        LDHRI_2x128_S(curr_in, v_in_stride, av_in1, av_in2);
        // Input point 8: x(7) & Input point 9: x(8)
        curr_in = in_cp + in_strides[7];
        LDHRI_2x128_S(curr_in, v_in_stride, av_in3, av_in4);

        av_s1 = _mm_add_ps(av_in1, av_in3);
        av_s2 = _mm_sub_ps(av_in1, av_in3);
        av_t1 = _mm_mul_ps(v_CRTM_5_4, av_s1);
        av_t2 = _mm_mul_ps(v_CRTM_5_1, av_s2);
        av_s3 = _mm_sub_ps(av_in0, av_t1);
        av_s4 = _mm_add_ps(av_s3, av_t2);
        av_s5 = _mm_sub_ps(av_s3, av_t2);

        av_t3 = _mm_mul_ps(v_CRTM_5_2, av_in4);
        av_t4 = _mm_mul_ps(v_CRTM_5_3, av_in2);
        av_t5 = _mm_mul_ps(v_CRTM_5_3, av_in4);
        av_t6 = _mm_mul_ps(v_CRTM_5_2, av_in2);
        av_t7 = _mm_mul_ps(v_CRTM_5_5, av_s1);

        av_s6 = _mm_add_ps(av_t6, av_t5);
        av_s7 = _mm_sub_ps(av_t3, av_t4);

        // Output pt 1: X(0)
        v_out0 = _mm_add_ps(av_in0, av_t7);
        STHR_128_S(curr_out, v_out_stride, v_out0);
        // Output pt 3: X(2)
        curr_out = out_r + out_strides[2];
        v_out2 = _mm_sub_ps(av_s4, av_s6);
        STHR_128_S(curr_out, v_out_stride, v_out2);
        // Output pt 5: X(4)
        curr_out = out_r + out_strides[4];
        v_out4 = _mm_add_ps(av_s5, av_s7);
        STHR_128_S(curr_out, v_out_stride, v_out4);
        // Output pt 7: X(6)
        curr_out = out_r + out_strides[6];
        v_out6 = _mm_sub_ps(av_s5, av_s7);
        STHR_128_S(curr_out, v_out_stride, v_out6);
        // Output pt 9: X(8)
        curr_out = out_r + out_strides[8];
        v_out8 = _mm_add_ps(av_s4, av_s6);
        STHR_128_S(curr_out, v_out_stride, v_out8);

        /* Shifted DFT */
        __m128 bv_in0, bv_in1, bv_in2, bv_in3, bv_in4;
        __m128 bv_s1, bv_s2, bv_s3, bv_s4, bv_s5, bv_s6, bv_s7;
        __m128 bv_t1, bv_t2, bv_t3, bv_t4, bv_t5, bv_t6, bv_t7;

        // Input point 2: x(1) & Input point 3: x(2)
        curr_in = in_cp + in_strides[1];
        LDHRI_2x128_S(curr_in, v_in_stride, bv_in0, bv_in1);
        // Input point 6: x(5) & Input point 7: x(6)
        curr_in = in_cp + in_strides[5];
        LDHRI_2x128_S(curr_in, v_in_stride, bv_in2, bv_in3);
        // Input point 10: x(9)
        curr_in = in_r + in_strides[9];
        LDHR_128_S(curr_in, v_in_dc_nyq_stride, bv_in4);

        bv_s1 = _mm_add_ps(bv_in0, bv_in2);
        bv_s2 = _mm_sub_ps(bv_in2, bv_in0);
        bv_t1 = _mm_mul_ps(v_CRTM_5_1, bv_s2);
        bv_t2 = _mm_mul_ps(v_CRTM_5_4, bv_s1);
        bv_s3 = _mm_sub_ps(bv_in4, bv_t2);
        bv_s4 = _mm_add_ps(bv_s3, bv_t1);
        bv_s5 = _mm_sub_ps(bv_s3, bv_t1);

        bv_t3 = _mm_mul_ps(v_CRTM_5_3, bv_in1);
        bv_t4 = _mm_mul_ps(v_CRTM_5_2, bv_in3);
        bv_t5 = _mm_mul_ps(v_CRTM_5_2, bv_in1);
        bv_t6 = _mm_mul_ps(v_CRTM_5_3, bv_in3);
        bv_s6 = _mm_add_ps(bv_t3, bv_t4);

        bv_s7 = _mm_sub_ps(bv_t6, bv_t5);
        bv_t7 = _mm_mul_ps(v_CRTM_5_5, bv_s1);

        // Output pt 2: X(1)
        curr_out = out_r + out_strides[1];
        v_out1 = _mm_add_ps(bv_in4, bv_t7);
        STHR_128_S(curr_out, v_out_stride, v_out1);
        // Output pt 4: X(3)
        curr_out = out_r + out_strides[3];
        v_out3 = _mm_sub_ps(NEGATE_128_S(bv_s6), bv_s4);
        STHR_128_S(curr_out, v_out_stride, v_out3);
        // Output pt 6: X(5)
        curr_out = out_r + out_strides[5];
        v_out5 = _mm_add_ps(bv_s5, bv_s7);
        STHR_128_S(curr_out, v_out_stride, v_out5);
        // Output pt 8: X(7)
        curr_out = out_r + out_strides[7];
        v_out7 = _mm_sub_ps(bv_s7, bv_s5);
        STHR_128_S(curr_out, v_out_stride, v_out7);
        // Output pt 10: X(9)
        curr_out = out_r + out_strides[9];
        v_out9 = _mm_sub_ps(bv_s4, bv_s6);
        STHR_128_S(curr_out, v_out_stride, v_out9);

        in_cp = in_cp + (v_in_stride << 1);
        in_r = in_r + (v_in_dc_nyq_stride << 1);
        out_r = out_r + (v_out_stride << 1);
    }
    // tail cases
    if (n & 1)
    {
        /* Standard DFT */
        FFTZ_FLOAT a_in0, a_in1, a_in2, a_in3, a_in4;
        FFTZ_FLOAT a_s1, a_s2, a_s3, a_s4, a_s5, a_s6, a_s7;
        FFTZ_FLOAT a_t1, a_t2, a_t3, a_t4, a_t5, a_t6, a_t7;

        a_in0 = *in_r;               // Input point 1: x(0)
        a_in1 = in_cp[in_strides[3]]; // Input point 4: x(3)
        a_in2 = in_cp[in_strides[4]]; // Input point 5: x(4)
        a_in3 = in_cp[in_strides[7]]; // Input point 8: x(7)
        a_in4 = in_cp[in_strides[8]]; // Input point 9: x(8)

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

        *out_r = a_in0 + a_t7;               // Output point 1: X(0)
        out_r[out_strides[2]] = a_s4 - a_s6; // Output point 3: X(2)
        out_r[out_strides[4]] = a_s5 + a_s7; // Output point 5: X(4)
        out_r[out_strides[6]] = a_s5 - a_s7; // Output point 7: X(6)
        out_r[out_strides[8]] = a_s4 + a_s6; // Output point 9: X(8)

        /* Shifted DFT */
        FFTZ_FLOAT b_in0, b_in1, b_in2, b_in3, b_in4;
        FFTZ_FLOAT b_s1, b_s2, b_s3, b_s4, b_s5, b_s6, b_s7;
        FFTZ_FLOAT b_t1, b_t2, b_t3, b_t4, b_t5, b_t6, b_t7;

        b_in0 = in_cp[in_strides[1]]; // Input point 2: x(1)
        b_in1 = in_cp[in_strides[2]]; // Input point 3: x(2)
        b_in2 = in_cp[in_strides[5]]; // Input point 6: x(5)
        b_in3 = in_cp[in_strides[6]]; // Input point 7: x(6)
        b_in4 = in_r[in_strides[9]]; // Input point 10: x(9)

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

        out_r[out_strides[1]] = b_in4 + b_s7; // Output point 2: X(1)
        out_r[out_strides[3]] = -b_t7 - b_s4; // Output point 4: X(3)
        out_r[out_strides[5]] = b_s5 + b_s6;  // Output point 6: X(5)
        out_r[out_strides[7]] = b_s6 - b_s5;  // Output point 8: X(7)
        out_r[out_strides[9]] = b_s4 - b_t7;  // Output point 10: X(9)
    }
    AOCLFFTZ_LOG(DEBUG, global_logger_mode, "Exit");
}

static FFTZ_VOID r2hcf_rfft5avx128_fp64_fwd(FFTZ_VOID *in_real,
                                            FFTZ_VOID *in_complex,
                                            FFTZ_VOID *out_real,
                                            FFTZ_VOID *out_complex,
                                            FFTZ_INTP n,
                                            aoclfftz_strides_t *strides,
                                            FFTZ_VOID *twd, FFTZ_UINT8 flag)
{
    AOCLFFTZ_LOG(DEBUG, global_logger_mode, "Enter");
    const FFTZ_DOUBLE CRTM_5_1 =
        0.559016994374947424102293417182819058860154590;
    const FFTZ_DOUBLE CRTM_5_2 =
        0.951056516295153572116439333379382143405698632;
    const FFTZ_DOUBLE CRTM_5_3 =
        0.587785252292473129168705954639072768597652438;
    const FFTZ_DOUBLE CRTM_5_4 =
        0.250000000000000000000000000000000000000000000;

    FFTZ_DOUBLE *in_r = (FFTZ_DOUBLE *)in_real;
    FFTZ_DOUBLE *out_r = (FFTZ_DOUBLE *)out_real;
    FFTZ_DOUBLE *out_cp = (FFTZ_DOUBLE *)out_complex;

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
    FFTZ_INTP v_out_dc_nyq_stride = strides->v_out_sym_stride;
    FFTZ_UINT8 is_contiguous_out_dc_nyq = (v_out_dc_nyq_stride == 1);

    FFTZ_INTP cnt;
    FFTZ_DOUBLE *curr_in, *curr_out;
    FFTZ_INTP N = n / NUM_SETS_REAL_128_D;

    __m128d v_CRTM_5_1 = _mm_set1_pd(CRTM_5_1);
    __m128d v_CRTM_5_2 = _mm_set1_pd(CRTM_5_2);
    __m128d v_CRTM_5_3 = _mm_set1_pd(CRTM_5_3);
    __m128d v_CRTM_5_4 = _mm_set1_pd(CRTM_5_4);

    for (cnt = 0; cnt < N; cnt++)
    {
        /* Standard DFT */
        __m128d av_in0, av_in1, av_in2, av_in3, av_in4;
        __m128d av_s1, av_s2, av_s3, av_s4, av_s5, av_s6, av_s7;
        __m128d av_t1, av_t2, av_t3, av_t4, av_t5, av_t6;
        __m128d v_out0, v_out1, v_out2, v_out3, v_out4, v_out5, v_out6, v_out7,
                v_out8, v_out9;

        curr_in = in_r;
        curr_out = out_cp;

        // Input point 1: x(0)
        LDR_128_D(curr_in, v_in_stride, av_in0, is_contiguous_in);
        // Input point 3: x(2)
        curr_in = in_r + in_strides[2];
        LDR_128_D(curr_in, v_in_stride, av_in1, is_contiguous_in);
        // Input point 5: x(4)
        curr_in = in_r + in_strides[4];
        LDR_128_D(curr_in, v_in_stride, av_in2, is_contiguous_in);
        // Input point 7: x(6)
        curr_in = in_r + in_strides[6];
        LDR_128_D(curr_in, v_in_stride, av_in3, is_contiguous_in);
        // Input point 9: x(8)
        curr_in = in_r + in_strides[8];
        LDR_128_D(curr_in, v_in_stride, av_in4, is_contiguous_in);

        av_s1 = _mm_add_pd(av_in1, av_in4);
        av_s2 = _mm_sub_pd(av_in4, av_in1);
        av_s3 = _mm_add_pd(av_in2, av_in3);
        av_s4 = _mm_sub_pd(av_in2, av_in3);
        av_s5 = _mm_add_pd(av_s1, av_s3);

        av_t1 = _mm_mul_pd(v_CRTM_5_4, av_s5);
        av_s6 = _mm_sub_pd(av_s1, av_s3);

        av_s7 = _mm_sub_pd(av_in0, av_t1);
        av_t2 = _mm_mul_pd(v_CRTM_5_1, av_s6);

        av_t3 = _mm_mul_pd(v_CRTM_5_3, av_s4);
        av_t4 = _mm_mul_pd(v_CRTM_5_2, av_s2);
        av_t5 = _mm_mul_pd(v_CRTM_5_2, av_s4);
        av_t6 = _mm_mul_pd(v_CRTM_5_3, av_s2);

        // Output point 1: X(0)
        v_out0 = _mm_add_pd(av_in0, av_s5);
        curr_out = out_r;
        STR_128_D(curr_out, v_out_dc_nyq_stride, v_out0, is_contiguous_out_dc_nyq);
        // Output point 4: X(3) & Output point 5: X(4)
        curr_out = out_cp + out_strides[3];
        v_out3 = _mm_add_pd(av_s7, av_t2);
        v_out4 = _mm_sub_pd(av_t4, av_t3);
        STRI_2x128_D(curr_out, v_out_stride, v_out3, v_out4);
        // Output point 8: X(7) & Output point 9: X(8)
        curr_out = out_cp + out_strides[7];
        v_out7 = _mm_sub_pd(av_s7, av_t2);
        v_out8 = _mm_add_pd(av_t5, av_t6);
        STRI_2x128_D(curr_out, v_out_stride, v_out7, v_out8);

        /* Shifted DFT */
        __m128d bv_in0, bv_in1, bv_in2, bv_in3, bv_in4;
        __m128d bv_s1, bv_s2, bv_s3, bv_s4, bv_s5, bv_s6, bv_s7;
        __m128d bv_t1, bv_t2, bv_t3, bv_t4, bv_t5, bv_t6;

        // Input point 2: x(1)
        curr_in = in_r + in_strides[1];
        LDR_128_D(curr_in, v_in_stride, bv_in0, is_contiguous_in);
        // Input point 4: x(3)
        curr_in = in_r + in_strides[3];
        LDR_128_D(curr_in, v_in_stride, bv_in1, is_contiguous_in);
        // Input point 6: x(5)
        curr_in = in_r + in_strides[5];
        LDR_128_D(curr_in, v_in_stride, bv_in2, is_contiguous_in);
        // Input point 8: x(7)
        curr_in = in_r + in_strides[7];
        LDR_128_D(curr_in, v_in_stride, bv_in3, is_contiguous_in);
        // Input point 10: x(9)
        curr_in = in_r + in_strides[9];
        LDR_128_D(curr_in, v_in_stride, bv_in4, is_contiguous_in);

        bv_s1 = _mm_add_pd(bv_in1, bv_in4);
        bv_s2 = _mm_sub_pd(bv_in1, bv_in4);
        bv_s3 = _mm_add_pd(bv_in2, bv_in3);
        bv_s4 = _mm_sub_pd(bv_in2, bv_in3);
        bv_s5 = _mm_sub_pd(bv_s2, bv_s4);

        bv_s6 = _mm_add_pd(bv_s2, bv_s4);
        bv_t1 = _mm_mul_pd(v_CRTM_5_4, bv_s5);
        bv_s7 = _mm_add_pd(bv_in0, bv_t1);
        bv_t2 = _mm_mul_pd(v_CRTM_5_1, bv_s6);

        bv_t3 = _mm_mul_pd(v_CRTM_5_2, bv_s3);
        bv_t4 = _mm_mul_pd(v_CRTM_5_3, bv_s1);
        bv_t5 = _mm_mul_pd(v_CRTM_5_3, bv_s3);
        bv_t6 = _mm_mul_pd(v_CRTM_5_2, bv_s1);

        // Output point 2: X(1) & Output point 3: X(2)
        curr_out = out_cp + out_strides[1];
        v_out1 = _mm_add_pd(bv_s7, bv_t2);
        v_out2 = _mm_sub_pd(NEGATE_128_D(bv_t4), bv_t3);
        STRI_2x128_D(curr_out, v_out_stride, v_out1, v_out2);
        // Output point 6: X(5) & Output point 7: X(6)
        curr_out = out_cp + out_strides[5];
        v_out5 = _mm_sub_pd(bv_s7, bv_t2);
        v_out6 = _mm_sub_pd(bv_t5, bv_t6);
        STRI_2x128_D(curr_out, v_out_stride, v_out5, v_out6);
        // Output point 10: X(9)
        curr_out = out_r + out_strides[9];
        v_out9 = _mm_sub_pd(bv_in0, bv_s5);
        STR_128_D(curr_out, v_out_dc_nyq_stride, v_out9, is_contiguous_out_dc_nyq);

        in_r += v_in_stride * NUM_SETS_REAL_128_D;
        out_cp += v_out_stride * NUM_SETS_REAL_128_D;
        out_r += v_out_dc_nyq_stride * NUM_SETS_REAL_128_D;
    }
    // tail cases
    if (n & 1)
    {
        /* Standard DFT */
        FFTZ_DOUBLE a_in0, a_in1, a_in2, a_in3, a_in4;
        FFTZ_DOUBLE a_s1, a_s2, a_s3, a_s4, a_s5, a_s6, a_s7;
        FFTZ_DOUBLE a_t1, a_t2, a_t3, a_t4, a_t5, a_t6;

        a_in0 = *in_r;               // Input point 1: x(0)
        a_in1 = in_r[in_strides[2]]; // Input point 3: x(2)
        a_in2 = in_r[in_strides[4]]; // Input point 5: x(4)
        a_in3 = in_r[in_strides[6]]; // Input point 7: x(6)
        a_in4 = in_r[in_strides[8]]; // Input point 9: x(8)

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

        *out_r = a_in0 + a_s5;               // Output point 1: X(0)
        out_cp[out_strides[3]] = a_s7 + a_t2; // Output point 4: X(3)
        out_cp[out_strides[4]] = a_t4 - a_t3; // Output point 5: X(4)
        out_cp[out_strides[7]] = a_s7 - a_t2; // Output point 8: X(7)
        out_cp[out_strides[8]] = a_t5 + a_t6; // Output point 9: X(8)

        /* Shifted DFT */
        FFTZ_DOUBLE b_in0, b_in1, b_in2, b_in3, b_in4;
        FFTZ_DOUBLE b_s1, b_s2, b_s3, b_s4, b_s5, b_s6, b_s7;
        FFTZ_DOUBLE b_t1, b_t2, b_t3, b_t4, b_t5, b_t6;

        b_in0 = in_r[in_strides[1]]; // Input point 2: x(1)
        b_in1 = in_r[in_strides[3]]; // Input point 4: x(3)
        b_in2 = in_r[in_strides[5]]; // Input point 6: x(5)
        b_in3 = in_r[in_strides[7]]; // Input point 8: x(7)
        b_in4 = in_r[in_strides[9]]; // Input point 10: x(9)

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

        out_cp[out_strides[1]] = b_s7 + b_t2;  // Output point 2: X(1)
        out_cp[out_strides[2]] = -b_t3 - b_t4; // Output point 3: X(2)
        out_cp[out_strides[5]] = b_s7 - b_t2;  // Output point 6: X(5)
        out_cp[out_strides[6]] = b_t5 - b_t6;  // Output point 7: X(6)
        out_r[out_strides[9]] = b_in0 - b_s5; // Output point 10: X(9)
    }
    AOCLFFTZ_LOG(DEBUG, global_logger_mode, "Exit");
}

static FFTZ_VOID r2hcf_rfft5avx128_fp64_bwd(FFTZ_VOID *in_real,
                                            FFTZ_VOID *in_complex,
                                            FFTZ_VOID *out_real,
                                            FFTZ_VOID *out_complex,
                                            FFTZ_INTP n,
                                            aoclfftz_strides_t *strides,
                                            FFTZ_VOID *twd, FFTZ_UINT8 flag)
{
    AOCLFFTZ_LOG(DEBUG, global_logger_mode, "Enter");
    const FFTZ_DOUBLE CRTM_5_1 = 1.11803398874989484820458683436563811772030918;
    const FFTZ_DOUBLE CRTM_5_2 = 1.90211303259030714423287866675876428681139726;
    const FFTZ_DOUBLE CRTM_5_3 = 1.17557050458494625833741190927814553719530488;
    const FFTZ_DOUBLE CRTM_5_4 = 0.50000000000000000000000000000000000000000000;
    const FFTZ_DOUBLE CRTM_5_5 =
        2.000000000000000000000000000000000000000000000;

    FFTZ_DOUBLE *in_r = (FFTZ_DOUBLE *)in_real;
    FFTZ_DOUBLE *in_cp = (FFTZ_DOUBLE *)in_complex;
    FFTZ_DOUBLE *out_r = (FFTZ_DOUBLE *)out_real;
#ifdef VOLATILE_STRIDE_ARRAY
    volatile FFTZ_INTP *in_strides = strides->in_strides;
    volatile FFTZ_INTP *out_strides = strides->out_strides;
#else
    FFTZ_INTP *in_strides = strides->in_strides;
    FFTZ_INTP *out_strides = strides->out_strides;
#endif
    FFTZ_INTP v_in_stride = strides->v_in_stride;
    FFTZ_INTP v_out_stride = strides->v_out_stride;
    FFTZ_UINT8 is_contiguous_out = (v_out_stride == 1);
    FFTZ_INTP v_in_dc_nyq_stride = strides->v_in_sym_stride;
    FFTZ_UINT8 is_contiguous_in_dc_nyq = (v_in_dc_nyq_stride == 1);

    FFTZ_INTP cnt;
    FFTZ_DOUBLE *curr_in, *curr_out;
    FFTZ_INTP N = n / NUM_SETS_REAL_128_D;

    __m128d v_CRTM_5_1 = _mm_set1_pd(CRTM_5_1);
    __m128d v_CRTM_5_2 = _mm_set1_pd(CRTM_5_2);
    __m128d v_CRTM_5_3 = _mm_set1_pd(CRTM_5_3);
    __m128d v_CRTM_5_4 = _mm_set1_pd(CRTM_5_4);
    __m128d v_CRTM_5_5 = _mm_set1_pd(CRTM_5_5);

    for (cnt = 0; cnt < N; cnt++)
    {
        /* Standard DFT */
        __m128d av_in0, av_in1, av_in2, av_in3, av_in4;
        __m128d av_s1, av_s2, av_s3, av_s4, av_s5, av_s6, av_s7;
        __m128d av_t1, av_t2, av_t3, av_t4, av_t5, av_t6, av_t7;
        __m128d v_out0, v_out1, v_out2, v_out3, v_out4, v_out5, v_out6, v_out7,
                v_out8, v_out9;

        curr_in = in_r;
        curr_out = out_r;

        // Input point 1: x(0)
        LDR_128_D(curr_in, v_in_dc_nyq_stride, av_in0, is_contiguous_in_dc_nyq);
        // Input point 4: x(3) & Input point 5: x(4)
        curr_in = in_cp + in_strides[3];
        LDRI_2x128_D(curr_in, v_in_stride, av_in1, av_in2);
        // Input point 8: x(7) & Input point 9: x(8)
        curr_in = in_cp + in_strides[7];
        LDRI_2x128_D(curr_in, v_in_stride, av_in3, av_in4);

        av_s1 = _mm_add_pd(av_in1, av_in3);
        av_s2 = _mm_sub_pd(av_in1, av_in3);
        av_t1 = _mm_mul_pd(v_CRTM_5_4, av_s1);
        av_t2 = _mm_mul_pd(v_CRTM_5_1, av_s2);
        av_s3 = _mm_sub_pd(av_in0, av_t1);
        av_s4 = _mm_add_pd(av_s3, av_t2);
        av_s5 = _mm_sub_pd(av_s3, av_t2);

        av_t3 = _mm_mul_pd(v_CRTM_5_2, av_in4);
        av_t4 = _mm_mul_pd(v_CRTM_5_3, av_in2);
        av_t5 = _mm_mul_pd(v_CRTM_5_3, av_in4);
        av_t6 = _mm_mul_pd(v_CRTM_5_2, av_in2);
        av_t7 = _mm_mul_pd(v_CRTM_5_5, av_s1);

        av_s6 = _mm_add_pd(av_t6, av_t5);
        av_s7 = _mm_sub_pd(av_t3, av_t4);

        // Output pt 1: X(0)
        v_out0 = _mm_add_pd(av_in0, av_t7);
        STR_128_D(curr_out, v_out_stride, v_out0, is_contiguous_out);
        // Output pt 3: X(2)
        curr_out = out_r + out_strides[2];
        v_out2 = _mm_sub_pd(av_s4, av_s6);
        STR_128_D(curr_out, v_out_stride, v_out2, is_contiguous_out);
        // Output pt 5: X(4)
        curr_out = out_r + out_strides[4];
        v_out4 = _mm_add_pd(av_s5, av_s7);
        STR_128_D(curr_out, v_out_stride, v_out4, is_contiguous_out);
        // Output pt 7: X(6)
        curr_out = out_r + out_strides[6];
        v_out6 = _mm_sub_pd(av_s5, av_s7);
        STR_128_D(curr_out, v_out_stride, v_out6, is_contiguous_out);
        // Output pt 9: X(8)
        curr_out = out_r + out_strides[8];
        v_out8 = _mm_add_pd(av_s4, av_s6);
        STR_128_D(curr_out, v_out_stride, v_out8, is_contiguous_out);

        /* Shifted DFT */
        __m128d bv_in0, bv_in1, bv_in2, bv_in3, bv_in4;
        __m128d bv_s1, bv_s2, bv_s3, bv_s4, bv_s5, bv_s6, bv_s7;
        __m128d bv_t1, bv_t2, bv_t3, bv_t4, bv_t5, bv_t6, bv_t7;

        // Input point 2: x(1) & Input point 3: x(2)
        curr_in = in_cp + in_strides[1];
        LDRI_2x128_D(curr_in, v_in_stride, bv_in0, bv_in1);
        // Input point 6: x(5) & Input point 7: x(6)
        curr_in = in_cp + in_strides[5];
        LDRI_2x128_D(curr_in, v_in_stride, bv_in2, bv_in3);
        // Input point 10: x(9)
        curr_in = in_r + in_strides[9];
        LDR_128_D(curr_in, v_in_dc_nyq_stride, bv_in4, is_contiguous_in_dc_nyq);

        bv_s1 = _mm_add_pd(bv_in0, bv_in2);
        bv_s2 = _mm_sub_pd(bv_in2, bv_in0);
        bv_t1 = _mm_mul_pd(v_CRTM_5_1, bv_s2);
        bv_t2 = _mm_mul_pd(v_CRTM_5_4, bv_s1);
        bv_s3 = _mm_sub_pd(bv_in4, bv_t2);
        bv_s4 = _mm_add_pd(bv_s3, bv_t1);
        bv_s5 = _mm_sub_pd(bv_s3, bv_t1);

        bv_t3 = _mm_mul_pd(v_CRTM_5_3, bv_in1);
        bv_t4 = _mm_mul_pd(v_CRTM_5_2, bv_in3);
        bv_t5 = _mm_mul_pd(v_CRTM_5_2, bv_in1);
        bv_t6 = _mm_mul_pd(v_CRTM_5_3, bv_in3);
        bv_s6 = _mm_add_pd(bv_t3, bv_t4);

        bv_s7 = _mm_sub_pd(bv_t6, bv_t5);
        bv_t7 = _mm_mul_pd(v_CRTM_5_5, bv_s1);

        // Output pt 2: X(1)
        curr_out = out_r + out_strides[1];
        v_out1 = _mm_add_pd(bv_in4, bv_t7);
        STR_128_D(curr_out, v_out_stride, v_out1, is_contiguous_out);
        // Output pt 4: X(3)
        curr_out = out_r + out_strides[3];
        v_out3 = _mm_sub_pd(NEGATE_128_D(bv_s6), bv_s4);
        STR_128_D(curr_out, v_out_stride, v_out3, is_contiguous_out);
        // Output pt 6: X(5)
        curr_out = out_r + out_strides[5];
        v_out5 = _mm_add_pd(bv_s5, bv_s7);
        STR_128_D(curr_out, v_out_stride, v_out5, is_contiguous_out);
        // Output pt 8: X(7)
        curr_out = out_r + out_strides[7];
        v_out7 = _mm_sub_pd(bv_s7, bv_s5);
        STR_128_D(curr_out, v_out_stride, v_out7, is_contiguous_out);
        // Output pt 10: X(9)
        curr_out = out_r + out_strides[9];
        v_out9 = _mm_sub_pd(bv_s4, bv_s6);
        STR_128_D(curr_out, v_out_stride, v_out9, is_contiguous_out);

        in_cp += v_in_stride * NUM_SETS_REAL_128_D;
        in_r += v_in_dc_nyq_stride * NUM_SETS_REAL_128_D;
        out_r += v_out_stride * NUM_SETS_REAL_128_D;
    }
    // tail cases
    if (n & 1)
    {
        /* Standard DFT */
        FFTZ_DOUBLE a_in0, a_in1, a_in2, a_in3, a_in4;
        FFTZ_DOUBLE a_s1, a_s2, a_s3, a_s4, a_s5, a_s6, a_s7;
        FFTZ_DOUBLE a_t1, a_t2, a_t3, a_t4, a_t5, a_t6, a_t7;

        a_in0 = *in_r;               // Input point 1: x(0)
        a_in1 = in_cp[in_strides[3]]; // Input point 4: x(3)
        a_in2 = in_cp[in_strides[4]]; // Input point 5: x(4)
        a_in3 = in_cp[in_strides[7]]; // Input point 8: x(7)
        a_in4 = in_cp[in_strides[8]]; // Input point 9: x(8)

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

        *out_r = a_in0 + a_t7;               // Output point 1: X(0)
        out_r[out_strides[2]] = a_s4 - a_s6; // Output point 3: X(2)
        out_r[out_strides[4]] = a_s5 + a_s7; // Output point 5: X(4)
        out_r[out_strides[6]] = a_s5 - a_s7; // Output point 7: X(6)
        out_r[out_strides[8]] = a_s4 + a_s6; // Output point 9: X(8)

        /* Shifted DFT */
        FFTZ_DOUBLE b_in0, b_in1, b_in2, b_in3, b_in4;
        FFTZ_DOUBLE b_s1, b_s2, b_s3, b_s4, b_s5, b_s6, b_s7;
        FFTZ_DOUBLE b_t1, b_t2, b_t3, b_t4, b_t5, b_t6, b_t7;

        b_in0 = in_cp[in_strides[1]]; // Input point 2: x(1)
        b_in1 = in_cp[in_strides[2]]; // Input point 3: x(2)
        b_in2 = in_cp[in_strides[5]]; // Input point 6: x(5)
        b_in3 = in_cp[in_strides[6]]; // Input point 7: x(6)
        b_in4 = in_r[in_strides[9]]; // Input point 10: x(9)

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

        out_r[out_strides[1]] = b_in4 + b_s7; // Output point 2: X(1)
        out_r[out_strides[3]] = -b_t7 - b_s4; // Output point 4: X(3)
        out_r[out_strides[5]] = b_s5 + b_s6;  // Output point 6: X(5)
        out_r[out_strides[7]] = b_s6 - b_s5;  // Output point 8: X(7)
        out_r[out_strides[9]] = b_s4 - b_t7;  // Output point 10: X(9)
    }
    AOCLFFTZ_LOG(DEBUG, global_logger_mode, "Exit");
}

kfft_ register_kernel_r2hcf_rfft5avx128(FFTZ_UINT8 precision,
                                        FFTZ_UINT8 direction)
{
    if (direction == FORWARD_FFT_DIR)
    {
        if (precision == DT_FLOAT)
        {
            return r2hcf_rfft5avx128_fp32_fwd;
        }
        else if (precision == DT_DOUBLE)
        {
            return r2hcf_rfft5avx128_fp64_fwd;
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
            return r2hcf_rfft5avx128_fp32_bwd;
        }
        else if (precision == DT_DOUBLE)
        {
            return r2hcf_rfft5avx128_fp64_bwd;
        }
        else
        {
            return NULL;
        }
    }
}
