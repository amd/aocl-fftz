// Copyright Advanced Micro Devices, Inc.
// SPDX-License-Identifier: BSD-3-Clause

/** @file rfft3avx512.c
 *
 *  @brief Radix-3 r2hc_fused Real-FFT kernel with with AVX-512 operations using
 *  x86 SIMD intrinsics.
 *
 *  This file contains the DIT radix-3 real-to-halfcomplex fused of two
 *  different implementations (Standard DFT and Shifted DFT that differs in DFT
 *  weight matrix) using x86 SIMD intrinsics for single-precision and
 *  double-precision inputs
 *
 *  @author Jeevanantham N
 */

#include "core/kernels/kernel.h"
#include "core/kernels/simd_includes/r2hc_simd_avx512.h"

static const ops_cycles_t ops_cnt[NUM_PRECISIONS][NUM_FFT_DIRS] =
                                                {{{0, 4, 8, 160, 100, 36},
                                                  {0, 4, 8, 160, 116, 36}},
                                                 {{0, 4, 8,  80,   4, 36},
                                                  {0, 4, 8,  80,   4, 36}}};

ops_cycles_t get_ops_cnt_r2hcf_rfft3avx512(UINT8 precision, UINT8 direction)
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

static VOID r2hcf_rfft3avx512_fp32_fwd(VOID *in_real, VOID *in_imag,
                                       VOID *out_real, VOID *out_imag, INTP n,
                                       aoclfftz_strides_t *strides, VOID *twd, UINT8 flag)
{
    AOCLFFTZ_LOG(DEBUG, global_logger_mode, "Enter");
    const FLOAT CRTM_3_1 =  0.500000000000000000000000000000000000000000000f;
    const FLOAT CRTM_3_2 = -0.866025403784438646763723170752936183471402627f;

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

    __m512 v_CRTM_3_1 = _mm512_set1_ps(CRTM_3_1);
    __m512 v_CRTM_3_2 = _mm512_set1_ps(CRTM_3_2);

    for (cnt = 0; cnt < N; cnt++)
    {
        // Standard DFT
        __m512 av_in0, av_in1, av_in2;
        __m512 av_s0, av_s1;
        __m512 av_t0, av_t1;
        __m512 v_out0, v_out1, v_out2, v_out3, v_out4, v_out5;

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

        av_s0 = _mm512_add_ps(av_in1, av_in2);
        av_s1 = _mm512_sub_ps(av_in1, av_in2);
        av_t0 = _mm512_mul_ps(v_CRTM_3_1, av_s0);
        av_t1 = _mm512_mul_ps(v_CRTM_3_2, av_s1);

        // Output point 1: x(0)
        v_out0 = _mm512_add_ps(av_in0, av_s0);
        STR_512_S(curr_out, v_out_stride, v_out0);
        // Output point 4: x(3)
        curr_out = out + out_strides[3];
        v_out3 = _mm512_sub_ps(av_in0, av_t0);
        // Output point 5: x(4)
        v_out4 = av_t1;
        STRI_2x512_S(curr_out, v_out_stride, v_out3, v_out4);

        // Shifted DFT
        __m512 bv_in0, bv_in1, bv_in2;
        __m512 bv_s0, bv_s1;
        __m512 bv_t0, bv_t1;

        // Input point 2: x(1)
        curr_in = in + in_strides[1];
        LDR_512_S(curr_in, v_in_stride, bv_in0);
        // Input point 4: x(3)
        curr_in = in + in_strides[3];
        LDR_512_S(curr_in, v_in_stride, bv_in1);
        // Input point 6: x(5)
        curr_in = in + in_strides[5];
        LDR_512_S(curr_in, v_in_stride, bv_in2);

        bv_s0 = _mm512_sub_ps(bv_in1, bv_in2);
        bv_s1 = _mm512_add_ps(bv_in1, bv_in2);
        bv_t0 = _mm512_mul_ps(v_CRTM_3_1, bv_s0);
        bv_t1 = _mm512_mul_ps(v_CRTM_3_2, bv_s1);

        // Output point 2: x(1)
        curr_out = out + out_strides[1];
        v_out1 = _mm512_add_ps(bv_in0, bv_t0);
        v_out2 = bv_t1;
        STRI_2x512_S(curr_out, v_out_stride, v_out1, v_out2);
        // Output point 6: x(5)
        curr_out = out + out_strides[5];
        v_out5 = _mm512_sub_ps(bv_in0, bv_s0);
        STR_512_S(curr_out, v_out_stride, v_out5);

        in = in + (v_in_stride << 4);
        out = out + (v_out_stride << 4);
    }
    // tail cases
    if (remaining_sets & NUM_SETS_REAL_256_S)
    {
        // Standard DFT
        __m256 av_in0, av_in1, av_in2;
        __m256 av_s0, av_s1;
        __m256 av_t0, av_t1;
        __m256 v_out0, v_out1, v_out2, v_out3, v_out4, v_out5;

        curr_in = in;
        curr_out = out;

        __m256 v256_CRTM_3_1 = _mm512_castps512_ps256(v_CRTM_3_1);
        __m256 v256_CRTM_3_2 = _mm512_castps512_ps256(v_CRTM_3_2);

        // Input point 1: x(0)
        LDR_256_S(curr_in, v_in_stride, av_in0);
        // Input point 3: x(2)
        curr_in = in + in_strides[2];
        LDR_256_S(curr_in, v_in_stride, av_in1);
        // Input point 5: x(4)
        curr_in = in + in_strides[4];
        LDR_256_S(curr_in, v_in_stride, av_in2);

        av_s0 = _mm256_add_ps(av_in1, av_in2);
        av_s1 = _mm256_sub_ps(av_in1, av_in2);
        av_t0 = _mm256_mul_ps(v256_CRTM_3_1, av_s0);
        av_t1 = _mm256_mul_ps(v256_CRTM_3_2, av_s1);

        // Output point 1: x(0)
        v_out0 = _mm256_add_ps(av_in0, av_s0);
        STR_256_S(curr_out, v_out_stride, v_out0);
        // Output point 4: x(3)
        curr_out = out + out_strides[3];
        v_out3 = _mm256_sub_ps(av_in0, av_t0);
        // Output point 5: x(4)
        v_out4 = av_t1;
        STRI_2x256_S(curr_out, v_out_stride, v_out3, v_out4);

        // Shifted DFT
        __m256 bv_in0, bv_in1, bv_in2;
        __m256 bv_s0, bv_s1;
        __m256 bv_t0, bv_t1;

        // Input point 2: x(1)
        curr_in = in + in_strides[1];
        LDR_256_S(curr_in, v_in_stride, bv_in0);
        // Input point 4: x(3)
        curr_in = in + in_strides[3];
        LDR_256_S(curr_in, v_in_stride, bv_in1);
        // Input point 6: x(5)
        curr_in = in + in_strides[5];
        LDR_256_S(curr_in, v_in_stride, bv_in2);

        bv_s0 = _mm256_sub_ps(bv_in1, bv_in2);
        bv_s1 = _mm256_add_ps(bv_in1, bv_in2);
        bv_t0 = _mm256_mul_ps(v256_CRTM_3_1, bv_s0);
        bv_t1 = _mm256_mul_ps(v256_CRTM_3_2, bv_s1);

        // Output point 2: x(1)
        curr_out = out + out_strides[1];
        v_out1 = _mm256_add_ps(bv_in0, bv_t0);
        // Output point 3: x(2)
        v_out2 = bv_t1;
        STRI_2x256_S(curr_out, v_out_stride, v_out1, v_out2);
        // Output point 6: x(5)
        curr_out = out + out_strides[5];
        v_out5 = _mm256_sub_ps(bv_in0, bv_s0);
        STR_256_S(curr_out, v_out_stride, v_out5);

        in = in + (v_in_stride << 3);
        out = out + (v_out_stride << 3);
    }
    // tail cases
    if (remaining_sets & NUM_SETS_REAL_128_S)
    {
        // Standard DFT
        __m128 av_in0, av_in1, av_in2;
        __m128 av_s0, av_s1;
        __m128 av_t0, av_t1;
        __m128 v_out0, v_out1, v_out2, v_out3, v_out4, v_out5;

        curr_in = in;
        curr_out = out;

        __m128 v128_CRTM_3_1 = _mm512_castps512_ps128(v_CRTM_3_1);
        __m128 v128_CRTM_3_2 = _mm512_castps512_ps128(v_CRTM_3_2);

        // Input point 1: x(0)
        LDR_128_S(curr_in, v_in_stride, av_in0);
        // Input point 3: x(2)
        curr_in = in + in_strides[2];
        LDR_128_S(curr_in, v_in_stride, av_in1);
        // Input point 5: x(4)
        curr_in = in + in_strides[4];
        LDR_128_S(curr_in, v_in_stride, av_in2);

        av_s0 = _mm_add_ps(av_in1, av_in2);
        av_s1 = _mm_sub_ps(av_in1, av_in2);
        av_t0 = _mm_mul_ps(v128_CRTM_3_1, av_s0);
        av_t1 = _mm_mul_ps(v128_CRTM_3_2, av_s1);

        // Output point 1: x(0)
        v_out0 = _mm_add_ps(av_in0, av_s0);
        STR_128_S(curr_out, v_out_stride, v_out0);
        // Output point 4: x(3)
        curr_out = out + out_strides[3];
        v_out3 = _mm_sub_ps(av_in0, av_t0);
        // Output point 5: x(4)
        v_out4 = av_t1;
        STRI_2x128_S(curr_out, v_out_stride, v_out3, v_out4);

        // Shifted DFT
        __m128 bv_in0, bv_in1, bv_in2;
        __m128 bv_s0, bv_s1;
        __m128 bv_t0, bv_t1;

        // Input point 2: x(1)
        curr_in = in + in_strides[1];
        LDR_128_S(curr_in, v_in_stride, bv_in0);
        // Input point 4: x(3)
        curr_in = in + in_strides[3];
        LDR_128_S(curr_in, v_in_stride, bv_in1);
        // Input point 6: x(5)
        curr_in = in + in_strides[5];
        LDR_128_S(curr_in, v_in_stride, bv_in2);

        bv_s0 = _mm_sub_ps(bv_in1, bv_in2);
        bv_s1 = _mm_add_ps(bv_in1, bv_in2);
        bv_t0 = _mm_mul_ps(v128_CRTM_3_1, bv_s0);
        bv_t1 = _mm_mul_ps(v128_CRTM_3_2, bv_s1);

        // Output point 2: x(1)
        curr_out = out + out_strides[1];
        v_out1 = _mm_add_ps(bv_in0, bv_t0);
        // Output point 3: x(2)
        v_out2 = bv_t1;
        STRI_2x128_S(curr_out, v_out_stride, v_out1, v_out2);
        // Output point 6: x(5)
        curr_out = out + out_strides[5];
        v_out5 = _mm_sub_ps(bv_in0, bv_s0);
        STR_128_S(curr_out, v_out_stride, v_out5);

        in = in + (v_in_stride << 2);
        out = out + (v_out_stride << 2);
    }
    // tail cases
    if (remaining_sets & 2)
    {
        // Standard DFT
        __m128 av_in0, av_in1, av_in2;
        __m128 av_s0, av_s1;
        __m128 av_t0, av_t1;
        __m128 v_out0, v_out1, v_out2, v_out3, v_out4, v_out5;

        curr_in = in;
        curr_out = out;

        __m128 v128_CRTM_3_1 = _mm512_castps512_ps128(v_CRTM_3_1);
        __m128 v128_CRTM_3_2 = _mm512_castps512_ps128(v_CRTM_3_2);

        // Input point 1: x(0)
        LDHR_128_S(curr_in, v_in_stride, av_in0);
        // Input point 3: x(2)
        curr_in = in + in_strides[2];
        LDHR_128_S(curr_in, v_in_stride, av_in1);
        // Input point 5: x(4)
        curr_in = in + in_strides[4];
        LDHR_128_S(curr_in, v_in_stride, av_in2);

        av_s0 = _mm_add_ps(av_in1, av_in2);
        av_s1 = _mm_sub_ps(av_in1, av_in2);
        av_t0 = _mm_mul_ps(v128_CRTM_3_1, av_s0);
        av_t1 = _mm_mul_ps(v128_CRTM_3_2, av_s1);

        // Output point 1: x(0)
        v_out0 = _mm_add_ps(av_in0, av_s0);
        STHR_128_S(curr_out, v_out_stride, v_out0);
        // Output point 4: x(3)
        curr_out = out + out_strides[3];
        v_out3 = _mm_sub_ps(av_in0, av_t0);
        // Output point 5: x(4)
        v_out4 = av_t1;
        STHRI_2x128_S(curr_out, v_out_stride, v_out3, v_out4);

        // Shifted DFT
        __m128 bv_in0, bv_in1, bv_in2;
        __m128 bv_s0, bv_s1;
        __m128 bv_t0, bv_t1;

        // Input point 2: x(1)
        curr_in = in + in_strides[1];
        LDHR_128_S(curr_in, v_in_stride, bv_in0);
        // Input point 4: x(3)
        curr_in = in + in_strides[3];
        LDHR_128_S(curr_in, v_in_stride, bv_in1);
        // Input point 6: x(5)
        curr_in = in + in_strides[5];
        LDHR_128_S(curr_in, v_in_stride, bv_in2);

        bv_s0 = _mm_sub_ps(bv_in1, bv_in2);
        bv_s1 = _mm_add_ps(bv_in1, bv_in2);
        bv_t0 = _mm_mul_ps(v128_CRTM_3_1, bv_s0);
        bv_t1 = _mm_mul_ps(v128_CRTM_3_2, bv_s1);

        // Output point 2: x(1)
        curr_out = out + out_strides[1];
        v_out1 = _mm_add_ps(bv_in0, bv_t0);
        // Output point 3: x(2)
        v_out2 = bv_t1;
        STHRI_2x128_S(curr_out, v_out_stride, v_out1, v_out2);
        // Output point 6: x(5)
        curr_out = out + out_strides[5];
        v_out5 = _mm_sub_ps(bv_in0, bv_s0);
        STHR_128_S(curr_out, v_out_stride, v_out5);

        in = in + (v_in_stride << 1);
        out = out + (v_out_stride << 1);
    }
    // tail cases
    if (remaining_sets & 1)
    {
        // Standard DFT
        FLOAT a_in0, a_in1, a_in2;
        FLOAT a_s0, a_s1, a_t0, a_t1;

        a_in0 = *in;               // Input point 1: x(0)
        a_in1 = in[in_strides[2]]; // Input point 3: x(2)
        a_in2 = in[in_strides[4]]; // Input point 5: x(4)

        a_s0 = a_in1 + a_in2;
        a_s1 = a_in1 - a_in2;
        a_t0 = CRTM_3_1 * a_s0;
        a_t1 = CRTM_3_2 * a_s1;

        *out = a_in0 + a_s0;                // Output point 1: X(0)
        out[out_strides[3]] = a_in0 - a_t0; // Output point 4: X(3)
        out[out_strides[4]] = a_t1;         // Output point 5: X(4)

        // Shifted DFT
        FLOAT b_in0, b_in1, b_in2;
        FLOAT b_s0, b_s1, b_t0, b_t1;

        b_in0 = in[in_strides[1]]; // Input point 2: x(1)
        b_in1 = in[in_strides[3]]; // Input point 4: x(3)
        b_in2 = in[in_strides[5]]; // Input point 6: x(5)

        b_s0 = b_in1 - b_in2;
        b_s1 = b_in1 + b_in2;
        b_t0 = CRTM_3_1 * b_s0;
        b_t1 = CRTM_3_2 * b_s1;

        out[out_strides[1]] = b_in0 + b_t0; // Output point 2: X(1)
        out[out_strides[2]] = b_t1;         // Output point 3: X(2)
        out[out_strides[5]] = b_in0 - b_s0; // Output point 6: X(5)

        in = in + v_in_stride;
        out = out + v_out_stride;
    }
    AOCLFFTZ_LOG(DEBUG, global_logger_mode, "Exit");
}

static VOID r2hcf_rfft3avx512_fp32_bwd(VOID *in_real, VOID *in_imag,
                                       VOID *out_real, VOID *out_imag, INTP n,
                                       aoclfftz_strides_t *strides, VOID *twd, UINT8 flag)
{
    AOCLFFTZ_LOG(DEBUG, global_logger_mode, "Enter");
    const FLOAT CRTM_3_1 = -1.732050807568877293527446341505872366942805254f;
    const FLOAT CRTM_3_2 =  2.000000000000000000000000000000000000000000000f;

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

    __m512 v_CRTM_3_1 = _mm512_set1_ps(CRTM_3_1);
    __m512 v_CRTM_3_2 = _mm512_set1_ps(CRTM_3_2);

    for (cnt = 0; cnt < N; cnt++)
    {
        // Standard DFT
        __m512 av_in0, av_in1, av_in2;
        __m512 av_s0;
        __m512 av_t0, av_t1;
        __m512 v_out0, v_out1, v_out2, v_out3, v_out4, v_out5;

        curr_in = in;
        curr_out = out;

        // Input point 1: x(0)
        LDR_512_S(curr_in, v_in_stride, av_in0);
        // Input point 4: x(3) & Input point 5: x(4)
        curr_in = in + in_strides[3];
        LDRI_2x512_S(curr_in, v_in_stride, av_in1, av_in2);

        av_s0 = _mm512_sub_ps(av_in0, av_in1);
        av_t0 = _mm512_mul_ps(v_CRTM_3_1, av_in2);
        av_t1 = _mm512_mul_ps(v_CRTM_3_2, av_in1);

        // Output point 1: x(0)
        v_out0 = _mm512_add_ps(av_in0, av_t1);
        STR_512_S(curr_out, v_out_stride, v_out0);
        // Output point 3: x(2)
        curr_out = out + out_strides[2];
        v_out2 = _mm512_add_ps(av_s0, av_t0);
        STR_512_S(curr_out, v_out_stride, v_out2);
        // Output point 5: x(4)
        curr_out = out + out_strides[4];
        v_out4 = _mm512_sub_ps(av_s0, av_t0);
        STR_512_S(curr_out, v_out_stride, v_out4);

        // Shifted DFT
        __m512 bv_in0, bv_in1, bv_in2;
        __m512 bv_s0;
        __m512 bv_t0, bv_t1;

        // Input point 2: x(1) & Input point 3: x(2)
        curr_in = in + in_strides[1];
        LDRI_2x512_S(curr_in, v_in_stride, bv_in0, bv_in1);
        // Input point 6: x(5)
        curr_in = in + in_strides[5];
        LDR_512_S(curr_in, v_in_stride, bv_in2);

        bv_s0 = _mm512_sub_ps(bv_in0, bv_in2);
        bv_t0 = _mm512_mul_ps(v_CRTM_3_1, bv_in1);
        bv_t1 = _mm512_mul_ps(v_CRTM_3_2, bv_in0);

        // Output point 2: x(1)
        curr_out = out + out_strides[1];
        v_out1 = _mm512_add_ps(bv_in2, bv_t1);
        STR_512_S(curr_out, v_out_stride, v_out1);
        // Output point 4: x(3)
        curr_out = out + out_strides[3];
        v_out3 = _mm512_add_ps(bv_s0, bv_t0);
        STR_512_S(curr_out, v_out_stride, v_out3);
        // Output point 6: x(5)
        curr_out = out + out_strides[5];
        v_out5 = _mm512_sub_ps(bv_t0, bv_s0);
        STR_512_S(curr_out, v_out_stride, v_out5);

        in = in + (v_in_stride << 4);
        out = out + (v_out_stride << 4);
    }
    // tail cases
    if (remaining_sets & NUM_SETS_REAL_256_S)
    {
        // Standard DFT
        __m256 av_in0, av_in1, av_in2;
        __m256 av_s0;
        __m256 av_t0, av_t1;
        __m256 v_out0, v_out1, v_out2, v_out3, v_out4, v_out5;

        curr_in = in;
        curr_out = out;

        __m256 v256_CRTM_3_1 = _mm512_castps512_ps256(v_CRTM_3_1);
        __m256 v256_CRTM_3_2 = _mm512_castps512_ps256(v_CRTM_3_2);

        // Input point 1: x(0)
        LDR_256_S(curr_in, v_in_stride, av_in0);
        // Input point 4: x(3) & Input point 5: x(4)
        curr_in = in + in_strides[3];
        LDRI_2x256_S(curr_in, v_in_stride, av_in1, av_in2);

        av_s0 = _mm256_sub_ps(av_in0, av_in1);
        av_t0 = _mm256_mul_ps(v256_CRTM_3_1, av_in2);
        av_t1 = _mm256_mul_ps(v256_CRTM_3_2, av_in1);

        // Output point 1: x(0)
        v_out0 = _mm256_add_ps(av_in0, av_t1);
        STR_256_S(curr_out, v_out_stride, v_out0);
        // Output point 3: x(2)
        curr_out = out + out_strides[2];
        v_out2 = _mm256_add_ps(av_s0, av_t0);
        STR_256_S(curr_out, v_out_stride, v_out2);
        // Output point 5: x(4)
        curr_out = out + out_strides[4];
        v_out4 = _mm256_sub_ps(av_s0, av_t0);
        STR_256_S(curr_out, v_out_stride, v_out4);

        // Shifted DFT
        __m256 bv_in0, bv_in1, bv_in2;
        __m256 bv_s0;
        __m256 bv_t0, bv_t1;

        // Input point 2: x(1) & Input point 3: x(2)
        curr_in = in + in_strides[1];
        LDRI_2x256_S(curr_in, v_in_stride, bv_in0, bv_in1);
        // Input point 6: x(5)
        curr_in = in + in_strides[5];
        LDR_256_S(curr_in, v_in_stride, bv_in2);

        bv_s0 = _mm256_sub_ps(bv_in0, bv_in2);
        bv_t0 = _mm256_mul_ps(v256_CRTM_3_1, bv_in1);
        bv_t1 = _mm256_mul_ps(v256_CRTM_3_2, bv_in0);

        // Output point 2: x(1)
        curr_out = out + out_strides[1];
        v_out1 = _mm256_add_ps(bv_in2, bv_t1);
        STR_256_S(curr_out, v_out_stride, v_out1);
        // Output point 4: x(3)
        curr_out = out + out_strides[3];
        v_out3 = _mm256_add_ps(bv_s0, bv_t0);
        STR_256_S(curr_out, v_out_stride, v_out3);
        // Output point 6: x(5)
        curr_out = out + out_strides[5];
        v_out5 = _mm256_sub_ps(bv_t0, bv_s0);
        STR_256_S(curr_out, v_out_stride, v_out5);

        in = in + (v_in_stride << 3);
        out = out + (v_out_stride << 3);
    }
    // tail cases
    if (remaining_sets & NUM_SETS_REAL_128_S)
    {
        // Standard DFT
        __m128 av_in0, av_in1, av_in2;
        __m128 av_s0;
        __m128 av_t0, av_t1;
        __m128 v_out0, v_out1, v_out2, v_out3, v_out4, v_out5;

        curr_in = in;
        curr_out = out;

        __m128 v128_CRTM_3_1 = _mm512_castps512_ps128(v_CRTM_3_1);
        __m128 v128_CRTM_3_2 = _mm512_castps512_ps128(v_CRTM_3_2);

        // Input point 1: x(0)
        LDR_128_S(curr_in, v_in_stride, av_in0);
        // Input point 4: x(3) & Input point 5: x(4)
        curr_in = in + in_strides[3];
        LDRI_2x128_S(curr_in, v_in_stride, av_in1, av_in2);

        av_s0 = _mm_sub_ps(av_in0, av_in1);
        av_t0 = _mm_mul_ps(v128_CRTM_3_1, av_in2);
        av_t1 = _mm_mul_ps(v128_CRTM_3_2, av_in1);

        // Output point 1: x(0)
        v_out0 = _mm_add_ps(av_in0, av_t1);
        STR_128_S(curr_out, v_out_stride, v_out0);
        // Output point 3: x(2)
        curr_out = out + out_strides[2];
        v_out2 = _mm_add_ps(av_s0, av_t0);
        STR_128_S(curr_out, v_out_stride, v_out2);
        // Output point 5: x(4)
        curr_out = out + out_strides[4];
        v_out4 = _mm_sub_ps(av_s0, av_t0);
        STR_128_S(curr_out, v_out_stride, v_out4);

        // Shifted DFT
        __m128 bv_in0, bv_in1, bv_in2;
        __m128 bv_s0;
        __m128 bv_t0, bv_t1;

        // Input point 2: x(1) & Input point 3: x(2)
        curr_in = in + in_strides[1];
        LDRI_2x128_S(curr_in, v_in_stride, bv_in0, bv_in1);
        // Input point 6: x(5)
        curr_in = in + in_strides[5];
        LDR_128_S(curr_in, v_in_stride, bv_in2);

        bv_s0 = _mm_sub_ps(bv_in0, bv_in2);
        bv_t0 = _mm_mul_ps(v128_CRTM_3_1, bv_in1);
        bv_t1 = _mm_mul_ps(v128_CRTM_3_2, bv_in0);

        // Output point 2: x(1)
        curr_out = out + out_strides[1];
        v_out1 = _mm_add_ps(bv_in2, bv_t1);
        STR_128_S(curr_out, v_out_stride, v_out1);
        // Output point 4: x(3)
        curr_out = out + out_strides[3];
        v_out3 = _mm_add_ps(bv_s0, bv_t0);
        STR_128_S(curr_out, v_out_stride, v_out3);
        // Output point 6: x(5)
        curr_out = out + out_strides[5];
        v_out5 = _mm_sub_ps(bv_t0, bv_s0);
        STR_128_S(curr_out, v_out_stride, v_out5);

        in = in + (v_in_stride << 2);
        out = out + (v_out_stride << 2);
    }
    // tail cases
    if (remaining_sets & 2)
    {
        // Standard DFT
        __m128 av_in0, av_in1, av_in2;
        __m128 av_s0;
        __m128 av_t0, av_t1;
        __m128 v_out0, v_out1, v_out2, v_out3, v_out4, v_out5;

        curr_in = in;
        curr_out = out;

        __m128 v128_CRTM_3_1 = _mm512_castps512_ps128(v_CRTM_3_1);
        __m128 v128_CRTM_3_2 = _mm512_castps512_ps128(v_CRTM_3_2);

        // Input point 1: x(0)
        LDHR_128_S(curr_in, v_in_stride, av_in0);
        // Input point 4: x(3) & Input point 5: x(4)
        curr_in = in + in_strides[3];
        LDHRI_2x128_S(curr_in, v_in_stride, av_in1, av_in2);

        av_s0 = _mm_sub_ps(av_in0, av_in1);
        av_t0 = _mm_mul_ps(v128_CRTM_3_1, av_in2);
        av_t1 = _mm_mul_ps(v128_CRTM_3_2, av_in1);

        // Output point 1: x(0)
        v_out0 = _mm_add_ps(av_in0, av_t1);
        STHR_128_S(curr_out, v_out_stride, v_out0);
        // Output point 3: x(2)
        curr_out = out + out_strides[2];
        v_out2 = _mm_add_ps(av_s0, av_t0);
        STHR_128_S(curr_out, v_out_stride, v_out2);
        // Output point 5: x(4)
        curr_out = out + out_strides[4];
        v_out4 = _mm_sub_ps(av_s0, av_t0);
        STHR_128_S(curr_out, v_out_stride, v_out4);

        // Shifted DFT
        __m128 bv_in0, bv_in1, bv_in2;
        __m128 bv_s0;
        __m128 bv_t0, bv_t1;

        // Input point 2: x(1) & Input point 3: x(2)
        curr_in = in + in_strides[1];
        LDHRI_2x128_S(curr_in, v_in_stride, bv_in0, bv_in1);
        // Input point 6: x(5)
        curr_in = in + in_strides[5];
        LDHR_128_S(curr_in, v_in_stride, bv_in2);

        bv_s0 = _mm_sub_ps(bv_in0, bv_in2);
        bv_t0 = _mm_mul_ps(v128_CRTM_3_1, bv_in1);
        bv_t1 = _mm_mul_ps(v128_CRTM_3_2, bv_in0);

        // Output point 2: x(1)
        curr_out = out + out_strides[1];
        v_out1 = _mm_add_ps(bv_in2, bv_t1);
        STHR_128_S(curr_out, v_out_stride, v_out1);
        // Output point 4: x(3)
        curr_out = out + out_strides[3];
        v_out3 = _mm_add_ps(bv_s0, bv_t0);
        STHR_128_S(curr_out, v_out_stride, v_out3);
        // Output point 6: x(5)
        curr_out = out + out_strides[5];
        v_out5 = _mm_sub_ps(bv_t0, bv_s0);
        STHR_128_S(curr_out, v_out_stride, v_out5);

        in = in + (v_in_stride << 1);
        out = out + (v_out_stride << 1);
    }
    // tail cases
    if (remaining_sets & 1)
    {
        // Standard DFT
        FLOAT a_in0, a_in1, a_in2;
        FLOAT a_s0, a_t0, a_t1;

        a_in0 = *in;               // Input point 1: x(0)
        a_in1 = in[in_strides[3]]; // Input point 4: x(3)
        a_in2 = in[in_strides[4]]; // Input point 5: x(4)

        a_s0 = a_in0 - a_in1;
        a_t0 = CRTM_3_1 * a_in2;
        a_t1 = CRTM_3_2 * a_in1;

        *out = a_in0 + a_t1;               // Output point 1: X(0)
        out[out_strides[2]] = a_s0 + a_t0; // Output point 3: X(2)
        out[out_strides[4]] = a_s0 - a_t0; // Output point 5: X(4)

        // Shifted DFT
        FLOAT b_in0, b_in1, b_in2;
        FLOAT b_s0, b_t0, b_t1;

        b_in0 = in[in_strides[1]]; // Input point 2: x(1)
        b_in1 = in[in_strides[2]]; // Input point 3: x(2)
        b_in2 = in[in_strides[5]]; // Input point 6: x(5)

        b_s0 = b_in0 - b_in2;
        b_t0 = CRTM_3_1 * b_in1;
        b_t1 = CRTM_3_2 * b_in0;

        out[out_strides[1]] = b_in2 + b_t1; // Output point 2: X(1)
        out[out_strides[3]] = b_s0 + b_t0;  // Output point 4: X(3)
        out[out_strides[5]] = b_t0 - b_s0;  // Output point 6: X(5)
    }
    AOCLFFTZ_LOG(DEBUG, global_logger_mode, "Exit");
}

static VOID r2hcf_rfft3avx512_fp64_fwd(VOID *in_real, VOID *in_imag,
                                       VOID *out_real, VOID *out_imag, INTP n,
                                       aoclfftz_strides_t *strides, VOID *twd, UINT8 flag)
{
    AOCLFFTZ_LOG(DEBUG, global_logger_mode, "Enter");
    const DOUBLE CRTM_3_1 =  0.500000000000000000000000000000000000000000000;
    const DOUBLE CRTM_3_2 = -0.866025403784438646763723170752936183471402627;

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

    __m512d v_CRTM_3_1 = _mm512_set1_pd(CRTM_3_1);
    __m512d v_CRTM_3_2 = _mm512_set1_pd(CRTM_3_2);

    for (cnt = 0; cnt < N; cnt++)
    {
        // Standard DFT
        __m512d av_in0, av_in1, av_in2;
        __m512d av_s0, av_s1;
        __m512d av_t0, av_t1;
        __m512d v_out0, v_out1, v_out2, v_out3, v_out4, v_out5;

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

        av_s0 = _mm512_add_pd(av_in1, av_in2);
        av_s1 = _mm512_sub_pd(av_in1, av_in2);
        av_t0 = _mm512_mul_pd(v_CRTM_3_1, av_s0);
        av_t1 = _mm512_mul_pd(v_CRTM_3_2, av_s1);

        // Output point 1: x(0)
        v_out0 = _mm512_add_pd(av_in0, av_s0);
        STR_512_D(curr_out, v_out_stride, v_out0);
        // Output point 4: x(3)
        curr_out = out + out_strides[3];
        v_out3 = _mm512_sub_pd(av_in0, av_t0);
        // Output point 5: x(4)
        v_out4 = av_t1;
        STRI_2x512_D(curr_out, v_out_stride, v_out3, v_out4);

        // Shifted DFT
        __m512d bv_in0, bv_in1, bv_in2;
        __m512d bv_s0, bv_s1;
        __m512d bv_t0, bv_t1;

        // Input point 2: x(1)
        curr_in = in + in_strides[1];
        LDR_512_D(curr_in, v_in_stride, bv_in0);
        // Input point 4: x(3)
        curr_in = in + in_strides[3];
        LDR_512_D(curr_in, v_in_stride, bv_in1);
        // Input point 6: x(5)
        curr_in = in + in_strides[5];
        LDR_512_D(curr_in, v_in_stride, bv_in2);

        bv_s0 = _mm512_sub_pd(bv_in1, bv_in2);
        bv_s1 = _mm512_add_pd(bv_in1, bv_in2);
        bv_t0 = _mm512_mul_pd(v_CRTM_3_1, bv_s0);
        bv_t1 = _mm512_mul_pd(v_CRTM_3_2, bv_s1);

        // Output point 2: x(1)
        curr_out = out + out_strides[1];
        v_out1 = _mm512_add_pd(bv_in0, bv_t0);
        v_out2 = bv_t1;
        STRI_2x512_D(curr_out, v_out_stride, v_out1, v_out2);
        // Output point 6: x(5)
        curr_out = out + out_strides[5];
        v_out5 = _mm512_sub_pd(bv_in0, bv_s0);
        STR_512_D(curr_out, v_out_stride, v_out5);

        in = in + (v_in_stride << 3);
        out = out + (v_out_stride << 3);
    }
    // tail cases
    if (remaining_sets & NUM_SETS_REAL_256_D)
    {
        // Standard DFT
        __m256d av_in0, av_in1, av_in2;
        __m256d av_s0, av_s1;
        __m256d av_t0, av_t1;
        __m256d v_out0, v_out1, v_out2, v_out3, v_out4, v_out5;

        curr_in = in;
        curr_out = out;

        __m256d v256_CRTM_3_1 = _mm512_castpd512_pd256(v_CRTM_3_1);
        __m256d v256_CRTM_3_2 = _mm512_castpd512_pd256(v_CRTM_3_2);

        // Input point 1: x(0)
        LDR_256_D(curr_in, v_in_stride, av_in0);
        // Input point 3: x(2)
        curr_in = in + in_strides[2];
        LDR_256_D(curr_in, v_in_stride, av_in1);
        // Input point 5: x(4)
        curr_in = in + in_strides[4];
        LDR_256_D(curr_in, v_in_stride, av_in2);

        av_s0 = _mm256_add_pd(av_in1, av_in2);
        av_s1 = _mm256_sub_pd(av_in1, av_in2);
        av_t0 = _mm256_mul_pd(v256_CRTM_3_1, av_s0);
        av_t1 = _mm256_mul_pd(v256_CRTM_3_2, av_s1);

        // Output point 1: x(0)
        v_out0 = _mm256_add_pd(av_in0, av_s0);
        STR_256_D(curr_out, v_out_stride, v_out0);
        // Output point 4: x(3)
        curr_out = out + out_strides[3];
        v_out3 = _mm256_sub_pd(av_in0, av_t0);
        // Output point 5: x(4)
        v_out4 = av_t1;
        STRI_2x256_D(curr_out, v_out_stride, v_out3, v_out4);

        // Shifted DFT
        __m256d bv_in0, bv_in1, bv_in2;
        __m256d bv_s0, bv_s1;
        __m256d bv_t0, bv_t1;

        // Input point 2: x(1)
        curr_in = in + in_strides[1];
        LDR_256_D(curr_in, v_in_stride, bv_in0);
        // Input point 4: x(3)
        curr_in = in + in_strides[3];
        LDR_256_D(curr_in, v_in_stride, bv_in1);
        // Input point 6: x(5)
        curr_in = in + in_strides[5];
        LDR_256_D(curr_in, v_in_stride, bv_in2);

        bv_s0 = _mm256_sub_pd(bv_in1, bv_in2);
        bv_s1 = _mm256_add_pd(bv_in1, bv_in2);
        bv_t0 = _mm256_mul_pd(v256_CRTM_3_1, bv_s0);
        bv_t1 = _mm256_mul_pd(v256_CRTM_3_2, bv_s1);

        // Output point 2: x(1)
        curr_out = out + out_strides[1];
        v_out1 = _mm256_add_pd(bv_in0, bv_t0);
        // Output point 3: x(2)
        v_out2 = bv_t1;
        STRI_2x256_D(curr_out, v_out_stride, v_out1, v_out2);
        // Output point 6: x(5)
        curr_out = out + out_strides[5];
        v_out5 = _mm256_sub_pd(bv_in0, bv_s0);
        STR_256_D(curr_out, v_out_stride, v_out5);

        in = in + (v_in_stride << 2);
        out = out + (v_out_stride << 2);
    }
    // tail cases
    if (remaining_sets & NUM_SETS_REAL_128_D)
    {
        // Standard DFT
        __m128d av_in0, av_in1, av_in2;
        __m128d av_s0, av_s1;
        __m128d av_t0, av_t1;
        __m128d v_out0, v_out1, v_out2, v_out3, v_out4, v_out5;

        curr_in = in;
        curr_out = out;

        __m128d v128_CRTM_3_1 = _mm512_castpd512_pd128(v_CRTM_3_1);
        __m128d v128_CRTM_3_2 = _mm512_castpd512_pd128(v_CRTM_3_2);

        // Input point 1: x(0)
        LDR_128_D(curr_in, v_in_stride, av_in0);
        // Input point 3: x(2)
        curr_in = in + in_strides[2];
        LDR_128_D(curr_in, v_in_stride, av_in1);
        // Input point 5: x(4)
        curr_in = in + in_strides[4];
        LDR_128_D(curr_in, v_in_stride, av_in2);

        av_s0 = _mm_add_pd(av_in1, av_in2);
        av_s1 = _mm_sub_pd(av_in1, av_in2);
        av_t0 = _mm_mul_pd(v128_CRTM_3_1, av_s0);
        av_t1 = _mm_mul_pd(v128_CRTM_3_2, av_s1);

        // Output point 1: x(0)
        v_out0 = _mm_add_pd(av_in0, av_s0);
        STR_128_D(curr_out, v_out_stride, v_out0);
        // Output point 4: x(3)
        curr_out = out + out_strides[3];
        v_out3 = _mm_sub_pd(av_in0, av_t0);
        // Output point 5: x(4)
        v_out4 = av_t1;
        STRI_2x128_D(curr_out, v_out_stride, v_out3, v_out4);

        // Shifted DFT
        __m128d bv_in0, bv_in1, bv_in2;
        __m128d bv_s0, bv_s1;
        __m128d bv_t0, bv_t1;

        // Input point 2: x(1)
        curr_in = in + in_strides[1];
        LDR_128_D(curr_in, v_in_stride, bv_in0);
        // Input point 4: x(3)
        curr_in = in + in_strides[3];
        LDR_128_D(curr_in, v_in_stride, bv_in1);
        // Input point 6: x(5)
        curr_in = in + in_strides[5];
        LDR_128_D(curr_in, v_in_stride, bv_in2);

        bv_s0 = _mm_sub_pd(bv_in1, bv_in2);
        bv_s1 = _mm_add_pd(bv_in1, bv_in2);
        bv_t0 = _mm_mul_pd(v128_CRTM_3_1, bv_s0);
        bv_t1 = _mm_mul_pd(v128_CRTM_3_2, bv_s1);

        // Output point 2: x(1)
        curr_out = out + out_strides[1];
        v_out1 = _mm_add_pd(bv_in0, bv_t0);
        // Output point 3: x(2)
        v_out2 = bv_t1;
        STRI_2x128_D(curr_out, v_out_stride, v_out1, v_out2);
        // Output point 6: x(5)
        curr_out = out + out_strides[5];
        v_out5 = _mm_sub_pd(bv_in0, bv_s0);
        STR_128_D(curr_out, v_out_stride, v_out5);

        in = in + (v_in_stride << 1);
        out = out + (v_out_stride << 1);
    }
    // tail cases
    if (remaining_sets & 1)
    {
        // Standard DFT
        DOUBLE a_in0, a_in1, a_in2;
        DOUBLE a_s0, a_s1, a_t0, a_t1;

        a_in0 = *in;               // Input point 1: x(0)
        a_in1 = in[in_strides[2]]; // Input point 3: x(2)
        a_in2 = in[in_strides[4]]; // Input point 5: x(4)

        a_s0 = a_in1 + a_in2;
        a_s1 = a_in1 - a_in2;
        a_t0 = CRTM_3_1 * a_s0;
        a_t1 = CRTM_3_2 * a_s1;

        *out = a_in0 + a_s0;                // Output point 1: X(0)
        out[out_strides[3]] = a_in0 - a_t0; // Output point 4: X(3)
        out[out_strides[4]] = a_t1;         // Output point 5: X(4)

        // Shifted DFT
        DOUBLE b_in0, b_in1, b_in2;
        DOUBLE b_s0, b_s1, b_t0, b_t1;

        b_in0 = in[in_strides[1]]; // Input point 2: x(1)
        b_in1 = in[in_strides[3]]; // Input point 4: x(3)
        b_in2 = in[in_strides[5]]; // Input point 6: x(5)

        b_s0 = b_in1 - b_in2;
        b_s1 = b_in1 + b_in2;
        b_t0 = CRTM_3_1 * b_s0;
        b_t1 = CRTM_3_2 * b_s1;

        out[out_strides[1]] = b_in0 + b_t0; // Output point 2: X(1)
        out[out_strides[2]] = b_t1;         // Output point 3: X(2)
        out[out_strides[5]] = b_in0 - b_s0; // Output point 6: X(5)

        in = in + v_in_stride;
        out = out + v_out_stride;
    }
    AOCLFFTZ_LOG(DEBUG, global_logger_mode, "Exit");
}

static VOID r2hcf_rfft3avx512_fp64_bwd(VOID *in_real, VOID *in_imag,
                                       VOID *out_real, VOID *out_imag, INTP n,
                                       aoclfftz_strides_t *strides, VOID *twd, UINT8 flag)
{
    AOCLFFTZ_LOG(DEBUG, global_logger_mode, "Enter");
    const DOUBLE CRTM_3_1 = -1.732050807568877293527446341505872366942805254;
    const DOUBLE CRTM_3_2 =  2.000000000000000000000000000000000000000000000;

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

    __m512d v_CRTM_3_1 = _mm512_set1_pd(CRTM_3_1);
    __m512d v_CRTM_3_2 = _mm512_set1_pd(CRTM_3_2);

    for (cnt = 0; cnt < N; cnt++)
    {
        // Standard DFT
        __m512d av_in0, av_in1, av_in2;
        __m512d av_s0;
        __m512d av_t0, av_t1;
        __m512d v_out0, v_out1, v_out2, v_out3, v_out4, v_out5;

        curr_in = in;
        curr_out = out;

        // Input point 1: x(0)
        LDR_512_D(curr_in, v_in_stride, av_in0);
        // Input point 4: x(3) & Input point 5: x(4)
        curr_in = in + in_strides[3];
        LDRI_2x512_D(curr_in, v_in_stride, av_in1, av_in2);

        av_s0 = _mm512_sub_pd(av_in0, av_in1);
        av_t0 = _mm512_mul_pd(v_CRTM_3_1, av_in2);
        av_t1 = _mm512_mul_pd(v_CRTM_3_2, av_in1);

        // Output point 1: x(0)
        v_out0 = _mm512_add_pd(av_in0, av_t1);
        STR_512_D(curr_out, v_out_stride, v_out0);
        // Output point 3: x(2)
        curr_out = out + out_strides[2];
        v_out2 = _mm512_add_pd(av_s0, av_t0);
        STR_512_D(curr_out, v_out_stride, v_out2);
        // Output point 5: x(4)
        curr_out = out + out_strides[4];
        v_out4 = _mm512_sub_pd(av_s0, av_t0);
        STR_512_D(curr_out, v_out_stride, v_out4);

        // Shifted DFT
        __m512d bv_in0, bv_in1, bv_in2;
        __m512d bv_s0;
        __m512d bv_t0, bv_t1;

        // Input point 2: x(1) & Input point 3: x(2)
        curr_in = in + in_strides[1];
        LDRI_2x512_D(curr_in, v_in_stride, bv_in0, bv_in1);
        // Input point 6: x(5)
        curr_in = in + in_strides[5];
        LDR_512_D(curr_in, v_in_stride, bv_in2);

        bv_s0 = _mm512_sub_pd(bv_in0, bv_in2);
        bv_t0 = _mm512_mul_pd(v_CRTM_3_1, bv_in1);
        bv_t1 = _mm512_mul_pd(v_CRTM_3_2, bv_in0);

        // Output point 2: x(1)
        curr_out = out + out_strides[1];
        v_out1 = _mm512_add_pd(bv_in2, bv_t1);
        STR_512_D(curr_out, v_out_stride, v_out1);
        // Output point 4: x(3)
        curr_out = out + out_strides[3];
        v_out3 = _mm512_add_pd(bv_s0, bv_t0);
        STR_512_D(curr_out, v_out_stride, v_out3);
        // Output point 6: x(5)
        curr_out = out + out_strides[5];
        v_out5 = _mm512_sub_pd(bv_t0, bv_s0);
        STR_512_D(curr_out, v_out_stride, v_out5);

        in = in + (v_in_stride << 3);
        out = out + (v_out_stride << 3);
    }
    // tail cases
    if (remaining_sets & NUM_SETS_REAL_256_D)
    {
        // Standard DFT
        __m256d av_in0, av_in1, av_in2;
        __m256d av_s0;
        __m256d av_t0, av_t1;
        __m256d v_out0, v_out1, v_out2, v_out3, v_out4, v_out5;

        curr_in = in;
        curr_out = out;

        __m256d v256_CRTM_3_1 = _mm512_castpd512_pd256(v_CRTM_3_1);
        __m256d v256_CRTM_3_2 = _mm512_castpd512_pd256(v_CRTM_3_2);

        // Input point 1: x(0)
        LDR_256_D(curr_in, v_in_stride, av_in0);
        // Input point 4: x(3) & Input point 5: x(4)
        curr_in = in + in_strides[3];
        LDRI_2x256_D(curr_in, v_in_stride, av_in1, av_in2);

        av_s0 = _mm256_sub_pd(av_in0, av_in1);
        av_t0 = _mm256_mul_pd(v256_CRTM_3_1, av_in2);
        av_t1 = _mm256_mul_pd(v256_CRTM_3_2, av_in1);

        // Output point 1: x(0)
        v_out0 = _mm256_add_pd(av_in0, av_t1);
        STR_256_D(curr_out, v_out_stride, v_out0);
        // Output point 3: x(2)
        curr_out = out + out_strides[2];
        v_out2 = _mm256_add_pd(av_s0, av_t0);
        STR_256_D(curr_out, v_out_stride, v_out2);
        // Output point 5: x(4)
        curr_out = out + out_strides[4];
        v_out4 = _mm256_sub_pd(av_s0, av_t0);
        STR_256_D(curr_out, v_out_stride, v_out4);

        // Shifted DFT
        __m256d bv_in0, bv_in1, bv_in2;
        __m256d bv_s0;
        __m256d bv_t0, bv_t1;

        // Input point 2: x(1) & Input point 3: x(2)
        curr_in = in + in_strides[1];
        LDRI_2x256_D(curr_in, v_in_stride, bv_in0, bv_in1);
        // Input point 6: x(5)
        curr_in = in + in_strides[5];
        LDR_256_D(curr_in, v_in_stride, bv_in2);

        bv_s0 = _mm256_sub_pd(bv_in0, bv_in2);
        bv_t0 = _mm256_mul_pd(v256_CRTM_3_1, bv_in1);
        bv_t1 = _mm256_mul_pd(v256_CRTM_3_2, bv_in0);

        // Output point 2: x(1)
        curr_out = out + out_strides[1];
        v_out1 = _mm256_add_pd(bv_in2, bv_t1);
        STR_256_D(curr_out, v_out_stride, v_out1);
        // Output point 4: x(3)
        curr_out = out + out_strides[3];
        v_out3 = _mm256_add_pd(bv_s0, bv_t0);
        STR_256_D(curr_out, v_out_stride, v_out3);
        // Output point 6: x(5)
        curr_out = out + out_strides[5];
        v_out5 = _mm256_sub_pd(bv_t0, bv_s0);
        STR_256_D(curr_out, v_out_stride, v_out5);

        in = in + (v_in_stride << 2);
        out = out + (v_out_stride << 2);
    }
    // tail cases
    if (remaining_sets & NUM_SETS_REAL_128_D)
    {
        // Standard DFT
        __m128d av_in0, av_in1, av_in2;
        __m128d av_s0;
        __m128d av_t0, av_t1;
        __m128d v_out0, v_out1, v_out2, v_out3, v_out4, v_out5;

        curr_in = in;
        curr_out = out;

        __m128d v128_CRTM_3_1 = _mm512_castpd512_pd128(v_CRTM_3_1);
        __m128d v128_CRTM_3_2 = _mm512_castpd512_pd128(v_CRTM_3_2);

        // Input point 1: x(0)
        LDR_128_D(curr_in, v_in_stride, av_in0);
        // Input point 4: x(3) & Input point 5: x(4)
        curr_in = in + in_strides[3];
        LDRI_2x128_D(curr_in, v_in_stride, av_in1, av_in2);

        av_s0 = _mm_sub_pd(av_in0, av_in1);
        av_t0 = _mm_mul_pd(v128_CRTM_3_1, av_in2);
        av_t1 = _mm_mul_pd(v128_CRTM_3_2, av_in1);

        // Output point 1: x(0)
        v_out0 = _mm_add_pd(av_in0, av_t1);
        STR_128_D(curr_out, v_out_stride, v_out0);
        // Output point 3: x(2)
        curr_out = out + out_strides[2];
        v_out2 = _mm_add_pd(av_s0, av_t0);
        STR_128_D(curr_out, v_out_stride, v_out2);
        // Output point 5: x(4)
        curr_out = out + out_strides[4];
        v_out4 = _mm_sub_pd(av_s0, av_t0);
        STR_128_D(curr_out, v_out_stride, v_out4);

        // Shifted DFT
        __m128d bv_in0, bv_in1, bv_in2;
        __m128d bv_s0;
        __m128d bv_t0, bv_t1;

        // Input point 2: x(1) & Input point 3: x(2)
        curr_in = in + in_strides[1];
        LDRI_2x128_D(curr_in, v_in_stride, bv_in0, bv_in1);
        // Input point 6: x(5)
        curr_in = in + in_strides[5];
        LDR_128_D(curr_in, v_in_stride, bv_in2);

        bv_s0 = _mm_sub_pd(bv_in0, bv_in2);
        bv_t0 = _mm_mul_pd(v128_CRTM_3_1, bv_in1);
        bv_t1 = _mm_mul_pd(v128_CRTM_3_2, bv_in0);

        // Output point 2: x(1)
        curr_out = out + out_strides[1];
        v_out1 = _mm_add_pd(bv_in2, bv_t1);
        STR_128_D(curr_out, v_out_stride, v_out1);
        // Output point 4: x(3)
        curr_out = out + out_strides[3];
        v_out3 = _mm_add_pd(bv_s0, bv_t0);
        STR_128_D(curr_out, v_out_stride, v_out3);
        // Output point 6: x(5)
        curr_out = out + out_strides[5];
        v_out5 = _mm_sub_pd(bv_t0, bv_s0);
        STR_128_D(curr_out, v_out_stride, v_out5);

        in = in + (v_in_stride << 1);
        out = out + (v_out_stride << 1);
    }
    // tail cases
    if (remaining_sets & 1)
    {
        // Standard DFT
        DOUBLE a_in0, a_in1, a_in2;
        DOUBLE a_s0, a_t0, a_t1;

        a_in0 = *in;               // Input point 1: x(0)
        a_in1 = in[in_strides[3]]; // Input point 4: x(3)
        a_in2 = in[in_strides[4]]; // Input point 5: x(4)

        a_s0 = a_in0 - a_in1;
        a_t0 = CRTM_3_1 * a_in2;
        a_t1 = CRTM_3_2 * a_in1;

        *out = a_in0 + a_t1;               // Output point 1: X(0)
        out[out_strides[2]] = a_s0 + a_t0; // Output point 3: X(2)
        out[out_strides[4]] = a_s0 - a_t0; // Output point 5: X(4)

        // Shifted DFT
        DOUBLE b_in0, b_in1, b_in2;
        DOUBLE b_s0, b_t0, b_t1;

        b_in0 = in[in_strides[1]]; // Input point 2: x(1)
        b_in1 = in[in_strides[2]]; // Input point 3: x(2)
        b_in2 = in[in_strides[5]]; // Input point 6: x(5)

        b_s0 = b_in0 - b_in2;
        b_t0 = CRTM_3_1 * b_in1;
        b_t1 = CRTM_3_2 * b_in0;

        out[out_strides[1]] = b_in2 + b_t1; // Output point 2: X(1)
        out[out_strides[3]] = b_s0 + b_t0;  // Output point 4: X(3)
        out[out_strides[5]] = b_t0 - b_s0;  // Output point 6: X(5)
    }
    AOCLFFTZ_LOG(DEBUG, global_logger_mode, "Exit");
}

kfft_ register_kernel_r2hcf_rfft3avx512(UINT8 precision, UINT8 direction)
{
    if (direction == FORWARD_FFT_DIR)
    {
        if (precision == DT_FLOAT)
        {
            return r2hcf_rfft3avx512_fp32_fwd;
        }
        else if (precision == DT_DOUBLE)
        {
            return r2hcf_rfft3avx512_fp64_fwd;
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
            return r2hcf_rfft3avx512_fp32_bwd;
        }
        else if (precision == DT_DOUBLE)
        {
            return r2hcf_rfft3avx512_fp64_bwd;
        }
        else
        {
            return NULL;
        }
    }
}
