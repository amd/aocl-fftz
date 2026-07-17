// Copyright Advanced Micro Devices, Inc.
// SPDX-License-Identifier: BSD-3-Clause

/** @file rfft6avx512.c
 *
 *  @brief Radix-6 r2hc Real-FFT kernel with AVX-512 operations using x86 SIMD
 *  intrinsics
 *
 *  This file contains the DIT radix-6 real-to-halfcomplex implementations using
 *  AVX512 SIMD operations for single-precision and double-precision inputs.
 *
 *  @author Jeevanantham N
 */

#include "core/kernels/kernel.h"
#include "core/kernels/simd_includes/r2hc_simd_avx512.h"

static const ops_cycles_t ops_cnt[NUM_PRECISIONS][NUM_FFT_DIRS] =
                                                  {{{0, 4, 14, 160, 100, 36},
                                                    {0, 2, 16, 160, 116, 36}},
                                                   {{0, 4, 14,  80,   4, 36},
                                                    {0, 2, 16,  80,   4, 36}}};

ops_cycles_t get_ops_cnt_r2hc_rfft6avx512(FFTZ_UINT8 precision,
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

static FFTZ_VOID r2hc_rfft6avx512_fp32_fwd(FFTZ_VOID *in_real,
                                           FFTZ_VOID *in_imag,
                                           FFTZ_VOID *out_real,
                                           FFTZ_VOID *out_imag, FFTZ_INTP n,
                                           aoclfftz_strides_t *strides,
                                           FFTZ_VOID *twd, FFTZ_UINT8 flag)
{
    AOCLFFTZ_LOG(DEBUG, global_logger_mode, "Enter");
    const FFTZ_FLOAT CRTM_6_1 =
        0.500000000000000000000000000000000000000000000f;
    const FFTZ_FLOAT CRTM_6_2 =
        0.866025403784438646763723170752936183471402627f;

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
    FFTZ_INTP N = n / NUM_SETS_REAL_512_S;
    FFTZ_INTP remaining_sets = n % NUM_SETS_REAL_512_S;

    __m512 v_CRTM_6_1 = _mm512_set1_ps(CRTM_6_1);
    __m512 v_CRTM_6_2 = _mm512_set1_ps(CRTM_6_2);

    for (cnt = 0; cnt < N; cnt++)
    {
        __m512 v_in0, v_in1, v_in2, v_in3, v_in4, v_in5;
        __m512 v_s0, v_s1, v_s2, v_s3, v_s4, v_s5, v_s6, v_s7;
        __m512 v_out0, v_out1, v_out2, v_out3, v_out4, v_out5;

        curr_in = in;
        curr_out = out;

        // Input point 1: x(0)
        LDR_512_S(curr_in, v_in_stride, v_in0, is_contiguous_in);
        // Input point 2: x(1)
        curr_in = in + in_strides[1];
        LDR_512_S(curr_in, v_in_stride, v_in1, is_contiguous_in);
        // Input point 3: x(2)
        curr_in = in + in_strides[2];
        LDR_512_S(curr_in, v_in_stride, v_in2, is_contiguous_in);
        // Input point 4: x(3)
        curr_in = in + in_strides[3];
        LDR_512_S(curr_in, v_in_stride, v_in3, is_contiguous_in);
        // Input point 5: x(4)
        curr_in = in + in_strides[4];
        LDR_512_S(curr_in, v_in_stride, v_in4, is_contiguous_in);
        // Input point 6: x(5)
        curr_in = in + in_strides[5];
        LDR_512_S(curr_in, v_in_stride, v_in5, is_contiguous_in);

        v_s0 = _mm512_add_ps(v_in0, v_in3);
        v_s1 = _mm512_sub_ps(v_in0, v_in3);
        v_s2 = _mm512_add_ps(v_in1, v_in2);
        v_s3 = _mm512_sub_ps(v_in2, v_in1);
        v_s4 = _mm512_add_ps(v_in4, v_in5);
        v_s5 = _mm512_sub_ps(v_in5, v_in4);

        v_s6 = _mm512_add_ps(v_s2, v_s4);
        v_s7 = _mm512_sub_ps(v_s5, v_s3);

        // Output point 1: X(0)
        v_out0 = _mm512_add_ps(v_s0, v_s6);

        // Output point 2: X(1)
        v_out1 = _mm512_add_ps(v_s1, _mm512_mul_ps(v_CRTM_6_1, v_s7));

        // Output point 3: X(2)
        v_out2 = _mm512_mul_ps(v_CRTM_6_2, _mm512_sub_ps(v_s4, v_s2));

        // Output point 4: X(3)
        v_out3 = _mm512_sub_ps(v_s0, _mm512_mul_ps(v_CRTM_6_1, v_s6));

        // Output point 5: X(4)
        v_out4 = _mm512_mul_ps(v_CRTM_6_2, _mm512_add_ps(v_s5, v_s3));

        // Output point 6: X(5)
        v_out5 = _mm512_sub_ps(v_s1, v_s7);

        STR_512_S(curr_out, v_out_stride, v_out0, is_contiguous_out);
        curr_out = out + out_strides[1];
        STRI_2x512_S(curr_out, v_out_stride, v_out1, v_out2);
        curr_out = out + out_strides[3];
        STRI_2x512_S(curr_out, v_out_stride, v_out3, v_out4);
        curr_out = out + out_strides[5];
        STR_512_S(curr_out, v_out_stride, v_out5, is_contiguous_out);

        in = in + (v_in_stride << 4);
        out = out + (v_out_stride << 4);
    }
    // tail cases
    if (remaining_sets & NUM_SETS_REAL_256_S)
    {
        __m256 v_in0, v_in1, v_in2, v_in3, v_in4, v_in5;
        __m256 v_s0, v_s1, v_s2, v_s3, v_s4, v_s5, v_s6, v_s7;
        __m256 v_out0, v_out1, v_out2, v_out3, v_out4, v_out5;

        curr_in = in;
        curr_out = out;

        __m256 v256_CRTM_6_1 = _mm512_castps512_ps256(v_CRTM_6_1);
        __m256 v256_CRTM_6_2 = _mm512_castps512_ps256(v_CRTM_6_2);

        // Input point 1: x(0)
        LDR_256_S(curr_in, v_in_stride, v_in0, is_contiguous_in);
        // Input point 2: x(1)
        curr_in = in + in_strides[1];
        LDR_256_S(curr_in, v_in_stride, v_in1, is_contiguous_in);
        // Input point 3: x(2)
        curr_in = in + in_strides[2];
        LDR_256_S(curr_in, v_in_stride, v_in2, is_contiguous_in);
        // Input point 4: x(3)
        curr_in = in + in_strides[3];
        LDR_256_S(curr_in, v_in_stride, v_in3, is_contiguous_in);
        // Input point 5: x(4)
        curr_in = in + in_strides[4];
        LDR_256_S(curr_in, v_in_stride, v_in4, is_contiguous_in);
        // Input point 6: x(5)
        curr_in = in + in_strides[5];
        LDR_256_S(curr_in, v_in_stride, v_in5, is_contiguous_in);

        v_s0 = _mm256_add_ps(v_in0, v_in3);
        v_s1 = _mm256_sub_ps(v_in0, v_in3);
        v_s2 = _mm256_add_ps(v_in1, v_in2);
        v_s3 = _mm256_sub_ps(v_in2, v_in1);
        v_s4 = _mm256_add_ps(v_in4, v_in5);
        v_s5 = _mm256_sub_ps(v_in5, v_in4);

        v_s6 = _mm256_add_ps(v_s2, v_s4);
        v_s7 = _mm256_sub_ps(v_s5, v_s3);

        // Output point 1: X(0)
        v_out0 = _mm256_add_ps(v_s0, v_s6);

        // Output point 2: X(1)
        v_out1 = _mm256_add_ps(v_s1, _mm256_mul_ps(v256_CRTM_6_1, v_s7));

        // Output point 3: X(2)
        v_out2 = _mm256_mul_ps(v256_CRTM_6_2, _mm256_sub_ps(v_s4, v_s2));

        // Output point 4: X(3)
        v_out3 = _mm256_sub_ps(v_s0, _mm256_mul_ps(v256_CRTM_6_1, v_s6));

        // Output point 5: X(4)
        v_out4 = _mm256_mul_ps(v256_CRTM_6_2, _mm256_add_ps(v_s5, v_s3));

        // Output point 6: X(5)
        v_out5 = _mm256_sub_ps(v_s1, v_s7);

        STR_256_S(curr_out, v_out_stride, v_out0, is_contiguous_out);
        curr_out = out + out_strides[1];
        STRI_2x256_S(curr_out, v_out_stride, v_out1, v_out2);
        curr_out = out + out_strides[3];
        STRI_2x256_S(curr_out, v_out_stride, v_out3, v_out4);
        curr_out = out + out_strides[5];
        STR_256_S(curr_out, v_out_stride, v_out5, is_contiguous_out);

        in = in + (v_in_stride << 3);
        out = out + (v_out_stride << 3);
    }
    // tail cases
    if (remaining_sets & NUM_SETS_REAL_128_S)
    {
        __m128 v_in0, v_in1, v_in2, v_in3, v_in4, v_in5;
        __m128 v_s0, v_s1, v_s2, v_s3, v_s4, v_s5, v_s6, v_s7;
        __m128 v_out0, v_out1, v_out2, v_out3, v_out4, v_out5;

        curr_in = in;
        curr_out = out;

        __m128 v128_CRTM_6_1 = _mm512_castps512_ps128(v_CRTM_6_1);
        __m128 v128_CRTM_6_2 = _mm512_castps512_ps128(v_CRTM_6_2);

        // Input point 1: x(0)
        LDR_128_S(curr_in, v_in_stride, v_in0, is_contiguous_in);
        // Input point 2: x(1)
        curr_in = in + in_strides[1];
        LDR_128_S(curr_in, v_in_stride, v_in1, is_contiguous_in);
        // Input point 3: x(2)
        curr_in = in + in_strides[2];
        LDR_128_S(curr_in, v_in_stride, v_in2, is_contiguous_in);
        // Input point 4: x(3)
        curr_in = in + in_strides[3];
        LDR_128_S(curr_in, v_in_stride, v_in3, is_contiguous_in);
        // Input point 5: x(4)
        curr_in = in + in_strides[4];
        LDR_128_S(curr_in, v_in_stride, v_in4, is_contiguous_in);
        // Input point 6: x(5)
        curr_in = in + in_strides[5];
        LDR_128_S(curr_in, v_in_stride, v_in5, is_contiguous_in);

        v_s0 = _mm_add_ps(v_in0, v_in3);
        v_s1 = _mm_sub_ps(v_in0, v_in3);
        v_s2 = _mm_add_ps(v_in1, v_in2);
        v_s3 = _mm_sub_ps(v_in2, v_in1);
        v_s4 = _mm_add_ps(v_in4, v_in5);
        v_s5 = _mm_sub_ps(v_in5, v_in4);

        v_s6 = _mm_add_ps(v_s2, v_s4);
        v_s7 = _mm_sub_ps(v_s5, v_s3);

        // Output point 1: X(0)
        v_out0 = _mm_add_ps(v_s0, v_s6);

        // Output point 2: X(1)
        v_out1 = _mm_add_ps(v_s1, _mm_mul_ps(v128_CRTM_6_1, v_s7));

        // Output point 3: X(2)
        v_out2 = _mm_mul_ps(v128_CRTM_6_2, _mm_sub_ps(v_s4, v_s2));

        // Output point 4: X(3)
        v_out3 = _mm_sub_ps(v_s0, _mm_mul_ps(v128_CRTM_6_1, v_s6));

        // Output point 5: X(4)
        v_out4 = _mm_mul_ps(v128_CRTM_6_2, _mm_add_ps(v_s5, v_s3));

        // Output point 6: X(5)
        v_out5 = _mm_sub_ps(v_s1, v_s7);

        STR_128_S(curr_out, v_out_stride, v_out0, is_contiguous_out);
        curr_out = out + out_strides[1];
        STRI_2x128_S(curr_out, v_out_stride, v_out1, v_out2);
        curr_out = out + out_strides[3];
        STRI_2x128_S(curr_out, v_out_stride, v_out3, v_out4);
        curr_out = out + out_strides[5];
        STR_128_S(curr_out, v_out_stride, v_out5, is_contiguous_out);

        in = in + (v_in_stride << 2);
        out = out + (v_out_stride << 2);
    }
    // tail cases
    if (remaining_sets & 2)
    {
        __m128 v_in0, v_in1, v_in2, v_in3, v_in4, v_in5;
        __m128 v_s0, v_s1, v_s2, v_s3, v_s4, v_s5, v_s6, v_s7;
        __m128 v_out0, v_out1, v_out2, v_out3, v_out4, v_out5;

        curr_in = in;
        curr_out = out;

        __m128 v128_CRTM_6_1 = _mm512_castps512_ps128(v_CRTM_6_1);
        __m128 v128_CRTM_6_2 = _mm512_castps512_ps128(v_CRTM_6_2);

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

        v_s0 = _mm_add_ps(v_in0, v_in3);
        v_s1 = _mm_sub_ps(v_in0, v_in3);
        v_s2 = _mm_add_ps(v_in1, v_in2);
        v_s3 = _mm_sub_ps(v_in2, v_in1);
        v_s4 = _mm_add_ps(v_in4, v_in5);
        v_s5 = _mm_sub_ps(v_in5, v_in4);

        v_s6 = _mm_add_ps(v_s2, v_s4);
        v_s7 = _mm_sub_ps(v_s5, v_s3);

        // Output point 1: X(0)
        v_out0 = _mm_add_ps(v_s0, v_s6);

        // Output point 2: X(1)
        v_out1 = _mm_add_ps(v_s1, _mm_mul_ps(v128_CRTM_6_1, v_s7));

        // Output point 3: X(2)
        v_out2 = _mm_mul_ps(v128_CRTM_6_2, _mm_sub_ps(v_s4, v_s2));

        // Output point 4: X(3)
        v_out3 = _mm_sub_ps(v_s0, _mm_mul_ps(v128_CRTM_6_1, v_s6));

        // Output point 5: X(4)
        v_out4 = _mm_mul_ps(v128_CRTM_6_2, _mm_add_ps(v_s5, v_s3));

        // Output point 6: X(5)
        v_out5 = _mm_sub_ps(v_s1, v_s7);

        STHR_128_S(curr_out, v_out_stride, v_out0);
        curr_out = out + out_strides[1];
        STHRI_2x128_S(curr_out, v_out_stride, v_out1, v_out2);
        curr_out = out + out_strides[3];
        STHRI_2x128_S(curr_out, v_out_stride, v_out3, v_out4);
        curr_out = out + out_strides[5];
        STHR_128_S(curr_out, v_out_stride, v_out5);

        in = in + (v_in_stride << 1);
        out = out + (v_out_stride << 1);
    }
    // tail cases
    if (remaining_sets & 1)
    {
        FFTZ_FLOAT in0, in1, in2, in3, in4, in5;
        FFTZ_FLOAT s0, s1, s2, s3, s4, s5, s6, s7;

        // Input point 1: x(0)
        in0 = *in;
        // Input point 2: x(1)
        in1 = in[in_strides[1]];
        // Input point 3: x(2)
        in2 = in[in_strides[2]];
        // Input point 4: x(3)
        in3 = in[in_strides[3]];
        // Input point 2: x(4)
        in4 = in[in_strides[4]];
        // Input point 3: x(5)
        in5 = in[in_strides[5]];

        s0 = in0 + in3;
        s1 = in0 - in3;
        s2 = in1 + in2;
        s3 = in2 - in1;
        s4 = in4 + in5;
        s5 = in5 - in4;

        s6 = s2 + s4;
        s7 = s5 - s3;

        // Output point 1: X(0)
        *out = s0 + s6;

        // Output point 2: X(1)
        out[out_strides[1]] = s1 + (CRTM_6_1 * s7);

        // Output point 3: X(2)
        out[out_strides[2]] = CRTM_6_2 * (s4 - s2);

        // Output point 4: X(3)
        out[out_strides[3]] = s0 - (CRTM_6_1 * s6);

        // Output point 5: X(4)
        out[out_strides[4]] = CRTM_6_2 * (s5 + s3);

        // Output point 6: X(5)
        out[out_strides[5]] = s1 - s7;
    }
    AOCLFFTZ_LOG(DEBUG, global_logger_mode, "Exit");
}

static FFTZ_VOID r2hc_rfft6avx512_fp32_bwd(FFTZ_VOID *in_real,
                                           FFTZ_VOID *in_imag,
                                           FFTZ_VOID *out_real,
                                           FFTZ_VOID *out_imag, FFTZ_INTP n,
                                           aoclfftz_strides_t *strides,
                                           FFTZ_VOID *twd, FFTZ_UINT8 flag)
{
    AOCLFFTZ_LOG(DEBUG, global_logger_mode, "Enter");
    const FFTZ_FLOAT CRTM_6_1 =
        1.732050807568877293527446341505872366942805253f;

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
    FFTZ_INTP N = n / NUM_SETS_REAL_512_S;
    FFTZ_INTP remaining_sets = n % NUM_SETS_REAL_512_S;

    __m512 v_CRTM_6_1 = _mm512_set1_ps(CRTM_6_1);

    for (cnt = 0; cnt < N; cnt++)
    {
        __m512 v_in0, v_in1, v_in2, v_in3, v_in4, v_in5;
        __m512 v_s0, v_s1, v_s2, v_s3, v_s4, v_s5, v_s6, v_s7;
        __m512 v_t0, v_t1;
        __m512 v_out0, v_out1, v_out2, v_out3, v_out4, v_out5;

        curr_in = in;
        curr_out = out;

        // Input point 1: x(0)
        LDR_512_S(curr_in, v_in_stride, v_in0, is_contiguous_in);
        // Input point 2: x(1) & Input point 3: x(2)
        curr_in = in + in_strides[1];
        LDRI_2x512_S(curr_in, v_in_stride, v_in1, v_in2);
        // Input point 4: x(3) & Input point 5: x(4)
        curr_in = in + in_strides[3];
        LDRI_2x512_S(curr_in, v_in_stride, v_in3, v_in4);
        // Input point 6: x(5)
        curr_in = in + in_strides[5];
        LDR_512_S(curr_in, v_in_stride, v_in5, is_contiguous_in);

        v_s0 = _mm512_add_ps(v_in0, v_in5);
        v_s1 = _mm512_sub_ps(v_in0, v_in5);
        v_s2 = _mm512_add_ps(v_in1, v_in3);
        v_s3 = _mm512_sub_ps(v_in1, v_in3);
        v_s4 = _mm512_add_ps(v_in2, v_in4);
        v_s5 = _mm512_sub_ps(v_in2, v_in4);

        v_t0 = _mm512_mul_ps(v_CRTM_6_1, v_s4);
        v_t1 = _mm512_mul_ps(v_CRTM_6_1, v_s5);
        v_s6 = _mm512_add_ps(v_s1, v_s3);
        v_s7 = _mm512_sub_ps(v_s0, v_s2);

        // Output point 1: X(0)
        v_out0 = _mm512_add_ps(v_s0, _mm512_add_ps(v_s2, v_s2));

        // Output point 2: X(1)
        v_out1 = _mm512_sub_ps(v_s6, v_t0);

        // Output point 3: X(2)
        v_out2 = _mm512_sub_ps(v_s7, v_t1);

        // Output point 4: X(3)
        v_out3 = _mm512_sub_ps(v_s1, _mm512_add_ps(v_s3, v_s3));

        // Output point 5: X(4)
        v_out4 = _mm512_add_ps(v_s7, v_t1);

        // Output point 6: X(5)
        v_out5 = _mm512_add_ps(v_s6, v_t0);

        STR_512_S(curr_out, v_out_stride, v_out0, is_contiguous_out);
        curr_out = out + out_strides[1];
        STR_512_S(curr_out, v_out_stride, v_out1, is_contiguous_out);
        curr_out = out + out_strides[2];
        STR_512_S(curr_out, v_out_stride, v_out2, is_contiguous_out);
        curr_out = out + out_strides[3];
        STR_512_S(curr_out, v_out_stride, v_out3, is_contiguous_out);
        curr_out = out + out_strides[4];
        STR_512_S(curr_out, v_out_stride, v_out4, is_contiguous_out);
        curr_out = out + out_strides[5];
        STR_512_S(curr_out, v_out_stride, v_out5, is_contiguous_out);

        in = in + (v_in_stride << 4);
        out = out + (v_out_stride << 4);
    }
    // tail cases
    if (remaining_sets & NUM_SETS_REAL_256_S)
    {
        __m256 v_in0, v_in1, v_in2, v_in3, v_in4, v_in5;
        __m256 v_s0, v_s1, v_s2, v_s3, v_s4, v_s5, v_s6, v_s7;
        __m256 v_t0, v_t1;
        __m256 v_out0, v_out1, v_out2, v_out3, v_out4, v_out5;

        curr_in = in;
        curr_out = out;

        __m256 v256_CRTM_6_1 = _mm512_castps512_ps256(v_CRTM_6_1);

        // Input point 1: x(0)
        LDR_256_S(curr_in, v_in_stride, v_in0, is_contiguous_in);
        // Input point 2: x(1) & Input point 3: x(2)
        curr_in = in + in_strides[1];
        LDRI_2x256_S(curr_in, v_in_stride, v_in1, v_in2);
        // Input point 4: x(3) & Input point 5: x(4)
        curr_in = in + in_strides[3];
        LDRI_2x256_S(curr_in, v_in_stride, v_in3, v_in4);
        // Input point 6: x(5)
        curr_in = in + in_strides[5];
        LDR_256_S(curr_in, v_in_stride, v_in5, is_contiguous_in);

        v_s0 = _mm256_add_ps(v_in0, v_in5);
        v_s1 = _mm256_sub_ps(v_in0, v_in5);
        v_s2 = _mm256_add_ps(v_in1, v_in3);
        v_s3 = _mm256_sub_ps(v_in1, v_in3);
        v_s4 = _mm256_add_ps(v_in2, v_in4);
        v_s5 = _mm256_sub_ps(v_in2, v_in4);

        v_t0 = _mm256_mul_ps(v256_CRTM_6_1, v_s4);
        v_t1 = _mm256_mul_ps(v256_CRTM_6_1, v_s5);
        v_s6 = _mm256_add_ps(v_s1, v_s3);
        v_s7 = _mm256_sub_ps(v_s0, v_s2);

        // Output point 1: X(0)
        v_out0 = _mm256_add_ps(v_s0, _mm256_add_ps(v_s2, v_s2));

        // Output point 2: X(1)
        v_out1 = _mm256_sub_ps(v_s6, v_t0);

        // Output point 3: X(2)
        v_out2 = _mm256_sub_ps(v_s7, v_t1);

        // Output point 4: X(3)
        v_out3 = _mm256_sub_ps(v_s1, _mm256_add_ps(v_s3, v_s3));

        // Output point 5: X(4)
        v_out4 = _mm256_add_ps(v_s7, v_t1);

        // Output point 6: X(5)
        v_out5 = _mm256_add_ps(v_s6, v_t0);

        STR_256_S(curr_out, v_out_stride, v_out0, is_contiguous_out);
        curr_out = out + out_strides[1];
        STR_256_S(curr_out, v_out_stride, v_out1, is_contiguous_out);
        curr_out = out + out_strides[2];
        STR_256_S(curr_out, v_out_stride, v_out2, is_contiguous_out);
        curr_out = out + out_strides[3];
        STR_256_S(curr_out, v_out_stride, v_out3, is_contiguous_out);
        curr_out = out + out_strides[4];
        STR_256_S(curr_out, v_out_stride, v_out4, is_contiguous_out);
        curr_out = out + out_strides[5];
        STR_256_S(curr_out, v_out_stride, v_out5, is_contiguous_out);

        in = in + (v_in_stride << 3);
        out = out + (v_out_stride << 3);
    }
    // tail cases
    if (remaining_sets & NUM_SETS_REAL_128_S)
    {
        __m128 v_in0, v_in1, v_in2, v_in3, v_in4, v_in5;
        __m128 v_s0, v_s1, v_s2, v_s3, v_s4, v_s5, v_s6, v_s7;
        __m128 v_t0, v_t1;
        __m128 v_out0, v_out1, v_out2, v_out3, v_out4, v_out5;

        curr_in = in;
        curr_out = out;

        __m128 v128_CRTM_6_1 = _mm512_castps512_ps128(v_CRTM_6_1);

        // Input point 1: x(0)
        LDR_128_S(curr_in, v_in_stride, v_in0, is_contiguous_in);
        // Input point 2: x(1) & Input point 3: x(2)
        curr_in = in + in_strides[1];
        LDRI_2x128_S(curr_in, v_in_stride, v_in1, v_in2);
        // Input point 4: x(3) & Input point 5: x(4)
        curr_in = in + in_strides[3];
        LDRI_2x128_S(curr_in, v_in_stride, v_in3, v_in4);
        // Input point 6: x(5)
        curr_in = in + in_strides[5];
        LDR_128_S(curr_in, v_in_stride, v_in5, is_contiguous_in);

        v_s0 = _mm_add_ps(v_in0, v_in5);
        v_s1 = _mm_sub_ps(v_in0, v_in5);
        v_s2 = _mm_add_ps(v_in1, v_in3);
        v_s3 = _mm_sub_ps(v_in1, v_in3);
        v_s4 = _mm_add_ps(v_in2, v_in4);
        v_s5 = _mm_sub_ps(v_in2, v_in4);

        v_t0 = _mm_mul_ps(v128_CRTM_6_1, v_s4);
        v_t1 = _mm_mul_ps(v128_CRTM_6_1, v_s5);
        v_s6 = _mm_add_ps(v_s1, v_s3);
        v_s7 = _mm_sub_ps(v_s0, v_s2);

        // Output point 1: X(0)
        v_out0 = _mm_add_ps(v_s0, _mm_add_ps(v_s2, v_s2));

        // Output point 2: X(1)
        v_out1 = _mm_sub_ps(v_s6, v_t0);

        // Output point 3: X(2)
        v_out2 = _mm_sub_ps(v_s7, v_t1);

        // Output point 4: X(3)
        v_out3 = _mm_sub_ps(v_s1, _mm_add_ps(v_s3, v_s3));

        // Output point 5: X(4)
        v_out4 = _mm_add_ps(v_s7, v_t1);

        // Output point 6: X(5)
        v_out5 = _mm_add_ps(v_s6, v_t0);

        STR_128_S(curr_out, v_out_stride, v_out0, is_contiguous_out);
        curr_out = out + out_strides[1];
        STR_128_S(curr_out, v_out_stride, v_out1, is_contiguous_out);
        curr_out = out + out_strides[2];
        STR_128_S(curr_out, v_out_stride, v_out2, is_contiguous_out);
        curr_out = out + out_strides[3];
        STR_128_S(curr_out, v_out_stride, v_out3, is_contiguous_out);
        curr_out = out + out_strides[4];
        STR_128_S(curr_out, v_out_stride, v_out4, is_contiguous_out);
        curr_out = out + out_strides[5];
        STR_128_S(curr_out, v_out_stride, v_out5, is_contiguous_out);

        in = in + (v_in_stride << 2);
        out = out + (v_out_stride << 2);
    }
    // tail cases
    if (remaining_sets & 2)
    {
        __m128 v_in0, v_in1, v_in2, v_in3, v_in4, v_in5;
        __m128 v_s0, v_s1, v_s2, v_s3, v_s4, v_s5, v_s6, v_s7;
        __m128 v_t0, v_t1;
        __m128 v_out0, v_out1, v_out2, v_out3, v_out4, v_out5;

        curr_in = in;
        curr_out = out;

        __m128 v128_CRTM_6_1 = _mm512_castps512_ps128(v_CRTM_6_1);

        // Input point 1: x(0)
        LDHR_128_S(curr_in, v_in_stride, v_in0);
        // Input point 2: x(1) & Input point 3: x(2)
        curr_in = in + in_strides[1];
        LDHRI_2x128_S(curr_in, v_in_stride, v_in1, v_in2);
        // Input point 4: x(3) & Input point 5: x(4)
        curr_in = in + in_strides[3];
        LDHRI_2x128_S(curr_in, v_in_stride, v_in3, v_in4);
        // Input point 6: x(5)
        curr_in = in + in_strides[5];
        LDHR_128_S(curr_in, v_in_stride, v_in5);

        v_s0 = _mm_add_ps(v_in0, v_in5);
        v_s1 = _mm_sub_ps(v_in0, v_in5);
        v_s2 = _mm_add_ps(v_in1, v_in3);
        v_s3 = _mm_sub_ps(v_in1, v_in3);
        v_s4 = _mm_add_ps(v_in2, v_in4);
        v_s5 = _mm_sub_ps(v_in2, v_in4);

        v_t0 = _mm_mul_ps(v128_CRTM_6_1, v_s4);
        v_t1 = _mm_mul_ps(v128_CRTM_6_1, v_s5);
        v_s6 = _mm_add_ps(v_s1, v_s3);
        v_s7 = _mm_sub_ps(v_s0, v_s2);

        // Output point 1: X(0)
        v_out0 = _mm_add_ps(v_s0, _mm_add_ps(v_s2, v_s2));

        // Output point 2: X(1)
        v_out1 = _mm_sub_ps(v_s6, v_t0);

        // Output point 3: X(2)
        v_out2 = _mm_sub_ps(v_s7, v_t1);

        // Output point 4: X(3)
        v_out3 = _mm_sub_ps(v_s1, _mm_add_ps(v_s3, v_s3));

        // Output point 5: X(4)
        v_out4 = _mm_add_ps(v_s7, v_t1);

        // Output point 6: X(5)
        v_out5 = _mm_add_ps(v_s6, v_t0);

        STHR_128_S(curr_out, v_out_stride, v_out0);
        curr_out = out + out_strides[1];
        STHR_128_S(curr_out, v_out_stride, v_out1);
        curr_out = out + out_strides[2];
        STHR_128_S(curr_out, v_out_stride, v_out2);
        curr_out = out + out_strides[3];
        STHR_128_S(curr_out, v_out_stride, v_out3);
        curr_out = out + out_strides[4];
        STHR_128_S(curr_out, v_out_stride, v_out4);
        curr_out = out + out_strides[5];
        STHR_128_S(curr_out, v_out_stride, v_out5);

        in = in + (v_in_stride << 1);
        out = out + (v_out_stride << 1);
    }
    // tail cases
    if (remaining_sets & 1)
    {
        FFTZ_FLOAT in0, in1, in2, in3, in4, in5;
        FFTZ_FLOAT s0, s1, s2, s3, s4, s5, s6, s7;
        FFTZ_FLOAT t0, t1;

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

        s0 = in0 + in5;
        s1 = in0 - in5;
        s2 = in1 + in3;
        s3 = in1 - in3;
        s4 = in2 + in4;
        s5 = in2 - in4;

        t0 = CRTM_6_1 * s4;
        t1 = CRTM_6_1 * s5;
        s6 = s1 + s3;
        s7 = s0 - s2;

        // Output point 1: X(0)
        *out = s0 + (2 * s2);

        // Output point 2: X(1)
        out[out_strides[1]] = s6 - t0;

        // Output point 3: X(2)
        out[out_strides[2]] = s7 - t1;

        // Output point 4: X(3)
        out[out_strides[3]] = s1 - (2 * s3);

        // Output point 5: X(4)
        out[out_strides[4]] = s7 + t1;

        // Output point 6: X(5)
        out[out_strides[5]] = s6 + t0;
    }
    AOCLFFTZ_LOG(DEBUG, global_logger_mode, "Exit");
}

static FFTZ_VOID r2hc_rfft6avx512_fp64_fwd(FFTZ_VOID *in_real,
                                           FFTZ_VOID *in_imag,
                                           FFTZ_VOID *out_real,
                                           FFTZ_VOID *out_imag, FFTZ_INTP n,
                                           aoclfftz_strides_t *strides,
                                           FFTZ_VOID *twd, FFTZ_UINT8 flag)
{
    AOCLFFTZ_LOG(DEBUG, global_logger_mode, "Enter");
    const FFTZ_DOUBLE CRTM_6_1 =
        0.500000000000000000000000000000000000000000000;
    const FFTZ_DOUBLE CRTM_6_2 =
        0.866025403784438646763723170752936183471402627;

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
    FFTZ_INTP N = n / NUM_SETS_REAL_512_D;
    FFTZ_INTP remaining_sets = n % NUM_SETS_REAL_512_D;

    __m512d v_CRTM_6_1 = _mm512_set1_pd(CRTM_6_1);
    __m512d v_CRTM_6_2 = _mm512_set1_pd(CRTM_6_2);

    for (cnt = 0; cnt < N; cnt++)
    {
        __m512d v_in0, v_in1, v_in2, v_in3, v_in4, v_in5;
        __m512d v_s0, v_s1, v_s2, v_s3, v_s4, v_s5, v_s6, v_s7;
        __m512d v_out0, v_out1, v_out2, v_out3, v_out4, v_out5;

        curr_in = in;
        curr_out = out;

        // Input point 1: x(0)
        LDR_512_D(curr_in, v_in_stride, v_in0, is_contiguous_in);
        // Input point 2: x(1)
        curr_in = in + in_strides[1];
        LDR_512_D(curr_in, v_in_stride, v_in1, is_contiguous_in);
        // Input point 3: x(2)
        curr_in = in + in_strides[2];
        LDR_512_D(curr_in, v_in_stride, v_in2, is_contiguous_in);
        // Input point 4: x(3)
        curr_in = in + in_strides[3];
        LDR_512_D(curr_in, v_in_stride, v_in3, is_contiguous_in);
        // Input point 5: x(4)
        curr_in = in + in_strides[4];
        LDR_512_D(curr_in, v_in_stride, v_in4, is_contiguous_in);
        // Input point 6: x(5)
        curr_in = in + in_strides[5];
        LDR_512_D(curr_in, v_in_stride, v_in5, is_contiguous_in);

        v_s0 = _mm512_add_pd(v_in0, v_in3);
        v_s1 = _mm512_sub_pd(v_in0, v_in3);
        v_s2 = _mm512_add_pd(v_in1, v_in2);
        v_s3 = _mm512_sub_pd(v_in2, v_in1);
        v_s4 = _mm512_add_pd(v_in4, v_in5);
        v_s5 = _mm512_sub_pd(v_in5, v_in4);

        v_s6 = _mm512_add_pd(v_s2, v_s4);
        v_s7 = _mm512_sub_pd(v_s5, v_s3);

        // Output point 1: X(0)
        v_out0 = _mm512_add_pd(v_s0, v_s6);

        // Output point 2: X(1)
        v_out1 = _mm512_add_pd(v_s1, _mm512_mul_pd(v_CRTM_6_1, v_s7));

        // Output point 3: X(2)
        v_out2 = _mm512_mul_pd(v_CRTM_6_2, _mm512_sub_pd(v_s4, v_s2));

        // Output point 4: X(3)
        v_out3 = _mm512_sub_pd(v_s0, _mm512_mul_pd(v_CRTM_6_1, v_s6));

        // Output point 5: X(4)
        v_out4 = _mm512_mul_pd(v_CRTM_6_2, _mm512_add_pd(v_s5, v_s3));

        // Output point 6: X(5)
        v_out5 = _mm512_sub_pd(v_s1, v_s7);

        STR_512_D(curr_out, v_out_stride, v_out0, is_contiguous_out);
        curr_out = out + out_strides[1];
        STRI_2x512_D(curr_out, v_out_stride, v_out1, v_out2);
        curr_out = out + out_strides[3];
        STRI_2x512_D(curr_out, v_out_stride, v_out3, v_out4);
        curr_out = out + out_strides[5];
        STR_512_D(curr_out, v_out_stride, v_out5, is_contiguous_out);

        in = in + (v_in_stride << 3);
        out = out + (v_out_stride << 3);
    }
    // tail cases
    if (remaining_sets & NUM_SETS_REAL_256_D)
    {
        __m256d v_in0, v_in1, v_in2, v_in3, v_in4, v_in5;
        __m256d v_s0, v_s1, v_s2, v_s3, v_s4, v_s5, v_s6, v_s7;
        __m256d v_out0, v_out1, v_out2, v_out3, v_out4, v_out5;

        curr_in = in;
        curr_out = out;

        __m256d v256_CRTM_6_1 = _mm512_castpd512_pd256(v_CRTM_6_1);
        __m256d v256_CRTM_6_2 = _mm512_castpd512_pd256(v_CRTM_6_2);

        // Input point 1: x(0)
        LDR_256_D(curr_in, v_in_stride, v_in0, is_contiguous_in);
        // Input point 2: x(1)
        curr_in = in + in_strides[1];
        LDR_256_D(curr_in, v_in_stride, v_in1, is_contiguous_in);
        // Input point 3: x(2)
        curr_in = in + in_strides[2];
        LDR_256_D(curr_in, v_in_stride, v_in2, is_contiguous_in);
        // Input point 4: x(3)
        curr_in = in + in_strides[3];
        LDR_256_D(curr_in, v_in_stride, v_in3, is_contiguous_in);
        // Input point 5: x(4)
        curr_in = in + in_strides[4];
        LDR_256_D(curr_in, v_in_stride, v_in4, is_contiguous_in);
        // Input point 6: x(5)
        curr_in = in + in_strides[5];
        LDR_256_D(curr_in, v_in_stride, v_in5, is_contiguous_in);

        v_s0 = _mm256_add_pd(v_in0, v_in3);
        v_s1 = _mm256_sub_pd(v_in0, v_in3);
        v_s2 = _mm256_add_pd(v_in1, v_in2);
        v_s3 = _mm256_sub_pd(v_in2, v_in1);
        v_s4 = _mm256_add_pd(v_in4, v_in5);
        v_s5 = _mm256_sub_pd(v_in5, v_in4);

        v_s6 = _mm256_add_pd(v_s2, v_s4);
        v_s7 = _mm256_sub_pd(v_s5, v_s3);

        // Output point 1: X(0)
        v_out0 = _mm256_add_pd(v_s0, v_s6);

        // Output point 2: X(1)
        v_out1 = _mm256_add_pd(v_s1, _mm256_mul_pd(v256_CRTM_6_1, v_s7));

        // Output point 3: X(2)
        v_out2 = _mm256_mul_pd(v256_CRTM_6_2, _mm256_sub_pd(v_s4, v_s2));

        // Output point 4: X(3)
        v_out3 = _mm256_sub_pd(v_s0, _mm256_mul_pd(v256_CRTM_6_1, v_s6));

        // Output point 5: X(4)
        v_out4 = _mm256_mul_pd(v256_CRTM_6_2, _mm256_add_pd(v_s5, v_s3));

        // Output point 6: X(5)
        v_out5 = _mm256_sub_pd(v_s1, v_s7);

        STR_256_D(curr_out, v_out_stride, v_out0, is_contiguous_out);
        curr_out = out + out_strides[1];
        STRI_2x256_D(curr_out, v_out_stride, v_out1, v_out2);
        curr_out = out + out_strides[3];
        STRI_2x256_D(curr_out, v_out_stride, v_out3, v_out4);
        curr_out = out + out_strides[5];
        STR_256_D(curr_out, v_out_stride, v_out5, is_contiguous_out);

        in = in + (v_in_stride << 2);
        out = out + (v_out_stride << 2);
    }
    // tail cases
    if (remaining_sets & 2)
    {
        __m128d v_in0, v_in1, v_in2, v_in3, v_in4, v_in5;
        __m128d v_s0, v_s1, v_s2, v_s3, v_s4, v_s5, v_s6, v_s7;
        __m128d v_out0, v_out1, v_out2, v_out3, v_out4, v_out5;

        curr_in = in;
        curr_out = out;

        __m128d v128_CRTM_6_1 = _mm512_castpd512_pd128(v_CRTM_6_1);
        __m128d v128_CRTM_6_2 = _mm512_castpd512_pd128(v_CRTM_6_2);

        // Input point 1: x(0)
        LDR_128_D(curr_in, v_in_stride, v_in0, is_contiguous_in);
        // Input point 2: x(1)
        curr_in = in + in_strides[1];
        LDR_128_D(curr_in, v_in_stride, v_in1, is_contiguous_in);
        // Input point 3: x(2)
        curr_in = in + in_strides[2];
        LDR_128_D(curr_in, v_in_stride, v_in2, is_contiguous_in);
        // Input point 4: x(3)
        curr_in = in + in_strides[3];
        LDR_128_D(curr_in, v_in_stride, v_in3, is_contiguous_in);
        // Input point 5: x(4)
        curr_in = in + in_strides[4];
        LDR_128_D(curr_in, v_in_stride, v_in4, is_contiguous_in);
        // Input point 6: x(5)
        curr_in = in + in_strides[5];
        LDR_128_D(curr_in, v_in_stride, v_in5, is_contiguous_in);

        v_s0 = _mm_add_pd(v_in0, v_in3);
        v_s1 = _mm_sub_pd(v_in0, v_in3);
        v_s2 = _mm_add_pd(v_in1, v_in2);
        v_s3 = _mm_sub_pd(v_in2, v_in1);
        v_s4 = _mm_add_pd(v_in4, v_in5);
        v_s5 = _mm_sub_pd(v_in5, v_in4);

        v_s6 = _mm_add_pd(v_s2, v_s4);
        v_s7 = _mm_sub_pd(v_s5, v_s3);

        // Output point 1: X(0)
        v_out0 = _mm_add_pd(v_s0, v_s6);

        // Output point 2: X(1)
        v_out1 = _mm_add_pd(v_s1, _mm_mul_pd(v128_CRTM_6_1, v_s7));

        // Output point 3: X(2)
        v_out2 = _mm_mul_pd(v128_CRTM_6_2, _mm_sub_pd(v_s4, v_s2));

        // Output point 4: X(3)
        v_out3 = _mm_sub_pd(v_s0, _mm_mul_pd(v128_CRTM_6_1, v_s6));

        // Output point 5: X(4)
        v_out4 = _mm_mul_pd(v128_CRTM_6_2, _mm_add_pd(v_s5, v_s3));

        // Output point 6: X(5)
        v_out5 = _mm_sub_pd(v_s1, v_s7);

        STR_128_D(curr_out, v_out_stride, v_out0, is_contiguous_out);
        curr_out = out + out_strides[1];
        STRI_2x128_D(curr_out, v_out_stride, v_out1, v_out2);
        curr_out = out + out_strides[3];
        STRI_2x128_D(curr_out, v_out_stride, v_out3, v_out4);
        curr_out = out + out_strides[5];
        STR_128_D(curr_out, v_out_stride, v_out5, is_contiguous_out);

        in = in + (v_in_stride << 1);
        out = out + (v_out_stride << 1);
    }
    // tail cases
    if (remaining_sets & 1)
    {
        FFTZ_DOUBLE in0, in1, in2, in3, in4, in5;
        FFTZ_DOUBLE s0, s1, s2, s3, s4, s5, s6, s7;

        // Input point 1: x(0)
        in0 = *in;
        // Input point 2: x(1)
        in1 = in[in_strides[1]];
        // Input point 3: x(2)
        in2 = in[in_strides[2]];
        // Input point 4: x(3)
        in3 = in[in_strides[3]];
        // Input point 2: x(4)
        in4 = in[in_strides[4]];
        // Input point 3: x(5)
        in5 = in[in_strides[5]];

        s0 = in0 + in3;
        s1 = in0 - in3;
        s2 = in1 + in2;
        s3 = in2 - in1;
        s4 = in4 + in5;
        s5 = in5 - in4;
        s6 = s2 + s4;
        s7 = s5 - s3;

        // Output point 1: X(0)
        *out = s0 + s6;

        // Output point 2: X(1)
        out[out_strides[1]] = s1 + (CRTM_6_1 * s7);

        // Output point 3: X(2)
        out[out_strides[2]] = CRTM_6_2 * (s4 - s2);

        // Output point 4: X(3)
        out[out_strides[3]] = s0 - (CRTM_6_1 * s6);

        // Output point 5: X(4)
        out[out_strides[4]] = CRTM_6_2 * (s5 + s3);

        // Output point 6: X(5)
        out[out_strides[5]] = s1 - s7;
    }
    AOCLFFTZ_LOG(DEBUG, global_logger_mode, "Exit");
}

static FFTZ_VOID r2hc_rfft6avx512_fp64_bwd(FFTZ_VOID *in_real,
                                           FFTZ_VOID *in_imag,
                                           FFTZ_VOID *out_real,
                                           FFTZ_VOID *out_imag, FFTZ_INTP n,
                                           aoclfftz_strides_t *strides,
                                           FFTZ_VOID *twd, FFTZ_UINT8 flag)
{
    AOCLFFTZ_LOG(DEBUG, global_logger_mode, "Enter");
    const FFTZ_DOUBLE CRTM_6_1 =
        1.732050807568877293527446341505872366942805253;

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
    FFTZ_INTP N = n / NUM_SETS_REAL_512_D;
    FFTZ_INTP remaining_sets = n % NUM_SETS_REAL_512_D;

    __m512d v_CRTM_6_1 = _mm512_set1_pd(CRTM_6_1);

    for (cnt = 0; cnt < N; cnt++)
    {
        __m512d v_in0, v_in1, v_in2, v_in3, v_in4, v_in5;
        __m512d v_s0, v_s1, v_s2, v_s3, v_s4, v_s5, v_s6, v_s7;
        __m512d v_t0, v_t1;
        __m512d v_out0, v_out1, v_out2, v_out3, v_out4, v_out5;

        curr_in = in;
        curr_out = out;

        // Input point 1: x(0)
        LDR_512_D(curr_in, v_in_stride, v_in0, is_contiguous_in);
        // Input point 2: x(1) & Input point 3: x(2)
        curr_in = in + in_strides[1];
        LDRI_2x512_D(curr_in, v_in_stride, v_in1, v_in2);
        // Input point 4: x(3) & Input point 5: x(4)
        curr_in = in + in_strides[3];
        LDRI_2x512_D(curr_in, v_in_stride, v_in3, v_in4);
        // Input point 6: x(5)
        curr_in = in + in_strides[5];
        LDR_512_D(curr_in, v_in_stride, v_in5, is_contiguous_in);

        v_s0 = _mm512_add_pd(v_in0, v_in5);
        v_s1 = _mm512_sub_pd(v_in0, v_in5);
        v_s2 = _mm512_add_pd(v_in1, v_in3);
        v_s3 = _mm512_sub_pd(v_in1, v_in3);
        v_s4 = _mm512_add_pd(v_in2, v_in4);
        v_s5 = _mm512_sub_pd(v_in2, v_in4);
        v_t0 = _mm512_mul_pd(v_CRTM_6_1, v_s4);
        v_t1 = _mm512_mul_pd(v_CRTM_6_1, v_s5);
        v_s6 = _mm512_add_pd(v_s1, v_s3);
        v_s7 = _mm512_sub_pd(v_s0, v_s2);

        // Output point 1: X(0)
        v_out0 = _mm512_add_pd(v_s0, _mm512_add_pd(v_s2, v_s2));

        // Output point 2: X(1)
        v_out1 = _mm512_sub_pd(v_s6, v_t0);

        // Output point 3: X(2)
        v_out2 = _mm512_sub_pd(v_s7, v_t1);

        // Output point 4: X(3)
        v_out3 = _mm512_sub_pd(v_s1, _mm512_add_pd(v_s3, v_s3));

        // Output point 5: X(4)
        v_out4 = _mm512_add_pd(v_s7, v_t1);

        // Output point 6: X(5)
        v_out5 = _mm512_add_pd(v_s6, v_t0);

        STR_512_D(curr_out, v_out_stride, v_out0, is_contiguous_out);
        curr_out = out + out_strides[1];
        STR_512_D(curr_out, v_out_stride, v_out1, is_contiguous_out);
        curr_out = out + out_strides[2];
        STR_512_D(curr_out, v_out_stride, v_out2, is_contiguous_out);
        curr_out = out + out_strides[3];
        STR_512_D(curr_out, v_out_stride, v_out3, is_contiguous_out);
        curr_out = out + out_strides[4];
        STR_512_D(curr_out, v_out_stride, v_out4, is_contiguous_out);
        curr_out = out + out_strides[5];
        STR_512_D(curr_out, v_out_stride, v_out5, is_contiguous_out);

        in = in + (v_in_stride << 3);
        out = out + (v_out_stride << 3);
    }
    // tail cases
    if (remaining_sets & NUM_SETS_REAL_256_D)
    {
        __m256d v_in0, v_in1, v_in2, v_in3, v_in4, v_in5;
        __m256d v_s0, v_s1, v_s2, v_s3, v_s4, v_s5, v_s6, v_s7;
        __m256d v_t0, v_t1;
        __m256d v_out0, v_out1, v_out2, v_out3, v_out4, v_out5;

        curr_in = in;
        curr_out = out;

        __m256d v256_CRTM_6_1 = _mm512_castpd512_pd256(v_CRTM_6_1);

        // Input point 1: x(0)
        LDR_256_D(curr_in, v_in_stride, v_in0, is_contiguous_in);
        // Input point 2: x(1) & Input point 3: x(2)
        curr_in = in + in_strides[1];
        LDRI_2x256_D(curr_in, v_in_stride, v_in1, v_in2);
        // Input point 4: x(3) & Input point 5: x(4)
        curr_in = in + in_strides[3];
        LDRI_2x256_D(curr_in, v_in_stride, v_in3, v_in4);
        // Input point 6: x(5)
        curr_in = in + in_strides[5];
        LDR_256_D(curr_in, v_in_stride, v_in5, is_contiguous_in);

        v_s0 = _mm256_add_pd(v_in0, v_in5);
        v_s1 = _mm256_sub_pd(v_in0, v_in5);
        v_s2 = _mm256_add_pd(v_in1, v_in3);
        v_s3 = _mm256_sub_pd(v_in1, v_in3);
        v_s4 = _mm256_add_pd(v_in2, v_in4);
        v_s5 = _mm256_sub_pd(v_in2, v_in4);
        v_t0 = _mm256_mul_pd(v256_CRTM_6_1, v_s4);
        v_t1 = _mm256_mul_pd(v256_CRTM_6_1, v_s5);
        v_s6 = _mm256_add_pd(v_s1, v_s3);
        v_s7 = _mm256_sub_pd(v_s0, v_s2);

        // Output point 1: X(0)
        v_out0 = _mm256_add_pd(v_s0, _mm256_add_pd(v_s2, v_s2));

        // Output point 2: X(1)
        v_out1 = _mm256_sub_pd(v_s6, v_t0);

        // Output point 3: X(2)
        v_out2 = _mm256_sub_pd(v_s7, v_t1);

        // Output point 4: X(3)
        v_out3 = _mm256_sub_pd(v_s1, _mm256_add_pd(v_s3, v_s3));

        // Output point 5: X(4)
        v_out4 = _mm256_add_pd(v_s7, v_t1);

        // Output point 6: X(5)
        v_out5 = _mm256_add_pd(v_s6, v_t0);

        STR_256_D(curr_out, v_out_stride, v_out0, is_contiguous_out);
        curr_out = out + out_strides[1];
        STR_256_D(curr_out, v_out_stride, v_out1, is_contiguous_out);
        curr_out = out + out_strides[2];
        STR_256_D(curr_out, v_out_stride, v_out2, is_contiguous_out);
        curr_out = out + out_strides[3];
        STR_256_D(curr_out, v_out_stride, v_out3, is_contiguous_out);
        curr_out = out + out_strides[4];
        STR_256_D(curr_out, v_out_stride, v_out4, is_contiguous_out);
        curr_out = out + out_strides[5];
        STR_256_D(curr_out, v_out_stride, v_out5, is_contiguous_out);

        in = in + (v_in_stride << 2);
        out = out + (v_out_stride << 2);
    }
    // tail cases
    if (remaining_sets & 2)
    {
        __m128d v_in0, v_in1, v_in2, v_in3, v_in4, v_in5;
        __m128d v_s0, v_s1, v_s2, v_s3, v_s4, v_s5, v_s6, v_s7;
        __m128d v_t0, v_t1;
        __m128d v_out0, v_out1, v_out2, v_out3, v_out4, v_out5;

        curr_in = in;
        curr_out = out;

        __m128d v128_CRTM_6_1 = _mm512_castpd512_pd128(v_CRTM_6_1);

        // Input point 1: x(0)
        LDR_128_D(curr_in, v_in_stride, v_in0, is_contiguous_in);
        // Input point 2: x(1) & Input point 3: x(2)
        curr_in = in + in_strides[1];
        LDRI_2x128_D(curr_in, v_in_stride, v_in1, v_in2);
        // Input point 4: x(3) & Input point 5: x(4)
        curr_in = in + in_strides[3];
        LDRI_2x128_D(curr_in, v_in_stride, v_in3, v_in4);
        // Input point 6: x(5)
        curr_in = in + in_strides[5];
        LDR_128_D(curr_in, v_in_stride, v_in5, is_contiguous_in);

        v_s0 = _mm_add_pd(v_in0, v_in5);
        v_s1 = _mm_sub_pd(v_in0, v_in5);
        v_s2 = _mm_add_pd(v_in1, v_in3);
        v_s3 = _mm_sub_pd(v_in1, v_in3);
        v_s4 = _mm_add_pd(v_in2, v_in4);
        v_s5 = _mm_sub_pd(v_in2, v_in4);
        v_t0 = _mm_mul_pd(v128_CRTM_6_1, v_s4);
        v_t1 = _mm_mul_pd(v128_CRTM_6_1, v_s5);

        v_s6 = _mm_add_pd(v_s1, v_s3);
        v_s7 = _mm_sub_pd(v_s0, v_s2);

        // Output point 1: X(0)
        v_out0 = _mm_add_pd(v_s0, _mm_add_pd(v_s2, v_s2));

        // Output point 2: X(1)
        v_out1 = _mm_sub_pd(v_s6, v_t0);

        // Output point 3: X(2)
        v_out2 = _mm_sub_pd(v_s7, v_t1);

        // Output point 4: X(3)
        v_out3 = _mm_sub_pd(v_s1, _mm_add_pd(v_s3, v_s3));

        // Output point 5: X(4)
        v_out4 = _mm_add_pd(v_s7, v_t1);

        // Output point 6: X(5)
        v_out5 = _mm_add_pd(v_s6, v_t0);

        STR_128_D(curr_out, v_out_stride, v_out0, is_contiguous_out);
        curr_out = out + out_strides[1];
        STR_128_D(curr_out, v_out_stride, v_out1, is_contiguous_out);
        curr_out = out + out_strides[2];
        STR_128_D(curr_out, v_out_stride, v_out2, is_contiguous_out);
        curr_out = out + out_strides[3];
        STR_128_D(curr_out, v_out_stride, v_out3, is_contiguous_out);
        curr_out = out + out_strides[4];
        STR_128_D(curr_out, v_out_stride, v_out4, is_contiguous_out);
        curr_out = out + out_strides[5];
        STR_128_D(curr_out, v_out_stride, v_out5, is_contiguous_out);

        in = in + (v_in_stride << 1);
        out = out + (v_out_stride << 1);
    }
    // tail cases
    if (remaining_sets & 1)
    {
        FFTZ_DOUBLE in0, in1, in2, in3, in4, in5;
        FFTZ_DOUBLE s0, s1, s2, s3, s4, s5, s6, s7;
        FFTZ_DOUBLE t0, t1;

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

        s0 = in0 + in5;
        s1 = in0 - in5;
        s2 = in1 + in3;
        s3 = in1 - in3;
        s4 = in2 + in4;
        s5 = in2 - in4;

        t0 = CRTM_6_1 * s4;
        t1 = CRTM_6_1 * s5;
        s6 = s1 + s3;
        s7 = s0 - s2;

        // Output point 1: X(0)
        *out = s0 + (2 * s2);

        // Output point 2: X(1)
        out[out_strides[1]] = s6 - t0;

        // Output point 3: X(2)
        out[out_strides[2]] = s7 - t1;

        // Output point 4: X(3)
        out[out_strides[3]] = s1 - (2 * s3);

        // Output point 5: X(4)
        out[out_strides[4]] = s7 + t1;

        // Output point 6: X(5)
        out[out_strides[5]] = s6 + t0;
    }
    AOCLFFTZ_LOG(DEBUG, global_logger_mode, "Exit");
}

kfft_ register_kernel_r2hc_rfft6avx512(FFTZ_UINT8 precision,
                                       FFTZ_UINT8 direction)
{
    if (direction == FORWARD_FFT_DIR)
    {
        if (precision == DT_FLOAT)
        {
            return r2hc_rfft6avx512_fp32_fwd;
        }
        else if (precision == DT_DOUBLE)
        {
            return r2hc_rfft6avx512_fp64_fwd;
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
            return r2hc_rfft6avx512_fp32_bwd;
        }
        else if (precision == DT_DOUBLE)
        {
            return r2hc_rfft6avx512_fp64_bwd;
        }
        else
        {
            return NULL;
        }
    }
}
