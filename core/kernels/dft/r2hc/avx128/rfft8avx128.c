// Copyright Advanced Micro Devices, Inc.
// SPDX-License-Identifier: BSD-3-Clause

/** @file rfft8avx128.c
 *
 *  @brief Radix-8 r2hc Real-FFT kernel with AVX-128 operations using x86 SIMD
 *  intrinsics
 *
 *  This file contains the DIT radix-8 real-to-halfcomplex implementations using
 *  AVX128 SIMD operations for single-precision and double-precision inputs.
 *
 *  @author Jeevanantham N
 */

#include "core/kernels/kernel.h"
#include "core/kernels/simd_includes/r2hc_simd_avx128.h"

static const ops_cycles_t ops_cnt[NUM_PRECISIONS][NUM_FFT_DIRS] =
                                                    {{{0, 2, 20, 52, 36, 0},
                                                      {0, 4, 22, 52, 39, 1}},
                                                     {{0, 2, 20, 26,  6, 0},
                                                      {0, 4, 22, 26,  6, 1}}};

ops_cycles_t get_ops_cnt_r2hc_rfft8avx128(FFTZ_UINT8 precision,
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

static FFTZ_VOID r2hc_rfft8avx128_fp32_fwd(FFTZ_VOID *in_real,
                                           FFTZ_VOID *in_imag,
                                           FFTZ_VOID *out_real,
                                           FFTZ_VOID *out_imag, FFTZ_INTP n,
                                           aoclfftz_strides_t *strides,
                                           FFTZ_VOID *twd, FFTZ_UINT8 flag)
{
    AOCLFFTZ_LOG(DEBUG, global_logger_mode, "Enter");
    const FFTZ_FLOAT CRTM_8_1 =
        0.7071067811865475244008443621048490392848359377f;

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
    FFTZ_INTP N = n / NUM_SETS_REAL_128_S;

    __m128 v_CRTM_8_1 = _mm_broadcast_ss(&CRTM_8_1);

    for (cnt = 0; cnt < N; cnt++)
    {
        __m128 v_in0, v_in1, v_in2, v_in3, v_in4, v_in5, v_in6, v_in7;
        __m128 v_s0, v_s1, v_s2, v_s3, v_s4, v_s5, v_s6, v_s7, v_s8, v_s9,
               v_s10, v_s11;
        __m128 v_t0, v_t1;
        __m128 v_out0, v_out1, v_out2, v_out3, v_out4, v_out5, v_out6, v_out7;

        curr_in = in;
        curr_out = out;

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

        v_s0 = _mm_add_ps(v_in7, v_in5);
        v_s1 = _mm_sub_ps(v_in7, v_in5);
        v_s2 = _mm_add_ps(v_in6, v_in2);
        v_s3 = _mm_sub_ps(v_in2, v_in6);
        v_s4 = _mm_add_ps(v_in4, v_in0);
        v_s5 = _mm_sub_ps(v_in0, v_in4);
        v_s6 = _mm_add_ps(v_in3, v_in1);
        v_s7 = _mm_sub_ps(v_in3, v_in1);

        v_s8 = _mm_add_ps(v_s6, v_s0);
        v_s9 = _mm_add_ps(v_s4, v_s2);
        v_s10 = _mm_sub_ps(v_s1, v_s7);
        v_s11 = _mm_sub_ps(v_s0, v_s6);

        v_t0 = _mm_mul_ps(v_CRTM_8_1, v_s10);
        v_t1 = _mm_mul_ps(v_CRTM_8_1, v_s11);

        // Output point 1: X(0)
        v_out0 = _mm_add_ps(v_s9, v_s8);

        // Output point 2: X(1)
        v_out1 = _mm_add_ps(v_s5, v_t0);

        // Output point 3: X(2)
        v_out2 = _mm_sub_ps(v_t1, v_s3);

        // Output point 4: X(3)
        v_out3 = _mm_sub_ps(v_s4, v_s2);

        // Output point 5: X(4)
        v_out4 = _mm_add_ps(v_s7, v_s1);

        // Output point 6: X(5)
        v_out5 = _mm_sub_ps(v_s5, v_t0);

        // Output point 7: X(6)
        v_out6 = _mm_add_ps(v_s3, v_t1);

        // Output point 8: X(7)
        v_out7 = _mm_sub_ps(v_s9, v_s8);

        STR_128_S(curr_out, v_out_stride, v_out0);
        curr_out = out + out_strides[1];
        STRI_2x128_S(curr_out, v_out_stride, v_out1, v_out2);
        curr_out = out + out_strides[3];
        STRI_2x128_S(curr_out, v_out_stride, v_out3, v_out4);
        curr_out = out + out_strides[5];
        STRI_2x128_S(curr_out, v_out_stride, v_out5, v_out6);
        curr_out = out + out_strides[7];
        STR_128_S(curr_out, v_out_stride, v_out7);

        in += v_in_stride * NUM_SETS_REAL_128_S;
        out += v_out_stride * NUM_SETS_REAL_128_S;
    }
    // tail cases
    if (n & 2)
    {
        __m128 v_in0, v_in1, v_in2, v_in3, v_in4, v_in5, v_in6, v_in7;
        __m128 v_s0, v_s1, v_s2, v_s3, v_s4, v_s5, v_s6, v_s7, v_s8, v_s9,
               v_s10, v_s11;
        __m128 v_t0, v_t1;
        __m128 v_out0, v_out1, v_out2, v_out3, v_out4, v_out5, v_out6, v_out7;

        curr_in = in;
        curr_out = out;

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

        v_s0 = _mm_add_ps(v_in7, v_in5);
        v_s1 = _mm_sub_ps(v_in7, v_in5);
        v_s2 = _mm_add_ps(v_in6, v_in2);
        v_s3 = _mm_sub_ps(v_in2, v_in6);
        v_s4 = _mm_add_ps(v_in4, v_in0);
        v_s5 = _mm_sub_ps(v_in0, v_in4);
        v_s6 = _mm_add_ps(v_in3, v_in1);
        v_s7 = _mm_sub_ps(v_in3, v_in1);

        v_s8 = _mm_add_ps(v_s6, v_s0);
        v_s9 = _mm_add_ps(v_s4, v_s2);
        v_s10 = _mm_sub_ps(v_s1, v_s7);
        v_s11 = _mm_sub_ps(v_s0, v_s6);

        v_t0 = _mm_mul_ps(v_CRTM_8_1, v_s10);
        v_t1 = _mm_mul_ps(v_CRTM_8_1, v_s11);

        // Output point 1: X(0)
        v_out0 = _mm_add_ps(v_s9, v_s8);

        // Output point 2: X(1)
        v_out1 = _mm_add_ps(v_t0, v_s5);

        // Output point 3: X(2)
        v_out2 = _mm_sub_ps(v_t1, v_s3);

        // Output point 4: X(3)
        v_out3 = _mm_sub_ps(v_s4, v_s2);

        // Output point 5: X(4)
        v_out4 = _mm_add_ps(v_s7, v_s1);

        // Output point 6: X(5)
        v_out5 = _mm_sub_ps(v_s5, v_t0);

        // Output point 7: X(6)
        v_out6 = _mm_add_ps(v_s3, v_t1);

        // Output point 8: X(7)
        v_out7 = _mm_sub_ps(v_s9, v_s8);

        STHR_128_S(curr_out, v_out_stride, v_out0);
        curr_out = out + out_strides[1];
        STHRI_2x128_S(curr_out, v_out_stride, v_out1, v_out2);
        curr_out = out + out_strides[3];
        STHRI_2x128_S(curr_out, v_out_stride, v_out3, v_out4);
        curr_out = out + out_strides[5];
        STHRI_2x128_S(curr_out, v_out_stride, v_out5, v_out6);
        curr_out = out + out_strides[7];
        STHR_128_S(curr_out, v_out_stride, v_out7);

        in = in + (v_in_stride << 1);
        out = out + (v_out_stride << 1);
    }
    // tail cases
    if (n & 1)
    {
        FFTZ_FLOAT in0, in1, in2, in3, in4, in5, in6, in7;
        FFTZ_FLOAT s0, s1, s2, s3, s4, s5, s6, s7, s8, s9, s10, s11;
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
        // Input point 7: x(6)
        in6 = in[in_strides[6]];
        // Input point 8: x(7)
        in7 = in[in_strides[7]];

        s0 = in7 + in5;
        s1 = in7 - in5;
        s2 = in6 + in2;
        s3 = in2 - in6;
        s4 = in4 + in0;
        s5 = in0 - in4;
        s6 = in3 + in1;
        s7 = in3 - in1;

        s8 = s6 + s0;
        s9 = s4 + s2;
        s10 = s1 - s7;
        s11 = s0 - s6;

        t0 = CRTM_8_1 * s10;
        t1 = CRTM_8_1 * s11;

        // Output point 1: X(0)
        *out = s9 + s8;

        // Output point 2: X(1)
        out[out_strides[1]] = s5 + t0;

        // Output point 3: X(2)
        out[out_strides[2]] = t1 - s3;

        // Output point 4: X(3)
        out[out_strides[3]] = s4 - s2;

        // Output point 5: X(4)
        out[out_strides[4]] = s7 + s1;

        // Output point 6: X(5)
        out[out_strides[5]] = s5 - t0;

        // Output point 7: X(6)
        out[out_strides[6]] = s3 + t1;

        // Output point 8: X(7)
        out[out_strides[7]] = s9 - s8;
    }
    AOCLFFTZ_LOG(DEBUG, global_logger_mode, "Exit");
}

static FFTZ_VOID r2hc_rfft8avx128_fp32_bwd(FFTZ_VOID *in_real,
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

    FFTZ_INTP cnt;
    FFTZ_FLOAT *curr_in, *curr_out;
    FFTZ_INTP N = n / NUM_SETS_REAL_128_S;

    __m128 v_CRTM_8_1 = _mm_broadcast_ss(&CRTM_8_1);
    __m128 v_CRTM_8_2 = _mm_broadcast_ss(&CRTM_8_2);

    for (cnt = 0; cnt < N; cnt++)
    {
        __m128 v_in0, v_in1, v_in2, v_in3, v_in4, v_in5, v_in6, v_in7;
        __m128 v_s0, v_s1, v_s2, v_s3, v_s4, v_s5, v_s6, v_s7, v_s8, v_s9;
        __m128 v_t0, v_t1, v_t2, v_t3;
        __m128 v_out0, v_out1, v_out2, v_out3, v_out4, v_out5, v_out6, v_out7;

        curr_in = in;
        curr_out = out;

        // Input point 1: x(0)
        LDR_128_S(curr_in, v_in_stride, v_in0);
        // Input point 2: x(1) & Input point 3: x(2)
        curr_in = in + in_strides[1];
        LDRI_2x128_S(curr_in, v_in_stride, v_in1, v_in2);
        // Input point 4: x(3) & Input point 5: x(4)
        curr_in = in + in_strides[3];
        LDRI_2x128_S(curr_in, v_in_stride, v_in3, v_in4);
        // Input point 6: x(5) & Input point 7: x(6)
        curr_in = in + in_strides[5];
        LDRI_2x128_S(curr_in, v_in_stride, v_in5, v_in6);
        // Input point 8: x(7)
        curr_in = in + in_strides[7];
        LDR_128_S(curr_in, v_in_stride, v_in7);

        v_s0 = _mm_add_ps(v_in7, v_in0);
        v_s1 = _mm_sub_ps(v_in7, v_in0);
        v_s2 = _mm_add_ps(v_in6, v_in2);
        v_t0 = _mm_mul_ps(v_CRTM_8_2, _mm_sub_ps(v_in2, v_in6));
        v_t1 = _mm_mul_ps(v_CRTM_8_2, _mm_add_ps(v_in5, v_in1));
        v_s3 = _mm_sub_ps(v_in5, v_in1);
        v_s4 = _mm_add_ps(v_in4, v_in4);
        v_s5 = _mm_add_ps(v_in3, v_in3);

        v_t2 = _mm_mul_ps(v_CRTM_8_1, _mm_add_ps(v_s3, v_s2));
        v_t3 = _mm_mul_ps(v_CRTM_8_1, _mm_sub_ps(v_s3, v_s2));
        v_s6 = _mm_add_ps(v_s5, v_s0);
        v_s7 = _mm_sub_ps(v_s0, v_s5);
        v_s8 = _mm_add_ps(v_s4, v_s1);
        v_s9 = _mm_sub_ps(v_s4, v_s1);

        // Output point 1: X(0)
        v_out0 = _mm_add_ps(v_s6, v_t1);

        // Output point 2: X(1)
        v_out1 = NEGATE_128_S(_mm_add_ps(v_s8, v_t2));

        // Output point 3: X(2)
        v_out2 = _mm_sub_ps(v_s7, v_t0);

        // Output point 4: X(3)
        v_out3 = _mm_add_ps(v_s9, v_t3);

        // Output point 5: X(4)
        v_out4 = _mm_sub_ps(v_s6, v_t1);

        // Output point 6: X(5)
        v_out5 = _mm_sub_ps(v_t2, v_s8);

        // Output point 7: X(6)
        v_out6 = _mm_add_ps(v_s7, v_t0);

        // Output point 8: X(7)
        v_out7 = _mm_sub_ps(v_s9, v_t3);

        STR_128_S(curr_out, v_out_stride, v_out0);
        curr_out = out + out_strides[1];
        STR_128_S(curr_out, v_out_stride, v_out1);
        curr_out = out + out_strides[2];
        STR_128_S(curr_out, v_out_stride, v_out2);
        curr_out = out + out_strides[3];
        STR_128_S(curr_out, v_out_stride, v_out3);
        curr_out = out + out_strides[4];
        STR_128_S(curr_out, v_out_stride, v_out4);
        curr_out = out + out_strides[5];
        STR_128_S(curr_out, v_out_stride, v_out5);
        curr_out = out + out_strides[6];
        STR_128_S(curr_out, v_out_stride, v_out6);
        curr_out = out + out_strides[7];
        STR_128_S(curr_out, v_out_stride, v_out7);

        in += v_in_stride * NUM_SETS_REAL_128_S;
        out += v_out_stride * NUM_SETS_REAL_128_S;
    }
    // tail cases
    if (n & 2)
    {
        __m128 v_in0, v_in1, v_in2, v_in3, v_in4, v_in5, v_in6, v_in7;
        __m128 v_s0, v_s1, v_s2, v_s3, v_s4, v_s5, v_s6, v_s7, v_s8, v_s9;
        __m128 v_t0, v_t1, v_t2, v_t3;
        __m128 v_out0, v_out1, v_out2, v_out3, v_out4, v_out5, v_out6, v_out7;

        curr_in = in;
        curr_out = out;

        // Input point 1: x(0)
        LDHR_128_S(curr_in, v_in_stride, v_in0);
        // Input point 2: x(1) & Input point 3: x(2)
        curr_in = in + in_strides[1];
        LDHRI_2x128_S(curr_in, v_in_stride, v_in1, v_in2);
        // Input point 4: x(3) & Input point 5: x(4)
        curr_in = in + in_strides[3];
        LDHRI_2x128_S(curr_in, v_in_stride, v_in3, v_in4);
        // Input point 6: x(5) & Input point 7: x(6)
        curr_in = in + in_strides[5];
        LDHRI_2x128_S(curr_in, v_in_stride, v_in5, v_in6);
        // Input point 8: x(7)
        curr_in = in + in_strides[7];
        LDHR_128_S(curr_in, v_in_stride, v_in7);

        v_s0 = _mm_add_ps(v_in7, v_in0);
        v_s1 = _mm_sub_ps(v_in7, v_in0);
        v_s2 = _mm_add_ps(v_in6, v_in2);
        v_t0 = _mm_mul_ps(v_CRTM_8_2, _mm_sub_ps(v_in2, v_in6));
        v_t1 = _mm_mul_ps(v_CRTM_8_2, _mm_add_ps(v_in5, v_in1));
        v_s3 = _mm_sub_ps(v_in5, v_in1);
        v_s4 = _mm_add_ps(v_in4, v_in4);
        v_s5 = _mm_add_ps(v_in3, v_in3);

        v_t2 = _mm_mul_ps(v_CRTM_8_1, _mm_add_ps(v_s3, v_s2));
        v_t3 = _mm_mul_ps(v_CRTM_8_1, _mm_sub_ps(v_s3, v_s2));
        v_s6 = _mm_add_ps(v_s5, v_s0);
        v_s7 = _mm_sub_ps(v_s0, v_s5);
        v_s8 = _mm_add_ps(v_s4, v_s1);
        v_s9 = _mm_sub_ps(v_s4, v_s1);

        // Output point 1: X(0)
        v_out0 = _mm_add_ps(v_s6, v_t1);

        // Output point 2: X(1)
        v_out1 = NEGATE_128_S(_mm_add_ps(v_s8, v_t2));

        // Output point 3: X(2)
        v_out2 = _mm_sub_ps(v_s7, v_t0);

        // Output point 4: X(3)
        v_out3 = _mm_add_ps(v_s9, v_t3);

        // Output point 5: X(4)
        v_out4 = _mm_sub_ps(v_s6, v_t1);

        // Output point 6: X(5)
        v_out5 = _mm_sub_ps(v_t2, v_s8);

        // Output point 7: X(6)
        v_out6 = _mm_add_ps(v_s7, v_t0);

        // Output point 8: X(7)
        v_out7 = _mm_sub_ps(v_s9, v_t3);

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
        curr_out = out + out_strides[6];
        STHR_128_S(curr_out, v_out_stride, v_out6);
        curr_out = out + out_strides[7];
        STHR_128_S(curr_out, v_out_stride, v_out7);

        in = in + (v_in_stride << 1);
        out = out + (v_out_stride << 1);
    }
    // tail cases
    if (n & 1)
    {
        FFTZ_FLOAT in0, in1, in2, in3, in4, in5, in6, in7;
        FFTZ_FLOAT s0, s1, s2, s3, s4, s5, s6, s7, s8, s9;
        FFTZ_FLOAT t0, t1, t2, t3;

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

        s0 = in7 + in0;
        s1 = in7 - in0;
        s2 = in6 + in2;
        t0 = CRTM_8_2 * (in6 - in2);
        t1 = CRTM_8_2 * (in5 + in1);
        s3 = in5 - in1;
        s4 = in4 + in4;
        s5 = in3 + in3;

        t2 = CRTM_8_1 * (s3 + s2);
        t3 = CRTM_8_1 * (s3 - s2);
        s6 = s5 + s0;
        s7 = s5 - s0;
        s8 = s4 + s1;
        s9 = s4 - s1;

        // Output point 1: X(0)
        *out = s6 + t1;

        // Output point 2: X(1)
        out[out_strides[1]] = -s8 - t2;

        // Output point 3: X(2)
        out[out_strides[2]] = -s7 + t0;

        // Output point 4: X(3)
        out[out_strides[3]] = s9 + t3;

        // Output point 5: X(4)
        out[out_strides[4]] = s6 - t1;

        // Output point 6: X(5)
        out[out_strides[5]] = t2 - s8;

        // Output point 7: X(6)
        out[out_strides[6]] = -s7 - t0;

        // Output point 8: X(7)
        out[out_strides[7]] = s9 - t3;
    }
    AOCLFFTZ_LOG(DEBUG, global_logger_mode, "Exit");
}

static FFTZ_VOID r2hc_rfft8avx128_fp64_fwd(FFTZ_VOID *in_real,
                                           FFTZ_VOID *in_imag,
                                           FFTZ_VOID *out_real,
                                           FFTZ_VOID *out_imag, FFTZ_INTP n,
                                           aoclfftz_strides_t *strides,
                                           FFTZ_VOID *twd, FFTZ_UINT8 flag)
{
    AOCLFFTZ_LOG(DEBUG, global_logger_mode, "Enter");
    const FFTZ_DOUBLE CRTM_8_1 =
        0.7071067811865475244008443621048490392848359377;

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
    FFTZ_INTP N = n / NUM_SETS_REAL_128_D;

    __m128d v_CRTM_8_1 = _mm_set1_pd(CRTM_8_1);

    for (cnt = 0; cnt < N; cnt++)
    {
        __m128d v_in0, v_in1, v_in2, v_in3, v_in4, v_in5, v_in6, v_in7;
        __m128d v_s0, v_s1, v_s2, v_s3, v_s4, v_s5, v_s6, v_s7, v_s8, v_s9,
                v_s10, v_s11;
        __m128d v_t0, v_t1;
        __m128d v_out0, v_out1, v_out2, v_out3, v_out4, v_out5, v_out6, v_out7;

        curr_in = in;
        curr_out = out;

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

        v_s0 = _mm_add_pd(v_in7, v_in5);
        v_s1 = _mm_sub_pd(v_in7, v_in5);
        v_s2 = _mm_add_pd(v_in6, v_in2);
        v_s3 = _mm_sub_pd(v_in2, v_in6);
        v_s4 = _mm_add_pd(v_in4, v_in0);
        v_s5 = _mm_sub_pd(v_in0, v_in4);
        v_s6 = _mm_add_pd(v_in3, v_in1);
        v_s7 = _mm_sub_pd(v_in3, v_in1);

        v_s8 = _mm_add_pd(v_s6, v_s0);
        v_s9 = _mm_add_pd(v_s4, v_s2);
        v_s10 = _mm_sub_pd(v_s1, v_s7);
        v_s11 = _mm_sub_pd(v_s0, v_s6);

        v_t0 = _mm_mul_pd(v_CRTM_8_1, v_s10);
        v_t1 = _mm_mul_pd(v_CRTM_8_1, v_s11);

        // Output point 1: X(0)
        v_out0 = _mm_add_pd(v_s9, v_s8);

        // Output point 2: X(1)
        v_out1 = _mm_add_pd(v_t0, v_s5);

        // Output point 3: X(2)
        v_out2 = _mm_sub_pd(v_t1, v_s3);

        // Output point 4: X(3)
        v_out3 = _mm_sub_pd(v_s4, v_s2);

        // Output point 5: X(4)
        v_out4 = _mm_add_pd(v_s7, v_s1);

        // Output point 6: X(5)
        v_out5 = _mm_sub_pd(v_s5, v_t0);

        // Output point 7: X(6)
        v_out6 = _mm_add_pd(v_s3, v_t1);

        // Output point 8: X(7)
        v_out7 = _mm_sub_pd(v_s9, v_s8);

        STR_128_D(curr_out, v_out_stride, v_out0);
        curr_out = out + out_strides[1];
        STRI_2x128_D(curr_out, v_out_stride, v_out1, v_out2);
        curr_out = out + out_strides[3];
        STRI_2x128_D(curr_out, v_out_stride, v_out3, v_out4);
        curr_out = out + out_strides[5];
        STRI_2x128_D(curr_out, v_out_stride, v_out5, v_out6);
        curr_out = out + out_strides[7];
        STR_128_D(curr_out, v_out_stride, v_out7);

        in += v_in_stride * NUM_SETS_REAL_128_D;
        out += v_out_stride * NUM_SETS_REAL_128_D;
    }
    // tail cases
    if (n & 1)
    {
        FFTZ_DOUBLE in0, in1, in2, in3, in4, in5, in6, in7;
        FFTZ_DOUBLE s0, s1, s2, s3, s4, s5, s6, s7, s8, s9, s10, s11;
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
        // Input point 7: x(6)
        in6 = in[in_strides[6]];
        // Input point 8: x(7)
        in7 = in[in_strides[7]];

        s0 = in7 + in5;
        s1 = in7 - in5;
        s2 = in6 + in2;
        s3 = in2 - in6;
        s4 = in4 + in0;
        s5 = in0 - in4;
        s6 = in3 + in1;
        s7 = in3 - in1;

        s8 = s6 + s0;
        s9 = s4 + s2;
        s10 = s1 - s7;
        s11 = s0 - s6;

        t0 = CRTM_8_1 * s10;
        t1 = CRTM_8_1 * s11;

        // Output point 1: X(0)
        *out = s9 + s8;

        // Output point 2: X(1)
        out[out_strides[1]] = s5 + t0;

        // Output point 3: X(2)
        out[out_strides[2]] = t1 - s3;

        // Output point 4: X(3)
        out[out_strides[3]] = s4 - s2;

        // Output point 5: X(4)
        out[out_strides[4]] = s7 + s1;

        // Output point 6: X(5)
        out[out_strides[5]] = s5 - t0;

        // Output point 7: X(6)
        out[out_strides[6]] = s3 + t1;

        // Output point 8: X(7)
        out[out_strides[7]] = s9 - s8;
    }
    AOCLFFTZ_LOG(DEBUG, global_logger_mode, "Exit");
}

static FFTZ_VOID r2hc_rfft8avx128_fp64_bwd(FFTZ_VOID *in_real,
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

    FFTZ_INTP cnt;
    FFTZ_DOUBLE *curr_in, *curr_out;
    FFTZ_INTP N = n / NUM_SETS_REAL_128_D;

    __m128d v_CRTM_8_1 = _mm_set1_pd(CRTM_8_1);
    __m128d v_CRTM_8_2 = _mm_set1_pd(CRTM_8_2);

    for (cnt = 0; cnt < N; cnt++)
    {
        __m128d v_in0, v_in1, v_in2, v_in3, v_in4, v_in5, v_in6, v_in7;
        __m128d v_s0, v_s1, v_s2, v_s3, v_s4, v_s5, v_s6, v_s7, v_s8, v_s9;
        __m128d v_t0, v_t1, v_t2, v_t3;
        __m128d v_out0, v_out1, v_out2, v_out3, v_out4, v_out5, v_out6, v_out7;

        curr_in = in;
        curr_out = out;

        // Input point 1: x(0)
        LDR_128_D(curr_in, v_in_stride, v_in0);
        // Input point 2: x(1) & Input point 3: x(2)
        curr_in = in + in_strides[1];
        LDRI_2x128_D(curr_in, v_in_stride, v_in1, v_in2);
        // Input point 4: x(3) & Input point 5: x(4)
        curr_in = in + in_strides[3];
        LDRI_2x128_D(curr_in, v_in_stride, v_in3, v_in4);
        // Input point 6: x(5) & Input point 7: x(6)
        curr_in = in + in_strides[5];
        LDRI_2x128_D(curr_in, v_in_stride, v_in5, v_in6);
        // Input point 8: x(7)
        curr_in = in + in_strides[7];
        LDR_128_D(curr_in, v_in_stride, v_in7);

        v_s0 = _mm_add_pd(v_in7, v_in0);
        v_s1 = _mm_sub_pd(v_in7, v_in0);
        v_s2 = _mm_add_pd(v_in6, v_in2);
        v_t0 = _mm_mul_pd(v_CRTM_8_2, _mm_sub_pd(v_in2, v_in6));
        v_t1 = _mm_mul_pd(v_CRTM_8_2, _mm_add_pd(v_in5, v_in1));
        v_s3 = _mm_sub_pd(v_in5, v_in1);
        v_s4 = _mm_add_pd(v_in4, v_in4);
        v_s5 = _mm_add_pd(v_in3, v_in3);

        v_t2 = _mm_mul_pd(v_CRTM_8_1, _mm_add_pd(v_s3, v_s2));
        v_t3 = _mm_mul_pd(v_CRTM_8_1, _mm_sub_pd(v_s3, v_s2));
        v_s6 = _mm_add_pd(v_s5, v_s0);
        v_s7 = _mm_sub_pd(v_s0, v_s5);
        v_s8 = _mm_add_pd(v_s4, v_s1);
        v_s9 = _mm_sub_pd(v_s4, v_s1);

        // Output point 1: X(0)
        v_out0 = _mm_add_pd(v_s6, v_t1);

        // Output point 2: X(1)
        v_out1 = NEGATE_128_D(_mm_add_pd(v_s8, v_t2));

        // Output point 3: X(2)
        v_out2 = _mm_sub_pd(v_s7, v_t0);

        // Output point 4: X(3)
        v_out3 = _mm_add_pd(v_s9, v_t3);

        // Output point 5: X(4)
        v_out4 = _mm_sub_pd(v_s6, v_t1);

        // Output point 6: X(5)
        v_out5 = _mm_sub_pd(v_t2, v_s8);

        // Output point 7: X(6)
        v_out6 = _mm_add_pd(v_s7, v_t0);

        // Output point 8: X(7)
        v_out7 = _mm_sub_pd(v_s9, v_t3);

        STR_128_D(curr_out, v_out_stride, v_out0);
        curr_out = out + out_strides[1];
        STR_128_D(curr_out, v_out_stride, v_out1);
        curr_out = out + out_strides[2];
        STR_128_D(curr_out, v_out_stride, v_out2);
        curr_out = out + out_strides[3];
        STR_128_D(curr_out, v_out_stride, v_out3);
        curr_out = out + out_strides[4];
        STR_128_D(curr_out, v_out_stride, v_out4);
        curr_out = out + out_strides[5];
        STR_128_D(curr_out, v_out_stride, v_out5);
        curr_out = out + out_strides[6];
        STR_128_D(curr_out, v_out_stride, v_out6);
        curr_out = out + out_strides[7];
        STR_128_D(curr_out, v_out_stride, v_out7);

        in += v_in_stride * NUM_SETS_REAL_128_D;
        out += v_out_stride * NUM_SETS_REAL_128_D;
    }
    // tail cases
    if (n & 1)
    {
        FFTZ_DOUBLE in0, in1, in2, in3, in4, in5, in6, in7;
        FFTZ_DOUBLE s0, s1, s2, s3, s4, s5, s6, s7, s8, s9;
        FFTZ_DOUBLE t0, t1, t2, t3;

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

        s0 = in7 + in0;
        s1 = in7 - in0;
        s2 = in6 + in2;
        t0 = CRTM_8_2 * (in6 - in2);
        t1 = CRTM_8_2 * (in5 + in1);
        s3 = in5 - in1;
        s4 = in4 + in4;
        s5 = in3 + in3;

        t2 = CRTM_8_1 * (s3 + s2);
        t3 = CRTM_8_1 * (s3 - s2);
        s6 = s5 + s0;
        s7 = s5 - s0;
        s8 = s4 + s1;
        s9 = s4 - s1;

        // Output point 1: X(0)
        *out = s6 + t1;

        // Output point 2: X(1)
        out[out_strides[1]] = -s8 - t2;

        // Output point 3: X(2)
        out[out_strides[2]] = -s7 + t0;

        // Output point 4: X(3)
        out[out_strides[3]] = s9 + t3;

        // Output point 5: X(4)
        out[out_strides[4]] = s6 - t1;

        // Output point 6: X(5)
        out[out_strides[5]] = t2 - s8;

        // Output point 7: X(6)
        out[out_strides[6]] = -s7 - t0;

        // Output point 8: X(7)
        out[out_strides[7]] = s9 - t3;
    }
    AOCLFFTZ_LOG(DEBUG, global_logger_mode, "Exit");
}

kfft_ register_kernel_r2hc_rfft8avx128(FFTZ_UINT8 precision,
                                       FFTZ_UINT8 direction)
{
    if (direction == FORWARD_FFT_DIR)
    {
        if (precision == DT_FLOAT)
        {
            return r2hc_rfft8avx128_fp32_fwd;
        }
        else if (precision == DT_DOUBLE)
        {
            return r2hc_rfft8avx128_fp64_fwd;
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
            return r2hc_rfft8avx128_fp32_bwd;
        }
        else if (precision == DT_DOUBLE)
        {
            return r2hc_rfft8avx128_fp64_bwd;
        }
        else
        {
            return NULL;
        }
    }
}
