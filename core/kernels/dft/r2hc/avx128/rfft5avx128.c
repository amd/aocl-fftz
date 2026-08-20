// Copyright Advanced Micro Devices, Inc.
// SPDX-License-Identifier: BSD-3-Clause

/** @file rfft5avx128.c
 *
 *  @brief Radix-5 r2hc Real-FFT kernel with AVX-128 operations using x86 SIMD
 *  intrinsics
 *
 *  This file contains the DIT radix-5 real-to-halfcomplex implementations using
 *  AVX128 SIMD operations for single-precision and double-precision inputs.
 *
 *  @author Jeya R
 */

#include "core/kernels/kernel.h"
#include "core/kernels/simd_includes/r2hc_simd_avx128.h"

static const ops_cycles_t ops_cnt[NUM_PRECISIONS][NUM_FFT_DIRS] =
                                                    {{{0, 6, 12, 32, 22, 0},
                                                      {0, 7, 12, 32, 24, 0}},
                                                     {{0, 6, 12, 16, 4, 0},
                                                      {0, 7, 12, 16, 4, 0}}};

ops_cycles_t get_ops_cnt_r2hc_rfft5avx128(FFTZ_UINT8 precision,
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

static FFTZ_VOID r2hc_rfft5avx128_fp32_fwd(FFTZ_VOID *in_real,
                                           FFTZ_VOID *in_imag,
                                           FFTZ_VOID *out_real,
                                           FFTZ_VOID *out_imag, FFTZ_INTP n,
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

    FFTZ_FLOAT *in = (FFTZ_FLOAT *)in_real;
    FFTZ_FLOAT *out = (FFTZ_FLOAT *)out_real;
#ifdef VOLATILE_STRIDE_ARRAY
    volatile FFTZ_INTP *in_strides  = strides->in_strides;
    volatile FFTZ_INTP *out_strides = strides->out_strides;
#else
    FFTZ_INTP *in_strides  = strides->in_strides;
    FFTZ_INTP *out_strides = strides->out_strides;
#endif
    FFTZ_INTP v_in_stride  = strides->v_in_stride;
    FFTZ_INTP v_out_stride = strides->v_out_stride;

    FFTZ_INTP cnt;
    FFTZ_FLOAT *curr_in, *curr_out;
    FFTZ_INTP N = n / NUM_SETS_REAL_128_S;

    __m128 v_CRTM_5_1 = _mm_broadcast_ss(&CRTM_5_1);
    __m128 v_CRTM_5_2 = _mm_broadcast_ss(&CRTM_5_2);
    __m128 v_CRTM_5_3 = _mm_broadcast_ss(&CRTM_5_3);
    __m128 v_CRTM_5_4 = _mm_broadcast_ss(&CRTM_5_4);

    for (cnt = 0; cnt < N; cnt++)
    {
        __m128 v_in0, v_in1, v_in2, v_in3, v_in4;
        __m128 v_s1, v_s2, v_s3, v_s4, v_s5, v_t1, v_s6;
        __m128 v_out0, v_out1, v_out2, v_out3, v_out4;

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

        v_s1 = _mm_add_ps(v_in1, v_in4);
        v_s2 = _mm_sub_ps(v_in4, v_in1);
        v_s3 = _mm_add_ps(v_in2, v_in3);
        v_s4 = _mm_sub_ps(v_in2, v_in3);
        v_s6 = _mm_add_ps(v_s1, v_s3);

        v_s5 = _mm_sub_ps(v_in0, _mm_mul_ps(v_CRTM_5_4, v_s6));
        v_t1 = _mm_mul_ps(v_CRTM_5_1, _mm_sub_ps(v_s1, v_s3));

        // Output point 1: X(0)
        v_out0 = _mm_add_ps(v_in0, v_s6);

        // Output point 2: X(1)
        v_out1 = _mm_add_ps(v_s5, v_t1);

        // Output point 3: X(2)
        v_out2 = _mm_sub_ps(_mm_mul_ps(v_CRTM_5_2, v_s2),
                            _mm_mul_ps(v_CRTM_5_3, v_s4));

        // Output point 4: X(3)
        v_out3 = _mm_sub_ps(v_s5, v_t1);

        // Output point 5: X(4)
        v_out4 = _mm_add_ps(_mm_mul_ps(v_CRTM_5_2, v_s4),
                            _mm_mul_ps(v_CRTM_5_3, v_s2));

        STR_128_S(curr_out, v_out_stride, v_out0);
        curr_out = out + out_strides[1];
        STRI_2x128_S(curr_out, v_out_stride, v_out1, v_out2);
        curr_out = out + out_strides[3];
        STRI_2x128_S(curr_out, v_out_stride, v_out3, v_out4);

        in += v_in_stride * NUM_SETS_REAL_128_S;
        out += v_out_stride * NUM_SETS_REAL_128_S;
    }
    // tail cases
    if (n & 2)
    {
        __m128 v_in0, v_in1, v_in2, v_in3, v_in4;
        __m128 v_s1, v_s2, v_s3, v_s4, v_s5, v_t1, v_s6;
        __m128 v_out0, v_out1, v_out2, v_out3, v_out4;

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

        v_s1 = _mm_add_ps(v_in1, v_in4);
        v_s2 = _mm_sub_ps(v_in4, v_in1);
        v_s3 = _mm_add_ps(v_in2, v_in3);
        v_s4 = _mm_sub_ps(v_in2, v_in3);
        v_s6 = _mm_add_ps(v_s1, v_s3);

        v_s5 = _mm_sub_ps(v_in0, _mm_mul_ps(v_CRTM_5_4, v_s6));
        v_t1 = _mm_mul_ps(v_CRTM_5_1, _mm_sub_ps(v_s1, v_s3));

        // Output point 1: X(0)
        v_out0 = _mm_add_ps(v_in0, v_s6);

        // Output point 2: X(1)
        v_out1 = _mm_add_ps(v_s5, v_t1);

        // Output point 3: X(2)
        v_out2 = _mm_sub_ps(_mm_mul_ps(v_CRTM_5_2, v_s2),
                            _mm_mul_ps(v_CRTM_5_3, v_s4));

        // Output point 4: X(3)
        v_out3 = _mm_sub_ps(v_s5, v_t1);

        // Output point 5: X(4)
        v_out4 = _mm_add_ps(_mm_mul_ps(v_CRTM_5_2, v_s4),
                            _mm_mul_ps(v_CRTM_5_3, v_s2));

        STHR_128_S(curr_out, v_out_stride, v_out0);
        curr_out = out + out_strides[1];
        STHRI_2x128_S(curr_out, v_out_stride, v_out1, v_out2);
        curr_out = out + out_strides[3];
        STHRI_2x128_S(curr_out, v_out_stride, v_out3, v_out4);

        in = in + (v_in_stride << 1);
        out = out + (v_out_stride << 1);
    }
    // tail cases
    if (n & 1)
    {
        FFTZ_FLOAT in0, in1, in2, in3, in4;
        FFTZ_FLOAT s1, s2, s3, s4, s5, s6, t1;

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

        s1 = in1 + in4;
        s2 = in4 - in1;
        s3 = in2 + in3;
        s4 = in2 - in3;
        s6 = s1 + s3;

        s5 = in0 - (CRTM_5_4 * s6);
        t1 = CRTM_5_1 * (s1 - s3);

        // Output point 1: X(0)
        *out = in0 + s6;
        // Output point 2: X(1)
        out[out_strides[1]] = s5 + t1;
        // Output point 3: X(2)
        out[out_strides[2]] = (CRTM_5_2 * s2) - (CRTM_5_3 * s4);
        // Output point 4: X(3)
        out[out_strides[3]] = s5 - t1;
        // Output point 5: X(4)
        out[out_strides[4]] = (CRTM_5_2 * s4) + (CRTM_5_3 * s2);
    }
    AOCLFFTZ_LOG(DEBUG, global_logger_mode, "Exit");
}

static FFTZ_VOID r2hc_rfft5avx128_fp32_bwd(FFTZ_VOID *in_real,
                                           FFTZ_VOID *in_imag,
                                           FFTZ_VOID *out_real,
                                           FFTZ_VOID *out_imag, FFTZ_INTP n,
                                           aoclfftz_strides_t *strides,
                                           FFTZ_VOID *twd, FFTZ_UINT8 flag)
{
    AOCLFFTZ_LOG(DEBUG, global_logger_mode, "Enter");
    const FFTZ_FLOAT CRTM_5_1 = 1.11803398874989484820458683436563811772030918f;
    const FFTZ_FLOAT CRTM_5_2 = 1.90211303259030714423287866675876428681139726f;
    const FFTZ_FLOAT CRTM_5_3 = 1.17557050458494625833741190927814553719530488f;
    const FFTZ_FLOAT CRTM_5_4 = 0.50000000000000000000000000000000000000000000f;
    const FFTZ_FLOAT CRTM_5_5 = 2.00000000000000000000000000000000000000000000f;

    FFTZ_FLOAT *in = (FFTZ_FLOAT *)in_real;
    FFTZ_FLOAT *out = (FFTZ_FLOAT *)out_real;
#ifdef VOLATILE_STRIDE_ARRAY
    volatile FFTZ_INTP *in_strides  = strides->in_strides;
    volatile FFTZ_INTP *out_strides = strides->out_strides;
#else
    FFTZ_INTP *in_strides  = strides->in_strides;
    FFTZ_INTP *out_strides = strides->out_strides;
#endif
    FFTZ_INTP v_in_stride  = strides->v_in_stride;
    FFTZ_INTP v_out_stride = strides->v_out_stride;

    FFTZ_INTP cnt;
    FFTZ_FLOAT *curr_in, *curr_out;
    FFTZ_INTP N = n / NUM_SETS_REAL_128_S;

    __m128 v_CRTM_5_1 = _mm_broadcast_ss(&CRTM_5_1);
    __m128 v_CRTM_5_2 = _mm_broadcast_ss(&CRTM_5_2);
    __m128 v_CRTM_5_3 = _mm_broadcast_ss(&CRTM_5_3);
    __m128 v_CRTM_5_4 = _mm_broadcast_ss(&CRTM_5_4);
    __m128 v_CRTM_5_5 = _mm_broadcast_ss(&CRTM_5_5);

    for (cnt = 0; cnt < N; cnt++)
    {
        __m128 v_in0, v_in1, v_in2, v_in3, v_in4;
        __m128 v_s1, v_s2, v_s3, v_s4, v_s5, v_t1, v_s6;
        __m128 v_out0, v_out1, v_out2, v_out3, v_out4;

        curr_in = in;
        curr_out = out;

        // Input point 1: x(0)
        LDR_128_S(curr_in, v_in_stride, v_in0);
        // Input point 2: x(1) & x(2)
        curr_in = in + in_strides[1];
        LDRI_2x128_S(curr_in, v_in_stride, v_in1, v_in2);
        // Input point 4: x(3) & x(4)
        curr_in = in + in_strides[3];
        LDRI_2x128_S(curr_in, v_in_stride, v_in3, v_in4);

        v_s6 = _mm_add_ps(v_in1, v_in3);
        v_t1 = _mm_mul_ps(v_CRTM_5_1, _mm_sub_ps(v_in1, v_in3));
        v_s5 = _mm_sub_ps(v_in0, _mm_mul_ps(v_CRTM_5_4, v_s6));
        v_s1 = _mm_add_ps(v_s5, v_t1);
        v_s2 = _mm_sub_ps(v_s5, v_t1);
        v_s3 = _mm_add_ps(_mm_mul_ps(v_CRTM_5_2, v_in2),
                          _mm_mul_ps(v_CRTM_5_3, v_in4));
        v_s4 = _mm_sub_ps(_mm_mul_ps(v_CRTM_5_2, v_in4),
                          _mm_mul_ps(v_CRTM_5_3, v_in2));

        // Output point 1: X(0)
        v_out0 = _mm_add_ps(v_in0, _mm_mul_ps(v_CRTM_5_5, v_s6));

        // Output point 2: X(1)
        v_out1 = _mm_sub_ps(v_s1, v_s3);

        // Output point 3: X(2)
        v_out2 = _mm_add_ps(v_s2, v_s4);

        // Output point 4: X(3)
        v_out3 = _mm_sub_ps(v_s2, v_s4);

        // Output point 5: X(4)
        v_out4 = _mm_add_ps(v_s1, v_s3);

        STR_128_S(curr_out, v_out_stride, v_out0);
        curr_out = out + out_strides[1];
        STR_128_S(curr_out, v_out_stride, v_out1);
        curr_out = out + out_strides[2];
        STR_128_S(curr_out, v_out_stride, v_out2);
        curr_out = out + out_strides[3];
        STR_128_S(curr_out, v_out_stride, v_out3);
        curr_out = out + out_strides[4];
        STR_128_S(curr_out, v_out_stride, v_out4);

        in += v_in_stride * NUM_SETS_REAL_128_S;
        out += v_out_stride * NUM_SETS_REAL_128_S;
    }
    // tail cases
    if (n & 2)
    {
        __m128 v_in0, v_in1, v_in2, v_in3, v_in4;
        __m128 v_s1, v_s2, v_s3, v_s4, v_s5, v_t1, v_s6;
        __m128 v_out0, v_out1, v_out2, v_out3, v_out4;

        curr_in = in;
        curr_out = out;

        // Input point 1: x(0)
        LDHR_128_S(curr_in, v_in_stride, v_in0);
        // Input point 2: x(1) & x(2)
        curr_in = in + in_strides[1];
        LDHRI_2x128_S(curr_in, v_in_stride, v_in1, v_in2);
        // Input point 4: x(3) & x(4)
        curr_in = in + in_strides[3];
        LDHRI_2x128_S(curr_in, v_in_stride, v_in3, v_in4);

        v_s6 = _mm_add_ps(v_in1, v_in3);
        v_t1 = _mm_mul_ps(v_CRTM_5_1, _mm_sub_ps(v_in1, v_in3));
        v_s5 = _mm_sub_ps(v_in0, _mm_mul_ps(v_CRTM_5_4, v_s6));
        v_s1 = _mm_add_ps(v_s5, v_t1);
        v_s2 = _mm_sub_ps(v_s5, v_t1);
        v_s3 = _mm_add_ps(_mm_mul_ps(v_CRTM_5_2, v_in2),
                          _mm_mul_ps(v_CRTM_5_3, v_in4));
        v_s4 = _mm_sub_ps(_mm_mul_ps(v_CRTM_5_2, v_in4),
                          _mm_mul_ps(v_CRTM_5_3, v_in2));

        // Output point 1: X(0)
        v_out0 = _mm_add_ps(v_in0, _mm_mul_ps(v_CRTM_5_5, v_s6));

        // Output point 2: X(1)
        v_out1 = _mm_sub_ps(v_s1, v_s3);

        // Output point 3: X(2)
        v_out2 = _mm_add_ps(v_s2, v_s4);

        // Output point 4: X(3)
        v_out3 = _mm_sub_ps(v_s2, v_s4);

        // Output point 5: X(4)
        v_out4 = _mm_add_ps(v_s1, v_s3);

        STHR_128_S(curr_out, v_out_stride, v_out0);
        curr_out = out + out_strides[1];
        STHR_128_S(curr_out, v_out_stride, v_out1);
        curr_out = out + out_strides[2];
        STHR_128_S(curr_out, v_out_stride, v_out2);
        curr_out = out + out_strides[3];
        STHR_128_S(curr_out, v_out_stride, v_out3);
        curr_out = out + out_strides[4];
        STHR_128_S(curr_out, v_out_stride, v_out4);

        in = in + (v_in_stride << 1);
        out = out + (v_out_stride << 1);
    }
    // tail cases
    if (n & 1)
    {
        FFTZ_DOUBLE in0, in1, in2, in3, in4;
        FFTZ_DOUBLE s1, s2, s3, s4, s5, s6, t1;

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

        s6 = in1 + in3;
        t1 = CRTM_5_1 * (in1 - in3);
        s5 = in0 - (CRTM_5_4 * s6);
        s1 = s5 + t1;
        s2 = s5 - t1;
        s3 = (CRTM_5_2 * in2) + (CRTM_5_3 * in4);
        s4 = (CRTM_5_2 * in4) - (CRTM_5_3 * in2);

        // Output point 1: X(0)
        *out = in0 + CRTM_5_5 * s6;

        // Output point 2: X(1)
        out[out_strides[1]] = s1 - s3;

        // Output point 3: X(2)
        out[out_strides[2]] = s2 + s4;

        // Output point 4: X(3)
        out[out_strides[3]] = s2 - s4;

        // Output point 5: X(4)
        out[out_strides[4]] = s1 + s3;
    }
    AOCLFFTZ_LOG(DEBUG, global_logger_mode, "Exit");
}

static FFTZ_VOID r2hc_rfft5avx128_fp64_fwd(FFTZ_VOID *in_real,
                                           FFTZ_VOID *in_imag,
                                           FFTZ_VOID *out_real,
                                           FFTZ_VOID *out_imag, FFTZ_INTP n,
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

    FFTZ_DOUBLE *in = (FFTZ_DOUBLE *)in_real;
    FFTZ_DOUBLE *out = (FFTZ_DOUBLE *)out_real;
#ifdef VOLATILE_STRIDE_ARRAY
    volatile FFTZ_INTP *in_strides  = strides->in_strides;
    volatile FFTZ_INTP *out_strides = strides->out_strides;
#else
    FFTZ_INTP *in_strides  = strides->in_strides;
    FFTZ_INTP *out_strides = strides->out_strides;
#endif
    FFTZ_INTP v_in_stride  = strides->v_in_stride;
    FFTZ_INTP v_out_stride = strides->v_out_stride;

    FFTZ_INTP cnt;
    FFTZ_DOUBLE *curr_in, *curr_out;
    FFTZ_INTP N = n / NUM_SETS_REAL_128_D;

    __m128d v_CRTM_5_1 = _mm_set1_pd(CRTM_5_1);
    __m128d v_CRTM_5_2 = _mm_set1_pd(CRTM_5_2);
    __m128d v_CRTM_5_3 = _mm_set1_pd(CRTM_5_3);
    __m128d v_CRTM_5_4 = _mm_set1_pd(CRTM_5_4);

    for (cnt = 0; cnt < N; cnt++)
    {
        __m128d v_in0, v_in1, v_in2, v_in3, v_in4;
        __m128d v_s1, v_s2, v_s3, v_s4, v_s5, v_t1, v_s6;
        __m128d v_out0, v_out1, v_out2, v_out3, v_out4;

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

        v_s1 = _mm_add_pd(v_in1, v_in4);
        v_s2 = _mm_sub_pd(v_in4, v_in1);
        v_s3 = _mm_add_pd(v_in2, v_in3);
        v_s4 = _mm_sub_pd(v_in2, v_in3);
        v_s6 = _mm_add_pd(v_s1, v_s3);

        v_s5 = _mm_sub_pd(v_in0, _mm_mul_pd(v_CRTM_5_4, v_s6));
        v_t1 = _mm_mul_pd(v_CRTM_5_1, _mm_sub_pd(v_s1, v_s3));

        // Output point 1: X(0)
        v_out0 = _mm_add_pd(v_in0, v_s6);

        // Output point 2: X(1)
        v_out1 = _mm_add_pd(v_s5, v_t1);

        // Output point 3: X(2)
        v_out2 = _mm_sub_pd(_mm_mul_pd(v_CRTM_5_2, v_s2),
                            _mm_mul_pd(v_CRTM_5_3, v_s4));

        // Output point 4: X(3)
        v_out3 = _mm_sub_pd(v_s5, v_t1);

        // Output point 5: X(4)
        v_out4 = _mm_add_pd(_mm_mul_pd(v_CRTM_5_2, v_s4),
                            _mm_mul_pd(v_CRTM_5_3, v_s2));

        STR_128_D(curr_out, v_out_stride, v_out0);
        curr_out = out + out_strides[1];
        STRI_2x128_D(curr_out, v_out_stride, v_out1, v_out2);
        curr_out = out + out_strides[3];
        STRI_2x128_D(curr_out, v_out_stride, v_out3, v_out4);

        in += v_in_stride * NUM_SETS_REAL_128_D;
        out += v_out_stride * NUM_SETS_REAL_128_D;
    }
    // tail cases
    if (n & 1)
    {
        FFTZ_DOUBLE in0, in1, in2, in3, in4;
        FFTZ_DOUBLE s1, s2, s3, s4, s5, s6, t1;

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

        s1 = in1 + in4;
        s2 = in4 - in1;
        s3 = in2 + in3;
        s4 = in2 - in3;
        s6 = s1 + s3;

        s5 = in0 - (CRTM_5_4 * s6);
        t1 = CRTM_5_1 * (s1 - s3);

        // Output point 1: X(0)
        *out = in0 + s6;
        // Output point 2: X(1)
        out[out_strides[1]] = s5 + t1;
        // Output point 3: X(2)
        out[out_strides[2]] = (CRTM_5_2 * s2) - (CRTM_5_3 * s4);
        // Output point 4: X(3)
        out[out_strides[3]] = s5 - t1;
        // Output point 5: X(4)
        out[out_strides[4]] = (CRTM_5_2 * s4) + (CRTM_5_3 * s2);
    }
    AOCLFFTZ_LOG(DEBUG, global_logger_mode, "Exit");
}

static FFTZ_VOID r2hc_rfft5avx128_fp64_bwd(FFTZ_VOID *in_real,
                                           FFTZ_VOID *in_imag,
                                           FFTZ_VOID *out_real,
                                           FFTZ_VOID *out_imag, FFTZ_INTP n,
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

    FFTZ_DOUBLE *in = (FFTZ_DOUBLE *)in_real;
    FFTZ_DOUBLE *out = (FFTZ_DOUBLE *)out_real;
#ifdef VOLATILE_STRIDE_ARRAY
    volatile FFTZ_INTP *in_strides  = strides->in_strides;
    volatile FFTZ_INTP *out_strides = strides->out_strides;
#else
    FFTZ_INTP *in_strides  = strides->in_strides;
    FFTZ_INTP *out_strides = strides->out_strides;
#endif
    FFTZ_INTP v_in_stride  = strides->v_in_stride;
    FFTZ_INTP v_out_stride = strides->v_out_stride;

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
        __m128d v_in0, v_in1, v_in2, v_in3, v_in4;
        __m128d v_s1, v_s2, v_s3, v_s4, v_s5, v_t1, v_s6;
        __m128d v_out0, v_out1, v_out2, v_out3, v_out4;

        curr_in = in;
        curr_out = out;

        // Input point 1: x(0)
        LDR_128_D(curr_in, v_in_stride, v_in0);
        // Input point 2: x(1) & x(2)
        curr_in = in + in_strides[1];
        LDRI_2x128_D(curr_in, v_in_stride, v_in1, v_in2);
        // Input point 4: x(3) & x(4)
        curr_in = in + in_strides[3];
        LDRI_2x128_D(curr_in, v_in_stride, v_in3, v_in4);

        v_s6 = _mm_add_pd(v_in1, v_in3);
        v_t1 = _mm_mul_pd(v_CRTM_5_1, _mm_sub_pd(v_in1, v_in3));
        v_s5 = _mm_sub_pd(v_in0, _mm_mul_pd(v_CRTM_5_4, v_s6));
        v_s1 = _mm_add_pd(v_s5, v_t1);
        v_s2 = _mm_sub_pd(v_s5, v_t1);
        v_s3 = _mm_add_pd(_mm_mul_pd(v_CRTM_5_2, v_in2),
                          _mm_mul_pd(v_CRTM_5_3, v_in4));
        v_s4 = _mm_sub_pd(_mm_mul_pd(v_CRTM_5_2, v_in4),
                          _mm_mul_pd(v_CRTM_5_3, v_in2));

        // Output point 1: X(0)
        v_out0 = _mm_add_pd(v_in0, _mm_mul_pd(v_CRTM_5_5, v_s6));

        // Output point 2: X(1)
        v_out1 = _mm_sub_pd(v_s1, v_s3);

        // Output point 3: X(2)
        v_out2 = _mm_add_pd(v_s2, v_s4);

        // Output point 4: X(3)
        v_out3 = _mm_sub_pd(v_s2, v_s4);

        // Output point 5: X(4)
        v_out4 = _mm_add_pd(v_s1, v_s3);

        STR_128_D(curr_out, v_out_stride, v_out0);
        curr_out = out + out_strides[1];
        STR_128_D(curr_out, v_out_stride, v_out1);
        curr_out = out + out_strides[2];
        STR_128_D(curr_out, v_out_stride, v_out2);
        curr_out = out + out_strides[3];
        STR_128_D(curr_out, v_out_stride, v_out3);
        curr_out = out + out_strides[4];
        STR_128_D(curr_out, v_out_stride, v_out4);

        in += v_in_stride * NUM_SETS_REAL_128_D;
        out += v_out_stride * NUM_SETS_REAL_128_D;
    }
    // tail cases
    if (n & 1)
    {
        FFTZ_DOUBLE in0, in1, in2, in3, in4;
        FFTZ_DOUBLE s1, s2, s3, s4, s5, s6, t1;

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

        s6 = in1 + in3;
        t1 = CRTM_5_1 * (in1 - in3);
        s5 = in0 - (CRTM_5_4 * s6);
        s1 = s5 + t1;
        s2 = s5 - t1;
        s3 = (CRTM_5_2 * in2) + (CRTM_5_3 * in4);
        s4 = (CRTM_5_2 * in4) - (CRTM_5_3 * in2);

        // Output point 1: X(0)
        *out = in0 + CRTM_5_5 * s6;

        // Output point 2: X(1)
        out[out_strides[1]] = s1 - s3;

        // Output point 3: X(2)
        out[out_strides[2]] = s2 + s4;

        // Output point 4: X(3)
        out[out_strides[3]] = s2 - s4;

        // Output point 5: X(4)
        out[out_strides[4]] = s1 + s3;
    }
    AOCLFFTZ_LOG(DEBUG, global_logger_mode, "Exit");
}

kfft_ register_kernel_r2hc_rfft5avx128(FFTZ_UINT8 precision,
                                       FFTZ_UINT8 direction)
{
    if (direction == FORWARD_FFT_DIR)
    {
        if (precision == DT_FLOAT)
        {
            return r2hc_rfft5avx128_fp32_fwd;
        }
        else if (precision == DT_DOUBLE)
        {
            return r2hc_rfft5avx128_fp64_fwd;
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
            return r2hc_rfft5avx128_fp32_bwd;
        }
        else if (precision == DT_DOUBLE)
        {
            return r2hc_rfft5avx128_fp64_bwd;
        }
        else
        {
            return NULL;
        }
    }
}
