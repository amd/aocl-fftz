// Copyright Advanced Micro Devices, Inc.
// SPDX-License-Identifier: BSD-3-Clause

/** @file rfft7avx128.c
 *
 *  @brief Radix-7 r2hc Real-FFT kernel with AVX-128 operations using x86 SIMD
 *  intrinsics
 *
 *  This file contains the DIT radix-7 real-to-halfcomplex implementations using
 *  AVX128 SIMD operations for single-precision and double-precision inputs.
 *
 *  @author Jeevanantham N
 */

#include "core/kernels/kernel.h"
#include "core/kernels/simd_includes/r2hc_simd_avx128.h"

static const ops_cycles_t ops_cnt[NUM_PRECISIONS][NUM_FFT_DIRS] =
                                                    {{{0, 18, 24, 44, 30, 0},
                                                      {0, 19, 24, 44, 33, 0}},
                                                     {{0, 18, 24, 22, 6, 0},
                                                      {0, 19, 24, 22, 6, 0}}};

ops_cycles_t get_ops_cnt_r2hc_rfft7avx128(FFTZ_UINT8 precision,
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

static FFTZ_VOID r2hc_rfft7avx128_fp32_fwd(FFTZ_VOID *in_real,
                                           FFTZ_VOID *in_imag,
                                           FFTZ_VOID *out_real,
                                           FFTZ_VOID *out_imag, FFTZ_INTP n,
                                           aoclfftz_strides_t *strides,
                                           FFTZ_VOID *twd, FFTZ_UINT8 flag)
{
    AOCLFFTZ_LOG(DEBUG, global_logger_mode, "Enter");
    const FFTZ_FLOAT CRTM_7_1 =
        0.900968867902419126236102319507445051165919162f;
    const FFTZ_FLOAT CRTM_7_2 =
        0.433883739117558120475768332848358754609990728f;
    const FFTZ_FLOAT CRTM_7_3 =
        0.623489801858733530525004884004239810632274731f;
    const FFTZ_FLOAT CRTM_7_4 =
        0.781831482468029808708444526674057750232334519f;
    const FFTZ_FLOAT CRTM_7_5 =
        0.222520933956314404288902564496794759466355569f;
    const FFTZ_FLOAT CRTM_7_6 =
        0.974927912181823607018131682993931217232785801f;

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

    __m128 v_CRTM_7_1 = _mm_broadcast_ss(&CRTM_7_1);
    __m128 v_CRTM_7_2 = _mm_broadcast_ss(&CRTM_7_2);
    __m128 v_CRTM_7_3 = _mm_broadcast_ss(&CRTM_7_3);
    __m128 v_CRTM_7_4 = _mm_broadcast_ss(&CRTM_7_4);
    __m128 v_CRTM_7_5 = _mm_broadcast_ss(&CRTM_7_5);
    __m128 v_CRTM_7_6 = _mm_broadcast_ss(&CRTM_7_6);

    for (cnt = 0; cnt < N; cnt++)
    {
        __m128 v_in0, v_in1, v_in2, v_in3, v_in4, v_in5, v_in6;
        __m128 v_s0, v_s1, v_s2, v_s3, v_s4, v_s5, v_s6, v_s7,  v_s8, v_s9,
               v_s10, v_s11, v_s12, v_s13,  v_s14, v_s15, v_s16;
        __m128 v_t0, v_t1, v_t2, v_t3, v_t4, v_t5,  v_t6, v_t7, v_t8, v_t9,
               v_t10, v_t11, v_t12, v_t13, v_t14, v_t15, v_t16, v_t17;
        __m128 v_out0, v_out1, v_out2, v_out3, v_out4, v_out5, v_out6;

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

        v_s0 = _mm_add_ps(v_in6, v_in1);
        v_s1 = _mm_sub_ps(v_in6, v_in1);
        v_s2 = _mm_add_ps(v_in5, v_in2);
        v_s3 = _mm_sub_ps(v_in5, v_in2);
        v_s4 = _mm_add_ps(v_in4, v_in3);
        v_s5 = _mm_sub_ps(v_in4, v_in3);

        v_s6 = _mm_add_ps(v_in0, v_s0);
        v_s7 = _mm_add_ps(v_s2, v_s4);
        // Output point 1: X(0)
        v_out0 = _mm_add_ps(v_s6, v_s7);

        v_t0 = _mm_mul_ps(v_CRTM_7_1, v_s4);
        v_t1 = _mm_mul_ps(v_CRTM_7_3, v_s0);
        v_t2 = _mm_mul_ps(v_CRTM_7_5, v_s2);
        v_s8 = _mm_sub_ps(v_in0, v_t0);
        v_s9 = _mm_sub_ps(v_t1, v_t2);
        // Output point 2: X(1)
        v_out1 = _mm_add_ps(v_s8, v_s9);

        v_t3 = _mm_mul_ps(v_CRTM_7_2, v_s5);
        v_t4 = _mm_mul_ps(v_CRTM_7_4, v_s1);
        v_t5 = _mm_mul_ps(v_CRTM_7_6, v_s3);
        v_s10 = _mm_add_ps(v_t3, v_t4);
        // Output point 3: X(2)
        v_out2 = _mm_add_ps(v_t5, v_s10);

        v_t6 = _mm_mul_ps(v_CRTM_7_1, v_s2);
        v_t7 = _mm_mul_ps(v_CRTM_7_3, v_s4);
        v_t8 = _mm_mul_ps(v_CRTM_7_5, v_s0);
        v_s11 = _mm_sub_ps(v_in0, v_t6);
        v_s12 = _mm_sub_ps(v_t7, v_t8);
        // Output point 4: X(3)
        v_out3 = _mm_add_ps(v_s11, v_s12);

        v_t9 = _mm_mul_ps(v_CRTM_7_2, v_s3);
        v_t10 = _mm_mul_ps(v_CRTM_7_4, v_s5);
        v_t11 = _mm_mul_ps(v_CRTM_7_6, v_s1);
        v_s13 = _mm_add_ps(v_t9, v_t10);
        // Output point 5: X(4)
        v_out4 = _mm_sub_ps(v_t11, v_s13);

        v_t12 = _mm_mul_ps(v_CRTM_7_1, v_s0);
        v_t13 = _mm_mul_ps(v_CRTM_7_3, v_s2);
        v_t14 = _mm_mul_ps(v_CRTM_7_5, v_s4);
        v_s14 = _mm_sub_ps(v_in0, v_t12);
        v_s15 = _mm_sub_ps(v_t13, v_t14);
        // Output point 6: X(5)
        v_out5 = _mm_add_ps(v_s14, v_s15);

        v_t15 = _mm_mul_ps(v_CRTM_7_2, v_s1);
        v_t16 = _mm_mul_ps(v_CRTM_7_4, v_s3);
        v_t17 = _mm_mul_ps(v_CRTM_7_6, v_s5);
        v_s16 = _mm_sub_ps(v_t15, v_t16);

        // Output point 7: X(6)
        v_out6 = _mm_add_ps(v_s16, v_t17);

        STR_128_S(curr_out, v_out_stride, v_out0);
        curr_out = out + out_strides[1];
        STRI_2x128_S(curr_out, v_out_stride, v_out1, v_out2);
        curr_out = out + out_strides[3];
        STRI_2x128_S(curr_out, v_out_stride, v_out3, v_out4);
        curr_out = out + out_strides[5];
        STRI_2x128_S(curr_out, v_out_stride, v_out5, v_out6);

        in += v_in_stride * NUM_SETS_REAL_128_S;
        out += v_out_stride * NUM_SETS_REAL_128_S;
    }
    // tail cases
    if (n & 2)
    {
        __m128 v_in0, v_in1, v_in2, v_in3, v_in4, v_in5, v_in6;
        __m128 v_s0, v_s1, v_s2, v_s3, v_s4, v_s5, v_s6, v_s7,  v_s8, v_s9,
               v_s10, v_s11, v_s12, v_s13,  v_s14, v_s15, v_s16;
        __m128 v_t0, v_t1, v_t2, v_t3, v_t4, v_t5,  v_t6, v_t7, v_t8, v_t9,
               v_t10, v_t11, v_t12, v_t13, v_t14, v_t15, v_t16, v_t17;
        __m128 v_out0, v_out1, v_out2, v_out3, v_out4, v_out5, v_out6;

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

        v_s0 = _mm_add_ps(v_in6, v_in1);
        v_s1 = _mm_sub_ps(v_in6, v_in1);
        v_s2 = _mm_add_ps(v_in5, v_in2);
        v_s3 = _mm_sub_ps(v_in5, v_in2);
        v_s4 = _mm_add_ps(v_in4, v_in3);
        v_s5 = _mm_sub_ps(v_in4, v_in3);

        v_s6 = _mm_add_ps(v_in0, v_s0);
        v_s7 = _mm_add_ps(v_s2, v_s4);
        // Output point 1: X(0)
        v_out0 = _mm_add_ps(v_s6, v_s7);

        v_t0 = _mm_mul_ps(v_CRTM_7_1, v_s4);
        v_t1 = _mm_mul_ps(v_CRTM_7_3, v_s0);
        v_t2 = _mm_mul_ps(v_CRTM_7_5, v_s2);
        v_s8 = _mm_sub_ps(v_in0, v_t0);
        v_s9 = _mm_sub_ps(v_t1, v_t2);
        // Output point 2: X(1)
        v_out1 = _mm_add_ps(v_s8, v_s9);

        v_t3 = _mm_mul_ps(v_CRTM_7_2, v_s5);
        v_t4 = _mm_mul_ps(v_CRTM_7_4, v_s1);
        v_t5 = _mm_mul_ps(v_CRTM_7_6, v_s3);
        v_s10 = _mm_add_ps(v_t3, v_t4);
        // Output point 3: X(2)
        v_out2 = _mm_add_ps(v_t5, v_s10);

        v_t6 = _mm_mul_ps(v_CRTM_7_1, v_s2);
        v_t7 = _mm_mul_ps(v_CRTM_7_3, v_s4);
        v_t8 = _mm_mul_ps(v_CRTM_7_5, v_s0);
        v_s11 = _mm_sub_ps(v_in0, v_t6);
        v_s12 = _mm_sub_ps(v_t7, v_t8);
        // Output point 4: X(3)
        v_out3 = _mm_add_ps(v_s11, v_s12);

        v_t9 = _mm_mul_ps(v_CRTM_7_2, v_s3);
        v_t10 = _mm_mul_ps(v_CRTM_7_4, v_s5);
        v_t11 = _mm_mul_ps(v_CRTM_7_6, v_s1);
        v_s13 = _mm_add_ps(v_t9, v_t10);
        // Output point 5: X(4)
        v_out4 = _mm_sub_ps(v_t11, v_s13);

        v_t12 = _mm_mul_ps(v_CRTM_7_1, v_s0);
        v_t13 = _mm_mul_ps(v_CRTM_7_3, v_s2);
        v_t14 = _mm_mul_ps(v_CRTM_7_5, v_s4);
        v_s14 = _mm_sub_ps(v_in0, v_t12);
        v_s15 = _mm_sub_ps(v_t13, v_t14);
        // Output point 6: X(5)
        v_out5 = _mm_add_ps(v_s14, v_s15);

        v_t15 = _mm_mul_ps(v_CRTM_7_2, v_s1);
        v_t16 = _mm_mul_ps(v_CRTM_7_4, v_s3);
        v_t17 = _mm_mul_ps(v_CRTM_7_6, v_s5);
        v_s16 = _mm_sub_ps(v_t15, v_t16);

        // Output point 7: X(6)
        v_out6 = _mm_add_ps(v_s16, v_t17);
        STHR_128_S(curr_out, v_out_stride, v_out0);
        curr_out = out + out_strides[1];
        STHRI_2x128_S(curr_out, v_out_stride, v_out1, v_out2);
        curr_out = out + out_strides[3];
        STHRI_2x128_S(curr_out, v_out_stride, v_out3, v_out4);
        curr_out = out + out_strides[5];
        STHRI_2x128_S(curr_out, v_out_stride, v_out5, v_out6);

        in = in + (v_in_stride << 1);
        out = out + (v_out_stride << 1);
    }
    // tail cases
    if (n & 1)
    {
        FFTZ_FLOAT in0, in1, in2, in3, in4, in5, in6;
        FFTZ_FLOAT s0, s1, s2, s3, s4, s5, s6, s7, s8, s9, s10, s11, s12, s13,
            s14, s15, s16, s17, s18, s19, s20, s21, s22, s23, s24, s25, s26,
            s27, s28, s29, s30, s31, s32, s33, s34;

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

        s0 = in6 + in1;
        s1 = in6 - in1;
        s2 = in5 + in2;
        s3 = in5 - in2;
        s4 = in4 + in3;
        s5 = in4 - in3;

        s6 = in0 + s0;
        s7 = s2 + s4;

        // Output point 1: X(0)
        *out = s6 + s7;

        // Output point 2: X(1)
        s8 = CRTM_7_1 * s4;
        s9 = CRTM_7_3 * s0;
        s10 = CRTM_7_5 * s2;
        s11 = in0 - s8;
        s12 = s9 - s10;
        out[out_strides[1]] = s11 + s12;

        // Output point 3: X(2)
        s13 = CRTM_7_2 * s5;
        s14 = CRTM_7_4 * s1;
        s15 = CRTM_7_6 * s3;
        s16 = s13 + s14;
        out[out_strides[2]] = s15 + s16;

        // Output point 4: X(3)
        s17 = CRTM_7_1 * s2;
        s18 = CRTM_7_3 * s4;
        s19 = CRTM_7_5 * s0;
        s20 = in0 - s17;
        s21 = s18 - s19;
        out[out_strides[3]] = s20 + s21;

        // Output point 5: X(4)
        s22 = CRTM_7_2 * s3;
        s23 = CRTM_7_4 * s5;
        s24 = CRTM_7_6 * s1;
        s25 = -s22 - s23;
        out[out_strides[4]] = s25 + s24;

        // Output point 6: X(5)
        s26 = CRTM_7_1 * s0;
        s27 = CRTM_7_3 * s2;
        s28 = CRTM_7_5 * s4;
        s29 = in0 - s26;
        s30 = s27 - s28;
        out[out_strides[5]] = s29 + s30;

        // Output point 7: X(6)
        s31 = CRTM_7_2 * s1;
        s32 = CRTM_7_4 * s3;
        s33 = CRTM_7_6 * s5;
        s34 = s31 - s32;
        out[out_strides[6]] = s34 + s33;
    }
    AOCLFFTZ_LOG(DEBUG, global_logger_mode, "Exit");
}

static FFTZ_VOID r2hc_rfft7avx128_fp32_bwd(FFTZ_VOID *in_real,
                                           FFTZ_VOID *in_imag,
                                           FFTZ_VOID *out_real,
                                           FFTZ_VOID *out_imag, FFTZ_INTP n,
                                           aoclfftz_strides_t *strides,
                                           FFTZ_VOID *twd, FFTZ_UINT8 flag)
{
    AOCLFFTZ_LOG(DEBUG, global_logger_mode, "Enter");
    const FFTZ_FLOAT CRTM_7_1 =
        1.801937735804838252472204639014890102331838324f;
    const FFTZ_FLOAT CRTM_7_2 =
        0.867767478235116240951536665696717509219981456f;
    const FFTZ_FLOAT CRTM_7_3 =
        1.246979603717467061050009768008479621264549462f;
    const FFTZ_FLOAT CRTM_7_4 =
        1.563662964936059617416889053348115500464669038f;
    const FFTZ_FLOAT CRTM_7_5 =
        0.445041867912628808577802568993589518932711138f;
    const FFTZ_FLOAT CRTM_7_6 =
        1.949855824363647214036263365987862434465571602f;
    const FFTZ_FLOAT CRTM_7_7 =
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

    __m128 v_CRTM_7_1 = _mm_broadcast_ss(&CRTM_7_1);
    __m128 v_CRTM_7_2 = _mm_broadcast_ss(&CRTM_7_2);
    __m128 v_CRTM_7_3 = _mm_broadcast_ss(&CRTM_7_3);
    __m128 v_CRTM_7_4 = _mm_broadcast_ss(&CRTM_7_4);
    __m128 v_CRTM_7_5 = _mm_broadcast_ss(&CRTM_7_5);
    __m128 v_CRTM_7_6 = _mm_broadcast_ss(&CRTM_7_6);
    __m128 v_CRTM_7_7 = _mm_broadcast_ss(&CRTM_7_7);

    for (cnt = 0; cnt < N; cnt++)
    {
        __m128 v_in0, v_in1, v_in2, v_in3, v_in4, v_in5, v_in6;
        __m128 v_s0, v_s1, v_s2, v_s3, v_s4,  v_s5, v_s6, v_s7, v_s8, v_s9,
               v_s10, v_s11, v_s12, v_s13, v_s14, v_s15, v_s16;
        __m128 v_t0, v_t1, v_t2, v_t3, v_t4, v_t5, v_t6, v_t7, v_t8, v_t9,
               v_t10, v_t11, v_t12, v_t13, v_t14, v_t15, v_t16, v_t17, v_t18;
        __m128 v_out0, v_out1, v_out2, v_out3, v_out4, v_out5, v_out6;

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

        v_t0 = _mm_mul_ps(v_CRTM_7_1, v_in5);
        v_t1 = _mm_mul_ps(v_CRTM_7_3, v_in1);
        v_t2 = _mm_mul_ps(v_CRTM_7_5, v_in3);
        v_s0 = _mm_sub_ps(v_in0, v_t0);
        v_s1 = _mm_sub_ps(v_t1, v_t2);
        v_s2 = _mm_add_ps(v_s0, v_s1);

        v_t3 = _mm_mul_ps(v_CRTM_7_2, v_in6);
        v_t4 = _mm_mul_ps(v_CRTM_7_4, v_in2);
        v_t5 = _mm_mul_ps(v_CRTM_7_6, v_in4);
        v_s3 = _mm_add_ps(v_t3, v_t4);
        v_s4 = _mm_add_ps(v_s3, v_t5);
        // Output point 2: X(1)
        curr_out = out + out_strides[1];
        v_out1 = _mm_sub_ps(v_s2, v_s4);
        STR_128_S(curr_out, v_out_stride, v_out1);
        // Output point 7: X(6)
        curr_out = out + out_strides[6];
        v_out6 = _mm_add_ps(v_s2, v_s4);
        STR_128_S(curr_out, v_out_stride, v_out6);

        v_t6 = _mm_mul_ps(v_CRTM_7_1, v_in3);
        v_t7 = _mm_mul_ps(v_CRTM_7_3, v_in5);
        v_t8 = _mm_mul_ps(v_CRTM_7_5, v_in1);
        v_s5 = _mm_sub_ps(v_in0, v_t6);
        v_s6 = _mm_sub_ps(v_t7, v_t8);
        v_s7 = _mm_add_ps(v_s5, v_s6);

        v_t9 = _mm_mul_ps(v_CRTM_7_2, v_in4);
        v_t10 = _mm_mul_ps(v_CRTM_7_4, v_in6);
        v_t11 = _mm_mul_ps(v_CRTM_7_6, v_in2);
        v_s8 = _mm_add_ps(v_t9, v_t10);
        v_s9 = _mm_sub_ps(v_t11, v_s8);
        // Output point 3: X(2)
        curr_out = out + out_strides[2];
        v_out2 = _mm_sub_ps(v_s7, v_s9);
        STR_128_S(curr_out, v_out_stride, v_out2);
        // Output point 6: X(5)
        v_out5 = _mm_add_ps(v_s7, v_s9);
        curr_out = out + out_strides[5];
        STR_128_S(curr_out, v_out_stride, v_out5);

        v_t12 = _mm_mul_ps(v_CRTM_7_1, v_in1);
        v_t13 = _mm_mul_ps(v_CRTM_7_3, v_in3);
        v_t14 = _mm_mul_ps(v_CRTM_7_5, v_in5);
        v_s10 = _mm_sub_ps(v_in0, v_t12);
        v_s11 = _mm_sub_ps(v_t13, v_t14);
        v_s12 = _mm_add_ps(v_s10, v_s11);

        v_t15 = _mm_mul_ps(v_CRTM_7_2, v_in2);
        v_t16 = _mm_mul_ps(v_CRTM_7_4, v_in4);
        v_t17 = _mm_mul_ps(v_CRTM_7_6, v_in6);
        v_s13 = _mm_sub_ps(v_t15, v_t16);
        v_s14 = _mm_add_ps(v_s13, v_t17);
        // Output point 4: X(3)
        curr_out = out + out_strides[3];
        v_out3 = _mm_sub_ps(v_s12, v_s14);
        STR_128_S(curr_out, v_out_stride, v_out3);
        // Output point 5: X(4)
        curr_out = out + out_strides[4];
        v_out4 = _mm_add_ps(v_s12, v_s14);
        STR_128_S(curr_out, v_out_stride, v_out4);

        v_s15 = _mm_add_ps(v_in1, v_in3);
        v_s16 = _mm_add_ps(v_s15, v_in5);
        v_t18 = _mm_mul_ps(v_CRTM_7_7, v_s16);
        // Output point 1: X(0)
        curr_out = out + out_strides[0];
        v_out0 = _mm_add_ps(v_in0, v_t18);
        STR_128_S(curr_out, v_out_stride, v_out0);

        in += v_in_stride * NUM_SETS_REAL_128_S;
        out += v_out_stride * NUM_SETS_REAL_128_S;
    }
    // tail cases
    if (n & 2)
    {
        __m128 v_in0, v_in1, v_in2, v_in3, v_in4, v_in5, v_in6;
        __m128 v_s0, v_s1, v_s2, v_s3, v_s4,  v_s5, v_s6, v_s7, v_s8, v_s9,
               v_s10, v_s11, v_s12, v_s13, v_s14, v_s15, v_s16;
        __m128 v_t0, v_t1, v_t2, v_t3, v_t4, v_t5, v_t6, v_t7, v_t8, v_t9,
               v_t10, v_t11, v_t12, v_t13, v_t14, v_t15, v_t16, v_t17, v_t18;
        __m128 v_out0, v_out1, v_out2, v_out3, v_out4, v_out5, v_out6;

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

        v_t0 = _mm_mul_ps(v_CRTM_7_1, v_in5);
        v_t1 = _mm_mul_ps(v_CRTM_7_3, v_in1);
        v_t2 = _mm_mul_ps(v_CRTM_7_5, v_in3);
        v_s0 = _mm_sub_ps(v_in0, v_t0);
        v_s1 = _mm_sub_ps(v_t1, v_t2);
        v_s2 = _mm_add_ps(v_s0, v_s1);

        v_t3 = _mm_mul_ps(v_CRTM_7_2, v_in6);
        v_t4 = _mm_mul_ps(v_CRTM_7_4, v_in2);
        v_t5 = _mm_mul_ps(v_CRTM_7_6, v_in4);
        v_s3 = _mm_add_ps(v_t3, v_t4);
        v_s4 = _mm_add_ps(v_s3, v_t5);
        // Output point 2: X(1)
        curr_out = out + out_strides[1];
        v_out1 = _mm_sub_ps(v_s2, v_s4);
        STHR_128_S(curr_out, v_out_stride, v_out1);
        // Output point 7: X(6)
        curr_out = out + out_strides[6];
        v_out6 = _mm_add_ps(v_s2, v_s4);
        STHR_128_S(curr_out, v_out_stride, v_out6);

        v_t6 = _mm_mul_ps(v_CRTM_7_1, v_in3);
        v_t7 = _mm_mul_ps(v_CRTM_7_3, v_in5);
        v_t8 = _mm_mul_ps(v_CRTM_7_5, v_in1);
        v_s5 = _mm_sub_ps(v_in0, v_t6);
        v_s6 = _mm_sub_ps(v_t7, v_t8);
        v_s7 = _mm_add_ps(v_s5, v_s6);

        v_t9 = _mm_mul_ps(v_CRTM_7_2, v_in4);
        v_t10 = _mm_mul_ps(v_CRTM_7_4, v_in6);
        v_t11 = _mm_mul_ps(v_CRTM_7_6, v_in2);
        v_s8 = _mm_add_ps(v_t9, v_t10);
        v_s9 = _mm_sub_ps(v_t11, v_s8);
        // Output point 3: X(2)
        curr_out = out + out_strides[2];
        v_out2 = _mm_sub_ps(v_s7, v_s9);
        STHR_128_S(curr_out, v_out_stride, v_out2);
        // Output point 6: X(5)
        v_out5 = _mm_add_ps(v_s7, v_s9);
        curr_out = out + out_strides[5];
        STHR_128_S(curr_out, v_out_stride, v_out5);

        v_t12 = _mm_mul_ps(v_CRTM_7_1, v_in1);
        v_t13 = _mm_mul_ps(v_CRTM_7_3, v_in3);
        v_t14 = _mm_mul_ps(v_CRTM_7_5, v_in5);
        v_s10 = _mm_sub_ps(v_in0, v_t12);
        v_s11 = _mm_sub_ps(v_t13, v_t14);
        v_s12 = _mm_add_ps(v_s10, v_s11);

        v_t15 = _mm_mul_ps(v_CRTM_7_2, v_in2);
        v_t16 = _mm_mul_ps(v_CRTM_7_4, v_in4);
        v_t17 = _mm_mul_ps(v_CRTM_7_6, v_in6);
        v_s13 = _mm_sub_ps(v_t15, v_t16);
        v_s14 = _mm_add_ps(v_s13, v_t17);
        // Output point 4: X(3)
        curr_out = out + out_strides[3];
        v_out3 = _mm_sub_ps(v_s12, v_s14);
        STHR_128_S(curr_out, v_out_stride, v_out3);
        // Output point 5: X(4)
        curr_out = out + out_strides[4];
        v_out4 = _mm_add_ps(v_s12, v_s14);
        STHR_128_S(curr_out, v_out_stride, v_out4);

        v_s15 = _mm_add_ps(v_in1, v_in3);
        v_s16 = _mm_add_ps(v_s15, v_in5);
        v_t18 = _mm_mul_ps(v_CRTM_7_7, v_s16);
        // Output point 1: X(0)
        curr_out = out + out_strides[0];
        v_out0 = _mm_add_ps(v_in0, v_t18);
        STHR_128_S(curr_out, v_out_stride, v_out0);

        in = in + (v_in_stride << 1);
        out = out + (v_out_stride << 1);
    }
    // tail cases
    if (n & 1)
    {
        FFTZ_FLOAT in0, in1, in2, in3, in4, in5, in6;
        FFTZ_FLOAT s0, s1, s2, s3, s4, s5, s6, s7, s8,
              s9, s10, s11, s12, s13, s14, s15, s16,
              s17, s18, s19, s20, s21, s22, s23, s24,
              s25, s26, s27, s28, s29, s30, s31, s32,
              s33, s34, s35;

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

        s0 = CRTM_7_1 * in5;
        s1 = CRTM_7_3 * in1;
        s2 = CRTM_7_5 * in3;
        s3 = in0 - s0;
        s4 = s1 - s2;
        s5 = s3 + s4;

        s6 = CRTM_7_2 * in6;
        s7 = CRTM_7_4 * in2;
        s8 = CRTM_7_6 * in4;
        s9 = s6 + s7;
        s10 = s9 + s8;

        s11 = CRTM_7_1 * in3;
        s12 = CRTM_7_3 * in5;
        s13 = CRTM_7_5 * in1;
        s14 = in0 - s11;
        s15 = s12 - s13;
        s16 = s14 + s15;

        s17 = CRTM_7_2 * in4;
        s18 = CRTM_7_4 * in6;
        s19 = CRTM_7_6 * in2;
        s20 = -s17 - s18;
        s21 = s20 + s19;

        s22 = CRTM_7_1 * in1;
        s23 = CRTM_7_3 * in3;
        s24 = CRTM_7_5 * in5;
        s25 = in0 - s22;
        s26 = s23 - s24;
        s27 = s25 + s26;

        s28 = CRTM_7_2 * in2;
        s29 = CRTM_7_4 * in4;
        s30 = CRTM_7_6 * in6;
        s31 = s28 - s29;
        s32 = s31 + s30;

        s33 = in1 + in3;
        s34 = s33 + in5;
        s35 = CRTM_7_7 * s34;

        // Output point 1: X(0)
        *out = in0 + s35;

        // Output point 2: X(1)
        out[out_strides[1]] = s5 - s10;

        // Output point 3: X(2)
        out[out_strides[2]] = s16 - s21;

        // Output point 4: X(3)
        out[out_strides[3]] = s27 - s32;

        // Output point 5: X(4)
        out[out_strides[4]] = s27 + s32;

        // Output point 6: X(5)
        out[out_strides[5]] = s16 + s21;

        // Output point 7: X(6)
        out[out_strides[6]] = s5 + s10;
    }
    AOCLFFTZ_LOG(DEBUG, global_logger_mode, "Exit");
}

static FFTZ_VOID r2hc_rfft7avx128_fp64_fwd(FFTZ_VOID *in_real,
                                           FFTZ_VOID *in_imag,
                                           FFTZ_VOID *out_real,
                                           FFTZ_VOID *out_imag, FFTZ_INTP n,
                                           aoclfftz_strides_t *strides,
                                           FFTZ_VOID *twd, FFTZ_UINT8 flag)
{
    AOCLFFTZ_LOG(DEBUG, global_logger_mode, "Enter");
    const FFTZ_DOUBLE CRTM_7_1 =
        0.900968867902419126236102319507445051165919162;
    const FFTZ_DOUBLE CRTM_7_2 =
        0.433883739117558120475768332848358754609990728;
    const FFTZ_DOUBLE CRTM_7_3 =
        0.623489801858733530525004884004239810632274731;
    const FFTZ_DOUBLE CRTM_7_4 =
        0.781831482468029808708444526674057750232334519;
    const FFTZ_DOUBLE CRTM_7_5 =
        0.222520933956314404288902564496794759466355569;
    const FFTZ_DOUBLE CRTM_7_6 =
        0.974927912181823607018131682993931217232785801;

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

    __m128d v_CRTM_7_1 = _mm_set1_pd(CRTM_7_1);
    __m128d v_CRTM_7_2 = _mm_set1_pd(CRTM_7_2);
    __m128d v_CRTM_7_3 = _mm_set1_pd(CRTM_7_3);
    __m128d v_CRTM_7_4 = _mm_set1_pd(CRTM_7_4);
    __m128d v_CRTM_7_5 = _mm_set1_pd(CRTM_7_5);
    __m128d v_CRTM_7_6 = _mm_set1_pd(CRTM_7_6);

    for (cnt = 0; cnt < N; cnt++)
    {
        __m128d v_in0, v_in1, v_in2, v_in3, v_in4, v_in5, v_in6;
        __m128d v_s0, v_s1, v_s2, v_s3, v_s4, v_s5, v_s6, v_s7,  v_s8, v_s9,
               v_s10, v_s11, v_s12, v_s13,  v_s14, v_s15, v_s16;
        __m128d v_t0, v_t1, v_t2, v_t3, v_t4, v_t5,  v_t6, v_t7, v_t8, v_t9,
               v_t10, v_t11, v_t12, v_t13, v_t14, v_t15, v_t16, v_t17;
        __m128d v_out0, v_out1, v_out2, v_out3, v_out4, v_out5, v_out6;

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

        v_s0 = _mm_add_pd(v_in6, v_in1);
        v_s1 = _mm_sub_pd(v_in6, v_in1);
        v_s2 = _mm_add_pd(v_in5, v_in2);
        v_s3 = _mm_sub_pd(v_in5, v_in2);
        v_s4 = _mm_add_pd(v_in4, v_in3);
        v_s5 = _mm_sub_pd(v_in4, v_in3);

        v_s6 = _mm_add_pd(v_in0, v_s0);
        v_s7 = _mm_add_pd(v_s2, v_s4);
        // Output point 1: X(0)
        v_out0 = _mm_add_pd(v_s6, v_s7);

        v_t0 = _mm_mul_pd(v_CRTM_7_1, v_s4);
        v_t1 = _mm_mul_pd(v_CRTM_7_3, v_s0);
        v_t2 = _mm_mul_pd(v_CRTM_7_5, v_s2);
        v_s8 = _mm_sub_pd(v_in0, v_t0);
        v_s9 = _mm_sub_pd(v_t1, v_t2);
        // Output point 2: X(1)
        v_out1 = _mm_add_pd(v_s8, v_s9);

        v_t3 = _mm_mul_pd(v_CRTM_7_2, v_s5);
        v_t4 = _mm_mul_pd(v_CRTM_7_4, v_s1);
        v_t5 = _mm_mul_pd(v_CRTM_7_6, v_s3);
        v_s10 = _mm_add_pd(v_t3, v_t4);
        // Output point 3: X(2)
        v_out2 = _mm_add_pd(v_t5, v_s10);

        v_t6 = _mm_mul_pd(v_CRTM_7_1, v_s2);
        v_t7 = _mm_mul_pd(v_CRTM_7_3, v_s4);
        v_t8 = _mm_mul_pd(v_CRTM_7_5, v_s0);
        v_s11 = _mm_sub_pd(v_in0, v_t6);
        v_s12 = _mm_sub_pd(v_t7, v_t8);
        // Output point 4: X(3)
        v_out3 = _mm_add_pd(v_s11, v_s12);

        v_t9 = _mm_mul_pd(v_CRTM_7_2, v_s3);
        v_t10 = _mm_mul_pd(v_CRTM_7_4, v_s5);
        v_t11 = _mm_mul_pd(v_CRTM_7_6, v_s1);
        v_s13 = _mm_add_pd(v_t9, v_t10);
        // Output point 5: X(4)
        v_out4 = _mm_sub_pd(v_t11, v_s13);

        v_t12 = _mm_mul_pd(v_CRTM_7_1, v_s0);
        v_t13 = _mm_mul_pd(v_CRTM_7_3, v_s2);
        v_t14 = _mm_mul_pd(v_CRTM_7_5, v_s4);
        v_s14 = _mm_sub_pd(v_in0, v_t12);
        v_s15 = _mm_sub_pd(v_t13, v_t14);
        // Output point 6: X(5)
        v_out5 = _mm_add_pd(v_s14, v_s15);

        v_t15 = _mm_mul_pd(v_CRTM_7_2, v_s1);
        v_t16 = _mm_mul_pd(v_CRTM_7_4, v_s3);
        v_t17 = _mm_mul_pd(v_CRTM_7_6, v_s5);
        v_s16 = _mm_sub_pd(v_t15, v_t16);

        // Output point 7: X(6)
        v_out6 = _mm_add_pd(v_s16, v_t17);

        STR_128_D(curr_out, v_out_stride, v_out0);
        curr_out = out + out_strides[1];
        STRI_2x128_D(curr_out, v_out_stride, v_out1, v_out2);
        curr_out = out + out_strides[3];
        STRI_2x128_D(curr_out, v_out_stride, v_out3, v_out4);
        curr_out = out + out_strides[5];
        STRI_2x128_D(curr_out, v_out_stride, v_out5, v_out6);

        in += v_in_stride * NUM_SETS_REAL_128_D;
        out += v_out_stride * NUM_SETS_REAL_128_D;
    }
    // tail cases
    if (n & 1)
    {
        FFTZ_DOUBLE in0, in1, in2, in3, in4, in5, in6;
        FFTZ_DOUBLE s0, s1, s2, s3, s4, s5, s6, s7, s8, s9, s10, s11, s12, s13,
            s14, s15, s16, s17, s18, s19, s20, s21, s22, s23, s24, s25, s26,
            s27, s28, s29, s30, s31, s32, s33, s34;

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

        s0 = in6 + in1;
        s1 = in6 - in1;
        s2 = in5 + in2;
        s3 = in5 - in2;
        s4 = in4 + in3;
        s5 = in4 - in3;

        s6 = in0 + s0;
        s7 = s2 + s4;

        // Output point 1: X(0)
        *out = s6 + s7;

        // Output point 2: X(1)
        s8 = CRTM_7_1 * s4;
        s9 = CRTM_7_3 * s0;
        s10 = CRTM_7_5 * s2;
        s11 = in0 - s8;
        s12 = s9 - s10;
        out[out_strides[1]] = s11 + s12;

        // Output point 3: X(2)
        s13 = CRTM_7_2 * s5;
        s14 = CRTM_7_4 * s1;
        s15 = CRTM_7_6 * s3;
        s16 = s13 + s14;
        out[out_strides[2]] = s15 + s16;

        // Output point 4: X(3)
        s17 = CRTM_7_1 * s2;
        s18 = CRTM_7_3 * s4;
        s19 = CRTM_7_5 * s0;
        s20 = in0 - s17;
        s21 = s18 - s19;
        out[out_strides[3]] = s20 + s21;

        // Output point 5: X(4)
        s22 = CRTM_7_2 * s3;
        s23 = CRTM_7_4 * s5;
        s24 = CRTM_7_6 * s1;
        s25 = -s22 - s23;
        out[out_strides[4]] = s25 + s24;

        // Output point 6: X(5)
        s26 = CRTM_7_1 * s0;
        s27 = CRTM_7_3 * s2;
        s28 = CRTM_7_5 * s4;
        s29 = in0 - s26;
        s30 = s27 - s28;
        out[out_strides[5]] = s29 + s30;

        // Output point 7: X(6)
        s31 = CRTM_7_2 * s1;
        s32 = CRTM_7_4 * s3;
        s33 = CRTM_7_6 * s5;
        s34 = s31 - s32;
        out[out_strides[6]] = s34 + s33;
    }
    AOCLFFTZ_LOG(DEBUG, global_logger_mode, "Exit");
}

static FFTZ_VOID r2hc_rfft7avx128_fp64_bwd(FFTZ_VOID *in_real,
                                           FFTZ_VOID *in_imag,
                                           FFTZ_VOID *out_real,
                                           FFTZ_VOID *out_imag, FFTZ_INTP n,
                                           aoclfftz_strides_t *strides,
                                           FFTZ_VOID *twd, FFTZ_UINT8 flag)
{
    AOCLFFTZ_LOG(DEBUG, global_logger_mode, "Enter");
    const FFTZ_DOUBLE CRTM_7_1 =
        1.801937735804838252472204639014890102331838324;
    const FFTZ_DOUBLE CRTM_7_2 =
        0.867767478235116240951536665696717509219981456;
    const FFTZ_DOUBLE CRTM_7_3 =
        1.246979603717467061050009768008479621264549462;
    const FFTZ_DOUBLE CRTM_7_4 =
        1.563662964936059617416889053348115500464669038;
    const FFTZ_DOUBLE CRTM_7_5 =
        0.445041867912628808577802568993589518932711138;
    const FFTZ_DOUBLE CRTM_7_6 =
        1.949855824363647214036263365987862434465571602;
    const FFTZ_DOUBLE CRTM_7_7 =
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

    __m128d v_CRTM_7_1 = _mm_set1_pd(CRTM_7_1);
    __m128d v_CRTM_7_2 = _mm_set1_pd(CRTM_7_2);
    __m128d v_CRTM_7_3 = _mm_set1_pd(CRTM_7_3);
    __m128d v_CRTM_7_4 = _mm_set1_pd(CRTM_7_4);
    __m128d v_CRTM_7_5 = _mm_set1_pd(CRTM_7_5);
    __m128d v_CRTM_7_6 = _mm_set1_pd(CRTM_7_6);
    __m128d v_CRTM_7_7 = _mm_set1_pd(CRTM_7_7);

    for (cnt = 0; cnt < N; cnt++)
    {
        __m128d v_in0, v_in1, v_in2, v_in3, v_in4, v_in5, v_in6;
        __m128d v_s0, v_s1, v_s2, v_s3, v_s4,  v_s5, v_s6, v_s7, v_s8, v_s9,
                v_s10, v_s11, v_s12, v_s13, v_s14, v_s15, v_s16;
        __m128d v_t0, v_t1, v_t2, v_t3, v_t4, v_t5, v_t6, v_t7, v_t8, v_t9,
                v_t10, v_t11, v_t12, v_t13, v_t14, v_t15, v_t16, v_t17, v_t18;
        __m128d v_out0, v_out1, v_out2, v_out3, v_out4, v_out5, v_out6;

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

        v_t0 = _mm_mul_pd(v_CRTM_7_1, v_in5);
        v_t1 = _mm_mul_pd(v_CRTM_7_3, v_in1);
        v_t2 = _mm_mul_pd(v_CRTM_7_5, v_in3);
        v_s0 = _mm_sub_pd(v_in0, v_t0);
        v_s1 = _mm_sub_pd(v_t1, v_t2);
        v_s2 = _mm_add_pd(v_s0, v_s1);

        v_t3 = _mm_mul_pd(v_CRTM_7_2, v_in6);
        v_t4 = _mm_mul_pd(v_CRTM_7_4, v_in2);
        v_t5 = _mm_mul_pd(v_CRTM_7_6, v_in4);
        v_s3 = _mm_add_pd(v_t3, v_t4);
        v_s4 = _mm_add_pd(v_s3, v_t5);
        // Output point 2: X(1)
        curr_out = out + out_strides[1];
        v_out1 = _mm_sub_pd(v_s2, v_s4);
        STR_128_D(curr_out, v_out_stride, v_out1);
        // Output point 7: X(6)
        curr_out = out + out_strides[6];
        v_out6 = _mm_add_pd(v_s2, v_s4);
        STR_128_D(curr_out, v_out_stride, v_out6);

        v_t6 = _mm_mul_pd(v_CRTM_7_1, v_in3);
        v_t7 = _mm_mul_pd(v_CRTM_7_3, v_in5);
        v_t8 = _mm_mul_pd(v_CRTM_7_5, v_in1);
        v_s5 = _mm_sub_pd(v_in0, v_t6);
        v_s6 = _mm_sub_pd(v_t7, v_t8);
        v_s7 = _mm_add_pd(v_s5, v_s6);

        v_t9 = _mm_mul_pd(v_CRTM_7_2, v_in4);
        v_t10 = _mm_mul_pd(v_CRTM_7_4, v_in6);
        v_t11 = _mm_mul_pd(v_CRTM_7_6, v_in2);
        v_s8 = _mm_add_pd(v_t9, v_t10);
        v_s9 = _mm_sub_pd(v_t11, v_s8);
        // Output point 3: X(2)
        curr_out = out + out_strides[2];
        v_out2 = _mm_sub_pd(v_s7, v_s9);
        STR_128_D(curr_out, v_out_stride, v_out2);
        // Output point 6: X(5)
        v_out5 = _mm_add_pd(v_s7, v_s9);
        curr_out = out + out_strides[5];
        STR_128_D(curr_out, v_out_stride, v_out5);

        v_t12 = _mm_mul_pd(v_CRTM_7_1, v_in1);
        v_t13 = _mm_mul_pd(v_CRTM_7_3, v_in3);
        v_t14 = _mm_mul_pd(v_CRTM_7_5, v_in5);
        v_s10 = _mm_sub_pd(v_in0, v_t12);
        v_s11 = _mm_sub_pd(v_t13, v_t14);
        v_s12 = _mm_add_pd(v_s10, v_s11);

        v_t15 = _mm_mul_pd(v_CRTM_7_2, v_in2);
        v_t16 = _mm_mul_pd(v_CRTM_7_4, v_in4);
        v_t17 = _mm_mul_pd(v_CRTM_7_6, v_in6);
        v_s13 = _mm_sub_pd(v_t15, v_t16);
        v_s14 = _mm_add_pd(v_s13, v_t17);
        // Output point 4: X(3)
        curr_out = out + out_strides[3];
        v_out3 = _mm_sub_pd(v_s12, v_s14);
        STR_128_D(curr_out, v_out_stride, v_out3);
        // Output point 5: X(4)
        curr_out = out + out_strides[4];
        v_out4 = _mm_add_pd(v_s12, v_s14);
        STR_128_D(curr_out, v_out_stride, v_out4);

        v_s15 = _mm_add_pd(v_in1, v_in3);
        v_s16 = _mm_add_pd(v_s15, v_in5);
        v_t18 = _mm_mul_pd(v_CRTM_7_7, v_s16);
        // Output point 1: X(0)
        curr_out = out + out_strides[0];
        v_out0 = _mm_add_pd(v_in0, v_t18);
        STR_128_D(curr_out, v_out_stride, v_out0);

        in += v_in_stride * NUM_SETS_REAL_128_D;
        out += v_out_stride * NUM_SETS_REAL_128_D;
    }
    // tail cases
    if (n & 1)
    {
        FFTZ_DOUBLE in0, in1, in2, in3, in4, in5, in6;
        FFTZ_DOUBLE s0, s1, s2, s3, s4, s5, s6, s7, s8,
              s9, s10, s11, s12, s13, s14, s15, s16,
              s17, s18, s19, s20, s21, s22, s23, s24,
              s25, s26, s27, s28, s29, s30, s31, s32,
              s33, s34, s35;

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

        s0 = CRTM_7_1 * in5;
        s1 = CRTM_7_3 * in1;
        s2 = CRTM_7_5 * in3;
        s3 = in0 - s0;
        s4 = s1 - s2;
        s5 = s3 + s4;

        s6 = CRTM_7_2 * in6;
        s7 = CRTM_7_4 * in2;
        s8 = CRTM_7_6 * in4;
        s9 = s6 + s7;
        s10 = s9 + s8;

        s11 = CRTM_7_1 * in3;
        s12 = CRTM_7_3 * in5;
        s13 = CRTM_7_5 * in1;
        s14 = in0 - s11;
        s15 = s12 - s13;
        s16 = s14 + s15;

        s17 = CRTM_7_2 * in4;
        s18 = CRTM_7_4 * in6;
        s19 = CRTM_7_6 * in2;
        s20 = -s17 - s18;
        s21 = s20 + s19;

        s22 = CRTM_7_1 * in1;
        s23 = CRTM_7_3 * in3;
        s24 = CRTM_7_5 * in5;
        s25 = in0 - s22;
        s26 = s23 - s24;
        s27 = s25 + s26;

        s28 = CRTM_7_2 * in2;
        s29 = CRTM_7_4 * in4;
        s30 = CRTM_7_6 * in6;
        s31 = s28 - s29;
        s32 = s31 + s30;

        s33 = in1 + in3;
        s34 = s33 + in5;
        s35 = CRTM_7_7 * s34;

        // Output point 1: X(0)
        *out = in0 + s35;

        // Output point 2: X(1)
        out[out_strides[1]] = s5 - s10;

        // Output point 3: X(2)
        out[out_strides[2]] = s16 - s21;

        // Output point 4: X(3)
        out[out_strides[3]] = s27 - s32;

        // Output point 5: X(4)
        out[out_strides[4]] = s27 + s32;

        // Output point 6: X(5)
        out[out_strides[5]] = s16 + s21;

        // Output point 7: X(6)
        out[out_strides[6]] = s5 + s10;
    }
    AOCLFFTZ_LOG(DEBUG, global_logger_mode, "Exit");
}

kfft_ register_kernel_r2hc_rfft7avx128(FFTZ_UINT8 precision,
                                       FFTZ_UINT8 direction)
{
    if (direction == FORWARD_FFT_DIR)
    {
        if (precision == DT_FLOAT)
        {
            return r2hc_rfft7avx128_fp32_fwd;
        }
        else if (precision == DT_DOUBLE)
        {
            return r2hc_rfft7avx128_fp64_fwd;
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
            return r2hc_rfft7avx128_fp32_bwd;
        }
        else if (precision == DT_DOUBLE)
        {
            return r2hc_rfft7avx128_fp64_bwd;
        }
        else
        {
            return NULL;
        }
    }
}
