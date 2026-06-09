// Copyright Advanced Micro Devices, Inc.
// SPDX-License-Identifier: BSD-3-Clause

/** @file rfft11c.c
 *
 *  @brief Radix-11 r2hc Real-FFT kernel with scalar operations in C
 *
 *  This file contains the DIT radix-11 real-to-halfcomplex implementations
 *  using scalar operations for single-precision and double-precision inputs.
 *
 *  @author Jeevanantham N
 */

#include "core/kernels/kernel.h"

static const ops_cycles_t ops_cnt[NUM_PRECISIONS][NUM_FFT_DIRS] =
                                                    {{{0, 50, 60, 22, 0, 0},
                                                      {0, 51, 60, 22, 0, 0}},
                                                     {{0, 50, 60, 22, 0, 0},
                                                      {0, 51, 60, 22, 0, 0}}};

ops_cycles_t get_ops_cnt_r2hc_rfft11c(UINT8 precision, UINT8 direction)
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

static VOID r2hc_rfft11c_fp32_fwd(VOID *in_real, VOID *in_imag, VOID *out_real,
                                  VOID *out_imag, INTP n,
                                  aoclfftz_strides_t *strides, VOID *twd,
                                  UINT8 flag)
{
    AOCLFFTZ_LOG(DEBUG, global_logger_mode, "Enter");

    const FLOAT CRTM_11_1 = 0.841253532831181168861811648919367717513292498f;
    const FLOAT CRTM_11_2 = 0.540640817455597582107635954318691695431770608f;
    const FLOAT CRTM_11_3 = 0.415415013001886425529274149229623203524004910f;
    const FLOAT CRTM_11_4 = 0.909631995354518371411715383079028460060241051f;
    const FLOAT CRTM_11_5 = 0.142314838273285140443792668616369668791051361f;
    const FLOAT CRTM_11_6 = 0.989821441880932732376092037776718787376519372f;
    const FLOAT CRTM_11_7 = 0.654860733945285064056925072466293553183791199f;
    const FLOAT CRTM_11_8 = 0.755749574354258283774035843972344420179717445f;
    const FLOAT CRTM_11_9 = 0.959492973614497389890368057066327699062454848f;
    const FLOAT CRTM_11_10 = 0.281732556841429697711417915346616899035777899f;

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

    for (cnt = 0; cnt < n; cnt++)
    {
        FLOAT in0, in1, in2, in3, in4, in5, in6, in7, in8, in9, in10;
        FLOAT s0, s1, s2, s3, s4, s5, s6, s7, s8, s9, s10, s11, s12, s13, s14,
              s15, s16, s17, s18, s19, s20, s21, s22, s23, s24, s25, s26, s27,
              s28, s29, s30, s31, s32, s33, s34, s35, s36, s37, s38, s39, s40,
              s41, s42, s43, s44, s45, s46, s47, s48;
        FLOAT m0, m1, m2, m3, m4, m5, m6, m7, m8, m9, m10, m11, m12, m13, m14,
              m15, m16, m17, m18, m19, m20, m21, m22, m23, m24, m25, m26, m27,
              m28, m29, m30, m31, m32, m33, m34, m35, m36, m37, m38, m39, m40,
              m41, m42, m43, m44, m45, m46, m47, m48, m49;

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

        s0 = in1 + in10;
        s1 = in2 + in9;
        s2 = in3 + in8;
        s3 = in4 + in7;
        s4 = in5 + in6;
        s5 = in1 - in10;
        s6 = in2 - in9;
        s7 = in3 - in8;
        s8 = in4 - in7;
        s9 = in5 - in6;

        s10 = s0 + s1;
        s11 = s2 + s3;
        s12 = s4 + in0;
        s13 = s10 + s11;

        // Output point 1: X(0)
        *out = s12 + s13;

        m0 = CRTM_11_1 * s0;
        m1 = CRTM_11_3 * s1;
        m2 = CRTM_11_5 * s2;
        m3 = CRTM_11_7 * s3;
        m4 = CRTM_11_9 * s4;

        s14 = m0 + m1;
        s15 = m2 + m3;
        s16 = in0 - m4;
        s17 = s14 - s15;

        // Output point 2: X(1)
        out[out_strides[1]] = s16 + s17;

        m5 = CRTM_11_2 * s5;
        m6 = CRTM_11_4 * s6;
        m7 = CRTM_11_6 * s7;
        m8 = CRTM_11_8 * s8;
        m9 = CRTM_11_10 * s9;

        s18 = m5 + m6;
        s19 = m7 + m8;
        s20 = s19 + m9;

        // Output point 3: X(2)
        out[out_strides[2]] = -(s18 + s20);

        m10 = CRTM_11_1 * s4;
        m11 = CRTM_11_3 * s0;
        m12 = CRTM_11_5 * s3;
        m13 = CRTM_11_7 * s1;
        m14 = CRTM_11_9 * s2;

        s21 = m10 + m11;
        s22 = m12 + m13;
        s23 = in0 - m14;
        s24 = s21 - s22;

        // Output point 4: X(3)
        out[out_strides[3]] = s23 + s24;

        m15 = CRTM_11_2 * s9;
        m16 = CRTM_11_4 * s5;
        m17 = CRTM_11_6 * s8;
        m18 = CRTM_11_8 * s6;
        m19 = CRTM_11_10 * s7;

        s25 = m15 - m16;
        s26 = m17 - m18;
        s27 = s26 + m19;

        // Output point 5: X(4)
        out[out_strides[4]] = s25 + s27;

        m20 = CRTM_11_1 * s3;
        m21 = CRTM_11_3 * s2;
        m22 = CRTM_11_5 * s0;
        m23 = CRTM_11_7 * s4;
        m24 = CRTM_11_9 * s1;

        s28 = m20 + m21;
        s29 = m22 + m23;
        s30 = in0 - m24;
        s31 = s30 - s29;

        // Output point 6: X(5)
        out[out_strides[5]] = s28 + s31;

        m25 = CRTM_11_2 * s8;
        m26 = CRTM_11_4 * s7;
        m27 = CRTM_11_6 * s5;
        m28 = CRTM_11_8 * s9;
        m29 = CRTM_11_10 * s6;

        s32 = m26 - m25;
        s33 = m27 + m28;
        s34 = s32 - s33;

        // Output point 7: X(6)
        out[out_strides[6]] = s34 + m29;

        m30 = CRTM_11_1 * s2;
        m31 = CRTM_11_3 * s4;
        m32 = CRTM_11_5 * s1;
        m33 = CRTM_11_7 * s0;
        m34 = CRTM_11_9 * s3;

        s35 = m30 + m31;
        s36 = m32 + m33;
        s37 = in0 - m34;
        s38 = s35 - s36;

        // Output point 8: X(7)
        out[out_strides[7]] = s38 + s37;

        m35 = CRTM_11_2 * s7;
        m36 = CRTM_11_4 * s9;
        m37 = CRTM_11_6 * s6;
        m38 = CRTM_11_8 * s5;
        m39 = CRTM_11_10 * s8;

        s39 = m36 - m35;
        s40 = m37 - m38;
        s41 = s40 - m39;

        // Output point 9: X(8)
        out[out_strides[8]] = s39 + s41;

        m40 = CRTM_11_1 * s1;
        m41 = CRTM_11_3 * s3;
        m42 = CRTM_11_5 * s4;
        m43 = CRTM_11_7 * s2;
        m44 = CRTM_11_9 * s0;

        s42 = m40 + m41;
        s43 = m42 + m43;
        s44 = in0 - m44;
        s45 = s42 - s43;

        // Output point 10: X(9)
        out[out_strides[9]] = s44 + s45;

        m45 = CRTM_11_2 * s6;
        m46 = CRTM_11_4 * s8;
        m47 = CRTM_11_6 * s9;
        m48 = CRTM_11_8 * s7;
        m49 = CRTM_11_10 * s5;

        s46 = m45 + m46;
        s47 = m47 + m48;
        s48 = s46 - s47;

        // Output point 11: X(10)
        out[out_strides[10]] = s48 - m49;

        in = in + v_in_stride;
        out = out + v_out_stride;
    }
    AOCLFFTZ_LOG(DEBUG, global_logger_mode, "Exit");
}

static VOID r2hc_rfft11c_fp32_bwd(VOID *in_real, VOID *in_imag, VOID *out_real,
                                  VOID *out_imag, INTP n,
                                  aoclfftz_strides_t *strides, VOID *twd,
                                  UINT8 flag)
{
    AOCLFFTZ_LOG(DEBUG, global_logger_mode, "Enter");

    const FLOAT CRTM_11_1 = 1.682507065662362337723623297838735435026584997f;
    const FLOAT CRTM_11_2 = 1.081281634911195164215271908637383390863541216f;
    const FLOAT CRTM_11_3 = 0.830830026003772851058548298459246407048009821f;
    const FLOAT CRTM_11_4 = 1.819263990709036742823430766158056920120482102f;
    const FLOAT CRTM_11_5 = 0.284629676546570280887585337232739337582102722f;
    const FLOAT CRTM_11_6 = 1.979642883761865464752184075553437574753038744f;
    const FLOAT CRTM_11_7 = 1.309721467890570128113850144932587106367582399f;
    const FLOAT CRTM_11_8 = 1.511499148708516567548071687944688840359434890f;
    const FLOAT CRTM_11_9 = 1.918985947228994779780736114132655398124909697f;
    const FLOAT CRTM_11_10 = 0.563465113682859395422835830693233798071555798f;
    const FLOAT CRTM_11_11 = 2.000000000000000000000000000000000000000000000f;

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

    for (cnt = 0; cnt < n; cnt++)
    {
        FLOAT in0, in1, in2, in3, in4, in5, in6, in7, in8, in9, in10;
        FLOAT s0, s1, s2, s3, s4, s5, s6, s7, s8, s9, s10, s11, s12, s13, s14,
              s15, s16, s17, s18, s19, s20, s21, s22, s23, s24, s25, s26, s27,
              s28, s29, s30, s31, s32, s33, s34, s35, s36, s37, s38, s39, s40,
              s41, s42, s43, s44, s45, s46, s47, s48;
        FLOAT m0, m1, m2, m3, m4, m5, m6, m7, m8, m9, m10, m11, m12, m13, m14,
              m15, m16, m17, m18, m19, m20, m21, m22, m23, m24, m25, m26, m27,
              m28, m29, m30, m31, m32, m33, m34, m35, m36, m37, m38, m39, m40,
              m41, m42, m43, m44, m45, m46, m47, m48, m49, m50;

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

        s0 = in1 + in3;
        s1 = in5 + in7;
        s2 = s0 + s1;
        s3 = s2 + in9;
        m0 = CRTM_11_11 * s3;

        // Output point 1: x(0)
        *out = in0 + m0;

        m1 = CRTM_11_1 * in1;
        m2 = CRTM_11_2 * in2;
        m3 = CRTM_11_3 * in3;
        m4 = CRTM_11_4 * in4;
        m5 = CRTM_11_5 * in5;
        m6 = CRTM_11_6 * in6;
        m7 = CRTM_11_7 * in7;
        m8 = CRTM_11_8 * in8;
        m9 = CRTM_11_9 * in9;
        m10 = CRTM_11_10 * in10;

        s4 = m1 + m3;
        s5 = s4 + in0;
        s6 = m5 + m7;
        s7 = s6 + m9;
        s8 = s5 - s7;

        s9 = m2 + m4;
        s10 = m6 + m8;
        s11 = s9 + m10;
        s12 = s10 + s11;

        // Output point 2: x(1)
        out[out_strides[1]] = s8 - s12;
        // Output point 11: x(10)
        out[out_strides[10]] = s8 + s12;

        m11 = CRTM_11_1 * in9;
        m12 = CRTM_11_2 * in10;
        m13 = CRTM_11_3 * in1;
        m14 = CRTM_11_4 * in2;
        m15 = CRTM_11_5 * in7;
        m16 = CRTM_11_6 * in8;
        m17 = CRTM_11_7 * in3;
        m18 = CRTM_11_8 * in4;
        m19 = CRTM_11_9 * in5;
        m20 = CRTM_11_10 * in6;

        s13 = m11 + m13;
        s14 = s13 + in0;
        s15 = m15 + m17;
        s16 = s15 + m19;
        s17 = s14 - s16;

        s18 = m12 - m14;
        s19 = m16 - m18;
        s20 = s19 + m20;
        s21 = s18 + s20;

        // Output point 3: x(2)
        out[out_strides[2]] = s17 + s21;
        // Output point 10: x(9)
        out[out_strides[9]] = s17 - s21;

        m21 = CRTM_11_1 * in7;
        m22 = CRTM_11_2 * in8;
        m23 = CRTM_11_3 * in5;
        m24 = CRTM_11_4 * in6;
        m25 = CRTM_11_5 * in1;
        m26 = CRTM_11_6 * in2;
        m27 = CRTM_11_7 * in9;
        m28 = CRTM_11_8 * in10;
        m29 = CRTM_11_9 * in3;
        m30 = CRTM_11_10 * in4;

        s22 = m21 + m23;
        s23 = s22 + in0;
        s24 = m25 + m27;
        s25 = s24 + m29;
        s26 = s23 - s25;

        s27 = m22 - m24;
        s28 = m26 + m28;
        s29 = s28 - m30;
        s30 = s27 + s29;

        // Output point 4: x(3)
        out[out_strides[3]] = s26 - s30;
        // Output point 9: x(8)
        out[out_strides[8]] = s26 + s30;

        m31 = CRTM_11_1 * in5;
        m32 = CRTM_11_2 * in6;
        m33 = CRTM_11_3 * in9;
        m34 = CRTM_11_4 * in10;
        m35 = CRTM_11_5 * in3;
        m36 = CRTM_11_6 * in4;
        m37 = CRTM_11_7 * in1;
        m38 = CRTM_11_8 * in2;
        m39 = CRTM_11_9 * in7;
        m40 = CRTM_11_10 * in8;

        s31 = m31 + m33;
        s32 = s31 + in0;
        s33 = m35 + m37;
        s34 = s33 + m39;
        s35 = s32 - s34;

        s36 = m32 - m34;
        s37 = m38 - m36;
        s38 = s37 + m40;
        s39 = s36 + s38;

        // Output point 5: x(4)
        out[out_strides[4]] = s35 - s39;
        // Output point 8: x(7)
        out[out_strides[7]] = s35 + s39;

        m41 = CRTM_11_1 * in3;
        m42 = CRTM_11_2 * in4;
        m43 = CRTM_11_3 * in7;
        m44 = CRTM_11_4 * in8;
        m45 = CRTM_11_5 * in9;
        m46 = CRTM_11_6 * in10;
        m47 = CRTM_11_7 * in5;
        m48 = CRTM_11_8 * in6;
        m49 = CRTM_11_9 * in1;
        m50 = CRTM_11_10 * in2;

        s40 = m41 + m43;
        s41 = s40 + in0;
        s42 = m45 + m47;
        s43 = s42 + m49;
        s44 = s41 - s43;

        s45 = m42 + m44;
        s46 = m46 + m48;
        s47 = s45 - s46;
        s48 = s47 - m50;

        // Output point 6: x(5)
        out[out_strides[5]] = s44 + s48;
        // Output point 7: x(6)
        out[out_strides[6]] = s44 - s48;

        in = in + v_in_stride;
        out = out + v_out_stride;
    }
    AOCLFFTZ_LOG(DEBUG, global_logger_mode, "Exit");
}

static VOID r2hc_rfft11c_fp64_fwd(VOID *in_real, VOID *in_imag, VOID *out_real,
                                  VOID *out_imag, INTP n,
                                  aoclfftz_strides_t *strides, VOID *twd,
                                  UINT8 flag)
{
    AOCLFFTZ_LOG(DEBUG, global_logger_mode, "Enter");

    const DOUBLE CRTM_11_1 = 0.841253532831181168861811648919367717513292498;
    const DOUBLE CRTM_11_2 = 0.540640817455597582107635954318691695431770608;
    const DOUBLE CRTM_11_3 = 0.415415013001886425529274149229623203524004910;
    const DOUBLE CRTM_11_4 = 0.909631995354518371411715383079028460060241051;
    const DOUBLE CRTM_11_5 = 0.142314838273285140443792668616369668791051361;
    const DOUBLE CRTM_11_6 = 0.989821441880932732376092037776718787376519372;
    const DOUBLE CRTM_11_7 = 0.654860733945285064056925072466293553183791199;
    const DOUBLE CRTM_11_8 = 0.755749574354258283774035843972344420179717445;
    const DOUBLE CRTM_11_9 = 0.959492973614497389890368057066327699062454848;
    const DOUBLE CRTM_11_10 = 0.281732556841429697711417915346616899035777899;

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

    for (cnt = 0; cnt < n; cnt++)
    {
        DOUBLE in0, in1, in2, in3, in4, in5, in6, in7, in8, in9, in10;
        DOUBLE s0, s1, s2, s3, s4, s5, s6, s7, s8, s9, s10, s11, s12, s13, s14,
               s15, s16, s17, s18, s19, s20, s21, s22, s23, s24, s25, s26, s27,
               s28, s29, s30, s31, s32, s33, s34, s35, s36, s37, s38, s39, s40,
               s41, s42, s43, s44, s45, s46, s47, s48;
        DOUBLE m0, m1, m2, m3, m4, m5, m6, m7, m8, m9, m10, m11, m12, m13, m14,
               m15, m16, m17, m18, m19, m20, m21, m22, m23, m24, m25, m26, m27,
               m28, m29, m30, m31, m32, m33, m34, m35, m36, m37, m38, m39, m40,
               m41, m42, m43, m44, m45, m46, m47, m48, m49;

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

        s0 = in1 + in10;
        s1 = in2 + in9;
        s2 = in3 + in8;
        s3 = in4 + in7;
        s4 = in5 + in6;
        s5 = in1 - in10;
        s6 = in2 - in9;
        s7 = in3 - in8;
        s8 = in4 - in7;
        s9 = in5 - in6;

        s10 = s0 + s1;
        s11 = s2 + s3;
        s12 = s4 + in0;
        s13 = s10 + s11;

        // Output point 1: X(0)
        *out = s12 + s13;

        m0 = CRTM_11_1 * s0;
        m1 = CRTM_11_3 * s1;
        m2 = CRTM_11_5 * s2;
        m3 = CRTM_11_7 * s3;
        m4 = CRTM_11_9 * s4;

        s14 = m0 + m1;
        s15 = m2 + m3;
        s16 = in0 - m4;
        s17 = s14 - s15;

        // Output point 2: X(1)
        out[out_strides[1]] = s16 + s17;

        m5 = CRTM_11_2 * s5;
        m6 = CRTM_11_4 * s6;
        m7 = CRTM_11_6 * s7;
        m8 = CRTM_11_8 * s8;
        m9 = CRTM_11_10 * s9;

        s18 = m5 + m6;
        s19 = m7 + m8;
        s20 = s19 + m9;

        // Output point 3: X(2)
        out[out_strides[2]] = -(s18 + s20);

        m10 = CRTM_11_1 * s4;
        m11 = CRTM_11_3 * s0;
        m12 = CRTM_11_5 * s3;
        m13 = CRTM_11_7 * s1;
        m14 = CRTM_11_9 * s2;

        s21 = m10 + m11;
        s22 = m12 + m13;
        s23 = in0 - m14;
        s24 = s21 - s22;

        // Output point 4: X(3)
        out[out_strides[3]] = s23 + s24;

        m15 = CRTM_11_2 * s9;
        m16 = CRTM_11_4 * s5;
        m17 = CRTM_11_6 * s8;
        m18 = CRTM_11_8 * s6;
        m19 = CRTM_11_10 * s7;

        s25 = m15 - m16;
        s26 = m17 - m18;
        s27 = s26 + m19;

        // Output point 5: X(4)
        out[out_strides[4]] = s25 + s27;

        m20 = CRTM_11_1 * s3;
        m21 = CRTM_11_3 * s2;
        m22 = CRTM_11_5 * s0;
        m23 = CRTM_11_7 * s4;
        m24 = CRTM_11_9 * s1;

        s28 = m20 + m21;
        s29 = m22 + m23;
        s30 = in0 - m24;
        s31 = s30 - s29;

        // Output point 6: X(5)
        out[out_strides[5]] = s28 + s31;

        m25 = CRTM_11_2 * s8;
        m26 = CRTM_11_4 * s7;
        m27 = CRTM_11_6 * s5;
        m28 = CRTM_11_8 * s9;
        m29 = CRTM_11_10 * s6;

        s32 = m26 - m25;
        s33 = m27 + m28;
        s34 = s32 - s33;

        // Output point 7: X(6)
        out[out_strides[6]] = s34 + m29;

        m30 = CRTM_11_1 * s2;
        m31 = CRTM_11_3 * s4;
        m32 = CRTM_11_5 * s1;
        m33 = CRTM_11_7 * s0;
        m34 = CRTM_11_9 * s3;

        s35 = m30 + m31;
        s36 = m32 + m33;
        s37 = in0 - m34;
        s38 = s35 - s36;

        // Output point 8: X(7)
        out[out_strides[7]] = s38 + s37;

        m35 = CRTM_11_2 * s7;
        m36 = CRTM_11_4 * s9;
        m37 = CRTM_11_6 * s6;
        m38 = CRTM_11_8 * s5;
        m39 = CRTM_11_10 * s8;

        s39 = m36 - m35;
        s40 = m37 - m38;
        s41 = s40 - m39;

        // Output point 9: X(8)
        out[out_strides[8]] = s39 + s41;

        m40 = CRTM_11_1 * s1;
        m41 = CRTM_11_3 * s3;
        m42 = CRTM_11_5 * s4;
        m43 = CRTM_11_7 * s2;
        m44 = CRTM_11_9 * s0;

        s42 = m40 + m41;
        s43 = m42 + m43;
        s44 = in0 - m44;
        s45 = s42 - s43;

        // Output point 10: X(9)
        out[out_strides[9]] = s44 + s45;

        m45 = CRTM_11_2 * s6;
        m46 = CRTM_11_4 * s8;
        m47 = CRTM_11_6 * s9;
        m48 = CRTM_11_8 * s7;
        m49 = CRTM_11_10 * s5;

        s46 = m45 + m46;
        s47 = m47 + m48;
        s48 = s46 - s47;

        // Output point 11: X(10)
        out[out_strides[10]] = s48 - m49;

        in = in + v_in_stride;
        out = out + v_out_stride;
    }
    AOCLFFTZ_LOG(DEBUG, global_logger_mode, "Exit");
}

static VOID r2hc_rfft11c_fp64_bwd(VOID *in_real, VOID *in_imag, VOID *out_real,
                                  VOID *out_imag, INTP n,
                                  aoclfftz_strides_t *strides, VOID *twd,
                                  UINT8 flag)
{
    AOCLFFTZ_LOG(DEBUG, global_logger_mode, "Enter");

    const DOUBLE CRTM_11_1 = 1.682507065662362337723623297838735435026584997;
    const DOUBLE CRTM_11_2 = 1.081281634911195164215271908637383390863541216;
    const DOUBLE CRTM_11_3 = 0.830830026003772851058548298459246407048009821;
    const DOUBLE CRTM_11_4 = 1.819263990709036742823430766158056920120482102;
    const DOUBLE CRTM_11_5 = 0.284629676546570280887585337232739337582102722;
    const DOUBLE CRTM_11_6 = 1.979642883761865464752184075553437574753038744;
    const DOUBLE CRTM_11_7 = 1.309721467890570128113850144932587106367582399;
    const DOUBLE CRTM_11_8 = 1.511499148708516567548071687944688840359434890;
    const DOUBLE CRTM_11_9 = 1.918985947228994779780736114132655398124909697;
    const DOUBLE CRTM_11_10 = 0.563465113682859395422835830693233798071555798;
    const DOUBLE CRTM_11_11 = 2.000000000000000000000000000000000000000000000;

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

    for (cnt = 0; cnt < n; cnt++)
    {
        DOUBLE in0, in1, in2, in3, in4, in5, in6, in7, in8, in9, in10;
        DOUBLE s0, s1, s2, s3, s4, s5, s6, s7, s8, s9, s10, s11, s12, s13, s14,
               s15, s16, s17, s18, s19, s20, s21, s22, s23, s24, s25, s26, s27,
               s28, s29, s30, s31, s32, s33, s34, s35, s36, s37, s38, s39, s40,
               s41, s42, s43, s44, s45, s46, s47, s48;
        DOUBLE m0, m1, m2, m3, m4, m5, m6, m7, m8, m9, m10, m11, m12, m13, m14,
               m15, m16, m17, m18, m19, m20, m21, m22, m23, m24, m25, m26, m27,
               m28, m29, m30, m31, m32, m33, m34, m35, m36, m37, m38, m39, m40,
               m41, m42, m43, m44, m45, m46, m47, m48, m49, m50;

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

        s0 = in1 + in3;
        s1 = in5 + in7;
        s2 = s0 + s1;
        s3 = s2 + in9;
        m0 = CRTM_11_11 * s3;

        // Output point 1: x(0)
        *out = in0 + m0;

        m1 = CRTM_11_1 * in1;
        m2 = CRTM_11_2 * in2;
        m3 = CRTM_11_3 * in3;
        m4 = CRTM_11_4 * in4;
        m5 = CRTM_11_5 * in5;
        m6 = CRTM_11_6 * in6;
        m7 = CRTM_11_7 * in7;
        m8 = CRTM_11_8 * in8;
        m9 = CRTM_11_9 * in9;
        m10 = CRTM_11_10 * in10;

        s4 = m1 + m3;
        s5 = s4 + in0;
        s6 = m5 + m7;
        s7 = s6 + m9;
        s8 = s5 - s7;

        s9 = m2 + m4;
        s10 = m6 + m8;
        s11 = s9 + m10;
        s12 = s10 + s11;

        // Output point 2: x(1)
        out[out_strides[1]] = s8 - s12;
        // Output point 11: x(10)
        out[out_strides[10]] = s8 + s12;

        m11 = CRTM_11_1 * in9;
        m12 = CRTM_11_2 * in10;
        m13 = CRTM_11_3 * in1;
        m14 = CRTM_11_4 * in2;
        m15 = CRTM_11_5 * in7;
        m16 = CRTM_11_6 * in8;
        m17 = CRTM_11_7 * in3;
        m18 = CRTM_11_8 * in4;
        m19 = CRTM_11_9 * in5;
        m20 = CRTM_11_10 * in6;

        s13 = m11 + m13;
        s14 = s13 + in0;
        s15 = m15 + m17;
        s16 = s15 + m19;
        s17 = s14 - s16;

        s18 = m12 - m14;
        s19 = m16 - m18;
        s20 = s19 + m20;
        s21 = s18 + s20;

        // Output point 3: x(2)
        out[out_strides[2]] = s17 + s21;
        // Output point 10: x(9)
        out[out_strides[9]] = s17 - s21;

        m21 = CRTM_11_1 * in7;
        m22 = CRTM_11_2 * in8;
        m23 = CRTM_11_3 * in5;
        m24 = CRTM_11_4 * in6;
        m25 = CRTM_11_5 * in1;
        m26 = CRTM_11_6 * in2;
        m27 = CRTM_11_7 * in9;
        m28 = CRTM_11_8 * in10;
        m29 = CRTM_11_9 * in3;
        m30 = CRTM_11_10 * in4;

        s22 = m21 + m23;
        s23 = s22 + in0;
        s24 = m25 + m27;
        s25 = s24 + m29;
        s26 = s23 - s25;

        s27 = m22 - m24;
        s28 = m26 + m28;
        s29 = s28 - m30;
        s30 = s27 + s29;

        // Output point 4: x(3)
        out[out_strides[3]] = s26 - s30;
        // Output point 9: x(8)
        out[out_strides[8]] = s26 + s30;

        m31 = CRTM_11_1 * in5;
        m32 = CRTM_11_2 * in6;
        m33 = CRTM_11_3 * in9;
        m34 = CRTM_11_4 * in10;
        m35 = CRTM_11_5 * in3;
        m36 = CRTM_11_6 * in4;
        m37 = CRTM_11_7 * in1;
        m38 = CRTM_11_8 * in2;
        m39 = CRTM_11_9 * in7;
        m40 = CRTM_11_10 * in8;

        s31 = m31 + m33;
        s32 = s31 + in0;
        s33 = m35 + m37;
        s34 = s33 + m39;
        s35 = s32 - s34;

        s36 = m32 - m34;
        s37 = m38 - m36;
        s38 = s37 + m40;
        s39 = s36 + s38;

        // Output point 5: x(4)
        out[out_strides[4]] = s35 - s39;
        // Output point 8: x(7)
        out[out_strides[7]] = s35 + s39;

        m41 = CRTM_11_1 * in3;
        m42 = CRTM_11_2 * in4;
        m43 = CRTM_11_3 * in7;
        m44 = CRTM_11_4 * in8;
        m45 = CRTM_11_5 * in9;
        m46 = CRTM_11_6 * in10;
        m47 = CRTM_11_7 * in5;
        m48 = CRTM_11_8 * in6;
        m49 = CRTM_11_9 * in1;
        m50 = CRTM_11_10 * in2;

        s40 = m41 + m43;
        s41 = s40 + in0;
        s42 = m45 + m47;
        s43 = s42 + m49;
        s44 = s41 - s43;

        s45 = m42 + m44;
        s46 = m46 + m48;
        s47 = s45 - s46;
        s48 = s47 - m50;

        // Output point 6: x(5)
        out[out_strides[5]] = s44 + s48;
        // Output point 7: x(6)
        out[out_strides[6]] = s44 - s48;

        in = in + v_in_stride;
        out = out + v_out_stride;
    }
    AOCLFFTZ_LOG(DEBUG, global_logger_mode, "Exit");
}

kfft_ register_kernel_r2hc_rfft11c(UINT8 precision, UINT8 direction)
{
    if (direction == FORWARD_FFT_DIR)
    {
        if (precision == DT_FLOAT)
        {
            return r2hc_rfft11c_fp32_fwd;
        }
        else if (precision == DT_DOUBLE)
        {
            return r2hc_rfft11c_fp64_fwd;
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
            return r2hc_rfft11c_fp32_bwd;
        }
        else if (precision == DT_DOUBLE)
        {
            return r2hc_rfft11c_fp64_bwd;
        }
        else
        {
            return NULL;
        }
    }
}
