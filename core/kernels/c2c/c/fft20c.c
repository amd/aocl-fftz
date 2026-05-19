/**
 * Copyright (C) 2026, Advanced Micro Devices. All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 * 1. Redistributions of source code must retain the above copyright notice,
 * this list of conditions and the following disclaimer.
 * 2. Redistributions in binary form must reproduce the above copyright notice,
 * this list of conditions and the following disclaimer in the documentation
 * and/or other materials provided with the distribution.
 * 3. Neither the name of the copyright holder nor the names of its
 * contributors may be used to endorse or promote products derived from this
 * software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
 * ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
 * LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
 * CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
 * SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
 * INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
 * CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
 * ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
 * POSSIBILITY OF SUCH DAMAGE.
 */

/** @file fft20c.c
 *
 *  @brief Radix-20 FFT kernel using direct DFT equations in C
 *
 *  This file contains the DIT radix-20 FFT implementations using direct
 *  DFT equations generated from kernel_generation.py script.
 *
 *  @author Jeya R
 *  @author Jeevanantham N
 */

#include "core/kernels/kernel.h"

static const ops_cycles_t ops_cnt[NUM_PRECISIONS] = {{0, 48, 208, 80, 0, 0},
                                                     {0, 48, 208, 80, 0, 0}};

ops_cycles_t get_ops_cnt_fft20c(UINT8 precision, UINT8 direction)
{
    if (precision == DT_FLOAT)
    {
        return ops_cnt[0];
    }
    else
    {
        return ops_cnt[1];
    }
}

static VOID fft20c_fp64(VOID *in_real, VOID *in_imag, VOID *out_real,
                        VOID *out_imag, INTP n, aoclfftz_strides_t *strides,
                        VOID *twd, UINT8 flag)
{
    AOCLFFTZ_LOG(DEBUG, global_logger_mode, "Enter");

    const DOUBLE CRTM_20_0 =
        0.95105651629515357211643933337876088917255401611328;
    const DOUBLE CRTM_20_3 =
        0.58778525229247312916870595463505014777183532714844;
    const DOUBLE CRTM_20_K1 =
        0.55901699437494742410229341718281905886015458990288;
    const DOUBLE CRTM_20_K2 =
        0.25000000000000000000000000000000000000000000000000;

    DOUBLE *in_r = NULL;
    DOUBLE *in_i = NULL;
    DOUBLE *out_r = NULL;
    DOUBLE *out_i = NULL;

#ifdef VOLATILE_STRIDE_ARRAY
    volatile INTP *in_strides = strides->in_strides;
    volatile INTP *out_strides = strides->out_strides;
#else
    INTP *in_strides = strides->in_strides;
    INTP *out_strides = strides->out_strides;
#endif
    INTP v_in_stride = strides->v_in_stride;
    INTP v_out_stride = strides->v_out_stride;
    INTP cnt = 0;

    if (flag) /* non-zero flag indicates that the fft is inverse */
    {
        in_r = (DOUBLE *)in_imag;
        in_i = (DOUBLE *)in_real;
        out_r = (DOUBLE *)out_imag;
        out_i = (DOUBLE *)out_real;
    }
    else
    {
        in_r = (DOUBLE *)in_real;
        in_i = (DOUBLE *)in_imag;
        out_r = (DOUBLE *)out_real;
        out_i = (DOUBLE *)out_imag;
    }

    for (cnt = 0; cnt < n; cnt++)
    {
        /* Load all 20 complex inputs */
        DOUBLE x_0_r  = in_r[in_strides[0]];
        DOUBLE x_0_i  = in_i[in_strides[0]];
        DOUBLE x_1_r  = in_r[in_strides[1]];
        DOUBLE x_1_i  = in_i[in_strides[1]];
        DOUBLE x_2_r  = in_r[in_strides[2]];
        DOUBLE x_2_i  = in_i[in_strides[2]];
        DOUBLE x_3_r  = in_r[in_strides[3]];
        DOUBLE x_3_i  = in_i[in_strides[3]];
        DOUBLE x_4_r  = in_r[in_strides[4]];
        DOUBLE x_4_i  = in_i[in_strides[4]];
        DOUBLE x_5_r  = in_r[in_strides[5]];
        DOUBLE x_5_i  = in_i[in_strides[5]];
        DOUBLE x_6_r  = in_r[in_strides[6]];
        DOUBLE x_6_i  = in_i[in_strides[6]];
        DOUBLE x_7_r  = in_r[in_strides[7]];
        DOUBLE x_7_i  = in_i[in_strides[7]];
        DOUBLE x_8_r  = in_r[in_strides[8]];
        DOUBLE x_8_i  = in_i[in_strides[8]];
        DOUBLE x_9_r  = in_r[in_strides[9]];
        DOUBLE x_9_i  = in_i[in_strides[9]];
        DOUBLE x_10_r = in_r[in_strides[10]];
        DOUBLE x_10_i = in_i[in_strides[10]];
        DOUBLE x_11_r = in_r[in_strides[11]];
        DOUBLE x_11_i = in_i[in_strides[11]];
        DOUBLE x_12_r = in_r[in_strides[12]];
        DOUBLE x_12_i = in_i[in_strides[12]];
        DOUBLE x_13_r = in_r[in_strides[13]];
        DOUBLE x_13_i = in_i[in_strides[13]];
        DOUBLE x_14_r = in_r[in_strides[14]];
        DOUBLE x_14_i = in_i[in_strides[14]];
        DOUBLE x_15_r = in_r[in_strides[15]];
        DOUBLE x_15_i = in_i[in_strides[15]];
        DOUBLE x_16_r = in_r[in_strides[16]];
        DOUBLE x_16_i = in_i[in_strides[16]];
        DOUBLE x_17_r = in_r[in_strides[17]];
        DOUBLE x_17_i = in_i[in_strides[17]];
        DOUBLE x_18_r = in_r[in_strides[18]];
        DOUBLE x_18_i = in_i[in_strides[18]];
        DOUBLE x_19_r = in_r[in_strides[19]];
        DOUBLE x_19_i = in_i[in_strides[19]];

        DOUBLE t1 = x_0_r + x_10_r;
        DOUBLE t2 = x_0_r - x_10_r;
        DOUBLE t3 = x_5_r + x_15_r;
        DOUBLE t4 = x_5_r - x_15_r;
        DOUBLE t5 = x_0_i + x_10_i;
        DOUBLE t6 = x_0_i - x_10_i;
        DOUBLE t7 = x_5_i + x_15_i;
        DOUBLE t8 = x_5_i - x_15_i;
        DOUBLE t9 = x_14_r + x_16_r;
        DOUBLE t10 = x_14_r - x_16_r;
        DOUBLE t11 = x_4_r + x_6_r;
        DOUBLE t12 = x_4_r - x_6_r;
        DOUBLE t13 = x_11_i + x_19_i;
        DOUBLE t14 = x_11_i - x_19_i;
        DOUBLE t15 = x_1_i + x_9_i;
        DOUBLE t16 = x_9_i - x_1_i;
        DOUBLE t17 = x_14_i + x_16_i;
        DOUBLE t18 = x_14_i - x_16_i;
        DOUBLE t19 = x_4_i + x_6_i;
        DOUBLE t20 = x_4_i - x_6_i;
        DOUBLE t21 = x_11_r + x_19_r;
        DOUBLE t22 = x_11_r - x_19_r;
        DOUBLE t23 = x_1_r + x_9_r;
        DOUBLE t24 = x_9_r - x_1_r;
        DOUBLE t25 = x_12_r + x_18_r;
        DOUBLE t26 = x_18_r - x_12_r;
        DOUBLE t27 = x_12_i + x_18_i;
        DOUBLE t28 = x_18_i - x_12_i;
        DOUBLE t29 = x_2_r + x_8_r;
        DOUBLE t30 = x_8_r - x_2_r;
        DOUBLE t31 = x_2_i + x_8_i;
        DOUBLE t32 = x_8_i - x_2_i;
        DOUBLE t33 = x_3_r + x_7_r;
        DOUBLE t34 = x_3_r - x_7_r;
        DOUBLE t35 = x_3_i + x_7_i;
        DOUBLE t36 = x_3_i - x_7_i;
        DOUBLE t37 = x_13_r + x_17_r;
        DOUBLE t38 = x_13_r - x_17_r;
        DOUBLE t39 = x_13_i + x_17_i;
        DOUBLE t40 = x_13_i - x_17_i;

        DOUBLE cv1 = t1 + t3;
        DOUBLE cv2 = t1 - t3;
        DOUBLE cv3 = t5 + t7;
        DOUBLE cv4 = t5 - t7;
        DOUBLE cv5 = t2 + t8;
        DOUBLE cv6 = t2 - t8;
        DOUBLE cv7 = t6 + t4;
        DOUBLE cv8 = t6 - t4;
        DOUBLE cv9 = t9 + t11;
        DOUBLE cv10 = t9 - t11;
        DOUBLE cv11 = t10 + t12;
        DOUBLE cv12 = t10 - t12;
        DOUBLE cv13 = t14 + t16;
        DOUBLE cv14 = t14 - t16;
        DOUBLE cv15 = t13 + t15;
        DOUBLE cv16 = t13 - t15;
        DOUBLE cv17 = t17 + t19;
        DOUBLE cv18 = t17 - t19;
        DOUBLE cv19 = t20 + t18;
        DOUBLE cv20 = t20 - t18;
        DOUBLE cv21 = t22 + t24;
        DOUBLE cv22 = t22 - t24;
        DOUBLE cv23 = t21 + t23;
        DOUBLE cv24 = t21 - t23;
        DOUBLE cv25 = t25 + t29;
        DOUBLE cv26 = t25 - t29;
        DOUBLE cv27 = t26 + t30;
        DOUBLE cv28 = t26 - t30;
        DOUBLE cv29 = t27 + t31;
        DOUBLE cv30 = t27 - t31;
        DOUBLE cv31 = t28 + t32;
        DOUBLE cv32 = t28 - t32;
        DOUBLE cv33 = t33 + t37;
        DOUBLE cv34 = t33 - t37;
        DOUBLE cv35 = t35 + t39;
        DOUBLE cv36 = t35 - t39;
        DOUBLE cv37 = t36 + t40;
        DOUBLE cv38 = t36 - t40;
        DOUBLE cv39 = t34 + t38;
        DOUBLE cv40 = t34 - t38;
        DOUBLE cv41 = cv21 + cv18;
        DOUBLE cv42 = cv21 - cv18;
        DOUBLE cv43 = cv16 + cv12;
        DOUBLE cv44 = cv16 - cv12;
        DOUBLE cv45 = cv28 + cv36;
        DOUBLE cv46 = cv28 - cv36;
        DOUBLE cv47 = cv30 + cv40;
        DOUBLE cv48 = cv30 - cv40;
        DOUBLE cv49 = cv10 + cv13;
        DOUBLE cv50 = cv10 - cv13;
        DOUBLE cv51 = cv24 + cv20;
        DOUBLE cv52 = cv24 - cv20;
        DOUBLE cv53 = cv31 + cv37;
        DOUBLE cv54 = cv37 - cv31;
        DOUBLE cv55 = cv32 + cv34;
        DOUBLE cv56 = cv32 - cv34;
        DOUBLE cv57 = cv26 + cv38;
        DOUBLE cv58 = cv26 - cv38;
        DOUBLE cv59 = cv25 + cv33;
        DOUBLE cv60 = cv25 - cv33;
        DOUBLE cv61 = cv23 + cv9;
        DOUBLE cv62 = cv23 - cv9;
        DOUBLE cv63 = cv14 + cv19;
        DOUBLE cv64 = cv14 - cv19;
        DOUBLE cv65 = cv27 + cv39;
        DOUBLE cv66 = cv27 - cv39;
        DOUBLE cv67 = cv29 + cv35;
        DOUBLE cv68 = cv29 - cv35;
        DOUBLE cv69 = cv15 + cv17;
        DOUBLE cv70 = cv15 - cv17;
        DOUBLE cv71 = cv11 + cv22;
        DOUBLE cv72 = cv11 - cv22;

        DOUBLE s1 = cv43 + cv45;
        DOUBLE d1 = cv45 - cv43;
        DOUBLE s2 = cv61 + cv59;
        DOUBLE d2 = cv61 - cv59;
        DOUBLE s3 = cv69 + cv67;
        DOUBLE d3 = cv69 - cv67;
        DOUBLE s4 = cv55 + cv52;
        DOUBLE d4 = cv52 - cv55;
        DOUBLE s5 = cv51 + cv56;
        DOUBLE d5 = cv51 - cv56;
        DOUBLE s6 = cv60 + cv62;
        DOUBLE d6 = cv60 - cv62;
        DOUBLE s7 = cv68 + cv70;
        DOUBLE d7 = cv68 - cv70;
        DOUBLE s8 = cv44 + cv46;
        DOUBLE d8 = cv46 - cv44;

        DOUBLE k1_d1 = CRTM_20_K1 * d1;
        DOUBLE k2_s1 = CRTM_20_K2 * s1;
        DOUBLE k1_d2 = CRTM_20_K1 * d2;
        DOUBLE k2_s2 = CRTM_20_K2 * s2;
        DOUBLE k1_d3 = CRTM_20_K1 * d3;
        DOUBLE k2_s3 = CRTM_20_K2 * s3;
        DOUBLE k1_d4 = CRTM_20_K1 * d4;
        DOUBLE k2_s4 = CRTM_20_K2 * s4;
        DOUBLE k1_s5 = CRTM_20_K1 * s5;
        DOUBLE k2_d5 = CRTM_20_K2 * d5;
        DOUBLE k1_s6 = CRTM_20_K1 * s6;
        DOUBLE k2_d6 = CRTM_20_K2 * d6;
        DOUBLE k1_s7 = CRTM_20_K1 * s7;
        DOUBLE k2_d7 = CRTM_20_K2 * d7;
        DOUBLE k1_s8 = CRTM_20_K1 * s8;
        DOUBLE k2_d8 = CRTM_20_K2 * d8;

        DOUBLE base1 = k2_s1 + cv5;
        DOUBLE base2 = cv1 - k2_s2;
        DOUBLE base3 = cv3 - k2_s3;
        DOUBLE base4 = cv7 + k2_s4;
        DOUBLE base5 = cv8 - k2_d5;
        DOUBLE base6 = cv2 - k2_d6;
        DOUBLE base7 = cv4 - k2_d7;
        DOUBLE base8 = cv6 + k2_d8;

        DOUBLE cv73 = base1 + k1_d1;
        DOUBLE cv101 = base1 - k1_d1;
        DOUBLE cv85 = base2 + k1_d2;
        DOUBLE cv93 = base2 - k1_d2;
        DOUBLE cv87 = base3 + k1_d3;
        DOUBLE cv95 = base3 - k1_d3;
        DOUBLE cv83 = base4 + k1_d4;
        DOUBLE cv99 = base4 - k1_d4;
        DOUBLE cv75 = base5 + k1_s5;
        DOUBLE cv103 = base5 - k1_s5;
        DOUBLE cv77 = base6 + k1_s6;
        DOUBLE cv89 = base6 - k1_s6;
        DOUBLE cv79 = base7 + k1_s7;
        DOUBLE cv91 = base7 - k1_s7;
        DOUBLE cv81 = base8 - k1_s8;
        DOUBLE cv97 = base8 + k1_s8;

        DOUBLE cv74 = CRTM_20_0*(-cv41) + CRTM_20_3*(-cv48);
        DOUBLE cv76 = CRTM_20_0*(cv50) + CRTM_20_3*(cv57);
        DOUBLE cv78 = CRTM_20_0*(cv54) + CRTM_20_3*(cv63);
        DOUBLE cv80 = CRTM_20_0*(cv66) + CRTM_20_3*(-cv71);
        DOUBLE cv82 = CRTM_20_0*(-cv47) + CRTM_20_3*(-cv42);
        DOUBLE cv84 = CRTM_20_0*(cv58) + CRTM_20_3*(-cv49);
        DOUBLE cv86 = CRTM_20_0*(cv64) + CRTM_20_3*(-cv53);
        DOUBLE cv88 = CRTM_20_0*(cv72) + CRTM_20_3*(cv65);
        DOUBLE cv90 = CRTM_20_0*(cv63) + CRTM_20_3*(-cv54);
        DOUBLE cv92 = CRTM_20_0*(-cv71) + CRTM_20_3*(-cv66);
        DOUBLE cv94 = CRTM_20_0*(cv53) + CRTM_20_3*(cv64);
        DOUBLE cv96 = CRTM_20_0*(-cv65) + CRTM_20_3*(cv72);
        DOUBLE cv98 = CRTM_20_0*(cv42) + CRTM_20_3*(-cv47);
        DOUBLE cv100 = CRTM_20_0*(cv49) + CRTM_20_3*(cv58);
        DOUBLE cv102 = CRTM_20_0*(-cv48) + CRTM_20_3*(cv41);
        DOUBLE cv104 = CRTM_20_0*(cv57) + CRTM_20_3*(-cv50);

        /* X[0] */
        out_r[out_strides[0]] = cv1 + s2;
        out_i[out_strides[0]] = cv3 + s3;

        /* X[1] */
        out_r[out_strides[1]] = cv73 + cv74;
        out_i[out_strides[1]] = cv75 + cv76;

        /* X[2] */
        out_r[out_strides[2]] = cv77 + cv78;
        out_i[out_strides[2]] = cv79 + cv80;

        /* X[3] */
        out_r[out_strides[3]] = cv81 + cv82;
        out_i[out_strides[3]] = cv83 + cv84;

        /* X[4] */
        out_r[out_strides[4]] = cv85 + cv86;
        out_i[out_strides[4]] = cv87 + cv88;

        /* X[5] */
        out_r[out_strides[5]] = cv5 - s1;
        out_i[out_strides[5]] = cv8 + d5;

        /* X[6] */
        out_r[out_strides[6]] = cv89 + cv90;
        out_i[out_strides[6]] = cv91 + cv92;

        /* X[7] */
        out_r[out_strides[7]] = cv81 - cv82;
        out_i[out_strides[7]] = cv83 - cv84;

        /* X[8] */
        out_r[out_strides[8]] = cv93 + cv94;
        out_i[out_strides[8]] = cv95 + cv96;

        /* X[9] */
        out_r[out_strides[9]] = cv73 - cv74;
        out_i[out_strides[9]] = cv75 - cv76;

        /* X[10] */
        out_r[out_strides[10]] = cv2 + d6;
        out_i[out_strides[10]] = cv4 + d7;

        /* X[11] */
        out_r[out_strides[11]] = cv97 + cv98;
        out_i[out_strides[11]] = cv99 + cv100;

        /* X[12] */
        out_r[out_strides[12]] = cv93 - cv94;
        out_i[out_strides[12]] = cv95 - cv96;

        /* X[13] */
        out_r[out_strides[13]] = cv101 + cv102;
        out_i[out_strides[13]] = cv103 + cv104;

        /* X[14] */
        out_r[out_strides[14]] = cv89 - cv90;
        out_i[out_strides[14]] = cv91 - cv92;

        /* X[15] */
        out_r[out_strides[15]] = cv6 - d8;
        out_i[out_strides[15]] = cv7 - s4;

        /* X[16] */
        out_r[out_strides[16]] = cv85 - cv86;
        out_i[out_strides[16]] = cv87 - cv88;

        /* X[17] */
        out_r[out_strides[17]] = cv101 - cv102;
        out_i[out_strides[17]] = cv103 - cv104;

        /* X[18] */
        out_r[out_strides[18]] = cv77 - cv78;
        out_i[out_strides[18]] = cv79 - cv80;

        /* X[19] */
        out_r[out_strides[19]] = cv97 - cv98;
        out_i[out_strides[19]] = cv99 - cv100;

        in_r += v_in_stride;
        in_i += v_in_stride;
        out_r += v_out_stride;
        out_i += v_out_stride;
    }

    AOCLFFTZ_LOG(DEBUG, global_logger_mode, "Exit");
}

static VOID fft20c_fp32(VOID *in_real, VOID *in_imag, VOID *out_real,
                        VOID *out_imag, INTP n, aoclfftz_strides_t *strides,
                        VOID *twd, UINT8 flag)
{
    AOCLFFTZ_LOG(DEBUG, global_logger_mode, "Enter");

    const FLOAT CRTM_20_0 =
        0.95105651629515357211643933337876088917255401611328f;
    const FLOAT CRTM_20_3 =
        0.58778525229247312916870595463505014777183532714844f;
    const FLOAT CRTM_20_K1 =
        0.55901699437494742410229341718281905886015458990288f;
    const FLOAT CRTM_20_K2 =
        0.25000000000000000000000000000000000000000000000000f;

    FLOAT *in_r = NULL;
    FLOAT *in_i = NULL;
    FLOAT *out_r = NULL;
    FLOAT *out_i = NULL;

#ifdef VOLATILE_STRIDE_ARRAY
    volatile INTP *in_strides = strides->in_strides;
    volatile INTP *out_strides = strides->out_strides;
#else
    INTP *in_strides = strides->in_strides;
    INTP *out_strides = strides->out_strides;
#endif
    INTP v_in_stride = strides->v_in_stride;
    INTP v_out_stride = strides->v_out_stride;
    INTP cnt = 0;

    if (flag) /* non-zero flag indicates that the fft is inverse */
    {
        in_r = (FLOAT *)in_imag;
        in_i = (FLOAT *)in_real;
        out_r = (FLOAT *)out_imag;
        out_i = (FLOAT *)out_real;
    }
    else
    {
        in_r = (FLOAT *)in_real;
        in_i = (FLOAT *)in_imag;
        out_r = (FLOAT *)out_real;
        out_i = (FLOAT *)out_imag;
    }

    for (cnt = 0; cnt < n; cnt++)
    {
        /* Load all 20 complex inputs */
        FLOAT x_0_r  = in_r[in_strides[0]];
        FLOAT x_0_i  = in_i[in_strides[0]];
        FLOAT x_1_r  = in_r[in_strides[1]];
        FLOAT x_1_i  = in_i[in_strides[1]];
        FLOAT x_2_r  = in_r[in_strides[2]];
        FLOAT x_2_i  = in_i[in_strides[2]];
        FLOAT x_3_r  = in_r[in_strides[3]];
        FLOAT x_3_i  = in_i[in_strides[3]];
        FLOAT x_4_r  = in_r[in_strides[4]];
        FLOAT x_4_i  = in_i[in_strides[4]];
        FLOAT x_5_r  = in_r[in_strides[5]];
        FLOAT x_5_i  = in_i[in_strides[5]];
        FLOAT x_6_r  = in_r[in_strides[6]];
        FLOAT x_6_i  = in_i[in_strides[6]];
        FLOAT x_7_r  = in_r[in_strides[7]];
        FLOAT x_7_i  = in_i[in_strides[7]];
        FLOAT x_8_r  = in_r[in_strides[8]];
        FLOAT x_8_i  = in_i[in_strides[8]];
        FLOAT x_9_r  = in_r[in_strides[9]];
        FLOAT x_9_i  = in_i[in_strides[9]];
        FLOAT x_10_r = in_r[in_strides[10]];
        FLOAT x_10_i = in_i[in_strides[10]];
        FLOAT x_11_r = in_r[in_strides[11]];
        FLOAT x_11_i = in_i[in_strides[11]];
        FLOAT x_12_r = in_r[in_strides[12]];
        FLOAT x_12_i = in_i[in_strides[12]];
        FLOAT x_13_r = in_r[in_strides[13]];
        FLOAT x_13_i = in_i[in_strides[13]];
        FLOAT x_14_r = in_r[in_strides[14]];
        FLOAT x_14_i = in_i[in_strides[14]];
        FLOAT x_15_r = in_r[in_strides[15]];
        FLOAT x_15_i = in_i[in_strides[15]];
        FLOAT x_16_r = in_r[in_strides[16]];
        FLOAT x_16_i = in_i[in_strides[16]];
        FLOAT x_17_r = in_r[in_strides[17]];
        FLOAT x_17_i = in_i[in_strides[17]];
        FLOAT x_18_r = in_r[in_strides[18]];
        FLOAT x_18_i = in_i[in_strides[18]];
        FLOAT x_19_r = in_r[in_strides[19]];
        FLOAT x_19_i = in_i[in_strides[19]];

        FLOAT t1 = x_0_r + x_10_r;
        FLOAT t2 = x_0_r - x_10_r;
        FLOAT t3 = x_5_r + x_15_r;
        FLOAT t4 = x_5_r - x_15_r;
        FLOAT t5 = x_0_i + x_10_i;
        FLOAT t6 = x_0_i - x_10_i;
        FLOAT t7 = x_5_i + x_15_i;
        FLOAT t8 = x_5_i - x_15_i;
        FLOAT t9 = x_14_r + x_16_r;
        FLOAT t10 = x_14_r - x_16_r;
        FLOAT t11 = x_4_r + x_6_r;
        FLOAT t12 = x_4_r - x_6_r;
        FLOAT t13 = x_11_i + x_19_i;
        FLOAT t14 = x_11_i - x_19_i;
        FLOAT t15 = x_1_i + x_9_i;
        FLOAT t16 = x_9_i - x_1_i;
        FLOAT t17 = x_14_i + x_16_i;
        FLOAT t18 = x_14_i - x_16_i;
        FLOAT t19 = x_4_i + x_6_i;
        FLOAT t20 = x_4_i - x_6_i;
        FLOAT t21 = x_11_r + x_19_r;
        FLOAT t22 = x_11_r - x_19_r;
        FLOAT t23 = x_1_r + x_9_r;
        FLOAT t24 = x_9_r - x_1_r;
        FLOAT t25 = x_12_r + x_18_r;
        FLOAT t26 = x_18_r - x_12_r;
        FLOAT t27 = x_12_i + x_18_i;
        FLOAT t28 = x_18_i - x_12_i;
        FLOAT t29 = x_2_r + x_8_r;
        FLOAT t30 = x_8_r - x_2_r;
        FLOAT t31 = x_2_i + x_8_i;
        FLOAT t32 = x_8_i - x_2_i;
        FLOAT t33 = x_3_r + x_7_r;
        FLOAT t34 = x_3_r - x_7_r;
        FLOAT t35 = x_3_i + x_7_i;
        FLOAT t36 = x_3_i - x_7_i;
        FLOAT t37 = x_13_r + x_17_r;
        FLOAT t38 = x_13_r - x_17_r;
        FLOAT t39 = x_13_i + x_17_i;
        FLOAT t40 = x_13_i - x_17_i;

        FLOAT cv1 = t1 + t3;
        FLOAT cv2 = t1 - t3;
        FLOAT cv3 = t5 + t7;
        FLOAT cv4 = t5 - t7;
        FLOAT cv5 = t2 + t8;
        FLOAT cv6 = t2 - t8;
        FLOAT cv7 = t6 + t4;
        FLOAT cv8 = t6 - t4;
        FLOAT cv9 = t9 + t11;
        FLOAT cv10 = t9 - t11;
        FLOAT cv11 = t10 + t12;
        FLOAT cv12 = t10 - t12;
        FLOAT cv13 = t14 + t16;
        FLOAT cv14 = t14 - t16;
        FLOAT cv15 = t13 + t15;
        FLOAT cv16 = t13 - t15;
        FLOAT cv17 = t17 + t19;
        FLOAT cv18 = t17 - t19;
        FLOAT cv19 = t20 + t18;
        FLOAT cv20 = t20 - t18;
        FLOAT cv21 = t22 + t24;
        FLOAT cv22 = t22 - t24;
        FLOAT cv23 = t21 + t23;
        FLOAT cv24 = t21 - t23;
        FLOAT cv25 = t25 + t29;
        FLOAT cv26 = t25 - t29;
        FLOAT cv27 = t26 + t30;
        FLOAT cv28 = t26 - t30;
        FLOAT cv29 = t27 + t31;
        FLOAT cv30 = t27 - t31;
        FLOAT cv31 = t28 + t32;
        FLOAT cv32 = t28 - t32;
        FLOAT cv33 = t33 + t37;
        FLOAT cv34 = t33 - t37;
        FLOAT cv35 = t35 + t39;
        FLOAT cv36 = t35 - t39;
        FLOAT cv37 = t36 + t40;
        FLOAT cv38 = t36 - t40;
        FLOAT cv39 = t34 + t38;
        FLOAT cv40 = t34 - t38;
        FLOAT cv41 = cv21 + cv18;
        FLOAT cv42 = cv21 - cv18;
        FLOAT cv43 = cv16 + cv12;
        FLOAT cv44 = cv16 - cv12;
        FLOAT cv45 = cv28 + cv36;
        FLOAT cv46 = cv28 - cv36;
        FLOAT cv47 = cv30 + cv40;
        FLOAT cv48 = cv30 - cv40;
        FLOAT cv49 = cv10 + cv13;
        FLOAT cv50 = cv10 - cv13;
        FLOAT cv51 = cv24 + cv20;
        FLOAT cv52 = cv24 - cv20;
        FLOAT cv53 = cv31 + cv37;
        FLOAT cv54 = cv37 - cv31;
        FLOAT cv55 = cv32 + cv34;
        FLOAT cv56 = cv32 - cv34;
        FLOAT cv57 = cv26 + cv38;
        FLOAT cv58 = cv26 - cv38;
        FLOAT cv59 = cv25 + cv33;
        FLOAT cv60 = cv25 - cv33;
        FLOAT cv61 = cv23 + cv9;
        FLOAT cv62 = cv23 - cv9;
        FLOAT cv63 = cv14 + cv19;
        FLOAT cv64 = cv14 - cv19;
        FLOAT cv65 = cv27 + cv39;
        FLOAT cv66 = cv27 - cv39;
        FLOAT cv67 = cv29 + cv35;
        FLOAT cv68 = cv29 - cv35;
        FLOAT cv69 = cv15 + cv17;
        FLOAT cv70 = cv15 - cv17;
        FLOAT cv71 = cv11 + cv22;
        FLOAT cv72 = cv11 - cv22;

        FLOAT s1 = cv43 + cv45;
        FLOAT d1 = cv45 - cv43;
        FLOAT s2 = cv61 + cv59;
        FLOAT d2 = cv61 - cv59;
        FLOAT s3 = cv69 + cv67;
        FLOAT d3 = cv69 - cv67;
        FLOAT s4 = cv55 + cv52;
        FLOAT d4 = cv52 - cv55;
        FLOAT s5 = cv51 + cv56;
        FLOAT d5 = cv51 - cv56;
        FLOAT s6 = cv60 + cv62;
        FLOAT d6 = cv60 - cv62;
        FLOAT s7 = cv68 + cv70;
        FLOAT d7 = cv68 - cv70;
        FLOAT s8 = cv44 + cv46;
        FLOAT d8 = cv46 - cv44;

        FLOAT k1_d1 = CRTM_20_K1 * d1;
        FLOAT k2_s1 = CRTM_20_K2 * s1;
        FLOAT k1_d2 = CRTM_20_K1 * d2;
        FLOAT k2_s2 = CRTM_20_K2 * s2;
        FLOAT k1_d3 = CRTM_20_K1 * d3;
        FLOAT k2_s3 = CRTM_20_K2 * s3;
        FLOAT k1_d4 = CRTM_20_K1 * d4;
        FLOAT k2_s4 = CRTM_20_K2 * s4;
        FLOAT k1_s5 = CRTM_20_K1 * s5;
        FLOAT k2_d5 = CRTM_20_K2 * d5;
        FLOAT k1_s6 = CRTM_20_K1 * s6;
        FLOAT k2_d6 = CRTM_20_K2 * d6;
        FLOAT k1_s7 = CRTM_20_K1 * s7;
        FLOAT k2_d7 = CRTM_20_K2 * d7;
        FLOAT k1_s8 = CRTM_20_K1 * s8;
        FLOAT k2_d8 = CRTM_20_K2 * d8;

        FLOAT base1 = k2_s1 + cv5;
        FLOAT base2 = cv1 - k2_s2;
        FLOAT base3 = cv3 - k2_s3;
        FLOAT base4 = cv7 + k2_s4;
        FLOAT base5 = cv8 - k2_d5;
        FLOAT base6 = cv2 - k2_d6;
        FLOAT base7 = cv4 - k2_d7;
        FLOAT base8 = cv6 + k2_d8;

        FLOAT cv73 = base1 + k1_d1;
        FLOAT cv101 = base1 - k1_d1;
        FLOAT cv85 = base2 + k1_d2;
        FLOAT cv93 = base2 - k1_d2;
        FLOAT cv87 = base3 + k1_d3;
        FLOAT cv95 = base3 - k1_d3;
        FLOAT cv83 = base4 + k1_d4;
        FLOAT cv99 = base4 - k1_d4;
        FLOAT cv75 = base5 + k1_s5;
        FLOAT cv103 = base5 - k1_s5;
        FLOAT cv77 = base6 + k1_s6;
        FLOAT cv89 = base6 - k1_s6;
        FLOAT cv79 = base7 + k1_s7;
        FLOAT cv91 = base7 - k1_s7;
        FLOAT cv81 = base8 - k1_s8;
        FLOAT cv97 = base8 + k1_s8;

        FLOAT cv74 = CRTM_20_0*(-cv41) + CRTM_20_3*(-cv48);
        FLOAT cv76 = CRTM_20_0*(cv50) + CRTM_20_3*(cv57);
        FLOAT cv78 = CRTM_20_0*(cv54) + CRTM_20_3*(cv63);
        FLOAT cv80 = CRTM_20_0*(cv66) + CRTM_20_3*(-cv71);
        FLOAT cv82 = CRTM_20_0*(-cv47) + CRTM_20_3*(-cv42);
        FLOAT cv84 = CRTM_20_0*(cv58) + CRTM_20_3*(-cv49);
        FLOAT cv86 = CRTM_20_0*(cv64) + CRTM_20_3*(-cv53);
        FLOAT cv88 = CRTM_20_0*(cv72) + CRTM_20_3*(cv65);
        FLOAT cv90 = CRTM_20_0*(cv63) + CRTM_20_3*(-cv54);
        FLOAT cv92 = CRTM_20_0*(-cv71) + CRTM_20_3*(-cv66);
        FLOAT cv94 = CRTM_20_0*(cv53) + CRTM_20_3*(cv64);
        FLOAT cv96 = CRTM_20_0*(-cv65) + CRTM_20_3*(cv72);
        FLOAT cv98 = CRTM_20_0*(cv42) + CRTM_20_3*(-cv47);
        FLOAT cv100 = CRTM_20_0*(cv49) + CRTM_20_3*(cv58);
        FLOAT cv102 = CRTM_20_0*(-cv48) + CRTM_20_3*(cv41);
        FLOAT cv104 = CRTM_20_0*(cv57) + CRTM_20_3*(-cv50);

        out_r[out_strides[0]] = cv1 + s2;
        out_i[out_strides[0]] = cv3 + s3;

        out_r[out_strides[1]] = cv73 + cv74;
        out_i[out_strides[1]] = cv75 + cv76;

        out_r[out_strides[2]] = cv77 + cv78;
        out_i[out_strides[2]] = cv79 + cv80;

        out_r[out_strides[3]] = cv81 + cv82;
        out_i[out_strides[3]] = cv83 + cv84;

        out_r[out_strides[4]] = cv85 + cv86;
        out_i[out_strides[4]] = cv87 + cv88;

        out_r[out_strides[5]] = cv5 - s1;
        out_i[out_strides[5]] = cv8 + d5;

        out_r[out_strides[6]] = cv89 + cv90;
        out_i[out_strides[6]] = cv91 + cv92;

        out_r[out_strides[7]] = cv81 - cv82;
        out_i[out_strides[7]] = cv83 - cv84;

        out_r[out_strides[8]] = cv93 + cv94;
        out_i[out_strides[8]] = cv95 + cv96;

        out_r[out_strides[9]] = cv73 - cv74;
        out_i[out_strides[9]] = cv75 - cv76;

        out_r[out_strides[10]] = cv2 + d6;
        out_i[out_strides[10]] = cv4 + d7;

        out_r[out_strides[11]] = cv97 + cv98;
        out_i[out_strides[11]] = cv99 + cv100;

        out_r[out_strides[12]] = cv93 - cv94;
        out_i[out_strides[12]] = cv95 - cv96;

        out_r[out_strides[13]] = cv101 + cv102;
        out_i[out_strides[13]] = cv103 + cv104;

        out_r[out_strides[14]] = cv89 - cv90;
        out_i[out_strides[14]] = cv91 - cv92;

        out_r[out_strides[15]] = cv6 - d8;
        out_i[out_strides[15]] = cv7 - s4;

        out_r[out_strides[16]] = cv85 - cv86;
        out_i[out_strides[16]] = cv87 - cv88;

        out_r[out_strides[17]] = cv101 - cv102;
        out_i[out_strides[17]] = cv103 - cv104;

        out_r[out_strides[18]] = cv77 - cv78;
        out_i[out_strides[18]] = cv79 - cv80;

        out_r[out_strides[19]] = cv97 - cv98;
        out_i[out_strides[19]] = cv99 - cv100;

        in_r += v_in_stride;
        in_i += v_in_stride;
        out_r += v_out_stride;
        out_i += v_out_stride;
    }

    AOCLFFTZ_LOG(DEBUG, global_logger_mode, "Exit");
}

kfft_ register_kernel_fft20c(UINT8 precision, UINT8 direction /* unused */)
{
    if (precision == DT_FLOAT)
    {
        return fft20c_fp32;
    }
    else if (precision == DT_DOUBLE)
    {
        return fft20c_fp64;
    }
    else
    {
        return NULL;
    }
}
