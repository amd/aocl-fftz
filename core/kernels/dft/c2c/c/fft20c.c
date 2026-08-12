// Copyright Advanced Micro Devices, Inc.
// SPDX-License-Identifier: BSD-3-Clause

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

ops_cycles_t get_ops_cnt_fft20c(FFTZ_UINT8 precision, FFTZ_UINT8 direction)
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

static FFTZ_VOID fft20c_fp64(FFTZ_VOID *in_real, FFTZ_VOID *in_imag,
                             FFTZ_VOID *out_real, FFTZ_VOID *out_imag,
                             FFTZ_INTP n, aoclfftz_strides_t *strides,
                             FFTZ_VOID *twd, FFTZ_UINT8 flag)
{
    AOCLFFTZ_LOG(DEBUG, global_logger_mode, "Enter");

    const FFTZ_DOUBLE CRTM_20_0 =
        0.95105651629515357211643933337876088917255401611328;
    const FFTZ_DOUBLE CRTM_20_3 =
        0.58778525229247312916870595463505014777183532714844;
    const FFTZ_DOUBLE CRTM_20_K1 =
        0.55901699437494742410229341718281905886015458990288;
    const FFTZ_DOUBLE CRTM_20_K2 =
        0.25000000000000000000000000000000000000000000000000;

    FFTZ_DOUBLE *in_r = NULL;
    FFTZ_DOUBLE *in_i = NULL;
    FFTZ_DOUBLE *out_r = NULL;
    FFTZ_DOUBLE *out_i = NULL;

#ifdef VOLATILE_STRIDE_ARRAY
    volatile FFTZ_INTP *in_strides = strides->in_strides;
    volatile FFTZ_INTP *out_strides = strides->out_strides;
#else
    FFTZ_INTP *in_strides = strides->in_strides;
    FFTZ_INTP *out_strides = strides->out_strides;
#endif
    FFTZ_INTP v_in_stride = strides->v_in_stride;
    FFTZ_INTP v_out_stride = strides->v_out_stride;
    FFTZ_INTP cnt = 0;

    if (flag) /* non-zero flag indicates that the fft is inverse */
    {
        in_r = (FFTZ_DOUBLE *)in_imag;
        in_i = (FFTZ_DOUBLE *)in_real;
        out_r = (FFTZ_DOUBLE *)out_imag;
        out_i = (FFTZ_DOUBLE *)out_real;
    }
    else
    {
        in_r = (FFTZ_DOUBLE *)in_real;
        in_i = (FFTZ_DOUBLE *)in_imag;
        out_r = (FFTZ_DOUBLE *)out_real;
        out_i = (FFTZ_DOUBLE *)out_imag;
    }

    for (cnt = 0; cnt < n; cnt++)
    {
        /* Load all 20 complex inputs */
        FFTZ_DOUBLE x_0_r  = in_r[in_strides[0]];
        FFTZ_DOUBLE x_0_i  = in_i[in_strides[0]];
        FFTZ_DOUBLE x_1_r  = in_r[in_strides[1]];
        FFTZ_DOUBLE x_1_i  = in_i[in_strides[1]];
        FFTZ_DOUBLE x_2_r  = in_r[in_strides[2]];
        FFTZ_DOUBLE x_2_i  = in_i[in_strides[2]];
        FFTZ_DOUBLE x_3_r  = in_r[in_strides[3]];
        FFTZ_DOUBLE x_3_i  = in_i[in_strides[3]];
        FFTZ_DOUBLE x_4_r  = in_r[in_strides[4]];
        FFTZ_DOUBLE x_4_i  = in_i[in_strides[4]];
        FFTZ_DOUBLE x_5_r  = in_r[in_strides[5]];
        FFTZ_DOUBLE x_5_i  = in_i[in_strides[5]];
        FFTZ_DOUBLE x_6_r  = in_r[in_strides[6]];
        FFTZ_DOUBLE x_6_i  = in_i[in_strides[6]];
        FFTZ_DOUBLE x_7_r  = in_r[in_strides[7]];
        FFTZ_DOUBLE x_7_i  = in_i[in_strides[7]];
        FFTZ_DOUBLE x_8_r  = in_r[in_strides[8]];
        FFTZ_DOUBLE x_8_i  = in_i[in_strides[8]];
        FFTZ_DOUBLE x_9_r  = in_r[in_strides[9]];
        FFTZ_DOUBLE x_9_i  = in_i[in_strides[9]];
        FFTZ_DOUBLE x_10_r = in_r[in_strides[10]];
        FFTZ_DOUBLE x_10_i = in_i[in_strides[10]];
        FFTZ_DOUBLE x_11_r = in_r[in_strides[11]];
        FFTZ_DOUBLE x_11_i = in_i[in_strides[11]];
        FFTZ_DOUBLE x_12_r = in_r[in_strides[12]];
        FFTZ_DOUBLE x_12_i = in_i[in_strides[12]];
        FFTZ_DOUBLE x_13_r = in_r[in_strides[13]];
        FFTZ_DOUBLE x_13_i = in_i[in_strides[13]];
        FFTZ_DOUBLE x_14_r = in_r[in_strides[14]];
        FFTZ_DOUBLE x_14_i = in_i[in_strides[14]];
        FFTZ_DOUBLE x_15_r = in_r[in_strides[15]];
        FFTZ_DOUBLE x_15_i = in_i[in_strides[15]];
        FFTZ_DOUBLE x_16_r = in_r[in_strides[16]];
        FFTZ_DOUBLE x_16_i = in_i[in_strides[16]];
        FFTZ_DOUBLE x_17_r = in_r[in_strides[17]];
        FFTZ_DOUBLE x_17_i = in_i[in_strides[17]];
        FFTZ_DOUBLE x_18_r = in_r[in_strides[18]];
        FFTZ_DOUBLE x_18_i = in_i[in_strides[18]];
        FFTZ_DOUBLE x_19_r = in_r[in_strides[19]];
        FFTZ_DOUBLE x_19_i = in_i[in_strides[19]];

        FFTZ_DOUBLE t1 = x_0_r + x_10_r;
        FFTZ_DOUBLE t2 = x_0_r - x_10_r;
        FFTZ_DOUBLE t3 = x_5_r + x_15_r;
        FFTZ_DOUBLE t4 = x_5_r - x_15_r;
        FFTZ_DOUBLE t5 = x_0_i + x_10_i;
        FFTZ_DOUBLE t6 = x_0_i - x_10_i;
        FFTZ_DOUBLE t7 = x_5_i + x_15_i;
        FFTZ_DOUBLE t8 = x_5_i - x_15_i;
        FFTZ_DOUBLE t9 = x_14_r + x_16_r;
        FFTZ_DOUBLE t10 = x_14_r - x_16_r;
        FFTZ_DOUBLE t11 = x_4_r + x_6_r;
        FFTZ_DOUBLE t12 = x_4_r - x_6_r;
        FFTZ_DOUBLE t13 = x_11_i + x_19_i;
        FFTZ_DOUBLE t14 = x_11_i - x_19_i;
        FFTZ_DOUBLE t15 = x_1_i + x_9_i;
        FFTZ_DOUBLE t16 = x_9_i - x_1_i;
        FFTZ_DOUBLE t17 = x_14_i + x_16_i;
        FFTZ_DOUBLE t18 = x_14_i - x_16_i;
        FFTZ_DOUBLE t19 = x_4_i + x_6_i;
        FFTZ_DOUBLE t20 = x_4_i - x_6_i;
        FFTZ_DOUBLE t21 = x_11_r + x_19_r;
        FFTZ_DOUBLE t22 = x_11_r - x_19_r;
        FFTZ_DOUBLE t23 = x_1_r + x_9_r;
        FFTZ_DOUBLE t24 = x_9_r - x_1_r;
        FFTZ_DOUBLE t25 = x_12_r + x_18_r;
        FFTZ_DOUBLE t26 = x_18_r - x_12_r;
        FFTZ_DOUBLE t27 = x_12_i + x_18_i;
        FFTZ_DOUBLE t28 = x_18_i - x_12_i;
        FFTZ_DOUBLE t29 = x_2_r + x_8_r;
        FFTZ_DOUBLE t30 = x_8_r - x_2_r;
        FFTZ_DOUBLE t31 = x_2_i + x_8_i;
        FFTZ_DOUBLE t32 = x_8_i - x_2_i;
        FFTZ_DOUBLE t33 = x_3_r + x_7_r;
        FFTZ_DOUBLE t34 = x_3_r - x_7_r;
        FFTZ_DOUBLE t35 = x_3_i + x_7_i;
        FFTZ_DOUBLE t36 = x_3_i - x_7_i;
        FFTZ_DOUBLE t37 = x_13_r + x_17_r;
        FFTZ_DOUBLE t38 = x_13_r - x_17_r;
        FFTZ_DOUBLE t39 = x_13_i + x_17_i;
        FFTZ_DOUBLE t40 = x_13_i - x_17_i;

        FFTZ_DOUBLE cv1 = t1 + t3;
        FFTZ_DOUBLE cv2 = t1 - t3;
        FFTZ_DOUBLE cv3 = t5 + t7;
        FFTZ_DOUBLE cv4 = t5 - t7;
        FFTZ_DOUBLE cv5 = t2 + t8;
        FFTZ_DOUBLE cv6 = t2 - t8;
        FFTZ_DOUBLE cv7 = t6 + t4;
        FFTZ_DOUBLE cv8 = t6 - t4;
        FFTZ_DOUBLE cv9 = t9 + t11;
        FFTZ_DOUBLE cv10 = t9 - t11;
        FFTZ_DOUBLE cv11 = t10 + t12;
        FFTZ_DOUBLE cv12 = t10 - t12;
        FFTZ_DOUBLE cv13 = t14 + t16;
        FFTZ_DOUBLE cv14 = t14 - t16;
        FFTZ_DOUBLE cv15 = t13 + t15;
        FFTZ_DOUBLE cv16 = t13 - t15;
        FFTZ_DOUBLE cv17 = t17 + t19;
        FFTZ_DOUBLE cv18 = t17 - t19;
        FFTZ_DOUBLE cv19 = t20 + t18;
        FFTZ_DOUBLE cv20 = t20 - t18;
        FFTZ_DOUBLE cv21 = t22 + t24;
        FFTZ_DOUBLE cv22 = t22 - t24;
        FFTZ_DOUBLE cv23 = t21 + t23;
        FFTZ_DOUBLE cv24 = t21 - t23;
        FFTZ_DOUBLE cv25 = t25 + t29;
        FFTZ_DOUBLE cv26 = t25 - t29;
        FFTZ_DOUBLE cv27 = t26 + t30;
        FFTZ_DOUBLE cv28 = t26 - t30;
        FFTZ_DOUBLE cv29 = t27 + t31;
        FFTZ_DOUBLE cv30 = t27 - t31;
        FFTZ_DOUBLE cv31 = t28 + t32;
        FFTZ_DOUBLE cv32 = t28 - t32;
        FFTZ_DOUBLE cv33 = t33 + t37;
        FFTZ_DOUBLE cv34 = t33 - t37;
        FFTZ_DOUBLE cv35 = t35 + t39;
        FFTZ_DOUBLE cv36 = t35 - t39;
        FFTZ_DOUBLE cv37 = t36 + t40;
        FFTZ_DOUBLE cv38 = t36 - t40;
        FFTZ_DOUBLE cv39 = t34 + t38;
        FFTZ_DOUBLE cv40 = t34 - t38;
        FFTZ_DOUBLE cv41 = cv21 + cv18;
        FFTZ_DOUBLE cv42 = cv21 - cv18;
        FFTZ_DOUBLE cv43 = cv16 + cv12;
        FFTZ_DOUBLE cv44 = cv16 - cv12;
        FFTZ_DOUBLE cv45 = cv28 + cv36;
        FFTZ_DOUBLE cv46 = cv28 - cv36;
        FFTZ_DOUBLE cv47 = cv30 + cv40;
        FFTZ_DOUBLE cv48 = cv30 - cv40;
        FFTZ_DOUBLE cv49 = cv10 + cv13;
        FFTZ_DOUBLE cv50 = cv10 - cv13;
        FFTZ_DOUBLE cv51 = cv24 + cv20;
        FFTZ_DOUBLE cv52 = cv24 - cv20;
        FFTZ_DOUBLE cv53 = cv31 + cv37;
        FFTZ_DOUBLE cv54 = cv37 - cv31;
        FFTZ_DOUBLE cv55 = cv32 + cv34;
        FFTZ_DOUBLE cv56 = cv32 - cv34;
        FFTZ_DOUBLE cv57 = cv26 + cv38;
        FFTZ_DOUBLE cv58 = cv26 - cv38;
        FFTZ_DOUBLE cv59 = cv25 + cv33;
        FFTZ_DOUBLE cv60 = cv25 - cv33;
        FFTZ_DOUBLE cv61 = cv23 + cv9;
        FFTZ_DOUBLE cv62 = cv23 - cv9;
        FFTZ_DOUBLE cv63 = cv14 + cv19;
        FFTZ_DOUBLE cv64 = cv14 - cv19;
        FFTZ_DOUBLE cv65 = cv27 + cv39;
        FFTZ_DOUBLE cv66 = cv27 - cv39;
        FFTZ_DOUBLE cv67 = cv29 + cv35;
        FFTZ_DOUBLE cv68 = cv29 - cv35;
        FFTZ_DOUBLE cv69 = cv15 + cv17;
        FFTZ_DOUBLE cv70 = cv15 - cv17;
        FFTZ_DOUBLE cv71 = cv11 + cv22;
        FFTZ_DOUBLE cv72 = cv11 - cv22;

        FFTZ_DOUBLE s1 = cv43 + cv45;
        FFTZ_DOUBLE d1 = cv45 - cv43;
        FFTZ_DOUBLE s2 = cv61 + cv59;
        FFTZ_DOUBLE d2 = cv61 - cv59;
        FFTZ_DOUBLE s3 = cv69 + cv67;
        FFTZ_DOUBLE d3 = cv69 - cv67;
        FFTZ_DOUBLE s4 = cv55 + cv52;
        FFTZ_DOUBLE d4 = cv52 - cv55;
        FFTZ_DOUBLE s5 = cv51 + cv56;
        FFTZ_DOUBLE d5 = cv51 - cv56;
        FFTZ_DOUBLE s6 = cv60 + cv62;
        FFTZ_DOUBLE d6 = cv60 - cv62;
        FFTZ_DOUBLE s7 = cv68 + cv70;
        FFTZ_DOUBLE d7 = cv68 - cv70;
        FFTZ_DOUBLE s8 = cv44 + cv46;
        FFTZ_DOUBLE d8 = cv46 - cv44;

        FFTZ_DOUBLE k1_d1 = CRTM_20_K1 * d1;
        FFTZ_DOUBLE k2_s1 = CRTM_20_K2 * s1;
        FFTZ_DOUBLE k1_d2 = CRTM_20_K1 * d2;
        FFTZ_DOUBLE k2_s2 = CRTM_20_K2 * s2;
        FFTZ_DOUBLE k1_d3 = CRTM_20_K1 * d3;
        FFTZ_DOUBLE k2_s3 = CRTM_20_K2 * s3;
        FFTZ_DOUBLE k1_d4 = CRTM_20_K1 * d4;
        FFTZ_DOUBLE k2_s4 = CRTM_20_K2 * s4;
        FFTZ_DOUBLE k1_s5 = CRTM_20_K1 * s5;
        FFTZ_DOUBLE k2_d5 = CRTM_20_K2 * d5;
        FFTZ_DOUBLE k1_s6 = CRTM_20_K1 * s6;
        FFTZ_DOUBLE k2_d6 = CRTM_20_K2 * d6;
        FFTZ_DOUBLE k1_s7 = CRTM_20_K1 * s7;
        FFTZ_DOUBLE k2_d7 = CRTM_20_K2 * d7;
        FFTZ_DOUBLE k1_s8 = CRTM_20_K1 * s8;
        FFTZ_DOUBLE k2_d8 = CRTM_20_K2 * d8;

        FFTZ_DOUBLE base1 = k2_s1 + cv5;
        FFTZ_DOUBLE base2 = cv1 - k2_s2;
        FFTZ_DOUBLE base3 = cv3 - k2_s3;
        FFTZ_DOUBLE base4 = cv7 + k2_s4;
        FFTZ_DOUBLE base5 = cv8 - k2_d5;
        FFTZ_DOUBLE base6 = cv2 - k2_d6;
        FFTZ_DOUBLE base7 = cv4 - k2_d7;
        FFTZ_DOUBLE base8 = cv6 + k2_d8;

        FFTZ_DOUBLE cv73 = base1 + k1_d1;
        FFTZ_DOUBLE cv101 = base1 - k1_d1;
        FFTZ_DOUBLE cv85 = base2 + k1_d2;
        FFTZ_DOUBLE cv93 = base2 - k1_d2;
        FFTZ_DOUBLE cv87 = base3 + k1_d3;
        FFTZ_DOUBLE cv95 = base3 - k1_d3;
        FFTZ_DOUBLE cv83 = base4 + k1_d4;
        FFTZ_DOUBLE cv99 = base4 - k1_d4;
        FFTZ_DOUBLE cv75 = base5 + k1_s5;
        FFTZ_DOUBLE cv103 = base5 - k1_s5;
        FFTZ_DOUBLE cv77 = base6 + k1_s6;
        FFTZ_DOUBLE cv89 = base6 - k1_s6;
        FFTZ_DOUBLE cv79 = base7 + k1_s7;
        FFTZ_DOUBLE cv91 = base7 - k1_s7;
        FFTZ_DOUBLE cv81 = base8 - k1_s8;
        FFTZ_DOUBLE cv97 = base8 + k1_s8;

        FFTZ_DOUBLE cv74 = CRTM_20_0*(-cv41) + CRTM_20_3*(-cv48);
        FFTZ_DOUBLE cv76 = CRTM_20_0*(cv50) + CRTM_20_3*(cv57);
        FFTZ_DOUBLE cv78 = CRTM_20_0*(cv54) + CRTM_20_3*(cv63);
        FFTZ_DOUBLE cv80 = CRTM_20_0*(cv66) + CRTM_20_3*(-cv71);
        FFTZ_DOUBLE cv82 = CRTM_20_0*(-cv47) + CRTM_20_3*(-cv42);
        FFTZ_DOUBLE cv84 = CRTM_20_0*(cv58) + CRTM_20_3*(-cv49);
        FFTZ_DOUBLE cv86 = CRTM_20_0*(cv64) + CRTM_20_3*(-cv53);
        FFTZ_DOUBLE cv88 = CRTM_20_0*(cv72) + CRTM_20_3*(cv65);
        FFTZ_DOUBLE cv90 = CRTM_20_0*(cv63) + CRTM_20_3*(-cv54);
        FFTZ_DOUBLE cv92 = CRTM_20_0*(-cv71) + CRTM_20_3*(-cv66);
        FFTZ_DOUBLE cv94 = CRTM_20_0*(cv53) + CRTM_20_3*(cv64);
        FFTZ_DOUBLE cv96 = CRTM_20_0*(-cv65) + CRTM_20_3*(cv72);
        FFTZ_DOUBLE cv98 = CRTM_20_0*(cv42) + CRTM_20_3*(-cv47);
        FFTZ_DOUBLE cv100 = CRTM_20_0*(cv49) + CRTM_20_3*(cv58);
        FFTZ_DOUBLE cv102 = CRTM_20_0*(-cv48) + CRTM_20_3*(cv41);
        FFTZ_DOUBLE cv104 = CRTM_20_0*(cv57) + CRTM_20_3*(-cv50);

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

static FFTZ_VOID fft20c_fp32(FFTZ_VOID *in_real, FFTZ_VOID *in_imag,
                             FFTZ_VOID *out_real, FFTZ_VOID *out_imag,
                             FFTZ_INTP n, aoclfftz_strides_t *strides,
                             FFTZ_VOID *twd, FFTZ_UINT8 flag)
{
    AOCLFFTZ_LOG(DEBUG, global_logger_mode, "Enter");

    const FFTZ_FLOAT CRTM_20_0 =
        0.95105651629515357211643933337876088917255401611328f;
    const FFTZ_FLOAT CRTM_20_3 =
        0.58778525229247312916870595463505014777183532714844f;
    const FFTZ_FLOAT CRTM_20_K1 =
        0.55901699437494742410229341718281905886015458990288f;
    const FFTZ_FLOAT CRTM_20_K2 =
        0.25000000000000000000000000000000000000000000000000f;

    FFTZ_FLOAT *in_r = NULL;
    FFTZ_FLOAT *in_i = NULL;
    FFTZ_FLOAT *out_r = NULL;
    FFTZ_FLOAT *out_i = NULL;

#ifdef VOLATILE_STRIDE_ARRAY
    volatile FFTZ_INTP *in_strides = strides->in_strides;
    volatile FFTZ_INTP *out_strides = strides->out_strides;
#else
    FFTZ_INTP *in_strides = strides->in_strides;
    FFTZ_INTP *out_strides = strides->out_strides;
#endif
    FFTZ_INTP v_in_stride = strides->v_in_stride;
    FFTZ_INTP v_out_stride = strides->v_out_stride;
    FFTZ_INTP cnt = 0;

    if (flag) /* non-zero flag indicates that the fft is inverse */
    {
        in_r = (FFTZ_FLOAT *)in_imag;
        in_i = (FFTZ_FLOAT *)in_real;
        out_r = (FFTZ_FLOAT *)out_imag;
        out_i = (FFTZ_FLOAT *)out_real;
    }
    else
    {
        in_r = (FFTZ_FLOAT *)in_real;
        in_i = (FFTZ_FLOAT *)in_imag;
        out_r = (FFTZ_FLOAT *)out_real;
        out_i = (FFTZ_FLOAT *)out_imag;
    }

    for (cnt = 0; cnt < n; cnt++)
    {
        /* Load all 20 complex inputs */
        FFTZ_FLOAT x_0_r  = in_r[in_strides[0]];
        FFTZ_FLOAT x_0_i  = in_i[in_strides[0]];
        FFTZ_FLOAT x_1_r  = in_r[in_strides[1]];
        FFTZ_FLOAT x_1_i  = in_i[in_strides[1]];
        FFTZ_FLOAT x_2_r  = in_r[in_strides[2]];
        FFTZ_FLOAT x_2_i  = in_i[in_strides[2]];
        FFTZ_FLOAT x_3_r  = in_r[in_strides[3]];
        FFTZ_FLOAT x_3_i  = in_i[in_strides[3]];
        FFTZ_FLOAT x_4_r  = in_r[in_strides[4]];
        FFTZ_FLOAT x_4_i  = in_i[in_strides[4]];
        FFTZ_FLOAT x_5_r  = in_r[in_strides[5]];
        FFTZ_FLOAT x_5_i  = in_i[in_strides[5]];
        FFTZ_FLOAT x_6_r  = in_r[in_strides[6]];
        FFTZ_FLOAT x_6_i  = in_i[in_strides[6]];
        FFTZ_FLOAT x_7_r  = in_r[in_strides[7]];
        FFTZ_FLOAT x_7_i  = in_i[in_strides[7]];
        FFTZ_FLOAT x_8_r  = in_r[in_strides[8]];
        FFTZ_FLOAT x_8_i  = in_i[in_strides[8]];
        FFTZ_FLOAT x_9_r  = in_r[in_strides[9]];
        FFTZ_FLOAT x_9_i  = in_i[in_strides[9]];
        FFTZ_FLOAT x_10_r = in_r[in_strides[10]];
        FFTZ_FLOAT x_10_i = in_i[in_strides[10]];
        FFTZ_FLOAT x_11_r = in_r[in_strides[11]];
        FFTZ_FLOAT x_11_i = in_i[in_strides[11]];
        FFTZ_FLOAT x_12_r = in_r[in_strides[12]];
        FFTZ_FLOAT x_12_i = in_i[in_strides[12]];
        FFTZ_FLOAT x_13_r = in_r[in_strides[13]];
        FFTZ_FLOAT x_13_i = in_i[in_strides[13]];
        FFTZ_FLOAT x_14_r = in_r[in_strides[14]];
        FFTZ_FLOAT x_14_i = in_i[in_strides[14]];
        FFTZ_FLOAT x_15_r = in_r[in_strides[15]];
        FFTZ_FLOAT x_15_i = in_i[in_strides[15]];
        FFTZ_FLOAT x_16_r = in_r[in_strides[16]];
        FFTZ_FLOAT x_16_i = in_i[in_strides[16]];
        FFTZ_FLOAT x_17_r = in_r[in_strides[17]];
        FFTZ_FLOAT x_17_i = in_i[in_strides[17]];
        FFTZ_FLOAT x_18_r = in_r[in_strides[18]];
        FFTZ_FLOAT x_18_i = in_i[in_strides[18]];
        FFTZ_FLOAT x_19_r = in_r[in_strides[19]];
        FFTZ_FLOAT x_19_i = in_i[in_strides[19]];

        FFTZ_FLOAT t1 = x_0_r + x_10_r;
        FFTZ_FLOAT t2 = x_0_r - x_10_r;
        FFTZ_FLOAT t3 = x_5_r + x_15_r;
        FFTZ_FLOAT t4 = x_5_r - x_15_r;
        FFTZ_FLOAT t5 = x_0_i + x_10_i;
        FFTZ_FLOAT t6 = x_0_i - x_10_i;
        FFTZ_FLOAT t7 = x_5_i + x_15_i;
        FFTZ_FLOAT t8 = x_5_i - x_15_i;
        FFTZ_FLOAT t9 = x_14_r + x_16_r;
        FFTZ_FLOAT t10 = x_14_r - x_16_r;
        FFTZ_FLOAT t11 = x_4_r + x_6_r;
        FFTZ_FLOAT t12 = x_4_r - x_6_r;
        FFTZ_FLOAT t13 = x_11_i + x_19_i;
        FFTZ_FLOAT t14 = x_11_i - x_19_i;
        FFTZ_FLOAT t15 = x_1_i + x_9_i;
        FFTZ_FLOAT t16 = x_9_i - x_1_i;
        FFTZ_FLOAT t17 = x_14_i + x_16_i;
        FFTZ_FLOAT t18 = x_14_i - x_16_i;
        FFTZ_FLOAT t19 = x_4_i + x_6_i;
        FFTZ_FLOAT t20 = x_4_i - x_6_i;
        FFTZ_FLOAT t21 = x_11_r + x_19_r;
        FFTZ_FLOAT t22 = x_11_r - x_19_r;
        FFTZ_FLOAT t23 = x_1_r + x_9_r;
        FFTZ_FLOAT t24 = x_9_r - x_1_r;
        FFTZ_FLOAT t25 = x_12_r + x_18_r;
        FFTZ_FLOAT t26 = x_18_r - x_12_r;
        FFTZ_FLOAT t27 = x_12_i + x_18_i;
        FFTZ_FLOAT t28 = x_18_i - x_12_i;
        FFTZ_FLOAT t29 = x_2_r + x_8_r;
        FFTZ_FLOAT t30 = x_8_r - x_2_r;
        FFTZ_FLOAT t31 = x_2_i + x_8_i;
        FFTZ_FLOAT t32 = x_8_i - x_2_i;
        FFTZ_FLOAT t33 = x_3_r + x_7_r;
        FFTZ_FLOAT t34 = x_3_r - x_7_r;
        FFTZ_FLOAT t35 = x_3_i + x_7_i;
        FFTZ_FLOAT t36 = x_3_i - x_7_i;
        FFTZ_FLOAT t37 = x_13_r + x_17_r;
        FFTZ_FLOAT t38 = x_13_r - x_17_r;
        FFTZ_FLOAT t39 = x_13_i + x_17_i;
        FFTZ_FLOAT t40 = x_13_i - x_17_i;

        FFTZ_FLOAT cv1 = t1 + t3;
        FFTZ_FLOAT cv2 = t1 - t3;
        FFTZ_FLOAT cv3 = t5 + t7;
        FFTZ_FLOAT cv4 = t5 - t7;
        FFTZ_FLOAT cv5 = t2 + t8;
        FFTZ_FLOAT cv6 = t2 - t8;
        FFTZ_FLOAT cv7 = t6 + t4;
        FFTZ_FLOAT cv8 = t6 - t4;
        FFTZ_FLOAT cv9 = t9 + t11;
        FFTZ_FLOAT cv10 = t9 - t11;
        FFTZ_FLOAT cv11 = t10 + t12;
        FFTZ_FLOAT cv12 = t10 - t12;
        FFTZ_FLOAT cv13 = t14 + t16;
        FFTZ_FLOAT cv14 = t14 - t16;
        FFTZ_FLOAT cv15 = t13 + t15;
        FFTZ_FLOAT cv16 = t13 - t15;
        FFTZ_FLOAT cv17 = t17 + t19;
        FFTZ_FLOAT cv18 = t17 - t19;
        FFTZ_FLOAT cv19 = t20 + t18;
        FFTZ_FLOAT cv20 = t20 - t18;
        FFTZ_FLOAT cv21 = t22 + t24;
        FFTZ_FLOAT cv22 = t22 - t24;
        FFTZ_FLOAT cv23 = t21 + t23;
        FFTZ_FLOAT cv24 = t21 - t23;
        FFTZ_FLOAT cv25 = t25 + t29;
        FFTZ_FLOAT cv26 = t25 - t29;
        FFTZ_FLOAT cv27 = t26 + t30;
        FFTZ_FLOAT cv28 = t26 - t30;
        FFTZ_FLOAT cv29 = t27 + t31;
        FFTZ_FLOAT cv30 = t27 - t31;
        FFTZ_FLOAT cv31 = t28 + t32;
        FFTZ_FLOAT cv32 = t28 - t32;
        FFTZ_FLOAT cv33 = t33 + t37;
        FFTZ_FLOAT cv34 = t33 - t37;
        FFTZ_FLOAT cv35 = t35 + t39;
        FFTZ_FLOAT cv36 = t35 - t39;
        FFTZ_FLOAT cv37 = t36 + t40;
        FFTZ_FLOAT cv38 = t36 - t40;
        FFTZ_FLOAT cv39 = t34 + t38;
        FFTZ_FLOAT cv40 = t34 - t38;
        FFTZ_FLOAT cv41 = cv21 + cv18;
        FFTZ_FLOAT cv42 = cv21 - cv18;
        FFTZ_FLOAT cv43 = cv16 + cv12;
        FFTZ_FLOAT cv44 = cv16 - cv12;
        FFTZ_FLOAT cv45 = cv28 + cv36;
        FFTZ_FLOAT cv46 = cv28 - cv36;
        FFTZ_FLOAT cv47 = cv30 + cv40;
        FFTZ_FLOAT cv48 = cv30 - cv40;
        FFTZ_FLOAT cv49 = cv10 + cv13;
        FFTZ_FLOAT cv50 = cv10 - cv13;
        FFTZ_FLOAT cv51 = cv24 + cv20;
        FFTZ_FLOAT cv52 = cv24 - cv20;
        FFTZ_FLOAT cv53 = cv31 + cv37;
        FFTZ_FLOAT cv54 = cv37 - cv31;
        FFTZ_FLOAT cv55 = cv32 + cv34;
        FFTZ_FLOAT cv56 = cv32 - cv34;
        FFTZ_FLOAT cv57 = cv26 + cv38;
        FFTZ_FLOAT cv58 = cv26 - cv38;
        FFTZ_FLOAT cv59 = cv25 + cv33;
        FFTZ_FLOAT cv60 = cv25 - cv33;
        FFTZ_FLOAT cv61 = cv23 + cv9;
        FFTZ_FLOAT cv62 = cv23 - cv9;
        FFTZ_FLOAT cv63 = cv14 + cv19;
        FFTZ_FLOAT cv64 = cv14 - cv19;
        FFTZ_FLOAT cv65 = cv27 + cv39;
        FFTZ_FLOAT cv66 = cv27 - cv39;
        FFTZ_FLOAT cv67 = cv29 + cv35;
        FFTZ_FLOAT cv68 = cv29 - cv35;
        FFTZ_FLOAT cv69 = cv15 + cv17;
        FFTZ_FLOAT cv70 = cv15 - cv17;
        FFTZ_FLOAT cv71 = cv11 + cv22;
        FFTZ_FLOAT cv72 = cv11 - cv22;

        FFTZ_FLOAT s1 = cv43 + cv45;
        FFTZ_FLOAT d1 = cv45 - cv43;
        FFTZ_FLOAT s2 = cv61 + cv59;
        FFTZ_FLOAT d2 = cv61 - cv59;
        FFTZ_FLOAT s3 = cv69 + cv67;
        FFTZ_FLOAT d3 = cv69 - cv67;
        FFTZ_FLOAT s4 = cv55 + cv52;
        FFTZ_FLOAT d4 = cv52 - cv55;
        FFTZ_FLOAT s5 = cv51 + cv56;
        FFTZ_FLOAT d5 = cv51 - cv56;
        FFTZ_FLOAT s6 = cv60 + cv62;
        FFTZ_FLOAT d6 = cv60 - cv62;
        FFTZ_FLOAT s7 = cv68 + cv70;
        FFTZ_FLOAT d7 = cv68 - cv70;
        FFTZ_FLOAT s8 = cv44 + cv46;
        FFTZ_FLOAT d8 = cv46 - cv44;

        FFTZ_FLOAT k1_d1 = CRTM_20_K1 * d1;
        FFTZ_FLOAT k2_s1 = CRTM_20_K2 * s1;
        FFTZ_FLOAT k1_d2 = CRTM_20_K1 * d2;
        FFTZ_FLOAT k2_s2 = CRTM_20_K2 * s2;
        FFTZ_FLOAT k1_d3 = CRTM_20_K1 * d3;
        FFTZ_FLOAT k2_s3 = CRTM_20_K2 * s3;
        FFTZ_FLOAT k1_d4 = CRTM_20_K1 * d4;
        FFTZ_FLOAT k2_s4 = CRTM_20_K2 * s4;
        FFTZ_FLOAT k1_s5 = CRTM_20_K1 * s5;
        FFTZ_FLOAT k2_d5 = CRTM_20_K2 * d5;
        FFTZ_FLOAT k1_s6 = CRTM_20_K1 * s6;
        FFTZ_FLOAT k2_d6 = CRTM_20_K2 * d6;
        FFTZ_FLOAT k1_s7 = CRTM_20_K1 * s7;
        FFTZ_FLOAT k2_d7 = CRTM_20_K2 * d7;
        FFTZ_FLOAT k1_s8 = CRTM_20_K1 * s8;
        FFTZ_FLOAT k2_d8 = CRTM_20_K2 * d8;

        FFTZ_FLOAT base1 = k2_s1 + cv5;
        FFTZ_FLOAT base2 = cv1 - k2_s2;
        FFTZ_FLOAT base3 = cv3 - k2_s3;
        FFTZ_FLOAT base4 = cv7 + k2_s4;
        FFTZ_FLOAT base5 = cv8 - k2_d5;
        FFTZ_FLOAT base6 = cv2 - k2_d6;
        FFTZ_FLOAT base7 = cv4 - k2_d7;
        FFTZ_FLOAT base8 = cv6 + k2_d8;

        FFTZ_FLOAT cv73 = base1 + k1_d1;
        FFTZ_FLOAT cv101 = base1 - k1_d1;
        FFTZ_FLOAT cv85 = base2 + k1_d2;
        FFTZ_FLOAT cv93 = base2 - k1_d2;
        FFTZ_FLOAT cv87 = base3 + k1_d3;
        FFTZ_FLOAT cv95 = base3 - k1_d3;
        FFTZ_FLOAT cv83 = base4 + k1_d4;
        FFTZ_FLOAT cv99 = base4 - k1_d4;
        FFTZ_FLOAT cv75 = base5 + k1_s5;
        FFTZ_FLOAT cv103 = base5 - k1_s5;
        FFTZ_FLOAT cv77 = base6 + k1_s6;
        FFTZ_FLOAT cv89 = base6 - k1_s6;
        FFTZ_FLOAT cv79 = base7 + k1_s7;
        FFTZ_FLOAT cv91 = base7 - k1_s7;
        FFTZ_FLOAT cv81 = base8 - k1_s8;
        FFTZ_FLOAT cv97 = base8 + k1_s8;

        FFTZ_FLOAT cv74 = CRTM_20_0*(-cv41) + CRTM_20_3*(-cv48);
        FFTZ_FLOAT cv76 = CRTM_20_0*(cv50) + CRTM_20_3*(cv57);
        FFTZ_FLOAT cv78 = CRTM_20_0*(cv54) + CRTM_20_3*(cv63);
        FFTZ_FLOAT cv80 = CRTM_20_0*(cv66) + CRTM_20_3*(-cv71);
        FFTZ_FLOAT cv82 = CRTM_20_0*(-cv47) + CRTM_20_3*(-cv42);
        FFTZ_FLOAT cv84 = CRTM_20_0*(cv58) + CRTM_20_3*(-cv49);
        FFTZ_FLOAT cv86 = CRTM_20_0*(cv64) + CRTM_20_3*(-cv53);
        FFTZ_FLOAT cv88 = CRTM_20_0*(cv72) + CRTM_20_3*(cv65);
        FFTZ_FLOAT cv90 = CRTM_20_0*(cv63) + CRTM_20_3*(-cv54);
        FFTZ_FLOAT cv92 = CRTM_20_0*(-cv71) + CRTM_20_3*(-cv66);
        FFTZ_FLOAT cv94 = CRTM_20_0*(cv53) + CRTM_20_3*(cv64);
        FFTZ_FLOAT cv96 = CRTM_20_0*(-cv65) + CRTM_20_3*(cv72);
        FFTZ_FLOAT cv98 = CRTM_20_0*(cv42) + CRTM_20_3*(-cv47);
        FFTZ_FLOAT cv100 = CRTM_20_0*(cv49) + CRTM_20_3*(cv58);
        FFTZ_FLOAT cv102 = CRTM_20_0*(-cv48) + CRTM_20_3*(cv41);
        FFTZ_FLOAT cv104 = CRTM_20_0*(cv57) + CRTM_20_3*(-cv50);

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

kfft_ register_kernel_fft20c(FFTZ_UINT8 precision,
                             FFTZ_UINT8 direction /* unused */)
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
