// Copyright Advanced Micro Devices, Inc.
// SPDX-License-Identifier: BSD-3-Clause

/** @file rfft10c.c
 *
 *  @brief Radix-10 r2hc Real-FFT kernel with scalar operations in C
 *
 *  This file contains the DIT radix-10 real-to-halfcomplex implementations
 *  using scalar operations for single-precision and double-precision inputs.
 *
 *  @author Jeevanantham N
 */

#include "core/kernels/kernel.h"

static const ops_cycles_t ops_cnt[NUM_PRECISIONS][NUM_FFT_DIRS] =
                                                    {{{0, 12, 34, 20, 0, 0},
                                                      {0, 14, 34, 20, 0, 0}},
                                                     {{0, 12, 34, 20, 0, 0},
                                                      {0, 14, 34, 20, 0, 0}}};

ops_cycles_t get_ops_cnt_r2hc_rfft10c(FFTZ_UINT8 precision,
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

static FFTZ_VOID r2hc_rfft10c_fp32_fwd(FFTZ_VOID *in_real,
                                       FFTZ_VOID *in_complex,
                                       FFTZ_VOID *out_real,
                                       FFTZ_VOID *out_complex, FFTZ_INTP n,
                                       aoclfftz_strides_t *strides,
                                       FFTZ_VOID *twd, FFTZ_UINT8 flag)
{
    AOCLFFTZ_LOG(DEBUG, global_logger_mode, "Enter");
    const FFTZ_FLOAT CRTM_10_1 =
        0.55901699437494742410229341718281905886015458990288f;
    const FFTZ_FLOAT CRTM_10_2 =
        0.25000000000000000000000000000000000000000000000000f;
    const FFTZ_FLOAT CRTM_10_3 =
        0.58778525229247315738615484497912915412138427663885f;
    const FFTZ_FLOAT CRTM_10_4 =
        0.95105651629515357211643933337938214340569863400000f;

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
    FFTZ_INTP v_out_dc_nyq_stride = strides->v_out_sym_stride;
    FFTZ_INTP cnt;

    for (cnt = 0; cnt < n; cnt++)
    {
        FFTZ_FLOAT v0, v1, v2, v3, v4, v5, v6, v7, v8, v9;
        FFTZ_FLOAT t0, t1, t2, t3, t4, t5, t6, t7, t8, t9, t10, t11, t12, t13,
            t14, t15, t16, t17, t18, t19, t20, t21, t22, t23, t24, t25, t26,
            t27, t28, t29, t30, t31, t32, t33, t34, t35;

        // Input point 1: x(0)
        v0 = *in_r;
        // Input point 2: x(1)
        v1 = in_r[in_strides[1]];
        // Input point 3: x(2)
        v2 = in_r[in_strides[2]];
        // Input point 4: x(3)
        v3 = in_r[in_strides[3]];
        // Input point 5: x(4)
        v4 = in_r[in_strides[4]];
        // Input point 6: x(5)
        v5 = in_r[in_strides[5]];
        // Input point 7: x(6)
        v6 = in_r[in_strides[6]];
        // Input point 8: x(7)
        v7 = in_r[in_strides[7]];
        // Input point 9: x(8)
        v8 = in_r[in_strides[8]];
        // Input point 10: x(9)
        v9 = in_r[in_strides[9]];

        t0 = v0 + v5;
        t1 = v0 - v5;
        t2 = v1 + v9;
        t3 = v1 - v9;
        t4 = v2 + v3;
        t5 = v2 - v3;
        t6 = v4 + v6;
        t7 = v4 - v6;
        t8 = v7 + v8;
        t9 = v7 - v8;

        t10 = t2 + t6;
        t14 = t4 + t8;
        t18 = t10 + t14;
        // Output point 1: X(0)
        *out_r = t0 + t18;

        t11 = t2 - t6;
        t17 = t5 - t9;
        t20 = t11 + t17;
        t21 = t11 - t17;
        // Output point 10: X(9)
        out_r[out_strides[9]] = t1 - t21;

        t22 = CRTM_10_2 * t21;
        t23 = CRTM_10_1 * t20;
        t26 = t22 + t1;
        // Output point 2: X(1)
        out_cp[out_strides[1]] = t26 + t23;
        // Output point 6: X(5)
        out_cp[out_strides[5]] = t26 - t23;

        t12 = t3 + t7;
        t15 = t4 - t8;
        t28 = CRTM_10_3 * t12;
        t34 = CRTM_10_4 * t15;
        // Output point 3: X(2)
        out_cp[out_strides[2]] = -(t28 + t34);
        t30 = CRTM_10_3 * t15;
        t32 = CRTM_10_4 * t12;
        // Output point 7: X(6)
        out_cp[out_strides[6]] = t30 - t32;

        t13 = t7 - t3;
        t16 = t5 + t9;
        t29 = CRTM_10_3 * t13;
        t35 = CRTM_10_4 * t16;
        // Output point 9: X(8)
        out_cp[out_strides[8]] = t35 + t29;
        t31 = CRTM_10_3 * t16;
        t33 = CRTM_10_4 * t13;
        // Output point 5: X(4)
        out_cp[out_strides[4]] = t33 - t31;

        t19 = t10 - t14;
        t24 = CRTM_10_2 * t18;
        t25 = CRTM_10_1 * t19;
        t27 = t0 - t24;
        // Output point 4: X(3)
        out_cp[out_strides[3]] = t27 + t25;
        // Output point 8: X(7)
        out_cp[out_strides[7]] = t27 - t25;

        in_r = in_r + v_in_stride;
        out_cp = out_cp + v_out_stride;
        out_r = out_r + v_out_dc_nyq_stride;
    }
    AOCLFFTZ_LOG(DEBUG, global_logger_mode, "Exit");
}

static FFTZ_VOID r2hc_rfft10c_fp32_bwd(FFTZ_VOID *in_real,
                                       FFTZ_VOID *in_complex,
                                       FFTZ_VOID *out_real,
                                       FFTZ_VOID *out_complex, FFTZ_INTP n,
                                       aoclfftz_strides_t *strides,
                                       FFTZ_VOID *twd, FFTZ_UINT8 flag)
{
    AOCLFFTZ_LOG(DEBUG, global_logger_mode, "Enter");
    const FFTZ_FLOAT CRTM_10_1 =
        1.118033988749894848204586834365638117720309180f;
    const FFTZ_FLOAT CRTM_10_2 =
        0.500000000000000000000000000000000000000000000f;
    const FFTZ_FLOAT CRTM_10_3 =
        2.000000000000000000000000000000000000000000000f;
    const FFTZ_FLOAT CRTM_10_4 =
        1.175570504584946258337411909278145537195304875f;
    const FFTZ_FLOAT CRTM_10_5 =
        1.902113032590307144232878666758764286811397268f;

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
    FFTZ_INTP v_in_dc_nyq_stride = strides->v_in_sym_stride;
    FFTZ_INTP v_out_stride = strides->v_out_stride;
    FFTZ_INTP cnt;

    for (cnt = 0; cnt < n; cnt++)
    {
        FFTZ_FLOAT v0, v1, v2, v3, v4, v5, v6, v7, v8, v9;
        FFTZ_FLOAT t0, t1, t2, t3, t4, t5, t6, t7, t8, t9, t10, t11, t12, t13,
            t14, t15, t16, t17, t18, t19, t20, t21, t22, t23, t24, t25, t26,
            t27, t28, t29, t30, t31, t32, t33, t34, t35, t36, t37;

        // Input point 1: x(0)
        v0 = *in_r;
        // Input point 2: x(1)
        v1 = in_cp[in_strides[1]];
        // Input point 3: x(2)
        v2 = in_cp[in_strides[2]];
        // Input point 4: x(3)
        v3 = in_cp[in_strides[3]];
        // Input point 5: x(4)
        v4 = in_cp[in_strides[4]];
        // Input point 6: x(5)
        v5 = in_cp[in_strides[5]];
        // Input point 7: x(6)
        v6 = in_cp[in_strides[6]];
        // Input point 8: x(7)
        v7 = in_cp[in_strides[7]];
        // Input point 9: x(8)
        v8 = in_cp[in_strides[8]];
        // Input point 10: x(9)
        v9 = in_r[in_strides[9]];

        t0 = v0 + v9;
        t1 = v0 - v9;
        t2 = v1 + v7;
        t3 = v1 - v7;
        t4 = v2 + v8;
        t5 = v2 - v8;
        t30 = CRTM_10_4 * t5;
        t8 = v4 + v6;

        t28 = CRTM_10_4 * t8;
        t29 = CRTM_10_5 * t4;
        t36 = t28 - t29;
        t9 = v4 - v6;
        t31 = CRTM_10_5 * t9;
        t37 = t31 - t30;

        t6 = v3 + v5;
        t7 = v3 - v5;
        t13 = t3 - t7;
        t33 = CRTM_10_3 * t13;
        // Output point 6: X(5)
        out_r[out_strides[5]] = t1 - t33;

        t10 = t2 + t6;
        t32 = CRTM_10_3 * t10;
        // Output point 1: X(0)
        *out_r = t0 + t32;

        t14 = CRTM_10_2 * t10;
        t19 = t0 - t14;
        t11 = t2 - t6;
        t16 = CRTM_10_1 * t11;
        t22 = t19 + t16;
        t23 = t19 - t16;
        // Output point 5: X(4)
        out_r[out_strides[4]] = t23 + t37;
        // Output point 7: X(6)
        out_r[out_strides[6]] = t23 - t37;

        t12 = t3 + t7;
        t17 = CRTM_10_1 * t12;
        t15 = CRTM_10_2 * t13;
        t18 = t1 + t15;
        t20 = t18 + t17;
        t21 = t18 - t17;
        // Output point 4: X(3)
        out_r[out_strides[3]] = t21 + t36;
        // Output point 8: X(7)
        out_r[out_strides[7]] = t21 - t36;

        t24 = CRTM_10_4 * t4;
        t25 = CRTM_10_5 * t8;
        t34 = t24 + t25;
        // Output point 2: X(1)
        out_r[out_strides[1]] = t20 - t34;
        // Output point 10: X(9)
        out_r[out_strides[9]] = t20 + t34;

        t26 = CRTM_10_4 * t9;
        t27 = CRTM_10_5 * t5;
        t35 = t26 + t27;
        // Output point 3: X(2)
        out_r[out_strides[2]] = t22 - t35;
        // Output point 9: X(8)
        out_r[out_strides[8]] = t22 + t35;

        in_cp = in_cp + v_in_stride;
        in_r = in_r + v_in_dc_nyq_stride;
        out_r = out_r + v_out_stride;
    }
    AOCLFFTZ_LOG(DEBUG, global_logger_mode, "Exit");
}

static FFTZ_VOID r2hc_rfft10c_fp64_fwd(FFTZ_VOID *in_real,
                                       FFTZ_VOID *in_complex,
                                       FFTZ_VOID *out_real,
                                       FFTZ_VOID *out_complex, FFTZ_INTP n,
                                       aoclfftz_strides_t *strides,
                                       FFTZ_VOID *twd, FFTZ_UINT8 flag)
{
    AOCLFFTZ_LOG(DEBUG, global_logger_mode, "Enter");
    const FFTZ_DOUBLE CRTM_10_1 =
        0.55901699437494742410229341718281905886015458990288;
    const FFTZ_DOUBLE CRTM_10_2 =
        0.25000000000000000000000000000000000000000000000000;
    const FFTZ_DOUBLE CRTM_10_3 =
        0.58778525229247315738615484497912915412138427663885;
    const FFTZ_DOUBLE CRTM_10_4 =
        0.95105651629515357211643933337938214340569863400000;

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
    FFTZ_INTP v_out_dc_nyq_stride = strides->v_out_sym_stride;
    FFTZ_INTP cnt;

    for (cnt = 0; cnt < n; cnt++)
    {
        FFTZ_DOUBLE v0, v1, v2, v3, v4, v5, v6, v7, v8, v9;
        FFTZ_DOUBLE t0, t1, t2, t3, t4, t5, t6, t7, t8, t9, t10, t11, t12, t13,
            t14, t15, t16, t17, t18, t19, t20, t21, t22, t23, t24, t25, t26,
            t27, t28, t29, t30, t31, t32, t33, t34, t35;

        // Input point 1: x(0)
        v0 = *in_r;
        // Input point 2: x(1)
        v1 = in_r[in_strides[1]];
        // Input point 3: x(2)
        v2 = in_r[in_strides[2]];
        // Input point 4: x(3)
        v3 = in_r[in_strides[3]];
        // Input point 5: x(4)
        v4 = in_r[in_strides[4]];
        // Input point 6: x(5)
        v5 = in_r[in_strides[5]];
        // Input point 7: x(6)
        v6 = in_r[in_strides[6]];
        // Input point 8: x(7)
        v7 = in_r[in_strides[7]];
        // Input point 9: x(8)
        v8 = in_r[in_strides[8]];
        // Input point 10: x(9)
        v9 = in_r[in_strides[9]];

        t0 = v0 + v5;
        t1 = v0 - v5;
        t2 = v1 + v9;
        t3 = v1 - v9;
        t4 = v2 + v3;
        t5 = v2 - v3;
        t6 = v4 + v6;
        t7 = v4 - v6;
        t8 = v7 + v8;
        t9 = v7 - v8;

        t10 = t2 + t6;
        t14 = t4 + t8;
        t18 = t10 + t14;
        // Output point 1: X(0)
        *out_r = t0 + t18;

        t11 = t2 - t6;
        t17 = t5 - t9;
        t20 = t11 + t17;
        t21 = t11 - t17;
        // Output point 10: X(9)
        out_r[out_strides[9]] = t1 - t21;

        t22 = CRTM_10_2 * t21;
        t23 = CRTM_10_1 * t20;
        t26 = t22 + t1;
        // Output point 2: X(1)
        out_cp[out_strides[1]] = t26 + t23;
        // Output point 6: X(5)
        out_cp[out_strides[5]] = t26 - t23;

        t12 = t3 + t7;
        t15 = t4 - t8;
        t28 = CRTM_10_3 * t12;
        t34 = CRTM_10_4 * t15;
        // Output point 3: X(2)
        out_cp[out_strides[2]] = -(t28 + t34);
        t30 = CRTM_10_3 * t15;
        t32 = CRTM_10_4 * t12;
        // Output point 7: X(6)
        out_cp[out_strides[6]] = t30 - t32;

        t13 = t7 - t3;
        t16 = t5 + t9;
        t29 = CRTM_10_3 * t13;
        t35 = CRTM_10_4 * t16;
        // Output point 9: X(8)
        out_cp[out_strides[8]] = t35 + t29;
        t31 = CRTM_10_3 * t16;
        t33 = CRTM_10_4 * t13;
        // Output point 5: X(4)
        out_cp[out_strides[4]] = t33 - t31;

        t19 = t10 - t14;
        t24 = CRTM_10_2 * t18;
        t25 = CRTM_10_1 * t19;
        t27 = t0 - t24;
        // Output point 4: X(3)
        out_cp[out_strides[3]] = t27 + t25;
        // Output point 8: X(7)
        out_cp[out_strides[7]] = t27 - t25;

        in_r = in_r + v_in_stride;
        out_cp = out_cp + v_out_stride;
        out_r = out_r + v_out_dc_nyq_stride;
    }
    AOCLFFTZ_LOG(DEBUG, global_logger_mode, "Exit");
}

static FFTZ_VOID r2hc_rfft10c_fp64_bwd(FFTZ_VOID *in_real,
                                       FFTZ_VOID *in_complex,
                                       FFTZ_VOID *out_real,
                                       FFTZ_VOID *out_complex, FFTZ_INTP n,
                                       aoclfftz_strides_t *strides,
                                       FFTZ_VOID *twd, FFTZ_UINT8 flag)
{
    AOCLFFTZ_LOG(DEBUG, global_logger_mode, "Enter");
    const FFTZ_DOUBLE CRTM_10_1 =
        1.118033988749894848204586834365638117720309180;
    const FFTZ_DOUBLE CRTM_10_2 =
        0.500000000000000000000000000000000000000000000;
    const FFTZ_DOUBLE CRTM_10_3 =
        2.000000000000000000000000000000000000000000000;
    const FFTZ_DOUBLE CRTM_10_4 =
        1.175570504584946258337411909278145537195304875;
    const FFTZ_DOUBLE CRTM_10_5 =
        1.902113032590307144232878666758764286811397268;

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
    FFTZ_INTP v_in_dc_nyq_stride = strides->v_in_sym_stride;
    FFTZ_INTP v_out_stride = strides->v_out_stride;
    FFTZ_INTP cnt;

    for (cnt = 0; cnt < n; cnt++)
    {
        FFTZ_DOUBLE v0, v1, v2, v3, v4, v5, v6, v7, v8, v9;
        FFTZ_DOUBLE t0, t1, t2, t3, t4, t5, t6, t7, t8, t9, t10, t11, t12, t13,
            t14, t15, t16, t17, t18, t19, t20, t21, t22, t23, t24, t25, t26,
            t27, t28, t29, t30, t31, t32, t33, t34, t35, t36, t37;

        // Input point 1: x(0)
        v0 = *in_r;
        // Input point 2: x(1)
        v1 = in_cp[in_strides[1]];
        // Input point 3: x(2)
        v2 = in_cp[in_strides[2]];
        // Input point 4: x(3)
        v3 = in_cp[in_strides[3]];
        // Input point 5: x(4)
        v4 = in_cp[in_strides[4]];
        // Input point 6: x(5)
        v5 = in_cp[in_strides[5]];
        // Input point 7: x(6)
        v6 = in_cp[in_strides[6]];
        // Input point 8: x(7)
        v7 = in_cp[in_strides[7]];
        // Input point 9: x(8)
        v8 = in_cp[in_strides[8]];
        // Input point 10: x(9)
        v9 = in_r[in_strides[9]];

        t0 = v0 + v9;
        t1 = v0 - v9;
        t2 = v1 + v7;
        t3 = v1 - v7;
        t4 = v2 + v8;
        t5 = v2 - v8;
        t30 = CRTM_10_4 * t5;
        t8 = v4 + v6;

        t28 = CRTM_10_4 * t8;
        t29 = CRTM_10_5 * t4;
        t36 = t28 - t29;
        t9 = v4 - v6;
        t31 = CRTM_10_5 * t9;
        t37 = t31 - t30;

        t6 = v3 + v5;
        t7 = v3 - v5;
        t13 = t3 - t7;
        t33 = CRTM_10_3 * t13;
        // Output point 6: X(5)
        out_r[out_strides[5]] = t1 - t33;

        t10 = t2 + t6;
        t32 = CRTM_10_3 * t10;
        // Output point 1: X(0)
        *out_r = t0 + t32;

        t14 = CRTM_10_2 * t10;
        t19 = t0 - t14;
        t11 = t2 - t6;
        t16 = CRTM_10_1 * t11;
        t22 = t19 + t16;
        t23 = t19 - t16;
        // Output point 5: X(4)
        out_r[out_strides[4]] = t23 + t37;
        // Output point 7: X(6)
        out_r[out_strides[6]] = t23 - t37;

        t12 = t3 + t7;
        t17 = CRTM_10_1 * t12;
        t15 = CRTM_10_2 * t13;
        t18 = t1 + t15;
        t20 = t18 + t17;
        t21 = t18 - t17;
        // Output point 4: X(3)
        out_r[out_strides[3]] = t21 + t36;
        // Output point 8: X(7)
        out_r[out_strides[7]] = t21 - t36;

        t24 = CRTM_10_4 * t4;
        t25 = CRTM_10_5 * t8;
        t34 = t24 + t25;
        // Output point 2: X(1)
        out_r[out_strides[1]] = t20 - t34;
        // Output point 10: X(9)
        out_r[out_strides[9]] = t20 + t34;

        t26 = CRTM_10_4 * t9;
        t27 = CRTM_10_5 * t5;
        t35 = t26 + t27;
        // Output point 3: X(2)
        out_r[out_strides[2]] = t22 - t35;
        // Output point 9: X(8)
        out_r[out_strides[8]] = t22 + t35;

        in_cp = in_cp + v_in_stride;
        in_r = in_r + v_in_dc_nyq_stride;
        out_r = out_r + v_out_stride;
    }
    AOCLFFTZ_LOG(DEBUG, global_logger_mode, "Exit");
}

kfft_ register_kernel_r2hc_rfft10c(FFTZ_UINT8 precision, FFTZ_UINT8 direction)
{
    if (direction == FORWARD_FFT_DIR)
    {
        if (precision == DT_FLOAT)
        {
            return r2hc_rfft10c_fp32_fwd;
        }
        else if (precision == DT_DOUBLE)
        {
            return r2hc_rfft10c_fp64_fwd;
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
            return r2hc_rfft10c_fp32_bwd;
        }
        else if (precision == DT_DOUBLE)
        {
            return r2hc_rfft10c_fp64_bwd;
        }
        else
        {
            return NULL;
        }
    }
}
