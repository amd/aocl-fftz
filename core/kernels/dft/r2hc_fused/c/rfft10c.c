// Copyright Advanced Micro Devices, Inc.
// SPDX-License-Identifier: BSD-3-Clause

/** @file rfft10c.c
 *
 *  @brief Radix-10 r2hc_fused Real-FFT kernel with scalar operations in C
 *
 *  This file contains the DIT radix-10 real-to-halfcomplex fused of two
 *  different implementations (Standard DFT and Shifted DFT that differs in DFT
 *  weight matrix) using scalar operations for single-precision
 *  and double-precision inputs.
 *
 *  @author Jeevanantham N
 */

#include "core/kernels/kernel.h"

static const ops_cycles_t ops_cnt[NUM_PRECISIONS][NUM_FFT_DIRS] =
                                                    {{{0, 24, 66, 40, 0, 0},
                                                      {0, 28, 68, 40, 0, 0}},
                                                     {{0, 24, 66, 40, 0, 0},
                                                      {0, 28, 68, 40, 0, 0}}};

ops_cycles_t get_ops_cnt_r2hcf_rfft10c(FFTZ_UINT8 precision,
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

static FFTZ_VOID r2hcf_rfft10c_fp32_fwd(FFTZ_VOID *in_real, FFTZ_VOID *in_imag,
                                        FFTZ_VOID *out_real,
                                        FFTZ_VOID *out_imag, FFTZ_INTP n,
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

    for (cnt = 0; cnt < n; cnt++)
    {
        /* Standard DFT */
        FFTZ_FLOAT av0, av1, av2, av3, av4, av5, av6, av7, av8, av9;
        FFTZ_FLOAT at0, at1, at2, at3, at4, at5, at6, at7, at8, at9, at10, at11,
              at12, at13, at14, at15, at16, at17, at18, at19, at20, at21, at22,
              at23, at24, at25, at26, at27, at28, at29, at30, at31, at32, at33,
              at34, at35;

        // Input point 1: x(0)
        av0 = *in;
        // Input point 3: x(2)
        av1 = in[in_strides[2]];
        // Input point 5: x(4)
        av2 = in[in_strides[4]];
        // Input point 7: x(6)
        av3 = in[in_strides[6]];
        // Input point 9: x(8)
        av4 = in[in_strides[8]];
        // Input point 11: x(10)
        av5 = in[in_strides[10]];
        // Input point 13: x(12)
        av6 = in[in_strides[12]];
        // Input point 15: x(14)
        av7 = in[in_strides[14]];
        // Input point 17: x(16)
        av8 = in[in_strides[16]];
        // Input point 19: x(18)
        av9 = in[in_strides[18]];

        at0 = av0 + av5;
        at1 = av0 - av5;
        at2 = av1 + av9;
        at3 = av1 - av9;
        at4 = av2 + av3;
        at5 = av2 - av3;
        at6 = av4 + av6;
        at7 = av4 - av6;
        at8 = av7 + av8;
        at9 = av7 - av8;

        at10 = at2 + at6;
        at14 = at4 + at8;
        at18 = at10 + at14;
        // Output point 1: X(0)
        *out = at0 + at18;

        at11 = at2 - at6;
        at17 = at5 - at9;
        at20 = at11 + at17;
        at21 = at11 - at17;
        // Output point 20: X(19)
        out[out_strides[19]] = at1 - at21;

        at22 = CRTM_10_2 * at21;
        at23 = CRTM_10_1 * at20;
        at26 = at22 + at1;
        // Output point 4: X(3)
        out[out_strides[3]] = at26 + at23;
        // Output point 12: X(11)
        out[out_strides[11]] = at26 - at23;

        at12 = at3 + at7;
        at15 = at4 - at8;
        at28 = CRTM_10_3 * at12;
        at34 = CRTM_10_4 * at15;
        // Output point 5: X(4)
        out[out_strides[4]] = -(at28 + at34);

        at30 = CRTM_10_3 * at15;
        at32 = CRTM_10_4 * at12;
        // Output point 13: X(12)
        out[out_strides[12]] = at30 - at32;

        at13 = at7 - at3;
        at16 = at5 + at9;
        at29 = CRTM_10_3 * at13;
        at35 = CRTM_10_4 * at16;
        // Output point 17: X(16)
        out[out_strides[16]] = at35 + at29;

        at31 = CRTM_10_3 * at16;
        at33 = CRTM_10_4 * at13;
        // Output point 9: X(8)
        out[out_strides[8]] = at33 - at31;

        at19 = at10 - at14;
        at24 = CRTM_10_2 * at18;
        at25 = CRTM_10_1 * at19;
        at27 = at0 - at24;
        // Output point 8: X(7)
        out[out_strides[7]] = at27 + at25;
        // Output point 16: X(15)
        out[out_strides[15]] = at27 - at25;

        /* Shifted DFT */
        FFTZ_FLOAT bv0, bv1, bv2, bv3, bv4, bv5, bv6, bv7, bv8, bv9;
        FFTZ_FLOAT bt0, bt1, bt2, bt3, bt4, bt5, bt6, bt7, bt8, bt9, bt10, bt11,
              bt12, bt13, bt14, bt15, bt16, bt17, bt18, bt19, bt20, bt21, bt22,
              bt23, bt24, bt25, bt26, bt27, bt28, bt29, bt30, bt31, bt32, bt33;

        // Input point 2: x(1)
        bv0 = in[in_strides[1]];
        // Input point 4: x(3)
        bv1 = in[in_strides[3]];
        // Input point 6: x(5)
        bv2 = in[in_strides[5]];
        // Input point 8: x(7)
        bv3 = in[in_strides[7]];
        // Input point 10: x(9)
        bv4 = in[in_strides[9]];
        // Input point 12: x(11)
        bv5 = in[in_strides[11]];
        // Input point 14: x(13)
        bv6 = in[in_strides[13]];
        // Input point 16: x(15)
        bv7 = in[in_strides[15]];
        // Input point 18: x(17)
        bv8 = in[in_strides[17]];
        // Input point 20: x(19)
        bv9 = in[in_strides[19]];

        bt0 = bv1 + bv9;
        bt1 = bv1 - bv9;
        bt2 = bv2 + bv8;
        bt3 = bv2 - bv8;
        bt4 = bv3 + bv7;
        bt5 = bv3 - bv7;
        bt6 = bv4 + bv6;
        bt7 = bv4 - bv6;

        bt8 = bt0 + bt4;
        bt9 = bt0 - bt4;
        bt10 = bt3 + bt7;
        bt11 = bt3 - bt7;
        // Output point 11: X(10)
        out[out_strides[10]] = -(bt9 + bv5);
        // Output point 10: X(9)
        out[out_strides[9]] = bv0 - bt11;

        bt12 = CRTM_10_2 * bt11;
        bt13 = CRTM_10_1 * bt10;
        bt14 = CRTM_10_1 * bt8;
        bt15 = CRTM_10_2 * bt9;

        bt16 = bv0 + bt12;
        bt17 = bt15 - bv5;

        bt18 = CRTM_10_4 * bt1;
        bt19 = CRTM_10_3 * bt5;
        bt20 = CRTM_10_4 * bt6;
        bt21 = CRTM_10_3 * bt2;
        bt22 = CRTM_10_4 * bt5;
        bt23 = CRTM_10_3 * bt1;
        bt24 = CRTM_10_4 * bt2;
        bt25 = CRTM_10_3 * bt6;

        bt26 = bt16 + bt13;
        bt27 = bt18 + bt19;
        // Output point 2: X(1)
        out[out_strides[1]] = bt26 + bt27;
        // Output point 18: X(17)
        out[out_strides[17]] = bt26 - bt27;

        bt28 = bt17 - bt14;
        bt29 = bt20 + bt21;
        // Output point 3: X(2)
        out[out_strides[2]] = bt28 - bt29;
        // Output point 19: X(18)
        out[out_strides[18]] = bt28 + bt29;

        bt30 = bt16 - bt13;
        bt31 = bt23 - bt22;
        // Output point 6: X(5)
        out[out_strides[5]] = bt30 + bt31;
        // Output point 14: X(13)
        out[out_strides[13]] = bt30 - bt31;

        bt32 = bt17 + bt14;
        bt33 = bt25 - bt24;
        // Output point 7: X(6)
        out[out_strides[6]] = bt33 - bt32;
        // Output point 15: X(14)
        out[out_strides[14]] = -(bt33 + bt32);

        in = in + v_in_stride;
        out = out + v_out_stride;
    }
    AOCLFFTZ_LOG(DEBUG, global_logger_mode, "Exit");
}

static FFTZ_VOID r2hcf_rfft10c_fp32_bwd(FFTZ_VOID *in_real, FFTZ_VOID *in_imag,
                                        FFTZ_VOID *out_real,
                                        FFTZ_VOID *out_imag, FFTZ_INTP n,
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

    for (cnt = 0; cnt < n; cnt++)
    {
        /* Standard DFT */
        FFTZ_FLOAT av0, av1, av2, av3, av4, av5, av6, av7, av8, av9;
        FFTZ_FLOAT at0, at1, at2, at3, at4, at5, at6, at7, at8, at9, at10, at11,
              at12, at13, at14, at15, at16, at17, at18, at19, at20, at21, at22,
              at23, at24, at25, at26, at27, at28, at29, at30, at31, at32, at33,
              at34, at35, at36, at37;

        //  Input point 1: x(0)
        av0 = *in;
        // Input point 4: x(3)
        av1 = in[in_strides[3]];
        // Input point 5: x(4)
        av2 = in[in_strides[4]];
        // Input point 8: x(7)
        av3 = in[in_strides[7]];
        // Input point 9: x(8)
        av4 = in[in_strides[8]];
        // Input point 12: x(11)
        av5 = in[in_strides[11]];
        // Input point 13: x(12)
        av6 = in[in_strides[12]];
        // Input point 16: x(15)
        av7 = in[in_strides[15]];
        // Input point 17: x(16)
        av8 = in[in_strides[16]];
        // Input point 20: x(19)
        av9 = in[in_strides[19]];

        at0 = av0 + av9;
        at1 = av0 - av9;
        at2 = av1 + av7;
        at3 = av1 - av7;
        at4 = av2 + av8;
        at5 = av2 - av8;
        at30 = CRTM_10_4 * at5;
        at8 = av4 + av6;

        at28 = CRTM_10_4 * at8;
        at29 = CRTM_10_5 * at4;
        at36 = at28 - at29;
        at9 = av4 - av6;
        at31 = CRTM_10_5 * at9;
        at37 = at31 - at30;

        at6 = av3 + av5;
        at7 = av3 - av5;
        at13 = at3 - at7;
        at33 = CRTM_10_3 * at13;
        // Output point 11: X(10)
        out[out_strides[10]] = at1 - at33;

        at10 = at2 + at6;
        at32 = CRTM_10_3 * at10;
        // Output point 1: X(0)
        *out = at0 + at32;

        at14 = CRTM_10_2 * at10;
        at19 = at0 - at14;
        at11 = at2 - at6;
        at16 = CRTM_10_1 * at11;
        at22 = at19 + at16;
        at23 = at19 - at16;
        // Output point 9: X(8)
        out[out_strides[8]] = at23 + at37;
        // Output point 13: X(12)
        out[out_strides[12]] = at23 - at37;

        at12 = at3 + at7;
        at17 = CRTM_10_1 * at12;
        at15 = CRTM_10_2 * at13;
        at18 = at1 + at15;
        at20 = at18 + at17;
        at21 = at18 - at17;
        // Output point 7: X(6)
        out[out_strides[6]] = at21 + at36;
        // Output point 15: X(14)
        out[out_strides[14]] = at21 - at36;

        at24 = CRTM_10_4 * at4;
        at25 = CRTM_10_5 * at8;
        at34 = at24 + at25;
        // Output point 3: X(2)
        out[out_strides[2]] = at20 - at34;
        // Output point 19: X(18)
        out[out_strides[18]] = at20 + at34;

        at26 = CRTM_10_4 * at9;
        at27 = CRTM_10_5 * at5;
        at35 = at26 + at27;
        // Output point 5: X(4)
        out[out_strides[4]] = at22 - at35;
        // Output point 17: X(16)
        out[out_strides[16]] = at22 + at35;

        /* Shifted DFT */
        FFTZ_FLOAT bv0, bv1, bv2, bv3, bv4, bv5, bv6, bv7, bv8, bv9;
        FFTZ_FLOAT bt0, bt1, bt2, bt3, bt4, bt5, bt6, bt7, bt8, bt9, bt10, bt11,
              bt12, bt13, bt14, bt15, bt16, bt17, bt18, bt19, bt20, bt21, bt22,
              bt23, bt24, bt25, bt26, bt27, bt28, bt29, bt30, bt31, bt32, bt33,
              bt34, bt35, bt36, bt37;

        // Input point 2: x(1)
        bv0 = in[in_strides[1]];
        // Input point 3: x(2)
        bv1 = in[in_strides[2]];
        // Input point 6: x(5)
        bv2 = in[in_strides[5]];
        // Input point 7: x(6)
        bv3 = in[in_strides[6]];
        // Input point 10: x(9)
        bv4 = in[in_strides[9]];
        // Input point 11: x(10)
        bv5 = in[in_strides[10]];
        // Input point 14: x(13)
        bv6 = in[in_strides[13]];
        // Input point 15: x(14)
        bv7 = in[in_strides[14]];
        // Input point 18: x(17)
        bv8 = in[in_strides[17]];
        // Input point 19: x(18)
        bv9 = in[in_strides[18]];

        bt0 = bv0 + bv8;
        bt1 = bv0 - bv8;
        bt2 = bv1 + bv9;
        bt3 = bv1 - bv9;
        bt4 = bv2 + bv6;
        bt5 = bv2 - bv6;
        bt6 = bv3 + bv7;
        bt7 = bv3 - bv7;

        bt8 = bt0 + bt4;
        bt9 = bt0 - bt4;
        bt10 = bt2 + bt6;
        bt11 = bt2 - bt6;

        bt12 = CRTM_10_1 * bt10;
        bt13 = CRTM_10_2 * bt11;
        bt14 = CRTM_10_1 * bt9;
        bt15 = CRTM_10_2 * bt8;
        bt16 = CRTM_10_3 * bv4;
        bt17 = CRTM_10_3 * bv5;
        bt36 = bt8 + bt8;
        // Output point 2: X(1)
        out[out_strides[1]] = bt16 + bt36;
        bt37 = bt11 + bt11;
        // Output point 12: X(11)
        out[out_strides[11]] = -(bt17 + bt37);

        bt18 = bt13 - bt17;
        bt19 = bt15 - bt16;

        bt20 = bt18 - bt12;
        bt21 = bt18 + bt12;
        bt22 = bt14 + bt19;
        bt23 = bt14 - bt19;

        bt24 = CRTM_10_5 * bt1;
        bt25 = CRTM_10_5 * bt3;
        bt26 = CRTM_10_5 * bt5;
        bt27 = CRTM_10_5 * bt7;
        bt28 = CRTM_10_4 * bt1;
        bt29 = CRTM_10_4 * bt3;
        bt30 = CRTM_10_4 * bt5;
        bt31 = CRTM_10_4 * bt7;

        bt32 = bt24 + bt30;
        // Output point 4: X(3)
        out[out_strides[3]] = bt20 + bt32;
        // Output point 20: X(19)
        out[out_strides[19]] = bt20 - bt32;

        bt33 = bt27 + bt29;
        // Output point 6: X(5)
        out[out_strides[5]] = bt22 - bt33;
        // Output point 18: X(17)
        out[out_strides[17]] = -(bt22 + bt33);

        bt34 = bt28 - bt26;
        // Output point 8: X(7)
        out[out_strides[7]] = bt34 - bt21;
        // Output point 16: X(15)
        out[out_strides[15]] = -(bt21 + bt34);

        bt35 = bt31 - bt25;
        // Output point 10: X(9)
        out[out_strides[9]] = bt35 + bt23;
        // Output point 14: X(13)
        out[out_strides[13]] = bt35 - bt23;

        in = in + v_in_stride;
        out = out + v_out_stride;
    }
    AOCLFFTZ_LOG(DEBUG, global_logger_mode, "Exit");
}

static FFTZ_VOID r2hcf_rfft10c_fp64_fwd(FFTZ_VOID *in_real, FFTZ_VOID *in_imag,
                                        FFTZ_VOID *out_real,
                                        FFTZ_VOID *out_imag, FFTZ_INTP n,
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

    for (cnt = 0; cnt < n; cnt++)
    {
        /* Standard DFT */
        FFTZ_DOUBLE av0, av1, av2, av3, av4, av5, av6, av7, av8, av9;
        FFTZ_DOUBLE at0, at1, at2, at3, at4, at5, at6, at7, at8, at9, at10,
            at11, at12, at13, at14, at15, at16, at17, at18, at19, at20, at21,
            at22, at23, at24, at25, at26, at27, at28, at29, at30, at31, at32,
            at33, at34, at35;

        // Input point 1: x(0)
        av0 = *in;
        // Input point 3: x(2)
        av1 = in[in_strides[2]];
        // Input point 5: x(4)
        av2 = in[in_strides[4]];
        // Input point 7: x(6)
        av3 = in[in_strides[6]];
        // Input point 9: x(8)
        av4 = in[in_strides[8]];
        // Input point 11: x(10)
        av5 = in[in_strides[10]];
        // Input point 13: x(12)
        av6 = in[in_strides[12]];
        // Input point 15: x(14)
        av7 = in[in_strides[14]];
        // Input point 17: x(16)
        av8 = in[in_strides[16]];
        // Input point 19: x(18)
        av9 = in[in_strides[18]];

        at0 = av0 + av5;
        at1 = av0 - av5;
        at2 = av1 + av9;
        at3 = av1 - av9;
        at4 = av2 + av3;
        at5 = av2 - av3;
        at6 = av4 + av6;
        at7 = av4 - av6;
        at8 = av7 + av8;
        at9 = av7 - av8;

        at10 = at2 + at6;
        at14 = at4 + at8;
        at18 = at10 + at14;
        // Output point 1: X(0)
        *out = at0 + at18;

        at11 = at2 - at6;
        at17 = at5 - at9;
        at20 = at11 + at17;
        at21 = at11 - at17;
        // Output point 20: X(19)
        out[out_strides[19]] = at1 - at21;

        at22 = CRTM_10_2 * at21;
        at23 = CRTM_10_1 * at20;
        at26 = at22 + at1;
        // Output point 4: X(3)
        out[out_strides[3]] = at26 + at23;
        // Output point 12: X(11)
        out[out_strides[11]] = at26 - at23;

        at12 = at3 + at7;
        at15 = at4 - at8;
        at28 = CRTM_10_3 * at12;
        at34 = CRTM_10_4 * at15;
        // Output point 5: X(4)
        out[out_strides[4]] = -(at28 + at34);

        at30 = CRTM_10_3 * at15;
        at32 = CRTM_10_4 * at12;
        // Output point 13: X(12)
        out[out_strides[12]] = at30 - at32;

        at13 = at7 - at3;
        at16 = at5 + at9;
        at29 = CRTM_10_3 * at13;
        at35 = CRTM_10_4 * at16;
        // Output point 17: X(16)
        out[out_strides[16]] = at35 + at29;

        at31 = CRTM_10_3 * at16;
        at33 = CRTM_10_4 * at13;
        // Output point 9: X(8)
        out[out_strides[8]] = at33 - at31;

        at19 = at10 - at14;
        at24 = CRTM_10_2 * at18;
        at25 = CRTM_10_1 * at19;
        at27 = at0 - at24;
        // Output point 8: X(7)
        out[out_strides[7]] = at27 + at25;
        // Output point 16: X(15)
        out[out_strides[15]] = at27 - at25;

        /* Shifted DFT */
        FFTZ_DOUBLE bv0, bv1, bv2, bv3, bv4, bv5, bv6, bv7, bv8, bv9;
        FFTZ_DOUBLE bt0, bt1, bt2, bt3, bt4, bt5, bt6, bt7, bt8, bt9, bt10,
            bt11, bt12, bt13, bt14, bt15, bt16, bt17, bt18, bt19, bt20, bt21,
            bt22, bt23, bt24, bt25, bt26, bt27, bt28, bt29, bt30, bt31, bt32,
            bt33;

        // Input point 2: x(1)
        bv0 = in[in_strides[1]];
        // Input point 4: x(3)
        bv1 = in[in_strides[3]];
        // Input point 6: x(5)
        bv2 = in[in_strides[5]];
        // Input point 8: x(7)
        bv3 = in[in_strides[7]];
        // Input point 10: x(9)
        bv4 = in[in_strides[9]];
        // Input point 12: x(11)
        bv5 = in[in_strides[11]];
        // Input point 14: x(13)
        bv6 = in[in_strides[13]];
        // Input point 16: x(15)
        bv7 = in[in_strides[15]];
        // Input point 18: x(17)
        bv8 = in[in_strides[17]];
        // Input point 20: x(19)
        bv9 = in[in_strides[19]];

        bt0 = bv1 + bv9;
        bt1 = bv1 - bv9;
        bt2 = bv2 + bv8;
        bt3 = bv2 - bv8;
        bt4 = bv3 + bv7;
        bt5 = bv3 - bv7;
        bt6 = bv4 + bv6;
        bt7 = bv4 - bv6;

        bt8 = bt0 + bt4;
        bt9 = bt0 - bt4;
        bt10 = bt3 + bt7;
        bt11 = bt3 - bt7;
        // Output point 11: X(10)
        out[out_strides[10]] = -(bt9 + bv5);
        // Output point 10: X(9)
        out[out_strides[9]] = bv0 - bt11;

        bt12 = CRTM_10_2 * bt11;
        bt13 = CRTM_10_1 * bt10;
        bt14 = CRTM_10_1 * bt8;
        bt15 = CRTM_10_2 * bt9;

        bt16 = bv0 + bt12;
        bt17 = bt15 - bv5;

        bt18 = CRTM_10_4 * bt1;
        bt19 = CRTM_10_3 * bt5;
        bt20 = CRTM_10_4 * bt6;
        bt21 = CRTM_10_3 * bt2;
        bt22 = CRTM_10_4 * bt5;
        bt23 = CRTM_10_3 * bt1;
        bt24 = CRTM_10_4 * bt2;
        bt25 = CRTM_10_3 * bt6;

        bt26 = bt16 + bt13;
        bt27 = bt18 + bt19;
        // Output point 2: X(1)
        out[out_strides[1]] = bt26 + bt27;
        // Output point 18: X(17)
        out[out_strides[17]] = bt26 - bt27;

        bt28 = bt17 - bt14;
        bt29 = bt20 + bt21;
        // Output point 3: X(2)
        out[out_strides[2]] = bt28 - bt29;
        // Output point 19: X(18)
        out[out_strides[18]] = bt28 + bt29;

        bt30 = bt16 - bt13;
        bt31 = bt23 - bt22;
        // Output point 6: X(5)
        out[out_strides[5]] = bt30 + bt31;
        // Output point 14: X(13)
        out[out_strides[13]] = bt30 - bt31;

        bt32 = bt17 + bt14;
        bt33 = bt25 - bt24;
        // Output point 7: X(6)
        out[out_strides[6]] = bt33 - bt32;
        // Output point 15: X(14)
        out[out_strides[14]] = -(bt33 + bt32);

        in = in + v_in_stride;
        out = out + v_out_stride;
    }
    AOCLFFTZ_LOG(DEBUG, global_logger_mode, "Exit");
}

static FFTZ_VOID r2hcf_rfft10c_fp64_bwd(FFTZ_VOID *in_real, FFTZ_VOID *in_imag,
                                        FFTZ_VOID *out_real,
                                        FFTZ_VOID *out_imag, FFTZ_INTP n,
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

    for (cnt = 0; cnt < n; cnt++)
    {
        /* Standard DFT */
        FFTZ_DOUBLE av0, av1, av2, av3, av4, av5, av6, av7, av8, av9;
        FFTZ_DOUBLE at0, at1, at2, at3, at4, at5, at6, at7, at8, at9, at10,
            at11, at12, at13, at14, at15, at16, at17, at18, at19, at20, at21,
            at22, at23, at24, at25, at26, at27, at28, at29, at30, at31, at32,
            at33, at34, at35, at36, at37;

        //  Input point 1: x(0)
        av0 = *in;
        // Input point 4: x(3)
        av1 = in[in_strides[3]];
        // Input point 5: x(4)
        av2 = in[in_strides[4]];
        // Input point 8: x(7)
        av3 = in[in_strides[7]];
        // Input point 9: x(8)
        av4 = in[in_strides[8]];
        // Input point 12: x(11)
        av5 = in[in_strides[11]];
        // Input point 13: x(12)
        av6 = in[in_strides[12]];
        // Input point 16: x(15)
        av7 = in[in_strides[15]];
        // Input point 17: x(16)
        av8 = in[in_strides[16]];
        // Input point 20: x(19)
        av9 = in[in_strides[19]];

        at0 = av0 + av9;
        at1 = av0 - av9;
        at2 = av1 + av7;
        at3 = av1 - av7;
        at4 = av2 + av8;
        at5 = av2 - av8;
        at30 = CRTM_10_4 * at5;
        at8 = av4 + av6;

        at28 = CRTM_10_4 * at8;
        at29 = CRTM_10_5 * at4;
        at36 = at28 - at29;
        at9 = av4 - av6;
        at31 = CRTM_10_5 * at9;
        at37 = at31 - at30;

        at6 = av3 + av5;
        at7 = av3 - av5;
        at13 = at3 - at7;
        at33 = CRTM_10_3 * at13;
        // Output point 11: X(10)
        out[out_strides[10]] = at1 - at33;

        at10 = at2 + at6;
        at32 = CRTM_10_3 * at10;
        // Output point 1: X(0)
        *out = at0 + at32;

        at14 = CRTM_10_2 * at10;
        at19 = at0 - at14;
        at11 = at2 - at6;
        at16 = CRTM_10_1 * at11;
        at22 = at19 + at16;
        at23 = at19 - at16;
        // Output point 9: X(8)
        out[out_strides[8]] = at23 + at37;
        // Output point 13: X(12)
        out[out_strides[12]] = at23 - at37;

        at12 = at3 + at7;
        at17 = CRTM_10_1 * at12;
        at15 = CRTM_10_2 * at13;
        at18 = at1 + at15;
        at20 = at18 + at17;
        at21 = at18 - at17;
        // Output point 7: X(6)
        out[out_strides[6]] = at21 + at36;
        // Output point 15: X(14)
        out[out_strides[14]] = at21 - at36;

        at24 = CRTM_10_4 * at4;
        at25 = CRTM_10_5 * at8;
        at34 = at24 + at25;
        // Output point 3: X(2)
        out[out_strides[2]] = at20 - at34;
        // Output point 19: X(18)
        out[out_strides[18]] = at20 + at34;

        at26 = CRTM_10_4 * at9;
        at27 = CRTM_10_5 * at5;
        at35 = at26 + at27;
        // Output point 5: X(4)
        out[out_strides[4]] = at22 - at35;
        // Output point 17: X(16)
        out[out_strides[16]] = at22 + at35;

        /* Shifted DFT */
        FFTZ_DOUBLE bv0, bv1, bv2, bv3, bv4, bv5, bv6, bv7, bv8, bv9;
        FFTZ_DOUBLE bt0, bt1, bt2, bt3, bt4, bt5, bt6, bt7, bt8, bt9, bt10,
            bt11, bt12, bt13, bt14, bt15, bt16, bt17, bt18, bt19, bt20, bt21,
            bt22, bt23, bt24, bt25, bt26, bt27, bt28, bt29, bt30, bt31, bt32,
            bt33, bt34, bt35, bt36, bt37;

        // Input point 2: x(1)
        bv0 = in[in_strides[1]];
        // Input point 3: x(2)
        bv1 = in[in_strides[2]];
        // Input point 6: x(5)
        bv2 = in[in_strides[5]];
        // Input point 7: x(6)
        bv3 = in[in_strides[6]];
        // Input point 10: x(9)
        bv4 = in[in_strides[9]];
        // Input point 11: x(10)
        bv5 = in[in_strides[10]];
        // Input point 14: x(13)
        bv6 = in[in_strides[13]];
        // Input point 15: x(14)
        bv7 = in[in_strides[14]];
        // Input point 18: x(17)
        bv8 = in[in_strides[17]];
        // Input point 19: x(18)
        bv9 = in[in_strides[18]];

        bt0 = bv0 + bv8;
        bt1 = bv0 - bv8;
        bt2 = bv1 + bv9;
        bt3 = bv1 - bv9;
        bt4 = bv2 + bv6;
        bt5 = bv2 - bv6;
        bt6 = bv3 + bv7;
        bt7 = bv3 - bv7;

        bt8 = bt0 + bt4;
        bt9 = bt0 - bt4;
        bt10 = bt2 + bt6;
        bt11 = bt2 - bt6;

        bt12 = CRTM_10_1 * bt10;
        bt13 = CRTM_10_2 * bt11;
        bt14 = CRTM_10_1 * bt9;
        bt15 = CRTM_10_2 * bt8;
        bt16 = CRTM_10_3 * bv4;
        bt17 = CRTM_10_3 * bv5;
        bt36 = bt8 + bt8;
        // Output point 2: X(1)
        out[out_strides[1]] = bt16 + bt36;
        bt37 = bt11 + bt11;
        // Output point 12: X(11)
        out[out_strides[11]] = -(bt17 + bt37);

        bt18 = bt13 - bt17;
        bt19 = bt15 - bt16;

        bt20 = bt18 - bt12;
        bt21 = bt18 + bt12;
        bt22 = bt14 + bt19;
        bt23 = bt14 - bt19;

        bt24 = CRTM_10_5 * bt1;
        bt25 = CRTM_10_5 * bt3;
        bt26 = CRTM_10_5 * bt5;
        bt27 = CRTM_10_5 * bt7;
        bt28 = CRTM_10_4 * bt1;
        bt29 = CRTM_10_4 * bt3;
        bt30 = CRTM_10_4 * bt5;
        bt31 = CRTM_10_4 * bt7;

        bt32 = bt24 + bt30;
        // Output point 4: X(3)
        out[out_strides[3]] = bt20 + bt32;
        // Output point 20: X(19)
        out[out_strides[19]] = bt20 - bt32;

        bt33 = bt27 + bt29;
        // Output point 6: X(5)
        out[out_strides[5]] = bt22 - bt33;
        // Output point 18: X(17)
        out[out_strides[17]] = -(bt22 + bt33);

        bt34 = bt28 - bt26;
        // Output point 8: X(7)
        out[out_strides[7]] = bt34 - bt21;
        // Output point 16: X(15)
        out[out_strides[15]] = -(bt21 + bt34);

        bt35 = bt31 - bt25;
        // Output point 10: X(9)
        out[out_strides[9]] = bt35 + bt23;
        // Output point 14: X(13)
        out[out_strides[13]] = bt35 - bt23;

        in = in + v_in_stride;
        out = out + v_out_stride;
    }
    AOCLFFTZ_LOG(DEBUG, global_logger_mode, "Exit");
}

kfft_ register_kernel_r2hcf_rfft10c(FFTZ_UINT8 precision, FFTZ_UINT8 direction)
{
    if (direction == FORWARD_FFT_DIR)
    {
        if (precision == DT_FLOAT)
        {
            return r2hcf_rfft10c_fp32_fwd;
        }
        else if (precision == DT_DOUBLE)
        {
            return r2hcf_rfft10c_fp64_fwd;
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
            return r2hcf_rfft10c_fp32_bwd;
        }
        else if (precision == DT_DOUBLE)
        {
            return r2hcf_rfft10c_fp64_bwd;
        }
        else
        {
            return NULL;
        }
    }
}
