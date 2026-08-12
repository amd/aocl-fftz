// Copyright Advanced Micro Devices, Inc.
// SPDX-License-Identifier: BSD-3-Clause

/** @file rfft12c.c
 *
 *  @brief Radix-12 r2hc_fused Real-FFT kernel with scalar operations in C
 *
 *  This file contains the DIT radix-12 real-to-halfcomplex fused of two
 *  different implementations (Standard DFT and Shifted DFT that differs in DFT
 *  weight matrix) using scalar operations for single-precision
 *  and double-precision inputs.
 *
 *  @author Partiksha
 */

#include "core/kernels/kernel.h"

static const ops_cycles_t ops_cnt[NUM_PRECISIONS][NUM_FFT_DIRS] =
                                                    {{{0, 20, 80, 72, 0, 0},
                                                      {0, 30, 80, 72, 0, 0}},
                                                     {{0, 20, 80, 72, 0, 0},
                                                      {0, 30, 80, 72, 0, 0}}};

ops_cycles_t get_ops_cnt_r2hcf_rfft12c(FFTZ_UINT8 precision,
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

static FFTZ_VOID r2hcf_rfft12c_fp32_fwd(FFTZ_VOID *in_real, FFTZ_VOID *in_imag,
                                        FFTZ_VOID *out_real,
                                        FFTZ_VOID *out_imag, FFTZ_INTP n,
                                        aoclfftz_strides_t *strides,
                                        FFTZ_VOID *twd, FFTZ_UINT8 flag)
{
    AOCLFFTZ_LOG(DEBUG, global_logger_mode, "Enter");
    const FFTZ_FLOAT CRTM_12_1 =
        0.612372435695794524549321018676472847991486870f;
    const FFTZ_FLOAT CRTM_12_2 =
        0.353553390593273762200422181052424519642417969f;
    const FFTZ_FLOAT CRTM_12_3 =
        0.866025403784438646763723170752936183471402627f;
    const FFTZ_FLOAT CRTM_12_4 =
        0.500000000000000000000000000000000000000000000f;
    const FFTZ_FLOAT CRTM_12_5 =
        0.707106781186547524400844362104849039284835937f;

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
        FFTZ_FLOAT av0, av1, av2, av3, av4, av5, av6, av7, av8, av9, av10, av11;
        FFTZ_FLOAT at0, at1, at2, at3, at4, at5, at6, at7, at8, at9, at10, at11,
              at12, at13, at14, at15, at16, at17, at18, at19, at20, at21, at22,
              at23, at24, at25, at26, at27, at28, at29, at30, at31, at32, at33;

        av0 = *in;                    // Input point 1: x(0)
        av1 = in[in_strides[2]];      // Input point 3: x(2)
        av2 = in[in_strides[4]];      // Input point 5: x(4)
        av3 = in[in_strides[6]];      // Input point 7: x(6)
        av4 = in[in_strides[8]];      // Input point 9: x(8)
        av5 = in[in_strides[10]];     // Input point 11: x(10)
        av6 = in[in_strides[12]];     // Input point 13: x(12)
        av7 = in[in_strides[14]];     // Input point 15: x(14)
        av8 = in[in_strides[16]];     // Input point 17: x(16)
        av9 = in[in_strides[18]];     // Input point 19: x(18)
        av10 = in[in_strides[20]];    // Input point 21: x(20)
        av11 = in[in_strides[22]];    // Input point 23: x(22)

        at0 = av11 + av1;
        at1 = av11 - av1;
        at2 = av5 + av7;
        at3 = av5 - av7;
        at4 = av0 + av6;
        at5 = av0 - av6;
        at6 = av10 + av2;
        at7 = av10 - av2;
        at8 = av4 + av8;
        at9 = av4 - av8;
        at10 = av9 + av3;
        at11 = av9 - av3;

        at12 = at0 + at2;
        at13 = at0 - at2;
        at14 = at6 + at8;
        at15 = at6 - at8;
        at16 = at1 + at3;
        at17 = at1 - at3;
        at18 = at4 + at10;
        at19 = at4 - at10;
        at20 = at12 + at14;
        at21 = at12 - at14;

        at22 = CRTM_12_4 * at15;
        at23 = CRTM_12_4 * at17;
        at24 = at5 + at22;
        at25 = at23 + at11;
        at26 = at7 + at9;
        at27 = at7 - at9;
        at28 = at16 + at26;
        at29 = at16 - at26;

        at30 = CRTM_12_3 * at13;
        at31 = CRTM_12_4 * at20;
        at32 = CRTM_12_4 * at21;
        at33 = CRTM_12_3 * at27;

        *out = at20 + at18;                           // Output pt 1: X(0)
        out[out_strides[3]]  = at24 + at30;           // Output pt 4: X(3)
        out[out_strides[4]]  = at25 + at33;           // Output pt 5: X(4)
        out[out_strides[7]]  = at32 + at19;           // Output pt 8: X(7)
        out[out_strides[8]]  = CRTM_12_3 * at28;      // Output pt 9: X(8)
        out[out_strides[11]] = at5 - at15;            // Output pt 12: X(11)
        out[out_strides[12]] = at17 - at11;           // Output pt 13: X(12)
        out[out_strides[15]] = at18 - at31;           // Output pt 16: X(15)
        out[out_strides[16]] = CRTM_12_3 * at29;      // Output pt 17: X(16)
        out[out_strides[19]] = at24 - at30;           // Output pt 20: X(19)
        out[out_strides[20]] = at25 - at33;           // Output pt 21: X(20)
        out[out_strides[23]] = at19 - at21;           // Output pt 24: X(23)

        /* Shifted DFT */
        FFTZ_FLOAT bv0, bv1, bv2, bv3, bv4, bv5, bv6, bv7, bv8, bv9, bv10, bv11;
        FFTZ_FLOAT bt0, bt1, bt2, bt3, bt4, bt5, bt6, bt7, bt8, bt9, bt10, bt11,
              bt12, bt13, bt14, bt15, bt16, bt17, bt18, bt19, bt20, bt21,
              bt22, bt23, bt24, bt25, bt26, bt27, bt28, bt29, bt30, bt31,
              bt32, bt33, bt34, bt35, bt36, bt37, bt38, bt39, bt40, bt41;

        bv0  = in[in_strides[1]];    // Input point 2: x(1)
        bv1  = in[in_strides[3]];    // Input point 4: x(3)
        bv2  = in[in_strides[5]];    // Input point 6: x(5)
        bv3  = in[in_strides[7]];    // Input point 8: x(7)
        bv4  = in[in_strides[9]];    // Input point 10: x(9)
        bv5  = in[in_strides[11]];   // Input point 12: x(11)
        bv6  = in[in_strides[13]];   // Input point 14: x(13)
        bv7  = in[in_strides[15]];   // Input point 16: x(15)
        bv8  = in[in_strides[17]];   // Input point 18: x(17)
        bv9  = in[in_strides[19]];   // Input point 20: x(19)
        bv10 = in[in_strides[21]];   // Input point 22: x(21)
        bv11 = in[in_strides[23]];   // Input point 24: x(23)

        bt0 = bv1 + bv11;
        bt1 = bv1 - bv11;
        bt2 = bv5 + bv7;
        bt3 = bv5 - bv7;
        bt4 = bv2 + bv10;
        bt5 = bv2 - bv10;
        bt6 = bv4 + bv8;
        bt7 = bv4 - bv8;
        bt8 = bv3 + bv9;
        bt9 = bv3 - bv9;

        bt10 = bv0 - bt7;
        bt11 = bv6 - bt4;

        bt12 = bt2 + bt0;
        bt13 = bt2 - bt0;
        bt14 = bt1 + bt3;
        bt15 = bt1 - bt3;
        bt16 = bt13 - bt8;
        bt17 = bt15 - bt9;
        bt18 = CRTM_12_5 * bt16;
        bt19 = CRTM_12_5 * bt17;

        bt20 = CRTM_12_1 * bt12;
        bt21 = CRTM_12_2 * bt13;
        bt22 = CRTM_12_1 * bt14;
        bt23 = CRTM_12_2 * bt15;

        bt24 = CRTM_12_4 * bt4;
        bt25 = CRTM_12_3 * bt5;
        bt26 = CRTM_12_3 * bt6;
        bt27 = CRTM_12_4 * bt7;
        bt28 = CRTM_12_5 * bt8;
        bt29 = CRTM_12_5 * bt9;

        bt30 = bt27 + bv0;
        bt31 = bt23 + bt29;
        bt32 = bt21 + bt28;
        bt33 = bt24 + bv6;

        bt34 = bt25 + bt22;
        bt35 = bt25 - bt22;
        bt36 = bt26 + bt20;
        bt37 = bt26 - bt20;

        bt38 = bt30 + bt31;
        bt39 = bt30 - bt31;
        bt40 = bt32 + bt33;
        bt41 = bt32 - bt33;

        out[out_strides[1]]  = bt38 + bt34;      // Output pt 2: X(1)
        out[out_strides[2]]  = -bt40 - bt36;     // Output pt 3: X(2)
        out[out_strides[5]]  = bt10 + bt19;      // Output pt 6: X(5)
        out[out_strides[6]]  = bt18 + bt11;      // Output pt 7: X(6)
        out[out_strides[9]]  = bt39 - bt35;      // Output pt 10: X(9)
        out[out_strides[10]] = bt37 + bt41;      // Output pt 11: X(10)
        out[out_strides[13]] = bt38 - bt34;      // Output pt 14: X(13)
        out[out_strides[14]] = bt40 - bt36;      // Output pt 15: X(14)
        out[out_strides[17]] = bt10 - bt19;      // Output pt 18: X(17)
        out[out_strides[18]] = bt18 - bt11;      // Output pt 19: X(18)
        out[out_strides[21]] = bt39 + bt35;      // Output pt 22: X(21)
        out[out_strides[22]] = bt37 - bt41;      // Output pt 23: X(22)

        in = in + v_in_stride;
        out = out + v_out_stride;
    }
    AOCLFFTZ_LOG(DEBUG, global_logger_mode, "Exit");
}

static FFTZ_VOID r2hcf_rfft12c_fp32_bwd(FFTZ_VOID *in_real, FFTZ_VOID *in_imag,
                                        FFTZ_VOID *out_real,
                                        FFTZ_VOID *out_imag, FFTZ_INTP n,
                                        aoclfftz_strides_t *strides,
                                        FFTZ_VOID *twd, FFTZ_UINT8 flag)
{
    AOCLFFTZ_LOG(DEBUG, global_logger_mode, "Enter");
    const FFTZ_FLOAT CRTM_12_1 =
        1.931851652578136573499486399457794735267809678f;
    const FFTZ_FLOAT CRTM_12_2 =
        0.517638090205041524697797675248096656698137802f;
    const FFTZ_FLOAT CRTM_12_3 =
        1.732050807568877293527446341505872366942805254f;
    const FFTZ_FLOAT CRTM_12_4 =
        1.414213562373095048801688724209698078569671875f;
    const FFTZ_FLOAT CRTM_12_5 =
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

    for (cnt = 0; cnt < n; cnt++)
    {
        /* Standard DFT */
        FFTZ_FLOAT av0, av1, av2, av3, av4, av5, av6, av7, av8, av9, av10, av11;
        FFTZ_FLOAT at0, at1, at2, at3, at4, at5, at6, at7, at8, at9, at10, at11,
              at12, at13, at14, at15, at16, at17, at18, at19, at20, at21,
              at22, at23, at24, at25, at26, at27, at28, at29, at30, at31,
              at32, at33, at34, at35;

        av0 = *in;                  // Input point 1: X(0)
        av1 = in[in_strides[3]];    // Input point 4: X(3)
        av2 = in[in_strides[4]];    // Input point 5: X(4)
        av3 = in[in_strides[7]];    // Input point 8: X(7)
        av4 = in[in_strides[8]];    // Input point 9: X(8)
        av5 = in[in_strides[11]];   // Input point 12: X(11)
        av6 = in[in_strides[12]];   // Input point 13: X(12)
        av7 = in[in_strides[15]];   // Input point 16: X(15)
        av8 = in[in_strides[16]];   // Input point 18: X(16)
        av9 = in[in_strides[19]];   // Input point 20: X(19)
        av10 = in[in_strides[20]];  // Input point 21: X(20)
        av11 = in[in_strides[23]];  // Input point 24: X(23)

        at0 = av0 + av11;
        at1 = av0 - av11;
        at2 = av1 + av9;
        at3 = av1 - av9;
        at4 = av3 + av7;
        at5 = av3 - av7;
        at6 = av4 + av8;
        at7 = av4 - av8;
        at8 = av10 + av2;
        at9 = av10 - av2;

        at10 = CRTM_12_5 * av5;
        at11 = CRTM_12_5 * av6;

        at12 = at2 + at4;
        at13 = at0 + at10;
        at14 = at0 - at10;
        at15 = CRTM_12_5 * at12;

        at16 = at3 + at6;
        at17 = at3 - at6;
        at18 = CRTM_12_3 * at16;
        at19 = CRTM_12_3 * at17;

        at20 = at9 + at7;
        at21 = at9 - at7;
        at22 = CRTM_12_3 * at20;
        at23 = CRTM_12_3 * at21;

        at24 = at5 + at8;
        at25 = at5 - at8;
        at26 = at2 - at4;
        at27 = CRTM_12_5 * at24;
        at28 = CRTM_12_5 * at25;
        at29 = CRTM_12_5 * at26;

        at30 = at1 - at11;
        at31 = at1 + at11;
        at32 = at25 + at30;
        at33 = at31 + at24;
        at34 = at13 - at12;
        at35 = at14 + at26;

        *out = at13 + at15;                    // Output pt 1: x(0)
        out[out_strides[2]]  = at19 + at32;    // Output pt 2: x(2)
        out[out_strides[4]]  = at23 + at35;    // Output pt 3: x(4)
        out[out_strides[6]]  = at31 - at27;    // Output pt 7: x(6)
        out[out_strides[8]]  = at34 + at22;    // Output pt 9: x(8)
        out[out_strides[10]] = at32 - at19;    // Output pt 11: x(10)
        out[out_strides[12]] = at14 - at29;    // Output pt 13: x(12)
        out[out_strides[14]] = at33 - at18;    // Output pt 15: x(14)
        out[out_strides[16]] = at34 - at22;    // Output pt 17: x(16)
        out[out_strides[18]] = at30 - at28;    // Output pt 19: x(18)
        out[out_strides[20]] = at35 - at23;    // Output pt 21: x(20)
        out[out_strides[22]] = at33 + at18;    // Output pt 23: x(22)

        /* Shifted DFT */
        FFTZ_FLOAT bv0, bv1, bv2, bv3, bv4, bv5, bv6, bv7, bv8, bv9, bv10, bv11;
        FFTZ_FLOAT bt0, bt1, bt2, bt3, bt4, bt5, bt6, bt7, bt8, bt9, bt10, bt11,
              bt12, bt13, bt14, bt15, bt16, bt17, bt18, bt19, bt20, bt21, bt22,
              bt23, bt24, bt25, bt26, bt27, bt28, bt29, bt30, bt31, bt32, bt33,
              bt34, bt35, bt36, bt37, bt38, bt39, bt40, bt41, bt42, bt43, bt44,
              bt45, bt46, bt47, bt48, bt49;

        bv0  = in[in_strides[1]];    // Input point 2: X(1)
        bv1  = in[in_strides[2]];    // Input point 3: X(2)
        bv2  = in[in_strides[5]];    // Input point 6: X(5)
        bv3  = in[in_strides[6]];    // Input point 7: X(6)
        bv4  = in[in_strides[9]];    // Input point 10: X(9)
        bv5  = in[in_strides[10]];   // Input point 11: X(10)
        bv6  = in[in_strides[13]];   // Input point 14: X(13)
        bv7  = in[in_strides[14]];   // Input point 15: X(14)
        bv8  = in[in_strides[17]];   // Input point 18: X(17)
        bv9  = in[in_strides[18]];   // Input point 19: X(18)
        bv10 = in[in_strides[21]];   // Input point 22: X(21)
        bv11 = in[in_strides[22]];   // Input point 23: X(22)

        bt0 = bv0 + bv10;
        bt1 = bv0 - bv10;
        bt2 = bv2 + bv8;
        bt3 = bv2 - bv8;
        bt4 = bv4 + bv6;
        bt5 = bv4 - bv6;
        bt6 = bv7 + bv5;
        bt7 = bv7 - bv5;
        bt8 = bv11 + bv1;
        bt9 = bv11 - bv1;
        bt10 = bv9 + bv3;
        bt11 = bv9 - bv3;
        bt12 = bt0 + bt4;
        bt13 = bt0 - bt4;

        bt14 = bt12 + bt2;
        bt15 = bt9 + bt7;
        bt16 = bt9 - bt7;
        bt17 = CRTM_12_5 * bt2;
        bt18 = CRTM_12_5 * bt11;
        bt19 = CRTM_12_3 * bt13;
        bt20 = CRTM_12_3 * bt16;

        bt21 = bt15 + bt18;
        bt22 = bt12 - bt17;
        bt23 = bt15 - bt11;

        bt24 = bt1 + bt6;
        bt25 = bt1 - bt6;
        bt26 = bt5 + bt8;
        bt27 = bt5 - bt8;
        bt28 = bt10 + bt3;
        bt29 = bt10 - bt3;

        bt30 = CRTM_12_1 * bt24;
        bt31 = CRTM_12_2 * bt24;
        bt32 = CRTM_12_4 * bt24;

        bt33 = CRTM_12_1 * bt25;
        bt34 = CRTM_12_2 * bt25;
        bt35 = CRTM_12_4 * bt25;

        bt42 = CRTM_12_1 * bt26;
        bt41 = CRTM_12_2 * bt26;
        bt43 = CRTM_12_4 * bt26;

        bt36 = CRTM_12_1 * bt27;
        bt37 = CRTM_12_2 * bt27;
        bt38 = CRTM_12_4 * bt27;

        bt39 = CRTM_12_4 * bt28;
        bt40 = CRTM_12_4 * bt29;

        bt44 = bt33 + bt37;
        bt45 = bt32 - bt43;
        bt46 = bt36 + bt34;
        bt47 = bt42 + bt31;
        bt48 = bt38 - bt35;
        bt49 = -bt30 - bt41;

        out[out_strides[1]]  = CRTM_12_5 * bt14;     // Output pt 2: x(1)
        out[out_strides[3]]  = bt44 - bt40;          // Output pt 4: x(3)
        out[out_strides[5]]  = bt19 + bt21;          // Output pt 6: x(5)
        out[out_strides[7]]  = bt45 - bt39;          // Output pt 8: x(7)
        out[out_strides[9]]  = bt20 + bt22;          // Output pt 10: x(9)
        out[out_strides[11]] = bt46 + bt40;          // Output pt 12: x(11)
        out[out_strides[13]] = CRTM_12_5 * bt23;     // Output pt 14: x(13)
        out[out_strides[15]] = bt39 - bt47;          // Output pt 16: x(15)
        out[out_strides[17]] = bt20 - bt22;          // Output pt 18: x(17)
        out[out_strides[19]] = bt48 - bt40;          // Output pt 20: x(19)
        out[out_strides[21]] = bt21 - bt19;          // Output pt 22: x(21)
        out[out_strides[23]] = bt49 - bt39;          // Output pt 24: x(23)

        in = in + v_in_stride;
        out = out + v_out_stride;
    }
    AOCLFFTZ_LOG(DEBUG, global_logger_mode, "Exit");
}

static FFTZ_VOID r2hcf_rfft12c_fp64_fwd(FFTZ_VOID *in_real, FFTZ_VOID *in_imag,
                                        FFTZ_VOID *out_real,
                                        FFTZ_VOID *out_imag, FFTZ_INTP n,
                                        aoclfftz_strides_t *strides,
                                        FFTZ_VOID *twd, FFTZ_UINT8 flag)
 {
    AOCLFFTZ_LOG(DEBUG, global_logger_mode, "Enter");
    const FFTZ_DOUBLE CRTM_12_1 =
        0.612372435695794524549321018676472847991486870;
    const FFTZ_DOUBLE CRTM_12_2 =
        0.353553390593273762200422181052424519642417969;
    const FFTZ_DOUBLE CRTM_12_3 =
        0.866025403784438646763723170752936183471402627;
    const FFTZ_DOUBLE CRTM_12_4 =
        0.500000000000000000000000000000000000000000000;
    const FFTZ_DOUBLE CRTM_12_5 =
        0.707106781186547524400844362104849039284835937;

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
        FFTZ_DOUBLE av0, av1, av2, av3, av4, av5, av6, av7, av8, av9, av10,
            av11;
        FFTZ_DOUBLE at0, at1, at2, at3, at4, at5, at6, at7, at8, at9, at10,
            at11, at12, at13, at14, at15, at16, at17, at18, at19, at20, at21,
            at22, at23, at24, at25, at26, at27, at28, at29, at30, at31, at32,
            at33;

        av0 = *in;                    // Input point 1: x(0)
        av1 = in[in_strides[2]];      // Input point 3: x(2)
        av2 = in[in_strides[4]];      // Input point 5: x(4)
        av3 = in[in_strides[6]];      // Input point 7: x(6)
        av4 = in[in_strides[8]];      // Input point 9: x(8)
        av5 = in[in_strides[10]];     // Input point 11: x(10)
        av6 = in[in_strides[12]];     // Input point 13: x(12)
        av7 = in[in_strides[14]];     // Input point 15: x(14)
        av8 = in[in_strides[16]];     // Input point 17: x(16)
        av9 = in[in_strides[18]];     // Input point 19: x(18)
        av10 = in[in_strides[20]];    // Input point 21: x(20)
        av11 = in[in_strides[22]];    // Input point 23: x(22)

        at0 = av11 + av1;
        at1 = av11 - av1;
        at2 = av5 + av7;
        at3 = av5 - av7;
        at4 = av0 + av6;
        at5 = av0 - av6;
        at6 = av10 + av2;
        at7 = av10 - av2;
        at8 = av4 + av8;
        at9 = av4 - av8;
        at10 = av9 + av3;
        at11 = av9 - av3;

        at12 = at0 + at2;
        at13 = at0 - at2;
        at14 = at6 + at8;
        at15 = at6 - at8;
        at16 = at1 + at3;
        at17 = at1 - at3;
        at18 = at4 + at10;
        at19 = at4 - at10;
        at20 = at12 + at14;
        at21 = at12 - at14;

        at22 = CRTM_12_4 * at15;
        at23 = CRTM_12_4 * at17;
        at24 = at5 + at22;
        at25 = at23 + at11;
        at26 = at7 + at9;
        at27 = at7 - at9;
        at28 = at16 + at26;
        at29 = at16 - at26;

        at30 = CRTM_12_3 * at13;
        at31 = CRTM_12_4 * at20;
        at32 = CRTM_12_4 * at21;
        at33 = CRTM_12_3 * at27;

        *out = at20 + at18;                           // Output pt 1: X(0)
        out[out_strides[3]]  = at24 + at30;           // Output pt 4: X(3)
        out[out_strides[4]]  = at25 + at33;           // Output pt 5: X(4)
        out[out_strides[7]]  = at32 + at19;           // Output pt 8: X(7)
        out[out_strides[8]]  = CRTM_12_3 * at28;      // Output pt 9: X(8)
        out[out_strides[11]] = at5 - at15;            // Output pt 12: X(11)
        out[out_strides[12]] = at17 - at11;           // Output pt 13: X(12)
        out[out_strides[15]] = at18 - at31;           // Output pt 16: X(15)
        out[out_strides[16]] = CRTM_12_3 * at29;      // Output pt 17: X(16)
        out[out_strides[19]] = at24 - at30;           // Output pt 20: X(19)
        out[out_strides[20]] = at25 - at33;           // Output pt 21: X(20)
        out[out_strides[23]] = at19 - at21;           // Output pt 24: X(23)

        /* Shifted DFT */
        FFTZ_DOUBLE bv0, bv1, bv2, bv3, bv4, bv5, bv6, bv7, bv8, bv9, bv10,
            bv11;
        FFTZ_DOUBLE bt0, bt1, bt2, bt3, bt4, bt5, bt6, bt7, bt8, bt9, bt10,
            bt11, bt12, bt13, bt14, bt15, bt16, bt17, bt18, bt19, bt20, bt21,
            bt22, bt23, bt24, bt25, bt26, bt27, bt28, bt29, bt30, bt31, bt32,
            bt33, bt34, bt35, bt36, bt37, bt38, bt39, bt40, bt41;

        bv0  = in[in_strides[1]];    // Input point 2: x(1)
        bv1  = in[in_strides[3]];    // Input point 4: x(3)
        bv2  = in[in_strides[5]];    // Input point 6: x(5)
        bv3  = in[in_strides[7]];    // Input point 8: x(7)
        bv4  = in[in_strides[9]];    // Input point 10: x(9)
        bv5  = in[in_strides[11]];   // Input point 12: x(11)
        bv6  = in[in_strides[13]];   // Input point 14: x(13)
        bv7  = in[in_strides[15]];   // Input point 16: x(15)
        bv8  = in[in_strides[17]];   // Input point 18: x(17)
        bv9  = in[in_strides[19]];   // Input point 20: x(19)
        bv10 = in[in_strides[21]];   // Input point 22: x(21)
        bv11 = in[in_strides[23]];   // Input point 24: x(23)

        bt0 = bv1 + bv11;
        bt1 = bv1 - bv11;
        bt2 = bv5 + bv7;
        bt3 = bv5 - bv7;
        bt4 = bv2 + bv10;
        bt5 = bv2 - bv10;
        bt6 = bv4 + bv8;
        bt7 = bv4 - bv8;
        bt8 = bv3 + bv9;
        bt9 = bv3 - bv9;

        bt10 = bv0 - bt7;
        bt11 = bv6 - bt4;

        bt12 = bt2 + bt0;
        bt13 = bt2 - bt0;
        bt14 = bt1 + bt3;
        bt15 = bt1 - bt3;
        bt16 = bt13 - bt8;
        bt17 = bt15 - bt9;
        bt18 = CRTM_12_5 * bt16;
        bt19 = CRTM_12_5 * bt17;

        bt20 = CRTM_12_1 * bt12;
        bt21 = CRTM_12_2 * bt13;
        bt22 = CRTM_12_1 * bt14;
        bt23 = CRTM_12_2 * bt15;

        bt24 = CRTM_12_4 * bt4;
        bt25 = CRTM_12_3 * bt5;
        bt26 = CRTM_12_3 * bt6;
        bt27 = CRTM_12_4 * bt7;
        bt28 = CRTM_12_5 * bt8;
        bt29 = CRTM_12_5 * bt9;

        bt30 = bt27 + bv0;
        bt31 = bt23 + bt29;
        bt32 = bt21 + bt28;
        bt33 = bt24 + bv6;

        bt34 = bt25 + bt22;
        bt35 = bt25 - bt22;
        bt36 = bt26 + bt20;
        bt37 = bt26 - bt20;

        bt38 = bt30 + bt31;
        bt39 = bt30 - bt31;
        bt40 = bt32 + bt33;
        bt41 = bt32 - bt33;

        out[out_strides[1]]  = bt38 + bt34;      // Output pt 2: X(1)
        out[out_strides[2]]  = -bt40 - bt36;     // Output pt 3: X(2)
        out[out_strides[5]]  = bt10 + bt19;      // Output pt 6: X(5)
        out[out_strides[6]]  = bt18 + bt11;      // Output pt 7: X(6)
        out[out_strides[9]]  = bt39 - bt35;      // Output pt 10: X(9)
        out[out_strides[10]] = bt37 + bt41;      // Output pt 11: X(10)
        out[out_strides[13]] = bt38 - bt34;      // Output pt 14: X(13)
        out[out_strides[14]] = bt40 - bt36;      // Output pt 15: X(14)
        out[out_strides[17]] = bt10 - bt19;      // Output pt 18: X(17)
        out[out_strides[18]] = bt18 - bt11;      // Output pt 19: X(18)
        out[out_strides[21]] = bt39 + bt35;      // Output pt 22: X(21)
        out[out_strides[22]] = bt37 - bt41;      // Output pt 23: X(22)

        in = in + v_in_stride;
        out = out + v_out_stride;

    }
    AOCLFFTZ_LOG(DEBUG, global_logger_mode, "Exit");
 }

static FFTZ_VOID r2hcf_rfft12c_fp64_bwd(FFTZ_VOID *in_real, FFTZ_VOID *in_imag,
                                        FFTZ_VOID *out_real,
                                        FFTZ_VOID *out_imag, FFTZ_INTP n,
                                        aoclfftz_strides_t *strides,
                                        FFTZ_VOID *twd, FFTZ_UINT8 flag)
 {
    AOCLFFTZ_LOG(DEBUG, global_logger_mode, "Enter");
    const FFTZ_DOUBLE CRTM_12_1 =
        1.931851652578136573499486399457794735267809678;
    const FFTZ_DOUBLE CRTM_12_2 =
        0.5176380902050415246977976752480966566981378026;
    const FFTZ_DOUBLE CRTM_12_3 =
        1.732050807568877293527446341505872366942805254;
    const FFTZ_DOUBLE CRTM_12_4 =
        1.414213562373095048801688724209698078569671875;
    const FFTZ_DOUBLE CRTM_12_5 =
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

    for (cnt = 0; cnt < n; cnt++)
    {
        /* Standard DFT */
        FFTZ_DOUBLE av0, av1, av2, av3, av4, av5, av6, av7, av8, av9, av10,
            av11;
        FFTZ_DOUBLE at0, at1, at2, at3, at4, at5, at6, at7, at8, at9, at10,
            at11, at12, at13, at14, at15, at16, at17, at18, at19, at20, at21,
            at22, at23, at24, at25, at26, at27, at28, at29, at30, at31, at32,
            at33, at34, at35;

        av0 = *in;                  // Input point 1: X(0)
        av1 = in[in_strides[3]];    // Input point 4: X(3)
        av2 = in[in_strides[4]];    // Input point 5: X(4)
        av3 = in[in_strides[7]];    // Input point 8: X(7)
        av4 = in[in_strides[8]];    // Input point 9: X(8)
        av5 = in[in_strides[11]];   // Input point 12: X(11)
        av6 = in[in_strides[12]];   // Input point 13: X(12)
        av7 = in[in_strides[15]];   // Input point 16: X(15)
        av8 = in[in_strides[16]];   // Input point 17: X(16)
        av9 = in[in_strides[19]];   // Input point 20: X(19)
        av10 = in[in_strides[20]];  // Input point 21: X(20)
        av11 = in[in_strides[23]];  // Input point 24: X(23)

        at0 = av0 + av11;
        at1 = av0 - av11;
        at2 = av1 + av9;
        at3 = av1 - av9;
        at4 = av3 + av7;
        at5 = av3 - av7;
        at6 = av4 + av8;
        at7 = av4 - av8;
        at8 = av10 + av2;
        at9 = av10 - av2;

        at10 = CRTM_12_5 * av5;
        at11 = CRTM_12_5 * av6;

        at12 = at2 + at4;
        at13 = at0 + at10;
        at14 = at0 - at10;
        at15 = CRTM_12_5 * at12;

        at16 = at3 + at6;
        at17 = at3 - at6;
        at18 = CRTM_12_3 * at16;
        at19 = CRTM_12_3 * at17;

        at20 = at9 + at7;
        at21 = at9 - at7;
        at22 = CRTM_12_3 * at20;
        at23 = CRTM_12_3 * at21;

        at24 = at5 + at8;
        at25 = at5 - at8;
        at26 = at2 - at4;
        at27 = CRTM_12_5 * at24;
        at28 = CRTM_12_5 * at25;
        at29 = CRTM_12_5 * at26;

        at30 = at1 - at11;
        at31 = at1 + at11;
        at32 = at25 + at30;
        at33 = at31 + at24;
        at34 = at13 - at12;
        at35 = at14 + at26;

        *out = at13 + at15;                    // Output pt 1: x(0)
        out[out_strides[2]]  = at19 + at32;    // Output pt 3: x(2)
        out[out_strides[4]]  = at23 + at35;    // Output pt 5: x(4)
        out[out_strides[6]]  = at31 - at27;    // Output pt 7: x(6)
        out[out_strides[8]]  = at34 + at22;    // Output pt 9: x(8)
        out[out_strides[10]] = at32 - at19;    // Output pt 11: x(10)
        out[out_strides[12]] = at14 - at29;    // Output pt 13: x(12)
        out[out_strides[14]] = at33 - at18;    // Output pt 15: x(14)
        out[out_strides[16]] = at34 - at22;    // Output pt 17: x(16)
        out[out_strides[18]] = at30 - at28;    // Output pt 19: x(18)
        out[out_strides[20]] = at35 - at23;    // Output pt 21: x(20)
        out[out_strides[22]] = at33 + at18;    // Output pt 23: x(22)

        /* Shifted DFT */
        FFTZ_DOUBLE bv0, bv1, bv2, bv3, bv4, bv5, bv6, bv7, bv8, bv9, bv10,
            bv11;
        FFTZ_DOUBLE bt0, bt1, bt2, bt3, bt4, bt5, bt6, bt7, bt8, bt9, bt10,
            bt11, bt12, bt13, bt14, bt15, bt16, bt17, bt18, bt19, bt20, bt21,
            bt22, bt23, bt24, bt25, bt26, bt27, bt28, bt29, bt30, bt31, bt32,
            bt33, bt34, bt35, bt36, bt37, bt38, bt39, bt40, bt41, bt42, bt43,
            bt44, bt45, bt46, bt47, bt48, bt49;

        bv0  = in[in_strides[1]];    // Input point 2: X(1)
        bv1  = in[in_strides[2]];    // Input point 3: X(2)
        bv2  = in[in_strides[5]];    // Input point 6: X(5)
        bv3  = in[in_strides[6]];    // Input point 7: X(6)
        bv4  = in[in_strides[9]];    // Input point 10: X(9)
        bv5  = in[in_strides[10]];   // Input point 11: X(10)
        bv6  = in[in_strides[13]];   // Input point 14: X(13)
        bv7  = in[in_strides[14]];   // Input point 15: X(14)
        bv8  = in[in_strides[17]];   // Input point 18: X(17)
        bv9  = in[in_strides[18]];   // Input point 19: X(18)
        bv10 = in[in_strides[21]];   // Input point 22: X(21)
        bv11 = in[in_strides[22]];   // Input point 23: X(22)

        bt0 = bv0 + bv10;
        bt1 = bv0 - bv10;
        bt2 = bv2 + bv8;
        bt3 = bv2 - bv8;
        bt4 = bv4 + bv6;
        bt5 = bv4 - bv6;
        bt6 = bv7 + bv5;
        bt7 = bv7 - bv5;
        bt8 = bv11 + bv1;
        bt9 = bv11 - bv1;
        bt10 = bv9 + bv3;
        bt11 = bv9 - bv3;
        bt12 = bt0 + bt4;
        bt13 = bt0 - bt4;

        bt14 = bt12 + bt2;
        bt15 = bt9 + bt7;
        bt16 = bt9 - bt7;
        bt17 = CRTM_12_5 * bt2;
        bt18 = CRTM_12_5 * bt11;
        bt19 = CRTM_12_3 * bt13;
        bt20 = CRTM_12_3 * bt16;

        bt21 = bt15 + bt18;
        bt22 = bt12 - bt17;
        bt23 = bt15 - bt11;

        bt24 = bt1 + bt6;
        bt25 = bt1 - bt6;
        bt26 = bt5 + bt8;
        bt27 = bt5 - bt8;
        bt28 = bt10 + bt3;
        bt29 = bt10 - bt3;

        bt30 = CRTM_12_1 * bt24;
        bt31 = CRTM_12_2 * bt24;
        bt32 = CRTM_12_4 * bt24;

        bt33 = CRTM_12_1 * bt25;
        bt34 = CRTM_12_2 * bt25;
        bt35 = CRTM_12_4 * bt25;

        bt42 = CRTM_12_1 * bt26;
        bt41 = CRTM_12_2 * bt26;
        bt43 = CRTM_12_4 * bt26;

        bt36 = CRTM_12_1 * bt27;
        bt37 = CRTM_12_2 * bt27;
        bt38 = CRTM_12_4 * bt27;

        bt39 = CRTM_12_4 * bt28;
        bt40 = CRTM_12_4 * bt29;

        bt44 = bt33 + bt37;
        bt45 = bt32 - bt43;
        bt46 = bt36 + bt34;
        bt47 = bt42 + bt31;
        bt48 = bt38 - bt35;
        bt49 = -bt30 - bt41;

        out[out_strides[1]]  = CRTM_12_5 * bt14;     // Output pt 2: x(1)
        out[out_strides[3]]  = bt44 - bt40;          // Output pt 4: x(3)
        out[out_strides[5]]  = bt19 + bt21;          // Output pt 6: x(5)
        out[out_strides[7]]  = bt45 - bt39;          // Output pt 8: x(7)
        out[out_strides[9]]  = bt20 + bt22;          // Output pt 10: x(9)
        out[out_strides[11]] = bt46 + bt40;          // Output pt 12: x(11)
        out[out_strides[13]] = CRTM_12_5 * bt23;     // Output pt 14: x(13)
        out[out_strides[15]] = bt39 - bt47;          // Output pt 16: x(15)
        out[out_strides[17]] = bt20 - bt22;          // Output pt 18: x(17)
        out[out_strides[19]] = bt48 - bt40;          // Output pt 20: x(19)
        out[out_strides[21]] = bt21 - bt19;          // Output pt 22: x(21)
        out[out_strides[23]] = bt49 - bt39;          // Output pt 24: x(23)

        in = in + v_in_stride;
        out = out + v_out_stride;
    }
    AOCLFFTZ_LOG(DEBUG, global_logger_mode, "Exit");
}

kfft_ register_kernel_r2hcf_rfft12c(FFTZ_UINT8 precision, FFTZ_UINT8 direction)
{
    if (direction == FORWARD_FFT_DIR)
    {
        if (precision == DT_FLOAT)
        {
            return r2hcf_rfft12c_fp32_fwd;
        }
        else if (precision == DT_DOUBLE)
        {
            return r2hcf_rfft12c_fp64_fwd;
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
            return r2hcf_rfft12c_fp32_bwd;
        }
        else if (precision == DT_DOUBLE)
        {
            return r2hcf_rfft12c_fp64_bwd;
        }
        else
        {
            return NULL;
        }
    }
}
