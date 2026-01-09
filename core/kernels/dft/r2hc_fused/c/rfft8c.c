// Copyright Advanced Micro Devices, Inc.
// SPDX-License-Identifier: BSD-3-Clause

/** @file rfft8c.c
 *
 *  @brief Radix-8 r2hc_fused Real-FFT kernel with scalar operations in C
 *
 *  This file contains the DIT radix-8 real-to-halfcomplex fused of two
 *  different implementations (Standard DFT and Shifted DFT that differs in DFT
 *  weight matrix) using scalar operations for single-precision
 *  and double-precision inputs.
 *
 *  @author Dr. Pritam Giri
 */

#include "core/kernels/kernel.h"

static const ops_cycles_t ops_cnt[NUM_PRECISIONS][NUM_FFT_DIRS] =
                                                    {{{0, 12, 42, 32, 0, 0},
                                                      {0, 16, 44, 32, 0, 0}},
                                                     {{0, 12, 42, 32, 0, 0},
                                                      {0, 16, 44, 32, 0, 0}}};

ops_cycles_t get_ops_cnt_r2hcf_rfft8c(UINT8 precision, UINT8 direction)
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

static VOID r2hcf_rfft8c_fp32_fwd(VOID *in_real, VOID *in_imag, VOID *out_real,
                              VOID *out_imag, INTP n,
                              aoclfftz_strides_t *strides, VOID *twd, UINT8 flag)
{
    AOCLFFTZ_LOG(DEBUG, global_logger_mode, "Enter");
    const FLOAT CRTM_8_1 = 0.7071067811865475244008443621048490392848359377f;
    const FLOAT CRTM_8_2 = 0.9238795325112867561281831893967882868224166259f;
    const FLOAT CRTM_8_3 = 0.3826834323650897717284599840303988667613445625f;

    FLOAT *in = (FLOAT *)in_real;
    FLOAT *out = (FLOAT *)out_real;
#ifdef VOLATILE_STRIDE_ARRAY
    volatile INTP *in_strides = strides->in_strides;
    volatile INTP *out_strides = strides->out_strides;
#else
    INTP *in_strides = strides->in_strides;
    INTP *out_strides = strides->out_strides;
#endif
    INTP v_in_stride = (strides->v_in_stride);
    INTP v_out_stride = (strides->v_out_stride);
    INTP cnt;

    for (cnt = 0; cnt < n; cnt++)
    {
        /* Standard DFT */
        FLOAT av0, av1, av2, av3, av4, av5, av6, av7;
        FLOAT at0, at1, at2, at3, at4, at5, at6, at7, at8, at9,
              at10, at11, at12, at13;

        av0 = *in;                  // Input point 1: x(0)
        av1 = in[in_strides[2]];    // Input point 3: x(2)
        av2 = in[in_strides[4]];    // Input point 5: x(4)
        av3 = in[in_strides[6]];    // Input point 7: x(6)
        av4 = in[in_strides[8]];    // Input point 9: x(8)
        av5 = in[in_strides[10]];   // Input point 11: x(10)
        av6 = in[in_strides[12]];   // Input point 13: x(12)
        av7 = in[in_strides[14]];   // Input point 13: x(12)

        at0 = av7 + av5;
        at1 = av7 - av5;
        at2 = av6 + av2;
        at3 = av6 - av2;
        at4 = av0 + av4;
        at5 = av0 - av4;
        at6 = av3 + av1;
        at7 = av3 - av1;

        at8 = at7 - at1;
        at9 = at0 - at6;
        at10 = at0 + at6;
        at11 = at4 + at2;
        at12 = CRTM_8_1 * at8;
        at13 = CRTM_8_1 * at9;

        *out = at11 + at10;                   // Output pt 1: X(0)
        out[out_strides[3]]  = at5 - at12;    // Output pt 4: X(3)
        out[out_strides[4]]  = at3 + at13;    // Output pt 5: X(4)
        out[out_strides[7]]  = at4 - at2;     // Output pt 8: X(7)
        out[out_strides[8]]  = at7 + at1;     // Output pt 9: X(8)
        out[out_strides[11]] = at12 + at5;    // Output pt 12: X(11)
        out[out_strides[12]] = at13 - at3;    // Output pt 13: X(12)
        out[out_strides[15]] = at11 - at10;   // Output pt 16: X(15)

        /* Shifted DFT */
        FLOAT bv0, bv1, bv2, bv3, bv4, bv5, bv6, bv7;
        FLOAT bt0, bt1, bt2, bt3, bt4, bt5, bt6, bt7, bt8, bt9, bt10, bt11,
              bt12, bt13, bt14, bt15, bt16, bt17, bt18, bt19, bt20, bt21,
              bt22, bt23;

        bv0 = in[in_strides[1]];    // Input point 2: x(1)
        bv1 = in[in_strides[3]];    // Input point 4: x(3)
        bv2 = in[in_strides[5]];    // Input point 6: x(5)
        bv3 = in[in_strides[7]];    // Input point 8: x(7)
        bv4 = in[in_strides[9]];    // Input point 10: x(9)
        bv5 = in[in_strides[11]];   // Input point 12: x(11)
        bv6 = in[in_strides[13]];   // Input point 14: x(13)
        bv7 = in[in_strides[15]];   // Input point 14: x(13)

        bt0 = bv6 + bv2;
        bt1 = bv6 - bv2;
        bt2 = bv7 + bv1;
        bt3 = bv7 - bv1;
        bt4 = bv5 + bv3;
        bt5 = bv5 - bv3;

        bt6 = CRTM_8_1 * bt0;
        bt7 = CRTM_8_1 * bt1;
        bt8 = CRTM_8_2 * bt2;
        bt9 = CRTM_8_3 * bt2;
        bt10 = CRTM_8_2 * bt3;
        bt11 = CRTM_8_3 * bt3;
        bt12 = CRTM_8_2 * bt4;
        bt13 = CRTM_8_3 * bt4;
        bt14 = CRTM_8_2 * bt5;
        bt15 = CRTM_8_3 * bt5;

        bt16 = bv0 - bt7;
        bt17 = bv0 + bt7;
        bt18 = bv4 - bt6;
        bt19 = bt6 + bv4;
        bt20 = bt10 + bt15;
        bt21 = bt12 + bt9;
        bt22 = bt14 - bt11;
        bt23 = bt13 - bt8;

        out[out_strides[1]]  = bt16 - bt20;   // Output pt 2: X(1)
        out[out_strides[2]]  = -bt19 - bt21;  // Output pt 3: X(2)
        out[out_strides[5]]  = bt17 + bt22;   // Output pt 6: X(5)
        out[out_strides[6]]  = bt18 + bt23;   // Output pt 7: X(6)
        out[out_strides[9]]  = bt17 - bt22;   // Output pt 10: X(9)
        out[out_strides[10]] = bt23 - bt18;   // Output pt 11: X(10)
        out[out_strides[13]] = bt16 + bt20;   // Output pt 14: X(12)
        out[out_strides[14]] = bt19 - bt21;   // Output pt 15: X(14)

        in = in + v_in_stride;
        out = out + v_out_stride;
    }
    AOCLFFTZ_LOG(DEBUG, global_logger_mode, "Exit");
}

static VOID r2hcf_rfft8c_fp32_bwd(VOID *in_real, VOID *in_imag, VOID *out_real,
                              VOID *out_imag, INTP n,
                              aoclfftz_strides_t *strides, VOID *twd, UINT8 flag)
{
    AOCLFFTZ_LOG(DEBUG, global_logger_mode, "Enter");
    const FLOAT CRTM_8_1 = 1.414213562373095048801688724209698078569671875f;
    const FLOAT CRTM_8_2 = 1.847759065022573512256366378793576573644833252f;
    const FLOAT CRTM_8_3 = 0.765366864730179543456919968060797733522689125f;
    const FLOAT CRTM_8_4 = 2.000000000000000000000000000000000000000000000f;

    FLOAT *in = (FLOAT *)in_real;
    FLOAT *out = (FLOAT *)out_real;
#ifdef VOLATILE_STRIDE_ARRAY
    volatile INTP *in_strides = strides->in_strides;
    volatile INTP *out_strides = strides->out_strides;
#else
    INTP *in_strides = strides->in_strides;
    INTP *out_strides = strides->out_strides;
#endif
    INTP v_in_stride = (strides->v_in_stride);
    INTP v_out_stride = (strides->v_out_stride);
    INTP cnt;

    for (cnt = 0; cnt < n; cnt++)
    {
        /* Standard DFT */
        FLOAT av0, av1, av2, av3, av4, av5, av6, av7;
        FLOAT at0, at1, at2, at3, at4, at5, at6, at7, at8, at9,
              at10, at11, at12, at13, at14, at15, at16, at17;

        av0 = *in;                  // Input point 1: x(0)
        av1 = in[in_strides[3]];    // Input point 4: x(3)
        av2 = in[in_strides[4]];    // Input point 5: x(4)
        av3 = in[in_strides[7]];    // Input point 8: x(7)
        av4 = in[in_strides[8]];    // Input point 9: x(8)
        av5 = in[in_strides[11]];   // Input point 12: x(11)
        av6 = in[in_strides[12]];   // Input point 13: x(12)
        av7 = in[in_strides[15]];   // Input point 14: x(13)

        at0 = av7 + av0;
        at1 = av7 - av0;
        at2 = av6 + av2;
        at3 = av6 - av2;
        at4 = av5 + av1;
        at5 = av5 - av1;
        at6 = CRTM_8_4 * at3;
        at7 = CRTM_8_4 * at4;
        at8 = CRTM_8_4 * av3;
        at9 = CRTM_8_4 * av4;

        at10 = at5 + at2;
        at11 = at5 - at2;
        at12 = CRTM_8_1 * at10;
        at13 = CRTM_8_1 * at11;
        at14 = at0 + at8;
        at15 = at0 - at8;
        at16 = at9 + at1;
        at17 = at9 - at1;

        *out = at14 + at7;                   // Output pt 1: X(0)
        out[out_strides[2]]  = -at12 - at16; // Output pt 3: X(2)
        out[out_strides[4]]  = at15 + at6;   // Output pt 5: X(4)
        out[out_strides[6]]  = at17 + at13;  // Output pt 7: X(6)
        out[out_strides[8]]  = at14 - at7;   // Output pt 9: X(8)
        out[out_strides[10]] = at12 - at16;  // Output pt 11: X(10)
        out[out_strides[12]] = at15 - at6;   // Output pt 13: X(12)
        out[out_strides[14]] = at17 - at13;  // Output pt 15: X(14)

        /* Shifted DFT */
        FLOAT bv0, bv1, bv2, bv3, bv4, bv5, bv6, bv7;
        FLOAT bt0, bt1, bt2, bt3, bt4, bt5, bt6, bt7, bt8, bt9,
              bt10, bt11, bt12, bt13, bt14, bt15, bt16, bt17, bt18, bt19,
              bt20, bt21, bt22, bt23, bt24, bt25;

        bv0 = in[in_strides[1]];    // Input point 2: x(1)
        bv1 = in[in_strides[2]];    // Input point 3: x(2)
        bv2 = in[in_strides[5]];    // Input point 6: x(5)
        bv3 = in[in_strides[6]];    // Input point 7: x(6)
        bv4 = in[in_strides[9]];    // Input point 10: x(9)
        bv5 = in[in_strides[10]];   // Input point 11: x(10)
        bv6 = in[in_strides[13]];   // Input point 14: x(13)
        bv7 = in[in_strides[14]];   // Input point 13: x(12)

        bt0 = bv6 + bv0;
        bt1 = bv6 - bv0;
        bt2 = bv4 + bv2;
        bt3 = bv4 - bv2;
        bt4 = bv7 + bv1;
        bt5 = bv7 - bv1;
        bt6 = bv5 + bv3;
        bt7 = bv5 - bv3;
        bt8 = bt5 + bt7;
        bt9 = bt5 - bt7;
        bt10 = bt0 - bt2;
        bt11 = bt0 + bt2;

        bt12 = CRTM_8_1 * bt8;
        bt13 = CRTM_8_1 * bt10;
        bt14 = bt6 + bt1;
        bt15 = bt6 - bt1;
        bt16 = bt3 + bt4;
        bt17 = bt3 - bt4;
        bt18 = CRTM_8_2 * bt14;
        bt19 = CRTM_8_3 * bt15;
        bt20 = CRTM_8_3 * bt16;
        bt21 = CRTM_8_2 * bt17;
        bt22 = CRTM_8_3 * bt14;
        bt23 = CRTM_8_2 * bt15;
        bt24 = CRTM_8_2 * bt16;
        bt25 = CRTM_8_3 * bt17;

        out[out_strides[1]]  = CRTM_8_4 * bt11; // Output pt 2: X(1)
        out[out_strides[3]]  = -bt18 - bt20;    // Output pt 4: X(3)
        out[out_strides[5]]  = bt12 + bt13;     // Output pt 6: X(5)
        out[out_strides[7]]  = bt19 + bt21;     // Output pt 8: X(7)
        out[out_strides[9]]  = CRTM_8_4 * bt9;  // Output pt 10: X(9)
        out[out_strides[11]] = bt22 - bt24;     // Output pt 12: X(11)
        out[out_strides[13]] = bt12 - bt13;     // Output pt 14: X(12)
        out[out_strides[15]] = bt25 - bt23;     // Output pt 16: X(15)

        in = in + v_in_stride;
        out = out + v_out_stride;
    }
    AOCLFFTZ_LOG(DEBUG, global_logger_mode, "Exit");
}

static VOID r2hcf_rfft8c_fp64_fwd(VOID *in_real, VOID *in_imag, VOID *out_real,
                              VOID *out_imag, INTP n,
                              aoclfftz_strides_t *strides, VOID *twd, UINT8 flag)
{
    AOCLFFTZ_LOG(DEBUG, global_logger_mode, "Enter");
    const DOUBLE CRTM_8_1 = 0.7071067811865475244008443621048490392848359377;
    const DOUBLE CRTM_8_2 = 0.9238795325112867561281831893967882868224166259;
    const DOUBLE CRTM_8_3 = 0.3826834323650897717284599840303988667613445625;

    DOUBLE *in = (DOUBLE *)in_real;
    DOUBLE *out = (DOUBLE *)out_real;
#ifdef VOLATILE_STRIDE_ARRAY
    volatile INTP *in_strides = strides->in_strides;
    volatile INTP *out_strides = strides->out_strides;
#else
    INTP *in_strides = strides->in_strides;
    INTP *out_strides = strides->out_strides;
#endif
    INTP v_in_stride = (strides->v_in_stride);
    INTP v_out_stride = (strides->v_out_stride);
    INTP cnt;

    for (cnt = 0; cnt < n; cnt++)
    {
        /* Standard DFT */
        DOUBLE av0, av1, av2, av3, av4, av5, av6, av7;
        DOUBLE at0, at1, at2, at3, at4, at5, at6, at7, at8, at9,
               at10, at11, at12, at13;

        av0 = *in;                  // Input point 1: x(0)
        av1 = in[in_strides[2]];    // Input point 3: x(2)
        av2 = in[in_strides[4]];    // Input point 5: x(4)
        av3 = in[in_strides[6]];    // Input point 7: x(6)
        av4 = in[in_strides[8]];    // Input point 9: x(8)
        av5 = in[in_strides[10]];   // Input point 11: x(10)
        av6 = in[in_strides[12]];   // Input point 13: x(12)
        av7 = in[in_strides[14]];   // Input point 13: x(12)

        at0 = av7 + av5;
        at1 = av7 - av5;
        at2 = av6 + av2;
        at3 = av6 - av2;
        at4 = av0 + av4;
        at5 = av0 - av4;
        at6 = av3 + av1;
        at7 = av3 - av1;

        at8 = at7 - at1;
        at9 = at0 - at6;
        at10 = at0 + at6;
        at11 = at4 + at2;
        at12 = CRTM_8_1 * at8;
        at13 = CRTM_8_1 * at9;

        *out = at11 + at10;                   // Output pt 1: X(0)
        out[out_strides[3]]  = at5 - at12;    // Output pt 4: X(3)
        out[out_strides[4]]  = at3 + at13;    // Output pt 5: X(4)
        out[out_strides[7]]  = at4 - at2;     // Output pt 8: X(7)
        out[out_strides[8]]  = at7 + at1;     // Output pt 9: X(8)
        out[out_strides[11]] = at12 + at5;    // Output pt 12: X(11)
        out[out_strides[12]] = at13 - at3;    // Output pt 13: X(12)
        out[out_strides[15]] = at11 - at10;   // Output pt 16: X(15)

        /* Shifted DFT */
        DOUBLE bv0, bv1, bv2, bv3, bv4, bv5, bv6, bv7;
        DOUBLE bt0, bt1, bt2, bt3, bt4, bt5, bt6, bt7, bt8, bt9, bt10, bt11,
               bt12, bt13, bt14, bt15, bt16, bt17, bt18, bt19, bt20, bt21,
               bt22, bt23;

        bv0 = in[in_strides[1]];    // Input point 2: x(1)
        bv1 = in[in_strides[3]];    // Input point 4: x(3)
        bv2 = in[in_strides[5]];    // Input point 6: x(5)
        bv3 = in[in_strides[7]];    // Input point 8: x(7)
        bv4 = in[in_strides[9]];    // Input point 10: x(9)
        bv5 = in[in_strides[11]];   // Input point 12: x(11)
        bv6 = in[in_strides[13]];   // Input point 14: x(13)
        bv7 = in[in_strides[15]];   // Input point 14: x(13)

        bt0 = bv6 + bv2;
        bt1 = bv6 - bv2;
        bt2 = bv7 + bv1;
        bt3 = bv7 - bv1;
        bt4 = bv5 + bv3;
        bt5 = bv5 - bv3;

        bt6 = CRTM_8_1 * bt0;
        bt7 = CRTM_8_1 * bt1;
        bt8 = CRTM_8_2 * bt2;
        bt9 = CRTM_8_3 * bt2;
        bt10 = CRTM_8_2 * bt3;
        bt11 = CRTM_8_3 * bt3;
        bt12 = CRTM_8_2 * bt4;
        bt13 = CRTM_8_3 * bt4;
        bt14 = CRTM_8_2 * bt5;
        bt15 = CRTM_8_3 * bt5;

        bt16 = bv0 - bt7;
        bt17 = bv0 + bt7;
        bt18 = bv4 - bt6;
        bt19 = bt6 + bv4;
        bt20 = bt10 + bt15;
        bt21 = bt12 + bt9;
        bt22 = bt14 - bt11;
        bt23 = bt13 - bt8;

        out[out_strides[1]]  = bt16 - bt20;   // Output pt 2: X(1)
        out[out_strides[2]]  = -bt19 - bt21;  // Output pt 3: X(2)
        out[out_strides[5]]  = bt17 + bt22;   // Output pt 6: X(5)
        out[out_strides[6]]  = bt18 + bt23;   // Output pt 7: X(6)
        out[out_strides[9]]  = bt17 - bt22;   // Output pt 10: X(9)
        out[out_strides[10]] = bt23 - bt18;   // Output pt 11: X(10)
        out[out_strides[13]] = bt16 + bt20;   // Output pt 14: X(12)
        out[out_strides[14]] = bt19 - bt21;   // Output pt 15: X(14)

        in = in + v_in_stride;
        out = out + v_out_stride;
    }
    AOCLFFTZ_LOG(DEBUG, global_logger_mode, "Exit");
}

static VOID r2hcf_rfft8c_fp64_bwd(VOID *in_real, VOID *in_imag, VOID *out_real,
                              VOID *out_imag, INTP n,
                              aoclfftz_strides_t *strides, VOID *twd, UINT8 flag)
{
    AOCLFFTZ_LOG(DEBUG, global_logger_mode, "Enter");
    const DOUBLE CRTM_8_1 = 1.414213562373095048801688724209698078569671875;
    const DOUBLE CRTM_8_2 = 1.847759065022573512256366378793576573644833252;
    const DOUBLE CRTM_8_3 = 0.765366864730179543456919968060797733522689125;
    const DOUBLE CRTM_8_4 = 2.000000000000000000000000000000000000000000000;

    DOUBLE *in = (DOUBLE *)in_real;
    DOUBLE *out = (DOUBLE *)out_real;
#ifdef VOLATILE_STRIDE_ARRAY
    volatile INTP *in_strides = strides->in_strides;
    volatile INTP *out_strides = strides->out_strides;
#else
    INTP *in_strides = strides->in_strides;
    INTP *out_strides = strides->out_strides;
#endif
    INTP v_in_stride = (strides->v_in_stride);
    INTP v_out_stride = (strides->v_out_stride);
    INTP cnt;

    for (cnt = 0; cnt < n; cnt++)
    {
        /* Standard DFT */
        DOUBLE av0, av1, av2, av3, av4, av5, av6, av7;
        DOUBLE at0, at1, at2, at3, at4, at5, at6, at7, at8, at9,
               at10, at11, at12, at13, at14, at15, at16, at17;

        av0 = *in;                  // Input point 1: x(0)
        av1 = in[in_strides[3]];    // Input point 4: x(3)
        av2 = in[in_strides[4]];    // Input point 5: x(4)
        av3 = in[in_strides[7]];    // Input point 8: x(7)
        av4 = in[in_strides[8]];    // Input point 9: x(8)
        av5 = in[in_strides[11]];   // Input point 12: x(11)
        av6 = in[in_strides[12]];   // Input point 13: x(12)
        av7 = in[in_strides[15]];   // Input point 14: x(13)

        at0 = av7 + av0;
        at1 = av7 - av0;
        at2 = av6 + av2;
        at3 = av6 - av2;
        at4 = av5 + av1;
        at5 = av5 - av1;
        at6 = CRTM_8_4 * at3;
        at7 = CRTM_8_4 * at4;
        at8 = CRTM_8_4 * av3;
        at9 = CRTM_8_4 * av4;

        at10 = at5 + at2;
        at11 = at5 - at2;
        at12 = CRTM_8_1 * at10;
        at13 = CRTM_8_1 * at11;
        at14 = at0 + at8;
        at15 = at0 - at8;
        at16 = at9 + at1;
        at17 = at9 - at1;

        *out = at14 + at7;                   // Output pt 1: X(0)
        out[out_strides[2]]  = -at12 - at16; // Output pt 3: X(2)
        out[out_strides[4]]  = at15 + at6;   // Output pt 5: X(4)
        out[out_strides[6]]  = at17 + at13;  // Output pt 7: X(6)
        out[out_strides[8]]  = at14 - at7;   // Output pt 9: X(8)
        out[out_strides[10]] = at12 - at16;  // Output pt 11: X(10)
        out[out_strides[12]] = at15 - at6;   // Output pt 13: X(12)
        out[out_strides[14]] = at17 - at13;  // Output pt 15: X(14)

        /* Shifted DFT */
        DOUBLE bv0, bv1, bv2, bv3, bv4, bv5, bv6, bv7;
        DOUBLE bt0, bt1, bt2, bt3, bt4, bt5, bt6, bt7, bt8, bt9,
               bt10, bt11, bt12, bt13, bt14, bt15, bt16, bt17, bt18, bt19,
               bt20, bt21, bt22, bt23, bt24, bt25;

        bv0 = in[in_strides[1]];    // Input point 2: x(1)
        bv1 = in[in_strides[2]];    // Input point 3: x(2)
        bv2 = in[in_strides[5]];    // Input point 6: x(5)
        bv3 = in[in_strides[6]];    // Input point 7: x(6)
        bv4 = in[in_strides[9]];    // Input point 10: x(9)
        bv5 = in[in_strides[10]];   // Input point 11: x(10)
        bv6 = in[in_strides[13]];   // Input point 14: x(13)
        bv7 = in[in_strides[14]];   // Input point 13: x(12)

        bt0 = bv6 + bv0;
        bt1 = bv6 - bv0;
        bt2 = bv4 + bv2;
        bt3 = bv4 - bv2;
        bt4 = bv7 + bv1;
        bt5 = bv7 - bv1;
        bt6 = bv5 + bv3;
        bt7 = bv5 - bv3;
        bt8 = bt5 + bt7;
        bt9 = bt5 - bt7;
        bt10 = bt0 - bt2;
        bt11 = bt0 + bt2;

        bt12 = CRTM_8_1 * bt8;
        bt13 = CRTM_8_1 * bt10;
        bt14 = bt6 + bt1;
        bt15 = bt6 - bt1;
        bt16 = bt3 + bt4;
        bt17 = bt3 - bt4;
        bt18 = CRTM_8_2 * bt14;
        bt19 = CRTM_8_3 * bt15;
        bt20 = CRTM_8_3 * bt16;
        bt21 = CRTM_8_2 * bt17;
        bt22 = CRTM_8_3 * bt14;
        bt23 = CRTM_8_2 * bt15;
        bt24 = CRTM_8_2 * bt16;
        bt25 = CRTM_8_3 * bt17;

        out[out_strides[1]]  = CRTM_8_4 * bt11; // Output pt 2: X(1)
        out[out_strides[3]]  = -bt18 - bt20;    // Output pt 4: X(3)
        out[out_strides[5]]  = bt12 + bt13;     // Output pt 6: X(5)
        out[out_strides[7]]  = bt19 + bt21;     // Output pt 8: X(7)
        out[out_strides[9]]  = CRTM_8_4 * bt9;  // Output pt 10: X(9)
        out[out_strides[11]] = bt22 - bt24;     // Output pt 12: X(11)
        out[out_strides[13]] = bt12 - bt13;     // Output pt 14: X(12)
        out[out_strides[15]] = bt25 - bt23;     // Output pt 16: X(15)

        in = in + v_in_stride;
        out = out + v_out_stride;
    }
    AOCLFFTZ_LOG(DEBUG, global_logger_mode, "Exit");
}

kfft_ register_kernel_r2hcf_rfft8c(UINT8 precision, UINT8 direction)
{
    if (direction == FORWARD_FFT_DIR)
    {
        if (precision == DT_FLOAT)
        {
            return r2hcf_rfft8c_fp32_fwd;
        }
        else if (precision == DT_DOUBLE)
        {
            return r2hcf_rfft8c_fp64_fwd;
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
            return r2hcf_rfft8c_fp32_bwd;
        }
        else if (precision == DT_DOUBLE)
        {
            return r2hcf_rfft8c_fp64_bwd;
        }
        else
        {
            return NULL;
        }
    }
}
