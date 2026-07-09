// Copyright Advanced Micro Devices, Inc.
// SPDX-License-Identifier: BSD-3-Clause

/** @file rfft6c.c
 *
 *  @brief Radix-6 r2hc_fused Real-FFT kernel with scalar operations in C
 *
 *  This file contains the DIT radix-6 real-to-halfcomplex fused of two
 *  different implementations (Standard DFT and Shifted DFT that differs in DFT
 *  weight matrix) using scalar operations for single-precision
 *  and double-precision inputs.
 *
 *  @author Srirammaswamy Srinivasan
 */

#include "core/kernels/kernel.h"

static const ops_cycles_t ops_cnt[NUM_PRECISIONS][NUM_FFT_DIRS] =
                                                    {{{0, 8, 26, 24, 0, 0},
                                                      {0, 10, 26, 24, 0, 0}},
                                                     {{0, 8, 26, 24, 0, 0},
                                                      {0, 10, 26, 24, 0, 0}}};

ops_cycles_t get_ops_cnt_r2hcf_rfft6c(FFTZ_UINT8 precision,
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

static FFTZ_VOID r2hcf_rfft6c_fp32_fwd(FFTZ_VOID *in_real, FFTZ_VOID *in_imag,
                                       FFTZ_VOID *out_real, FFTZ_VOID *out_imag,
                                       FFTZ_INTP n, aoclfftz_strides_t *strides,
                                       FFTZ_VOID *twd, FFTZ_UINT8 flag)
{
    AOCLFFTZ_LOG(DEBUG, global_logger_mode, "Enter");
    const FFTZ_FLOAT CRTM_6_1 =
        0.500000000000000000000000000000000000000000000f;
    const FFTZ_FLOAT CRTM_6_2 =
        0.866025403784438646763723170752936183471402627f;

    FFTZ_FLOAT *in = (FFTZ_FLOAT *)in_real;
    FFTZ_FLOAT *out = (FFTZ_FLOAT *)out_real;
#ifdef VOLATILE_STRIDE_ARRAY
    volatile FFTZ_INTP *in_strides = strides->in_strides;
    volatile FFTZ_INTP *out_strides = strides->out_strides;
#else
    FFTZ_INTP *in_strides = strides->in_strides;
    FFTZ_INTP *out_strides = strides->out_strides;
#endif
    FFTZ_INTP v_in_stride = (strides->v_in_stride);
    FFTZ_INTP v_out_stride = (strides->v_out_stride);
    FFTZ_INTP cnt;

    for (cnt = 0; cnt < n; cnt++)
    {
        /* Standard DFT */
        FFTZ_FLOAT av0, av1, av2, av3, av4, av5;
        FFTZ_FLOAT at0, at1, at2, at3, at4, at5, at6, at7, at8, at9, at10, at11;

        av0 = *in;                  // Input point 1: x(0)
        av1 = in[in_strides[2]];    // Input point 3: x(2)
        av2 = in[in_strides[4]];    // Input point 5: x(4)
        av3 = in[in_strides[6]];    // Input point 7: x(6)
        av4 = in[in_strides[8]];    // Input point 9: x(8)
        av5 = in[in_strides[10]];   // Input point 11: x(10)

        at0 = av0 + av3;
        at1 = av0 - av3;
        at2 = av1 + av2;
        at3 = av1 - av2;
        at4 = av5 + av4;
        at5 = av5 - av4;
        at6 = at2 + at4;
        at7 = at3 + at5;

        at8  = CRTM_6_1 * at7;
        at9  = at4 - at2;
        at10 = CRTM_6_1 * at6;
        at11 = at5 - at3;

        *out = at0 + at6;                       // Output pt 1: X(0)
        out[out_strides[3]]  = at1 + at8;       // Output pt 4: X(3)
        out[out_strides[4]]  = CRTM_6_2 * at9;  // Output pt 5: X(4)
        out[out_strides[7]]  = at0 - at10;      // Output pt 8: X(7)
        out[out_strides[8]]  = CRTM_6_2 * at11; // Output pt 9: X(8)
        out[out_strides[11]] = at1 - at7;       // Output pt 12: X(11)

        /* Shifted DFT */
        FFTZ_FLOAT bv0, bv1, bv2, bv3, bv4, bv5;
        FFTZ_FLOAT bt0, bt1, bt2, bt3, bt4, bt5, bt6, bt7, bt8, bt9;

        bv0 = in[in_strides[1]];    // Input point 2: x(1)
        bv1 = in[in_strides[3]];    // Input point 4: x(3)
        bv2 = in[in_strides[5]];    // Input point 6: x(5)
        bv3 = in[in_strides[7]];    // Input point 8: x(7)
        bv4 = in[in_strides[9]];    // Input point 10: x(9)
        bv5 = in[in_strides[11]];   // Input point 12: x(11)

        bt0 = bv1 - bv5;
        bt1 = bv1 + bv5;
        bt2 = bv2 - bv4;
        bt3 = bv2 + bv4;

        bt4 = CRTM_6_2 * bt0;
        bt5 = CRTM_6_1 * bt1;
        bt6 = CRTM_6_1 * bt2;
        bt7 = CRTM_6_2 * bt3;

        bt8 = bv0 + bt6;
        bt9 = bv3 + bt5;

        out[out_strides[1]]  = bt8 + bt4;   // Output pt 2: X(1)
        out[out_strides[2]]  = -bt7 - bt9;  // Output pt 3: X(2)
        out[out_strides[5]]  = bv0 - bt2;   // Output pt 6: X(5)
        out[out_strides[6]]  = bv3 - bt1;   // Output pt 7: X(6)
        out[out_strides[9]]  = bt8 - bt4;   // Output pt 10: X(9)
        out[out_strides[10]] = bt7 - bt9;   // Output pt 11: X(10)

        in = in + v_in_stride;
        out = out + v_out_stride;
    }
    AOCLFFTZ_LOG(DEBUG, global_logger_mode, "Exit");
}

static FFTZ_VOID r2hcf_rfft6c_fp32_bwd(FFTZ_VOID *in_real, FFTZ_VOID *in_imag,
                                       FFTZ_VOID *out_real, FFTZ_VOID *out_imag,
                                       FFTZ_INTP n, aoclfftz_strides_t *strides,
                                       FFTZ_VOID *twd, FFTZ_UINT8 flag)
{
    AOCLFFTZ_LOG(DEBUG, global_logger_mode, "Enter");
    const FFTZ_FLOAT CRTM_6_1 =
        1.732050807568877293527446341505872366942805253f;
    const FFTZ_FLOAT CRTM_6_2 = 2.0f;

    FFTZ_FLOAT *in = (FFTZ_FLOAT *)in_real;
    FFTZ_FLOAT *out = (FFTZ_FLOAT *)out_real;
#ifdef VOLATILE_STRIDE_ARRAY
    volatile FFTZ_INTP *in_strides = strides->in_strides;
    volatile FFTZ_INTP *out_strides = strides->out_strides;
#else
    FFTZ_INTP *in_strides = strides->in_strides;
    FFTZ_INTP *out_strides = strides->out_strides;
#endif
    FFTZ_INTP v_in_stride = (strides->v_in_stride);
    FFTZ_INTP v_out_stride = (strides->v_out_stride);
    FFTZ_INTP cnt;

    for (cnt = 0; cnt < n; cnt++)
    {
        /* Standard DFT */
        FFTZ_FLOAT av0, av1, av2, av3, av4, av5;
        FFTZ_FLOAT at0, at1, at2, at3, at4, at5, at6, at7, at8, at9, at10, at11;

        av0 = *in;                  // Input point 1: x(0)
        av1 = in[in_strides[3]];    // Input point 3: x(2)
        av2 = in[in_strides[4]];    // Input point 5: x(4)
        av3 = in[in_strides[7]];    // Input point 7: x(6)
        av4 = in[in_strides[8]];    // Input point 9: x(8)
        av5 = in[in_strides[11]];   // Input point 11: x(10)

        at0 = av0 + av5;
        at1 = av0 - av5;
        at2 = av1 + av3;
        at3 = av1 - av3;
        at4 = av2 + av4;
        at5 = av2 - av4;

        at6 = CRTM_6_1 * at4;
        at7 = CRTM_6_1 * at5;
        at8 = at1 + at3;
        at9 = at0 - at2;

        at10 = CRTM_6_2 * at2;
        at11 = CRTM_6_2 * at3;

        *out = at0 + at10;                  // Output pt 1: X(0)
        out[out_strides[2]]  = at8 - at6;   // Output pt 3: X(2)
        out[out_strides[4]]  = at9 - at7;   // Output pt 5: X(4)
        out[out_strides[6]]  = at1 - at11;  // Output pt 7: X(6)
        out[out_strides[8]]  = at9 + at7;   // Output pt 9: X(8)
        out[out_strides[10]] = at8 + at6;   // Output pt 11: X(10)

        /* Shifted DFT */
        FFTZ_FLOAT bv0, bv1, bv2, bv3, bv4, bv5;
        FFTZ_FLOAT bt0, bt1, bt2, bt3, bt4, bt5, bt6, bt7, bt8, bt9, bt10, bt11;

        bv0 = in[in_strides[1]];    // Input point 2: x(1)
        bv1 = in[in_strides[2]];    // Input point 4: x(3)
        bv2 = in[in_strides[5]];    // Input point 6: x(5)
        bv3 = in[in_strides[6]];    // Input point 8: x(7)
        bv4 = in[in_strides[9]];    // Input point 10: x(9)
        bv5 = in[in_strides[10]];   // Input point 12: x(11)

        bt0 = bv0 + bv4;
        bt1 = bv0 - bv4;
        bt2 = CRTM_6_1 * bt1;
        bt3 = bv1 + bv5;
        bt4 = bv1 - bv5;
        bt5 = CRTM_6_1 * bt4;

        bt6 = CRTM_6_2 * bt0;
        bt7 = CRTM_6_2 * bt3;
        bt8 = CRTM_6_2 * bv2;
        bt9 = CRTM_6_2 * bv3;

        bt10 = bt3 + bt9;
        bt11 = bt0 - bt8;

        out[out_strides[1]]  = bt8 + bt6;   // Output pt 2: X(1)
        out[out_strides[3]]  = bt2 - bt10;  // Output pt 4: X(3)
        out[out_strides[5]]  = bt11 - bt5;  // Output pt 6: X(5)
        out[out_strides[7]]  = bt9 - bt7;   // Output pt 8: X(7)
        out[out_strides[9]]  = -bt11 - bt5; // Output pt 10: X(9)
        out[out_strides[11]] = -bt2 - bt10; // Output pt 12: X(11)

        in = in + v_in_stride;
        out = out + v_out_stride;
    }
    AOCLFFTZ_LOG(DEBUG, global_logger_mode, "Exit");
}

static FFTZ_VOID r2hcf_rfft6c_fp64_fwd(FFTZ_VOID *in_real, FFTZ_VOID *in_imag,
                                       FFTZ_VOID *out_real, FFTZ_VOID *out_imag,
                                       FFTZ_INTP n, aoclfftz_strides_t *strides,
                                       FFTZ_VOID *twd, FFTZ_UINT8 flag)
{
    AOCLFFTZ_LOG(DEBUG, global_logger_mode, "Enter");
    const FFTZ_DOUBLE CRTM_6_1 =
        0.500000000000000000000000000000000000000000000;
    const FFTZ_DOUBLE CRTM_6_2 =
        0.866025403784438646763723170752936183471402627;

    FFTZ_DOUBLE *in = (FFTZ_DOUBLE *)in_real;
    FFTZ_DOUBLE *out = (FFTZ_DOUBLE *)out_real;
#ifdef VOLATILE_STRIDE_ARRAY
    volatile FFTZ_INTP *in_strides = strides->in_strides;
    volatile FFTZ_INTP *out_strides = strides->out_strides;
#else
    FFTZ_INTP *in_strides = strides->in_strides;
    FFTZ_INTP *out_strides = strides->out_strides;
#endif
    FFTZ_INTP v_in_stride = (strides->v_in_stride);
    FFTZ_INTP v_out_stride = (strides->v_out_stride);
    FFTZ_INTP cnt;

    for (cnt = 0; cnt < n; cnt++)
    {
        /* Standard DFT */
        FFTZ_DOUBLE av0, av1, av2, av3, av4, av5;
        FFTZ_DOUBLE at0, at1, at2, at3, at4, at5, at6, at7, at8, at9, at10,
            at11;

        av0 = *in;                  // Input point 1: x(0)
        av1 = in[in_strides[2]];    // Input point 3: x(2)
        av2 = in[in_strides[4]];    // Input point 5: x(4)
        av3 = in[in_strides[6]];    // Input point 7: x(6)
        av4 = in[in_strides[8]];    // Input point 9: x(8)
        av5 = in[in_strides[10]];   // Input point 11: x(10)

        at0 = av0 + av3;
        at1 = av0 - av3;
        at2 = av1 + av2;
        at3 = av1 - av2;
        at4 = av5 + av4;
        at5 = av5 - av4;
        at6 = at2 + at4;
        at7 = at3 + at5;

        at8  = CRTM_6_1 * at7;
        at9  = at4 - at2;
        at10 = CRTM_6_1 * at6;
        at11 = at5 - at3;

        *out = at0 + at6;                       // Output pt 1: X(0)
        out[out_strides[3]]  = at1 + at8;       // Output pt 4: X(3)
        out[out_strides[4]]  = CRTM_6_2 * at9;  // Output pt 5: X(4)
        out[out_strides[7]]  = at0 - at10;      // Output pt 8: X(7)
        out[out_strides[8]]  = CRTM_6_2 * at11; // Output pt 9: X(8)
        out[out_strides[11]] = at1 - at7;       // Output pt 12: X(11)

        /* Shifted DFT */
        FFTZ_DOUBLE bv0, bv1, bv2, bv3, bv4, bv5;
        FFTZ_DOUBLE bt0, bt1, bt2, bt3, bt4, bt5, bt6, bt7, bt8, bt9;

        bv0 = in[in_strides[1]];    // Input point 2: x(1)
        bv1 = in[in_strides[3]];    // Input point 4: x(3)
        bv2 = in[in_strides[5]];    // Input point 6: x(5)
        bv3 = in[in_strides[7]];    // Input point 8: x(7)
        bv4 = in[in_strides[9]];    // Input point 10: x(9)
        bv5 = in[in_strides[11]];   // Input point 12: x(11)

        bt0 = bv1 - bv5;
        bt1 = bv1 + bv5;
        bt2 = bv2 - bv4;
        bt3 = bv2 + bv4;
        bt4 = CRTM_6_2 * bt0;
        bt5 = CRTM_6_1 * bt1;
        bt6 = CRTM_6_1 * bt2;
        bt7 = CRTM_6_2 * bt3;

        bt8 = bv0 + bt6;
        bt9 = bv3 + bt5;

        out[out_strides[1]]  = bt8 + bt4;   // Output pt 2: X(1)
        out[out_strides[2]]  = -bt7 - bt9;  // Output pt 3: X(2)
        out[out_strides[5]]  = bv0 - bt2;   // Output pt 6: X(5)
        out[out_strides[6]]  = bv3 - bt1;   // Output pt 7: X(6)
        out[out_strides[9]]  = bt8 - bt4;   // Output pt 10: X(9)
        out[out_strides[10]] = bt7 - bt9;   // Output pt 11: X(10)

        in = in + v_in_stride;
        out = out + v_out_stride;
    }
    AOCLFFTZ_LOG(DEBUG, global_logger_mode, "Exit");
}

static FFTZ_VOID r2hcf_rfft6c_fp64_bwd(FFTZ_VOID *in_real, FFTZ_VOID *in_imag,
                                       FFTZ_VOID *out_real, FFTZ_VOID *out_imag,
                                       FFTZ_INTP n, aoclfftz_strides_t *strides,
                                       FFTZ_VOID *twd, FFTZ_UINT8 flag)
{
    AOCLFFTZ_LOG(DEBUG, global_logger_mode, "Enter");
    const FFTZ_DOUBLE CRTM_6_1 =
        1.732050807568877293527446341505872366942805253;
    const FFTZ_DOUBLE CRTM_6_2 = 2.0;

    FFTZ_DOUBLE *in = (FFTZ_DOUBLE *)in_real;
    FFTZ_DOUBLE *out = (FFTZ_DOUBLE *)out_real;
#ifdef VOLATILE_STRIDE_ARRAY
    volatile FFTZ_INTP *in_strides = strides->in_strides;
    volatile FFTZ_INTP *out_strides = strides->out_strides;
#else
    FFTZ_INTP *in_strides = strides->in_strides;
    FFTZ_INTP *out_strides = strides->out_strides;
#endif
    FFTZ_INTP v_in_stride = (strides->v_in_stride);
    FFTZ_INTP v_out_stride = (strides->v_out_stride);
    FFTZ_INTP cnt;

    for (cnt = 0; cnt < n; cnt++)
    {
        /* Standard DFT */
        FFTZ_DOUBLE av0, av1, av2, av3, av4, av5;
        FFTZ_DOUBLE at0, at1, at2, at3, at4, at5, at6, at7, at8, at9, at10,
            at11;

        av0 = *in;                  // Input point 1: x(0)
        av1 = in[in_strides[3]];    // Input point 3: x(2)
        av2 = in[in_strides[4]];    // Input point 5: x(4)
        av3 = in[in_strides[7]];    // Input point 7: x(6)
        av4 = in[in_strides[8]];    // Input point 9: x(8)
        av5 = in[in_strides[11]];   // Input point 11: x(10)

        at0 = av0 + av5;
        at1 = av0 - av5;
        at2 = av1 + av3;
        at3 = av1 - av3;
        at4 = av2 + av4;
        at5 = av2 - av4;

        at6 = CRTM_6_1 * at4;
        at7 = CRTM_6_1 * at5;
        at8 = at1 + at3;
        at9 = at0 - at2;

        at10 = CRTM_6_2 * at2;
        at11 = CRTM_6_2 * at3;

        *out = at0 + at10;                  // Output pt 1: X(0)
        out[out_strides[2]] = at8 - at6;    // Output pt 3: X(2)
        out[out_strides[4]] = at9 - at7;    // Output pt 5: X(4)
        out[out_strides[6]] = at1 - at11;   // Output pt 7: X(6)
        out[out_strides[8]] = at9 + at7;    // Output pt 9: X(8)
        out[out_strides[10]] = at8 + at6;   // Output pt 11: X(10)

        /* Shifted DFT */
        FFTZ_DOUBLE bv0, bv1, bv2, bv3, bv4, bv5;
        FFTZ_DOUBLE bt0, bt1, bt2, bt3, bt4, bt5, bt6, bt7, bt8, bt9, bt10,
            bt11;

        bv0 = in[in_strides[1]];    // Input point 2: x(1)
        bv1 = in[in_strides[2]];    // Input point 4: x(3)
        bv2 = in[in_strides[5]];    // Input point 6: x(5)
        bv3 = in[in_strides[6]];    // Input point 8: x(7)
        bv4 = in[in_strides[9]];    // Input point 10: x(9)
        bv5 = in[in_strides[10]];   // Input point 12: x(11)

        bt0 = bv0 + bv4;
        bt1 = bv0 - bv4;
        bt2 = CRTM_6_1 * bt1;
        bt3 = bv1 + bv5;
        bt4 = bv1 - bv5;
        bt5 = CRTM_6_1 * bt4;

        bt6 = CRTM_6_2 * bt0;
        bt7 = CRTM_6_2 * bt3;
        bt8 = CRTM_6_2 * bv2;

        bt9  = CRTM_6_2 * bv3;
        bt10 = bt3 + bt9;
        bt11 = bt0 - bt8;

        out[out_strides[1]]  = bt8 + bt6;    // Output pt 2: X(1)
        out[out_strides[3]]  = bt2 - bt10;   // Output pt 4: X(3)
        out[out_strides[5]]  = bt11 - bt5;   // Output pt 6: X(5)
        out[out_strides[7]]  = bt9 - bt7;    // Output pt 8: X(7)
        out[out_strides[9]]  = -bt11 - bt5;  // Output pt 10: X(9)
        out[out_strides[11]] = -bt2 - bt10;  // Output pt 12: X(11)

        in = in + v_in_stride;
        out = out + v_out_stride;
    }
    AOCLFFTZ_LOG(DEBUG, global_logger_mode, "Exit");
}

kfft_ register_kernel_r2hcf_rfft6c(FFTZ_UINT8 precision, FFTZ_UINT8 direction)
{
    if (direction == FORWARD_FFT_DIR)
    {
        if (precision == DT_FLOAT)
        {
            return r2hcf_rfft6c_fp32_fwd;
        }
        else if (precision == DT_DOUBLE)
        {
            return r2hcf_rfft6c_fp64_fwd;
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
            return r2hcf_rfft6c_fp32_bwd;
        }
        else if (precision == DT_DOUBLE)
        {
            return r2hcf_rfft6c_fp64_bwd;
        }
        else
        {
            return NULL;
        }
    }
}
