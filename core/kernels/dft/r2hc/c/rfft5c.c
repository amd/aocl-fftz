// Copyright Advanced Micro Devices, Inc.
// SPDX-License-Identifier: BSD-3-Clause

/** @file rfft5c.c
 *
 *  @brief Radix-5 r2hc Real-FFT kernel with scalar operations in C
 *
 *  This file contains the DIT radix-5 real-to-halfcomplex implementations
 *  using scalar operations for single-precision and double-precision inputs.
 *
 *  @author Srirammaswamy Srinivasan
 */

#include "core/kernels/kernel.h"

static const ops_cycles_t ops_cnt[NUM_PRECISIONS][NUM_FFT_DIRS] =
                                                    {{{0, 6, 12, 10, 0, 0},
                                                      {0, 7, 12, 10, 0, 0}},
                                                     {{0, 6, 12, 10, 0, 0},
                                                      {0, 7, 12, 10, 0, 0}}};

ops_cycles_t get_ops_cnt_r2hc_rfft5c(UINT8 precision, UINT8 direction)
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

static VOID r2hc_rfft5c_fp32_fwd(VOID *in_real, VOID *in_imag, VOID *out_real,
                                 VOID *out_imag, INTP n,
                                 aoclfftz_strides_t *strides, VOID *twd, UINT8 flag)
{
    AOCLFFTZ_LOG(DEBUG, global_logger_mode, "Enter");
    const FLOAT CRTM_5_1 = 0.559016994374947424102293417182819058860154590f;
    const FLOAT CRTM_5_2 = 0.951056516295153572116439333379382143405698632f;
    const FLOAT CRTM_5_3 = 0.587785252292473129168705954639072768597652438f;
    const FLOAT CRTM_5_4 = 0.250000000000000000000000000000000000000000000f;

    FLOAT *in  = (FLOAT *)in_real;
    FLOAT *out = (FLOAT *)out_real;
#ifdef VOLATILE_STRIDE_ARRAY
    volatile INTP *in_strides  = strides->in_strides;
    volatile INTP *out_strides = strides->out_strides;
#else
    INTP *in_strides  = strides->in_strides;
    INTP *out_strides = strides->out_strides;
#endif
    INTP v_in_stride  = (strides->v_in_stride);
    INTP v_out_stride = (strides->v_out_stride);
    INTP cnt;

    for (cnt = 0; cnt < n; cnt++)
    {
        FLOAT v0, v1, v2, v3, v4;
        FLOAT t0, t1, t2, t3, t4, t5, t6, t7, t8, t9, t10, t11, t12;

        // Input point 1: x(0)
        v0 = *in;
        // Input point 2: x(1)
        v1 = in[in_strides[1]];
        // Input point 3: x(2)
        v2 = in[in_strides[2]];
        // Input point 4: x(3)
        v3 = in[in_strides[3]];
        // Input point 5: x(4)
        v4 = in[in_strides[4]];

        t0 = v1 + v4;
        t1 = v1 - v4;
        t2 = v2 + v3;
        t3 = v2 - v3;
        t4 = t0 + t2;

        t5 = CRTM_5_4 * t4;
        t6 = t0 - t2;

        t7 = v0 - t5;
        t8 = CRTM_5_1 * t6;

        t9  = CRTM_5_3 * t3;
        t10 = CRTM_5_2 * t1;
        t11 = CRTM_5_2 * t3;
        t12 = CRTM_5_3 * t1;

        // Output point 1: X(0)
        *out = v0 + t4;

        // Output point 2: X(1)
        out[out_strides[1]] = t7 + t8;

        // Output point 3: X(2)
        out[out_strides[2]] = -t9 - t10;

        // Output point 4: X(3)
        out[out_strides[3]] = t7 - t8;

        // Output point 5: X(4)
        out[out_strides[4]] = t11 - t12;

        in  = in + v_in_stride;
        out = out + v_out_stride;
    }
    AOCLFFTZ_LOG(DEBUG, global_logger_mode, "Exit");
}

static VOID r2hc_rfft5c_fp32_bwd(VOID *in_real, VOID *in_imag, VOID *out_real,
                                 VOID *out_imag, INTP n,
                                 aoclfftz_strides_t *strides, VOID *twd, UINT8 flag)
{
    AOCLFFTZ_LOG(DEBUG, global_logger_mode, "Enter");
    const FLOAT CRTM_5_1 = 1.11803398874989484820458683436563811772030918f;
    const FLOAT CRTM_5_2 = 1.90211303259030714423287866675876428681139726f;
    const FLOAT CRTM_5_3 = 1.17557050458494625833741190927814553719530488f;
    const FLOAT CRTM_5_4 = 0.50000000000000000000000000000000000000000000f;
    const FLOAT CRTM_5_5 = 2.000000000000000000000000000000000000000000000f;

    FLOAT *in  = (FLOAT *)in_real;
    FLOAT *out = (FLOAT *)out_real;
#ifdef VOLATILE_STRIDE_ARRAY
    volatile INTP *in_strides  = strides->in_strides;
    volatile INTP *out_strides = strides->out_strides;
#else
    INTP *in_strides  = strides->in_strides;
    INTP *out_strides = strides->out_strides;
#endif
    INTP v_in_stride  = (strides->v_in_stride);
    INTP v_out_stride = (strides->v_out_stride);
    INTP cnt;

    for (cnt = 0; cnt < n; cnt++)
    {
        FLOAT v0, v1, v2, v3, v4;
        FLOAT t0, t1, t2, t3, t4, t5, t6, t7, t8, t9, t10, t11, t12, t13;

        // Input point 1: x(0)
        v0 = *in;
        // Input point 2: x(1)
        v1 = in[in_strides[1]];
        // Input point 3: x(2)
        v2 = in[in_strides[2]];
        // Input point 4: x(3)
        v3 = in[in_strides[3]];
        // Input point 5: x(4)
        v4 = in[in_strides[4]];

        t0 = v1 + v3;
        t1 = v1 - v3;
        t2 = CRTM_5_4 * t0;
        t3 = CRTM_5_1 * t1;
        t4 = v0 - t2;
        t5 = t4 + t3;
        t6 = t4 - t3;

        t7  = CRTM_5_2 * v4;
        t8  = CRTM_5_3 * v2;
        t9  = CRTM_5_3 * v4;
        t10 = CRTM_5_2 * v2;
        t11 = CRTM_5_5 * t0;

        t12 = t10 + t9;
        t13 = t7 - t8;

        // Output point 1: X(0)
        *out = v0 + t11;

        // Output point 2: X(1)
        out[out_strides[1]] = t5 - t12;

        // Output point 3: X(2)
        out[out_strides[2]] = t6 + t13;

        // Output point 4: X(3)
        out[out_strides[3]] = t6 - t13;

        // Output point 5: X(4)
        out[out_strides[4]] = t5 + t12;

        in  = in + v_in_stride;
        out = out + v_out_stride;
    }
    AOCLFFTZ_LOG(DEBUG, global_logger_mode, "Exit");
}

static VOID r2hc_rfft5c_fp64_fwd(VOID *in_real, VOID *in_imag, VOID *out_real,
                                 VOID *out_imag, INTP n,
                                 aoclfftz_strides_t *strides, VOID *twd, UINT8 flag)
{
    AOCLFFTZ_LOG(DEBUG, global_logger_mode, "Enter");
    const DOUBLE CRTM_5_1 = 0.559016994374947424102293417182819058860154590;
    const DOUBLE CRTM_5_2 = 0.951056516295153572116439333379382143405698632;
    const DOUBLE CRTM_5_3 = 0.587785252292473129168705954639072768597652438;
    const DOUBLE CRTM_5_4 = 0.250000000000000000000000000000000000000000000;

    DOUBLE *in  = (DOUBLE *)in_real;
    DOUBLE *out = (DOUBLE *)out_real;
#ifdef VOLATILE_STRIDE_ARRAY
    volatile INTP *in_strides  = strides->in_strides;
    volatile INTP *out_strides = strides->out_strides;
#else
    INTP *in_strides  = strides->in_strides;
    INTP *out_strides = strides->out_strides;
#endif
    INTP v_in_stride  = (strides->v_in_stride);
    INTP v_out_stride = (strides->v_out_stride);
    INTP cnt;

    for (cnt = 0; cnt < n; cnt++)
    {
        DOUBLE v0, v1, v2, v3, v4;
        DOUBLE t0, t1, t2, t3, t4, t5, t6, t7, t8, t9, t10, t11, t12;

        // Input point 1: x(0)
        v0 = *in;
        // Input point 2: x(1)
        v1 = in[in_strides[1]];
        // Input point 3: x(2)
        v2 = in[in_strides[2]];
        // Input point 4: x(3)
        v3 = in[in_strides[3]];
        // Input point 5: x(4)
        v4 = in[in_strides[4]];

        t0 = v1 + v4;
        t1 = v1 - v4;
        t2 = v2 + v3;
        t3 = v2 - v3;
        t4 = t0 + t2;

        t5 = CRTM_5_4 * t4;
        t6 = t0 - t2;

        t7 = v0 - t5;
        t8 = CRTM_5_1 * t6;

        t9  = CRTM_5_3 * t3;
        t10 = CRTM_5_2 * t1;
        t11 = CRTM_5_2 * t3;
        t12 = CRTM_5_3 * t1;

        // Output point 1: X(0)
        *out = v0 + t4;

        // Output point 2: X(1)
        out[out_strides[1]] = t7 + t8;

        // Output point 3: X(2)
        out[out_strides[2]] = -t9 - t10;

        // Output point 4: X(3)
        out[out_strides[3]] = t7 - t8;

        // Output point 5: X(4)
        out[out_strides[4]] = t11 - t12;

        in  = in + v_in_stride;
        out = out + v_out_stride;
    }
    AOCLFFTZ_LOG(DEBUG, global_logger_mode, "Exit");
}

static VOID r2hc_rfft5c_fp64_bwd(VOID *in_real, VOID *in_imag, VOID *out_real,
                                 VOID *out_imag, INTP n,
                                 aoclfftz_strides_t *strides, VOID *twd, UINT8 flag)
{
    AOCLFFTZ_LOG(DEBUG, global_logger_mode, "Enter");
    const DOUBLE CRTM_5_1 = 1.11803398874989484820458683436563811772030918;
    const DOUBLE CRTM_5_2 = 1.90211303259030714423287866675876428681139726;
    const DOUBLE CRTM_5_3 = 1.17557050458494625833741190927814553719530488;
    const DOUBLE CRTM_5_4 = 0.50000000000000000000000000000000000000000000;
    const DOUBLE CRTM_5_5 = 2.000000000000000000000000000000000000000000000;

    DOUBLE *in  = (DOUBLE *)in_real;
    DOUBLE *out = (DOUBLE *)out_real;
#ifdef VOLATILE_STRIDE_ARRAY
    volatile INTP *in_strides  = strides->in_strides;
    volatile INTP *out_strides = strides->out_strides;
#else
    INTP *in_strides  = strides->in_strides;
    INTP *out_strides = strides->out_strides;
#endif
    INTP v_in_stride  = (strides->v_in_stride);
    INTP v_out_stride = (strides->v_out_stride);
    INTP cnt;

    for (cnt = 0; cnt < n; cnt++)
    {
        DOUBLE v0, v1, v2, v3, v4;
        DOUBLE t0, t1, t2, t3, t4, t5, t6, t7, t8, t9, t10, t11, t12, t13;

        // Input point 1: x(0)
        v0 = *in;
        // Input point 2: x(1)
        v1 = in[in_strides[1]];
        // Input point 3: x(2)
        v2 = in[in_strides[2]];
        // Input point 4: x(3)
        v3 = in[in_strides[3]];
        // Input point 5: x(4)
        v4 = in[in_strides[4]];

        t0 = v1 + v3;
        t1 = v1 - v3;
        t2 = CRTM_5_4 * t0;
        t3 = CRTM_5_1 * t1;
        t4 = v0 - t2;
        t5 = t4 + t3;
        t6 = t4 - t3;

        t7  = CRTM_5_2 * v4;
        t8  = CRTM_5_3 * v2;
        t9  = CRTM_5_3 * v4;
        t10 = CRTM_5_2 * v2;
        t11 = CRTM_5_5 * t0;

        t12 = t10 + t9;
        t13 = t7 - t8;

        // Output point 1: X(0)
        *out = v0 + t11;

        // Output point 2: X(1)
        out[out_strides[1]] = t5 - t12;

        // Output point 3: X(2)
        out[out_strides[2]] = t6 + t13;

        // Output point 4: X(3)
        out[out_strides[3]] = t6 - t13;

        // Output point 5: X(4)
        out[out_strides[4]] = t5 + t12;

        in  = in + v_in_stride;
        out = out + v_out_stride;
    }
    AOCLFFTZ_LOG(DEBUG, global_logger_mode, "Exit");
}

kfft_ register_kernel_r2hc_rfft5c(UINT8 precision, UINT8 direction)
{
    if (direction == FORWARD_FFT_DIR)
    {
        if (precision == DT_FLOAT)
        {
            return r2hc_rfft5c_fp32_fwd;
        }
        else if (precision == DT_DOUBLE)
        {
            return r2hc_rfft5c_fp64_fwd;
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
            return r2hc_rfft5c_fp32_bwd;
        }
        else if (precision == DT_DOUBLE)
        {
            return r2hc_rfft5c_fp64_bwd;
        }
        else
        {
            return NULL;
        }
    }
}
