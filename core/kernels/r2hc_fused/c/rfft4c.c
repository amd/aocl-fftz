/**
 * Copyright (C) 2025, Advanced Micro Devices. All rights reserved.
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

/** @file rfft4c.c
 *
 *  @brief Radix-4 r2hc_fused Real-FFT kernel with scalar operations in C
 *
 *  This file contains the DIT radix-4 real-to-halfcomplex fused of two
 *  different implementations (Standard DFT and Shifted DFT that differs in DFT
 *  weight matrix) using scalar operations for single-precision
 *  and double-precision inputs.
 *
 *  @author Srirammaswamy Srinivasan
 */

#include "core/kernels/kernel.h"

static const ops_cycles_t ops_cnt[NUM_PRECISIONS][NUM_FFT_DIRS] =
                                                    {{{0, 2, 12, 16, 0, 0},
                                                      {0, 6, 12, 16, 0, 0}},
                                                     {{0, 2, 12, 16, 0, 0},
                                                      {0, 6, 12, 16, 0, 0}}};

ops_cycles_t get_ops_cnt_r2hcf_rfft4c(UINT8 precision, UINT8 direction)
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

static VOID r2hcf_rfft4c_fp32_fwd(VOID *in_real, VOID *in_imag, VOID *out_real,
                                  VOID *out_imag, INTP n,
                                  aoclfftz_strides_t *strides, UINT8 flag)
{
#ifdef AOCL_ENABLE_LOG
    AOCLFFTZ_LOG_UNFORMATTED(TRACE, TRACE, "Enter");
#endif
    const FLOAT CRTM_4_1 = 0.707106781186547524400844362104849039284835935f;

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
        FLOAT av0, av1, av2, av3;
        FLOAT at0, at1;

        av0 = *in;                  // Input point 1: x(0)
        av1 = in[in_strides[2]];    // Input point 3: x(2)
        av2 = in[in_strides[4]];    // Input point 5: x(4)
        av3 = in[in_strides[6]];    // Input point 7: x(6)

        at0 = av0 + av2;
        at1 = av1 + av3;

        *out = at0 + at1;                   // Output point 1: X(0)
        out[out_strides[3]] = av0 - av2;    // Output point 4: X(3)
        out[out_strides[4]] = av3 - av1;    // Output point 5: X(4)
        out[out_strides[7]] = at0 - at1;    // Output point 8: X(7)

        /* Shifted DFT */
        FLOAT bv0, bv1, bv2, bv3;
        FLOAT bt0, bt1, bt2, bt3;

        bv0 = in[in_strides[1]];    // Input point 2: x(1)
        bv1 = in[in_strides[3]];    // Input point 4: x(3)
        bv2 = in[in_strides[5]];    // Input point 6: x(5)
        bv3 = in[in_strides[7]];    // Input point 8: x(7)

        bt0 = bv1 + bv3;
        bt1 = bv1 - bv3;
        bt2 = CRTM_4_1 * bt0;
        bt3 = CRTM_4_1 * bt1;

        out[out_strides[1]] = bv0 + bt3;    // Output point 2: X(1)
        out[out_strides[2]] = -bv2 - bt2;   // Output point 3: X(2)
        out[out_strides[5]] = bv0 - bt3;    // Output point 6: X(5)
        out[out_strides[6]] = bv2 - bt2;    // Output point 7: X(6)

        in = in + v_in_stride;
        out = out + v_out_stride;
    }
#ifdef AOCL_ENABLE_LOG
    AOCLFFTZ_LOG_UNFORMATTED(TRACE, TRACE, "Exit");
#endif
}

static VOID r2hcf_rfft4c_fp32_bwd(VOID *in_real, VOID *in_imag, VOID *out_real,
                                  VOID *out_imag, INTP n,
                                  aoclfftz_strides_t *strides, UINT8 flag)
{
#ifdef AOCL_ENABLE_LOG
    AOCLFFTZ_LOG_UNFORMATTED(TRACE, TRACE, "Enter");
#endif
    const FLOAT CRTM_4_1 = 1.414213562373095048801688724209698078569671875f;
    const FLOAT CRTM_4_2 = 2.000000000000000000000000000000000000000000000f;

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
        FLOAT av0, av1, av2, av3;
        FLOAT at0, at1, at2, at3;

        av0 = *in;                  // Input point 1: x(0)
        av1 = in[in_strides[3]];    // Input point 4: x(3)
        av2 = in[in_strides[4]];    // Input point 5: x(4)
        av3 = in[in_strides[7]];    // Input point 8: x(7)

        at0 = av0 + av3;
        at1 = CRTM_4_2 * av1;
        at2 = av0 - av3;
        at3 = CRTM_4_2 * av2;

        *out = at0 + at1;                     // Output point 1: X(0)
        out[out_strides[2]] = at2 - at3;      // Output point 3: X(2)
        out[out_strides[4]] = at0 - at1;      // Output point 5: X(4)
        out[out_strides[6]] = at2 + at3;      // Output point 7: X(6)

        /* Shifted DFT */
        FLOAT bv0, bv1, bv2, bv3;
        FLOAT bt0, bt1, bt2, bt3, bt4, bt5;

        bv0 = in[in_strides[1]];    // Input point 2: x(1)
        bv1 = in[in_strides[2]];    // Input point 3: x(2)
        bv2 = in[in_strides[5]];    // Input point 6: x(5)
        bv3 = in[in_strides[6]];    // Input point 7: x(6)

        bt0 = bv0 - bv2;
        bt1 = bv0 + bv2;
        bt2 = bv1 + bv3;
        bt3 = bv3 - bv1;
        bt4 = CRTM_4_1 * bt0;
        bt5 = CRTM_4_1 * bt2;

        out[out_strides[1]] = CRTM_4_2 * bt1; // Output point 2: X(1)
        out[out_strides[3]] = bt4 - bt5;      // Output point 4: X(3)
        out[out_strides[5]] = CRTM_4_2 * bt3; // Output point 6: X(5)
        out[out_strides[7]] = -bt4 - bt5;     // Output point 8: X(7)

        in = in + v_in_stride;
        out = out + v_out_stride;
    }
#ifdef AOCL_ENABLE_LOG
    AOCLFFTZ_LOG_UNFORMATTED(TRACE, TRACE, "Exit");
#endif
}

static VOID r2hcf_rfft4c_fp64_fwd(VOID *in_real, VOID *in_imag, VOID *out_real,
                                  VOID *out_imag, INTP n,
                                  aoclfftz_strides_t *strides, UINT8 flag)
{
#ifdef AOCL_ENABLE_LOG
    AOCLFFTZ_LOG_UNFORMATTED(TRACE, TRACE, "Enter");
#endif
    const DOUBLE CRTM_4_1 = 0.707106781186547524400844362104849039284835935;

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
        DOUBLE av0, av1, av2, av3;
        DOUBLE at0, at1;

        av0 = *in;                  // Input point 1: x(0)
        av1 = in[in_strides[2]];    // Input point 3: x(2)
        av2 = in[in_strides[4]];    // Input point 5: x(4)
        av3 = in[in_strides[6]];    // Input point 7: x(6)

        at0 = av0 + av2;
        at1 = av1 + av3;

        *out = at0 + at1;                   // Output point 1: X(0)
        out[out_strides[3]] = av0 - av2;    // Output point 4: X(3)
        out[out_strides[4]] = av3 - av1;    // Output point 5: X(4)
        out[out_strides[7]] = at0 - at1;    // Output point 8: X(7)

        /* Shifted DFT */
        DOUBLE bv0, bv1, bv2, bv3;
        DOUBLE bt0, bt1, bt2, bt3;

        bv0 = in[in_strides[1]];    // Input point 2: x(1)
        bv1 = in[in_strides[3]];    // Input point 4: x(3)
        bv2 = in[in_strides[5]];    // Input point 6: x(5)
        bv3 = in[in_strides[7]];    // Input point 8: x(7)

        bt0 = bv1 + bv3;
        bt1 = bv1 - bv3;
        bt2 = CRTM_4_1 * bt0;
        bt3 = CRTM_4_1 * bt1;

        out[out_strides[1]] = bv0 + bt3;    // Output point 2: X(1)
        out[out_strides[2]] = -bv2 - bt2;   // Output point 3: X(2)
        out[out_strides[5]] = bv0 - bt3;    // Output point 6: X(5)
        out[out_strides[6]] = bv2 - bt2;    // Output point 7: X(6)


        in = in + v_in_stride;
        out = out + v_out_stride;
    }
#ifdef AOCL_ENABLE_LOG
    AOCLFFTZ_LOG_UNFORMATTED(TRACE, TRACE, "Exit");
#endif
}

static VOID r2hcf_rfft4c_fp64_bwd(VOID *in_real, VOID *in_imag, VOID *out_real,
                                  VOID *out_imag, INTP n,
                                  aoclfftz_strides_t *strides, UINT8 flag)
{
#ifdef AOCL_ENABLE_LOG
    AOCLFFTZ_LOG_UNFORMATTED(TRACE, TRACE, "Enter");
#endif
    const DOUBLE CRTM_4_1 = 1.414213562373095048801688724209698078569671875;
    const DOUBLE CRTM_4_2 = 2.000000000000000000000000000000000000000000000;

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
        DOUBLE av0, av1, av2, av3;
        DOUBLE at0, at1, at2, at3;

        av0 = *in;                  // Input point 1: x(0)
        av1 = in[in_strides[3]];    // Input point 4: x(3)
        av2 = in[in_strides[4]];    // Input point 5: x(4)
        av3 = in[in_strides[7]];    // Input point 8: x(7)

        at0 = av0 + av3;
        at1 = CRTM_4_2 * av1;
        at2 = av0 - av3;
        at3 = CRTM_4_2 * av2;

        *out = at0 + at1;                     // Output point 1: X(0)
        out[out_strides[2]] = at2 - at3;      // Output point 3: X(2)
        out[out_strides[4]] = at0 - at1;      // Output point 5: X(4)
        out[out_strides[6]] = at2 + at3;      // Output point 7: X(6)

        /* Shifted DFT */
        DOUBLE bv0, bv1, bv2, bv3;
        DOUBLE bt0, bt1, bt2, bt3, bt4, bt5;

        bv0 = in[in_strides[1]];    // Input point 2: x(1)
        bv1 = in[in_strides[2]];    // Input point 3: x(2)
        bv2 = in[in_strides[5]];    // Input point 6: x(5)
        bv3 = in[in_strides[6]];    // Input point 7: x(6)

        bt0 = bv0 - bv2;
        bt1 = bv0 + bv2;
        bt2 = bv1 + bv3;
        bt3 = bv3 - bv1;
        bt4 = CRTM_4_1 * bt0;
        bt5 = CRTM_4_1 * bt2;

        out[out_strides[1]] = CRTM_4_2 * bt1; // Output point 2: X(1)
        out[out_strides[3]] = bt4 - bt5;      // Output point 4: X(3)
        out[out_strides[5]] = CRTM_4_2 * bt3; // Output point 6: X(5)
        out[out_strides[7]] = -bt4 - bt5;     // Output point 8: X(7)

        in = in + v_in_stride;
        out = out + v_out_stride;
    }
#ifdef AOCL_ENABLE_LOG
    AOCLFFTZ_LOG_UNFORMATTED(TRACE, TRACE, "Exit");
#endif
}

kfft_ register_kernel_r2hcf_rfft4c(UINT8 precision, UINT8 direction)
{
    if (direction == FORWARD_FFT_DIR)
    {
        if (precision == DT_FLOAT)
        {
            return r2hcf_rfft4c_fp32_fwd;
        }
        else if (precision == DT_DOUBLE)
        {
            return r2hcf_rfft4c_fp64_fwd;
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
            return r2hcf_rfft4c_fp32_bwd;
        }
        else if (precision == DT_DOUBLE)
        {
            return r2hcf_rfft4c_fp64_bwd;
        }
        else
        {
            return NULL;
        }
    }
}
