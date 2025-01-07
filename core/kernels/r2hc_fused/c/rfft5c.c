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

/** @file rfft5c.c
 *
 *  @brief Radix-5 r2hc_fused Real-FFT kernel with scalar operations in C
 *
 *  This file contains the DIT radix-5 real-to-halfcomplex fused of two
 *  different implementations (Standard DFT and Shifted DFT that differs in DFT
 *  weight matrix) using scalar operations for single-precision
 *  and double-precision inputs.
 *
 *  @author Srirammaswamy Srinivasan
 */

#include "core/kernels/kernel.h"

static const ops_cycles_t ops_cnt[NUM_PRECISIONS][NUM_FFT_DIRS] =
                                                    {{{0, 12, 24, 20, 0, 0},
                                                      {0, 14, 24, 20, 0, 0}},
                                                     {{0, 12, 24, 20, 0, 0},
                                                      {0, 14, 24, 20, 0, 0}}};

ops_cycles_t get_ops_cnt_r2hcf_rfft5c(UINT8 precision, UINT8 direction)
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

static VOID r2hcf_rfft5c_fp32_fwd(VOID *in_real, VOID *in_imag, VOID *out_real,
                                  VOID *out_imag, INTP n,
                                  aoclfftz_strides_t *strides, UINT8 flag)
{
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
        /* Standard DFT */
        FLOAT av0, av1, av2, av3, av4;
        FLOAT at0, at1, at2, at3, at4, at5, at6, at7, at8, at9, at10, at11,
              at12;

        av0 = *in;                  // Input point 1: x(0)
        av1 = in[in_strides[2]];    // Input point 3: x(2)
        av2 = in[in_strides[4]];    // Input point 5: x(4)
        av3 = in[in_strides[6]];    // Input point 7: x(6)
        av4 = in[in_strides[8]];    // Input point 9: x(8)

        at0 = av1 + av4;
        at1 = av1 - av4;
        at2 = av2 + av3;
        at3 = av2 - av3;
        at4 = at0 + at2;

        at5 = CRTM_5_4 * at4;
        at6 = at0 - at2;

        at7 = av0 - at5;
        at8 = CRTM_5_1 * at6;

        at9  = CRTM_5_3 * at3;
        at10 = CRTM_5_2 * at1;
        at11 = CRTM_5_2 * at3;
        at12 = CRTM_5_3 * at1;

        *out = av0 + at4;                   // Output point 1: X(0)
        out[out_strides[3]] = at7 + at8;    // Output point 4: X(3)
        out[out_strides[4]] = -at9 - at10;  // Output point 5: X(4)
        out[out_strides[7]] = at7 - at8;    // Output point 8: X(7)
        out[out_strides[8]] = at11 - at12;  // Output point 9: X(8)

        /* Shifted DFT */
        FLOAT bv0, bv1, bv2, bv3, bv4;
        FLOAT bt0, bt1, bt2, bt3, bt4, bt5, bt6, bt7, bt8, bt9, bt10, bt11,
              bt12;

        bv0 = in[in_strides[1]];    // Input point 2: x(1)
        bv1 = in[in_strides[3]];    // Input point 4: x(3)
        bv2 = in[in_strides[5]];    // Input point 6: x(5)
        bv3 = in[in_strides[7]];    // Input point 8: x(7)
        bv4 = in[in_strides[9]];    // Input point 10: x(9)

        bt0 = bv1 + bv4;
        bt1 = bv1 - bv4;
        bt2 = bv2 + bv3;
        bt3 = bv2 - bv3;
        bt4 = bt1 - bt3;

        bt5 = bt1 + bt3;
        bt6 = CRTM_5_4 * bt4;
        bt7 = bv0 + bt6;
        bt8 = CRTM_5_1 * bt5;

        bt9  = CRTM_5_2 * bt2;
        bt10 = CRTM_5_3 * bt0;
        bt11 = CRTM_5_3 * bt2;
        bt12 = CRTM_5_2 * bt0;

        out[out_strides[1]] = bt7 + bt8;    // Output point 2: X(1)
        out[out_strides[2]] = -bt9 - bt10;  // Output point 3: X(2)
        out[out_strides[5]] = bt7 - bt8;    // Output point 6: X(5)
        out[out_strides[6]] = bt11 - bt12;  // Output point 7: X(6)
        out[out_strides[9]] = bv0 - bt4;    // Output point 10: X(9)

        in  = in + v_in_stride;
        out = out + v_out_stride;
    }
}

static VOID r2hcf_rfft5c_fp32_bwd(VOID *in_real, VOID *in_imag, VOID *out_real,
                                  VOID *out_imag, INTP n,
                                  aoclfftz_strides_t *strides, UINT8 flag)
{
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
        /* Standard DFT */
        FLOAT av0, av1, av2, av3, av4;
        FLOAT at0, at1, at2, at3, at4, at5, at6, at7, at8, at9, at10, at11,
              at12, at13;

        av0 = *in;                  // Input point 1: x(0)
        av1 = in[in_strides[3]];    // Input point 4: x(3)
        av2 = in[in_strides[4]];    // Input point 5: x(4)
        av3 = in[in_strides[7]];    // Input point 8: x(7)
        av4 = in[in_strides[8]];    // Input point 9: x(8)

        at0 = av1 + av3;
        at1 = av1 - av3;
        at2 = CRTM_5_4 * at0;
        at3 = CRTM_5_1 * at1;
        at4 = av0 - at2;
        at5 = at4 + at3;
        at6 = at4 - at3;


        at7  = CRTM_5_2 * av4;
        at8  = CRTM_5_3 * av2;
        at9  = CRTM_5_3 * av4;
        at10 = CRTM_5_2 * av2;
        at11 = CRTM_5_5 * at0;

        at12 = at10 + at9;
        at13 = at7 - at8;

        *out = av0 + at11;                  // Output point 1: X(0)
        out[out_strides[2]] = at5 - at12;   // Output point 3: X(2)
        out[out_strides[4]] = at6 + at13;   // Output point 5: X(4)
        out[out_strides[6]] = at6 - at13;   // Output point 7: X(6)
        out[out_strides[8]] = at5 + at12;   // Output point 9: X(8)

        /* Shifted DFT */
        FLOAT bv0, bv1, bv2, bv3, bv4;
        FLOAT bt0, bt1, bt2, bt3, bt4, bt5, bt6, bt7, bt8, bt9, bt10, bt11,
        bt12, bt13;

        bv0 = in[in_strides[1]];    // Input point 2: x(1)
        bv1 = in[in_strides[2]];    // Input point 3: x(2)
        bv2 = in[in_strides[5]];    // Input point 6: x(5)
        bv3 = in[in_strides[6]];    // Input point 7: x(6)
        bv4 = in[in_strides[9]];    // Input point 10: x(9)

        bt0 = bv0 + bv2;
        bt1 = bv2 - bv0;
        bt2 = CRTM_5_1 * bt1;
        bt3 = CRTM_5_4 * bt0;
        bt4 = bv4 - bt3;
        bt5 = bt4 + bt2;
        bt6 = bt4 - bt2;

        bt7  = CRTM_5_3 * bv1;
        bt8  = CRTM_5_2 * bv3;
        bt9  = CRTM_5_2 * bv1;
        bt10 = CRTM_5_3 * bv3;
        bt11 = -bt8 - bt7;

        bt12 = bt9 - bt10;
        bt13 = CRTM_5_5 * bt0;

        out[out_strides[1]] = bv4 + bt13;   // Output point 2: X(1)
        out[out_strides[3]] = bt11 - bt5;  // Output point 4: X(3)
        out[out_strides[5]] = bt6 - bt12;   // Output point 6: X(5)
        out[out_strides[7]] = -bt6 - bt12;  // Output point 8: X(7)
        out[out_strides[9]] = bt11 + bt5;   // Output point 10: X(9)

        in  = in + v_in_stride;
        out = out + v_out_stride;
    }
}

static VOID r2hcf_rfft5c_fp64_fwd(VOID *in_real, VOID *in_imag, VOID *out_real,
                                  VOID *out_imag, INTP n,
                                  aoclfftz_strides_t *strides, UINT8 flag)
{
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
        /* Standard DFT */
        DOUBLE av0, av1, av2, av3, av4;
        DOUBLE at0, at1, at2, at3, at4, at5, at6, at7, at8, at9, at10, at11,
               at12;

        av0 = *in;                  // Input point 1: x(0)
        av1 = in[in_strides[2]];    // Input point 3: x(2)
        av2 = in[in_strides[4]];    // Input point 5: x(4)
        av3 = in[in_strides[6]];    // Input point 7: x(6)
        av4 = in[in_strides[8]];    // Input point 9: x(8)

        at0 = av1 + av4;
        at1 = av1 - av4;
        at2 = av2 + av3;
        at3 = av2 - av3;
        at4 = at0 + at2;

        at5 = CRTM_5_4 * at4;
        at6 = at0 - at2;

        at7 = av0 - at5;
        at8 = CRTM_5_1 * at6;

        at9  = CRTM_5_3 * at3;
        at10 = CRTM_5_2 * at1;
        at11 = CRTM_5_2 * at3;
        at12 = CRTM_5_3 * at1;

        *out = av0 + at4;                   // Output point 1: X(0)
        out[out_strides[3]] = at7 + at8;    // Output point 4: X(3)
        out[out_strides[4]] = -at9 - at10;  // Output point 5: X(4)
        out[out_strides[7]] = at7 - at8;    // Output point 8: X(7)
        out[out_strides[8]] = at11 - at12;  // Output point 9: X(8)

        /* Shifted DFT */
        DOUBLE bv0, bv1, bv2, bv3, bv4;
        DOUBLE bt0, bt1, bt2, bt3, bt4, bt5, bt6, bt7, bt8, bt9, bt10, bt11,
               bt12;

        bv0 = in[in_strides[1]];    // Input point 2: x(1)
        bv1 = in[in_strides[3]];    // Input point 4: x(3)
        bv2 = in[in_strides[5]];    // Input point 6: x(5)
        bv3 = in[in_strides[7]];    // Input point 8: x(7)
        bv4 = in[in_strides[9]];    // Input point 10: x(9)

        bt0 = bv1 + bv4;
        bt1 = bv1 - bv4;
        bt2 = bv2 + bv3;
        bt3 = bv2 - bv3;
        bt4 = bt1 - bt3;

        bt5 = bt1 + bt3;
        bt6 = CRTM_5_4 * bt4;
        bt7 = bv0 + bt6;
        bt8 = CRTM_5_1 * bt5;

        bt9  = CRTM_5_2 * bt2;
        bt10 = CRTM_5_3 * bt0;
        bt11 = CRTM_5_3 * bt2;
        bt12 = CRTM_5_2 * bt0;

        out[out_strides[1]] = bt7 + bt8;    // Output point 2: X(1)
        out[out_strides[2]] = -bt9 - bt10;  // Output point 3: X(2)
        out[out_strides[5]] = bt7 - bt8;    // Output point 6: X(5)
        out[out_strides[6]] = bt11 - bt12;  // Output point 7: X(6)
        out[out_strides[9]] = bv0 - bt4;    // Output point 10: X(9)

        in  = in + v_in_stride;
        out = out + v_out_stride;
    }
}

static VOID r2hcf_rfft5c_fp64_bwd(VOID *in_real, VOID *in_imag, VOID *out_real,
                                  VOID *out_imag, INTP n,
                                  aoclfftz_strides_t *strides, UINT8 flag)
{
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
        /* Standard DFT */
        DOUBLE av0, av1, av2, av3, av4;
        DOUBLE at0, at1, at2, at3, at4, at5, at6, at7, at8, at9, at10, at11,
               at12, at13;

        av0 = *in;                  // Input point 1: x(0)
        av1 = in[in_strides[3]];    // Input point 4: x(3)
        av2 = in[in_strides[4]];    // Input point 5: x(4)
        av3 = in[in_strides[7]];    // Input point 8: x(7)
        av4 = in[in_strides[8]];    // Input point 9: x(8)

        at0 = av1 + av3;
        at1 = av1 - av3;
        at2 = CRTM_5_4 * at0;
        at3 = CRTM_5_1 * at1;
        at4 = av0 - at2;
        at5 = at4 + at3;
        at6 = at4 - at3;

        at7  = CRTM_5_2 * av4;
        at8  = CRTM_5_3 * av2;
        at9  = CRTM_5_3 * av4;
        at10 = CRTM_5_2 * av2;
        at11 = CRTM_5_5 * at0;

        at12 = at10 + at9;
        at13 = at7 - at8;

        *out = av0 + at11;                  // Output point 1: X(0)
        out[out_strides[2]] = at5 - at12;   // Output point 3: X(2)
        out[out_strides[4]] = at6 + at13;   // Output point 5: X(4)
        out[out_strides[6]] = at6 - at13;   // Output point 7: X(6)
        out[out_strides[8]] = at5 + at12;   // Output point 9: X(8)

        /* Shifted DFT */
        DOUBLE bv0, bv1, bv2, bv3, bv4;
        DOUBLE bt0, bt1, bt2, bt3, bt4, bt5, bt6, bt7, bt8, bt9, bt10, bt11,
               bt12, bt13;

        bv0 = in[in_strides[1]];    // Input point 2: x(1)
        bv1 = in[in_strides[2]];    // Input point 3: x(2)
        bv2 = in[in_strides[5]];    // Input point 6: x(5)
        bv3 = in[in_strides[6]];    // Input point 7: x(6)
        bv4 = in[in_strides[9]];    // Input point 10: x(9)

        bt0 = bv0 + bv2;
        bt1 = bv2 - bv0;
        bt2 = CRTM_5_1 * bt1;
        bt3 = CRTM_5_4 * bt0;
        bt4 = bv4 - bt3;
        bt5 = bt4 + bt2;
        bt6 = bt4 - bt2;

        bt7  = CRTM_5_3 * bv1;
        bt8  = CRTM_5_2 * bv3;
        bt9  = CRTM_5_2 * bv1;
        bt10 = CRTM_5_3 * bv3;
        bt11 = -bt8 - bt7;

        bt12 = bt9 - bt10;
        bt13 = CRTM_5_5 * bt0;

        out[out_strides[1]] = bv4 + bt13;   // Output point 2: X(1)
        out[out_strides[3]] = bt11 - bt5;   // Output point 4: X(3)
        out[out_strides[5]] = bt6 - bt12;   // Output point 6: X(5)
        out[out_strides[7]] = -bt6 - bt12;  // Output point 8: X(7)
        out[out_strides[9]] = bt11 + bt5;   // Output point 10: X(9)

        in  = in + v_in_stride;
        out = out + v_out_stride;
    }
}

kfft_ register_kernel_r2hcf_rfft5c(UINT8 precision, UINT8 direction)
{
    if (direction == FORWARD_FFT_DIR)
    {
        if (precision == DT_FLOAT)
        {
            return r2hcf_rfft5c_fp32_fwd;
        }
        else if (precision == DT_DOUBLE)
        {
            return r2hcf_rfft5c_fp64_fwd;
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
            return r2hcf_rfft5c_fp32_bwd;
        }
        else if (precision == DT_DOUBLE)
        {
            return r2hcf_rfft5c_fp64_bwd;
        }
        else
        {
            return NULL;
        }
    }
}
