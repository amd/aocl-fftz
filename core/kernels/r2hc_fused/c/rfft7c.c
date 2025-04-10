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

/** @file rfft7c.c
 *
 *  @brief Radix-7 r2hc_fused Real-FFT kernel with scalar operations in C
 *
 *  This file contains the DIT radix-7 real-to-halfcomplex fused of two
 *  different implementations (Standard DFT and Shifted DFT that differs in DFT
 *  weight matrix) using scalar operations for single-precision
 *  and double-precision inputs.
 *
 *  @author Dr. Pritam Giri
 */

#include "core/kernels/kernel.h"

static const ops_cycles_t ops_cnt[NUM_PRECISIONS][NUM_FFT_DIRS] =
                                                    {{{0, 36, 48, 28, 0, 0},
                                                      {0, 38, 48, 28, 0, 0}},
                                                     {{0, 36, 48, 28, 0, 0},
                                                      {0, 38, 48, 28, 0, 0}}};

ops_cycles_t get_ops_cnt_r2hcf_rfft7c(UINT8 precision, UINT8 direction)
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

static VOID r2hcf_rfft7c_fp32_fwd(VOID *in_real, VOID *in_imag, VOID *out_real,
                              VOID *out_imag, INTP n,
                              aoclfftz_strides_t *strides, UINT8 flag)
{
#ifdef AOCL_ENABLE_LOG
    AOCLFFTZ_LOG_UNFORMATTED(TRACE, TRACE, "Enter");
#endif
    const FLOAT CRTM_7_1 = 0.900968867902419126236102319507445051165919162f;
    const FLOAT CRTM_7_2 = 0.433883739117558120475768332848358754609990728f;
    const FLOAT CRTM_7_3 = 0.623489801858733530525004884004239810632274731f;
    const FLOAT CRTM_7_4 = 0.781831482468029808708444526674057750232334519f;
    const FLOAT CRTM_7_5 = 0.222520933956314404288902564496794759466355569f;
    const FLOAT CRTM_7_6 = 0.974927912181823607018131682993931217232785801f;

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
        FLOAT av0, av1, av2, av3, av4, av5, av6;
        FLOAT at0, at1, at2, at3, at4, at5, at6, at7, at8, at9,
              at10, at11, at12, at13, at14, at15, at16, at17, at18, at19,
              at20, at21, at22, at23, at24, at25, at26, at27, at28, at29,
              at30, at31, at32, at33, at34;

        av0 = *in;                  // Input point 1: x(0)
        av1 = in[in_strides[2]];    // Input point 3: x(2)
        av2 = in[in_strides[4]];    // Input point 5: x(4)
        av3 = in[in_strides[6]];    // Input point 7: x(6)
        av4 = in[in_strides[8]];    // Input point 9: x(8)
        av5 = in[in_strides[10]];   // Input point 11: x(10)
        av6 = in[in_strides[12]];   // Input point 13: x(12)

        at0 = av6 + av1;
        at1 = av6 - av1;
        at2 = av5 + av2;
        at3 = av5 - av2;
        at4 = av4 + av3;
        at5 = av4 - av3;
        at6 = av0 + at0;
        at7 = at2 + at4;

        at8 = CRTM_7_1 * at4;
        at9 = CRTM_7_3 * at0;
        at10 = CRTM_7_5 * at2;
        at11 = CRTM_7_2 * at5;
        at12 = CRTM_7_4 * at1;
        at13 = av0 - at8;
        at14 = at9 - at10;

        at15 = CRTM_7_6 * at3;
        at16 = at11 + at12;
        at17 = CRTM_7_1 * at2;
        at18 = CRTM_7_3 * at4;
        at19 = CRTM_7_5 * at0;

        at20 = av0 - at17;
        at21 = at18 - at19;
        at22 = CRTM_7_2 * at3;
        at23 = CRTM_7_4 * at5;

        at24 = CRTM_7_6 * at1;
        at25 = CRTM_7_1 * at0;
        at26 = CRTM_7_3 * at2;
        at27 = CRTM_7_5 * at4;
        at28 = at22 + at23;

        at29 = av0 - at25;
        at30 = at26 - at27;
        at31 = CRTM_7_2 * at1;
        at32 = CRTM_7_4 * at3;
        at33 = CRTM_7_6 * at5;
        at34 = at31 - at32;

        *out = at6 + at7;                       // Output pt 1: X(0)
        out[out_strides[3]]  = at13 + at14;     // Output pt 4: X(3)
        out[out_strides[4]]  = at15 + at16;     // Output pt 5: X(4)
        out[out_strides[7]]  = at20 + at21;     // Output pt 8: X(7)
        out[out_strides[8]]  = at24 - at28;     // Output pt 9: X(8)
        out[out_strides[11]] = at29 + at30;     // Output pt 12: X(11)
        out[out_strides[12]] = at34 + at33;     // Output pt 13: X(12)

        /* Shifted DFT */
        FLOAT bv0, bv1, bv2, bv3, bv4, bv5, bv6;
        FLOAT bt0, bt1, bt2, bt3, bt4, bt5, bt6, bt7, bt8, bt9,
              bt10, bt11, bt12, bt13, bt14, bt15, bt16, bt17, bt18, bt19,
              bt20, bt21, bt22, bt23, bt24, bt25, bt26, bt27, bt28, bt29,
              bt30, bt31, bt32, bt33, bt34;

        bv0 = in[in_strides[1]];    // Input point 2: x(1)
        bv1 = in[in_strides[3]];    // Input point 4: x(3)
        bv2 = in[in_strides[5]];    // Input point 6: x(5)
        bv3 = in[in_strides[7]];    // Input point 8: x(7)
        bv4 = in[in_strides[9]];    // Input point 10: x(9)
        bv5 = in[in_strides[11]];   // Input point 12: x(11)
        bv6 = in[in_strides[13]];   // Input point 14: x(13)

        bt0 = bv6 + bv1;
        bt1 = bv6 - bv1;
        bt2 = bv5 + bv2;
        bt3 = bv5 - bv2;
        bt4 = bv4 + bv3;
        bt5 = bv4 - bv3;

        bt6 = CRTM_7_1 * bt1;
        bt7 = CRTM_7_3 * bt3;

        bt8  = CRTM_7_5 * bt5;
        bt9  = bv0 - bt6;
        bt10 = bt7 + bt8;
        bt11 = CRTM_7_2 * bt0;
        bt12 = CRTM_7_4 * bt2;
        bt13 = CRTM_7_6 * bt4;
        bt14 = -bt11 - bt12;

        bt15 = CRTM_7_1 * bt3;
        bt16 = CRTM_7_3 * bt5;

        bt17 = CRTM_7_5 * bt1;
        bt18 = bv0 + bt15;
        bt20 = CRTM_7_2 * bt2;
        bt19 = bt16 - bt17;
        bt21 = CRTM_7_4 * bt4;
        bt22 = CRTM_7_6 * bt0;
        bt23 = bt21 - bt20;

        bt24 = CRTM_7_1 * bt5;
        bt25 = CRTM_7_3 * bt1;
        bt26 = CRTM_7_5 * bt3;
        bt27 = bv0 - bt24;
        bt28 = bt25 + bt26;

        bt29 = CRTM_7_2 * bt4;
        bt30 = CRTM_7_4 * bt0;
        bt31 = CRTM_7_6 * bt2;
        bt32 = bt29 + bt30;
        bt33 = bv0 + bt1;
        bt34 = bt5 - bt3;

        out[out_strides[1]]  = bt9 - bt10;     // Output pt 2: X(1)
        out[out_strides[2]]  = bt14 - bt13;    // Output pt 3: X(2)
        out[out_strides[5]]  = bt18 + bt19;    // Output pt 6: X(5)
        out[out_strides[6]]  = bt23 - bt22;    // Output pt 7: X(6)
        out[out_strides[9]]  = bt27 + bt28;    // Output pt 10: X(9)
        out[out_strides[10]] = bt31 - bt32;    // Output pt 11: X(10)
        out[out_strides[13]] = bt33 + bt34;    // Output pt 14: X(13)

        in = in + v_in_stride;
        out = out + v_out_stride;
    }
#ifdef AOCL_ENABLE_LOG
    AOCLFFTZ_LOG_UNFORMATTED(TRACE, TRACE, "Exit");
#endif
}

static VOID r2hcf_rfft7c_fp32_bwd(VOID *in_real, VOID *in_imag, VOID *out_real,
                              VOID *out_imag, INTP n,
                              aoclfftz_strides_t *strides, UINT8 flag)
{
#ifdef AOCL_ENABLE_LOG
    AOCLFFTZ_LOG_UNFORMATTED(TRACE, TRACE, "Enter");
#endif
    const FLOAT CRTM_7_1 = 1.801937735804838252472204639014890102331838324f;
    const FLOAT CRTM_7_2 = 0.867767478235116240951536665696717509219981456f;
    const FLOAT CRTM_7_3 = 1.246979603717467061050009768008479621264549462f;
    const FLOAT CRTM_7_4 = 1.563662964936059617416889053348115500464669038f;
    const FLOAT CRTM_7_5 = 0.445041867912628808577805128993589518932711138f;
    const FLOAT CRTM_7_6 = 1.949855824363647214036263365987862434465571602f;
    const FLOAT CRTM_7_7 = 2.000000000000000000000000000000000000000000000f;

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
        FLOAT av0, av1, av2, av3, av4, av5, av6;
        FLOAT at0, at1, at2, at3, at4, at5, at6, at7, at8, at9,
              at10, at11, at12, at13, at14, at15, at16, at17, at18, at19,
              at20, at21, at22, at23, at24, at25, at26, at27, at28, at29,
              at30, at31, at32, at33, at34, at35;

        av0 = *in;                  // Input point 1: x(0)
        av1 = in[in_strides[3]];    // Input point 4: x(3)
        av2 = in[in_strides[4]];    // Input point 5: x(4)
        av3 = in[in_strides[7]];    // Input point 8: x(7)
        av4 = in[in_strides[8]];    // Input point 9: x(8)
        av5 = in[in_strides[11]];   // Input point 12: x(11)
        av6 = in[in_strides[12]];   // Input point 13: x(12)

        at0 = CRTM_7_1 * av5;
        at1 = CRTM_7_3 * av1;
        at2 = CRTM_7_5 * av3;
        at3 = av0 - at0;
        at4 = at1 - at2;

        at5 = CRTM_7_2 * av6;
        at6 = CRTM_7_4 * av2;
        at7 = CRTM_7_6 * av4;
        at8 = at5 + at6;
        at9 = at3 + at4;
        at10 = at8 + at7;

        at11 = CRTM_7_1 * av3;
        at12 = CRTM_7_3 * av5;
        at13 = CRTM_7_5 * av1;
        at14 = av0 - at11;
        at15 = at12 - at13;

        at16 = CRTM_7_2 * av4;
        at17 = CRTM_7_4 * av6;
        at18 = CRTM_7_6 * av2;
        at19 = at16 + at17;

        at20 = at14 + at15;
        at21 = at18 - at19;

        at22 = CRTM_7_1 * av1;
        at23 = CRTM_7_3 * av3;
        at24 = CRTM_7_5 * av5;
        at25 = av0 - at22;
        at26 = at23 - at24;

        at27 = CRTM_7_2 * av2;
        at28 = CRTM_7_4 * av4;
        at29 = CRTM_7_6 * av6;
        at30 = at27 - at28;

        at31 = av1 + av3;
        at32 = at25 + at26;
        at33 = at31 + av5;
        at34 = at30 + at29;
        at35 = CRTM_7_7 * at33;

        *out = av0 + at35;                     // Output pt 1: X(0)
        out[out_strides[2]]  = at9 - at10;     // Output pt 3: X(2)
        out[out_strides[4]]  = at20 - at21;    // Output pt 5: X(4)
        out[out_strides[6]]  = at32 - at34;    // Output pt 7: X(6)
        out[out_strides[8]]  = at32 + at34;    // Output pt 9: X(8)
        out[out_strides[10]] = at20 + at21;    // Output pt 11: X(10)
        out[out_strides[12]] = at9 + at10;     // Output pt 13: X(12)

        /* Shifted DFT */
        FLOAT bv0, bv1, bv2, bv3, bv4, bv5, bv6;
        FLOAT bt0, bt1, bt2, bt3, bt4, bt5, bt6, bt7, bt8, bt9,
              bt10, bt11, bt12, bt13, bt14, bt15, bt16, bt17, bt18, bt19,
              bt20, bt21, bt22, bt23, bt24, bt25, bt26, bt27, bt28, bt29,
              bt30, bt31, bt32, bt33, bt34, bt35;

        bv0 = in[in_strides[1]];    // Input point 2: x(1)
        bv1 = in[in_strides[2]];    // Input point 3: x(2)
        bv2 = in[in_strides[5]];    // Input point 6: x(5)
        bv3 = in[in_strides[6]];    // Input point 7: x(6)
        bv4 = in[in_strides[9]];    // Input point 10: x(9)
        bv5 = in[in_strides[10]];   // Input point 11: x(10)
        bv6 = in[in_strides[13]];   // Input point 14: x(13)

        bt0 = CRTM_7_1 * bv0;
        bt1 = CRTM_7_3 * bv4;
        bt2 = CRTM_7_5 * bv2;
        bt3 = bv6 - bt0;
        bt4 = bt1 - bt2;
        bt5 = bt3 + bt4;

        bt6 = CRTM_7_2 * bv1;
        bt7 = CRTM_7_4 * bv5;
        bt8 = CRTM_7_6 * bv3;
        bt9 = bt6 + bt7;

        bt10 = bt9 + bt8;
        bt11 = CRTM_7_1 * bv2;
        bt12 = CRTM_7_3 * bv0;
        bt13 = CRTM_7_5 * bv4;
        bt14 = bv6 - bt11;
        bt15 = bt12 - bt13;

        bt16 = CRTM_7_2 * bv3;
        bt17 = CRTM_7_4 * bv1;
        bt18 = CRTM_7_6 * bv5;
        bt19 = bt16 + bt17;

        bt20 = bt14 + bt15;
        bt21 = bt18 - bt19;

        bt22 = CRTM_7_1 * bv4;
        bt23 = CRTM_7_3 * bv2;
        bt24 = CRTM_7_5 * bv0;
        bt25 = bv6 - bt22;
        bt26 = bt23 - bt24;

        bt27 = CRTM_7_2 * bv5;
        bt28 = CRTM_7_4 * bv3;
        bt29 = CRTM_7_6 * bv1;
        bt30 = bt27 - bt28;

        bt31 = bv0 + bv2;
        bt32 = bt25 + bt26;
        bt33 = bt31 + bv4;
        bt34 = bt30 + bt29;
        bt35 = bt33 * CRTM_7_7;

        out[out_strides[1]]  = bv6 + bt35;     // Output pt 2: X(1)
        out[out_strides[3]]  = -bt5 - bt10;    // Output pt 4: X(3)
        out[out_strides[5]]  = bt20 + bt21;    // Output pt 6: X(5)
        out[out_strides[7]]  = -bt32 - bt34;   // Output pt 8: X(7)
        out[out_strides[9]]  = bt32 - bt34;    // Output pt 10: X(9)
        out[out_strides[11]] = bt21 - bt20;    // Output pt 12: X(11)
        out[out_strides[13]] = bt5 - bt10;     // Output pt 14: X(13)

        in  = in + v_in_stride;
        out = out + v_out_stride;
    }
#ifdef AOCL_ENABLE_LOG
    AOCLFFTZ_LOG_UNFORMATTED(TRACE, TRACE, "Exit");
#endif
}

static VOID r2hcf_rfft7c_fp64_fwd(VOID *in_real, VOID *in_imag, VOID *out_real,
                              VOID *out_imag, INTP n,
                              aoclfftz_strides_t *strides, UINT8 flag)
{
#ifdef AOCL_ENABLE_LOG
    AOCLFFTZ_LOG_UNFORMATTED(TRACE, TRACE, "Enter");
#endif
    const DOUBLE CRTM_7_1 = 0.900968867902419126236102319507445051165919162;
    const DOUBLE CRTM_7_2 = 0.433883739117558120475768332848358754609990728;
    const DOUBLE CRTM_7_3 = 0.623489801858733530525004884004239810632274731;
    const DOUBLE CRTM_7_4 = 0.781831482468029808708444526674057750232334519;
    const DOUBLE CRTM_7_5 = 0.222520933956314404288902564496794759466355569;
    const DOUBLE CRTM_7_6 = 0.974927912181823607018131682993931217232785801;

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
        DOUBLE av0, av1, av2, av3, av4, av5, av6;
        DOUBLE at0, at1, at2, at3, at4, at5, at6, at7, at8, at9,
              at10, at11, at12, at13, at14, at15, at16, at17, at18, at19,
              at20, at21, at22, at23, at24, at25, at26, at27, at28, at29,
              at30, at31, at32, at33, at34;

        av0 = *in;                  // Input point 1: x(0)
        av1 = in[in_strides[2]];    // Input point 3: x(2)
        av2 = in[in_strides[4]];    // Input point 5: x(4)
        av3 = in[in_strides[6]];    // Input point 7: x(6)
        av4 = in[in_strides[8]];    // Input point 9: x(8)
        av5 = in[in_strides[10]];   // Input point 11: x(10)
        av6 = in[in_strides[12]];   // Input point 13: x(12)

        at0 = av6 + av1;
        at1 = av6 - av1;
        at2 = av5 + av2;
        at3 = av5 - av2;
        at4 = av4 + av3;
        at5 = av4 - av3;
        at6 = av0 + at0;
        at7 = at2 + at4;

        at8 = CRTM_7_1 * at4;
        at9 = CRTM_7_3 * at0;
        at10 = CRTM_7_5 * at2;
        at11 = CRTM_7_2 * at5;
        at12 = CRTM_7_4 * at1;
        at13 = av0 - at8;
        at14 = at9 - at10;

        at15 = CRTM_7_6 * at3;
        at16 = at11 + at12;
        at17 = CRTM_7_1 * at2;
        at18 = CRTM_7_3 * at4;
        at19 = CRTM_7_5 * at0;

        at20 = av0 - at17;
        at21 = at18 - at19;
        at22 = CRTM_7_2 * at3;
        at23 = CRTM_7_4 * at5;

        at24 = CRTM_7_6 * at1;
        at25 = at22 + at23;
        at26 = CRTM_7_1 * at0;
        at27 = CRTM_7_3 * at2;
        at28 = CRTM_7_5 * at4;

        at29 = av0 - at26;
        at30 = at27 - at28;
        at31 = CRTM_7_2 * at1;
        at32 = CRTM_7_4 * at3;
        at33 = CRTM_7_6 * at5;
        at34 = at31 - at32;

        *out = at6 + at7;                       // Output pt 1: X(0)
        out[out_strides[3]]  = at13 + at14;     // Output pt 4: X(3)
        out[out_strides[4]]  = at15 + at16;     // Output pt 5: X(4)
        out[out_strides[7]]  = at20 + at21;     // Output pt 8: X(7)
        out[out_strides[8]]  = at24 - at25;     // Output pt 9: X(8)
        out[out_strides[11]] = at29 + at30;     // Output pt 12: X(11)
        out[out_strides[12]] = at34 + at33;     // Output pt 13: X(12)

        /* Shifted DFT */
        DOUBLE bv0, bv1, bv2, bv3, bv4, bv5, bv6;
        DOUBLE bt0, bt1, bt2, bt3, bt4, bt5, bt6, bt7, bt8, bt9,
              bt10, bt11, bt12, bt13, bt14, bt15, bt16, bt17, bt18, bt19,
              bt20, bt21, bt22, bt23, bt24, bt25, bt26, bt27, bt28, bt29,
              bt30, bt31, bt32, bt33, bt34;

        bv0 = in[in_strides[1]];    // Input point 2: x(1)
        bv1 = in[in_strides[3]];    // Input point 4: x(3)
        bv2 = in[in_strides[5]];    // Input point 6: x(5)
        bv3 = in[in_strides[7]];    // Input point 8: x(7)
        bv4 = in[in_strides[9]];    // Input point 10: x(9)
        bv5 = in[in_strides[11]];   // Input point 12: x(11)
        bv6 = in[in_strides[13]];   // Input point 14: x(13)

        bt0 = bv6 + bv1;
        bt1 = bv6 - bv1;
        bt2 = bv5 + bv2;
        bt3 = bv5 - bv2;
        bt4 = bv4 + bv3;
        bt5 = bv4 - bv3;

        bt6 = CRTM_7_1 * bt1;
        bt7 = CRTM_7_3 * bt3;

        bt8  = CRTM_7_5 * bt5;
        bt9  = bv0 - bt6;
        bt10 = bt7 + bt8;
        bt11 = CRTM_7_2 * bt0;
        bt12 = CRTM_7_4 * bt2;
        bt13 = CRTM_7_6 * bt4;
        bt14 = -bt11 - bt12;

        bt15 = CRTM_7_1 * bt3;
        bt16 = CRTM_7_3 * bt5;
        bt17 = CRTM_7_5 * bt1;
        bt18 = bv0 + bt15;
        bt19 = bt16 - bt17;
        bt20 = CRTM_7_2 * bt2;
        bt21 = CRTM_7_4 * bt4;
        bt22 = CRTM_7_6 * bt0;
        bt23 = bt21 - bt20;

        bt24 = CRTM_7_1 * bt5;
        bt25 = CRTM_7_3 * bt1;
        bt26 = CRTM_7_5 * bt3;
        bt27 = bv0 - bt24;
        bt28 = bt25 + bt26;

        bt29 = CRTM_7_2 * bt4;
        bt30 = CRTM_7_4 * bt0;
        bt31 = CRTM_7_6 * bt2;
        bt32 = bt29 + bt30;
        bt33 = bv0 + bt1;
        bt34 = bt5 - bt3;

        out[out_strides[1]]  = bt9 - bt10;     // Output pt 2: X(1)
        out[out_strides[2]]  = bt14 - bt13;    // Output pt 3: X(2)
        out[out_strides[5]]  = bt18 + bt19;    // Output pt 6: X(5)
        out[out_strides[6]]  = bt23 - bt22;    // Output pt 7: X(6)
        out[out_strides[9]]  = bt27 + bt28;    // Output pt 10: X(9)
        out[out_strides[10]] = bt31 - bt32;    // Output pt 11: X(10)
        out[out_strides[13]] = bt33 + bt34;    // Output pt 14: X(13)

        in = in + v_in_stride;
        out = out + v_out_stride;
    }
#ifdef AOCL_ENABLE_LOG
    AOCLFFTZ_LOG_UNFORMATTED(TRACE, TRACE, "Exit");
#endif
}

static VOID r2hcf_rfft7c_fp64_bwd(VOID *in_real, VOID *in_imag, VOID *out_real,
                              VOID *out_imag, INTP n,
                              aoclfftz_strides_t *strides, UINT8 flag)
{
#ifdef AOCL_ENABLE_LOG
    AOCLFFTZ_LOG_UNFORMATTED(TRACE, TRACE, "Enter");
#endif
    const DOUBLE CRTM_7_1 = 1.801937735804838252472204639014890102331838324;
    const DOUBLE CRTM_7_2 = 0.867767478235116240951536665696717509219981456;
    const DOUBLE CRTM_7_3 = 1.246979603717467061050009768008479621264549462;
    const DOUBLE CRTM_7_4 = 1.563662964936059617416889053348115500464669038;
    const DOUBLE CRTM_7_5 = 0.445041867912628808577805128993589518932711138;
    const DOUBLE CRTM_7_6 = 1.949855824363647214036263365987862434465571602;
    const DOUBLE CRTM_7_7 = 2.000000000000000000000000000000000000000000000;

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
        DOUBLE av0, av1, av2, av3, av4, av5, av6;
        DOUBLE at0, at1, at2, at3, at4, at5, at6, at7, at8, at9,
              at10, at11, at12, at13, at14, at15, at16, at17, at18, at19,
              at20, at21, at22, at23, at24, at25, at26, at27, at28, at29,
              at30, at31, at32, at33, at34, at35;

        av0 = *in;                  // Input point 1: x(0)
        av1 = in[in_strides[3]];    // Input point 4: x(3)
        av2 = in[in_strides[4]];    // Input point 5: x(4)
        av3 = in[in_strides[7]];    // Input point 8: x(7)
        av4 = in[in_strides[8]];    // Input point 9: x(8)
        av5 = in[in_strides[11]];   // Input point 12: x(11)
        av6 = in[in_strides[12]];   // Input point 13: x(12)

        at0 = CRTM_7_1 * av5;
        at1 = CRTM_7_3 * av1;
        at2 = CRTM_7_5 * av3;
        at3 = av0 - at0;
        at4 = at1 - at2;

        at5 = CRTM_7_2 * av6;
        at6 = CRTM_7_4 * av2;
        at7 = CRTM_7_6 * av4;
        at8 = at5 + at6;
        at9 = at3 + at4;
        at10 = at8 + at7;

        at11 = CRTM_7_1 * av3;
        at12 = CRTM_7_3 * av5;
        at13 = CRTM_7_5 * av1;
        at14 = av0 - at11;
        at15 = at12 - at13;

        at16 = CRTM_7_2 * av4;
        at17 = CRTM_7_4 * av6;
        at18 = CRTM_7_6 * av2;
        at19 = at16 + at17;

        at20 = at14 + at15;
        at21 = at18 - at19;

        at22 = CRTM_7_1 * av1;
        at23 = CRTM_7_3 * av3;
        at24 = CRTM_7_5 * av5;
        at25 = av0 - at22;
        at26 = at23 - at24;

        at27 = CRTM_7_2 * av2;
        at28 = CRTM_7_4 * av4;
        at29 = CRTM_7_6 * av6;
        at30 = at27 - at28;

        at31 = av1 + av3;
        at32 = at25 + at26;
        at33 = at31 + av5;
        at34 = at30 + at29;
        at35 = CRTM_7_7 * at33;

        *out = av0 + at35;                     // Output pt 1: X(0)
        out[out_strides[2]]  = at9 - at10;     // Output pt 3: X(2)
        out[out_strides[4]]  = at20 - at21;    // Output pt 5: X(4)
        out[out_strides[6]]  = at32 - at34;    // Output pt 7: X(6)
        out[out_strides[8]]  = at32 + at34;    // Output pt 9: X(8)
        out[out_strides[10]] = at20 + at21;    // Output pt 11: X(10)
        out[out_strides[12]] = at9 + at10;     // Output pt 13: X(12)

        /* Shifted DFT */
        DOUBLE bv0, bv1, bv2, bv3, bv4, bv5, bv6;
        DOUBLE bt0, bt1, bt2, bt3, bt4, bt5, bt6, bt7, bt8, bt9,
              bt10, bt11, bt12, bt13, bt14, bt15, bt16, bt17, bt18, bt19,
              bt20, bt21, bt22, bt23, bt24, bt25, bt26, bt27, bt28, bt29,
              bt30, bt31, bt32, bt33, bt34, bt35;

        bv0 = in[in_strides[1]];    // Input point 2: x(1)
        bv1 = in[in_strides[2]];    // Input point 3: x(2)
        bv2 = in[in_strides[5]];    // Input point 6: x(5)
        bv3 = in[in_strides[6]];    // Input point 7: x(6)
        bv4 = in[in_strides[9]];    // Input point 10: x(9)
        bv5 = in[in_strides[10]];   // Input point 11: x(10)
        bv6 = in[in_strides[13]];   // Input point 14: x(13)

        bt0 = CRTM_7_1 * bv0;
        bt1 = CRTM_7_3 * bv4;
        bt2 = CRTM_7_5 * bv2;
        bt3 = bv6 - bt0;
        bt4 = bt1 - bt2;
        bt5 = bt3 + bt4;

        bt6 = CRTM_7_2 * bv1;
        bt7 = CRTM_7_4 * bv5;
        bt8 = CRTM_7_6 * bv3;
        bt9 = bt6 + bt7;

        bt10 = bt9 + bt8;
        bt11 = CRTM_7_1 * bv2;
        bt12 = CRTM_7_3 * bv0;
        bt13 = CRTM_7_5 * bv4;
        bt14 = bv6 - bt11;
        bt15 = bt12 - bt13;

        bt16 = CRTM_7_2 * bv3;
        bt17 = CRTM_7_4 * bv1;
        bt18 = CRTM_7_6 * bv5;
        bt19 = bt16 + bt17;

        bt20 = bt14 + bt15;
        bt21 = bt18 - bt19;

        bt22 = CRTM_7_1 * bv4;
        bt23 = CRTM_7_3 * bv2;
        bt24 = CRTM_7_5 * bv0;
        bt25 = bv6 - bt22;
        bt26 = bt23 - bt24;

        bt27 = CRTM_7_2 * bv5;
        bt28 = CRTM_7_4 * bv3;
        bt29 = CRTM_7_6 * bv1;
        bt30 = bt27 - bt28;

        bt31 = bv0 + bv2;
        bt32 = bt25 + bt26;
        bt33 = bt31 + bv4;
        bt34 = bt30 + bt29;
        bt35 = bt33 * CRTM_7_7;

        out[out_strides[1]]  = bv6 + bt35;     // Output pt 2: X(1)
        out[out_strides[3]]  = -bt5 - bt10;    // Output pt 4: X(3)
        out[out_strides[5]]  = bt20 + bt21;    // Output pt 6: X(5)
        out[out_strides[7]]  = -bt32 - bt34;   // Output pt 8: X(7)
        out[out_strides[9]]  = bt32 - bt34;    // Output pt 10: X(9)
        out[out_strides[11]] = bt21 - bt20;    // Output pt 12: X(11)
        out[out_strides[13]] = bt5 - bt10;     // Output pt 14: X(13)

        in = in + v_in_stride;
        out = out + v_out_stride;
    }
#ifdef AOCL_ENABLE_LOG
    AOCLFFTZ_LOG_UNFORMATTED(TRACE, TRACE, "Exit");
#endif
}

kfft_ register_kernel_r2hcf_rfft7c(UINT8 precision, UINT8 direction)
{
    if (direction == FORWARD_FFT_DIR)
    {
        if (precision == DT_FLOAT)
        {
            return r2hcf_rfft7c_fp32_fwd;
        }
        else if (precision == DT_DOUBLE)
        {
            return r2hcf_rfft7c_fp64_fwd;
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
            return r2hcf_rfft7c_fp32_bwd;
        }
        else if (precision == DT_DOUBLE)
        {
            return r2hcf_rfft7c_fp64_bwd;
        }
        else
        {
            return NULL;
        }
    }
}
