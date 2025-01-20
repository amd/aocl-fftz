/**
 * Copyright (C) 2023, Advanced Micro Devices. All rights reserved.
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

/** @file fft15c.c
 *
 *  @brief Radix-15 FFT kernel with scalar operations in C
 *
 *  This file contains the DIT radix-15 FFT implementations using scalar
 *  operations for single-precision and double-precision inputs.
 *
 * @author S. Biplab Raut
 * @author Varun Sanjay
 * @author Jeya R
 *
 */
#include "core/kernels/kernel.h"

#ifdef USE_OPT_KERNEL_VARIANT
/* --------------- optimized C kernel variant --------------- */
static const ops_cycles_t ops_cnt[NUM_PRECISIONS] = {{0, 50, 156, 60, 0, 0},
                                                     {0, 50, 156, 60, 0, 0}};

ops_cycles_t get_ops_cnt_fft15c(UINT8 precision, UINT8 direction)
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

static VOID fft15c_fp64(VOID *in_real, VOID *in_imag, VOID *out_real,
                        VOID *out_imag, INTP n, aoclfftz_strides_t *strides,
                        UINT8 flag)
{
#ifdef AOCL_ENABLE_LOG
    AOCLFFTZ_LOG_UNFORMATTED(TRACE, TRACE, "Enter");
#endif
    const DOUBLE CRTM_15_1 =
        +0.55901699437494742410229341718281905886015458990288;
    const DOUBLE CRTM_15_2 =
        +0.25000000000000000000000000000000000000000000000000;
    const DOUBLE CRTM_15_3 =
        +0.95105651629515357211643933337938214340569863400000;
    const DOUBLE CRTM_15_4 =
        +0.58778525229247301629891039327884007596190389052978;
    const DOUBLE CRTM_15_5 =
        +0.50000000000000000000000000000000000000000000000000;
    const DOUBLE CRTM_15_6 =
        +0.86602540378443864676372317075293618347140262690519;
    // Below CRTMs are the product of the above CRTMs, Precomputed to save
    // multiplications on the fly.
    // CRTM_15_7 = CRTM_15_6 * CRTM_15_4
    const DOUBLE CRTM_15_7 =
        +0.50903696045512706468216979248996715975105181034577;
    // CRTM_15_8 = CRTM_15_6 * CRTM_15_3
    const DOUBLE CRTM_15_8 =
        +0.82363910354633184270744116161596601637855195182647;
    // CRTM_15_9 = CRTM_15_6 * CRTM_15_1
    const DOUBLE CRTM_15_9 =
        +0.48412291827592710612024388657479988457787393064252;
    // CRTM_15_10 = CRTM_15_6 * CRTM_15_2
    const DOUBLE CRTM_15_10 =
        +0.21650635094610964914707551542960572987794876098633;

    DOUBLE *in_r = (DOUBLE *)in_real;
    DOUBLE *in_i = (DOUBLE *)in_imag;
    DOUBLE *out_r = (DOUBLE *)out_real;
    DOUBLE *out_i = (DOUBLE *)out_imag;
#ifdef VOLATILE_STRIDE_ARRAY
    volatile INTP *in_strides = strides->in_strides;
    volatile INTP *out_strides = strides->out_strides;
#else
    INTP *in_strides = strides->in_strides;
    INTP *out_strides = strides->out_strides;
#endif
    INTP v_in_stride = strides->v_in_stride;
    INTP v_out_stride = strides->v_out_stride;
    INTP cnt;

    for (cnt = 0; cnt < n; cnt++)
    {
        DOUBLE v1r, v1i, v2r, v2i, v3r, v3i, av1, av2, av3, av4, av5, av6, av7,
               av8, av9, av10, av11, av12, av13, av14, av15, av16, av17, av18,
               av19, tv1, tv2, tv3, tv4, tv5, tv6, tv7, tv8, tv9, tv10, tv11,
               tv12, cv1r, cv1i, cv1, cv2, cv3, cv4, cv5, cv6, cv7, cv8, cv9,
               cv10, cv11, cv12, cv13, cv14, cv15, cv16;

        v1r = in_r[in_strides[2]];
        v2r = in_r[in_strides[7]];
        v3r = in_r[in_strides[12]];

        av1 = v1r + v2r;
        av2 = v1r - v2r;
        av3 = v3r + av1;
        av4 = v3r - (CRTM_15_5 * av1);

        v1i = in_i[in_strides[2]];
        v2i = in_i[in_strides[7]];
        v3i = in_i[in_strides[12]];

        av1 = v1i + v2i;
        av5 = v1i - v2i;
        av6 = v3i + av1;
        av7 = v3i - (CRTM_15_5 * av1);

        v1r = in_r[in_strides[8]];
        v2r = in_r[in_strides[13]];
        v3r = in_r[in_strides[3]];

        av1 = v1r + v2r;
        av8 = v2r - v1r;
        av9 = v3r + av1;
        av10 = v3r - (CRTM_15_5 * av1);

        v1i = in_i[in_strides[8]];
        v2i = in_i[in_strides[13]];
        v3i = in_i[in_strides[3]];

        av1 = v1i + v2i;
        av11 = v2i - v1i;
        av12 = v3i + av1;
        av13 = v3i - (CRTM_15_5 * av1);

        cv1 = av10 + av4;
        cv2 = av10 - av4;
        cv3 = av13 + av7;
        cv4 = av13 - av7;
        cv5 = av8 + av2;
        cv6 = av8 - av2;
        cv7 = av11 + av5;
        cv8 = av11 - av5;

        v1r = in_r[in_strides[1]];
        v2r = in_r[in_strides[11]];
        v3r = in_r[in_strides[6]];

        av1 = v1r + v2r;
        av2 = v2r - v1r;
        av14 = v3r + av1;
        av4 = v3r - (CRTM_15_5 * av1);

        v1i = in_i[in_strides[1]];
        v2i = in_i[in_strides[11]];
        v3i = in_i[in_strides[6]];

        av1 = v1i + v2i;
        av5 = v2i - v1i;
        av15 = v3i + av1;
        av7 = v3i - (CRTM_15_5 * av1);

        v1r = in_r[in_strides[4]];
        v2r = in_r[in_strides[14]];
        v3r = in_r[in_strides[9]];

        av1 = v1r + v2r;
        av8 = v2r - v1r;
        av16 = v3r + av1;
        av10 = v3r - (CRTM_15_5 * av1);

        v1i = in_i[in_strides[4]];
        v2i = in_i[in_strides[14]];
        v3i = in_i[in_strides[9]];

        av1 = v1i + v2i;
        av11 = v2i - v1i;
        av17 = v3i + av1;
        av13 = v3i - (CRTM_15_5 * av1);

        cv9 = av10 + av4;
        cv10 = av4 - av10;
        cv11 = av13 + av7;
        cv12 = av7 - av13;
        cv13 = av8 + av2;
        cv14 = av2 - av8;
        cv15 = av11 + av5;
        cv16 = av5 - av11;

        v1r = in_r[in_strides[5]];
        v2r = in_r[in_strides[10]];
        v3r = *in_r;

        av1 = v1r + v2r;
        av2 = CRTM_15_6 * (v2r - v1r);
        av18 = v3r + av1;
        av4 = v3r - (CRTM_15_5 * av1);

        v1i = in_i[in_strides[5]];
        v2i = in_i[in_strides[10]];
        v3i = *in_i;

        av1 = v1i + v2i;
        av5 = CRTM_15_6 * (v2i - v1i);
        av19 = v3i + av1;
        av7 = v3i - (CRTM_15_5 * av1);

        tv1 = av16 + av14;
        tv2 = av3 + av9;
        tv11 = av9 - av3;
        tv12 = av16 - av14;
        tv3 = tv1 + tv2;
        tv4 = CRTM_15_1 * (tv1 - tv2);
        tv5 = av18 - (CRTM_15_2 * tv3);

        *out_r = av18 + tv3;

        tv6 = av6 - av12;
        tv8 = av6 + av12;
        tv1 = av15 - av17;
        tv7 = av17 + av15;
        tv2 = (CRTM_15_3 * tv1) + (CRTM_15_4 * tv6);
        cv1r = tv5 + tv4;

        out_r[out_strides[3]] = cv1r + tv2;
        out_r[out_strides[12]] = cv1r - tv2;

        tv3 = tv7 + tv8;
        tv9 = CRTM_15_1 * (tv7 - tv8);
        tv10 = av19 - (CRTM_15_2 * tv3);

        *out_i = av19 + tv3;

        tv8 = (CRTM_15_3 * tv12) + (CRTM_15_4 * tv11);
        cv1i = tv10 + tv9;
        out_i[out_strides[3]] = cv1i + tv8;
        out_i[out_strides[12]] = cv1i - tv8;

        cv1r = tv5 - tv4;
        tv2 = (CRTM_15_4 * tv1) - (CRTM_15_3 * tv6);
        cv1i = tv10 - tv9;
        tv8 = (CRTM_15_4 * tv12) - (CRTM_15_3 * tv11);

        out_r[out_strides[6]] = cv1r + tv2;
        out_r[out_strides[9]] = cv1r - tv2;
        out_i[out_strides[6]] = cv1i + tv8;
        out_i[out_strides[9]] = cv1i - tv8;

        tv1 = CRTM_15_1 * (cv9 - cv1);
        tv2 = CRTM_15_9 * (cv15 + cv8);
        tv3 = cv9 + cv1;
        tv4 = cv8 - cv15;
        tv5 = tv3 + av4;
        tv6 = CRTM_15_6 * tv4 + av5;

        out_r[out_strides[5]] = tv5 + tv6;
        out_r[out_strides[10]] = tv5 - tv6;

        tv7 = CRTM_15_1 * (cv11 - cv3);
        tv8 = CRTM_15_9 * (cv13 + cv6);
        tv9 = cv11 + cv3;
        tv10 = cv13 - cv6;
        tv5 = tv9 + av7;
        tv6 = CRTM_15_6 * tv10 - av2;

        out_i[out_strides[5]] = tv5 + tv6;
        out_i[out_strides[10]] = tv5 - tv6;

        tv5 = av4 - (CRTM_15_2 * tv3);
        tv6 = (CRTM_15_10 * tv4) - av5;
        tv11 = av7 - (CRTM_15_2 * tv9);
        tv12 = av2 + (CRTM_15_10 * tv10);
        tv3 = tv6 + tv5;
        tv4 = tv5 - tv6;
        tv9 = tv2 - tv1;
        tv10 = tv1 + tv2;

        cv1r = tv3 - tv10;
        cv1 = (CRTM_15_4 * cv12) + (CRTM_15_3 * cv4);
        cv3 = (CRTM_15_8 * cv5) - (CRTM_15_7 * cv14);
        cv6 = cv1 + cv3;
        cv8 = cv1 - cv3;

        out_r[out_strides[1]] = cv1r + cv6;
        out_r[out_strides[4]] = cv1r - cv6;

        cv1r = tv3 + tv10;
        cv1 = (CRTM_15_8 * cv14) + (CRTM_15_7 * cv5);
        cv3 = (CRTM_15_4 * cv4) - (CRTM_15_3 * cv12);
        cv6 = cv3 + cv1;

        out_r[out_strides[7]] = cv1r + cv6;
        out_r[out_strides[13]] = cv1r - cv6;

        tv1 = tv12 + tv11;
        tv2 = tv11 - tv12;
        tv3 = tv7 - tv8;
        tv5 = tv8 + tv7;
        cv1i = tv1 - tv3;

        tv6 = (CRTM_15_4 * cv10) + (CRTM_15_3 * cv2);
        tv10 = (CRTM_15_8 * cv7) - (CRTM_15_7 * cv16);
        cv6 = tv10 - tv6;
        tv7 = tv6 + tv10;

        out_i[out_strides[1]] = cv1i + cv6;
        out_i[out_strides[4]] = cv1i - cv6;

        cv1i = tv1 + tv3;
        tv8 = (CRTM_15_8 * cv16) + (CRTM_15_7 * cv7);
        tv1 = (CRTM_15_3 * cv10) - (CRTM_15_4 * cv2);
        tv3 = tv1 + tv8;

        out_i[out_strides[7]] = cv1i + tv3;
        out_i[out_strides[13]] = cv1i - tv3;

        tv3 = tv1 - tv8;
        cv1r = tv4 - tv9;
        cv6 = cv3 - cv1;
        out_r[out_strides[2]] = cv1r + cv6;
        out_r[out_strides[8]] = cv1r - cv6;

        cv1i = tv2 + tv5;
        out_i[out_strides[2]] = cv1i + tv3;
        out_i[out_strides[8]] = cv1i - tv3;

        cv1r = tv4 + tv9;
        cv1i = tv2 - tv5;

        out_r[out_strides[11]] = cv1r + cv8;
        out_i[out_strides[11]] = cv1i - tv7;
        out_r[out_strides[14]] = cv1r - cv8;
        out_i[out_strides[14]] = cv1i + tv7;

        in_r += v_in_stride;
        in_i += v_in_stride;
        out_r += v_out_stride;
        out_i += v_out_stride;
    }
#ifdef AOCL_ENABLE_LOG
    AOCLFFTZ_LOG_UNFORMATTED(TRACE, TRACE, "Exit");
#endif
}

static VOID fft15c_fp32(VOID *in_real, VOID *in_imag, VOID *out_real,
                        VOID *out_imag, INTP n, aoclfftz_strides_t *strides,
                        UINT8 flag)
{
#ifdef AOCL_ENABLE_LOG
    AOCLFFTZ_LOG_UNFORMATTED(TRACE, TRACE, "Enter");
#endif
    const FLOAT CRTM_15_1 =
        +0.55901699437494742410229341718281905886015458990288;
    const FLOAT CRTM_15_2 =
        +0.25000000000000000000000000000000000000000000000000;
    const FLOAT CRTM_15_3 =
        +0.95105651629515357211643933337938214340569863400000;
    const FLOAT CRTM_15_4 =
        +0.58778525229247301629891039327884007596190389052978;
    const FLOAT CRTM_15_5 =
        +0.50000000000000000000000000000000000000000000000000;
    const FLOAT CRTM_15_6 =
        +0.86602540378443864676372317075293618347140262690519;
    // Below CRTMs are the product of the above CRTMs, Precomputed to save
    // multiplications on the fly.
    // CRTM_15_7 = CRTM_15_6 * CRTM_15_4
    const FLOAT CRTM_15_7 =
        +0.50903696045512706468216979248996715975105181034577;
    // CRTM_15_8 = CRTM_15_6 * CRTM_15_3
    const FLOAT CRTM_15_8 =
        +0.82363910354633184270744116161596601637855195182647;
    // CRTM_15_9 = CRTM_15_6 * CRTM_15_1
    const FLOAT CRTM_15_9 =
        +0.48412291827592710612024388657479988457787393064252;
    // CRTM_15_10 = CRTM_15_6 * CRTM_15_2
    const FLOAT CRTM_15_10 =
        +0.21650635094610964914707551542960572987794876098633;

    FLOAT *in_r = (FLOAT *)in_real;
    FLOAT *in_i = (FLOAT *)in_imag;
    FLOAT *out_r = (FLOAT *)out_real;
    FLOAT *out_i = (FLOAT *)out_imag;
#ifdef VOLATILE_STRIDE_ARRAY
    volatile INTP *in_strides = strides->in_strides;
    volatile INTP *out_strides = strides->out_strides;
#else
    INTP *in_strides = strides->in_strides;
    INTP *out_strides = strides->out_strides;
#endif
    INTP v_in_stride = strides->v_in_stride;
    INTP v_out_stride = strides->v_out_stride;
    INTP cnt;

    for (cnt = 0; cnt < n; cnt++)
    {
        FLOAT v1r, v1i, v2r, v2i, v3r, v3i, av1, av2, av3, av4, av5, av6, av7,
              av8, av9, av10, av11, av12, av13, av14, av15, av16, av17, av18,
              av19, tv1, tv2, tv3, tv4, tv5, tv6, tv7, tv8, tv9, tv10, tv11,
              tv12, cv1r, cv1i, cv1, cv2, cv3, cv4, cv5, cv6, cv7, cv8, cv9,
              cv10, cv11, cv12, cv13, cv14, cv15, cv16;

        v1r = in_r[in_strides[2]];
        v2r = in_r[in_strides[7]];
        v3r = in_r[in_strides[12]];

        av1 = v1r + v2r;
        av2 = v1r - v2r;
        av3 = v3r + av1;
        av4 = v3r - (CRTM_15_5 * av1);

        v1i = in_i[in_strides[2]];
        v2i = in_i[in_strides[7]];
        v3i = in_i[in_strides[12]];

        av1 = v1i + v2i;
        av5 = v1i - v2i;
        av6 = v3i + av1;
        av7 = v3i - (CRTM_15_5 * av1);

        v1r = in_r[in_strides[8]];
        v2r = in_r[in_strides[13]];
        v3r = in_r[in_strides[3]];

        av1 = v1r + v2r;
        av8 = v2r - v1r;
        av9 = v3r + av1;
        av10 = v3r - (CRTM_15_5 * av1);

        v1i = in_i[in_strides[8]];
        v2i = in_i[in_strides[13]];
        v3i = in_i[in_strides[3]];

        av1 = v1i + v2i;
        av11 = v2i - v1i;
        av12 = v3i + av1;
        av13 = v3i - (CRTM_15_5 * av1);

        cv1 = av10 + av4;
        cv2 = av10 - av4;
        cv3 = av13 + av7;
        cv4 = av13 - av7;
        cv5 = av8 + av2;
        cv6 = av8 - av2;
        cv7 = av11 + av5;
        cv8 = av11 - av5;

        v1r = in_r[in_strides[1]];
        v2r = in_r[in_strides[11]];
        v3r = in_r[in_strides[6]];

        av1 = v1r + v2r;
        av2 = v2r - v1r;
        av14 = v3r + av1;
        av4 = v3r - (CRTM_15_5 * av1);

        v1i = in_i[in_strides[1]];
        v2i = in_i[in_strides[11]];
        v3i = in_i[in_strides[6]];

        av1 = v1i + v2i;
        av5 = v2i - v1i;
        av15 = v3i + av1;
        av7 = v3i - (CRTM_15_5 * av1);

        v1r = in_r[in_strides[4]];
        v2r = in_r[in_strides[14]];
        v3r = in_r[in_strides[9]];

        av1 = v1r + v2r;
        av8 = v2r - v1r;
        av16 = v3r + av1;
        av10 = v3r - (CRTM_15_5 * av1);

        v1i = in_i[in_strides[4]];
        v2i = in_i[in_strides[14]];
        v3i = in_i[in_strides[9]];

        av1 = v1i + v2i;
        av11 = v2i - v1i;
        av17 = v3i + av1;
        av13 = v3i - (CRTM_15_5 * av1);

        cv9 = av10 + av4;
        cv10 = av4 - av10;
        cv11 = av13 + av7;
        cv12 = av7 - av13;
        cv13 = av8 + av2;
        cv14 = av2 - av8;
        cv15 = av11 + av5;
        cv16 = av5 - av11;

        v1r = in_r[in_strides[5]];
        v2r = in_r[in_strides[10]];
        v3r = *in_r;

        av1 = v1r + v2r;
        av2 = CRTM_15_6 * (v2r - v1r);
        av18 = v3r + av1;
        av4 = v3r - (CRTM_15_5 * av1);

        v1i = in_i[in_strides[5]];
        v2i = in_i[in_strides[10]];
        v3i = *in_i;

        av1 = v1i + v2i;
        av5 = CRTM_15_6 * (v2i - v1i);
        av19 = v3i + av1;
        av7 = v3i - (CRTM_15_5 * av1);

        tv1 = av16 + av14;
        tv2 = av3 + av9;
        tv11 = av9 - av3;
        tv12 = av16 - av14;
        tv3 = tv1 + tv2;
        tv4 = CRTM_15_1 * (tv1 - tv2);
        tv5 = av18 - (CRTM_15_2 * tv3);

        *out_r = av18 + tv3;

        tv6 = av6 - av12;
        tv8 = av6 + av12;
        tv1 = av15 - av17;
        tv7 = av17 + av15;
        tv2 = (CRTM_15_3 * tv1) + (CRTM_15_4 * tv6);
        cv1r = tv5 + tv4;

        out_r[out_strides[3]] = cv1r + tv2;
        out_r[out_strides[12]] = cv1r - tv2;

        tv3 = tv7 + tv8;
        tv9 = CRTM_15_1 * (tv7 - tv8);
        tv10 = av19 - (CRTM_15_2 * tv3);

        *out_i = av19 + tv3;

        tv8 = (CRTM_15_3 * tv12) + (CRTM_15_4 * tv11);
        cv1i = tv10 + tv9;
        out_i[out_strides[3]] = cv1i + tv8;
        out_i[out_strides[12]] = cv1i - tv8;

        cv1r = tv5 - tv4;
        tv2 = (CRTM_15_4 * tv1) - (CRTM_15_3 * tv6);
        cv1i = tv10 - tv9;
        tv8 = (CRTM_15_4 * tv12) - (CRTM_15_3 * tv11);

        out_r[out_strides[6]] = cv1r + tv2;
        out_r[out_strides[9]] = cv1r - tv2;
        out_i[out_strides[6]] = cv1i + tv8;
        out_i[out_strides[9]] = cv1i - tv8;

        tv1 = CRTM_15_1 * (cv9 - cv1);
        tv2 = CRTM_15_9 * (cv15 + cv8);
        tv3 = cv9 + cv1;
        tv4 = cv8 - cv15;
        tv5 = tv3 + av4;
        tv6 = CRTM_15_6 * tv4 + av5;

        out_r[out_strides[5]] = tv5 + tv6;
        out_r[out_strides[10]] = tv5 - tv6;

        tv7 = CRTM_15_1 * (cv11 - cv3);
        tv8 = CRTM_15_9 * (cv13 + cv6);
        tv9 = cv11 + cv3;
        tv10 = cv13 - cv6;
        tv5 = tv9 + av7;
        tv6 = CRTM_15_6 * tv10 - av2;

        out_i[out_strides[5]] = tv5 + tv6;
        out_i[out_strides[10]] = tv5 - tv6;

        tv5 = av4 - (CRTM_15_2 * tv3);
        tv6 = (CRTM_15_10 * tv4) - av5;
        tv11 = av7 - (CRTM_15_2 * tv9);
        tv12 = av2 + (CRTM_15_10 * tv10);
        tv3 = tv6 + tv5;
        tv4 = tv5 - tv6;
        tv9 = tv2 - tv1;
        tv10 = tv1 + tv2;

        cv1r = tv3 - tv10;
        cv1 = (CRTM_15_4 * cv12) + (CRTM_15_3 * cv4);
        cv3 = (CRTM_15_8 * cv5) - (CRTM_15_7 * cv14);
        cv6 = cv1 + cv3;
        cv8 = cv1 - cv3;

        out_r[out_strides[1]] = cv1r + cv6;
        out_r[out_strides[4]] = cv1r - cv6;

        cv1r = tv3 + tv10;
        cv1 = (CRTM_15_8 * cv14) + (CRTM_15_7 * cv5);
        cv3 = (CRTM_15_4 * cv4) - (CRTM_15_3 * cv12);
        cv6 = cv3 + cv1;

        out_r[out_strides[7]] = cv1r + cv6;
        out_r[out_strides[13]] = cv1r - cv6;

        tv1 = tv12 + tv11;
        tv2 = tv11 - tv12;
        tv3 = tv7 - tv8;
        tv5 = tv8 + tv7;
        cv1i = tv1 - tv3;

        tv6 = (CRTM_15_4 * cv10) + (CRTM_15_3 * cv2);
        tv10 = (CRTM_15_8 * cv7) - (CRTM_15_7 * cv16);
        cv6 = tv10 - tv6;
        tv7 = tv6 + tv10;

        out_i[out_strides[1]] = cv1i + cv6;
        out_i[out_strides[4]] = cv1i - cv6;

        cv1i = tv1 + tv3;
        tv8 = (CRTM_15_8 * cv16) + (CRTM_15_7 * cv7);
        tv1 = (CRTM_15_3 * cv10) - (CRTM_15_4 * cv2);
        tv3 = tv1 + tv8;

        out_i[out_strides[7]] = cv1i + tv3;
        out_i[out_strides[13]] = cv1i - tv3;

        tv3 = tv1 - tv8;
        cv1r = tv4 - tv9;
        cv6 = cv3 - cv1;
        out_r[out_strides[2]] = cv1r + cv6;
        out_r[out_strides[8]] = cv1r - cv6;

        cv1i = tv2 + tv5;
        out_i[out_strides[2]] = cv1i + tv3;
        out_i[out_strides[8]] = cv1i - tv3;

        cv1r = tv4 + tv9;
        cv1i = tv2 - tv5;

        out_r[out_strides[11]] = cv1r + cv8;
        out_i[out_strides[11]] = cv1i - tv7;
        out_r[out_strides[14]] = cv1r - cv8;
        out_i[out_strides[14]] = cv1i + tv7;

        in_r += v_in_stride;
        in_i += v_in_stride;
        out_r += v_out_stride;
        out_i += v_out_stride;
    }
#ifdef AOCL_ENABLE_LOG
    AOCLFFTZ_LOG_UNFORMATTED(TRACE, TRACE, "Exit");
#endif
}
#else
/* --------------- non-optimized C kernel variant --------------- */
#include "utils/complex_utils.h"

static const ops_cycles_t ops_cnt[NUM_PRECISIONS] = {
    {0, 368, 1520, 150, 0, 1681}, {0, 368, 1520, 150, 0, 1681}};

ops_cycles_t get_ops_cnt_fft15c(INT32 precision)
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

const DOUBLE CRTM_15[RADIX_15][2] = {{1.0, 0.0},
                                     {0.913545457642601, -0.4067366430758},
                                     {0.669130606358858, -0.743144825477394},
                                     {0.309016994374947, -0.951056516295154},
                                     {-0.104528463267653, -0.994521895368273},
                                     {-0.5, -0.866025403784439},
                                     {-0.809016994374947, -0.587785252292473},
                                     {-0.978147600733806, -0.207911690817759},
                                     {-0.978147600733806, 0.207911690817759},
                                     {-0.809016994374948, 0.587785252292473},
                                     {-0.5, 0.866025403784438},
                                     {-0.104528463267654, 0.994521895368273},
                                     {0.309016994374947, 0.951056516295154},
                                     {0.669130606358858, 0.743144825477394},
                                     {0.913545457642601, 0.4067366430758}};

static VOID fft15c_fp64(VOID *in_real, VOID *in_imag, VOID *out_real,
                        VOID *out_imag, INTP n, aoclfftz_strides_t *strides,
                        UINT8 flag)
{
#ifdef AOCL_ENABLE_LOG
    AOCLFFTZ_LOG_UNFORMATTED(TRACE, TRACE, "Enter");
#endif
    INTP *in_strides = strides->in_strides;
    INTP *out_strides = strides->out_strides;
    INTP v_in_stride = strides->v_in_stride;
    INTP v_out_stride = strides->v_out_stride;
    // temp variable to store power (constant_multiplier)
    DOUBLE pow_cm[2] = {0.0, 0.0};
    // temp variable to store pow_cm * input
    DOUBLE temp_out[2] = {0.0, 0.0};
    // buffer to store intermediate CMUL results
    DOUBLE cmul_temp[2] = {0.0, 0.0};
    // buffer to store intermediate CPOW results
    DOUBLE cpow_temp[2] = {0.0, 0.0};
    // buffer to store current input
    DOUBLE *in_dr = (DOUBLE *)in_real;
    DOUBLE *in_di = (DOUBLE *)in_imag;
    DOUBLE *input_r = (DOUBLE *)in_real;
    DOUBLE *input_i = (DOUBLE *)in_imag;
    // buffer to store current output
    DOUBLE *output_r = (DOUBLE *)out_real;
    DOUBLE *output_i = (DOUBLE *)out_imag;
    DOUBLE *out_dr = (DOUBLE *)out_real;
    DOUBLE *out_di = (DOUBLE *)out_imag;
    // local buffer to store input
    DOUBLE local_in[RADIX_15][2] = {0};

    for (INTP i = 0; i < n; i++)
    {
        /******************** load input **********************/
        input_r = in_dr;
        input_i = in_di;
        LOAD_INPUT(input_r, input_i, local_in[0]);
        LOAD_INPUT(input_r + in_strides[1], input_i + in_strides[1],
                   local_in[1]);
        LOAD_INPUT(input_r + in_strides[2], input_i + in_strides[2],
                   local_in[2]);
        LOAD_INPUT(input_r + in_strides[3], input_i + in_strides[3],
                   local_in[3]);
        LOAD_INPUT(input_r + in_strides[4], input_i + in_strides[4],
                   local_in[4]);
        LOAD_INPUT(input_r + in_strides[5], input_i + in_strides[5],
                   local_in[5]);
        LOAD_INPUT(input_r + in_strides[6], input_i + in_strides[6],
                   local_in[6]);
        LOAD_INPUT(input_r + in_strides[7], input_i + in_strides[7],
                   local_in[7]);
        LOAD_INPUT(input_r + in_strides[8], input_i + in_strides[8],
                   local_in[8]);
        LOAD_INPUT(input_r + in_strides[9], input_i + in_strides[9],
                   local_in[9]);
        LOAD_INPUT(input_r + in_strides[10], input_i + in_strides[10],
                   local_in[10]);
        LOAD_INPUT(input_r + in_strides[11], input_i + in_strides[11],
                   local_in[11]);
        LOAD_INPUT(input_r + in_strides[12], input_i + in_strides[12],
                   local_in[12]);
        LOAD_INPUT(input_r + in_strides[13], input_i + in_strides[13],
                   local_in[13]);
        LOAD_INPUT(input_r + in_strides[14], input_i + in_strides[14],
                   local_in[14]);

        output_r = out_dr;
        output_i = out_di;
        /******************** Output 15i ********************/
        pow_cm[0] = pow_cm[1] = temp_out[0] = temp_out[1] = 0.0;
        // CADD since (constant_multiplier)^0 = 1
        CADD(local_in[0], temp_out, temp_out);
        CADD(local_in[1], temp_out, temp_out);
        CADD(local_in[2], temp_out, temp_out);
        CADD(local_in[3], temp_out, temp_out);
        CADD(local_in[4], temp_out, temp_out);
        CADD(local_in[5], temp_out, temp_out);
        CADD(local_in[6], temp_out, temp_out);
        CADD(local_in[7], temp_out, temp_out);
        CADD(local_in[8], temp_out, temp_out);
        CADD(local_in[9], temp_out, temp_out);
        CADD(local_in[10], temp_out, temp_out);
        CADD(local_in[11], temp_out, temp_out);
        CADD(local_in[12], temp_out, temp_out);
        CADD(local_in[13], temp_out, temp_out);
        CADD(local_in[14], temp_out, temp_out);
        STORE_OUTPUT(temp_out, output_r, output_i);

        /******************** Output 15i+1 ********************/
        pow_cm[0] = pow_cm[1] = temp_out[0] = temp_out[1] = 0.0;
        // Using CMUL_CADD since pow(constant_multiplier, 1) =
        // constant_multiplier
        CADD(local_in[0], temp_out, temp_out);
        CMUL_CADD(local_in[1], CRTM_15[1], pow_cm, temp_out, cmul_temp);
        CMUL_CADD(local_in[2], CRTM_15[2], pow_cm, temp_out, cmul_temp);
        CMUL_CADD(local_in[3], CRTM_15[3], pow_cm, temp_out, cmul_temp);
        CMUL_CADD(local_in[4], CRTM_15[4], pow_cm, temp_out, cmul_temp);
        CMUL_CADD(local_in[5], CRTM_15[5], pow_cm, temp_out, cmul_temp);
        CMUL_CADD(local_in[6], CRTM_15[6], pow_cm, temp_out, cmul_temp);
        CMUL_CADD(local_in[7], CRTM_15[7], pow_cm, temp_out, cmul_temp);
        CMUL_CADD(local_in[8], CRTM_15[8], pow_cm, temp_out, cmul_temp);
        CMUL_CADD(local_in[9], CRTM_15[9], pow_cm, temp_out, cmul_temp);
        CMUL_CADD(local_in[10], CRTM_15[10], pow_cm, temp_out, cmul_temp);
        CMUL_CADD(local_in[11], CRTM_15[11], pow_cm, temp_out, cmul_temp);
        CMUL_CADD(local_in[12], CRTM_15[12], pow_cm, temp_out, cmul_temp);
        CMUL_CADD(local_in[13], CRTM_15[13], pow_cm, temp_out, cmul_temp);
        CMUL_CADD(local_in[14], CRTM_15[14], pow_cm, temp_out, cmul_temp);
        STORE_OUTPUT(temp_out, output_r + out_strides[1],
                     output_i + out_strides[1]);

        /******************** Output 15i+2 ********************/
        pow_cm[0] = pow_cm[1] = temp_out[0] = temp_out[1] = 0.0;
        CADD(local_in[0], temp_out, temp_out);
        CPOW(CRTM_15[1], 2, pow_cm, cmul_temp, cpow_temp);
        CMUL_CADD_CPOW(local_in[1], CRTM_15[2], 2, pow_cm, temp_out, cmul_temp,
                       cpow_temp);
        CMUL_CADD_CPOW(local_in[2], CRTM_15[3], 2, pow_cm, temp_out, cmul_temp,
                       cpow_temp);
        CMUL_CADD_CPOW(local_in[3], CRTM_15[4], 2, pow_cm, temp_out, cmul_temp,
                       cpow_temp);
        CMUL_CADD_CPOW(local_in[4], CRTM_15[5], 2, pow_cm, temp_out, cmul_temp,
                       cpow_temp);
        CMUL_CADD_CPOW(local_in[5], CRTM_15[6], 2, pow_cm, temp_out, cmul_temp,
                       cpow_temp);
        CMUL_CADD_CPOW(local_in[6], CRTM_15[7], 2, pow_cm, temp_out, cmul_temp,
                       cpow_temp);
        CMUL_CADD_CPOW(local_in[7], CRTM_15[8], 2, pow_cm, temp_out, cmul_temp,
                       cpow_temp);
        CMUL_CADD_CPOW(local_in[8], CRTM_15[9], 2, pow_cm, temp_out, cmul_temp,
                       cpow_temp);
        CMUL_CADD_CPOW(local_in[9], CRTM_15[10], 2, pow_cm, temp_out, cmul_temp,
                       cpow_temp);
        CMUL_CADD_CPOW(local_in[10], CRTM_15[11], 2, pow_cm, temp_out,
                       cmul_temp, cpow_temp);
        CMUL_CADD_CPOW(local_in[11], CRTM_15[12], 2, pow_cm, temp_out,
                       cmul_temp, cpow_temp);
        CMUL_CADD_CPOW(local_in[12], CRTM_15[13], 2, pow_cm, temp_out,
                       cmul_temp, cpow_temp);
        CMUL_CADD_CPOW(local_in[13], CRTM_15[14], 2, pow_cm, temp_out,
                       cmul_temp, cpow_temp);
        CMUL_CADD(local_in[14], pow_cm, pow_cm, temp_out, cmul_temp);
        STORE_OUTPUT(temp_out, output_r + out_strides[2],
                     output_i + out_strides[2]);

        /******************** Output 15i+3 ********************/
        pow_cm[0] = pow_cm[1] = temp_out[0] = temp_out[1] = 0.0;
        CADD(local_in[0], temp_out, temp_out);
        CPOW(CRTM_15[1], 3, pow_cm, cmul_temp, cpow_temp);
        CMUL_CADD_CPOW(local_in[1], CRTM_15[2], 3, pow_cm, temp_out, cmul_temp,
                       cpow_temp);
        CMUL_CADD_CPOW(local_in[2], CRTM_15[3], 3, pow_cm, temp_out, cmul_temp,
                       cpow_temp);
        CMUL_CADD_CPOW(local_in[3], CRTM_15[4], 3, pow_cm, temp_out, cmul_temp,
                       cpow_temp);
        CMUL_CADD_CPOW(local_in[4], CRTM_15[5], 3, pow_cm, temp_out, cmul_temp,
                       cpow_temp);
        CMUL_CADD_CPOW(local_in[5], CRTM_15[6], 3, pow_cm, temp_out, cmul_temp,
                       cpow_temp);
        CMUL_CADD_CPOW(local_in[6], CRTM_15[7], 3, pow_cm, temp_out, cmul_temp,
                       cpow_temp);
        CMUL_CADD_CPOW(local_in[7], CRTM_15[8], 3, pow_cm, temp_out, cmul_temp,
                       cpow_temp);
        CMUL_CADD_CPOW(local_in[8], CRTM_15[9], 3, pow_cm, temp_out, cmul_temp,
                       cpow_temp);
        CMUL_CADD_CPOW(local_in[9], CRTM_15[10], 3, pow_cm, temp_out, cmul_temp,
                       cpow_temp);
        CMUL_CADD_CPOW(local_in[10], CRTM_15[11], 3, pow_cm, temp_out,
                       cmul_temp, cpow_temp);
        CMUL_CADD_CPOW(local_in[11], CRTM_15[12], 3, pow_cm, temp_out,
                       cmul_temp, cpow_temp);
        CMUL_CADD_CPOW(local_in[12], CRTM_15[13], 3, pow_cm, temp_out,
                       cmul_temp, cpow_temp);
        CMUL_CADD_CPOW(local_in[13], CRTM_15[14], 3, pow_cm, temp_out,
                       cmul_temp, cpow_temp);
        CMUL_CADD(local_in[14], pow_cm, pow_cm, temp_out, cmul_temp);
        STORE_OUTPUT(temp_out, output_r + out_strides[3],
                     output_i + out_strides[3]);

        /******************** Output 15i+4 ********************/
        pow_cm[0] = pow_cm[1] = temp_out[0] = temp_out[1] = 0.0;
        CADD(local_in[0], temp_out, temp_out);
        CPOW(CRTM_15[1], 4, pow_cm, cmul_temp, cpow_temp);
        CMUL_CADD_CPOW(local_in[1], CRTM_15[2], 4, pow_cm, temp_out, cmul_temp,
                       cpow_temp);
        CMUL_CADD_CPOW(local_in[2], CRTM_15[3], 4, pow_cm, temp_out, cmul_temp,
                       cpow_temp);
        CMUL_CADD_CPOW(local_in[3], CRTM_15[4], 4, pow_cm, temp_out, cmul_temp,
                       cpow_temp);
        CMUL_CADD_CPOW(local_in[4], CRTM_15[5], 4, pow_cm, temp_out, cmul_temp,
                       cpow_temp);
        CMUL_CADD_CPOW(local_in[5], CRTM_15[6], 4, pow_cm, temp_out, cmul_temp,
                       cpow_temp);
        CMUL_CADD_CPOW(local_in[6], CRTM_15[7], 4, pow_cm, temp_out, cmul_temp,
                       cpow_temp);
        CMUL_CADD_CPOW(local_in[7], CRTM_15[8], 4, pow_cm, temp_out, cmul_temp,
                       cpow_temp);
        CMUL_CADD_CPOW(local_in[8], CRTM_15[9], 4, pow_cm, temp_out, cmul_temp,
                       cpow_temp);
        CMUL_CADD_CPOW(local_in[9], CRTM_15[10], 4, pow_cm, temp_out, cmul_temp,
                       cpow_temp);
        CMUL_CADD_CPOW(local_in[10], CRTM_15[11], 4, pow_cm, temp_out,
                       cmul_temp, cpow_temp);
        CMUL_CADD_CPOW(local_in[11], CRTM_15[12], 4, pow_cm, temp_out,
                       cmul_temp, cpow_temp);
        CMUL_CADD_CPOW(local_in[12], CRTM_15[13], 4, pow_cm, temp_out,
                       cmul_temp, cpow_temp);
        CMUL_CADD_CPOW(local_in[13], CRTM_15[14], 4, pow_cm, temp_out,
                       cmul_temp, cpow_temp);
        CMUL_CADD(local_in[14], pow_cm, pow_cm, temp_out, cmul_temp);
        STORE_OUTPUT(temp_out, output_r + out_strides[4],
                     output_i + out_strides[4]);

        /******************** Output 15i+5 ********************/
        pow_cm[0] = pow_cm[1] = temp_out[0] = temp_out[1] = 0.0;
        CADD(local_in[0], temp_out, temp_out);
        CPOW(CRTM_15[1], 5, pow_cm, cmul_temp, cpow_temp);
        CMUL_CADD_CPOW(local_in[1], CRTM_15[2], 5, pow_cm, temp_out, cmul_temp,
                       cpow_temp);
        CMUL_CADD_CPOW(local_in[2], CRTM_15[3], 5, pow_cm, temp_out, cmul_temp,
                       cpow_temp);
        CMUL_CADD_CPOW(local_in[3], CRTM_15[4], 5, pow_cm, temp_out, cmul_temp,
                       cpow_temp);
        CMUL_CADD_CPOW(local_in[4], CRTM_15[5], 5, pow_cm, temp_out, cmul_temp,
                       cpow_temp);
        CMUL_CADD_CPOW(local_in[5], CRTM_15[6], 5, pow_cm, temp_out, cmul_temp,
                       cpow_temp);
        CMUL_CADD_CPOW(local_in[6], CRTM_15[7], 5, pow_cm, temp_out, cmul_temp,
                       cpow_temp);
        CMUL_CADD_CPOW(local_in[7], CRTM_15[8], 5, pow_cm, temp_out, cmul_temp,
                       cpow_temp);
        CMUL_CADD_CPOW(local_in[8], CRTM_15[9], 5, pow_cm, temp_out, cmul_temp,
                       cpow_temp);
        CMUL_CADD_CPOW(local_in[9], CRTM_15[10], 5, pow_cm, temp_out, cmul_temp,
                       cpow_temp);
        CMUL_CADD_CPOW(local_in[10], CRTM_15[11], 5, pow_cm, temp_out,
                       cmul_temp, cpow_temp);
        CMUL_CADD_CPOW(local_in[11], CRTM_15[12], 5, pow_cm, temp_out,
                       cmul_temp, cpow_temp);
        CMUL_CADD_CPOW(local_in[12], CRTM_15[13], 5, pow_cm, temp_out,
                       cmul_temp, cpow_temp);
        CMUL_CADD_CPOW(local_in[13], CRTM_15[14], 5, pow_cm, temp_out,
                       cmul_temp, cpow_temp);
        CMUL_CADD(local_in[14], pow_cm, pow_cm, temp_out, cmul_temp);
        STORE_OUTPUT(temp_out, output_r + out_strides[5],
                     output_i + out_strides[5]);

        /******************** Output 15i+6 ********************/
        pow_cm[0] = pow_cm[1] = temp_out[0] = temp_out[1] = 0.0;
        CADD(local_in[0], temp_out, temp_out);
        CPOW(CRTM_15[1], 6, pow_cm, cmul_temp, cpow_temp);
        CMUL_CADD_CPOW(local_in[1], CRTM_15[2], 6, pow_cm, temp_out, cmul_temp,
                       cpow_temp);
        CMUL_CADD_CPOW(local_in[2], CRTM_15[3], 6, pow_cm, temp_out, cmul_temp,
                       cpow_temp);
        CMUL_CADD_CPOW(local_in[3], CRTM_15[4], 6, pow_cm, temp_out, cmul_temp,
                       cpow_temp);
        CMUL_CADD_CPOW(local_in[4], CRTM_15[5], 6, pow_cm, temp_out, cmul_temp,
                       cpow_temp);
        CMUL_CADD_CPOW(local_in[5], CRTM_15[6], 6, pow_cm, temp_out, cmul_temp,
                       cpow_temp);
        CMUL_CADD_CPOW(local_in[6], CRTM_15[7], 6, pow_cm, temp_out, cmul_temp,
                       cpow_temp);
        CMUL_CADD_CPOW(local_in[7], CRTM_15[8], 6, pow_cm, temp_out, cmul_temp,
                       cpow_temp);
        CMUL_CADD_CPOW(local_in[8], CRTM_15[9], 6, pow_cm, temp_out, cmul_temp,
                       cpow_temp);
        CMUL_CADD_CPOW(local_in[9], CRTM_15[10], 6, pow_cm, temp_out, cmul_temp,
                       cpow_temp);
        CMUL_CADD_CPOW(local_in[10], CRTM_15[11], 6, pow_cm, temp_out,
                       cmul_temp, cpow_temp);
        CMUL_CADD_CPOW(local_in[11], CRTM_15[12], 6, pow_cm, temp_out,
                       cmul_temp, cpow_temp);
        CMUL_CADD_CPOW(local_in[12], CRTM_15[13], 6, pow_cm, temp_out,
                       cmul_temp, cpow_temp);
        CMUL_CADD_CPOW(local_in[13], CRTM_15[14], 6, pow_cm, temp_out,
                       cmul_temp, cpow_temp);
        CMUL_CADD(local_in[14], pow_cm, pow_cm, temp_out, cmul_temp);
        STORE_OUTPUT(temp_out, output_r + out_strides[6],
                     output_i + out_strides[6]);

        /******************** Output 15i+7 ********************/
        pow_cm[0] = pow_cm[1] = temp_out[0] = temp_out[1] = 0.0;
        CADD(local_in[0], temp_out, temp_out);
        CPOW(CRTM_15[1], 7, pow_cm, cmul_temp, cpow_temp);
        CMUL_CADD_CPOW(local_in[1], CRTM_15[2], 7, pow_cm, temp_out, cmul_temp,
                       cpow_temp);
        CMUL_CADD_CPOW(local_in[2], CRTM_15[3], 7, pow_cm, temp_out, cmul_temp,
                       cpow_temp);
        CMUL_CADD_CPOW(local_in[3], CRTM_15[4], 7, pow_cm, temp_out, cmul_temp,
                       cpow_temp);
        CMUL_CADD_CPOW(local_in[4], CRTM_15[5], 7, pow_cm, temp_out, cmul_temp,
                       cpow_temp);
        CMUL_CADD_CPOW(local_in[5], CRTM_15[6], 7, pow_cm, temp_out, cmul_temp,
                       cpow_temp);
        CMUL_CADD_CPOW(local_in[6], CRTM_15[7], 7, pow_cm, temp_out, cmul_temp,
                       cpow_temp);
        CMUL_CADD_CPOW(local_in[7], CRTM_15[8], 7, pow_cm, temp_out, cmul_temp,
                       cpow_temp);
        CMUL_CADD_CPOW(local_in[8], CRTM_15[9], 7, pow_cm, temp_out, cmul_temp,
                       cpow_temp);
        CMUL_CADD_CPOW(local_in[9], CRTM_15[10], 7, pow_cm, temp_out, cmul_temp,
                       cpow_temp);
        CMUL_CADD_CPOW(local_in[10], CRTM_15[11], 7, pow_cm, temp_out,
                       cmul_temp, cpow_temp);
        CMUL_CADD_CPOW(local_in[11], CRTM_15[12], 7, pow_cm, temp_out,
                       cmul_temp, cpow_temp);
        CMUL_CADD_CPOW(local_in[12], CRTM_15[13], 7, pow_cm, temp_out,
                       cmul_temp, cpow_temp);
        CMUL_CADD_CPOW(local_in[13], CRTM_15[14], 7, pow_cm, temp_out,
                       cmul_temp, cpow_temp);
        CMUL_CADD(local_in[14], pow_cm, pow_cm, temp_out, cmul_temp);
        STORE_OUTPUT(temp_out, output_r + out_strides[7],
                     output_i + out_strides[7]);

        /******************** Output 15i+8 ********************/
        pow_cm[0] = pow_cm[1] = temp_out[0] = temp_out[1] = 0.0;
        CADD(local_in[0], temp_out, temp_out);
        CPOW(CRTM_15[1], 8, pow_cm, cmul_temp, cpow_temp);
        CMUL_CADD_CPOW(local_in[1], CRTM_15[2], 8, pow_cm, temp_out, cmul_temp,
                       cpow_temp);
        CMUL_CADD_CPOW(local_in[2], CRTM_15[3], 8, pow_cm, temp_out, cmul_temp,
                       cpow_temp);
        CMUL_CADD_CPOW(local_in[3], CRTM_15[4], 8, pow_cm, temp_out, cmul_temp,
                       cpow_temp);
        CMUL_CADD_CPOW(local_in[4], CRTM_15[5], 8, pow_cm, temp_out, cmul_temp,
                       cpow_temp);
        CMUL_CADD_CPOW(local_in[5], CRTM_15[6], 8, pow_cm, temp_out, cmul_temp,
                       cpow_temp);
        CMUL_CADD_CPOW(local_in[6], CRTM_15[7], 8, pow_cm, temp_out, cmul_temp,
                       cpow_temp);
        CMUL_CADD_CPOW(local_in[7], CRTM_15[8], 8, pow_cm, temp_out, cmul_temp,
                       cpow_temp);
        CMUL_CADD_CPOW(local_in[8], CRTM_15[9], 8, pow_cm, temp_out, cmul_temp,
                       cpow_temp);
        CMUL_CADD_CPOW(local_in[9], CRTM_15[10], 8, pow_cm, temp_out, cmul_temp,
                       cpow_temp);
        CMUL_CADD_CPOW(local_in[10], CRTM_15[11], 8, pow_cm, temp_out,
                       cmul_temp, cpow_temp);
        CMUL_CADD_CPOW(local_in[11], CRTM_15[12], 8, pow_cm, temp_out,
                       cmul_temp, cpow_temp);
        CMUL_CADD_CPOW(local_in[12], CRTM_15[13], 8, pow_cm, temp_out,
                       cmul_temp, cpow_temp);
        CMUL_CADD_CPOW(local_in[13], CRTM_15[14], 8, pow_cm, temp_out,
                       cmul_temp, cpow_temp);
        CMUL_CADD(local_in[14], pow_cm, pow_cm, temp_out, cmul_temp);
        STORE_OUTPUT(temp_out, output_r + out_strides[8],
                     output_i + out_strides[8]);

        /******************** Output 15i+9 ********************/
        pow_cm[0] = pow_cm[1] = temp_out[0] = temp_out[1] = 0.0;
        CADD(local_in[0], temp_out, temp_out);
        CPOW(CRTM_15[1], 9, pow_cm, cmul_temp, cpow_temp);
        CMUL_CADD_CPOW(local_in[1], CRTM_15[2], 9, pow_cm, temp_out, cmul_temp,
                       cpow_temp);
        CMUL_CADD_CPOW(local_in[2], CRTM_15[3], 9, pow_cm, temp_out, cmul_temp,
                       cpow_temp);
        CMUL_CADD_CPOW(local_in[3], CRTM_15[4], 9, pow_cm, temp_out, cmul_temp,
                       cpow_temp);
        CMUL_CADD_CPOW(local_in[4], CRTM_15[5], 9, pow_cm, temp_out, cmul_temp,
                       cpow_temp);
        CMUL_CADD_CPOW(local_in[5], CRTM_15[6], 9, pow_cm, temp_out, cmul_temp,
                       cpow_temp);
        CMUL_CADD_CPOW(local_in[6], CRTM_15[7], 9, pow_cm, temp_out, cmul_temp,
                       cpow_temp);
        CMUL_CADD_CPOW(local_in[7], CRTM_15[8], 9, pow_cm, temp_out, cmul_temp,
                       cpow_temp);
        CMUL_CADD_CPOW(local_in[8], CRTM_15[9], 9, pow_cm, temp_out, cmul_temp,
                       cpow_temp);
        CMUL_CADD_CPOW(local_in[9], CRTM_15[10], 9, pow_cm, temp_out, cmul_temp,
                       cpow_temp);
        CMUL_CADD_CPOW(local_in[10], CRTM_15[11], 9, pow_cm, temp_out,
                       cmul_temp, cpow_temp);
        CMUL_CADD_CPOW(local_in[11], CRTM_15[12], 9, pow_cm, temp_out,
                       cmul_temp, cpow_temp);
        CMUL_CADD_CPOW(local_in[12], CRTM_15[13], 9, pow_cm, temp_out,
                       cmul_temp, cpow_temp);
        CMUL_CADD_CPOW(local_in[13], CRTM_15[14], 9, pow_cm, temp_out,
                       cmul_temp, cpow_temp);
        CMUL_CADD(local_in[14], pow_cm, pow_cm, temp_out, cmul_temp);
        STORE_OUTPUT(temp_out, output_r + out_strides[9],
                     output_i + out_strides[9]);

        /******************** Output 15i+10 ********************/
        pow_cm[0] = pow_cm[1] = temp_out[0] = temp_out[1] = 0.0;
        CADD(local_in[0], temp_out, temp_out);
        CPOW(CRTM_15[1], 10, pow_cm, cmul_temp, cpow_temp);
        CMUL_CADD_CPOW(local_in[1], CRTM_15[2], 10, pow_cm, temp_out, cmul_temp,
                       cpow_temp);
        CMUL_CADD_CPOW(local_in[2], CRTM_15[3], 10, pow_cm, temp_out, cmul_temp,
                       cpow_temp);
        CMUL_CADD_CPOW(local_in[3], CRTM_15[4], 10, pow_cm, temp_out, cmul_temp,
                       cpow_temp);
        CMUL_CADD_CPOW(local_in[4], CRTM_15[5], 10, pow_cm, temp_out, cmul_temp,
                       cpow_temp);
        CMUL_CADD_CPOW(local_in[5], CRTM_15[6], 10, pow_cm, temp_out, cmul_temp,
                       cpow_temp);
        CMUL_CADD_CPOW(local_in[6], CRTM_15[7], 10, pow_cm, temp_out, cmul_temp,
                       cpow_temp);
        CMUL_CADD_CPOW(local_in[7], CRTM_15[8], 10, pow_cm, temp_out, cmul_temp,
                       cpow_temp);
        CMUL_CADD_CPOW(local_in[8], CRTM_15[9], 10, pow_cm, temp_out, cmul_temp,
                       cpow_temp);
        CMUL_CADD_CPOW(local_in[9], CRTM_15[10], 10, pow_cm, temp_out,
                       cmul_temp, cpow_temp);
        CMUL_CADD_CPOW(local_in[10], CRTM_15[11], 10, pow_cm, temp_out,
                       cmul_temp, cpow_temp);
        CMUL_CADD_CPOW(local_in[11], CRTM_15[12], 10, pow_cm, temp_out,
                       cmul_temp, cpow_temp);
        CMUL_CADD_CPOW(local_in[12], CRTM_15[13], 10, pow_cm, temp_out,
                       cmul_temp, cpow_temp);
        CMUL_CADD_CPOW(local_in[13], CRTM_15[14], 10, pow_cm, temp_out,
                       cmul_temp, cpow_temp);
        CMUL_CADD(local_in[14], pow_cm, pow_cm, temp_out, cmul_temp);
        STORE_OUTPUT(temp_out, output_r + out_strides[10],
                     output_i + out_strides[10]);

        /******************** Output 15i+11 ********************/
        pow_cm[0] = pow_cm[1] = temp_out[0] = temp_out[1] = 0.0;
        CADD(local_in[0], temp_out, temp_out);
        CPOW(CRTM_15[1], 11, pow_cm, cmul_temp, cpow_temp);
        CMUL_CADD_CPOW(local_in[1], CRTM_15[2], 11, pow_cm, temp_out, cmul_temp,
                       cpow_temp);
        CMUL_CADD_CPOW(local_in[2], CRTM_15[3], 11, pow_cm, temp_out, cmul_temp,
                       cpow_temp);
        CMUL_CADD_CPOW(local_in[3], CRTM_15[4], 11, pow_cm, temp_out, cmul_temp,
                       cpow_temp);
        CMUL_CADD_CPOW(local_in[4], CRTM_15[5], 11, pow_cm, temp_out, cmul_temp,
                       cpow_temp);
        CMUL_CADD_CPOW(local_in[5], CRTM_15[6], 11, pow_cm, temp_out, cmul_temp,
                       cpow_temp);
        CMUL_CADD_CPOW(local_in[6], CRTM_15[7], 11, pow_cm, temp_out, cmul_temp,
                       cpow_temp);
        CMUL_CADD_CPOW(local_in[7], CRTM_15[8], 11, pow_cm, temp_out, cmul_temp,
                       cpow_temp);
        CMUL_CADD_CPOW(local_in[8], CRTM_15[9], 11, pow_cm, temp_out, cmul_temp,
                       cpow_temp);
        CMUL_CADD_CPOW(local_in[9], CRTM_15[10], 11, pow_cm, temp_out,
                       cmul_temp, cpow_temp);
        CMUL_CADD_CPOW(local_in[10], CRTM_15[11], 11, pow_cm, temp_out,
                       cmul_temp, cpow_temp);
        CMUL_CADD_CPOW(local_in[11], CRTM_15[12], 11, pow_cm, temp_out,
                       cmul_temp, cpow_temp);
        CMUL_CADD_CPOW(local_in[12], CRTM_15[13], 11, pow_cm, temp_out,
                       cmul_temp, cpow_temp);
        CMUL_CADD_CPOW(local_in[13], CRTM_15[14], 11, pow_cm, temp_out,
                       cmul_temp, cpow_temp);
        CMUL_CADD(local_in[14], pow_cm, pow_cm, temp_out, cmul_temp);
        STORE_OUTPUT(temp_out, output_r + out_strides[11],
                     output_i + out_strides[11]);

        /******************** Output 15i+12 ********************/
        pow_cm[0] = pow_cm[1] = temp_out[0] = temp_out[1] = 0.0;
        CADD(local_in[0], temp_out, temp_out);
        CPOW(CRTM_15[1], 12, pow_cm, cmul_temp, cpow_temp);
        CMUL_CADD_CPOW(local_in[1], CRTM_15[2], 12, pow_cm, temp_out, cmul_temp,
                       cpow_temp);
        CMUL_CADD_CPOW(local_in[2], CRTM_15[3], 12, pow_cm, temp_out, cmul_temp,
                       cpow_temp);
        CMUL_CADD_CPOW(local_in[3], CRTM_15[4], 12, pow_cm, temp_out, cmul_temp,
                       cpow_temp);
        CMUL_CADD_CPOW(local_in[4], CRTM_15[5], 12, pow_cm, temp_out, cmul_temp,
                       cpow_temp);
        CMUL_CADD_CPOW(local_in[5], CRTM_15[6], 12, pow_cm, temp_out, cmul_temp,
                       cpow_temp);
        CMUL_CADD_CPOW(local_in[6], CRTM_15[7], 12, pow_cm, temp_out, cmul_temp,
                       cpow_temp);
        CMUL_CADD_CPOW(local_in[7], CRTM_15[8], 12, pow_cm, temp_out, cmul_temp,
                       cpow_temp);
        CMUL_CADD_CPOW(local_in[8], CRTM_15[9], 12, pow_cm, temp_out, cmul_temp,
                       cpow_temp);
        CMUL_CADD_CPOW(local_in[9], CRTM_15[10], 12, pow_cm, temp_out,
                       cmul_temp, cpow_temp);
        CMUL_CADD_CPOW(local_in[10], CRTM_15[11], 12, pow_cm, temp_out,
                       cmul_temp, cpow_temp);
        CMUL_CADD_CPOW(local_in[11], CRTM_15[12], 12, pow_cm, temp_out,
                       cmul_temp, cpow_temp);
        CMUL_CADD_CPOW(local_in[12], CRTM_15[13], 12, pow_cm, temp_out,
                       cmul_temp, cpow_temp);
        CMUL_CADD_CPOW(local_in[13], CRTM_15[14], 12, pow_cm, temp_out,
                       cmul_temp, cpow_temp);
        CMUL_CADD(local_in[14], pow_cm, pow_cm, temp_out, cmul_temp);
        STORE_OUTPUT(temp_out, output_r + out_strides[12],
                     output_i + out_strides[12]);

        /******************** Output 15i+13 ********************/
        pow_cm[0] = pow_cm[1] = temp_out[0] = temp_out[1] = 0.0;
        CADD(local_in[0], temp_out, temp_out);
        CPOW(CRTM_15[1], 13, pow_cm, cmul_temp, cpow_temp);
        CMUL_CADD_CPOW(local_in[1], CRTM_15[2], 13, pow_cm, temp_out, cmul_temp,
                       cpow_temp);
        CMUL_CADD_CPOW(local_in[2], CRTM_15[3], 13, pow_cm, temp_out, cmul_temp,
                       cpow_temp);
        CMUL_CADD_CPOW(local_in[3], CRTM_15[4], 13, pow_cm, temp_out, cmul_temp,
                       cpow_temp);
        CMUL_CADD_CPOW(local_in[4], CRTM_15[5], 13, pow_cm, temp_out, cmul_temp,
                       cpow_temp);
        CMUL_CADD_CPOW(local_in[5], CRTM_15[6], 13, pow_cm, temp_out, cmul_temp,
                       cpow_temp);
        CMUL_CADD_CPOW(local_in[6], CRTM_15[7], 13, pow_cm, temp_out, cmul_temp,
                       cpow_temp);
        CMUL_CADD_CPOW(local_in[7], CRTM_15[8], 13, pow_cm, temp_out, cmul_temp,
                       cpow_temp);
        CMUL_CADD_CPOW(local_in[8], CRTM_15[9], 13, pow_cm, temp_out, cmul_temp,
                       cpow_temp);
        CMUL_CADD_CPOW(local_in[9], CRTM_15[10], 13, pow_cm, temp_out,
                       cmul_temp, cpow_temp);
        CMUL_CADD_CPOW(local_in[10], CRTM_15[11], 13, pow_cm, temp_out,
                       cmul_temp, cpow_temp);
        CMUL_CADD_CPOW(local_in[11], CRTM_15[12], 13, pow_cm, temp_out,
                       cmul_temp, cpow_temp);
        CMUL_CADD_CPOW(local_in[12], CRTM_15[13], 13, pow_cm, temp_out,
                       cmul_temp, cpow_temp);
        CMUL_CADD_CPOW(local_in[13], CRTM_15[14], 13, pow_cm, temp_out,
                       cmul_temp, cpow_temp);
        CMUL_CADD(local_in[14], pow_cm, pow_cm, temp_out, cmul_temp);
        STORE_OUTPUT(temp_out, output_r + out_strides[13],
                     output_i + out_strides[13]);

        /******************** Output 15i+14 ********************/
        pow_cm[0] = pow_cm[1] = temp_out[0] = temp_out[1] = 0.0;
        CADD(local_in[0], temp_out, temp_out);
        CPOW(CRTM_15[1], 14, pow_cm, cmul_temp, cpow_temp);
        CMUL_CADD_CPOW(local_in[1], CRTM_15[2], 14, pow_cm, temp_out, cmul_temp,
                       cpow_temp);
        CMUL_CADD_CPOW(local_in[2], CRTM_15[3], 14, pow_cm, temp_out, cmul_temp,
                       cpow_temp);
        CMUL_CADD_CPOW(local_in[3], CRTM_15[4], 14, pow_cm, temp_out, cmul_temp,
                       cpow_temp);
        CMUL_CADD_CPOW(local_in[4], CRTM_15[5], 14, pow_cm, temp_out, cmul_temp,
                       cpow_temp);
        CMUL_CADD_CPOW(local_in[5], CRTM_15[6], 14, pow_cm, temp_out, cmul_temp,
                       cpow_temp);
        CMUL_CADD_CPOW(local_in[6], CRTM_15[7], 14, pow_cm, temp_out, cmul_temp,
                       cpow_temp);
        CMUL_CADD_CPOW(local_in[7], CRTM_15[8], 14, pow_cm, temp_out, cmul_temp,
                       cpow_temp);
        CMUL_CADD_CPOW(local_in[8], CRTM_15[9], 14, pow_cm, temp_out, cmul_temp,
                       cpow_temp);
        CMUL_CADD_CPOW(local_in[9], CRTM_15[10], 14, pow_cm, temp_out,
                       cmul_temp, cpow_temp);
        CMUL_CADD_CPOW(local_in[10], CRTM_15[11], 14, pow_cm, temp_out,
                       cmul_temp, cpow_temp);
        CMUL_CADD_CPOW(local_in[11], CRTM_15[12], 14, pow_cm, temp_out,
                       cmul_temp, cpow_temp);
        CMUL_CADD_CPOW(local_in[12], CRTM_15[13], 14, pow_cm, temp_out,
                       cmul_temp, cpow_temp);
        CMUL_CADD_CPOW(local_in[13], CRTM_15[14], 14, pow_cm, temp_out,
                       cmul_temp, cpow_temp);
        CMUL_CADD(local_in[14], pow_cm, pow_cm, temp_out, cmul_temp);
        STORE_OUTPUT(temp_out, output_r + out_strides[14],
                     output_i + out_strides[14]);

        // next set
        in_dr += v_in_stride;
        in_di += v_in_stride;
        out_dr += v_out_stride;
        out_di += v_out_stride;
    }
}

static VOID fft15c_fp32(VOID *in_real, VOID *in_imag, VOID *out_real,
                        VOID *out_imag, INTP n, aoclfftz_strides_t *strides,
                        UINT8 flag)
{
#ifdef AOCL_ENABLE_LOG
    AOCLFFTZ_LOG_UNFORMATTED(TRACE, TRACE, "Enter");
#endif
    INTP *in_strides = strides->in_strides;
    INTP *out_strides = strides->out_strides;
    INTP v_in_stride = strides->v_in_stride;
    INTP v_out_stride = strides->v_out_stride;

    // temp variable to store power (constant_multiplier)
    FLOAT pow_cm[2] = {0.0, 0.0};
    // temp variable to store pow_cm * input
    FLOAT temp_out[2] = {0.0, 0.0};
    // buffer to store intermediate CMUL results
    FLOAT cmul_temp[2] = {0.0, 0.0};
    // buffer to store intermediate CPOW results
    FLOAT cpow_temp[2] = {0.0, 0.0};
    // buffer to store current input
    FLOAT *in_fr = (FLOAT *)in_real;
    FLOAT *in_fi = (FLOAT *)in_imag;
    FLOAT *input_r = (FLOAT *)in_real;
    FLOAT *input_i = (FLOAT *)in_imag;
    // buffer to store current output
    FLOAT *output_r = (FLOAT *)out_real;
    FLOAT *output_i = (FLOAT *)out_imag;
    FLOAT *out_fr = (FLOAT *)out_real;
    FLOAT *out_fi = (FLOAT *)out_imag;
    // local buffer to store input
    FLOAT local_in[RADIX_15][2] = {0};

    for (INTP i = 0; i < n; i++)
    {
        /******************** load input **********************/
        input_r = in_fr;
        input_i = in_fi;
        LOAD_INPUT(input_r, input_i, local_in[0]);
        LOAD_INPUT(input_r + in_strides[1], input_i + in_strides[1],
                   local_in[1]);
        LOAD_INPUT(input_r + in_strides[2], input_i + in_strides[2],
                   local_in[2]);
        LOAD_INPUT(input_r + in_strides[3], input_i + in_strides[3],
                   local_in[3]);
        LOAD_INPUT(input_r + in_strides[4], input_i + in_strides[4],
                   local_in[4]);
        LOAD_INPUT(input_r + in_strides[5], input_i + in_strides[5],
                   local_in[5]);
        LOAD_INPUT(input_r + in_strides[6], input_i + in_strides[6],
                   local_in[6]);
        LOAD_INPUT(input_r + in_strides[7], input_i + in_strides[7],
                   local_in[7]);
        LOAD_INPUT(input_r + in_strides[8], input_i + in_strides[8],
                   local_in[8]);
        LOAD_INPUT(input_r + in_strides[9], input_i + in_strides[9],
                   local_in[9]);
        LOAD_INPUT(input_r + in_strides[10], input_i + in_strides[10],
                   local_in[10]);
        LOAD_INPUT(input_r + in_strides[11], input_i + in_strides[11],
                   local_in[11]);
        LOAD_INPUT(input_r + in_strides[12], input_i + in_strides[12],
                   local_in[12]);
        LOAD_INPUT(input_r + in_strides[13], input_i + in_strides[13],
                   local_in[13]);
        LOAD_INPUT(input_r + in_strides[14], input_i + in_strides[14],
                   local_in[14]);

        output_r = out_fr;
        output_i = out_fi;
        /******************** Output 15i ********************/
        pow_cm[0] = pow_cm[1] = temp_out[0] = temp_out[1] = 0.0;
        // CADD since (constant_multiplier)^0 = 1
        CADD(local_in[0], temp_out, temp_out);
        CADD(local_in[1], temp_out, temp_out);
        CADD(local_in[2], temp_out, temp_out);
        CADD(local_in[3], temp_out, temp_out);
        CADD(local_in[4], temp_out, temp_out);
        CADD(local_in[5], temp_out, temp_out);
        CADD(local_in[6], temp_out, temp_out);
        CADD(local_in[7], temp_out, temp_out);
        CADD(local_in[8], temp_out, temp_out);
        CADD(local_in[9], temp_out, temp_out);
        CADD(local_in[10], temp_out, temp_out);
        CADD(local_in[11], temp_out, temp_out);
        CADD(local_in[12], temp_out, temp_out);
        CADD(local_in[13], temp_out, temp_out);
        CADD(local_in[14], temp_out, temp_out);
        STORE_OUTPUT(temp_out, output_r, output_i);

        /******************** Output 15i+1 ********************/
        pow_cm[0] = pow_cm[1] = temp_out[0] = temp_out[1] = 0.0;
        // Using CMUL_CADD since pow(constant_multiplier, 1) =
        // constant_multiplier
        CADD(local_in[0], temp_out, temp_out);
        CMUL_CADD(local_in[1], CRTM_15[1], pow_cm, temp_out, cmul_temp);
        CMUL_CADD(local_in[2], CRTM_15[2], pow_cm, temp_out, cmul_temp);
        CMUL_CADD(local_in[3], CRTM_15[3], pow_cm, temp_out, cmul_temp);
        CMUL_CADD(local_in[4], CRTM_15[4], pow_cm, temp_out, cmul_temp);
        CMUL_CADD(local_in[5], CRTM_15[5], pow_cm, temp_out, cmul_temp);
        CMUL_CADD(local_in[6], CRTM_15[6], pow_cm, temp_out, cmul_temp);
        CMUL_CADD(local_in[7], CRTM_15[7], pow_cm, temp_out, cmul_temp);
        CMUL_CADD(local_in[8], CRTM_15[8], pow_cm, temp_out, cmul_temp);
        CMUL_CADD(local_in[9], CRTM_15[9], pow_cm, temp_out, cmul_temp);
        CMUL_CADD(local_in[10], CRTM_15[10], pow_cm, temp_out, cmul_temp);
        CMUL_CADD(local_in[11], CRTM_15[11], pow_cm, temp_out, cmul_temp);
        CMUL_CADD(local_in[12], CRTM_15[12], pow_cm, temp_out, cmul_temp);
        CMUL_CADD(local_in[13], CRTM_15[13], pow_cm, temp_out, cmul_temp);
        CMUL_CADD(local_in[14], CRTM_15[14], pow_cm, temp_out, cmul_temp);
        STORE_OUTPUT(temp_out, output_r + out_strides[1],
                     output_i + out_strides[1]);

        /******************** Output 15i+2 ********************/
        pow_cm[0] = pow_cm[1] = temp_out[0] = temp_out[1] = 0.0;
        CADD(local_in[0], temp_out, temp_out);
        CPOW(CRTM_15[1], 2, pow_cm, cmul_temp, cpow_temp);
        CMUL_CADD_CPOW(local_in[1], CRTM_15[2], 2, pow_cm, temp_out, cmul_temp,
                       cpow_temp);
        CMUL_CADD_CPOW(local_in[2], CRTM_15[3], 2, pow_cm, temp_out, cmul_temp,
                       cpow_temp);
        CMUL_CADD_CPOW(local_in[3], CRTM_15[4], 2, pow_cm, temp_out, cmul_temp,
                       cpow_temp);
        CMUL_CADD_CPOW(local_in[4], CRTM_15[5], 2, pow_cm, temp_out, cmul_temp,
                       cpow_temp);
        CMUL_CADD_CPOW(local_in[5], CRTM_15[6], 2, pow_cm, temp_out, cmul_temp,
                       cpow_temp);
        CMUL_CADD_CPOW(local_in[6], CRTM_15[7], 2, pow_cm, temp_out, cmul_temp,
                       cpow_temp);
        CMUL_CADD_CPOW(local_in[7], CRTM_15[8], 2, pow_cm, temp_out, cmul_temp,
                       cpow_temp);
        CMUL_CADD_CPOW(local_in[8], CRTM_15[9], 2, pow_cm, temp_out, cmul_temp,
                       cpow_temp);
        CMUL_CADD_CPOW(local_in[9], CRTM_15[10], 2, pow_cm, temp_out, cmul_temp,
                       cpow_temp);
        CMUL_CADD_CPOW(local_in[10], CRTM_15[11], 2, pow_cm, temp_out,
                       cmul_temp, cpow_temp);
        CMUL_CADD_CPOW(local_in[11], CRTM_15[12], 2, pow_cm, temp_out,
                       cmul_temp, cpow_temp);
        CMUL_CADD_CPOW(local_in[12], CRTM_15[13], 2, pow_cm, temp_out,
                       cmul_temp, cpow_temp);
        CMUL_CADD_CPOW(local_in[13], CRTM_15[14], 2, pow_cm, temp_out,
                       cmul_temp, cpow_temp);
        CMUL_CADD(local_in[14], pow_cm, pow_cm, temp_out, cmul_temp);
        STORE_OUTPUT(temp_out, output_r + out_strides[2],
                     output_i + out_strides[2]);

        /******************** Output 15i+3 ********************/
        pow_cm[0] = pow_cm[1] = temp_out[0] = temp_out[1] = 0.0;
        CADD(local_in[0], temp_out, temp_out);
        CPOW(CRTM_15[1], 3, pow_cm, cmul_temp, cpow_temp);
        CMUL_CADD_CPOW(local_in[1], CRTM_15[2], 3, pow_cm, temp_out, cmul_temp,
                       cpow_temp);
        CMUL_CADD_CPOW(local_in[2], CRTM_15[3], 3, pow_cm, temp_out, cmul_temp,
                       cpow_temp);
        CMUL_CADD_CPOW(local_in[3], CRTM_15[4], 3, pow_cm, temp_out, cmul_temp,
                       cpow_temp);
        CMUL_CADD_CPOW(local_in[4], CRTM_15[5], 3, pow_cm, temp_out, cmul_temp,
                       cpow_temp);
        CMUL_CADD_CPOW(local_in[5], CRTM_15[6], 3, pow_cm, temp_out, cmul_temp,
                       cpow_temp);
        CMUL_CADD_CPOW(local_in[6], CRTM_15[7], 3, pow_cm, temp_out, cmul_temp,
                       cpow_temp);
        CMUL_CADD_CPOW(local_in[7], CRTM_15[8], 3, pow_cm, temp_out, cmul_temp,
                       cpow_temp);
        CMUL_CADD_CPOW(local_in[8], CRTM_15[9], 3, pow_cm, temp_out, cmul_temp,
                       cpow_temp);
        CMUL_CADD_CPOW(local_in[9], CRTM_15[10], 3, pow_cm, temp_out, cmul_temp,
                       cpow_temp);
        CMUL_CADD_CPOW(local_in[10], CRTM_15[11], 3, pow_cm, temp_out,
                       cmul_temp, cpow_temp);
        CMUL_CADD_CPOW(local_in[11], CRTM_15[12], 3, pow_cm, temp_out,
                       cmul_temp, cpow_temp);
        CMUL_CADD_CPOW(local_in[12], CRTM_15[13], 3, pow_cm, temp_out,
                       cmul_temp, cpow_temp);
        CMUL_CADD_CPOW(local_in[13], CRTM_15[14], 3, pow_cm, temp_out,
                       cmul_temp, cpow_temp);
        CMUL_CADD(local_in[14], pow_cm, pow_cm, temp_out, cmul_temp);
        STORE_OUTPUT(temp_out, output_r + out_strides[3],
                     output_i + out_strides[3]);

        /******************** Output 15i+4 ********************/
        pow_cm[0] = pow_cm[1] = temp_out[0] = temp_out[1] = 0.0;
        CADD(local_in[0], temp_out, temp_out);
        CPOW(CRTM_15[1], 4, pow_cm, cmul_temp, cpow_temp);
        CMUL_CADD_CPOW(local_in[1], CRTM_15[2], 4, pow_cm, temp_out, cmul_temp,
                       cpow_temp);
        CMUL_CADD_CPOW(local_in[2], CRTM_15[3], 4, pow_cm, temp_out, cmul_temp,
                       cpow_temp);
        CMUL_CADD_CPOW(local_in[3], CRTM_15[4], 4, pow_cm, temp_out, cmul_temp,
                       cpow_temp);
        CMUL_CADD_CPOW(local_in[4], CRTM_15[5], 4, pow_cm, temp_out, cmul_temp,
                       cpow_temp);
        CMUL_CADD_CPOW(local_in[5], CRTM_15[6], 4, pow_cm, temp_out, cmul_temp,
                       cpow_temp);
        CMUL_CADD_CPOW(local_in[6], CRTM_15[7], 4, pow_cm, temp_out, cmul_temp,
                       cpow_temp);
        CMUL_CADD_CPOW(local_in[7], CRTM_15[8], 4, pow_cm, temp_out, cmul_temp,
                       cpow_temp);
        CMUL_CADD_CPOW(local_in[8], CRTM_15[9], 4, pow_cm, temp_out, cmul_temp,
                       cpow_temp);
        CMUL_CADD_CPOW(local_in[9], CRTM_15[10], 4, pow_cm, temp_out, cmul_temp,
                       cpow_temp);
        CMUL_CADD_CPOW(local_in[10], CRTM_15[11], 4, pow_cm, temp_out,
                       cmul_temp, cpow_temp);
        CMUL_CADD_CPOW(local_in[11], CRTM_15[12], 4, pow_cm, temp_out,
                       cmul_temp, cpow_temp);
        CMUL_CADD_CPOW(local_in[12], CRTM_15[13], 4, pow_cm, temp_out,
                       cmul_temp, cpow_temp);
        CMUL_CADD_CPOW(local_in[13], CRTM_15[14], 4, pow_cm, temp_out,
                       cmul_temp, cpow_temp);
        CMUL_CADD(local_in[14], pow_cm, pow_cm, temp_out, cmul_temp);
        STORE_OUTPUT(temp_out, output_r + out_strides[4],
                     output_i + out_strides[4]);

        /******************** Output 15i+5 ********************/
        pow_cm[0] = pow_cm[1] = temp_out[0] = temp_out[1] = 0.0;
        CADD(local_in[0], temp_out, temp_out);
        CPOW(CRTM_15[1], 5, pow_cm, cmul_temp, cpow_temp);
        CMUL_CADD_CPOW(local_in[1], CRTM_15[2], 5, pow_cm, temp_out, cmul_temp,
                       cpow_temp);
        CMUL_CADD_CPOW(local_in[2], CRTM_15[3], 5, pow_cm, temp_out, cmul_temp,
                       cpow_temp);
        CMUL_CADD_CPOW(local_in[3], CRTM_15[4], 5, pow_cm, temp_out, cmul_temp,
                       cpow_temp);
        CMUL_CADD_CPOW(local_in[4], CRTM_15[5], 5, pow_cm, temp_out, cmul_temp,
                       cpow_temp);
        CMUL_CADD_CPOW(local_in[5], CRTM_15[6], 5, pow_cm, temp_out, cmul_temp,
                       cpow_temp);
        CMUL_CADD_CPOW(local_in[6], CRTM_15[7], 5, pow_cm, temp_out, cmul_temp,
                       cpow_temp);
        CMUL_CADD_CPOW(local_in[7], CRTM_15[8], 5, pow_cm, temp_out, cmul_temp,
                       cpow_temp);
        CMUL_CADD_CPOW(local_in[8], CRTM_15[9], 5, pow_cm, temp_out, cmul_temp,
                       cpow_temp);
        CMUL_CADD_CPOW(local_in[9], CRTM_15[10], 5, pow_cm, temp_out, cmul_temp,
                       cpow_temp);
        CMUL_CADD_CPOW(local_in[10], CRTM_15[11], 5, pow_cm, temp_out,
                       cmul_temp, cpow_temp);
        CMUL_CADD_CPOW(local_in[11], CRTM_15[12], 5, pow_cm, temp_out,
                       cmul_temp, cpow_temp);
        CMUL_CADD_CPOW(local_in[12], CRTM_15[13], 5, pow_cm, temp_out,
                       cmul_temp, cpow_temp);
        CMUL_CADD_CPOW(local_in[13], CRTM_15[14], 5, pow_cm, temp_out,
                       cmul_temp, cpow_temp);
        CMUL_CADD(local_in[14], pow_cm, pow_cm, temp_out, cmul_temp);
        STORE_OUTPUT(temp_out, output_r + out_strides[5],
                     output_i + out_strides[5]);

        /******************** Output 15i+6 ********************/
        pow_cm[0] = pow_cm[1] = temp_out[0] = temp_out[1] = 0.0;
        CADD(local_in[0], temp_out, temp_out);
        CPOW(CRTM_15[1], 6, pow_cm, cmul_temp, cpow_temp);
        CMUL_CADD_CPOW(local_in[1], CRTM_15[2], 6, pow_cm, temp_out, cmul_temp,
                       cpow_temp);
        CMUL_CADD_CPOW(local_in[2], CRTM_15[3], 6, pow_cm, temp_out, cmul_temp,
                       cpow_temp);
        CMUL_CADD_CPOW(local_in[3], CRTM_15[4], 6, pow_cm, temp_out, cmul_temp,
                       cpow_temp);
        CMUL_CADD_CPOW(local_in[4], CRTM_15[5], 6, pow_cm, temp_out, cmul_temp,
                       cpow_temp);
        CMUL_CADD_CPOW(local_in[5], CRTM_15[6], 6, pow_cm, temp_out, cmul_temp,
                       cpow_temp);
        CMUL_CADD_CPOW(local_in[6], CRTM_15[7], 6, pow_cm, temp_out, cmul_temp,
                       cpow_temp);
        CMUL_CADD_CPOW(local_in[7], CRTM_15[8], 6, pow_cm, temp_out, cmul_temp,
                       cpow_temp);
        CMUL_CADD_CPOW(local_in[8], CRTM_15[9], 6, pow_cm, temp_out, cmul_temp,
                       cpow_temp);
        CMUL_CADD_CPOW(local_in[9], CRTM_15[10], 6, pow_cm, temp_out, cmul_temp,
                       cpow_temp);
        CMUL_CADD_CPOW(local_in[10], CRTM_15[11], 6, pow_cm, temp_out,
                       cmul_temp, cpow_temp);
        CMUL_CADD_CPOW(local_in[11], CRTM_15[12], 6, pow_cm, temp_out,
                       cmul_temp, cpow_temp);
        CMUL_CADD_CPOW(local_in[12], CRTM_15[13], 6, pow_cm, temp_out,
                       cmul_temp, cpow_temp);
        CMUL_CADD_CPOW(local_in[13], CRTM_15[14], 6, pow_cm, temp_out,
                       cmul_temp, cpow_temp);
        CMUL_CADD(local_in[14], pow_cm, pow_cm, temp_out, cmul_temp);
        STORE_OUTPUT(temp_out, output_r + out_strides[6],
                     output_i + out_strides[6]);

        /******************** Output 15i+7 ********************/
        pow_cm[0] = pow_cm[1] = temp_out[0] = temp_out[1] = 0.0;
        CADD(local_in[0], temp_out, temp_out);
        CPOW(CRTM_15[1], 7, pow_cm, cmul_temp, cpow_temp);
        CMUL_CADD_CPOW(local_in[1], CRTM_15[2], 7, pow_cm, temp_out, cmul_temp,
                       cpow_temp);
        CMUL_CADD_CPOW(local_in[2], CRTM_15[3], 7, pow_cm, temp_out, cmul_temp,
                       cpow_temp);
        CMUL_CADD_CPOW(local_in[3], CRTM_15[4], 7, pow_cm, temp_out, cmul_temp,
                       cpow_temp);
        CMUL_CADD_CPOW(local_in[4], CRTM_15[5], 7, pow_cm, temp_out, cmul_temp,
                       cpow_temp);
        CMUL_CADD_CPOW(local_in[5], CRTM_15[6], 7, pow_cm, temp_out, cmul_temp,
                       cpow_temp);
        CMUL_CADD_CPOW(local_in[6], CRTM_15[7], 7, pow_cm, temp_out, cmul_temp,
                       cpow_temp);
        CMUL_CADD_CPOW(local_in[7], CRTM_15[8], 7, pow_cm, temp_out, cmul_temp,
                       cpow_temp);
        CMUL_CADD_CPOW(local_in[8], CRTM_15[9], 7, pow_cm, temp_out, cmul_temp,
                       cpow_temp);
        CMUL_CADD_CPOW(local_in[9], CRTM_15[10], 7, pow_cm, temp_out, cmul_temp,
                       cpow_temp);
        CMUL_CADD_CPOW(local_in[10], CRTM_15[11], 7, pow_cm, temp_out,
                       cmul_temp, cpow_temp);
        CMUL_CADD_CPOW(local_in[11], CRTM_15[12], 7, pow_cm, temp_out,
                       cmul_temp, cpow_temp);
        CMUL_CADD_CPOW(local_in[12], CRTM_15[13], 7, pow_cm, temp_out,
                       cmul_temp, cpow_temp);
        CMUL_CADD_CPOW(local_in[13], CRTM_15[14], 7, pow_cm, temp_out,
                       cmul_temp, cpow_temp);
        CMUL_CADD(local_in[14], pow_cm, pow_cm, temp_out, cmul_temp);
        STORE_OUTPUT(temp_out, output_r + out_strides[7],
                     output_i + out_strides[7]);

        /******************** Output 15i+8 ********************/
        pow_cm[0] = pow_cm[1] = temp_out[0] = temp_out[1] = 0.0;
        CADD(local_in[0], temp_out, temp_out);
        CPOW(CRTM_15[1], 8, pow_cm, cmul_temp, cpow_temp);
        CMUL_CADD_CPOW(local_in[1], CRTM_15[2], 8, pow_cm, temp_out, cmul_temp,
                       cpow_temp);
        CMUL_CADD_CPOW(local_in[2], CRTM_15[3], 8, pow_cm, temp_out, cmul_temp,
                       cpow_temp);
        CMUL_CADD_CPOW(local_in[3], CRTM_15[4], 8, pow_cm, temp_out, cmul_temp,
                       cpow_temp);
        CMUL_CADD_CPOW(local_in[4], CRTM_15[5], 8, pow_cm, temp_out, cmul_temp,
                       cpow_temp);
        CMUL_CADD_CPOW(local_in[5], CRTM_15[6], 8, pow_cm, temp_out, cmul_temp,
                       cpow_temp);
        CMUL_CADD_CPOW(local_in[6], CRTM_15[7], 8, pow_cm, temp_out, cmul_temp,
                       cpow_temp);
        CMUL_CADD_CPOW(local_in[7], CRTM_15[8], 8, pow_cm, temp_out, cmul_temp,
                       cpow_temp);
        CMUL_CADD_CPOW(local_in[8], CRTM_15[9], 8, pow_cm, temp_out, cmul_temp,
                       cpow_temp);
        CMUL_CADD_CPOW(local_in[9], CRTM_15[10], 8, pow_cm, temp_out, cmul_temp,
                       cpow_temp);
        CMUL_CADD_CPOW(local_in[10], CRTM_15[11], 8, pow_cm, temp_out,
                       cmul_temp, cpow_temp);
        CMUL_CADD_CPOW(local_in[11], CRTM_15[12], 8, pow_cm, temp_out,
                       cmul_temp, cpow_temp);
        CMUL_CADD_CPOW(local_in[12], CRTM_15[13], 8, pow_cm, temp_out,
                       cmul_temp, cpow_temp);
        CMUL_CADD_CPOW(local_in[13], CRTM_15[14], 8, pow_cm, temp_out,
                       cmul_temp, cpow_temp);
        CMUL_CADD(local_in[14], pow_cm, pow_cm, temp_out, cmul_temp);
        STORE_OUTPUT(temp_out, output_r + out_strides[8],
                     output_i + out_strides[8]);

        /******************** Output 15i+9 ********************/
        pow_cm[0] = pow_cm[1] = temp_out[0] = temp_out[1] = 0.0;
        CADD(local_in[0], temp_out, temp_out);
        CPOW(CRTM_15[1], 9, pow_cm, cmul_temp, cpow_temp);
        CMUL_CADD_CPOW(local_in[1], CRTM_15[2], 9, pow_cm, temp_out, cmul_temp,
                       cpow_temp);
        CMUL_CADD_CPOW(local_in[2], CRTM_15[3], 9, pow_cm, temp_out, cmul_temp,
                       cpow_temp);
        CMUL_CADD_CPOW(local_in[3], CRTM_15[4], 9, pow_cm, temp_out, cmul_temp,
                       cpow_temp);
        CMUL_CADD_CPOW(local_in[4], CRTM_15[5], 9, pow_cm, temp_out, cmul_temp,
                       cpow_temp);
        CMUL_CADD_CPOW(local_in[5], CRTM_15[6], 9, pow_cm, temp_out, cmul_temp,
                       cpow_temp);
        CMUL_CADD_CPOW(local_in[6], CRTM_15[7], 9, pow_cm, temp_out, cmul_temp,
                       cpow_temp);
        CMUL_CADD_CPOW(local_in[7], CRTM_15[8], 9, pow_cm, temp_out, cmul_temp,
                       cpow_temp);
        CMUL_CADD_CPOW(local_in[8], CRTM_15[9], 9, pow_cm, temp_out, cmul_temp,
                       cpow_temp);
        CMUL_CADD_CPOW(local_in[9], CRTM_15[10], 9, pow_cm, temp_out, cmul_temp,
                       cpow_temp);
        CMUL_CADD_CPOW(local_in[10], CRTM_15[11], 9, pow_cm, temp_out,
                       cmul_temp, cpow_temp);
        CMUL_CADD_CPOW(local_in[11], CRTM_15[12], 9, pow_cm, temp_out,
                       cmul_temp, cpow_temp);
        CMUL_CADD_CPOW(local_in[12], CRTM_15[13], 9, pow_cm, temp_out,
                       cmul_temp, cpow_temp);
        CMUL_CADD_CPOW(local_in[13], CRTM_15[14], 9, pow_cm, temp_out,
                       cmul_temp, cpow_temp);
        CMUL_CADD(local_in[14], pow_cm, pow_cm, temp_out, cmul_temp);
        STORE_OUTPUT(temp_out, output_r + out_strides[9],
                     output_i + out_strides[9]);

        /******************** Output 15i+10 ********************/
        pow_cm[0] = pow_cm[1] = temp_out[0] = temp_out[1] = 0.0;
        CADD(local_in[0], temp_out, temp_out);
        CPOW(CRTM_15[1], 10, pow_cm, cmul_temp, cpow_temp);
        CMUL_CADD_CPOW(local_in[1], CRTM_15[2], 10, pow_cm, temp_out, cmul_temp,
                       cpow_temp);
        CMUL_CADD_CPOW(local_in[2], CRTM_15[3], 10, pow_cm, temp_out, cmul_temp,
                       cpow_temp);
        CMUL_CADD_CPOW(local_in[3], CRTM_15[4], 10, pow_cm, temp_out, cmul_temp,
                       cpow_temp);
        CMUL_CADD_CPOW(local_in[4], CRTM_15[5], 10, pow_cm, temp_out, cmul_temp,
                       cpow_temp);
        CMUL_CADD_CPOW(local_in[5], CRTM_15[6], 10, pow_cm, temp_out, cmul_temp,
                       cpow_temp);
        CMUL_CADD_CPOW(local_in[6], CRTM_15[7], 10, pow_cm, temp_out, cmul_temp,
                       cpow_temp);
        CMUL_CADD_CPOW(local_in[7], CRTM_15[8], 10, pow_cm, temp_out, cmul_temp,
                       cpow_temp);
        CMUL_CADD_CPOW(local_in[8], CRTM_15[9], 10, pow_cm, temp_out, cmul_temp,
                       cpow_temp);
        CMUL_CADD_CPOW(local_in[9], CRTM_15[10], 10, pow_cm, temp_out,
                       cmul_temp, cpow_temp);
        CMUL_CADD_CPOW(local_in[10], CRTM_15[11], 10, pow_cm, temp_out,
                       cmul_temp, cpow_temp);
        CMUL_CADD_CPOW(local_in[11], CRTM_15[12], 10, pow_cm, temp_out,
                       cmul_temp, cpow_temp);
        CMUL_CADD_CPOW(local_in[12], CRTM_15[13], 10, pow_cm, temp_out,
                       cmul_temp, cpow_temp);
        CMUL_CADD_CPOW(local_in[13], CRTM_15[14], 10, pow_cm, temp_out,
                       cmul_temp, cpow_temp);
        CMUL_CADD(local_in[14], pow_cm, pow_cm, temp_out, cmul_temp);
        STORE_OUTPUT(temp_out, output_r + out_strides[10],
                     output_i + out_strides[10]);

        /******************** Output 15i+11 ********************/
        pow_cm[0] = pow_cm[1] = temp_out[0] = temp_out[1] = 0.0;
        CADD(local_in[0], temp_out, temp_out);
        CPOW(CRTM_15[1], 11, pow_cm, cmul_temp, cpow_temp);
        CMUL_CADD_CPOW(local_in[1], CRTM_15[2], 11, pow_cm, temp_out, cmul_temp,
                       cpow_temp);
        CMUL_CADD_CPOW(local_in[2], CRTM_15[3], 11, pow_cm, temp_out, cmul_temp,
                       cpow_temp);
        CMUL_CADD_CPOW(local_in[3], CRTM_15[4], 11, pow_cm, temp_out, cmul_temp,
                       cpow_temp);
        CMUL_CADD_CPOW(local_in[4], CRTM_15[5], 11, pow_cm, temp_out, cmul_temp,
                       cpow_temp);
        CMUL_CADD_CPOW(local_in[5], CRTM_15[6], 11, pow_cm, temp_out, cmul_temp,
                       cpow_temp);
        CMUL_CADD_CPOW(local_in[6], CRTM_15[7], 11, pow_cm, temp_out, cmul_temp,
                       cpow_temp);
        CMUL_CADD_CPOW(local_in[7], CRTM_15[8], 11, pow_cm, temp_out, cmul_temp,
                       cpow_temp);
        CMUL_CADD_CPOW(local_in[8], CRTM_15[9], 11, pow_cm, temp_out, cmul_temp,
                       cpow_temp);
        CMUL_CADD_CPOW(local_in[9], CRTM_15[10], 11, pow_cm, temp_out,
                       cmul_temp, cpow_temp);
        CMUL_CADD_CPOW(local_in[10], CRTM_15[11], 11, pow_cm, temp_out,
                       cmul_temp, cpow_temp);
        CMUL_CADD_CPOW(local_in[11], CRTM_15[12], 11, pow_cm, temp_out,
                       cmul_temp, cpow_temp);
        CMUL_CADD_CPOW(local_in[12], CRTM_15[13], 11, pow_cm, temp_out,
                       cmul_temp, cpow_temp);
        CMUL_CADD_CPOW(local_in[13], CRTM_15[14], 11, pow_cm, temp_out,
                       cmul_temp, cpow_temp);
        CMUL_CADD(local_in[14], pow_cm, pow_cm, temp_out, cmul_temp);
        STORE_OUTPUT(temp_out, output_r + out_strides[11],
                     output_i + out_strides[11]);

        /******************** Output 15i+12 ********************/
        pow_cm[0] = pow_cm[1] = temp_out[0] = temp_out[1] = 0.0;
        CADD(local_in[0], temp_out, temp_out);
        CPOW(CRTM_15[1], 12, pow_cm, cmul_temp, cpow_temp);
        CMUL_CADD_CPOW(local_in[1], CRTM_15[2], 12, pow_cm, temp_out, cmul_temp,
                       cpow_temp);
        CMUL_CADD_CPOW(local_in[2], CRTM_15[3], 12, pow_cm, temp_out, cmul_temp,
                       cpow_temp);
        CMUL_CADD_CPOW(local_in[3], CRTM_15[4], 12, pow_cm, temp_out, cmul_temp,
                       cpow_temp);
        CMUL_CADD_CPOW(local_in[4], CRTM_15[5], 12, pow_cm, temp_out, cmul_temp,
                       cpow_temp);
        CMUL_CADD_CPOW(local_in[5], CRTM_15[6], 12, pow_cm, temp_out, cmul_temp,
                       cpow_temp);
        CMUL_CADD_CPOW(local_in[6], CRTM_15[7], 12, pow_cm, temp_out, cmul_temp,
                       cpow_temp);
        CMUL_CADD_CPOW(local_in[7], CRTM_15[8], 12, pow_cm, temp_out, cmul_temp,
                       cpow_temp);
        CMUL_CADD_CPOW(local_in[8], CRTM_15[9], 12, pow_cm, temp_out, cmul_temp,
                       cpow_temp);
        CMUL_CADD_CPOW(local_in[9], CRTM_15[10], 12, pow_cm, temp_out,
                       cmul_temp, cpow_temp);
        CMUL_CADD_CPOW(local_in[10], CRTM_15[11], 12, pow_cm, temp_out,
                       cmul_temp, cpow_temp);
        CMUL_CADD_CPOW(local_in[11], CRTM_15[12], 12, pow_cm, temp_out,
                       cmul_temp, cpow_temp);
        CMUL_CADD_CPOW(local_in[12], CRTM_15[13], 12, pow_cm, temp_out,
                       cmul_temp, cpow_temp);
        CMUL_CADD_CPOW(local_in[13], CRTM_15[14], 12, pow_cm, temp_out,
                       cmul_temp, cpow_temp);
        CMUL_CADD(local_in[14], pow_cm, pow_cm, temp_out, cmul_temp);
        STORE_OUTPUT(temp_out, output_r + out_strides[12],
                     output_i + out_strides[12]);

        /******************** Output 15i+13 ********************/
        pow_cm[0] = pow_cm[1] = temp_out[0] = temp_out[1] = 0.0;
        CADD(local_in[0], temp_out, temp_out);
        CPOW(CRTM_15[1], 13, pow_cm, cmul_temp, cpow_temp);
        CMUL_CADD_CPOW(local_in[1], CRTM_15[2], 13, pow_cm, temp_out, cmul_temp,
                       cpow_temp);
        CMUL_CADD_CPOW(local_in[2], CRTM_15[3], 13, pow_cm, temp_out, cmul_temp,
                       cpow_temp);
        CMUL_CADD_CPOW(local_in[3], CRTM_15[4], 13, pow_cm, temp_out, cmul_temp,
                       cpow_temp);
        CMUL_CADD_CPOW(local_in[4], CRTM_15[5], 13, pow_cm, temp_out, cmul_temp,
                       cpow_temp);
        CMUL_CADD_CPOW(local_in[5], CRTM_15[6], 13, pow_cm, temp_out, cmul_temp,
                       cpow_temp);
        CMUL_CADD_CPOW(local_in[6], CRTM_15[7], 13, pow_cm, temp_out, cmul_temp,
                       cpow_temp);
        CMUL_CADD_CPOW(local_in[7], CRTM_15[8], 13, pow_cm, temp_out, cmul_temp,
                       cpow_temp);
        CMUL_CADD_CPOW(local_in[8], CRTM_15[9], 13, pow_cm, temp_out, cmul_temp,
                       cpow_temp);
        CMUL_CADD_CPOW(local_in[9], CRTM_15[10], 13, pow_cm, temp_out,
                       cmul_temp, cpow_temp);
        CMUL_CADD_CPOW(local_in[10], CRTM_15[11], 13, pow_cm, temp_out,
                       cmul_temp, cpow_temp);
        CMUL_CADD_CPOW(local_in[11], CRTM_15[12], 13, pow_cm, temp_out,
                       cmul_temp, cpow_temp);
        CMUL_CADD_CPOW(local_in[12], CRTM_15[13], 13, pow_cm, temp_out,
                       cmul_temp, cpow_temp);
        CMUL_CADD_CPOW(local_in[13], CRTM_15[14], 13, pow_cm, temp_out,
                       cmul_temp, cpow_temp);
        CMUL_CADD(local_in[14], pow_cm, pow_cm, temp_out, cmul_temp);
        STORE_OUTPUT(temp_out, output_r + out_strides[13],
                     output_i + out_strides[13]);

        /******************** Output 15i+14 ********************/
        pow_cm[0] = pow_cm[1] = temp_out[0] = temp_out[1] = 0.0;
        CADD(local_in[0], temp_out, temp_out);
        CPOW(CRTM_15[1], 14, pow_cm, cmul_temp, cpow_temp);
        CMUL_CADD_CPOW(local_in[1], CRTM_15[2], 14, pow_cm, temp_out, cmul_temp,
                       cpow_temp);
        CMUL_CADD_CPOW(local_in[2], CRTM_15[3], 14, pow_cm, temp_out, cmul_temp,
                       cpow_temp);
        CMUL_CADD_CPOW(local_in[3], CRTM_15[4], 14, pow_cm, temp_out, cmul_temp,
                       cpow_temp);
        CMUL_CADD_CPOW(local_in[4], CRTM_15[5], 14, pow_cm, temp_out, cmul_temp,
                       cpow_temp);
        CMUL_CADD_CPOW(local_in[5], CRTM_15[6], 14, pow_cm, temp_out, cmul_temp,
                       cpow_temp);
        CMUL_CADD_CPOW(local_in[6], CRTM_15[7], 14, pow_cm, temp_out, cmul_temp,
                       cpow_temp);
        CMUL_CADD_CPOW(local_in[7], CRTM_15[8], 14, pow_cm, temp_out, cmul_temp,
                       cpow_temp);
        CMUL_CADD_CPOW(local_in[8], CRTM_15[9], 14, pow_cm, temp_out, cmul_temp,
                       cpow_temp);
        CMUL_CADD_CPOW(local_in[9], CRTM_15[10], 14, pow_cm, temp_out,
                       cmul_temp, cpow_temp);
        CMUL_CADD_CPOW(local_in[10], CRTM_15[11], 14, pow_cm, temp_out,
                       cmul_temp, cpow_temp);
        CMUL_CADD_CPOW(local_in[11], CRTM_15[12], 14, pow_cm, temp_out,
                       cmul_temp, cpow_temp);
        CMUL_CADD_CPOW(local_in[12], CRTM_15[13], 14, pow_cm, temp_out,
                       cmul_temp, cpow_temp);
        CMUL_CADD_CPOW(local_in[13], CRTM_15[14], 14, pow_cm, temp_out,
                       cmul_temp, cpow_temp);
        CMUL_CADD(local_in[14], pow_cm, pow_cm, temp_out, cmul_temp);
        STORE_OUTPUT(temp_out, output_r + out_strides[14],
                     output_i + out_strides[14]);

        // next set
        in_fr += v_in_stride;
        in_fi += v_in_stride;
        out_fr += v_out_stride;
        out_fi += v_out_stride;
    }
#ifdef AOCL_ENABLE_LOG
    AOCLFFTZ_LOG_UNFORMATTED(TRACE, TRACE, "Exit");
#endif
}
#endif // USE_OPT_KERNEL_VARIANT

kfft_ register_kernel_fft15c(UINT8 precision, UINT8 direction /* unused */)
{
    if (precision == DT_FLOAT)
    {
        return fft15c_fp32;
    }
    else if (precision == DT_DOUBLE)
    {
        return fft15c_fp64;
    }
    else
    {
        return NULL;
    }
}
