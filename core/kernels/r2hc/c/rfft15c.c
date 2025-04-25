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

/** @file rfft15c.c
 *
 *  @brief Radix-15 r2hc Real-FFT kernel with scalar operations in C
 *
 *  This file contains the DIT radix-15 real-to-halfcomplex implementations
 *  using scalar operations for single-precision and double-precision inputs.
 *
 *  @author Jeya R
 */

#include "core/kernels/kernel.h"

static const ops_cycles_t ops_cnt[NUM_PRECISIONS][NUM_FFT_DIRS] =
                                                    {{{0, 25, 64, 30, 0, 0},
                                                      {0, 27, 64, 30, 0, 0}},
                                                     {{0, 25, 64, 30, 0, 0},
                                                      {0, 27, 64, 30, 0, 0}}};

ops_cycles_t get_ops_cnt_r2hc_rfft15c(UINT8 precision, UINT8 direction)
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

static VOID r2hc_rfft15c_fp32_fwd(VOID *in_real, VOID *in_imag, VOID *out_real,
                                  VOID *out_imag, INTP n,
                                  aoclfftz_strides_t *strides, UINT8 flag)
{
#ifdef AOCL_ENABLE_LOG
    AOCLFFTZ_LOG_UNFORMATTED(TRACE, TRACE, "Enter");
#endif

    const FLOAT CRTM_15_1 =
        +0.55901699437494742410229341718281905886015458990288f;
    const FLOAT CRTM_15_2 =
        +0.25000000000000000000000000000000000000000000000000f;
    const FLOAT CRTM_15_3 =
        +0.95105651629515357211643933337938214340569863400000f;
    const FLOAT CRTM_15_4 =
        +0.58778525229247301629891039327884007596190389052978f;
    const FLOAT CRTM_15_5 =
        +0.50000000000000000000000000000000000000000000000000f;
    const FLOAT CRTM_15_6 =
        +0.86602540378443864676372317075293618347140262690519f;
    // Below CRTMs are the product of the above CRTMs, Precomputed to save
    // multiplications on the fly.
    // CRTM_15_7 = CRTM_15_6 * CRTM_15_4
    const FLOAT CRTM_15_7 =
        +0.50903696045512706468216979248996715975105181034577f;
    // CRTM_15_8 = CRTM_15_6 * CRTM_15_3
    const FLOAT CRTM_15_8 =
        +0.82363910354633184270744116161596601637855195182647f;
    // CRTM_15_9 = CRTM_15_6 * CRTM_15_1
    const FLOAT CRTM_15_9 =
        +0.48412291827592710612024388657479988457787393064252f;
    // CRTM_15_10 = CRTM_15_6 * CRTM_15_2
    const FLOAT CRTM_15_10 =
        +0.21650635094610964914707551542960572987794876098633f;
    FLOAT *in = (FLOAT *)in_real;
    FLOAT *out = (FLOAT *)out_real;
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
        FLOAT in0, in1, in2, in3, s0, s1, s2, s3, s4, s5, s6, s7, s8, s9, s10,
              s11, s12, s13, s15, s16, s17, s19, s20, s21, s22, s23, s24,
              s25, s26, s27, s28, s29, s30, s31, t0, t1, t2, t3;

        in0 = *in;
        in1 = in[in_strides[2]];
        in2 = in[in_strides[7]];
        in3 = in[in_strides[12]];

        s0 = in1 + in2;
        s1 = in1 - in2;
        s2 = in3 + s0;
        s3 = in3 - (CRTM_15_5 * s0);

        in1 = in[in_strides[8]];
        in2 = in[in_strides[13]];
        in3 = in[in_strides[3]];

        s0 = in1 + in2;
        s4 = in2 - in1;
        s5 = in3 + s0;
        s6 = in3 - (CRTM_15_5 * s0);

        s23 = s6 + s3;
        s24 = s6 - s3;
        s26 = s4 + s1;
        s27 = s4 - s1;

        in1 = in[in_strides[1]];
        in2 = in[in_strides[11]];
        in3 = in[in_strides[6]];

        s0 = in1 + in2;
        s1 = in2 - in1;
        s7 = in3 + s0;
        s3 = in3 - (CRTM_15_5 * s0);

        in1 = in[in_strides[4]];
        in2 = in[in_strides[14]];
        in3 = in[in_strides[9]];

        s0 = in1 + in2;
        s4 = in2 - in1;
        s8 = in3 + s0;
        s6 = in3 - (CRTM_15_5 * s0);

        s28 = s6 + s3;
        s29 = s3 - s6;
        s30 = s4 + s1;
        s31 = s1 - s4;

        in1 = in[in_strides[5]];
        in2 = in[in_strides[10]];

        s0 = in1 + in2;
        t0 = CRTM_15_6 * (in2 - in1);
        s9 = in0 + s0;
        s3 = in0 - (CRTM_15_5 * s0);

        s11 = s8 + s7;
        s12 = s2 + s5;
        s19 = s5 - s2;
        s20 = s8 - s7;
        s13 = s11 + s12;
        t1 = CRTM_15_1 * (s11 - s12);
        s15 = s9 - (CRTM_15_2 * s13);

        *out = s9 + s13;

        out[out_strides[5]] = s15 + t1;
        out[out_strides[6]] = (CRTM_15_3 * s20) + (CRTM_15_4 * s19);

        out[out_strides[11]] = s15 - t1;
        out[out_strides[12]] = (CRTM_15_4 * s20) - (CRTM_15_3 * s19);

        t2 = CRTM_15_1 * (s28 - s23);
        s13 = s28 + s23;

        out[out_strides[9]] = s13 + s3;

        t3 = CRTM_15_9 * (s30 + s27);
        s17 = s30 - s27;

        out[out_strides[10]] = CRTM_15_6 * s17 - t0;

        s15 = s3 - (CRTM_15_2 * s13);
        s20 = t0 + (CRTM_15_10 * s17);
        s21 = s15 - t2;
        s25 = (CRTM_15_8 * s26) - (CRTM_15_7 * s31);

        out[out_strides[1]] = s21 + s25;
        out[out_strides[7]] = s21 - s25;

        s21 = s15 + t2;
        s23 = (CRTM_15_8 * s31) + (CRTM_15_7 * s26);

        out[out_strides[13]] = s21 + s23;

        s22 = s20 + t3;
        s16 = (CRTM_15_4 * s29) + (CRTM_15_3 * s24);

        out[out_strides[2]] = s22 - s16;
        out[out_strides[8]] = s22 + s16;

        s22 = s20 - t3;
        s10 = (CRTM_15_3 * s29) - (CRTM_15_4 * s24);
        out[out_strides[14]] = s22 + s10;

        out[out_strides[3]] = s21 - s23;
        out[out_strides[4]] = s10 - s22;

        in = in + v_in_stride;
        out = out + v_out_stride;
    }
#ifdef AOCL_ENABLE_LOG
    AOCLFFTZ_LOG_UNFORMATTED(TRACE, TRACE, "Exit");
#endif
}

static VOID r2hc_rfft15c_fp32_bwd(VOID *in_real, VOID *in_imag, VOID *out_real,
                                  VOID *out_imag, INTP n,
                                  aoclfftz_strides_t *strides, UINT8 flag)
{
#ifdef AOCL_ENABLE_LOG
    AOCLFFTZ_LOG_UNFORMATTED(TRACE, TRACE, "Enter");
#endif
    const FLOAT CRTM_15_1 = +1.118033988749894848204586834365638117720309180f;
    const FLOAT CRTM_15_2 = +0.500000000000000000000000000000000000000000000f;
    const FLOAT CRTM_15_3 = +1.902113032590307144232878666758764286811397268f;
    const FLOAT CRTM_15_4 = +1.175570504584946258337411909278145537195304875f;
    const FLOAT CRTM_15_5 = +2.000000000000000000000000000000000000000000000f;
    const FLOAT CRTM_15_6 = +0.250000000000000000000000000000000000000000000f;
    // Below CRTMs are the product of the above CRTMs, Precomputed to save
    // multiplications on the fly.
    // CRTM_15_7 = CRTM_15_6 * CRTM_15_4
    const FLOAT CRTM_15_7 =
        +1.01807392091025412936433958497993431950210362069154f;
    // CRTM_15_8 = CRTM_15_6 * CRTM_15_3
    const FLOAT CRTM_15_8 =
        +1.64727820709266368541488232323193203275710390365294f;
    // CRTM_15_9 = CRTM_15_6 * CRTM_15_1
    const FLOAT CRTM_15_9 =
        +0.96824583655185421224048777314959976915574786128504f;
    // CRTM_15_10 = CRTM_15_6 * CRTM_15_5
    const FLOAT CRTM_15_10 = +1.732050807568877293527446341505872366942805254f;

    FLOAT *in = (FLOAT *)in_real;
    FLOAT *out = (FLOAT *)out_real;
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
        FLOAT in0, in1, in2, in3, s0, s1, s2, s3, s4, s5, s6, s7, s8, s9, s10,
              s11, s12, s13, s14, s15, s16, s17, s18, s19, s20, s21, s22, s23,
              s24, s25, s26, s27, s28, s29, s30, s31, s32, s33, s34, s35, s36,
              s37, s38, s39, t0, t1, t2, t3, t4, t5;

        in0 = *in;
        in1 = in[in_strides[3]];
        in2 = in[in_strides[13]];
        in3 = in[in_strides[5]];

        s0 = in1 + in2;
        s1 = in1 - in2;
        s2 = in3 + s0;
        s3 = in3 - (CRTM_15_2 * s0);

        in1 = in[in_strides[4]];
        in2 = in[in_strides[14]];
        in3 = in[in_strides[6]];

        s0 = in1 + in2;
        s4 = in1 - in2;
        s5 = s0 - in3;
        s6 = in3 + (CRTM_15_2 * s0);

        in1 = in[in_strides[1]];
        in2 = in[in_strides[7]];
        in3 = in[in_strides[11]];

        s0 = in1 + in2;
        s7 = in2 - in1;
        s8 = in3 + s0;
        s9 = in3 - (CRTM_15_2 * s0);

        in1 = in[in_strides[2]];
        in2 = in[in_strides[8]];
        in3 = in[in_strides[12]];

        s0 = in1 - in2;
        s10 = in2 + in1;
        s11 = in3 + s0;
        s12 = in3 - (CRTM_15_2 * s0);

        in1 = in[in_strides[9]];
        in2 = in[in_strides[10]];

        s13 = in0 + CRTM_15_5 * in1;
        s14 = in0 - in1;
        s15 = s8 + s2;
        t0 = CRTM_15_1 * (s8 - s2);
        s16 = s13 - (CRTM_15_2 * s15);

        *out = s13 + CRTM_15_5 * s15;

        s17 = (CRTM_15_3 * s11) + (CRTM_15_4 * s5);
        s18 = s16 + t0;

        out[out_strides[3]] = s18 - s17;
        out[out_strides[12]] = s18 + s17;

        s19 = s16 - t0;
        s20 = (CRTM_15_4 * s11) - (CRTM_15_3 * s5);

        out[out_strides[6]] = s19 - s20;
        out[out_strides[9]] = s19 + s20;

        t1 = CRTM_15_1 * (s9 - s3);
        t2 = CRTM_15_9 * (s10 + s4);
        s21 = (s9 + s3);
        s22 = s4 - s10;
        t3 = CRTM_15_5 * s21 + s14;
        t4 = CRTM_15_10 * (s22 + in2);

        out[out_strides[5]] = t3 + t4;
        out[out_strides[10]] = t3 - t4;

        s23 = s14 - (CRTM_15_2 * s21);
        t5 = CRTM_15_10 * ((CRTM_15_6 * s22) - in2);
        s24 = t5 + s23;
        s25 = s23 - t5;
        s26 = t2 - t1;
        s27 = t1 + t2;

        s28 = s24 - s27;
        s29 = (CRTM_15_4 * s12) + (CRTM_15_3 * s6);
        s30 = (CRTM_15_8 * s1) - (CRTM_15_7 * s7);
        s31 = s30 - s29;
        s32 = s29 + s30;

        out[out_strides[1]] = s28 + s31;
        out[out_strides[4]] = s28 - s31;

        s33 = s24 + s27;
        s34 = (CRTM_15_8 * s7) + (CRTM_15_7 * s1);
        s35 = (CRTM_15_3 * s12) - (CRTM_15_4 * s6);
        s36 = s35 + s34;

        out[out_strides[7]] = s33 + s36;
        out[out_strides[13]] = s33 - s36;

        s37 = s25 - s26;
        s38 = s35 - s34;
        out[out_strides[2]] = s37 + s38;
        out[out_strides[8]] = s37 - s38;

        s39 = s25 + s26;

        out[out_strides[11]] = s39 - s32;
        out[out_strides[14]] = s39 + s32;

        in = in + v_in_stride;
        out = out + v_out_stride;
    }
#ifdef AOCL_ENABLE_LOG
    AOCLFFTZ_LOG_UNFORMATTED(TRACE, TRACE, "Exit");
#endif
}

static VOID r2hc_rfft15c_fp64_fwd(VOID *in_real, VOID *in_imag, VOID *out_real,
                                  VOID *out_imag, INTP n,
                                  aoclfftz_strides_t *strides, UINT8 flag)
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

    DOUBLE *in = (DOUBLE *)in_real;
    DOUBLE *out = (DOUBLE *)out_real;
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
        DOUBLE in0, in1, in2, in3, s0, s1, s2, s3, s4, s5, s6, s7, s8, s9, s10,
               s11, s12, s13, s15, s16, s17, s19, s20, s21, s22, s23, s24,
               s25, s26, s27, s28, s29, s30, s31, t0, t1, t2, t3;

        in0 = *in;
        in1 = in[in_strides[2]];
        in2 = in[in_strides[7]];
        in3 = in[in_strides[12]];

        s0 = in1 + in2;
        s1 = in1 - in2;
        s2 = in3 + s0;
        s3 = in3 - (CRTM_15_5 * s0);

        in1 = in[in_strides[8]];
        in2 = in[in_strides[13]];
        in3 = in[in_strides[3]];

        s0 = in1 + in2;
        s4 = in2 - in1;
        s5 = in3 + s0;
        s6 = in3 - (CRTM_15_5 * s0);

        s23 = s6 + s3;
        s24 = s6 - s3;
        s26 = s4 + s1;
        s27 = s4 - s1;

        in1 = in[in_strides[1]];
        in2 = in[in_strides[11]];
        in3 = in[in_strides[6]];

        s0 = in1 + in2;
        s1 = in2 - in1;
        s7 = in3 + s0;
        s3 = in3 - (CRTM_15_5 * s0);

        in1 = in[in_strides[4]];
        in2 = in[in_strides[14]];
        in3 = in[in_strides[9]];

        s0 = in1 + in2;
        s4 = in2 - in1;
        s8 = in3 + s0;
        s6 = in3 - (CRTM_15_5 * s0);

        s28 = s6 + s3;
        s29 = s3 - s6;
        s30 = s4 + s1;
        s31 = s1 - s4;

        in1 = in[in_strides[5]];
        in2 = in[in_strides[10]];

        s0 = in1 + in2;
        t0 = CRTM_15_6 * (in2 - in1);
        s9 = in0 + s0;
        s3 = in0 - (CRTM_15_5 * s0);

        s11 = s8 + s7;
        s12 = s2 + s5;
        s19 = s5 - s2;
        s20 = s8 - s7;
        s13 = s11 + s12;
        t1 = CRTM_15_1 * (s11 - s12);
        s15 = s9 - (CRTM_15_2 * s13);

        *out = s9 + s13;

        out[out_strides[5]] = s15 + t1;
        out[out_strides[6]] = (CRTM_15_3 * s20) + (CRTM_15_4 * s19);

        out[out_strides[11]] = s15 - t1;
        out[out_strides[12]] = (CRTM_15_4 * s20) - (CRTM_15_3 * s19);

        t2 = CRTM_15_1 * (s28 - s23);
        s13 = s28 + s23;

        out[out_strides[9]] = s13 + s3;

        t3 = CRTM_15_9 * (s30 + s27);
        s17 = s30 - s27;

        out[out_strides[10]] = CRTM_15_6 * s17 - t0;

        s15 = s3 - (CRTM_15_2 * s13);
        s20 = t0 + (CRTM_15_10 * s17);
        s21 = s15 - t2;
        s25 = (CRTM_15_8 * s26) - (CRTM_15_7 * s31);

        out[out_strides[1]] = s21 + s25;
        out[out_strides[7]] = s21 - s25;

        s21 = s15 + t2;
        s23 = (CRTM_15_8 * s31) + (CRTM_15_7 * s26);

        out[out_strides[13]] = s21 + s23;

        s22 = s20 + t3;
        s16 = (CRTM_15_4 * s29) + (CRTM_15_3 * s24);

        out[out_strides[2]] = s22 - s16;
        out[out_strides[8]] = s22 + s16;

        s22 = s20 - t3;
        s10 = (CRTM_15_3 * s29) - (CRTM_15_4 * s24);
        out[out_strides[14]] = s22 + s10;

        out[out_strides[3]] = s21 - s23;
        out[out_strides[4]] = s10 - s22;

        in = in + v_in_stride;
        out = out + v_out_stride;
    }
#ifdef AOCL_ENABLE_LOG
    AOCLFFTZ_LOG_UNFORMATTED(TRACE, TRACE, "Exit");
#endif
}

static VOID r2hc_rfft15c_fp64_bwd(VOID *in_real, VOID *in_imag, VOID *out_real,
                                  VOID *out_imag, INTP n,
                                  aoclfftz_strides_t *strides, UINT8 flag)
{
#ifdef AOCL_ENABLE_LOG
    AOCLFFTZ_LOG_UNFORMATTED(TRACE, TRACE, "Enter");
#endif
    const DOUBLE CRTM_15_1 = +1.118033988749894848204586834365638117720309180;
    const DOUBLE CRTM_15_2 = +0.500000000000000000000000000000000000000000000;
    const DOUBLE CRTM_15_3 = +1.902113032590307144232878666758764286811397268;
    const DOUBLE CRTM_15_4 = +1.175570504584946258337411909278145537195304875;
    const DOUBLE CRTM_15_5 = +2.000000000000000000000000000000000000000000000;
    const DOUBLE CRTM_15_6 = +0.250000000000000000000000000000000000000000000;
    // Below CRTMs are the product of the above CRTMs, Precomputed to save
    // multiplications on the fly.
    // CRTM_15_7 = CRTM_15_6 * CRTM_15_4
    const DOUBLE CRTM_15_7 =
        +1.01807392091025412936433958497993431950210362069154;
    // CRTM_15_8 = CRTM_15_6 * CRTM_15_3
    const DOUBLE CRTM_15_8 =
        +1.64727820709266368541488232323193203275710390365294;
    // CRTM_15_9 = CRTM_15_6 * CRTM_15_1
    const DOUBLE CRTM_15_9 =
        +0.96824583655185421224048777314959976915574786128504;
    // CRTM_15_10 = CRTM_15_6 * CRTM_15_5
    const DOUBLE CRTM_15_10 = +1.732050807568877293527446341505872366942805254;

    DOUBLE *in = (DOUBLE *)in_real;
    DOUBLE *out = (DOUBLE *)out_real;
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

        DOUBLE in0, in1, in2, in3, s0, s1, s2, s3, s4, s5, s6, s7, s8, s9, s10,
               s11, s12, s13, s14, s15, s16, s17, s18, s19, s20, s21, s22, s23,
               s24, s25, s26, s27, s28, s29, s30, s31, s32, s33, s34, s35, s36,
               s37, s38, s39, t0, t1, t2, t3, t4, t5;

        in0 = *in;
        in1 = in[in_strides[3]];
        in2 = in[in_strides[13]];
        in3 = in[in_strides[5]];

        s0 = in1 + in2;
        s1 = in1 - in2;
        s2 = in3 + s0;
        s3 = in3 - (CRTM_15_2 * s0);

        in1 = in[in_strides[4]];
        in2 = in[in_strides[14]];
        in3 = in[in_strides[6]];

        s0 = in1 + in2;
        s4 = in1 - in2;
        s5 = s0 - in3;
        s6 = in3 + (CRTM_15_2 * s0);

        in1 = in[in_strides[1]];
        in2 = in[in_strides[7]];
        in3 = in[in_strides[11]];

        s0 = in1 + in2;
        s7 = in2 - in1;
        s8 = in3 + s0;
        s9 = in3 - (CRTM_15_2 * s0);

        in1 = in[in_strides[2]];
        in2 = in[in_strides[8]];
        in3 = in[in_strides[12]];

        s0 = in1 - in2;
        s10 = in2 + in1;
        s11 = in3 + s0;
        s12 = in3 - (CRTM_15_2 * s0);

        in1 = in[in_strides[9]];
        in2 = in[in_strides[10]];

        s13 = in0 + CRTM_15_5 * in1;
        s14 = in0 - in1;
        s15 = s8 + s2;
        t0 = CRTM_15_1 * (s8 - s2);
        s16 = s13 - (CRTM_15_2 * s15);

        *out = s13 + CRTM_15_5 * s15;

        s17 = (CRTM_15_3 * s11) + (CRTM_15_4 * s5);
        s18 = s16 + t0;

        out[out_strides[3]] = s18 - s17;
        out[out_strides[12]] = s18 + s17;

        s19 = s16 - t0;
        s20 = (CRTM_15_4 * s11) - (CRTM_15_3 * s5);

        out[out_strides[6]] = s19 - s20;
        out[out_strides[9]] = s19 + s20;

        t1 = CRTM_15_1 * (s9 - s3);
        t2 = CRTM_15_9 * (s10 + s4);
        s21 = (s9 + s3);
        s22 = s4 - s10;
        t3 = CRTM_15_5 * s21 + s14;
        t4 = CRTM_15_10 * (s22 + in2);

        out[out_strides[5]] = t3 + t4;
        out[out_strides[10]] = t3 - t4;

        s23 = s14 - (CRTM_15_2 * s21);
        t5 = CRTM_15_10 * ((CRTM_15_6 * s22) - in2);
        s24 = t5 + s23;
        s25 = s23 - t5;
        s26 = t2 - t1;
        s27 = t1 + t2;

        s28 = s24 - s27;
        s29 = (CRTM_15_4 * s12) + (CRTM_15_3 * s6);
        s30 = (CRTM_15_8 * s1) - (CRTM_15_7 * s7);
        s31 = s30 - s29;
        s32 = s29 + s30;

        out[out_strides[1]] = s28 + s31;
        out[out_strides[4]] = s28 - s31;

        s33 = s24 + s27;
        s34 = (CRTM_15_8 * s7) + (CRTM_15_7 * s1);
        s35 = (CRTM_15_3 * s12) - (CRTM_15_4 * s6);
        s36 = s35 + s34;

        out[out_strides[7]] = s33 + s36;
        out[out_strides[13]] = s33 - s36;

        s37 = s25 - s26;
        s38 = s35 - s34;
        out[out_strides[2]] = s37 + s38;
        out[out_strides[8]] = s37 - s38;

        s39 = s25 + s26;

        out[out_strides[11]] = s39 - s32;
        out[out_strides[14]] = s39 + s32;

        in = in + v_in_stride;
        out = out + v_out_stride;
    }
#ifdef AOCL_ENABLE_LOG
    AOCLFFTZ_LOG_UNFORMATTED(TRACE, TRACE, "Exit");
#endif
}

kfft_ register_kernel_r2hc_rfft15c(UINT8 precision, UINT8 direction)
{
    if (direction == FORWARD_FFT_DIR)
    {
        if (precision == DT_FLOAT)
        {
            return r2hc_rfft15c_fp32_fwd;
        }
        else if (precision == DT_DOUBLE)
        {
            return r2hc_rfft15c_fp64_fwd;
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
            return r2hc_rfft15c_fp32_bwd;
        }
        else if (precision == DT_DOUBLE)
        {
            return r2hc_rfft15c_fp64_bwd;
        }
        else
        {
            return NULL;
        }
    }
}
