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
 *  @brief Radix-15 r2hc_fused Real-FFT kernel with scalar operations in C
 *
 *  This file contains the DIT radix-15 real-to-halfcomplex fused of two
 *  different implementations (Standard DFT and Shifted DFT that differs in DFT
 *  weight matrix) using scalar operations for single-precision
 *  and double-precision inputs.
 *
 *  @author Jeya R
 *  @author Jeevanantham N
 */

#include "core/kernels/kernel.h"

static const ops_cycles_t ops_cnt[NUM_PRECISIONS][NUM_FFT_DIRS] =
                                                    {{{0, 58, 136, 60, 0, 0},
                                                      {0, 53, 128, 60, 0, 0}},
                                                     {{0, 58, 136, 60, 0, 0},
                                                      {0, 53, 128, 60, 0, 0}}};

ops_cycles_t get_ops_cnt_r2hcf_rfft15c(UINT8 precision, UINT8 direction)
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

static VOID r2hcf_rfft15c_fp32_fwd(VOID *in_real, VOID *in_imag, VOID *out_real,
                                   VOID *out_imag, INTP n,
                                   aoclfftz_strides_t *strides, VOID *twd, UINT8 flag)
{
    AOCLFFTZ_LOG(DEBUG, global_logger_mode, "Enter");
    const FLOAT CRTM_15_1 =
        0.55901699437494742410229341718281905886015458990288f;
    const FLOAT CRTM_15_2 =
        0.25000000000000000000000000000000000000000000000000f;
    const FLOAT CRTM_15_3 =
        0.95105651629515357211643933337938214340569863400000f;
    const FLOAT CRTM_15_4 =
        0.58778525229247301629891039327884007596190389052978f;
    const FLOAT CRTM_15_5 =
        0.50000000000000000000000000000000000000000000000000f;
    const FLOAT CRTM_15_6 =
        0.86602540378443864676372317075293618347140262690519f;
    const FLOAT CRTM_15_11 = 0.309016994374947424102293417182819058860154590f;
    const FLOAT CRTM_15_12 = 0.809016994374947424102293417182819058860154590f;
    // Below CRTMs are the product of the above CRTMs, Precomputed to save
    // multiplications on the fly.
    // CRTM_15_7 = CRTM_15_6 * CRTM_15_4
    const FLOAT CRTM_15_7 =
        0.50903696045512706468216979248996715975105181034577f;
    // CRTM_15_8 = CRTM_15_6 * CRTM_15_3
    const FLOAT CRTM_15_8 =
        0.82363910354633184270744116161596601637855195182647f;
    // CRTM_15_9 = CRTM_15_6 * CRTM_15_1
    const FLOAT CRTM_15_9 =
        0.48412291827592710612024388657479988457787393064252f;
    // CRTM_15_10 = CRTM_15_6 * CRTM_15_2
    const FLOAT CRTM_15_10 =
        0.21650635094610964914707551542960572987794876098633f;

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
        /* Standard DFT*/
        FLOAT a_in0, a_in1, a_in2, a_in3, a_s0, a_s1, a_s2, a_s3, a_s4, a_s5,
              a_s6, a_s7, a_s8, a_s9, a_s10, a_s11, a_s12, a_s13, a_s15, a_s16,
              a_s17, a_s19, a_s20, a_s21, a_s22, a_s23, a_s24, a_s25, a_s26,
              a_s27, a_s28, a_s29, a_s30, a_s31, a_t0, a_t1, a_t2, a_t3;

        // Input point 1: x(0)
        a_in0 = *in;
        // Input point 5: x(4)
        a_in1 = in[in_strides[4]];
        // Input point 15: x(14)
        a_in2 = in[in_strides[14]];
        // Input point 25: x(24)
        a_in3 = in[in_strides[24]];

        a_s0 = a_in1 + a_in2;
        a_s1 = a_in1 - a_in2;
        a_s2 = a_in3 + a_s0;
        a_s3 = a_in3 - (CRTM_15_5 * a_s0);

        // Input point 17: x(16)
        a_in1 = in[in_strides[16]];
        // Input point 27: x(26)
        a_in2 = in[in_strides[26]];
        // Input point 7: x(6)
        a_in3 = in[in_strides[6]];

        a_s0 = a_in1 + a_in2;
        a_s4 = a_in2 - a_in1;
        a_s5 = a_in3 + a_s0;
        a_s6 = a_in3 - (CRTM_15_5 * a_s0);

        a_s23 = a_s6 + a_s3;
        a_s24 = a_s6 - a_s3;
        a_s26 = a_s4 + a_s1;
        a_s27 = a_s4 - a_s1;

        // Input point 3: x(2)
        a_in1 = in[in_strides[2]];
        // Input point 23: x(22)
        a_in2 = in[in_strides[22]];
        // Input point 13: x(12)
        a_in3 = in[in_strides[12]];

        a_s0 = a_in1 + a_in2;
        a_s1 = a_in2 - a_in1;
        a_s7 = a_in3 + a_s0;
        a_s3 = a_in3 - (CRTM_15_5 * a_s0);

        // Input point 9: x(8)
        a_in1 = in[in_strides[8]];
        // Input point 29: x(28)
        a_in2 = in[in_strides[28]];
        // Input point 19: x(18)
        a_in3 = in[in_strides[18]];

        a_s0 = a_in1 + a_in2;
        a_s4 = a_in2 - a_in1;
        a_s8 = a_in3 + a_s0;
        a_s6 = a_in3 - (CRTM_15_5 * a_s0);

        a_s28 = a_s6 + a_s3;
        a_s29 = a_s3 - a_s6;
        a_s30 = a_s4 + a_s1;
        a_s31 = a_s1 - a_s4;

        // Input point 11: x(10)
        a_in1 = in[in_strides[10]];
        // Input point 21: x(20)
        a_in2 = in[in_strides[20]];

        a_s0 = a_in1 + a_in2;
        a_t0 = CRTM_15_6 * (a_in2 - a_in1);
        a_s9 = a_in0 + a_s0;
        a_s3 = a_in0 - (CRTM_15_5 * a_s0);

        a_s11 = a_s8 + a_s7;
        a_s12 = a_s2 + a_s5;
        a_s19 = a_s5 - a_s2;
        a_s20 = a_s8 - a_s7;
        a_s13 = a_s11 + a_s12;
        a_t1 = CRTM_15_1 * (a_s11 - a_s12);
        a_s15 = a_s9 - (CRTM_15_2 * a_s13);
        // Output point 1: X(0)
        *out = a_s9 + a_s13;

        // Output point 12: X(11)
        out[out_strides[11]] = a_s15 + a_t1;
        // Output point 13: X(12)
        out[out_strides[12]] = (CRTM_15_3 * a_s20) + (CRTM_15_4 * a_s19);

        // Output point 24: X(23)
        out[out_strides[23]] = a_s15 - a_t1;
        // Output point 25: X(24)
        out[out_strides[24]] = (CRTM_15_4 * a_s20) - (CRTM_15_3 * a_s19);

        a_t2 = CRTM_15_1 * (a_s28 - a_s23);
        a_s13 = a_s28 + a_s23;

        // Output point 20: X(19)
        out[out_strides[19]] = a_s13 + a_s3;

        a_t3 = CRTM_15_9 * (a_s30 + a_s27);
        a_s17 = a_s30 - a_s27;

        // Output point 21: X(20)
        out[out_strides[20]] = CRTM_15_6 * a_s17 - a_t0;

        a_s15 = a_s3 - (CRTM_15_2 * a_s13);
        a_s20 = a_t0 + (CRTM_15_10 * a_s17);
        a_s21 = a_s15 - a_t2;
        a_s25 = (CRTM_15_8 * a_s26) - (CRTM_15_7 * a_s31);

        // Output point 4: X(3)
        out[out_strides[3]] = a_s21 + a_s25;
        // Output point 16: X(15)
        out[out_strides[15]] = a_s21 - a_s25;

        a_s21 = a_s15 + a_t2;
        a_s23 = (CRTM_15_8 * a_s31) + (CRTM_15_7 * a_s26);

        // Output point 28: X(27)
        out[out_strides[27]] = a_s21 + a_s23;

        a_s22 = a_s20 + a_t3;
        a_s16 = (CRTM_15_4 * a_s29) + (CRTM_15_3 * a_s24);

        // Output point 5: X(4)
        out[out_strides[4]] = a_s22 - a_s16;
        // Output point 17: X(16)
        out[out_strides[16]] = a_s22 + a_s16;

        a_s22 = a_s20 - a_t3;
        a_s10 = (CRTM_15_3 * a_s29) - (CRTM_15_4 * a_s24);
        // Output point 29: X(28)
        out[out_strides[28]] = a_s22 + a_s10;

        // Output point 8: X(7)
        out[out_strides[7]] = a_s21 - a_s23;
        // Output point 9: X(8)
        out[out_strides[8]] = a_s10 - a_s22;

        /* Shifted DFT */
        FLOAT b_in0, b_in1, b_in2, b_in3, b_in4, b_in5, b_in6, b_in7, b_in8,
              b_in9, b_in10, b_in11, b_in12, b_in13, b_in14, b_s0, b_s1, b_s2,
              b_s3, b_s4, b_s5, b_s6, b_s7, b_s8, b_s9, b_s10, b_s11, b_s12,
              b_s13, b_s14, b_s15, b_s16, b_s17, b_s18, b_s19, b_s20, b_s21,
              b_s22, b_s23, b_s24, b_s25, b_s26, b_s27,b_s28, b_s29, b_s30,
              b_s31, b_s32, b_s33, b_s34, b_s35, b_s36, b_s37, b_s38, b_s39,
              b_s40, b_s41, b_s42, b_s43, b_s44, b_s45, b_s46, b_s47, b_s48,
              b_s49, b_s50, b_s51, b_t0, b_t1, b_t2, b_t3, b_t4, b_t5, b_t6,
              b_t7, b_t8, b_t9, b_t10;

        // Input point 2: x(1)
        b_in0 = in[in_strides[1]];
        // Input point 4: x(3)
        b_in1 = in[in_strides[3]];
        // Input point 6: x(5)
        b_in2 = in[in_strides[5]];
        // Input point 8: x(7)
        b_in3 = in[in_strides[7]];
        // Input point 10: x(9)
        b_in4 = in[in_strides[9]];
        // Input point 12: x(11)
        b_in5 = in[in_strides[11]];
        // Input point 14: x(13)
        b_in6 = in[in_strides[13]];
        // Input point 16: x(15)
        b_in7 = in[in_strides[15]];
        // Input point 18: x(17)
        b_in8 = in[in_strides[17]];
        // Input point 20: x(19)
        b_in9 = in[in_strides[19]];
        // Input point 22: x(21)
        b_in10 = in[in_strides[21]];
        // Input point 24: x(23)
        b_in11 = in[in_strides[23]];
        // Input point 26: x(25)
        b_in12 = in[in_strides[25]];
        // Input point 28: x(27)
        b_in13 = in[in_strides[27]];
        // Input point 30: x(29)
        b_in14 = in[in_strides[29]];

        b_s0 = b_in1 + b_in4;
        b_s1 = b_in7 + b_in13;
        b_s2 = b_in7 - b_in13;
        b_s3 = b_in6 + b_in9;
        b_s4 = b_in12 + b_in3;
        b_s5 = b_in14 + b_in11;
        b_s6 = b_in2 + b_in8;
        b_s7 = b_in2 - b_in8;
        b_s8 = b_in9 + b_in3;
        b_s9 = b_in6 + b_in12;

        b_s10 = b_s8 - b_s9;
        b_s11 = b_in1 + b_s1;
        b_s12 = b_in14 + b_s6;
        b_s13 = b_in0 - b_s10;

        b_s14 = b_in10 + b_in4;
        b_s15 = b_s14 - b_s11;
        b_s16 = b_in11 + b_in5;
        b_s17 = b_s12 - b_s16;
        // Output point 11: X(10)
        out[out_strides[10]] = CRTM_15_6 * (b_s15 - b_s17);

        b_s18 = b_s15 + b_s17;
        // Output point 10: X(9)
        out[out_strides[9]]  = b_s13 - CRTM_15_5 * b_s18;
        // Output point 30: X(29)
        out[out_strides[29]] = b_s18 + b_s13;

        b_s19 = b_in0 + CRTM_15_2 * (b_s10);
        b_s20 = b_in9 + b_in12;
        b_s21 = b_in6 + b_in3;
        b_s22 = b_s20 - b_s21;
        b_s23 = CRTM_15_1 * (b_s22);
        b_s24 = b_s19 + b_s23;
        b_s25 = CRTM_15_3 * b_s2 - CRTM_15_4 * b_s0;
        b_s26 = CRTM_15_4 * b_s5 + CRTM_15_3 * b_s7;
        b_s27 = b_s25 - b_s26;
        b_s28 = CRTM_15_4 * b_s3 - CRTM_15_3 * b_s4;
        // Output point 7: X(6)
        out[out_strides[6]]  = b_s28 + b_s27;

        b_s29 = CRTM_15_5 * b_s27 - b_s28;
        b_s30 = CRTM_15_12 * b_in11 - b_in5;
        b_t0 = CRTM_15_1 * (b_in14 - b_s6);
        b_s31 = CRTM_15_2 * b_s12 + b_t0;
        b_s32 = b_s30 - b_s31;
        b_t1 = CRTM_15_1 * (b_in1 - b_s1);
        b_s33 = b_in10 + b_t1;
        b_t2 = CRTM_15_2 * b_s11;
        b_s34 = b_t2 - CRTM_15_12 * b_in4;
        b_s35  = b_s33 + b_s34;
        b_s36 = b_s32 + b_s35;
        // Output point 6: X(5)
        out[out_strides[5]]  = b_s24 + b_s36;

        b_t3 = CRTM_15_6 * (b_s32 - b_s35);
        // Output point 15: X(14)
        out[out_strides[14]]  = b_t3 + b_s29;
        // Output point 27: X(26)
        out[out_strides[26]] = b_t3 - b_s29;

        b_s37 = b_s24 - CRTM_15_5 * b_s36;
        b_t4 = CRTM_15_6 * (b_s25 + b_s26);
        // Output point 14: X(13)
        out[out_strides[13]] = b_s37 - b_t4;
        // Output point 26: X(25)
        out[out_strides[25]] = b_s37 + b_t4;

        b_t5 = CRTM_15_4 * b_s2;
        b_t6 = CRTM_15_3 * b_s0;
        b_s38 = b_t6 + b_t5;
        b_t7 = CRTM_15_4 * b_s7;
        b_t8 = CRTM_15_3 * b_s5;
        b_s39 = b_t7 - b_t8;
        b_t9 = CRTM_15_6 * (b_s38 + b_s39);
        b_s40 = b_s19 - b_s23;
        b_s41 = b_s39 - b_s38;
        b_s42 = (CRTM_15_4 * b_s4) + (CRTM_15_3 * b_s3);
        // Output point 19: X(18)
        out[out_strides[18]]  = b_s42 + b_s41;

        b_s43 = CRTM_15_5 * b_s41 - b_s42;
        b_s44 = b_t2 - b_t1;
        b_s45 = b_in10 + CRTM_15_11 * b_in4;
        b_s46 = b_s45 + b_s44;
        b_s47 = b_t0 - CRTM_15_11 * b_in11;
        b_s48 = CRTM_15_2 * b_s12 + b_in5;
        b_s49 = b_s47 - b_s48;
        b_s50 = b_s46 + b_s49;
        // Output point 18: X(17)
        out[out_strides[17]]  = b_s40 + b_s50;

        b_t10 = CRTM_15_6 * (b_s49 - b_s46);
        // Output point 3: X(2)
        out[out_strides[2]]  = b_s43 + b_t10;
        // Output point 23: X(22)
        out[out_strides[22]]  = b_s43 - b_t10;

        b_s51 = b_s40 - CRTM_15_5 * b_s50;
        // Output point 2: X(1)
        out[out_strides[1]]  = b_s51 + b_t9;
        // Output point 22: X(21)
        out[out_strides[21]]  = b_s51 - b_t9;

        in = in + v_in_stride;
        out = out + v_out_stride;
    }
    AOCLFFTZ_LOG(DEBUG, global_logger_mode, "Exit");
}

static VOID r2hcf_rfft15c_fp32_bwd(VOID *in_real, VOID *in_imag, VOID *out_real,
                                   VOID *out_imag, INTP n,
                                   aoclfftz_strides_t *strides, VOID *twd, UINT8 flag)
{
    AOCLFFTZ_LOG(DEBUG, global_logger_mode, "Enter");
    const FLOAT CRTM_15_1 = 1.118033988749894848204586834365638117720309180f;
    const FLOAT CRTM_15_2 = 0.500000000000000000000000000000000000000000000f;
    const FLOAT CRTM_15_3 = 1.902113032590307144232878666758764286811397268f;
    const FLOAT CRTM_15_4 = 1.175570504584946258337411909278145537195304875f;
    const FLOAT CRTM_15_5 = 2.000000000000000000000000000000000000000000000f;
    const FLOAT CRTM_15_6 = 0.250000000000000000000000000000000000000000000f;
    const FLOAT CRTM_15_7 =
        1.01807392091025412936433958497993431950210362069154f;
    const FLOAT CRTM_15_8 =
        1.64727820709266368541488232323193203275710390365294f;
    const FLOAT CRTM_15_9 =
        0.96824583655185421224048777314959976915574786128504f;
    const FLOAT CRTM_15_10 = 1.732050807568877293527446341505872366942805254f;
    const FLOAT CRTM_15_11 = 0.433012701892219323381861585376468091735701313f;
    const FLOAT CRTM_15_12 = 0.587785252292473129168705954639072768597652438f;
    const FLOAT CRTM_15_13 = 0.951056516295153572116439333379382143405698634f;
    const FLOAT CRTM_15_14 = 0.559016994374947424102293417182819058860154590f;

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
        /* Standard DFT */
        FLOAT a_in0, a_in1, a_in2, a_in3, a_s0, a_s1, a_s2, a_s3, a_s4, a_s5,
              a_s6, a_s7, a_s8, a_s9, a_s10, a_s11, a_s12, a_s13, a_s14, a_s15,
              a_s16, a_s17, a_s18, a_s19, a_s20, a_s21, a_s22, a_s23, a_s24,
              a_s25, a_s26, a_s27, a_s28, a_s29, a_s30, a_s31, a_s32, a_s33,
              a_s34, a_s35, a_s36, a_s37, a_s38, a_s39, a_t0, a_t1, a_t2, a_t3,
              a_t4, a_t5;

        //  Input point 1: x(0)
        a_in0 = *in;
        //  Input point 8: x(7)
        a_in1 = in[in_strides[7]];
        //  Input point 28: x(27)
        a_in2 = in[in_strides[27]];
        //  Input point 12: x(11)
        a_in3 = in[in_strides[11]];

        a_s0 = a_in1 + a_in2;
        a_s1 = a_in1 - a_in2;
        a_s2 = a_in3 + a_s0;
        a_s3 = a_in3 - (CRTM_15_2 * a_s0);

        //  Input point 9: x(8)
        a_in1 = in[in_strides[8]];
        //  Input point 29: x(28)
        a_in2 = in[in_strides[28]];
        //  Input point 13: x(12)
        a_in3 = in[in_strides[12]];

        a_s0 = a_in1 + a_in2;
        a_s4 = a_in1 - a_in2;
        a_s5 = a_s0 - a_in3;
        a_s6 = a_in3 + (CRTM_15_2 * a_s0);

        //  Input point 4: x(3)
        a_in1 = in[in_strides[3]];
        //  Input point 16: x(15)
        a_in2 = in[in_strides[15]];
        //  Input point 24: x(23)
        a_in3 = in[in_strides[23]];

        a_s0 = a_in1 + a_in2;
        a_s7 = a_in2 - a_in1;
        a_s8 = a_in3 + a_s0;
        a_s9 = a_in3 - (CRTM_15_2 * a_s0);

        //  Input point 5: x(4)
        a_in1 = in[in_strides[4]];
        //  Input point 17: x(16)
        a_in2 = in[in_strides[16]];
        //  Input point 25: x(24)
        a_in3 = in[in_strides[24]];
        a_s0 = a_in1 - a_in2;
        a_s10 = a_in2 + a_in1;
        a_s11 = a_in3 + a_s0;
        a_s12 = a_in3 - (CRTM_15_2 * a_s0);

        //  Input point 20: x(19)
        a_in1 = in[in_strides[19]];
        //  Input point 21: x(20)
        a_in2 = in[in_strides[20]];

        a_s13 = a_in0 + CRTM_15_5 * a_in1;
        a_s14 = a_in0 - a_in1;
        a_s15 = a_s8 + a_s2;
        a_t0 = CRTM_15_1 * (a_s8 - a_s2);
        a_s16 = a_s13 - (CRTM_15_2 * a_s15);
        // Output point 1: X(0)
        *out = a_s13 + CRTM_15_5 * a_s15;

        a_s17 = (CRTM_15_3 * a_s11) + (CRTM_15_4 * a_s5);
        a_s18 = a_s16 + a_t0;

        // Output point 7: X(6)
        out[out_strides[6]] = a_s18 - a_s17;
        // Output point 25: X(24)
        out[out_strides[24]] = a_s18 + a_s17;

        a_s19 = a_s16 - a_t0;
        a_s20 = (CRTM_15_4 * a_s11) - (CRTM_15_3 * a_s5);

        // Output point 13: X(12)
        out[out_strides[12]] = a_s19 - a_s20;
        // Output point 19: X(18)
        out[out_strides[18]] = a_s19 + a_s20;

        a_t1 = CRTM_15_1 * (a_s9 - a_s3);
        a_t2 = CRTM_15_9 * (a_s10 + a_s4);
        a_s21 = (a_s9 + a_s3);
        a_s22 = a_s4 - a_s10;
        a_t3 = CRTM_15_5 * a_s21 + a_s14;
        a_t4 = CRTM_15_10 * (a_s22 + a_in2);

        // Output point 11: X(10)
        out[out_strides[10]] = a_t3 + a_t4;
        // Output point 21: X(20)
        out[out_strides[20]] = a_t3 - a_t4;

        a_s23 = a_s14 - (CRTM_15_2 * a_s21);
        a_t5 = CRTM_15_10 * ((CRTM_15_6 * a_s22) - a_in2);
        a_s24 = a_t5 + a_s23;
        a_s25 = a_s23 - a_t5;
        a_s26 = a_t2 - a_t1;
        a_s27 = a_t1 + a_t2;

        a_s28 = a_s24 - a_s27;
        a_s29 = (CRTM_15_4 * a_s12) + (CRTM_15_3 * a_s6);
        a_s30 = (CRTM_15_8 * a_s1) - (CRTM_15_7 * a_s7);
        a_s31 = a_s30 - a_s29;
        a_s32 = a_s29 + a_s30;

        // Output point 3: X(2)
        out[out_strides[2]] = a_s28 + a_s31;
        // Output point 9: X(8)
        out[out_strides[8]] = a_s28 - a_s31;

        a_s33 = a_s24 + a_s27;
        a_s34 = (CRTM_15_8 * a_s7) + (CRTM_15_7 * a_s1);
        a_s35 = (CRTM_15_3 * a_s12) - (CRTM_15_4 * a_s6);
        a_s36 = a_s35 + a_s34;

        // Output point 15: X(14)
        out[out_strides[14]] = a_s33 + a_s36;
        // Output point 27: X(26)
        out[out_strides[26]] = a_s33 - a_s36;

        a_s37 = a_s25 - a_s26;
        a_s38 = a_s35 - a_s34;
        // Output point 5: X(4)
        out[out_strides[4]] = a_s37 + a_s38;
        // Output point 17: X(16)
        out[out_strides[16]] = a_s37 - a_s38;

        a_s39 = a_s25 + a_s26;

        // Output point 23: X(22)
        out[out_strides[22]] = a_s39 - a_s32;
        // Output point 29: X(28)
        out[out_strides[28]] = a_s39 + a_s32;

        /* Shifted DFT */
        FLOAT b_in0, b_in1, b_in2, b_in3, b_in4, b_in5, b_in6, b_in7, b_in8,
              b_in9, b_in10, b_in11, b_in12, b_in13, b_in14;
        FLOAT b_s0, b_s1, b_s2, b_s3, b_s4, b_s5, b_s6, b_s7, b_s8, b_s9, b_s10,
              b_s11, b_s12, b_s13, b_s14, b_s15, b_s16, b_s17, b_s18, b_s19,
              b_s20, b_s21, b_s22, b_s23, b_s24, b_s25, b_s26, b_s27, b_s28,
              b_s29, b_s30, b_s31, b_s32, b_s33, b_s34, b_s35, b_s36, b_s37,
              b_s38, b_s39, b_s40, b_s41, b_s42, b_s43, b_s44, b_s45, b_s46,
              b_s47, b_s48, b_t0, b_t1, b_t2, b_t3, b_t4, b_t5, b_t6, b_t7,
              b_t8, b_t9, b_t10, b_t11, b_t12, b_t13, b_t14, b_t15, b_t16,
              b_t17, b_t18, b_t19, b_t20, b_t21, b_t22, b_t23, b_t24, b_t25;

        // Input point 2: x(1)
        b_in0 = in[in_strides[1]];
        // Input point 3: x(2)
        b_in1 = in[in_strides[2]];
        // Input point 6: x(5)
        b_in2 = in[in_strides[5]];
        // Input point 7: x(6)
        b_in3 = in[in_strides[6]];
        // Input point 10: x(9)
        b_in4 = in[in_strides[9]];
        // Input point 11: x(10)
        b_in5 = in[in_strides[10]];
        // Input point 14: x(13)
        b_in6 = in[in_strides[13]];
        // Input point 15: x(14)
        b_in7 = in[in_strides[14]];
        // Input point 18: x(17)
        b_in8 = in[in_strides[17]];
        // Input point 19: x(18)
        b_in9 = in[in_strides[18]];
        // Input point 22: x(21)
        b_in10 = in[in_strides[21]];
        // Input point 23: x(22)
        b_in11 = in[in_strides[22]];
        // Input point 26: x(25)
        b_in12 = in[in_strides[25]];
        // Input point 27: x(26)
        b_in13 = in[in_strides[26]];
        // Input point 30: x(29)
        b_in14 = in[in_strides[29]];

        b_s0 = b_in0 + b_in10;
        b_s1 = b_in0 - b_in10;
        b_s2 = b_in1 + b_in11;
        b_s3 = b_in1 - b_in11;
        b_s4 = b_in6 + b_in12;
        b_s5 = b_in6 - b_in12;
        b_s6 = b_in7 + b_in13;
        b_s7 = b_in7 - b_in13;
        b_s8 = b_in2 + b_in8;
        b_s9 = b_in2 - b_in8;

        b_s10 = b_s0 - b_s4;
        b_t0 = CRTM_15_14 * b_s10;
        b_t1 = CRTM_15_1 * b_s9;
        b_t2 = CRTM_15_5 * b_t0;
        b_s11 = b_t2 - b_t1;
        b_t3 = CRTM_15_2 * b_s8;
        b_s12 = b_in14 - b_t3;
        b_s13 = b_s0 + b_s4;

        b_t4 = CRTM_15_6 * b_s13;
        b_s14 = b_in4 - b_t4;
        b_t5 = CRTM_15_5 * b_s14;
        b_s15 = b_s12 + b_t5;
        b_s16 = b_s11 - b_s15;
        b_s17 = b_s11 + b_s15;

        b_t6 = CRTM_15_13 * b_s7;
        b_t7 = CRTM_15_12 * b_s2;
        b_t8 = CRTM_15_4 * b_in9;
        b_t9 = CRTM_15_3 * b_in3;
        b_s18 = b_t6 - b_t7;
        b_t10 = CRTM_15_5 * b_s18;
        b_s19 = b_t8 - b_t9;
        b_s20 = b_t10 + b_s19;
        // Output pt 8: X(7)
        out[out_strides[7]] = b_s20 + b_s16;
        // Output pt 26: X(25)
        out[out_strides[25]] = b_s20 - b_s16;

        b_s21 = b_s3 + b_s6;
        b_s22 = b_in5 - b_s21;
        b_t11 = CRTM_15_10 * b_s22;
        b_s23 = b_s13 + b_in4;
        b_t12 = CRTM_15_5 * b_s8;
        b_s24 = b_in14 + b_t12;
        b_s25 = b_s23 - b_s24;
        // Output pt 12: X(11)
        out[out_strides[11]] = b_t11 + b_s25;
        // Output pt 22: X(21)
        out[out_strides[21]] = b_t11 - b_s25;

        b_t13 = CRTM_15_5 * b_s23;
        // Output pt 2: X(1)
        out[out_strides[1]] = b_t13 + b_s24;

        b_t14 = CRTM_15_4 * b_in3;
        b_t15 = CRTM_15_3 * b_in9;
        b_s26 = b_t14 + b_t15;

        b_t16 = CRTM_15_13 * b_s2;
        b_t17 = CRTM_15_12 * b_s7;
        b_s27 = b_t16 + b_t17;
        b_t18 = CRTM_15_5 * b_s27;
        b_s28 = b_s26 - b_t18;
        // Output pt 14: X(13)
        out[out_strides[13]] = b_s28 + b_s17;
        // Output pt 20: X(19)
        out[out_strides[19]] = b_s28 - b_s17;

        b_t19 = CRTM_15_11 * b_s21;
        b_t20 = CRTM_15_10 * b_in5;
        b_s29 = b_t19 + b_t20;
        b_s30 = b_s3 - b_s6;
        b_t21 = CRTM_15_9 * b_s30;
        b_s31 = b_t21 - b_s29;
        b_s32 = b_s29 + b_t21;
        b_s33 = b_t0 + b_t1;
        b_s34 = b_s12 - b_s14;
        b_s35 = b_s33 + b_s34;
        b_s36 = b_s33 - b_s34;
        b_s37 = b_s27 + b_s26;

        b_t22 = CRTM_15_8 * b_s1;
        b_t23 = CRTM_15_7 * b_s5;
        b_s38 = b_t22 + b_t23;
        b_t24 = CRTM_15_8 * b_s5;
        b_t25 = CRTM_15_7 * b_s1;
        b_s39 = b_t24 - b_t25;
        b_s40 = b_s18 - b_s19;
        b_s41 = b_s36 - b_s37;
        b_s42 = b_s36 + b_s37;
        b_s43 = b_s38 + b_s31;
        // Output pt 4: X(3)
        out[out_strides[3]] = b_s41 + b_s43;
        // Output pt 24: X(23)
        out[out_strides[23]] = b_s41 - b_s43;

        b_s44 = b_s38 - b_s31;
        // Output pt 10: X(9)
        out[out_strides[9]] = b_s44 - b_s42;
        // Output pt 30: X(29)
        out[out_strides[29]] = -(b_s42 + b_s44);

        b_s45 = b_s39 - b_s32;
        b_s46 = b_s32 + b_s39;
        b_s47 = b_s35 + b_s40;
        // Output pt 18: X(17)
        out[out_strides[17]] = b_s45 + b_s47;
        // Output pt 28: X(27)
        out[out_strides[27]] = b_s45 - b_s47;

        b_s48 = b_s35 - b_s40;
        // Output pt 6: X(5)
        out[out_strides[5]] = b_s48 - b_s46;
        // Output pt 16: X(15)
        out[out_strides[15]] = -(b_s46 + b_s48);

        in  = in + v_in_stride;
        out = out + v_out_stride;
    }
    AOCLFFTZ_LOG(DEBUG, global_logger_mode, "Exit");
}

static VOID r2hcf_rfft15c_fp64_fwd(VOID *in_real, VOID *in_imag, VOID *out_real,
                                   VOID *out_imag, INTP n,
                                   aoclfftz_strides_t *strides, VOID *twd, UINT8 flag)
{
    AOCLFFTZ_LOG(DEBUG, global_logger_mode, "Enter");
    const DOUBLE CRTM_15_1 =
        0.55901699437494742410229341718281905886015458990288;
    const DOUBLE CRTM_15_2 =
        0.25000000000000000000000000000000000000000000000000;
    const DOUBLE CRTM_15_3 =
        0.95105651629515357211643933337938214340569863400000;
    const DOUBLE CRTM_15_4 =
        0.58778525229247301629891039327884007596190389052978;
    const DOUBLE CRTM_15_5 =
        0.50000000000000000000000000000000000000000000000000;
    const DOUBLE CRTM_15_6 =
        0.86602540378443864676372317075293618347140262690519;
    const DOUBLE CRTM_15_11 = 0.309016994374947424102293417182819058860154590;
    const DOUBLE CRTM_15_12 = 0.809016994374947424102293417182819058860154590;
    // Below CRTMs are the product of the above CRTMs, Precomputed to save
    // multiplications on the fly.
    // CRTM_15_7 = CRTM_15_6 * CRTM_15_4
    const DOUBLE CRTM_15_7 =
        0.50903696045512706468216979248996715975105181034577;
    // CRTM_15_8 = CRTM_15_6 * CRTM_15_3
    const DOUBLE CRTM_15_8 =
        0.82363910354633184270744116161596601637855195182647;
    // CRTM_15_9 = CRTM_15_6 * CRTM_15_1
    const DOUBLE CRTM_15_9 =
        0.48412291827592710612024388657479988457787393064252;
    // CRTM_15_10 = CRTM_15_6 * CRTM_15_2
    const DOUBLE CRTM_15_10 =
        0.21650635094610964914707551542960572987794876098633;

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
        /* Standard DFT*/
        DOUBLE a_in0, a_in1, a_in2, a_in3, a_s0, a_s1, a_s2, a_s3, a_s4, a_s5,
               a_s6, a_s7, a_s8, a_s9, a_s10, a_s11, a_s12, a_s13, a_s15, a_s16,
               a_s17, a_s19, a_s20, a_s21, a_s22, a_s23, a_s24, a_s25, a_s26,
               a_s27, a_s28, a_s29, a_s30, a_s31, a_t0, a_t1, a_t2, a_t3;

        // Input point 1: x(0)
        a_in0 = *in;
        // Input point 5: x(4)
        a_in1 = in[in_strides[4]];
        // Input point 15: x(14)
        a_in2 = in[in_strides[14]];
        // Input point 25: x(24)
        a_in3 = in[in_strides[24]];

        a_s0 = a_in1 + a_in2;
        a_s1 = a_in1 - a_in2;
        a_s2 = a_in3 + a_s0;
        a_s3 = a_in3 - (CRTM_15_5 * a_s0);

        // Input point 17: x(16)
        a_in1 = in[in_strides[16]];
        // Input point 27: x(26)
        a_in2 = in[in_strides[26]];
        // Input point 7: x(6)
        a_in3 = in[in_strides[6]];

        a_s0 = a_in1 + a_in2;
        a_s4 = a_in2 - a_in1;
        a_s5 = a_in3 + a_s0;
        a_s6 = a_in3 - (CRTM_15_5 * a_s0);

        a_s23 = a_s6 + a_s3;
        a_s24 = a_s6 - a_s3;
        a_s26 = a_s4 + a_s1;
        a_s27 = a_s4 - a_s1;

        // Input point 3: x(2)
        a_in1 = in[in_strides[2]];
        // Input point 23: x(22)
        a_in2 = in[in_strides[22]];
        // Input point 13: x(12)
        a_in3 = in[in_strides[12]];

        a_s0 = a_in1 + a_in2;
        a_s1 = a_in2 - a_in1;
        a_s7 = a_in3 + a_s0;
        a_s3 = a_in3 - (CRTM_15_5 * a_s0);

        // Input point 9: x(8)
        a_in1 = in[in_strides[8]];
        // Input point 29: x(28)
        a_in2 = in[in_strides[28]];
        // Input point 19: x(18)
        a_in3 = in[in_strides[18]];

        a_s0 = a_in1 + a_in2;
        a_s4 = a_in2 - a_in1;
        a_s8 = a_in3 + a_s0;
        a_s6 = a_in3 - (CRTM_15_5 * a_s0);

        a_s28 = a_s6 + a_s3;
        a_s29 = a_s3 - a_s6;
        a_s30 = a_s4 + a_s1;
        a_s31 = a_s1 - a_s4;

        // Input point 11: x(10)
        a_in1 = in[in_strides[10]];
        // Input point 21: x(20)
        a_in2 = in[in_strides[20]];

        a_s0 = a_in1 + a_in2;
        a_t0 = CRTM_15_6 * (a_in2 - a_in1);
        a_s9 = a_in0 + a_s0;
        a_s3 = a_in0 - (CRTM_15_5 * a_s0);

        a_s11 = a_s8 + a_s7;
        a_s12 = a_s2 + a_s5;
        a_s19 = a_s5 - a_s2;
        a_s20 = a_s8 - a_s7;
        a_s13 = a_s11 + a_s12;
        a_t1 = CRTM_15_1 * (a_s11 - a_s12);
        a_s15 = a_s9 - (CRTM_15_2 * a_s13);
        // Output point 1: X(0)
        *out = a_s9 + a_s13;

        // Output point 12: X(11)
        out[out_strides[11]] = a_s15 + a_t1;
        // Output point 13: X(12)
        out[out_strides[12]] = (CRTM_15_3 * a_s20) + (CRTM_15_4 * a_s19);

        // Output point 24: X(23)
        out[out_strides[23]] = a_s15 - a_t1;
        // Output point 25: X(24)
        out[out_strides[24]] = (CRTM_15_4 * a_s20) - (CRTM_15_3 * a_s19);

        a_t2 = CRTM_15_1 * (a_s28 - a_s23);
        a_s13 = a_s28 + a_s23;

        // Output point 20: X(19)
        out[out_strides[19]] = a_s13 + a_s3;

        a_t3 = CRTM_15_9 * (a_s30 + a_s27);
        a_s17 = a_s30 - a_s27;

        // Output point 21: X(20)
        out[out_strides[20]] = CRTM_15_6 * a_s17 - a_t0;

        a_s15 = a_s3 - (CRTM_15_2 * a_s13);
        a_s20 = a_t0 + (CRTM_15_10 * a_s17);
        a_s21 = a_s15 - a_t2;
        a_s25 = (CRTM_15_8 * a_s26) - (CRTM_15_7 * a_s31);

        // Output point 4: X(3)
        out[out_strides[3]] = a_s21 + a_s25;
        // Output point 16: X(15)
        out[out_strides[15]] = a_s21 - a_s25;

        a_s21 = a_s15 + a_t2;
        a_s23 = (CRTM_15_8 * a_s31) + (CRTM_15_7 * a_s26);

        // Output point 28: X(27)
        out[out_strides[27]] = a_s21 + a_s23;

        a_s22 = a_s20 + a_t3;
        a_s16 = (CRTM_15_4 * a_s29) + (CRTM_15_3 * a_s24);

        // Output point 5: X(4)
        out[out_strides[4]] = a_s22 - a_s16;
        // Output point 17: X(16)
        out[out_strides[16]] = a_s22 + a_s16;

        a_s22 = a_s20 - a_t3;
        a_s10 = (CRTM_15_3 * a_s29) - (CRTM_15_4 * a_s24);
        // Output point 29: X(28)
        out[out_strides[28]] = a_s22 + a_s10;

        // Output point 8: X(7)
        out[out_strides[7]] = a_s21 - a_s23;
        // Output point 9: X(8)
        out[out_strides[8]] = a_s10 - a_s22;

        /* Shifted DFT */
        DOUBLE b_in0, b_in1, b_in2, b_in3, b_in4, b_in5, b_in6, b_in7, b_in8,
               b_in9, b_in10, b_in11, b_in12, b_in13, b_in14, b_s0, b_s1, b_s2,
               b_s3, b_s4, b_s5, b_s6, b_s7, b_s8, b_s9, b_s10, b_s11, b_s12,
               b_s13, b_s14, b_s15, b_s16, b_s17, b_s18, b_s19, b_s20, b_s21,
               b_s22, b_s23, b_s24, b_s25, b_s26, b_s27,b_s28, b_s29, b_s30,
               b_s31, b_s32, b_s33, b_s34, b_s35, b_s36, b_s37, b_s38, b_s39,
               b_s40, b_s41, b_s42, b_s43, b_s44, b_s45, b_s46, b_s47, b_s48,
               b_s49, b_s50, b_s51, b_t0, b_t1, b_t2, b_t3, b_t4, b_t5, b_t6,
               b_t7, b_t8, b_t9, b_t10;

        // Input point 2: x(1)
        b_in0 = in[in_strides[1]];
        // Input point 4: x(3)
        b_in1 = in[in_strides[3]];
        // Input point 6: x(5)
        b_in2 = in[in_strides[5]];
        // Input point 8: x(7)
        b_in3 = in[in_strides[7]];
        // Input point 10: x(9)
        b_in4 = in[in_strides[9]];
        // Input point 12: x(11)
        b_in5 = in[in_strides[11]];
        // Input point 14: x(13)
        b_in6 = in[in_strides[13]];
        // Input point 16: x(15)
        b_in7 = in[in_strides[15]];
        // Input point 18: x(17)
        b_in8 = in[in_strides[17]];
        // Input point 20: x(19)
        b_in9 = in[in_strides[19]];
        // Input point 22: x(21)
        b_in10 = in[in_strides[21]];
        // Input point 24: x(23)
        b_in11 = in[in_strides[23]];
        // Input point 26: x(25)
        b_in12 = in[in_strides[25]];
        // Input point 28: x(27)
        b_in13 = in[in_strides[27]];
        // Input point 30: x(29)
        b_in14 = in[in_strides[29]];

        b_s0 = b_in1 + b_in4;
        b_s1 = b_in7 + b_in13;
        b_s2 = b_in7 - b_in13;
        b_s3 = b_in6 + b_in9;
        b_s4 = b_in12 + b_in3;
        b_s5 = b_in14 + b_in11;
        b_s6 = b_in2 + b_in8;
        b_s7 = b_in2 - b_in8;
        b_s8 = b_in9 + b_in3;
        b_s9 = b_in6 + b_in12;

        b_s10 = b_s8 - b_s9;
        b_s11 = b_in1 + b_s1;
        b_s12 = b_in14 + b_s6;
        b_s13 = b_in0 - b_s10;

        b_s14 = b_in10 + b_in4;
        b_s15 = b_s14 - b_s11;
        b_s16 = b_in11 + b_in5;
        b_s17 = b_s12 - b_s16;
        // Output point 11: X(10)
        out[out_strides[10]] = CRTM_15_6 * (b_s15 - b_s17);

        b_s18 = b_s15 + b_s17;
        // Output point 10: X(9)
        out[out_strides[9]]  = b_s13 - CRTM_15_5 * b_s18;
        // Output point 30: X(29)
        out[out_strides[29]] = b_s18 + b_s13;

        b_s19 = b_in0 + CRTM_15_2 * (b_s10);
        b_s20 = b_in9 + b_in12;
        b_s21 = b_in6 + b_in3;
        b_s22 = b_s20 - b_s21;
        b_s23 = CRTM_15_1 * (b_s22);
        b_s24 = b_s19 + b_s23;
        b_s25 = CRTM_15_3 * b_s2 - CRTM_15_4 * b_s0;
        b_s26 = CRTM_15_4 * b_s5 + CRTM_15_3 * b_s7;
        b_s27 = b_s25 - b_s26;
        b_s28 = CRTM_15_4 * b_s3 - CRTM_15_3 * b_s4;
        // Output point 7: X(6)
        out[out_strides[6]]  = b_s28 + b_s27;

        b_s29 = CRTM_15_5 * b_s27 - b_s28;
        b_s30 = CRTM_15_12 * b_in11 - b_in5;
        b_t0 = CRTM_15_1 * (b_in14 - b_s6);
        b_s31 = CRTM_15_2 * b_s12 + b_t0;
        b_s32 = b_s30 - b_s31;
        b_t1 = CRTM_15_1 * (b_in1 - b_s1);
        b_s33 = b_in10 + b_t1;
        b_t2 = CRTM_15_2 * b_s11;
        b_s34 = b_t2 - CRTM_15_12 * b_in4;
        b_s35  = b_s33 + b_s34;
        b_s36 = b_s32 + b_s35;
        // Output point 6: X(5)
        out[out_strides[5]]  = b_s24 + b_s36;

        b_t3 = CRTM_15_6 * (b_s32 - b_s35);
        // Output point 15: X(14)
        out[out_strides[14]]  = b_t3 + b_s29;
        // Output point 27: X(26)
        out[out_strides[26]] = b_t3 - b_s29;

        b_s37 = b_s24 - CRTM_15_5 * b_s36;
        b_t4 = CRTM_15_6 * (b_s25 + b_s26);
        // Output point 14: X(13)
        out[out_strides[13]] = b_s37 - b_t4;
        // Output point 26: X(25)
        out[out_strides[25]] = b_s37 + b_t4;

        b_t5 = CRTM_15_4 * b_s2;
        b_t6 = CRTM_15_3 * b_s0;
        b_s38 = b_t6 + b_t5;
        b_t7 = CRTM_15_4 * b_s7;
        b_t8 = CRTM_15_3 * b_s5;
        b_s39 = b_t7 - b_t8;
        b_t9 = CRTM_15_6 * (b_s38 + b_s39);
        b_s40 = b_s19 - b_s23;
        b_s41 = b_s39 - b_s38;
        b_s42 = (CRTM_15_4 * b_s4) + (CRTM_15_3 * b_s3);
        // Output point 19: X(18)
        out[out_strides[18]]  = b_s42 + b_s41;

        b_s43 = CRTM_15_5 * b_s41 - b_s42;
        b_s44 = b_t2 - b_t1;
        b_s45 = b_in10 + CRTM_15_11 * b_in4;
        b_s46 = b_s45 + b_s44;
        b_s47 = b_t0 - CRTM_15_11 * b_in11;
        b_s48 = CRTM_15_2 * b_s12 + b_in5;
        b_s49 = b_s47 - b_s48;
        b_s50 = b_s46 + b_s49;
        // Output point 18: X(17)
        out[out_strides[17]]  = b_s40 + b_s50;

        b_t10 = CRTM_15_6 * (b_s49 - b_s46);
        // Output point 3: X(2)
        out[out_strides[2]]  = b_s43 + b_t10;
        // Output point 23: X(22)
        out[out_strides[22]]  = b_s43 - b_t10;

        b_s51 = b_s40 - CRTM_15_5 * b_s50;
        // Output point 2: X(1)
        out[out_strides[1]]  = b_s51 + b_t9;
        // Output point 22: X(21)
        out[out_strides[21]]  = b_s51 - b_t9;

        in = in + v_in_stride;
        out = out + v_out_stride;
    }
    AOCLFFTZ_LOG(DEBUG, global_logger_mode, "Exit");
}

static VOID r2hcf_rfft15c_fp64_bwd(VOID *in_real, VOID *in_imag, VOID *out_real,
                                   VOID *out_imag, INTP n,
                                   aoclfftz_strides_t *strides, VOID *twd, UINT8 flag)
{
    AOCLFFTZ_LOG(DEBUG, global_logger_mode, "Enter");
    const DOUBLE CRTM_15_1 = 1.118033988749894848204586834365638117720309180;
    const DOUBLE CRTM_15_2 = 0.500000000000000000000000000000000000000000000;
    const DOUBLE CRTM_15_3 = 1.902113032590307144232878666758764286811397268;
    const DOUBLE CRTM_15_4 = 1.175570504584946258337411909278145537195304875;
    const DOUBLE CRTM_15_5 = 2.000000000000000000000000000000000000000000000;
    const DOUBLE CRTM_15_6 = 0.250000000000000000000000000000000000000000000;
    const DOUBLE CRTM_15_7 =
        1.01807392091025412936433958497993431950210362069154;
    const DOUBLE CRTM_15_8 =
        1.64727820709266368541488232323193203275710390365294;
    const DOUBLE CRTM_15_9 =
        0.96824583655185421224048777314959976915574786128504;
    const DOUBLE CRTM_15_10 = 1.732050807568877293527446341505872366942805254;
    const DOUBLE CRTM_15_11 = 0.433012701892219323381861585376468091735701313;
    const DOUBLE CRTM_15_12 = 0.587785252292473129168705954639072768597652438;
    const DOUBLE CRTM_15_13 = 0.951056516295153572116439333379382143405698634;
    const DOUBLE CRTM_15_14 = 0.559016994374947424102293417182819058860154590;

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
        /* Standard DFT */
        DOUBLE a_in0, a_in1, a_in2, a_in3, a_s0, a_s1, a_s2, a_s3, a_s4, a_s5,
               a_s6, a_s7, a_s8, a_s9, a_s10, a_s11, a_s12, a_s13, a_s14, a_s15,
               a_s16, a_s17, a_s18, a_s19, a_s20, a_s21, a_s22, a_s23, a_s24,
               a_s25, a_s26, a_s27, a_s28, a_s29, a_s30, a_s31, a_s32, a_s33,
               a_s34, a_s35, a_s36, a_s37, a_s38, a_s39, a_t0, a_t1, a_t2, a_t3,
               a_t4, a_t5;

        //  Input point 1: x(0)
        a_in0 = *in;
        //  Input point 8: x(7)
        a_in1 = in[in_strides[7]];
        //  Input point 28: x(27)
        a_in2 = in[in_strides[27]];
        //  Input point 12: x(11)
        a_in3 = in[in_strides[11]];

        a_s0 = a_in1 + a_in2;
        a_s1 = a_in1 - a_in2;
        a_s2 = a_in3 + a_s0;
        a_s3 = a_in3 - (CRTM_15_2 * a_s0);

        //  Input point 9: x(8)
        a_in1 = in[in_strides[8]];
        //  Input point 29: x(28)
        a_in2 = in[in_strides[28]];
        //  Input point 13: x(12)
        a_in3 = in[in_strides[12]];

        a_s0 = a_in1 + a_in2;
        a_s4 = a_in1 - a_in2;
        a_s5 = a_s0 - a_in3;
        a_s6 = a_in3 + (CRTM_15_2 * a_s0);

        //  Input point 4: x(3)
        a_in1 = in[in_strides[3]];
        //  Input point 16: x(15)
        a_in2 = in[in_strides[15]];
        //  Input point 24: x(23)
        a_in3 = in[in_strides[23]];

        a_s0 = a_in1 + a_in2;
        a_s7 = a_in2 - a_in1;
        a_s8 = a_in3 + a_s0;
        a_s9 = a_in3 - (CRTM_15_2 * a_s0);

        //  Input point 5: x(4)
        a_in1 = in[in_strides[4]];
        //  Input point 17: x(16)
        a_in2 = in[in_strides[16]];
        //  Input point 25: x(24)
        a_in3 = in[in_strides[24]];
        a_s0 = a_in1 - a_in2;
        a_s10 = a_in2 + a_in1;
        a_s11 = a_in3 + a_s0;
        a_s12 = a_in3 - (CRTM_15_2 * a_s0);

        //  Input point 20: x(19)
        a_in1 = in[in_strides[19]];
        //  Input point 21: x(20)
        a_in2 = in[in_strides[20]];

        a_s13 = a_in0 + CRTM_15_5 * a_in1;
        a_s14 = a_in0 - a_in1;
        a_s15 = a_s8 + a_s2;
        a_t0 = CRTM_15_1 * (a_s8 - a_s2);
        a_s16 = a_s13 - (CRTM_15_2 * a_s15);
        // Output point 1: X(0)
        *out = a_s13 + CRTM_15_5 * a_s15;

        a_s17 = (CRTM_15_3 * a_s11) + (CRTM_15_4 * a_s5);
        a_s18 = a_s16 + a_t0;

        // Output point 7: X(6)
        out[out_strides[6]] = a_s18 - a_s17;
        // Output point 25: X(24)
        out[out_strides[24]] = a_s18 + a_s17;

        a_s19 = a_s16 - a_t0;
        a_s20 = (CRTM_15_4 * a_s11) - (CRTM_15_3 * a_s5);

        // Output point 13: X(12)
        out[out_strides[12]] = a_s19 - a_s20;
        // Output point 19: X(18)
        out[out_strides[18]] = a_s19 + a_s20;

        a_t1 = CRTM_15_1 * (a_s9 - a_s3);
        a_t2 = CRTM_15_9 * (a_s10 + a_s4);
        a_s21 = (a_s9 + a_s3);
        a_s22 = a_s4 - a_s10;
        a_t3 = CRTM_15_5 * a_s21 + a_s14;
        a_t4 = CRTM_15_10 * (a_s22 + a_in2);

        // Output point 11: X(10)
        out[out_strides[10]] = a_t3 + a_t4;
        // Output point 21: X(20)
        out[out_strides[20]] = a_t3 - a_t4;

        a_s23 = a_s14 - (CRTM_15_2 * a_s21);
        a_t5 = CRTM_15_10 * ((CRTM_15_6 * a_s22) - a_in2);
        a_s24 = a_t5 + a_s23;
        a_s25 = a_s23 - a_t5;
        a_s26 = a_t2 - a_t1;
        a_s27 = a_t1 + a_t2;

        a_s28 = a_s24 - a_s27;
        a_s29 = (CRTM_15_4 * a_s12) + (CRTM_15_3 * a_s6);
        a_s30 = (CRTM_15_8 * a_s1) - (CRTM_15_7 * a_s7);
        a_s31 = a_s30 - a_s29;
        a_s32 = a_s29 + a_s30;

        // Output point 3: X(2)
        out[out_strides[2]] = a_s28 + a_s31;
        // Output point 9: X(8)
        out[out_strides[8]] = a_s28 - a_s31;

        a_s33 = a_s24 + a_s27;
        a_s34 = (CRTM_15_8 * a_s7) + (CRTM_15_7 * a_s1);
        a_s35 = (CRTM_15_3 * a_s12) - (CRTM_15_4 * a_s6);
        a_s36 = a_s35 + a_s34;

        // Output point 15: X(14)
        out[out_strides[14]] = a_s33 + a_s36;
        // Output point 27: X(26)
        out[out_strides[26]] = a_s33 - a_s36;

        a_s37 = a_s25 - a_s26;
        a_s38 = a_s35 - a_s34;
        // Output point 5: X(4)
        out[out_strides[4]] = a_s37 + a_s38;
        // Output point 17: X(16)
        out[out_strides[16]] = a_s37 - a_s38;

        a_s39 = a_s25 + a_s26;

        // Output point 23: X(22)
        out[out_strides[22]] = a_s39 - a_s32;
        // Output point 29: X(28)
        out[out_strides[28]] = a_s39 + a_s32;

        /* Shifted DFT */
        DOUBLE b_in0, b_in1, b_in2, b_in3, b_in4, b_in5, b_in6, b_in7, b_in8,
               b_in9, b_in10, b_in11, b_in12, b_in13, b_in14;
        DOUBLE b_s0, b_s1, b_s2, b_s3, b_s4, b_s5, b_s6, b_s7, b_s8, b_s9,
               b_s10, b_s11, b_s12, b_s13, b_s14, b_s15, b_s16, b_s17, b_s18,
               b_s19, b_s20, b_s21, b_s22, b_s23, b_s24, b_s25, b_s26, b_s27,
               b_s28, b_s29, b_s30, b_s31, b_s32, b_s33, b_s34, b_s35, b_s36,
               b_s37, b_s38, b_s39, b_s40, b_s41, b_s42, b_s43, b_s44, b_s45,
               b_s46, b_s47, b_s48, b_t0, b_t1, b_t2, b_t3, b_t4, b_t5, b_t6,
               b_t7, b_t8, b_t9, b_t10, b_t11, b_t12, b_t13, b_t14, b_t15,
               b_t16, b_t17, b_t18, b_t19, b_t20, b_t21, b_t22, b_t23, b_t24,
               b_t25;

        // Input point 2: x(1)
        b_in0 = in[in_strides[1]];
        // Input point 3: x(2)
        b_in1 = in[in_strides[2]];
        // Input point 6: x(5)
        b_in2 = in[in_strides[5]];
        // Input point 7: x(6)
        b_in3 = in[in_strides[6]];
        // Input point 10: x(9)
        b_in4 = in[in_strides[9]];
        // Input point 11: x(10)
        b_in5 = in[in_strides[10]];
        // Input point 14: x(13)
        b_in6 = in[in_strides[13]];
        // Input point 15: x(14)
        b_in7 = in[in_strides[14]];
        // Input point 18: x(17)
        b_in8 = in[in_strides[17]];
        // Input point 19: x(18)
        b_in9 = in[in_strides[18]];
        // Input point 22: x(21)
        b_in10 = in[in_strides[21]];
        // Input point 23: x(22)
        b_in11 = in[in_strides[22]];
        // Input point 26: x(25)
        b_in12 = in[in_strides[25]];
        // Input point 27: x(26)
        b_in13 = in[in_strides[26]];
        // Input point 30: x(29)
        b_in14 = in[in_strides[29]];

        b_s0 = b_in0 + b_in10;
        b_s1 = b_in0 - b_in10;
        b_s2 = b_in1 + b_in11;
        b_s3 = b_in1 - b_in11;
        b_s4 = b_in6 + b_in12;
        b_s5 = b_in6 - b_in12;
        b_s6 = b_in7 + b_in13;
        b_s7 = b_in7 - b_in13;
        b_s8 = b_in2 + b_in8;
        b_s9 = b_in2 - b_in8;

        b_s10 = b_s0 - b_s4;
        b_t0 = CRTM_15_14 * b_s10;
        b_t1 = CRTM_15_1 * b_s9;
        b_t2 = CRTM_15_5 * b_t0;
        b_s11 = b_t2 - b_t1;
        b_t3 = CRTM_15_2 * b_s8;
        b_s12 = b_in14 - b_t3;
        b_s13 = b_s0 + b_s4;

        b_t4 = CRTM_15_6 * b_s13;
        b_s14 = b_in4 - b_t4;
        b_t5 = CRTM_15_5 * b_s14;
        b_s15 = b_s12 + b_t5;
        b_s16 = b_s11 - b_s15;
        b_s17 = b_s11 + b_s15;

        b_t6 = CRTM_15_13 * b_s7;
        b_t7 = CRTM_15_12 * b_s2;
        b_t8 = CRTM_15_4 * b_in9;
        b_t9 = CRTM_15_3 * b_in3;
        b_s18 = b_t6 - b_t7;
        b_t10 = CRTM_15_5 * b_s18;
        b_s19 = b_t8 - b_t9;
        b_s20 = b_t10 + b_s19;
        // Output pt 8: X(7)
        out[out_strides[7]] = b_s20 + b_s16;
        // Output pt 26: X(25)
        out[out_strides[25]] = b_s20 - b_s16;

        b_s21 = b_s3 + b_s6;
        b_s22 = b_in5 - b_s21;
        b_t11 = CRTM_15_10 * b_s22;
        b_s23 = b_s13 + b_in4;
        b_t12 = CRTM_15_5 * b_s8;
        b_s24 = b_in14 + b_t12;
        b_s25 = b_s23 - b_s24;
        // Output pt 12: X(11)
        out[out_strides[11]] = b_t11 + b_s25;
        // Output pt 22: X(21)
        out[out_strides[21]] = b_t11 - b_s25;

        b_t13 = CRTM_15_5 * b_s23;
        // Output pt 2: X(1)
        out[out_strides[1]] = b_t13 + b_s24;

        b_t14 = CRTM_15_4 * b_in3;
        b_t15 = CRTM_15_3 * b_in9;
        b_s26 = b_t14 + b_t15;

        b_t16 = CRTM_15_13 * b_s2;
        b_t17 = CRTM_15_12 * b_s7;
        b_s27 = b_t16 + b_t17;
        b_t18 = CRTM_15_5 * b_s27;
        b_s28 = b_s26 - b_t18;
        // Output pt 14: X(13)
        out[out_strides[13]] = b_s28 + b_s17;
        // Output pt 20: X(19)
        out[out_strides[19]] = b_s28 - b_s17;

        b_t19 = CRTM_15_11 * b_s21;
        b_t20 = CRTM_15_10 * b_in5;
        b_s29 = b_t19 + b_t20;
        b_s30 = b_s3 - b_s6;
        b_t21 = CRTM_15_9 * b_s30;
        b_s31 = b_t21 - b_s29;
        b_s32 = b_s29 + b_t21;
        b_s33 = b_t0 + b_t1;
        b_s34 = b_s12 - b_s14;
        b_s35 = b_s33 + b_s34;
        b_s36 = b_s33 - b_s34;
        b_s37 = b_s27 + b_s26;

        b_t22 = CRTM_15_8 * b_s1;
        b_t23 = CRTM_15_7 * b_s5;
        b_s38 = b_t22 + b_t23;
        b_t24 = CRTM_15_8 * b_s5;
        b_t25 = CRTM_15_7 * b_s1;
        b_s39 = b_t24 - b_t25;
        b_s40 = b_s18 - b_s19;
        b_s41 = b_s36 - b_s37;
        b_s42 = b_s36 + b_s37;
        b_s43 = b_s38 + b_s31;
        // Output pt 4: X(3)
        out[out_strides[3]] = b_s41 + b_s43;
        // Output pt 24: X(23)
        out[out_strides[23]] = b_s41 - b_s43;

        b_s44 = b_s38 - b_s31;
        // Output pt 10: X(9)
        out[out_strides[9]] = b_s44 - b_s42;
        // Output pt 30: X(29)
        out[out_strides[29]] = -(b_s42 + b_s44);

        b_s45 = b_s39 - b_s32;
        b_s46 = b_s32 + b_s39;
        b_s47 = b_s35 + b_s40;
        // Output pt 18: X(17)
        out[out_strides[17]] = b_s45 + b_s47;
        // Output pt 28: X(27)
        out[out_strides[27]] = b_s45 - b_s47;

        b_s48 = b_s35 - b_s40;
        // Output pt 6: X(5)
        out[out_strides[5]] = b_s48 - b_s46;
        // Output pt 16: X(15)
        out[out_strides[15]] = -(b_s46 + b_s48);

        in  = in + v_in_stride;
        out = out + v_out_stride;
    }
    AOCLFFTZ_LOG(DEBUG, global_logger_mode, "Exit");
}

kfft_ register_kernel_r2hcf_rfft15c(UINT8 precision, UINT8 direction)
{
    if (direction == FORWARD_FFT_DIR)
    {
        if (precision == DT_FLOAT)
        {
            return r2hcf_rfft15c_fp32_fwd;
        }
        else if (precision == DT_DOUBLE)
        {
            return r2hcf_rfft15c_fp64_fwd;
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
            return r2hcf_rfft15c_fp32_bwd;
        }
        else if (precision == DT_DOUBLE)
        {
            return r2hcf_rfft15c_fp64_bwd;
        }
        else
        {
            return NULL;
        }
    }
}
