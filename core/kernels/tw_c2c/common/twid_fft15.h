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

/** @file twid_fft15.h
 *
 *  @brief The ISA generic kernel template for the radix 15 twiddle kernel
 *
 *  This file contains the DIT twiddle radix-15 FFT implementations for
 *  single-precision and double-precision inputs.
 *
 *  @author Ashwin K. Godbole
 */

// This header has no include guards.
// This is intentional.
// The functions defined in this file are not usable by default.
// They are "instantiated" only when "included" in another file.

#include "core/kernels/simd_includes/generic_kernels_common.h"

static VOID TWID_KNAME_FP32(VOID *in_real, VOID *in_imag, VOID *out_real,
                            VOID *out_imag, INTP n, aoclfftz_strides_t *strides,
                            VOID *twd, UINT8 flag)
{
#ifdef AOCL_ENABLE_LOG
    AOCLFFTZ_LOG_UNFORMATTED(TRACE, TRACE, "Enter");
#endif

    const FLOAT CRTM_15[6] = {
        0.55901699437494742410229341718281905886015458990288f,
        0.25000000000000000000000000000000000000000000000000f,
        0.95105651629515357211643933337938214340569863400000f,
        0.58778525229247301629891039327884007596190389052978f,
        0.50000000000000000000000000000000000000000000000000f,
        0.86602540378443864676372317075293618347140262690519f};

    FLOAT *in_r = in_real;
    FLOAT *out_r = out_real;
    FLOAT *curr_in, *curr_out;
#ifdef VOLATILE_STRIDE_ARRAY
    volatile INTP *in_strides = strides->in_strides;
    volatile INTP *out_strides = strides->out_strides;
#else
    INTP *in_strides = strides->in_strides;
    INTP *out_strides = strides->out_strides;
#endif
    INTP v_in_stride = strides->v_in_stride;
    INTP v_out_stride = strides->v_out_stride;

    aoclfftz_twiddle_t *tws = (aoclfftz_twiddle_t *)twd;
    FLOAT *tw = (FLOAT *)(tws->TW);
    UINTP cols = tws->cols;

    INTP N = n / NUM_SETS_S;
    INTP remaining_sets = n % NUM_SETS_S;

#if defined(KERNEL_USE_AVX512)
    INTP do_256_whole = (INTP)(remaining_sets >= NUM_SETS_256_S);
    INTP do_128_whole =
        (INTP)(remaining_sets % NUM_SETS_256_S >= NUM_SETS_128_S);
    INTP cnt_256 = N * NUM_SETS_512_S;
    INTP cnt_128 = N * NUM_SETS_512_S + do_256_whole * NUM_SETS_256_S;
    INTP cnt_128_low = N * NUM_SETS_512_S + do_256_whole * NUM_SETS_256_S +
                       do_128_whole * NUM_SETS_128_S;
#elif defined(KERNEL_USE_AVX256)
    INTP do_128_whole = (INTP)(remaining_sets >= NUM_SETS_128_S);
    INTP cnt_128 = N * NUM_SETS_256_S;
    INTP cnt_128_low = N * NUM_SETS_256_S + do_128_whole * NUM_SETS_128_S;
#elif defined(KERNEL_USE_AVX128)
    INTP cnt_128_low = N * NUM_SETS_128_S;
#endif

    VREGTYPE_S v_C1 = BCAST_S(CRTM_15[0]);
    VREGTYPE_S v_C2 = BCAST_S(CRTM_15[1]);
    VREGTYPE_S v_C3 = BCAST_S(CRTM_15[2]);
    VREGTYPE_S v_C4 = BCAST_S(CRTM_15[3]);
    VREGTYPE_S v_C5 = BCAST_S(CRTM_15[4]);
    VREGTYPE_S v_C6 = BCAST_S(CRTM_15[5]);

    INTP count;

    if (flag)
    {
        in_r = in_imag;
        out_r = out_imag;

        v_C3 = NEG_S(v_C3);
        v_C4 = NEG_S(v_C4);
        v_C6 = NEG_S(v_C6);
    }

    for (count = 0; count < N; count++)
    {
        VREGTYPE_S v_in0, v_in1, v_in2, v_in3, v_in4, v_in5, v_in6, v_in7,
            v_in8;
        VREGTYPE_S v_in9, v_in10, v_in11, v_in12, v_in13, v_in14;
        VREGTYPE_S v_cv1, v_cv2, v_cv3, v_cv4, v_cv5, v_cv6, v_cv7, v_cv8,
            v_cv9, v_cv10;
        VREGTYPE_S v_cv11, v_cv12;
        VREGTYPE_S v_av1, v_av2, v_av3, v_av4, v_av5, v_av6, v_av7, v_av8,
            v_av9;
        VREGTYPE_S v_av11, v_av12, v_av13, v_av14, v_av15, v_av16, v_av17;
        VREGTYPE_S v_av18, v_av19, v_av20, v_av21, v_av22, v_av23, v_av24,
            v_av25;
        VREGTYPE_S v_av26, v_av27, v_av28, v_av29, v_av30, v_av31, v_av32,
            v_av33;
        VREGTYPE_S v_av34, v_av35, v_av36, v_av37, v_av38, v_av39, v_av40,
            v_av41;
        VREGTYPE_S v_av42, v_av44, v_av45, v_av46, v_av47, v_av48, v_av49;
        VREGTYPE_S v_tv1, v_tv2, v_tv3, v_tv4, v_tv5, v_tv6, v_tv7, v_tv8,
            v_tv9, v_tv10;
        VREGTYPE_S v_tv11, v_tv16, v_tv17, v_tv18;
        VREGTYPE_S v_tv19, v_tv20, v_tv21, v_tv22, v_tv23, v_tv24, v_tv25,
            v_tv27;
        VREGTYPE_S v_out0, v_out1, v_out2, v_out3, v_out4, v_out5;
        VREGTYPE_S v_out6, v_out7, v_out8, v_out9, v_out10, v_out11;
        VREGTYPE_S v_out12, v_out13, v_out14;

        curr_in = in_r;
        curr_out = out_r;

        if (flag)
        {
            ITW_GATHER_S(curr_in, in_strides, 1, v_in_stride, v_in1, tw, cols,
                         (count * NUM_SETS_S));
            ITW_GATHER_S(curr_in, in_strides, 2, v_in_stride, v_in2, tw, cols,
                         (count * NUM_SETS_S));
            ITW_GATHER_S(curr_in, in_strides, 3, v_in_stride, v_in3, tw, cols,
                         (count * NUM_SETS_S));
            ITW_GATHER_S(curr_in, in_strides, 4, v_in_stride, v_in4, tw, cols,
                         (count * NUM_SETS_S));
            ITW_GATHER_S(curr_in, in_strides, 5, v_in_stride, v_in5, tw, cols,
                         (count * NUM_SETS_S));
            ITW_GATHER_S(curr_in, in_strides, 6, v_in_stride, v_in6, tw, cols,
                         (count * NUM_SETS_S));
            ITW_GATHER_S(curr_in, in_strides, 7, v_in_stride, v_in7, tw, cols,
                         (count * NUM_SETS_S));
            ITW_GATHER_S(curr_in, in_strides, 8, v_in_stride, v_in8, tw, cols,
                         (count * NUM_SETS_S));
            ITW_GATHER_S(curr_in, in_strides, 9, v_in_stride, v_in9, tw, cols,
                         (count * NUM_SETS_S));
            ITW_GATHER_S(curr_in, in_strides, 10, v_in_stride, v_in10, tw, cols,
                         (count * NUM_SETS_S));
            ITW_GATHER_S(curr_in, in_strides, 11, v_in_stride, v_in11, tw, cols,
                         (count * NUM_SETS_S));
            ITW_GATHER_S(curr_in, in_strides, 12, v_in_stride, v_in12, tw, cols,
                         (count * NUM_SETS_S));
            ITW_GATHER_S(curr_in, in_strides, 13, v_in_stride, v_in13, tw, cols,
                         (count * NUM_SETS_S));
            ITW_GATHER_S(curr_in, in_strides, 14, v_in_stride, v_in14, tw, cols,
                         (count * NUM_SETS_S));
        }
        else
        {
            TW_GATHER_S(curr_in, in_strides, 1, v_in_stride, v_in1, tw, cols,
                        (count * NUM_SETS_S));
            TW_GATHER_S(curr_in, in_strides, 2, v_in_stride, v_in2, tw, cols,
                        (count * NUM_SETS_S));
            TW_GATHER_S(curr_in, in_strides, 3, v_in_stride, v_in3, tw, cols,
                        (count * NUM_SETS_S));
            TW_GATHER_S(curr_in, in_strides, 4, v_in_stride, v_in4, tw, cols,
                        (count * NUM_SETS_S));
            TW_GATHER_S(curr_in, in_strides, 5, v_in_stride, v_in5, tw, cols,
                        (count * NUM_SETS_S));
            TW_GATHER_S(curr_in, in_strides, 6, v_in_stride, v_in6, tw, cols,
                        (count * NUM_SETS_S));
            TW_GATHER_S(curr_in, in_strides, 7, v_in_stride, v_in7, tw, cols,
                        (count * NUM_SETS_S));
            TW_GATHER_S(curr_in, in_strides, 8, v_in_stride, v_in8, tw, cols,
                        (count * NUM_SETS_S));
            TW_GATHER_S(curr_in, in_strides, 9, v_in_stride, v_in9, tw, cols,
                        (count * NUM_SETS_S));
            TW_GATHER_S(curr_in, in_strides, 10, v_in_stride, v_in10, tw, cols,
                        (count * NUM_SETS_S));
            TW_GATHER_S(curr_in, in_strides, 11, v_in_stride, v_in11, tw, cols,
                        (count * NUM_SETS_S));
            TW_GATHER_S(curr_in, in_strides, 12, v_in_stride, v_in12, tw, cols,
                        (count * NUM_SETS_S));
            TW_GATHER_S(curr_in, in_strides, 13, v_in_stride, v_in13, tw, cols,
                        (count * NUM_SETS_S));
            TW_GATHER_S(curr_in, in_strides, 14, v_in_stride, v_in14, tw, cols,
                        (count * NUM_SETS_S));
        }

        GATHER_S(curr_in, v_in_stride, v_in0);

        // common calculations
        v_cv1 = ADD_S(v_in10, v_in5);
        v_cv2 = ADD_S(v_in0, v_cv1);
        v_cv3 = ADD_S(v_in14, v_in4);
        v_cv4 = ADD_S(v_in9, v_cv3);
        v_cv5 = ADD_S(v_in11, v_in1);
        v_cv6 = ADD_S(v_in6, v_cv5);
        v_av1 = ADD_S(v_cv4, v_cv6);
        v_cv7 = ADD_S(v_in2, v_in7);
        v_cv8 = ADD_S(v_in12, v_cv7);
        v_cv9 = ADD_S(v_in13, v_in8);
        v_cv10 = ADD_S(v_in3, v_cv9);
        v_av2 = ADD_S(v_cv8, v_cv10);
        v_av3 = ADD_S(v_av1, v_av2);

        // Output point 1 : X[0]
        v_out0 = ADD_S(v_cv2, v_av3);

        v_tv1 = MUL_S(v_C2, v_av3);
        v_av4 = SUB_S(v_cv2, v_tv1);
        v_av5 = SUB_S(v_av1, v_av2);
        v_tv2 = MUL_S(v_C1, v_av5);
        v_av6 = ADD_S(v_av4, v_tv2);
        v_av7 = SUB_S(v_cv6, v_cv4);
        v_av8 = SUB_S(v_cv8, v_cv10);
        v_tv3 = MUL_S(v_C3, v_av7);
        v_tv4 = MUL_S(v_C4, v_av8);
        v_av9 = ADD_S(v_tv3, v_tv4);

        v_av9 = SWAP_RI_S(CONJ_S(v_av9));
        // Output point 13 : X[12]
        v_out12 = ADD_S(v_av6, v_av9);
        // Output point 4 : X[3]
        v_out3 = SUB_S(v_av6, v_av9);

        v_av6 = SUB_S(v_av4, v_tv2);
        v_tv3 = MUL_S(v_C3, v_av8);
        v_tv4 = MUL_S(v_C4, v_av7);
        v_av9 = SUB_S(v_tv4, v_tv3);

        v_av9 = SWAP_RI_S(CONJ_S(v_av9));
        // Output point 10 : X[9]
        v_out9 = ADD_S(v_av6, v_av9);
        // Output point 7 : X[6]
        v_out6 = SUB_S(v_av6, v_av9);

        v_tv5 = MUL_S(v_C5, v_cv1);
        v_av11 = SUB_S(v_in0, v_tv5);

        v_tv6 = MUL_S(v_C5, v_cv5);
        v_av12 = SUB_S(v_in6, v_tv6);
        v_tv7 = MUL_S(v_C5, v_cv3);
        v_av13 = SUB_S(v_in9, v_tv7);
        v_av14 = ADD_S(v_av12, v_av13);
        v_tv8 = MUL_S(v_C5, v_cv7);
        v_av15 = SUB_S(v_in12, v_tv8);
        v_tv9 = MUL_S(v_C5, v_cv9);
        v_av16 = SUB_S(v_in3, v_tv9);
        v_av17 = ADD_S(v_av15, v_av16);
        v_av18 = ADD_S(v_av14, v_av17);
        v_tv10 = MUL_S(v_C2, v_av18);
        v_av19 = SUB_S(v_av11, v_tv10);
        v_av20 = SUB_S(v_av14, v_av17);
        v_tv11 = MUL_S(v_C1, v_av20);
        v_av21 = ADD_S(v_av19, v_tv11);
        v_cv11 = SUB_S(v_in14, v_in4);
        v_av29 = MUL_S(v_C6, v_cv11);
        v_cv12 = SUB_S(v_in11, v_in1);
        v_av28 = MUL_S(v_C6, v_cv12);
        v_av22 = SUB_S(v_av29, v_av28);
        v_av23 = SUB_S(v_in8, v_in13);
        v_av32 = MUL_S(v_C6, v_av23);
        v_av24 = SUB_S(v_in2, v_in7);
        v_av31 = MUL_S(v_C6, v_av24);
        v_av25 = SUB_S(v_av32, v_av31);
        v_tv16 = MUL_S(v_C3, v_av22);
        v_tv17 = MUL_S(v_C4, v_av25);
        v_av26 = ADD_S(v_tv16, v_tv17);
        v_av27 = SUB_S(v_in10, v_in5);
        v_tv18 = MUL_S(v_C6, v_av27);
        v_av30 = ADD_S(v_av28, v_av29);
        v_av33 = ADD_S(v_av31, v_av32);
        v_av34 = ADD_S(v_av30, v_av33);
        v_tv19 = MUL_S(v_C2, v_av34);
        v_av35 = ADD_S(v_tv18, v_tv19);
        v_av36 = SUB_S(v_av33, v_av30);
        v_tv20 = MUL_S(v_C1, v_av36);
        v_av37 = ADD_S(v_av35, v_tv20);
        v_av38 = SUB_S(v_av13, v_av12);
        v_av39 = SUB_S(v_av16, v_av15);
        v_tv21 = MUL_S(v_C3, v_av38);
        v_tv22 = MUL_S(v_C4, v_av39);
        v_av40 = ADD_S(v_tv21, v_tv22);

        v_av41 = ADD_S(v_av11, v_av18);
        v_av42 = SUB_S(v_tv18, v_av34);
        v_av42 = SWAP_RI_S(CONJ_S(v_av42));
        // Output point 11 : X[10]
        v_out10 = ADD_S(v_av41, v_av42);
        // Output point 6 : X[5]
        v_out5 = SUB_S(v_av41, v_av42);

        v_av41 = ADD_S(v_av21, v_av26);
        v_av42 = ADD_S(v_av37, v_av40);
        v_av42 = SWAP_RI_S(CONJ_S(v_av42));
        // Output point 14 : X[13]
        v_out13 = ADD_S(v_av41, v_av42);
        // Output point 3 : X[2]
        v_out2 = SUB_S(v_av41, v_av42);

        v_av44 = SUB_S(v_av19, v_tv11);
        v_tv23 = MUL_S(v_C3, v_av25);
        v_tv24 = MUL_S(v_C4, v_av22);
        v_av45 = SUB_S(v_tv23, v_tv24);
        v_av46 = SUB_S(v_av35, v_tv20);
        v_tv25 = MUL_S(v_C3, v_av39);
        v_tv27 = MUL_S(v_C4, v_av38);
        v_av47 = SUB_S(v_tv25, v_tv27);

        v_av41 = ADD_S(v_av44, v_av45);
        v_av42 = ADD_S(v_av46, v_av47);
        v_av42 = SWAP_RI_S(CONJ_S(v_av42));
        // Output point 5 : X[4]
        v_out4 = ADD_S(v_av41, v_av42);
        // Output point 12 : X[11]
        v_out11 = SUB_S(v_av41, v_av42);

        v_av48 = SUB_S(v_av44, v_av45);
        v_av49 = SUB_S(v_av47, v_av46);
        v_av49 = SWAP_RI_S(CONJ_S(v_av49));
        // Output point 15 : X[14]
        v_out14 = ADD_S(v_av48, v_av49);
        // Output point 2 : X[1]
        v_out1 = SUB_S(v_av48, v_av49);

        v_av48 = SUB_S(v_av21, v_av26);
        v_av49 = SUB_S(v_av40, v_av37);
        v_av49 = SWAP_RI_S(CONJ_S(v_av49));
        // Output point 15 : X[14]
        v_out8 = ADD_S(v_av48, v_av49);
        // Output point 2 : X[1]
        v_out7 = SUB_S(v_av48, v_av49);

        SCATTER_S(curr_out, v_out_stride, v_out0);
        SCATTER_S(curr_out + out_strides[1], v_out_stride, v_out1);
        SCATTER_S(curr_out + out_strides[2], v_out_stride, v_out2);
        SCATTER_S(curr_out + out_strides[3], v_out_stride, v_out3);
        SCATTER_S(curr_out + out_strides[4], v_out_stride, v_out4);
        SCATTER_S(curr_out + out_strides[5], v_out_stride, v_out5);
        SCATTER_S(curr_out + out_strides[6], v_out_stride, v_out6);
        SCATTER_S(curr_out + out_strides[7], v_out_stride, v_out7);
        SCATTER_S(curr_out + out_strides[8], v_out_stride, v_out8);
        SCATTER_S(curr_out + out_strides[9], v_out_stride, v_out9);
        SCATTER_S(curr_out + out_strides[10], v_out_stride, v_out10);
        SCATTER_S(curr_out + out_strides[11], v_out_stride, v_out11);
        SCATTER_S(curr_out + out_strides[12], v_out_stride, v_out12);
        SCATTER_S(curr_out + out_strides[13], v_out_stride, v_out13);
        SCATTER_S(curr_out + out_strides[14], v_out_stride, v_out14);

        in_r += NUM_SETS_S * v_in_stride;
        out_r += NUM_SETS_S * v_out_stride;
    }

    // The following contains code that performs the FFT on the tail cases.
    // These tails are conditionally "instantiated" based on the ISA requested
    // by the "includee" file.

#if defined(KERNEL_USE_AVX512)
    if (remaining_sets >= NUM_SETS_256_S)
    {
        __m256 v_in0, v_in1, v_in2, v_in3, v_in4, v_in5, v_in6, v_in7, v_in8;
        __m256 v_in9, v_in10, v_in11, v_in12, v_in13, v_in14;
        __m256 v_cv1, v_cv2, v_cv3, v_cv4, v_cv5, v_cv6, v_cv7, v_cv8, v_cv9,
            v_cv10;
        __m256 v_cv11, v_cv12;
        __m256 v_av1, v_av2, v_av3, v_av4, v_av5, v_av6, v_av7, v_av8, v_av9;
        __m256 v_av11, v_av12, v_av13, v_av14, v_av15, v_av16, v_av17;
        __m256 v_av18, v_av19, v_av20, v_av21, v_av22, v_av23, v_av24, v_av25;
        __m256 v_av26, v_av27, v_av28, v_av29, v_av30, v_av31, v_av32, v_av33;
        __m256 v_av34, v_av35, v_av36, v_av37, v_av38, v_av39, v_av40, v_av41;
        __m256 v_av42, v_av44, v_av45, v_av46, v_av47, v_av48, v_av49;
        __m256 v_tv1, v_tv2, v_tv3, v_tv4, v_tv5, v_tv6, v_tv7, v_tv8, v_tv9,
            v_tv10;
        __m256 v_tv11, v_tv16, v_tv17, v_tv18;
        __m256 v_tv19, v_tv20, v_tv21, v_tv22, v_tv23, v_tv24, v_tv25, v_tv27;
        __m256 v_out0, v_out1, v_out2, v_out3, v_out4, v_out5;
        __m256 v_out6, v_out7, v_out8, v_out9, v_out10, v_out11;
        __m256 v_out12, v_out13, v_out14;

        __m256 v_K1 = CAST_512_TO_256_S(v_C1);
        __m256 v_K2 = CAST_512_TO_256_S(v_C2);
        __m256 v_K3 = CAST_512_TO_256_S(v_C3);
        __m256 v_K4 = CAST_512_TO_256_S(v_C4);
        __m256 v_K5 = CAST_512_TO_256_S(v_C5);
        __m256 v_K6 = CAST_512_TO_256_S(v_C6);

        curr_in = in_r;
        curr_out = out_r;

        if (flag)
        {
            ITW_GATHER_256_S(curr_in, in_strides, 1, v_in_stride, v_in1, tw,
                             cols, cnt_256);
            ITW_GATHER_256_S(curr_in, in_strides, 2, v_in_stride, v_in2, tw,
                             cols, cnt_256);
            ITW_GATHER_256_S(curr_in, in_strides, 3, v_in_stride, v_in3, tw,
                             cols, cnt_256);
            ITW_GATHER_256_S(curr_in, in_strides, 4, v_in_stride, v_in4, tw,
                             cols, cnt_256);
            ITW_GATHER_256_S(curr_in, in_strides, 5, v_in_stride, v_in5, tw,
                             cols, cnt_256);
            ITW_GATHER_256_S(curr_in, in_strides, 6, v_in_stride, v_in6, tw,
                             cols, cnt_256);
            ITW_GATHER_256_S(curr_in, in_strides, 7, v_in_stride, v_in7, tw,
                             cols, cnt_256);
            ITW_GATHER_256_S(curr_in, in_strides, 8, v_in_stride, v_in8, tw,
                             cols, cnt_256);
            ITW_GATHER_256_S(curr_in, in_strides, 9, v_in_stride, v_in9, tw,
                             cols, cnt_256);
            ITW_GATHER_256_S(curr_in, in_strides, 10, v_in_stride, v_in10, tw,
                             cols, cnt_256);
            ITW_GATHER_256_S(curr_in, in_strides, 11, v_in_stride, v_in11, tw,
                             cols, cnt_256);
            ITW_GATHER_256_S(curr_in, in_strides, 12, v_in_stride, v_in12, tw,
                             cols, cnt_256);
            ITW_GATHER_256_S(curr_in, in_strides, 13, v_in_stride, v_in13, tw,
                             cols, cnt_256);
            ITW_GATHER_256_S(curr_in, in_strides, 14, v_in_stride, v_in14, tw,
                             cols, cnt_256);
        }
        else
        {
            TW_GATHER_256_S(curr_in, in_strides, 1, v_in_stride, v_in1, tw,
                            cols, cnt_256);
            TW_GATHER_256_S(curr_in, in_strides, 2, v_in_stride, v_in2, tw,
                            cols, cnt_256);
            TW_GATHER_256_S(curr_in, in_strides, 3, v_in_stride, v_in3, tw,
                            cols, cnt_256);
            TW_GATHER_256_S(curr_in, in_strides, 4, v_in_stride, v_in4, tw,
                            cols, cnt_256);
            TW_GATHER_256_S(curr_in, in_strides, 5, v_in_stride, v_in5, tw,
                            cols, cnt_256);
            TW_GATHER_256_S(curr_in, in_strides, 6, v_in_stride, v_in6, tw,
                            cols, cnt_256);
            TW_GATHER_256_S(curr_in, in_strides, 7, v_in_stride, v_in7, tw,
                            cols, cnt_256);
            TW_GATHER_256_S(curr_in, in_strides, 8, v_in_stride, v_in8, tw,
                            cols, cnt_256);
            TW_GATHER_256_S(curr_in, in_strides, 9, v_in_stride, v_in9, tw,
                            cols, cnt_256);
            TW_GATHER_256_S(curr_in, in_strides, 10, v_in_stride, v_in10, tw,
                            cols, cnt_256);
            TW_GATHER_256_S(curr_in, in_strides, 11, v_in_stride, v_in11, tw,
                            cols, cnt_256);
            TW_GATHER_256_S(curr_in, in_strides, 12, v_in_stride, v_in12, tw,
                            cols, cnt_256);
            TW_GATHER_256_S(curr_in, in_strides, 13, v_in_stride, v_in13, tw,
                            cols, cnt_256);
            TW_GATHER_256_S(curr_in, in_strides, 14, v_in_stride, v_in14, tw,
                            cols, cnt_256);
        }

        GATHER4_256_S(curr_in, v_in_stride, v_in0);

        // common calculations
        v_cv1 = _mm256_add_ps(v_in10, v_in5);
        v_cv2 = _mm256_add_ps(v_in0, v_cv1);
        v_cv3 = _mm256_add_ps(v_in14, v_in4);
        v_cv4 = _mm256_add_ps(v_in9, v_cv3);
        v_cv5 = _mm256_add_ps(v_in11, v_in1);
        v_cv6 = _mm256_add_ps(v_in6, v_cv5);
        v_av1 = _mm256_add_ps(v_cv4, v_cv6);
        v_cv7 = _mm256_add_ps(v_in2, v_in7);
        v_cv8 = _mm256_add_ps(v_in12, v_cv7);
        v_cv9 = _mm256_add_ps(v_in13, v_in8);
        v_cv10 = _mm256_add_ps(v_in3, v_cv9);
        v_av2 = _mm256_add_ps(v_cv8, v_cv10);
        v_av3 = _mm256_add_ps(v_av1, v_av2);

        // Output point 1 : X[0]
        v_out0 = _mm256_add_ps(v_cv2, v_av3);

        v_tv1 = _mm256_mul_ps(v_K2, v_av3);
        v_av4 = _mm256_sub_ps(v_cv2, v_tv1);
        v_av5 = _mm256_sub_ps(v_av1, v_av2);
        v_tv2 = _mm256_mul_ps(v_K1, v_av5);
        v_av6 = _mm256_add_ps(v_av4, v_tv2);
        v_av7 = _mm256_sub_ps(v_cv6, v_cv4);
        v_av8 = _mm256_sub_ps(v_cv8, v_cv10);
        v_tv3 = _mm256_mul_ps(v_K3, v_av7);
        v_tv4 = _mm256_mul_ps(v_K4, v_av8);
        v_av9 = _mm256_add_ps(v_tv3, v_tv4);

        v_av9 = SWAP_RI_256_S(CONJ_256_S(v_av9));
        // Output point 13 : X[12]
        v_out12 = _mm256_add_ps(v_av6, v_av9);
        // Output point 4 : X[3]
        v_out3 = _mm256_sub_ps(v_av6, v_av9);

        v_av6 = _mm256_sub_ps(v_av4, v_tv2);
        v_tv3 = _mm256_mul_ps(v_K3, v_av8);
        v_tv4 = _mm256_mul_ps(v_K4, v_av7);
        v_av9 = _mm256_sub_ps(v_tv4, v_tv3);

        v_av9 = SWAP_RI_256_S(CONJ_256_S(v_av9));
        // Output point 10 : X[9]
        v_out9 = _mm256_add_ps(v_av6, v_av9);
        // Output point 7 : X[6]
        v_out6 = _mm256_sub_ps(v_av6, v_av9);

        v_tv5 = _mm256_mul_ps(v_K5, v_cv1);
        v_av11 = _mm256_sub_ps(v_in0, v_tv5);

        v_tv6 = _mm256_mul_ps(v_K5, v_cv5);
        v_av12 = _mm256_sub_ps(v_in6, v_tv6);
        v_tv7 = _mm256_mul_ps(v_K5, v_cv3);
        v_av13 = _mm256_sub_ps(v_in9, v_tv7);
        v_av14 = _mm256_add_ps(v_av12, v_av13);
        v_tv8 = _mm256_mul_ps(v_K5, v_cv7);
        v_av15 = _mm256_sub_ps(v_in12, v_tv8);
        v_tv9 = _mm256_mul_ps(v_K5, v_cv9);
        v_av16 = _mm256_sub_ps(v_in3, v_tv9);
        v_av17 = _mm256_add_ps(v_av15, v_av16);
        v_av18 = _mm256_add_ps(v_av14, v_av17);
        v_tv10 = _mm256_mul_ps(v_K2, v_av18);
        v_av19 = _mm256_sub_ps(v_av11, v_tv10);
        v_av20 = _mm256_sub_ps(v_av14, v_av17);
        v_tv11 = _mm256_mul_ps(v_K1, v_av20);
        v_av21 = _mm256_add_ps(v_av19, v_tv11);
        v_cv11 = _mm256_sub_ps(v_in14, v_in4);
        v_av29 = _mm256_mul_ps(v_K6, v_cv11);
        v_cv12 = _mm256_sub_ps(v_in11, v_in1);
        v_av28 = _mm256_mul_ps(v_K6, v_cv12);
        v_av22 = _mm256_sub_ps(v_av29, v_av28);
        v_av23 = _mm256_sub_ps(v_in8, v_in13);
        v_av32 = _mm256_mul_ps(v_K6, v_av23);
        v_av24 = _mm256_sub_ps(v_in2, v_in7);
        v_av31 = _mm256_mul_ps(v_K6, v_av24);
        v_av25 = _mm256_sub_ps(v_av32, v_av31);
        v_tv16 = _mm256_mul_ps(v_K3, v_av22);
        v_tv17 = _mm256_mul_ps(v_K4, v_av25);
        v_av26 = _mm256_add_ps(v_tv16, v_tv17);
        v_av27 = _mm256_sub_ps(v_in10, v_in5);
        v_tv18 = _mm256_mul_ps(v_K6, v_av27);
        v_av30 = _mm256_add_ps(v_av28, v_av29);
        v_av33 = _mm256_add_ps(v_av31, v_av32);
        v_av34 = _mm256_add_ps(v_av30, v_av33);
        v_tv19 = _mm256_mul_ps(v_K2, v_av34);
        v_av35 = _mm256_add_ps(v_tv18, v_tv19);
        v_av36 = _mm256_sub_ps(v_av33, v_av30);
        v_tv20 = _mm256_mul_ps(v_K1, v_av36);
        v_av37 = _mm256_add_ps(v_av35, v_tv20);
        v_av38 = _mm256_sub_ps(v_av13, v_av12);
        v_av39 = _mm256_sub_ps(v_av16, v_av15);
        v_tv21 = _mm256_mul_ps(v_K3, v_av38);
        v_tv22 = _mm256_mul_ps(v_K4, v_av39);
        v_av40 = _mm256_add_ps(v_tv21, v_tv22);

        // real part
        v_av41 = _mm256_add_ps(v_av11, v_av18);

        // imag part
        v_av42 = _mm256_sub_ps(v_tv18, v_av34);

        v_av42 = SWAP_RI_256_S(CONJ_256_S(v_av42));
        // Output point 11 : X[10]
        v_out10 = _mm256_add_ps(v_av41, v_av42);
        // Output point 6 : X[5]
        v_out5 = _mm256_sub_ps(v_av41, v_av42);

        // real part
        v_av41 = _mm256_add_ps(v_av21, v_av26);

        // imag part
        v_av42 = _mm256_add_ps(v_av37, v_av40);

        v_av42 = SWAP_RI_256_S(CONJ_256_S(v_av42));
        // Output point 14 : X[13]
        v_out13 = _mm256_add_ps(v_av41, v_av42);
        // Output point 3 : X[2]
        v_out2 = _mm256_sub_ps(v_av41, v_av42);

        v_av44 = _mm256_sub_ps(v_av19, v_tv11);
        v_tv23 = _mm256_mul_ps(v_K3, v_av25);
        v_tv24 = _mm256_mul_ps(v_K4, v_av22);
        v_av45 = _mm256_sub_ps(v_tv23, v_tv24);
        v_av46 = _mm256_sub_ps(v_av35, v_tv20);
        v_tv25 = _mm256_mul_ps(v_K3, v_av39);
        v_tv27 = _mm256_mul_ps(v_K4, v_av38);
        v_av47 = _mm256_sub_ps(v_tv25, v_tv27);

        // real part
        v_av41 = _mm256_add_ps(v_av44, v_av45);

        // imag part
        v_av42 = _mm256_add_ps(v_av46, v_av47);

        v_av42 = SWAP_RI_256_S(CONJ_256_S(v_av42));
        // Output point 5 : X[4]
        v_out4 = _mm256_add_ps(v_av41, v_av42);
        // Output point 12 : X[11]
        v_out11 = _mm256_sub_ps(v_av41, v_av42);

        // real part
        v_av48 = _mm256_sub_ps(v_av44, v_av45);

        // imag part
        v_av49 = _mm256_sub_ps(v_av47, v_av46);

        v_av49 = SWAP_RI_256_S(CONJ_256_S(v_av49));
        // Output point 15 : X[14]
        v_out14 = _mm256_add_ps(v_av48, v_av49);
        // Output point 2 : X[1]
        v_out1 = _mm256_sub_ps(v_av48, v_av49);

        // real part
        v_av48 = _mm256_sub_ps(v_av21, v_av26);

        // imag part
        v_av49 = _mm256_sub_ps(v_av40, v_av37);

        v_av49 = SWAP_RI_256_S(CONJ_256_S(v_av49));
        // Output point 15 : X[14]
        v_out8 = _mm256_add_ps(v_av48, v_av49);
        // Output point 2 : X[1]
        v_out7 = _mm256_sub_ps(v_av48, v_av49);

        SCATTER4_256_S(curr_out, v_out_stride, v_out0);
        SCATTER4_256_S(curr_out + out_strides[1], v_out_stride, v_out1);
        SCATTER4_256_S(curr_out + out_strides[2], v_out_stride, v_out2);
        SCATTER4_256_S(curr_out + out_strides[3], v_out_stride, v_out3);
        SCATTER4_256_S(curr_out + out_strides[4], v_out_stride, v_out4);
        SCATTER4_256_S(curr_out + out_strides[5], v_out_stride, v_out5);
        SCATTER4_256_S(curr_out + out_strides[6], v_out_stride, v_out6);
        SCATTER4_256_S(curr_out + out_strides[7], v_out_stride, v_out7);
        SCATTER4_256_S(curr_out + out_strides[8], v_out_stride, v_out8);
        SCATTER4_256_S(curr_out + out_strides[9], v_out_stride, v_out9);
        SCATTER4_256_S(curr_out + out_strides[10], v_out_stride, v_out10);
        SCATTER4_256_S(curr_out + out_strides[11], v_out_stride, v_out11);
        SCATTER4_256_S(curr_out + out_strides[12], v_out_stride, v_out12);
        SCATTER4_256_S(curr_out + out_strides[13], v_out_stride, v_out13);
        SCATTER4_256_S(curr_out + out_strides[14], v_out_stride, v_out14);

        in_r += NUM_SETS_256_S * v_in_stride;
        out_r += NUM_SETS_256_S * v_out_stride;
        remaining_sets = remaining_sets - NUM_SETS_256_S;
    }
#endif

#if defined(KERNEL_USE_AVX512) || defined(KERNEL_USE_AVX256)
    if (remaining_sets >= NUM_SETS_128_S)
    {
        __m128 v_in0, v_in1, v_in2, v_in3, v_in4, v_in5, v_in6, v_in7, v_in8;
        __m128 v_in9, v_in10, v_in11, v_in12, v_in13, v_in14;
        __m128 v_cv1, v_cv2, v_cv3, v_cv4, v_cv5, v_cv6, v_cv7, v_cv8, v_cv9,
            v_cv10;
        __m128 v_cv11, v_cv12;
        __m128 v_av1, v_av2, v_av3, v_av4, v_av5, v_av6, v_av7, v_av8, v_av9;
        __m128 v_av11, v_av12, v_av13, v_av14, v_av15, v_av16, v_av17;
        __m128 v_av18, v_av19, v_av20, v_av21, v_av22, v_av23, v_av24, v_av25;
        __m128 v_av26, v_av27, v_av28, v_av29, v_av30, v_av31, v_av32, v_av33;
        __m128 v_av34, v_av35, v_av36, v_av37, v_av38, v_av39, v_av40, v_av41;
        __m128 v_av42, v_av44, v_av45, v_av46, v_av47, v_av48, v_av49;
        __m128 v_tv1, v_tv2, v_tv3, v_tv4, v_tv5, v_tv6, v_tv7, v_tv8, v_tv9,
            v_tv10;
        __m128 v_tv11, v_tv16, v_tv17, v_tv18;
        __m128 v_tv19, v_tv20, v_tv21, v_tv22, v_tv23, v_tv24, v_tv25, v_tv27;
        __m128 v_out0, v_out1, v_out2, v_out3, v_out4, v_out5;
        __m128 v_out6, v_out7, v_out8, v_out9, v_out10, v_out11;
        __m128 v_out12, v_out13, v_out14;

#if defined(KERNEL_USE_AVX512)
        __m128 v_K1 = CAST_512_TO_128_S(v_C1);
        __m128 v_K2 = CAST_512_TO_128_S(v_C2);
        __m128 v_K3 = CAST_512_TO_128_S(v_C3);
        __m128 v_K4 = CAST_512_TO_128_S(v_C4);
        __m128 v_K5 = CAST_512_TO_128_S(v_C5);
        __m128 v_K6 = CAST_512_TO_128_S(v_C6);
#elif defined(KERNEL_USE_AVX256)
        __m128 v_K1 = CAST_256_TO_128_S(v_C1);
        __m128 v_K2 = CAST_256_TO_128_S(v_C2);
        __m128 v_K3 = CAST_256_TO_128_S(v_C3);
        __m128 v_K4 = CAST_256_TO_128_S(v_C4);
        __m128 v_K5 = CAST_256_TO_128_S(v_C5);
        __m128 v_K6 = CAST_256_TO_128_S(v_C6);
#endif

        curr_in = in_r;
        curr_out = out_r;

        if (flag)
        {
            ITW_GATHER_128_S(curr_in, in_strides, 1, v_in_stride, v_in1, tw,
                             cols, cnt_128);
            ITW_GATHER_128_S(curr_in, in_strides, 2, v_in_stride, v_in2, tw,
                             cols, cnt_128);
            ITW_GATHER_128_S(curr_in, in_strides, 3, v_in_stride, v_in3, tw,
                             cols, cnt_128);
            ITW_GATHER_128_S(curr_in, in_strides, 4, v_in_stride, v_in4, tw,
                             cols, cnt_128);
            ITW_GATHER_128_S(curr_in, in_strides, 5, v_in_stride, v_in5, tw,
                             cols, cnt_128);
            ITW_GATHER_128_S(curr_in, in_strides, 6, v_in_stride, v_in6, tw,
                             cols, cnt_128);
            ITW_GATHER_128_S(curr_in, in_strides, 7, v_in_stride, v_in7, tw,
                             cols, cnt_128);
            ITW_GATHER_128_S(curr_in, in_strides, 8, v_in_stride, v_in8, tw,
                             cols, cnt_128);
            ITW_GATHER_128_S(curr_in, in_strides, 9, v_in_stride, v_in9, tw,
                             cols, cnt_128);
            ITW_GATHER_128_S(curr_in, in_strides, 10, v_in_stride, v_in10, tw,
                             cols, cnt_128);
            ITW_GATHER_128_S(curr_in, in_strides, 11, v_in_stride, v_in11, tw,
                             cols, cnt_128);
            ITW_GATHER_128_S(curr_in, in_strides, 12, v_in_stride, v_in12, tw,
                             cols, cnt_128);
            ITW_GATHER_128_S(curr_in, in_strides, 13, v_in_stride, v_in13, tw,
                             cols, cnt_128);
            ITW_GATHER_128_S(curr_in, in_strides, 14, v_in_stride, v_in14, tw,
                             cols, cnt_128);
        }
        else
        {
            TW_GATHER_128_S(curr_in, in_strides, 1, v_in_stride, v_in1, tw,
                            cols, cnt_128);
            TW_GATHER_128_S(curr_in, in_strides, 2, v_in_stride, v_in2, tw,
                            cols, cnt_128);
            TW_GATHER_128_S(curr_in, in_strides, 3, v_in_stride, v_in3, tw,
                            cols, cnt_128);
            TW_GATHER_128_S(curr_in, in_strides, 4, v_in_stride, v_in4, tw,
                            cols, cnt_128);
            TW_GATHER_128_S(curr_in, in_strides, 5, v_in_stride, v_in5, tw,
                            cols, cnt_128);
            TW_GATHER_128_S(curr_in, in_strides, 6, v_in_stride, v_in6, tw,
                            cols, cnt_128);
            TW_GATHER_128_S(curr_in, in_strides, 7, v_in_stride, v_in7, tw,
                            cols, cnt_128);
            TW_GATHER_128_S(curr_in, in_strides, 8, v_in_stride, v_in8, tw,
                            cols, cnt_128);
            TW_GATHER_128_S(curr_in, in_strides, 9, v_in_stride, v_in9, tw,
                            cols, cnt_128);
            TW_GATHER_128_S(curr_in, in_strides, 10, v_in_stride, v_in10, tw,
                            cols, cnt_128);
            TW_GATHER_128_S(curr_in, in_strides, 11, v_in_stride, v_in11, tw,
                            cols, cnt_128);
            TW_GATHER_128_S(curr_in, in_strides, 12, v_in_stride, v_in12, tw,
                            cols, cnt_128);
            TW_GATHER_128_S(curr_in, in_strides, 13, v_in_stride, v_in13, tw,
                            cols, cnt_128);
            TW_GATHER_128_S(curr_in, in_strides, 14, v_in_stride, v_in14, tw,
                            cols, cnt_128);
        }

        GATHER2_128_S(curr_in, v_in_stride, v_in0);

        // common calculations
        v_cv1 = _mm_add_ps(v_in10, v_in5);
        v_cv2 = _mm_add_ps(v_in0, v_cv1);
        v_cv3 = _mm_add_ps(v_in14, v_in4);
        v_cv4 = _mm_add_ps(v_in9, v_cv3);
        v_cv5 = _mm_add_ps(v_in11, v_in1);
        v_cv6 = _mm_add_ps(v_in6, v_cv5);
        v_av1 = _mm_add_ps(v_cv4, v_cv6);
        v_cv7 = _mm_add_ps(v_in2, v_in7);
        v_cv8 = _mm_add_ps(v_in12, v_cv7);
        v_cv9 = _mm_add_ps(v_in13, v_in8);
        v_cv10 = _mm_add_ps(v_in3, v_cv9);
        v_av2 = _mm_add_ps(v_cv8, v_cv10);
        v_av3 = _mm_add_ps(v_av1, v_av2);

        // Output point 1 : X[0]
        v_out0 = _mm_add_ps(v_cv2, v_av3);

        v_tv1 = _mm_mul_ps(v_K2, v_av3);
        v_av4 = _mm_sub_ps(v_cv2, v_tv1);
        v_av5 = _mm_sub_ps(v_av1, v_av2);
        v_tv2 = _mm_mul_ps(v_K1, v_av5);
        v_av6 = _mm_add_ps(v_av4, v_tv2);
        v_av7 = _mm_sub_ps(v_cv6, v_cv4);
        v_av8 = _mm_sub_ps(v_cv8, v_cv10);
        v_tv3 = _mm_mul_ps(v_K3, v_av7);
        v_tv4 = _mm_mul_ps(v_K4, v_av8);
        v_av9 = _mm_add_ps(v_tv3, v_tv4);

        v_av9 = SWAP_RI_128_S(CONJ_128_S(v_av9));
        // Output point 13 : X[12]
        v_out12 = _mm_add_ps(v_av6, v_av9);
        // Output point 4 : X[3]
        v_out3 = _mm_sub_ps(v_av6, v_av9);

        v_av6 = _mm_sub_ps(v_av4, v_tv2);
        v_tv3 = _mm_mul_ps(v_K3, v_av8);
        v_tv4 = _mm_mul_ps(v_K4, v_av7);
        v_av9 = _mm_sub_ps(v_tv4, v_tv3);

        v_av9 = SWAP_RI_128_S(CONJ_128_S(v_av9));
        // Output point 10 : X[9]
        v_out9 = _mm_add_ps(v_av6, v_av9);
        // Output point 7 : X[6]
        v_out6 = _mm_sub_ps(v_av6, v_av9);

        v_tv5 = _mm_mul_ps(v_K5, v_cv1);
        v_av11 = _mm_sub_ps(v_in0, v_tv5);

        v_tv6 = _mm_mul_ps(v_K5, v_cv5);
        v_av12 = _mm_sub_ps(v_in6, v_tv6);
        v_tv7 = _mm_mul_ps(v_K5, v_cv3);
        v_av13 = _mm_sub_ps(v_in9, v_tv7);
        v_av14 = _mm_add_ps(v_av12, v_av13);
        v_tv8 = _mm_mul_ps(v_K5, v_cv7);
        v_av15 = _mm_sub_ps(v_in12, v_tv8);
        v_tv9 = _mm_mul_ps(v_K5, v_cv9);
        v_av16 = _mm_sub_ps(v_in3, v_tv9);
        v_av17 = _mm_add_ps(v_av15, v_av16);
        v_av18 = _mm_add_ps(v_av14, v_av17);
        v_tv10 = _mm_mul_ps(v_K2, v_av18);
        v_av19 = _mm_sub_ps(v_av11, v_tv10);
        v_av20 = _mm_sub_ps(v_av14, v_av17);
        v_tv11 = _mm_mul_ps(v_K1, v_av20);
        v_av21 = _mm_add_ps(v_av19, v_tv11);
        v_cv11 = _mm_sub_ps(v_in14, v_in4);
        v_av29 = _mm_mul_ps(v_K6, v_cv11);
        v_cv12 = _mm_sub_ps(v_in11, v_in1);
        v_av28 = _mm_mul_ps(v_K6, v_cv12);
        v_av22 = _mm_sub_ps(v_av29, v_av28);
        v_av23 = _mm_sub_ps(v_in8, v_in13);
        v_av32 = _mm_mul_ps(v_K6, v_av23);
        v_av24 = _mm_sub_ps(v_in2, v_in7);
        v_av31 = _mm_mul_ps(v_K6, v_av24);
        v_av25 = _mm_sub_ps(v_av32, v_av31);
        v_tv16 = _mm_mul_ps(v_K3, v_av22);
        v_tv17 = _mm_mul_ps(v_K4, v_av25);
        v_av26 = _mm_add_ps(v_tv16, v_tv17);
        v_av27 = _mm_sub_ps(v_in10, v_in5);
        v_tv18 = _mm_mul_ps(v_K6, v_av27);
        v_av30 = _mm_add_ps(v_av28, v_av29);
        v_av33 = _mm_add_ps(v_av31, v_av32);
        v_av34 = _mm_add_ps(v_av30, v_av33);
        v_tv19 = _mm_mul_ps(v_K2, v_av34);
        v_av35 = _mm_add_ps(v_tv18, v_tv19);
        v_av36 = _mm_sub_ps(v_av33, v_av30);
        v_tv20 = _mm_mul_ps(v_K1, v_av36);
        v_av37 = _mm_add_ps(v_av35, v_tv20);
        v_av38 = _mm_sub_ps(v_av13, v_av12);
        v_av39 = _mm_sub_ps(v_av16, v_av15);
        v_tv21 = _mm_mul_ps(v_K3, v_av38);
        v_tv22 = _mm_mul_ps(v_K4, v_av39);
        v_av40 = _mm_add_ps(v_tv21, v_tv22);

        // real part
        v_av41 = _mm_add_ps(v_av11, v_av18);

        // imag part
        v_av42 = _mm_sub_ps(v_tv18, v_av34);

        v_av42 = SWAP_RI_128_S(CONJ_128_S(v_av42));
        // Output point 11 : X[10]
        v_out10 = _mm_add_ps(v_av41, v_av42);
        // Output point 6 : X[5]
        v_out5 = _mm_sub_ps(v_av41, v_av42);

        // real part
        v_av41 = _mm_add_ps(v_av21, v_av26);

        // imag part
        v_av42 = _mm_add_ps(v_av37, v_av40);

        v_av42 = SWAP_RI_128_S(CONJ_128_S(v_av42));
        // Output point 14 : X[13]
        v_out13 = _mm_add_ps(v_av41, v_av42);
        // Output point 3 : X[2]
        v_out2 = _mm_sub_ps(v_av41, v_av42);

        v_av44 = _mm_sub_ps(v_av19, v_tv11);
        v_tv23 = _mm_mul_ps(v_K3, v_av25);
        v_tv24 = _mm_mul_ps(v_K4, v_av22);
        v_av45 = _mm_sub_ps(v_tv23, v_tv24);
        v_av46 = _mm_sub_ps(v_av35, v_tv20);
        v_tv25 = _mm_mul_ps(v_K3, v_av39);
        v_tv27 = _mm_mul_ps(v_K4, v_av38);
        v_av47 = _mm_sub_ps(v_tv25, v_tv27);

        // real part
        v_av41 = _mm_add_ps(v_av44, v_av45);

        // imag part
        v_av42 = _mm_add_ps(v_av46, v_av47);

        v_av42 = SWAP_RI_128_S(CONJ_128_S(v_av42));
        // Output point 5 : X[4]
        v_out4 = _mm_add_ps(v_av41, v_av42);
        // Output point 12 : X[11]
        v_out11 = _mm_sub_ps(v_av41, v_av42);

        // real part
        v_av48 = _mm_sub_ps(v_av44, v_av45);

        // imag part
        v_av49 = _mm_sub_ps(v_av47, v_av46);

        v_av49 = SWAP_RI_128_S(CONJ_128_S(v_av49));
        // Output point 15 : X[14]
        v_out14 = _mm_add_ps(v_av48, v_av49);
        // Output point 2 : X[1]
        v_out1 = _mm_sub_ps(v_av48, v_av49);

        // real part
        v_av48 = _mm_sub_ps(v_av21, v_av26);

        // imag part
        v_av49 = _mm_sub_ps(v_av40, v_av37);

        v_av49 = SWAP_RI_128_S(CONJ_128_S(v_av49));
        // Output point 15 : X[14]
        v_out8 = _mm_add_ps(v_av48, v_av49);
        // Output point 2 : X[1]
        v_out7 = _mm_sub_ps(v_av48, v_av49);

        SCATTER2_128_S(curr_out, v_out_stride, v_out0);
        SCATTER2_128_S(curr_out + out_strides[1], v_out_stride, v_out1);
        SCATTER2_128_S(curr_out + out_strides[2], v_out_stride, v_out2);
        SCATTER2_128_S(curr_out + out_strides[3], v_out_stride, v_out3);
        SCATTER2_128_S(curr_out + out_strides[4], v_out_stride, v_out4);
        SCATTER2_128_S(curr_out + out_strides[5], v_out_stride, v_out5);
        SCATTER2_128_S(curr_out + out_strides[6], v_out_stride, v_out6);
        SCATTER2_128_S(curr_out + out_strides[7], v_out_stride, v_out7);
        SCATTER2_128_S(curr_out + out_strides[8], v_out_stride, v_out8);
        SCATTER2_128_S(curr_out + out_strides[9], v_out_stride, v_out9);
        SCATTER2_128_S(curr_out + out_strides[10], v_out_stride, v_out10);
        SCATTER2_128_S(curr_out + out_strides[11], v_out_stride, v_out11);
        SCATTER2_128_S(curr_out + out_strides[12], v_out_stride, v_out12);
        SCATTER2_128_S(curr_out + out_strides[13], v_out_stride, v_out13);
        SCATTER2_128_S(curr_out + out_strides[14], v_out_stride, v_out14);

        in_r = in_r + (v_in_stride << 1);
        out_r = out_r + (v_out_stride << 1);
        remaining_sets = remaining_sets - NUM_SETS_128_S;
    }
#endif

#if defined(KERNEL_USE_AVX512) || defined(KERNEL_USE_AVX256) ||                      \
    defined(KERNEL_USE_AVX128)
    if (remaining_sets & 1)
    {
        __m128 v_in0, v_in1, v_in2, v_in3, v_in4, v_in5, v_in6, v_in7, v_in8;
        __m128 v_in9, v_in10, v_in11, v_in12, v_in13, v_in14;
        __m128 v_cv1, v_cv2, v_cv3, v_cv4, v_cv5, v_cv6, v_cv7, v_cv8, v_cv9,
            v_cv10;
        __m128 v_cv11, v_cv12;
        __m128 v_av1, v_av2, v_av3, v_av4, v_av5, v_av6, v_av7, v_av8, v_av9;
        __m128 v_av11, v_av12, v_av13, v_av14, v_av15, v_av16, v_av17;
        __m128 v_av18, v_av19, v_av20, v_av21, v_av22, v_av23, v_av24, v_av25;
        __m128 v_av26, v_av27, v_av28, v_av29, v_av30, v_av31, v_av32, v_av33;
        __m128 v_av34, v_av35, v_av36, v_av37, v_av38, v_av39, v_av40, v_av41;
        __m128 v_av42, v_av44, v_av45, v_av46, v_av47, v_av48, v_av49;
        __m128 v_tv1, v_tv2, v_tv3, v_tv4, v_tv5, v_tv6, v_tv7, v_tv8, v_tv9,
            v_tv10;
        __m128 v_tv11, v_tv16, v_tv17, v_tv18;
        __m128 v_tv19, v_tv20, v_tv21, v_tv22, v_tv23, v_tv24, v_tv25, v_tv27;
        __m128 v_out0, v_out1, v_out2, v_out3, v_out4, v_out5;
        __m128 v_out6, v_out7, v_out8, v_out9, v_out10, v_out11;
        __m128 v_out12, v_out13, v_out14;

#if defined(KERNEL_USE_AVX512)
        __m128 v_K1 = CAST_512_TO_128_S(v_C1);
        __m128 v_K2 = CAST_512_TO_128_S(v_C2);
        __m128 v_K3 = CAST_512_TO_128_S(v_C3);
        __m128 v_K4 = CAST_512_TO_128_S(v_C4);
        __m128 v_K5 = CAST_512_TO_128_S(v_C5);
        __m128 v_K6 = CAST_512_TO_128_S(v_C6);
#elif defined(KERNEL_USE_AVX256)
        __m128 v_K1 = CAST_256_TO_128_S(v_C1);
        __m128 v_K2 = CAST_256_TO_128_S(v_C2);
        __m128 v_K3 = CAST_256_TO_128_S(v_C3);
        __m128 v_K4 = CAST_256_TO_128_S(v_C4);
        __m128 v_K5 = CAST_256_TO_128_S(v_C5);
        __m128 v_K6 = CAST_256_TO_128_S(v_C6);
#elif defined(KERNEL_USE_AVX128)
        __m128 v_K1 = v_C1;
        __m128 v_K2 = v_C2;
        __m128 v_K3 = v_C3;
        __m128 v_K4 = v_C4;
        __m128 v_K5 = v_C5;
        __m128 v_K6 = v_C6;
#endif

        curr_in = in_r;
        curr_out = out_r;

        if (flag)
        {
            ITW_GATHER_LOW_128_S(curr_in, in_strides, 1, v_in1, tw, cols,
                                 cnt_128_low);
            ITW_GATHER_LOW_128_S(curr_in, in_strides, 2, v_in2, tw, cols,
                                 cnt_128_low);
            ITW_GATHER_LOW_128_S(curr_in, in_strides, 3, v_in3, tw, cols,
                                 cnt_128_low);
            ITW_GATHER_LOW_128_S(curr_in, in_strides, 4, v_in4, tw, cols,
                                 cnt_128_low);
            ITW_GATHER_LOW_128_S(curr_in, in_strides, 5, v_in5, tw, cols,
                                 cnt_128_low);
            ITW_GATHER_LOW_128_S(curr_in, in_strides, 6, v_in6, tw, cols,
                                 cnt_128_low);
            ITW_GATHER_LOW_128_S(curr_in, in_strides, 7, v_in7, tw, cols,
                                 cnt_128_low);
            ITW_GATHER_LOW_128_S(curr_in, in_strides, 8, v_in8, tw, cols,
                                 cnt_128_low);
            ITW_GATHER_LOW_128_S(curr_in, in_strides, 9, v_in9, tw, cols,
                                 cnt_128_low);
            ITW_GATHER_LOW_128_S(curr_in, in_strides, 10, v_in10, tw, cols,
                                 cnt_128_low);
            ITW_GATHER_LOW_128_S(curr_in, in_strides, 11, v_in11, tw, cols,
                                 cnt_128_low);
            ITW_GATHER_LOW_128_S(curr_in, in_strides, 12, v_in12, tw, cols,
                                 cnt_128_low);
            ITW_GATHER_LOW_128_S(curr_in, in_strides, 13, v_in13, tw, cols,
                                 cnt_128_low);
            ITW_GATHER_LOW_128_S(curr_in, in_strides, 14, v_in14, tw, cols,
                                 cnt_128_low);
        }
        else
        {
            TW_GATHER_LOW_128_S(curr_in, in_strides, 1, v_in1, tw, cols,
                                cnt_128_low);
            TW_GATHER_LOW_128_S(curr_in, in_strides, 2, v_in2, tw, cols,
                                cnt_128_low);
            TW_GATHER_LOW_128_S(curr_in, in_strides, 3, v_in3, tw, cols,
                                cnt_128_low);
            TW_GATHER_LOW_128_S(curr_in, in_strides, 4, v_in4, tw, cols,
                                cnt_128_low);
            TW_GATHER_LOW_128_S(curr_in, in_strides, 5, v_in5, tw, cols,
                                cnt_128_low);
            TW_GATHER_LOW_128_S(curr_in, in_strides, 6, v_in6, tw, cols,
                                cnt_128_low);
            TW_GATHER_LOW_128_S(curr_in, in_strides, 7, v_in7, tw, cols,
                                cnt_128_low);
            TW_GATHER_LOW_128_S(curr_in, in_strides, 8, v_in8, tw, cols,
                                cnt_128_low);
            TW_GATHER_LOW_128_S(curr_in, in_strides, 9, v_in9, tw, cols,
                                cnt_128_low);
            TW_GATHER_LOW_128_S(curr_in, in_strides, 10, v_in10, tw, cols,
                                cnt_128_low);
            TW_GATHER_LOW_128_S(curr_in, in_strides, 11, v_in11, tw, cols,
                                cnt_128_low);
            TW_GATHER_LOW_128_S(curr_in, in_strides, 12, v_in12, tw, cols,
                                cnt_128_low);
            TW_GATHER_LOW_128_S(curr_in, in_strides, 13, v_in13, tw, cols,
                                cnt_128_low);
            TW_GATHER_LOW_128_S(curr_in, in_strides, 14, v_in14, tw, cols,
                                cnt_128_low);
        }

        LD_LOW_128_S(curr_in, v_in0);

        // common calculations
        v_cv1 = _mm_add_ps(v_in10, v_in5);
        v_cv2 = _mm_add_ps(v_in0, v_cv1);
        v_cv3 = _mm_add_ps(v_in14, v_in4);
        v_cv4 = _mm_add_ps(v_in9, v_cv3);
        v_cv5 = _mm_add_ps(v_in11, v_in1);
        v_cv6 = _mm_add_ps(v_in6, v_cv5);
        v_av1 = _mm_add_ps(v_cv4, v_cv6);
        v_cv7 = _mm_add_ps(v_in2, v_in7);
        v_cv8 = _mm_add_ps(v_in12, v_cv7);
        v_cv9 = _mm_add_ps(v_in13, v_in8);
        v_cv10 = _mm_add_ps(v_in3, v_cv9);
        v_av2 = _mm_add_ps(v_cv8, v_cv10);
        v_av3 = _mm_add_ps(v_av1, v_av2);

        // Output point 1 : X[0]
        v_out0 = _mm_add_ps(v_cv2, v_av3);

        v_tv1 = _mm_mul_ps(v_K2, v_av3);
        v_av4 = _mm_sub_ps(v_cv2, v_tv1);
        v_av5 = _mm_sub_ps(v_av1, v_av2);
        v_tv2 = _mm_mul_ps(v_K1, v_av5);
        v_av6 = _mm_add_ps(v_av4, v_tv2);
        v_av7 = _mm_sub_ps(v_cv6, v_cv4);
        v_av8 = _mm_sub_ps(v_cv8, v_cv10);
        v_tv3 = _mm_mul_ps(v_K3, v_av7);
        v_tv4 = _mm_mul_ps(v_K4, v_av8);
        v_av9 = _mm_add_ps(v_tv3, v_tv4);

        v_av9 = SWAP_RI_128_S(CONJ_128_S(v_av9));
        // Output point 13 : X[12]
        v_out12 = _mm_add_ps(v_av6, v_av9);
        // Output point 4 : X[3]
        v_out3 = _mm_sub_ps(v_av6, v_av9);

        v_av6 = _mm_sub_ps(v_av4, v_tv2);
        v_tv3 = _mm_mul_ps(v_K3, v_av8);
        v_tv4 = _mm_mul_ps(v_K4, v_av7);
        v_av9 = _mm_sub_ps(v_tv4, v_tv3);

        v_av9 = SWAP_RI_128_S(CONJ_128_S(v_av9));
        // Output point 10 : X[9]
        v_out9 = _mm_add_ps(v_av6, v_av9);
        // Output point 7 : X[6]
        v_out6 = _mm_sub_ps(v_av6, v_av9);

        v_tv5 = _mm_mul_ps(v_K5, v_cv1);
        v_av11 = _mm_sub_ps(v_in0, v_tv5);

        v_tv6 = _mm_mul_ps(v_K5, v_cv5);
        v_av12 = _mm_sub_ps(v_in6, v_tv6);
        v_tv7 = _mm_mul_ps(v_K5, v_cv3);
        v_av13 = _mm_sub_ps(v_in9, v_tv7);
        v_av14 = _mm_add_ps(v_av12, v_av13);
        v_tv8 = _mm_mul_ps(v_K5, v_cv7);
        v_av15 = _mm_sub_ps(v_in12, v_tv8);
        v_tv9 = _mm_mul_ps(v_K5, v_cv9);
        v_av16 = _mm_sub_ps(v_in3, v_tv9);
        v_av17 = _mm_add_ps(v_av15, v_av16);
        v_av18 = _mm_add_ps(v_av14, v_av17);
        v_tv10 = _mm_mul_ps(v_K2, v_av18);
        v_av19 = _mm_sub_ps(v_av11, v_tv10);
        v_av20 = _mm_sub_ps(v_av14, v_av17);
        v_tv11 = _mm_mul_ps(v_K1, v_av20);
        v_av21 = _mm_add_ps(v_av19, v_tv11);
        v_cv11 = _mm_sub_ps(v_in14, v_in4);
        v_av29 = _mm_mul_ps(v_K6, v_cv11);
        v_cv12 = _mm_sub_ps(v_in11, v_in1);
        v_av28 = _mm_mul_ps(v_K6, v_cv12);
        v_av22 = _mm_sub_ps(v_av29, v_av28);
        v_av23 = _mm_sub_ps(v_in8, v_in13);
        v_av32 = _mm_mul_ps(v_K6, v_av23);
        v_av24 = _mm_sub_ps(v_in2, v_in7);
        v_av31 = _mm_mul_ps(v_K6, v_av24);
        v_av25 = _mm_sub_ps(v_av32, v_av31);
        v_tv16 = _mm_mul_ps(v_K3, v_av22);
        v_tv17 = _mm_mul_ps(v_K4, v_av25);
        v_av26 = _mm_add_ps(v_tv16, v_tv17);
        v_av27 = _mm_sub_ps(v_in10, v_in5);
        v_tv18 = _mm_mul_ps(v_K6, v_av27);
        v_av30 = _mm_add_ps(v_av28, v_av29);
        v_av33 = _mm_add_ps(v_av31, v_av32);
        v_av34 = _mm_add_ps(v_av30, v_av33);
        v_tv19 = _mm_mul_ps(v_K2, v_av34);
        v_av35 = _mm_add_ps(v_tv18, v_tv19);
        v_av36 = _mm_sub_ps(v_av33, v_av30);
        v_tv20 = _mm_mul_ps(v_K1, v_av36);
        v_av37 = _mm_add_ps(v_av35, v_tv20);
        v_av38 = _mm_sub_ps(v_av13, v_av12);
        v_av39 = _mm_sub_ps(v_av16, v_av15);
        v_tv21 = _mm_mul_ps(v_K3, v_av38);
        v_tv22 = _mm_mul_ps(v_K4, v_av39);
        v_av40 = _mm_add_ps(v_tv21, v_tv22);

        // real part
        v_av41 = _mm_add_ps(v_av11, v_av18);

        // imag part
        v_av42 = _mm_sub_ps(v_tv18, v_av34);

        v_av42 = SWAP_RI_128_S(CONJ_128_S(v_av42));
        // Output point 11 : X[10]
        v_out10 = _mm_add_ps(v_av41, v_av42);
        // Output point 6 : X[5]
        v_out5 = _mm_sub_ps(v_av41, v_av42);

        // real part
        v_av41 = _mm_add_ps(v_av21, v_av26);

        // imag part
        v_av42 = _mm_add_ps(v_av37, v_av40);

        v_av42 = SWAP_RI_128_S(CONJ_128_S(v_av42));
        // Output point 14 : X[13]
        v_out13 = _mm_add_ps(v_av41, v_av42);
        // Output point 3 : X[2]
        v_out2 = _mm_sub_ps(v_av41, v_av42);

        v_av44 = _mm_sub_ps(v_av19, v_tv11);
        v_tv23 = _mm_mul_ps(v_K3, v_av25);
        v_tv24 = _mm_mul_ps(v_K4, v_av22);
        v_av45 = _mm_sub_ps(v_tv23, v_tv24);
        v_av46 = _mm_sub_ps(v_av35, v_tv20);
        v_tv25 = _mm_mul_ps(v_K3, v_av39);
        v_tv27 = _mm_mul_ps(v_K4, v_av38);
        v_av47 = _mm_sub_ps(v_tv25, v_tv27);

        // real part
        v_av41 = _mm_add_ps(v_av44, v_av45);

        // imag part
        v_av42 = _mm_add_ps(v_av46, v_av47);

        v_av42 = SWAP_RI_128_S(CONJ_128_S(v_av42));
        // Output point 5 : X[4]
        v_out4 = _mm_add_ps(v_av41, v_av42);
        // Output point 12 : X[11]
        v_out11 = _mm_sub_ps(v_av41, v_av42);

        // real part
        v_av48 = _mm_sub_ps(v_av44, v_av45);

        // imag part
        v_av49 = _mm_sub_ps(v_av47, v_av46);

        v_av49 = SWAP_RI_128_S(CONJ_128_S(v_av49));
        // Output point 15 : X[14]
        v_out14 = _mm_add_ps(v_av48, v_av49);
        // Output point 2 : X[1]
        v_out1 = _mm_sub_ps(v_av48, v_av49);

        // real part
        v_av48 = _mm_sub_ps(v_av21, v_av26);

        // imag part
        v_av49 = _mm_sub_ps(v_av40, v_av37);

        v_av49 = SWAP_RI_128_S(CONJ_128_S(v_av49));
        // Output point 15 : X[14]
        v_out8 = _mm_add_ps(v_av48, v_av49);
        // Output point 2 : X[1]
        v_out7 = _mm_sub_ps(v_av48, v_av49);

        ST_LOW_128_S(curr_out, v_out0);
        ST_LOW_128_S(curr_out + out_strides[1], v_out1);
        ST_LOW_128_S(curr_out + out_strides[2], v_out2);
        ST_LOW_128_S(curr_out + out_strides[3], v_out3);
        ST_LOW_128_S(curr_out + out_strides[4], v_out4);
        ST_LOW_128_S(curr_out + out_strides[5], v_out5);
        ST_LOW_128_S(curr_out + out_strides[6], v_out6);
        ST_LOW_128_S(curr_out + out_strides[7], v_out7);
        ST_LOW_128_S(curr_out + out_strides[8], v_out8);
        ST_LOW_128_S(curr_out + out_strides[9], v_out9);
        ST_LOW_128_S(curr_out + out_strides[10], v_out10);
        ST_LOW_128_S(curr_out + out_strides[11], v_out11);
        ST_LOW_128_S(curr_out + out_strides[12], v_out12);
        ST_LOW_128_S(curr_out + out_strides[13], v_out13);
        ST_LOW_128_S(curr_out + out_strides[14], v_out14);
    }
#endif

#ifdef AOCL_ENABLE_LOG
    AOCLFFTZ_LOG_UNFORMATTED(TRACE, TRACE, "Exit");
#endif
}

static VOID TWID_KNAME_FP64(VOID *in_real, VOID *in_imag, VOID *out_real,
                            VOID *out_imag, INTP n, aoclfftz_strides_t *strides,
                            VOID *twd, UINT8 flag)
{
#ifdef AOCL_ENABLE_LOG
    AOCLFFTZ_LOG_UNFORMATTED(TRACE, TRACE, "Enter");
#endif

    const DOUBLE CRTM_15[6] = {
        0.55901699437494742410229341718281905886015458990288,
        0.25000000000000000000000000000000000000000000000000,
        0.95105651629515357211643933337938214340569863400000,
        0.58778525229247301629891039327884007596190389052978,
        0.50000000000000000000000000000000000000000000000000,
        0.86602540378443864676372317075293618347140262690519};

    DOUBLE *in_r = in_real;
    DOUBLE *out_r = out_real;
    DOUBLE *curr_in, *curr_out;
#ifdef VOLATILE_STRIDE_ARRAY
    volatile INTP *in_strides = strides->in_strides;
    volatile INTP *out_strides = strides->out_strides;
#else
    INTP *in_strides = strides->in_strides;
    INTP *out_strides = strides->out_strides;
#endif
    INTP v_in_stride = strides->v_in_stride;
    INTP v_out_stride = strides->v_out_stride;

    aoclfftz_twiddle_t *tws = (aoclfftz_twiddle_t *)twd;
    DOUBLE *tw = (DOUBLE *)(tws->TW);
    UINTP cols = tws->cols;

    INTP N = n / NUM_SETS_D;
    INTP count;

#if defined(KERNEL_USE_AVX512)
    INTP remaining_sets = n % NUM_SETS_D;
    INTP do_256_whole = (INTP)(remaining_sets >= NUM_SETS_256_D);
    INTP cnt_256 = N * NUM_SETS_512_D;
    INTP cnt_128 = N * NUM_SETS_512_D + do_256_whole * NUM_SETS_256_D;
#elif defined(KERNEL_USE_AVX256)
    INTP remaining_sets = n % NUM_SETS_D;
    INTP cnt_128 = N * NUM_SETS_256_D;
#elif defined(KERNEL_USE_AVX128)
    // nothing, since double doesn't have any tail cases to process for AVX128
#endif

    VREGTYPE_D v_C1 = BCAST_D(CRTM_15[0]);
    VREGTYPE_D v_C2 = BCAST_D(CRTM_15[1]);
    VREGTYPE_D v_C3 = BCAST_D(CRTM_15[2]);
    VREGTYPE_D v_C4 = BCAST_D(CRTM_15[3]);
    VREGTYPE_D v_C5 = BCAST_D(CRTM_15[4]);
    VREGTYPE_D v_C6 = BCAST_D(CRTM_15[5]);

    if (flag)
    {
        in_r = in_imag;
        out_r = out_imag;

        v_C3 = NEG_D(v_C3);
        v_C4 = NEG_D(v_C4);
        v_C6 = NEG_D(v_C6);
    }

    for (count = 0; count < N; count++)
    {
        VREGTYPE_D v_in0, v_in1, v_in2, v_in3, v_in4, v_in5, v_in6, v_in7,
            v_in8;
        VREGTYPE_D v_in9, v_in10, v_in11, v_in12, v_in13, v_in14;
        VREGTYPE_D v_cv1, v_cv2, v_cv3, v_cv4, v_cv5, v_cv6, v_cv7, v_cv8,
            v_cv9, v_cv10;
        VREGTYPE_D v_cv11, v_cv12;
        VREGTYPE_D v_av1, v_av2, v_av3, v_av4, v_av5, v_av6, v_av7, v_av8,
            v_av9;
        VREGTYPE_D v_av11, v_av12, v_av13, v_av14, v_av15, v_av16, v_av17;
        VREGTYPE_D v_av18, v_av19, v_av20, v_av21, v_av22, v_av23, v_av24,
            v_av25;
        VREGTYPE_D v_av26, v_av27, v_av28, v_av29, v_av30, v_av31, v_av32,
            v_av33;
        VREGTYPE_D v_av34, v_av35, v_av36, v_av37, v_av38, v_av39, v_av40,
            v_av41;
        VREGTYPE_D v_av42, v_av44, v_av45, v_av46, v_av47, v_av48, v_av49;
        VREGTYPE_D v_tv1, v_tv2, v_tv3, v_tv4, v_tv5, v_tv6, v_tv7, v_tv8,
            v_tv9, v_tv10;
        VREGTYPE_D v_tv11, v_tv16, v_tv17, v_tv18;
        VREGTYPE_D v_tv19, v_tv20, v_tv21, v_tv22, v_tv23, v_tv24, v_tv25,
            v_tv27;
        VREGTYPE_D v_out0, v_out1, v_out2, v_out3, v_out4, v_out5;
        VREGTYPE_D v_out6, v_out7, v_out8, v_out9, v_out10, v_out11;
        VREGTYPE_D v_out12, v_out13, v_out14;

        curr_in = in_r;
        curr_out = out_r;

        if (flag)
        {
            ITW_GATHER_D(curr_in, in_strides, 1, v_in_stride, v_in1, tw, cols,
                         (count * NUM_SETS_D));
            ITW_GATHER_D(curr_in, in_strides, 2, v_in_stride, v_in2, tw, cols,
                         (count * NUM_SETS_D));
            ITW_GATHER_D(curr_in, in_strides, 3, v_in_stride, v_in3, tw, cols,
                         (count * NUM_SETS_D));
            ITW_GATHER_D(curr_in, in_strides, 4, v_in_stride, v_in4, tw, cols,
                         (count * NUM_SETS_D));
            ITW_GATHER_D(curr_in, in_strides, 5, v_in_stride, v_in5, tw, cols,
                         (count * NUM_SETS_D));
            ITW_GATHER_D(curr_in, in_strides, 6, v_in_stride, v_in6, tw, cols,
                         (count * NUM_SETS_D));
            ITW_GATHER_D(curr_in, in_strides, 7, v_in_stride, v_in7, tw, cols,
                         (count * NUM_SETS_D));
            ITW_GATHER_D(curr_in, in_strides, 8, v_in_stride, v_in8, tw, cols,
                         (count * NUM_SETS_D));
            ITW_GATHER_D(curr_in, in_strides, 9, v_in_stride, v_in9, tw, cols,
                         (count * NUM_SETS_D));
            ITW_GATHER_D(curr_in, in_strides, 10, v_in_stride, v_in10, tw, cols,
                         (count * NUM_SETS_D));
            ITW_GATHER_D(curr_in, in_strides, 11, v_in_stride, v_in11, tw, cols,
                         (count * NUM_SETS_D));
            ITW_GATHER_D(curr_in, in_strides, 12, v_in_stride, v_in12, tw, cols,
                         (count * NUM_SETS_D));
            ITW_GATHER_D(curr_in, in_strides, 13, v_in_stride, v_in13, tw, cols,
                         (count * NUM_SETS_D));
            ITW_GATHER_D(curr_in, in_strides, 14, v_in_stride, v_in14, tw, cols,
                         (count * NUM_SETS_D));
        }
        else
        {
            TW_GATHER_D(curr_in, in_strides, 1, v_in_stride, v_in1, tw, cols,
                        (count * NUM_SETS_D));
            TW_GATHER_D(curr_in, in_strides, 2, v_in_stride, v_in2, tw, cols,
                        (count * NUM_SETS_D));
            TW_GATHER_D(curr_in, in_strides, 3, v_in_stride, v_in3, tw, cols,
                        (count * NUM_SETS_D));
            TW_GATHER_D(curr_in, in_strides, 4, v_in_stride, v_in4, tw, cols,
                        (count * NUM_SETS_D));
            TW_GATHER_D(curr_in, in_strides, 5, v_in_stride, v_in5, tw, cols,
                        (count * NUM_SETS_D));
            TW_GATHER_D(curr_in, in_strides, 6, v_in_stride, v_in6, tw, cols,
                        (count * NUM_SETS_D));
            TW_GATHER_D(curr_in, in_strides, 7, v_in_stride, v_in7, tw, cols,
                        (count * NUM_SETS_D));
            TW_GATHER_D(curr_in, in_strides, 8, v_in_stride, v_in8, tw, cols,
                        (count * NUM_SETS_D));
            TW_GATHER_D(curr_in, in_strides, 9, v_in_stride, v_in9, tw, cols,
                        (count * NUM_SETS_D));
            TW_GATHER_D(curr_in, in_strides, 10, v_in_stride, v_in10, tw, cols,
                        (count * NUM_SETS_D));
            TW_GATHER_D(curr_in, in_strides, 11, v_in_stride, v_in11, tw, cols,
                        (count * NUM_SETS_D));
            TW_GATHER_D(curr_in, in_strides, 12, v_in_stride, v_in12, tw, cols,
                        (count * NUM_SETS_D));
            TW_GATHER_D(curr_in, in_strides, 13, v_in_stride, v_in13, tw, cols,
                        (count * NUM_SETS_D));
            TW_GATHER_D(curr_in, in_strides, 14, v_in_stride, v_in14, tw, cols,
                        (count * NUM_SETS_D));
        }

        GATHER_D(curr_in, v_in_stride, v_in0);

        // common calculations
        v_cv1 = ADD_D(v_in10, v_in5);
        v_cv2 = ADD_D(v_in0, v_cv1);
        v_cv3 = ADD_D(v_in14, v_in4);
        v_cv4 = ADD_D(v_in9, v_cv3);
        v_cv5 = ADD_D(v_in11, v_in1);
        v_cv6 = ADD_D(v_in6, v_cv5);
        v_av1 = ADD_D(v_cv4, v_cv6);
        v_cv7 = ADD_D(v_in2, v_in7);
        v_cv8 = ADD_D(v_in12, v_cv7);
        v_cv9 = ADD_D(v_in13, v_in8);
        v_cv10 = ADD_D(v_in3, v_cv9);
        v_av2 = ADD_D(v_cv8, v_cv10);
        v_av3 = ADD_D(v_av1, v_av2);

        // Output point 1 : X[0]
        v_out0 = ADD_D(v_cv2, v_av3);

        v_tv1 = MUL_D(v_C2, v_av3);
        v_av4 = SUB_D(v_cv2, v_tv1);
        v_av5 = SUB_D(v_av1, v_av2);
        v_tv2 = MUL_D(v_C1, v_av5);
        v_av6 = ADD_D(v_av4, v_tv2);
        v_av7 = SUB_D(v_cv6, v_cv4);
        v_av8 = SUB_D(v_cv8, v_cv10);
        v_tv3 = MUL_D(v_C3, v_av7);
        v_tv4 = MUL_D(v_C4, v_av8);
        v_av9 = ADD_D(v_tv3, v_tv4);

        v_av9 = SWAP_RI_D(CONJ_D(v_av9));
        // Output point 13 : X[12]
        v_out12 = ADD_D(v_av6, v_av9);
        // Output point 4 : X[3]
        v_out3 = SUB_D(v_av6, v_av9);

        v_av6 = SUB_D(v_av4, v_tv2);
        v_tv3 = MUL_D(v_C3, v_av8);
        v_tv4 = MUL_D(v_C4, v_av7);
        v_av9 = SUB_D(v_tv4, v_tv3);

        v_av9 = SWAP_RI_D(CONJ_D(v_av9));
        // Output point 10 : X[9]
        v_out9 = ADD_D(v_av6, v_av9);
        // Output point 7 : X[6]
        v_out6 = SUB_D(v_av6, v_av9);

        v_tv5 = MUL_D(v_C5, v_cv1);
        v_av11 = SUB_D(v_in0, v_tv5);

        v_tv6 = MUL_D(v_C5, v_cv5);
        v_av12 = SUB_D(v_in6, v_tv6);
        v_tv7 = MUL_D(v_C5, v_cv3);
        v_av13 = SUB_D(v_in9, v_tv7);
        v_av14 = ADD_D(v_av12, v_av13);
        v_tv8 = MUL_D(v_C5, v_cv7);
        v_av15 = SUB_D(v_in12, v_tv8);
        v_tv9 = MUL_D(v_C5, v_cv9);
        v_av16 = SUB_D(v_in3, v_tv9);
        v_av17 = ADD_D(v_av15, v_av16);
        v_av18 = ADD_D(v_av14, v_av17);
        v_tv10 = MUL_D(v_C2, v_av18);
        v_av19 = SUB_D(v_av11, v_tv10);
        v_av20 = SUB_D(v_av14, v_av17);
        v_tv11 = MUL_D(v_C1, v_av20);
        v_av21 = ADD_D(v_av19, v_tv11);
        v_cv11 = SUB_D(v_in14, v_in4);
        v_av29 = MUL_D(v_C6, v_cv11);
        v_cv12 = SUB_D(v_in11, v_in1);
        v_av28 = MUL_D(v_C6, v_cv12);
        v_av22 = SUB_D(v_av29, v_av28);
        v_av23 = SUB_D(v_in8, v_in13);
        v_av32 = MUL_D(v_C6, v_av23);
        v_av24 = SUB_D(v_in2, v_in7);
        v_av31 = MUL_D(v_C6, v_av24);
        v_av25 = SUB_D(v_av32, v_av31);
        v_tv16 = MUL_D(v_C3, v_av22);
        v_tv17 = MUL_D(v_C4, v_av25);
        v_av26 = ADD_D(v_tv16, v_tv17);
        v_av27 = SUB_D(v_in10, v_in5);
        v_tv18 = MUL_D(v_C6, v_av27);
        v_av30 = ADD_D(v_av28, v_av29);
        v_av33 = ADD_D(v_av31, v_av32);
        v_av34 = ADD_D(v_av30, v_av33);
        v_tv19 = MUL_D(v_C2, v_av34);
        v_av35 = ADD_D(v_tv18, v_tv19);
        v_av36 = SUB_D(v_av33, v_av30);
        v_tv20 = MUL_D(v_C1, v_av36);
        v_av37 = ADD_D(v_av35, v_tv20);
        v_av38 = SUB_D(v_av13, v_av12);
        v_av39 = SUB_D(v_av16, v_av15);
        v_tv21 = MUL_D(v_C3, v_av38);
        v_tv22 = MUL_D(v_C4, v_av39);
        v_av40 = ADD_D(v_tv21, v_tv22);

        v_av41 = ADD_D(v_av11, v_av18);
        v_av42 = SUB_D(v_tv18, v_av34);
        v_av42 = SWAP_RI_D(CONJ_D(v_av42));
        // Output point 11 : X[10]
        v_out10 = ADD_D(v_av41, v_av42);
        // Output point 6 : X[5]
        v_out5 = SUB_D(v_av41, v_av42);

        v_av41 = ADD_D(v_av21, v_av26);
        v_av42 = ADD_D(v_av37, v_av40);
        v_av42 = SWAP_RI_D(CONJ_D(v_av42));
        // Output point 14 : X[13]
        v_out13 = ADD_D(v_av41, v_av42);
        // Output point 3 : X[2]
        v_out2 = SUB_D(v_av41, v_av42);

        v_av44 = SUB_D(v_av19, v_tv11);
        v_tv23 = MUL_D(v_C3, v_av25);
        v_tv24 = MUL_D(v_C4, v_av22);
        v_av45 = SUB_D(v_tv23, v_tv24);
        v_av46 = SUB_D(v_av35, v_tv20);
        v_tv25 = MUL_D(v_C3, v_av39);
        v_tv27 = MUL_D(v_C4, v_av38);
        v_av47 = SUB_D(v_tv25, v_tv27);

        v_av41 = ADD_D(v_av44, v_av45);
        v_av42 = ADD_D(v_av46, v_av47);
        v_av42 = SWAP_RI_D(CONJ_D(v_av42));
        // Output point 5 : X[4]
        v_out4 = ADD_D(v_av41, v_av42);
        // Output point 12 : X[11]
        v_out11 = SUB_D(v_av41, v_av42);

        v_av48 = SUB_D(v_av44, v_av45);
        v_av49 = SUB_D(v_av47, v_av46);
        v_av49 = SWAP_RI_D(CONJ_D(v_av49));
        // Output point 15 : X[14]
        v_out14 = ADD_D(v_av48, v_av49);
        // Output point 2 : X[1]
        v_out1 = SUB_D(v_av48, v_av49);

        v_av48 = SUB_D(v_av21, v_av26);
        v_av49 = SUB_D(v_av40, v_av37);
        v_av49 = SWAP_RI_D(CONJ_D(v_av49));
        // Output point 15 : X[14]
        v_out8 = ADD_D(v_av48, v_av49);
        // Output point 2 : X[1]
        v_out7 = SUB_D(v_av48, v_av49);

        SCATTER_D(curr_out, v_out_stride, v_out0);
        SCATTER_D(curr_out + out_strides[1], v_out_stride, v_out1);
        SCATTER_D(curr_out + out_strides[2], v_out_stride, v_out2);
        SCATTER_D(curr_out + out_strides[3], v_out_stride, v_out3);
        SCATTER_D(curr_out + out_strides[4], v_out_stride, v_out4);
        SCATTER_D(curr_out + out_strides[5], v_out_stride, v_out5);
        SCATTER_D(curr_out + out_strides[6], v_out_stride, v_out6);
        SCATTER_D(curr_out + out_strides[7], v_out_stride, v_out7);
        SCATTER_D(curr_out + out_strides[8], v_out_stride, v_out8);
        SCATTER_D(curr_out + out_strides[9], v_out_stride, v_out9);
        SCATTER_D(curr_out + out_strides[10], v_out_stride, v_out10);
        SCATTER_D(curr_out + out_strides[11], v_out_stride, v_out11);
        SCATTER_D(curr_out + out_strides[12], v_out_stride, v_out12);
        SCATTER_D(curr_out + out_strides[13], v_out_stride, v_out13);
        SCATTER_D(curr_out + out_strides[14], v_out_stride, v_out14);

        in_r += NUM_SETS_D * v_in_stride;
        out_r += NUM_SETS_D * v_out_stride;
    }

    // The following contains code that performs the FFT on the tail cases.
    // These tails are conditionally "instantiated" based on the ISA requested
    // by the "includee" file.

#if defined(KERNEL_USE_AVX512)
    if (remaining_sets >= NUM_SETS_256_D)
    {
        __m256d v_in0, v_in1, v_in2, v_in3, v_in4, v_in5, v_in6, v_in7, v_in8;
        __m256d v_in9, v_in10, v_in11, v_in12, v_in13, v_in14;
        __m256d v_cv1, v_cv2, v_cv3, v_cv4, v_cv5, v_cv6, v_cv7, v_cv8, v_cv9,
            v_cv10;
        __m256d v_cv11, v_cv12;
        __m256d v_av1, v_av2, v_av3, v_av4, v_av5, v_av6, v_av7, v_av8, v_av9;
        __m256d v_av11, v_av12, v_av13, v_av14, v_av15, v_av16, v_av17;
        __m256d v_av18, v_av19, v_av20, v_av21, v_av22, v_av23, v_av24, v_av25;
        __m256d v_av26, v_av27, v_av28, v_av29, v_av30, v_av31, v_av32, v_av33;
        __m256d v_av34, v_av35, v_av36, v_av37, v_av38, v_av39, v_av40, v_av41;
        __m256d v_av42, v_av44, v_av45, v_av46, v_av47, v_av48, v_av49;
        __m256d v_tv1, v_tv2, v_tv3, v_tv4, v_tv5, v_tv6, v_tv7, v_tv8, v_tv9,
            v_tv10;
        __m256d v_tv11, v_tv16, v_tv17, v_tv18;
        __m256d v_tv19, v_tv20, v_tv21, v_tv22, v_tv23, v_tv24, v_tv25, v_tv27;
        __m256d v_out0, v_out1, v_out2, v_out3, v_out4, v_out5;
        __m256d v_out6, v_out7, v_out8, v_out9, v_out10, v_out11;
        __m256d v_out12, v_out13, v_out14;

        __m256d v_K1 = CAST_512_TO_256_D(v_C1);
        __m256d v_K2 = CAST_512_TO_256_D(v_C2);
        __m256d v_K3 = CAST_512_TO_256_D(v_C3);
        __m256d v_K4 = CAST_512_TO_256_D(v_C4);
        __m256d v_K5 = CAST_512_TO_256_D(v_C5);
        __m256d v_K6 = CAST_512_TO_256_D(v_C6);

        curr_in = in_r;
        curr_out = out_r;

        if (flag)
        {
            ITW_GATHER_256_D(curr_in, in_strides, 1, v_in_stride, v_in1, tw,
                             cols, cnt_256);
            ITW_GATHER_256_D(curr_in, in_strides, 2, v_in_stride, v_in2, tw,
                             cols, cnt_256);
            ITW_GATHER_256_D(curr_in, in_strides, 3, v_in_stride, v_in3, tw,
                             cols, cnt_256);
            ITW_GATHER_256_D(curr_in, in_strides, 4, v_in_stride, v_in4, tw,
                             cols, cnt_256);
            ITW_GATHER_256_D(curr_in, in_strides, 5, v_in_stride, v_in5, tw,
                             cols, cnt_256);
            ITW_GATHER_256_D(curr_in, in_strides, 6, v_in_stride, v_in6, tw,
                             cols, cnt_256);
            ITW_GATHER_256_D(curr_in, in_strides, 7, v_in_stride, v_in7, tw,
                             cols, cnt_256);
            ITW_GATHER_256_D(curr_in, in_strides, 8, v_in_stride, v_in8, tw,
                             cols, cnt_256);
            ITW_GATHER_256_D(curr_in, in_strides, 9, v_in_stride, v_in9, tw,
                             cols, cnt_256);
            ITW_GATHER_256_D(curr_in, in_strides, 10, v_in_stride, v_in10, tw,
                             cols, cnt_256);
            ITW_GATHER_256_D(curr_in, in_strides, 11, v_in_stride, v_in11, tw,
                             cols, cnt_256);
            ITW_GATHER_256_D(curr_in, in_strides, 12, v_in_stride, v_in12, tw,
                             cols, cnt_256);
            ITW_GATHER_256_D(curr_in, in_strides, 13, v_in_stride, v_in13, tw,
                             cols, cnt_256);
            ITW_GATHER_256_D(curr_in, in_strides, 14, v_in_stride, v_in14, tw,
                             cols, cnt_256);
        }
        else
        {
            TW_GATHER_256_D(curr_in, in_strides, 1, v_in_stride, v_in1, tw,
                            cols, cnt_256);
            TW_GATHER_256_D(curr_in, in_strides, 2, v_in_stride, v_in2, tw,
                            cols, cnt_256);
            TW_GATHER_256_D(curr_in, in_strides, 3, v_in_stride, v_in3, tw,
                            cols, cnt_256);
            TW_GATHER_256_D(curr_in, in_strides, 4, v_in_stride, v_in4, tw,
                            cols, cnt_256);
            TW_GATHER_256_D(curr_in, in_strides, 5, v_in_stride, v_in5, tw,
                            cols, cnt_256);
            TW_GATHER_256_D(curr_in, in_strides, 6, v_in_stride, v_in6, tw,
                            cols, cnt_256);
            TW_GATHER_256_D(curr_in, in_strides, 7, v_in_stride, v_in7, tw,
                            cols, cnt_256);
            TW_GATHER_256_D(curr_in, in_strides, 8, v_in_stride, v_in8, tw,
                            cols, cnt_256);
            TW_GATHER_256_D(curr_in, in_strides, 9, v_in_stride, v_in9, tw,
                            cols, cnt_256);
            TW_GATHER_256_D(curr_in, in_strides, 10, v_in_stride, v_in10, tw,
                            cols, cnt_256);
            TW_GATHER_256_D(curr_in, in_strides, 11, v_in_stride, v_in11, tw,
                            cols, cnt_256);
            TW_GATHER_256_D(curr_in, in_strides, 12, v_in_stride, v_in12, tw,
                            cols, cnt_256);
            TW_GATHER_256_D(curr_in, in_strides, 13, v_in_stride, v_in13, tw,
                            cols, cnt_256);
            TW_GATHER_256_D(curr_in, in_strides, 14, v_in_stride, v_in14, tw,
                            cols, cnt_256);
        }

        GATHER2_256_D(curr_in, v_in_stride, v_in0);

        // common calculations
        v_cv1 = _mm256_add_pd(v_in10, v_in5);
        v_cv2 = _mm256_add_pd(v_in0, v_cv1);
        v_cv3 = _mm256_add_pd(v_in14, v_in4);
        v_cv4 = _mm256_add_pd(v_in9, v_cv3);
        v_cv5 = _mm256_add_pd(v_in11, v_in1);
        v_cv6 = _mm256_add_pd(v_in6, v_cv5);
        v_av1 = _mm256_add_pd(v_cv4, v_cv6);
        v_cv7 = _mm256_add_pd(v_in2, v_in7);
        v_cv8 = _mm256_add_pd(v_in12, v_cv7);
        v_cv9 = _mm256_add_pd(v_in13, v_in8);
        v_cv10 = _mm256_add_pd(v_in3, v_cv9);
        v_av2 = _mm256_add_pd(v_cv8, v_cv10);
        v_av3 = _mm256_add_pd(v_av1, v_av2);

        // Output point 1 : X[0]
        v_out0 = _mm256_add_pd(v_cv2, v_av3);

        v_tv1 = _mm256_mul_pd(v_K2, v_av3);
        v_av4 = _mm256_sub_pd(v_cv2, v_tv1);
        v_av5 = _mm256_sub_pd(v_av1, v_av2);
        v_tv2 = _mm256_mul_pd(v_K1, v_av5);
        v_av6 = _mm256_add_pd(v_av4, v_tv2);
        v_av7 = _mm256_sub_pd(v_cv6, v_cv4);
        v_av8 = _mm256_sub_pd(v_cv8, v_cv10);
        v_tv3 = _mm256_mul_pd(v_K3, v_av7);
        v_tv4 = _mm256_mul_pd(v_K4, v_av8);
        v_av9 = _mm256_add_pd(v_tv3, v_tv4);

        v_av9 = SWAP_RI_256_D(CONJ_256_D(v_av9));
        // Output point 13 : X[12]
        v_out12 = _mm256_add_pd(v_av6, v_av9);
        // Output point 4 : X[3]
        v_out3 = _mm256_sub_pd(v_av6, v_av9);

        v_av6 = _mm256_sub_pd(v_av4, v_tv2);
        v_tv3 = _mm256_mul_pd(v_K3, v_av8);
        v_tv4 = _mm256_mul_pd(v_K4, v_av7);
        v_av9 = _mm256_sub_pd(v_tv4, v_tv3);

        v_av9 = SWAP_RI_256_D(CONJ_256_D(v_av9));
        // Output point 10 : X[9]
        v_out9 = _mm256_add_pd(v_av6, v_av9);
        // Output point 7 : X[6]
        v_out6 = _mm256_sub_pd(v_av6, v_av9);

        v_tv5 = _mm256_mul_pd(v_K5, v_cv1);
        v_av11 = _mm256_sub_pd(v_in0, v_tv5);

        v_tv6 = _mm256_mul_pd(v_K5, v_cv5);
        v_av12 = _mm256_sub_pd(v_in6, v_tv6);
        v_tv7 = _mm256_mul_pd(v_K5, v_cv3);
        v_av13 = _mm256_sub_pd(v_in9, v_tv7);
        v_av14 = _mm256_add_pd(v_av12, v_av13);
        v_tv8 = _mm256_mul_pd(v_K5, v_cv7);
        v_av15 = _mm256_sub_pd(v_in12, v_tv8);
        v_tv9 = _mm256_mul_pd(v_K5, v_cv9);
        v_av16 = _mm256_sub_pd(v_in3, v_tv9);
        v_av17 = _mm256_add_pd(v_av15, v_av16);
        v_av18 = _mm256_add_pd(v_av14, v_av17);
        v_tv10 = _mm256_mul_pd(v_K2, v_av18);
        v_av19 = _mm256_sub_pd(v_av11, v_tv10);
        v_av20 = _mm256_sub_pd(v_av14, v_av17);
        v_tv11 = _mm256_mul_pd(v_K1, v_av20);
        v_av21 = _mm256_add_pd(v_av19, v_tv11);
        v_cv11 = _mm256_sub_pd(v_in14, v_in4);
        v_av29 = _mm256_mul_pd(v_K6, v_cv11);
        v_cv12 = _mm256_sub_pd(v_in11, v_in1);
        v_av28 = _mm256_mul_pd(v_K6, v_cv12);
        v_av22 = _mm256_sub_pd(v_av29, v_av28);
        v_av23 = _mm256_sub_pd(v_in8, v_in13);
        v_av32 = _mm256_mul_pd(v_K6, v_av23);
        v_av24 = _mm256_sub_pd(v_in2, v_in7);
        v_av31 = _mm256_mul_pd(v_K6, v_av24);
        v_av25 = _mm256_sub_pd(v_av32, v_av31);
        v_tv16 = _mm256_mul_pd(v_K3, v_av22);
        v_tv17 = _mm256_mul_pd(v_K4, v_av25);
        v_av26 = _mm256_add_pd(v_tv16, v_tv17);
        v_av27 = _mm256_sub_pd(v_in10, v_in5);
        v_tv18 = _mm256_mul_pd(v_K6, v_av27);
        v_av30 = _mm256_add_pd(v_av28, v_av29);
        v_av33 = _mm256_add_pd(v_av31, v_av32);
        v_av34 = _mm256_add_pd(v_av30, v_av33);
        v_tv19 = _mm256_mul_pd(v_K2, v_av34);
        v_av35 = _mm256_add_pd(v_tv18, v_tv19);
        v_av36 = _mm256_sub_pd(v_av33, v_av30);
        v_tv20 = _mm256_mul_pd(v_K1, v_av36);
        v_av37 = _mm256_add_pd(v_av35, v_tv20);
        v_av38 = _mm256_sub_pd(v_av13, v_av12);
        v_av39 = _mm256_sub_pd(v_av16, v_av15);
        v_tv21 = _mm256_mul_pd(v_K3, v_av38);
        v_tv22 = _mm256_mul_pd(v_K4, v_av39);
        v_av40 = _mm256_add_pd(v_tv21, v_tv22);

        // real part
        v_av41 = _mm256_add_pd(v_av11, v_av18);

        // imag part
        v_av42 = _mm256_sub_pd(v_tv18, v_av34);

        v_av42 = SWAP_RI_256_D(CONJ_256_D(v_av42));
        // Output point 11 : X[10]
        v_out10 = _mm256_add_pd(v_av41, v_av42);
        // Output point 6 : X[5]
        v_out5 = _mm256_sub_pd(v_av41, v_av42);

        // real part
        v_av41 = _mm256_add_pd(v_av21, v_av26);

        // imag part
        v_av42 = _mm256_add_pd(v_av37, v_av40);

        v_av42 = SWAP_RI_256_D(CONJ_256_D(v_av42));
        // Output point 14 : X[13]
        v_out13 = _mm256_add_pd(v_av41, v_av42);
        // Output point 3 : X[2]
        v_out2 = _mm256_sub_pd(v_av41, v_av42);

        v_av44 = _mm256_sub_pd(v_av19, v_tv11);
        v_tv23 = _mm256_mul_pd(v_K3, v_av25);
        v_tv24 = _mm256_mul_pd(v_K4, v_av22);
        v_av45 = _mm256_sub_pd(v_tv23, v_tv24);
        v_av46 = _mm256_sub_pd(v_av35, v_tv20);
        v_tv25 = _mm256_mul_pd(v_K3, v_av39);
        v_tv27 = _mm256_mul_pd(v_K4, v_av38);
        v_av47 = _mm256_sub_pd(v_tv25, v_tv27);

        // real part
        v_av41 = _mm256_add_pd(v_av44, v_av45);

        // imag part
        v_av42 = _mm256_add_pd(v_av46, v_av47);

        v_av42 = SWAP_RI_256_D(CONJ_256_D(v_av42));
        // Output point 5 : X[4]
        v_out4 = _mm256_add_pd(v_av41, v_av42);
        // Output point 12 : X[11]
        v_out11 = _mm256_sub_pd(v_av41, v_av42);

        // real part
        v_av48 = _mm256_sub_pd(v_av44, v_av45);

        // imag part
        v_av49 = _mm256_sub_pd(v_av47, v_av46);

        v_av49 = SWAP_RI_256_D(CONJ_256_D(v_av49));
        // Output point 15 : X[14]
        v_out14 = _mm256_add_pd(v_av48, v_av49);
        // Output point 2 : X[1]
        v_out1 = _mm256_sub_pd(v_av48, v_av49);

        // real part
        v_av48 = _mm256_sub_pd(v_av21, v_av26);

        // imag part
        v_av49 = _mm256_sub_pd(v_av40, v_av37);

        v_av49 = SWAP_RI_256_D(CONJ_256_D(v_av49));
        // Output point 15 : X[14]
        v_out8 = _mm256_add_pd(v_av48, v_av49);
        // Output point 2 : X[1]
        v_out7 = _mm256_sub_pd(v_av48, v_av49);

        SCATTER2_256_D(curr_out, v_out_stride, v_out0);
        SCATTER2_256_D(curr_out + out_strides[1], v_out_stride, v_out1);
        SCATTER2_256_D(curr_out + out_strides[2], v_out_stride, v_out2);
        SCATTER2_256_D(curr_out + out_strides[3], v_out_stride, v_out3);
        SCATTER2_256_D(curr_out + out_strides[4], v_out_stride, v_out4);
        SCATTER2_256_D(curr_out + out_strides[5], v_out_stride, v_out5);
        SCATTER2_256_D(curr_out + out_strides[6], v_out_stride, v_out6);
        SCATTER2_256_D(curr_out + out_strides[7], v_out_stride, v_out7);
        SCATTER2_256_D(curr_out + out_strides[8], v_out_stride, v_out8);
        SCATTER2_256_D(curr_out + out_strides[9], v_out_stride, v_out9);
        SCATTER2_256_D(curr_out + out_strides[10], v_out_stride, v_out10);
        SCATTER2_256_D(curr_out + out_strides[11], v_out_stride, v_out11);
        SCATTER2_256_D(curr_out + out_strides[12], v_out_stride, v_out12);
        SCATTER2_256_D(curr_out + out_strides[13], v_out_stride, v_out13);
        SCATTER2_256_D(curr_out + out_strides[14], v_out_stride, v_out14);

        in_r += NUM_SETS_256_D * v_in_stride;
        out_r += NUM_SETS_256_D * v_out_stride;
    }
#endif

#if defined(KERNEL_USE_AVX512) || defined(KERNEL_USE_AVX256)
    if (remaining_sets & 1)
    {
        __m128d v_in0, v_in1, v_in2, v_in3, v_in4, v_in5, v_in6, v_in7, v_in8;
        __m128d v_in9, v_in10, v_in11, v_in12, v_in13, v_in14;
        __m128d v_cv1, v_cv2, v_cv3, v_cv4, v_cv5, v_cv6, v_cv7, v_cv8, v_cv9,
            v_cv10;
        __m128d v_cv11, v_cv12;
        __m128d v_av1, v_av2, v_av3, v_av4, v_av5, v_av6, v_av7, v_av8, v_av9;
        __m128d v_av11, v_av12, v_av13, v_av14, v_av15, v_av16, v_av17;
        __m128d v_av18, v_av19, v_av20, v_av21, v_av22, v_av23, v_av24, v_av25;
        __m128d v_av26, v_av27, v_av28, v_av29, v_av30, v_av31, v_av32, v_av33;
        __m128d v_av34, v_av35, v_av36, v_av37, v_av38, v_av39, v_av40, v_av41;
        __m128d v_av42, v_av44, v_av45, v_av46, v_av47, v_av48, v_av49;
        __m128d v_tv1, v_tv2, v_tv3, v_tv4, v_tv5, v_tv6, v_tv7, v_tv8, v_tv9,
            v_tv10;
        __m128d v_tv11, v_tv16, v_tv17, v_tv18;
        __m128d v_tv19, v_tv20, v_tv21, v_tv22, v_tv23, v_tv24, v_tv25, v_tv27;
        __m128d v_out0, v_out1, v_out2, v_out3, v_out4, v_out5;
        __m128d v_out6, v_out7, v_out8, v_out9, v_out10, v_out11;
        __m128d v_out12, v_out13, v_out14;

#if defined(KERNEL_USE_AVX512)
        __m128d v_K1 = CAST_512_TO_128_D(v_C1);
        __m128d v_K2 = CAST_512_TO_128_D(v_C2);
        __m128d v_K3 = CAST_512_TO_128_D(v_C3);
        __m128d v_K4 = CAST_512_TO_128_D(v_C4);
        __m128d v_K5 = CAST_512_TO_128_D(v_C5);
        __m128d v_K6 = CAST_512_TO_128_D(v_C6);
#elif defined(KERNEL_USE_AVX256)
        __m128d v_K1 = CAST_256_TO_128_D(v_C1);
        __m128d v_K2 = CAST_256_TO_128_D(v_C2);
        __m128d v_K3 = CAST_256_TO_128_D(v_C3);
        __m128d v_K4 = CAST_256_TO_128_D(v_C4);
        __m128d v_K5 = CAST_256_TO_128_D(v_C5);
        __m128d v_K6 = CAST_256_TO_128_D(v_C6);
#endif

        curr_in = in_r;
        curr_out = out_r;

        if (flag)
        {
            ITW_GATHER_128_D(curr_in, in_strides, 1, /* unused */ 0, v_in1, tw,
                             cols, cnt_128);
            ITW_GATHER_128_D(curr_in, in_strides, 2, /* unused */ 0, v_in2, tw,
                             cols, cnt_128);
            ITW_GATHER_128_D(curr_in, in_strides, 3, /* unused */ 0, v_in3, tw,
                             cols, cnt_128);
            ITW_GATHER_128_D(curr_in, in_strides, 4, /* unused */ 0, v_in4, tw,
                             cols, cnt_128);
            ITW_GATHER_128_D(curr_in, in_strides, 5, /* unused */ 0, v_in5, tw,
                             cols, cnt_128);
            ITW_GATHER_128_D(curr_in, in_strides, 6, /* unused */ 0, v_in6, tw,
                             cols, cnt_128);
            ITW_GATHER_128_D(curr_in, in_strides, 7, /* unused */ 0, v_in7, tw,
                             cols, cnt_128);
            ITW_GATHER_128_D(curr_in, in_strides, 8, /* unused */ 0, v_in8, tw,
                             cols, cnt_128);
            ITW_GATHER_128_D(curr_in, in_strides, 9, /* unused */ 0, v_in9, tw,
                             cols, cnt_128);
            ITW_GATHER_128_D(curr_in, in_strides, 10, /* unused */ 0, v_in10,
                             tw, cols, cnt_128);
            ITW_GATHER_128_D(curr_in, in_strides, 11, /* unused */ 0, v_in11,
                             tw, cols, cnt_128);
            ITW_GATHER_128_D(curr_in, in_strides, 12, /* unused */ 0, v_in12,
                             tw, cols, cnt_128);
            ITW_GATHER_128_D(curr_in, in_strides, 13, /* unused */ 0, v_in13,
                             tw, cols, cnt_128);
            ITW_GATHER_128_D(curr_in, in_strides, 14, /* unused */ 0, v_in14,
                             tw, cols, cnt_128);
        }
        else
        {
            TW_GATHER_128_D(curr_in, in_strides, 1, /* unused */ 0, v_in1, tw,
                            cols, cnt_128);
            TW_GATHER_128_D(curr_in, in_strides, 2, /* unused */ 0, v_in2, tw,
                            cols, cnt_128);
            TW_GATHER_128_D(curr_in, in_strides, 3, /* unused */ 0, v_in3, tw,
                            cols, cnt_128);
            TW_GATHER_128_D(curr_in, in_strides, 4, /* unused */ 0, v_in4, tw,
                            cols, cnt_128);
            TW_GATHER_128_D(curr_in, in_strides, 5, /* unused */ 0, v_in5, tw,
                            cols, cnt_128);
            TW_GATHER_128_D(curr_in, in_strides, 6, /* unused */ 0, v_in6, tw,
                            cols, cnt_128);
            TW_GATHER_128_D(curr_in, in_strides, 7, /* unused */ 0, v_in7, tw,
                            cols, cnt_128);
            TW_GATHER_128_D(curr_in, in_strides, 8, /* unused */ 0, v_in8, tw,
                            cols, cnt_128);
            TW_GATHER_128_D(curr_in, in_strides, 9, /* unused */ 0, v_in9, tw,
                            cols, cnt_128);
            TW_GATHER_128_D(curr_in, in_strides, 10, /* unused */ 0, v_in10, tw,
                            cols, cnt_128);
            TW_GATHER_128_D(curr_in, in_strides, 11, /* unused */ 0, v_in11, tw,
                            cols, cnt_128);
            TW_GATHER_128_D(curr_in, in_strides, 12, /* unused */ 0, v_in12, tw,
                            cols, cnt_128);
            TW_GATHER_128_D(curr_in, in_strides, 13, /* unused */ 0, v_in13, tw,
                            cols, cnt_128);
            TW_GATHER_128_D(curr_in, in_strides, 14, /* unused */ 0, v_in14, tw,
                            cols, cnt_128);
        }

        LD_128_D(curr_in, v_in0);

        // common calculations
        v_cv1 = _mm_add_pd(v_in10, v_in5);
        v_cv2 = _mm_add_pd(v_in0, v_cv1);
        v_cv3 = _mm_add_pd(v_in14, v_in4);
        v_cv4 = _mm_add_pd(v_in9, v_cv3);
        v_cv5 = _mm_add_pd(v_in11, v_in1);
        v_cv6 = _mm_add_pd(v_in6, v_cv5);
        v_av1 = _mm_add_pd(v_cv4, v_cv6);
        v_cv7 = _mm_add_pd(v_in2, v_in7);
        v_cv8 = _mm_add_pd(v_in12, v_cv7);
        v_cv9 = _mm_add_pd(v_in13, v_in8);
        v_cv10 = _mm_add_pd(v_in3, v_cv9);
        v_av2 = _mm_add_pd(v_cv8, v_cv10);
        v_av3 = _mm_add_pd(v_av1, v_av2);

        // Output point 1 : X[0]
        v_out0 = _mm_add_pd(v_cv2, v_av3);

        v_tv1 = _mm_mul_pd(v_K2, v_av3);
        v_av4 = _mm_sub_pd(v_cv2, v_tv1);
        v_av5 = _mm_sub_pd(v_av1, v_av2);
        v_tv2 = _mm_mul_pd(v_K1, v_av5);
        v_av6 = _mm_add_pd(v_av4, v_tv2);
        v_av7 = _mm_sub_pd(v_cv6, v_cv4);
        v_av8 = _mm_sub_pd(v_cv8, v_cv10);
        v_tv3 = _mm_mul_pd(v_K3, v_av7);
        v_tv4 = _mm_mul_pd(v_K4, v_av8);
        v_av9 = _mm_add_pd(v_tv3, v_tv4);

        v_av9 = SWAP_RI_128_D(CONJ_128_D(v_av9));
        // Output point 13 : X[12]
        v_out12 = _mm_add_pd(v_av6, v_av9);
        // Output point 4 : X[3]
        v_out3 = _mm_sub_pd(v_av6, v_av9);

        v_av6 = _mm_sub_pd(v_av4, v_tv2);
        v_tv3 = _mm_mul_pd(v_K3, v_av8);
        v_tv4 = _mm_mul_pd(v_K4, v_av7);
        v_av9 = _mm_sub_pd(v_tv4, v_tv3);

        v_av9 = SWAP_RI_128_D(CONJ_128_D(v_av9));
        // Output point 10 : X[9]
        v_out9 = _mm_add_pd(v_av6, v_av9);
        // Output point 7 : X[6]
        v_out6 = _mm_sub_pd(v_av6, v_av9);

        v_tv5 = _mm_mul_pd(v_K5, v_cv1);
        v_av11 = _mm_sub_pd(v_in0, v_tv5);

        v_tv6 = _mm_mul_pd(v_K5, v_cv5);
        v_av12 = _mm_sub_pd(v_in6, v_tv6);
        v_tv7 = _mm_mul_pd(v_K5, v_cv3);
        v_av13 = _mm_sub_pd(v_in9, v_tv7);
        v_av14 = _mm_add_pd(v_av12, v_av13);
        v_tv8 = _mm_mul_pd(v_K5, v_cv7);
        v_av15 = _mm_sub_pd(v_in12, v_tv8);
        v_tv9 = _mm_mul_pd(v_K5, v_cv9);
        v_av16 = _mm_sub_pd(v_in3, v_tv9);
        v_av17 = _mm_add_pd(v_av15, v_av16);
        v_av18 = _mm_add_pd(v_av14, v_av17);
        v_tv10 = _mm_mul_pd(v_K2, v_av18);
        v_av19 = _mm_sub_pd(v_av11, v_tv10);
        v_av20 = _mm_sub_pd(v_av14, v_av17);
        v_tv11 = _mm_mul_pd(v_K1, v_av20);
        v_av21 = _mm_add_pd(v_av19, v_tv11);
        v_cv11 = _mm_sub_pd(v_in14, v_in4);
        v_av29 = _mm_mul_pd(v_K6, v_cv11);
        v_cv12 = _mm_sub_pd(v_in11, v_in1);
        v_av28 = _mm_mul_pd(v_K6, v_cv12);
        v_av22 = _mm_sub_pd(v_av29, v_av28);
        v_av23 = _mm_sub_pd(v_in8, v_in13);
        v_av32 = _mm_mul_pd(v_K6, v_av23);
        v_av24 = _mm_sub_pd(v_in2, v_in7);
        v_av31 = _mm_mul_pd(v_K6, v_av24);
        v_av25 = _mm_sub_pd(v_av32, v_av31);
        v_tv16 = _mm_mul_pd(v_K3, v_av22);
        v_tv17 = _mm_mul_pd(v_K4, v_av25);
        v_av26 = _mm_add_pd(v_tv16, v_tv17);
        v_av27 = _mm_sub_pd(v_in10, v_in5);
        v_tv18 = _mm_mul_pd(v_K6, v_av27);
        v_av30 = _mm_add_pd(v_av28, v_av29);
        v_av33 = _mm_add_pd(v_av31, v_av32);
        v_av34 = _mm_add_pd(v_av30, v_av33);
        v_tv19 = _mm_mul_pd(v_K2, v_av34);
        v_av35 = _mm_add_pd(v_tv18, v_tv19);
        v_av36 = _mm_sub_pd(v_av33, v_av30);
        v_tv20 = _mm_mul_pd(v_K1, v_av36);
        v_av37 = _mm_add_pd(v_av35, v_tv20);
        v_av38 = _mm_sub_pd(v_av13, v_av12);
        v_av39 = _mm_sub_pd(v_av16, v_av15);
        v_tv21 = _mm_mul_pd(v_K3, v_av38);
        v_tv22 = _mm_mul_pd(v_K4, v_av39);
        v_av40 = _mm_add_pd(v_tv21, v_tv22);

        // real part
        v_av41 = _mm_add_pd(v_av11, v_av18);

        // imag part
        v_av42 = _mm_sub_pd(v_tv18, v_av34);

        v_av42 = SWAP_RI_128_D(CONJ_128_D(v_av42));
        // Output point 11 : X[10]
        v_out10 = _mm_add_pd(v_av41, v_av42);
        // Output point 6 : X[5]
        v_out5 = _mm_sub_pd(v_av41, v_av42);

        // real part
        v_av41 = _mm_add_pd(v_av21, v_av26);

        // imag part
        v_av42 = _mm_add_pd(v_av37, v_av40);

        v_av42 = SWAP_RI_128_D(CONJ_128_D(v_av42));
        // Output point 14 : X[13]
        v_out13 = _mm_add_pd(v_av41, v_av42);
        // Output point 3 : X[2]
        v_out2 = _mm_sub_pd(v_av41, v_av42);

        v_av44 = _mm_sub_pd(v_av19, v_tv11);
        v_tv23 = _mm_mul_pd(v_K3, v_av25);
        v_tv24 = _mm_mul_pd(v_K4, v_av22);
        v_av45 = _mm_sub_pd(v_tv23, v_tv24);
        v_av46 = _mm_sub_pd(v_av35, v_tv20);
        v_tv25 = _mm_mul_pd(v_K3, v_av39);
        v_tv27 = _mm_mul_pd(v_K4, v_av38);
        v_av47 = _mm_sub_pd(v_tv25, v_tv27);

        // real part
        v_av41 = _mm_add_pd(v_av44, v_av45);

        // imag part
        v_av42 = _mm_add_pd(v_av46, v_av47);

        v_av42 = SWAP_RI_128_D(CONJ_128_D(v_av42));
        // Output point 5 : X[4]
        v_out4 = _mm_add_pd(v_av41, v_av42);
        // Output point 12 : X[11]
        v_out11 = _mm_sub_pd(v_av41, v_av42);

        // real part
        v_av48 = _mm_sub_pd(v_av44, v_av45);

        // imag part
        v_av49 = _mm_sub_pd(v_av47, v_av46);

        v_av49 = SWAP_RI_128_D(CONJ_128_D(v_av49));
        // Output point 15 : X[14]
        v_out14 = _mm_add_pd(v_av48, v_av49);
        // Output point 2 : X[1]
        v_out1 = _mm_sub_pd(v_av48, v_av49);

        // real part
        v_av48 = _mm_sub_pd(v_av21, v_av26);

        // imag part
        v_av49 = _mm_sub_pd(v_av40, v_av37);

        v_av49 = SWAP_RI_128_D(CONJ_128_D(v_av49));
        // Output point 15 : X[14]
        v_out8 = _mm_add_pd(v_av48, v_av49);
        // Output point 2 : X[1]
        v_out7 = _mm_sub_pd(v_av48, v_av49);

        ST_128_D(curr_out, v_out0);
        ST_128_D(curr_out + out_strides[1], v_out1);
        ST_128_D(curr_out + out_strides[2], v_out2);
        ST_128_D(curr_out + out_strides[3], v_out3);
        ST_128_D(curr_out + out_strides[4], v_out4);
        ST_128_D(curr_out + out_strides[5], v_out5);
        ST_128_D(curr_out + out_strides[6], v_out6);
        ST_128_D(curr_out + out_strides[7], v_out7);
        ST_128_D(curr_out + out_strides[8], v_out8);
        ST_128_D(curr_out + out_strides[9], v_out9);
        ST_128_D(curr_out + out_strides[10], v_out10);
        ST_128_D(curr_out + out_strides[11], v_out11);
        ST_128_D(curr_out + out_strides[12], v_out12);
        ST_128_D(curr_out + out_strides[13], v_out13);
        ST_128_D(curr_out + out_strides[14], v_out14);
    }
#endif
#ifdef AOCL_ENABLE_LOG
    AOCLFFTZ_LOG_UNFORMATTED(TRACE, TRACE, "Exit");
#endif
}

kfft_ REGISTER_KERNEL(UINT8 precision, UINT8 direction)
{
    if (precision == DT_FLOAT)
    {
        return TWID_KNAME_FP32;
    }
    else if (precision == DT_DOUBLE)
    {
        return TWID_KNAME_FP64;
    }
    else
    {
        return NULL;
    }
}

ops_cycles_t GET_OPS_COUNT(UINT8 precision, UINT8 direction)
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
