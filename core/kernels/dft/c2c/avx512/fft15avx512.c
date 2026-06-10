// Copyright Advanced Micro Devices, Inc.
// SPDX-License-Identifier: BSD-3-Clause

/** @file fft15avx512.c
 *
 *  @brief Radix-15 FFT kernel with AVX-512 operations using x86 SIMD intrinsics
 *
 *  This file contains the DIT radix-15 FFT implementations using AVX-512
 *  operations for single-precision and double-precision inputs.
 *
 *  @author D. Vijay Krishna
 *
 */

#include "core/kernels/kernel.h"
#include "core/kernels/simd_includes/simd_common_avx512.h"

static const ops_cycles_t ops_cnt[NUM_PRECISIONS] = {{0, 28, 78, 240, 52, 97},
                                                     {0, 28, 78, 120,  7, 97}};

ops_cycles_t get_ops_cnt_fft15avx512(UINT8 precision, UINT8 direction)
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

static VOID fft15avx512fp32(VOID *in_real, VOID *in_imag, VOID *out_real,
                            VOID *out_imag, INTP n, aoclfftz_strides_t *strides,
                            VOID *twd, UINT8 flag)
{
    AOCLFFTZ_LOG(DEBUG, global_logger_mode, "Enter");
    const FLOAT CRTM_15[6] = {
        0.55901699437494742410229341718281905886015458990288f,
        0.25000000000000000000000000000000000000000000000000f,
        0.95105651629515357211643933337938214340569863400000f,
        0.58778525229247301629891039327884007596190389052978f,
        0.50000000000000000000000000000000000000000000000000f,
        0.86602540378443864676372317075293618347140262690519f};

    FLOAT *in_r = (FLOAT *)in_real;
    FLOAT *out_r = (FLOAT *)out_real;
    FLOAT *curr_in, *curr_out;

#ifdef VOLATILE_STRIDE_ARRAY
    volatile INTP *in_strides = strides->in_strides;
    volatile INTP *out_strides = strides->out_strides;
#else
    INTP *in_strides = strides->in_strides;
    INTP *out_strides = strides->out_strides;
#endif
    INTP v_in_stride = strides->v_in_stride;
    UINT8 is_contiguous_in = (v_in_stride == DATA_STRIDE);
    INTP v_out_stride = strides->v_out_stride;
    UINT8 is_contiguous_out = (v_out_stride == DATA_STRIDE);

    INTP N = n / NUM_SETS_512_S;
    INTP remaining_sets = n % NUM_SETS_512_S;
    INTP count;

    __m512 v_C1 = _mm512_set1_ps(CRTM_15[0]);
    __m512 v_C2 = _mm512_set1_ps(CRTM_15[1]);
    __m512 v_C3 = _mm512_set1_ps(CRTM_15[2]);
    v_C3 = _mm512_xor_ps(v_C3, _neg_512_f[flag].s);
    __m512 v_C4 = _mm512_set1_ps(CRTM_15[3]);
    v_C4 = _mm512_xor_ps(v_C4, _neg_512_f[flag].s);
    __m512 v_C5 = _mm512_set1_ps(CRTM_15[4]);
    __m512 v_C6 = _mm512_set1_ps(CRTM_15[5]);
    v_C6 = _mm512_xor_ps(v_C6, _neg_512_f[flag].s);

    for (count = 0; count < N; count++)
    {
        __m512 v_in0, v_in1, v_in2, v_in3, v_in4, v_in5, v_in6, v_in7, v_in8;
        __m512 v_in9, v_in10, v_in11, v_in12, v_in13, v_in14;
        __m512 v_cv1, v_cv2, v_cv3, v_cv4, v_cv5, v_cv6, v_cv7, v_cv8, v_cv9,
               v_cv10;
        __m512 v_cv11, v_cv12;
        __m512 v_av1, v_av2, v_av3, v_av4, v_av5, v_av6, v_av7, v_av8, v_av9;
        __m512 v_av11, v_av12, v_av13, v_av14, v_av15, v_av16, v_av17;
        __m512 v_av18, v_av19, v_av20, v_av21, v_av22, v_av23, v_av24, v_av25;
        __m512 v_av26, v_av27, v_av28, v_av29, v_av30, v_av31, v_av32, v_av33;
        __m512 v_av34, v_av35, v_av36, v_av37, v_av38, v_av39, v_av40, v_av41;
        __m512 v_av42, v_av44, v_av45, v_av46, v_av47, v_av48, v_av49;
        __m512 v_tv1, v_tv2, v_tv3, v_tv4, v_tv5, v_tv6, v_tv7, v_tv8, v_tv9,
               v_tv10;
        __m512 v_tv11, v_tv16, v_tv17, v_tv18;
        __m512 v_tv19, v_tv20, v_tv21, v_tv22, v_tv23, v_tv24, v_tv25, v_tv27;
        __m512 v_out0, v_out1, v_out2, v_out3, v_out4, v_out5;
        __m512 v_out6, v_out7, v_out8, v_out9, v_out10, v_out11;
        __m512 v_out12, v_out13, v_out14;

        curr_in = in_r;
        curr_out = out_r;

        GATHER8_512_S(curr_in, v_in_stride, v_in0, is_contiguous_in);
        curr_in = in_r + in_strides[1];
        GATHER8_512_S(curr_in, v_in_stride, v_in1, is_contiguous_in);
        curr_in = in_r + in_strides[2];
        GATHER8_512_S(curr_in, v_in_stride, v_in2, is_contiguous_in);
        curr_in = in_r + in_strides[3];
        GATHER8_512_S(curr_in, v_in_stride, v_in3, is_contiguous_in);
        curr_in = in_r + in_strides[4];
        GATHER8_512_S(curr_in, v_in_stride, v_in4, is_contiguous_in);
        curr_in = in_r + in_strides[5];
        GATHER8_512_S(curr_in, v_in_stride, v_in5, is_contiguous_in);
        curr_in = in_r + in_strides[6];
        GATHER8_512_S(curr_in, v_in_stride, v_in6, is_contiguous_in);
        curr_in = in_r + in_strides[7];
        GATHER8_512_S(curr_in, v_in_stride, v_in7, is_contiguous_in);
        curr_in = in_r + in_strides[8];
        GATHER8_512_S(curr_in, v_in_stride, v_in8, is_contiguous_in);
        curr_in = in_r + in_strides[9];
        GATHER8_512_S(curr_in, v_in_stride, v_in9, is_contiguous_in);
        curr_in = in_r + in_strides[10];
        GATHER8_512_S(curr_in, v_in_stride, v_in10, is_contiguous_in);
        curr_in = in_r + in_strides[11];
        GATHER8_512_S(curr_in, v_in_stride, v_in11, is_contiguous_in);
        curr_in = in_r + in_strides[12];
        GATHER8_512_S(curr_in, v_in_stride, v_in12, is_contiguous_in);
        curr_in = in_r + in_strides[13];
        GATHER8_512_S(curr_in, v_in_stride, v_in13, is_contiguous_in);
        curr_in = in_r + in_strides[14];
        GATHER8_512_S(curr_in, v_in_stride, v_in14, is_contiguous_in);

        // common calculations
        v_cv1 = _mm512_add_ps(v_in10, v_in5);
        v_cv2 = _mm512_add_ps(v_in0, v_cv1);
        v_cv3 = _mm512_add_ps(v_in14, v_in4);
        v_cv4 = _mm512_add_ps(v_in9, v_cv3);
        v_cv5 = _mm512_add_ps(v_in11, v_in1);
        v_cv6 = _mm512_add_ps(v_in6, v_cv5);
        v_av1 = _mm512_add_ps(v_cv4, v_cv6);
        v_cv7 = _mm512_add_ps(v_in2, v_in7);
        v_cv8 = _mm512_add_ps(v_in12, v_cv7);
        v_cv9 = _mm512_add_ps(v_in13, v_in8);
        v_cv10 = _mm512_add_ps(v_in3, v_cv9);
        v_av2 = _mm512_add_ps(v_cv8, v_cv10);
        v_av3 = _mm512_add_ps(v_av1, v_av2);

        // Output point 1 : X[0]
        v_out0 = _mm512_add_ps(v_cv2, v_av3);

        v_tv1 = _mm512_mul_ps(v_C2, v_av3);
        v_av4 = _mm512_sub_ps(v_cv2, v_tv1);
        v_av5 = _mm512_sub_ps(v_av1, v_av2);
        v_tv2 = _mm512_mul_ps(v_C1, v_av5);
        v_av6 = _mm512_add_ps(v_av4, v_tv2);
        v_av7 = _mm512_sub_ps(v_cv6, v_cv4);
        v_av8 = _mm512_sub_ps(v_cv8, v_cv10);
        v_tv3 = _mm512_mul_ps(v_C3, v_av7);
        v_tv4 = _mm512_mul_ps(v_C4, v_av8);
        v_av9 = _mm512_add_ps(v_tv3, v_tv4);

        v_av9 = SWAP_RI_512_S(CONJ_512_S(v_av9));
        // Output point 13 : X[12]
        v_out12 = _mm512_add_ps(v_av6, v_av9);
        // Output point 4 : X[3]
        v_out3 = _mm512_sub_ps(v_av6, v_av9);

        v_av6 = _mm512_sub_ps(v_av4, v_tv2);
        v_tv3 = _mm512_mul_ps(v_C3, v_av8);
        v_tv4 = _mm512_mul_ps(v_C4, v_av7);
        v_av9 = _mm512_sub_ps(v_tv4, v_tv3);

        v_av9 = SWAP_RI_512_S(CONJ_512_S(v_av9));
        // Output point 10 : X[9]
        v_out9 = _mm512_add_ps(v_av6, v_av9);
        // Output point 7 : X[6]
        v_out6 = _mm512_sub_ps(v_av6, v_av9);

        v_tv5 = _mm512_mul_ps(v_C5, v_cv1);
        v_av11 = _mm512_sub_ps(v_in0, v_tv5);

        v_tv6 = _mm512_mul_ps(v_C5, v_cv5);
        v_av12 = _mm512_sub_ps(v_in6, v_tv6);
        v_tv7 = _mm512_mul_ps(v_C5, v_cv3);
        v_av13 = _mm512_sub_ps(v_in9, v_tv7);
        v_av14 = _mm512_add_ps(v_av12, v_av13);
        v_tv8 = _mm512_mul_ps(v_C5, v_cv7);
        v_av15 = _mm512_sub_ps(v_in12, v_tv8);
        v_tv9 = _mm512_mul_ps(v_C5, v_cv9);
        v_av16 = _mm512_sub_ps(v_in3, v_tv9);
        v_av17 = _mm512_add_ps(v_av15, v_av16);
        v_av18 = _mm512_add_ps(v_av14, v_av17);
        v_tv10 = _mm512_mul_ps(v_C2, v_av18);
        v_av19 = _mm512_sub_ps(v_av11, v_tv10);
        v_av20 = _mm512_sub_ps(v_av14, v_av17);
        v_tv11 = _mm512_mul_ps(v_C1, v_av20);
        v_av21 = _mm512_add_ps(v_av19, v_tv11);
        v_cv11 = _mm512_sub_ps(v_in14, v_in4);
        v_av29 = _mm512_mul_ps(v_C6, v_cv11);
        v_cv12 = _mm512_sub_ps(v_in11, v_in1);
        v_av28 = _mm512_mul_ps(v_C6, v_cv12);
        v_av22 = _mm512_sub_ps(v_av29, v_av28);
        v_av23 = _mm512_sub_ps(v_in8, v_in13);
        v_av32 = _mm512_mul_ps(v_C6, v_av23);
        v_av24 = _mm512_sub_ps(v_in2, v_in7);
        v_av31 = _mm512_mul_ps(v_C6, v_av24);
        v_av25 = _mm512_sub_ps(v_av32, v_av31);
        v_tv16 = _mm512_mul_ps(v_C3, v_av22);
        v_tv17 = _mm512_mul_ps(v_C4, v_av25);
        v_av26 = _mm512_add_ps(v_tv16, v_tv17);
        v_av27 = _mm512_sub_ps(v_in10, v_in5);
        v_tv18 = _mm512_mul_ps(v_C6, v_av27);
        v_av30 = _mm512_add_ps(v_av28, v_av29);
        v_av33 = _mm512_add_ps(v_av31, v_av32);
        v_av34 = _mm512_add_ps(v_av30, v_av33);
        v_tv19 = _mm512_mul_ps(v_C2, v_av34);
        v_av35 = _mm512_add_ps(v_tv18, v_tv19);
        v_av36 = _mm512_sub_ps(v_av33, v_av30);
        v_tv20 = _mm512_mul_ps(v_C1, v_av36);
        v_av37 = _mm512_add_ps(v_av35, v_tv20);
        v_av38 = _mm512_sub_ps(v_av13, v_av12);
        v_av39 = _mm512_sub_ps(v_av16, v_av15);
        v_tv21 = _mm512_mul_ps(v_C3, v_av38);
        v_tv22 = _mm512_mul_ps(v_C4, v_av39);
        v_av40 = _mm512_add_ps(v_tv21, v_tv22);

        v_av41 = _mm512_add_ps(v_av11, v_av18);
        v_av42 = _mm512_sub_ps(v_tv18, v_av34);
        v_av42 = SWAP_RI_512_S(CONJ_512_S(v_av42));
        // Output point 11 : X[10]
        v_out10 = _mm512_add_ps(v_av41, v_av42);
        // Output point 6 : X[5]
        v_out5 = _mm512_sub_ps(v_av41, v_av42);

        v_av41 = _mm512_add_ps(v_av21, v_av26);
        v_av42 = _mm512_add_ps(v_av37, v_av40);
        v_av42 = SWAP_RI_512_S(CONJ_512_S(v_av42));
        // Output point 14 : X[13]
        v_out13 = _mm512_add_ps(v_av41, v_av42);
        // Output point 3 : X[2]
        v_out2 = _mm512_sub_ps(v_av41, v_av42);

        v_av44 = _mm512_sub_ps(v_av19, v_tv11);
        v_tv23 = _mm512_mul_ps(v_C3, v_av25);
        v_tv24 = _mm512_mul_ps(v_C4, v_av22);
        v_av45 = _mm512_sub_ps(v_tv23, v_tv24);
        v_av46 = _mm512_sub_ps(v_av35, v_tv20);
        v_tv25 = _mm512_mul_ps(v_C3, v_av39);
        v_tv27 = _mm512_mul_ps(v_C4, v_av38);
        v_av47 = _mm512_sub_ps(v_tv25, v_tv27);

        v_av41 = _mm512_add_ps(v_av44, v_av45);
        v_av42 = _mm512_add_ps(v_av46, v_av47);
        v_av42 = SWAP_RI_512_S(CONJ_512_S(v_av42));
        // Output point 5 : X[4]
        v_out4 = _mm512_add_ps(v_av41, v_av42);
        // Output point 12 : X[11]
        v_out11 = _mm512_sub_ps(v_av41, v_av42);

        v_av48 = _mm512_sub_ps(v_av44, v_av45);
        v_av49 = _mm512_sub_ps(v_av47, v_av46);
        v_av49 = SWAP_RI_512_S(CONJ_512_S(v_av49));
        // Output point 15 : X[14]
        v_out14 = _mm512_add_ps(v_av48, v_av49);
        // Output point 2 : X[1]
        v_out1 = _mm512_sub_ps(v_av48, v_av49);

        v_av48 = _mm512_sub_ps(v_av21, v_av26);
        v_av49 = _mm512_sub_ps(v_av40, v_av37);
        v_av49 = SWAP_RI_512_S(CONJ_512_S(v_av49));
        // Output point 15 : X[14]
        v_out8 = _mm512_add_ps(v_av48, v_av49);
        // Output point 2 : X[1]
        v_out7 = _mm512_sub_ps(v_av48, v_av49);

        if (out_strides[1] == DATA_STRIDE)
        {
            TRANSPOSE_ST8_512_S(curr_out, v_out_stride, v_out0, v_out1, v_out2,
                                v_out3, v_out4, v_out5, v_out6, v_out7);
            curr_out = curr_out + NUM_SETS_512_S * DATA_STRIDE;
            TRANSPOSE_ST4_512_S(curr_out, v_out_stride, v_out8, v_out9, v_out10,
                                v_out11);
            curr_out = curr_out + 4 * DATA_STRIDE;
            TRANSPOSE_ST2_512_S(curr_out, v_out_stride, v_out12, v_out13);
            curr_out = curr_out + 2 * DATA_STRIDE;
            SCATTER8_512_S_STRIDED(curr_out, v_out_stride, v_out14);
        }
        else
        {
            SCATTER8_512_S(curr_out, v_out_stride, v_out0, is_contiguous_out);
            curr_out = out_r + out_strides[1];
            SCATTER8_512_S(curr_out, v_out_stride, v_out1, is_contiguous_out);
            curr_out = out_r + out_strides[2];
            SCATTER8_512_S(curr_out, v_out_stride, v_out2, is_contiguous_out);
            curr_out = out_r + out_strides[3];
            SCATTER8_512_S(curr_out, v_out_stride, v_out3, is_contiguous_out);
            curr_out = out_r + out_strides[4];
            SCATTER8_512_S(curr_out, v_out_stride, v_out4, is_contiguous_out);
            curr_out = out_r + out_strides[5];
            SCATTER8_512_S(curr_out, v_out_stride, v_out5, is_contiguous_out);
            curr_out = out_r + out_strides[6];
            SCATTER8_512_S(curr_out, v_out_stride, v_out6, is_contiguous_out);
            curr_out = out_r + out_strides[7];
            SCATTER8_512_S(curr_out, v_out_stride, v_out7, is_contiguous_out);
            curr_out = out_r + out_strides[8];
            SCATTER8_512_S(curr_out, v_out_stride, v_out8, is_contiguous_out);
            curr_out = out_r + out_strides[9];
            SCATTER8_512_S(curr_out, v_out_stride, v_out9, is_contiguous_out);
            curr_out = out_r + out_strides[10];
            SCATTER8_512_S(curr_out, v_out_stride, v_out10, is_contiguous_out);
            curr_out = out_r + out_strides[11];
            SCATTER8_512_S(curr_out, v_out_stride, v_out11, is_contiguous_out);
            curr_out = out_r + out_strides[12];
            SCATTER8_512_S(curr_out, v_out_stride, v_out12, is_contiguous_out);
            curr_out = out_r + out_strides[13];
            SCATTER8_512_S(curr_out, v_out_stride, v_out13, is_contiguous_out);
            curr_out = out_r + out_strides[14];
            SCATTER8_512_S(curr_out, v_out_stride, v_out14, is_contiguous_out);
        }

        in_r += NUM_SETS_512_S * v_in_stride;
        out_r += NUM_SETS_512_S * v_out_stride;
    }
    // tail cases
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

        __m256 v_K1 = _mm512_castps512_ps256(v_C1);
        __m256 v_K2 = _mm512_castps512_ps256(v_C2);
        __m256 v_K3 = _mm512_castps512_ps256(v_C3);
        __m256 v_K4 = _mm512_castps512_ps256(v_C4);
        __m256 v_K5 = _mm512_castps512_ps256(v_C5);
        __m256 v_K6 = _mm512_castps512_ps256(v_C6);

        curr_in = in_r;
        curr_out = out_r;

        GATHER4_256_S(curr_in, v_in_stride, v_in0, is_contiguous_in);
        curr_in = in_r + in_strides[1];
        GATHER4_256_S(curr_in, v_in_stride, v_in1, is_contiguous_in);
        curr_in = in_r + in_strides[2];
        GATHER4_256_S(curr_in, v_in_stride, v_in2, is_contiguous_in);
        curr_in = in_r + in_strides[3];
        GATHER4_256_S(curr_in, v_in_stride, v_in3, is_contiguous_in);
        curr_in = in_r + in_strides[4];
        GATHER4_256_S(curr_in, v_in_stride, v_in4, is_contiguous_in);
        curr_in = in_r + in_strides[5];
        GATHER4_256_S(curr_in, v_in_stride, v_in5, is_contiguous_in);
        curr_in = in_r + in_strides[6];
        GATHER4_256_S(curr_in, v_in_stride, v_in6, is_contiguous_in);
        curr_in = in_r + in_strides[7];
        GATHER4_256_S(curr_in, v_in_stride, v_in7, is_contiguous_in);
        curr_in = in_r + in_strides[8];
        GATHER4_256_S(curr_in, v_in_stride, v_in8, is_contiguous_in);
        curr_in = in_r + in_strides[9];
        GATHER4_256_S(curr_in, v_in_stride, v_in9, is_contiguous_in);
        curr_in = in_r + in_strides[10];
        GATHER4_256_S(curr_in, v_in_stride, v_in10, is_contiguous_in);
        curr_in = in_r + in_strides[11];
        GATHER4_256_S(curr_in, v_in_stride, v_in11, is_contiguous_in);
        curr_in = in_r + in_strides[12];
        GATHER4_256_S(curr_in, v_in_stride, v_in12, is_contiguous_in);
        curr_in = in_r + in_strides[13];
        GATHER4_256_S(curr_in, v_in_stride, v_in13, is_contiguous_in);
        curr_in = in_r + in_strides[14];
        GATHER4_256_S(curr_in, v_in_stride, v_in14, is_contiguous_in);

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

        if (out_strides[1] == DATA_STRIDE)
        {
            TRANSPOSE_ST4_256_S(curr_out, v_out_stride, v_out0, v_out1, v_out2,
                                v_out3);
            curr_out = curr_out + NUM_SETS_256_S * DATA_STRIDE;
            TRANSPOSE_ST4_256_S(curr_out, v_out_stride, v_out4, v_out5, v_out6,
                                v_out7);
            curr_out = curr_out + NUM_SETS_256_S * DATA_STRIDE;
            TRANSPOSE_ST4_256_S(curr_out, v_out_stride, v_out8, v_out9, v_out10,
                                v_out11);
            curr_out = curr_out + NUM_SETS_256_S * DATA_STRIDE;
            TRANSPOSE_ST2_256_S(curr_out, v_out_stride, v_out12, v_out13);
            curr_out = curr_out + 2 * DATA_STRIDE;
            SCATTER4_256_S_STRIDED(curr_out, v_out_stride, v_out14);
        }
        else
        {
            SCATTER4_256_S(curr_out, v_out_stride, v_out0, is_contiguous_out);
            curr_out = out_r + out_strides[1];
            SCATTER4_256_S(curr_out, v_out_stride, v_out1, is_contiguous_out);
            curr_out = out_r + out_strides[2];
            SCATTER4_256_S(curr_out, v_out_stride, v_out2, is_contiguous_out);
            curr_out = out_r + out_strides[3];
            SCATTER4_256_S(curr_out, v_out_stride, v_out3, is_contiguous_out);
            curr_out = out_r + out_strides[4];
            SCATTER4_256_S(curr_out, v_out_stride, v_out4, is_contiguous_out);
            curr_out = out_r + out_strides[5];
            SCATTER4_256_S(curr_out, v_out_stride, v_out5, is_contiguous_out);
            curr_out = out_r + out_strides[6];
            SCATTER4_256_S(curr_out, v_out_stride, v_out6, is_contiguous_out);
            curr_out = out_r + out_strides[7];
            SCATTER4_256_S(curr_out, v_out_stride, v_out7, is_contiguous_out);
            curr_out = out_r + out_strides[8];
            SCATTER4_256_S(curr_out, v_out_stride, v_out8, is_contiguous_out);
            curr_out = out_r + out_strides[9];
            SCATTER4_256_S(curr_out, v_out_stride, v_out9, is_contiguous_out);
            curr_out = out_r + out_strides[10];
            SCATTER4_256_S(curr_out, v_out_stride, v_out10, is_contiguous_out);
            curr_out = out_r + out_strides[11];
            SCATTER4_256_S(curr_out, v_out_stride, v_out11, is_contiguous_out);
            curr_out = out_r + out_strides[12];
            SCATTER4_256_S(curr_out, v_out_stride, v_out12, is_contiguous_out);
            curr_out = out_r + out_strides[13];
            SCATTER4_256_S(curr_out, v_out_stride, v_out13, is_contiguous_out);
            curr_out = out_r + out_strides[14];
            SCATTER4_256_S(curr_out, v_out_stride, v_out14, is_contiguous_out);
        }

        in_r += NUM_SETS_256_S * v_in_stride;
        out_r += NUM_SETS_256_S * v_out_stride;
        remaining_sets = remaining_sets % NUM_SETS_256_S;
    }
    // tail cases
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

        __m128 v_K1 = _mm512_castps512_ps128(v_C1);
        __m128 v_K2 = _mm512_castps512_ps128(v_C2);
        __m128 v_K3 = _mm512_castps512_ps128(v_C3);
        __m128 v_K4 = _mm512_castps512_ps128(v_C4);
        __m128 v_K5 = _mm512_castps512_ps128(v_C5);
        __m128 v_K6 = _mm512_castps512_ps128(v_C6);

        curr_in = in_r;
        curr_out = out_r;

        GATHER2_128_S(curr_in, v_in_stride, v_in0, is_contiguous_in);
        curr_in = in_r + in_strides[1];
        GATHER2_128_S(curr_in, v_in_stride, v_in1, is_contiguous_in);
        curr_in = in_r + in_strides[2];
        GATHER2_128_S(curr_in, v_in_stride, v_in2, is_contiguous_in);
        curr_in = in_r + in_strides[3];
        GATHER2_128_S(curr_in, v_in_stride, v_in3, is_contiguous_in);
        curr_in = in_r + in_strides[4];
        GATHER2_128_S(curr_in, v_in_stride, v_in4, is_contiguous_in);
        curr_in = in_r + in_strides[5];
        GATHER2_128_S(curr_in, v_in_stride, v_in5, is_contiguous_in);
        curr_in = in_r + in_strides[6];
        GATHER2_128_S(curr_in, v_in_stride, v_in6, is_contiguous_in);
        curr_in = in_r + in_strides[7];
        GATHER2_128_S(curr_in, v_in_stride, v_in7, is_contiguous_in);
        curr_in = in_r + in_strides[8];
        GATHER2_128_S(curr_in, v_in_stride, v_in8, is_contiguous_in);
        curr_in = in_r + in_strides[9];
        GATHER2_128_S(curr_in, v_in_stride, v_in9, is_contiguous_in);
        curr_in = in_r + in_strides[10];
        GATHER2_128_S(curr_in, v_in_stride, v_in10, is_contiguous_in);
        curr_in = in_r + in_strides[11];
        GATHER2_128_S(curr_in, v_in_stride, v_in11, is_contiguous_in);
        curr_in = in_r + in_strides[12];
        GATHER2_128_S(curr_in, v_in_stride, v_in12, is_contiguous_in);
        curr_in = in_r + in_strides[13];
        GATHER2_128_S(curr_in, v_in_stride, v_in13, is_contiguous_in);
        curr_in = in_r + in_strides[14];
        GATHER2_128_S(curr_in, v_in_stride, v_in14, is_contiguous_in);

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

        if (out_strides[1] == DATA_STRIDE)
        {
            TRANSPOSE_ST2_128_S(curr_out, v_out_stride, v_out0, v_out1);
            curr_out = curr_out + NUM_SETS_128_S * DATA_STRIDE;
            TRANSPOSE_ST2_128_S(curr_out, v_out_stride, v_out2, v_out3);
            curr_out = curr_out + NUM_SETS_128_S * DATA_STRIDE;
            TRANSPOSE_ST2_128_S(curr_out, v_out_stride, v_out4, v_out5);
            curr_out = curr_out + NUM_SETS_128_S * DATA_STRIDE;
            TRANSPOSE_ST2_128_S(curr_out, v_out_stride, v_out6, v_out7);
            curr_out = curr_out + NUM_SETS_128_S * DATA_STRIDE;
            TRANSPOSE_ST2_128_S(curr_out, v_out_stride, v_out8, v_out9);
            curr_out = curr_out + NUM_SETS_128_S * DATA_STRIDE;
            TRANSPOSE_ST2_128_S(curr_out, v_out_stride, v_out10, v_out11);
            curr_out = curr_out + NUM_SETS_128_S * DATA_STRIDE;
            TRANSPOSE_ST2_128_S(curr_out, v_out_stride, v_out12, v_out13);
            curr_out = curr_out + NUM_SETS_128_S * DATA_STRIDE;
            SCATTER2_128_S_STRIDED(curr_out, v_out_stride, v_out14);
        }
        else
        {
            SCATTER2_128_S(curr_out, v_out_stride, v_out0, is_contiguous_out);
            curr_out = out_r + out_strides[1];
            SCATTER2_128_S(curr_out, v_out_stride, v_out1, is_contiguous_out);
            curr_out = out_r + out_strides[2];
            SCATTER2_128_S(curr_out, v_out_stride, v_out2, is_contiguous_out);
            curr_out = out_r + out_strides[3];
            SCATTER2_128_S(curr_out, v_out_stride, v_out3, is_contiguous_out);
            curr_out = out_r + out_strides[4];
            SCATTER2_128_S(curr_out, v_out_stride, v_out4, is_contiguous_out);
            curr_out = out_r + out_strides[5];
            SCATTER2_128_S(curr_out, v_out_stride, v_out5, is_contiguous_out);
            curr_out = out_r + out_strides[6];
            SCATTER2_128_S(curr_out, v_out_stride, v_out6, is_contiguous_out);
            curr_out = out_r + out_strides[7];
            SCATTER2_128_S(curr_out, v_out_stride, v_out7, is_contiguous_out);
            curr_out = out_r + out_strides[8];
            SCATTER2_128_S(curr_out, v_out_stride, v_out8, is_contiguous_out);
            curr_out = out_r + out_strides[9];
            SCATTER2_128_S(curr_out, v_out_stride, v_out9, is_contiguous_out);
            curr_out = out_r + out_strides[10];
            SCATTER2_128_S(curr_out, v_out_stride, v_out10, is_contiguous_out);
            curr_out = out_r + out_strides[11];
            SCATTER2_128_S(curr_out, v_out_stride, v_out11, is_contiguous_out);
            curr_out = out_r + out_strides[12];
            SCATTER2_128_S(curr_out, v_out_stride, v_out12, is_contiguous_out);
            curr_out = out_r + out_strides[13];
            SCATTER2_128_S(curr_out, v_out_stride, v_out13, is_contiguous_out);
            curr_out = out_r + out_strides[14];
            SCATTER2_128_S(curr_out, v_out_stride, v_out14, is_contiguous_out);
        }

        in_r = in_r + (v_in_stride << 1);
        out_r = out_r + (v_out_stride << 1);
    }
    // tail case
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

        __m128 v_K1 = _mm512_castps512_ps128(v_C1);
        __m128 v_K2 = _mm512_castps512_ps128(v_C2);
        __m128 v_K3 = _mm512_castps512_ps128(v_C3);
        __m128 v_K4 = _mm512_castps512_ps128(v_C4);
        __m128 v_K5 = _mm512_castps512_ps128(v_C5);
        __m128 v_K6 = _mm512_castps512_ps128(v_C6);

        curr_in = in_r;
        curr_out = out_r;

        LD_LOW_128_S(curr_in, v_in0);
        curr_in = in_r + in_strides[1];
        LD_LOW_128_S(curr_in, v_in1);
        curr_in = in_r + in_strides[2];
        LD_LOW_128_S(curr_in, v_in2);
        curr_in = in_r + in_strides[3];
        LD_LOW_128_S(curr_in, v_in3);
        curr_in = in_r + in_strides[4];
        LD_LOW_128_S(curr_in, v_in4);
        curr_in = in_r + in_strides[5];
        LD_LOW_128_S(curr_in, v_in5);
        curr_in = in_r + in_strides[6];
        LD_LOW_128_S(curr_in, v_in6);
        curr_in = in_r + in_strides[7];
        LD_LOW_128_S(curr_in, v_in7);
        curr_in = in_r + in_strides[8];
        LD_LOW_128_S(curr_in, v_in8);
        curr_in = in_r + in_strides[9];
        LD_LOW_128_S(curr_in, v_in9);
        curr_in = in_r + in_strides[10];
        LD_LOW_128_S(curr_in, v_in10);
        curr_in = in_r + in_strides[11];
        LD_LOW_128_S(curr_in, v_in11);
        curr_in = in_r + in_strides[12];
        LD_LOW_128_S(curr_in, v_in12);
        curr_in = in_r + in_strides[13];
        LD_LOW_128_S(curr_in, v_in13);
        curr_in = in_r + in_strides[14];
        LD_LOW_128_S(curr_in, v_in14);

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

        if (out_strides[1] == DATA_STRIDE)
        {
            ST_128_S(curr_out, v_out0, v_out1);
            curr_out = out_r + 2 * DATA_STRIDE;
            ST_128_S(curr_out, v_out2, v_out3);
            curr_out = out_r + 4 * DATA_STRIDE;
            ST_128_S(curr_out, v_out4, v_out5);
            curr_out = out_r + 6 * DATA_STRIDE;
            ST_128_S(curr_out, v_out6, v_out7);
            curr_out = out_r + 8 * DATA_STRIDE;
            ST_128_S(curr_out, v_out8, v_out9);
            curr_out = out_r + 10 * DATA_STRIDE;
            ST_128_S(curr_out, v_out10, v_out11);
            curr_out = out_r + 12 * DATA_STRIDE;
            ST_128_S(curr_out, v_out12, v_out13);
            curr_out = out_r + 14 * DATA_STRIDE;
            ST_LOW_128_S(curr_out, v_out14);
        }
        else
        {
            ST_LOW_128_S(curr_out, v_out0);
            curr_out = out_r + out_strides[1];
            ST_LOW_128_S(curr_out, v_out1);
            curr_out = out_r + out_strides[2];
            ST_LOW_128_S(curr_out, v_out2);
            curr_out = out_r + out_strides[3];
            ST_LOW_128_S(curr_out, v_out3);
            curr_out = out_r + out_strides[4];
            ST_LOW_128_S(curr_out, v_out4);
            curr_out = out_r + out_strides[5];
            ST_LOW_128_S(curr_out, v_out5);
            curr_out = out_r + out_strides[6];
            ST_LOW_128_S(curr_out, v_out6);
            curr_out = out_r + out_strides[7];
            ST_LOW_128_S(curr_out, v_out7);
            curr_out = out_r + out_strides[8];
            ST_LOW_128_S(curr_out, v_out8);
            curr_out = out_r + out_strides[9];
            ST_LOW_128_S(curr_out, v_out9);
            curr_out = out_r + out_strides[10];
            ST_LOW_128_S(curr_out, v_out10);
            curr_out = out_r + out_strides[11];
            ST_LOW_128_S(curr_out, v_out11);
            curr_out = out_r + out_strides[12];
            ST_LOW_128_S(curr_out, v_out12);
            curr_out = out_r + out_strides[13];
            ST_LOW_128_S(curr_out, v_out13);
            curr_out = out_r + out_strides[14];
            ST_LOW_128_S(curr_out, v_out14);
        }
    }
    AOCLFFTZ_LOG(DEBUG, global_logger_mode, "Exit");
}

static VOID fft15avx512fp64(VOID *in_real, VOID *in_imag, VOID *out_real,
                            VOID *out_imag, INTP n, aoclfftz_strides_t *strides,
                            VOID *twd, UINT8 flag)
{
    AOCLFFTZ_LOG(DEBUG, global_logger_mode, "Enter");
    const DOUBLE CRTM_15[6] = {
        0.55901699437494742410229341718281905886015458990288,
        0.25000000000000000000000000000000000000000000000000,
        0.95105651629515357211643933337938214340569863400000,
        0.58778525229247301629891039327884007596190389052978,
        0.50000000000000000000000000000000000000000000000000,
        0.86602540378443864676372317075293618347140262690519};

    DOUBLE *in_r = (DOUBLE *)in_real;
    DOUBLE *out_r = (DOUBLE *)out_real;
    DOUBLE *curr_in, *curr_out;

#ifdef VOLATILE_STRIDE_ARRAY
    volatile INTP *in_strides = strides->in_strides;
    volatile INTP *out_strides = strides->out_strides;
#else
    INTP *in_strides = strides->in_strides;
    INTP *out_strides = strides->out_strides;
#endif
    INTP v_in_stride = strides->v_in_stride;
    UINT8 is_contiguous_in = (v_in_stride == DATA_STRIDE);
    INTP v_out_stride = strides->v_out_stride;
    UINT8 is_contiguous_out = (v_out_stride == DATA_STRIDE);

    INTP N = n / NUM_SETS_512_D;
    INTP remaining_sets = n % NUM_SETS_512_D;
    INTP count;

    __m512d v_C1 = _mm512_set1_pd(CRTM_15[0]);
    __m512d v_C2 = _mm512_set1_pd(CRTM_15[1]);
    __m512d v_C3 = _mm512_set1_pd(CRTM_15[2]);
    v_C3 = _mm512_xor_pd(v_C3, _neg_512_d[flag].d);
    __m512d v_C4 = _mm512_set1_pd(CRTM_15[3]);
    v_C4 = _mm512_xor_pd(v_C4, _neg_512_d[flag].d);
    __m512d v_C5 = _mm512_set1_pd(CRTM_15[4]);
    __m512d v_C6 = _mm512_set1_pd(CRTM_15[5]);
    v_C6 = _mm512_xor_pd(v_C6, _neg_512_d[flag].d);

    for (count = 0; count < N; count++)
    {
        __m512d v_in0, v_in1, v_in2, v_in3, v_in4, v_in5, v_in6, v_in7, v_in8;
        __m512d v_in9, v_in10, v_in11, v_in12, v_in13, v_in14;
        __m512d v_cv1, v_cv2, v_cv3, v_cv4, v_cv5, v_cv6, v_cv7, v_cv8, v_cv9,
                v_cv10;
        __m512d v_cv11, v_cv12;
        __m512d v_av1, v_av2, v_av3, v_av4, v_av5, v_av6, v_av7, v_av8, v_av9;
        __m512d v_av11, v_av12, v_av13, v_av14, v_av15, v_av16, v_av17;
        __m512d v_av18, v_av19, v_av20, v_av21, v_av22, v_av23, v_av24, v_av25;
        __m512d v_av26, v_av27, v_av28, v_av29, v_av30, v_av31, v_av32, v_av33;
        __m512d v_av34, v_av35, v_av36, v_av37, v_av38, v_av39, v_av40, v_av41;
        __m512d v_av42, v_av44, v_av45, v_av46, v_av47, v_av48, v_av49;
        __m512d v_tv1, v_tv2, v_tv3, v_tv4, v_tv5, v_tv6, v_tv7, v_tv8, v_tv9,
                v_tv10;
        __m512d v_tv11, v_tv16, v_tv17, v_tv18;
        __m512d v_tv19, v_tv20, v_tv21, v_tv22, v_tv23, v_tv24, v_tv25, v_tv27;
        __m512d v_out0, v_out1, v_out2, v_out3, v_out4, v_out5;
        __m512d v_out6, v_out7, v_out8, v_out9, v_out10, v_out11;
        __m512d v_out12, v_out13, v_out14;

        curr_in = in_r;
        curr_out = out_r;

        GATHER4_512_D(curr_in, v_in_stride, v_in0, is_contiguous_in);
        curr_in = in_r + in_strides[1];
        GATHER4_512_D(curr_in, v_in_stride, v_in1, is_contiguous_in);
        curr_in = in_r + in_strides[2];
        GATHER4_512_D(curr_in, v_in_stride, v_in2, is_contiguous_in);
        curr_in = in_r + in_strides[3];
        GATHER4_512_D(curr_in, v_in_stride, v_in3, is_contiguous_in);
        curr_in = in_r + in_strides[4];
        GATHER4_512_D(curr_in, v_in_stride, v_in4, is_contiguous_in);
        curr_in = in_r + in_strides[5];
        GATHER4_512_D(curr_in, v_in_stride, v_in5, is_contiguous_in);
        curr_in = in_r + in_strides[6];
        GATHER4_512_D(curr_in, v_in_stride, v_in6, is_contiguous_in);
        curr_in = in_r + in_strides[7];
        GATHER4_512_D(curr_in, v_in_stride, v_in7, is_contiguous_in);
        curr_in = in_r + in_strides[8];
        GATHER4_512_D(curr_in, v_in_stride, v_in8, is_contiguous_in);
        curr_in = in_r + in_strides[9];
        GATHER4_512_D(curr_in, v_in_stride, v_in9, is_contiguous_in);
        curr_in = in_r + in_strides[10];
        GATHER4_512_D(curr_in, v_in_stride, v_in10, is_contiguous_in);
        curr_in = in_r + in_strides[11];
        GATHER4_512_D(curr_in, v_in_stride, v_in11, is_contiguous_in);
        curr_in = in_r + in_strides[12];
        GATHER4_512_D(curr_in, v_in_stride, v_in12, is_contiguous_in);
        curr_in = in_r + in_strides[13];
        GATHER4_512_D(curr_in, v_in_stride, v_in13, is_contiguous_in);
        curr_in = in_r + in_strides[14];
        GATHER4_512_D(curr_in, v_in_stride, v_in14, is_contiguous_in);

        // common calculations
        v_cv1 = _mm512_add_pd(v_in10, v_in5);
        v_cv2 = _mm512_add_pd(v_in0, v_cv1);
        v_cv3 = _mm512_add_pd(v_in14, v_in4);
        v_cv4 = _mm512_add_pd(v_in9, v_cv3);
        v_cv5 = _mm512_add_pd(v_in11, v_in1);
        v_cv6 = _mm512_add_pd(v_in6, v_cv5);
        v_av1 = _mm512_add_pd(v_cv4, v_cv6);
        v_cv7 = _mm512_add_pd(v_in2, v_in7);
        v_cv8 = _mm512_add_pd(v_in12, v_cv7);
        v_cv9 = _mm512_add_pd(v_in13, v_in8);
        v_cv10 = _mm512_add_pd(v_in3, v_cv9);
        v_av2 = _mm512_add_pd(v_cv8, v_cv10);
        v_av3 = _mm512_add_pd(v_av1, v_av2);

        // Output point 1 : X[0]
        v_out0 = _mm512_add_pd(v_cv2, v_av3);

        v_tv1 = _mm512_mul_pd(v_C2, v_av3);
        v_av4 = _mm512_sub_pd(v_cv2, v_tv1);
        v_av5 = _mm512_sub_pd(v_av1, v_av2);
        v_tv2 = _mm512_mul_pd(v_C1, v_av5);
        v_av6 = _mm512_add_pd(v_av4, v_tv2);
        v_av7 = _mm512_sub_pd(v_cv6, v_cv4);
        v_av8 = _mm512_sub_pd(v_cv8, v_cv10);
        v_tv3 = _mm512_mul_pd(v_C3, v_av7);
        v_tv4 = _mm512_mul_pd(v_C4, v_av8);
        v_av9 = _mm512_add_pd(v_tv3, v_tv4);

        v_av9 = SWAP_RI_512_D(CONJ_512_D(v_av9));
        // Output point 13 : X[12]
        v_out12 = _mm512_add_pd(v_av6, v_av9);
        // Output point 4 : X[3]
        v_out3 = _mm512_sub_pd(v_av6, v_av9);

        v_av6 = _mm512_sub_pd(v_av4, v_tv2);
        v_tv3 = _mm512_mul_pd(v_C3, v_av8);
        v_tv4 = _mm512_mul_pd(v_C4, v_av7);
        v_av9 = _mm512_sub_pd(v_tv4, v_tv3);

        v_av9 = SWAP_RI_512_D(CONJ_512_D(v_av9));
        // Output point 10 : X[9]
        v_out9 = _mm512_add_pd(v_av6, v_av9);
        // Output point 7 : X[6]
        v_out6 = _mm512_sub_pd(v_av6, v_av9);

        v_tv5 = _mm512_mul_pd(v_C5, v_cv1);
        v_av11 = _mm512_sub_pd(v_in0, v_tv5);

        v_tv6 = _mm512_mul_pd(v_C5, v_cv5);
        v_av12 = _mm512_sub_pd(v_in6, v_tv6);
        v_tv7 = _mm512_mul_pd(v_C5, v_cv3);
        v_av13 = _mm512_sub_pd(v_in9, v_tv7);
        v_av14 = _mm512_add_pd(v_av12, v_av13);
        v_tv8 = _mm512_mul_pd(v_C5, v_cv7);
        v_av15 = _mm512_sub_pd(v_in12, v_tv8);
        v_tv9 = _mm512_mul_pd(v_C5, v_cv9);
        v_av16 = _mm512_sub_pd(v_in3, v_tv9);
        v_av17 = _mm512_add_pd(v_av15, v_av16);
        v_av18 = _mm512_add_pd(v_av14, v_av17);
        v_tv10 = _mm512_mul_pd(v_C2, v_av18);
        v_av19 = _mm512_sub_pd(v_av11, v_tv10);
        v_av20 = _mm512_sub_pd(v_av14, v_av17);
        v_tv11 = _mm512_mul_pd(v_C1, v_av20);
        v_av21 = _mm512_add_pd(v_av19, v_tv11);
        v_cv11 = _mm512_sub_pd(v_in14, v_in4);
        v_av29 = _mm512_mul_pd(v_C6, v_cv11);
        v_cv12 = _mm512_sub_pd(v_in11, v_in1);
        v_av28 = _mm512_mul_pd(v_C6, v_cv12);
        v_av22 = _mm512_sub_pd(v_av29, v_av28);
        v_av23 = _mm512_sub_pd(v_in8, v_in13);
        v_av32 = _mm512_mul_pd(v_C6, v_av23);
        v_av24 = _mm512_sub_pd(v_in2, v_in7);
        v_av31 = _mm512_mul_pd(v_C6, v_av24);
        v_av25 = _mm512_sub_pd(v_av32, v_av31);
        v_tv16 = _mm512_mul_pd(v_C3, v_av22);
        v_tv17 = _mm512_mul_pd(v_C4, v_av25);
        v_av26 = _mm512_add_pd(v_tv16, v_tv17);
        v_av27 = _mm512_sub_pd(v_in10, v_in5);
        v_tv18 = _mm512_mul_pd(v_C6, v_av27);
        v_av30 = _mm512_add_pd(v_av28, v_av29);
        v_av33 = _mm512_add_pd(v_av31, v_av32);
        v_av34 = _mm512_add_pd(v_av30, v_av33);
        v_tv19 = _mm512_mul_pd(v_C2, v_av34);
        v_av35 = _mm512_add_pd(v_tv18, v_tv19);
        v_av36 = _mm512_sub_pd(v_av33, v_av30);
        v_tv20 = _mm512_mul_pd(v_C1, v_av36);
        v_av37 = _mm512_add_pd(v_av35, v_tv20);
        v_av38 = _mm512_sub_pd(v_av13, v_av12);
        v_av39 = _mm512_sub_pd(v_av16, v_av15);
        v_tv21 = _mm512_mul_pd(v_C3, v_av38);
        v_tv22 = _mm512_mul_pd(v_C4, v_av39);
        v_av40 = _mm512_add_pd(v_tv21, v_tv22);

        v_av41 = _mm512_add_pd(v_av11, v_av18);
        v_av42 = _mm512_sub_pd(v_tv18, v_av34);
        v_av42 = SWAP_RI_512_D(CONJ_512_D(v_av42));
        // Output point 11 : X[10]
        v_out10 = _mm512_add_pd(v_av41, v_av42);
        // Output point 6 : X[5]
        v_out5 = _mm512_sub_pd(v_av41, v_av42);

        v_av41 = _mm512_add_pd(v_av21, v_av26);
        v_av42 = _mm512_add_pd(v_av37, v_av40);
        v_av42 = SWAP_RI_512_D(CONJ_512_D(v_av42));
        // Output point 14 : X[13]
        v_out13 = _mm512_add_pd(v_av41, v_av42);
        // Output point 3 : X[2]
        v_out2 = _mm512_sub_pd(v_av41, v_av42);

        v_av44 = _mm512_sub_pd(v_av19, v_tv11);
        v_tv23 = _mm512_mul_pd(v_C3, v_av25);
        v_tv24 = _mm512_mul_pd(v_C4, v_av22);
        v_av45 = _mm512_sub_pd(v_tv23, v_tv24);
        v_av46 = _mm512_sub_pd(v_av35, v_tv20);
        v_tv25 = _mm512_mul_pd(v_C3, v_av39);
        v_tv27 = _mm512_mul_pd(v_C4, v_av38);
        v_av47 = _mm512_sub_pd(v_tv25, v_tv27);

        v_av41 = _mm512_add_pd(v_av44, v_av45);
        v_av42 = _mm512_add_pd(v_av46, v_av47);
        v_av42 = SWAP_RI_512_D(CONJ_512_D(v_av42));
        // Output point 5 : X[4]
        v_out4 = _mm512_add_pd(v_av41, v_av42);
        // Output point 12 : X[11]
        v_out11 = _mm512_sub_pd(v_av41, v_av42);

        v_av48 = _mm512_sub_pd(v_av44, v_av45);
        v_av49 = _mm512_sub_pd(v_av47, v_av46);
        v_av49 = SWAP_RI_512_D(CONJ_512_D(v_av49));
        // Output point 15 : X[14]
        v_out14 = _mm512_add_pd(v_av48, v_av49);
        // Output point 2 : X[1]
        v_out1 = _mm512_sub_pd(v_av48, v_av49);

        v_av48 = _mm512_sub_pd(v_av21, v_av26);
        v_av49 = _mm512_sub_pd(v_av40, v_av37);
        v_av49 = SWAP_RI_512_D(CONJ_512_D(v_av49));
        // Output point 15 : X[14]
        v_out8 = _mm512_add_pd(v_av48, v_av49);
        // Output point 2 : X[1]
        v_out7 = _mm512_sub_pd(v_av48, v_av49);

        if (out_strides[1] == DATA_STRIDE)
        {
            TRANSPOSE_ST4_512_D(curr_out, v_out_stride, v_out0, v_out1, v_out2,
                                v_out3);
            curr_out = curr_out + NUM_SETS_512_D * DATA_STRIDE;
            TRANSPOSE_ST4_512_D(curr_out, v_out_stride, v_out4, v_out5, v_out6,
                                v_out7);
            curr_out = curr_out + NUM_SETS_512_D * DATA_STRIDE;
            TRANSPOSE_ST4_512_D(curr_out, v_out_stride, v_out8, v_out9, v_out10,
                                v_out11);
            curr_out = curr_out + NUM_SETS_512_D * DATA_STRIDE;
            TRANSPOSE_ST2_512_D(curr_out, v_out_stride, v_out12, v_out13);
            curr_out = curr_out + 2 * DATA_STRIDE;
            SCATTER4_512_D_STRIDED(curr_out, v_out_stride, v_out14);
        }
        else
        {
            SCATTER4_512_D(curr_out, v_out_stride, v_out0, is_contiguous_out);
            curr_out = out_r + out_strides[1];
            SCATTER4_512_D(curr_out, v_out_stride, v_out1, is_contiguous_out);
            curr_out = out_r + out_strides[2];
            SCATTER4_512_D(curr_out, v_out_stride, v_out2, is_contiguous_out);
            curr_out = out_r + out_strides[3];
            SCATTER4_512_D(curr_out, v_out_stride, v_out3, is_contiguous_out);
            curr_out = out_r + out_strides[4];
            SCATTER4_512_D(curr_out, v_out_stride, v_out4, is_contiguous_out);
            curr_out = out_r + out_strides[5];
            SCATTER4_512_D(curr_out, v_out_stride, v_out5, is_contiguous_out);
            curr_out = out_r + out_strides[6];
            SCATTER4_512_D(curr_out, v_out_stride, v_out6, is_contiguous_out);
            curr_out = out_r + out_strides[7];
            SCATTER4_512_D(curr_out, v_out_stride, v_out7, is_contiguous_out);
            curr_out = out_r + out_strides[8];
            SCATTER4_512_D(curr_out, v_out_stride, v_out8, is_contiguous_out);
            curr_out = out_r + out_strides[9];
            SCATTER4_512_D(curr_out, v_out_stride, v_out9, is_contiguous_out);
            curr_out = out_r + out_strides[10];
            SCATTER4_512_D(curr_out, v_out_stride, v_out10, is_contiguous_out);
            curr_out = out_r + out_strides[11];
            SCATTER4_512_D(curr_out, v_out_stride, v_out11, is_contiguous_out);
            curr_out = out_r + out_strides[12];
            SCATTER4_512_D(curr_out, v_out_stride, v_out12, is_contiguous_out);
            curr_out = out_r + out_strides[13];
            SCATTER4_512_D(curr_out, v_out_stride, v_out13, is_contiguous_out);
            curr_out = out_r + out_strides[14];
            SCATTER4_512_D(curr_out, v_out_stride, v_out14, is_contiguous_out);
        }

        in_r += NUM_SETS_512_D * v_in_stride;
        out_r += NUM_SETS_512_D * v_out_stride;
    }
    // tail cases
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

        __m256d v_K1 = _mm512_castpd512_pd256(v_C1);
        __m256d v_K2 = _mm512_castpd512_pd256(v_C2);
        __m256d v_K3 = _mm512_castpd512_pd256(v_C3);
        __m256d v_K4 = _mm512_castpd512_pd256(v_C4);
        __m256d v_K5 = _mm512_castpd512_pd256(v_C5);
        __m256d v_K6 = _mm512_castpd512_pd256(v_C6);

        curr_in = in_r;
        curr_out = out_r;

        GATHER2_256_D(curr_in, v_in_stride, v_in0, is_contiguous_in);
        curr_in = in_r + in_strides[1];
        GATHER2_256_D(curr_in, v_in_stride, v_in1, is_contiguous_in);
        curr_in = in_r + in_strides[2];
        GATHER2_256_D(curr_in, v_in_stride, v_in2, is_contiguous_in);
        curr_in = in_r + in_strides[3];
        GATHER2_256_D(curr_in, v_in_stride, v_in3, is_contiguous_in);
        curr_in = in_r + in_strides[4];
        GATHER2_256_D(curr_in, v_in_stride, v_in4, is_contiguous_in);
        curr_in = in_r + in_strides[5];
        GATHER2_256_D(curr_in, v_in_stride, v_in5, is_contiguous_in);
        curr_in = in_r + in_strides[6];
        GATHER2_256_D(curr_in, v_in_stride, v_in6, is_contiguous_in);
        curr_in = in_r + in_strides[7];
        GATHER2_256_D(curr_in, v_in_stride, v_in7, is_contiguous_in);
        curr_in = in_r + in_strides[8];
        GATHER2_256_D(curr_in, v_in_stride, v_in8, is_contiguous_in);
        curr_in = in_r + in_strides[9];
        GATHER2_256_D(curr_in, v_in_stride, v_in9, is_contiguous_in);
        curr_in = in_r + in_strides[10];
        GATHER2_256_D(curr_in, v_in_stride, v_in10, is_contiguous_in);
        curr_in = in_r + in_strides[11];
        GATHER2_256_D(curr_in, v_in_stride, v_in11, is_contiguous_in);
        curr_in = in_r + in_strides[12];
        GATHER2_256_D(curr_in, v_in_stride, v_in12, is_contiguous_in);
        curr_in = in_r + in_strides[13];
        GATHER2_256_D(curr_in, v_in_stride, v_in13, is_contiguous_in);
        curr_in = in_r + in_strides[14];
        GATHER2_256_D(curr_in, v_in_stride, v_in14, is_contiguous_in);

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

        if (out_strides[1] == DATA_STRIDE)
        {
            TRANSPOSE_ST2_256_D(curr_out, v_out_stride, v_out0, v_out1);
            curr_out = curr_out + NUM_SETS_256_D * DATA_STRIDE;
            TRANSPOSE_ST2_256_D(curr_out, v_out_stride, v_out2, v_out3);
            curr_out = curr_out + NUM_SETS_256_D * DATA_STRIDE;
            TRANSPOSE_ST2_256_D(curr_out, v_out_stride, v_out4, v_out5);
            curr_out = curr_out + NUM_SETS_256_D * DATA_STRIDE;
            TRANSPOSE_ST2_256_D(curr_out, v_out_stride, v_out6, v_out7);
            curr_out = curr_out + NUM_SETS_256_D * DATA_STRIDE;
            TRANSPOSE_ST2_256_D(curr_out, v_out_stride, v_out8, v_out9);
            curr_out = curr_out + NUM_SETS_256_D * DATA_STRIDE;
            TRANSPOSE_ST2_256_D(curr_out, v_out_stride, v_out10, v_out11);
            curr_out = curr_out + NUM_SETS_256_D * DATA_STRIDE;
            TRANSPOSE_ST2_256_D(curr_out, v_out_stride, v_out12, v_out13);
            curr_out = curr_out + 2 * DATA_STRIDE;
            SCATTER2_256_D_STRIDED(curr_out, v_out_stride, v_out14);
        }
        else
        {
            SCATTER2_256_D(curr_out, v_out_stride, v_out0, is_contiguous_out);
            curr_out = out_r + out_strides[1];
            SCATTER2_256_D(curr_out, v_out_stride, v_out1, is_contiguous_out);
            curr_out = out_r + out_strides[2];
            SCATTER2_256_D(curr_out, v_out_stride, v_out2, is_contiguous_out);
            curr_out = out_r + out_strides[3];
            SCATTER2_256_D(curr_out, v_out_stride, v_out3, is_contiguous_out);
            curr_out = out_r + out_strides[4];
            SCATTER2_256_D(curr_out, v_out_stride, v_out4, is_contiguous_out);
            curr_out = out_r + out_strides[5];
            SCATTER2_256_D(curr_out, v_out_stride, v_out5, is_contiguous_out);
            curr_out = out_r + out_strides[6];
            SCATTER2_256_D(curr_out, v_out_stride, v_out6, is_contiguous_out);
            curr_out = out_r + out_strides[7];
            SCATTER2_256_D(curr_out, v_out_stride, v_out7, is_contiguous_out);
            curr_out = out_r + out_strides[8];
            SCATTER2_256_D(curr_out, v_out_stride, v_out8, is_contiguous_out);
            curr_out = out_r + out_strides[9];
            SCATTER2_256_D(curr_out, v_out_stride, v_out9, is_contiguous_out);
            curr_out = out_r + out_strides[10];
            SCATTER2_256_D(curr_out, v_out_stride, v_out10, is_contiguous_out);
            curr_out = out_r + out_strides[11];
            SCATTER2_256_D(curr_out, v_out_stride, v_out11, is_contiguous_out);
            curr_out = out_r + out_strides[12];
            SCATTER2_256_D(curr_out, v_out_stride, v_out12, is_contiguous_out);
            curr_out = out_r + out_strides[13];
            SCATTER2_256_D(curr_out, v_out_stride, v_out13, is_contiguous_out);
            curr_out = out_r + out_strides[14];
            SCATTER2_256_D(curr_out, v_out_stride, v_out14, is_contiguous_out);
        }

        in_r += NUM_SETS_256_D * v_in_stride;
        out_r += NUM_SETS_256_D * v_out_stride;
    }
    // tail case
    if (n & 1)
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

        __m128d v_K1 = _mm512_castpd512_pd128(v_C1);
        __m128d v_K2 = _mm512_castpd512_pd128(v_C2);
        __m128d v_K3 = _mm512_castpd512_pd128(v_C3);
        __m128d v_K4 = _mm512_castpd512_pd128(v_C4);
        __m128d v_K5 = _mm512_castpd512_pd128(v_C5);
        __m128d v_K6 = _mm512_castpd512_pd128(v_C6);

        curr_in = in_r;
        curr_out = out_r;

        LD_128_D(curr_in, v_in0);
        curr_in = in_r + in_strides[1];
        LD_128_D(curr_in, v_in1);
        curr_in = in_r + in_strides[2];
        LD_128_D(curr_in, v_in2);
        curr_in = in_r + in_strides[3];
        LD_128_D(curr_in, v_in3);
        curr_in = in_r + in_strides[4];
        LD_128_D(curr_in, v_in4);
        curr_in = in_r + in_strides[5];
        LD_128_D(curr_in, v_in5);
        curr_in = in_r + in_strides[6];
        LD_128_D(curr_in, v_in6);
        curr_in = in_r + in_strides[7];
        LD_128_D(curr_in, v_in7);
        curr_in = in_r + in_strides[8];
        LD_128_D(curr_in, v_in8);
        curr_in = in_r + in_strides[9];
        LD_128_D(curr_in, v_in9);
        curr_in = in_r + in_strides[10];
        LD_128_D(curr_in, v_in10);
        curr_in = in_r + in_strides[11];
        LD_128_D(curr_in, v_in11);
        curr_in = in_r + in_strides[12];
        LD_128_D(curr_in, v_in12);
        curr_in = in_r + in_strides[13];
        LD_128_D(curr_in, v_in13);
        curr_in = in_r + in_strides[14];
        LD_128_D(curr_in, v_in14);

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
        curr_out = out_r + out_strides[1];
        ST_128_D(curr_out, v_out1);
        curr_out = out_r + out_strides[2];
        ST_128_D(curr_out, v_out2);
        curr_out = out_r + out_strides[3];
        ST_128_D(curr_out, v_out3);
        curr_out = out_r + out_strides[4];
        ST_128_D(curr_out, v_out4);
        curr_out = out_r + out_strides[5];
        ST_128_D(curr_out, v_out5);
        curr_out = out_r + out_strides[6];
        ST_128_D(curr_out, v_out6);
        curr_out = out_r + out_strides[7];
        ST_128_D(curr_out, v_out7);
        curr_out = out_r + out_strides[8];
        ST_128_D(curr_out, v_out8);
        curr_out = out_r + out_strides[9];
        ST_128_D(curr_out, v_out9);
        curr_out = out_r + out_strides[10];
        ST_128_D(curr_out, v_out10);
        curr_out = out_r + out_strides[11];
        ST_128_D(curr_out, v_out11);
        curr_out = out_r + out_strides[12];
        ST_128_D(curr_out, v_out12);
        curr_out = out_r + out_strides[13];
        ST_128_D(curr_out, v_out13);
        curr_out = out_r + out_strides[14];
        ST_128_D(curr_out, v_out14);
    }
    AOCLFFTZ_LOG(DEBUG, global_logger_mode, "Exit");
}

kfft_ register_kernel_fft15avx512(UINT8 precision, UINT8 direction /* unused */)
{
    if (precision == DT_FLOAT)
    {
        return fft15avx512fp32;
    }
    else if (precision == DT_DOUBLE)
    {
        return fft15avx512fp64;
    }
    else
    {
        return NULL;
    }
}
