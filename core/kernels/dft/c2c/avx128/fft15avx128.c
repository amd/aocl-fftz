// Copyright Advanced Micro Devices, Inc.
// SPDX-License-Identifier: BSD-3-Clause

/** @file fft15avx128.c
 *
 *  @brief Radix-15 FFT kernel with avx128 operations using x86 SIMD intrinsics
 *
 *  This file contains the DIT radix-15 FFT implementations using avx128
 *  operations for single-precision and double-precision inputs.
 *
 *  @author Varun Sanjay
 *
 */

#include "core/kernels/kernel.h"
#include "core/kernels/simd_includes/simd_common.h"

static const ops_cycles_t ops_cnt[NUM_PRECISIONS] = {{0, 28, 78, 60, 7, 7},
                                                     {0, 28, 78, 30, 7, 7}};

ops_cycles_t get_ops_cnt_fft15avx128(UINT8 precision, UINT8 direction)
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

static VOID fft15avx128fp32(VOID *in_real, VOID *in_imag, VOID *out_real,
                            VOID *out_imag, INTP n, aoclfftz_strides_t *strides,
                            VOID *twd, UINT8 flag)
{
    AOCLFFTZ_LOG(DEBUG, global_logger_mode, "Enter");
    const FLOAT CRTM_15[10] = {
        0.55901699437494742410229341718281905886015458990288,
        0.25000000000000000000000000000000000000000000000000,
        0.95105651629515357211643933337938214340569863400000,
        0.58778525229247301629891039327884007596190389052978,
        0.50000000000000000000000000000000000000000000000000,
        0.86602540378443864676372317075293618347140262690519,
        0.48412291827592710612024388657479988457787393064252,
        0.21650635094610964914707551542960572987794876098633,
        0.82363910354633184270744116161596601637855195182647,
        0.50903696045512706468216979248996715975105181034577};

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
    INTP N = n / NUM_SETS_128_S;
    INTP count;

    __m128 v_in0, v_in1, v_in2, v_in3, v_in4, v_in5, v_in6, v_in7, v_in8;
    __m128 v_in9, v_in10, v_in11, v_in12, v_in13, v_in14;
    __m128 v_cv1, v_cv2, v_cv3, v_cv4, v_cv5, v_cv6, v_cv7, v_cv8, v_cv9,
           v_cv10, v_cv11, v_cv12;
    __m128 v_av1, v_av2, v_av3, v_av4, v_av5, v_av6, v_av7, v_av8, v_av9;
    __m128 v_av11, v_av12, v_av13, v_av14, v_av15, v_av16, v_av17;
    __m128 v_av18, v_av19, v_av20, v_av21, v_av22, v_av23, v_av24, v_av25;
    __m128 v_av26, v_av27, v_av30, v_av33;
    __m128 v_av34, v_av35, v_av36, v_av37, v_av38, v_av39, v_av40, v_av41;
    __m128 v_av42, v_av44, v_av45, v_av46, v_av47, v_av48, v_av49;
    __m128 v_tv1, v_tv2, v_tv3, v_tv4, v_tv5, v_tv6, v_tv7, v_tv8, v_tv9,
           v_tv10, v_tv11, v_tv16, v_tv17, v_tv18;
    __m128 v_tv19, v_tv20, v_tv21, v_tv22, v_tv23, v_tv24, v_tv25, v_tv27;
    __m128 v_out0, v_out1, v_out2, v_out3, v_out4, v_out5;
    __m128 v_out6, v_out7, v_out8, v_out9, v_out10, v_out11;
    __m128 v_out12, v_out13, v_out14;

    __m128 v_K1 = _mm_broadcast_ss(&CRTM_15[0]);
    __m128 v_K2 = _mm_broadcast_ss(&CRTM_15[1]);
    __m128 v_K3 = _mm_broadcast_ss(&CRTM_15[2]);
    v_K3 = _mm_xor_ps(v_K3, _neg_128_f[flag].s);
    __m128 v_K4 = _mm_broadcast_ss(&CRTM_15[3]);
    v_K4 = _mm_xor_ps(v_K4, _neg_128_f[flag].s);
    __m128 v_K5 = _mm_broadcast_ss(&CRTM_15[4]);
    __m128 v_K6 = _mm_broadcast_ss(&CRTM_15[5]);
    v_K6 = _mm_xor_ps(v_K6, _neg_128_f[flag].s);
    __m128 v_K7 = _mm_broadcast_ss(&CRTM_15[6]);
    v_K7 = _mm_xor_ps(v_K7, _neg_128_f[flag].s);
    __m128 v_K8 = _mm_broadcast_ss(&CRTM_15[7]);
    v_K8 = _mm_xor_ps(v_K8, _neg_128_f[flag].s);
    __m128 v_K9 = _mm_broadcast_ss(&CRTM_15[8]);
    __m128 v_K10 = _mm_broadcast_ss(&CRTM_15[9]);

    for (count = 0; count < N; count++)
    {
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

        // common operations
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
        v_cv12 = _mm_sub_ps(v_in11, v_in1);
        v_av22 = _mm_sub_ps(v_cv11, v_cv12);
        v_av23 = _mm_sub_ps(v_in8, v_in13);
        v_av24 = _mm_sub_ps(v_in2, v_in7);
        v_av25 = _mm_sub_ps(v_av23, v_av24);
        v_tv16 = _mm_mul_ps(v_K9, v_av22);
        v_tv17 = _mm_mul_ps(v_K10, v_av25);
        v_av26 = _mm_add_ps(v_tv16, v_tv17);
        v_av27 = _mm_sub_ps(v_in10, v_in5);
        v_tv18 = _mm_mul_ps(v_K6, v_av27);
        v_av30 = _mm_add_ps(v_cv12, v_cv11);
        v_av33 = _mm_add_ps(v_av24, v_av23);
        v_av34 = _mm_add_ps(v_av30, v_av33);
        v_tv19 = _mm_mul_ps(v_K8, v_av34);
        v_av35 = _mm_add_ps(v_tv18, v_tv19);
        v_av36 = _mm_sub_ps(v_av33, v_av30);
        v_tv20 = _mm_mul_ps(v_K7, v_av36);
        v_av37 = _mm_add_ps(v_av35, v_tv20);
        v_av38 = _mm_sub_ps(v_av13, v_av12);
        v_av39 = _mm_sub_ps(v_av16, v_av15);
        v_tv21 = _mm_mul_ps(v_K3, v_av38);
        v_tv22 = _mm_mul_ps(v_K4, v_av39);
        v_av40 = _mm_add_ps(v_tv21, v_tv22);

        // real part
        v_av41 = _mm_add_ps(v_av11, v_av18);

        // imag part
        v_av34 = _mm_mul_ps(v_K6, v_av34);
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
        v_tv23 = _mm_mul_ps(v_K9, v_av25);
        v_tv24 = _mm_mul_ps(v_K10, v_av22);
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

        in_r += NUM_SETS_128_S * v_in_stride;
        out_r += NUM_SETS_128_S * v_out_stride;
    }
    // tail case
    if (n & 1)
    {
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

        // common operations
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
        v_cv12 = _mm_sub_ps(v_in11, v_in1);
        v_av22 = _mm_sub_ps(v_cv11, v_cv12);
        v_av23 = _mm_sub_ps(v_in8, v_in13);
        v_av24 = _mm_sub_ps(v_in2, v_in7);
        v_av25 = _mm_sub_ps(v_av23, v_av24);
        v_tv16 = _mm_mul_ps(v_K9, v_av22);
        v_tv17 = _mm_mul_ps(v_K10, v_av25);
        v_av26 = _mm_add_ps(v_tv16, v_tv17);
        v_av27 = _mm_sub_ps(v_in10, v_in5);
        v_tv18 = _mm_mul_ps(v_K6, v_av27);
        v_av30 = _mm_add_ps(v_cv12, v_cv11);
        v_av33 = _mm_add_ps(v_av24, v_av23);
        v_av34 = _mm_add_ps(v_av30, v_av33);
        v_tv19 = _mm_mul_ps(v_K8, v_av34);
        v_av35 = _mm_add_ps(v_tv18, v_tv19);
        v_av36 = _mm_sub_ps(v_av33, v_av30);
        v_tv20 = _mm_mul_ps(v_K7, v_av36);
        v_av37 = _mm_add_ps(v_av35, v_tv20);
        v_av38 = _mm_sub_ps(v_av13, v_av12);
        v_av39 = _mm_sub_ps(v_av16, v_av15);
        v_tv21 = _mm_mul_ps(v_K3, v_av38);
        v_tv22 = _mm_mul_ps(v_K4, v_av39);
        v_av40 = _mm_add_ps(v_tv21, v_tv22);

        // real part
        v_av41 = _mm_add_ps(v_av11, v_av18);

        // imag part
        v_av34 = _mm_mul_ps(v_K6, v_av34);
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
        v_tv23 = _mm_mul_ps(v_K9, v_av25);
        v_tv24 = _mm_mul_ps(v_K10, v_av22);
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
    AOCLFFTZ_LOG(DEBUG, global_logger_mode, "Exit");
}

static VOID fft15avx128fp64(VOID *in_real, VOID *in_imag, VOID *out_real,
                            VOID *out_imag, INTP n, aoclfftz_strides_t *strides,
                            VOID *twd, UINT8 flag)
{
    AOCLFFTZ_LOG(DEBUG, global_logger_mode, "Enter");
    const DOUBLE CRTM_15[10] = {
        0.55901699437494742410229341718281905886015458990288,
        0.25000000000000000000000000000000000000000000000000,
        0.95105651629515357211643933337938214340569863400000,
        0.58778525229247301629891039327884007596190389052978,
        0.50000000000000000000000000000000000000000000000000,
        0.86602540378443864676372317075293618347140262690519,
        0.48412291827592710612024388657479988457787393064252,
        0.21650635094610964914707551542960572987794876098633,
        0.82363910354633184270744116161596601637855195182647,
        0.50903696045512706468216979248996715975105181034577};

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
    INTP v_out_stride = strides->v_out_stride;
    INTP N = n / NUM_SETS_128_D;
    INTP count;

    __m128d v_in0, v_in1, v_in2, v_in3, v_in4, v_in5, v_in6, v_in7, v_in8;
    __m128d v_in9, v_in10, v_in11, v_in12, v_in13, v_in14;
    __m128d v_cv1, v_cv2, v_cv3, v_cv4, v_cv5, v_cv6, v_cv7, v_cv8, v_cv9,
            v_cv10, v_cv11, v_cv12;
    __m128d v_av1, v_av2, v_av3, v_av4, v_av5, v_av6, v_av7, v_av8, v_av9;
    __m128d v_av11, v_av12, v_av13, v_av14, v_av15, v_av16, v_av17;
    __m128d v_av18, v_av19, v_av20, v_av21, v_av22, v_av23, v_av24, v_av25;
    __m128d v_av26, v_av27, v_av30, v_av33;
    __m128d v_av34, v_av35, v_av36, v_av37, v_av38, v_av39, v_av40, v_av41;
    __m128d v_av42, v_av44, v_av45, v_av46, v_av47, v_av48, v_av49;
    __m128d v_tv1, v_tv2, v_tv3, v_tv4, v_tv5, v_tv6, v_tv7, v_tv8, v_tv9,
            v_tv10, v_tv11, v_tv16, v_tv17, v_tv18;
    __m128d v_tv19, v_tv20, v_tv21, v_tv22, v_tv23, v_tv24, v_tv25, v_tv27;
    __m128d v_out0, v_out1, v_out2, v_out3, v_out4, v_out5;
    __m128d v_out6, v_out7, v_out8, v_out9, v_out10, v_out11;
    __m128d v_out12, v_out13, v_out14;

    __m128d v_K1 = _mm_set1_pd(CRTM_15[0]);
    __m128d v_K2 = _mm_set1_pd(CRTM_15[1]);
    __m128d v_K3 = _mm_set1_pd(CRTM_15[2]);
    v_K3 = _mm_xor_pd(v_K3, _neg_128_d[flag].d);
    __m128d v_K4 = _mm_set1_pd(CRTM_15[3]);
    v_K4 = _mm_xor_pd(v_K4, _neg_128_d[flag].d);
    __m128d v_K5 = _mm_set1_pd(CRTM_15[4]);
    __m128d v_K6 = _mm_set1_pd(CRTM_15[5]);
    v_K6 = _mm_xor_pd(v_K6, _neg_128_d[flag].d);
    __m128d v_K7 = _mm_set1_pd(CRTM_15[6]);
    v_K7 = _mm_xor_pd(v_K7, _neg_128_d[flag].d);
    __m128d v_K8 = _mm_set1_pd(CRTM_15[7]);
    v_K8 = _mm_xor_pd(v_K8, _neg_128_d[flag].d);
    __m128d v_K9 = _mm_set1_pd(CRTM_15[8]);
    __m128d v_K10 = _mm_set1_pd(CRTM_15[9]);

    for (count = 0; count < N; count++)
    {
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

        // common operations
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
        v_cv12 = _mm_sub_pd(v_in11, v_in1);
        v_av22 = _mm_sub_pd(v_cv11, v_cv12);
        v_av23 = _mm_sub_pd(v_in8, v_in13);
        v_av24 = _mm_sub_pd(v_in2, v_in7);
        v_av25 = _mm_sub_pd(v_av23, v_av24);
        v_tv16 = _mm_mul_pd(v_K9, v_av22);
        v_tv17 = _mm_mul_pd(v_K10, v_av25);
        v_av26 = _mm_add_pd(v_tv16, v_tv17);
        v_av27 = _mm_sub_pd(v_in10, v_in5);
        v_tv18 = _mm_mul_pd(v_K6, v_av27);
        v_av30 = _mm_add_pd(v_cv12, v_cv11);
        v_av33 = _mm_add_pd(v_av24, v_av23);
        v_av34 = _mm_add_pd(v_av30, v_av33);
        v_tv19 = _mm_mul_pd(v_K8, v_av34);
        v_av35 = _mm_add_pd(v_tv18, v_tv19);
        v_av36 = _mm_sub_pd(v_av33, v_av30);
        v_tv20 = _mm_mul_pd(v_K7, v_av36);
        v_av37 = _mm_add_pd(v_av35, v_tv20);
        v_av38 = _mm_sub_pd(v_av13, v_av12);
        v_av39 = _mm_sub_pd(v_av16, v_av15);
        v_tv21 = _mm_mul_pd(v_K3, v_av38);
        v_tv22 = _mm_mul_pd(v_K4, v_av39);
        v_av40 = _mm_add_pd(v_tv21, v_tv22);

        // real part
        v_av41 = _mm_add_pd(v_av11, v_av18);

        // imag part
        v_av34 = _mm_mul_pd(v_K6, v_av34);
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
        v_tv23 = _mm_mul_pd(v_K9, v_av25);
        v_tv24 = _mm_mul_pd(v_K10, v_av22);
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

        in_r += NUM_SETS_128_D * v_in_stride;
        out_r += NUM_SETS_128_D * v_out_stride;
    }
    AOCLFFTZ_LOG(DEBUG, global_logger_mode, "Exit");
}

kfft_ register_kernel_fft15avx128(UINT8 precision, UINT8 direction /* unused */)
{
    if (precision == DT_FLOAT)
    {
        return fft15avx128fp32;
    }
    else if (precision == DT_DOUBLE)
    {
        return fft15avx128fp64;
    }
    else
    {
        return NULL;
    }
}
