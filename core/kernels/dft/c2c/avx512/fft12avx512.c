// Copyright Advanced Micro Devices, Inc.
// SPDX-License-Identifier: BSD-3-Clause

/** @file fft12avx512.c
 *
 *  @brief Radix-12 FFT kernel with AVX-512 operations using x86 SIMD intrinsics
 *
 *  This file contains the DIT radix-12 FFT implementations using AVX-512
 *  operations for single-precision and double-precision inputs.
 *
 *  @author Jeya R
 *
 */

#include "core/kernels/kernel.h"
#include "core/kernels/simd_includes/simd_common_avx512.h"

static const ops_cycles_t ops_cnt[NUM_PRECISIONS] = {{0, 10, 48, 192, 41, 78},
                                                     {4,  4, 44,  96,  5, 74}};

ops_cycles_t get_ops_cnt_fft12avx512(FFTZ_UINT8 precision, FFTZ_UINT8 direction)
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

static FFTZ_VOID fft12avx512fp32(FFTZ_VOID *in_real, FFTZ_VOID *in_imag,
                                 FFTZ_VOID *out_real, FFTZ_VOID *out_imag,
                                 FFTZ_INTP n, aoclfftz_strides_t *strides,
                                 FFTZ_VOID *twd, FFTZ_UINT8 flag)
{
    AOCLFFTZ_LOG(DEBUG, global_logger_mode, "Enter");
    const FFTZ_FLOAT CRTM_12[3] = {
        0.86602540378443864676372317075293618347140262700000f,
        0.50000000000000000000000000000000000000000000000000f,
        1.00000000000000000000000000000000000000000000000000f};

    FFTZ_FLOAT *in_r = (FFTZ_FLOAT *)in_real;
    FFTZ_FLOAT *out_r = (FFTZ_FLOAT *)out_real;
    FFTZ_FLOAT *curr_in, *curr_out;

#ifdef VOLATILE_STRIDE_ARRAY
    volatile FFTZ_INTP *in_strides = strides->in_strides;
    volatile FFTZ_INTP *out_strides = strides->out_strides;
#else
    FFTZ_INTP *in_strides = strides->in_strides;
    FFTZ_INTP *out_strides = strides->out_strides;
#endif
    FFTZ_INTP v_in_stride = strides->v_in_stride;
    FFTZ_UINT8 is_contiguous_in = (v_in_stride == DATA_STRIDE);
    FFTZ_INTP v_out_stride = strides->v_out_stride;
    FFTZ_UINT8 is_contiguous_out = (v_out_stride == DATA_STRIDE);

    FFTZ_INTP N = n / NUM_SETS_512_S;
    FFTZ_INTP remaining_sets = n % NUM_SETS_512_S;
    FFTZ_INTP count;

    __m512 v_C1 = _mm512_set1_ps(CRTM_12[0]);
    __m512 v_C2 = _mm512_set1_ps(CRTM_12[1]);
    __m512 v_C3 = _mm512_set1_ps(CRTM_12[2]);
    v_C3 = _mm512_xor_ps(v_C3, _neg_512_f[flag].s);
    __m512 v_C4 = _mm512_set1_ps(CRTM_12[0]);
    v_C4 = _mm512_xor_ps(v_C4, _neg_512_f[flag].s);
    __m512 v_C5 = _mm512_set1_ps(CRTM_12[1]);
    v_C5 = _mm512_xor_ps(v_C5, _neg_512_f[flag].s);

    for (count = 0; count < N; count++)
    {
        __m512 v_in0, v_in1, v_in2, v_in3, v_in4, v_in5, v_in6, v_in7, v_in8,
               v_in9, v_in10, v_in11;
        __m512 v_av1, v_av2, v_av3, v_av4, v_av5, v_av6, v_av7, v_av8, v_av9,
               v_av10, v_av11, v_av12;
        __m512 v_cv1, v_cv2, v_cv3, v_cv4, v_cv5, v_cv6, v_cv7, v_cv8;
        __m512 v_tv1, v_tv2, v_tv3, v_tv4, v_tv5;
        __m512 v_out0, v_out1, v_out2, v_out3, v_out4, v_out5, v_out6, v_out7,
               v_out8, v_out9, v_out10, v_out11;

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

        // Common operations
        v_av1 = _mm512_add_ps(v_in0, v_in6);
        v_av2 = _mm512_add_ps(v_in2, v_in4);
        v_av3 = _mm512_add_ps(v_in8, v_in10);
        v_av4 = _mm512_add_ps(v_in1, v_in5);
        v_av5 = _mm512_add_ps(v_in7, v_in11);
        v_av6 = _mm512_add_ps(v_in3, v_in9);

        v_cv1 = _mm512_add_ps(v_av2, v_av3);
        v_cv2 = _mm512_add_ps(v_av4, v_av5);
        v_cv3 = _mm512_add_ps(v_av1, v_av6);
        v_cv4 = _mm512_sub_ps(v_av1, v_av6);

        v_cv5 = _mm512_add_ps(v_cv1, v_cv2);
        v_cv6 = _mm512_sub_ps(v_cv1, v_cv2);

        // output point 1 & 7
        v_out0 = _mm512_add_ps(v_cv3, v_cv5);
        v_out6 = _mm512_add_ps(v_cv4, v_cv6);

        // Common operations
        v_av7 = _mm512_sub_ps(v_in0, v_in6);
        v_av8 = _mm512_sub_ps(v_in2, v_in4);
        v_av9 = _mm512_sub_ps(v_in8, v_in10);
        v_av10 = _mm512_sub_ps(v_in1, v_in5);
        v_av11 = _mm512_sub_ps(v_in7, v_in11);
        v_av12 = _mm512_sub_ps(v_in3, v_in9);

        v_cv1 = _mm512_sub_ps(v_av8, v_av9);
        v_cv2 = _mm512_sub_ps(v_av4, v_av5);
        v_cv7 = _mm512_sub_ps(v_cv2, v_av12);
        v_cv8 = _mm512_sub_ps(v_av7, v_cv1);

        v_tv1 = _mm512_mul_ps(v_C3, v_cv7);
        v_tv1 = SWAP_RI_512_S(CONJ_512_S(v_tv1));

        // output point 4 & 10
        v_out3 = _mm512_sub_ps(v_cv8, v_tv1);
        v_out9 = _mm512_add_ps(v_cv8, v_tv1);

        v_tv1 = _mm512_mul_ps(v_C3, v_av12);
        v_tv1 = CONJ_512_S(v_tv1);

        v_tv2 = _mm512_mul_ps(v_C2, v_cv1);
        v_cv1 = _mm512_add_ps(v_av7, v_tv2);
        v_cv7 = _mm512_sub_ps(v_av10, v_av11);

        v_tv3 = _mm512_mul_ps(v_C1, v_cv7);
        v_cv8 = _mm512_add_ps(v_cv1, v_tv3);

        v_tv4 = _mm512_mul_ps(v_C5, v_cv2);
        v_tv4 = CONJ_512_S(v_tv4);

        v_cv2 = _mm512_sub_ps(v_av2, v_av3);
        v_tv5 = _mm512_mul_ps(v_C4, v_cv2);
        v_tv5 = CONJ_512_S(v_tv5);

        v_cv2 = _mm512_add_ps(v_tv1, v_tv4);
        v_cv7 = _mm512_add_ps(v_cv2, v_tv5);
        v_cv7 = SWAP_RI_512_S(v_cv7);

        // output point 2 & 12
        v_out1 = _mm512_sub_ps(v_cv8, v_cv7);
        v_out11 = _mm512_add_ps(v_cv8, v_cv7);

        v_cv7 = _mm512_sub_ps(v_cv1, v_tv3);
        v_cv8 = _mm512_sub_ps(v_cv2, v_tv5);
        v_cv8 = SWAP_RI_512_S(v_cv8);

        // output point 6 & 8
        v_out5 = _mm512_sub_ps(v_cv7, v_cv8);
        v_out7 = _mm512_add_ps(v_cv7, v_cv8);

        v_tv1 = _mm512_mul_ps(v_C2, v_cv6);
        v_cv1 = _mm512_sub_ps(v_cv4, v_tv1);
        v_cv2 = _mm512_add_ps(v_av8, v_av9);
        v_cv4 = _mm512_add_ps(v_av10, v_av11);
        v_cv6 = _mm512_add_ps(v_cv4, v_cv2);
        v_tv2 = _mm512_mul_ps(v_C4, v_cv6);
        v_tv2 = SWAP_RI_512_S(CONJ_512_S(v_tv2));

        // output point 3 & 11
        v_out2 = _mm512_sub_ps(v_cv1, v_tv2);
        v_out10 = _mm512_add_ps(v_cv1, v_tv2);

        v_tv1 = _mm512_mul_ps(v_C2, v_cv5);
        v_cv1 = _mm512_sub_ps(v_cv3, v_tv1);
        v_cv6 = _mm512_sub_ps(v_cv4, v_cv2);
        v_tv2 = _mm512_mul_ps(v_C4, v_cv6);
        v_tv2 = SWAP_RI_512_S(CONJ_512_S(v_tv2));

        // output point 5 & 9
        v_out4 = _mm512_sub_ps(v_cv1, v_tv2);
        v_out8 = _mm512_add_ps(v_cv1, v_tv2);

        if (out_strides[1] == DATA_STRIDE)
        {
            TRANSPOSE_ST8_512_S(curr_out, v_out_stride, v_out0, v_out1, v_out2,
                                v_out3, v_out4, v_out5, v_out6, v_out7);
            curr_out = curr_out + NUM_SETS_512_S * DATA_STRIDE;
            TRANSPOSE_ST4_512_S(curr_out, v_out_stride, v_out8, v_out9, v_out10,
                                v_out11);
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
        }

        in_r += NUM_SETS_512_S * v_in_stride;
        out_r += NUM_SETS_512_S * v_out_stride;
    }
    // tail cases
    if (remaining_sets >= NUM_SETS_256_S)
    {
        __m256 v_in0, v_in1, v_in2, v_in3, v_in4, v_in5, v_in6, v_in7, v_in8,
               v_in9, v_in10, v_in11;
        __m256 v_av1, v_av2, v_av3, v_av4, v_av5, v_av6, v_av7, v_av8, v_av9,
               v_av10, v_av11, v_av12;
        __m256 v_cv1, v_cv2, v_cv3, v_cv4, v_cv5, v_cv6, v_cv7, v_cv8;
        __m256 v_tv1, v_tv2, v_tv3, v_tv4, v_tv5;
        __m256 v_out0, v_out1, v_out2, v_out3, v_out4, v_out5, v_out6, v_out7,
               v_out8, v_out9, v_out10, v_out11;
        __m256 v_K1 = _mm512_castps512_ps256(v_C1);
        __m256 v_K2 = _mm512_castps512_ps256(v_C2);
        __m256 v_K3 = _mm512_castps512_ps256(v_C3);
        __m256 v_K4 = _mm512_castps512_ps256(v_C4);
        __m256 v_K5 = _mm512_castps512_ps256(v_C5);

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

        // Common operations
        v_av1 = _mm256_add_ps(v_in0, v_in6);
        v_av2 = _mm256_add_ps(v_in2, v_in4);
        v_av3 = _mm256_add_ps(v_in8, v_in10);
        v_av4 = _mm256_add_ps(v_in1, v_in5);
        v_av5 = _mm256_add_ps(v_in7, v_in11);
        v_av6 = _mm256_add_ps(v_in3, v_in9);

        v_cv1 = _mm256_add_ps(v_av2, v_av3);
        v_cv2 = _mm256_add_ps(v_av4, v_av5);
        v_cv3 = _mm256_add_ps(v_av1, v_av6);
        v_cv4 = _mm256_sub_ps(v_av1, v_av6);

        v_cv5 = _mm256_add_ps(v_cv1, v_cv2);
        v_cv6 = _mm256_sub_ps(v_cv1, v_cv2);

        // output point 1 & 7
        v_out0 = _mm256_add_ps(v_cv3, v_cv5);
        v_out6 = _mm256_add_ps(v_cv4, v_cv6);

        // Common operations
        v_av7 = _mm256_sub_ps(v_in0, v_in6);
        v_av8 = _mm256_sub_ps(v_in2, v_in4);
        v_av9 = _mm256_sub_ps(v_in8, v_in10);
        v_av10 = _mm256_sub_ps(v_in1, v_in5);
        v_av11 = _mm256_sub_ps(v_in7, v_in11);
        v_av12 = _mm256_sub_ps(v_in3, v_in9);

        v_cv1 = _mm256_sub_ps(v_av8, v_av9);
        v_cv2 = _mm256_sub_ps(v_av4, v_av5);
        v_cv7 = _mm256_sub_ps(v_cv2, v_av12);
        v_cv8 = _mm256_sub_ps(v_av7, v_cv1);

        v_tv1 = _mm256_mul_ps(v_K3, v_cv7);
        v_tv1 = SWAP_RI_256_S(CONJ_256_S(v_tv1));

        // output point 4 & 10
        v_out3 = _mm256_sub_ps(v_cv8, v_tv1);
        v_out9 = _mm256_add_ps(v_cv8, v_tv1);

        v_tv1 = _mm256_mul_ps(v_K3, v_av12);
        v_tv1 = CONJ_256_S(v_tv1);

        v_tv2 = _mm256_mul_ps(v_K2, v_cv1);
        v_cv1 = _mm256_add_ps(v_av7, v_tv2);
        v_cv7 = _mm256_sub_ps(v_av10, v_av11);

        v_tv3 = _mm256_mul_ps(v_K1, v_cv7);
        v_cv8 = _mm256_add_ps(v_cv1, v_tv3);

        v_tv4 = _mm256_mul_ps(v_K5, v_cv2);
        v_tv4 = CONJ_256_S(v_tv4);

        v_cv2 = _mm256_sub_ps(v_av2, v_av3);
        v_tv5 = _mm256_mul_ps(v_K4, v_cv2);
        v_tv5 = CONJ_256_S(v_tv5);

        v_cv2 = _mm256_add_ps(v_tv1, v_tv4);
        v_cv7 = _mm256_add_ps(v_cv2, v_tv5);
        v_cv7 = SWAP_RI_256_S(v_cv7);

        // output point 2 & 12
        v_out1 = _mm256_sub_ps(v_cv8, v_cv7);
        v_out11 = _mm256_add_ps(v_cv8, v_cv7);

        v_cv7 = _mm256_sub_ps(v_cv1, v_tv3);
        v_cv8 = _mm256_sub_ps(v_cv2, v_tv5);
        v_cv8 = SWAP_RI_256_S(v_cv8);

        // output point 6 & 8
        v_out5 = _mm256_sub_ps(v_cv7, v_cv8);
        v_out7 = _mm256_add_ps(v_cv7, v_cv8);

        v_tv1 = _mm256_mul_ps(v_K2, v_cv6);
        v_cv1 = _mm256_sub_ps(v_cv4, v_tv1);
        v_cv2 = _mm256_add_ps(v_av8, v_av9);
        v_cv4 = _mm256_add_ps(v_av10, v_av11);
        v_cv6 = _mm256_add_ps(v_cv4, v_cv2);
        v_tv2 = _mm256_mul_ps(v_K4, v_cv6);
        v_tv2 = SWAP_RI_256_S(CONJ_256_S(v_tv2));

        // output point 3 & 11
        v_out2 = _mm256_sub_ps(v_cv1, v_tv2);
        v_out10 = _mm256_add_ps(v_cv1, v_tv2);

        v_tv1 = _mm256_mul_ps(v_K2, v_cv5);
        v_cv1 = _mm256_sub_ps(v_cv3, v_tv1);
        v_cv6 = _mm256_sub_ps(v_cv4, v_cv2);
        v_tv2 = _mm256_mul_ps(v_K4, v_cv6);
        v_tv2 = SWAP_RI_256_S(CONJ_256_S(v_tv2));

        // output point 5 & 9
        v_out4 = _mm256_sub_ps(v_cv1, v_tv2);
        v_out8 = _mm256_add_ps(v_cv1, v_tv2);

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
        }

        in_r += NUM_SETS_256_S * v_in_stride;
        out_r += NUM_SETS_256_S * v_out_stride;
        remaining_sets = remaining_sets % NUM_SETS_256_S;
    }
    // tail cases
    if (remaining_sets >= NUM_SETS_128_S)
    {
        __m128 v_in0, v_in1, v_in2, v_in3, v_in4, v_in5, v_in6, v_in7, v_in8,
               v_in9, v_in10, v_in11;
        __m128 v_av1, v_av2, v_av3, v_av4, v_av5, v_av6, v_av7, v_av8, v_av9,
               v_av10, v_av11, v_av12;
        __m128 v_cv1, v_cv2, v_cv3, v_cv4, v_cv5, v_cv6, v_cv7, v_cv8;
        __m128 v_tv1, v_tv2, v_tv3, v_tv4, v_tv5;
        __m128 v_out0, v_out1, v_out2, v_out3, v_out4, v_out5, v_out6,
               v_out7, v_out8, v_out9, v_out10, v_out11;

        __m128 v_K1 = _mm512_castps512_ps128(v_C1);
        __m128 v_K2 = _mm512_castps512_ps128(v_C2);
        __m128 v_K3 = _mm512_castps512_ps128(v_C3);
        __m128 v_K4 = _mm512_castps512_ps128(v_C4);
        __m128 v_K5 = _mm512_castps512_ps128(v_C5);

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

        // Common operations
        v_av1 = _mm_add_ps(v_in0, v_in6);
        v_av2 = _mm_add_ps(v_in2, v_in4);
        v_av3 = _mm_add_ps(v_in8, v_in10);
        v_av4 = _mm_add_ps(v_in1, v_in5);
        v_av5 = _mm_add_ps(v_in7, v_in11);
        v_av6 = _mm_add_ps(v_in3, v_in9);

        v_cv1 = _mm_add_ps(v_av2, v_av3);
        v_cv2 = _mm_add_ps(v_av4, v_av5);
        v_cv3 = _mm_add_ps(v_av1, v_av6);
        v_cv4 = _mm_sub_ps(v_av1, v_av6);

        v_cv5 = _mm_add_ps(v_cv1, v_cv2);
        v_cv6 = _mm_sub_ps(v_cv1, v_cv2);

        // output point 1 & 7
        v_out0 = _mm_add_ps(v_cv3, v_cv5);
        v_out6 = _mm_add_ps(v_cv4, v_cv6);

        // Common operations
        v_av7 = _mm_sub_ps(v_in0, v_in6);
        v_av8 = _mm_sub_ps(v_in2, v_in4);
        v_av9 = _mm_sub_ps(v_in8, v_in10);
        v_av10 = _mm_sub_ps(v_in1, v_in5);
        v_av11 = _mm_sub_ps(v_in7, v_in11);
        v_av12 = _mm_sub_ps(v_in3, v_in9);

        v_cv1 = _mm_sub_ps(v_av8, v_av9);
        v_cv2 = _mm_sub_ps(v_av4, v_av5);
        v_cv7 = _mm_sub_ps(v_cv2, v_av12);
        v_cv8 = _mm_sub_ps(v_av7, v_cv1);

        v_tv1 = _mm_mul_ps(v_K3, v_cv7);
        v_tv1 = SWAP_RI_128_S(CONJ_128_S(v_tv1));

        // output point 4 & 10
        v_out3 = _mm_sub_ps(v_cv8, v_tv1);
        v_out9 = _mm_add_ps(v_cv8, v_tv1);

        v_tv1 = _mm_mul_ps(v_K3, v_av12);
        v_tv1 = CONJ_128_S(v_tv1);

        v_tv2 = _mm_mul_ps(v_K2, v_cv1);
        v_cv1 = _mm_add_ps(v_av7, v_tv2);
        v_cv7 = _mm_sub_ps(v_av10, v_av11);

        v_tv3 = _mm_mul_ps(v_K1, v_cv7);
        v_cv8 = _mm_add_ps(v_cv1, v_tv3);

        v_tv4 = _mm_mul_ps(v_K5, v_cv2);
        v_tv4 = CONJ_128_S(v_tv4);

        v_cv2 = _mm_sub_ps(v_av2, v_av3);
        v_tv5 = _mm_mul_ps(v_K4, v_cv2);
        v_tv5 = CONJ_128_S(v_tv5);

        v_cv2 = _mm_add_ps(v_tv1, v_tv4);
        v_cv7 = _mm_add_ps(v_cv2, v_tv5);
        v_cv7 = SWAP_RI_128_S(v_cv7);

        // output point 2 & 12
        v_out1 = _mm_sub_ps(v_cv8, v_cv7);
        v_out11 = _mm_add_ps(v_cv8, v_cv7);

        v_cv7 = _mm_sub_ps(v_cv1, v_tv3);
        v_cv8 = _mm_sub_ps(v_cv2, v_tv5);
        v_cv8 = SWAP_RI_128_S(v_cv8);

        // output point 6 & 8
        v_out5 = _mm_sub_ps(v_cv7, v_cv8);
        v_out7 = _mm_add_ps(v_cv7, v_cv8);

        v_tv1 = _mm_mul_ps(v_K2, v_cv6);
        v_cv1 = _mm_sub_ps(v_cv4, v_tv1);
        v_cv2 = _mm_add_ps(v_av8, v_av9);
        v_cv4 = _mm_add_ps(v_av10, v_av11);
        v_cv6 = _mm_add_ps(v_cv4, v_cv2);
        v_tv2 = _mm_mul_ps(v_K4, v_cv6);
        v_tv2 = SWAP_RI_128_S(CONJ_128_S(v_tv2));

        // output point 3 & 11
        v_out2 = _mm_sub_ps(v_cv1, v_tv2);
        v_out10 = _mm_add_ps(v_cv1, v_tv2);

        v_tv1 = _mm_mul_ps(v_K2, v_cv5);
        v_cv1 = _mm_sub_ps(v_cv3, v_tv1);
        v_cv6 = _mm_sub_ps(v_cv4, v_cv2);
        v_tv2 = _mm_mul_ps(v_K4, v_cv6);
        v_tv2 = SWAP_RI_128_S(CONJ_128_S(v_tv2));

        // output point 5 & 9
        v_out4 = _mm_sub_ps(v_cv1, v_tv2);
        v_out8 = _mm_add_ps(v_cv1, v_tv2);

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
        }

        in_r = in_r + (v_in_stride << 1);
        out_r = out_r + (v_out_stride << 1);
    }
    // tail case
    if (remaining_sets & 1)
    {
        __m128 v_in0, v_in1, v_in2, v_in3, v_in4, v_in5, v_in6, v_in7, v_in8,
               v_in9, v_in10, v_in11;
        __m128 v_av1, v_av2, v_av3, v_av4, v_av5, v_av6, v_av7, v_av8, v_av9,
               v_av10, v_av11, v_av12;
        __m128 v_cv1, v_cv2, v_cv3, v_cv4, v_cv5, v_cv6, v_cv7, v_cv8;
        __m128 v_tv1, v_tv2, v_tv3, v_tv4, v_tv5;
        __m128 v_out0, v_out1, v_out2, v_out3, v_out4, v_out5, v_out6, v_out7,
               v_out8, v_out9, v_out10, v_out11;

        __m128 v_K1 = _mm512_castps512_ps128(v_C1);
        __m128 v_K2 = _mm512_castps512_ps128(v_C2);
        __m128 v_K3 = _mm512_castps512_ps128(v_C3);
        __m128 v_K4 = _mm512_castps512_ps128(v_C4);
        __m128 v_K5 = _mm512_castps512_ps128(v_C5);

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

        // Common operations
        v_av1 = _mm_add_ps(v_in0, v_in6);
        v_av2 = _mm_add_ps(v_in2, v_in4);
        v_av3 = _mm_add_ps(v_in8, v_in10);
        v_av4 = _mm_add_ps(v_in1, v_in5);
        v_av5 = _mm_add_ps(v_in7, v_in11);
        v_av6 = _mm_add_ps(v_in3, v_in9);

        v_cv1 = _mm_add_ps(v_av2, v_av3);
        v_cv2 = _mm_add_ps(v_av4, v_av5);
        v_cv3 = _mm_add_ps(v_av1, v_av6);
        v_cv4 = _mm_sub_ps(v_av1, v_av6);

        v_cv5 = _mm_add_ps(v_cv1, v_cv2);
        v_cv6 = _mm_sub_ps(v_cv1, v_cv2);

        // output point 1 & 7
        v_out0 = _mm_add_ps(v_cv3, v_cv5);
        v_out6 = _mm_add_ps(v_cv4, v_cv6);

        // Common operations
        v_av7 = _mm_sub_ps(v_in0, v_in6);
        v_av8 = _mm_sub_ps(v_in2, v_in4);
        v_av9 = _mm_sub_ps(v_in8, v_in10);
        v_av10 = _mm_sub_ps(v_in1, v_in5);
        v_av11 = _mm_sub_ps(v_in7, v_in11);
        v_av12 = _mm_sub_ps(v_in3, v_in9);

        v_cv1 = _mm_sub_ps(v_av8, v_av9);
        v_cv2 = _mm_sub_ps(v_av4, v_av5);
        v_cv7 = _mm_sub_ps(v_cv2, v_av12);
        v_cv8 = _mm_sub_ps(v_av7, v_cv1);

        v_tv1 = _mm_mul_ps(v_K3, v_cv7);
        v_tv1 = SWAP_RI_128_S(CONJ_128_S(v_tv1));

        // output point 4 & 10
        v_out3 = _mm_sub_ps(v_cv8, v_tv1);
        v_out9 = _mm_add_ps(v_cv8, v_tv1);

        v_tv1 = _mm_mul_ps(v_K3, v_av12);
        v_tv1 = CONJ_128_S(v_tv1);

        v_tv2 = _mm_mul_ps(v_K2, v_cv1);
        v_cv1 = _mm_add_ps(v_av7, v_tv2);
        v_cv7 = _mm_sub_ps(v_av10, v_av11);

        v_tv3 = _mm_mul_ps(v_K1, v_cv7);
        v_cv8 = _mm_add_ps(v_cv1, v_tv3);

        v_tv4 = _mm_mul_ps(v_K5, v_cv2);
        v_tv4 = CONJ_128_S(v_tv4);

        v_cv2 = _mm_sub_ps(v_av2, v_av3);
        v_tv5 = _mm_mul_ps(v_K4, v_cv2);
        v_tv5 = CONJ_128_S(v_tv5);

        v_cv2 = _mm_add_ps(v_tv1, v_tv4);
        v_cv7 = _mm_add_ps(v_cv2, v_tv5);
        v_cv7 = SWAP_RI_128_S(v_cv7);

        // output point 2 & 12
        v_out1 = _mm_sub_ps(v_cv8, v_cv7);
        v_out11 = _mm_add_ps(v_cv8, v_cv7);

        v_cv7 = _mm_sub_ps(v_cv1, v_tv3);
        v_cv8 = _mm_sub_ps(v_cv2, v_tv5);
        v_cv8 = SWAP_RI_128_S(v_cv8);

        // output point 6 & 8
        v_out5 = _mm_sub_ps(v_cv7, v_cv8);
        v_out7 = _mm_add_ps(v_cv7, v_cv8);

        v_tv1 = _mm_mul_ps(v_K2, v_cv6);
        v_cv1 = _mm_sub_ps(v_cv4, v_tv1);
        v_cv2 = _mm_add_ps(v_av8, v_av9);
        v_cv4 = _mm_add_ps(v_av10, v_av11);
        v_cv6 = _mm_add_ps(v_cv4, v_cv2);
        v_tv2 = _mm_mul_ps(v_K4, v_cv6);
        v_tv2 = SWAP_RI_128_S(CONJ_128_S(v_tv2));

        // output point 3 & 11
        v_out2 = _mm_sub_ps(v_cv1, v_tv2);
        v_out10 = _mm_add_ps(v_cv1, v_tv2);

        v_tv1 = _mm_mul_ps(v_K2, v_cv5);
        v_cv1 = _mm_sub_ps(v_cv3, v_tv1);
        v_cv6 = _mm_sub_ps(v_cv4, v_cv2);
        v_tv2 = _mm_mul_ps(v_K4, v_cv6);
        v_tv2 = SWAP_RI_128_S(CONJ_128_S(v_tv2));

        // output point 5 & 9
        v_out4 = _mm_sub_ps(v_cv1, v_tv2);
        v_out8 = _mm_add_ps(v_cv1, v_tv2);

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
        }
    }
    AOCLFFTZ_LOG(DEBUG, global_logger_mode, "Exit");
}

static FFTZ_VOID fft12avx512fp64(FFTZ_VOID *in_real, FFTZ_VOID *in_imag,
                                 FFTZ_VOID *out_real, FFTZ_VOID *out_imag,
                                 FFTZ_INTP n, aoclfftz_strides_t *strides,
                                 FFTZ_VOID *twd, FFTZ_UINT8 flag)
{
    AOCLFFTZ_LOG(DEBUG, global_logger_mode, "Enter");
    const FFTZ_DOUBLE CRTM_12[3] = {
        0.86602540378443864676372317075293618347140262700000,
        0.50000000000000000000000000000000000000000000000000,
        1.00000000000000000000000000000000000000000000000000};

    FFTZ_DOUBLE *in_r = (FFTZ_DOUBLE *)in_real;
    FFTZ_DOUBLE *out_r = (FFTZ_DOUBLE *)out_real;
    FFTZ_DOUBLE *curr_in, *curr_out;

#ifdef VOLATILE_STRIDE_ARRAY
    volatile FFTZ_INTP *in_strides = strides->in_strides;
    volatile FFTZ_INTP *out_strides = strides->out_strides;
#else
    FFTZ_INTP *in_strides = strides->in_strides;
    FFTZ_INTP *out_strides = strides->out_strides;
#endif
    FFTZ_INTP v_in_stride = strides->v_in_stride;
    FFTZ_UINT8 is_contiguous_in = (v_in_stride == DATA_STRIDE);
    FFTZ_INTP v_out_stride = strides->v_out_stride;
    FFTZ_UINT8 is_contiguous_out = (v_out_stride == DATA_STRIDE);

    FFTZ_INTP N = n / NUM_SETS_512_D;
    FFTZ_INTP remaining_sets = n % NUM_SETS_512_D;
    FFTZ_INTP count;

    __m512d v_C1 = _mm512_set1_pd(CRTM_12[0]);
    __m512d v_C2 = _mm512_set1_pd(CRTM_12[1]);
    __m512d v_sign_conj = _mm512_xor_pd(_neg_512_d[flag].d, _conj_512_d.d);

    __m512d v_C4_conj =
        _mm512_xor_pd(_mm512_set1_pd(CRTM_12[0]), _conj_512_d.d);
    v_C4_conj = _mm512_xor_pd(v_C4_conj, _neg_512_d[flag].d);
    __m512d v_C5_conj =
        _mm512_xor_pd(_mm512_set1_pd(CRTM_12[1]), _conj_512_d.d);
    v_C5_conj = _mm512_xor_pd(v_C5_conj, _neg_512_d[flag].d);

    for (count = 0; count < N; count++)
    {
        __m512d v_in0, v_in1, v_in2, v_in3, v_in4, v_in5, v_in6, v_in7, v_in8,
                v_in9, v_in10, v_in11;
        __m512d v_av1, v_av2, v_av3, v_av4, v_av5, v_av6, v_av7, v_av8, v_av9,
                v_av10, v_av11, v_av12;
        __m512d v_out0, v_out1, v_out2, v_out3, v_out4, v_out5, v_out6, v_out7,
                v_out8, v_out9, v_out10, v_out11;
        __m512d v_cv1, v_cv2, v_cv3, v_cv4, v_cv5, v_cv6, v_cv7, v_cv8;
        __m512d v_tv1, v_tv2, v_tv3, v_tv5;

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

        // Common operations
        v_av1 = _mm512_add_pd(v_in0, v_in6);
        v_av2 = _mm512_add_pd(v_in2, v_in4);
        v_av3 = _mm512_add_pd(v_in8, v_in10);
        v_av4 = _mm512_add_pd(v_in1, v_in5);
        v_av5 = _mm512_add_pd(v_in7, v_in11);
        v_av6 = _mm512_add_pd(v_in3, v_in9);

        v_cv1 = _mm512_add_pd(v_av2, v_av3);
        v_cv2 = _mm512_add_pd(v_av4, v_av5);
        v_cv3 = _mm512_add_pd(v_av1, v_av6);
        v_cv4 = _mm512_sub_pd(v_av1, v_av6);

        v_cv5 = _mm512_add_pd(v_cv1, v_cv2);
        v_cv6 = _mm512_sub_pd(v_cv1, v_cv2);

        // output point 1 & 7
        v_out0 = _mm512_add_pd(v_cv3, v_cv5);
        v_out6 = _mm512_add_pd(v_cv4, v_cv6);

        // Common operations
        v_av7 = _mm512_sub_pd(v_in0, v_in6);
        v_av8 = _mm512_sub_pd(v_in2, v_in4);
        v_av9 = _mm512_sub_pd(v_in8, v_in10);
        v_av10 = _mm512_sub_pd(v_in1, v_in5);
        v_av11 = _mm512_sub_pd(v_in7, v_in11);
        v_av12 = _mm512_sub_pd(v_in3, v_in9);

        v_cv1 = _mm512_sub_pd(v_av8, v_av9);
        v_cv2 = _mm512_sub_pd(v_av4, v_av5);
        v_cv7 = _mm512_sub_pd(v_cv2, v_av12);
        v_cv8 = _mm512_sub_pd(v_av7, v_cv1);

        v_tv1 = _mm512_xor_pd(v_cv7, v_sign_conj);
        v_tv1 = SWAP_RI_512_D(v_tv1);

        // output point 4 & 10
        v_out3 = _mm512_sub_pd(v_cv8, v_tv1);
        v_out9 = _mm512_add_pd(v_cv8, v_tv1);

        v_tv1 = _mm512_xor_pd(v_av12, v_sign_conj);
        v_cv1 = _mm512_fmadd_pd(v_C2, v_cv1, v_av7);
        v_cv7 = _mm512_sub_pd(v_av10, v_av11);

        v_tv3 = _mm512_mul_pd(v_C1, v_cv7);
        v_cv8 = _mm512_add_pd(v_cv1, v_tv3);

        const __m512d cv2_save = v_cv2;
        v_cv2 = _mm512_sub_pd(v_av2, v_av3);
        v_tv5 = _mm512_mul_pd(v_cv2, v_C4_conj);

        v_cv2 = _mm512_fmadd_pd(cv2_save, v_C5_conj, v_tv1);
        v_cv7 = _mm512_add_pd(v_cv2, v_tv5);
        v_cv7 = SWAP_RI_512_D(v_cv7);

        // output point 2 & 12
        v_out1 = _mm512_sub_pd(v_cv8, v_cv7);
        v_out11 = _mm512_add_pd(v_cv8, v_cv7);

        v_cv7 = _mm512_sub_pd(v_cv1, v_tv3);
        v_cv8 = _mm512_sub_pd(v_cv2, v_tv5);
        v_cv8 = SWAP_RI_512_D(v_cv8);

        // output point 6 & 8
        v_out5 = _mm512_sub_pd(v_cv7, v_cv8);
        v_out7 = _mm512_add_pd(v_cv7, v_cv8);

        v_cv1 = _mm512_fnmadd_pd(v_C2, v_cv6, v_cv4);
        v_cv2 = _mm512_add_pd(v_av8, v_av9);
        v_cv4 = _mm512_add_pd(v_av10, v_av11);
        v_cv6 = _mm512_add_pd(v_cv4, v_cv2);
        v_tv2 = _mm512_mul_pd(v_cv6, v_C4_conj);
        v_tv2 = SWAP_RI_512_D(v_tv2);

        // output point 3 & 11
        v_out2 = _mm512_sub_pd(v_cv1, v_tv2);
        v_out10 = _mm512_add_pd(v_cv1, v_tv2);

        v_cv1 = _mm512_fnmadd_pd(v_C2, v_cv5, v_cv3);
        v_cv6 = _mm512_sub_pd(v_cv4, v_cv2);
        v_tv2 = _mm512_mul_pd(v_cv6, v_C4_conj);
        v_tv2 = SWAP_RI_512_D(v_tv2);

        // output point 5 & 9
        v_out4 = _mm512_sub_pd(v_cv1, v_tv2);
        v_out8 = _mm512_add_pd(v_cv1, v_tv2);

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
        }

        in_r += NUM_SETS_512_D * v_in_stride;
        out_r += NUM_SETS_512_D * v_out_stride;
    }
    // tail cases
    if (remaining_sets >= NUM_SETS_256_D)
    {
        __m256d v_in0, v_in1, v_in2, v_in3, v_in4, v_in5, v_in6, v_in7, v_in8,
                v_in9, v_in10, v_in11;
        __m256d v_av1, v_av2, v_av3, v_av4, v_av5, v_av6, v_av7, v_av8, v_av9,
                v_av10, v_av11, v_av12;
        __m256d v_out0, v_out1, v_out2, v_out3, v_out4, v_out5, v_out6, v_out7,
                v_out8, v_out9, v_out10, v_out11;
        __m256d v_cv1, v_cv2, v_cv3, v_cv4, v_cv5, v_cv6, v_cv7, v_cv8;
        __m256d v_tv1, v_tv2, v_tv3, v_tv4, v_tv5;
        __m256d v_K1 = _mm512_castpd512_pd256(v_C1);
        __m256d v_K2 = _mm512_castpd512_pd256(v_C2);

        __m256d v_conj_256 = _mm512_castpd512_pd256(_conj_512_d.d);
        __m256d v_neg_256 = _mm512_castpd512_pd256(_neg_512_d[flag].d);
        __m256d v_sign_conj_256 = _mm256_xor_pd(v_neg_256, v_conj_256);

        __m256d v_K4_conj =
            _mm256_xor_pd(_mm256_set1_pd(CRTM_12[0]), v_conj_256);
        v_K4_conj = _mm256_xor_pd(v_K4_conj, v_neg_256);
        __m256d v_K5_conj =
            _mm256_xor_pd(_mm256_set1_pd(CRTM_12[1]), v_conj_256);
        v_K5_conj = _mm256_xor_pd(v_K5_conj, v_neg_256);

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

        // Common operations
        v_av1 = _mm256_add_pd(v_in0, v_in6);
        v_av2 = _mm256_add_pd(v_in2, v_in4);
        v_av3 = _mm256_add_pd(v_in8, v_in10);
        v_av4 = _mm256_add_pd(v_in1, v_in5);
        v_av5 = _mm256_add_pd(v_in7, v_in11);
        v_av6 = _mm256_add_pd(v_in3, v_in9);

        v_cv1 = _mm256_add_pd(v_av2, v_av3);
        v_cv2 = _mm256_add_pd(v_av4, v_av5);
        v_cv3 = _mm256_add_pd(v_av1, v_av6);
        v_cv4 = _mm256_sub_pd(v_av1, v_av6);

        v_cv5 = _mm256_add_pd(v_cv1, v_cv2);
        v_cv6 = _mm256_sub_pd(v_cv1, v_cv2);

        // output point 1 & 7
        v_out0 = _mm256_add_pd(v_cv3, v_cv5);
        v_out6 = _mm256_add_pd(v_cv4, v_cv6);

        // Common operations
        v_av7 = _mm256_sub_pd(v_in0, v_in6);
        v_av8 = _mm256_sub_pd(v_in2, v_in4);
        v_av9 = _mm256_sub_pd(v_in8, v_in10);
        v_av10 = _mm256_sub_pd(v_in1, v_in5);
        v_av11 = _mm256_sub_pd(v_in7, v_in11);
        v_av12 = _mm256_sub_pd(v_in3, v_in9);

        v_cv1 = _mm256_sub_pd(v_av8, v_av9);
        v_cv2 = _mm256_sub_pd(v_av4, v_av5);
        v_cv7 = _mm256_sub_pd(v_cv2, v_av12);
        v_cv8 = _mm256_sub_pd(v_av7, v_cv1);

        v_tv1 = _mm256_xor_pd(v_cv7, v_sign_conj_256);
        v_tv1 = SWAP_RI_256_D(v_tv1);

        // output point 4 & 10
        v_out3 = _mm256_sub_pd(v_cv8, v_tv1);
        v_out9 = _mm256_add_pd(v_cv8, v_tv1);

        v_tv1 = _mm256_xor_pd(v_av12, v_sign_conj_256);
        v_tv2 = _mm256_mul_pd(v_K2, v_cv1);
        v_cv1 = _mm256_add_pd(v_av7, v_tv2);
        v_cv7 = _mm256_sub_pd(v_av10, v_av11);

        v_tv3 = _mm256_mul_pd(v_K1, v_cv7);
        v_cv8 = _mm256_add_pd(v_cv1, v_tv3);

        const __m256d cv2_save = v_cv2;
        v_cv2 = _mm256_sub_pd(v_av2, v_av3);
        v_tv5 = _mm256_mul_pd(v_cv2, v_K4_conj);

        v_tv4 = _mm256_mul_pd(cv2_save, v_K5_conj);
        v_cv2 = _mm256_add_pd(v_tv1, v_tv4);
        v_cv7 = _mm256_add_pd(v_cv2, v_tv5);
        v_cv7 = SWAP_RI_256_D(v_cv7);

        // output point 2 & 12
        v_out1 = _mm256_sub_pd(v_cv8, v_cv7);
        v_out11 = _mm256_add_pd(v_cv8, v_cv7);

        v_cv7 = _mm256_sub_pd(v_cv1, v_tv3);
        v_cv8 = _mm256_sub_pd(v_cv2, v_tv5);
        v_cv8 = SWAP_RI_256_D(v_cv8);

        // output point 6 & 8
        v_out5 = _mm256_sub_pd(v_cv7, v_cv8);
        v_out7 = _mm256_add_pd(v_cv7, v_cv8);

        v_tv1 = _mm256_mul_pd(v_K2, v_cv6);
        v_cv1 = _mm256_sub_pd(v_cv4, v_tv1);
        v_cv2 = _mm256_add_pd(v_av8, v_av9);
        v_cv4 = _mm256_add_pd(v_av10, v_av11);
        v_cv6 = _mm256_add_pd(v_cv4, v_cv2);
        v_tv2 = _mm256_mul_pd(v_cv6, v_K4_conj);
        v_tv2 = SWAP_RI_256_D(v_tv2);

        // output point 3 & 11
        v_out2 = _mm256_sub_pd(v_cv1, v_tv2);
        v_out10 = _mm256_add_pd(v_cv1, v_tv2);

        v_tv1 = _mm256_mul_pd(v_K2, v_cv5);
        v_cv1 = _mm256_sub_pd(v_cv3, v_tv1);
        v_cv6 = _mm256_sub_pd(v_cv4, v_cv2);
        v_tv2 = _mm256_mul_pd(v_cv6, v_K4_conj);
        v_tv2 = SWAP_RI_256_D(v_tv2);

        // output point 5 & 9
        v_out4 = _mm256_sub_pd(v_cv1, v_tv2);
        v_out8 = _mm256_add_pd(v_cv1, v_tv2);

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
        }

        in_r += NUM_SETS_256_D * v_in_stride;
        out_r += NUM_SETS_256_D * v_out_stride;
    }
    // tail case
    if (n & 1)
    {
        __m128d v_in0, v_in1, v_in2, v_in3, v_in4, v_in5, v_in6, v_in7, v_in8,
                v_in9, v_in10, v_in11;
        __m128d v_av1, v_av2, v_av3, v_av4, v_av5, v_av6, v_av7, v_av8, v_av9,
                v_av10, v_av11, v_av12;
        __m128d v_out0, v_out1, v_out2, v_out3, v_out4, v_out5, v_out6, v_out7,
                v_out8, v_out9, v_out10, v_out11;
        __m128d v_cv1, v_cv2, v_cv3, v_cv4, v_cv5, v_cv6, v_cv7, v_cv8;
        __m128d v_tv1, v_tv2, v_tv3, v_tv4, v_tv5;

        __m128d v_K1 = _mm512_castpd512_pd128(v_C1);
        __m128d v_K2 = _mm512_castpd512_pd128(v_C2);

        __m128d v_conj_128 = _mm512_castpd512_pd128(_conj_512_d.d);
        __m128d v_neg_128 = _mm512_castpd512_pd128(_neg_512_d[flag].d);
        __m128d v_sign_conj_128 = _mm_xor_pd(v_neg_128, v_conj_128);

        __m128d v_K4_conj = _mm_xor_pd(_mm_set1_pd(CRTM_12[0]), v_conj_128);
        v_K4_conj = _mm_xor_pd(v_K4_conj, v_neg_128);
        __m128d v_K5_conj = _mm_xor_pd(_mm_set1_pd(CRTM_12[1]), v_conj_128);
        v_K5_conj = _mm_xor_pd(v_K5_conj, v_neg_128);

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

        // Common operations
        v_av1 = _mm_add_pd(v_in0, v_in6);
        v_av2 = _mm_add_pd(v_in2, v_in4);
        v_av3 = _mm_add_pd(v_in8, v_in10);
        v_av4 = _mm_add_pd(v_in1, v_in5);
        v_av5 = _mm_add_pd(v_in7, v_in11);
        v_av6 = _mm_add_pd(v_in3, v_in9);

        v_cv1 = _mm_add_pd(v_av2, v_av3);
        v_cv2 = _mm_add_pd(v_av4, v_av5);
        v_cv3 = _mm_add_pd(v_av1, v_av6);
        v_cv4 = _mm_sub_pd(v_av1, v_av6);

        v_cv5 = _mm_add_pd(v_cv1, v_cv2);
        v_cv6 = _mm_sub_pd(v_cv1, v_cv2);

        // output point 1 & 7
        v_out0 = _mm_add_pd(v_cv3, v_cv5);
        v_out6 = _mm_add_pd(v_cv4, v_cv6);

        // Common operations
        v_av7 = _mm_sub_pd(v_in0, v_in6);
        v_av8 = _mm_sub_pd(v_in2, v_in4);
        v_av9 = _mm_sub_pd(v_in8, v_in10);
        v_av10 = _mm_sub_pd(v_in1, v_in5);
        v_av11 = _mm_sub_pd(v_in7, v_in11);
        v_av12 = _mm_sub_pd(v_in3, v_in9);

        v_cv1 = _mm_sub_pd(v_av8, v_av9);
        v_cv2 = _mm_sub_pd(v_av4, v_av5);
        v_cv7 = _mm_sub_pd(v_cv2, v_av12);
        v_cv8 = _mm_sub_pd(v_av7, v_cv1);

        v_tv1 = _mm_xor_pd(v_cv7, v_sign_conj_128);
        v_tv1 = SWAP_RI_128_D(v_tv1);

        // output point 4 & 10
        v_out3 = _mm_sub_pd(v_cv8, v_tv1);
        v_out9 = _mm_add_pd(v_cv8, v_tv1);

        v_tv1 = _mm_xor_pd(v_av12, v_sign_conj_128);
        v_tv2 = _mm_mul_pd(v_K2, v_cv1);
        v_cv1 = _mm_add_pd(v_av7, v_tv2);
        v_cv7 = _mm_sub_pd(v_av10, v_av11);

        v_tv3 = _mm_mul_pd(v_K1, v_cv7);
        v_cv8 = _mm_add_pd(v_cv1, v_tv3);

        const __m128d cv2_save = v_cv2;
        v_cv2 = _mm_sub_pd(v_av2, v_av3);
        v_tv5 = _mm_mul_pd(v_cv2, v_K4_conj);

        v_tv4 = _mm_mul_pd(cv2_save, v_K5_conj);
        v_cv2 = _mm_add_pd(v_tv1, v_tv4);
        v_cv7 = _mm_add_pd(v_cv2, v_tv5);
        v_cv7 = SWAP_RI_128_D(v_cv7);

        // output point 2 & 12
        v_out1 = _mm_sub_pd(v_cv8, v_cv7);
        v_out11 = _mm_add_pd(v_cv8, v_cv7);

        v_cv7 = _mm_sub_pd(v_cv1, v_tv3);
        v_cv8 = _mm_sub_pd(v_cv2, v_tv5);
        v_cv8 = SWAP_RI_128_D(v_cv8);

        // output point 6 & 8
        v_out5 = _mm_sub_pd(v_cv7, v_cv8);
        v_out7 = _mm_add_pd(v_cv7, v_cv8);

        v_tv1 = _mm_mul_pd(v_K2, v_cv6);
        v_cv1 = _mm_sub_pd(v_cv4, v_tv1);
        v_cv2 = _mm_add_pd(v_av8, v_av9);
        v_cv4 = _mm_add_pd(v_av10, v_av11);
        v_cv6 = _mm_add_pd(v_cv4, v_cv2);
        v_tv2 = _mm_mul_pd(v_cv6, v_K4_conj);
        v_tv2 = SWAP_RI_128_D(v_tv2);

        // output point 3 & 11
        v_out2 = _mm_sub_pd(v_cv1, v_tv2);
        v_out10 = _mm_add_pd(v_cv1, v_tv2);

        v_tv1 = _mm_mul_pd(v_K2, v_cv5);
        v_cv1 = _mm_sub_pd(v_cv3, v_tv1);
        v_cv6 = _mm_sub_pd(v_cv4, v_cv2);
        v_tv2 = _mm_mul_pd(v_cv6, v_K4_conj);
        v_tv2 = SWAP_RI_128_D(v_tv2);

        // output point 5 & 9
        v_out4 = _mm_sub_pd(v_cv1, v_tv2);
        v_out8 = _mm_add_pd(v_cv1, v_tv2);

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
    }
    AOCLFFTZ_LOG(DEBUG, global_logger_mode, "Exit");
}

kfft_ register_kernel_fft12avx512(FFTZ_UINT8 precision,
                                  FFTZ_UINT8 direction /* unused */)
{
    if (precision == DT_FLOAT)
    {
        return fft12avx512fp32;
    }
    else if (precision == DT_DOUBLE)
    {
        return fft12avx512fp64;
    }
    else
    {
        return NULL;
    }
}
