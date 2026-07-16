// Copyright Advanced Micro Devices, Inc.
// SPDX-License-Identifier: BSD-3-Clause

/** @file fft12avx128.c
 *
 *  @brief Radix-12 FFT kernel with avx128 operations using x86 SIMD intrinsics
 *
 *  This file contains the DIT radix-12 FFT implementations using avx128
 *  operations for single-precision and double-precision inputs.
 *
 *  @author Murugan Vairavel
 *  @author Srirammaswamy S
 *
 */

#include "core/kernels/kernel.h"
#include "core/kernels/simd_includes/simd_common.h"

static const ops_cycles_t ops_cnt[NUM_PRECISIONS] = {{0, 10, 48, 48, 5, 6},
                                                     {0, 10, 48, 24, 5, 6}};

ops_cycles_t get_ops_cnt_fft12avx128(UINT8 precision, UINT8 direction)
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

static VOID fft12avx128fp32(VOID *in_real, VOID *in_imag, VOID *out_real,
                            VOID *out_imag, INTP n, aoclfftz_strides_t *strides,
                            VOID *twd, UINT8 flag)
{
    AOCLFFTZ_LOG(DEBUG, global_logger_mode, "Enter");
    const FLOAT CRTM_12[5] = {
        0.86602540378443864676372317075293618347140262700000,
        0.50000000000000000000000000000000000000000000000000,
        1.00000000000000000000000000000000000000000000000000,
        0.86602540378443864676372317075293618347140262700000,
        0.50000000000000000000000000000000000000000000000000};

    FLOAT *in_r = (FLOAT *)in_real;
    FLOAT *out_r = (FLOAT *)out_real;
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
    FLOAT *curr_set;

    __m128 v_in0, v_in1, v_in2, v_in3, v_in4, v_in5, v_in6, v_in7, v_in8, v_in9,
           v_in10, v_in11;
    __m128 v_av1, v_av2, v_av3, v_av4, v_av5, v_av6, v_av7, v_av8, v_av9,
           v_av10, v_av11, v_av12;
    __m128 v_cv1, v_cv2, v_cv3, v_cv4, v_cv5, v_cv6, v_cv7, v_cv8;
    __m128 v_tv1, v_tv2, v_tv3, v_tv4, v_tv5;
    __m128 v_out0, v_out1, v_out2, v_out3, v_out4, v_out5, v_out6, v_out7,
           v_out8, v_out9, v_out10, v_out11;

    __m128 v_K1 = _mm_broadcast_ss(&CRTM_12[0]);
    __m128 v_K2 = _mm_broadcast_ss(&CRTM_12[1]);
    __m128 v_K3 = _mm_broadcast_ss(&CRTM_12[2]);
    v_K3 = _mm_xor_ps(v_K3, _neg_128_f[flag].s);
    __m128 v_K4 = _mm_broadcast_ss(&CRTM_12[3]);
    v_K4 = _mm_xor_ps(v_K4, _neg_128_f[flag].s);
    __m128 v_K5 = _mm_broadcast_ss(&CRTM_12[4]);
    v_K5 = _mm_xor_ps(v_K5, _neg_128_f[flag].s);

    for (count = 0; count < N; count++)
    {
        curr_set = in_r;
        GATHER2_128_S(curr_set, v_in_stride, v_in0, is_contiguous_in);
        curr_set = in_r + in_strides[1];
        GATHER2_128_S(curr_set, v_in_stride, v_in1, is_contiguous_in);
        curr_set = in_r + in_strides[2];
        GATHER2_128_S(curr_set, v_in_stride, v_in2, is_contiguous_in);
        curr_set = in_r + in_strides[3];
        GATHER2_128_S(curr_set, v_in_stride, v_in3, is_contiguous_in);
        curr_set = in_r + in_strides[4];
        GATHER2_128_S(curr_set, v_in_stride, v_in4, is_contiguous_in);
        curr_set = in_r + in_strides[5];
        GATHER2_128_S(curr_set, v_in_stride, v_in5, is_contiguous_in);
        curr_set = in_r + in_strides[6];
        GATHER2_128_S(curr_set, v_in_stride, v_in6, is_contiguous_in);
        curr_set = in_r + in_strides[7];
        GATHER2_128_S(curr_set, v_in_stride, v_in7, is_contiguous_in);
        curr_set = in_r + in_strides[8];
        GATHER2_128_S(curr_set, v_in_stride, v_in8, is_contiguous_in);
        curr_set = in_r + in_strides[9];
        GATHER2_128_S(curr_set, v_in_stride, v_in9, is_contiguous_in);
        curr_set = in_r + in_strides[10];
        GATHER2_128_S(curr_set, v_in_stride, v_in10, is_contiguous_in);
        curr_set = in_r + in_strides[11];
        GATHER2_128_S(curr_set, v_in_stride, v_in11, is_contiguous_in);

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

        curr_set = out_r;
        if (out_strides[1] == DATA_STRIDE)
        {
            TRANSPOSE_ST2_128_S(curr_set, v_out_stride, v_out0, v_out1);
            curr_set = curr_set + NUM_SETS_128_S * DATA_STRIDE;
            TRANSPOSE_ST2_128_S(curr_set, v_out_stride, v_out2, v_out3);
            curr_set = curr_set + NUM_SETS_128_S * DATA_STRIDE;
            TRANSPOSE_ST2_128_S(curr_set, v_out_stride, v_out4, v_out5);
            curr_set = curr_set + NUM_SETS_128_S * DATA_STRIDE;
            TRANSPOSE_ST2_128_S(curr_set, v_out_stride, v_out6, v_out7);
            curr_set = curr_set + NUM_SETS_128_S * DATA_STRIDE;
            TRANSPOSE_ST2_128_S(curr_set, v_out_stride, v_out8, v_out9);
            curr_set = curr_set + NUM_SETS_128_S * DATA_STRIDE;
            TRANSPOSE_ST2_128_S(curr_set, v_out_stride, v_out10, v_out11);
        }
        else
        {
            SCATTER2_128_S(curr_set, v_out_stride, v_out0, is_contiguous_out);
            curr_set = out_r + out_strides[1];
            SCATTER2_128_S(curr_set, v_out_stride, v_out1, is_contiguous_out);
            curr_set = out_r + out_strides[2];
            SCATTER2_128_S(curr_set, v_out_stride, v_out2, is_contiguous_out);
            curr_set = out_r + out_strides[3];
            SCATTER2_128_S(curr_set, v_out_stride, v_out3, is_contiguous_out);
            curr_set = out_r + out_strides[4];
            SCATTER2_128_S(curr_set, v_out_stride, v_out4, is_contiguous_out);
            curr_set = out_r + out_strides[5];
            SCATTER2_128_S(curr_set, v_out_stride, v_out5, is_contiguous_out);
            curr_set = out_r + out_strides[6];
            SCATTER2_128_S(curr_set, v_out_stride, v_out6, is_contiguous_out);
            curr_set = out_r + out_strides[7];
            SCATTER2_128_S(curr_set, v_out_stride, v_out7, is_contiguous_out);
            curr_set = out_r + out_strides[8];
            SCATTER2_128_S(curr_set, v_out_stride, v_out8, is_contiguous_out);
            curr_set = out_r + out_strides[9];
            SCATTER2_128_S(curr_set, v_out_stride, v_out9, is_contiguous_out);
            curr_set = out_r + out_strides[10];
            SCATTER2_128_S(curr_set, v_out_stride, v_out10, is_contiguous_out);
            curr_set = out_r + out_strides[11];
            SCATTER2_128_S(curr_set, v_out_stride, v_out11, is_contiguous_out);
        }

        in_r += NUM_SETS_128_S * v_in_stride;
        out_r += NUM_SETS_128_S * v_out_stride;
    }

    // tail case
    if (n & 1)
    {
        curr_set = in_r;
        LD_LOW_128_S(curr_set, v_in0);
        curr_set = in_r + in_strides[1];
        LD_LOW_128_S(curr_set, v_in1);
        curr_set = in_r + in_strides[2];
        LD_LOW_128_S(curr_set, v_in2);
        curr_set = in_r + in_strides[3];
        LD_LOW_128_S(curr_set, v_in3);
        curr_set = in_r + in_strides[4];
        LD_LOW_128_S(curr_set, v_in4);
        curr_set = in_r + in_strides[5];
        LD_LOW_128_S(curr_set, v_in5);
        curr_set = in_r + in_strides[6];
        LD_LOW_128_S(curr_set, v_in6);
        curr_set = in_r + in_strides[7];
        LD_LOW_128_S(curr_set, v_in7);
        curr_set = in_r + in_strides[8];
        LD_LOW_128_S(curr_set, v_in8);
        curr_set = in_r + in_strides[9];
        LD_LOW_128_S(curr_set, v_in9);
        curr_set = in_r + in_strides[10];
        LD_LOW_128_S(curr_set, v_in10);
        curr_set = in_r + in_strides[11];
        LD_LOW_128_S(curr_set, v_in11);

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

        curr_set = out_r;
        if (out_strides[1] == DATA_STRIDE)
        {
            ST_128_S(curr_set, v_out0, v_out1);
            curr_set = out_r + 2 * DATA_STRIDE;
            ST_128_S(curr_set, v_out2, v_out3);
            curr_set = out_r + 4 * DATA_STRIDE;
            ST_128_S(curr_set, v_out4, v_out5);
            curr_set = out_r + 6 * DATA_STRIDE;
            ST_128_S(curr_set, v_out6, v_out7);
            curr_set = out_r + 8 * DATA_STRIDE;
            ST_128_S(curr_set, v_out8, v_out9);
            curr_set = out_r + 10 * DATA_STRIDE;
            ST_128_S(curr_set, v_out10, v_out11);
        }
        else
        {
            ST_LOW_128_S(curr_set, v_out0);
            curr_set = out_r + out_strides[1];
            ST_LOW_128_S(curr_set, v_out1);
            curr_set = out_r + out_strides[2];
            ST_LOW_128_S(curr_set, v_out2);
            curr_set = out_r + out_strides[3];
            ST_LOW_128_S(curr_set, v_out3);
            curr_set = out_r + out_strides[4];
            ST_LOW_128_S(curr_set, v_out4);
            curr_set = out_r + out_strides[5];
            ST_LOW_128_S(curr_set, v_out5);
            curr_set = out_r + out_strides[6];
            ST_LOW_128_S(curr_set, v_out6);
            curr_set = out_r + out_strides[7];
            ST_LOW_128_S(curr_set, v_out7);
            curr_set = out_r + out_strides[8];
            ST_LOW_128_S(curr_set, v_out8);
            curr_set = out_r + out_strides[9];
            ST_LOW_128_S(curr_set, v_out9);
            curr_set = out_r + out_strides[10];
            ST_LOW_128_S(curr_set, v_out10);
            curr_set = out_r + out_strides[11];
            ST_LOW_128_S(curr_set, v_out11);
        }
    }
    AOCLFFTZ_LOG(DEBUG, global_logger_mode, "Exit");
}

static VOID fft12avx128fp64(VOID *in_real, VOID *in_imag, VOID *out_real,
                            VOID *out_imag, INTP n, aoclfftz_strides_t *strides,
                            VOID *twd, UINT8 flag)
{
    AOCLFFTZ_LOG(DEBUG, global_logger_mode, "Enter");
    const DOUBLE CRTM_12[5] = {
        0.86602540378443864676372317075293618347140262700000,
        0.50000000000000000000000000000000000000000000000000,
        1.00000000000000000000000000000000000000000000000000,
        0.86602540378443864676372317075293618347140262700000,
        0.50000000000000000000000000000000000000000000000000};

    DOUBLE *in_r = (DOUBLE *)in_real;
    DOUBLE *out_r = (DOUBLE *)out_real;
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
    DOUBLE *curr_set;

    __m128d v_in0, v_in1, v_in2, v_in3, v_in4, v_in5, v_in6, v_in7, v_in8,
            v_in9, v_in10, v_in11;
    __m128d v_av1, v_av2, v_av3, v_av4, v_av5, v_av6, v_av7, v_av8, v_av9,
            v_av10, v_av11, v_av12;
    __m128d v_out0, v_out1, v_out2, v_out3, v_out4, v_out5, v_out6, v_out7,
            v_out8, v_out9, v_out10, v_out11;
    __m128d v_cv1, v_cv2, v_cv3, v_cv4, v_cv5, v_cv6, v_cv7, v_cv8;
    __m128d v_tv1, v_tv2, v_tv3, v_tv4, v_tv5;

    __m128d v_K1 = _mm_set1_pd(CRTM_12[0]);
    __m128d v_K2 = _mm_set1_pd(CRTM_12[1]);
    __m128d v_K3 = _mm_set1_pd(CRTM_12[2]);
    v_K3 = _mm_xor_pd(v_K3, _neg_128_d[flag].d);
    __m128d v_K4 = _mm_set1_pd(CRTM_12[3]);
    v_K4 = _mm_xor_pd(v_K4, _neg_128_d[flag].d);
    __m128d v_K5 = _mm_set1_pd(CRTM_12[4]);
    v_K5 = _mm_xor_pd(v_K5, _neg_128_d[flag].d);

    for (count = 0; count < N; count++)
    {
        curr_set = in_r;
        LD_128_D(curr_set, v_in0);
        curr_set = in_r + in_strides[1];
        LD_128_D(curr_set, v_in1);
        curr_set = in_r + in_strides[2];
        LD_128_D(curr_set, v_in2);
        curr_set = in_r + in_strides[3];
        LD_128_D(curr_set, v_in3);
        curr_set = in_r + in_strides[4];
        LD_128_D(curr_set, v_in4);
        curr_set = in_r + in_strides[5];
        LD_128_D(curr_set, v_in5);
        curr_set = in_r + in_strides[6];
        LD_128_D(curr_set, v_in6);
        curr_set = in_r + in_strides[7];
        LD_128_D(curr_set, v_in7);
        curr_set = in_r + in_strides[8];
        LD_128_D(curr_set, v_in8);
        curr_set = in_r + in_strides[9];
        LD_128_D(curr_set, v_in9);
        curr_set = in_r + in_strides[10];
        LD_128_D(curr_set, v_in10);
        curr_set = in_r + in_strides[11];
        LD_128_D(curr_set, v_in11);

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

        v_tv1 = _mm_mul_pd(v_K3, v_cv7);
        v_tv1 = SWAP_RI_128_D(CONJ_128_D(v_tv1));

        // output point 4 & 10
        v_out3 = _mm_sub_pd(v_cv8, v_tv1);
        v_out9 = _mm_add_pd(v_cv8, v_tv1);

        v_tv1 = _mm_mul_pd(v_K3, v_av12);
        v_tv1 = CONJ_128_D(v_tv1);

        v_tv2 = _mm_mul_pd(v_K2, v_cv1);
        v_cv1 = _mm_add_pd(v_av7, v_tv2);
        v_cv7 = _mm_sub_pd(v_av10, v_av11);

        v_tv3 = _mm_mul_pd(v_K1, v_cv7);
        v_cv8 = _mm_add_pd(v_cv1, v_tv3);

        v_tv4 = _mm_mul_pd(v_K5, v_cv2);
        v_tv4 = CONJ_128_D(v_tv4);

        v_cv2 = _mm_sub_pd(v_av2, v_av3);
        v_tv5 = _mm_mul_pd(v_K4, v_cv2);
        v_tv5 = CONJ_128_D(v_tv5);

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
        v_tv2 = _mm_mul_pd(v_K4, v_cv6);
        v_tv2 = SWAP_RI_128_D(CONJ_128_D(v_tv2));

        // output point 3 & 11
        v_out2 = _mm_sub_pd(v_cv1, v_tv2);
        v_out10 = _mm_add_pd(v_cv1, v_tv2);

        v_tv1 = _mm_mul_pd(v_K2, v_cv5);
        v_cv1 = _mm_sub_pd(v_cv3, v_tv1);
        v_cv6 = _mm_sub_pd(v_cv4, v_cv2);
        v_tv2 = _mm_mul_pd(v_K4, v_cv6);
        v_tv2 = SWAP_RI_128_D(CONJ_128_D(v_tv2));

        // output point 5 & 9
        v_out4 = _mm_sub_pd(v_cv1, v_tv2);
        v_out8 = _mm_add_pd(v_cv1, v_tv2);

        curr_set = out_r;
        ST_128_D(curr_set, v_out0);
        curr_set = out_r + out_strides[1];
        ST_128_D(curr_set, v_out1);
        curr_set = out_r + out_strides[2];
        ST_128_D(curr_set, v_out2);
        curr_set = out_r + out_strides[3];
        ST_128_D(curr_set, v_out3);
        curr_set = out_r + out_strides[4];
        ST_128_D(curr_set, v_out4);
        curr_set = out_r + out_strides[5];
        ST_128_D(curr_set, v_out5);
        curr_set = out_r + out_strides[6];
        ST_128_D(curr_set, v_out6);
        curr_set = out_r + out_strides[7];
        ST_128_D(curr_set, v_out7);
        curr_set = out_r + out_strides[8];
        ST_128_D(curr_set, v_out8);
        curr_set = out_r + out_strides[9];
        ST_128_D(curr_set, v_out9);
        curr_set = out_r + out_strides[10];
        ST_128_D(curr_set, v_out10);
        curr_set = out_r + out_strides[11];
        ST_128_D(curr_set, v_out11);

        in_r += NUM_SETS_128_D * v_in_stride;
        out_r += NUM_SETS_128_D * v_out_stride;
    }
    AOCLFFTZ_LOG(DEBUG, global_logger_mode, "Exit");
}

kfft_ register_kernel_fft12avx128(UINT8 precision, UINT8 direction /* unused */)
{
    if (precision == DT_FLOAT)
    {
        return fft12avx128fp32;
    }
    else if (precision == DT_DOUBLE)
    {
        return fft12avx128fp64;
    }
    else
    {
        return NULL;
    }
}
