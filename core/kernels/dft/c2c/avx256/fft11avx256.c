// Copyright Advanced Micro Devices, Inc.
// SPDX-License-Identifier: BSD-3-Clause

/** @file fft11avx256.c
 *
 *  @brief Radix-11 FFT kernel with AVX-256 operations using x86 SIMD intrinsics
 *
 *  This file contains the DIT radix-11 FFT implementations using AVX-256
 *  operations for single-precision and double-precision inputs.
 *
 *  @author Jeya R
 *
 */

#include "core/kernels/kernel.h"
#include "core/kernels/simd_includes/simd_common.h"

static const ops_cycles_t ops_cnt[NUM_PRECISIONS] = {{0, 50, 70, 88, 16, 27},
                                                     {0, 50, 70, 44,  5, 27}};

ops_cycles_t get_ops_cnt_fft11avx256(FFTZ_UINT8 precision, FFTZ_UINT8 direction)
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

static FFTZ_VOID fft11avx256fp32(FFTZ_VOID *in_real, FFTZ_VOID *in_imag,
                                 FFTZ_VOID *out_real, FFTZ_VOID *out_imag,
                                 FFTZ_INTP n, aoclfftz_strides_t *strides,
                                 FFTZ_VOID *twd, FFTZ_UINT8 flag)
{
    AOCLFFTZ_LOG(DEBUG, global_logger_mode, "Enter");
    const FFTZ_FLOAT CRTM_11[10] = {
        0.84125353283118116029052039464203089547681594330064,
        0.54064081745559759544482548159299693174139803024473,
        0.41541501300188639668675795488636098054966524290126,
        0.90963199535451838458365117807108162835411650732265,
        0.14231483827328501490317354898047094957684096668515,
        0.98982144188093275042610808187068914262031166769031,
        0.65486073394528511198338203198719613618953603946564,
        0.75574957435425824224552448923467521721665586591805,
        0.95949297361449738989036805706632769906245484800000,
        0.28173255684142978898192655345478532989004751779983};

    FFTZ_FLOAT *in_r = (FFTZ_FLOAT *)in_real;
    FFTZ_FLOAT *out_r = (FFTZ_FLOAT *)out_real;
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
    FFTZ_INTP N = n / NUM_SETS_256_S;
    FFTZ_INTP remaining_sets = n % NUM_SETS_256_S;
    FFTZ_INTP count;
    FFTZ_FLOAT *curr_in, *curr_out;

    __m256 v_in0, v_in1, v_in2, v_in3, v_in4, v_in5, v_in6, v_in7, v_in8,
           v_in9, v_in10;
    __m256 v_av1, v_av2, v_av3, v_av4, v_av5, v_av6, v_av7, v_av8, v_av9,
           v_av10, v_tv1;
    __m256 v_tv2, v_tv3, v_tv4, v_tv5, v_av11, v_av12;
    __m256 v_out0, v_out1, v_out2, v_out3, v_out4, v_out5, v_out6, v_out7,
           v_out8, v_out9, v_out10;

    __m256 v_C1 = _mm256_broadcast_ss(&CRTM_11[0]);
    __m256 v_C2 = _mm256_broadcast_ss(&CRTM_11[1]);
    v_C2 = _mm256_xor_ps(v_C2, _neg_256_f[flag].s);
    __m256 v_C3 = _mm256_broadcast_ss(&CRTM_11[2]);
    __m256 v_C4 = _mm256_broadcast_ss(&CRTM_11[3]);
    v_C4 = _mm256_xor_ps(v_C4, _neg_256_f[flag].s);
    __m256 v_C5 = _mm256_broadcast_ss(&CRTM_11[4]);
    __m256 v_C6 = _mm256_broadcast_ss(&CRTM_11[5]);
    v_C6 = _mm256_xor_ps(v_C6, _neg_256_f[flag].s);
    __m256 v_C7 = _mm256_broadcast_ss(&CRTM_11[6]);
    __m256 v_C8 = _mm256_broadcast_ss(&CRTM_11[7]);
    v_C8 = _mm256_xor_ps(v_C8, _neg_256_f[flag].s);
    __m256 v_C9 = _mm256_broadcast_ss(&CRTM_11[8]);
    __m256 v_C10 = _mm256_broadcast_ss(&CRTM_11[9]);
    v_C10 = _mm256_xor_ps(v_C10, _neg_256_f[flag].s);

    for (count = 0; count < N; count++)
    {
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

        // common calculations
        v_av1 = _mm256_add_ps(v_in1, v_in10);
        v_av2 = _mm256_add_ps(v_in2, v_in9);
        v_av3 = _mm256_add_ps(v_in3, v_in8);
        v_av4 = _mm256_add_ps(v_in4, v_in7);
        v_av5 = _mm256_add_ps(v_in5, v_in6);
        v_av6 = _mm256_sub_ps(v_in10, v_in1);
        v_av7 = _mm256_sub_ps(v_in9, v_in2);
        v_av8 = _mm256_sub_ps(v_in8, v_in3);
        v_av9 = _mm256_sub_ps(v_in7, v_in4);
        v_av10 = _mm256_sub_ps(v_in6, v_in5);

        v_tv1 = _mm256_mul_ps(v_C1, v_av1);
        v_tv2 = _mm256_mul_ps(v_C3, v_av2);
        v_tv3 = _mm256_mul_ps(v_C5, v_av3);
        v_tv4 = _mm256_mul_ps(v_C7, v_av4);
        v_tv5 = _mm256_mul_ps(v_C9, v_av5);

        v_av11 = _mm256_add_ps(v_in0, v_tv1);
        v_av11 = _mm256_add_ps(v_av11, v_tv2);
        v_av11 = _mm256_sub_ps(v_av11, v_tv3);
        v_av11 = _mm256_sub_ps(v_av11, v_tv4);
        v_av11 = _mm256_sub_ps(v_av11, v_tv5);

        v_tv1 = _mm256_mul_ps(v_C2, v_av6);
        v_tv2 = _mm256_mul_ps(v_C4, v_av7);
        v_tv3 = _mm256_mul_ps(v_C6, v_av8);
        v_tv4 = _mm256_mul_ps(v_C8, v_av9);
        v_tv5 = _mm256_mul_ps(v_C10, v_av10);

        v_av12 = _mm256_add_ps(v_tv1, v_tv2);
        v_av12 = _mm256_add_ps(v_av12, v_tv3);
        v_av12 = _mm256_add_ps(v_av12, v_tv4);
        v_av12 = _mm256_add_ps(v_av12, v_tv5);
        v_av12 = SWAP_RI_256_S(CONJ_256_S(v_av12));

        // Output point 2:X[1]
        v_out1 = _mm256_add_ps(v_av11, v_av12);
        // Output point 11:X[10]
        v_out10 = _mm256_sub_ps(v_av11, v_av12);

        v_tv1 = _mm256_mul_ps(v_C3, v_av1);
        v_tv2 = _mm256_mul_ps(v_C7, v_av2);
        v_tv3 = _mm256_mul_ps(v_C9, v_av3);
        v_tv4 = _mm256_mul_ps(v_C5, v_av4);
        v_tv5 = _mm256_mul_ps(v_C1, v_av5);

        v_av11 = _mm256_add_ps(v_in0, v_tv1);
        v_av11 = _mm256_sub_ps(v_av11, v_tv2);
        v_av11 = _mm256_sub_ps(v_av11, v_tv3);
        v_av11 = _mm256_sub_ps(v_av11, v_tv4);
        v_av11 = _mm256_add_ps(v_av11, v_tv5);

        v_tv1 = _mm256_mul_ps(v_C4, v_av6);
        v_tv2 = _mm256_mul_ps(v_C8, v_av7);
        v_tv3 = _mm256_mul_ps(v_C10, v_av8);
        v_tv4 = _mm256_mul_ps(v_C6, v_av9);
        v_tv5 = _mm256_mul_ps(v_C2, v_av10);

        v_av12 = _mm256_add_ps(v_tv1, v_tv2);
        v_av12 = _mm256_sub_ps(v_av12, v_tv3);
        v_av12 = _mm256_sub_ps(v_av12, v_tv4);
        v_av12 = _mm256_sub_ps(v_av12, v_tv5);
        v_av12 = SWAP_RI_256_S(CONJ_256_S(v_av12));

        // Output point 3:X[2]
        v_out2 = _mm256_add_ps(v_av11, v_av12);
        // Output point 10:X[9]
        v_out9 = _mm256_sub_ps(v_av11, v_av12);

        v_tv1 = _mm256_mul_ps(v_C5, v_av1);
        v_tv2 = _mm256_mul_ps(v_C9, v_av2);
        v_tv3 = _mm256_mul_ps(v_C3, v_av3);
        v_tv4 = _mm256_mul_ps(v_C1, v_av4);
        v_tv5 = _mm256_mul_ps(v_C7, v_av5);

        v_av11 = _mm256_sub_ps(v_in0, v_tv1);
        v_av11 = _mm256_sub_ps(v_av11, v_tv2);
        v_av11 = _mm256_add_ps(v_av11, v_tv3);
        v_av11 = _mm256_add_ps(v_av11, v_tv4);
        v_av11 = _mm256_sub_ps(v_av11, v_tv5);

        v_tv1 = _mm256_mul_ps(v_C6, v_av6);
        v_tv2 = _mm256_mul_ps(v_C10, v_av7);
        v_tv3 = _mm256_mul_ps(v_C4, v_av8);
        v_tv4 = _mm256_mul_ps(v_C2, v_av9);
        v_tv5 = _mm256_mul_ps(v_C8, v_av10);

        v_av12 = _mm256_sub_ps(v_tv1, v_tv2);
        v_av12 = _mm256_sub_ps(v_av12, v_tv3);
        v_av12 = _mm256_add_ps(v_av12, v_tv4);
        v_av12 = _mm256_add_ps(v_av12, v_tv5);
        v_av12 = SWAP_RI_256_S(CONJ_256_S(v_av12));

        // Output point 4:X[3]
        v_out3 = _mm256_add_ps(v_av11, v_av12);
        // Output point 9:X[8]
        v_out8 = _mm256_sub_ps(v_av11, v_av12);

        v_tv1 = _mm256_mul_ps(v_C7, v_av1);
        v_tv2 = _mm256_mul_ps(v_C5, v_av2);
        v_tv3 = _mm256_mul_ps(v_C1, v_av3);
        v_tv4 = _mm256_mul_ps(v_C9, v_av4);
        v_tv5 = _mm256_mul_ps(v_C3, v_av5);

        v_av11 = _mm256_sub_ps(v_in0, v_tv1);
        v_av11 = _mm256_sub_ps(v_av11, v_tv2);
        v_av11 = _mm256_add_ps(v_av11, v_tv3);
        v_av11 = _mm256_sub_ps(v_av11, v_tv4);
        v_av11 = _mm256_add_ps(v_av11, v_tv5);

        v_tv1 = _mm256_mul_ps(v_C8, v_av6);
        v_tv2 = _mm256_mul_ps(v_C6, v_av7);
        v_tv3 = _mm256_mul_ps(v_C2, v_av8);
        v_tv4 = _mm256_mul_ps(v_C10, v_av9);
        v_tv5 = _mm256_mul_ps(v_C4, v_av10);

        v_av12 = _mm256_sub_ps(v_tv1, v_tv2);
        v_av12 = _mm256_add_ps(v_av12, v_tv3);
        v_av12 = _mm256_add_ps(v_av12, v_tv4);
        v_av12 = _mm256_sub_ps(v_av12, v_tv5);
        v_av12 = SWAP_RI_256_S(CONJ_256_S(v_av12));

        // Output point 5:X[4]
        v_out4 = _mm256_add_ps(v_av11, v_av12);
        // Output point 8:X[7]
        v_out7 = _mm256_sub_ps(v_av11, v_av12);

        v_tv1 = _mm256_mul_ps(v_C9, v_av1);
        v_tv2 = _mm256_mul_ps(v_C1, v_av2);
        v_tv3 = _mm256_mul_ps(v_C7, v_av3);
        v_tv4 = _mm256_mul_ps(v_C3, v_av4);
        v_tv5 = _mm256_mul_ps(v_C5, v_av5);

        v_av11 = _mm256_sub_ps(v_in0, v_tv1);
        v_av11 = _mm256_add_ps(v_av11, v_tv2);
        v_av11 = _mm256_sub_ps(v_av11, v_tv3);
        v_av11 = _mm256_add_ps(v_av11, v_tv4);
        v_av11 = _mm256_sub_ps(v_av11, v_tv5);

        v_tv1 = _mm256_mul_ps(v_C10, v_av6);
        v_tv2 = _mm256_mul_ps(v_C2, v_av7);
        v_tv3 = _mm256_mul_ps(v_C8, v_av8);
        v_tv4 = _mm256_mul_ps(v_C4, v_av9);
        v_tv5 = _mm256_mul_ps(v_C6, v_av10);

        v_av12 = _mm256_sub_ps(v_tv1, v_tv2);
        v_av12 = _mm256_add_ps(v_av12, v_tv3);
        v_av12 = _mm256_sub_ps(v_av12, v_tv4);
        v_av12 = _mm256_add_ps(v_av12, v_tv5);
        v_av12 = SWAP_RI_256_S(CONJ_256_S(v_av12));

        // Output point 6:X[5]
        v_out5 = _mm256_add_ps(v_av11, v_av12);
        // Output point 7:X[6]
        v_out6 = _mm256_sub_ps(v_av11, v_av12);

        // Output point 1:X[0]
        v_out0 = _mm256_add_ps(v_in0, v_av1);
        v_out0 = _mm256_add_ps(v_out0, v_av2);
        v_out0 = _mm256_add_ps(v_out0, v_av3);
        v_out0 = _mm256_add_ps(v_out0, v_av4);
        v_out0 = _mm256_add_ps(v_out0, v_av5);

        if (out_strides[1] == DATA_STRIDE)
        {
            TRANSPOSE_ST4_256_S(curr_out, v_out_stride, v_out0, v_out1, v_out2,
                                v_out3);
            curr_out = curr_out + NUM_SETS_256_S * DATA_STRIDE;
            TRANSPOSE_ST4_256_S(curr_out, v_out_stride, v_out4, v_out5, v_out6,
                                v_out7);
            curr_out = curr_out + NUM_SETS_256_S * DATA_STRIDE;
            TRANSPOSE_ST2_256_S(curr_out, v_out_stride, v_out8, v_out9);
            curr_out = curr_out + 2 * DATA_STRIDE;
            SCATTER4_256_S_STRIDED(curr_out, v_out_stride, v_out10);
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
        }

        in_r += NUM_SETS_256_S * v_in_stride;
        out_r += NUM_SETS_256_S * v_out_stride;
    }
    // tail cases
    if (remaining_sets >= NUM_SETS_128_S)
    {
        __m128 v_in0, v_in1, v_in2, v_in3, v_in4, v_in5, v_in6, v_in7, v_in8,
               v_in9, v_in10;
        __m128 v_av1, v_av2, v_av3, v_av4, v_av5, v_av6, v_av7, v_av8, v_av9,
               v_av10, v_tv1;
        __m128 v_tv2, v_tv3, v_tv4, v_tv5, v_av11, v_av12;
        __m128 v_out0, v_out1, v_out2, v_out3, v_out4, v_out5, v_out6, v_out7,
               v_out8, v_out9, v_out10;

        __m128 v_K1 = _mm256_castps256_ps128(v_C1);
        __m128 v_K2 = _mm256_castps256_ps128(v_C2);
        __m128 v_K3 = _mm256_castps256_ps128(v_C3);
        __m128 v_K4 = _mm256_castps256_ps128(v_C4);
        __m128 v_K5 = _mm256_castps256_ps128(v_C5);
        __m128 v_K6 = _mm256_castps256_ps128(v_C6);
        __m128 v_K7 = _mm256_castps256_ps128(v_C7);
        __m128 v_K8 = _mm256_castps256_ps128(v_C8);
        __m128 v_K9 = _mm256_castps256_ps128(v_C9);
        __m128 v_K10 = _mm256_castps256_ps128(v_C10);

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

        // common calculations
        v_av1 = _mm_add_ps(v_in1, v_in10);
        v_av2 = _mm_add_ps(v_in2, v_in9);
        v_av3 = _mm_add_ps(v_in3, v_in8);
        v_av4 = _mm_add_ps(v_in4, v_in7);
        v_av5 = _mm_add_ps(v_in5, v_in6);
        v_av6 = _mm_sub_ps(v_in10, v_in1);
        v_av7 = _mm_sub_ps(v_in9, v_in2);
        v_av8 = _mm_sub_ps(v_in8, v_in3);
        v_av9 = _mm_sub_ps(v_in7, v_in4);
        v_av10 = _mm_sub_ps(v_in6, v_in5);

        v_tv1 = _mm_mul_ps(v_K1, v_av1);
        v_tv2 = _mm_mul_ps(v_K3, v_av2);
        v_tv3 = _mm_mul_ps(v_K5, v_av3);
        v_tv4 = _mm_mul_ps(v_K7, v_av4);
        v_tv5 = _mm_mul_ps(v_K9, v_av5);

        v_av11 = _mm_add_ps(v_in0, v_tv1);
        v_av11 = _mm_add_ps(v_av11, v_tv2);
        v_av11 = _mm_sub_ps(v_av11, v_tv3);
        v_av11 = _mm_sub_ps(v_av11, v_tv4);
        v_av11 = _mm_sub_ps(v_av11, v_tv5);

        v_tv1 = _mm_mul_ps(v_K2, v_av6);
        v_tv2 = _mm_mul_ps(v_K4, v_av7);
        v_tv3 = _mm_mul_ps(v_K6, v_av8);
        v_tv4 = _mm_mul_ps(v_K8, v_av9);
        v_tv5 = _mm_mul_ps(v_K10, v_av10);

        v_av12 = _mm_add_ps(v_tv1, v_tv2);
        v_av12 = _mm_add_ps(v_av12, v_tv3);
        v_av12 = _mm_add_ps(v_av12, v_tv4);
        v_av12 = _mm_add_ps(v_av12, v_tv5);
        v_av12 = SWAP_RI_128_S(CONJ_128_S(v_av12));

        // Output point 2:X[1]
        v_out1 = _mm_add_ps(v_av11, v_av12);
        // Output point 11:X[10]
        v_out10 = _mm_sub_ps(v_av11, v_av12);

        v_tv1 = _mm_mul_ps(v_K3, v_av1);
        v_tv2 = _mm_mul_ps(v_K7, v_av2);
        v_tv3 = _mm_mul_ps(v_K9, v_av3);
        v_tv4 = _mm_mul_ps(v_K5, v_av4);
        v_tv5 = _mm_mul_ps(v_K1, v_av5);

        v_av11 = _mm_add_ps(v_in0, v_tv1);
        v_av11 = _mm_sub_ps(v_av11, v_tv2);
        v_av11 = _mm_sub_ps(v_av11, v_tv3);
        v_av11 = _mm_sub_ps(v_av11, v_tv4);
        v_av11 = _mm_add_ps(v_av11, v_tv5);

        v_tv1 = _mm_mul_ps(v_K4, v_av6);
        v_tv2 = _mm_mul_ps(v_K8, v_av7);
        v_tv3 = _mm_mul_ps(v_K10, v_av8);
        v_tv4 = _mm_mul_ps(v_K6, v_av9);
        v_tv5 = _mm_mul_ps(v_K2, v_av10);

        v_av12 = _mm_add_ps(v_tv1, v_tv2);
        v_av12 = _mm_sub_ps(v_av12, v_tv3);
        v_av12 = _mm_sub_ps(v_av12, v_tv4);
        v_av12 = _mm_sub_ps(v_av12, v_tv5);
        v_av12 = SWAP_RI_128_S(CONJ_128_S(v_av12));

        // Output point 3:X[2]
        v_out2 = _mm_add_ps(v_av11, v_av12);
        // Output point 10:X[9]
        v_out9 = _mm_sub_ps(v_av11, v_av12);

        v_tv1 = _mm_mul_ps(v_K5, v_av1);
        v_tv2 = _mm_mul_ps(v_K9, v_av2);
        v_tv3 = _mm_mul_ps(v_K3, v_av3);
        v_tv4 = _mm_mul_ps(v_K1, v_av4);
        v_tv5 = _mm_mul_ps(v_K7, v_av5);

        v_av11 = _mm_sub_ps(v_in0, v_tv1);
        v_av11 = _mm_sub_ps(v_av11, v_tv2);
        v_av11 = _mm_add_ps(v_av11, v_tv3);
        v_av11 = _mm_add_ps(v_av11, v_tv4);
        v_av11 = _mm_sub_ps(v_av11, v_tv5);

        v_tv1 = _mm_mul_ps(v_K6, v_av6);
        v_tv2 = _mm_mul_ps(v_K10, v_av7);
        v_tv3 = _mm_mul_ps(v_K4, v_av8);
        v_tv4 = _mm_mul_ps(v_K2, v_av9);
        v_tv5 = _mm_mul_ps(v_K8, v_av10);

        v_av12 = _mm_sub_ps(v_tv1, v_tv2);
        v_av12 = _mm_sub_ps(v_av12, v_tv3);
        v_av12 = _mm_add_ps(v_av12, v_tv4);
        v_av12 = _mm_add_ps(v_av12, v_tv5);
        v_av12 = SWAP_RI_128_S(CONJ_128_S(v_av12));

        // Output point 4:X[3]
        v_out3 = _mm_add_ps(v_av11, v_av12);
        // Output point 9:X[8]
        v_out8 = _mm_sub_ps(v_av11, v_av12);

        v_tv1 = _mm_mul_ps(v_K7, v_av1);
        v_tv2 = _mm_mul_ps(v_K5, v_av2);
        v_tv3 = _mm_mul_ps(v_K1, v_av3);
        v_tv4 = _mm_mul_ps(v_K9, v_av4);
        v_tv5 = _mm_mul_ps(v_K3, v_av5);

        v_av11 = _mm_sub_ps(v_in0, v_tv1);
        v_av11 = _mm_sub_ps(v_av11, v_tv2);
        v_av11 = _mm_add_ps(v_av11, v_tv3);
        v_av11 = _mm_sub_ps(v_av11, v_tv4);
        v_av11 = _mm_add_ps(v_av11, v_tv5);

        v_tv1 = _mm_mul_ps(v_K8, v_av6);
        v_tv2 = _mm_mul_ps(v_K6, v_av7);
        v_tv3 = _mm_mul_ps(v_K2, v_av8);
        v_tv4 = _mm_mul_ps(v_K10, v_av9);
        v_tv5 = _mm_mul_ps(v_K4, v_av10);

        v_av12 = _mm_sub_ps(v_tv1, v_tv2);
        v_av12 = _mm_add_ps(v_av12, v_tv3);
        v_av12 = _mm_add_ps(v_av12, v_tv4);
        v_av12 = _mm_sub_ps(v_av12, v_tv5);
        v_av12 = SWAP_RI_128_S(CONJ_128_S(v_av12));

        // Output point 5:X[4]
        v_out4 = _mm_add_ps(v_av11, v_av12);
        // Output point 8:X[7]
        v_out7 = _mm_sub_ps(v_av11, v_av12);

        v_tv1 = _mm_mul_ps(v_K9, v_av1);
        v_tv2 = _mm_mul_ps(v_K1, v_av2);
        v_tv3 = _mm_mul_ps(v_K7, v_av3);
        v_tv4 = _mm_mul_ps(v_K3, v_av4);
        v_tv5 = _mm_mul_ps(v_K5, v_av5);

        v_av11 = _mm_sub_ps(v_in0, v_tv1);
        v_av11 = _mm_add_ps(v_av11, v_tv2);
        v_av11 = _mm_sub_ps(v_av11, v_tv3);
        v_av11 = _mm_add_ps(v_av11, v_tv4);
        v_av11 = _mm_sub_ps(v_av11, v_tv5);

        v_tv1 = _mm_mul_ps(v_K10, v_av6);
        v_tv2 = _mm_mul_ps(v_K2, v_av7);
        v_tv3 = _mm_mul_ps(v_K8, v_av8);
        v_tv4 = _mm_mul_ps(v_K4, v_av9);
        v_tv5 = _mm_mul_ps(v_K6, v_av10);

        v_av12 = _mm_sub_ps(v_tv1, v_tv2);
        v_av12 = _mm_add_ps(v_av12, v_tv3);
        v_av12 = _mm_sub_ps(v_av12, v_tv4);
        v_av12 = _mm_add_ps(v_av12, v_tv5);
        v_av12 = SWAP_RI_128_S(CONJ_128_S(v_av12));

        // Output point 6:X[5]
        v_out5 = _mm_add_ps(v_av11, v_av12);
        // Output point 7:X[6]
        v_out6 = _mm_sub_ps(v_av11, v_av12);

        // Output point 1:X[0]
        v_out0 = _mm_add_ps(v_in0, v_av1);
        v_out0 = _mm_add_ps(v_out0, v_av2);
        v_out0 = _mm_add_ps(v_out0, v_av3);
        v_out0 = _mm_add_ps(v_out0, v_av4);
        v_out0 = _mm_add_ps(v_out0, v_av5);

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
            SCATTER2_128_S_STRIDED(curr_out, v_out_stride, v_out10);
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
        }

        in_r = in_r + (v_in_stride << 1);
        out_r = out_r + (v_out_stride << 1);
    }
    if (remaining_sets & 1)
    {
        __m128 v_in0, v_in1, v_in2, v_in3, v_in4, v_in5, v_in6, v_in7, v_in8,
               v_in9, v_in10;
        __m128 v_av1, v_av2, v_av3, v_av4, v_av5, v_av6, v_av7, v_av8, v_av9,
               v_av10, v_tv1;
        __m128 v_tv2, v_tv3, v_tv4, v_tv5, v_av11, v_av12;
        __m128 v_out0, v_out1, v_out2, v_out3, v_out4, v_out5, v_out6, v_out7,
               v_out8, v_out9, v_out10;

        __m128 v_K1 = _mm256_castps256_ps128(v_C1);
        __m128 v_K2 = _mm256_castps256_ps128(v_C2);
        __m128 v_K3 = _mm256_castps256_ps128(v_C3);
        __m128 v_K4 = _mm256_castps256_ps128(v_C4);
        __m128 v_K5 = _mm256_castps256_ps128(v_C5);
        __m128 v_K6 = _mm256_castps256_ps128(v_C6);
        __m128 v_K7 = _mm256_castps256_ps128(v_C7);
        __m128 v_K8 = _mm256_castps256_ps128(v_C8);
        __m128 v_K9 = _mm256_castps256_ps128(v_C9);
        __m128 v_K10 = _mm256_castps256_ps128(v_C10);

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

        // common calculations
        v_av1 = _mm_add_ps(v_in1, v_in10);
        v_av2 = _mm_add_ps(v_in2, v_in9);
        v_av3 = _mm_add_ps(v_in3, v_in8);
        v_av4 = _mm_add_ps(v_in4, v_in7);
        v_av5 = _mm_add_ps(v_in5, v_in6);
        v_av6 = _mm_sub_ps(v_in10, v_in1);
        v_av7 = _mm_sub_ps(v_in9, v_in2);
        v_av8 = _mm_sub_ps(v_in8, v_in3);
        v_av9 = _mm_sub_ps(v_in7, v_in4);
        v_av10 = _mm_sub_ps(v_in6, v_in5);

        v_tv1 = _mm_mul_ps(v_K1, v_av1);
        v_tv2 = _mm_mul_ps(v_K3, v_av2);
        v_tv3 = _mm_mul_ps(v_K5, v_av3);
        v_tv4 = _mm_mul_ps(v_K7, v_av4);
        v_tv5 = _mm_mul_ps(v_K9, v_av5);

        v_av11 = _mm_add_ps(v_in0, v_tv1);
        v_av11 = _mm_add_ps(v_av11, v_tv2);
        v_av11 = _mm_sub_ps(v_av11, v_tv3);
        v_av11 = _mm_sub_ps(v_av11, v_tv4);
        v_av11 = _mm_sub_ps(v_av11, v_tv5);

        v_tv1 = _mm_mul_ps(v_K2, v_av6);
        v_tv2 = _mm_mul_ps(v_K4, v_av7);
        v_tv3 = _mm_mul_ps(v_K6, v_av8);
        v_tv4 = _mm_mul_ps(v_K8, v_av9);
        v_tv5 = _mm_mul_ps(v_K10, v_av10);

        v_av12 = _mm_add_ps(v_tv1, v_tv2);
        v_av12 = _mm_add_ps(v_av12, v_tv3);
        v_av12 = _mm_add_ps(v_av12, v_tv4);
        v_av12 = _mm_add_ps(v_av12, v_tv5);
        v_av12 = SWAP_RI_128_S(CONJ_128_S(v_av12));

        // Output point 2:X[1]
        v_out1 = _mm_add_ps(v_av11, v_av12);
        // Output point 11:X[10]
        v_out10 = _mm_sub_ps(v_av11, v_av12);

        v_tv1 = _mm_mul_ps(v_K3, v_av1);
        v_tv2 = _mm_mul_ps(v_K7, v_av2);
        v_tv3 = _mm_mul_ps(v_K9, v_av3);
        v_tv4 = _mm_mul_ps(v_K5, v_av4);
        v_tv5 = _mm_mul_ps(v_K1, v_av5);

        v_av11 = _mm_add_ps(v_in0, v_tv1);
        v_av11 = _mm_sub_ps(v_av11, v_tv2);
        v_av11 = _mm_sub_ps(v_av11, v_tv3);
        v_av11 = _mm_sub_ps(v_av11, v_tv4);
        v_av11 = _mm_add_ps(v_av11, v_tv5);

        v_tv1 = _mm_mul_ps(v_K4, v_av6);
        v_tv2 = _mm_mul_ps(v_K8, v_av7);
        v_tv3 = _mm_mul_ps(v_K10, v_av8);
        v_tv4 = _mm_mul_ps(v_K6, v_av9);
        v_tv5 = _mm_mul_ps(v_K2, v_av10);

        v_av12 = _mm_add_ps(v_tv1, v_tv2);
        v_av12 = _mm_sub_ps(v_av12, v_tv3);
        v_av12 = _mm_sub_ps(v_av12, v_tv4);
        v_av12 = _mm_sub_ps(v_av12, v_tv5);
        v_av12 = SWAP_RI_128_S(CONJ_128_S(v_av12));

        // Output point 3:X[2]
        v_out2 = _mm_add_ps(v_av11, v_av12);
        // Output point 10:X[9]
        v_out9 = _mm_sub_ps(v_av11, v_av12);

        v_tv1 = _mm_mul_ps(v_K5, v_av1);
        v_tv2 = _mm_mul_ps(v_K9, v_av2);
        v_tv3 = _mm_mul_ps(v_K3, v_av3);
        v_tv4 = _mm_mul_ps(v_K1, v_av4);
        v_tv5 = _mm_mul_ps(v_K7, v_av5);

        v_av11 = _mm_sub_ps(v_in0, v_tv1);
        v_av11 = _mm_sub_ps(v_av11, v_tv2);
        v_av11 = _mm_add_ps(v_av11, v_tv3);
        v_av11 = _mm_add_ps(v_av11, v_tv4);
        v_av11 = _mm_sub_ps(v_av11, v_tv5);

        v_tv1 = _mm_mul_ps(v_K6, v_av6);
        v_tv2 = _mm_mul_ps(v_K10, v_av7);
        v_tv3 = _mm_mul_ps(v_K4, v_av8);
        v_tv4 = _mm_mul_ps(v_K2, v_av9);
        v_tv5 = _mm_mul_ps(v_K8, v_av10);

        v_av12 = _mm_sub_ps(v_tv1, v_tv2);
        v_av12 = _mm_sub_ps(v_av12, v_tv3);
        v_av12 = _mm_add_ps(v_av12, v_tv4);
        v_av12 = _mm_add_ps(v_av12, v_tv5);
        v_av12 = SWAP_RI_128_S(CONJ_128_S(v_av12));

        // Output point 4:X[3]
        v_out3 = _mm_add_ps(v_av11, v_av12);
        // Output point 9:X[8]
        v_out8 = _mm_sub_ps(v_av11, v_av12);

        v_tv1 = _mm_mul_ps(v_K7, v_av1);
        v_tv2 = _mm_mul_ps(v_K5, v_av2);
        v_tv3 = _mm_mul_ps(v_K1, v_av3);
        v_tv4 = _mm_mul_ps(v_K9, v_av4);
        v_tv5 = _mm_mul_ps(v_K3, v_av5);

        v_av11 = _mm_sub_ps(v_in0, v_tv1);
        v_av11 = _mm_sub_ps(v_av11, v_tv2);
        v_av11 = _mm_add_ps(v_av11, v_tv3);
        v_av11 = _mm_sub_ps(v_av11, v_tv4);
        v_av11 = _mm_add_ps(v_av11, v_tv5);

        v_tv1 = _mm_mul_ps(v_K8, v_av6);
        v_tv2 = _mm_mul_ps(v_K6, v_av7);
        v_tv3 = _mm_mul_ps(v_K2, v_av8);
        v_tv4 = _mm_mul_ps(v_K10, v_av9);
        v_tv5 = _mm_mul_ps(v_K4, v_av10);

        v_av12 = _mm_sub_ps(v_tv1, v_tv2);
        v_av12 = _mm_add_ps(v_av12, v_tv3);
        v_av12 = _mm_add_ps(v_av12, v_tv4);
        v_av12 = _mm_sub_ps(v_av12, v_tv5);
        v_av12 = SWAP_RI_128_S(CONJ_128_S(v_av12));

        // Output point 5:X[4]
        v_out4 = _mm_add_ps(v_av11, v_av12);
        // Output point 8:X[7]
        v_out7 = _mm_sub_ps(v_av11, v_av12);

        v_tv1 = _mm_mul_ps(v_K9, v_av1);
        v_tv2 = _mm_mul_ps(v_K1, v_av2);
        v_tv3 = _mm_mul_ps(v_K7, v_av3);
        v_tv4 = _mm_mul_ps(v_K3, v_av4);
        v_tv5 = _mm_mul_ps(v_K5, v_av5);

        v_av11 = _mm_sub_ps(v_in0, v_tv1);
        v_av11 = _mm_add_ps(v_av11, v_tv2);
        v_av11 = _mm_sub_ps(v_av11, v_tv3);
        v_av11 = _mm_add_ps(v_av11, v_tv4);
        v_av11 = _mm_sub_ps(v_av11, v_tv5);

        v_tv1 = _mm_mul_ps(v_K10, v_av6);
        v_tv2 = _mm_mul_ps(v_K2, v_av7);
        v_tv3 = _mm_mul_ps(v_K8, v_av8);
        v_tv4 = _mm_mul_ps(v_K4, v_av9);
        v_tv5 = _mm_mul_ps(v_K6, v_av10);

        v_av12 = _mm_sub_ps(v_tv1, v_tv2);
        v_av12 = _mm_add_ps(v_av12, v_tv3);
        v_av12 = _mm_sub_ps(v_av12, v_tv4);
        v_av12 = _mm_add_ps(v_av12, v_tv5);
        v_av12 = SWAP_RI_128_S(CONJ_128_S(v_av12));

        // Output point 6:X[5]
        v_out5 = _mm_add_ps(v_av11, v_av12);
        // Output point 7:X[6]
        v_out6 = _mm_sub_ps(v_av11, v_av12);

        // Output point 1:X[0]
        v_out0 = _mm_add_ps(v_in0, v_av1);
        v_out0 = _mm_add_ps(v_out0, v_av2);
        v_out0 = _mm_add_ps(v_out0, v_av3);
        v_out0 = _mm_add_ps(v_out0, v_av4);
        v_out0 = _mm_add_ps(v_out0, v_av5);

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
            ST_LOW_128_S(curr_out, v_out10);
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
        }
    }
    AOCLFFTZ_LOG(DEBUG, global_logger_mode, "Exit");
}

static FFTZ_VOID fft11avx256fp64(FFTZ_VOID *in_real, FFTZ_VOID *in_imag,
                                 FFTZ_VOID *out_real, FFTZ_VOID *out_imag,
                                 FFTZ_INTP n, aoclfftz_strides_t *strides,
                                 FFTZ_VOID *twd, FFTZ_UINT8 flag)
{
    AOCLFFTZ_LOG(DEBUG, global_logger_mode, "Enter");
    const FFTZ_DOUBLE CRTM_11[10] = {
        0.84125353283118116029052039464203089547681594330064,
        0.54064081745559759544482548159299693174139803024473,
        0.41541501300188639668675795488636098054966524290126,
        0.90963199535451838458365117807108162835411650732265,
        0.14231483827328501490317354898047094957684096668515,
        0.98982144188093275042610808187068914262031166769031,
        0.65486073394528511198338203198719613618953603946564,
        0.75574957435425824224552448923467521721665586591805,
        0.95949297361449738989036805706632769906245484800000,
        0.28173255684142978898192655345478532989004751779983};

    FFTZ_DOUBLE *in_r = (FFTZ_DOUBLE *)in_real;
    FFTZ_DOUBLE *out_r = (FFTZ_DOUBLE *)out_real;
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
    FFTZ_INTP N = n / NUM_SETS_256_D;
    FFTZ_INTP count;
    FFTZ_DOUBLE *curr_in, *curr_out;

    __m256d v_in0, v_in1, v_in2, v_in3, v_in4, v_in5, v_in6, v_in7, v_in8,
            v_in9, v_in10;
    __m256d v_av1, v_av2, v_av3, v_av4, v_av5, v_av6, v_av7, v_av8, v_av9,
            v_av10, v_tv1;
    __m256d v_tv2, v_tv3, v_tv4, v_tv5, v_av11, v_av12;
    __m256d v_out0, v_out1, v_out2, v_out3, v_out4, v_out5, v_out6, v_out7,
            v_out8, v_out9, v_out10;

    __m256d v_C1 = _mm256_broadcast_sd(&CRTM_11[0]);
    __m256d v_C2 = _mm256_broadcast_sd(&CRTM_11[1]);
    v_C2 = _mm256_xor_pd(v_C2, _neg_256_d[flag].d);
    __m256d v_C3 = _mm256_broadcast_sd(&CRTM_11[2]);
    __m256d v_C4 = _mm256_broadcast_sd(&CRTM_11[3]);
    v_C4 = _mm256_xor_pd(v_C4, _neg_256_d[flag].d);
    __m256d v_C5 = _mm256_broadcast_sd(&CRTM_11[4]);
    __m256d v_C6 = _mm256_broadcast_sd(&CRTM_11[5]);
    v_C6 = _mm256_xor_pd(v_C6, _neg_256_d[flag].d);
    __m256d v_C7 = _mm256_broadcast_sd(&CRTM_11[6]);
    __m256d v_C8 = _mm256_broadcast_sd(&CRTM_11[7]);
    v_C8 = _mm256_xor_pd(v_C8, _neg_256_d[flag].d);
    __m256d v_C9 = _mm256_broadcast_sd(&CRTM_11[8]);
    __m256d v_C10 = _mm256_broadcast_sd(&CRTM_11[9]);
    v_C10 = _mm256_xor_pd(v_C10, _neg_256_d[flag].d);

    for (count = 0; count < N; count++)
    {
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

        // common calculations
        v_av1 = _mm256_add_pd(v_in1, v_in10);
        v_av2 = _mm256_add_pd(v_in2, v_in9);
        v_av3 = _mm256_add_pd(v_in3, v_in8);
        v_av4 = _mm256_add_pd(v_in4, v_in7);
        v_av5 = _mm256_add_pd(v_in5, v_in6);
        v_av6 = _mm256_sub_pd(v_in10, v_in1);
        v_av7 = _mm256_sub_pd(v_in9, v_in2);
        v_av8 = _mm256_sub_pd(v_in8, v_in3);
        v_av9 = _mm256_sub_pd(v_in7, v_in4);
        v_av10 = _mm256_sub_pd(v_in6, v_in5);

        v_tv1 = _mm256_mul_pd(v_C1, v_av1);
        v_tv2 = _mm256_mul_pd(v_C3, v_av2);
        v_tv3 = _mm256_mul_pd(v_C5, v_av3);
        v_tv4 = _mm256_mul_pd(v_C7, v_av4);
        v_tv5 = _mm256_mul_pd(v_C9, v_av5);

        v_av11 = _mm256_add_pd(v_in0, v_tv1);
        v_av11 = _mm256_add_pd(v_av11, v_tv2);
        v_av11 = _mm256_sub_pd(v_av11, v_tv3);
        v_av11 = _mm256_sub_pd(v_av11, v_tv4);
        v_av11 = _mm256_sub_pd(v_av11, v_tv5);

        v_tv1 = _mm256_mul_pd(v_C2, v_av6);
        v_tv2 = _mm256_mul_pd(v_C4, v_av7);
        v_tv3 = _mm256_mul_pd(v_C6, v_av8);
        v_tv4 = _mm256_mul_pd(v_C8, v_av9);
        v_tv5 = _mm256_mul_pd(v_C10, v_av10);

        v_av12 = _mm256_add_pd(v_tv1, v_tv2);
        v_av12 = _mm256_add_pd(v_av12, v_tv3);
        v_av12 = _mm256_add_pd(v_av12, v_tv4);
        v_av12 = _mm256_add_pd(v_av12, v_tv5);
        v_av12 = SWAP_RI_256_D(CONJ_256_D(v_av12));

        // Output point 2:X[1]
        v_out1 = _mm256_add_pd(v_av11, v_av12);
        // Output point 11:X[10]
        v_out10 = _mm256_sub_pd(v_av11, v_av12);

        v_tv1 = _mm256_mul_pd(v_C3, v_av1);
        v_tv2 = _mm256_mul_pd(v_C7, v_av2);
        v_tv3 = _mm256_mul_pd(v_C9, v_av3);
        v_tv4 = _mm256_mul_pd(v_C5, v_av4);
        v_tv5 = _mm256_mul_pd(v_C1, v_av5);

        v_av11 = _mm256_add_pd(v_in0, v_tv1);
        v_av11 = _mm256_sub_pd(v_av11, v_tv2);
        v_av11 = _mm256_sub_pd(v_av11, v_tv3);
        v_av11 = _mm256_sub_pd(v_av11, v_tv4);
        v_av11 = _mm256_add_pd(v_av11, v_tv5);

        v_tv1 = _mm256_mul_pd(v_C4, v_av6);
        v_tv2 = _mm256_mul_pd(v_C8, v_av7);
        v_tv3 = _mm256_mul_pd(v_C10, v_av8);
        v_tv4 = _mm256_mul_pd(v_C6, v_av9);
        v_tv5 = _mm256_mul_pd(v_C2, v_av10);

        v_av12 = _mm256_add_pd(v_tv1, v_tv2);
        v_av12 = _mm256_sub_pd(v_av12, v_tv3);
        v_av12 = _mm256_sub_pd(v_av12, v_tv4);
        v_av12 = _mm256_sub_pd(v_av12, v_tv5);
        v_av12 = SWAP_RI_256_D(CONJ_256_D(v_av12));

        // Output point 3:X[2]
        v_out2 = _mm256_add_pd(v_av11, v_av12);
        // Output point 10:X[9]
        v_out9 = _mm256_sub_pd(v_av11, v_av12);

        v_tv1 = _mm256_mul_pd(v_C5, v_av1);
        v_tv2 = _mm256_mul_pd(v_C9, v_av2);
        v_tv3 = _mm256_mul_pd(v_C3, v_av3);
        v_tv4 = _mm256_mul_pd(v_C1, v_av4);
        v_tv5 = _mm256_mul_pd(v_C7, v_av5);

        v_av11 = _mm256_sub_pd(v_in0, v_tv1);
        v_av11 = _mm256_sub_pd(v_av11, v_tv2);
        v_av11 = _mm256_add_pd(v_av11, v_tv3);
        v_av11 = _mm256_add_pd(v_av11, v_tv4);
        v_av11 = _mm256_sub_pd(v_av11, v_tv5);

        v_tv1 = _mm256_mul_pd(v_C6, v_av6);
        v_tv2 = _mm256_mul_pd(v_C10, v_av7);
        v_tv3 = _mm256_mul_pd(v_C4, v_av8);
        v_tv4 = _mm256_mul_pd(v_C2, v_av9);
        v_tv5 = _mm256_mul_pd(v_C8, v_av10);

        v_av12 = _mm256_sub_pd(v_tv1, v_tv2);
        v_av12 = _mm256_sub_pd(v_av12, v_tv3);
        v_av12 = _mm256_add_pd(v_av12, v_tv4);
        v_av12 = _mm256_add_pd(v_av12, v_tv5);
        v_av12 = SWAP_RI_256_D(CONJ_256_D(v_av12));

        // Output point 4:X[3]
        v_out3 = _mm256_add_pd(v_av11, v_av12);
        // Output point 9:X[8]
        v_out8 = _mm256_sub_pd(v_av11, v_av12);

        v_tv1 = _mm256_mul_pd(v_C7, v_av1);
        v_tv2 = _mm256_mul_pd(v_C5, v_av2);
        v_tv3 = _mm256_mul_pd(v_C1, v_av3);
        v_tv4 = _mm256_mul_pd(v_C9, v_av4);
        v_tv5 = _mm256_mul_pd(v_C3, v_av5);

        v_av11 = _mm256_sub_pd(v_in0, v_tv1);
        v_av11 = _mm256_sub_pd(v_av11, v_tv2);
        v_av11 = _mm256_add_pd(v_av11, v_tv3);
        v_av11 = _mm256_sub_pd(v_av11, v_tv4);
        v_av11 = _mm256_add_pd(v_av11, v_tv5);

        v_tv1 = _mm256_mul_pd(v_C8, v_av6);
        v_tv2 = _mm256_mul_pd(v_C6, v_av7);
        v_tv3 = _mm256_mul_pd(v_C2, v_av8);
        v_tv4 = _mm256_mul_pd(v_C10, v_av9);
        v_tv5 = _mm256_mul_pd(v_C4, v_av10);

        v_av12 = _mm256_sub_pd(v_tv1, v_tv2);
        v_av12 = _mm256_add_pd(v_av12, v_tv3);
        v_av12 = _mm256_add_pd(v_av12, v_tv4);
        v_av12 = _mm256_sub_pd(v_av12, v_tv5);
        v_av12 = SWAP_RI_256_D(CONJ_256_D(v_av12));

        // Output point 5:X[4]
        v_out4 = _mm256_add_pd(v_av11, v_av12);
        // Output point 8:X[7]
        v_out7 = _mm256_sub_pd(v_av11, v_av12);

        v_tv1 = _mm256_mul_pd(v_C9, v_av1);
        v_tv2 = _mm256_mul_pd(v_C1, v_av2);
        v_tv3 = _mm256_mul_pd(v_C7, v_av3);
        v_tv4 = _mm256_mul_pd(v_C3, v_av4);
        v_tv5 = _mm256_mul_pd(v_C5, v_av5);

        v_av11 = _mm256_sub_pd(v_in0, v_tv1);
        v_av11 = _mm256_add_pd(v_av11, v_tv2);
        v_av11 = _mm256_sub_pd(v_av11, v_tv3);
        v_av11 = _mm256_add_pd(v_av11, v_tv4);
        v_av11 = _mm256_sub_pd(v_av11, v_tv5);

        v_tv1 = _mm256_mul_pd(v_C10, v_av6);
        v_tv2 = _mm256_mul_pd(v_C2, v_av7);
        v_tv3 = _mm256_mul_pd(v_C8, v_av8);
        v_tv4 = _mm256_mul_pd(v_C4, v_av9);
        v_tv5 = _mm256_mul_pd(v_C6, v_av10);

        v_av12 = _mm256_sub_pd(v_tv1, v_tv2);
        v_av12 = _mm256_add_pd(v_av12, v_tv3);
        v_av12 = _mm256_sub_pd(v_av12, v_tv4);
        v_av12 = _mm256_add_pd(v_av12, v_tv5);
        v_av12 = SWAP_RI_256_D(CONJ_256_D(v_av12));

        // Output point 6:X[5]
        v_out5 = _mm256_add_pd(v_av11, v_av12);
        // Output point 7:X[6]
        v_out6 = _mm256_sub_pd(v_av11, v_av12);

        // Output point 1:X[0]
        v_out0 = _mm256_add_pd(v_in0, v_av1);
        v_out0 = _mm256_add_pd(v_out0, v_av2);
        v_out0 = _mm256_add_pd(v_out0, v_av3);
        v_out0 = _mm256_add_pd(v_out0, v_av4);
        v_out0 = _mm256_add_pd(v_out0, v_av5);

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
            SCATTER2_256_D_STRIDED(curr_out, v_out_stride, v_out10);
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
        }

        in_r += NUM_SETS_256_D * v_in_stride;
        out_r += NUM_SETS_256_D * v_out_stride;
    }
    // tail case
    if (n & 1)
    {
        __m128d v_in0, v_in1, v_in2, v_in3, v_in4, v_in5, v_in6, v_in7, v_in8,
                v_in9, v_in10;
        __m128d v_av1, v_av2, v_av3, v_av4, v_av5, v_av6, v_av7, v_av8, v_av9,
                v_av10, v_tv1;
        __m128d v_tv2, v_tv3, v_tv4, v_tv5, v_av11, v_av12;
        __m128d v_out0, v_out1, v_out2, v_out3, v_out4, v_out5, v_out6, v_out7,
                v_out8, v_out9, v_out10;

        __m128d v_K1 = _mm256_castpd256_pd128(v_C1);
        __m128d v_K2 = _mm256_castpd256_pd128(v_C2);
        __m128d v_K3 = _mm256_castpd256_pd128(v_C3);
        __m128d v_K4 = _mm256_castpd256_pd128(v_C4);
        __m128d v_K5 = _mm256_castpd256_pd128(v_C5);
        __m128d v_K6 = _mm256_castpd256_pd128(v_C6);
        __m128d v_K7 = _mm256_castpd256_pd128(v_C7);
        __m128d v_K8 = _mm256_castpd256_pd128(v_C8);
        __m128d v_K9 = _mm256_castpd256_pd128(v_C9);
        __m128d v_K10 = _mm256_castpd256_pd128(v_C10);

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

        // common calculations
        v_av1 = _mm_add_pd(v_in1, v_in10);
        v_av2 = _mm_add_pd(v_in2, v_in9);
        v_av3 = _mm_add_pd(v_in3, v_in8);
        v_av4 = _mm_add_pd(v_in4, v_in7);
        v_av5 = _mm_add_pd(v_in5, v_in6);
        v_av6 = _mm_sub_pd(v_in10, v_in1);
        v_av7 = _mm_sub_pd(v_in9, v_in2);
        v_av8 = _mm_sub_pd(v_in8, v_in3);
        v_av9 = _mm_sub_pd(v_in7, v_in4);
        v_av10 = _mm_sub_pd(v_in6, v_in5);

        v_tv1 = _mm_mul_pd(v_K1, v_av1);
        v_tv2 = _mm_mul_pd(v_K3, v_av2);
        v_tv3 = _mm_mul_pd(v_K5, v_av3);
        v_tv4 = _mm_mul_pd(v_K7, v_av4);
        v_tv5 = _mm_mul_pd(v_K9, v_av5);

        v_av11 = _mm_add_pd(v_in0, v_tv1);
        v_av11 = _mm_add_pd(v_av11, v_tv2);
        v_av11 = _mm_sub_pd(v_av11, v_tv3);
        v_av11 = _mm_sub_pd(v_av11, v_tv4);
        v_av11 = _mm_sub_pd(v_av11, v_tv5);

        v_tv1 = _mm_mul_pd(v_K2, v_av6);
        v_tv2 = _mm_mul_pd(v_K4, v_av7);
        v_tv3 = _mm_mul_pd(v_K6, v_av8);
        v_tv4 = _mm_mul_pd(v_K8, v_av9);
        v_tv5 = _mm_mul_pd(v_K10, v_av10);

        v_av12 = _mm_add_pd(v_tv1, v_tv2);
        v_av12 = _mm_add_pd(v_av12, v_tv3);
        v_av12 = _mm_add_pd(v_av12, v_tv4);
        v_av12 = _mm_add_pd(v_av12, v_tv5);
        v_av12 = SWAP_RI_128_D(CONJ_128_D(v_av12));

        // Output point 2:X[1]
        v_out1 = _mm_add_pd(v_av11, v_av12);
        // Output point 11:X[10]
        v_out10 = _mm_sub_pd(v_av11, v_av12);

        v_tv1 = _mm_mul_pd(v_K3, v_av1);
        v_tv2 = _mm_mul_pd(v_K7, v_av2);
        v_tv3 = _mm_mul_pd(v_K9, v_av3);
        v_tv4 = _mm_mul_pd(v_K5, v_av4);
        v_tv5 = _mm_mul_pd(v_K1, v_av5);

        v_av11 = _mm_add_pd(v_in0, v_tv1);
        v_av11 = _mm_sub_pd(v_av11, v_tv2);
        v_av11 = _mm_sub_pd(v_av11, v_tv3);
        v_av11 = _mm_sub_pd(v_av11, v_tv4);
        v_av11 = _mm_add_pd(v_av11, v_tv5);

        v_tv1 = _mm_mul_pd(v_K4, v_av6);
        v_tv2 = _mm_mul_pd(v_K8, v_av7);
        v_tv3 = _mm_mul_pd(v_K10, v_av8);
        v_tv4 = _mm_mul_pd(v_K6, v_av9);
        v_tv5 = _mm_mul_pd(v_K2, v_av10);

        v_av12 = _mm_add_pd(v_tv1, v_tv2);
        v_av12 = _mm_sub_pd(v_av12, v_tv3);
        v_av12 = _mm_sub_pd(v_av12, v_tv4);
        v_av12 = _mm_sub_pd(v_av12, v_tv5);
        v_av12 = SWAP_RI_128_D(CONJ_128_D(v_av12));

        // Output point 3:X[2]
        v_out2 = _mm_add_pd(v_av11, v_av12);
        // Output point 10:X[9]
        v_out9 = _mm_sub_pd(v_av11, v_av12);

        v_tv1 = _mm_mul_pd(v_K5, v_av1);
        v_tv2 = _mm_mul_pd(v_K9, v_av2);
        v_tv3 = _mm_mul_pd(v_K3, v_av3);
        v_tv4 = _mm_mul_pd(v_K1, v_av4);
        v_tv5 = _mm_mul_pd(v_K7, v_av5);

        v_av11 = _mm_sub_pd(v_in0, v_tv1);
        v_av11 = _mm_sub_pd(v_av11, v_tv2);
        v_av11 = _mm_add_pd(v_av11, v_tv3);
        v_av11 = _mm_add_pd(v_av11, v_tv4);
        v_av11 = _mm_sub_pd(v_av11, v_tv5);

        v_tv1 = _mm_mul_pd(v_K6, v_av6);
        v_tv2 = _mm_mul_pd(v_K10, v_av7);
        v_tv3 = _mm_mul_pd(v_K4, v_av8);
        v_tv4 = _mm_mul_pd(v_K2, v_av9);
        v_tv5 = _mm_mul_pd(v_K8, v_av10);

        v_av12 = _mm_sub_pd(v_tv1, v_tv2);
        v_av12 = _mm_sub_pd(v_av12, v_tv3);
        v_av12 = _mm_add_pd(v_av12, v_tv4);
        v_av12 = _mm_add_pd(v_av12, v_tv5);
        v_av12 = SWAP_RI_128_D(CONJ_128_D(v_av12));

        // Output point 4:X[3]
        v_out3 = _mm_add_pd(v_av11, v_av12);
        // Output point 9:X[8]
        v_out8 = _mm_sub_pd(v_av11, v_av12);

        v_tv1 = _mm_mul_pd(v_K7, v_av1);
        v_tv2 = _mm_mul_pd(v_K5, v_av2);
        v_tv3 = _mm_mul_pd(v_K1, v_av3);
        v_tv4 = _mm_mul_pd(v_K9, v_av4);
        v_tv5 = _mm_mul_pd(v_K3, v_av5);

        v_av11 = _mm_sub_pd(v_in0, v_tv1);
        v_av11 = _mm_sub_pd(v_av11, v_tv2);
        v_av11 = _mm_add_pd(v_av11, v_tv3);
        v_av11 = _mm_sub_pd(v_av11, v_tv4);
        v_av11 = _mm_add_pd(v_av11, v_tv5);

        v_tv1 = _mm_mul_pd(v_K8, v_av6);
        v_tv2 = _mm_mul_pd(v_K6, v_av7);
        v_tv3 = _mm_mul_pd(v_K2, v_av8);
        v_tv4 = _mm_mul_pd(v_K10, v_av9);
        v_tv5 = _mm_mul_pd(v_K4, v_av10);

        v_av12 = _mm_sub_pd(v_tv1, v_tv2);
        v_av12 = _mm_add_pd(v_av12, v_tv3);
        v_av12 = _mm_add_pd(v_av12, v_tv4);
        v_av12 = _mm_sub_pd(v_av12, v_tv5);
        v_av12 = SWAP_RI_128_D(CONJ_128_D(v_av12));

        // Output point 5:X[4]
        v_out4 = _mm_add_pd(v_av11, v_av12);
        // Output point 8:X[7]
        v_out7 = _mm_sub_pd(v_av11, v_av12);

        v_tv1 = _mm_mul_pd(v_K9, v_av1);
        v_tv2 = _mm_mul_pd(v_K1, v_av2);
        v_tv3 = _mm_mul_pd(v_K7, v_av3);
        v_tv4 = _mm_mul_pd(v_K3, v_av4);
        v_tv5 = _mm_mul_pd(v_K5, v_av5);

        v_av11 = _mm_sub_pd(v_in0, v_tv1);
        v_av11 = _mm_add_pd(v_av11, v_tv2);
        v_av11 = _mm_sub_pd(v_av11, v_tv3);
        v_av11 = _mm_add_pd(v_av11, v_tv4);
        v_av11 = _mm_sub_pd(v_av11, v_tv5);

        v_tv1 = _mm_mul_pd(v_K10, v_av6);
        v_tv2 = _mm_mul_pd(v_K2, v_av7);
        v_tv3 = _mm_mul_pd(v_K8, v_av8);
        v_tv4 = _mm_mul_pd(v_K4, v_av9);
        v_tv5 = _mm_mul_pd(v_K6, v_av10);

        v_av12 = _mm_sub_pd(v_tv1, v_tv2);
        v_av12 = _mm_add_pd(v_av12, v_tv3);
        v_av12 = _mm_sub_pd(v_av12, v_tv4);
        v_av12 = _mm_add_pd(v_av12, v_tv5);
        v_av12 = SWAP_RI_128_D(CONJ_128_D(v_av12));

        // Output point 6:X[5]
        v_out5 = _mm_add_pd(v_av11, v_av12);
        // Output point 7:X[6]
        v_out6 = _mm_sub_pd(v_av11, v_av12);

        // Output point 1:X[0]
        v_out0 = _mm_add_pd(v_in0, v_av1);
        v_out0 = _mm_add_pd(v_out0, v_av2);
        v_out0 = _mm_add_pd(v_out0, v_av3);
        v_out0 = _mm_add_pd(v_out0, v_av4);
        v_out0 = _mm_add_pd(v_out0, v_av5);

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
    }
    AOCLFFTZ_LOG(DEBUG, global_logger_mode, "Exit");
}

kfft_ register_kernel_fft11avx256(FFTZ_UINT8 precision,
                                  FFTZ_UINT8 direction /* unused */)
{
    if (precision == DT_FLOAT)
    {
        return fft11avx256fp32;
    }
    else if (precision == DT_DOUBLE)
    {
        return fft11avx256fp64;
    }
    else
    {
        return NULL;
    }
}
