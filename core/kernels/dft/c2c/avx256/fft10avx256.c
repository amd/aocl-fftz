// Copyright Advanced Micro Devices, Inc.
// SPDX-License-Identifier: BSD-3-Clause

/** @file fft10avx256.c
 *
 *  @brief Radix-10 FFT kernel with AVX-256 operations using x86 SIMD intrinsics
 *
 *  This file contains the DIT radix-10 FFT implementations using AVX-256
 *  operations for single-precision and double-precision inputs.
 *
 *  @author Jeya R
 *
 */

#include "core/kernels/kernel.h"
#include "core/kernels/simd_includes/simd_common.h"

static const ops_cycles_t ops_cnt[NUM_PRECISIONS] = {{0, 12, 42, 80, 14, 24},
                                                     {0, 12, 42, 40,  4, 24}};

ops_cycles_t get_ops_cnt_fft10avx256(UINT8 precision, UINT8 direction)
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

static VOID fft10avx256fp32(VOID *in_real, VOID *in_imag, VOID *out_real,
                            VOID *out_imag, INTP n, aoclfftz_strides_t *strides,
                            VOID *twd, UINT8 flag)
{
    AOCLFFTZ_LOG(DEBUG, global_logger_mode, "Enter");
    const FLOAT CRTM_10[4] = {
        0.55901699437494742410229341718281905886015458990288,
        0.58778525229247315738615484497912915412138427663885,
        0.25000000000000000000000000000000000000000000000000,
        0.95105651629515357211643933337938214340569863400000};

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
    INTP N = n / NUM_SETS_256_S;
    INTP remaining_sets = n % NUM_SETS_256_S;
    INTP count;
    FLOAT *curr_in, *curr_out;

    __m256 v_in0, v_in1, v_in2, v_in3, v_in4, v_in5, v_in6, v_in7, v_in8, v_in9;
    __m256 v_out0, v_out1, v_out2, v_out3, v_out4, v_out5, v_out6, v_out7,
           v_out8;
    __m256 v_out9, v_cv1, v_cv2, v_cv3, v_cv4, v_tv1, v_tv2;
    __m256 v_av1, v_av2, v_av3, v_av4, v_av5, v_av6, v_av7, v_av8, v_av9;

    __m256 v_C1 = _mm256_broadcast_ss(&CRTM_10[0]);
    __m256 v_C2 = _mm256_broadcast_ss(&CRTM_10[1]);
    v_C2 = _mm256_xor_ps(v_C2, _neg_256_f[flag].s);
    __m256 v_C3 = _mm256_broadcast_ss(&CRTM_10[2]);
    __m256 v_C4 = _mm256_broadcast_ss(&CRTM_10[3]);
    v_C4 = _mm256_xor_ps(v_C4, _neg_256_f[flag].s);

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

        // common operations
        v_av1 = _mm256_add_ps(v_in1, v_in9);
        v_av2 = _mm256_add_ps(v_in2, v_in8);
        v_av3 = _mm256_add_ps(v_in3, v_in7);
        v_av4 = _mm256_add_ps(v_in4, v_in6);
        v_av5 = _mm256_sub_ps(v_in9, v_in1);
        v_av6 = _mm256_sub_ps(v_in8, v_in2);
        v_av7 = _mm256_sub_ps(v_in7, v_in3);
        v_av8 = _mm256_sub_ps(v_in6, v_in4);

        // Output point 6:X[5]
        v_av9 = _mm256_sub_ps(v_in0, v_in5);
        v_cv1 = _mm256_sub_ps(v_av2, v_av3);
        v_cv2 = _mm256_sub_ps(v_av1, v_av4);
        v_cv3 = _mm256_sub_ps(v_cv1, v_cv2);
        v_out5 = _mm256_add_ps(v_av9, v_cv3);

        // Output point 2:X[1]
        v_cv4 = _mm256_add_ps(v_cv1, v_cv2);
        v_cv1 = _mm256_sub_ps(v_av9, _mm256_mul_ps(v_C3, v_cv3));
        v_tv1 = _mm256_mul_ps(v_C1, v_cv4);
        v_cv2 = _mm256_add_ps(v_cv1, v_tv1);
        v_cv3 = _mm256_add_ps(v_av6, v_av7);
        v_cv4 = _mm256_add_ps(v_av5, v_av8);
        v_tv2 = _mm256_add_ps(_mm256_mul_ps(v_C4, v_cv3),
                              _mm256_mul_ps(v_C2, v_cv4));
        v_tv2 = SWAP_RI_256_S(CONJ_256_S(v_tv2));
        v_out1 = _mm256_add_ps(v_cv2, v_tv2);
        // Output point 10:X[9]
        v_out9 = _mm256_sub_ps(v_cv2, v_tv2);

        // Output point 4:X[3]
        v_cv2 = _mm256_sub_ps(v_cv1, v_tv1);
        v_tv2 = _mm256_sub_ps(_mm256_mul_ps(v_C4, v_cv4),
                              _mm256_mul_ps(v_C2, v_cv3));
        v_tv2 = SWAP_RI_256_S(CONJ_256_S(v_tv2));
        v_out3 = _mm256_add_ps(v_cv2, v_tv2);
        // Output point 8:X[7]
        v_out7 = _mm256_sub_ps(v_cv2, v_tv2);

        // Output point 1:X[0]
        v_av9 = _mm256_add_ps(v_in0, v_in5);
        v_cv1 = _mm256_add_ps(v_av2, v_av3);
        v_cv2 = _mm256_add_ps(v_av1, v_av4);
        v_cv3 = _mm256_add_ps(v_cv2, v_cv1);
        v_out0 = _mm256_add_ps(v_av9, v_cv3);

        // Output point 3:X[2]
        v_cv4 = _mm256_sub_ps(v_cv2, v_cv1);
        v_cv1 = _mm256_sub_ps(v_av9, _mm256_mul_ps(v_C3, v_cv3));
        v_tv1 = _mm256_mul_ps(v_C1, v_cv4);
        v_cv2 = _mm256_add_ps(v_cv1, v_tv1);

        v_cv3 = _mm256_sub_ps(v_av6, v_av7);
        v_cv4 = _mm256_sub_ps(v_av5, v_av8);
        v_tv2 = _mm256_add_ps(_mm256_mul_ps(v_C4, v_cv4),
                              _mm256_mul_ps(v_C2, v_cv3));
        v_tv2 = SWAP_RI_256_S(CONJ_256_S(v_tv2));
        v_out2 = _mm256_add_ps(v_cv2, v_tv2);
        // Output point 9:X[8]
        v_out8 = _mm256_sub_ps(v_cv2, v_tv2);

        // Output point 7:X[6]
        v_cv2 = _mm256_sub_ps(v_cv1, v_tv1);
        v_tv2 = _mm256_sub_ps(_mm256_mul_ps(v_C4, v_cv3),
                              _mm256_mul_ps(v_C2, v_cv4));
        v_tv2 = SWAP_RI_256_S(CONJ_256_S(v_tv2));
        v_out6 = _mm256_add_ps(v_cv2, v_tv2);
        // Output point 5:X[4]
        v_out4 = _mm256_sub_ps(v_cv2, v_tv2);

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

        in_r += NUM_SETS_256_S * v_in_stride;
        out_r += NUM_SETS_256_S * v_out_stride;
    }
    // tail cases
    if (remaining_sets >= NUM_SETS_128_S)
    {
        __m128 v_in0, v_in1, v_in2, v_in3, v_in4, v_in5, v_in6, v_in7, v_in8,
               v_in9;
        __m128 v_out0, v_out1, v_out2, v_out3, v_out4, v_out5, v_out6, v_out7,
               v_out8;
        __m128 v_out9, v_cv1, v_cv2, v_cv3, v_cv4, v_tv1, v_tv2;
        __m128 v_av1, v_av2, v_av3, v_av4, v_av5, v_av6, v_av7, v_av8, v_av9;

        __m128 v_K1 = _mm256_castps256_ps128(v_C1);
        __m128 v_K2 = _mm256_castps256_ps128(v_C2);
        __m128 v_K3 = _mm256_castps256_ps128(v_C3);
        __m128 v_K4 = _mm256_castps256_ps128(v_C4);

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

        // common operations
        v_av1 = _mm_add_ps(v_in1, v_in9);
        v_av2 = _mm_add_ps(v_in2, v_in8);
        v_av3 = _mm_add_ps(v_in3, v_in7);
        v_av4 = _mm_add_ps(v_in4, v_in6);
        v_av5 = _mm_sub_ps(v_in9, v_in1);
        v_av6 = _mm_sub_ps(v_in8, v_in2);
        v_av7 = _mm_sub_ps(v_in7, v_in3);
        v_av8 = _mm_sub_ps(v_in6, v_in4);
        // Output point 6:X[5]
        v_av9 = _mm_sub_ps(v_in0, v_in5);
        v_cv1 = _mm_sub_ps(v_av2, v_av3);
        v_cv2 = _mm_sub_ps(v_av1, v_av4);
        v_cv3 = _mm_sub_ps(v_cv1, v_cv2);
        v_out5 = _mm_add_ps(v_av9, v_cv3);

        // Output point 2:X[1]
        v_cv4 = _mm_add_ps(v_cv1, v_cv2);
        v_cv1 = _mm_sub_ps(v_av9, _mm_mul_ps(v_K3, v_cv3));
        v_tv1 = _mm_mul_ps(v_K1, v_cv4);
        v_cv2 = _mm_add_ps(v_cv1, v_tv1);
        v_cv3 = _mm_add_ps(v_av6, v_av7);
        v_cv4 = _mm_add_ps(v_av5, v_av8);
        v_tv2 = _mm_add_ps(_mm_mul_ps(v_K4, v_cv3), _mm_mul_ps(v_K2, v_cv4));
        v_tv2 = SWAP_RI_128_S(CONJ_128_S(v_tv2));
        v_out1 = _mm_add_ps(v_cv2, v_tv2);
        // Output point 10:X[9]
        v_out9 = _mm_sub_ps(v_cv2, v_tv2);

        // Output point 4:X[3]
        v_cv2 = _mm_sub_ps(v_cv1, v_tv1);
        v_tv2 = _mm_sub_ps(_mm_mul_ps(v_K4, v_cv4), _mm_mul_ps(v_K2, v_cv3));
        v_tv2 = SWAP_RI_128_S(CONJ_128_S(v_tv2));
        v_out3 = _mm_add_ps(v_cv2, v_tv2);
        // Output point 8:X[7]
        v_out7 = _mm_sub_ps(v_cv2, v_tv2);

        // Output point 1:X[0]
        v_av9 = _mm_add_ps(v_in0, v_in5);
        v_cv1 = _mm_add_ps(v_av2, v_av3);
        v_cv2 = _mm_add_ps(v_av1, v_av4);
        v_cv3 = _mm_add_ps(v_cv2, v_cv1);
        v_out0 = _mm_add_ps(v_av9, v_cv3);

        // Output point 3:X[2]
        v_cv4 = _mm_sub_ps(v_cv2, v_cv1);
        v_cv1 = _mm_sub_ps(v_av9, _mm_mul_ps(v_K3, v_cv3));
        v_tv1 = _mm_mul_ps(v_K1, v_cv4);
        v_cv2 = _mm_add_ps(v_cv1, v_tv1);

        v_cv3 = _mm_sub_ps(v_av6, v_av7);
        v_cv4 = _mm_sub_ps(v_av5, v_av8);
        v_tv2 = _mm_add_ps(_mm_mul_ps(v_K4, v_cv4), _mm_mul_ps(v_K2, v_cv3));
        v_tv2 = SWAP_RI_128_S(CONJ_128_S(v_tv2));
        v_out2 = _mm_add_ps(v_cv2, v_tv2);
        // Output point 9:X[8]
        v_out8 = _mm_sub_ps(v_cv2, v_tv2);

        // Output point 7:X[6]
        v_cv2 = _mm_sub_ps(v_cv1, v_tv1);
        v_tv2 = _mm_sub_ps(_mm_mul_ps(v_K4, v_cv3), _mm_mul_ps(v_K2, v_cv4));
        v_tv2 = SWAP_RI_128_S(CONJ_128_S(v_tv2));
        v_out6 = _mm_add_ps(v_cv2, v_tv2);
        // Output point 5:X[4]
        v_out4 = _mm_sub_ps(v_cv2, v_tv2);

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

        in_r = in_r + (v_in_stride << 1);
        out_r = out_r + (v_out_stride << 1);
    }
    if (remaining_sets & 1)
    {
        __m128 v_in0, v_in1, v_in2, v_in3, v_in4, v_in5, v_in6, v_in7, v_in8,
               v_in9;
        __m128 v_out0, v_out1, v_out2, v_out3, v_out4, v_out5, v_out6, v_out7,
               v_out8;
        __m128 v_out9, v_cv1, v_cv2, v_cv3, v_cv4, v_tv1, v_tv2;
        __m128 v_av1, v_av2, v_av3, v_av4, v_av5, v_av6, v_av7, v_av8, v_av9;

        __m128 v_K1 = _mm256_castps256_ps128(v_C1);
        __m128 v_K2 = _mm256_castps256_ps128(v_C2);
        __m128 v_K3 = _mm256_castps256_ps128(v_C3);
        __m128 v_K4 = _mm256_castps256_ps128(v_C4);

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

        // common operations
        v_av1 = _mm_add_ps(v_in1, v_in9);
        v_av2 = _mm_add_ps(v_in2, v_in8);
        v_av3 = _mm_add_ps(v_in3, v_in7);
        v_av4 = _mm_add_ps(v_in4, v_in6);
        v_av5 = _mm_sub_ps(v_in9, v_in1);
        v_av6 = _mm_sub_ps(v_in8, v_in2);
        v_av7 = _mm_sub_ps(v_in7, v_in3);
        v_av8 = _mm_sub_ps(v_in6, v_in4);
        // Output point 6:X[5]
        v_av9 = _mm_sub_ps(v_in0, v_in5);
        v_cv1 = _mm_sub_ps(v_av2, v_av3);
        v_cv2 = _mm_sub_ps(v_av1, v_av4);
        v_cv3 = _mm_sub_ps(v_cv1, v_cv2);
        v_out5 = _mm_add_ps(v_av9, v_cv3);

        // Output point 2:X[1]
        v_cv4 = _mm_add_ps(v_cv1, v_cv2);
        v_cv1 = _mm_sub_ps(v_av9, _mm_mul_ps(v_K3, v_cv3));
        v_tv1 = _mm_mul_ps(v_K1, v_cv4);
        v_cv2 = _mm_add_ps(v_cv1, v_tv1);
        v_cv3 = _mm_add_ps(v_av6, v_av7);
        v_cv4 = _mm_add_ps(v_av5, v_av8);
        v_tv2 = _mm_add_ps(_mm_mul_ps(v_K4, v_cv3), _mm_mul_ps(v_K2, v_cv4));
        v_tv2 = SWAP_RI_128_S(CONJ_128_S(v_tv2));
        v_out1 = _mm_add_ps(v_cv2, v_tv2);
        // Output point 10:X[9]
        v_out9 = _mm_sub_ps(v_cv2, v_tv2);

        // Output point 4:X[3]
        v_cv2 = _mm_sub_ps(v_cv1, v_tv1);
        v_tv2 = _mm_sub_ps(_mm_mul_ps(v_K4, v_cv4), _mm_mul_ps(v_K2, v_cv3));
        v_tv2 = SWAP_RI_128_S(CONJ_128_S(v_tv2));
        v_out3 = _mm_add_ps(v_cv2, v_tv2);
        // Output point 8:X[7]
        v_out7 = _mm_sub_ps(v_cv2, v_tv2);

        // Output point 1:X[0]
        v_av9 = _mm_add_ps(v_in0, v_in5);
        v_cv1 = _mm_add_ps(v_av2, v_av3);
        v_cv2 = _mm_add_ps(v_av1, v_av4);
        v_cv3 = _mm_add_ps(v_cv2, v_cv1);
        v_out0 = _mm_add_ps(v_av9, v_cv3);

        // Output point 3:X[2]
        v_cv4 = _mm_sub_ps(v_cv2, v_cv1);
        v_cv1 = _mm_sub_ps(v_av9, _mm_mul_ps(v_K3, v_cv3));
        v_tv1 = _mm_mul_ps(v_K1, v_cv4);
        v_cv2 = _mm_add_ps(v_cv1, v_tv1);

        v_cv3 = _mm_sub_ps(v_av6, v_av7);
        v_cv4 = _mm_sub_ps(v_av5, v_av8);
        v_tv2 = _mm_add_ps(_mm_mul_ps(v_K4, v_cv4), _mm_mul_ps(v_K2, v_cv3));
        v_tv2 = SWAP_RI_128_S(CONJ_128_S(v_tv2));
        v_out2 = _mm_add_ps(v_cv2, v_tv2);
        // Output point 9:X[8]
        v_out8 = _mm_sub_ps(v_cv2, v_tv2);

        // Output point 7:X[6]
        v_cv2 = _mm_sub_ps(v_cv1, v_tv1);
        v_tv2 = _mm_sub_ps(_mm_mul_ps(v_K4, v_cv3), _mm_mul_ps(v_K2, v_cv4));
        v_tv2 = SWAP_RI_128_S(CONJ_128_S(v_tv2));
        v_out6 = _mm_add_ps(v_cv2, v_tv2);
        // Output point 5:X[4]
        v_out4 = _mm_sub_ps(v_cv2, v_tv2);

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
    }
    AOCLFFTZ_LOG(DEBUG, global_logger_mode, "Exit");
}

static VOID fft10avx256fp64(VOID *in_real, VOID *in_imag, VOID *out_real,
                            VOID *out_imag, INTP n, aoclfftz_strides_t *strides,
                            VOID *twd, UINT8 flag)
{
    AOCLFFTZ_LOG(DEBUG, global_logger_mode, "Enter");
    const DOUBLE CRTM_10[4] = {
        0.55901699437494742410229341718281905886015458990288,
        0.58778525229247315738615484497912915412138427663885,
        0.25000000000000000000000000000000000000000000000000,
        0.95105651629515357211643933337938214340569863400000};

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
    UINT8 is_contiguous_in = (v_in_stride == DATA_STRIDE);
    INTP v_out_stride = strides->v_out_stride;
    UINT8 is_contiguous_out = (v_out_stride == DATA_STRIDE);
    INTP N = n / NUM_SETS_256_D;
    INTP count;
    DOUBLE *curr_in, *curr_out;

    __m256d v_in0, v_in1, v_in2, v_in3, v_in4, v_in5, v_in6, v_in7, v_in8,
            v_in9;
    __m256d v_out0, v_out1, v_out2, v_out3, v_out4, v_out5, v_out6, v_out7,
            v_out8;
    __m256d v_out9, v_cv1, v_cv2, v_cv3, v_cv4, v_tv1, v_tv2;
    __m256d v_av1, v_av2, v_av3, v_av4, v_av5, v_av6, v_av7, v_av8, v_av9;

    __m256d v_C1 = _mm256_broadcast_sd(&CRTM_10[0]);
    __m256d v_C2 = _mm256_broadcast_sd(&CRTM_10[1]);
    v_C2 = _mm256_xor_pd(v_C2, _neg_256_d[flag].d);
    __m256d v_C3 = _mm256_broadcast_sd(&CRTM_10[2]);
    __m256d v_C4 = _mm256_broadcast_sd(&CRTM_10[3]);
    v_C4 = _mm256_xor_pd(v_C4, _neg_256_d[flag].d);

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

        // common operations
        v_av1 = _mm256_add_pd(v_in1, v_in9);
        v_av2 = _mm256_add_pd(v_in2, v_in8);
        v_av3 = _mm256_add_pd(v_in3, v_in7);
        v_av4 = _mm256_add_pd(v_in4, v_in6);
        v_av5 = _mm256_sub_pd(v_in9, v_in1);
        v_av6 = _mm256_sub_pd(v_in8, v_in2);
        v_av7 = _mm256_sub_pd(v_in7, v_in3);
        v_av8 = _mm256_sub_pd(v_in6, v_in4);

        // Output point 6:X[5]
        v_av9 = _mm256_sub_pd(v_in0, v_in5);
        v_cv1 = _mm256_sub_pd(v_av2, v_av3);
        v_cv2 = _mm256_sub_pd(v_av1, v_av4);
        v_cv3 = _mm256_sub_pd(v_cv1, v_cv2);
        v_out5 = _mm256_add_pd(v_av9, v_cv3);

        // Output point 2:X[1]
        v_cv4 = _mm256_add_pd(v_cv1, v_cv2);
        v_cv1 = _mm256_sub_pd(v_av9, _mm256_mul_pd(v_C3, v_cv3));
        v_tv1 = _mm256_mul_pd(v_C1, v_cv4);
        v_cv2 = _mm256_add_pd(v_cv1, v_tv1);
        v_cv3 = _mm256_add_pd(v_av6, v_av7);
        v_cv4 = _mm256_add_pd(v_av5, v_av8);
        v_tv2 = _mm256_add_pd(_mm256_mul_pd(v_C4, v_cv3),
                              _mm256_mul_pd(v_C2, v_cv4));
        v_tv2 = SWAP_RI_256_D(CONJ_256_D(v_tv2));
        v_out1 = _mm256_add_pd(v_cv2, v_tv2);
        // Output point 10:X[9]
        v_out9 = _mm256_sub_pd(v_cv2, v_tv2);

        // Output point 4:X[3]
        v_cv2 = _mm256_sub_pd(v_cv1, v_tv1);
        v_tv2 = _mm256_sub_pd(_mm256_mul_pd(v_C4, v_cv4),
                              _mm256_mul_pd(v_C2, v_cv3));
        v_tv2 = SWAP_RI_256_D(CONJ_256_D(v_tv2));
        v_out3 = _mm256_add_pd(v_cv2, v_tv2);
        // Output point 8:X[7]
        v_out7 = _mm256_sub_pd(v_cv2, v_tv2);

        // Output point 1:X[0]
        v_av9 = _mm256_add_pd(v_in0, v_in5);
        v_cv1 = _mm256_add_pd(v_av2, v_av3);
        v_cv2 = _mm256_add_pd(v_av1, v_av4);
        v_cv3 = _mm256_add_pd(v_cv2, v_cv1);
        v_out0 = _mm256_add_pd(v_av9, v_cv3);

        // Output point 3:X[2]
        v_cv4 = _mm256_sub_pd(v_cv2, v_cv1);
        v_cv1 = _mm256_sub_pd(v_av9, _mm256_mul_pd(v_C3, v_cv3));
        v_tv1 = _mm256_mul_pd(v_C1, v_cv4);
        v_cv2 = _mm256_add_pd(v_cv1, v_tv1);

        v_cv3 = _mm256_sub_pd(v_av6, v_av7);
        v_cv4 = _mm256_sub_pd(v_av5, v_av8);
        v_tv2 = _mm256_add_pd(_mm256_mul_pd(v_C4, v_cv4),
                              _mm256_mul_pd(v_C2, v_cv3));
        v_tv2 = SWAP_RI_256_D(CONJ_256_D(v_tv2));
        v_out2 = _mm256_add_pd(v_cv2, v_tv2);
        // Output point 9:X[8]
        v_out8 = _mm256_sub_pd(v_cv2, v_tv2);

        // Output point 7:X[6]
        v_cv2 = _mm256_sub_pd(v_cv1, v_tv1);
        v_tv2 = _mm256_sub_pd(_mm256_mul_pd(v_C4, v_cv3),
                              _mm256_mul_pd(v_C2, v_cv4));
        v_tv2 = SWAP_RI_256_D(CONJ_256_D(v_tv2));
        v_out6 = _mm256_add_pd(v_cv2, v_tv2);
        // Output point 5:X[4]
        v_out4 = _mm256_sub_pd(v_cv2, v_tv2);

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

        in_r += NUM_SETS_256_D * v_in_stride;
        out_r += NUM_SETS_256_D * v_out_stride;
    }
    // tail case
    if (n & 1)
    {
        __m128d v_in0, v_in1, v_in2, v_in3, v_in4, v_in5, v_in6, v_in7, v_in8,
                v_in9;
        __m128d v_out0, v_out1, v_out2, v_out3, v_out4, v_out5, v_out6, v_out7,
                v_out8;
        __m128d v_out9, v_cv1, v_cv2, v_cv3, v_cv4, v_tv1, v_tv2;
        __m128d v_av1, v_av2, v_av3, v_av4, v_av5, v_av6, v_av7, v_av8, v_av9;

        __m128d v_K1 = _mm256_castpd256_pd128(v_C1);
        __m128d v_K2 = _mm256_castpd256_pd128(v_C2);
        __m128d v_K3 = _mm256_castpd256_pd128(v_C3);
        __m128d v_K4 = _mm256_castpd256_pd128(v_C4);

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

        // common operations
        v_av1 = _mm_add_pd(v_in1, v_in9);
        v_av2 = _mm_add_pd(v_in2, v_in8);
        v_av3 = _mm_add_pd(v_in3, v_in7);
        v_av4 = _mm_add_pd(v_in4, v_in6);
        v_av5 = _mm_sub_pd(v_in9, v_in1);
        v_av6 = _mm_sub_pd(v_in8, v_in2);
        v_av7 = _mm_sub_pd(v_in7, v_in3);
        v_av8 = _mm_sub_pd(v_in6, v_in4);
        // Output point 6:X[5]
        v_av9 = _mm_sub_pd(v_in0, v_in5);
        v_cv1 = _mm_sub_pd(v_av2, v_av3);
        v_cv2 = _mm_sub_pd(v_av1, v_av4);
        v_cv3 = _mm_sub_pd(v_cv1, v_cv2);
        v_out5 = _mm_add_pd(v_av9, v_cv3);

        // Output point 2:X[1]
        v_cv4 = _mm_add_pd(v_cv1, v_cv2);
        v_cv1 = _mm_sub_pd(v_av9, _mm_mul_pd(v_K3, v_cv3));
        v_tv1 = _mm_mul_pd(v_K1, v_cv4);
        v_cv2 = _mm_add_pd(v_cv1, v_tv1);
        v_cv3 = _mm_add_pd(v_av6, v_av7);
        v_cv4 = _mm_add_pd(v_av5, v_av8);
        v_tv2 = _mm_add_pd(_mm_mul_pd(v_K4, v_cv3), _mm_mul_pd(v_K2, v_cv4));
        v_tv2 = SWAP_RI_128_D(CONJ_128_D(v_tv2));
        v_out1 = _mm_add_pd(v_cv2, v_tv2);
        // Output point 10:X[9]
        v_out9 = _mm_sub_pd(v_cv2, v_tv2);

        // Output point 4:X[3]
        v_cv2 = _mm_sub_pd(v_cv1, v_tv1);
        v_tv2 = _mm_sub_pd(_mm_mul_pd(v_K4, v_cv4), _mm_mul_pd(v_K2, v_cv3));
        v_tv2 = SWAP_RI_128_D(CONJ_128_D(v_tv2));
        v_out3 = _mm_add_pd(v_cv2, v_tv2);
        // Output point 8:X[7]
        v_out7 = _mm_sub_pd(v_cv2, v_tv2);

        // Output point 1:X[0]
        v_av9 = _mm_add_pd(v_in0, v_in5);
        v_cv1 = _mm_add_pd(v_av2, v_av3);
        v_cv2 = _mm_add_pd(v_av1, v_av4);
        v_cv3 = _mm_add_pd(v_cv2, v_cv1);
        v_out0 = _mm_add_pd(v_av9, v_cv3);

        // Output point 3:X[2]
        v_cv4 = _mm_sub_pd(v_cv2, v_cv1);
        v_cv1 = _mm_sub_pd(v_av9, _mm_mul_pd(v_K3, v_cv3));
        v_tv1 = _mm_mul_pd(v_K1, v_cv4);
        v_cv2 = _mm_add_pd(v_cv1, v_tv1);

        v_cv3 = _mm_sub_pd(v_av6, v_av7);
        v_cv4 = _mm_sub_pd(v_av5, v_av8);
        v_tv2 = _mm_add_pd(_mm_mul_pd(v_K4, v_cv4), _mm_mul_pd(v_K2, v_cv3));
        v_tv2 = SWAP_RI_128_D(CONJ_128_D(v_tv2));
        v_out2 = _mm_add_pd(v_cv2, v_tv2);
        // Output point 9:X[8]
        v_out8 = _mm_sub_pd(v_cv2, v_tv2);

        // Output point 7:X[6]
        v_cv2 = _mm_sub_pd(v_cv1, v_tv1);
        v_tv2 = _mm_sub_pd(_mm_mul_pd(v_K4, v_cv3), _mm_mul_pd(v_K2, v_cv4));
        v_tv2 = SWAP_RI_128_D(CONJ_128_D(v_tv2));
        v_out6 = _mm_add_pd(v_cv2, v_tv2);
        // Output point 5:X[4]
        v_out4 = _mm_sub_pd(v_cv2, v_tv2);

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
    }
    AOCLFFTZ_LOG(DEBUG, global_logger_mode, "Exit");
}

kfft_ register_kernel_fft10avx256(UINT8 precision, UINT8 direction /* unused */)
{
    if (precision == DT_FLOAT)
    {
        return fft10avx256fp32;
    }
    else if (precision == DT_DOUBLE)
    {
        return fft10avx256fp64;
    }
    else
    {
        return NULL;
    }
}
