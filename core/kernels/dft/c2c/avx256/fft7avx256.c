// Copyright Advanced Micro Devices, Inc.
// SPDX-License-Identifier: BSD-3-Clause

/** @file fft7avx256.c
 *
 *  @brief Radix-7 FFT kernel with AVX-256 operations using x86 SIMD intrinsics
 *
 *  This file contains the DIT radix-7 FFT implementations using AVX-256
 *  operations for single-precision and double-precision inputs.
 *
 *  @author Varun Sanjay
 *
 */

#include "core/kernels/kernel.h"
#include "core/kernels/simd_includes/simd_common.h"

static const ops_cycles_t ops_cnt[NUM_PRECISIONS] = {{0, 18, 33, 56, 10, 17},
                                                     {0, 18, 33, 28,  3, 17}};

ops_cycles_t get_ops_cnt_fft7avx256(FFTZ_UINT8 precision, FFTZ_UINT8 direction)
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

static FFTZ_VOID fft7avx256fp32(FFTZ_VOID *in_real, FFTZ_VOID *in_imag,
                                FFTZ_VOID *out_real, FFTZ_VOID *out_imag,
                                FFTZ_INTP n, aoclfftz_strides_t *strides,
                                FFTZ_VOID *twd, FFTZ_UINT8 flag)
{
    AOCLFFTZ_LOG(DEBUG, global_logger_mode, "Enter");
    const FFTZ_FLOAT CRTM_7[6] = {
        +0.222520933956314404288902564496794759466355569,
        +0.900968867902419126236102319507445051165919162,
        +0.623489801858733530525004884004239810632274731,
        +0.433883739117558120475768332848358754609990728,
        +0.781831482468029808708444526674057750232334519,
        +0.974927912181823607018131682993931217232785801};

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

    __m256 v_in0, v_in1, v_in2, v_in3, v_in4, v_in5, v_in6;
    __m256 v_av1, v_av2, v_av3, v_av4, v_av5, v_av6, v_av7;
    __m256 v_tv1, v_tv2, v_tv3, v_tv4, v_tv5, v_tv6, v_tv7;
    __m256 v_tv8, v_tv9, v_tv10, v_tv11, v_tv12, v_tv13, v_tv14;
    __m256 v_tv16, v_tv17, v_tv18, v_tv19;
    __m256 v_cv1, v_cv2, v_cv3, v_cv4, v_cv5, v_cv6;
    __m256 v_out0, v_out1, v_out2, v_out3, v_out4, v_out5, v_out6;

    __m256 v_C1 = _mm256_broadcast_ss(&CRTM_7[0]);
    __m256 v_C2 = _mm256_broadcast_ss(&CRTM_7[1]);
    __m256 v_C3 = _mm256_broadcast_ss(&CRTM_7[2]);
    __m256 v_C4 = _mm256_broadcast_ss(&CRTM_7[3]);
    v_C4 = _mm256_xor_ps(v_C4, _neg_256_f[flag].s);
    __m256 v_C5 = _mm256_broadcast_ss(&CRTM_7[4]);
    v_C5 = _mm256_xor_ps(v_C5, _neg_256_f[flag].s);
    __m256 v_C6 = _mm256_broadcast_ss(&CRTM_7[5]);
    v_C6 = _mm256_xor_ps(v_C6, _neg_256_f[flag].s);

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

        // common calculations
        v_av1 = _mm256_add_ps(v_in2, v_in5);
        v_av2 = _mm256_sub_ps(v_in2, v_in5);
        v_av3 = _mm256_add_ps(v_in3, v_in4);
        v_av4 = _mm256_sub_ps(v_in3, v_in4);
        v_av5 = _mm256_add_ps(v_in1, v_in6);
        v_av6 = _mm256_sub_ps(v_in1, v_in6);
        v_av7 = _mm256_add_ps(_mm256_add_ps(v_av1, v_av3), v_av5);

        v_tv1 = _mm256_mul_ps(v_av1, v_C1);
        v_tv2 = _mm256_mul_ps(v_av3, v_C1);
        v_tv3 = _mm256_mul_ps(v_av5, v_C1);
        v_tv4 = _mm256_mul_ps(v_av1, v_C2);
        v_tv5 = _mm256_mul_ps(v_av3, v_C2);
        v_tv6 = _mm256_mul_ps(v_av5, v_C2);
        v_tv7 = _mm256_mul_ps(v_av1, v_C3);
        v_tv8 = _mm256_mul_ps(v_av3, v_C3);
        v_tv9 = _mm256_mul_ps(v_av5, v_C3);
        v_tv10 = _mm256_mul_ps(v_av2, v_C4);
        v_tv11 = _mm256_mul_ps(v_av4, v_C4);
        v_tv12 = _mm256_mul_ps(v_av6, v_C4);
        v_tv13 = _mm256_mul_ps(v_av2, v_C5);
        v_tv14 = _mm256_mul_ps(v_av4, v_C5);
        v_tv16 = _mm256_mul_ps(v_av6, v_C5);
        v_tv17 = _mm256_mul_ps(v_av2, v_C6);
        v_tv18 = _mm256_mul_ps(v_av4, v_C6);
        v_tv19 = _mm256_mul_ps(v_av6, v_C6);

        v_cv1 = _mm256_sub_ps(_mm256_add_ps(v_tv1, v_tv5), v_tv9);
        v_cv2 = _mm256_add_ps(_mm256_add_ps(v_tv11, v_tv16), v_tv17);
        v_cv3 = _mm256_sub_ps(_mm256_add_ps(v_tv3, v_tv4), v_tv8);
        v_cv4 = _mm256_sub_ps(_mm256_add_ps(v_tv10, v_tv14), v_tv19);
        v_cv5 = _mm256_sub_ps(_mm256_add_ps(v_tv2, v_tv6), v_tv7);
        v_cv6 = _mm256_add_ps(_mm256_sub_ps(v_tv12, v_tv13), v_tv18);

        v_cv2 = SWAP_RI_256_S(CONJ_256_S(v_cv2));
        v_cv4 = SWAP_RI_256_S(CONJ_256_S(v_cv4));
        v_cv6 = SWAP_RI_256_S(CONJ_256_S(v_cv6));

        // Output point 1:X[0]
        v_out0 = _mm256_add_ps(v_in0, v_av7);
        // Output point 7:X[6]
        v_out6 = _mm256_add_ps(_mm256_sub_ps(v_in0, v_cv1), v_cv2);
        // Output point 2:X[1]
        v_out1 = _mm256_sub_ps(_mm256_sub_ps(v_in0, v_cv1), v_cv2);
        // Output point 3:X[2]
        v_out2 = _mm256_add_ps(_mm256_sub_ps(v_in0, v_cv3), v_cv4);
        // Output point 6:X[5]
        v_out5 = _mm256_sub_ps(_mm256_sub_ps(v_in0, v_cv3), v_cv4);
        // Output point 5:X[4]
        v_out4 = _mm256_add_ps(_mm256_sub_ps(v_in0, v_cv5), v_cv6);
        // Output point 4:X[3]
        v_out3 = _mm256_sub_ps(_mm256_sub_ps(v_in0, v_cv5), v_cv6);

        if (out_strides[1] == DATA_STRIDE)
        {
            TRANSPOSE_ST4_256_S(curr_out, v_out_stride, v_out0, v_out1, v_out2,
                                v_out3);
            curr_out = curr_out + NUM_SETS_256_S * DATA_STRIDE;
            TRANSPOSE_ST2_256_S(curr_out, v_out_stride, v_out4, v_out5);
            curr_out = curr_out + 2 * DATA_STRIDE;
            SCATTER4_256_S_STRIDED(curr_out, v_out_stride, v_out6);
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
        }

        in_r += NUM_SETS_256_S * v_in_stride;
        out_r += NUM_SETS_256_S * v_out_stride;
    }
    // tail cases
    if (remaining_sets >= NUM_SETS_128_S)
    {
        __m128 v_in0, v_in1, v_in2, v_in3, v_in4, v_in5, v_in6;
        __m128 v_av1, v_av2, v_av3, v_av4, v_av5, v_av6, v_av7;
        __m128 v_tv1, v_tv2, v_tv3, v_tv4, v_tv5, v_tv6, v_tv7;
        __m128 v_tv8, v_tv9, v_tv10, v_tv11, v_tv12, v_tv13, v_tv14;
        __m128 v_tv16, v_tv17, v_tv18, v_tv19;
        __m128 v_cv1, v_cv2, v_cv3, v_cv4, v_cv5, v_cv6;
        __m128 v_out0, v_out1, v_out2, v_out3, v_out4, v_out5, v_out6;

        __m128 v_K1 = _mm256_castps256_ps128(v_C1);
        __m128 v_K2 = _mm256_castps256_ps128(v_C2);
        __m128 v_K3 = _mm256_castps256_ps128(v_C3);
        __m128 v_K4 = _mm256_castps256_ps128(v_C4);
        __m128 v_K5 = _mm256_castps256_ps128(v_C5);
        __m128 v_K6 = _mm256_castps256_ps128(v_C6);

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

        // common calculations
        v_av1 = _mm_add_ps(v_in2, v_in5);
        v_av2 = _mm_sub_ps(v_in2, v_in5);
        v_av3 = _mm_add_ps(v_in3, v_in4);
        v_av4 = _mm_sub_ps(v_in3, v_in4);
        v_av5 = _mm_add_ps(v_in1, v_in6);
        v_av6 = _mm_sub_ps(v_in1, v_in6);
        v_av7 = _mm_add_ps(_mm_add_ps(v_av1, v_av3), v_av5);

        v_tv1 = _mm_mul_ps(v_av1, v_K1);
        v_tv2 = _mm_mul_ps(v_av3, v_K1);
        v_tv3 = _mm_mul_ps(v_av5, v_K1);
        v_tv4 = _mm_mul_ps(v_av1, v_K2);
        v_tv5 = _mm_mul_ps(v_av3, v_K2);
        v_tv6 = _mm_mul_ps(v_av5, v_K2);
        v_tv7 = _mm_mul_ps(v_av1, v_K3);
        v_tv8 = _mm_mul_ps(v_av3, v_K3);
        v_tv9 = _mm_mul_ps(v_av5, v_K3);
        v_tv10 = _mm_mul_ps(v_av2, v_K4);
        v_tv11 = _mm_mul_ps(v_av4, v_K4);
        v_tv12 = _mm_mul_ps(v_av6, v_K4);
        v_tv13 = _mm_mul_ps(v_av2, v_K5);
        v_tv14 = _mm_mul_ps(v_av4, v_K5);
        v_tv16 = _mm_mul_ps(v_av6, v_K5);
        v_tv17 = _mm_mul_ps(v_av2, v_K6);
        v_tv18 = _mm_mul_ps(v_av4, v_K6);
        v_tv19 = _mm_mul_ps(v_av6, v_K6);

        v_cv1 = _mm_sub_ps(_mm_add_ps(v_tv1, v_tv5), v_tv9);
        v_cv2 = _mm_add_ps(_mm_add_ps(v_tv11, v_tv16), v_tv17);
        v_cv3 = _mm_sub_ps(_mm_add_ps(v_tv3, v_tv4), v_tv8);
        v_cv4 = _mm_sub_ps(_mm_add_ps(v_tv10, v_tv14), v_tv19);
        v_cv5 = _mm_sub_ps(_mm_add_ps(v_tv2, v_tv6), v_tv7);
        v_cv6 = _mm_add_ps(_mm_sub_ps(v_tv12, v_tv13), v_tv18);

        v_cv2 = SWAP_RI_128_S(CONJ_128_S(v_cv2));
        v_cv4 = SWAP_RI_128_S(CONJ_128_S(v_cv4));
        v_cv6 = SWAP_RI_128_S(CONJ_128_S(v_cv6));

        // Output point 1:X[0]
        v_out0 = _mm_add_ps(v_in0, v_av7);
        // Output point 7:X[6]
        v_out6 = _mm_add_ps(_mm_sub_ps(v_in0, v_cv1), v_cv2);
        // Output point 2:X[1]
        v_out1 = _mm_sub_ps(_mm_sub_ps(v_in0, v_cv1), v_cv2);
        // Output point 3:X[2]
        v_out2 = _mm_add_ps(_mm_sub_ps(v_in0, v_cv3), v_cv4);
        // Output point 6:X[5]
        v_out5 = _mm_sub_ps(_mm_sub_ps(v_in0, v_cv3), v_cv4);
        // Output point 5:X[4]
        v_out4 = _mm_add_ps(_mm_sub_ps(v_in0, v_cv5), v_cv6);
        // Output point 4:X[3]
        v_out3 = _mm_sub_ps(_mm_sub_ps(v_in0, v_cv5), v_cv6);

        if (out_strides[1] == DATA_STRIDE)
        {
            TRANSPOSE_ST2_128_S(curr_out, v_out_stride, v_out0, v_out1);
            curr_out = curr_out + NUM_SETS_128_S * DATA_STRIDE;
            TRANSPOSE_ST2_128_S(curr_out, v_out_stride, v_out2, v_out3);
            curr_out = curr_out + NUM_SETS_128_S * DATA_STRIDE;
            TRANSPOSE_ST2_128_S(curr_out, v_out_stride, v_out4, v_out5);
            curr_out = curr_out + NUM_SETS_128_S * DATA_STRIDE;
            SCATTER2_128_S_STRIDED(curr_out, v_out_stride, v_out6);
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
        }

        in_r = in_r + (v_in_stride << 1);
        out_r = out_r + (v_out_stride << 1);
    }
    if (remaining_sets & 1)
    {
        __m128 v_in0, v_in1, v_in2, v_in3, v_in4, v_in5, v_in6;
        __m128 v_av1, v_av2, v_av3, v_av4, v_av5, v_av6, v_av7;
        __m128 v_tv1, v_tv2, v_tv3, v_tv4, v_tv5, v_tv6, v_tv7;
        __m128 v_tv8, v_tv9, v_tv10, v_tv11, v_tv12, v_tv13, v_tv14;
        __m128 v_tv16, v_tv17, v_tv18, v_tv19;
        __m128 v_cv1, v_cv2, v_cv3, v_cv4, v_cv5, v_cv6;
        __m128 v_out0, v_out1, v_out2, v_out3, v_out4, v_out5, v_out6;

        __m128 v_K1 = _mm256_castps256_ps128(v_C1);
        __m128 v_K2 = _mm256_castps256_ps128(v_C2);
        __m128 v_K3 = _mm256_castps256_ps128(v_C3);
        __m128 v_K4 = _mm256_castps256_ps128(v_C4);
        __m128 v_K5 = _mm256_castps256_ps128(v_C5);
        __m128 v_K6 = _mm256_castps256_ps128(v_C6);

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

        v_av1 = _mm_add_ps(v_in2, v_in5);
        v_av2 = _mm_sub_ps(v_in2, v_in5);
        v_av3 = _mm_add_ps(v_in3, v_in4);
        v_av4 = _mm_sub_ps(v_in3, v_in4);
        v_av5 = _mm_add_ps(v_in1, v_in6);
        v_av6 = _mm_sub_ps(v_in1, v_in6);
        v_av7 = _mm_add_ps(_mm_add_ps(v_av1, v_av3), v_av5);

        v_tv1 = _mm_mul_ps(v_av1, v_K1);
        v_tv2 = _mm_mul_ps(v_av3, v_K1);
        v_tv3 = _mm_mul_ps(v_av5, v_K1);
        v_tv4 = _mm_mul_ps(v_av1, v_K2);
        v_tv5 = _mm_mul_ps(v_av3, v_K2);
        v_tv6 = _mm_mul_ps(v_av5, v_K2);
        v_tv7 = _mm_mul_ps(v_av1, v_K3);
        v_tv8 = _mm_mul_ps(v_av3, v_K3);
        v_tv9 = _mm_mul_ps(v_av5, v_K3);
        v_tv10 = _mm_mul_ps(v_av2, v_K4);
        v_tv11 = _mm_mul_ps(v_av4, v_K4);
        v_tv12 = _mm_mul_ps(v_av6, v_K4);
        v_tv13 = _mm_mul_ps(v_av2, v_K5);
        v_tv14 = _mm_mul_ps(v_av4, v_K5);
        v_tv16 = _mm_mul_ps(v_av6, v_K5);
        v_tv17 = _mm_mul_ps(v_av2, v_K6);
        v_tv18 = _mm_mul_ps(v_av4, v_K6);
        v_tv19 = _mm_mul_ps(v_av6, v_K6);

        v_cv1 = _mm_sub_ps(_mm_add_ps(v_tv1, v_tv5), v_tv9);
        v_cv2 = _mm_add_ps(_mm_add_ps(v_tv11, v_tv16), v_tv17);
        v_cv3 = _mm_sub_ps(_mm_add_ps(v_tv3, v_tv4), v_tv8);
        v_cv4 = _mm_sub_ps(_mm_add_ps(v_tv10, v_tv14), v_tv19);
        v_cv5 = _mm_sub_ps(_mm_add_ps(v_tv2, v_tv6), v_tv7);
        v_cv6 = _mm_add_ps(_mm_sub_ps(v_tv12, v_tv13), v_tv18);

        v_cv2 = SWAP_RI_128_S(CONJ_128_S(v_cv2));
        v_cv4 = SWAP_RI_128_S(CONJ_128_S(v_cv4));
        v_cv6 = SWAP_RI_128_S(CONJ_128_S(v_cv6));

        // Output point 1:X[0]
        v_out0 = _mm_add_ps(v_in0, v_av7);
        // Output point 7:X[6]
        v_out6 = _mm_add_ps(_mm_sub_ps(v_in0, v_cv1), v_cv2);
        // Output point 2:X[1]
        v_out1 = _mm_sub_ps(_mm_sub_ps(v_in0, v_cv1), v_cv2);
        // Output point 3:X[2]
        v_out2 = _mm_add_ps(_mm_sub_ps(v_in0, v_cv3), v_cv4);
        // Output point 6:X[5]
        v_out5 = _mm_sub_ps(_mm_sub_ps(v_in0, v_cv3), v_cv4);
        // Output point 5:X[4]
        v_out4 = _mm_add_ps(_mm_sub_ps(v_in0, v_cv5), v_cv6);
        // Output point 4:X[3]
        v_out3 = _mm_sub_ps(_mm_sub_ps(v_in0, v_cv5), v_cv6);

        if (out_strides[1] == DATA_STRIDE)
        {
            ST_128_S(curr_out, v_out0, v_out1);
            curr_out = out_r + 2 * DATA_STRIDE;
            ST_128_S(curr_out, v_out2, v_out3);
            curr_out = out_r + 4 * DATA_STRIDE;
            ST_128_S(curr_out, v_out4, v_out5);
            curr_out = out_r + 6 * DATA_STRIDE;
            ST_LOW_128_S(curr_out, v_out6);
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
        }
    }
    AOCLFFTZ_LOG(DEBUG, global_logger_mode, "Exit");
}

static FFTZ_VOID fft7avx256fp64(FFTZ_VOID *in_real, FFTZ_VOID *in_imag,
                                FFTZ_VOID *out_real, FFTZ_VOID *out_imag,
                                FFTZ_INTP n, aoclfftz_strides_t *strides,
                                FFTZ_VOID *twd, FFTZ_UINT8 flag)
{
    AOCLFFTZ_LOG(DEBUG, global_logger_mode, "Enter");
    const FFTZ_DOUBLE CRTM_7[6] = {
        +0.222520933956314404288902564496794759466355569,
        +0.900968867902419126236102319507445051165919162,
        +0.623489801858733530525004884004239810632274731,
        +0.433883739117558120475768332848358754609990728,
        +0.781831482468029808708444526674057750232334519,
        +0.974927912181823607018131682993931217232785801};

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

    __m256d v_in0, v_in1, v_in2, v_in3, v_in4, v_in5, v_in6;
    __m256d v_av1, v_av2, v_av3, v_av4, v_av5, v_av6, v_av7;
    __m256d v_tv1, v_tv2, v_tv3, v_tv4, v_tv5, v_tv6, v_tv7;
    __m256d v_tv8, v_tv9, v_tv10, v_tv11, v_tv12, v_tv13, v_tv14;
    __m256d v_tv16, v_tv17, v_tv18, v_tv19;
    __m256d v_cv1, v_cv2, v_cv3, v_cv4, v_cv5, v_cv6;
    __m256d v_out0, v_out1, v_out2, v_out3, v_out4, v_out5, v_out6;

    __m256d v_C1 = _mm256_broadcast_sd(&CRTM_7[0]);
    __m256d v_C2 = _mm256_broadcast_sd(&CRTM_7[1]);
    __m256d v_C3 = _mm256_broadcast_sd(&CRTM_7[2]);
    __m256d v_C4 = _mm256_broadcast_sd(&CRTM_7[3]);
    v_C4 = _mm256_xor_pd(v_C4, _neg_256_d[flag].d);
    __m256d v_C5 = _mm256_broadcast_sd(&CRTM_7[4]);
    v_C5 = _mm256_xor_pd(v_C5, _neg_256_d[flag].d);
    __m256d v_C6 = _mm256_broadcast_sd(&CRTM_7[5]);
    v_C6 = _mm256_xor_pd(v_C6, _neg_256_d[flag].d);

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

        // common calculations
        v_av1 = _mm256_add_pd(v_in2, v_in5);
        v_av2 = _mm256_sub_pd(v_in2, v_in5);
        v_av3 = _mm256_add_pd(v_in3, v_in4);
        v_av4 = _mm256_sub_pd(v_in3, v_in4);
        v_av5 = _mm256_add_pd(v_in1, v_in6);
        v_av6 = _mm256_sub_pd(v_in1, v_in6);
        v_av7 = _mm256_add_pd(_mm256_add_pd(v_av1, v_av3), v_av5);

        v_tv1 = _mm256_mul_pd(v_av1, v_C1);
        v_tv2 = _mm256_mul_pd(v_av3, v_C1);
        v_tv3 = _mm256_mul_pd(v_av5, v_C1);
        v_tv4 = _mm256_mul_pd(v_av1, v_C2);
        v_tv5 = _mm256_mul_pd(v_av3, v_C2);
        v_tv6 = _mm256_mul_pd(v_av5, v_C2);
        v_tv7 = _mm256_mul_pd(v_av1, v_C3);
        v_tv8 = _mm256_mul_pd(v_av3, v_C3);
        v_tv9 = _mm256_mul_pd(v_av5, v_C3);
        v_tv10 = _mm256_mul_pd(v_av2, v_C4);
        v_tv11 = _mm256_mul_pd(v_av4, v_C4);
        v_tv12 = _mm256_mul_pd(v_av6, v_C4);
        v_tv13 = _mm256_mul_pd(v_av2, v_C5);
        v_tv14 = _mm256_mul_pd(v_av4, v_C5);
        v_tv16 = _mm256_mul_pd(v_av6, v_C5);
        v_tv17 = _mm256_mul_pd(v_av2, v_C6);
        v_tv18 = _mm256_mul_pd(v_av4, v_C6);
        v_tv19 = _mm256_mul_pd(v_av6, v_C6);

        v_cv1 = _mm256_sub_pd(_mm256_add_pd(v_tv1, v_tv5), v_tv9);
        v_cv2 = _mm256_add_pd(_mm256_add_pd(v_tv11, v_tv16), v_tv17);
        v_cv3 = _mm256_sub_pd(_mm256_add_pd(v_tv3, v_tv4), v_tv8);
        v_cv4 = _mm256_sub_pd(_mm256_add_pd(v_tv10, v_tv14), v_tv19);
        v_cv5 = _mm256_sub_pd(_mm256_add_pd(v_tv2, v_tv6), v_tv7);
        v_cv6 = _mm256_add_pd(_mm256_sub_pd(v_tv12, v_tv13), v_tv18);

        v_cv2 = SWAP_RI_256_D(CONJ_256_D(v_cv2));
        v_cv4 = SWAP_RI_256_D(CONJ_256_D(v_cv4));
        v_cv6 = SWAP_RI_256_D(CONJ_256_D(v_cv6));

        // Output point 1:X[0]
        v_out0 = _mm256_add_pd(v_in0, v_av7);
        // Output point 7:X[6]
        v_out6 = _mm256_add_pd(_mm256_sub_pd(v_in0, v_cv1), v_cv2);
        // Output point 2:X[1]
        v_out1 = _mm256_sub_pd(_mm256_sub_pd(v_in0, v_cv1), v_cv2);
        // Output point 3:X[2]
        v_out2 = _mm256_add_pd(_mm256_sub_pd(v_in0, v_cv3), v_cv4);
        // Output point 6:X[5]
        v_out5 = _mm256_sub_pd(_mm256_sub_pd(v_in0, v_cv3), v_cv4);
        // Output point 5:X[4]
        v_out4 = _mm256_add_pd(_mm256_sub_pd(v_in0, v_cv5), v_cv6);
        // Output point 4:X[3]
        v_out3 = _mm256_sub_pd(_mm256_sub_pd(v_in0, v_cv5), v_cv6);

        if (out_strides[1] == DATA_STRIDE)
        {
            TRANSPOSE_ST2_256_D(curr_out, v_out_stride, v_out0, v_out1);
            curr_out = curr_out + NUM_SETS_256_D * DATA_STRIDE;
            TRANSPOSE_ST2_256_D(curr_out, v_out_stride, v_out2, v_out3);
            curr_out = curr_out + NUM_SETS_256_D * DATA_STRIDE;
            TRANSPOSE_ST2_256_D(curr_out, v_out_stride, v_out4, v_out5);
            curr_out = curr_out + NUM_SETS_256_D * DATA_STRIDE;
            SCATTER2_256_D_STRIDED(curr_out, v_out_stride, v_out6);
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
        }

        in_r += NUM_SETS_256_D * v_in_stride;
        out_r += NUM_SETS_256_D * v_out_stride;
    }
    // tail case
    if (n & 1)
    {
        __m128d v_in0, v_in1, v_in2, v_in3, v_in4, v_in5, v_in6;
        __m128d v_av1, v_av2, v_av3, v_av4, v_av5, v_av6, v_av7;
        __m128d v_tv1, v_tv2, v_tv3, v_tv4, v_tv5, v_tv6, v_tv7;
        __m128d v_tv8, v_tv9, v_tv10, v_tv11, v_tv12, v_tv13, v_tv14;
        __m128d v_tv16, v_tv17, v_tv18, v_tv19;
        __m128d v_cv1, v_cv2, v_cv3, v_cv4, v_cv5, v_cv6;
        __m128d v_out0, v_out1, v_out2, v_out3, v_out4, v_out5, v_out6;

        __m128d v_K1 = _mm256_castpd256_pd128(v_C1);
        __m128d v_K2 = _mm256_castpd256_pd128(v_C2);
        __m128d v_K3 = _mm256_castpd256_pd128(v_C3);
        __m128d v_K4 = _mm256_castpd256_pd128(v_C4);
        __m128d v_K5 = _mm256_castpd256_pd128(v_C5);
        __m128d v_K6 = _mm256_castpd256_pd128(v_C6);

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

        // common calculations
        v_av1 = _mm_add_pd(v_in2, v_in5);
        v_av2 = _mm_sub_pd(v_in2, v_in5);
        v_av3 = _mm_add_pd(v_in3, v_in4);
        v_av4 = _mm_sub_pd(v_in3, v_in4);
        v_av5 = _mm_add_pd(v_in1, v_in6);
        v_av6 = _mm_sub_pd(v_in1, v_in6);
        v_av7 = _mm_add_pd(_mm_add_pd(v_av1, v_av3), v_av5);

        v_tv1 = _mm_mul_pd(v_av1, v_K1);
        v_tv2 = _mm_mul_pd(v_av3, v_K1);
        v_tv3 = _mm_mul_pd(v_av5, v_K1);
        v_tv4 = _mm_mul_pd(v_av1, v_K2);
        v_tv5 = _mm_mul_pd(v_av3, v_K2);
        v_tv6 = _mm_mul_pd(v_av5, v_K2);
        v_tv7 = _mm_mul_pd(v_av1, v_K3);
        v_tv8 = _mm_mul_pd(v_av3, v_K3);
        v_tv9 = _mm_mul_pd(v_av5, v_K3);
        v_tv10 = _mm_mul_pd(v_av2, v_K4);
        v_tv11 = _mm_mul_pd(v_av4, v_K4);
        v_tv12 = _mm_mul_pd(v_av6, v_K4);
        v_tv13 = _mm_mul_pd(v_av2, v_K5);
        v_tv14 = _mm_mul_pd(v_av4, v_K5);
        v_tv16 = _mm_mul_pd(v_av6, v_K5);
        v_tv17 = _mm_mul_pd(v_av2, v_K6);
        v_tv18 = _mm_mul_pd(v_av4, v_K6);
        v_tv19 = _mm_mul_pd(v_av6, v_K6);

        v_cv1 = _mm_sub_pd(_mm_add_pd(v_tv1, v_tv5), v_tv9);
        v_cv2 = _mm_add_pd(_mm_add_pd(v_tv11, v_tv16), v_tv17);
        v_cv3 = _mm_sub_pd(_mm_add_pd(v_tv3, v_tv4), v_tv8);
        v_cv4 = _mm_sub_pd(_mm_add_pd(v_tv10, v_tv14), v_tv19);
        v_cv5 = _mm_sub_pd(_mm_add_pd(v_tv2, v_tv6), v_tv7);
        v_cv6 = _mm_add_pd(_mm_sub_pd(v_tv12, v_tv13), v_tv18);

        v_cv2 = SWAP_RI_128_D(CONJ_128_D(v_cv2));
        v_cv4 = SWAP_RI_128_D(CONJ_128_D(v_cv4));
        v_cv6 = SWAP_RI_128_D(CONJ_128_D(v_cv6));

        // Output point 1:X[0]
        v_out0 = _mm_add_pd(v_in0, v_av7);
        // Output point 7:X[6]
        v_out6 = _mm_add_pd(_mm_sub_pd(v_in0, v_cv1), v_cv2);
        // Output point 2:X[1]
        v_out1 = _mm_sub_pd(_mm_sub_pd(v_in0, v_cv1), v_cv2);
        // Output point 3:X[2]
        v_out2 = _mm_add_pd(_mm_sub_pd(v_in0, v_cv3), v_cv4);
        // Output point 6:X[5]
        v_out5 = _mm_sub_pd(_mm_sub_pd(v_in0, v_cv3), v_cv4);
        // Output point 5:X[4]
        v_out4 = _mm_add_pd(_mm_sub_pd(v_in0, v_cv5), v_cv6);
        // Output point 4:X[3]
        v_out3 = _mm_sub_pd(_mm_sub_pd(v_in0, v_cv5), v_cv6);

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
    }
    AOCLFFTZ_LOG(DEBUG, global_logger_mode, "Exit");
}

kfft_ register_kernel_fft7avx256(FFTZ_UINT8 precision,
                                 FFTZ_UINT8 direction /* unused */)
{
    if (precision == DT_FLOAT)
    {
        return fft7avx256fp32;
    }
    else if (precision == DT_DOUBLE)
    {
        return fft7avx256fp64;
    }
    else
    {
        return NULL;
    }
}
