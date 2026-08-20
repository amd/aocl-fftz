// Copyright Advanced Micro Devices, Inc.
// SPDX-License-Identifier: BSD-3-Clause

/** @file fft6avx512.c
 *
 *  @brief Radix-6 FFT kernel with AVX-512 operations using x86 SIMD intrinsics
 *
 *  This file contains the DIT radix-6 FFT implementations using AVX-512
 *  operations for single-precision and double-precision inputs.
 *
 *  @author Jeya R
 *
 */

#include "core/kernels/kernel.h"
#include "core/kernels/simd_includes/simd_common_avx512.h"

static const ops_cycles_t ops_cnt[NUM_PRECISIONS] = {{0, 4, 18, 96, 20, 38},
                                                     {0, 4, 18, 48,  2, 38}};

ops_cycles_t get_ops_cnt_fft6avx512(FFTZ_UINT8 precision, FFTZ_UINT8 direction)
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

static FFTZ_VOID fft6avx512fp32(FFTZ_VOID *in_real, FFTZ_VOID *in_imag,
                                FFTZ_VOID *out_real, FFTZ_VOID *out_imag,
                                FFTZ_INTP n, aoclfftz_strides_t *strides,
                                FFTZ_VOID *twd, FFTZ_UINT8 flag)
{
    AOCLFFTZ_LOG(DEBUG, global_logger_mode, "Enter");
    const FFTZ_FLOAT CRTM_6[2] = {
        0.500000000000000000000000000000000000000000000f,
        0.866025403784438646763723170752936183471402627f};

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

    __m512 v_C1 = _mm512_set1_ps(CRTM_6[0]);
    __m512 v_C2 = _mm512_set1_ps(CRTM_6[1]);
    v_C2 = _mm512_xor_ps(v_C2, _neg_512_f[flag].s);

    for (count = 0; count < N; count++)
    {
        __m512 v_in0, v_in1, v_in2, v_in3, v_in4, v_in5;
        __m512 v_av1, v_av2, v_av3, v_av4, v_cv1, v_cv2, v_cv3;
        __m512 v_out0, v_out1, v_out2, v_out3, v_out4, v_out5;

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

        v_av1 = _mm512_add_ps(v_in0, v_in3);
        v_av2 = _mm512_add_ps(v_in2, v_in4);
        v_av3 = _mm512_add_ps(v_in1, v_in5);
        v_av4 = _mm512_add_ps(v_av2, v_av3);
        v_cv1 = _mm512_sub_ps(v_av1, _mm512_mul_ps(v_C1, v_av4));

        // Output point 1:X[0]
        v_out0 = _mm512_add_ps(v_av1, v_av4);

        v_av1 = _mm512_sub_ps(v_av2, v_av3);
        v_av2 = _mm512_sub_ps(v_in0, v_in3);
        v_cv2 = _mm512_sub_ps(v_av2, _mm512_mul_ps(v_C1, v_av1));

        // Output point 4:X[3]
        v_out3 = _mm512_add_ps(v_av2, v_av1);

        v_av1 = _mm512_sub_ps(v_in1, v_in5);
        v_av2 = _mm512_sub_ps(v_in2, v_in4);
        v_av3 = _mm512_sub_ps(v_av1, v_av2);
        v_cv3 = _mm512_mul_ps(v_C2, v_av3);
        v_cv3 = CONJ_512_S(SWAP_RI_512_S(v_cv3));

        // Output point 3:X[2]
        v_out2 = _mm512_add_ps(v_cv1, v_cv3);
        // Output point 5:X[4]
        v_out4 = _mm512_sub_ps(v_cv1, v_cv3);

        v_av3 = _mm512_add_ps(v_av1, v_av2);
        v_cv3 = _mm512_mul_ps(v_C2, v_av3);
        v_cv3 = CONJ_512_S(SWAP_RI_512_S(v_cv3));

        // Output point 2:X[1]
        v_out1 = _mm512_add_ps(v_cv2, v_cv3);
        // Output point 6:X[5]
        v_out5 = _mm512_sub_ps(v_cv2, v_cv3);

        if (out_strides[1] == DATA_STRIDE)
        {
            TRANSPOSE_ST4_512_S(curr_out, v_out_stride, v_out0, v_out1, v_out2,
                                v_out3);
            curr_out = curr_out + 4 * DATA_STRIDE;
            TRANSPOSE_ST2_512_S(curr_out, v_out_stride, v_out4, v_out5);
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
        }

        in_r += NUM_SETS_512_S * v_in_stride;
        out_r += NUM_SETS_512_S * v_out_stride;
    }
    // tail cases
    if (remaining_sets >= NUM_SETS_256_S)
    {
        __m256 v_in0, v_in1, v_in2, v_in3, v_in4, v_in5;
        __m256 v_av1, v_av2, v_av3, v_av4, v_cv1, v_cv2, v_cv3;
        __m256 v_out0, v_out1, v_out2, v_out3, v_out4, v_out5;

        __m256 v_K1 = _mm512_castps512_ps256(v_C1);
        __m256 v_K2 = _mm512_castps512_ps256(v_C2);

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

        v_av1 = _mm256_add_ps(v_in0, v_in3);
        v_av2 = _mm256_add_ps(v_in2, v_in4);
        v_av3 = _mm256_add_ps(v_in1, v_in5);
        v_av4 = _mm256_add_ps(v_av2, v_av3);
        v_cv1 = _mm256_sub_ps(v_av1, _mm256_mul_ps(v_K1, v_av4));

        // Output point 1:X[0]
        v_out0 = _mm256_add_ps(v_av1, v_av4);

        v_av1 = _mm256_sub_ps(v_av2, v_av3);
        v_av2 = _mm256_sub_ps(v_in0, v_in3);
        v_cv2 = _mm256_sub_ps(v_av2, _mm256_mul_ps(v_K1, v_av1));

        // Output point 4:X[3]
        v_out3 = _mm256_add_ps(v_av2, v_av1);

        v_av1 = _mm256_sub_ps(v_in1, v_in5);
        v_av2 = _mm256_sub_ps(v_in2, v_in4);
        v_av3 = _mm256_sub_ps(v_av1, v_av2);
        v_cv3 = _mm256_mul_ps(v_K2, v_av3);
        v_cv3 = CONJ_256_S(SWAP_RI_256_S(v_cv3));

        // Output point 3:X[2]
        v_out2 = _mm256_add_ps(v_cv1, v_cv3);
        // Output point 5:X[4]
        v_out4 = _mm256_sub_ps(v_cv1, v_cv3);

        v_av3 = _mm256_add_ps(v_av1, v_av2);
        v_cv3 = _mm256_mul_ps(v_K2, v_av3);
        v_cv3 = CONJ_256_S(SWAP_RI_256_S(v_cv3));

        // Output point 2:X[1]
        v_out1 = _mm256_add_ps(v_cv2, v_cv3);
        // Output point 6:X[5]
        v_out5 = _mm256_sub_ps(v_cv2, v_cv3);

        if (out_strides[1] == DATA_STRIDE)
        {
            TRANSPOSE_ST4_256_S(curr_out, v_out_stride, v_out0, v_out1, v_out2,
                                v_out3);
            curr_out = curr_out + NUM_SETS_256_S * DATA_STRIDE;
            TRANSPOSE_ST2_256_S(curr_out, v_out_stride, v_out4, v_out5);
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
        }

        in_r += NUM_SETS_256_S * v_in_stride;
        out_r += NUM_SETS_256_S * v_out_stride;
        remaining_sets = remaining_sets % NUM_SETS_256_S;
    }
    // tail cases
    if (remaining_sets >= NUM_SETS_128_S)
    {
        __m128 v_in0, v_in1, v_in2, v_in3, v_in4, v_in5;
        __m128 v_av1, v_av2, v_av3, v_av4, v_cv1, v_cv2, v_cv3;
        __m128 v_out0, v_out1, v_out2, v_out3, v_out4, v_out5;

        __m128 v_K1 = _mm512_castps512_ps128(v_C1);
        __m128 v_K2 = _mm512_castps512_ps128(v_C2);

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

        v_av1 = _mm_add_ps(v_in0, v_in3);
        v_av2 = _mm_add_ps(v_in2, v_in4);
        v_av3 = _mm_add_ps(v_in1, v_in5);
        v_av4 = _mm_add_ps(v_av2, v_av3);
        v_cv1 = _mm_sub_ps(v_av1, _mm_mul_ps(v_K1, v_av4));

        // Output point 1:X[0]
        v_out0 = _mm_add_ps(v_av1, v_av4);

        v_av1 = _mm_sub_ps(v_av2, v_av3);
        v_av2 = _mm_sub_ps(v_in0, v_in3);
        v_cv2 = _mm_sub_ps(v_av2, _mm_mul_ps(v_K1, v_av1));

        // Output point 4:X[3]
        v_out3 = _mm_add_ps(v_av2, v_av1);

        v_av1 = _mm_sub_ps(v_in1, v_in5);
        v_av2 = _mm_sub_ps(v_in2, v_in4);
        v_av3 = _mm_sub_ps(v_av1, v_av2);
        v_cv3 = _mm_mul_ps(v_K2, v_av3);
        v_cv3 = CONJ_128_S(SWAP_RI_128_S(v_cv3));

        // Output point 3:X[2]
        v_out2 = _mm_add_ps(v_cv1, v_cv3);
        // Output point 5:X[4]
        v_out4 = _mm_sub_ps(v_cv1, v_cv3);

        v_av3 = _mm_add_ps(v_av1, v_av2);
        v_cv3 = _mm_mul_ps(v_K2, v_av3);
        v_cv3 = CONJ_128_S(SWAP_RI_128_S(v_cv3));

        // Output point 2:X[1]
        v_out1 = _mm_add_ps(v_cv2, v_cv3);
        // Output point 6:X[5]
        v_out5 = _mm_sub_ps(v_cv2, v_cv3);

        if (out_strides[1] == DATA_STRIDE)
        {
            TRANSPOSE_ST2_128_S(curr_out, v_out_stride, v_out0, v_out1);
            curr_out = curr_out + NUM_SETS_128_S * DATA_STRIDE;
            TRANSPOSE_ST2_128_S(curr_out, v_out_stride, v_out2, v_out3);
            curr_out = curr_out + NUM_SETS_128_S * DATA_STRIDE;
            TRANSPOSE_ST2_128_S(curr_out, v_out_stride, v_out4, v_out5);
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
        }

        in_r = in_r + (v_in_stride << 1);
        out_r = out_r + (v_out_stride << 1);
    }
    if (remaining_sets & 1)
    {
        __m128 v_in0, v_in1, v_in2, v_in3, v_in4, v_in5;
        __m128 v_av1, v_av2, v_av3, v_av4, v_cv1, v_cv2, v_cv3;
        __m128 v_out0, v_out1, v_out2, v_out3, v_out4, v_out5;

        __m128 v_K1 = _mm512_castps512_ps128(v_C1);
        __m128 v_K2 = _mm512_castps512_ps128(v_C2);

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

        v_av1 = _mm_add_ps(v_in0, v_in3);
        v_av2 = _mm_add_ps(v_in2, v_in4);
        v_av3 = _mm_add_ps(v_in1, v_in5);
        v_av4 = _mm_add_ps(v_av2, v_av3);
        v_cv1 = _mm_sub_ps(v_av1, _mm_mul_ps(v_K1, v_av4));

        // Output point 1:X[0]
        v_out0 = _mm_add_ps(v_av1, v_av4);

        v_av1 = _mm_sub_ps(v_av2, v_av3);
        v_av2 = _mm_sub_ps(v_in0, v_in3);
        v_cv2 = _mm_sub_ps(v_av2, _mm_mul_ps(v_K1, v_av1));

        // Output point 4:X[3]
        v_out3 = _mm_add_ps(v_av2, v_av1);

        v_av1 = _mm_sub_ps(v_in1, v_in5);
        v_av2 = _mm_sub_ps(v_in2, v_in4);
        v_av3 = _mm_sub_ps(v_av1, v_av2);
        v_cv3 = _mm_mul_ps(v_K2, v_av3);
        v_cv3 = CONJ_128_S(SWAP_RI_128_S(v_cv3));

        // Output point 3:X[2]
        v_out2 = _mm_add_ps(v_cv1, v_cv3);
        // Output point 5:X[4]
        v_out4 = _mm_sub_ps(v_cv1, v_cv3);

        v_av3 = _mm_add_ps(v_av1, v_av2);
        v_cv3 = _mm_mul_ps(v_K2, v_av3);
        v_cv3 = CONJ_128_S(SWAP_RI_128_S(v_cv3));

        // Output point 2:X[1]
        v_out1 = _mm_add_ps(v_cv2, v_cv3);
        // Output point 6:X[5]
        v_out5 = _mm_sub_ps(v_cv2, v_cv3);

        if (out_strides[1] == DATA_STRIDE)
        {
            ST_128_S(curr_out, v_out0, v_out1);
            curr_out = out_r + 2 * DATA_STRIDE;
            ST_128_S(curr_out, v_out2, v_out3);
            curr_out = out_r + 4 * DATA_STRIDE;
            ST_128_S(curr_out, v_out4, v_out5);
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
        }
    }
    AOCLFFTZ_LOG(DEBUG, global_logger_mode, "Exit");
}

static FFTZ_VOID fft6avx512fp64(FFTZ_VOID *in_real, FFTZ_VOID *in_imag,
                                FFTZ_VOID *out_real, FFTZ_VOID *out_imag,
                                FFTZ_INTP n, aoclfftz_strides_t *strides,
                                FFTZ_VOID *twd, FFTZ_UINT8 flag)
{
    AOCLFFTZ_LOG(DEBUG, global_logger_mode, "Enter");
    const FFTZ_DOUBLE CRTM_6[2] = {
        0.500000000000000000000000000000000000000000000,
        0.866025403784438646763723170752936183471402627};

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

    __m512d v_C1 = _mm512_set1_pd(CRTM_6[0]);
    __m512d v_C2 = _mm512_set1_pd(CRTM_6[1]);
    v_C2 = _mm512_xor_pd(v_C2, _neg_512_d[flag].d);

    for (count = 0; count < N; count++)
    {
        __m512d v_in0, v_in1, v_in2, v_in3, v_in4, v_in5;
        __m512d v_av1, v_av2, v_av3, v_av4, v_cv1, v_cv2, v_cv3;
        __m512d v_out0, v_out1, v_out2, v_out3, v_out4, v_out5;

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

        v_av1 = _mm512_add_pd(v_in0, v_in3);
        v_av2 = _mm512_add_pd(v_in2, v_in4);
        v_av3 = _mm512_add_pd(v_in1, v_in5);
        v_av4 = _mm512_add_pd(v_av2, v_av3);
        v_cv1 = _mm512_sub_pd(v_av1, _mm512_mul_pd(v_C1, v_av4));

        // Output point 1:X[0]
        v_out0 = _mm512_add_pd(v_av1, v_av4);

        v_av1 = _mm512_sub_pd(v_av2, v_av3);
        v_av2 = _mm512_sub_pd(v_in0, v_in3);
        v_cv2 = _mm512_sub_pd(v_av2, _mm512_mul_pd(v_C1, v_av1));

        // Output point 4:X[3]
        v_out3 = _mm512_add_pd(v_av2, v_av1);

        v_av1 = _mm512_sub_pd(v_in1, v_in5);
        v_av2 = _mm512_sub_pd(v_in2, v_in4);
        v_av3 = _mm512_sub_pd(v_av1, v_av2);
        v_cv3 = _mm512_mul_pd(v_C2, v_av3);
        v_cv3 = CONJ_512_D(SWAP_RI_512_D(v_cv3));

        // Output point 3:X[2]
        v_out2 = _mm512_add_pd(v_cv1, v_cv3);
        // Output point 5:X[4]
        v_out4 = _mm512_sub_pd(v_cv1, v_cv3);

        v_av3 = _mm512_add_pd(v_av1, v_av2);
        v_cv3 = _mm512_mul_pd(v_C2, v_av3);
        v_cv3 = CONJ_512_D(SWAP_RI_512_D(v_cv3));

        // Output point 2:X[1]
        v_out1 = _mm512_add_pd(v_cv2, v_cv3);
        // Output point 6:X[5]
        v_out5 = _mm512_sub_pd(v_cv2, v_cv3);

        if (out_strides[1] == DATA_STRIDE)
        {
            TRANSPOSE_ST4_512_D(curr_out, v_out_stride, v_out0, v_out1, v_out2,
                                v_out3);
            curr_out = curr_out + NUM_SETS_512_D * DATA_STRIDE;
            TRANSPOSE_ST2_512_D(curr_out, v_out_stride, v_out4, v_out5);
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
        }

        in_r += NUM_SETS_512_D * v_in_stride;
        out_r += NUM_SETS_512_D * v_out_stride;
    }
    // tail cases
    if (remaining_sets >= NUM_SETS_256_D)
    {
        __m256d v_in0, v_in1, v_in2, v_in3, v_in4, v_in5;
        __m256d v_av1, v_av2, v_av3, v_av4, v_cv1, v_cv2, v_cv3;
        __m256d v_out0, v_out1, v_out2, v_out3, v_out4, v_out5;

        __m256d v_K1 = _mm512_castpd512_pd256(v_C1);
        __m256d v_K2 = _mm512_castpd512_pd256(v_C2);

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

        v_av1 = _mm256_add_pd(v_in0, v_in3);
        v_av2 = _mm256_add_pd(v_in2, v_in4);
        v_av3 = _mm256_add_pd(v_in1, v_in5);
        v_av4 = _mm256_add_pd(v_av2, v_av3);
        v_cv1 = _mm256_sub_pd(v_av1, _mm256_mul_pd(v_K1, v_av4));

        // Output point 1:X[0]
        v_out0 = _mm256_add_pd(v_av1, v_av4);

        v_av1 = _mm256_sub_pd(v_av2, v_av3);
        v_av2 = _mm256_sub_pd(v_in0, v_in3);
        v_cv2 = _mm256_sub_pd(v_av2, _mm256_mul_pd(v_K1, v_av1));

        // Output point 4:X[3]
        v_out3 = _mm256_add_pd(v_av2, v_av1);

        v_av1 = _mm256_sub_pd(v_in1, v_in5);
        v_av2 = _mm256_sub_pd(v_in2, v_in4);
        v_av3 = _mm256_sub_pd(v_av1, v_av2);
        v_cv3 = _mm256_mul_pd(v_K2, v_av3);
        v_cv3 = CONJ_256_D(SWAP_RI_256_D(v_cv3));

        // Output point 3:X[2]
        v_out2 = _mm256_add_pd(v_cv1, v_cv3);
        // Output point 5:X[4]
        v_out4 = _mm256_sub_pd(v_cv1, v_cv3);

        v_av3 = _mm256_add_pd(v_av1, v_av2);
        v_cv3 = _mm256_mul_pd(v_K2, v_av3);
        v_cv3 = CONJ_256_D(SWAP_RI_256_D(v_cv3));

        // Output point 2:X[1]
        v_out1 = _mm256_add_pd(v_cv2, v_cv3);
        // Output point 6:X[5]
        v_out5 = _mm256_sub_pd(v_cv2, v_cv3);

        if (out_strides[1] == DATA_STRIDE)
        {
            TRANSPOSE_ST2_256_D(curr_out, v_out_stride, v_out0, v_out1);
            curr_out = curr_out + NUM_SETS_256_D * DATA_STRIDE;
            TRANSPOSE_ST2_256_D(curr_out, v_out_stride, v_out2, v_out3);
            curr_out = curr_out + NUM_SETS_256_D * DATA_STRIDE;
            TRANSPOSE_ST2_256_D(curr_out, v_out_stride, v_out4, v_out5);
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
        }

        in_r += NUM_SETS_256_D * v_in_stride;
        out_r += NUM_SETS_256_D * v_out_stride;
    }
    // tail case
    if (n & 1)
    {
        __m128d v_in0, v_in1, v_in2, v_in3, v_in4, v_in5;
        __m128d v_av1, v_av2, v_av3, v_av4, v_cv1, v_cv2, v_cv3;
        __m128d v_out0, v_out1, v_out2, v_out3, v_out4, v_out5;

        __m128d v_K1 = _mm512_castpd512_pd128(v_C1);
        __m128d v_K2 = _mm512_castpd512_pd128(v_C2);

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

        v_av1 = _mm_add_pd(v_in0, v_in3);
        v_av2 = _mm_add_pd(v_in2, v_in4);
        v_av3 = _mm_add_pd(v_in1, v_in5);
        v_av4 = _mm_add_pd(v_av2, v_av3);
        v_cv1 = _mm_sub_pd(v_av1, _mm_mul_pd(v_K1, v_av4));

        // Output point 1:X[0]
        v_out0 = _mm_add_pd(v_av1, v_av4);

        v_av1 = _mm_sub_pd(v_av2, v_av3);
        v_av2 = _mm_sub_pd(v_in0, v_in3);
        v_cv2 = _mm_sub_pd(v_av2, _mm_mul_pd(v_K1, v_av1));

        // Output point 4:X[3]
        v_out3 = _mm_add_pd(v_av2, v_av1);

        v_av1 = _mm_sub_pd(v_in1, v_in5);
        v_av2 = _mm_sub_pd(v_in2, v_in4);
        v_av3 = _mm_sub_pd(v_av1, v_av2);
        v_cv3 = _mm_mul_pd(v_K2, v_av3);
        v_cv3 = CONJ_128_D(SWAP_RI_128_D(v_cv3));

        // Output point 3:X[2]
        v_out2 = _mm_add_pd(v_cv1, v_cv3);
        // Output point 5:X[4]
        v_out4 = _mm_sub_pd(v_cv1, v_cv3);

        v_av3 = _mm_add_pd(v_av1, v_av2);
        v_cv3 = _mm_mul_pd(v_K2, v_av3);
        v_cv3 = CONJ_128_D(SWAP_RI_128_D(v_cv3));

        // Output point 2:X[1]
        v_out1 = _mm_add_pd(v_cv2, v_cv3);
        // Output point 6:X[5]
        v_out5 = _mm_sub_pd(v_cv2, v_cv3);

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
    }
    AOCLFFTZ_LOG(DEBUG, global_logger_mode, "Exit");
}

kfft_ register_kernel_fft6avx512(FFTZ_UINT8 precision,
                                 FFTZ_UINT8 direction /* unused */)
{
    if (precision == DT_FLOAT)
    {
        return fft6avx512fp32;
    }
    else if (precision == DT_DOUBLE)
    {
        return fft6avx512fp64;
    }
    else
    {
        return NULL;
    }
}
