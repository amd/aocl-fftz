// Copyright Advanced Micro Devices, Inc.
// SPDX-License-Identifier: BSD-3-Clause

/** @file fft7avx128.c
 *
 *  @brief Radix-7 DFT kernel with AVX-128 operations using x86 SIMD intrinsics
 *
 *  This file contains the DIT Radix-7 DFT implementations using AVX128
 *  operations for single-precision and double-precision inputs.
 *
 *  @author Prasandh Sankarankutty
 *  @author Varun Sanjay
 *
 */

#include "core/kernels/kernel.h"
#include "core/kernels/simd_includes/simd_common.h"

static const ops_cycles_t ops_cnt[NUM_PRECISIONS] = {{0, 18, 33, 28, 3, 3},
                                                     {0, 18, 33, 14, 3, 3}};

ops_cycles_t get_ops_cnt_fft7avx128(UINT8 precision, UINT8 direction)
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

static VOID fft7avx128fp32(VOID *in_real, VOID *in_imag, VOID *out_real,
                           VOID *out_imag, INTP n, aoclfftz_strides_t *strides,
                           VOID *twd, UINT8 flag)
{
    AOCLFFTZ_LOG(DEBUG, global_logger_mode, "Enter");
    const FLOAT CRTM_7[6] = {+0.222520933956314404288902564496794759466355569,
                             +0.900968867902419126236102319507445051165919162,
                             +0.623489801858733530525004884004239810632274731,
                             +0.433883739117558120475768332848358754609990728,
                             +0.781831482468029808708444526674057750232334519,
                             +0.974927912181823607018131682993931217232785801};

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
    FLOAT *curr_in, *curr_out;

    __m128 v_in0, v_in1, v_in2, v_in3, v_in4, v_in5, v_in6;
    __m128 v_av1, v_av2, v_av3, v_av4, v_av5, v_av6, v_av7;
    __m128 v_tv1, v_tv2, v_tv3, v_tv4, v_tv5, v_tv6, v_tv7;
    __m128 v_tv8, v_tv9, v_tv10, v_tv11, v_tv12, v_tv13, v_tv14;
    __m128 v_tv16, v_tv17, v_tv18, v_tv19;
    __m128 v_cv1, v_cv2, v_cv3, v_cv4, v_cv5, v_cv6;
    __m128 v_out0, v_out1, v_out2, v_out3, v_out4, v_out5, v_out6;

    __m128 v_K1 = _mm_broadcast_ss(&CRTM_7[0]);
    __m128 v_K2 = _mm_broadcast_ss(&CRTM_7[1]);
    __m128 v_K3 = _mm_broadcast_ss(&CRTM_7[2]);
    __m128 v_K4 = _mm_broadcast_ss(&CRTM_7[3]);
    v_K4 = _mm_xor_ps(v_K4, _neg_128_f[flag].s);
    __m128 v_K5 = _mm_broadcast_ss(&CRTM_7[4]);
    v_K5 = _mm_xor_ps(v_K5, _neg_128_f[flag].s);
    __m128 v_K6 = _mm_broadcast_ss(&CRTM_7[5]);
    v_K6 = _mm_xor_ps(v_K6, _neg_128_f[flag].s);

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
    AOCLFFTZ_LOG(DEBUG, global_logger_mode, "Exit");
}

static VOID fft7avx128fp64(VOID *in_real, VOID *in_imag, VOID *out_real,
                           VOID *out_imag, INTP n, aoclfftz_strides_t *strides,
                           VOID *twd, UINT8 flag)
{
    AOCLFFTZ_LOG(DEBUG, global_logger_mode, "Enter");
    const DOUBLE CRTM_7[6] = {+0.222520933956314404288902564496794759466355569,
                              +0.900968867902419126236102319507445051165919162,
                              +0.623489801858733530525004884004239810632274731,
                              +0.433883739117558120475768332848358754609990728,
                              +0.781831482468029808708444526674057750232334519,
                              +0.974927912181823607018131682993931217232785801};

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
    INTP count;
    DOUBLE *curr_in, *curr_out;

    __m128d v_in0, v_in1, v_in2, v_in3, v_in4, v_in5, v_in6;
    __m128d v_av1, v_av2, v_av3, v_av4, v_av5, v_av6, v_av7;
    __m128d v_tv1, v_tv2, v_tv3, v_tv4, v_tv5, v_tv6, v_tv7;
    __m128d v_tv8, v_tv9, v_tv10, v_tv11, v_tv12, v_tv13, v_tv14;
    __m128d v_tv16, v_tv17, v_tv18, v_tv19;
    __m128d v_cv1, v_cv2, v_cv3, v_cv4, v_cv5, v_cv6;
    __m128d v_out0, v_out1, v_out2, v_out3, v_out4, v_out5, v_out6;

    __m128d v_K1 = _mm_set1_pd(CRTM_7[0]);
    __m128d v_K2 = _mm_set1_pd(CRTM_7[1]);
    __m128d v_K3 = _mm_set1_pd(CRTM_7[2]);
    __m128d v_K4 = _mm_set1_pd(CRTM_7[3]);
    v_K4 = _mm_xor_pd(v_K4, _neg_128_d[flag].d);
    __m128d v_K5 = _mm_set1_pd(CRTM_7[4]);
    v_K5 = _mm_xor_pd(v_K5, _neg_128_d[flag].d);
    __m128d v_K6 = _mm_set1_pd(CRTM_7[5]);
    v_K6 = _mm_xor_pd(v_K6, _neg_128_d[flag].d);

    for (count = 0; count < n; count++)
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

        in_r += v_in_stride;
        out_r += v_out_stride;
    }
    AOCLFFTZ_LOG(DEBUG, global_logger_mode, "Exit");
}

kfft_ register_kernel_fft7avx128(UINT8 precision, UINT8 direction /* unused */)
{
    if (precision == DT_FLOAT)
    {
        return fft7avx128fp32;
    }
    else if (precision == DT_DOUBLE)
    {
        return fft7avx128fp64;
    }
    else
    {
        return NULL;
    }
}
