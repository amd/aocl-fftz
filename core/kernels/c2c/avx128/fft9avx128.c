// Copyright Advanced Micro Devices, Inc.
// SPDX-License-Identifier: BSD-3-Clause

/** @file fft9avx128.c
 *
 *  @brief Radix-9 FFT kernel with AVX-128 operations using x86 SIMD intrinsics
 *
 *  This file contains the DIT radix-9 FFT implementations using AVX128
 *  operations for single-precision and double-precision inputs.
 *
 *  @author Jeya R
 *
 */

#include "core/kernels/kernel.h"
#include "core/kernels/simd_includes/simd_common.h"

static const ops_cycles_t ops_cnt[NUM_PRECISIONS] = {{0, 20, 40, 36, 14, 6},
                                                     {0, 20, 40, 18, 14, 6}};

ops_cycles_t get_ops_cnt_fft9avx128(UINT8 precision, UINT8 direction)
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

static VOID fft9avx128fp32(VOID *in_real, VOID *in_imag, VOID *out_real,
                           VOID *out_imag, INTP n, aoclfftz_strides_t *strides,
                           VOID *twd, UINT8 flag)
{
    AOCLFFTZ_LOG(DEBUG, global_logger_mode, "Enter");
    const FLOAT CRTM_9[8] = {-0.939692620785908384054109277324731469936208134,
                             0.342020143325668733044099614682259580763083368,
                             0.984807753012208059366743024589523013670643252,
                             0.173648177666930348851716626769314796000375677,
                             0.642787609686539326322643409907263432907559884,
                             0.766044443118978035202392650555416673935832457,
                             0.500000000000000000000000000000000000000000000,
                             0.866025403784438646763723170752936183471402627};

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
    INTP v_out_stride = strides->v_out_stride;
    INTP N = n / NUM_SETS_128_S;
    INTP count;
    FLOAT *curr_in, *curr_out;

    __m128 v_in0, v_in1, v_in2, v_in3, v_in4, v_in5, v_in6, v_in7, v_in8;
    __m128 v_out0, v_out1, v_out2, v_out3, v_out4, v_out5, v_out6, v_out7,
           v_out8;
    __m128 v_tv1, v_tv2, v_tv3, v_tv4, v_av1;
    __m128 v_cv1, v_cv2, v_cv3, v_cv4, v_cv5, v_cv6, v_cv7, v_cv8, v_cv9;

    __m128 v_K1 = _mm_broadcast_ss(&CRTM_9[0]);
    __m128 v_K2 = _mm_broadcast_ss(&CRTM_9[1]);
    v_K2 = _mm_xor_ps(v_K2, _neg_128_f[flag].s);
    __m128 v_K3 = _mm_broadcast_ss(&CRTM_9[2]);
    v_K3 = _mm_xor_ps(v_K3, _neg_128_f[flag].s);
    __m128 v_K4 = _mm_broadcast_ss(&CRTM_9[3]);
    __m128 v_K5 = _mm_broadcast_ss(&CRTM_9[4]);
    v_K5 = _mm_xor_ps(v_K5, _neg_128_f[flag].s);
    __m128 v_K6 = _mm_broadcast_ss(&CRTM_9[5]);
    __m128 v_K7 = _mm_broadcast_ss(&CRTM_9[6]);
    __m128 v_K8 = _mm_broadcast_ss(&CRTM_9[7]);
    v_K8 = _mm_xor_ps(v_K8, _neg_128_f[flag].s);

    for (count = 0; count < N; count++)
    {
        curr_in = in_r;
        curr_out = out_r;

        GATHER2_128_S(curr_in, v_in_stride, v_in0);
        curr_in = in_r + in_strides[1];
        GATHER2_128_S(curr_in, v_in_stride, v_in1);
        curr_in = in_r + in_strides[2];
        GATHER2_128_S(curr_in, v_in_stride, v_in2);
        curr_in = in_r + in_strides[3];
        GATHER2_128_S(curr_in, v_in_stride, v_in3);
        curr_in = in_r + in_strides[4];
        GATHER2_128_S(curr_in, v_in_stride, v_in4);
        curr_in = in_r + in_strides[5];
        GATHER2_128_S(curr_in, v_in_stride, v_in5);
        curr_in = in_r + in_strides[6];
        GATHER2_128_S(curr_in, v_in_stride, v_in6);
        curr_in = in_r + in_strides[7];
        GATHER2_128_S(curr_in, v_in_stride, v_in7);
        curr_in = in_r + in_strides[8];
        GATHER2_128_S(curr_in, v_in_stride, v_in8);

        // common operations
        v_av1 = _mm_add_ps(v_in3, v_in6);
        v_cv1 = _mm_add_ps(v_in0, v_av1);
        v_tv1 = _mm_sub_ps(v_in0, _mm_mul_ps(v_K7, v_av1));
        v_tv2 = _mm_mul_ps(v_K8, _mm_sub_ps(v_in3, v_in6));
        v_tv2 = CONJ_128_S(SWAP_RI_128_S(v_tv2));
        v_cv2 = _mm_add_ps(v_tv1, v_tv2);
        v_cv3 = _mm_sub_ps(v_tv1, v_tv2);

        v_av1 = _mm_add_ps(v_in4, v_in7);
        v_cv4 = _mm_add_ps(v_in1, v_av1);
        v_tv1 = _mm_sub_ps(v_in1, _mm_mul_ps(v_K7, v_av1));
        v_tv2 = _mm_mul_ps(v_K8, _mm_sub_ps(v_in4, v_in7));
        v_tv2 = CONJ_128_S(SWAP_RI_128_S(v_tv2));
        v_cv5 = _mm_add_ps(v_tv1, v_tv2);
        v_cv6 = _mm_sub_ps(v_tv1, v_tv2);

        v_av1 = _mm_add_ps(v_in5, v_in8);
        v_cv7 = _mm_add_ps(v_in2, v_av1);
        v_tv1 = _mm_sub_ps(v_in2, _mm_mul_ps(v_K7, v_av1));
        v_tv2 = _mm_mul_ps(v_K8, _mm_sub_ps(v_in5, v_in8));
        v_tv2 = CONJ_128_S(SWAP_RI_128_S(v_tv2));
        v_cv8 = _mm_add_ps(v_tv1, v_tv2);
        v_cv9 = _mm_sub_ps(v_tv1, v_tv2);

        // Output point 1: X[0]
        v_av1 = _mm_add_ps(v_cv4, v_cv7);
        v_out0 = _mm_add_ps(v_cv1, v_av1);

        v_tv1 = _mm_sub_ps(v_cv1, _mm_mul_ps(v_K7, v_av1));
        v_tv2 = _mm_mul_ps(v_K8, _mm_sub_ps(v_cv4, v_cv7));
        v_tv2 = CONJ_128_S(SWAP_RI_128_S(v_tv2));

        // Output point 4: X[3]
        v_out3 = _mm_add_ps(v_tv1, v_tv2);
        // Output point 7: X[6]
        v_out6 = _mm_sub_ps(v_tv1, v_tv2);

        v_tv3 = _mm_mul_ps(v_cv5, v_K6);
        v_tv4 = _mm_mul_ps(v_cv5, v_K5);
        SUBADD_SWAPA_128_S(v_tv3, v_tv4, v_cv5);
        v_tv3 = _mm_mul_ps(v_cv8, v_K4);
        v_tv4 = _mm_mul_ps(v_cv8, v_K3);
        SUBADD_SWAPA_128_S(v_tv3, v_tv4, v_cv8);
        v_av1 = _mm_add_ps(v_cv5, v_cv8);

        // Output point 2: X[1]
        v_out1 = _mm_add_ps(v_cv2, v_av1);

        v_tv1 = _mm_sub_ps(v_cv2, _mm_mul_ps(v_K7, v_av1));
        v_tv2 = _mm_mul_ps(v_K8, _mm_sub_ps(v_cv5, v_cv8));
        v_tv2 = CONJ_128_S(SWAP_RI_128_S(v_tv2));

        // Output point 5: X[4]
        v_out4 = _mm_add_ps(v_tv1, v_tv2);
        // Output point 8: X[7]
        v_out7 = _mm_sub_ps(v_tv1, v_tv2);

        v_tv3 = _mm_mul_ps(v_cv6, v_K4);
        v_tv4 = _mm_mul_ps(v_cv6, v_K3);
        SUBADD_SWAPA_128_S(v_tv3, v_tv4, v_cv6);
        v_tv3 = _mm_mul_ps(v_cv9, v_K1);
        v_tv4 = _mm_mul_ps(v_cv9, v_K2);
        SUBADD_SWAPA_128_S(v_tv3, v_tv4, v_cv9);
        v_av1 = _mm_add_ps(v_cv6, v_cv9);

        // Output point 3: X[2]
        v_out2 = _mm_add_ps(v_cv3, v_av1);

        v_tv1 = _mm_sub_ps(v_cv3, _mm_mul_ps(v_K7, v_av1));
        v_tv2 = _mm_mul_ps(v_K8, _mm_sub_ps(v_cv6, v_cv9));
        v_tv2 = CONJ_128_S(SWAP_RI_128_S(v_tv2));

        // Output point 6: X[5]
        v_out5 = _mm_add_ps(v_tv1, v_tv2);
        // Output point 9: X[8]
        v_out8 = _mm_sub_ps(v_tv1, v_tv2);

        SCATTER2_128_S(curr_out, v_out_stride, v_out0);
        curr_out = out_r + out_strides[1];
        SCATTER2_128_S(curr_out, v_out_stride, v_out1);
        curr_out = out_r + out_strides[2];
        SCATTER2_128_S(curr_out, v_out_stride, v_out2);
        curr_out = out_r + out_strides[3];
        SCATTER2_128_S(curr_out, v_out_stride, v_out3);
        curr_out = out_r + out_strides[4];
        SCATTER2_128_S(curr_out, v_out_stride, v_out4);
        curr_out = out_r + out_strides[5];
        SCATTER2_128_S(curr_out, v_out_stride, v_out5);
        curr_out = out_r + out_strides[6];
        SCATTER2_128_S(curr_out, v_out_stride, v_out6);
        curr_out = out_r + out_strides[7];
        SCATTER2_128_S(curr_out, v_out_stride, v_out7);
        curr_out = out_r + out_strides[8];
        SCATTER2_128_S(curr_out, v_out_stride, v_out8);

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

        // common operations
        v_av1 = _mm_add_ps(v_in3, v_in6);
        v_cv1 = _mm_add_ps(v_in0, v_av1);
        v_tv1 = _mm_sub_ps(v_in0, _mm_mul_ps(v_K7, v_av1));
        v_tv2 = _mm_mul_ps(v_K8, _mm_sub_ps(v_in3, v_in6));
        v_tv2 = CONJ_128_S(SWAP_RI_128_S(v_tv2));
        v_cv2 = _mm_add_ps(v_tv1, v_tv2);
        v_cv3 = _mm_sub_ps(v_tv1, v_tv2);

        v_av1 = _mm_add_ps(v_in4, v_in7);
        v_cv4 = _mm_add_ps(v_in1, v_av1);
        v_tv1 = _mm_sub_ps(v_in1, _mm_mul_ps(v_K7, v_av1));
        v_tv2 = _mm_mul_ps(v_K8, _mm_sub_ps(v_in4, v_in7));
        v_tv2 = CONJ_128_S(SWAP_RI_128_S(v_tv2));
        v_cv5 = _mm_add_ps(v_tv1, v_tv2);
        v_cv6 = _mm_sub_ps(v_tv1, v_tv2);

        v_av1 = _mm_add_ps(v_in5, v_in8);
        v_cv7 = _mm_add_ps(v_in2, v_av1);
        v_tv1 = _mm_sub_ps(v_in2, _mm_mul_ps(v_K7, v_av1));
        v_tv2 = _mm_mul_ps(v_K8, _mm_sub_ps(v_in5, v_in8));
        v_tv2 = CONJ_128_S(SWAP_RI_128_S(v_tv2));
        v_cv8 = _mm_add_ps(v_tv1, v_tv2);
        v_cv9 = _mm_sub_ps(v_tv1, v_tv2);

        // Output point 1: X[0]
        v_av1 = _mm_add_ps(v_cv4, v_cv7);
        v_out0 = _mm_add_ps(v_cv1, v_av1);

        v_tv1 = _mm_sub_ps(v_cv1, _mm_mul_ps(v_K7, v_av1));
        v_tv2 = _mm_mul_ps(v_K8, _mm_sub_ps(v_cv4, v_cv7));
        v_tv2 = CONJ_128_S(SWAP_RI_128_S(v_tv2));

        // Output point 4: X[3]
        v_out3 = _mm_add_ps(v_tv1, v_tv2);
        // Output point 7: X[6]
        v_out6 = _mm_sub_ps(v_tv1, v_tv2);

        v_tv3 = _mm_mul_ps(v_cv5, v_K6);
        v_tv4 = _mm_mul_ps(v_cv5, v_K5);
        SUBADD_SWAPA_128_S(v_tv3, v_tv4, v_cv5);
        v_tv3 = _mm_mul_ps(v_cv8, v_K4);
        v_tv4 = _mm_mul_ps(v_cv8, v_K3);
        SUBADD_SWAPA_128_S(v_tv3, v_tv4, v_cv8);
        v_av1 = _mm_add_ps(v_cv5, v_cv8);

        // Output point 2: X[1]
        v_out1 = _mm_add_ps(v_cv2, v_av1);

        v_tv1 = _mm_sub_ps(v_cv2, _mm_mul_ps(v_K7, v_av1));
        v_tv2 = _mm_mul_ps(v_K8, _mm_sub_ps(v_cv5, v_cv8));
        v_tv2 = CONJ_128_S(SWAP_RI_128_S(v_tv2));

        // Output point 5: X[4]
        v_out4 = _mm_add_ps(v_tv1, v_tv2);
        // Output point 8: X[7]
        v_out7 = _mm_sub_ps(v_tv1, v_tv2);

        v_tv3 = _mm_mul_ps(v_cv6, v_K4);
        v_tv4 = _mm_mul_ps(v_cv6, v_K3);
        SUBADD_SWAPA_128_S(v_tv3, v_tv4, v_cv6);
        v_tv3 = _mm_mul_ps(v_cv9, v_K1);
        v_tv4 = _mm_mul_ps(v_cv9, v_K2);
        SUBADD_SWAPA_128_S(v_tv3, v_tv4, v_cv9);
        v_av1 = _mm_add_ps(v_cv6, v_cv9);

        // Output point 3: X[2]
        v_out2 = _mm_add_ps(v_cv3, v_av1);

        v_tv1 = _mm_sub_ps(v_cv3, _mm_mul_ps(v_K7, v_av1));
        v_tv2 = _mm_mul_ps(v_K8, _mm_sub_ps(v_cv6, v_cv9));
        v_tv2 = CONJ_128_S(SWAP_RI_128_S(v_tv2));

        // Output point 6: X[5]
        v_out5 = _mm_add_ps(v_tv1, v_tv2);
        // Output point 9: X[8]
        v_out8 = _mm_sub_ps(v_tv1, v_tv2);

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
    }
    AOCLFFTZ_LOG(DEBUG, global_logger_mode, "Exit");
}

static VOID fft9avx128fp64(VOID *in_real, VOID *in_imag, VOID *out_real,
                           VOID *out_imag, INTP n, aoclfftz_strides_t *strides,
                           VOID *twd, UINT8 flag)
{
    AOCLFFTZ_LOG(DEBUG, global_logger_mode, "Enter");
    const DOUBLE CRTM_9[8] = {-0.939692620785908384054109277324731469936208134,
                              0.342020143325668733044099614682259580763083368,
                              0.984807753012208059366743024589523013670643252,
                              0.173648177666930348851716626769314796000375677,
                              0.642787609686539326322643409907263432907559884,
                              0.766044443118978035202392650555416673935832457,
                              0.500000000000000000000000000000000000000000000,
                              0.866025403784438646763723170752936183471402627};

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

    __m128d v_in0, v_in1, v_in2, v_in3, v_in4, v_in5, v_in6, v_in7, v_in8;
    __m128d v_out0, v_out1, v_out2, v_out3, v_out4, v_out5, v_out6, v_out7,
            v_out8;
    __m128d v_tv1, v_tv2, v_tv3, v_tv4, v_av1;
    __m128d v_cv1, v_cv2, v_cv3, v_cv4, v_cv5, v_cv6, v_cv7, v_cv8, v_cv9;

    __m128d v_K1 = _mm_set1_pd(CRTM_9[0]);
    __m128d v_K2 = _mm_set1_pd(CRTM_9[1]);
    v_K2 = _mm_xor_pd(v_K2, _neg_128_d[flag].d);
    __m128d v_K3 = _mm_set1_pd(CRTM_9[2]);
    v_K3 = _mm_xor_pd(v_K3, _neg_128_d[flag].d);
    __m128d v_K4 = _mm_set1_pd(CRTM_9[3]);
    __m128d v_K5 = _mm_set1_pd(CRTM_9[4]);
    v_K5 = _mm_xor_pd(v_K5, _neg_128_d[flag].d);
    __m128d v_K6 = _mm_set1_pd(CRTM_9[5]);
    __m128d v_K7 = _mm_set1_pd(CRTM_9[6]);
    __m128d v_K8 = _mm_set1_pd(CRTM_9[7]);
    v_K8 = _mm_xor_pd(v_K8, _neg_128_d[flag].d);

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
        curr_in = in_r + in_strides[7];
        LD_128_D(curr_in, v_in7);
        curr_in = in_r + in_strides[8];
        LD_128_D(curr_in, v_in8);

        // common operations
        v_av1 = _mm_add_pd(v_in3, v_in6);
        v_cv1 = _mm_add_pd(v_in0, v_av1);
        v_tv1 = _mm_sub_pd(v_in0, _mm_mul_pd(v_K7, v_av1));
        v_tv2 = _mm_mul_pd(v_K8, _mm_sub_pd(v_in3, v_in6));
        v_tv2 = CONJ_128_D(SWAP_RI_128_D(v_tv2));
        v_cv2 = _mm_add_pd(v_tv1, v_tv2);
        v_cv3 = _mm_sub_pd(v_tv1, v_tv2);

        v_av1 = _mm_add_pd(v_in4, v_in7);
        v_cv4 = _mm_add_pd(v_in1, v_av1);
        v_tv1 = _mm_sub_pd(v_in1, _mm_mul_pd(v_K7, v_av1));
        v_tv2 = _mm_mul_pd(v_K8, _mm_sub_pd(v_in4, v_in7));
        v_tv2 = CONJ_128_D(SWAP_RI_128_D(v_tv2));
        v_cv5 = _mm_add_pd(v_tv1, v_tv2);
        v_cv6 = _mm_sub_pd(v_tv1, v_tv2);

        v_av1 = _mm_add_pd(v_in5, v_in8);
        v_cv7 = _mm_add_pd(v_in2, v_av1);
        v_tv1 = _mm_sub_pd(v_in2, _mm_mul_pd(v_K7, v_av1));
        v_tv2 = _mm_mul_pd(v_K8, _mm_sub_pd(v_in5, v_in8));
        v_tv2 = CONJ_128_D(SWAP_RI_128_D(v_tv2));
        v_cv8 = _mm_add_pd(v_tv1, v_tv2);
        v_cv9 = _mm_sub_pd(v_tv1, v_tv2);

        // Output point 1: X[0]
        v_av1 = _mm_add_pd(v_cv4, v_cv7);
        v_out0 = _mm_add_pd(v_cv1, v_av1);

        v_tv1 = _mm_sub_pd(v_cv1, _mm_mul_pd(v_K7, v_av1));
        v_tv2 = _mm_mul_pd(v_K8, _mm_sub_pd(v_cv4, v_cv7));
        v_tv2 = CONJ_128_D(SWAP_RI_128_D(v_tv2));

        // Output point 4: X[3]
        v_out3 = _mm_add_pd(v_tv1, v_tv2);
        // Output point 7: X[6]
        v_out6 = _mm_sub_pd(v_tv1, v_tv2);

        v_tv3 = _mm_mul_pd(v_cv5, v_K6);
        v_tv4 = _mm_mul_pd(v_cv5, v_K5);
        SUBADD_SWAPA_128_D(v_tv3, v_tv4, v_cv5);
        v_tv3 = _mm_mul_pd(v_cv8, v_K4);
        v_tv4 = _mm_mul_pd(v_cv8, v_K3);
        SUBADD_SWAPA_128_D(v_tv3, v_tv4, v_cv8);
        v_av1 = _mm_add_pd(v_cv5, v_cv8);

        // Output point 2: X[1]
        v_out1 = _mm_add_pd(v_cv2, v_av1);

        v_tv1 = _mm_sub_pd(v_cv2, _mm_mul_pd(v_K7, v_av1));
        v_tv2 = _mm_mul_pd(v_K8, _mm_sub_pd(v_cv5, v_cv8));
        v_tv2 = CONJ_128_D(SWAP_RI_128_D(v_tv2));

        // Output point 5: X[4]
        v_out4 = _mm_add_pd(v_tv1, v_tv2);
        // Output point 8: X[7]
        v_out7 = _mm_sub_pd(v_tv1, v_tv2);

        v_tv3 = _mm_mul_pd(v_cv6, v_K4);
        v_tv4 = _mm_mul_pd(v_cv6, v_K3);
        SUBADD_SWAPA_128_D(v_tv3, v_tv4, v_cv6);
        v_tv3 = _mm_mul_pd(v_cv9, v_K1);
        v_tv4 = _mm_mul_pd(v_cv9, v_K2);
        SUBADD_SWAPA_128_D(v_tv3, v_tv4, v_cv9);
        v_av1 = _mm_add_pd(v_cv6, v_cv9);

        // Output point 3: X[2]s
        v_out2 = _mm_add_pd(v_cv3, v_av1);

        v_tv1 = _mm_sub_pd(v_cv3, _mm_mul_pd(v_K7, v_av1));
        v_tv2 = _mm_mul_pd(v_K8, _mm_sub_pd(v_cv6, v_cv9));
        v_tv2 = CONJ_128_D(SWAP_RI_128_D(v_tv2));

        // Output point 6: X[5]
        v_out5 = _mm_add_pd(v_tv1, v_tv2);
        // Output point 9: X[8]
        v_out8 = _mm_sub_pd(v_tv1, v_tv2);

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

        in_r += v_in_stride;
        out_r += v_out_stride;
    }
    AOCLFFTZ_LOG(DEBUG, global_logger_mode, "Exit");
}

kfft_ register_kernel_fft9avx128(UINT8 precision, UINT8 direction /* unused */)
{
    if (precision == DT_FLOAT)
    {
        return fft9avx128fp32;
    }
    else if (precision == DT_DOUBLE)
    {
        return fft9avx128fp64;
    }
    else
    {
        return NULL;
    }
}
