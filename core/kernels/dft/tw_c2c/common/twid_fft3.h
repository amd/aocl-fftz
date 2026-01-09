// Copyright Advanced Micro Devices, Inc.
// SPDX-License-Identifier: BSD-3-Clause

/** @file twid_fft3.h
 *
 *  @brief The ISA generic kernel template for the radix 3 twiddle kernel
 *
 *  This file contains the DIT twiddle radix-3 FFT implementations for
 *  single-precision and double-precision inputs.
 *
 *  @author Ashwin K. Godbole
 */

// This header has no include guards.
// This is intentional.
// The functions defined in this file are not usable by default.
// They are "instantiated" only when "included" in another file.

#include "core/kernels/simd_includes/generic_kernels_common.h"

static VOID TWID_KNAME_FP32(VOID *in_real, VOID *in_imag, VOID *out_real,
                    VOID *out_imag, INTP n, aoclfftz_strides_t *strides,
                    VOID *twd, UINT8 flag)
{
    AOCLFFTZ_LOG(DEBUG, global_logger_mode, "Enter");
    const FLOAT CRTM_3[2] = {0.500000000000000000000000000000000000000000000f,
                             0.866025403784438646763723170752936183471402627f};

    FLOAT *in_r = in_real;
    FLOAT *out_r = out_real;
    FLOAT *curr_in, *curr_out;
#ifdef VOLATILE_STRIDE_ARRAY
    volatile INTP *in_strides = strides->in_strides;
    volatile INTP *out_strides = strides->out_strides;
#else
    INTP *in_strides = strides->in_strides;
    INTP *out_strides = strides->out_strides;
#endif
    INTP v_in_stride = strides->v_in_stride;
    INTP v_out_stride = strides->v_out_stride;

    aoclfftz_twiddle_t *tws = (aoclfftz_twiddle_t *)twd;
    FLOAT *tw = (FLOAT *)(tws->TW);
    UINTP cols = tws->cols;
    UINTP load_multi_cols = tws->load_multi_cols;

    INTP N = n / NUM_SETS_S;
    INTP remaining_sets = n % NUM_SETS_S;

#if defined(KERNEL_USE_AVX512)
    INTP do_256_whole = (INTP)(remaining_sets >= NUM_SETS_256_S);
    INTP do_128_whole =
        (INTP)(remaining_sets % NUM_SETS_256_S >= NUM_SETS_128_S);
    INTP cnt_256 = load_multi_cols * (N * NUM_SETS_512_S);
    INTP cnt_128 =
        load_multi_cols * (N * NUM_SETS_512_S + do_256_whole * NUM_SETS_256_S);
    INTP cnt_128_low =
        load_multi_cols * (N * NUM_SETS_512_S + do_256_whole * NUM_SETS_256_S +
                           do_128_whole * NUM_SETS_128_S);
#elif defined(KERNEL_USE_AVX256)
    INTP do_128_whole = (INTP)(remaining_sets >= NUM_SETS_128_S);
    INTP cnt_128 = load_multi_cols * (N * NUM_SETS_256_S);
    INTP cnt_128_low =
        load_multi_cols * (N * NUM_SETS_256_S + do_128_whole * NUM_SETS_128_S);
#elif defined(KERNEL_USE_AVX128)
    INTP cnt_128_low = load_multi_cols * (N * NUM_SETS_128_S);
#endif

    VREGTYPE_S v_C1 = BCAST_S(CRTM_3[0]);
    VREGTYPE_S v_C2 = BCAST_S(CRTM_3[1]);

    INTP count;

    v_C2 = NEG_S(v_C2, flag);

    for (count = 0; count < N; count++)
    {
        VREGTYPE_S v_in0, v_in1, v_in2;
        VREGTYPE_S v_av0, v_av1, v_tv0, v_tv1;
        VREGTYPE_S v_out0, v_out1, v_out2;

        curr_in = in_r;
        curr_out = out_r;

        if (flag)
        {
            ITW_GATHER_S(curr_in, in_strides, 1, v_in_stride, v_in1, tw, cols,
                         (count * load_multi_cols * NUM_SETS_S),
                         load_multi_cols);
            ITW_GATHER_S(curr_in, in_strides, 2, v_in_stride, v_in2, tw, cols,
                         (count * load_multi_cols * NUM_SETS_S),
                         load_multi_cols);
        }
        else
        {
            TW_GATHER_S(curr_in, in_strides, 1, v_in_stride, v_in1, tw, cols,
                        (count * load_multi_cols * NUM_SETS_S),
                        load_multi_cols);
            TW_GATHER_S(curr_in, in_strides, 2, v_in_stride, v_in2, tw, cols,
                        (count * load_multi_cols * NUM_SETS_S),
                        load_multi_cols);
        }

        GATHER_S(curr_in, v_in_stride, v_in0);

        v_av0 = ADD_S(v_in1, v_in2);
        v_av1 = SUB_S(v_in1, v_in2);

        v_tv0 = SUB_S(v_in0, MUL_S(v_C1, v_av0));

        // Output point 1: X[0]
        v_out0 = ADD_S(v_in0, v_av0);

        v_tv1 = MUL_S(v_C2, v_av1);
        v_tv1 = SWAP_RI_S(CONJ_S(v_tv1));

        // Output point 2: X[1]
        v_out1 = SUB_S(v_tv0, v_tv1);
        // Output point 3: X[2]
        v_out2 = ADD_S(v_tv0, v_tv1);

        SCATTER_S(curr_out, v_out_stride, v_out0);
        SCATTER_S(curr_out + out_strides[1], v_out_stride, v_out1);
        SCATTER_S(curr_out + out_strides[2], v_out_stride, v_out2);

        in_r += NUM_SETS_S * v_in_stride;
        out_r += NUM_SETS_S * v_out_stride;
    }

    // The following contains code that performs the FFT on the tail cases.
    // These tails are conditionally "instantiated" based on the ISA requested
    // by the "includee" file.

#if defined(KERNEL_USE_AVX512)
    if (remaining_sets >= NUM_SETS_256_S)
    {
        __m256 v_in0, v_in1, v_in2;
        __m256 v_av0, v_av1, v_tv0, v_tv1;
        __m256 v_out0, v_out1, v_out2;

        __m256 K1 = CAST_512_TO_256_S(v_C1);
        __m256 K2 = CAST_512_TO_256_S(v_C2);

        curr_in = in_r;
        curr_out = out_r;

        if (flag)
        {
            ITW_GATHER_256_S(curr_in, in_strides, 1, v_in_stride, v_in1, tw,
                             cols, cnt_256, load_multi_cols);
            ITW_GATHER_256_S(curr_in, in_strides, 2, v_in_stride, v_in2, tw,
                             cols, cnt_256, load_multi_cols);
        }
        else
        {
            TW_GATHER_256_S(curr_in, in_strides, 1, v_in_stride, v_in1, tw,
                            cols, cnt_256, load_multi_cols);
            TW_GATHER_256_S(curr_in, in_strides, 2, v_in_stride, v_in2, tw,
                            cols, cnt_256, load_multi_cols);
        }

        GATHER4_256_S(curr_in, v_in_stride, v_in0);

        v_av0 = _mm256_add_ps(v_in1, v_in2);
        v_av1 = _mm256_sub_ps(v_in1, v_in2);

        v_tv0 = _mm256_sub_ps(v_in0, _mm256_mul_ps(K1, v_av0));

        // Output point 1: X[0]
        v_out0 = _mm256_add_ps(v_in0, v_av0);

        v_tv1 = _mm256_mul_ps(K2, v_av1);
        v_tv1 = SWAP_RI_256_S(CONJ_256_S(v_tv1));

        // Output point 2: X[1]
        v_out1 = _mm256_sub_ps(v_tv0, v_tv1);
        // Output point 3: X[2]
        v_out2 = _mm256_add_ps(v_tv0, v_tv1);

        SCATTER4_256_S(curr_out, v_out_stride, v_out0);
        SCATTER4_256_S(curr_out + out_strides[1], v_out_stride, v_out1);
        SCATTER4_256_S(curr_out + out_strides[2], v_out_stride, v_out2);

        in_r += NUM_SETS_256_S * v_in_stride;
        out_r += NUM_SETS_256_S * v_out_stride;
        remaining_sets = remaining_sets - NUM_SETS_256_S;
    }
#endif

#if defined(KERNEL_USE_AVX512) || defined(KERNEL_USE_AVX256)
    if (remaining_sets >= NUM_SETS_128_S)
    {
        __m128 v_in0, v_in1, v_in2;
        __m128 v_av0, v_av1, v_tv0, v_tv1;
        __m128 v_out0, v_out1, v_out2;

#if defined(KERNEL_USE_AVX512)
        __m128 K1 = CAST_512_TO_128_S(v_C1);
        __m128 K2 = CAST_512_TO_128_S(v_C2);
#elif defined(KERNEL_USE_AVX256)
        __m128 K1 = CAST_256_TO_128_S(v_C1);
        __m128 K2 = CAST_256_TO_128_S(v_C2);
#endif

        curr_in = in_r;
        curr_out = out_r;

        if (flag)
        {
            ITW_GATHER_128_S(curr_in, in_strides, 1, v_in_stride, v_in1, tw,
                             cols, cnt_128, load_multi_cols);
            ITW_GATHER_128_S(curr_in, in_strides, 2, v_in_stride, v_in2, tw,
                             cols, cnt_128, load_multi_cols);
        }
        else
        {
            TW_GATHER_128_S(curr_in, in_strides, 1, v_in_stride, v_in1, tw,
                            cols, cnt_128, load_multi_cols);
            TW_GATHER_128_S(curr_in, in_strides, 2, v_in_stride, v_in2, tw,
                            cols, cnt_128, load_multi_cols);
        }

        GATHER2_128_S(curr_in, v_in_stride, v_in0);

        v_av0 = _mm_add_ps(v_in1, v_in2);
        v_av1 = _mm_sub_ps(v_in1, v_in2);

        v_tv0 = _mm_sub_ps(v_in0, _mm_mul_ps(K1, v_av0));

        // Output point 1: X[0]
        v_out0 = _mm_add_ps(v_in0, v_av0);

        v_tv1 = _mm_mul_ps(K2, v_av1);
        v_tv1 = SWAP_RI_128_S(CONJ_128_S(v_tv1));

        // Output point 2: X[1]
        v_out1 = _mm_sub_ps(v_tv0, v_tv1);
        // Output point 3: X[2]
        v_out2 = _mm_add_ps(v_tv0, v_tv1);

        SCATTER2_128_S(curr_out, v_out_stride, v_out0);
        SCATTER2_128_S(curr_out + out_strides[1], v_out_stride, v_out1);
        SCATTER2_128_S(curr_out + out_strides[2], v_out_stride, v_out2);

        in_r = in_r + (v_in_stride << 1);
        out_r = out_r + (v_out_stride << 1);
        remaining_sets = remaining_sets - NUM_SETS_128_S;
    }
#endif

#if defined(KERNEL_USE_AVX512) || defined(KERNEL_USE_AVX256) || defined(KERNEL_USE_AVX128)
    if (remaining_sets & 1)
    {
        __m128 v_in0, v_in1, v_in2;
        __m128 v_av0, v_av1, v_tv0, v_tv1;
        __m128 v_out0, v_out1, v_out2;

#if defined(KERNEL_USE_AVX512)
        __m128 K1 = CAST_512_TO_128_S(v_C1);
        __m128 K2 = CAST_512_TO_128_S(v_C2);
#elif defined(KERNEL_USE_AVX256)
        __m128 K1 = CAST_256_TO_128_S(v_C1);
        __m128 K2 = CAST_256_TO_128_S(v_C2);
#elif defined(KERNEL_USE_AVX128)
        __m128 K1 = v_C1;
        __m128 K2 = v_C2;
#endif

        curr_in = in_r;
        curr_out = out_r;

        if (flag)
        {
            ITW_GATHER_LOW_128_S(curr_in, in_strides, 1, v_in1, tw, cols,
                                 cnt_128_low, load_multi_cols);
            ITW_GATHER_LOW_128_S(curr_in, in_strides, 2, v_in2, tw, cols,
                                 cnt_128_low, load_multi_cols);
        }
        else
        {
            TW_GATHER_LOW_128_S(curr_in, in_strides, 1, v_in1, tw, cols,
                                cnt_128_low, load_multi_cols);
            TW_GATHER_LOW_128_S(curr_in, in_strides, 2, v_in2, tw, cols,
                                cnt_128_low, load_multi_cols);
        }

        LD_LOW_128_S(curr_in, v_in0);

        v_av0 = _mm_add_ps(v_in1, v_in2);
        v_av1 = _mm_sub_ps(v_in1, v_in2);

        v_tv0 = _mm_sub_ps(v_in0, _mm_mul_ps(K1, v_av0));

        // Output point 1: X[0]
        v_out0 = _mm_add_ps(v_in0, v_av0);

        v_tv1 = _mm_mul_ps(K2, v_av1);
        v_tv1 = SWAP_RI_128_S(CONJ_128_S(v_tv1));

        // Output point 2: X[1]
        v_out1 = _mm_sub_ps(v_tv0, v_tv1);
        // Output point 3: X[2]
        v_out2 = _mm_add_ps(v_tv0, v_tv1);

        ST_LOW_128_S(curr_out, v_out0);
        ST_LOW_128_S(curr_out + out_strides[1], v_out1);
        ST_LOW_128_S(curr_out + out_strides[2], v_out2);
    }
#endif

    AOCLFFTZ_LOG(DEBUG, global_logger_mode, "Exit");
}

static VOID TWID_KNAME_FP64(VOID *in_real, VOID *in_imag, VOID *out_real,
                    VOID *out_imag, INTP n, aoclfftz_strides_t *strides,
                    VOID *twd, UINT8 flag)
{
    AOCLFFTZ_LOG(DEBUG, global_logger_mode, "Enter");
    const DOUBLE CRTM_3[2] = {0.500000000000000000000000000000000000000000000,
                              0.866025403784438646763723170752936183471402627};

    DOUBLE *in_r = in_real;
    DOUBLE *out_r = out_real;
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

    aoclfftz_twiddle_t *tws = (aoclfftz_twiddle_t *)twd;
    DOUBLE *tw = (DOUBLE *)(tws->TW);
    UINTP cols = tws->cols;
    UINTP load_multi_cols = tws->load_multi_cols;

    INTP N = n / NUM_SETS_D;
    INTP count;

#if defined(KERNEL_USE_AVX512)
    INTP remaining_sets = n % NUM_SETS_D;
    INTP do_256_whole = (INTP)(remaining_sets >= NUM_SETS_256_D);
    INTP cnt_256 = load_multi_cols * (N * NUM_SETS_512_D);
    INTP cnt_128 =
        load_multi_cols * (N * NUM_SETS_512_D + do_256_whole * NUM_SETS_256_D);
#elif defined(KERNEL_USE_AVX256)
    INTP remaining_sets = n % NUM_SETS_D;
    INTP cnt_128 = load_multi_cols * (N * NUM_SETS_256_D);
#elif defined(KERNEL_USE_AVX128)
    // nothing, since double doesn't have any tail cases to process for AVX128
#endif

    VREGTYPE_D v_C1 = BCAST_D(CRTM_3[0]);
    VREGTYPE_D v_C2 = BCAST_D(CRTM_3[1]);

    v_C2 = NEG_D(v_C2, flag);

    for (count = 0; count < N; count++)
    {
        VREGTYPE_D v_in0, v_in1, v_in2;
        VREGTYPE_D v_av0, v_av1, v_tv0, v_tv1;
        VREGTYPE_D v_out0, v_out1, v_out2;

        curr_in = in_r;
        curr_out = out_r;

        if (flag)
        {
            ITW_GATHER_D(curr_in, in_strides, 1, v_in_stride, v_in1, tw, cols,
                         (count * load_multi_cols * NUM_SETS_D),
                         load_multi_cols);
            ITW_GATHER_D(curr_in, in_strides, 2, v_in_stride, v_in2, tw, cols,
                         (count * load_multi_cols * NUM_SETS_D),
                         load_multi_cols);
        }
        else
        {
            TW_GATHER_D(curr_in, in_strides, 1, v_in_stride, v_in1, tw, cols,
                        (count * load_multi_cols * NUM_SETS_D),
                        load_multi_cols);
            TW_GATHER_D(curr_in, in_strides, 2, v_in_stride, v_in2, tw, cols,
                        (count * load_multi_cols * NUM_SETS_D),
                        load_multi_cols);
        }

        GATHER_D(curr_in, v_in_stride, v_in0);

        v_av0 = ADD_D(v_in1, v_in2);
        v_av1 = SUB_D(v_in1, v_in2);

        v_tv0 = SUB_D(v_in0, MUL_D(v_C1, v_av0));

        // Output point 1: X[0]
        v_out0 = ADD_D(v_in0, v_av0);

        v_tv1 = MUL_D(v_C2, v_av1);
        v_tv1 = SWAP_RI_D(CONJ_D(v_tv1));

        // Output point 2: X[1]
        v_out1 = SUB_D(v_tv0, v_tv1);
        // Output point 3: X[2]
        v_out2 = ADD_D(v_tv0, v_tv1);

        SCATTER_D(curr_out, v_out_stride, v_out0);
        SCATTER_D(curr_out + out_strides[1], v_out_stride, v_out1);
        SCATTER_D(curr_out + out_strides[2], v_out_stride, v_out2);

        in_r += NUM_SETS_D * v_in_stride;
        out_r += NUM_SETS_D * v_out_stride;
    }

    // The following contains code that performs the FFT on the tail cases.
    // These tails are conditionally "instantiated" based on the ISA requested
    // by the "includee" file.

#if defined(KERNEL_USE_AVX512)
    if (remaining_sets >= NUM_SETS_256_D)
    {
        __m256d v_in0, v_in1, v_in2;
        __m256d v_av0, v_av1, v_tv0, v_tv1;
        __m256d v_out0, v_out1, v_out2;

        __m256d K1 = CAST_512_TO_256_D(v_C1);
        __m256d K2 = CAST_512_TO_256_D(v_C2);

        curr_in = in_r;
        curr_out = out_r;

        if (flag)
        {
            ITW_GATHER_256_D(curr_in, in_strides, 1, v_in_stride, v_in1, tw,
                             cols, cnt_256, load_multi_cols);
            ITW_GATHER_256_D(curr_in, in_strides, 2, v_in_stride, v_in2, tw,
                             cols, cnt_256, load_multi_cols);
        }
        else
        {
            TW_GATHER_256_D(curr_in, in_strides, 1, v_in_stride, v_in1, tw,
                            cols, cnt_256, load_multi_cols);
            TW_GATHER_256_D(curr_in, in_strides, 2, v_in_stride, v_in2, tw,
                            cols, cnt_256, load_multi_cols);
        }

        GATHER2_256_D(curr_in, v_in_stride, v_in0);

        v_av0 = _mm256_add_pd(v_in1, v_in2);
        v_av1 = _mm256_sub_pd(v_in1, v_in2);

        v_tv0 = _mm256_sub_pd(v_in0, _mm256_mul_pd(K1, v_av0));

        // Output point 1: X[0]
        v_out0 = _mm256_add_pd(v_in0, v_av0);

        v_tv1 = _mm256_mul_pd(K2, v_av1);
        v_tv1 = SWAP_RI_256_D(CONJ_256_D(v_tv1));

        // Output point 2: X[1]
        v_out1 = _mm256_sub_pd(v_tv0, v_tv1);
        // Output point 3: X[2]
        v_out2 = _mm256_add_pd(v_tv0, v_tv1);

        SCATTER2_256_D(curr_out, v_out_stride, v_out0);
        SCATTER2_256_D(curr_out + out_strides[1], v_out_stride, v_out1);
        SCATTER2_256_D(curr_out + out_strides[2], v_out_stride, v_out2);

        in_r += NUM_SETS_256_D * v_in_stride;
        out_r += NUM_SETS_256_D * v_out_stride;
    }
#endif

#if defined(KERNEL_USE_AVX512) || defined(KERNEL_USE_AVX256)
    if (remaining_sets & 1)
    {
        __m128d v_in0, v_in1, v_in2;
        __m128d v_av0, v_av1, v_tv0, v_tv1;
        __m128d v_out0, v_out1, v_out2;

#if defined(KERNEL_USE_AVX512)
        __m128d K1 = CAST_512_TO_128_D(v_C1);
        __m128d K2 = CAST_512_TO_128_D(v_C2);
#elif defined(KERNEL_USE_AVX256)
        __m128d K1 = CAST_256_TO_128_D(v_C1);
        __m128d K2 = CAST_256_TO_128_D(v_C2);
#endif

        curr_in = in_r;
        curr_out = out_r;

        if (flag)
        {
            ITW_GATHER_128_D(curr_in, in_strides, 1, /* unused */ 0, v_in1, tw,
                             cols, cnt_128, load_multi_cols);
            ITW_GATHER_128_D(curr_in, in_strides, 2, /* unused */ 0, v_in2, tw,
                             cols, cnt_128, load_multi_cols);
        }
        else
        {
            TW_GATHER_128_D(curr_in, in_strides, 1, /* unused */ 0, v_in1, tw,
                            cols, cnt_128, load_multi_cols);
            TW_GATHER_128_D(curr_in, in_strides, 2, /* unused */ 0, v_in2, tw,
                            cols, cnt_128, load_multi_cols);
        }

        LD_128_D(curr_in, v_in0);

        v_av0 = _mm_add_pd(v_in1, v_in2);
        v_av1 = _mm_sub_pd(v_in1, v_in2);

        v_tv0 = _mm_sub_pd(v_in0, _mm_mul_pd(K1, v_av0));

        // Output point 1: X[0]
        v_out0 = _mm_add_pd(v_in0, v_av0);

        v_tv1 = _mm_mul_pd(K2, v_av1);
        v_tv1 = SWAP_RI_128_D(CONJ_128_D(v_tv1));

        // Output point 2: X[1]
        v_out1 = _mm_sub_pd(v_tv0, v_tv1);
        // Output point 3: X[2]
        v_out2 = _mm_add_pd(v_tv0, v_tv1);

        ST_128_D(curr_out, v_out0);
        ST_128_D(curr_out + out_strides[1], v_out1);
        ST_128_D(curr_out + out_strides[2], v_out2);
    }
#endif
    AOCLFFTZ_LOG(DEBUG, global_logger_mode, "Exit");
}

kfft_ REGISTER_KERNEL(UINT8 precision, UINT8 direction)
{
    if (precision == DT_FLOAT)
    {
        return TWID_KNAME_FP32;
    }
    else if (precision == DT_DOUBLE)
    {
        return TWID_KNAME_FP64;
    }
    else
    {
        return NULL;
    }
}

ops_cycles_t GET_OPS_COUNT(UINT8 precision, UINT8 direction)
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
