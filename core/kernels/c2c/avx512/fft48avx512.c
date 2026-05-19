/**
 * Copyright (C) 2026, Advanced Micro Devices. All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 * 1. Redistributions of source code must retain the above copyright notice,
 * this list of conditions and the following disclaimer.
 * 2. Redistributions in binary form must reproduce the above copyright notice,
 * this list of conditions and the following disclaimer in the documentation
 * and/or other materials provided with the distribution.
 * 3. Neither the name of the copyright holder nor the names of its
 * contributors may be used to endorse or promote products derived from this
 * software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
 * ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
 * LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
 * CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
 * SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
 * INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
 * CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
 * ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
 * POSSIBILITY OF SUCH DAMAGE.
 */

/** @file fft48avx512.c
 *
 *  @brief Radix-48 FFT kernel with AVX-512 operations using x86 SIMD intrinsics
 *
 *  This file contains the DIT radix-48 FFT implementations using AVX-512
 *  operations for single-precision and double-precision inputs.
 *
 *  @author Ashwin K. Godbole
 *  @author Jeevanantham N
 *
 */

#include "core/kernels/kernel.h"
#include "core/kernels/simd_includes/simd_common_avx512.h"

static const ops_cycles_t ops_cnt[NUM_PRECISIONS] = {
                                                {5,  13, 44, 96,  46, 39},
                                                {11, 24, 72, 107, 65, 92}};

static const FLOAT twiddle_buf_fp32[5][16] __attribute__((aligned(64))) = {
    {
        1.0f,
        0.0f,
        0.99144486137381041114455752692856287127773827444810f,
        -0.13052619222005159154840622789548901019374070481173f,
        0.96592582628906828674974319972889736763390483900840f,
        -0.25881904510252076234889883762404832834906890131993f,
        0.92387953251128675612818318939678828682241662586364f,
        -0.38268343236508977172845998403039886676134456248563f,
        0.86602540378443864676372317075293618347140262690519f,
        -0.50000000000000000000000000000000000000000000000000f,
        0.79335334029123516457977696150129927662867592105191f,
        -0.60876142900872063941609754289816400451639371196248f,
        0.70710678118654752440084436210484903928483593768847f,
        -0.70710678118654752440084436210484903928483593768847f,
        0.60876142900872063941609754289816400451639371196248f,
        -0.79335334029123516457977696150129927662867592105191f
    },
    {
        1.0f,
        0.0f,
        0.96592582628906828674974319972889736763390483900840f,
        -0.25881904510252076234889883762404832834906890131993f,
        0.86602540378443864676372317075293618347140262690519f,
        -0.50000000000000000000000000000000000000000000000000f,
        0.70710678118654752440084436210484903928483593768847f,
        -0.70710678118654752440084436210484903928483593768847f,
        0.50000000000000000000000000000000000000000000000000f,
        -0.86602540378443864676372317075293618347140262690519f,
        0.25881904510252076234889883762404832834906890131993f,
        -0.96592582628906828674974319972889736763390483900840f,
        0.0f,
        -1.0f,
        -0.25881904510252076234889883762404832834906890131993f,
        -0.96592582628906828674974319972889736763390483900840f
    },
    {
        1.0f,
        0.0f,
        0.92387953251128675612818318939678828682241662586364f,
        -0.38268343236508977172845998403039886676134456248563f,
        0.70710678118654752440084436210484903928483593768847f,
        -0.70710678118654752440084436210484903928483593768847f,
        0.38268343236508977172845998403039886676134456248563f,
        -0.92387953251128675612818318939678828682241662586364f,
        0.0f,
        -1.0f,
        -0.38268343236508977172845998403039886676134456248563f,
        -0.92387953251128675612818318939678828682241662586364f,
        -0.70710678118654752440084436210484903928483593768847f,
        -0.70710678118654752440084436210484903928483593768847f,
        -0.92387953251128675612818318939678828682241662586364f,
        -0.38268343236508977172845998403039886676134456248563f
    },
    {
        1.0f,
        0.0f,
        0.86602540378443864676372317075293618347140262690519f,
        -0.50000000000000000000000000000000000000000000000000f,
        0.50000000000000000000000000000000000000000000000000f,
        -0.86602540378443864676372317075293618347140262690519f,
        0.0f,
        -1.0f,
        -0.50000000000000000000000000000000000000000000000000f,
        -0.86602540378443864676372317075293618347140262690519f,
        -0.86602540378443864676372317075293618347140262690519f,
        -0.50000000000000000000000000000000000000000000000000f,
        -1.0f,
        0.0f,
        -0.86602540378443864676372317075293618347140262690519f,
        0.50000000000000000000000000000000000000000000000000f
    },
    {
        1.0f,
        0.0f,
        0.79335334029123516457977696150129927662867592105191f,
        -0.60876142900872063941609754289816400451639371196248f,
        0.25881904510252076234889883762404832834906890131993f,
        -0.96592582628906828674974319972889736763390483900840f,
        -0.38268343236508977172845998403039886676134456248563f,
        -0.92387953251128675612818318939678828682241662586364f,
        -0.86602540378443864676372317075293618347140262690519f,
        -0.50000000000000000000000000000000000000000000000000f,
        -0.99144486137381041114455752692856287127773827444810f,
        0.13052619222005159154840622789548901019374070481173f,
        -0.70710678118654752440084436210484903928483593768847f,
        0.70710678118654752440084436210484903928483593768847f,
        -0.13052619222005159154840622789548901019374070481173f,
        0.99144486137381041114455752692856287127773827444810f
    }
};

/**
 * @brief store six complex numbers(real,imaginary) of 32 bit single precision
 * floating point number from 512 bit register into memory addresses
 * specified by base address and offset. Only stores positions 0-5, ignoring
 * positions 6-7.
 * Operations : 6 MOV(store), 2 OTHERS(extract).
 * Cast is excluded as it will be a compile time operation.
 */
// Cost: {fma: 0, mul: 0, add: 0, move: 6, perm: 0, other: 2}
#define SCATTER6_512_S(base, offset, src)                                      \
{                                                                              \
    if (offset == 2)                                                           \
    {                                                                          \
        _mm256_storeu_ps(base, _mm512_castps512_ps256(src));                   \
        __m128 lane2 = _mm512_extractf32x4_ps(src, 2);                         \
        _mm_storeu_ps((base) + 8, lane2);                                      \
    }                                                                          \
    else                                                                       \
    {                                                                          \
        __m256 lanes01 = _mm512_castps512_ps256(src);                          \
        __m128 lane0 = _mm256_castps256_ps128(lanes01);                        \
        __m128 lane1 = _mm256_extractf128_ps(lanes01, 1);                      \
        __m128 lane2 = _mm512_extractf32x4_ps(src, 2);                         \
        _mm_storel_pi((__m64 *)(base), lane0);                                 \
        _mm_storeh_pi((__m64 *)((base) + (offset)), lane0);                    \
        _mm_storel_pi((__m64 *)((base) + 2 * (offset)), lane1);                \
        _mm_storeh_pi((__m64 *)((base) + 3 * (offset)), lane1);                \
        _mm_storel_pi((__m64 *)((base) + 4 * (offset)), lane2);                \
        _mm_storeh_pi((__m64 *)((base) + 5 * (offset)), lane2);                \
    }                                                                          \
}

ops_cycles_t get_ops_cnt_fft48avx512(UINT8 precision, UINT8 direction)
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

// Radix-48 FFT kernel using 6x8 decomposition
static VOID fft48avx512fp32(VOID *in_real, VOID *in_imag, VOID *out_real,
                            VOID *out_imag, INTP n, aoclfftz_strides_t *strides,
                            VOID *twd, UINT8 flag)
{
    AOCLFFTZ_LOG(DEBUG, global_logger_mode, "Enter");
    
    const FLOAT CRTM_6[2] = {0.500000000000000000000000000000000000000000000f,
                             0.866025403784438646763723170752936183471402627f};
    const FLOAT CRTM_8[2] = {1.0f,
                             0.707106781186547524400844362104849039284835938f};

    FLOAT * in_r = (FLOAT *)in_real;
    FLOAT * out_r = (FLOAT *)out_real;

#ifdef VOLATILE_STRIDE_ARRAY
    volatile INTP *in_strides = strides->in_strides;
    volatile INTP *out_strides = strides->out_strides;
#else
    INTP *in_strides = strides->in_strides;
    INTP *out_strides = strides->out_strides;
#endif
    INTP v_in_stride = strides->v_in_stride;
    INTP v_out_stride = strides->v_out_stride;

    __m512 v_R6_C1 = _mm512_set1_ps(CRTM_6[0]);
    __m512 v_R6_C2 = _mm512_set1_ps(CRTM_6[1]);
    v_R6_C2 = _mm512_xor_ps(v_R6_C2, _neg_512_f[flag].s);

    __m512 v_R8_C1 = _mm512_set1_ps(CRTM_8[0]);
    v_R8_C1 = _mm512_xor_ps(v_R8_C1, _neg_512_f[flag].s);
    __m512 v_R8_C2 = _mm512_set1_ps(CRTM_8[1]);
    __m512 v_R8_C3 = _mm512_xor_ps(v_R8_C2, _neg_512_f[flag].s);

    // Pre-load twiddles for fast complex multiply
    __m512 twd1 = _mm512_load_ps(twiddle_buf_fp32[0]);
    __m512 twd2 = _mm512_load_ps(twiddle_buf_fp32[1]);
    __m512 twd3 = _mm512_load_ps(twiddle_buf_fp32[2]);
    __m512 twd4 = _mm512_load_ps(twiddle_buf_fp32[3]);
    __m512 twd5 = _mm512_load_ps(twiddle_buf_fp32[4]);

    if (flag)
    {
        twd1 = CONJ_512_S(twd1);
        twd2 = CONJ_512_S(twd2);
        twd3 = CONJ_512_S(twd3);
        twd4 = CONJ_512_S(twd4);
        twd5 = CONJ_512_S(twd5);
    }

    __m512 twd1_re = _mm512_moveldup_ps(twd1);
    __m512 twd1_im = _mm512_movehdup_ps(twd1);
    __m512 twd2_re = _mm512_moveldup_ps(twd2);
    __m512 twd2_im = _mm512_movehdup_ps(twd2);
    __m512 twd3_re = _mm512_moveldup_ps(twd3);
    __m512 twd3_im = _mm512_movehdup_ps(twd3);
    __m512 twd4_re = _mm512_moveldup_ps(twd4);
    __m512 twd4_im = _mm512_movehdup_ps(twd4);
    __m512 twd5_re = _mm512_moveldup_ps(twd5);
    __m512 twd5_im = _mm512_movehdup_ps(twd5);

    INTP group_stride = in_strides[1];
    INTP out_group_stride = out_strides[1];

    INTP in_off8 = in_strides[8];
    INTP in_off16 = in_strides[16];
    INTP in_off24 = in_strides[24];
    INTP in_off32 = in_strides[32];
    INTP in_off40 = in_strides[40];

    INTP out_off6 = out_strides[6];
    INTP out_off12 = out_strides[12];
    INTP out_off18 = out_strides[18];
    INTP out_off24 = out_strides[24];
    INTP out_off30 = out_strides[30];
    INTP out_off36 = out_strides[36];
    INTP out_off42 = out_strides[42];

    for (INTP i = 0; i < n; i++)
    {
        __m512 v_in0, v_in1, v_in2, v_in3, v_in4, v_in5;
        __m512 v_cv1, v_cv2, v_cv3, v_cv4, v_cv5, v_cv6;
        __m512 v_av1, v_av2, v_av3, v_av4;
        __m512 v_tv1, v_tv2, v_tv3;
        __m512 v_out0, v_out1, v_out2, v_out3, v_out4, v_out5;

        GATHER8_512_S(in_r, group_stride, v_in0);
        GATHER8_512_S(in_r + in_off8, group_stride, v_in1);
        GATHER8_512_S(in_r + in_off16, group_stride, v_in2);
        GATHER8_512_S(in_r + in_off24, group_stride, v_in3);
        GATHER8_512_S(in_r + in_off32, group_stride, v_in4);
        GATHER8_512_S(in_r + in_off40, group_stride, v_in5);

        v_cv1 = _mm512_add_ps(v_in0, v_in3);
        v_cv2 = _mm512_add_ps(v_in2, v_in4);
        v_cv3 = _mm512_add_ps(v_in1, v_in5);
        v_cv4 = _mm512_sub_ps(v_in0, v_in3);
        v_cv5 = _mm512_sub_ps(v_in1, v_in5);
        v_cv6 = _mm512_sub_ps(v_in2, v_in4);

        v_av1 = _mm512_add_ps(v_cv2, v_cv3);
        v_av2 = _mm512_sub_ps(v_cv2, v_cv3);
        v_av3 = _mm512_sub_ps(v_cv5, v_cv6);
        v_av4 = _mm512_add_ps(v_cv5, v_cv6);

        v_tv1 = _mm512_mul_ps(v_R6_C1, v_av1);
        v_tv2 = _mm512_mul_ps(v_R6_C2, v_av3);
        v_tv2 = CONJ_512_S(SWAP_RI_512_S(v_tv2));
        v_tv3 = _mm512_mul_ps(v_R6_C2, v_av4);
        v_tv3 = CONJ_512_S(SWAP_RI_512_S(v_tv3));

        v_out0 = _mm512_add_ps(v_cv1, v_av1);
        v_out3 = _mm512_add_ps(v_cv4, v_av2);

        v_av1 = _mm512_sub_ps(v_cv1, v_tv1);
        v_av2 = _mm512_sub_ps(v_cv4, _mm512_mul_ps(v_R6_C1, v_av2));

        v_out2 = _mm512_add_ps(v_av1, v_tv2);
        v_out4 = _mm512_sub_ps(v_av1, v_tv2);
        v_out1 = _mm512_add_ps(v_av2, v_tv3);
        v_out5 = _mm512_sub_ps(v_av2, v_tv3);

        v_out1 = _mm512_fmaddsub_ps(v_out1, twd1_re,
                     _mm512_mul_ps(SWAP_RI_512_S(v_out1), twd1_im));
        v_out2 = _mm512_fmaddsub_ps(v_out2, twd2_re,
                     _mm512_mul_ps(SWAP_RI_512_S(v_out2), twd2_im));
        v_out3 = _mm512_fmaddsub_ps(v_out3, twd3_re,
                     _mm512_mul_ps(SWAP_RI_512_S(v_out3), twd3_im));
        v_out4 = _mm512_fmaddsub_ps(v_out4, twd4_re,
                     _mm512_mul_ps(SWAP_RI_512_S(v_out4), twd4_im));
        v_out5 = _mm512_fmaddsub_ps(v_out5, twd5_re,
                     _mm512_mul_ps(SWAP_RI_512_S(v_out5), twd5_im));

        __m512 lo_01 = _mm512_shuffle_ps(v_out0, v_out1, 0x44);
        __m512 hi_01 = _mm512_shuffle_ps(v_out0, v_out1, 0xEE);
        __m512 lo_23 = _mm512_shuffle_ps(v_out2, v_out3, 0x44);
        __m512 hi_23 = _mm512_shuffle_ps(v_out2, v_out3, 0xEE);
        __m512 lo_45 = _mm512_shuffle_ps(v_out4, v_out5, 0x44);
        __m512 hi_45 = _mm512_shuffle_ps(v_out4, v_out5, 0xEE);

        __m512 temp_04_0123 = _mm512_shuffle_f32x4(lo_01, lo_23, 0x88);
        __m512 temp_26_0123 = _mm512_shuffle_f32x4(lo_01, lo_23, 0xDD);
        __m512 temp_15_0123 = _mm512_shuffle_f32x4(hi_01, hi_23, 0x88);
        __m512 temp_37_0123 = _mm512_shuffle_f32x4(hi_01, hi_23, 0xDD);

        __m512 t0 = _mm512_shuffle_f32x4(temp_04_0123, lo_45, 0x88);
        __m512 t4 = _mm512_shuffle_f32x4(temp_04_0123, lo_45, 0xAD);
        __m512 t2 = _mm512_shuffle_f32x4(temp_26_0123, lo_45, 0x98);
        __m512 t6 = _mm512_shuffle_f32x4(temp_26_0123, lo_45, 0xBD);
        __m512 t1 = _mm512_shuffle_f32x4(temp_15_0123, hi_45, 0x88);
        __m512 t5 = _mm512_shuffle_f32x4(temp_15_0123, hi_45, 0xAD);
        __m512 t3 = _mm512_shuffle_f32x4(temp_37_0123, hi_45, 0x98);
        __m512 t7 = _mm512_shuffle_f32x4(temp_37_0123, hi_45, 0xBD);

        __m512 r8_cv1 = _mm512_add_ps(t0, t4);
        __m512 r8_cv2 = _mm512_add_ps(t2, t6);
        __m512 r8_cv3 = _mm512_add_ps(t1, t5);
        __m512 r8_cv4 = _mm512_add_ps(t3, t7);
        __m512 r8_cv5 = _mm512_sub_ps(t0, t4);
        __m512 r8_cv6 = _mm512_sub_ps(t2, t6);
        __m512 r8_cv7 = _mm512_sub_ps(t1, t5);
        __m512 r8_cv8 = _mm512_sub_ps(t3, t7);

        __m512 r8_av1 = _mm512_add_ps(r8_cv1, r8_cv2);
        __m512 r8_av2 = _mm512_add_ps(r8_cv3, r8_cv4);
        __m512 r8_out0 = _mm512_add_ps(r8_av1, r8_av2);
        __m512 r8_out4 = _mm512_sub_ps(r8_av1, r8_av2);

        r8_av1 = _mm512_sub_ps(r8_cv3, r8_cv4);
        r8_av2 = _mm512_sub_ps(r8_cv1, r8_cv2);
        __m512 r8_tv1 = _mm512_mul_ps(v_R8_C1, r8_av1);
        r8_tv1 = CONJ_512_S(r8_tv1);
        r8_tv1 = SWAP_RI_512_S(r8_tv1);
        __m512 r8_out6 = _mm512_add_ps(r8_av2, r8_tv1);
        __m512 r8_out2 = _mm512_sub_ps(r8_av2, r8_tv1);

        r8_av1 = _mm512_sub_ps(r8_cv7, r8_cv8);
        r8_tv1 = _mm512_mul_ps(v_R8_C2, r8_av1);
        r8_av1 = _mm512_add_ps(r8_cv7, r8_cv8);
        __m512 r8_tv2 = _mm512_mul_ps(v_R8_C3, r8_av1);
        __m512 r8_tv3 = _mm512_mul_ps(v_R8_C1, r8_cv6);

        r8_av1 = _mm512_sub_ps(r8_tv3, r8_tv2);
        r8_av2 = _mm512_add_ps(r8_tv3, r8_tv2);
        r8_av1 = CONJ_512_S(r8_av1);
        r8_av1 = SWAP_RI_512_S(r8_av1);
        r8_av2 = CONJ_512_S(r8_av2);
        r8_av2 = SWAP_RI_512_S(r8_av2);

        __m512 r8_av3 = _mm512_sub_ps(r8_cv5, r8_tv1);
        __m512 r8_av4 = _mm512_add_ps(r8_cv5, r8_tv1);

        __m512 r8_out1 = _mm512_sub_ps(r8_av4, r8_av2);
        __m512 r8_out7 = _mm512_add_ps(r8_av4, r8_av2);
        __m512 r8_out5 = _mm512_sub_ps(r8_av3, r8_av1);
        __m512 r8_out3 = _mm512_add_ps(r8_av3, r8_av1);

        SCATTER6_512_S(out_r, out_group_stride, r8_out0);
        SCATTER6_512_S(out_r + out_off6,  out_group_stride, r8_out1);
        SCATTER6_512_S(out_r + out_off12, out_group_stride, r8_out2);
        SCATTER6_512_S(out_r + out_off18, out_group_stride, r8_out3);
        SCATTER6_512_S(out_r + out_off24, out_group_stride, r8_out4);
        SCATTER6_512_S(out_r + out_off30, out_group_stride, r8_out5);
        SCATTER6_512_S(out_r + out_off36, out_group_stride, r8_out6);
        SCATTER6_512_S(out_r + out_off42, out_group_stride, r8_out7);

        in_r += v_in_stride;
        out_r += v_out_stride;
    }

    // No tail cases: this kernel processes one batch at a time,
    // not multiple batches.

    AOCLFFTZ_LOG(DEBUG, global_logger_mode, "Exit");
}

/* Branch-free conjugate for FP64 kernel: [0] = identity, [1] = conjugate (flip sign of imag). */
static const union data_union_512 _conj_512_d_fp48[2] = {
    {.u = { 0x00000000, 0x00000000, 0x00000000, 0x00000000,
            0x00000000, 0x00000000, 0x00000000, 0x00000000,
            0x00000000, 0x00000000, 0x00000000, 0x00000000,
            0x00000000, 0x00000000, 0x00000000, 0x00000000 }},
    {.u = { 0x00000000, 0x00000000, 0x00000000, 0x80000000,
            0x00000000, 0x00000000, 0x00000000, 0x80000000,
            0x00000000, 0x00000000, 0x00000000, 0x80000000,
            0x00000000, 0x00000000, 0x00000000, 0x80000000 }}
};
#define CONJ_512_D_48(x, flag) _mm512_xor_pd(_conj_512_d_fp48[flag].d, (x))

static const DOUBLE twiddle_buf_fp64[11][8] __attribute__((aligned(64))) = {
    {
        1.0,
        0.0,
        0.99144486137381041114455752692856287127773827444810,
        -0.13052619222005159154840622789548901019374070481173,
        0.96592582628906828674974319972889736763390483900840,
        -0.25881904510252076234889883762404832834906890131993,
        0.92387953251128675612818318939678828682241662586364,
        -0.38268343236508977172845998403039886676134456248563
    },
    {
        1.0,
        0.0,
        0.96592582628906828674974319972889736763390483900840,
        -0.25881904510252076234889883762404832834906890131993,
        0.86602540378443864676372317075293618347140262690519,
        -0.50000000000000000000000000000000000000000000000000,
        0.70710678118654752440084436210484903928483593768847,
        -0.70710678118654752440084436210484903928483593768847
    },
    {
        1.0,
        0.0,
        0.92387953251128675612818318939678828682241662586364,
        -0.38268343236508977172845998403039886676134456248563,
        0.70710678118654752440084436210484903928483593768847,
        -0.70710678118654752440084436210484903928483593768847,
        0.38268343236508977172845998403039886676134456248563,
        -0.92387953251128675612818318939678828682241662586364
    },
    
    {
        1.0,
        0.0,
        0.86602540378443864676372317075293618347140262690519,
        -0.50000000000000000000000000000000000000000000000000,
        0.50000000000000000000000000000000000000000000000000,
        -0.86602540378443864676372317075293618347140262690519,
        0.0,
        -1.0
    },
    {
        1.0,
        0.0,
        0.79335334029123516457977696150129927662867592105191,
        -0.60876142900872063941609754289816400451639371196248,
        0.25881904510252076234889883762404832834906890131993,
        -0.96592582628906828674974319972889736763390483900840,
        -0.38268343236508977172845998403039886676134456248563,
        -0.92387953251128675612818318939678828682241662586364
    },
    {
        1.0,
        0.0,
        0.70710678118654752440084436210484903928483593768847,
        -0.70710678118654752440084436210484903928483593768847,
        0.0,
        -1.0,
        -0.70710678118654752440084436210484903928483593768847,
        -0.70710678118654752440084436210484903928483593768847
    },
    {
        1.0,
        0.0,
        0.60876142900872063941609754289816400451639371196248,
        -0.79335334029123516457977696150129927662867592105191,
        -0.25881904510252076234889883762404832834906890131993,
        -0.96592582628906828674974319972889736763390483900840,
        -0.92387953251128675612818318939678828682241662586364,
        -0.38268343236508977172845998403039886676134456248563
    },
    
    {
        1.0,
        0.0,
        0.50000000000000000000000000000000000000000000000000,
        -0.86602540378443864676372317075293618347140262690519,
        -0.50000000000000000000000000000000000000000000000000,
        -0.86602540378443864676372317075293618347140262690519,
        -1.0,
        0.0
    },
    {
        1.0,
        0.0,
        0.38268343236508977172845998403039886676134456248563,
        -0.92387953251128675612818318939678828682241662586364,
        -0.70710678118654752440084436210484903928483593768847,
        -0.70710678118654752440084436210484903928483593768847,
        -0.92387953251128675612818318939678828682241662586364,
        0.38268343236508977172845998403039886676134456248563
    },
    {
        1.0,
        0.0,
        0.25881904510252076234889883762404832834906890131993,
        -0.96592582628906828674974319972889736763390483900840,
        -0.86602540378443864676372317075293618347140262690519,
        -0.50000000000000000000000000000000000000000000000000,
        -0.70710678118654752440084436210484903928483593768847,
        0.70710678118654752440084436210484903928483593768847
    },
    {
        1.0,
        0.0,
        0.13052619222005159154840622789548901019374070481173,
        -0.99144486137381041114455752692856287127773827444810,
        -0.96592582628906828674974319972889736763390483900840,
        -0.25881904510252076234889883762404832834906890131993,
        -0.38268343236508977172845998403039886676134456248563,
        0.92387953251128675612818318939678828682241662586364}
};

// Radix-48 FFT kernel using 12x4 decomposition (AVX-512 double precision)
static VOID fft48avx512fp64(VOID *in_real, VOID *in_imag,
                            VOID *out_real, VOID *out_imag,
                            INTP n, aoclfftz_strides_t *strides,
                            VOID *twd, UINT8 flag)
{
    AOCLFFTZ_LOG(DEBUG, global_logger_mode, "Enter");
    
    const DOUBLE CRTM_12[3] = {
        0.86602540378443864676372317075293618347140262700000,
        0.50000000000000000000000000000000000000000000000000,
        1.00000000000000000000000000000000000000000000000000
    };

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

    __m512d v_C1 = _mm512_set1_pd(CRTM_12[0]);
    __m512d v_C2 = _mm512_set1_pd(CRTM_12[1]);

    __m512d v_sign_conj = _mm512_xor_pd(_neg_512_d[flag].d, _conj_512_d.d);

    __m512d v_C4_conj = _mm512_xor_pd(_mm512_set1_pd(CRTM_12[0]), _conj_512_d.d);
    v_C4_conj = _mm512_xor_pd(v_C4_conj, _neg_512_d[flag].d);
    __m512d v_C5_conj = _mm512_xor_pd(_mm512_set1_pd(CRTM_12[1]), _conj_512_d.d);
    v_C5_conj = _mm512_xor_pd(v_C5_conj, _neg_512_d[flag].d);
        
    INTP group_stride = in_strides[1];
    INTP out_group_stride = out_strides[1];

    __m512d twd1_r_  = CONJ_512_D_48(_mm512_load_pd(twiddle_buf_fp64[ 0]), flag);
    __m512d twd1_real = _mm512_movedup_pd(twd1_r_);
    __m512d twd1_imag = _mm512_unpackhi_pd(twd1_r_, twd1_r_);

    __m512d twd2_r_  = CONJ_512_D_48(_mm512_load_pd(twiddle_buf_fp64[ 1]), flag);
    __m512d twd2_real = _mm512_movedup_pd(twd2_r_);
    __m512d twd2_imag = _mm512_unpackhi_pd(twd2_r_, twd2_r_);

    __m512d twd3_r_  = CONJ_512_D_48(_mm512_load_pd(twiddle_buf_fp64[ 2]), flag);
    __m512d twd3_real = _mm512_movedup_pd(twd3_r_);
    __m512d twd3_imag = _mm512_unpackhi_pd(twd3_r_, twd3_r_);

    __m512d twd4_r_  = CONJ_512_D_48(_mm512_load_pd(twiddle_buf_fp64[ 3]), flag);
    __m512d twd4_real = _mm512_movedup_pd(twd4_r_);
    __m512d twd4_imag = _mm512_unpackhi_pd(twd4_r_, twd4_r_);

    __m512d twd5_r_  = CONJ_512_D_48(_mm512_load_pd(twiddle_buf_fp64[ 4]), flag);
    __m512d twd5_real = _mm512_movedup_pd(twd5_r_);
    __m512d twd5_imag = _mm512_unpackhi_pd(twd5_r_, twd5_r_);

    __m512d twd6_r_  = CONJ_512_D_48(_mm512_load_pd(twiddle_buf_fp64[ 5]), flag);
    __m512d twd6_real = _mm512_movedup_pd(twd6_r_);
    __m512d twd6_imag = _mm512_unpackhi_pd(twd6_r_, twd6_r_);

    __m512d twd7_r_  = CONJ_512_D_48(_mm512_load_pd(twiddle_buf_fp64[ 6]), flag);
    __m512d twd7_real = _mm512_movedup_pd(twd7_r_);
    __m512d twd7_imag = _mm512_unpackhi_pd(twd7_r_, twd7_r_);

    __m512d twd8_r_  = CONJ_512_D_48(_mm512_load_pd(twiddle_buf_fp64[ 7]), flag);
    __m512d twd8_real = _mm512_movedup_pd(twd8_r_);
    __m512d twd8_imag = _mm512_unpackhi_pd(twd8_r_, twd8_r_);

    __m512d twd9_r_  = CONJ_512_D_48(_mm512_load_pd(twiddle_buf_fp64[ 8]), flag);
    __m512d twd9_real = _mm512_movedup_pd(twd9_r_);
    __m512d twd9_imag = _mm512_unpackhi_pd(twd9_r_, twd9_r_);

    __m512d twd10_r_  = CONJ_512_D_48(_mm512_load_pd(twiddle_buf_fp64[ 9]), flag);
    __m512d twd10_real = _mm512_movedup_pd(twd10_r_);
    __m512d twd10_imag = _mm512_unpackhi_pd(twd10_r_, twd10_r_);

    __m512d twd11_r_  = CONJ_512_D_48(_mm512_load_pd(twiddle_buf_fp64[10]), flag);
    __m512d twd11_real = _mm512_movedup_pd(twd11_r_);
    __m512d twd11_imag = _mm512_unpackhi_pd(twd11_r_, twd11_r_);

    for (INTP i = 0; i < n; i++)
    {
        __m512d v_in[12] __attribute__((aligned(64)));
        __m512d v_av1, v_av2, v_av3, v_av4, v_av5, v_av6;
        __m512d v_av7, v_av8, v_av9, v_av10, v_av11, v_av12;
        __m512d v_cv1, v_cv2, v_cv3, v_cv4, v_cv5, v_cv6, v_cv7, v_cv8;
        __m512d v_tv1, v_tv2, v_tv3, v_tv5;
        
        GATHER4_512_D(in_r                  , group_stride, v_in[0]);
        GATHER4_512_D(in_r + in_strides[4]  , group_stride, v_in[1]);
        GATHER4_512_D(in_r + in_strides[8]  , group_stride, v_in[2]);
        GATHER4_512_D(in_r + in_strides[12] , group_stride, v_in[3]);
        GATHER4_512_D(in_r + in_strides[16] , group_stride, v_in[4]);
        GATHER4_512_D(in_r + in_strides[20] , group_stride, v_in[5]);
        GATHER4_512_D(in_r + in_strides[24] , group_stride, v_in[6]);
        GATHER4_512_D(in_r + in_strides[28] , group_stride, v_in[7]);
        GATHER4_512_D(in_r + in_strides[32] , group_stride, v_in[8]);
        GATHER4_512_D(in_r + in_strides[36] , group_stride, v_in[9]);
        GATHER4_512_D(in_r + in_strides[40] , group_stride, v_in[10]);
        GATHER4_512_D(in_r + in_strides[44] , group_stride, v_in[11]);

        v_av1 = _mm512_add_pd(v_in[0], v_in[6]);
        v_av2 = _mm512_add_pd(v_in[2], v_in[4]);
        v_av3 = _mm512_add_pd(v_in[8], v_in[10]);
        v_av4 = _mm512_add_pd(v_in[1], v_in[5]);
        v_av5 = _mm512_add_pd(v_in[7], v_in[11]);
        v_av6 = _mm512_add_pd(v_in[3], v_in[9]);

        v_cv1 = _mm512_add_pd(v_av2, v_av3);
        v_cv2 = _mm512_add_pd(v_av4, v_av5);
        v_cv3 = _mm512_add_pd(v_av1, v_av6);
        v_cv4 = _mm512_sub_pd(v_av1, v_av6);
        v_cv5 = _mm512_add_pd(v_cv1, v_cv2);
        v_cv6 = _mm512_sub_pd(v_cv1, v_cv2);

        v_tv1 = _mm512_add_pd(v_cv3, v_cv5);
        v_tv2 = _mm512_add_pd(v_cv4, v_cv6);

        v_av7 = _mm512_sub_pd(v_in[0], v_in[6]);
        v_av8 = _mm512_sub_pd(v_in[2], v_in[4]);
        v_av9 = _mm512_sub_pd(v_in[8], v_in[10]);
        v_av10 = _mm512_sub_pd(v_in[1], v_in[5]);
        v_av11 = _mm512_sub_pd(v_in[7], v_in[11]);
        v_av12 = _mm512_sub_pd(v_in[3], v_in[9]);

        v_in[0] = v_tv1;
        v_in[6] = v_tv2;

        v_cv1 = _mm512_sub_pd(v_av8, v_av9);
        v_cv2 = _mm512_sub_pd(v_av4, v_av5);
        v_cv7 = _mm512_sub_pd(v_cv2, v_av12);
        v_cv8 = _mm512_sub_pd(v_av7, v_cv1);

        v_tv3 = _mm512_xor_pd(v_cv7, v_sign_conj);
        v_tv3 = SWAP_RI_512_D(v_tv3);
        
        v_in[3] = _mm512_sub_pd(v_cv8, v_tv3);
        v_in[9] = _mm512_add_pd(v_cv8, v_tv3);
        
        v_tv3 = _mm512_xor_pd(v_av12, v_sign_conj);
        
        v_cv1 = _mm512_fmadd_pd(v_C2, v_cv1, v_av7);
        v_cv7 = _mm512_sub_pd(v_av10, v_av11);

        v_tv5 = _mm512_mul_pd(v_C1, v_cv7);
        v_cv8 = _mm512_add_pd(v_cv1, v_tv5);
        
        const __m512d cv2_save = v_cv2;
        v_cv2 = _mm512_sub_pd(v_av2, v_av3);
        v_tv2 = _mm512_mul_pd(v_cv2, v_C4_conj);
        
        v_cv2 = _mm512_fmadd_pd(cv2_save, v_C5_conj, v_tv3);
        v_cv7 = _mm512_add_pd(v_cv2, v_tv2);
        v_cv7 = SWAP_RI_512_D(v_cv7);

        v_in[1] = _mm512_sub_pd(v_cv8, v_cv7);
        v_in[11] = _mm512_add_pd(v_cv8, v_cv7);

        v_cv7 = _mm512_sub_pd(v_cv1, v_tv5);
        v_cv8 = _mm512_sub_pd(v_cv2, v_tv2);
        v_cv8 = SWAP_RI_512_D(v_cv8);

        v_in[5] = _mm512_sub_pd(v_cv7, v_cv8);
        v_in[7] = _mm512_add_pd(v_cv7, v_cv8);
        
        v_cv1 = _mm512_fnmadd_pd(v_C2, v_cv6, v_cv4);
        v_cv2 = _mm512_add_pd(v_av8, v_av9);
        v_cv4 = _mm512_add_pd(v_av10, v_av11);
        v_cv6 = _mm512_add_pd(v_cv4, v_cv2);
        v_tv2 = _mm512_mul_pd(v_cv6, v_C4_conj);
        v_tv2 = SWAP_RI_512_D(v_tv2);
        
        v_in[2] = _mm512_sub_pd(v_cv1, v_tv2);
        v_in[10] = _mm512_add_pd(v_cv1, v_tv2);
        
        v_cv1 = _mm512_fnmadd_pd(v_C2, v_cv5, v_cv3);
        v_cv6 = _mm512_sub_pd(v_cv4, v_cv2);
        v_tv2 = _mm512_mul_pd(v_cv6, v_C4_conj);
        v_tv2 = SWAP_RI_512_D(v_tv2);
        
        v_in[4] = _mm512_sub_pd(v_cv1, v_tv2);
        v_in[8] = _mm512_add_pd(v_cv1, v_tv2);
        
        __m512d sw1 = SWAP_RI_512_D(v_in[1]);
        v_in[1] = _mm512_fmaddsub_pd(v_in[1], twd1_real, _mm512_mul_pd(sw1, twd1_imag));
        
        __m512d sw2 = SWAP_RI_512_D(v_in[2]);
        v_in[2] = _mm512_fmaddsub_pd(v_in[2], twd2_real, _mm512_mul_pd(sw2, twd2_imag));
        
        __m512d sw3 = SWAP_RI_512_D(v_in[3]);
        v_in[3] = _mm512_fmaddsub_pd(v_in[3], twd3_real, _mm512_mul_pd(sw3, twd3_imag));
        
        __m512d t_b0, t_b1, t_b2, t_b3;
        __m512d ab_02, ab_13, cd_02, cd_13;

        ab_02 = _mm512_shuffle_f64x2(v_in[0], v_in[1], 0x88);
        ab_13 = _mm512_shuffle_f64x2(v_in[0], v_in[1], 0xDD);
        cd_02 = _mm512_shuffle_f64x2(v_in[2], v_in[3], 0x88);
        cd_13 = _mm512_shuffle_f64x2(v_in[2], v_in[3], 0xDD);
        t_b0 = _mm512_shuffle_f64x2(ab_02, cd_02, 0x88);
        t_b2 = _mm512_shuffle_f64x2(ab_02, cd_02, 0xDD);
        t_b1 = _mm512_shuffle_f64x2(ab_13, cd_13, 0x88);
        t_b3 = _mm512_shuffle_f64x2(ab_13, cd_13, 0xDD);

        __m512d r4_av1, r4_av2, r4_out0, r4_out1, r4_out2, r4_out3;

        r4_av1 = _mm512_add_pd(t_b0, t_b2);
        r4_av2 = _mm512_add_pd(t_b1, t_b3);
        r4_out0 = _mm512_add_pd(r4_av1, r4_av2);
        r4_out2 = _mm512_sub_pd(r4_av1, r4_av2);

        r4_av1 = _mm512_sub_pd(t_b3, t_b1);
        r4_av1 = _mm512_xor_pd(r4_av1, v_sign_conj);
        r4_av1 = SWAP_RI_512_D(r4_av1);
        r4_av2 = _mm512_sub_pd(t_b0, t_b2);
        r4_out1 = _mm512_add_pd(r4_av2, r4_av1);
        r4_out3 = _mm512_sub_pd(r4_av2, r4_av1);

        SCATTER4_512_D(out_r, out_group_stride, r4_out0);
        SCATTER4_512_D(out_r + out_strides[12], out_group_stride, r4_out1);
        SCATTER4_512_D(out_r + out_strides[24], out_group_stride, r4_out2);
        SCATTER4_512_D(out_r + out_strides[36], out_group_stride, r4_out3);
        
        __m512d sw4 = SWAP_RI_512_D(v_in[4]);
        v_in[4] = _mm512_fmaddsub_pd(v_in[4], twd4_real, _mm512_mul_pd(sw4, twd4_imag));
        
        __m512d sw5 = SWAP_RI_512_D(v_in[5]);
        v_in[5] = _mm512_fmaddsub_pd(v_in[5], twd5_real, _mm512_mul_pd(sw5, twd5_imag));
        
        __m512d sw6 = SWAP_RI_512_D(v_in[6]);
        v_in[6] = _mm512_fmaddsub_pd(v_in[6], twd6_real, _mm512_mul_pd(sw6, twd6_imag));
        
        __m512d sw7 = SWAP_RI_512_D(v_in[7]);
        v_in[7] = _mm512_fmaddsub_pd(v_in[7], twd7_real, _mm512_mul_pd(sw7, twd7_imag));
        
        ab_02 = _mm512_shuffle_f64x2(v_in[4], v_in[5], 0x88);
        ab_13 = _mm512_shuffle_f64x2(v_in[4], v_in[5], 0xDD);
        cd_02 = _mm512_shuffle_f64x2(v_in[6], v_in[7], 0x88);
        cd_13 = _mm512_shuffle_f64x2(v_in[6], v_in[7], 0xDD);
        t_b0 = _mm512_shuffle_f64x2(ab_02, cd_02, 0x88);
        t_b2 = _mm512_shuffle_f64x2(ab_02, cd_02, 0xDD);
        t_b1 = _mm512_shuffle_f64x2(ab_13, cd_13, 0x88);
        t_b3 = _mm512_shuffle_f64x2(ab_13, cd_13, 0xDD);
        
        r4_av1 = _mm512_add_pd(t_b0, t_b2);
        r4_av2 = _mm512_add_pd(t_b1, t_b3);
        r4_out0 = _mm512_add_pd(r4_av1, r4_av2);
        r4_out2 = _mm512_sub_pd(r4_av1, r4_av2);

        r4_av1 = _mm512_sub_pd(t_b1, t_b3);
        r4_av1 = _mm512_xor_pd(r4_av1, v_sign_conj);
        r4_av1 = SWAP_RI_512_D(r4_av1);
        r4_av2 = _mm512_sub_pd(t_b0, t_b2);
        r4_out1 = _mm512_add_pd(r4_av2, r4_av1);
        r4_out3 = _mm512_sub_pd(r4_av2, r4_av1);

        SCATTER4_512_D(out_r + out_strides[4], out_group_stride, r4_out0);
        SCATTER4_512_D(out_r + out_strides[16], out_group_stride, r4_out3);
        SCATTER4_512_D(out_r + out_strides[28], out_group_stride, r4_out2);
        SCATTER4_512_D(out_r + out_strides[40], out_group_stride, r4_out1);
        
        __m512d sw8 = SWAP_RI_512_D(v_in[8]);
        v_in[8] = _mm512_fmaddsub_pd(v_in[8], twd8_real, _mm512_mul_pd(sw8, twd8_imag));
        
        __m512d sw9 = SWAP_RI_512_D(v_in[9]);
        v_in[9] = _mm512_fmaddsub_pd(v_in[9], twd9_real, _mm512_mul_pd(sw9, twd9_imag));
        
        __m512d sw10 = SWAP_RI_512_D(v_in[10]);
        v_in[10] = _mm512_fmaddsub_pd(v_in[10], twd10_real, _mm512_mul_pd(sw10, twd10_imag));
        
        __m512d sw11 = SWAP_RI_512_D(v_in[11]);
        v_in[11] = _mm512_fmaddsub_pd(v_in[11], twd11_real, _mm512_mul_pd(sw11, twd11_imag));
        
        ab_02 = _mm512_shuffle_f64x2(v_in[8], v_in[9], 0x88);
        ab_13 = _mm512_shuffle_f64x2(v_in[8], v_in[9], 0xDD);
        cd_02 = _mm512_shuffle_f64x2(v_in[10], v_in[11], 0x88);
        cd_13 = _mm512_shuffle_f64x2(v_in[10], v_in[11], 0xDD);
        t_b0 = _mm512_shuffle_f64x2(ab_02, cd_02, 0x88);
        t_b2 = _mm512_shuffle_f64x2(ab_02, cd_02, 0xDD);
        t_b1 = _mm512_shuffle_f64x2(ab_13, cd_13, 0x88);
        t_b3 = _mm512_shuffle_f64x2(ab_13, cd_13, 0xDD);

        r4_av1 = _mm512_add_pd(t_b0, t_b2);
        r4_av2 = _mm512_add_pd(t_b1, t_b3);
        r4_out0 = _mm512_add_pd(r4_av1, r4_av2);
        r4_out2 = _mm512_sub_pd(r4_av1, r4_av2);

        r4_av1 = _mm512_sub_pd(t_b3, t_b1);
        r4_av1 = _mm512_xor_pd(r4_av1, v_sign_conj);
        r4_av1 = SWAP_RI_512_D(r4_av1);
        r4_av2 = _mm512_sub_pd(t_b0, t_b2);
        r4_out1 = _mm512_add_pd(r4_av2, r4_av1);
        r4_out3 = _mm512_sub_pd(r4_av2, r4_av1);

        SCATTER4_512_D(out_r + out_strides[8], out_group_stride, r4_out0);
        SCATTER4_512_D(out_r + out_strides[20], out_group_stride, r4_out1);
        SCATTER4_512_D(out_r + out_strides[32], out_group_stride, r4_out2);
        SCATTER4_512_D(out_r + out_strides[44], out_group_stride, r4_out3);

        in_r += v_in_stride;
        out_r += v_out_stride;
    }
    
    // No tail cases: this kernel processes one batch at a time,
    // not multiple batches.

    AOCLFFTZ_LOG(DEBUG, global_logger_mode, "Exit");
}

kfft_ register_kernel_fft48avx512(UINT8 precision, UINT8 direction /* unused */)
{
    if (precision == DT_FLOAT)
    {
        return fft48avx512fp32;
    }
    else if (precision == DT_DOUBLE)
    {
        return fft48avx512fp64;
    }
    else
    {
        return NULL;
    }
}
