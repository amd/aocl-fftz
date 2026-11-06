// Copyright Advanced Micro Devices, Inc.
// SPDX-License-Identifier: BSD-3-Clause

/** @file fused_twiddle_transpose_avx512.c
 *
 *  @brief Fused four-step inter-step twiddle + out-of-place transpose, AVX-512.
 *
 *  Replaces the four-step step-2 (inter-step twiddle multiply) + step-3
 *  (out-of-place transpose) with a single pass: each source micro-tile is
 *  multiplied by its dense twiddle tile (inverse path: by the conjugate) and
 *  transposed in registers before the full-cacheline store, removing the
 *  intermediate write-back and re-read of the n1 x n2 matrix.
 *
 *  Mirrors the tiling of transpose_outofplace_avx512.c: a 4x4 (fp64) / 8x8
 *  (fp32) register micro-tile, each dest row written as a 64 B cacheline. The
 *  complex multiply reuses the CMUL FMA macros; the twiddle table is consumed
 *  in blocked micro-tile order (see fused_twiddle_transpose.h). The fwd/bwd
 *  wrappers pass a literal `conjugate` to one static inline core so the
 *  compiler folds the branch away.
 *
 *  @author Ashwin K. Godbole
 */

#include "core/kernels/kernel.h"
#include "core/kernels/non_dft/fused_twiddle_transpose/fused_twiddle_transpose.h"
#include "core/kernels/simd_includes/simd_common_avx512.h"
#include "core/kernels/non_dft/fused_twiddle_transpose/fused_twiddle_transpose_cmul.h"
#include <immintrin.h>

// In-register transpose of a 4x4 tile of complex doubles (each complex is one
// 128-bit lane) via four-lane shuffles.
#define TRANSPOSE_4x4_FP64(r0, r1, r2, r3, o0, o1, o2, o3)                     \
    do                                                                         \
    {                                                                          \
        __m512d _t0 = _mm512_shuffle_f64x2(r0, r1, 0x88);                      \
        __m512d _t1 = _mm512_shuffle_f64x2(r0, r1, 0xDD);                      \
        __m512d _t2 = _mm512_shuffle_f64x2(r2, r3, 0x88);                      \
        __m512d _t3 = _mm512_shuffle_f64x2(r2, r3, 0xDD);                      \
        (o0) = _mm512_shuffle_f64x2(_t0, _t2, 0x88);                           \
        (o2) = _mm512_shuffle_f64x2(_t0, _t2, 0xDD);                           \
        (o1) = _mm512_shuffle_f64x2(_t1, _t3, 0x88);                           \
        (o3) = _mm512_shuffle_f64x2(_t1, _t3, 0xDD);                           \
    } while (0)

// Each row of the 4x4 micro-tile is one 512-bit register, multiplied by its
// blocked twiddle then transposed via the four-lane shuffles above.
static inline FFTZ_VOID fused_twiddle_transpose_fp64_avx512(
    const FFTZ_DOUBLE *in, FFTZ_DOUBLE *out, const FFTZ_DOUBLE *twiddle_ptr,
    FFTZ_INTP n1, FFTZ_INTP n2, FFTZ_INTP in_row_stride,
    FFTZ_INTP out_row_stride, FFTZ_INT32 conjugate)
{
    const FFTZ_INTP cache_block = FUSED_TWIDDLE_TRANSPOSE_FP64_CACHE_BLOCK;
    const FFTZ_INTP micro_tile = FUSED_TWIDDLE_TRANSPOSE_FP64_MICRO_TILE;

    for (FFTZ_INTP col_block = 0; col_block < n2; col_block += cache_block)
    {
        const FFTZ_INTP j_end =
            (col_block + cache_block < n2) ? col_block + cache_block : n2;

        for (FFTZ_INTP row_block = 0; row_block < n1; row_block += cache_block)
        {
            const FFTZ_INTP i_end =
                (row_block + cache_block < n1) ? row_block + cache_block : n1;

            for (FFTZ_INTP i = row_block; i < i_end; i += micro_tile)
            {
                for (FFTZ_INTP j = col_block; j < j_end; j += micro_tile)
                {
                    __m512d r0 = _mm512_loadu_pd(
                        &in[(i * in_row_stride + j) * DATA_STRIDE]);
                    __m512d r1 = _mm512_loadu_pd(
                        &in[((i + 1) * in_row_stride + j) * DATA_STRIDE]);
                    __m512d r2 = _mm512_loadu_pd(
                        &in[((i + 2) * in_row_stride + j) * DATA_STRIDE]);
                    __m512d r3 = _mm512_loadu_pd(
                        &in[((i + 3) * in_row_stride + j) * DATA_STRIDE]);

                    // twiddle micro-tile: 4 rows x 4 complex, contiguous
                    __m512d w0 = _mm512_loadu_pd(twiddle_ptr + 0);
                    __m512d w1 = _mm512_loadu_pd(twiddle_ptr + 8);
                    __m512d w2 = _mm512_loadu_pd(twiddle_ptr + 16);
                    __m512d w3 = _mm512_loadu_pd(twiddle_ptr + 24);
                    twiddle_ptr += micro_tile * micro_tile * 2;

                    if (conjugate)
                    {
                        r0 = CMUL_CONJ_512_FP64(r0, w0);
                        r1 = CMUL_CONJ_512_FP64(r1, w1);
                        r2 = CMUL_CONJ_512_FP64(r2, w2);
                        r3 = CMUL_CONJ_512_FP64(r3, w3);
                    }
                    else
                    {
                        r0 = CMUL_512_FP64(r0, w0);
                        r1 = CMUL_512_FP64(r1, w1);
                        r2 = CMUL_512_FP64(r2, w2);
                        r3 = CMUL_512_FP64(r3, w3);
                    }

                    __m512d o0, o1, o2, o3;
                    TRANSPOSE_4x4_FP64(r0, r1, r2, r3, o0, o1, o2, o3);
                    _mm512_storeu_pd(
                        &out[(j * out_row_stride + i) * DATA_STRIDE], o0);
                    _mm512_storeu_pd(
                        &out[((j + 1) * out_row_stride + i) * DATA_STRIDE], o1);
                    _mm512_storeu_pd(
                        &out[((j + 2) * out_row_stride + i) * DATA_STRIDE], o2);
                    _mm512_storeu_pd(
                        &out[((j + 3) * out_row_stride + i) * DATA_STRIDE], o3);
                }
            }
        }
    }
}

static FFTZ_VOID fused_twiddle_transpose_fp64_avx512_fwd(
    FFTZ_VOID *in_ptr, FFTZ_VOID *out_ptr, FFTZ_VOID *twiddles_ptr,
    FFTZ_INTP n1, FFTZ_INTP n2, FFTZ_INTP in_row_stride,
    FFTZ_INTP out_row_stride)
{
    AOCLFFTZ_LOG(DEBUG, global_logger_mode, "Enter");
    fused_twiddle_transpose_fp64_avx512((const FFTZ_DOUBLE *)in_ptr,
                                  (FFTZ_DOUBLE *)out_ptr,
                                  (const FFTZ_DOUBLE *)twiddles_ptr, n1, n2,
                                  in_row_stride, out_row_stride, 0);
}

static FFTZ_VOID fused_twiddle_transpose_fp64_avx512_bwd(
    FFTZ_VOID *in_ptr, FFTZ_VOID *out_ptr, FFTZ_VOID *twiddles_ptr,
    FFTZ_INTP n1, FFTZ_INTP n2, FFTZ_INTP in_row_stride,
    FFTZ_INTP out_row_stride)
{
    AOCLFFTZ_LOG(DEBUG, global_logger_mode, "Enter");
    fused_twiddle_transpose_fp64_avx512((const FFTZ_DOUBLE *)in_ptr,
                                  (FFTZ_DOUBLE *)out_ptr,
                                  (const FFTZ_DOUBLE *)twiddles_ptr, n1, n2,
                                  in_row_stride, out_row_stride, 1);
}

#undef TRANSPOSE_4x4_FP64

// One stage of the 8x8 (64-bit element) transpose butterfly: row0 takes both
// rows' low halves interleaved, row1 their high halves, with no unpack step.
#define TRANSPOSE_8x8_FP32_INTERLEAVE(row0, row1, idxlo, idxhi)                \
    do                                                                         \
    {                                                                          \
        __m512d _t = _mm512_permutex2var_pd(row0, idxlo, row1);                \
        (row1) = _mm512_permutex2var_pd(row0, idxhi, row1);                    \
        (row0) = _t;                                                           \
    } while (0)

// In-register transpose of an 8x8 tile of 64-bit elements (complex floats), in
// place on the eight rows[].
#define TRANSPOSE_8x8_FP32(rows, idxlo, idxhi)                                 \
    do                                                                         \
    {                                                                          \
        TRANSPOSE_8x8_FP32_INTERLEAVE((rows)[0], (rows)[4], idxlo, idxhi);     \
        TRANSPOSE_8x8_FP32_INTERLEAVE((rows)[1], (rows)[5], idxlo, idxhi);     \
        TRANSPOSE_8x8_FP32_INTERLEAVE((rows)[2], (rows)[6], idxlo, idxhi);     \
        TRANSPOSE_8x8_FP32_INTERLEAVE((rows)[3], (rows)[7], idxlo, idxhi);     \
        TRANSPOSE_8x8_FP32_INTERLEAVE((rows)[0], (rows)[2], idxlo, idxhi);     \
        TRANSPOSE_8x8_FP32_INTERLEAVE((rows)[1], (rows)[3], idxlo, idxhi);     \
        TRANSPOSE_8x8_FP32_INTERLEAVE((rows)[4], (rows)[6], idxlo, idxhi);     \
        TRANSPOSE_8x8_FP32_INTERLEAVE((rows)[5], (rows)[7], idxlo, idxhi);     \
        TRANSPOSE_8x8_FP32_INTERLEAVE((rows)[0], (rows)[1], idxlo, idxhi);     \
        TRANSPOSE_8x8_FP32_INTERLEAVE((rows)[2], (rows)[3], idxlo, idxhi);     \
        TRANSPOSE_8x8_FP32_INTERLEAVE((rows)[4], (rows)[5], idxlo, idxhi);     \
        TRANSPOSE_8x8_FP32_INTERLEAVE((rows)[6], (rows)[7], idxlo, idxhi);     \
    } while (0)

// 8x8 transpose treats each complex float as one 64-bit element (__m512d); the
// complex multiply runs in the float domain (__m512) via free cast intrinsics.
static inline FFTZ_VOID fused_twiddle_transpose_fp32_avx512(
    const FFTZ_FLOAT *in, FFTZ_FLOAT *out, const FFTZ_FLOAT *twiddle_ptr,
    FFTZ_INTP n1, FFTZ_INTP n2, FFTZ_INTP in_row_stride,
    FFTZ_INTP out_row_stride, FFTZ_INT32 conjugate)
{
    const FFTZ_INTP cache_block = FUSED_TWIDDLE_TRANSPOSE_FP32_CACHE_BLOCK;
    const FFTZ_INTP micro_tile = FUSED_TWIDDLE_TRANSPOSE_FP32_MICRO_TILE;
    const __m512i idxlo =
        _mm512_setr_epi64(0x0, 0x8, 0x1, 0x9, 0x2, 0xA, 0x3, 0xB);
    const __m512i idxhi =
        _mm512_setr_epi64(0x4, 0xC, 0x5, 0xD, 0x6, 0xE, 0x7, 0xF);

    for (FFTZ_INTP col_block = 0; col_block < n2; col_block += cache_block)
    {
        const FFTZ_INTP j_end =
            (col_block + cache_block < n2) ? col_block + cache_block : n2;

        for (FFTZ_INTP row_block = 0; row_block < n1; row_block += cache_block)
        {
            const FFTZ_INTP i_end =
                (row_block + cache_block < n1) ? row_block + cache_block : n1;

            for (FFTZ_INTP i = row_block; i < i_end; i += micro_tile)
            {
                for (FFTZ_INTP j = col_block; j < j_end; j += micro_tile)
                {
                    __m512d rows[8];
                    for (FFTZ_INTP r = 0; r < micro_tile; r++)
                    {
                        __m512 vr = _mm512_loadu_ps(
                            &in[((i + r) * in_row_stride + j) * DATA_STRIDE]);
                        __m512 vw =
                            _mm512_loadu_ps(twiddle_ptr + r * micro_tile * 2);
                        __m512 prod = conjugate
                                          ? CMUL_CONJ_512_FP32(vr, vw)
                                          : CMUL_512_FP32(vr, vw);
                        rows[r] = _mm512_castps_pd(prod);
                    }
                    twiddle_ptr += micro_tile * micro_tile * 2;

                    TRANSPOSE_8x8_FP32(rows, idxlo, idxhi);
                    for (FFTZ_INTP r = 0; r < micro_tile; r++)
                    {
                        _mm512_storeu_ps(
                            &out[((j + r) * out_row_stride + i) * DATA_STRIDE],
                            _mm512_castpd_ps(rows[r]));
                    }
                }
            }
        }
    }
}

static FFTZ_VOID fused_twiddle_transpose_fp32_avx512_fwd(
    FFTZ_VOID *in_ptr, FFTZ_VOID *out_ptr, FFTZ_VOID *twiddles_ptr,
    FFTZ_INTP n1, FFTZ_INTP n2, FFTZ_INTP in_row_stride,
    FFTZ_INTP out_row_stride)
{
    AOCLFFTZ_LOG(DEBUG, global_logger_mode, "Enter");
    fused_twiddle_transpose_fp32_avx512((const FFTZ_FLOAT *)in_ptr,
                                  (FFTZ_FLOAT *)out_ptr,
                                  (const FFTZ_FLOAT *)twiddles_ptr, n1, n2,
                                  in_row_stride, out_row_stride, 0);
}

static FFTZ_VOID fused_twiddle_transpose_fp32_avx512_bwd(
    FFTZ_VOID *in_ptr, FFTZ_VOID *out_ptr, FFTZ_VOID *twiddles_ptr,
    FFTZ_INTP n1, FFTZ_INTP n2, FFTZ_INTP in_row_stride,
    FFTZ_INTP out_row_stride)
{
    AOCLFFTZ_LOG(DEBUG, global_logger_mode, "Enter");
    fused_twiddle_transpose_fp32_avx512((const FFTZ_FLOAT *)in_ptr,
                                  (FFTZ_FLOAT *)out_ptr,
                                  (const FFTZ_FLOAT *)twiddles_ptr, n1, n2,
                                  in_row_stride, out_row_stride, 1);
}

#undef TRANSPOSE_8x8_FP32
#undef TRANSPOSE_8x8_FP32_INTERLEAVE

fused_twiddle_transpose_
register_fused_twiddle_transpose_avx512(FFTZ_UINT8 precision,
                                        FFTZ_UINT8 direction)
{
    if (direction == FORWARD_FFT_DIR)
    {
        if (precision == DT_FLOAT)
        {
            return fused_twiddle_transpose_fp32_avx512_fwd;
        }
        else if (precision == DT_DOUBLE)
        {
            return fused_twiddle_transpose_fp64_avx512_fwd;
        }
        return NULL;
    }
    else
    {
        if (precision == DT_FLOAT)
        {
            return fused_twiddle_transpose_fp32_avx512_bwd;
        }
        else if (precision == DT_DOUBLE)
        {
            return fused_twiddle_transpose_fp64_avx512_bwd;
        }
        return NULL;
    }
}

