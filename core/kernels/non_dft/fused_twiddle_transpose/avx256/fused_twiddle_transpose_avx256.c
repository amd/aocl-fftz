// Copyright Advanced Micro Devices, Inc.
// SPDX-License-Identifier: BSD-3-Clause

/** @file fused_twiddle_transpose_avx256.c
 *
 *  @brief Fused four-step inter-step twiddle + out-of-place transpose, AVX-256.
 *
 *  AVX-256 counterpart to fused_twiddle_transpose_avx512.c: multiplies each
 *  source micro-tile by its twiddles and transposes it in registers in a single
 *  pass, avoiding the standalone twiddle pass's write-back / re-read of the
 *  n1 x n2 matrix. Micro-tile edges match the AVX-512 variant (fp64 4x4, fp32
 *  8x8) so a single per-precision blocked twiddle layout (see
 *  fused_twiddle_transpose.h) serves every ISA. The fwd/bwd wrappers pass a
 *  literal `conjugate` to one static inline core so the compiler folds the
 *  branch away.
 *
 *  @author Ashwin K. Godbole
 */

#include "core/kernels/kernel.h"
#include "core/kernels/non_dft/fused_twiddle_transpose/fused_twiddle_transpose.h"
#include "core/kernels/simd_includes/simd_common.h"
#include "core/kernels/non_dft/fused_twiddle_transpose/fused_twiddle_transpose_cmul.h"
#include <immintrin.h>

// fp64 (4x4 micro-tile; cache-block edge from fused_twiddle_transpose.h)

// Each 4x4-tile source row spans two 256-bit lanes (cols 0-1, 2-3), multiplied
// before the register transpose (128-bit lane swaps emit full-cacheline rows).
static inline FFTZ_VOID fused_twiddle_transpose_fp64_avx256(
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
                    // each source row spans two 256-bit lanes (cols 0-1, 2-3)
                    __m256d r0 = _mm256_loadu_pd(
                        &in[(i * in_row_stride + j) * DATA_STRIDE]);
                    __m256d r0b = _mm256_loadu_pd(
                        &in[(i * in_row_stride + j + 2) * DATA_STRIDE]);
                    __m256d r1 = _mm256_loadu_pd(
                        &in[((i + 1) * in_row_stride + j) * DATA_STRIDE]);
                    __m256d r1b = _mm256_loadu_pd(
                        &in[((i + 1) * in_row_stride + j + 2) * DATA_STRIDE]);
                    __m256d r2 = _mm256_loadu_pd(
                        &in[((i + 2) * in_row_stride + j) * DATA_STRIDE]);
                    __m256d r2b = _mm256_loadu_pd(
                        &in[((i + 2) * in_row_stride + j + 2) * DATA_STRIDE]);
                    __m256d r3 = _mm256_loadu_pd(
                        &in[((i + 3) * in_row_stride + j) * DATA_STRIDE]);
                    __m256d r3b = _mm256_loadu_pd(
                        &in[((i + 3) * in_row_stride + j + 2) * DATA_STRIDE]);

                    // blocked twiddle: 4 rows x (2 + 2) complex
                    __m256d w0 = _mm256_loadu_pd(twiddle_ptr + 0);
                    __m256d w0b = _mm256_loadu_pd(twiddle_ptr + 4);
                    __m256d w1 = _mm256_loadu_pd(twiddle_ptr + 8);
                    __m256d w1b = _mm256_loadu_pd(twiddle_ptr + 12);
                    __m256d w2 = _mm256_loadu_pd(twiddle_ptr + 16);
                    __m256d w2b = _mm256_loadu_pd(twiddle_ptr + 20);
                    __m256d w3 = _mm256_loadu_pd(twiddle_ptr + 24);
                    __m256d w3b = _mm256_loadu_pd(twiddle_ptr + 28);
                    twiddle_ptr += micro_tile * micro_tile * 2;

                    if (conjugate)
                    {
                        r0 = CMUL_CONJ_256_FP64(r0, w0);
                        r0b = CMUL_CONJ_256_FP64(r0b, w0b);
                        r1 = CMUL_CONJ_256_FP64(r1, w1);
                        r1b = CMUL_CONJ_256_FP64(r1b, w1b);
                        r2 = CMUL_CONJ_256_FP64(r2, w2);
                        r2b = CMUL_CONJ_256_FP64(r2b, w2b);
                        r3 = CMUL_CONJ_256_FP64(r3, w3);
                        r3b = CMUL_CONJ_256_FP64(r3b, w3b);
                    }
                    else
                    {
                        r0 = CMUL_256_FP64(r0, w0);
                        r0b = CMUL_256_FP64(r0b, w0b);
                        r1 = CMUL_256_FP64(r1, w1);
                        r1b = CMUL_256_FP64(r1b, w1b);
                        r2 = CMUL_256_FP64(r2, w2);
                        r2b = CMUL_256_FP64(r2b, w2b);
                        r3 = CMUL_256_FP64(r3, w3);
                        r3b = CMUL_256_FP64(r3b, w3b);
                    }

                    // transpose (128-bit lane swaps); dest rows are full lines
                    _mm256_storeu_pd(
                        &out[(j * out_row_stride + i) * DATA_STRIDE],
                        _mm256_permute2f128_pd(r0, r1, 0x20));
                    _mm256_storeu_pd(
                        &out[(j * out_row_stride + i + 2) * DATA_STRIDE],
                        _mm256_permute2f128_pd(r2, r3, 0x20));
                    _mm256_storeu_pd(
                        &out[((j + 1) * out_row_stride + i) * DATA_STRIDE],
                        _mm256_permute2f128_pd(r0, r1, 0x31));
                    _mm256_storeu_pd(
                        &out[((j + 1) * out_row_stride + i + 2) * DATA_STRIDE],
                        _mm256_permute2f128_pd(r2, r3, 0x31));
                    _mm256_storeu_pd(
                        &out[((j + 2) * out_row_stride + i) * DATA_STRIDE],
                        _mm256_permute2f128_pd(r0b, r1b, 0x20));
                    _mm256_storeu_pd(
                        &out[((j + 2) * out_row_stride + i + 2) * DATA_STRIDE],
                        _mm256_permute2f128_pd(r2b, r3b, 0x20));
                    _mm256_storeu_pd(
                        &out[((j + 3) * out_row_stride + i) * DATA_STRIDE],
                        _mm256_permute2f128_pd(r0b, r1b, 0x31));
                    _mm256_storeu_pd(
                        &out[((j + 3) * out_row_stride + i + 2) * DATA_STRIDE],
                        _mm256_permute2f128_pd(r2b, r3b, 0x31));
                }
            }
        }
    }
}

static FFTZ_VOID fused_twiddle_transpose_fp64_avx256_fwd(
    FFTZ_VOID *in_ptr, FFTZ_VOID *out_ptr, FFTZ_VOID *twiddles_ptr,
    FFTZ_INTP n1, FFTZ_INTP n2, FFTZ_INTP in_row_stride,
    FFTZ_INTP out_row_stride)
{
    AOCLFFTZ_LOG(DEBUG, global_logger_mode, "Enter");
    fused_twiddle_transpose_fp64_avx256((const FFTZ_DOUBLE *)in_ptr,
                                  (FFTZ_DOUBLE *)out_ptr,
                                  (const FFTZ_DOUBLE *)twiddles_ptr, n1, n2,
                                  in_row_stride, out_row_stride, 0);
}

static FFTZ_VOID fused_twiddle_transpose_fp64_avx256_bwd(
    FFTZ_VOID *in_ptr, FFTZ_VOID *out_ptr, FFTZ_VOID *twiddles_ptr,
    FFTZ_INTP n1, FFTZ_INTP n2, FFTZ_INTP in_row_stride,
    FFTZ_INTP out_row_stride)
{
    AOCLFFTZ_LOG(DEBUG, global_logger_mode, "Enter");
    fused_twiddle_transpose_fp64_avx256((const FFTZ_DOUBLE *)in_ptr,
                                  (FFTZ_DOUBLE *)out_ptr,
                                  (const FFTZ_DOUBLE *)twiddles_ptr, n1, n2,
                                  in_row_stride, out_row_stride, 1);
}

// fp32 (8x8 micro-tile; cache-block edge from fused_twiddle_transpose.h)

// In-register transpose of a 4x4 tile of 64-bit elements (complex floats).
#define TRANSPOSE_4x4_FP32(r0, r1, r2, r3, o0, o1, o2, o3)                     \
    do                                                                         \
    {                                                                          \
        __m256d _t0 = _mm256_unpacklo_pd(r0, r1);                              \
        __m256d _t1 = _mm256_unpackhi_pd(r0, r1);                              \
        __m256d _t2 = _mm256_unpacklo_pd(r2, r3);                              \
        __m256d _t3 = _mm256_unpackhi_pd(r2, r3);                              \
        (o0) = _mm256_permute2f128_pd(_t0, _t2, 0x20);                         \
        (o1) = _mm256_permute2f128_pd(_t1, _t3, 0x20);                         \
        (o2) = _mm256_permute2f128_pd(_t0, _t2, 0x31);                         \
        (o3) = _mm256_permute2f128_pd(_t1, _t3, 0x31);                         \
    } while (0)

// The 8x8 tile holds two 256-bit lanes per row (cols 0-3, 4-7), multiplied then
// transposed as four 4x4 sub-tiles (each complex float is one 64-bit element).
static inline FFTZ_VOID fused_twiddle_transpose_fp32_avx256(
    const FFTZ_FLOAT *in, FFTZ_FLOAT *out, const FFTZ_FLOAT *twiddle_ptr,
    FFTZ_INTP n1, FFTZ_INTP n2, FFTZ_INTP in_row_stride,
    FFTZ_INTP out_row_stride, FFTZ_INT32 conjugate)
{
    const FFTZ_INTP cache_block = FUSED_TWIDDLE_TRANSPOSE_FP32_CACHE_BLOCK;
    const FFTZ_INTP micro_tile = FUSED_TWIDDLE_TRANSPOSE_FP32_MICRO_TILE;

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
                    // multiply the 8x8 tile; each row in a lo/hi 256-bit pair
                    __m256d lo[8], hi[8];
                    for (FFTZ_INTP r = 0; r < micro_tile; r++)
                    {
                        const FFTZ_INTP src =
                            ((i + r) * in_row_stride + j) * DATA_STRIDE;
                        __m256 vlo = _mm256_loadu_ps(&in[src]);
                        __m256 vhi =
                            _mm256_loadu_ps(&in[src + 4 * DATA_STRIDE]);
                        __m256 wlo = _mm256_loadu_ps(twiddle_ptr + r * 16);
                        __m256 whi = _mm256_loadu_ps(twiddle_ptr + r * 16 + 8);
                        __m256 plo = conjugate
                                         ? CMUL_CONJ_256_FP32(vlo, wlo)
                                         : CMUL_256_FP32(vlo, wlo);
                        __m256 phi = conjugate
                                         ? CMUL_CONJ_256_FP32(vhi, whi)
                                         : CMUL_256_FP32(vhi, whi);
                        lo[r] = _mm256_castps_pd(plo);
                        hi[r] = _mm256_castps_pd(phi);
                    }
                    twiddle_ptr += micro_tile * micro_tile * 2;

                    // transpose as four 4x4 sub-tiles of 64-bit elements
                    __m256d P0, P1, P2, P3, Q0, Q1, Q2, Q3;
                    __m256d R0, R1, R2, R3, S0, S1, S2, S3;
                    TRANSPOSE_4x4_FP32(lo[0], lo[1], lo[2], lo[3],
                                       P0, P1, P2, P3);
                    TRANSPOSE_4x4_FP32(lo[4], lo[5], lo[6], lo[7],
                                       Q0, Q1, Q2, Q3);
                    TRANSPOSE_4x4_FP32(hi[0], hi[1], hi[2], hi[3],
                                       R0, R1, R2, R3);
                    TRANSPOSE_4x4_FP32(hi[4], hi[5], hi[6], hi[7],
                                       S0, S1, S2, S3);

                    _mm256_storeu_ps(
                        &out[(j * out_row_stride + i) * DATA_STRIDE],
                        _mm256_castpd_ps(P0));
                    _mm256_storeu_ps(
                        &out[(j * out_row_stride + i + 4) * DATA_STRIDE],
                        _mm256_castpd_ps(Q0));
                    _mm256_storeu_ps(
                        &out[((j + 1) * out_row_stride + i) * DATA_STRIDE],
                        _mm256_castpd_ps(P1));
                    _mm256_storeu_ps(
                        &out[((j + 1) * out_row_stride + i + 4) * DATA_STRIDE],
                        _mm256_castpd_ps(Q1));
                    _mm256_storeu_ps(
                        &out[((j + 2) * out_row_stride + i) * DATA_STRIDE],
                        _mm256_castpd_ps(P2));
                    _mm256_storeu_ps(
                        &out[((j + 2) * out_row_stride + i + 4) * DATA_STRIDE],
                        _mm256_castpd_ps(Q2));
                    _mm256_storeu_ps(
                        &out[((j + 3) * out_row_stride + i) * DATA_STRIDE],
                        _mm256_castpd_ps(P3));
                    _mm256_storeu_ps(
                        &out[((j + 3) * out_row_stride + i + 4) * DATA_STRIDE],
                        _mm256_castpd_ps(Q3));
                    _mm256_storeu_ps(
                        &out[((j + 4) * out_row_stride + i) * DATA_STRIDE],
                        _mm256_castpd_ps(R0));
                    _mm256_storeu_ps(
                        &out[((j + 4) * out_row_stride + i + 4) * DATA_STRIDE],
                        _mm256_castpd_ps(S0));
                    _mm256_storeu_ps(
                        &out[((j + 5) * out_row_stride + i) * DATA_STRIDE],
                        _mm256_castpd_ps(R1));
                    _mm256_storeu_ps(
                        &out[((j + 5) * out_row_stride + i + 4) * DATA_STRIDE],
                        _mm256_castpd_ps(S1));
                    _mm256_storeu_ps(
                        &out[((j + 6) * out_row_stride + i) * DATA_STRIDE],
                        _mm256_castpd_ps(R2));
                    _mm256_storeu_ps(
                        &out[((j + 6) * out_row_stride + i + 4) * DATA_STRIDE],
                        _mm256_castpd_ps(S2));
                    _mm256_storeu_ps(
                        &out[((j + 7) * out_row_stride + i) * DATA_STRIDE],
                        _mm256_castpd_ps(R3));
                    _mm256_storeu_ps(
                        &out[((j + 7) * out_row_stride + i + 4) * DATA_STRIDE],
                        _mm256_castpd_ps(S3));
                }
            }
        }
    }
}

static FFTZ_VOID fused_twiddle_transpose_fp32_avx256_fwd(
    FFTZ_VOID *in_ptr, FFTZ_VOID *out_ptr, FFTZ_VOID *twiddles_ptr,
    FFTZ_INTP n1, FFTZ_INTP n2, FFTZ_INTP in_row_stride,
    FFTZ_INTP out_row_stride)
{
    AOCLFFTZ_LOG(DEBUG, global_logger_mode, "Enter");
    fused_twiddle_transpose_fp32_avx256((const FFTZ_FLOAT *)in_ptr,
                                  (FFTZ_FLOAT *)out_ptr,
                                  (const FFTZ_FLOAT *)twiddles_ptr, n1, n2,
                                  in_row_stride, out_row_stride, 0);
}

static FFTZ_VOID fused_twiddle_transpose_fp32_avx256_bwd(
    FFTZ_VOID *in_ptr, FFTZ_VOID *out_ptr, FFTZ_VOID *twiddles_ptr,
    FFTZ_INTP n1, FFTZ_INTP n2, FFTZ_INTP in_row_stride,
    FFTZ_INTP out_row_stride)
{
    AOCLFFTZ_LOG(DEBUG, global_logger_mode, "Enter");
    fused_twiddle_transpose_fp32_avx256((const FFTZ_FLOAT *)in_ptr,
                                  (FFTZ_FLOAT *)out_ptr,
                                  (const FFTZ_FLOAT *)twiddles_ptr, n1, n2,
                                  in_row_stride, out_row_stride, 1);
}

#undef TRANSPOSE_4x4_FP32

fused_twiddle_transpose_
register_fused_twiddle_transpose_avx256(FFTZ_UINT8 precision,
                                        FFTZ_UINT8 direction)
{
    if (direction == FORWARD_FFT_DIR)
    {
        if (precision == DT_FLOAT)
        {
            return fused_twiddle_transpose_fp32_avx256_fwd;
        }
        else if (precision == DT_DOUBLE)
        {
            return fused_twiddle_transpose_fp64_avx256_fwd;
        }
        return NULL;
    }
    else
    {
        if (precision == DT_FLOAT)
        {
            return fused_twiddle_transpose_fp32_avx256_bwd;
        }
        else if (precision == DT_DOUBLE)
        {
            return fused_twiddle_transpose_fp64_avx256_bwd;
        }
        return NULL;
    }
}

