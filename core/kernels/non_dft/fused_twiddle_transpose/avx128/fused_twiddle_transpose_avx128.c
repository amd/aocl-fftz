// Copyright Advanced Micro Devices, Inc.
// SPDX-License-Identifier: BSD-3-Clause

/** @file fused_twiddle_transpose_avx128.c
 *
 *  @brief Fused four-step inter-step twiddle + out-of-place transpose, AVX-128.
 *
 *  AVX-128 counterpart to fused_twiddle_transpose_avx{256,512}.c: multiplies
 *  each source micro-tile by its twiddles and writes the transpose in one pass,
 *  avoiding the standalone twiddle pass's write-back / re-read of the n1 x n2
 *  matrix. Micro-tile edges match the wider variants (fp64 4x4, fp32 8x8) so a
 *  single per-precision blocked twiddle layout (see
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

// Each complex double is one 128-bit lane, so the transpose is a direct strided
// store with the multiply fused on the loaded lane.
static inline FFTZ_VOID fused_twiddle_transpose_fp64_avx128(
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
                    for (FFTZ_INTP r = 0; r < micro_tile; r++)
                    {
                        for (FFTZ_INTP c = 0; c < micro_tile; c++)
                        {
                            __m128d va = _mm_loadu_pd(
                                &in[((i + r) * in_row_stride + (j + c))
                                    * DATA_STRIDE]);
                            __m128d vb = _mm_loadu_pd(twiddle_ptr);
                            twiddle_ptr += 2;

                            __m128d prod = conjugate
                                               ? CMUL_CONJ_128_FP64(va, vb)
                                               : CMUL_128_FP64(va, vb);
                            _mm_storeu_pd(
                                &out[((j + c) * out_row_stride + (i + r))
                                     * DATA_STRIDE],
                                prod);
                        }
                    }
                }
            }
        }
    }
}

static FFTZ_VOID fused_twiddle_transpose_fp64_avx128_fwd(
    FFTZ_VOID *in_ptr, FFTZ_VOID *out_ptr, FFTZ_VOID *twiddles_ptr,
    FFTZ_INTP n1, FFTZ_INTP n2, FFTZ_INTP in_row_stride,
    FFTZ_INTP out_row_stride)
{
    AOCLFFTZ_LOG(DEBUG, global_logger_mode, "Enter");
    fused_twiddle_transpose_fp64_avx128((const FFTZ_DOUBLE *)in_ptr,
                                  (FFTZ_DOUBLE *)out_ptr,
                                  (const FFTZ_DOUBLE *)twiddles_ptr, n1, n2,
                                  in_row_stride, out_row_stride, 0);
}

static FFTZ_VOID fused_twiddle_transpose_fp64_avx128_bwd(
    FFTZ_VOID *in_ptr, FFTZ_VOID *out_ptr, FFTZ_VOID *twiddles_ptr,
    FFTZ_INTP n1, FFTZ_INTP n2, FFTZ_INTP in_row_stride,
    FFTZ_INTP out_row_stride)
{
    AOCLFFTZ_LOG(DEBUG, global_logger_mode, "Enter");
    fused_twiddle_transpose_fp64_avx128((const FFTZ_DOUBLE *)in_ptr,
                                  (FFTZ_DOUBLE *)out_ptr,
                                  (const FFTZ_DOUBLE *)twiddles_ptr, n1, n2,
                                  in_row_stride, out_row_stride, 1);
}

// fp32 (8x8 micro-tile; cache-block edge from fused_twiddle_transpose.h)

// Two complex floats fill one 128-bit lane. The 8x8 micro-tile is walked a row
// pair at a time: both rows are multiplied into 8 registers and transposed with
// 2x2 movelh/movehl moves straight out of them. A staging tile would have to be
// addressed through loadu/storeu, which blocks the compiler from promoting it
// to registers and costs a full store/reload of the tile.
static inline FFTZ_VOID fused_twiddle_transpose_fp32_avx128(
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
                    for (FFTZ_INTP r = 0; r < micro_tile; r += 2)
                    {
                        // one row pair: 2 x (micro_tile / 2) 128-bit lanes
                        __m128 p0[FUSED_TWIDDLE_TRANSPOSE_FP32_MICRO_TILE / 2];
                        __m128 p1[FUSED_TWIDDLE_TRANSPOSE_FP32_MICRO_TILE / 2];

                        // twiddles are consumed row-major, so row r's lanes
                        // come before row r+1's
                        for (FFTZ_INTP cc = 0; cc < micro_tile; cc += 2)
                        {
                            __m128 va = _mm_loadu_ps(
                                &in[((i + r) * in_row_stride + (j + cc))
                                    * DATA_STRIDE]);
                            __m128 vb = _mm_loadu_ps(twiddle_ptr);
                            twiddle_ptr += 4;

                            p0[cc >> 1] = conjugate
                                              ? CMUL_CONJ_128_FP32(va, vb)
                                              : CMUL_128_FP32(va, vb);
                        }
                        for (FFTZ_INTP cc = 0; cc < micro_tile; cc += 2)
                        {
                            __m128 va = _mm_loadu_ps(
                                &in[((i + r + 1) * in_row_stride + (j + cc))
                                    * DATA_STRIDE]);
                            __m128 vb = _mm_loadu_ps(twiddle_ptr);
                            twiddle_ptr += 4;

                            p1[cc >> 1] = conjugate
                                              ? CMUL_CONJ_128_FP32(va, vb)
                                              : CMUL_128_FP32(va, vb);
                        }

                        // transpose the row pair via 2x2 movelh/movehl
                        for (FFTZ_INTP c = 0; c < micro_tile; c += 2)
                        {
                            __m128 t0 = p0[c >> 1];
                            __m128 t1 = p1[c >> 1];
                            _mm_storeu_ps(
                                &out[((j + c) * out_row_stride + (i + r))
                                     * DATA_STRIDE],
                                _mm_movelh_ps(t0, t1));
                            _mm_storeu_ps(
                                &out[((j + c + 1) * out_row_stride + (i + r))
                                     * DATA_STRIDE],
                                _mm_movehl_ps(t1, t0));
                        }
                    }
                }
            }
        }
    }
}

static FFTZ_VOID fused_twiddle_transpose_fp32_avx128_fwd(
    FFTZ_VOID *in_ptr, FFTZ_VOID *out_ptr, FFTZ_VOID *twiddles_ptr,
    FFTZ_INTP n1, FFTZ_INTP n2, FFTZ_INTP in_row_stride,
    FFTZ_INTP out_row_stride)
{
    AOCLFFTZ_LOG(DEBUG, global_logger_mode, "Enter");
    fused_twiddle_transpose_fp32_avx128((const FFTZ_FLOAT *)in_ptr,
                                  (FFTZ_FLOAT *)out_ptr,
                                  (const FFTZ_FLOAT *)twiddles_ptr, n1, n2,
                                  in_row_stride, out_row_stride, 0);
}

static FFTZ_VOID fused_twiddle_transpose_fp32_avx128_bwd(
    FFTZ_VOID *in_ptr, FFTZ_VOID *out_ptr, FFTZ_VOID *twiddles_ptr,
    FFTZ_INTP n1, FFTZ_INTP n2, FFTZ_INTP in_row_stride,
    FFTZ_INTP out_row_stride)
{
    AOCLFFTZ_LOG(DEBUG, global_logger_mode, "Enter");
    fused_twiddle_transpose_fp32_avx128((const FFTZ_FLOAT *)in_ptr,
                                  (FFTZ_FLOAT *)out_ptr,
                                  (const FFTZ_FLOAT *)twiddles_ptr, n1, n2,
                                  in_row_stride, out_row_stride, 1);
}

fused_twiddle_transpose_
register_fused_twiddle_transpose_avx128(FFTZ_UINT8 precision,
                                        FFTZ_UINT8 direction)
{
    if (direction == FORWARD_FFT_DIR)
    {
        if (precision == DT_FLOAT)
        {
            return fused_twiddle_transpose_fp32_avx128_fwd;
        }
        else if (precision == DT_DOUBLE)
        {
            return fused_twiddle_transpose_fp64_avx128_fwd;
        }
        return NULL;
    }
    else
    {
        if (precision == DT_FLOAT)
        {
            return fused_twiddle_transpose_fp32_avx128_bwd;
        }
        else if (precision == DT_DOUBLE)
        {
            return fused_twiddle_transpose_fp64_avx128_bwd;
        }
        return NULL;
    }
}

