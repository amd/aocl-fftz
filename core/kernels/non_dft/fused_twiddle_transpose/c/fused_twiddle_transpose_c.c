// Copyright Advanced Micro Devices, Inc.
// SPDX-License-Identifier: BSD-3-Clause

/** @file fused_twiddle_transpose_c.c
 *
 *  @brief Fused four-step inter-step twiddle + out-of-place transpose, scalar
 *         C.
 *
 *  Portable scalar baseline for the fused four-step step-2 (inter-step twiddle)
 *  + step-3 (out-of-place transpose) pass, used when no SIMD variant is
 *  available so the solver runs at every optimization level. Walk order and
 *  blocked twiddle-table consumption mirror the SIMD kernels (see
 *  fused_twiddle_transpose.h). `conjugate` selects the inverse path and is a
 *  compile-time constant so the fwd/bwd wrappers specialize the core.
 *
 *  @author Ashwin K. Godbole
 */

#include "core/kernels/kernel.h"
#include "core/kernels/non_dft/fused_twiddle_transpose/fused_twiddle_transpose.h"

static inline FFTZ_VOID fused_twiddle_transpose_fp64_c(
    aoclfftz_complex_d_t *in, aoclfftz_complex_d_t *out,
    const FFTZ_DOUBLE *twiddle_ptr, FFTZ_INTP n1, FFTZ_INTP n2,
    FFTZ_INTP in_row_stride, FFTZ_INTP out_row_stride, FFTZ_INT32 conjugate)
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
                            aoclfftz_complex_d_t a =
                                in[(i + r) * in_row_stride + (j + c)];
                            FFTZ_DOUBLE wr = twiddle_ptr[0];
                            FFTZ_DOUBLE wi = twiddle_ptr[1];
                            twiddle_ptr += 2;

                            aoclfftz_complex_d_t p;
                            if (conjugate)
                            {
                                p.real = a.real * wr + a.imag * wi;
                                p.imag = a.imag * wr - a.real * wi;
                            }
                            else
                            {
                                p.real = a.real * wr - a.imag * wi;
                                p.imag = a.real * wi + a.imag * wr;
                            }
                            out[(j + c) * out_row_stride + (i + r)] = p;
                        }
                    }
                }
            }
        }
    }
}

static FFTZ_VOID fused_twiddle_transpose_fp64_c_fwd(
    FFTZ_VOID *in_ptr, FFTZ_VOID *out_ptr, FFTZ_VOID *twiddles_ptr,
    FFTZ_INTP n1, FFTZ_INTP n2, FFTZ_INTP in_row_stride,
    FFTZ_INTP out_row_stride)
{
    AOCLFFTZ_LOG(DEBUG, global_logger_mode, "Enter");
    fused_twiddle_transpose_fp64_c((aoclfftz_complex_d_t *)in_ptr,
                             (aoclfftz_complex_d_t *)out_ptr,
                             (const FFTZ_DOUBLE *)twiddles_ptr, n1, n2,
                             in_row_stride, out_row_stride, 0);
}

static FFTZ_VOID fused_twiddle_transpose_fp64_c_bwd(
    FFTZ_VOID *in_ptr, FFTZ_VOID *out_ptr, FFTZ_VOID *twiddles_ptr,
    FFTZ_INTP n1, FFTZ_INTP n2, FFTZ_INTP in_row_stride,
    FFTZ_INTP out_row_stride)
{
    AOCLFFTZ_LOG(DEBUG, global_logger_mode, "Enter");
    fused_twiddle_transpose_fp64_c((aoclfftz_complex_d_t *)in_ptr,
                             (aoclfftz_complex_d_t *)out_ptr,
                             (const FFTZ_DOUBLE *)twiddles_ptr, n1, n2,
                             in_row_stride, out_row_stride, 1);
}

// fp32 (complex float)

static inline FFTZ_VOID fused_twiddle_transpose_fp32_c(
    aoclfftz_complex_f_t *in, aoclfftz_complex_f_t *out,
    const FFTZ_FLOAT *twiddle_ptr, FFTZ_INTP n1, FFTZ_INTP n2,
    FFTZ_INTP in_row_stride, FFTZ_INTP out_row_stride, FFTZ_INT32 conjugate)
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
                    for (FFTZ_INTP r = 0; r < micro_tile; r++)
                    {
                        for (FFTZ_INTP c = 0; c < micro_tile; c++)
                        {
                            aoclfftz_complex_f_t a =
                                in[(i + r) * in_row_stride + (j + c)];
                            FFTZ_FLOAT wr = twiddle_ptr[0];
                            FFTZ_FLOAT wi = twiddle_ptr[1];
                            twiddle_ptr += 2;

                            aoclfftz_complex_f_t p;
                            if (conjugate)
                            {
                                p.real = a.real * wr + a.imag * wi;
                                p.imag = a.imag * wr - a.real * wi;
                            }
                            else
                            {
                                p.real = a.real * wr - a.imag * wi;
                                p.imag = a.real * wi + a.imag * wr;
                            }
                            out[(j + c) * out_row_stride + (i + r)] = p;
                        }
                    }
                }
            }
        }
    }
}

static FFTZ_VOID fused_twiddle_transpose_fp32_c_fwd(
    FFTZ_VOID *in_ptr, FFTZ_VOID *out_ptr, FFTZ_VOID *twiddles_ptr,
    FFTZ_INTP n1, FFTZ_INTP n2, FFTZ_INTP in_row_stride,
    FFTZ_INTP out_row_stride)
{
    AOCLFFTZ_LOG(DEBUG, global_logger_mode, "Enter");
    fused_twiddle_transpose_fp32_c((aoclfftz_complex_f_t *)in_ptr,
                             (aoclfftz_complex_f_t *)out_ptr,
                             (const FFTZ_FLOAT *)twiddles_ptr, n1, n2,
                             in_row_stride, out_row_stride, 0);
}

static FFTZ_VOID fused_twiddle_transpose_fp32_c_bwd(
    FFTZ_VOID *in_ptr, FFTZ_VOID *out_ptr, FFTZ_VOID *twiddles_ptr,
    FFTZ_INTP n1, FFTZ_INTP n2, FFTZ_INTP in_row_stride,
    FFTZ_INTP out_row_stride)
{
    AOCLFFTZ_LOG(DEBUG, global_logger_mode, "Enter");
    fused_twiddle_transpose_fp32_c((aoclfftz_complex_f_t *)in_ptr,
                             (aoclfftz_complex_f_t *)out_ptr,
                             (const FFTZ_FLOAT *)twiddles_ptr, n1, n2,
                             in_row_stride, out_row_stride, 1);
}

fused_twiddle_transpose_
register_fused_twiddle_transpose_c(FFTZ_UINT8 precision, FFTZ_UINT8 direction)
{
    if (direction == FORWARD_FFT_DIR)
    {
        if (precision == DT_FLOAT)
        {
            return fused_twiddle_transpose_fp32_c_fwd;
        }
        else if (precision == DT_DOUBLE)
        {
            return fused_twiddle_transpose_fp64_c_fwd;
        }
        return NULL;
    }
    else
    {
        if (precision == DT_FLOAT)
        {
            return fused_twiddle_transpose_fp32_c_bwd;
        }
        else if (precision == DT_DOUBLE)
        {
            return fused_twiddle_transpose_fp64_c_bwd;
        }
        return NULL;
    }
}

