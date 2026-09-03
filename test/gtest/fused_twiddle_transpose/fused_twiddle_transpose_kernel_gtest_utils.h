// Copyright Advanced Micro Devices, Inc.
// SPDX-License-Identifier: BSD-3-Clause

/** @file fused_twiddle_transpose_kernel_gtest_utils.h
 *
 * @brief Utilities for the fused four-step twiddle + transpose kernel tests:
 * per-precision traits, the ISA-variant registry, a generator for the blocked
 * twiddle table (see fused_twiddle_transpose.h) and an independent reference.
 *
 * @author Ashwin K. Godbole
 *
 */

#ifndef FUSED_TWIDDLE_TRANSPOSE_KERNEL_GTEST_UTILS_H
#define FUSED_TWIDDLE_TRANSPOSE_KERNEL_GTEST_UTILS_H

#include <cmath>
#include <string>
#include <vector>
#include <gtest/gtest.h>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

#ifdef MULTI_THREADING
#include <omp.h>
#endif

extern "C"
{
#include "api/aoclfftz_internal.h"
#include "core/kernels/non_dft/fused_twiddle_transpose/fused_twiddle_transpose.h"
#include "test/gtest/aoclfftz_core_wrapper.h"
}

// Per-precision traits: element type, real scalar, precision enum, and tiling
// geometry (must match the kernels via fused_twiddle_transpose.h).
template <class C> struct tt_traits;

template <> struct tt_traits<aoclfftz_complex_f_t>
{
    using real_t = FFTZ_FLOAT;
    static constexpr FFTZ_UINT8 dt = DT_FLOAT;
    static constexpr FFTZ_INTP cache_block =
        FUSED_TWIDDLE_TRANSPOSE_FP32_CACHE_BLOCK;
    static constexpr FFTZ_INTP micro_tile =
        FUSED_TWIDDLE_TRANSPOSE_FP32_MICRO_TILE;
    // fp32: ~1e-3 tolerance (mirrors kernel_gtest_utils.h TOLERANCE_F)
    static constexpr double tolerance = 1e-3;
    static constexpr const char *name = "FP32";
};

template <> struct tt_traits<aoclfftz_complex_d_t>
{
    using real_t = FFTZ_DOUBLE;
    static constexpr FFTZ_UINT8 dt = DT_DOUBLE;
    static constexpr FFTZ_INTP cache_block =
        FUSED_TWIDDLE_TRANSPOSE_FP64_CACHE_BLOCK;
    static constexpr FFTZ_INTP micro_tile =
        FUSED_TWIDDLE_TRANSPOSE_FP64_MICRO_TILE;
    // fp64: ~1e-10 tolerance (mirrors kernel_gtest_utils.h TOLERANCE_D)
    static constexpr double tolerance = 1e-10;
    static constexpr const char *name = "FP64";
};

// ISA variants from register_fused_twiddle_transpose_kernel. Scalar (C) is
// always present; AVX variants are gated on the build macros.
enum tt_isa
{
    TT_ISA_C = 0,
#ifdef ENABLE_AVX128
    TT_ISA_AVX128,
#endif
#ifdef ENABLE_AVX256
    TT_ISA_AVX256,
#endif
#ifdef ENABLE_AVX512
    TT_ISA_AVX512,
#endif
};

static const std::vector<std::pair<FFTZ_INT32, std::string>> &tt_isa_variants()
{
    static const std::vector<std::pair<FFTZ_INT32, std::string>> variants = {
        {TT_ISA_C, "c"},
#ifdef ENABLE_AVX128
        {TT_ISA_AVX128, "avx128"},
#endif
#ifdef ENABLE_AVX256
        {TT_ISA_AVX256, "avx256"},
#endif
#ifdef ENABLE_AVX512
        {TT_ISA_AVX512, "avx512"},
#endif
    };
    return variants;
}

// Select the fused kernel for the given ISA index, precision and direction.
template <class C>
fused_twiddle_transpose_ get_fused_twiddle_transpose_kernel(FFTZ_INT32 isa,
                                                FFTZ_UINT8 direction)
{
    const FFTZ_UINT8 prec = tt_traits<C>::dt;
    switch (isa)
    {
    case TT_ISA_C:
        return register_fused_twiddle_transpose_c_wrapper(prec, direction);
#ifdef ENABLE_AVX128
    case TT_ISA_AVX128:
        return register_fused_twiddle_transpose_avx128_wrapper(prec, direction);
#endif
#ifdef ENABLE_AVX256
    case TT_ISA_AVX256:
        return register_fused_twiddle_transpose_avx256_wrapper(prec, direction);
#endif
#ifdef ENABLE_AVX512
    case TT_ISA_AVX512:
        return register_fused_twiddle_transpose_avx512_wrapper(prec, direction);
#endif
    default:
        return nullptr;
    }
}

// Per-element twiddle W_N^{i*j}, N = n1*n2 (forward). Computed in double so the
// SAME value feeds both the generated table and the reference.
static inline void tt_twiddle_value(FFTZ_INTP i, FFTZ_INTP j, FFTZ_INTP n1,
                                    FFTZ_INTP n2, double &wr, double &wi)
{
    const double N = static_cast<double>(n1) * static_cast<double>(n2);
    // reduce i*j modulo N to keep the angle well-conditioned for large indices
    const long long ij = static_cast<long long>(i) * static_cast<long long>(j);
    const double k = static_cast<double>(ij % static_cast<long long>(N));
    const double theta = -2.0 * M_PI * k / N;
    wr = std::cos(theta);
    wi = std::sin(theta);
}

// Bounded (|.| <= 1) input so results stay O(1) regardless of geometry, which
// keeps the absolute tolerances meaningful for both precisions.
template <class C>
static inline C tt_input_value(FFTZ_INTP i, FFTZ_INTP j)
{
    using R = typename tt_traits<C>::real_t;
    const double re = std::cos(0.1 * (i + 1) * (j + 2));
    const double im = std::sin(0.07 * (i + 2) + 0.03 * (j + 1));
    return C{static_cast<R>(re), static_cast<R>(im)};
}

// Build the blocked twiddle table in kernel-consumption order (see
// fused_twiddle_transpose.h): cache blocks of micro-tiles, (re, im) pairs.
template <class C>
std::vector<typename tt_traits<C>::real_t>
build_blocked_twiddle_table(FFTZ_INTP n1, FFTZ_INTP n2)
{
    using R = typename tt_traits<C>::real_t;
    const FFTZ_INTP cache_block = tt_traits<C>::cache_block;
    const FFTZ_INTP micro_tile = tt_traits<C>::micro_tile;

    std::vector<R> table;
    table.reserve(static_cast<size_t>(n1) * static_cast<size_t>(n2) * 2);

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
                            double wr, wi;
                            tt_twiddle_value(i + r, j + c, n1, n2, wr, wi);
                            table.push_back(static_cast<R>(wr));
                            table.push_back(static_cast<R>(wi));
                        }
                    }
                }
            }
        }
    }
    return table;
}

// Independent reference: per-element twiddle multiply + transpose, i.e.
// out[j][i] = in[i][j] * W(i,j) (forward) or * conj(W) (backward).
template <class C>
void fused_twiddle_transpose_reference(const C *in, C *out, FFTZ_INTP n1,
                                 FFTZ_INTP n2, FFTZ_INTP in_row_stride,
                                 FFTZ_INTP out_row_stride, bool conjugate)
{
    using R = typename tt_traits<C>::real_t;
    for (FFTZ_INTP i = 0; i < n1; i++)
    {
        for (FFTZ_INTP j = 0; j < n2; j++)
        {
            const C a = in[i * in_row_stride + j];
            double wr, wi;
            tt_twiddle_value(i, j, n1, n2, wr, wi);

            C p;
            if (conjugate)
            {
                p.real = static_cast<R>(a.real * wr + a.imag * wi);
                p.imag = static_cast<R>(a.imag * wr - a.real * wi);
            }
            else
            {
                p.real = static_cast<R>(a.real * wr - a.imag * wi);
                p.imag = static_cast<R>(a.real * wi + a.imag * wr);
            }
            out[j * out_row_stride + i] = p;
        }
    }
}

#endif // FUSED_TWIDDLE_TRANSPOSE_KERNEL_GTEST_UTILS_H

