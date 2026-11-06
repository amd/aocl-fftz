// Copyright Advanced Micro Devices, Inc.
// SPDX-License-Identifier: BSD-3-Clause

/** @file fused_twiddle_transpose_kernel_gtest_base.h
 *
 * @brief Base file for the fused four-step twiddle + transpose kernel tests.
 *
 * Parameterized fixture that drives the fused twiddle + transpose kernels
 * against an independent reference across precisions, directions, ISA variants,
 * geometries and (padded) row strides.
 *
 * @author Ashwin K. Godbole
 *
 */

#ifndef FUSED_TWIDDLE_TRANSPOSE_KERNEL_GTEST_BASE_H
#define FUSED_TWIDDLE_TRANSPOSE_KERNEL_GTEST_BASE_H

#include <cstdlib>
#include <tuple>
#include <vector>
#include <gtest/gtest.h>

#include "test/gtest/fused_twiddle_transpose/fused_twiddle_transpose_kernel_gtest_utils.h"

// Test parameter: ((n1, n2), in_row_pad, out_row_pad, direction, isa_index);
// in_row_stride = n2 + in_row_pad, out_row_stride = n1 + out_row_pad.
using FusedTwiddleTransposeParam =
    std::tuple<std::tuple<FFTZ_INTP, FFTZ_INTP>, FFTZ_INTP, FFTZ_INTP,
               FFTZ_INT32, FFTZ_INT32>;

template <class C>
class AoclfftzFusedTwiddleTransposeTestBase
    : public ::testing::TestWithParam<FusedTwiddleTransposeParam>
{
  protected:
    using R = typename tt_traits<C>::real_t;

    void test_kernel()
    {
        const FFTZ_INTP n1 = std::get<0>(std::get<0>(GetParam()));
        const FFTZ_INTP n2 = std::get<1>(std::get<0>(GetParam()));
        const FFTZ_INTP in_row_stride = n2 + std::get<1>(GetParam());
        const FFTZ_INTP out_row_stride = n1 + std::get<2>(GetParam());
        const FFTZ_INT32 direction = std::get<3>(GetParam());
        const FFTZ_INT32 isa = std::get<4>(GetParam());
        const bool conjugate = (direction != FORWARD_FFT_DIR);

        // Geometry must be a multiple of the micro-tile for the fused layout.
        ASSERT_EQ(n1 % tt_traits<C>::micro_tile, 0);
        ASSERT_EQ(n2 % tt_traits<C>::micro_tile, 0);

        fused_twiddle_transpose_ kernel =
            get_fused_twiddle_transpose_kernel<C>(isa, direction);
        ASSERT_NE(kernel, nullptr)
            << "registry returned NULL for isa=" << isa
            << " dir=" << direction;

        // in: n1 x n2 (stride in_row_stride); out: n2 x n1 (out_row_stride)
        const size_t in_size = static_cast<size_t>(n1) * in_row_stride;
        const size_t out_size = static_cast<size_t>(n2) * out_row_stride;

        // std::vector owns the buffers (zero-initialised like calloc) so an
        // early ASSERT/return releases them instead of leaking.
        std::vector<C> in(in_size);
        std::vector<C> out_ref(out_size);
        std::vector<C> out_ker(out_size);

        for (FFTZ_INTP i = 0; i < n1; i++)
        {
            for (FFTZ_INTP j = 0; j < n2; j++)
            {
                in[i * in_row_stride + j] = tt_input_value<C>(i, j);
            }
        }

        std::vector<R> twiddles = build_blocked_twiddle_table<C>(n1, n2);

        fused_twiddle_transpose_reference<C>(in.data(), out_ref.data(), n1, n2,
                                       in_row_stride, out_row_stride,
                                       conjugate);

        kernel((FFTZ_VOID *)in.data(), (FFTZ_VOID *)out_ker.data(),
               (FFTZ_VOID *)twiddles.data(), n1, n2, in_row_stride,
               out_row_stride);

        const double tol = tt_traits<C>::tolerance;
        for (FFTZ_INTP j = 0; j < n2; j++)
        {
            for (FFTZ_INTP i = 0; i < n1; i++)
            {
                const C e = out_ref[j * out_row_stride + i];
                const C g = out_ker[j * out_row_stride + i];
                EXPECT_NEAR(static_cast<double>(g.real),
                            static_cast<double>(e.real), tol)
                    << "real mismatch at out(" << j << ", " << i << ")";
                EXPECT_NEAR(static_cast<double>(g.imag),
                            static_cast<double>(e.imag), tol)
                    << "imag mismatch at out(" << j << ", " << i << ")";
            }
        }
    }
};

class AoclfftzFusedTwiddleTransposeKernelTestF32
    : public AoclfftzFusedTwiddleTransposeTestBase<aoclfftz_complex_f_t>
{
};

class AoclfftzFusedTwiddleTransposeKernelTestF64
    : public AoclfftzFusedTwiddleTransposeTestBase<aoclfftz_complex_d_t>
{
};

#endif // FUSED_TWIDDLE_TRANSPOSE_KERNEL_GTEST_BASE_H

