// Copyright Advanced Micro Devices, Inc.
// SPDX-License-Identifier: BSD-3-Clause

/**
 * @file fftw_wrapper_edge_gtest.cpp
 * @brief Edge-case tests: degenerate dimensions (N=1 in multi-dim),
 *        4D transforms, very small multi-dim, plan_dft(rank=1) vs
 *        plan_dft_1d, prime-size C2C (should succeed), and
 *        R2C with N=2 (smallest non-trivial R2C).
 */

#include <cstring>
#include "fftw_wrapper_test_utils.h"

template<typename T>
class FftwWrapperEdgeTest : public FftwWrapperTestBase<T>
{
};
TYPED_TEST_SUITE(FftwWrapperEdgeTest, FftwTestTypes);

/* =====================================================================
 * 1. 2D with N0=1: degenerate first dimension.
 *    Effectively a 1D transform along N1.
 * ===================================================================== */

TYPED_TEST(FftwWrapperEdgeTest,
           PTEST_2D_DEGENERATE_N0_IS_1)
{
    using F = FftwTypes<TypeParam>;
    using complex_t [[maybe_unused]] = typename F::complex_t;
    const int N0 = 1, N1 = 8;
    const int total = N0 * N1;

    auto *in  = F::alloc_complex(total);
    auto *out = F::alloc_complex(total);
    auto *ref = F::alloc_complex(N1);

    this->init_complex(in, total);

    auto p = F::plan_dft_2d(
        N0, N1, in, out, FFTW_FORWARD, FFTW_ESTIMATE);
    ASSERT_NE(p, nullptr);
    F::execute(p);

    dft_reference_1d(in, ref, N1, FFTW_FORWARD);

    compare_complex_arrays(ref, out, N1, dft_tolerance<F>(total));

    F::destroy_plan(p);
    F::free_fn(in);
    F::free_fn(out);
    F::free_fn(ref);
}

/* =====================================================================
 * 2. 2D with N1=1: degenerate last dimension.
 *    Effectively a 1D transform along N0.
 * ===================================================================== */

TYPED_TEST(FftwWrapperEdgeTest,
           PTEST_2D_DEGENERATE_N1_IS_1)
{
    using F = FftwTypes<TypeParam>;
    using complex_t [[maybe_unused]] = typename F::complex_t;
    const int N0 = 8, N1 = 1;
    const int total = N0 * N1;

    auto *in  = F::alloc_complex(total);
    auto *out = F::alloc_complex(total);
    auto *ref = F::alloc_complex(N0);

    this->init_complex(in, total);

    auto p = F::plan_dft_2d(
        N0, N1, in, out, FFTW_FORWARD, FFTW_ESTIMATE);
    ASSERT_NE(p, nullptr);
    F::execute(p);

    dft_reference_1d(in, ref, N0, FFTW_FORWARD);

    compare_complex_arrays(ref, out, N0, dft_tolerance<F>(total));

    F::destroy_plan(p);
    F::free_fn(in);
    F::free_fn(out);
    F::free_fn(ref);
}

/* =====================================================================
 * 3. 3D with two degenerate dimensions: plan_dft_3d(1, 1, 8, ...).
 * ===================================================================== */

TYPED_TEST(FftwWrapperEdgeTest,
           PTEST_3D_TWO_DEGENERATE_DIMS)
{
    using F = FftwTypes<TypeParam>;
    using complex_t [[maybe_unused]] = typename F::complex_t;
    const int N0 = 1, N1 = 1, N2 = 8;
    const int total = N0 * N1 * N2;

    auto *in  = F::alloc_complex(total);
    auto *out = F::alloc_complex(total);
    auto *ref = F::alloc_complex(N2);

    this->init_complex(in, total);

    auto p = F::plan_dft_3d(
        N0, N1, N2, in, out,
        FFTW_FORWARD, FFTW_ESTIMATE);
    ASSERT_NE(p, nullptr);
    F::execute(p);

    dft_reference_1d(in, ref, N2, FFTW_FORWARD);

    compare_complex_arrays(ref, out, N2, dft_tolerance<F>(total));

    F::destroy_plan(p);
    F::free_fn(in);
    F::free_fn(out);
    F::free_fn(ref);
}

/* =====================================================================
 * 4. 4D C2C transform: plan_dft(rank=4, ...).
 *    Verifies translator handles rank > 3 correctly.
 *    Roundtrip test: FORWARD -> BACKWARD = x * N.
 * ===================================================================== */

TYPED_TEST(FftwWrapperEdgeTest,
           PTEST_4D_C2C_ROUNDTRIP)
{
    using F = FftwTypes<TypeParam>;
    using complex_t [[maybe_unused]] = typename F::complex_t;
    const int N0 = 2, N1 = 3, N2 = 4, N3 = 5;
    const int total = N0 * N1 * N2 * N3;

    auto *in   = F::alloc_complex(total);
    auto *freq = F::alloc_complex(total);
    auto *back = F::alloc_complex(total);

    this->init_complex(in, total);

    int n_arr[] = {N0, N1, N2, N3};
    auto pf = F::plan_dft(
        4, n_arr, in, freq,
        FFTW_FORWARD, FFTW_ESTIMATE);
    auto pb = F::plan_dft(
        4, n_arr, freq, back,
        FFTW_BACKWARD, FFTW_ESTIMATE);
    ASSERT_NE(pf, nullptr);
    ASSERT_NE(pb, nullptr);

    F::execute(pf);
    F::execute(pb);

    compare_complex_scaled(in, back, total, total, dft_tolerance<F>(total));

    F::destroy_plan(pf);
    F::destroy_plan(pb);
    F::free_fn(in);
    F::free_fn(freq);
    F::free_fn(back);
}

/* =====================================================================
 * 5. 4D C2C Parseval: verify energy conservation for 4D.
 * ===================================================================== */

TYPED_TEST(FftwWrapperEdgeTest,
           PTEST_4D_C2C_PARSEVAL)
{
    using F = FftwTypes<TypeParam>;
    using complex_t [[maybe_unused]] = typename F::complex_t;
    const int N0 = 2, N1 = 3, N2 = 4, N3 = 5;
    const int total = N0 * N1 * N2 * N3;

    auto *in  = F::alloc_complex(total);
    auto *out = F::alloc_complex(total);

    this->init_complex(in, total);

    double energy_time = energy_complex(in, total);

    int n_arr[] = {N0, N1, N2, N3};
    auto p = F::plan_dft(
        4, n_arr, in, out,
        FFTW_FORWARD, FFTW_ESTIMATE);
    ASSERT_NE(p, nullptr);
    F::execute(p);

    double energy_freq = energy_complex(out, total) / total;

    double tol = F::tolerance * total * energy_time;
    EXPECT_NEAR(energy_time, energy_freq, tol)
        << "4D Parseval violation";

    F::destroy_plan(p);
    F::free_fn(in);
    F::free_fn(out);
}

/* =====================================================================
 * 6. Prime-size C2C: N=11 C2C should SUCCEED (unlike R2C).
 *    Verify correctness via Parseval's theorem.
 * ===================================================================== */

TYPED_TEST(FftwWrapperEdgeTest,
           PTEST_PRIME_SIZE_C2C_PARSEVAL)
{
    using F = FftwTypes<TypeParam>;
    using complex_t [[maybe_unused]] = typename F::complex_t;
    const int N = 11;

    auto *in  = F::alloc_complex(N);
    auto *out = F::alloc_complex(N);

    this->init_complex(in, N);

    double energy_time = energy_complex(in, N);

    auto p = F::plan_dft_1d(
        N, in, out, FFTW_FORWARD, FFTW_ESTIMATE);
    ASSERT_NE(p, nullptr);
    F::execute(p);

    double energy_freq = energy_complex(out, N) / N;

    double tol = F::tolerance * N * energy_time;
    EXPECT_NEAR(energy_time, energy_freq, tol)
        << "Parseval for prime-size C2C";

    F::destroy_plan(p);
    F::free_fn(in);
    F::free_fn(out);
}

/* =====================================================================
 * 7. Prime-size C2C correctness via roundtrip (N=13).
 * ===================================================================== */

TYPED_TEST(FftwWrapperEdgeTest,
           PTEST_PRIME_SIZE_C2C_ROUNDTRIP)
{
    using F = FftwTypes<TypeParam>;
    using complex_t [[maybe_unused]] = typename F::complex_t;
    const int N = 13;

    auto *in   = F::alloc_complex(N);
    auto *freq = F::alloc_complex(N);
    auto *back = F::alloc_complex(N);

    this->init_complex(in, N);

    auto pf = F::plan_dft_1d(
        N, in, freq, FFTW_FORWARD, FFTW_ESTIMATE);
    auto pb = F::plan_dft_1d(
        N, freq, back, FFTW_BACKWARD, FFTW_ESTIMATE);
    ASSERT_NE(pf, nullptr);
    ASSERT_NE(pb, nullptr);

    F::execute(pf);
    F::execute(pb);

    compare_complex_scaled(in, back, N, N, dft_tolerance<F>(N));

    F::destroy_plan(pf);
    F::destroy_plan(pb);
    F::free_fn(in);
    F::free_fn(freq);
    F::free_fn(back);
}

/* =====================================================================
 * 8. 2D R2C with N0=1 (degenerate first dim): effectively 1D R2C.
 * ===================================================================== */

TYPED_TEST(FftwWrapperEdgeTest,
           PTEST_R2C_2D_DEGENERATE_N0_IS_1)
{
    using F = FftwTypes<TypeParam>;
    using real_t = typename F::real_t;
    using complex_t [[maybe_unused]] = typename F::complex_t;
    const int N0 = 1, N1 = 8;
    const int total = N0 * N1;
    const int Nc = N0 * (N1 / 2 + 1);

    auto *in    = F::alloc_real(total);
    auto *out2d = F::alloc_complex(Nc);
    auto *out1d = F::alloc_complex(Nc);

    this->init_real_iota(in, total);

    auto p2d = F::plan_dft_r2c_2d(
        N0, N1, in, out2d, FFTW_ESTIMATE);
    ASSERT_NE(p2d, nullptr);
    F::execute(p2d);

    this->init_real_iota(in, total);

    auto p1d = F::plan_dft_r2c_1d(
        N1, in, out1d, FFTW_ESTIMATE);
    ASSERT_NE(p1d, nullptr);
    F::execute(p1d);

    compare_complex_arrays(out1d, out2d, Nc, dft_tolerance<F>(total));

    F::destroy_plan(p2d);
    F::destroy_plan(p1d);
    F::free_fn(in);
    F::free_fn(out2d);
    F::free_fn(out1d);
}

/* =====================================================================
 * 9. N=2 R2C: smallest non-trivial R2C.
 *    Nc = 2/2+1 = 2. Verify exact output.
 *    For in=[a,b]: X[0]=a+b, X[1]=a-b.
 * ===================================================================== */

TYPED_TEST(FftwWrapperEdgeTest,
           PTEST_R2C_N2_EXACT)
{
    using F = FftwTypes<TypeParam>;
    using real_t = typename F::real_t;
    using complex_t [[maybe_unused]] = typename F::complex_t;
    const int N = 2;
    const int Nc = 2;

    auto *in  = F::alloc_real(N);
    auto *out = F::alloc_complex(Nc);

    in[0] = static_cast<real_t>(3.0);
    in[1] = static_cast<real_t>(7.0);

    auto p = F::plan_dft_r2c_1d(N, in, out, FFTW_ESTIMATE);
    ASSERT_NE(p, nullptr);
    F::execute(p);

    EXPECT_NEAR(out[0][0], 10.0, F::tolerance)
        << "X[0] real = a+b = 3+7 = 10";
    EXPECT_NEAR(out[0][1], 0.0, F::tolerance)
        << "X[0] imag = 0";
    EXPECT_NEAR(out[1][0], -4.0, F::tolerance)
        << "X[1] real = a-b = 3-7 = -4";
    EXPECT_NEAR(out[1][1], 0.0, F::tolerance)
        << "X[1] imag = 0";

    F::destroy_plan(p);
    F::free_fn(in);
    F::free_fn(out);
}

/* =====================================================================
 * 10. 2D all-ones R2C: flat real -> delta in freq.
 *     X[0] = N0*N1, rest should be 0.
 *     Very sensitive to stride layout.
 * ===================================================================== */

TYPED_TEST(FftwWrapperEdgeTest,
           PTEST_R2C_2D_FLAT_TO_DELTA)
{
    using F = FftwTypes<TypeParam>;
    using real_t [[maybe_unused]] = typename F::real_t;
    using complex_t [[maybe_unused]] = typename F::complex_t;
    const int N0 = 4, N1 = 6;
    const int total = N0 * N1;
    const int Nc = N0 * (N1 / 2 + 1);

    auto *in  = F::alloc_real(total);
    auto *out = F::alloc_complex(Nc);

    this->fill_real_constant(in, total, 1.0);

    auto p = F::plan_dft_r2c_2d(
        N0, N1, in, out, FFTW_ESTIMATE);
    ASSERT_NE(p, nullptr);
    F::execute(p);

    double tol = dft_tolerance<F>(total);
    EXPECT_NEAR(out[0][0], static_cast<double>(total), tol)
        << "DC real";
    EXPECT_NEAR(out[0][1], 0.0, tol)
        << "DC imag";
    compare_complex_zero(&out[1], Nc - 1, tol);

    F::destroy_plan(p);
    F::free_fn(in);
    F::free_fn(out);
}

/* =====================================================================
 * 11. plan_dft(rank=1) vs plan_dft_1d: identical code paths.
 *     Should produce bit-identical results.
 * ===================================================================== */

TYPED_TEST(FftwWrapperEdgeTest,
           PTEST_PLAN_DFT_RANK1_VS_PLAN_DFT_1D)
{
    using F = FftwTypes<TypeParam>;
    using complex_t [[maybe_unused]] = typename F::complex_t;
    const int N = 16;

    auto *in   = F::alloc_complex(N);
    auto *out1 = F::alloc_complex(N);
    auto *out2 = F::alloc_complex(N);

    this->init_complex(in, N);

    auto p1 = F::plan_dft_1d(
        N, in, out1, FFTW_FORWARD, FFTW_ESTIMATE);
    ASSERT_NE(p1, nullptr);

    int n_arr[] = {N};
    auto p2 = F::plan_dft(
        1, n_arr, in, out2,
        FFTW_FORWARD, FFTW_ESTIMATE);
    ASSERT_NE(p2, nullptr);

    F::execute(p1);
    F::execute(p2);

    compare_complex_arrays(out1, out2, N, dft_tolerance<F>(N));

    F::destroy_plan(p1);
    F::destroy_plan(p2);
    F::free_fn(in);
    F::free_fn(out1);
    F::free_fn(out2);
}
