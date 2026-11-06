// Copyright Advanced Micro Devices, Inc.
// SPDX-License-Identifier: BSD-3-Clause

/**
 * @file fftw_wrapper_invariant_gtest.cpp
 * @brief Mathematical-invariant tests: Parseval's theorem, linearity,
 *        delta/flat patterns, guru R2C cross-API, new-array execute
 *        for R2C/C2R, guru64 vs guru, prime sizes.
 *        These detect silent data corruption that roundtrip tests miss.
 */

#include <cstring>
#include "fftw_wrapper_test_utils.h"

template<typename T>
class FftwWrapperInvariantTest : public FftwWrapperTestBase<T>
{
};
TYPED_TEST_SUITE(FftwWrapperInvariantTest, FftwTestTypes);

/* =====================================================================
 * 1. Parseval's theorem for 2D C2C: sum|x|^2 = (1/N)*sum|X|^2.
 *    Catches any dimension/stride bug that scrambles data.
 * ===================================================================== */

TYPED_TEST(FftwWrapperInvariantTest,
           PTEST_PARSEVAL_2D_C2C)
{
    using F = FftwTypes<TypeParam>;
    using complex_t [[maybe_unused]] = typename F::complex_t;
    const int N0 = 5, N1 = 7;
    const int total = N0 * N1;

    auto *in  = F::alloc_complex(total);
    auto *out = F::alloc_complex(total);

    this->init_complex(in, total);

    double energy_time = energy_complex(in, total);

    auto p = F::plan_dft_2d(
        N0, N1, in, out, FFTW_FORWARD, FFTW_ESTIMATE);
    ASSERT_NE(p, nullptr);
    F::execute(p);

    double energy_freq = energy_complex(out, total) / total;

    double tol = F::tolerance * total * energy_time;
    EXPECT_NEAR(energy_time, energy_freq, tol)
        << "Parseval violation";

    /* Parseval energy is invariant under any bin permutation or axis
     * transpose, so it cannot by itself catch a dimension-swap bug.  Anchor
     * the spectrum to an independent reference to give the test real teeth. */
    auto *ref = F::alloc_complex(total);
    dft_reference_2d(in, ref, N0, N1, FFTW_FORWARD);
    compare_complex_arrays(ref, out, total, dft_tolerance<F>(total));

    F::destroy_plan(p);
    F::free_fn(in);
    F::free_fn(out);
    F::free_fn(ref);
}

/* =====================================================================
 * 2. Parseval's theorem for 3D C2C.
 * ===================================================================== */

TYPED_TEST(FftwWrapperInvariantTest,
           PTEST_PARSEVAL_3D_C2C)
{
    using F = FftwTypes<TypeParam>;
    using complex_t [[maybe_unused]] = typename F::complex_t;
    const int N0 = 3, N1 = 4, N2 = 5;
    const int total = N0 * N1 * N2;

    auto *in  = F::alloc_complex(total);
    auto *out = F::alloc_complex(total);

    this->init_complex(in, total);

    double energy_time = energy_complex(in, total);

    auto p = F::plan_dft_3d(
        N0, N1, N2, in, out,
        FFTW_FORWARD, FFTW_ESTIMATE);
    ASSERT_NE(p, nullptr);
    F::execute(p);

    double energy_freq = energy_complex(out, total) / total;

    double tol = F::tolerance * total * energy_time;
    EXPECT_NEAR(energy_time, energy_freq, tol)
        << "Parseval violation";

    /* Anchor to an independent reference (Parseval alone is transpose-blind). */
    auto *ref = F::alloc_complex(total);
    dft_reference_3d(in, ref, N0, N1, N2, FFTW_FORWARD);
    compare_complex_arrays(ref, out, total, dft_tolerance<F>(total));

    F::destroy_plan(p);
    F::free_fn(in);
    F::free_fn(out);
    F::free_fn(ref);
}

/* =====================================================================
 * 3. Delta input -> flat output (2D C2C).
 *    x = delta at (0,0); DFT should be all 1+0i.
 * ===================================================================== */

TYPED_TEST(FftwWrapperInvariantTest,
           PTEST_DELTA_TO_FLAT_2D)
{
    using F = FftwTypes<TypeParam>;
    using complex_t [[maybe_unused]] = typename F::complex_t;
    const int N0 = 4, N1 = 6;
    const int total = N0 * N1;

    auto *in  = F::alloc_complex(total);
    auto *out = F::alloc_complex(total);

    this->zero_complex(in, total);
    in[0][0] = 1.0;
    in[0][1] = 0.0;

    auto p = F::plan_dft_2d(
        N0, N1, in, out, FFTW_FORWARD, FFTW_ESTIMATE);
    ASSERT_NE(p, nullptr);
    F::execute(p);

    double tol = dft_tolerance<F>(total);
    compare_complex_constant(out, total, 1.0, 0.0, tol);

    F::destroy_plan(p);
    F::free_fn(in);
    F::free_fn(out);
}

/* =====================================================================
 * 4. Flat input -> delta output (2D C2C).
 *    x = all 1+0i; DFT should have X[0]=N, rest 0.
 * ===================================================================== */

TYPED_TEST(FftwWrapperInvariantTest,
           PTEST_FLAT_TO_DELTA_2D)
{
    using F = FftwTypes<TypeParam>;
    using complex_t [[maybe_unused]] = typename F::complex_t;
    const int N0 = 4, N1 = 6;
    const int total = N0 * N1;

    auto *in  = F::alloc_complex(total);
    auto *out = F::alloc_complex(total);

    this->fill_complex_constant(in, total, 1.0, 0.0);

    auto p = F::plan_dft_2d(
        N0, N1, in, out, FFTW_FORWARD, FFTW_ESTIMATE);
    ASSERT_NE(p, nullptr);
    F::execute(p);

    double tol = dft_tolerance<F>(total);
    EXPECT_NEAR(out[0][0], static_cast<double>(total), tol)
        << "DC real";
    EXPECT_NEAR(out[0][1], 0.0, tol)
        << "DC imag";
    compare_complex_zero(&out[1], total - 1, tol);

    F::destroy_plan(p);
    F::free_fn(in);
    F::free_fn(out);
}

/* =====================================================================
 * 5. Linearity for 2D R2C: R2C(a*x+b*y) = a*R2C(x) + b*R2C(y).
 *    Catches stride bugs that corrupt only certain elements.
 * ===================================================================== */

TYPED_TEST(FftwWrapperInvariantTest,
           PTEST_LINEARITY_2D_R2C)
{
    using F = FftwTypes<TypeParam>;
    using real_t = typename F::real_t;
    using complex_t [[maybe_unused]] = typename F::complex_t;
    const int N0 = 4, N1 = 6;
    const int total = N0 * N1;
    const int Nc = N0 * (N1 / 2 + 1);
    const double a = 2.5, b = -1.3;

    auto *x    = F::alloc_real(total);
    auto *y    = F::alloc_real(total);
    auto *z    = F::alloc_real(total);
    auto *Fx   = F::alloc_complex(Nc);
    auto *Fy   = F::alloc_complex(Nc);
    auto *Fz   = F::alloc_complex(Nc);
    auto *ref  = F::alloc_complex(Nc);

    this->init_real_iota(x, total);
    this->fill_affine_real(y, total, static_cast<real_t>(total),
                           static_cast<real_t>(-1));
    this->combine_linear_real(z, x, y, a, b, total);

    auto px = F::plan_dft_r2c_2d(
        N0, N1, x, Fx, FFTW_ESTIMATE);
    ASSERT_NE(px, nullptr);
    F::execute(px);

    this->init_real_iota(x, total);
    auto py = F::plan_dft_r2c_2d(
        N0, N1, y, Fy, FFTW_ESTIMATE);
    ASSERT_NE(py, nullptr);
    F::execute(py);

    auto pz = F::plan_dft_r2c_2d(
        N0, N1, z, Fz, FFTW_ESTIMATE);
    ASSERT_NE(pz, nullptr);
    F::execute(pz);

    double tol = F::tolerance * total * total;
    expect_complex_linear_combo(Fz, Fx, Fy, a, b, Nc, tol);

    const double reference_tol = dft_tolerance<F>(total);
    dft_reference_r2c_2d(x, ref, N0, N1);
    compare_complex_arrays(ref, Fx, Nc, reference_tol);
    dft_reference_r2c_2d(y, ref, N0, N1);
    compare_complex_arrays(ref, Fy, Nc, reference_tol);
    dft_reference_r2c_2d(z, ref, N0, N1);
    compare_complex_arrays(ref, Fz, Nc, reference_tol);

    F::destroy_plan(px);
    F::destroy_plan(py);
    F::destroy_plan(pz);
    F::free_fn(x);
    F::free_fn(y);
    F::free_fn(z);
    F::free_fn(Fx);
    F::free_fn(Fy);
    F::free_fn(Fz);
    F::free_fn(ref);
}

/* =====================================================================
 * 6. Delta R2C: real delta at (0,0) -> all R2C output = 1+0i.
 *    Very sensitive to dimension layout for 2D R2C.
 * ===================================================================== */

TYPED_TEST(FftwWrapperInvariantTest,
           PTEST_DELTA_R2C_2D)
{
    using F = FftwTypes<TypeParam>;
    using real_t = typename F::real_t;
    using complex_t [[maybe_unused]] = typename F::complex_t;
    const int N0 = 5, N1 = 7;
    const int total = N0 * N1;
    const int Nc = N0 * (N1 / 2 + 1);

    auto *in  = F::alloc_real(total);
    auto *out = F::alloc_complex(Nc);

    this->zero_real(in, total);
    in[0] = static_cast<real_t>(1.0);

    auto p = F::plan_dft_r2c_2d(
        N0, N1, in, out, FFTW_ESTIMATE);
    ASSERT_NE(p, nullptr);
    F::execute(p);

    double tol = dft_tolerance<F>(total);
    for (int i = 0; i < Nc; i++)
    {
        EXPECT_NEAR(out[i][0], 1.0, tol)
            << "real at " << i;
        EXPECT_NEAR(out[i][1], 0.0, tol)
            << "imag at " << i;
    }

    F::destroy_plan(p);
    F::free_fn(in);
    F::free_fn(out);
}

/* =====================================================================
 * 7. Guru R2C cross-API: plan_guru_dft_r2c vs plan_dft_r2c_2d
 *    for 2D with asymmetric dims.
 *    Guru uses get_guru_dv_desc; plan_dft_r2c_2d uses get_r2c_dv_desc.
 *    The guru strides must match FFTW row-major R2C conventions.
 * ===================================================================== */

TYPED_TEST(FftwWrapperInvariantTest,
           PTEST_GURU_R2C_VS_PLAN_R2C_2D)
{
    using F = FftwTypes<TypeParam>;
    using real_t = typename F::real_t;
    using complex_t [[maybe_unused]] = typename F::complex_t;
    using iodim_t = typename F::iodim_t;
    const int N0 = 4, N1 = 6;
    const int total = N0 * N1;
    const int Nc = N0 * (N1 / 2 + 1);

    auto *in    = F::alloc_real(total);
    auto *out1  = F::alloc_complex(Nc);
    auto *out2  = F::alloc_complex(Nc);

    this->init_real_iota(in, total);

    auto p1 = F::plan_dft_r2c_2d(
        N0, N1, in, out1, FFTW_ESTIMATE);
    ASSERT_NE(p1, nullptr);
    F::execute(p1);

    this->init_real_iota(in, total);

    iodim_t dims[2] = {
        {N0, N1, (N1 / 2 + 1)},
        {N1, 1, 1}
    };
    iodim_t howmany[1] = {{1, 1, 1}};

    auto p2 = F::plan_guru_dft_r2c(
        2, dims, 1, howmany, in, out2, FFTW_ESTIMATE);
    ASSERT_NE(p2, nullptr);
    F::execute(p2);

    double tol = dft_tolerance<F>(total);
    compare_complex_arrays(out1, out2, Nc, tol);

    /* Guru and plan share the same translator, so out1==out2 alone would not
     * catch a bug common to both.  Anchor one path to an independent R2C
     * reference. */
    auto *ref = F::alloc_complex(Nc);
    dft_reference_r2c_2d(in, ref, N0, N1);
    compare_complex_arrays(ref, out1, Nc, tol);

    F::destroy_plan(p1);
    F::destroy_plan(p2);
    F::free_fn(in);
    F::free_fn(out1);
    F::free_fn(out2);
    F::free_fn(ref);
}

/* =====================================================================
 * 8. Guru C2R cross-API: plan_guru_dft_c2r vs plan_dft_c2r_2d.
 *    Roundtrip: R2C -> copy freq -> C2R via guru vs C2R via plan.
 * ===================================================================== */

TYPED_TEST(FftwWrapperInvariantTest,
           PTEST_GURU_C2R_VS_PLAN_C2R_2D)
{
    using F = FftwTypes<TypeParam>;
    using real_t = typename F::real_t;
    using complex_t [[maybe_unused]] = typename F::complex_t;
    using iodim_t = typename F::iodim_t;
    const int N0 = 4, N1 = 6;
    const int total = N0 * N1;
    const int Nc = N0 * (N1 / 2 + 1);

    auto *real_in = F::alloc_real(total);
    auto *freq    = F::alloc_complex(Nc);
    auto *freq2   = F::alloc_complex(Nc);
    auto *out1    = F::alloc_real(total);
    auto *out2    = F::alloc_real(total);

    this->init_real_iota(real_in, total);

    auto pr = F::plan_dft_r2c_2d(
        N0, N1, real_in, freq, FFTW_ESTIMATE);
    ASSERT_NE(pr, nullptr);
    F::execute(pr);
    std::memcpy(freq2, freq, sizeof(complex_t) * Nc);

    auto p1 = F::plan_dft_c2r_2d(
        N0, N1, freq, out1, FFTW_ESTIMATE);
    ASSERT_NE(p1, nullptr);
    F::execute(p1);

    iodim_t dims[2] = {
        {N0, (N1 / 2 + 1), N1},
        {N1, 1, 1}
    };
    iodim_t howmany[1] = {{1, 1, 1}};
    auto p2 = F::plan_guru_dft_c2r(
        2, dims, 1, howmany, freq2, out2, FFTW_ESTIMATE);
    ASSERT_NE(p2, nullptr);
    F::execute(p2);

    double tol = dft_tolerance<F>(total);
    compare_real_arrays(out1, out2, total, tol);

    /* Anchor the C2R output to an independent reference.  A roundtrip anchor
     * (out == real_in*total) would be transpose-blind because forward and
     * inverse apply the same wrong shape and cancel; c2r-of-freq does not. */
    auto *ref = F::alloc_real(total);
    dft_reference_c2r_2d(freq2, ref, N0, N1);
    compare_real_arrays(ref, out1, total, tol);

    F::destroy_plan(pr);
    F::destroy_plan(p1);
    F::destroy_plan(p2);
    F::free_fn(real_in);
    F::free_fn(freq);
    F::free_fn(freq2);
    F::free_fn(out1);
    F::free_fn(out2);
    F::free_fn(ref);
}

/* =====================================================================
 * 9. New-array execute_dft: plan with buffers A/B, execute with C/D.
 *    C contains different data. Result should match DFT(C), not DFT(A).
 * ===================================================================== */

TYPED_TEST(FftwWrapperInvariantTest,
           PTEST_EXECUTE_DFT_NEW_ARRAY)
{
    using F = FftwTypes<TypeParam>;
    using complex_t [[maybe_unused]] = typename F::complex_t;
    const int N = 16;

    auto *plan_in  = F::alloc_complex(N);
    auto *plan_out = F::alloc_complex(N);
    auto *new_in   = F::alloc_complex(N);
    auto *new_out  = F::alloc_complex(N);
    auto *ref      = F::alloc_complex(N);

    this->init_complex(plan_in, N);

    this->init_complex_desc_ramp(new_in, N, 2);

    auto p = F::plan_dft_1d(
        N, plan_in, plan_out,
        FFTW_FORWARD, FFTW_ESTIMATE);
    ASSERT_NE(p, nullptr);

    F::execute_dft(p, new_in, new_out);

    dft_reference_1d(new_in, ref, N, FFTW_FORWARD);

    double tol = dft_tolerance<F>(N);
    compare_complex_arrays(ref, new_out, N, tol);

    F::destroy_plan(p);
    F::free_fn(plan_in);
    F::free_fn(plan_out);
    F::free_fn(new_in);
    F::free_fn(new_out);
    F::free_fn(ref);
}

/* =====================================================================
 * 10. New-array execute_dft_r2c: plan with A/B, execute with C/D.
 * ===================================================================== */

TYPED_TEST(FftwWrapperInvariantTest,
           PTEST_EXECUTE_DFT_R2C_NEW_ARRAY)
{
    using F = FftwTypes<TypeParam>;
    using real_t = typename F::real_t;
    using complex_t [[maybe_unused]] = typename F::complex_t;
    const int N = 16;
    const int Nc = N / 2 + 1;

    auto *plan_in  = F::alloc_real(N);
    auto *plan_out = F::alloc_complex(Nc);
    auto *new_in   = F::alloc_real(N);
    auto *new_out  = F::alloc_complex(Nc);
    auto *ref      = F::alloc_complex(Nc);

    this->init_real(plan_in, N);

    this->fill_affine_real(new_in, N, static_cast<real_t>(N),
                           static_cast<real_t>(-1));

    auto p = F::plan_dft_r2c_1d(
        N, plan_in, plan_out, FFTW_ESTIMATE);
    ASSERT_NE(p, nullptr);

    F::execute_dft_r2c(p, new_in, new_out);

    dft_reference_r2c_1d(new_in, ref, N);

    double tol = dft_tolerance<F>(N);
    compare_complex_arrays(ref, new_out, Nc, tol);

    F::destroy_plan(p);
    F::free_fn(plan_in);
    F::free_fn(plan_out);
    F::free_fn(new_in);
    F::free_fn(new_out);
    F::free_fn(ref);
}

/* =====================================================================
 * 11. New-array execute_dft_c2r: plan with A/B, execute with C/D.
 *     Full roundtrip: R2C(new_data) -> C2R(result) = new_data * N.
 * ===================================================================== */

TYPED_TEST(FftwWrapperInvariantTest,
           PTEST_EXECUTE_DFT_C2R_NEW_ARRAY)
{
    using F = FftwTypes<TypeParam>;
    using real_t = typename F::real_t;
    using complex_t [[maybe_unused]] = typename F::complex_t;
    const int N = 16;
    const int Nc = N / 2 + 1;

    auto *plan_r   = F::alloc_real(N);
    auto *plan_c   = F::alloc_complex(Nc);
    auto *new_r_in = F::alloc_real(N);
    auto *new_freq = F::alloc_complex(Nc);
    auto *new_c_in = F::alloc_complex(Nc);
    auto *new_r_out = F::alloc_real(N);

    this->init_real(plan_r, N);

    this->init_real_iota(new_r_in, N);

    auto pr = F::plan_dft_r2c_1d(
        N, plan_r, plan_c, FFTW_ESTIMATE);
    ASSERT_NE(pr, nullptr);

    F::execute_dft_r2c(pr, new_r_in, new_freq);

    std::memcpy(new_c_in, new_freq, sizeof(complex_t) * Nc);

    auto pc = F::plan_dft_c2r_1d(
        N, plan_c, plan_r, FFTW_ESTIMATE);
    ASSERT_NE(pc, nullptr);

    F::execute_dft_c2r(pc, new_c_in, new_r_out);

    double tol = dft_tolerance<F>(N);
    compare_real_scaled(new_r_in, new_r_out, N, N, tol);

    F::destroy_plan(pr);
    F::destroy_plan(pc);
    F::free_fn(plan_r);
    F::free_fn(plan_c);
    F::free_fn(new_r_in);
    F::free_fn(new_freq);
    F::free_fn(new_c_in);
    F::free_fn(new_r_out);
}

/* =====================================================================
 * 12. Guru64 vs Guru: 64-bit and 32-bit guru produce same result.
 *     Catches type-conversion or reversal bugs in guru_64 path.
 * ===================================================================== */

TYPED_TEST(FftwWrapperInvariantTest,
           PTEST_GURU64_VS_GURU_C2C)
{
    using F = FftwTypes<TypeParam>;
    using complex_t [[maybe_unused]] = typename F::complex_t;
    using iodim_t = typename F::iodim_t;
    using iodim64_t = typename F::iodim64_t;
    const int N = 16;

    auto *in   = F::alloc_complex(N);
    auto *out1 = F::alloc_complex(N);
    auto *out2 = F::alloc_complex(N);

    this->init_complex(in, N);

    iodim_t dims32[1] = {{N, 1, 1}};
    iodim_t howmany32[1] = {{1, 1, 1}};

    auto p1 = F::plan_guru_dft(
        1, dims32, 1, howmany32,
        in, out1, FFTW_FORWARD, FFTW_ESTIMATE);
    ASSERT_NE(p1, nullptr);

    iodim64_t dims64[1] = {{N, 1, 1}};
    iodim64_t howmany64[1] = {{1, 1, 1}};

    auto p2 = F::plan_guru64_dft(
        1, dims64, 1, howmany64,
        in, out2, FFTW_FORWARD, FFTW_ESTIMATE);
    ASSERT_NE(p2, nullptr);

    F::execute(p1);
    F::execute(p2);

    double tol = dft_tolerance<F>(N);
    compare_complex_arrays(out1, out2, N, tol);
    /* Anchor to an independent reference so a bug shared by both guru paths
     * (e.g. a common reversal/type-conversion error) is still caught. */
    auto *ref = F::alloc_complex(N);
    dft_reference_1d(in, ref, N, FFTW_FORWARD);
    compare_complex_arrays(ref, out1, N, tol);

    F::destroy_plan(p1);
    F::destroy_plan(p2);
    F::free_fn(in);
    F::free_fn(out1);
    F::free_fn(out2);
    F::free_fn(ref);
}

/* =====================================================================
 * 13. Guru64 2D: compare guru64 with plan_dft_2d for 2D C2C.
 *     Catches guru64 dimension-reversal bugs for multi-dim.
 * ===================================================================== */

TYPED_TEST(FftwWrapperInvariantTest,
           PTEST_GURU64_2D_VS_PLAN_DFT_2D)
{
    using F = FftwTypes<TypeParam>;
    using complex_t [[maybe_unused]] = typename F::complex_t;
    using iodim64_t = typename F::iodim64_t;
    const int N0 = 5, N1 = 7;
    const int total = N0 * N1;

    auto *in   = F::alloc_complex(total);
    auto *out1 = F::alloc_complex(total);
    auto *out2 = F::alloc_complex(total);

    this->init_complex(in, total);

    auto p1 = F::plan_dft_2d(
        N0, N1, in, out1, FFTW_FORWARD, FFTW_ESTIMATE);
    ASSERT_NE(p1, nullptr);

    iodim64_t dims[2] = {
        {N0, N1, N1},
        {N1, 1, 1}
    };
    iodim64_t howmany[1] = {{1, 1, 1}};

    auto p2 = F::plan_guru64_dft(
        2, dims, 1, howmany,
        in, out2, FFTW_FORWARD, FFTW_ESTIMATE);
    ASSERT_NE(p2, nullptr);

    F::execute(p1);
    F::execute(p2);

    double tol = dft_tolerance<F>(total);
    compare_complex_arrays(out1, out2, total, tol);
    /* Anchor to an independent 2D reference (both paths share the translator). */
    auto *ref = F::alloc_complex(total);
    dft_reference_2d(in, ref, N0, N1, FFTW_FORWARD);
    compare_complex_arrays(ref, out1, total, tol);
    F::free_fn(ref);

    F::destroy_plan(p1);
    F::destroy_plan(p2);
    F::free_fn(in);
    F::free_fn(out1);
    F::free_fn(out2);
}

/* =====================================================================
 * Prime-size 1D R2C roundtrip (N=7).
 * ===================================================================== */

TYPED_TEST(FftwWrapperInvariantTest,
           PTEST_PRIME_1D_R2C_ROUNDTRIP)
{
    using F = FftwTypes<TypeParam>;
    using real_t = typename F::real_t;
    using complex_t [[maybe_unused]] = typename F::complex_t;
    const int N = 7;
    const int Nc = N / 2 + 1;

    auto *in   = F::alloc_real(N);
    auto *freq = F::alloc_complex(Nc);
    auto *ref  = F::alloc_complex(Nc);
    auto *out  = F::alloc_real(N);

    this->init_real_iota(in, N);

    auto pr = F::plan_dft_r2c_1d(N, in, freq, FFTW_ESTIMATE);
    auto pc = F::plan_dft_c2r_1d(N, freq, out, FFTW_ESTIMATE);
    ASSERT_NE(pr, nullptr);
    ASSERT_NE(pc, nullptr);

    F::execute(pr);
    dft_reference_r2c_1d(in, ref, N);
    compare_complex_arrays(ref, freq, Nc, dft_tolerance<F>(N));
    F::execute(pc);

    double tol = dft_tolerance<F>(N);
    compare_real_scaled(in, out, N, N, tol);

    F::destroy_plan(pr);
    F::destroy_plan(pc);
    F::free_fn(in);
    F::free_fn(freq);
    F::free_fn(ref);
    F::free_fn(out);
}

/* =====================================================================
 * Prime-size 2D R2C roundtrip (N0=7, N1=5).
 * ===================================================================== */

TYPED_TEST(FftwWrapperInvariantTest,
           PTEST_PRIME_2D_R2C_ROUNDTRIP)
{
    using F = FftwTypes<TypeParam>;
    using real_t = typename F::real_t;
    using complex_t [[maybe_unused]] = typename F::complex_t;
    const int N0 = 7, N1 = 5;
    const int total = N0 * N1;
    const int Nc = N0 * (N1 / 2 + 1);

    auto *in   = F::alloc_real(total);
    auto *freq = F::alloc_complex(Nc);
    auto *ref  = F::alloc_complex(Nc);
    auto *out  = F::alloc_real(total);

    this->init_real_iota(in, total);

    auto pr = F::plan_dft_r2c_2d(
        N0, N1, in, freq, FFTW_ESTIMATE);
    auto pc = F::plan_dft_c2r_2d(
        N0, N1, freq, out, FFTW_ESTIMATE);
    ASSERT_NE(pr, nullptr);
    ASSERT_NE(pc, nullptr);

    F::execute(pr);
    dft_reference_r2c_2d(in, ref, N0, N1);
    compare_complex_arrays(ref, freq, Nc, dft_tolerance<F>(total));
    F::execute(pc);

    double tol = dft_tolerance<F>(total);
    compare_real_scaled(in, out, total, total, tol);

    F::destroy_plan(pr);
    F::destroy_plan(pc);
    F::free_fn(in);
    F::free_fn(freq);
    F::free_fn(ref);
    F::free_fn(out);
}

/* =====================================================================
 * 17. Guru64 R2C: compare guru64_dft_r2c with plan_dft_r2c_2d.
 *     64-bit guru R2C uses get_guru_64_dv_desc, a DIFFERENT code path.
 * ===================================================================== */

TYPED_TEST(FftwWrapperInvariantTest,
           PTEST_GURU64_R2C_VS_PLAN_R2C_2D)
{
    using F = FftwTypes<TypeParam>;
    using real_t = typename F::real_t;
    using complex_t [[maybe_unused]] = typename F::complex_t;
    using iodim64_t = typename F::iodim64_t;
    const int N0 = 4, N1 = 6;
    const int total = N0 * N1;
    const int Nc = N0 * (N1 / 2 + 1);

    auto *in    = F::alloc_real(total);
    auto *out1  = F::alloc_complex(Nc);
    auto *out2  = F::alloc_complex(Nc);

    this->init_real_iota(in, total);

    auto p1 = F::plan_dft_r2c_2d(
        N0, N1, in, out1, FFTW_ESTIMATE);
    ASSERT_NE(p1, nullptr);
    F::execute(p1);

    this->init_real_iota(in, total);

    iodim64_t dims[2] = {
        {N0, N1, (N1 / 2 + 1)},
        {N1, 1, 1}
    };
    iodim64_t howmany[1] = {{1, 1, 1}};

    auto p2 = F::plan_guru64_dft_r2c(
        2, dims, 1, howmany, in, out2, FFTW_ESTIMATE);
    ASSERT_NE(p2, nullptr);
    F::execute(p2);

    double tol = dft_tolerance<F>(total);
    compare_complex_arrays(out1, out2, Nc, tol);

    /* Anchor to an independent R2C reference (both paths share the translator). */
    auto *ref = F::alloc_complex(Nc);
    dft_reference_r2c_2d(in, ref, N0, N1);
    compare_complex_arrays(ref, out1, Nc, tol);

    F::destroy_plan(p1);
    F::destroy_plan(p2);
    F::free_fn(in);
    F::free_fn(out1);
    F::free_fn(out2);
    F::free_fn(ref);
}

/* =====================================================================
 * 18. Guru64 C2R: compare guru64_dft_c2r with plan_dft_c2r_2d.
 * ===================================================================== */

TYPED_TEST(FftwWrapperInvariantTest,
           PTEST_GURU64_C2R_VS_PLAN_C2R_2D)
{
    using F = FftwTypes<TypeParam>;
    using real_t = typename F::real_t;
    using complex_t [[maybe_unused]] = typename F::complex_t;
    using iodim64_t = typename F::iodim64_t;
    const int N0 = 4, N1 = 6;
    const int total = N0 * N1;
    const int Nc = N0 * (N1 / 2 + 1);

    auto *real_in = F::alloc_real(total);
    auto *freq    = F::alloc_complex(Nc);
    auto *freq2   = F::alloc_complex(Nc);
    auto *out1    = F::alloc_real(total);
    auto *out2    = F::alloc_real(total);

    this->init_real_iota(real_in, total);

    auto pr = F::plan_dft_r2c_2d(
        N0, N1, real_in, freq, FFTW_ESTIMATE);
    ASSERT_NE(pr, nullptr);
    F::execute(pr);
    std::memcpy(freq2, freq, sizeof(complex_t) * Nc);

    auto p1 = F::plan_dft_c2r_2d(
        N0, N1, freq, out1, FFTW_ESTIMATE);
    ASSERT_NE(p1, nullptr);
    F::execute(p1);

    iodim64_t dims[2] = {
        {N0, (N1 / 2 + 1), N1},
        {N1, 1, 1}
    };
    iodim64_t howmany[1] = {{1, 1, 1}};
    auto p2 = F::plan_guru64_dft_c2r(
        2, dims, 1, howmany, freq2, out2, FFTW_ESTIMATE);
    ASSERT_NE(p2, nullptr);
    F::execute(p2);

    double tol = dft_tolerance<F>(total);
    compare_real_arrays(out1, out2, total, tol);

    /* Anchor the C2R output to an independent reference. */
    auto *ref = F::alloc_real(total);
    dft_reference_c2r_2d(freq2, ref, N0, N1);
    compare_real_arrays(ref, out1, total, tol);

    F::destroy_plan(pr);
    F::destroy_plan(p1);
    F::destroy_plan(p2);
    F::free_fn(real_in);
    F::free_fn(freq);
    F::free_fn(freq2);
    F::free_fn(out1);
    F::free_fn(out2);
    F::free_fn(ref);
}

/* =====================================================================
 * 19. plan_dft (multi-dim via rank array) vs plan_dft_3d.
 *     plan_dft and plan_dft_3d both use get_dv_desc but construct
 *     the n array differently. If the compound literal is ephemeral
 *     or mis-sized, results will differ.
 * ===================================================================== */

TYPED_TEST(FftwWrapperInvariantTest,
           PTEST_PLAN_DFT_VS_PLAN_DFT_3D)
{
    using F = FftwTypes<TypeParam>;
    using complex_t [[maybe_unused]] = typename F::complex_t;
    const int N0 = 3, N1 = 4, N2 = 5;
    const int total = N0 * N1 * N2;

    auto *in   = F::alloc_complex(total);
    auto *out1 = F::alloc_complex(total);
    auto *out2 = F::alloc_complex(total);

    this->init_complex(in, total);

    auto p1 = F::plan_dft_3d(
        N0, N1, N2, in, out1,
        FFTW_FORWARD, FFTW_ESTIMATE);
    ASSERT_NE(p1, nullptr);

    int n_arr[] = {N0, N1, N2};
    auto p2 = F::plan_dft(
        3, n_arr, in, out2,
        FFTW_FORWARD, FFTW_ESTIMATE);
    ASSERT_NE(p2, nullptr);

    F::execute(p1);
    F::execute(p2);

    double tol = dft_tolerance<F>(total);
    compare_complex_arrays(out1, out2, total, tol);
    /* Anchor to an independent 3D reference (both paths share get_dv_desc). */
    auto *ref = F::alloc_complex(total);
    dft_reference_3d(in, ref, N0, N1, N2, FFTW_FORWARD);
    compare_complex_arrays(ref, out1, total, tol);
    F::free_fn(ref);

    F::destroy_plan(p1);
    F::destroy_plan(p2);
    F::free_fn(in);
    F::free_fn(out1);
    F::free_fn(out2);
}

/* =====================================================================
 * 20. New-array execute_dft for 2D: plan with A/B, execute with C/D.
 *     A 2D new-array execute is more likely to expose stride bugs
 *     than 1D.
 * ===================================================================== */

TYPED_TEST(FftwWrapperInvariantTest,
           PTEST_EXECUTE_DFT_2D_NEW_ARRAY)
{
    using F = FftwTypes<TypeParam>;
    using complex_t [[maybe_unused]] = typename F::complex_t;
    const int N0 = 4, N1 = 6;
    const int total = N0 * N1;

    auto *plan_in  = F::alloc_complex(total);
    auto *plan_out = F::alloc_complex(total);
    auto *new_in   = F::alloc_complex(total);
    auto *new_out  = F::alloc_complex(total);
    auto *ref_out  = F::alloc_complex(total);

    this->init_complex(plan_in, total);

    this->init_complex_desc_ramp(new_in, total, 3);

    auto p_plan = F::plan_dft_2d(
        N0, N1, plan_in, plan_out,
        FFTW_FORWARD, FFTW_ESTIMATE);
    ASSERT_NE(p_plan, nullptr);

    auto p_ref = F::plan_dft_2d(
        N0, N1, new_in, ref_out,
        FFTW_FORWARD, FFTW_ESTIMATE);
    ASSERT_NE(p_ref, nullptr);
    F::execute(p_ref);

    this->init_complex_desc_ramp(new_in, total, 3);

    F::execute_dft(p_plan, new_in, new_out);

    double tol = dft_tolerance<F>(total);
    compare_complex_arrays(ref_out, new_out, total, tol);

    F::destroy_plan(p_plan);
    F::destroy_plan(p_ref);
    F::free_fn(plan_in);
    F::free_fn(plan_out);
    F::free_fn(new_in);
    F::free_fn(new_out);
    F::free_fn(ref_out);
}
