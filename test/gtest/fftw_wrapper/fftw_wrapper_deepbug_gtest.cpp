// Copyright Advanced Micro Devices, Inc.
// SPDX-License-Identifier: BSD-3-Clause

/**
 * @file fftw_wrapper_deepbug_gtest.cpp
 * @brief Deep bug-hunting tests: C2R input destruction behavior,
 *        flags being ignored, in-place C2R 3D, execute_dft_r2c 2D
 *        new-array, in-place C2C 2D, plan_many R2C in-place, and
 *        N=1 edge cases.
 */

#include <cstdio>
#include <cstring>
#include <vector>
#include "fftw_wrapper_test_utils.h"

template<typename T>
class FftwWrapperDeepBugTest : public FftwWrapperTestBase<T>
{
};
TYPED_TEST_SUITE(FftwWrapperDeepBugTest, FftwTestTypes);

/* =====================================================================
 * C2R input destruction: FFTW documents that C2R transforms
 *    destroy the input array by default. Verify wrapper behavior.
 * ===================================================================== */

TYPED_TEST(FftwWrapperDeepBugTest,
           PTEST_C2R_INPUT_DESTRUCTION)
{
    using F = FftwTypes<TypeParam>;
    using real_t [[maybe_unused]] = typename F::real_t;
    using complex_t [[maybe_unused]] = typename F::complex_t;
    const int N = 16;
    const int Nc = N / 2 + 1;

    auto *in_orig  = F::alloc_complex(Nc);
    auto *in_copy  = F::alloc_complex(Nc);
    auto *real_buf = F::alloc_real(N);
    auto *freq     = F::alloc_complex(Nc);
    auto *out      = F::alloc_real(N);

    this->init_real_iota(real_buf, N);
    auto pr = F::plan_dft_r2c_1d(
        N, real_buf, freq, FFTW_ESTIMATE);
    ASSERT_NE(pr, nullptr);
    F::execute(pr);

    std::memcpy(in_orig, freq, sizeof(complex_t) * Nc);
    std::memcpy(in_copy, freq, sizeof(complex_t) * Nc);

    auto pc = F::plan_dft_c2r_1d(
        N, in_orig, out, FFTW_ESTIMATE);
    ASSERT_NE(pc, nullptr);
    F::execute(pc);

    /* Functional teeth: r2c followed by c2r reproduces the original real input
     * scaled by N. */
    compare_real_scaled(real_buf, out, N, N, dft_tolerance<F>(N));

    /* Separately observe (not assert) whether C2R destroyed its input, since
     * that is a documented AOCL-vs-FFTW behavioral difference rather than a
     * correctness requirement. */
    bool input_destroyed =
        max_error_complex(in_orig, in_copy, Nc) > F::tolerance;

    if (input_destroyed)
    {
        printf("  [INFO] C2R destroys input (matches FFTW "
               "default behavior)\n");
    }
    else
    {
        printf("  [INFO] C2R preserves input (differs from "
               "FFTW default; FFTW destroys input for C2R "
               "unless FFTW_PRESERVE_INPUT is set)\n");
    }

    F::destroy_plan(pr);
    F::destroy_plan(pc);
    F::free_fn(in_orig);
    F::free_fn(in_copy);
    F::free_fn(real_buf);
    F::free_fn(freq);
    F::free_fn(out);
}

/* =====================================================================
 * FFTW flags ignored: wrapper ignores FFTW_DESTROY_INPUT flag.
 *    Verify that passing different flags produces the same plan.
 *    This documents the flags-being-ignored behavior.
 * ===================================================================== */

TYPED_TEST(FftwWrapperDeepBugTest,
           KNOWN_DIVERGENCE_FLAGS_IGNORED)
{
    using F = FftwTypes<TypeParam>;
    using complex_t [[maybe_unused]] = typename F::complex_t;
    const int N = 16;

    auto *in  = F::alloc_complex(N);
    auto *out = F::alloc_complex(N);

    this->init_complex(in, N);

    auto p_estimate = F::plan_dft_1d(
        N, in, out, FFTW_FORWARD, FFTW_ESTIMATE);
    ASSERT_NE(p_estimate, nullptr);

    auto p_measure = F::plan_dft_1d(
        N, in, out, FFTW_FORWARD, FFTW_MEASURE);

    EXPECT_NE(p_measure, nullptr)
        << "Wrapper should succeed with FFTW_MEASURE "
           "flag (even if it ignores it)";

    auto p_patient = F::plan_dft_1d(
        N, in, out, FFTW_FORWARD, FFTW_PATIENT);

    EXPECT_NE(p_patient, nullptr)
        << "Wrapper should succeed with FFTW_PATIENT "
           "flag (even if it ignores it)";

    auto p_exhaustive = F::plan_dft_1d(
        N, in, out, FFTW_FORWARD, FFTW_EXHAUSTIVE);

    EXPECT_NE(p_exhaustive, nullptr)
        << "Wrapper should succeed with FFTW_EXHAUSTIVE "
           "flag (even if it ignores it)";

    F::destroy_plan(p_estimate);
    if (p_measure)
    {
        F::destroy_plan(p_measure);
    }
    if (p_patient)
    {
        F::destroy_plan(p_patient);
    }
    if (p_exhaustive)
    {
        F::destroy_plan(p_exhaustive);
    }
    F::free_fn(in);
    F::free_fn(out);
}

/* =====================================================================
 * N=1 R2C and C2R: single-element transforms.
 *    R2C of 1 element: output should equal input (Nc=1).
 * ===================================================================== */

TYPED_TEST(FftwWrapperDeepBugTest,
           PTEST_R2C_N1_EDGE)
{
    using F = FftwTypes<TypeParam>;
    using real_t [[maybe_unused]] = typename F::real_t;
    using complex_t [[maybe_unused]] = typename F::complex_t;
    const int N = 1;
    const int Nc = 1;

    auto *in  = F::alloc_real(N);
    auto *out = F::alloc_complex(Nc);

    in[0] = static_cast<real_t>(42.0);

    auto p = F::plan_dft_r2c_1d(N, in, out, FFTW_ESTIMATE);
    ASSERT_NE(p, nullptr);
    F::execute(p);

    EXPECT_NEAR(out[0][0], 42.0, F::tolerance)
        << "N=1 R2C: output should equal input";
    EXPECT_NEAR(out[0][1], 0.0, F::tolerance)
        << "N=1 R2C: imaginary should be 0";

    F::destroy_plan(p);
    F::free_fn(in);
    F::free_fn(out);
}

/* =====================================================================
 * N=1 C2R edge case.
 * ===================================================================== */

TYPED_TEST(FftwWrapperDeepBugTest,
           PTEST_C2R_N1_EDGE)
{
    using F = FftwTypes<TypeParam>;
    using real_t [[maybe_unused]] = typename F::real_t;
    using complex_t [[maybe_unused]] = typename F::complex_t;
    const int N = 1;
    const int Nc = 1;

    auto *in  = F::alloc_complex(Nc);
    auto *out = F::alloc_real(N);

    in[0][0] = 42.0;
    in[0][1] = 0.0;

    auto p = F::plan_dft_c2r_1d(N, in, out, FFTW_ESTIMATE);
    ASSERT_NE(p, nullptr);
    F::execute(p);

    EXPECT_NEAR(static_cast<double>(out[0]), 42.0,
                F::tolerance)
        << "N=1 C2R: output should equal real part of input";

    F::destroy_plan(p);
    F::free_fn(in);
    F::free_fn(out);
}

/* =====================================================================
 * In-place C2R 3D roundtrip: tests get_c2r_dv_desc in-place
 *    path with 3 dimensions.
 * ===================================================================== */

TYPED_TEST(FftwWrapperDeepBugTest,
           PTEST_C2R_3D_INPLACE_ROUNDTRIP)
{
    using F = FftwTypes<TypeParam>;
    using real_t [[maybe_unused]] = typename F::real_t;
    using complex_t [[maybe_unused]] = typename F::complex_t;
    const int N0 = 3, N1 = 4, N2 = 6;
    const int padded_N2 = 2 * (N2 / 2 + 1);
    const int buf_reals = N0 * N1 * padded_N2;
    const int total = N0 * N1 * N2;

    auto *buf = (real_t *)F::malloc_fn(
        sizeof(real_t) * buf_reals);
    ASSERT_NE(buf, nullptr);

    std::memset(buf, 0, sizeof(real_t) * buf_reals);

    /* Contiguous ramp saved[p]=p+1 scattered into the padded in-place buffer. */
    auto *saved = F::alloc_real(total);
    this->init_real_iota(saved, total);
    this->scatter_padded_real(buf, saved, static_cast<long>(N0) * N1, N2,
                              padded_N2);

    complex_t *cbuf = (complex_t *)buf;

    auto pr = F::plan_dft_r2c_3d(
        N0, N1, N2, buf, cbuf, FFTW_ESTIMATE);
    ASSERT_NE(pr, nullptr);
    F::execute(pr);

    auto pc = F::plan_dft_c2r_3d(
        N0, N1, N2, cbuf, buf, FFTW_ESTIMATE);
    ASSERT_NE(pc, nullptr);
    F::execute(pc);

    double tol = dft_tolerance<F>(total);
    std::vector<real_t> result(total);
    this->gather_padded_real(result.data(), buf,
                             static_cast<long>(N0) * N1, N2, padded_N2);
    compare_real_scaled(saved, result.data(), total, total, tol);

    F::destroy_plan(pr);
    F::destroy_plan(pc);
    F::free_fn(buf);
    F::free_fn(saved);
}

/* =====================================================================
 * Execute_dft_r2c with 2D new arrays: plan with A/B, execute
 *    with C/D for a 2D R2C. Verifies new-array execute works for
 *    multi-dim R2C.
 * ===================================================================== */

TYPED_TEST(FftwWrapperDeepBugTest,
           PTEST_EXECUTE_DFT_R2C_2D_NEW_ARRAY)
{
    using F = FftwTypes<TypeParam>;
    using real_t [[maybe_unused]] = typename F::real_t;
    using complex_t [[maybe_unused]] = typename F::complex_t;
    const int N0 = 4, N1 = 6;
    const int total = N0 * N1;
    const int Nc = N0 * (N1 / 2 + 1);

    auto *plan_in  = F::alloc_real(total);
    auto *plan_out = F::alloc_complex(Nc);
    auto *new_in   = F::alloc_real(total);
    auto *new_out  = F::alloc_complex(Nc);
    auto *ref_out  = F::alloc_complex(Nc);

    this->init_real(plan_in, total);

    this->fill_affine_real(new_in, total, static_cast<real_t>(total),
                           static_cast<real_t>(-1));

    auto p1 = F::plan_dft_r2c_2d(
        N0, N1, plan_in, plan_out, FFTW_ESTIMATE);
    ASSERT_NE(p1, nullptr);

    auto p_ref = F::plan_dft_r2c_2d(
        N0, N1, new_in, ref_out, FFTW_ESTIMATE);
    ASSERT_NE(p_ref, nullptr);
    F::execute(p_ref);

    this->fill_affine_real(new_in, total, static_cast<real_t>(total),
                           static_cast<real_t>(-1));
    F::execute_dft_r2c(p1, new_in, new_out);

    compare_complex_arrays(ref_out, new_out, Nc, dft_tolerance<F>(total));

    F::destroy_plan(p1);
    F::destroy_plan(p_ref);
    F::free_fn(plan_in);
    F::free_fn(plan_out);
    F::free_fn(new_in);
    F::free_fn(new_out);
    F::free_fn(ref_out);
}

/* =====================================================================
 * Execute_dft_c2r with 2D new arrays.
 * ===================================================================== */

TYPED_TEST(FftwWrapperDeepBugTest,
           PTEST_EXECUTE_DFT_C2R_2D_NEW_ARRAY)
{
    using F = FftwTypes<TypeParam>;
    using real_t [[maybe_unused]] = typename F::real_t;
    using complex_t [[maybe_unused]] = typename F::complex_t;
    const int N0 = 4, N1 = 6;
    const int total = N0 * N1;
    const int Nc = N0 * (N1 / 2 + 1);

    auto *plan_c = F::alloc_complex(Nc);
    auto *plan_r = F::alloc_real(total);

    auto *src_r    = F::alloc_real(total);
    auto *new_freq = F::alloc_complex(Nc);
    auto *new_c_in = F::alloc_complex(Nc);
    auto *new_r_out = F::alloc_real(total);

    this->init_real_iota(src_r, total);

    auto pr_plan = F::plan_dft_r2c_2d(
        N0, N1, src_r, new_freq, FFTW_ESTIMATE);
    ASSERT_NE(pr_plan, nullptr);
    F::execute(pr_plan);
    std::memcpy(new_c_in, new_freq, sizeof(complex_t) * Nc);

    this->fill_complex_constant(plan_c, Nc, 1.0, 0.0);
    auto pc_plan = F::plan_dft_c2r_2d(
        N0, N1, plan_c, plan_r, FFTW_ESTIMATE);
    ASSERT_NE(pc_plan, nullptr);

    F::execute_dft_c2r(pc_plan, new_c_in, new_r_out);

    compare_real_scaled(src_r, new_r_out, total, total, dft_tolerance<F>(total));

    F::destroy_plan(pr_plan);
    F::destroy_plan(pc_plan);
    F::free_fn(plan_c);
    F::free_fn(plan_r);
    F::free_fn(src_r);
    F::free_fn(new_freq);
    F::free_fn(new_c_in);
    F::free_fn(new_r_out);
}

/* =====================================================================
 * In-place C2C 2D: verify roundtrip with asymmetric dims.
 *     Tests that in-place detection (in == out pointer) works
 *     correctly for C2C.
 * ===================================================================== */

TYPED_TEST(FftwWrapperDeepBugTest,
           PTEST_INPLACE_C2C_2D_ROUNDTRIP)
{
    using F = FftwTypes<TypeParam>;
    using complex_t [[maybe_unused]] = typename F::complex_t;
    const int N0 = 5, N1 = 7;
    const int total = N0 * N1;

    auto *buf  = F::alloc_complex(total);
    auto *saved = F::alloc_complex(total);

    this->init_complex(buf, total);
    std::memcpy(saved, buf, sizeof(complex_t) * total);

    auto pf = F::plan_dft_2d(
        N0, N1, buf, buf, FFTW_FORWARD, FFTW_ESTIMATE);
    auto pb = F::plan_dft_2d(
        N0, N1, buf, buf, FFTW_BACKWARD, FFTW_ESTIMATE);
    ASSERT_NE(pf, nullptr);
    ASSERT_NE(pb, nullptr);

    F::execute(pf);
    F::execute(pb);

    compare_complex_scaled(saved, buf, total, total, dft_tolerance<F>(total));

    F::destroy_plan(pf);
    F::destroy_plan(pb);
    F::free_fn(buf);
    F::free_fn(saved);
}

/* =====================================================================
 * plan_many_dft_r2c with rank=2, in-place.
 *     Tests get_many_r2c_dv_desc in-place path for 2D.
 *     Uses padded real buffer of size N0 * 2*(N1/2+1).
 * ===================================================================== */

TYPED_TEST(FftwWrapperDeepBugTest,
           PTEST_MANY_R2C_2D_INPLACE_ROUNDTRIP)
{
    using F = FftwTypes<TypeParam>;
    using real_t [[maybe_unused]] = typename F::real_t;
    using complex_t [[maybe_unused]] = typename F::complex_t;
    const int N0 = 4, N1 = 6;
    const int total = N0 * N1;
    const int padded_N1 = 2 * (N1 / 2 + 1);
    const int buf_reals = N0 * padded_N1;
    const int Nc = N0 * (N1 / 2 + 1);

    auto *buf = (real_t *)F::malloc_fn(
        sizeof(real_t) * buf_reals);
    ASSERT_NE(buf, nullptr);

    std::memset(buf, 0, sizeof(real_t) * buf_reals);

    /* Contiguous ramp saved[p]=p+1 scattered into the row-padded buffer. */
    auto *saved = F::alloc_real(total);
    this->init_real_iota(saved, total);
    this->scatter_padded_real(buf, saved, N0, N1, padded_N1);

    complex_t *cbuf = (complex_t *)buf;

    int n_arr[] = {N0, N1};
    int inembed[] = {N0, padded_N1};
    int onembed[] = {N0, (N1 / 2 + 1)};

    auto pr = F::plan_many_dft_r2c(
        2, n_arr, 1,
        buf, inembed, 1, buf_reals,
        cbuf, onembed, 1, Nc,
        FFTW_ESTIMATE);
    ASSERT_NE(pr, nullptr);

    auto pc = F::plan_many_dft_c2r(
        2, n_arr, 1,
        cbuf, onembed, 1, Nc,
        buf, inembed, 1, buf_reals,
        FFTW_ESTIMATE);
    ASSERT_NE(pc, nullptr);

    F::execute(pr);
    F::execute(pc);

    double tol = dft_tolerance<F>(total);
    std::vector<real_t> result(total);
    this->gather_padded_real(result.data(), buf, N0, N1, padded_N1);
    compare_real_scaled(saved, result.data(), total, total, tol);

    F::destroy_plan(pr);
    F::destroy_plan(pc);
    F::free_fn(buf);
    F::free_fn(saved);
}
