// Copyright Advanced Micro Devices, Inc.
// SPDX-License-Identifier: BSD-3-Clause

/** @file fftw_wrapper_sequence_gtest.cpp
 * @brief GTest cases for FFTW wrapper API sequence and workflow tests.
 */

#include <vector>
#include "fftw_wrapper_test_utils.h"

template<typename T>
class FftwWrapperSequenceTest : public FftwWrapperTestBase<T>
{
};
TYPED_TEST_SUITE(FftwWrapperSequenceTest, FftwTestTypes);

/* ====================================================================
 * Typical application workflows
 * ==================================================================== */

TYPED_TEST(FftwWrapperSequenceTest, PTEST_BASIC_WORKFLOW)
{
    using F = FftwTypes<TypeParam>;
    const int N = 32;
    auto *in  = F::alloc_complex(N);
    auto *out = F::alloc_complex(N);
    auto *ref = F::alloc_complex(N);

    this->init_complex(in, N);
    auto p = F::plan_dft_1d(N, in, out, FFTW_FORWARD, FFTW_ESTIMATE);
    ASSERT_NE(p, nullptr);
    F::execute(p);

    dft_reference_1d(in, ref, N, FFTW_FORWARD);
    compare_complex_arrays(ref, out, N, dft_tolerance<F>(N));

    F::destroy_plan(p);
    F::free_fn(in);
    F::free_fn(out);
    F::free_fn(ref);
}

TYPED_TEST(FftwWrapperSequenceTest, PTEST_MULTIPLE_PLANS_SAME_SIZE)
{
    using F = FftwTypes<TypeParam>;
    const int N = 16;
    auto *in1  = F::alloc_complex(N);
    auto *out1 = F::alloc_complex(N);
    auto *in2  = F::alloc_complex(N);
    auto *out2 = F::alloc_complex(N);

    this->init_complex(in1, N);
    this->init_complex(in2, N);

    auto p1 = F::plan_dft_1d(N, in1, out1, FFTW_FORWARD, FFTW_ESTIMATE);
    auto p2 = F::plan_dft_1d(N, in2, out2, FFTW_FORWARD, FFTW_ESTIMATE);
    ASSERT_NE(p1, nullptr);
    ASSERT_NE(p2, nullptr);

    F::execute(p1);
    F::execute(p2);

    compare_complex_arrays(out1, out2, N, dft_tolerance<F>(N));

    F::destroy_plan(p1);
    F::destroy_plan(p2);
    F::free_fn(in1);
    F::free_fn(out1);
    F::free_fn(in2);
    F::free_fn(out2);
}

TYPED_TEST(FftwWrapperSequenceTest, PTEST_MULTIPLE_PLANS_DIFFERENT_SIZES)
{
    using F = FftwTypes<TypeParam>;
    const int N1 = 16;
    const int N2 = 32;
    auto *in1  = F::alloc_complex(N1);
    auto *out1 = F::alloc_complex(N1);
    auto *ref1 = F::alloc_complex(N1);
    auto *in2  = F::alloc_complex(N2);
    auto *out2 = F::alloc_complex(N2);
    auto *ref2 = F::alloc_complex(N2);

    this->init_complex(in1, N1);
    this->init_complex(in2, N2);

    auto p1 = F::plan_dft_1d(N1, in1, out1, FFTW_FORWARD, FFTW_ESTIMATE);
    auto p2 = F::plan_dft_1d(N2, in2, out2, FFTW_FORWARD, FFTW_ESTIMATE);
    ASSERT_NE(p1, nullptr);
    ASSERT_NE(p2, nullptr);

    F::execute(p1);
    F::execute(p2);

    dft_reference_1d(in1, ref1, N1, FFTW_FORWARD);
    dft_reference_1d(in2, ref2, N2, FFTW_FORWARD);
    compare_complex_arrays(ref1, out1, N1, dft_tolerance<F>(N1));
    compare_complex_arrays(ref2, out2, N2, dft_tolerance<F>(N2));

    F::destroy_plan(p1);
    F::destroy_plan(p2);
    F::free_fn(in1);
    F::free_fn(out1);
    F::free_fn(ref1);
    F::free_fn(in2);
    F::free_fn(out2);
    F::free_fn(ref2);
}

TYPED_TEST(FftwWrapperSequenceTest, PTEST_PLAN_REUSE_MULTIPLE_EXECUTIONS)
{
    using F = FftwTypes<TypeParam>;
    const int N = 16;
    auto *in  = F::alloc_complex(N);
    auto *out = F::alloc_complex(N);
    auto *ref = F::alloc_complex(N);

    auto p = F::plan_dft_1d(N, in, out, FFTW_FORWARD, FFTW_ESTIMATE);
    ASSERT_NE(p, nullptr);

    for (int iter = 0; iter < 10; iter++)
    {
        this->init_complex_arithmetic(in, N, static_cast<TypeParam>(iter));
        F::execute(p);
        dft_reference_1d(in, ref, N, FFTW_FORWARD);
        compare_complex_arrays(ref, out, N, dft_tolerance<F>(N));
    }

    F::destroy_plan(p);
    F::free_fn(in);
    F::free_fn(out);
    F::free_fn(ref);
}

TYPED_TEST(FftwWrapperSequenceTest, PTEST_PLAN_DESTROY_RECREATE)
{
    using F = FftwTypes<TypeParam>;
    const int N = 16;
    auto *in  = F::alloc_complex(N);
    auto *out = F::alloc_complex(N);
    auto *ref = F::alloc_complex(N);
    this->init_complex(in, N);

    auto p1 = F::plan_dft_1d(N, in, out, FFTW_FORWARD, FFTW_ESTIMATE);
    ASSERT_NE(p1, nullptr);
    F::destroy_plan(p1);

    auto p2 = F::plan_dft_1d(N, in, out, FFTW_FORWARD, FFTW_ESTIMATE);
    ASSERT_NE(p2, nullptr);
    F::execute(p2);
    dft_reference_1d(in, ref, N, FFTW_FORWARD);
    compare_complex_arrays(ref, out, N, dft_tolerance<F>(N));

    F::destroy_plan(p2);
    F::free_fn(in);
    F::free_fn(out);
    F::free_fn(ref);
}

TYPED_TEST(FftwWrapperSequenceTest, PTEST_R2C_THEN_C2R_WORKFLOW)
{
    using F = FftwTypes<TypeParam>;
    using real_t = typename F::real_t;
    const int N = 32;
    int Nc = N / 2 + 1;

    auto *in   = F::alloc_real(N);
    auto *freq = F::alloc_complex(Nc);
    auto *out  = F::alloc_real(N);

    this->init_real(in, N);
    std::vector<real_t> orig(in, in + N);

    auto pr = F::plan_dft_r2c_1d(N, in, freq, FFTW_ESTIMATE);
    ASSERT_NE(pr, nullptr);
    F::execute(pr);

    auto pc = F::plan_dft_c2r_1d(N, freq, out, FFTW_ESTIMATE);
    ASSERT_NE(pc, nullptr);
    F::execute(pc);

    compare_real_scaled(orig.data(), out, N, N, dft_tolerance<F>(N));

    F::destroy_plan(pr);
    F::destroy_plan(pc);
    F::free_fn(in);
    F::free_fn(freq);
    F::free_fn(out);
}

TYPED_TEST(FftwWrapperSequenceTest, PTEST_MIXED_PRECISION_PLANS)
{
    const int N = 16;
    fftw_complex *d_in  = fftw_alloc_complex(N);
    fftw_complex *d_out = fftw_alloc_complex(N);
    fftwf_complex *f_in  = fftwf_alloc_complex(N);
    fftwf_complex *f_out = fftwf_alloc_complex(N);

    /* Same {i+1, 0} ramp into both precisions via the shared free-function
     * initializer (the fixture member is single-precision, so it cannot fill
     * the double and float buffers of this cross-precision test). */
    fill_complex_real_iota(d_in, N);
    fill_complex_real_iota(f_in, N);

    fftw_plan pd = fftw_plan_dft_1d(N, d_in, d_out,
                                    FFTW_FORWARD, FFTW_ESTIMATE);
    fftwf_plan pf = fftwf_plan_dft_1d(N, f_in, f_out,
                                      FFTW_FORWARD, FFTW_ESTIMATE);
    ASSERT_NE(pd, nullptr);
    ASSERT_NE(pf, nullptr);

    fftw_execute(pd);
    fftwf_execute(pf);

    /* Double vs single agreement is bounded by the single-precision epsilon,
     * so use the looser cross-precision tolerance rather than F::tolerance. */
    compare_complex_cross(d_out, f_out, N, 1e-4);

    fftw_destroy_plan(pd);
    fftwf_destroy_plan(pf);
    fftw_free(d_in);
    fftw_free(d_out);
    fftwf_free(f_in);
    fftwf_free(f_out);
}

/* ====================================================================
 * Threading sequences
 * ==================================================================== */

TYPED_TEST(FftwWrapperSequenceTest, PTEST_THREADED_WORKFLOW)
{
    using F = FftwTypes<TypeParam>;
    const int N = 32;
    F::init_threads();
    F::plan_with_nthreads(4);

    /* RAII guards: the buffers and plan free themselves on scope exit,
     * including the ASSERT_NE early-return path below (keeps the failure
     * path leak-clean under the project's AddressSanitizer build). */
    auto in  = make_complex_buf<F>(N);
    auto out = make_complex_buf<F>(N);
    auto ref = make_complex_buf<F>(N);
    this->init_complex(in, N);

    ScopedPlan<F> p = F::plan_dft_1d(N, in, out, FFTW_FORWARD, FFTW_ESTIMATE);
    ASSERT_NE(p.get(), nullptr);
    F::execute(p);

    /* Template helpers cannot deduce through the guard's implicit conversion,
     * so hand them the raw pointer via .get(). */
    dft_reference_1d(in.get(), ref.get(), N, FFTW_FORWARD);
    compare_complex_arrays(ref.get(), out.get(), N, dft_tolerance<F>(N));

    F::cleanup_threads();
}

TYPED_TEST(FftwWrapperSequenceTest, PTEST_THREAD_COUNT_CHANGE)
{
    using F = FftwTypes<TypeParam>;
    const int N = 16;

    F::init_threads();
    auto *in  = F::alloc_complex(N);
    auto *outA = F::alloc_complex(N);
    auto *outB = F::alloc_complex(N);
    this->init_complex(in, N);

    F::plan_with_nthreads(2);
    auto pA = F::plan_dft_1d(N, in, outA, FFTW_FORWARD, FFTW_ESTIMATE);
    ASSERT_NE(pA, nullptr);

    F::plan_with_nthreads(4);
    auto pB = F::plan_dft_1d(N, in, outB, FFTW_FORWARD, FFTW_ESTIMATE);
    ASSERT_NE(pB, nullptr);

    F::execute(pA);
    F::execute(pB);

    compare_complex_arrays(outA, outB, N, dft_tolerance<F>(N));

    F::destroy_plan(pA);
    F::destroy_plan(pB);
    F::free_fn(in);
    F::free_fn(outA);
    F::free_fn(outB);
    F::cleanup_threads();
}

TYPED_TEST(FftwWrapperSequenceTest, PTEST_THREAD_INIT_BEFORE_PLAN)
{
    using F = FftwTypes<TypeParam>;
    const int N = 16;

    F::init_threads();
    F::plan_with_nthreads(4);

    auto *in  = F::alloc_complex(N);
    auto *out = F::alloc_complex(N);
    this->init_complex(in, N);

    auto p = F::plan_dft_1d(N, in, out, FFTW_FORWARD, FFTW_ESTIMATE);
    ASSERT_NE(p, nullptr);
    F::execute(p);
    F::destroy_plan(p);
    F::free_fn(in);
    F::free_fn(out);
    F::cleanup_threads();
}

/* ====================================================================
 * Wisdom stub sequences
 * ==================================================================== */

TYPED_TEST(FftwWrapperSequenceTest, PTEST_WISDOM_STUB_SEQUENCE)
{
    using F = FftwTypes<TypeParam>;
    const int N = 16;
    auto *in  = F::alloc_complex(N);
    auto *out = F::alloc_complex(N);
    auto *ref = F::alloc_complex(N);
    this->init_complex(in, N);

    auto p1 = F::plan_dft_1d(N, in, out, FFTW_FORWARD, FFTW_ESTIMATE);
    ASSERT_NE(p1, nullptr);
    F::execute(p1);
    dft_reference_1d(in, ref, N, FFTW_FORWARD);
    compare_complex_arrays(ref, out, N, dft_tolerance<F>(N));
    F::destroy_plan(p1);

    int exp_ret = F::export_wisdom_to_filename("dummy_seq.wis");
    EXPECT_EQ(exp_ret, 1);
    F::cleanup();
    int imp_ret = F::import_wisdom_from_filename("dummy_seq.wis");
    EXPECT_EQ(imp_ret, 0);

    auto p2 = F::plan_dft_1d(N, in, out, FFTW_FORWARD, FFTW_ESTIMATE);
    ASSERT_NE(p2, nullptr);
    F::execute(p2);
    compare_complex_arrays(ref, out, N, dft_tolerance<F>(N));
    F::destroy_plan(p2);

    F::free_fn(in);
    F::free_fn(out);
    F::free_fn(ref);
}

TYPED_TEST(FftwWrapperSequenceTest, PTEST_WISDOM_WITH_THREADS)
{
    using F = FftwTypes<TypeParam>;
    const int N = 16;
    F::init_threads();
    F::plan_with_nthreads(4);

    auto *in  = F::alloc_complex(N);
    auto *out = F::alloc_complex(N);
    this->init_complex(in, N);

    auto p = F::plan_dft_1d(N, in, out, FFTW_FORWARD, FFTW_ESTIMATE);
    ASSERT_NE(p, nullptr);
    F::execute(p);
    F::destroy_plan(p);

    F::export_wisdom_to_filename("wis_thread.wis");
    F::forget_wisdom();
    F::import_wisdom_from_filename("wis_thread.wis");

    auto p2 = F::plan_dft_1d(N, in, out, FFTW_FORWARD, FFTW_ESTIMATE);
    ASSERT_NE(p2, nullptr);
    F::execute(p2);
    F::destroy_plan(p2);

    F::free_fn(in);
    F::free_fn(out);
    F::cleanup_threads();
}

/* ====================================================================
 * Edge-case sequences
 * ==================================================================== */

/* Uses the RAII guards (ScopedBuf/ScopedPlan) so the buffers and plan are
 * released on every exit path, including a failing ASSERT -- the leak-clean
 * pattern the rest of the suite can adopt incrementally. */
TYPED_TEST(FftwWrapperSequenceTest, PTEST_DESTROY_BEFORE_EXECUTE)
{
    using F = FftwTypes<TypeParam>;
    const int N = 16;
    auto in  = make_complex_buf<F>(N);
    auto out = make_complex_buf<F>(N);
    this->init_complex(in, N);

    ScopedPlan<F> p = F::plan_dft_1d(N, in, out, FFTW_FORWARD, FFTW_ESTIMATE);
    ASSERT_NE(p.get(), nullptr);
    /* p, in and out are freed automatically when the scope ends. */
}

/* CONTRACT NOTE (AOCL-FFTZ divergence from FFTW):
 * Native FFTW documents that fftw_cleanup() invalidates ALL existing plans, so
 * touching a plan (destroy or execute) afterwards is undefined behavior. This
 * test intentionally destroys a plan after cleanup() to characterize AOCL-FFTZ,
 * whose cleanup() does NOT invalidate outstanding plans -- so destroy remains
 * safe. It documents AOCL's more lenient behavior and must not be read as an
 * endorsement of the destroy-after-cleanup pattern for portable FFTW code. */
TYPED_TEST(FftwWrapperSequenceTest, PTEST_CLEANUP_BEFORE_DESTROY)
{
    using F = FftwTypes<TypeParam>;
    const int N = 16;
    auto *in  = F::alloc_complex(N);
    auto *out = F::alloc_complex(N);
    this->init_complex(in, N);

    auto p = F::plan_dft_1d(N, in, out, FFTW_FORWARD, FFTW_ESTIMATE);
    ASSERT_NE(p, nullptr);
    F::cleanup();
    F::destroy_plan(p);
    F::free_fn(in);
    F::free_fn(out);
}

/* CONTRACT NOTE (AOCL-FFTZ divergence from FFTW):
 * Executing an existing plan after fftw_cleanup() is undefined behavior in
 * native FFTW (cleanup invalidates the plan). AOCL-FFTZ plans are
 * self-contained and survive cleanup(), so execute() still produces the
 * correct transform -- which we now actually verify against the reference
 * rather than merely calling it. As above, this characterizes AOCL behavior
 * and is not a portable-FFTW usage pattern. */
TYPED_TEST(FftwWrapperSequenceTest, PTEST_EXECUTE_AFTER_CLEANUP)
{
    using F = FftwTypes<TypeParam>;
    using complex_t = typename F::complex_t;
    const int N = 16;
    auto *in  = F::alloc_complex(N);
    auto *out = F::alloc_complex(N);
    auto *ref = F::alloc_complex(N);
    this->init_complex(in, N);

    auto p = F::plan_dft_1d(N, in, out, FFTW_FORWARD, FFTW_ESTIMATE);
    ASSERT_NE(p, nullptr);
    F::cleanup();
    F::execute(p);

    dft_reference_1d(in, ref, N, FFTW_FORWARD);
    compare_complex_arrays<complex_t>(ref, out, N, dft_tolerance<F>(N));

    F::destroy_plan(p);
    F::free_fn(in);
    F::free_fn(out);
    F::free_fn(ref);
}
