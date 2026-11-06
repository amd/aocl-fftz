// Copyright Advanced Micro Devices, Inc.
// SPDX-License-Identifier: BSD-3-Clause

/**
 * @file fftw_wrapper_coverage_gtest.cpp
 * @brief API coverage gaps: positive corner-case tests and missing API
 *        variant tests.
 */

#include <cstring>
#include "fftw_wrapper_test_utils.h"

template<typename T>
class FftwWrapperCoverageTest : public FftwWrapperTestBase<T>
{
};
TYPED_TEST_SUITE(FftwWrapperCoverageTest, FftwTestTypes);

class FftwWrapperCoverageMiscTest : public ::testing::Test
{
};

/* ===== Section 1 -- Execute variants (TYPED_TEST) ===== */

TYPED_TEST(FftwWrapperCoverageTest, PTEST_EXECUTE_DFT_R2C_NEW_BUFFERS)
{
    using F = FftwTypes<TypeParam>;
    using real_t = typename F::real_t;
    const int N = 16;
    int Nc = N / 2 + 1;
    auto *in1 = F::alloc_real(N);
    auto *out1 = F::alloc_complex(Nc);
    this->init_real_iota(in1, N);
    auto p = F::plan_dft_r2c_1d(N, in1, out1, FFTW_ESTIMATE);
    ASSERT_NE(p, nullptr);

    auto *in2 = F::alloc_real(N);
    auto *out2 = F::alloc_complex(Nc);
    this->init_real_iota(in2, N);
    F::execute(p);
    F::execute_dft_r2c(p, in2, out2);
    compare_complex_arrays(out1, out2, Nc, dft_tolerance<F>(N));

    F::destroy_plan(p);
    F::free_fn(in1);
    F::free_fn(out1);
    F::free_fn(in2);
    F::free_fn(out2);
}

TYPED_TEST(FftwWrapperCoverageTest, PTEST_EXECUTE_DFT_C2R_NEW_BUFFERS)
{
    using F = FftwTypes<TypeParam>;
    using real_t = typename F::real_t;
    const int N = 16;
    int Nc = N / 2 + 1;
    auto *in = F::alloc_real(N);
    auto *freq = F::alloc_complex(Nc);
    auto *out = F::alloc_real(N);

    this->init_real_iota(in, N);

    auto pr = F::plan_dft_r2c_1d(N, in, freq, FFTW_ESTIMATE);
    auto pc = F::plan_dft_c2r_1d(N, freq, out, FFTW_ESTIMATE);
    ASSERT_NE(pr, nullptr);
    ASSERT_NE(pc, nullptr);

    F::execute(pr);

    auto *freq2 = F::alloc_complex(Nc);
    auto *out2 = F::alloc_real(N);
    std::memcpy(freq2, freq, sizeof(typename F::complex_t) * Nc);
    F::execute_dft_c2r(pc, freq2, out2);

    compare_real_scaled(in, out2, N, N, dft_tolerance<F>(N));

    F::destroy_plan(pr);
    F::destroy_plan(pc);
    F::free_fn(in);
    F::free_fn(freq);
    F::free_fn(out);
    F::free_fn(freq2);
    F::free_fn(out2);
}

TYPED_TEST(FftwWrapperCoverageTest, PTEST_EXECUTE_DFT_R2C_C2R_ROUNDTRIP)
{
    using F = FftwTypes<TypeParam>;
    using real_t = typename F::real_t;
    const int N = 32;
    int Nc = N / 2 + 1;

    auto *in = F::alloc_real(N);
    auto *freq = F::alloc_complex(Nc);
    auto *out = F::alloc_real(N);

    this->init_real_iota(in, N);

    auto pr = F::plan_dft_r2c_1d(N, in, freq, FFTW_ESTIMATE);
    auto pc = F::plan_dft_c2r_1d(N, freq, out, FFTW_ESTIMATE);
    ASSERT_NE(pr, nullptr);
    ASSERT_NE(pc, nullptr);

    auto *in2 = F::alloc_real(N);
    auto *freq2 = F::alloc_complex(Nc);
    auto *out2 = F::alloc_real(N);
    this->init_real_iota(in2, N);

    F::execute_dft_r2c(pr, in2, freq2);
    F::execute_dft_c2r(pc, freq2, out2);

    compare_real_scaled(in2, out2, N, N, dft_tolerance<F>(N));

    F::destroy_plan(pr);
    F::destroy_plan(pc);
    F::free_fn(in);
    F::free_fn(freq);
    F::free_fn(out);
    F::free_fn(in2);
    F::free_fn(freq2);
    F::free_fn(out2);
}

/* ===== Section 2 -- Misc float coverage (non-typed) ===== */

TEST_F(FftwWrapperCoverageMiscTest, TEST_FPRINT_PLAN_F)
{
    fftwf_complex *in = fftwf_alloc_complex(8);
    fftwf_complex *out = fftwf_alloc_complex(8);
    fftwf_plan p = fftwf_plan_dft_1d(
        8, in, out, FFTW_FORWARD, FFTW_ESTIMATE);
    ASSERT_NE(p, nullptr);
    fftwf_fprint_plan(p, stderr);
    fftwf_destroy_plan(p);
    fftwf_free(in);
    fftwf_free(out);
}

TEST_F(FftwWrapperCoverageMiscTest, TEST_SPRINT_PLAN_F_NULL)
{
    char *s = fftwf_sprint_plan(nullptr);
    EXPECT_NE(s, nullptr);
    if (s) fftwf_free(s);
}

TEST_F(FftwWrapperCoverageMiscTest, TEST_FFTW_ALIGNMENT_OF_FLOAT_EXPLICIT)
{
    float *p = fftwf_alloc_real(16);
    ASSERT_NE(p, nullptr);
    int a = fftwf_alignment_of(p);
    EXPECT_EQ(a, 0);
    fftwf_free(p);
}

/* ===== Section 3 -- Boundary correctness (TYPED_TEST) ===== */

TYPED_TEST(FftwWrapperCoverageTest, PTEST_EXECUTE_SIZE_1)
{
    using F = FftwTypes<TypeParam>;
    auto *in = F::alloc_complex(1);
    auto *out = F::alloc_complex(1);
    (*in)[0] = 3.0;
    (*in)[1] = 4.0;
    auto p = F::plan_dft_1d(
        1, in, out, FFTW_FORWARD, FFTW_ESTIMATE);
    ASSERT_NE(p, nullptr);
    F::execute(p);
    EXPECT_NEAR((*out)[0], 3.0, F::tolerance);
    EXPECT_NEAR((*out)[1], 4.0, F::tolerance);
    F::destroy_plan(p);
    F::free_fn(in);
    F::free_fn(out);
}

TYPED_TEST(FftwWrapperCoverageTest, PTEST_EXECUTE_SIZE_2)
{
    using F = FftwTypes<TypeParam>;
    auto *in = F::alloc_complex(2);
    auto *out = F::alloc_complex(2);
    in[0][0] = 1.0;
    in[0][1] = 0.0;
    in[1][0] = 1.0;
    in[1][1] = 0.0;
    auto p = F::plan_dft_1d(
        2, in, out, FFTW_FORWARD, FFTW_ESTIMATE);
    ASSERT_NE(p, nullptr);
    F::execute(p);
    EXPECT_NEAR(out[0][0], 2.0, F::tolerance);
    EXPECT_NEAR(out[0][1], 0.0, F::tolerance);
    EXPECT_NEAR(out[1][0], 0.0, F::tolerance);
    EXPECT_NEAR(out[1][1], 0.0, F::tolerance);
    F::destroy_plan(p);
    F::free_fn(in);
    F::free_fn(out);
}

TYPED_TEST(FftwWrapperCoverageTest, PTEST_EXECUTE_SIZE_3)
{
    using F = FftwTypes<TypeParam>;
    auto *in = F::alloc_complex(3);
    auto *out = F::alloc_complex(3);
    auto *ref = F::alloc_complex(3);
    this->init_complex(in, 3);
    dft_reference_1d(in, ref, 3, FFTW_FORWARD);
    auto p = F::plan_dft_1d(
        3, in, out, FFTW_FORWARD, FFTW_ESTIMATE);
    ASSERT_NE(p, nullptr);
    F::execute(p);
    compare_complex_arrays(ref, out, 3, dft_tolerance<F>(3));
    F::destroy_plan(p);
    F::free_fn(in);
    F::free_fn(out);
    F::free_fn(ref);
}

TYPED_TEST(FftwWrapperCoverageTest, PTEST_EXECUTE_SIZE_PRIME_7)
{
    using F = FftwTypes<TypeParam>;
    auto *in = F::alloc_complex(7);
    auto *out = F::alloc_complex(7);
    auto *ref = F::alloc_complex(7);
    this->init_complex(in, 7);
    dft_reference_1d(in, ref, 7, FFTW_FORWARD);
    auto p = F::plan_dft_1d(
        7, in, out, FFTW_FORWARD, FFTW_ESTIMATE);
    ASSERT_NE(p, nullptr);
    F::execute(p);
    compare_complex_arrays(ref, out, 7, dft_tolerance<F>(7));
    F::destroy_plan(p);
    F::free_fn(in);
    F::free_fn(out);
    F::free_fn(ref);
}

TYPED_TEST(FftwWrapperCoverageTest, PTEST_EXECUTE_SIZE_PRIME_13)
{
    using F = FftwTypes<TypeParam>;
    auto *in = F::alloc_complex(13);
    auto *out = F::alloc_complex(13);
    auto *ref = F::alloc_complex(13);
    this->init_complex(in, 13);
    dft_reference_1d(in, ref, 13, FFTW_FORWARD);
    auto p = F::plan_dft_1d(
        13, in, out, FFTW_FORWARD, FFTW_ESTIMATE);
    ASSERT_NE(p, nullptr);
    F::execute(p);
    compare_complex_arrays(ref, out, 13, dft_tolerance<F>(13));
    F::destroy_plan(p);
    F::free_fn(in);
    F::free_fn(out);
    F::free_fn(ref);
}

TYPED_TEST(FftwWrapperCoverageTest, PTEST_EXECUTE_THIN_2D_1XN)
{
    using F = FftwTypes<TypeParam>;
    const int N = 16;
    auto *in = F::alloc_complex(N);
    auto *out = F::alloc_complex(N);
    auto *ref = F::alloc_complex(N);
    this->init_complex(in, N);
    auto p = F::plan_dft_2d(
        1, N, in, out, FFTW_FORWARD, FFTW_ESTIMATE);
    ASSERT_NE(p, nullptr);
    F::execute(p);
    dft_reference_2d(in, ref, 1, N, FFTW_FORWARD);
    compare_complex_arrays(ref, out, N, dft_tolerance<F>(N));
    F::destroy_plan(p);
    F::free_fn(in);
    F::free_fn(out);
    F::free_fn(ref);
}

TYPED_TEST(FftwWrapperCoverageTest, PTEST_EXECUTE_THIN_2D_NX1)
{
    using F = FftwTypes<TypeParam>;
    const int N = 16;
    auto *in = F::alloc_complex(N);
    auto *out = F::alloc_complex(N);
    auto *ref = F::alloc_complex(N);
    this->init_complex(in, N);
    auto p = F::plan_dft_2d(
        N, 1, in, out, FFTW_FORWARD, FFTW_ESTIMATE);
    ASSERT_NE(p, nullptr);
    F::execute(p);
    dft_reference_2d(in, ref, N, 1, FFTW_FORWARD);
    compare_complex_arrays(ref, out, N, dft_tolerance<F>(N));
    F::destroy_plan(p);
    F::free_fn(in);
    F::free_fn(out);
    F::free_fn(ref);
}

/* ===== Section 4 -- Plan flag variants (TYPED_TEST) ===== */

TYPED_TEST(FftwWrapperCoverageTest, PTEST_PLAN_DFT_1D_MEASURE)
{
    using F = FftwTypes<TypeParam>;
    auto *in = F::alloc_complex(16);
    auto *out = F::alloc_complex(16);
    auto p = F::plan_dft_1d(
        16, in, out, FFTW_FORWARD, FFTW_MEASURE);
    ASSERT_NE(p, nullptr);
    F::destroy_plan(p);
    F::free_fn(in);
    F::free_fn(out);
}

TYPED_TEST(FftwWrapperCoverageTest, PTEST_PLAN_DFT_1D_PATIENT)
{
    using F = FftwTypes<TypeParam>;
    auto *in = F::alloc_complex(16);
    auto *out = F::alloc_complex(16);
    auto p = F::plan_dft_1d(
        16, in, out, FFTW_FORWARD, FFTW_PATIENT);
    ASSERT_NE(p, nullptr);
    F::destroy_plan(p);
    F::free_fn(in);
    F::free_fn(out);
}

TYPED_TEST(FftwWrapperCoverageTest, PTEST_PLAN_DFT_1D_EXHAUSTIVE)
{
    using F = FftwTypes<TypeParam>;
    auto *in = F::alloc_complex(16);
    auto *out = F::alloc_complex(16);
    auto p = F::plan_dft_1d(
        16, in, out, FFTW_FORWARD, FFTW_EXHAUSTIVE);
    ASSERT_NE(p, nullptr);
    F::destroy_plan(p);
    F::free_fn(in);
    F::free_fn(out);
}

/* ===== Section 5 -- State transition (TYPED_TEST) ===== */

TYPED_TEST(FftwWrapperCoverageTest, PTEST_DESTROY_NULL_PLAN)
{
    using F = FftwTypes<TypeParam>;
    F::destroy_plan(nullptr);
}

TYPED_TEST(FftwWrapperCoverageTest, PTEST_EXECUTE_NULL_PLAN)
{
    using F = FftwTypes<TypeParam>;
    /* Executing a NULL plan must be a safe no-op (not use-after-destroy). */
    F::execute(nullptr);
}
