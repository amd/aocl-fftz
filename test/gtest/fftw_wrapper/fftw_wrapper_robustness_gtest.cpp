// Copyright Advanced Micro Devices, Inc.
// SPDX-License-Identifier: BSD-3-Clause

/**
 * @file fftw_wrapper_robustness_gtest.cpp
 * @brief Negative, corner, and crash scenarios for the FFTW wrapper.
 */

#include <climits>
#include <cstdint>
#include "fftw_wrapper_test_utils.h"

/* AddressSanitizer intercepts oversized allocations and ABORTS the process
 * ("allocation-size-too-big") instead of letting the allocator return NULL.
 * That masks the graceful-NULL behavior the SIZE_MAX/huge allocation tests
 * verify, so those specific tests must be skipped when built with ASAN. */
#if defined(__SANITIZE_ADDRESS__)
#  define FFTW_WRAPPER_BUILT_WITH_ASAN 1
#elif defined(__has_feature)
#  if __has_feature(address_sanitizer)
#    define FFTW_WRAPPER_BUILT_WITH_ASAN 1
#  endif
#endif

#ifdef FFTW_WRAPPER_BUILT_WITH_ASAN
#  define SKIP_OVERSIZED_ALLOC_UNDER_ASAN()                                   \
    GTEST_SKIP() << "Skipped under AddressSanitizer: ASAN aborts on "         \
                    "oversized allocations instead of returning NULL, "       \
                    "masking the graceful-failure behavior under test."
#else
#  define SKIP_OVERSIZED_ALLOC_UNDER_ASAN() ((void)0)
#endif

template<typename T>
class FftwWrapperRobustnessTest : public FftwWrapperTestBase<T>
{
};

TYPED_TEST_SUITE(FftwWrapperRobustnessTest, FftwTestTypes);

/* ===== Section 1 -- Robustness: NULL pointer inputs (TYPED_TEST) ===== */
/* R2C with NULL complex output */
TYPED_TEST(FftwWrapperRobustnessTest, PTEST_PLAN_DFT_R2C_1D_NULL_OUTPUT)
{
    using F = FftwTypes<TypeParam>;
    auto *in = F::alloc_real(16);
    auto p = F::plan_dft_r2c_1d(16, in, nullptr, FFTW_ESTIMATE);
    EXPECT_NE(p, nullptr);
    if (p)
    {
        /* fftw_execute returns void; no segfault is expected. */
        F::execute(p);
        F::destroy_plan(p);
    }
    F::free_fn(in);
}

/* C2R with NULL complex input */
TYPED_TEST(FftwWrapperRobustnessTest, PTEST_PLAN_DFT_C2R_1D_NULL_INPUT)
{
    using F = FftwTypes<TypeParam>;
    auto *out = F::alloc_real(16);
    auto p = F::plan_dft_c2r_1d(16, nullptr, out, FFTW_ESTIMATE);
    EXPECT_NE(p, nullptr);
    if (p)
    {
        /* fftw_execute returns void; no segfault is expected. */
        F::execute(p);
        F::destroy_plan(p);
    }
    F::free_fn(out);
}

/* In-place C2C planned with both buffers NULL. */
TYPED_TEST(FftwWrapperRobustnessTest,
           PTEST_EXECUTE_DFT_INPLACE_NULL_BUFFERS_AT_SETUP)
{
    using F = FftwTypes<TypeParam>;
    const int N = 16;
    auto *buf = F::alloc_complex(N);
    auto *expected_output = F::alloc_complex(N);

    this->init_complex(buf, N);
    /* Compute the expected output before in-place execution overwrites the input. */
    dft_reference_1d(buf, expected_output, N, FFTW_FORWARD);

    /* NULL == NULL, so the plan is in-place */
    auto p = F::plan_dft_1d(N, nullptr, nullptr, FFTW_FORWARD, FFTW_ESTIMATE);
    ASSERT_NE(p, nullptr);

    F::execute_dft(p, buf, buf);
    compare_complex_arrays(expected_output, buf, N, dft_tolerance<F>(N));

    F::destroy_plan(p);
    F::free_fn(buf);
    F::free_fn(expected_output);
}

/* Out-of-place C2C planned with a NULL output. */
TYPED_TEST(FftwWrapperRobustnessTest,
           PTEST_EXECUTE_DFT_OUTOFPLACE_NULL_OUTPUT_AT_SETUP)
{
    using F = FftwTypes<TypeParam>;
    const int N = 16;
    auto *in = F::alloc_complex(N);
    auto *out = F::alloc_complex(N);
    auto *expected_output = F::alloc_complex(N);

    this->init_complex(in, N);
    /* Compute the expected output using the reference DFT. */
    dft_reference_1d(in, expected_output, N, FFTW_FORWARD);

    /* in != NULL; out == NULL, so the plan is out-of-place. */
    auto p = F::plan_dft_1d(N, in, nullptr, FFTW_FORWARD, FFTW_ESTIMATE);
    ASSERT_NE(p, nullptr);

    F::execute_dft(p, in, out);
    compare_complex_arrays(expected_output, out, N, dft_tolerance<F>(N));

    F::destroy_plan(p);
    F::free_fn(in);
    F::free_fn(out);
    F::free_fn(expected_output);
}

/* execute_dft_r2c with NULL plan - aoclfftz_execute_io guards NULL */
TYPED_TEST(FftwWrapperRobustnessTest, NTEST_EXECUTE_DFT_R2C_NULL_PLAN)
{
    using F = FftwTypes<TypeParam>;
    auto *in = F::alloc_real(16);
    auto *out = F::alloc_complex(16 / 2 + 1);
    F::execute_dft_r2c(nullptr, in, out);
    F::free_fn(in);
    F::free_fn(out);
}

/* execute_dft_c2r with NULL plan */
TYPED_TEST(FftwWrapperRobustnessTest, NTEST_EXECUTE_DFT_C2R_NULL_PLAN)
{
    using F = FftwTypes<TypeParam>;
    auto *in = F::alloc_complex(16 / 2 + 1);
    auto *out = F::alloc_real(16);
    F::execute_dft_c2r(nullptr, in, out);
    F::free_fn(in);
    F::free_fn(out);
}

/* execute_dft with valid plan but NULL buffers */
TYPED_TEST(FftwWrapperRobustnessTest, NTEST_EXECUTE_DFT_NULL_BUFFERS)
{
    using F = FftwTypes<TypeParam>;
    auto *in = F::alloc_complex(16);
    auto *out = F::alloc_complex(16);
    auto p = F::plan_dft_1d(16, in, out, FFTW_FORWARD, FFTW_ESTIMATE);
    ASSERT_NE(p, nullptr);
    F::execute_dft(p, nullptr, nullptr);
    F::destroy_plan(p);
    F::free_fn(in);
    F::free_fn(out);
}

/* ===== Section 2 -- BVA: Size boundaries (TYPED_TEST) ===== */

/* R2C zero size */
TYPED_TEST(FftwWrapperRobustnessTest, NTEST_PLAN_DFT_R2C_1D_ZERO_SIZE)
{
    using F = FftwTypes<TypeParam>;
    auto *in = F::alloc_real(1);
    auto *out = F::alloc_complex(1);
    auto p = F::plan_dft_r2c_1d(0, in, out, FFTW_ESTIMATE);
    EXPECT_EQ(p, nullptr);
    if (p)
    {
        F::destroy_plan(p);
    }
    F::free_fn(in);
    F::free_fn(out);
}

/* R2C negative size */
TYPED_TEST(FftwWrapperRobustnessTest, NTEST_PLAN_DFT_R2C_1D_NEGATIVE_SIZE)
{
    using F = FftwTypes<TypeParam>;
    auto *in = F::alloc_real(1);
    auto *out = F::alloc_complex(1);
    auto p = F::plan_dft_r2c_1d(-1, in, out, FFTW_ESTIMATE);
    EXPECT_EQ(p, nullptr);
    if (p)
    {
        F::destroy_plan(p);
    }
    F::free_fn(in);
    F::free_fn(out);
}

/* C2R zero size */
TYPED_TEST(FftwWrapperRobustnessTest, NTEST_PLAN_DFT_C2R_1D_ZERO_SIZE)
{
    using F = FftwTypes<TypeParam>;
    auto *in = F::alloc_complex(1);
    auto *out = F::alloc_real(1);
    auto p = F::plan_dft_c2r_1d(0, in, out, FFTW_ESTIMATE);
    EXPECT_EQ(p, nullptr);
    if (p)
    {
        F::destroy_plan(p);
    }
    F::free_fn(in);
    F::free_fn(out);
}

/* C2R negative size */
TYPED_TEST(FftwWrapperRobustnessTest, NTEST_PLAN_DFT_C2R_1D_NEGATIVE_SIZE)
{
    using F = FftwTypes<TypeParam>;
    auto *in = F::alloc_complex(1);
    auto *out = F::alloc_real(1);
    auto p = F::plan_dft_c2r_1d(-1, in, out, FFTW_ESTIMATE);
    EXPECT_EQ(p, nullptr);
    if (p)
    {
        F::destroy_plan(p);
    }
    F::free_fn(in);
    F::free_fn(out);
}

/* Smallest valid C2C: n=1 */
TYPED_TEST(FftwWrapperRobustnessTest, PTEST_PLAN_DFT_1D_SIZE_1)
{
    using F = FftwTypes<TypeParam>;
    auto *in = F::alloc_complex(1);
    auto *out = F::alloc_complex(1);
    auto p = F::plan_dft_1d(1, in, out, FFTW_FORWARD, FFTW_ESTIMATE);
    ASSERT_NE(p, nullptr);
    F::destroy_plan(p);
    F::free_fn(in);
    F::free_fn(out);
}

/* Smallest valid R2C: n=1 */
TYPED_TEST(FftwWrapperRobustnessTest, PTEST_PLAN_DFT_R2C_1D_SIZE_1)
{
    using F = FftwTypes<TypeParam>;
    auto *in = F::alloc_real(1);
    auto *out = F::alloc_complex(1);
    auto p = F::plan_dft_r2c_1d(1, in, out, FFTW_ESTIMATE);
    ASSERT_NE(p, nullptr);
    F::destroy_plan(p);
    F::free_fn(in);
    F::free_fn(out);
}

/* Smallest valid C2R: n=1 */
TYPED_TEST(FftwWrapperRobustnessTest, PTEST_PLAN_DFT_C2R_1D_SIZE_1)
{
    using F = FftwTypes<TypeParam>;
    auto *in = F::alloc_complex(1);
    auto *out = F::alloc_real(1);
    auto p = F::plan_dft_c2r_1d(1, in, out, FFTW_ESTIMATE);
    ASSERT_NE(p, nullptr);
    F::destroy_plan(p);
    F::free_fn(in);
    F::free_fn(out);
}

/* n=2 smallest even */
TYPED_TEST(FftwWrapperRobustnessTest, PTEST_PLAN_DFT_1D_SIZE_2)
{
    using F = FftwTypes<TypeParam>;
    auto *in = F::alloc_complex(2);
    auto *out = F::alloc_complex(2);
    auto p = F::plan_dft_1d(2, in, out, FFTW_FORWARD, FFTW_ESTIMATE);
    ASSERT_NE(p, nullptr);
    F::destroy_plan(p);
    F::free_fn(in);
    F::free_fn(out);
}

/* n=3 smallest odd prime */
TYPED_TEST(FftwWrapperRobustnessTest, PTEST_PLAN_DFT_1D_SIZE_3)
{
    using F = FftwTypes<TypeParam>;
    auto *in = F::alloc_complex(3);
    auto *out = F::alloc_complex(3);
    auto p = F::plan_dft_1d(3, in, out, FFTW_FORWARD, FFTW_ESTIMATE);
    ASSERT_NE(p, nullptr);
    F::destroy_plan(p);
    F::free_fn(in);
    F::free_fn(out);
}

/* ===== Section 3 -- BVA: Stride/batch boundaries (TYPED_TEST) ===== */

TYPED_TEST(FftwWrapperRobustnessTest, NTEST_MANY_DFT_NEGATIVE_ISTRIDE)
{
    using F = FftwTypes<TypeParam>;
    int n[] = {8};
    auto *in = F::alloc_complex(8);
    auto *out = F::alloc_complex(8);
    auto p = F::plan_many_dft(
        1, n, 1, in, nullptr, -1, 8,
        out, nullptr, 1, 8, FFTW_FORWARD, FFTW_ESTIMATE);
    /* A negative istride maps directly to dims[0].in_stride, which the
     * problem-descriptor validator rejects (in_stride must be > 0). */
    EXPECT_EQ(p, nullptr);
    if (p)
    {
        F::destroy_plan(p);
    }
    F::free_fn(in);
    F::free_fn(out);
}

TYPED_TEST(FftwWrapperRobustnessTest, NTEST_MANY_DFT_ZERO_ISTRIDE)
{
    using F = FftwTypes<TypeParam>;
    int n[] = {8};
    auto *in = F::alloc_complex(8);
    auto *out = F::alloc_complex(8);
    auto p = F::plan_many_dft(
        1, n, 1, in, nullptr, 0, 8,
        out, nullptr, 1, 8, FFTW_FORWARD, FFTW_ESTIMATE);
    /* istride == 0 is coerced to 1 by the translator (contiguous default),
     * so this is a VALID plan rather than a rejected one. */
    EXPECT_NE(p, nullptr);
    if (p)
    {
        F::destroy_plan(p);
    }
    F::free_fn(in);
    F::free_fn(out);
}

TYPED_TEST(FftwWrapperRobustnessTest, NTEST_MANY_DFT_NEGATIVE_IDIST)
{
    using F = FftwTypes<TypeParam>;
    int n[] = {8};
    auto *in = F::alloc_complex(8);
    auto *out = F::alloc_complex(8);
    auto p = F::plan_many_dft(
        1, n, 2, in, nullptr, 1, -1,
        out, nullptr, 1, 8, FFTW_FORWARD, FFTW_ESTIMATE);
    /* A negative idist maps to vecs[0].in_stride, which the validator
     * rejects (batch in_stride must be > 0). */
    EXPECT_EQ(p, nullptr);
    if (p)
    {
        F::destroy_plan(p);
    }
    F::free_fn(in);
    F::free_fn(out);
}

TYPED_TEST(FftwWrapperRobustnessTest, NTEST_MANY_DFT_ZERO_IDIST)
{
    using F = FftwTypes<TypeParam>;
    int n[] = {8};
    auto *in = F::alloc_complex(8);
    auto *out = F::alloc_complex(8);
    auto p = F::plan_many_dft(
        1, n, 2, in, nullptr, 1, 0,
        out, nullptr, 1, 8, FFTW_FORWARD, FFTW_ESTIMATE);
    /* idist == 0 is coerced to 1 by the translator, so this is a VALID
     * plan rather than a rejected one. */
    EXPECT_NE(p, nullptr);
    if (p)
    {
        F::destroy_plan(p);
    }
    F::free_fn(in);
    F::free_fn(out);
}

TYPED_TEST(FftwWrapperRobustnessTest, NTEST_MANY_DFT_NEGATIVE_HOWMANY)
{
    using F = FftwTypes<TypeParam>;
    int n[] = {8};
    auto *in = F::alloc_complex(8);
    auto *out = F::alloc_complex(8);
    auto p = F::plan_many_dft(
        1, n, -1, in, nullptr, 1, 8,
        out, nullptr, 1, 8, FFTW_FORWARD, FFTW_ESTIMATE);
    /* howmany maps to vecs[0].n; a negative batch count is rejected
     * (vec size must be >= 1). */
    EXPECT_EQ(p, nullptr);
    if (p)
    {
        F::destroy_plan(p);
    }
    F::free_fn(in);
    F::free_fn(out);
}

/* rank < 0 previously left dims[0] uninitialized -> could hang. Builders now
 * reject it up front -> deterministic NULL plan. */
TYPED_TEST(FftwWrapperRobustnessTest, NTEST_MANY_DFT_NEGATIVE_RANK)
{
    using F = FftwTypes<TypeParam>;
    const int n[] = {8};
    auto *in = F::alloc_complex(n[0]);
    auto *out = F::alloc_complex(n[0]);

    auto p = F::plan_many_dft(
        -1, n, 1, in, nullptr, 1, n[0],
        out, nullptr, 1, n[0], FFTW_FORWARD, FFTW_ESTIMATE);
    EXPECT_EQ(p, nullptr);

    if (p)
    {
        F::destroy_plan(p);
    }
    F::free_fn(in);
    F::free_fn(out);
}

/* NULL size array with positive rank -> builder rejects, NULL plan. */
TYPED_TEST(FftwWrapperRobustnessTest, NTEST_MANY_DFT_NULL_N)
{
    using F = FftwTypes<TypeParam>;
    auto *in = F::alloc_complex(8);
    auto *out = F::alloc_complex(8);
    auto p = F::plan_many_dft(
        1, nullptr, 1, in, nullptr, 1, 8,
        out, nullptr, 1, 8, FFTW_FORWARD, FFTW_ESTIMATE);
    EXPECT_EQ(p, nullptr);
    if (p)
    {
        F::destroy_plan(p);
    }
    F::free_fn(in);
    F::free_fn(out);
}

/* ===== Section 4 -- BVA: Guru iodim boundaries (TYPED_TEST) ===== */

TYPED_TEST(FftwWrapperRobustnessTest, NTEST_GURU_DFT_ZERO_N)
{
    using F = FftwTypes<TypeParam>;
    using iodim_t = typename F::iodim_t;
    iodim_t dims[1] = {{0, 1, 1}};
    iodim_t howmany[1] = {{1, 1, 1}};
    auto *in = F::alloc_complex(1);
    auto *out = F::alloc_complex(1);
    auto p = F::plan_guru_dft(
        1, dims, 1, howmany, in, out,
        FFTW_FORWARD, FFTW_ESTIMATE);
    /* dims[0].n == 0 -> rejected (dimension size must be >= 1). */
    EXPECT_EQ(p, nullptr);
    if (p)
    {
        F::destroy_plan(p);
    }
    F::free_fn(in);
    F::free_fn(out);
}

TYPED_TEST(FftwWrapperRobustnessTest, NTEST_GURU_DFT_NEGATIVE_N)
{
    using F = FftwTypes<TypeParam>;
    using iodim_t = typename F::iodim_t;
    iodim_t dims[1] = {{-1, 1, 1}};
    iodim_t howmany[1] = {{1, 1, 1}};
    auto *in = F::alloc_complex(1);
    auto *out = F::alloc_complex(1);
    auto p = F::plan_guru_dft(
        1, dims, 1, howmany, in, out,
        FFTW_FORWARD, FFTW_ESTIMATE);
    /* dims[0].n < 0 -> rejected (dimension size must be >= 1). */
    EXPECT_EQ(p, nullptr);
    if (p)
    {
        F::destroy_plan(p);
    }
    F::free_fn(in);
    F::free_fn(out);
}

TYPED_TEST(FftwWrapperRobustnessTest, NTEST_GURU_DFT_ZERO_STRIDES)
{
    using F = FftwTypes<TypeParam>;
    using iodim_t = typename F::iodim_t;
    iodim_t dims[1] = {{8, 0, 0}};
    iodim_t howmany[1] = {{1, 1, 1}};
    auto *in = F::alloc_complex(8);
    auto *out = F::alloc_complex(8);
    auto p = F::plan_guru_dft(
        1, dims, 1, howmany, in, out,
        FFTW_FORWARD, FFTW_ESTIMATE);
    /* The guru path copies iodim strides verbatim (no zero-coercion, unlike
     * plan_many), so a zero stride reaches the validator and is rejected. */
    EXPECT_EQ(p, nullptr);
    if (p)
    {
        F::destroy_plan(p);
    }
    F::free_fn(in);
    F::free_fn(out);
}

TYPED_TEST(FftwWrapperRobustnessTest, NTEST_GURU_DFT_NEGATIVE_STRIDES)
{
    using F = FftwTypes<TypeParam>;
    using iodim_t = typename F::iodim_t;
    iodim_t dims[1] = {{8, -1, -1}};
    iodim_t howmany[1] = {{1, 1, 1}};
    auto *in = F::alloc_complex(8);
    auto *out = F::alloc_complex(8);
    auto p = F::plan_guru_dft(
        1, dims, 1, howmany, in, out,
        FFTW_FORWARD, FFTW_ESTIMATE);
    /* Negative guru strides reach the validator and are rejected. */
    EXPECT_EQ(p, nullptr);
    if (p)
    {
        F::destroy_plan(p);
    }
    F::free_fn(in);
    F::free_fn(out);
}

TYPED_TEST(FftwWrapperRobustnessTest, NTEST_GURU_DFT_NEGATIVE_HOWMANY_RANK)
{
    using F = FftwTypes<TypeParam>;
    using iodim_t = typename F::iodim_t;
    iodim_t dims[1] = {{8, 1, 1}};
    auto *in = F::alloc_complex(8);
    auto *out = F::alloc_complex(8);
    auto p = F::plan_guru_dft(
        1, dims, -1, nullptr, in, out,
        FFTW_FORWARD, FFTW_ESTIMATE);
    /* howmany_rank <= 0 is treated as "no batch" (vec_rank=1, unit vec) by
     * the translator, so this yields a VALID single-transform plan. */
    EXPECT_NE(p, nullptr);
    if (p)
    {
        F::destroy_plan(p);
    }
    F::free_fn(in);
    F::free_fn(out);
}

/* NULL dims array with positive rank -> guru builder rejects, NULL plan. */
TYPED_TEST(FftwWrapperRobustnessTest, NTEST_GURU_DFT_NULL_DIMS)
{
    using F = FftwTypes<TypeParam>;
    using iodim_t = typename F::iodim_t;
    iodim_t howmany[1] = {{1, 1, 1}};
    auto *in = F::alloc_complex(8);
    auto *out = F::alloc_complex(8);
    auto p = F::plan_guru_dft(
        1, nullptr, 1, howmany, in, out,
        FFTW_FORWARD, FFTW_ESTIMATE);
    EXPECT_EQ(p, nullptr);
    if (p)
    {
        F::destroy_plan(p);
    }
    F::free_fn(in);
    F::free_fn(out);
}

/* NULL howmany_dims with positive howmany_rank -> NULL plan. */
TYPED_TEST(FftwWrapperRobustnessTest, NTEST_GURU_DFT_NULL_HOWMANY_DIMS)
{
    using F = FftwTypes<TypeParam>;
    using iodim_t = typename F::iodim_t;
    iodim_t dims[1] = {{8, 1, 1}};
    auto *in = F::alloc_complex(8);
    auto *out = F::alloc_complex(8);
    auto p = F::plan_guru_dft(
        1, dims, 1, nullptr, in, out,
        FFTW_FORWARD, FFTW_ESTIMATE);
    EXPECT_EQ(p, nullptr);
    if (p)
    {
        F::destroy_plan(p);
    }
    F::free_fn(in);
    F::free_fn(out);
}

/* ===== Section 5 -- Robustness: Memory (non-typed TEST_F) ===== */

class FftwWrapperMemRobustnessTest : public ::testing::Test
{
};

TEST_F(FftwWrapperMemRobustnessTest, NTEST_MALLOC_SIZE_MAX)
{
    SKIP_OVERSIZED_ALLOC_UNDER_ASAN();
    void *p = fftw_malloc(SIZE_MAX);
    EXPECT_EQ(p, nullptr);
    if (p)
    {
        fftw_free(p);
    }
}

TEST_F(FftwWrapperMemRobustnessTest, NTEST_MALLOC_HUGE)
{
    SKIP_OVERSIZED_ALLOC_UNDER_ASAN();
    void *p = fftw_malloc(SIZE_MAX / 2);
    EXPECT_EQ(p, nullptr);
    if (p)
    {
        fftw_free(p);
    }
}

TEST_F(FftwWrapperMemRobustnessTest, NTEST_ALLOC_REAL_SIZE_MAX)
{
    SKIP_OVERSIZED_ALLOC_UNDER_ASAN();
    double *p = fftw_alloc_real(SIZE_MAX);
    EXPECT_EQ(p, nullptr);
    if (p)
    {
        fftw_free(p);
    }
}

TEST_F(FftwWrapperMemRobustnessTest, NTEST_ALLOC_COMPLEX_SIZE_MAX)
{
    SKIP_OVERSIZED_ALLOC_UNDER_ASAN();
    fftw_complex *p = fftw_alloc_complex(SIZE_MAX);
    EXPECT_EQ(p, nullptr);
    if (p)
    {
        fftw_free(p);
    }
}

TEST_F(FftwWrapperMemRobustnessTest, NTEST_ALIGNMENT_OF_NULL)
{
    int r = fftw_alignment_of(nullptr);
    EXPECT_EQ(r, 0);
}

TEST_F(FftwWrapperMemRobustnessTest, NTEST_MALLOC_F_SIZE_MAX)
{
    SKIP_OVERSIZED_ALLOC_UNDER_ASAN();
    void *p = fftwf_malloc(SIZE_MAX);
    EXPECT_EQ(p, nullptr);
    if (p)
    {
        fftwf_free(p);
    }
}

TEST_F(FftwWrapperMemRobustnessTest, NTEST_MALLOC_F_HUGE)
{
    SKIP_OVERSIZED_ALLOC_UNDER_ASAN();
    void *p = fftwf_malloc(SIZE_MAX / 2);
    EXPECT_EQ(p, nullptr);
    if (p)
    {
        fftwf_free(p);
    }
}

TEST_F(FftwWrapperMemRobustnessTest, NTEST_ALLOC_REAL_F_SIZE_MAX)
{
    SKIP_OVERSIZED_ALLOC_UNDER_ASAN();
    float *p = fftwf_alloc_real(SIZE_MAX);
    EXPECT_EQ(p, nullptr);
    if (p)
    {
        fftwf_free(p);
    }
}

TEST_F(FftwWrapperMemRobustnessTest, NTEST_ALLOC_COMPLEX_F_SIZE_MAX)
{
    SKIP_OVERSIZED_ALLOC_UNDER_ASAN();
    fftwf_complex *p = fftwf_alloc_complex(SIZE_MAX);
    EXPECT_EQ(p, nullptr);
    if (p)
    {
        fftwf_free(p);
    }
}

/* Overflow boundary: counts above SIZE_MAX/sizeof(element) are rejected (NULL)
 * instead of wrapping small and under-allocating. At-limit is valid but huge,
 * so the OS refuses it. Under ASAN, skip the at-limit alloc (ASAN aborts on
 * impossibly large requests). */
TEST_F(FftwWrapperMemRobustnessTest,
       NTEST_ALLOC_REAL_MULTIPLICATION_BOUNDARY)
{
    constexpr size_t max_count = SIZE_MAX / sizeof(double);
#ifndef FFTW_WRAPPER_BUILT_WITH_ASAN
    double *at_limit = fftw_alloc_real(max_count);
    EXPECT_EQ(at_limit, nullptr);
    if (at_limit)
    {
        fftw_free(at_limit);
    }
#endif
    double *overflow = fftw_alloc_real(max_count + 1);
    EXPECT_EQ(overflow, nullptr);
    if (overflow)
    {
        fftw_free(overflow);
    }
}

TEST_F(FftwWrapperMemRobustnessTest,
       NTEST_ALLOC_COMPLEX_MULTIPLICATION_BOUNDARY)
{
    constexpr size_t max_count = SIZE_MAX / sizeof(fftw_complex);
#ifndef FFTW_WRAPPER_BUILT_WITH_ASAN
    fftw_complex *at_limit = fftw_alloc_complex(max_count);
    EXPECT_EQ(at_limit, nullptr);
    if (at_limit)
    {
        fftw_free(at_limit);
    }
#endif
    fftw_complex *overflow = fftw_alloc_complex(max_count + 1);
    EXPECT_EQ(overflow, nullptr);
    if (overflow)
    {
        fftw_free(overflow);
    }
}

TEST_F(FftwWrapperMemRobustnessTest,
       NTEST_ALLOC_REAL_F_MULTIPLICATION_BOUNDARY)
{
    constexpr size_t max_count = SIZE_MAX / sizeof(float);
#ifndef FFTW_WRAPPER_BUILT_WITH_ASAN
    float *at_limit = fftwf_alloc_real(max_count);
    EXPECT_EQ(at_limit, nullptr);
    if (at_limit)
    {
        fftwf_free(at_limit);
    }
#endif
    float *overflow = fftwf_alloc_real(max_count + 1);
    EXPECT_EQ(overflow, nullptr);
    if (overflow)
    {
        fftwf_free(overflow);
    }
}

TEST_F(FftwWrapperMemRobustnessTest,
       NTEST_ALLOC_COMPLEX_F_MULTIPLICATION_BOUNDARY)
{
    constexpr size_t max_count = SIZE_MAX / sizeof(fftwf_complex);
#ifndef FFTW_WRAPPER_BUILT_WITH_ASAN
    fftwf_complex *at_limit = fftwf_alloc_complex(max_count);
    EXPECT_EQ(at_limit, nullptr);
    if (at_limit)
    {
        fftwf_free(at_limit);
    }
#endif
    fftwf_complex *overflow = fftwf_alloc_complex(max_count + 1);
    EXPECT_EQ(overflow, nullptr);
    if (overflow)
    {
        fftwf_free(overflow);
    }
}

TEST_F(FftwWrapperMemRobustnessTest, NTEST_ALIGNMENT_OF_F_NULL)
{
    int r = fftwf_alignment_of(nullptr);
    EXPECT_EQ(r, 0);
}

/* ===== Section 6 -- BVA: Threading (non-typed TEST_F) ===== */

TEST_F(FftwWrapperMemRobustnessTest, NTEST_PLAN_WITH_NTHREADS_ZERO)
{
    EXPECT_EQ(fftw_init_threads(), 1);
    fftw_plan_with_nthreads(0);
    EXPECT_EQ(fftw_planner_nthreads(), 0);
    fftw_cleanup_threads();
}

TEST_F(FftwWrapperMemRobustnessTest, NTEST_PLAN_WITH_NTHREADS_NEGATIVE)
{
    EXPECT_EQ(fftw_init_threads(), 1);
    fftw_plan_with_nthreads(-1);
    EXPECT_EQ(fftw_planner_nthreads(), -1);
    fftw_cleanup_threads();
}

TEST_F(FftwWrapperMemRobustnessTest, NTEST_PLAN_WITH_NTHREADS_HUGE)
{
    EXPECT_EQ(fftw_init_threads(), 1);
    fftw_plan_with_nthreads(INT_MAX);
    EXPECT_EQ(fftw_planner_nthreads(), INT_MAX);
    fftw_cleanup_threads();
}

TEST_F(FftwWrapperMemRobustnessTest, NTEST_CLEANUP_THREADS_WITHOUT_INIT)
{
    fftw_cleanup_threads();
    EXPECT_EQ(fftw_planner_nthreads(), 1);
}

TEST_F(FftwWrapperMemRobustnessTest, NTEST_PLAN_WITH_NTHREADS_AFTER_CLEANUP)
{
    EXPECT_EQ(fftw_init_threads(), 1);
    fftw_plan_with_nthreads(4);
    fftw_cleanup_threads();
    fftw_plan_with_nthreads(8);
    EXPECT_EQ(fftw_planner_nthreads(), 8);
    fftw_cleanup_threads();
}

/* ===== Section 7 -- Wisdom with NULL (non-typed TEST_F) ===== */

TEST_F(FftwWrapperMemRobustnessTest, NTEST_EXPORT_WISDOM_TO_FILENAME_NULL)
{
    int r = fftw_export_wisdom_to_filename(nullptr);
    EXPECT_EQ(r, 1);
}

TEST_F(FftwWrapperMemRobustnessTest, NTEST_IMPORT_WISDOM_FROM_FILENAME_NULL)
{
    int r = fftw_import_wisdom_from_filename(nullptr);
    EXPECT_EQ(r, 0);
}

TEST_F(FftwWrapperMemRobustnessTest, NTEST_IMPORT_WISDOM_FROM_STRING_NULL)
{
    int r = fftw_import_wisdom_from_string(nullptr);
    EXPECT_EQ(r, 0);
}

TEST_F(FftwWrapperMemRobustnessTest, NTEST_EXPORT_WISDOM_TO_FILE_NULL)
{
    fftw_export_wisdom_to_file(nullptr);
}

TEST_F(FftwWrapperMemRobustnessTest, NTEST_IMPORT_WISDOM_FROM_FILE_NULL)
{
    int r = fftw_import_wisdom_from_file(nullptr);
    EXPECT_EQ(r, 0);
}

/* ===== Section 8 -- Misc with NULL plan (non-typed TEST_F) ===== */

TEST_F(FftwWrapperMemRobustnessTest, NTEST_FLOPS_NULL_PLAN)
{
    double add = 1, mul = 1, fma = 1;
    fftw_flops(nullptr, &add, &mul, &fma);
}

TEST_F(FftwWrapperMemRobustnessTest, NTEST_FLOPS_NULL_PTRS)
{
    fftw_complex *in = fftw_alloc_complex(8);
    fftw_complex *out = fftw_alloc_complex(8);
    fftw_plan p = fftw_plan_dft_1d(
        8, in, out, FFTW_FORWARD, FFTW_ESTIMATE);
    if (p)
    {
        fftw_flops(p, nullptr, nullptr, nullptr);
        fftw_destroy_plan(p);
    }
    fftw_free(in);
    fftw_free(out);
}

TEST_F(FftwWrapperMemRobustnessTest, NTEST_ESTIMATE_COST_NULL)
{
    double c = fftw_estimate_cost(nullptr);
    EXPECT_EQ(c, 0.0);
}

TEST_F(FftwWrapperMemRobustnessTest, NTEST_COST_NULL)
{
    double c = fftw_cost(nullptr);
    EXPECT_EQ(c, 0.0);
}

TEST_F(FftwWrapperMemRobustnessTest, NTEST_PRINT_PLAN_NULL)
{
    fftw_print_plan(nullptr);
}

TEST_F(FftwWrapperMemRobustnessTest, NTEST_FPRINT_PLAN_NULL)
{
    fftw_fprint_plan(nullptr, stderr);
}

TEST_F(FftwWrapperMemRobustnessTest, NTEST_FPRINT_PLAN_NULL_FILE)
{
    fftw_complex *in = fftw_alloc_complex(8);
    fftw_complex *out = fftw_alloc_complex(8);
    fftw_plan p = fftw_plan_dft_1d(
        8, in, out, FFTW_FORWARD, FFTW_ESTIMATE);
    if (p)
    {
        fftw_fprint_plan(p, nullptr);
        fftw_destroy_plan(p);
    }
    fftw_free(in);
    fftw_free(out);
}

TEST_F(FftwWrapperMemRobustnessTest, NTEST_SPRINT_PLAN_NULL)
{
    char *s = fftw_sprint_plan(nullptr);
    EXPECT_NE(s, nullptr);
    if (s) fftw_free(s);
}

/* ===== Section 9 -- Float variants: Wisdom NULL ===== */

TEST_F(FftwWrapperMemRobustnessTest, NTEST_EXPORT_WISDOM_TO_FILENAME_F_NULL)
{
    int r = fftwf_export_wisdom_to_filename(nullptr);
    EXPECT_EQ(r, 1);
}

TEST_F(FftwWrapperMemRobustnessTest, NTEST_IMPORT_WISDOM_FROM_FILENAME_F_NULL)
{
    int r = fftwf_import_wisdom_from_filename(nullptr);
    EXPECT_EQ(r, 0);
}

TEST_F(FftwWrapperMemRobustnessTest, NTEST_IMPORT_WISDOM_FROM_STRING_F_NULL)
{
    int r = fftwf_import_wisdom_from_string(nullptr);
    EXPECT_EQ(r, 0);
}

TEST_F(FftwWrapperMemRobustnessTest, NTEST_EXPORT_WISDOM_TO_FILE_F_NULL)
{
    fftwf_export_wisdom_to_file(nullptr);
}

TEST_F(FftwWrapperMemRobustnessTest, NTEST_IMPORT_WISDOM_FROM_FILE_F_NULL)
{
    int r = fftwf_import_wisdom_from_file(nullptr);
    EXPECT_EQ(r, 0);
}

/* ===== Section 10 -- Float variants: Misc NULL ===== */

TEST_F(FftwWrapperMemRobustnessTest, NTEST_FLOPS_F_NULL_PLAN)
{
    double add = 1, mul = 1, fma = 1;
    fftwf_flops(nullptr, &add, &mul, &fma);
}

TEST_F(FftwWrapperMemRobustnessTest, NTEST_ESTIMATE_COST_F_NULL)
{
    double c = fftwf_estimate_cost(nullptr);
    EXPECT_EQ(c, 0.0);
}

TEST_F(FftwWrapperMemRobustnessTest, NTEST_COST_F_NULL)
{
    double c = fftwf_cost(nullptr);
    EXPECT_EQ(c, 0.0);
}

TEST_F(FftwWrapperMemRobustnessTest, NTEST_PRINT_PLAN_F_NULL)
{
    fftwf_print_plan(nullptr);
}

TEST_F(FftwWrapperMemRobustnessTest, NTEST_FPRINT_PLAN_F_NULL)
{
    fftwf_fprint_plan(nullptr, stderr);
}

TEST_F(FftwWrapperMemRobustnessTest, NTEST_FPRINT_PLAN_F_NULL_FILE)
{
    fftwf_complex *in = fftwf_alloc_complex(8);
    fftwf_complex *out = fftwf_alloc_complex(8);
    fftwf_plan p = fftwf_plan_dft_1d(
        8, in, out, FFTW_FORWARD, FFTW_ESTIMATE);
    if (p)
    {
        fftwf_fprint_plan(p, nullptr);
        fftwf_destroy_plan(p);
    }
    fftwf_free(in);
    fftwf_free(out);
}

/* R2R stubs must return NULL (unsupported) without crashing. */
TEST_F(FftwWrapperMemRobustnessTest, NTEST_R2R_STUBS_RETURN_NULL_D)
{
    double buf[16] = {0};
    const int n = 8;
    fftw_r2r_kind kind = FFTW_R2HC;

    EXPECT_EQ(fftw_plan_r2r_1d(n, buf, buf, kind, FFTW_ESTIMATE), nullptr);
    EXPECT_EQ(fftw_plan_r2r_2d(n, n, buf, buf, kind, kind, FFTW_ESTIMATE),
              nullptr);
    EXPECT_EQ(fftw_plan_r2r_3d(n, n, n, buf, buf, kind, kind, kind,
                               FFTW_ESTIMATE), nullptr);
    EXPECT_EQ(fftw_plan_r2r(1, &n, buf, buf, &kind, FFTW_ESTIMATE), nullptr);
    EXPECT_EQ(fftw_plan_many_r2r(1, &n, 1, buf, nullptr, 1, n,
                                 buf, nullptr, 1, n, &kind, FFTW_ESTIMATE),
              nullptr);
}

TEST_F(FftwWrapperMemRobustnessTest, NTEST_R2R_STUBS_RETURN_NULL_F)
{
    float buf[16] = {0};
    const int n = 8;
    fftwf_r2r_kind kind = FFTW_R2HC;

    EXPECT_EQ(fftwf_plan_r2r_1d(n, buf, buf, kind, FFTW_ESTIMATE), nullptr);
    EXPECT_EQ(fftwf_plan_r2r_2d(n, n, buf, buf, kind, kind, FFTW_ESTIMATE),
              nullptr);
    EXPECT_EQ(fftwf_plan_r2r_3d(n, n, n, buf, buf, kind, kind, kind,
                                FFTW_ESTIMATE), nullptr);
    EXPECT_EQ(fftwf_plan_r2r(1, &n, buf, buf, &kind, FFTW_ESTIMATE), nullptr);
    EXPECT_EQ(fftwf_plan_many_r2r(1, &n, 1, buf, nullptr, 1, n,
                                  buf, nullptr, 1, n, &kind, FFTW_ESTIMATE),
              nullptr);
}
