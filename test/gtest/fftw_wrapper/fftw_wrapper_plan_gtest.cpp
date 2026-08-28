// Copyright Advanced Micro Devices, Inc.
// SPDX-License-Identifier: BSD-3-Clause

/** @file fftw_wrapper_plan_gtest.cpp
 * @brief GTest cases for FFTW wrapper plan creation APIs.
 */

#include "fftw_wrapper_test_utils.h"

template<typename T>
class FftwWrapperPlanTest : public FftwWrapperTestBase<T>
{
};

TYPED_TEST_SUITE(FftwWrapperPlanTest, FftwTestTypes);

/* ===== C2C basic plan creation ===== */

TYPED_TEST(FftwWrapperPlanTest, PTEST_PLAN_DFT_1D_FORWARD)
{
    using F = FftwTypes<TypeParam>;
    const int N = 16;
    auto *in  = F::alloc_complex(N);
    auto *out = F::alloc_complex(N);
    auto p = F::plan_dft_1d(N, in, out, FFTW_FORWARD, FFTW_ESTIMATE);
    ASSERT_NE(p, nullptr);
    F::destroy_plan(p);
    F::free_fn(in);
    F::free_fn(out);
}

TYPED_TEST(FftwWrapperPlanTest, PTEST_PLAN_DFT_1D_BACKWARD)
{
    using F = FftwTypes<TypeParam>;
    const int N = 16;
    auto *in  = F::alloc_complex(N);
    auto *out = F::alloc_complex(N);
    auto p = F::plan_dft_1d(N, in, out, FFTW_BACKWARD, FFTW_ESTIMATE);
    ASSERT_NE(p, nullptr);
    F::destroy_plan(p);
    F::free_fn(in);
    F::free_fn(out);
}

TYPED_TEST(FftwWrapperPlanTest, PTEST_PLAN_DFT_2D)
{
    using F = FftwTypes<TypeParam>;
    const int n0 = 4, n1 = 8;
    auto *in  = F::alloc_complex(n0 * n1);
    auto *out = F::alloc_complex(n0 * n1);
    auto p = F::plan_dft_2d(n0, n1, in, out, FFTW_FORWARD, FFTW_ESTIMATE);
    ASSERT_NE(p, nullptr);
    F::destroy_plan(p);
    F::free_fn(in);
    F::free_fn(out);
}

TYPED_TEST(FftwWrapperPlanTest, PTEST_PLAN_DFT_3D)
{
    using F = FftwTypes<TypeParam>;
    const int n0 = 2, n1 = 3, n2 = 4;
    auto *in  = F::alloc_complex(n0 * n1 * n2);
    auto *out = F::alloc_complex(n0 * n1 * n2);
    auto p = F::plan_dft_3d(n0, n1, n2, in, out, FFTW_FORWARD, FFTW_ESTIMATE);
    ASSERT_NE(p, nullptr);
    F::destroy_plan(p);
    F::free_fn(in);
    F::free_fn(out);
}

TYPED_TEST(FftwWrapperPlanTest, PTEST_PLAN_DFT_ND)
{
    using F = FftwTypes<TypeParam>;
    int n[] = {2, 3, 4};
    int total = 2 * 3 * 4;
    auto *in  = F::alloc_complex(total);
    auto *out = F::alloc_complex(total);
    auto p = F::plan_dft(3, n, in, out, FFTW_FORWARD, FFTW_ESTIMATE);
    ASSERT_NE(p, nullptr);
    F::destroy_plan(p);
    F::free_fn(in);
    F::free_fn(out);
}

TYPED_TEST(FftwWrapperPlanTest, PTEST_PLAN_DFT_1D_INPLACE)
{
    using F = FftwTypes<TypeParam>;
    const int N = 32;
    auto *buf = F::alloc_complex(N);
    auto p = F::plan_dft_1d(N, buf, buf, FFTW_FORWARD, FFTW_ESTIMATE);
    ASSERT_NE(p, nullptr);
    F::destroy_plan(p);
    F::free_fn(buf);
}

TYPED_TEST(FftwWrapperPlanTest, PTEST_PLAN_DFT_1D_OUT_OF_PLACE)
{
    using F = FftwTypes<TypeParam>;
    const int N = 32;
    auto *in  = F::alloc_complex(N);
    auto *out = F::alloc_complex(N);
    auto p = F::plan_dft_1d(N, in, out, FFTW_FORWARD, FFTW_ESTIMATE);
    ASSERT_NE(p, nullptr);
    F::destroy_plan(p);
    F::free_fn(in);
    F::free_fn(out);
}

TYPED_TEST(FftwWrapperPlanTest, PTEST_PLAN_DFT_1D_ALL_FLAGS)
{
    using F = FftwTypes<TypeParam>;
    const int N = 16;
    auto *in  = F::alloc_complex(N);
    auto *out = F::alloc_complex(N);

    unsigned flag_list[] = {FFTW_ESTIMATE, FFTW_MEASURE, FFTW_PATIENT,
                            FFTW_EXHAUSTIVE, FFTW_DESTROY_INPUT,
                            FFTW_PRESERVE_INPUT, FFTW_UNALIGNED};
    for (unsigned fl : flag_list)
    {
        auto p = F::plan_dft_1d(N, in, out, FFTW_FORWARD, fl);
        EXPECT_NE(p, nullptr) << "flag = " << fl;
        if (p)
        {
            F::destroy_plan(p);
        }
    }
    F::free_fn(in);
    F::free_fn(out);
}

TYPED_TEST(FftwWrapperPlanTest, PTEST_PLAN_DFT_1D_VARIOUS_SIZES)
{
    using F = FftwTypes<TypeParam>;
    for (int N : COMMON_SIZES)
    {
        auto *in  = F::alloc_complex(N);
        auto *out = F::alloc_complex(N);
        auto p = F::plan_dft_1d(N, in, out, FFTW_FORWARD, FFTW_ESTIMATE);
        EXPECT_NE(p, nullptr) << "N = " << N;
        if (p)
        {
            F::destroy_plan(p);
        }
        F::free_fn(in);
        F::free_fn(out);
    }
}

/* ===== Many / Advanced interface ===== */

TYPED_TEST(FftwWrapperPlanTest, PTEST_PLAN_MANY_DFT)
{
    using F = FftwTypes<TypeParam>;
    const int N = 16, howmany = 4;
    auto *in  = F::alloc_complex(N * howmany);
    auto *out = F::alloc_complex(N * howmany);
    int n[] = {N};
    auto p = F::plan_many_dft(1, n, howmany, in, nullptr, 1, N,
                              out, nullptr, 1, N, FFTW_FORWARD, FFTW_ESTIMATE);
    ASSERT_NE(p, nullptr);
    F::destroy_plan(p);
    F::free_fn(in);
    F::free_fn(out);
}

TYPED_TEST(FftwWrapperPlanTest, PTEST_PLAN_MANY_DFT_INTERLEAVED)
{
    using F = FftwTypes<TypeParam>;
    const int N = 8, howmany = 3;
    auto *in  = F::alloc_complex(N * howmany);
    auto *out = F::alloc_complex(N * howmany);
    int n[] = {N};
    auto p = F::plan_many_dft(1, n, howmany, in, nullptr, howmany, 1,
                              out, nullptr, howmany, 1,
                              FFTW_FORWARD, FFTW_ESTIMATE);
    ASSERT_NE(p, nullptr);
    F::destroy_plan(p);
    F::free_fn(in);
    F::free_fn(out);
}

/* ===== Guru interface ===== */

TYPED_TEST(FftwWrapperPlanTest, PTEST_PLAN_GURU_DFT)
{
    using F = FftwTypes<TypeParam>;
    using iodim_t = typename F::iodim_t;
    const int N = 16;
    iodim_t dims[1] = {{N, 1, 1}};
    iodim_t howmany_dims[1] = {{1, 1, 1}};
    auto *in  = F::alloc_complex(N);
    auto *out = F::alloc_complex(N);
    auto p = F::plan_guru_dft(1, dims, 1, howmany_dims, in, out,
                              FFTW_FORWARD, FFTW_ESTIMATE);
    ASSERT_NE(p, nullptr);
    F::destroy_plan(p);
    F::free_fn(in);
    F::free_fn(out);
}

TYPED_TEST(FftwWrapperPlanTest, PTEST_PLAN_GURU64_DFT)
{
    using F = FftwTypes<TypeParam>;
    using iodim64_t = typename F::iodim64_t;
    const int N = 16;
    iodim64_t dims[1] = {{N, 1, 1}};
    iodim64_t howmany_dims[1] = {{1, 1, 1}};
    auto *in  = F::alloc_complex(N);
    auto *out = F::alloc_complex(N);
    auto p = F::plan_guru64_dft(1, dims, 1, howmany_dims, in, out,
                                FFTW_FORWARD, FFTW_ESTIMATE);
    ASSERT_NE(p, nullptr);
    F::destroy_plan(p);
    F::free_fn(in);
    F::free_fn(out);
}

/* ===== R2C plan creation ===== */

TYPED_TEST(FftwWrapperPlanTest, PTEST_PLAN_DFT_R2C_1D)
{
    using F = FftwTypes<TypeParam>;
    const int N = 16;
    auto *in  = F::alloc_real(N);
    auto *out = F::alloc_complex(N / 2 + 1);
    auto p = F::plan_dft_r2c_1d(N, in, out, FFTW_ESTIMATE);
    ASSERT_NE(p, nullptr);
    F::destroy_plan(p);
    F::free_fn(in);
    F::free_fn(out);
}

TYPED_TEST(FftwWrapperPlanTest, PTEST_PLAN_DFT_R2C_2D)
{
    using F = FftwTypes<TypeParam>;
    const int n0 = 4, n1 = 8;
    auto *in  = F::alloc_real(n0 * n1);
    auto *out = F::alloc_complex(n0 * (n1 / 2 + 1));
    auto p = F::plan_dft_r2c_2d(n0, n1, in, out, FFTW_ESTIMATE);
    ASSERT_NE(p, nullptr);
    F::destroy_plan(p);
    F::free_fn(in);
    F::free_fn(out);
}

TYPED_TEST(FftwWrapperPlanTest, PTEST_PLAN_DFT_R2C_3D)
{
    using F = FftwTypes<TypeParam>;
    const int n0 = 2, n1 = 3, n2 = 4;
    auto *in  = F::alloc_real(n0 * n1 * n2);
    auto *out = F::alloc_complex(n0 * n1 * (n2 / 2 + 1));
    auto p = F::plan_dft_r2c_3d(n0, n1, n2, in, out, FFTW_ESTIMATE);
    ASSERT_NE(p, nullptr);
    F::destroy_plan(p);
    F::free_fn(in);
    F::free_fn(out);
}

TYPED_TEST(FftwWrapperPlanTest, PTEST_PLAN_DFT_R2C_ND)
{
    using F = FftwTypes<TypeParam>;
    int n[] = {4, 6};
    int total_real = 4 * 6;
    int total_complex = 4 * (6 / 2 + 1);
    auto *in  = F::alloc_real(total_real);
    auto *out = F::alloc_complex(total_complex);
    auto p = F::plan_dft_r2c(2, n, in, out, FFTW_ESTIMATE);
    ASSERT_NE(p, nullptr);
    F::destroy_plan(p);
    F::free_fn(in);
    F::free_fn(out);
}

TYPED_TEST(FftwWrapperPlanTest, PTEST_PLAN_MANY_DFT_R2C)
{
    using F = FftwTypes<TypeParam>;
    const int N = 16, howmany = 3;
    int n[] = {N};
    auto *in  = F::alloc_real(N * howmany);
    auto *out = F::alloc_complex((N / 2 + 1) * howmany);
    auto p = F::plan_many_dft_r2c(1, n, howmany, in, nullptr, 1, N,
                                  out, nullptr, 1, N / 2 + 1, FFTW_ESTIMATE);
    ASSERT_NE(p, nullptr);
    F::destroy_plan(p);
    F::free_fn(in);
    F::free_fn(out);
}

TYPED_TEST(FftwWrapperPlanTest, PTEST_PLAN_GURU_DFT_R2C)
{
    using F = FftwTypes<TypeParam>;
    using iodim_t = typename F::iodim_t;
    const int N = 16;
    iodim_t dims[1] = {{N, 1, 1}};
    iodim_t howmany_dims[1] = {{1, 1, 1}};
    auto *in  = F::alloc_real(N);
    auto *out = F::alloc_complex(N / 2 + 1);
    auto p = F::plan_guru_dft_r2c(1, dims, 1, howmany_dims, in, out,
                                  FFTW_ESTIMATE);
    ASSERT_NE(p, nullptr);
    F::destroy_plan(p);
    F::free_fn(in);
    F::free_fn(out);
}

TYPED_TEST(FftwWrapperPlanTest, PTEST_PLAN_GURU64_DFT_R2C)
{
    using F = FftwTypes<TypeParam>;
    using iodim64_t = typename F::iodim64_t;
    const int N = 16;
    iodim64_t dims[1] = {{N, 1, 1}};
    iodim64_t howmany_dims[1] = {{1, 1, 1}};
    auto *in  = F::alloc_real(N);
    auto *out = F::alloc_complex(N / 2 + 1);
    auto p = F::plan_guru64_dft_r2c(1, dims, 1, howmany_dims, in, out,
                                    FFTW_ESTIMATE);
    ASSERT_NE(p, nullptr);
    F::destroy_plan(p);
    F::free_fn(in);
    F::free_fn(out);
}

/* ===== C2R plan creation ===== */

TYPED_TEST(FftwWrapperPlanTest, PTEST_PLAN_DFT_C2R_1D)
{
    using F = FftwTypes<TypeParam>;
    const int N = 16;
    auto *in  = F::alloc_complex(N / 2 + 1);
    auto *out = F::alloc_real(N);
    auto p = F::plan_dft_c2r_1d(N, in, out, FFTW_ESTIMATE);
    ASSERT_NE(p, nullptr);
    F::destroy_plan(p);
    F::free_fn(in);
    F::free_fn(out);
}

TYPED_TEST(FftwWrapperPlanTest, PTEST_PLAN_DFT_C2R_2D)
{
    using F = FftwTypes<TypeParam>;
    const int n0 = 4, n1 = 8;
    auto *in  = F::alloc_complex(n0 * (n1 / 2 + 1));
    auto *out = F::alloc_real(n0 * n1);
    auto p = F::plan_dft_c2r_2d(n0, n1, in, out, FFTW_ESTIMATE);
    ASSERT_NE(p, nullptr);
    F::destroy_plan(p);
    F::free_fn(in);
    F::free_fn(out);
}

TYPED_TEST(FftwWrapperPlanTest, PTEST_PLAN_DFT_C2R_3D)
{
    using F = FftwTypes<TypeParam>;
    const int n0 = 2, n1 = 3, n2 = 4;
    auto *in  = F::alloc_complex(n0 * n1 * (n2 / 2 + 1));
    auto *out = F::alloc_real(n0 * n1 * n2);
    auto p = F::plan_dft_c2r_3d(n0, n1, n2, in, out, FFTW_ESTIMATE);
    ASSERT_NE(p, nullptr);
    F::destroy_plan(p);
    F::free_fn(in);
    F::free_fn(out);
}

TYPED_TEST(FftwWrapperPlanTest, PTEST_PLAN_DFT_C2R_ND)
{
    using F = FftwTypes<TypeParam>;
    int n[] = {4, 6};
    auto *in  = F::alloc_complex(4 * (6 / 2 + 1));
    auto *out = F::alloc_real(4 * 6);
    auto p = F::plan_dft_c2r(2, n, in, out, FFTW_ESTIMATE);
    ASSERT_NE(p, nullptr);
    F::destroy_plan(p);
    F::free_fn(in);
    F::free_fn(out);
}

TYPED_TEST(FftwWrapperPlanTest, PTEST_PLAN_MANY_DFT_C2R)
{
    using F = FftwTypes<TypeParam>;
    const int N = 16, howmany = 3;
    int n[] = {N};
    auto *in  = F::alloc_complex((N / 2 + 1) * howmany);
    auto *out = F::alloc_real(N * howmany);
    auto p = F::plan_many_dft_c2r(1, n, howmany, in, nullptr, 1, N / 2 + 1,
                                  out, nullptr, 1, N, FFTW_ESTIMATE);
    ASSERT_NE(p, nullptr);
    F::destroy_plan(p);
    F::free_fn(in);
    F::free_fn(out);
}

TYPED_TEST(FftwWrapperPlanTest, PTEST_PLAN_GURU_DFT_C2R)
{
    using F = FftwTypes<TypeParam>;
    using iodim_t = typename F::iodim_t;
    const int N = 16;
    iodim_t dims[1] = {{N, 1, 1}};
    iodim_t howmany_dims[1] = {{1, 1, 1}};
    auto *in  = F::alloc_complex(N / 2 + 1);
    auto *out = F::alloc_real(N);
    auto p = F::plan_guru_dft_c2r(1, dims, 1, howmany_dims, in, out,
                                  FFTW_ESTIMATE);
    ASSERT_NE(p, nullptr);
    F::destroy_plan(p);
    F::free_fn(in);
    F::free_fn(out);
}

TYPED_TEST(FftwWrapperPlanTest, PTEST_PLAN_GURU64_DFT_C2R)
{
    using F = FftwTypes<TypeParam>;
    using iodim64_t = typename F::iodim64_t;
    const int N = 16;
    iodim64_t dims[1] = {{N, 1, 1}};
    iodim64_t howmany_dims[1] = {{1, 1, 1}};
    auto *in  = F::alloc_complex(N / 2 + 1);
    auto *out = F::alloc_real(N);
    auto p = F::plan_guru64_dft_c2r(1, dims, 1, howmany_dims, in, out,
                                    FFTW_ESTIMATE);
    ASSERT_NE(p, nullptr);
    F::destroy_plan(p);
    F::free_fn(in);
    F::free_fn(out);
}

/* ===== In-place R2C / C2R (padded arrays) ===== */

TYPED_TEST(FftwWrapperPlanTest, PTEST_PLAN_DFT_R2C_INPLACE)
{
    using F = FftwTypes<TypeParam>;
    using real_t = typename F::real_t;
    using complex_t = typename F::complex_t;
    const int N = 16;
    int padded = 2 * (N / 2 + 1);
    auto *buf = (real_t *)F::malloc_fn(sizeof(real_t) * padded);
    ASSERT_NE(buf, nullptr);
    auto p = F::plan_dft_r2c_1d(N, buf, (complex_t *)buf, FFTW_ESTIMATE);
    ASSERT_NE(p, nullptr);
    F::destroy_plan(p);
    F::free_fn(buf);
}

TYPED_TEST(FftwWrapperPlanTest, PTEST_PLAN_DFT_C2R_INPLACE)
{
    using F = FftwTypes<TypeParam>;
    using real_t = typename F::real_t;
    using complex_t = typename F::complex_t;
    const int N = 16;
    int padded = 2 * (N / 2 + 1);
    auto *buf = (real_t *)F::malloc_fn(sizeof(real_t) * padded);
    ASSERT_NE(buf, nullptr);
    auto p = F::plan_dft_c2r_1d(N, (complex_t *)buf, buf, FFTW_ESTIMATE);
    ASSERT_NE(p, nullptr);
    F::destroy_plan(p);
    F::free_fn(buf);
}

/* ===== Negative tests =====
 * The FFTW wrapper forwards invalid parameters to AOCL-FFTZ, whose setup
 * validates the problem descriptor (see api/validate_problem.h) and returns
 * NULL for invalid sizes/strides. These tests are active and assert that a
 * NULL plan is returned for such inputs. */

TYPED_TEST(FftwWrapperPlanTest, NTEST_PLAN_DFT_1D_ZERO_SIZE)
{
    using F = FftwTypes<TypeParam>;
    auto *in  = F::alloc_complex(1);
    auto *out = F::alloc_complex(1);
    auto p = F::plan_dft_1d(0, in, out, FFTW_FORWARD, FFTW_ESTIMATE);
    EXPECT_EQ(p, nullptr);
    if (p)
    {
        F::destroy_plan(p);
    }
    F::free_fn(in);
    F::free_fn(out);
}

TYPED_TEST(FftwWrapperPlanTest, NTEST_PLAN_DFT_1D_NEGATIVE_SIZE)
{
    using F = FftwTypes<TypeParam>;
    auto *in  = F::alloc_complex(1);
    auto *out = F::alloc_complex(1);
    auto p = F::plan_dft_1d(-1, in, out, FFTW_FORWARD, FFTW_ESTIMATE);
    EXPECT_EQ(p, nullptr);
    if (p)
    {
        F::destroy_plan(p);
    }
    F::free_fn(in);
    F::free_fn(out);
}

TYPED_TEST(FftwWrapperPlanTest, PTEST_PLAN_DFT_1D_NULL_INPUT)
{
    using F = FftwTypes<TypeParam>;
    auto *out = F::alloc_complex(16);
    auto p = F::plan_dft_1d(16, nullptr, out, FFTW_FORWARD, FFTW_ESTIMATE);
    EXPECT_NE(p, nullptr);
    if (p)
    {
        F::destroy_plan(p);
    }
    F::free_fn(out);
}

TYPED_TEST(FftwWrapperPlanTest, PTEST_PLAN_DFT_1D_NULL_OUTPUT)
{
    using F = FftwTypes<TypeParam>;
    auto *in = F::alloc_complex(16);
    auto p = F::plan_dft_1d(16, in, nullptr, FFTW_FORWARD, FFTW_ESTIMATE);
    EXPECT_NE(p, nullptr);
    if (p)
    {
        F::destroy_plan(p);
    }
    F::free_fn(in);
}

TYPED_TEST(FftwWrapperPlanTest, NTEST_PLAN_R2C_1D_ZERO_SIZE)
{
    using F = FftwTypes<TypeParam>;
    auto *in  = F::alloc_real(1);
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

TYPED_TEST(FftwWrapperPlanTest, PTEST_PLAN_R2C_1D_NULL_INPUT)
{
    using F = FftwTypes<TypeParam>;
    auto *out = F::alloc_complex(9);
    auto p = F::plan_dft_r2c_1d(16, nullptr, out, FFTW_ESTIMATE);
    EXPECT_NE(p, nullptr);
    if (p)
    {
        F::destroy_plan(p);
    }
    F::free_fn(out);
}

TYPED_TEST(FftwWrapperPlanTest, NTEST_PLAN_C2R_1D_ZERO_SIZE)
{
    using F = FftwTypes<TypeParam>;
    auto *in  = F::alloc_complex(1);
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

TYPED_TEST(FftwWrapperPlanTest, PTEST_PLAN_C2R_1D_NULL_OUTPUT)
{
    using F = FftwTypes<TypeParam>;
    auto *in = F::alloc_complex(9);
    auto p = F::plan_dft_c2r_1d(16, in, nullptr, FFTW_ESTIMATE);
    EXPECT_NE(p, nullptr);
    if (p)
    {
        F::destroy_plan(p);
    }
    F::free_fn(in);
}

TYPED_TEST(FftwWrapperPlanTest, PTEST_PLAN_MANY_DFT_NULL_INPUT)
{
    using F = FftwTypes<TypeParam>;
    int n[] = {8};
    auto *out = F::alloc_complex(8);
    auto p = F::plan_many_dft(1, n, 1, nullptr, nullptr, 1, 8,
                              out, nullptr, 1, 8, FFTW_FORWARD, FFTW_ESTIMATE);
    EXPECT_NE(p, nullptr);
    if (p)
    {
        F::destroy_plan(p);
    }
    F::free_fn(out);
}

TYPED_TEST(FftwWrapperPlanTest,
           PTEST_PLAN_MANY_DFT_ZERO_HOWMANY)
{
    using F = FftwTypes<TypeParam>;
    int n[] = {8};
    auto *in  = F::alloc_complex(8);
    auto *out = F::alloc_complex(8);
    auto p = F::plan_many_dft(1, n, 0, in, nullptr, 1, 8,
                              out, nullptr, 1, 8, FFTW_FORWARD, FFTW_ESTIMATE);
    EXPECT_NE(p, nullptr);
    if (p)
    {
        F::destroy_plan(p);
    }
    F::free_fn(in);
    F::free_fn(out);
}

