// Copyright Advanced Micro Devices, Inc.
// SPDX-License-Identifier: BSD-3-Clause

/** @file fftw_wrapper_execute_gtest.cpp
 * @brief FFTW wrapper execute tests.
 */

#include "fftw_wrapper_test_utils.h"

template<typename T>
class FftwWrapperExecuteTest : public FftwWrapperTestBase<T>
{
};
TYPED_TEST_SUITE(FftwWrapperExecuteTest, FftwTestTypes);

/* ===== 1-D forward correctness ===== */

TYPED_TEST(FftwWrapperExecuteTest, PTEST_EXECUTE_1D_FORWARD)
{
    using F = FftwTypes<TypeParam>;
    using complex_t [[maybe_unused]] = typename F::complex_t;
    const int N = 32;
    auto *in   = F::alloc_complex(N);
    auto *out  = F::alloc_complex(N);
    auto *ref  = F::alloc_complex(N);

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

TYPED_TEST(FftwWrapperExecuteTest, PTEST_EXECUTE_1D_BACKWARD)
{
    using F = FftwTypes<TypeParam>;
    const int N = 32;
    auto *in  = F::alloc_complex(N);
    auto *out = F::alloc_complex(N);
    auto *ref = F::alloc_complex(N);

    this->init_complex(in, N);
    auto p = F::plan_dft_1d(N, in, out, FFTW_BACKWARD, FFTW_ESTIMATE);
    ASSERT_NE(p, nullptr);
    F::execute(p);
    dft_reference_1d(in, ref, N, FFTW_BACKWARD);
    compare_complex_arrays(ref, out, N, dft_tolerance<F>(N));

    F::destroy_plan(p);
    F::free_fn(in);
    F::free_fn(out);
    F::free_fn(ref);
}

/* ===== execute_dft with explicit arrays ===== */

TYPED_TEST(FftwWrapperExecuteTest, PTEST_EXECUTE_DFT_1D)
{
    using F = FftwTypes<TypeParam>;
    const int N = 16;
    auto *in   = F::alloc_complex(N);
    auto *out  = F::alloc_complex(N);
    auto *in2  = F::alloc_complex(N);
    auto *out2 = F::alloc_complex(N);

    this->init_complex(in, N);
    auto p = F::plan_dft_1d(N, in, out, FFTW_FORWARD, FFTW_ESTIMATE);
    ASSERT_NE(p, nullptr);

    this->init_complex(in2, N);
    F::execute_dft(p, in2, out2);

    auto *ref = F::alloc_complex(N);
    dft_reference_1d(in2, ref, N, FFTW_FORWARD);
    compare_complex_arrays(ref, out2, N, dft_tolerance<F>(N));

    F::destroy_plan(p);
    F::free_fn(in);
    F::free_fn(out);
    F::free_fn(in2);
    F::free_fn(out2);
    F::free_fn(ref);
}

/* ===== Multi-dimensional execution ===== */

TYPED_TEST(FftwWrapperExecuteTest, PTEST_EXECUTE_DFT_2D)
{
    using F = FftwTypes<TypeParam>;
    const int n0 = 4, n1 = 8, total = n0 * n1;
    auto *in  = F::alloc_complex(total);
    auto *out = F::alloc_complex(total);
    auto *ref = F::alloc_complex(total);
    this->init_complex(in, total);
    auto p = F::plan_dft_2d(n0, n1, in, out, FFTW_FORWARD, FFTW_ESTIMATE);
    ASSERT_NE(p, nullptr);
    F::execute(p);
    dft_reference_2d(in, ref, n0, n1, FFTW_FORWARD);
    compare_complex_arrays(ref, out, total, dft_tolerance<F>(total));
    F::destroy_plan(p);
    F::free_fn(in);
    F::free_fn(out);
    F::free_fn(ref);
}

TYPED_TEST(FftwWrapperExecuteTest, PTEST_EXECUTE_DFT_3D)
{
    using F = FftwTypes<TypeParam>;
    const int n0 = 2, n1 = 3, n2 = 4, total = n0 * n1 * n2;
    auto *in  = F::alloc_complex(total);
    auto *out = F::alloc_complex(total);
    auto *ref = F::alloc_complex(total);
    this->init_complex(in, total);
    auto p = F::plan_dft_3d(n0, n1, n2, in, out, FFTW_FORWARD, FFTW_ESTIMATE);
    ASSERT_NE(p, nullptr);
    F::execute(p);
    dft_reference_3d(in, ref, n0, n1, n2, FFTW_FORWARD);
    compare_complex_arrays(ref, out, total, dft_tolerance<F>(total));
    F::destroy_plan(p);
    F::free_fn(in);
    F::free_fn(out);
    F::free_fn(ref);
}

/* ===== In-place ===== */

TYPED_TEST(FftwWrapperExecuteTest, PTEST_EXECUTE_INPLACE)
{
    using F = FftwTypes<TypeParam>;
    const int N = 32;
    auto *buf = F::alloc_complex(N);
    auto *ref = F::alloc_complex(N);
    this->init_complex(buf, N);
    std::memcpy(ref, buf, sizeof(typename F::complex_t) * N);

    auto p = F::plan_dft_1d(N, buf, buf, FFTW_FORWARD, FFTW_ESTIMATE);
    ASSERT_NE(p, nullptr);
    F::execute(p);

    auto *expected = F::alloc_complex(N);
    dft_reference_1d(ref, expected, N, FFTW_FORWARD);
    compare_complex_arrays(expected, buf, N, dft_tolerance<F>(N));

    F::destroy_plan(p);
    F::free_fn(buf);
    F::free_fn(ref);
    F::free_fn(expected);
}

/* ===== Out-of-place ===== */

TYPED_TEST(FftwWrapperExecuteTest, PTEST_EXECUTE_OUT_OF_PLACE)
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

/* ===== Multiple executions on same plan ===== */

TYPED_TEST(FftwWrapperExecuteTest, PTEST_EXECUTE_MULTIPLE_EXECUTIONS)
{
    using F = FftwTypes<TypeParam>;
    const int N = 16;
    auto *in  = F::alloc_complex(N);
    auto *out = F::alloc_complex(N);

    auto p = F::plan_dft_1d(N, in, out, FFTW_FORWARD, FFTW_ESTIMATE);
    ASSERT_NE(p, nullptr);

    for (int iter = 0; iter < 5; iter++)
    {
        this->init_complex_arithmetic(in, N, static_cast<TypeParam>(iter));
        F::execute(p);
    }

    /* Verify the last iteration against a reference of that iteration's input:
     * proves the plan still computes correctly after repeated reuse (an empty
     * loop body would otherwise pass regardless of what execute does). */
    auto *ref = F::alloc_complex(N);
    dft_reference_1d(in, ref, N, FFTW_FORWARD);
    compare_complex_arrays(ref, out, N, dft_tolerance<F>(N));

    F::destroy_plan(p);
    F::free_fn(in);
    F::free_fn(out);
    F::free_fn(ref);
}

/* ===== C2C roundtrip: forward then backward, verify input * N ===== */

TYPED_TEST(FftwWrapperExecuteTest, PTEST_ROUNDTRIP_C2C)
{
    using F = FftwTypes<TypeParam>;
    const int N = 64;
    auto *in     = F::alloc_complex(N);
    auto *freq   = F::alloc_complex(N);
    auto *result = F::alloc_complex(N);

    this->init_complex(in, N);
    auto pf = F::plan_dft_1d(N, in, freq, FFTW_FORWARD, FFTW_ESTIMATE);
    auto pb = F::plan_dft_1d(N, freq, result, FFTW_BACKWARD, FFTW_ESTIMATE);
    ASSERT_NE(pf, nullptr);
    ASSERT_NE(pb, nullptr);

    F::execute(pf);
    F::execute(pb);

    compare_complex_scaled(in, result, N, static_cast<double>(N),
                           dft_tolerance<F>(N));

    F::destroy_plan(pf);
    F::destroy_plan(pb);
    F::free_fn(in);
    F::free_fn(freq);
    F::free_fn(result);
}

/* ===== R2C then C2R roundtrip ===== */

TYPED_TEST(FftwWrapperExecuteTest, PTEST_ROUNDTRIP_R2C_C2R)
{
    using F = FftwTypes<TypeParam>;
    using real_t [[maybe_unused]] = typename F::real_t;
    const int N = 32;
    int Nc = N / 2 + 1;

    auto *in     = F::alloc_real(N);
    auto *freq   = F::alloc_complex(Nc);
    auto *result = F::alloc_real(N);

    this->init_real(in, N);
    std::vector<real_t> orig(in, in + N);

    auto pr = F::plan_dft_r2c_1d(N, in, freq, FFTW_ESTIMATE);
    auto pc = F::plan_dft_c2r_1d(N, freq, result, FFTW_ESTIMATE);
    ASSERT_NE(pr, nullptr);
    ASSERT_NE(pc, nullptr);

    F::execute(pr);
    F::execute(pc);

    compare_real_scaled(orig.data(), result, N, N, dft_tolerance<F>(N));

    F::destroy_plan(pr);
    F::destroy_plan(pc);
    F::free_fn(in);
    F::free_fn(freq);
    F::free_fn(result);
}

/* ===== R2C 1-D correctness against reference ===== */

TYPED_TEST(FftwWrapperExecuteTest, PTEST_EXECUTE_DFT_R2C_1D)
{
    using F = FftwTypes<TypeParam>;
    using real_t [[maybe_unused]] = typename F::real_t;
    const int N = 32;
    int Nc = N / 2 + 1;

    auto *in  = F::alloc_real(N);
    auto *out = F::alloc_complex(Nc);
    auto *ref = F::alloc_complex(Nc);

    this->init_real(in, N);
    auto p = F::plan_dft_r2c_1d(N, in, out, FFTW_ESTIMATE);
    ASSERT_NE(p, nullptr);
    /* Exercise the new-array R2C entry point (matches the test name and makes
     * this distinct from PTEST_R2C_1D_CORRECTNESS, which uses plain execute). */
    F::execute_dft_r2c(p, in, out);

    dft_reference_r2c_1d(in, ref, N);
    compare_complex_arrays(ref, out, Nc, dft_tolerance<F>(N));

    F::destroy_plan(p);
    F::free_fn(in);
    F::free_fn(out);
    F::free_fn(ref);
}

/* ===== C2R 1-D correctness via roundtrip ===== */

TYPED_TEST(FftwWrapperExecuteTest, PTEST_EXECUTE_DFT_C2R_1D)
{
    using F = FftwTypes<TypeParam>;
    using real_t [[maybe_unused]] = typename F::real_t;
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
    F::execute_dft_c2r(pc, freq, out);

    compare_real_scaled(orig.data(), out, N, N, dft_tolerance<F>(N));

    F::destroy_plan(pr);
    F::destroy_plan(pc);
    F::free_fn(in);
    F::free_fn(freq);
    F::free_fn(out);
}

/* ===== Various sizes ===== */

TYPED_TEST(FftwWrapperExecuteTest, PTEST_EXECUTE_VARIOUS_SIZES)
{
    using F = FftwTypes<TypeParam>;
    for (int N : {1, 2, 4, 7, 8, 16, 32, 64, 128, 256})
    {
        auto *in  = F::alloc_complex(N);
        auto *out = F::alloc_complex(N);
        auto *ref = F::alloc_complex(N);
        this->init_complex(in, N);

        auto p = F::plan_dft_1d(N, in, out, FFTW_FORWARD, FFTW_ESTIMATE);
        ASSERT_NE(p, nullptr) << "N = " << N;
        F::execute(p);

        dft_reference_1d(in, ref, N, FFTW_FORWARD);
        compare_complex_arrays(ref, out, N, dft_tolerance<F>(N));

        F::destroy_plan(p);
        F::free_fn(in);
        F::free_fn(out);
        F::free_fn(ref);
    }
}

/* ===== Batched (many) execution ===== */

TYPED_TEST(FftwWrapperExecuteTest, PTEST_EXECUTE_MANY_DFT)
{
    using F = FftwTypes<TypeParam>;
    const int N = 8, howmany = 4;
    auto *in  = F::alloc_complex(N * howmany);
    auto *out = F::alloc_complex(N * howmany);
    auto *ref = F::alloc_complex(N);

    this->init_complex_arithmetic_strided(in, N, 1, N, howmany);

    int n[] = {N};
    auto p = F::plan_many_dft(1, n, howmany, in, nullptr, 1, N,
                              out, nullptr, 1, N, FFTW_FORWARD, FFTW_ESTIMATE);
    ASSERT_NE(p, nullptr);
    F::execute(p);

    for (int b = 0; b < howmany; b++)
    {
        dft_reference_1d(&in[b * N], ref, N, FFTW_FORWARD);
        compare_complex_arrays(ref, &out[b * N], N, dft_tolerance<F>(N));
    }

    F::destroy_plan(p);
    F::free_fn(in);
    F::free_fn(out);
    F::free_fn(ref);
}

/* ===== Guru execution ===== */

TYPED_TEST(FftwWrapperExecuteTest, PTEST_EXECUTE_GURU_DFT)
{
    using F = FftwTypes<TypeParam>;
    using iodim_t = typename F::iodim_t;
    const int N = 16;
    iodim_t dims[1] = {{N, 1, 1}};
    iodim_t howmany[1] = {{1, 1, 1}};

    auto *in  = F::alloc_complex(N);
    auto *out = F::alloc_complex(N);
    auto *ref = F::alloc_complex(N);
    this->init_complex(in, N);

    auto p = F::plan_guru_dft(1, dims, 1, howmany, in, out,
                              FFTW_FORWARD, FFTW_ESTIMATE);
    ASSERT_NE(p, nullptr);
    F::execute(p);

    dft_reference_1d(in, ref, N, FFTW_FORWARD);
    compare_complex_arrays(ref, out, N, dft_tolerance<F>(N));

    F::destroy_plan(p);
    F::free_fn(in);
    F::free_fn(out);
    F::free_fn(ref);
}

/* ===== Special values: zeros ===== */

TYPED_TEST(FftwWrapperExecuteTest, PTEST_EXECUTE_ALL_ZEROS)
{
    using F = FftwTypes<TypeParam>;
    const int N = 16;
    auto *in  = F::alloc_complex(N);
    auto *out = F::alloc_complex(N);
    this->zero_complex(in, N);

    auto p = F::plan_dft_1d(N, in, out, FFTW_FORWARD, FFTW_ESTIMATE);
    ASSERT_NE(p, nullptr);
    F::execute(p);

    compare_complex_zero(out, N, F::tolerance);

    F::destroy_plan(p);
    F::free_fn(in);
    F::free_fn(out);
}

TYPED_TEST(FftwWrapperExecuteTest, PTEST_EXECUTE_SPECIAL_VALUES)
{
    using F = FftwTypes<TypeParam>;
    const int N = 8;
    auto *in  = F::alloc_complex(N);
    auto *out = F::alloc_complex(N);
    auto *ref = F::alloc_complex(N);

    in[0][0] = 0.0;
    in[0][1] = 0.0;
    in[1][0] = 1e-15;
    in[1][1] = -1e-15;
    in[2][0] = 1.0;
    in[2][1] = 0.0;
    in[3][0] = -1.0;
    in[3][1] = 1.0;
    in[4][0] = 1e10;
    in[4][1] = -1e10;
    in[5][0] = 1e-10;
    in[5][1] = 1e-10;
    in[6][0] = 0.5;
    in[6][1] = -0.5;
    in[7][0] = 100.0;
    in[7][1] = 200.0;

    auto p = F::plan_dft_1d(N, in, out, FFTW_FORWARD, FFTW_ESTIMATE);
    ASSERT_NE(p, nullptr);
    F::execute(p);

    dft_reference_1d(in, ref, N, FFTW_FORWARD);
    double tol = std::is_same<TypeParam, float>::value ? 1e3 : 1e-2;
    compare_complex_arrays(ref, out, N, tol);

    F::destroy_plan(p);
    F::free_fn(in);
    F::free_fn(out);
    F::free_fn(ref);
}

/* ===== Negative tests ===== */

TYPED_TEST(FftwWrapperExecuteTest, NTEST_EXECUTE_NULL_PLAN)
{
    using F = FftwTypes<TypeParam>;
    F::execute(nullptr);
}

TYPED_TEST(FftwWrapperExecuteTest, NTEST_EXECUTE_DFT_NULL_PLAN)
{
    using F = FftwTypes<TypeParam>;
    auto *in  = F::alloc_complex(8);
    auto *out = F::alloc_complex(8);
    F::execute_dft(nullptr, in, out);
    F::free_fn(in);
    F::free_fn(out);
}
