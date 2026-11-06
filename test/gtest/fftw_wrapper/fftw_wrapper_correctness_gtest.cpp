// Copyright Advanced Micro Devices, Inc.
// SPDX-License-Identifier: BSD-3-Clause

/**
 * @file fftw_wrapper_correctness_gtest.cpp
 * @brief Correctness tests targeting translator logic: dimension reversal,
 *        R2C/C2R stride chains, batched independence, guru pass-through,
 *        and backward-direction transforms.  Each test compares wrapper
 *        output against a brute-force reference to catch silent data
 *        corruption bugs.
 */

#include <cstring>
#include "fftw_wrapper_test_utils.h"

template<typename T>
class FftwWrapperCorrectnessTest : public FftwWrapperTestBase<T>
{
};
TYPED_TEST_SUITE(FftwWrapperCorrectnessTest, FftwTestTypes);

/* =====================================================================
 * 1. Dimension reversal: 2D asymmetric C2C (n0 != n1)
 *    Targets: get_dv_desc dimension reversal n[rank-i-1]
 * ===================================================================== */

TYPED_TEST(FftwWrapperCorrectnessTest,
           PTEST_2D_ASYMMETRIC_C2C_FORWARD)
{
    using F = FftwTypes<TypeParam>;
    using complex_t [[maybe_unused]] = typename F::complex_t;
    const int N0 = 3, N1 = 7;
    const int total = N0 * N1;

    auto *in  = F::alloc_complex(total);
    auto *out = F::alloc_complex(total);
    auto *ref = F::alloc_complex(total);

    this->init_complex(in, total);

    dft_reference_2d(in, ref, N0, N1, FFTW_FORWARD);

    auto p = F::plan_dft_2d(
        N0, N1, in, out, FFTW_FORWARD, FFTW_ESTIMATE);
    ASSERT_NE(p, nullptr);
    F::execute(p);

    double tol = dft_tolerance<F>(total);
    compare_complex_arrays(ref, out, total, tol);

    F::destroy_plan(p);
    F::free_fn(in);
    F::free_fn(out);
    F::free_fn(ref);
}

/* =====================================================================
 * 2. Dimension reversal: 2D asymmetric C2C BACKWARD
 *    Targets: init_flag direction translation
 * ===================================================================== */

TYPED_TEST(FftwWrapperCorrectnessTest,
           PTEST_2D_ASYMMETRIC_C2C_BACKWARD)
{
    using F = FftwTypes<TypeParam>;
    using complex_t [[maybe_unused]] = typename F::complex_t;
    const int N0 = 5, N1 = 3;
    const int total = N0 * N1;

    auto *in  = F::alloc_complex(total);
    auto *out = F::alloc_complex(total);
    auto *ref = F::alloc_complex(total);

    this->init_complex(in, total);

    dft_reference_2d(in, ref, N0, N1, FFTW_BACKWARD);

    auto p = F::plan_dft_2d(
        N0, N1, in, out, FFTW_BACKWARD, FFTW_ESTIMATE);
    ASSERT_NE(p, nullptr);
    F::execute(p);

    double tol = dft_tolerance<F>(total);
    compare_complex_arrays(ref, out, total, tol);

    F::destroy_plan(p);
    F::free_fn(in);
    F::free_fn(out);
    F::free_fn(ref);
}

/* =====================================================================
 * 3. Dimension reversal: 3D all-different-dims C2C
 *    Targets: get_dv_desc 3-dim stride chain
 * ===================================================================== */

TYPED_TEST(FftwWrapperCorrectnessTest,
           PTEST_3D_ASYMMETRIC_C2C_FORWARD)
{
    using F = FftwTypes<TypeParam>;
    using complex_t [[maybe_unused]] = typename F::complex_t;
    const int N0 = 3, N1 = 4, N2 = 5;
    const int total = N0 * N1 * N2;

    auto *in  = F::alloc_complex(total);
    auto *out = F::alloc_complex(total);
    auto *ref = F::alloc_complex(total);

    this->init_complex(in, total);

    dft_reference_3d(in, ref, N0, N1, N2, FFTW_FORWARD);

    auto p = F::plan_dft_3d(
        N0, N1, N2, in, out, FFTW_FORWARD, FFTW_ESTIMATE);
    ASSERT_NE(p, nullptr);
    F::execute(p);

    double tol = dft_tolerance<F>(total);
    compare_complex_arrays(ref, out, total, tol);

    F::destroy_plan(p);
    F::free_fn(in);
    F::free_fn(out);
    F::free_fn(ref);
}

/* =====================================================================
 * 4. R2C 1D: verify against brute-force R2C reference
 *    Targets: get_r2c_dv_desc 1D path
 * ===================================================================== */

TYPED_TEST(FftwWrapperCorrectnessTest,
           PTEST_R2C_1D_CORRECTNESS)
{
    using F = FftwTypes<TypeParam>;
    using real_t [[maybe_unused]] = typename F::real_t;
    using complex_t [[maybe_unused]] = typename F::complex_t;
    const int N = 16;
    const int Nc = N / 2 + 1;

    auto *in  = F::alloc_real(N);
    auto *out = F::alloc_complex(Nc);
    auto *ref = F::alloc_complex(Nc);

    this->init_real(in, N);

    dft_reference_r2c_1d(in, ref, N);

    auto p = F::plan_dft_r2c_1d(N, in, out, FFTW_ESTIMATE);
    ASSERT_NE(p, nullptr);
    F::execute(p);

    double tol = dft_tolerance<F>(N);
    compare_complex_arrays(ref, out, Nc, tol);

    F::destroy_plan(p);
    F::free_fn(in);
    F::free_fn(out);
    F::free_fn(ref);
}

/* =====================================================================
 * 5. R2C 1D with ODD size: N/2+1 edge
 *    Targets: integer division (N/2+1) for odd N
 * ===================================================================== */

TYPED_TEST(FftwWrapperCorrectnessTest,
           PTEST_R2C_1D_ODD_SIZE_CORRECTNESS)
{
    using F = FftwTypes<TypeParam>;
    using real_t [[maybe_unused]] = typename F::real_t;
    using complex_t [[maybe_unused]] = typename F::complex_t;
    const int N = 15;
    const int Nc = N / 2 + 1;

    auto *in  = F::alloc_real(N);
    auto *out = F::alloc_complex(Nc);
    auto *ref = F::alloc_complex(Nc);

    this->init_real(in, N);

    dft_reference_r2c_1d(in, ref, N);

    auto p = F::plan_dft_r2c_1d(N, in, out, FFTW_ESTIMATE);
    ASSERT_NE(p, nullptr);
    F::execute(p);

    double tol = dft_tolerance<F>(N);
    compare_complex_arrays(ref, out, Nc, tol);

    F::destroy_plan(p);
    F::free_fn(in);
    F::free_fn(out);
    F::free_fn(ref);
}

/* =====================================================================
 * 6. R2C 2D out-of-place with odd last FFTW dimension
 *    Targets: get_r2c_dv_desc dims[1] out_stride = (dims[0].n/2+1)
 *    Verification via roundtrip: R2C -> C2R should recover data * N
 * ===================================================================== */

TYPED_TEST(FftwWrapperCorrectnessTest,
           PTEST_R2C_2D_ODD_LAST_DIM_ROUNDTRIP)
{
    using F = FftwTypes<TypeParam>;
    using real_t [[maybe_unused]] = typename F::real_t;
    using complex_t [[maybe_unused]] = typename F::complex_t;
    const int N0 = 4, N1 = 7;
    const int total = N0 * N1;
    const int Nc = N0 * (N1 / 2 + 1);

    auto *in   = F::alloc_real(total);
    auto *freq = F::alloc_complex(Nc);
    auto *out  = F::alloc_real(total);

    this->init_real_iota(in, total);

    auto pr = F::plan_dft_r2c_2d(
        N0, N1, in, freq, FFTW_ESTIMATE);
    auto pc = F::plan_dft_c2r_2d(
        N0, N1, freq, out, FFTW_ESTIMATE);
    ASSERT_NE(pr, nullptr);
    ASSERT_NE(pc, nullptr);

    F::execute(pr);
    F::execute(pc);

    compare_real_scaled(in, out, total, total, dft_tolerance<F>(total));

    F::destroy_plan(pr);
    F::destroy_plan(pc);
    F::free_fn(in);
    F::free_fn(freq);
    F::free_fn(out);
}

/* =====================================================================
 * 7. R2C 3D out-of-place roundtrip with asymmetric dims
 *    Targets: get_r2c_dv_desc / get_c2r_dv_desc 3D stride chain
 * ===================================================================== */

TYPED_TEST(FftwWrapperCorrectnessTest,
           PTEST_R2C_3D_ASYMMETRIC_ROUNDTRIP)
{
    using F = FftwTypes<TypeParam>;
    using real_t [[maybe_unused]] = typename F::real_t;
    using complex_t [[maybe_unused]] = typename F::complex_t;
    const int N0 = 3, N1 = 4, N2 = 5;
    const int total = N0 * N1 * N2;
    const int Nc = N0 * N1 * (N2 / 2 + 1);

    auto *in   = F::alloc_real(total);
    auto *freq = F::alloc_complex(Nc);
    auto *out  = F::alloc_real(total);

    this->init_real_iota(in, total);

    auto pr = F::plan_dft_r2c_3d(
        N0, N1, N2, in, freq, FFTW_ESTIMATE);
    auto pc = F::plan_dft_c2r_3d(
        N0, N1, N2, freq, out, FFTW_ESTIMATE);
    ASSERT_NE(pr, nullptr);
    ASSERT_NE(pc, nullptr);

    F::execute(pr);
    F::execute(pc);

    compare_real_scaled(in, out, total, static_cast<double>(total),
                        dft_tolerance<F>(total));

    F::destroy_plan(pr);
    F::destroy_plan(pc);
    F::free_fn(in);
    F::free_fn(freq);
    F::free_fn(out);
}

/* =====================================================================
 * 8. Forward-backward roundtrip: FORWARD then BACKWARD => x * N
 *    Targets: init_flag direction mapping for BACKWARD
 * ===================================================================== */

TYPED_TEST(FftwWrapperCorrectnessTest,
           PTEST_1D_FORWARD_BACKWARD_ROUNDTRIP)
{
    using F = FftwTypes<TypeParam>;
    using complex_t [[maybe_unused]] = typename F::complex_t;
    const int N = 32;

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

    compare_complex_scaled(in, back, N, static_cast<double>(N),
                           dft_tolerance<F>(N));

    F::destroy_plan(pf);
    F::destroy_plan(pb);
    F::free_fn(in);
    F::free_fn(freq);
    F::free_fn(back);
}

/* =====================================================================
 * 9. Batched C2C: verify each batch is independently correct
 *    Targets: get_many_dv_desc vecs[0].n / vecs[0].in_stride
 * ===================================================================== */

TYPED_TEST(FftwWrapperCorrectnessTest,
           PTEST_MANY_DFT_BATCH_INDEPENDENCE)
{
    using F = FftwTypes<TypeParam>;
    using complex_t [[maybe_unused]] = typename F::complex_t;
    const int N = 8;
    const int howmany = 3;
    const int total = N * howmany;

    auto *in  = F::alloc_complex(total);
    auto *out = F::alloc_complex(total);
    auto *ref = F::alloc_complex(N);

    this->init_complex_strided(in, N, 1, N, howmany);

    auto p = F::plan_many_dft(
        1, &N, howmany, in, nullptr, 1, N,
        out, nullptr, 1, N, FFTW_FORWARD, FFTW_ESTIMATE);
    ASSERT_NE(p, nullptr);
    F::execute(p);

    double tol = dft_tolerance<F>(N);
    for (int b = 0; b < howmany; b++)
    {
        dft_reference_1d(&in[b * N], ref, N, FFTW_FORWARD);
        compare_complex_arrays(ref, &out[b * N], N, tol);
    }

    F::destroy_plan(p);
    F::free_fn(in);
    F::free_fn(out);
    F::free_fn(ref);
}

/* =====================================================================
 * 10. Batched R2C: verify each batch against R2C reference
 *     Targets: get_many_r2c_dv_desc stride/dist for batches
 * ===================================================================== */

TYPED_TEST(FftwWrapperCorrectnessTest,
           PTEST_MANY_R2C_BATCH_CORRECTNESS)
{
    using F = FftwTypes<TypeParam>;
    using real_t [[maybe_unused]] = typename F::real_t;
    using complex_t [[maybe_unused]] = typename F::complex_t;
    const int N = 16;
    const int Nc = N / 2 + 1;
    const int howmany = 2;

    auto *in  = F::alloc_real(N * howmany);
    auto *out = F::alloc_complex(Nc * howmany);
    auto *ref = F::alloc_complex(Nc);

    this->init_real_strided(in, N, 1, N, howmany);

    int n_arr[] = {N};
    auto p = F::plan_many_dft_r2c(
        1, n_arr, howmany,
        in, nullptr, 1, N,
        out, nullptr, 1, Nc,
        FFTW_ESTIMATE);
    ASSERT_NE(p, nullptr);
    F::execute(p);

    double tol = dft_tolerance<F>(N);
    for (int b = 0; b < howmany; b++)
    {
        dft_reference_r2c_1d(&in[b * N], ref, N);
        compare_complex_arrays(ref, &out[b * Nc], Nc, tol);
    }

    F::destroy_plan(p);
    F::free_fn(in);
    F::free_fn(out);
    F::free_fn(ref);
}

/* =====================================================================
 * 11. Batched R2C -> C2R roundtrip
 *     Targets: get_many_r2c_dv_desc + get_many_c2r_dv_desc coherence
 * ===================================================================== */

TYPED_TEST(FftwWrapperCorrectnessTest,
           PTEST_MANY_R2C_C2R_ROUNDTRIP)
{
    using F = FftwTypes<TypeParam>;
    using real_t [[maybe_unused]] = typename F::real_t;
    using complex_t [[maybe_unused]] = typename F::complex_t;
    const int N = 16;
    const int Nc = N / 2 + 1;
    const int howmany = 3;

    auto *in   = F::alloc_real(N * howmany);
    auto *freq = F::alloc_complex(Nc * howmany);
    auto *out  = F::alloc_real(N * howmany);

    this->init_real_strided(in, N, 1, N, howmany);

    int n_arr[] = {N};
    auto pr = F::plan_many_dft_r2c(
        1, n_arr, howmany,
        in, nullptr, 1, N,
        freq, nullptr, 1, Nc,
        FFTW_ESTIMATE);
    auto pc = F::plan_many_dft_c2r(
        1, n_arr, howmany,
        freq, nullptr, 1, Nc,
        out, nullptr, 1, N,
        FFTW_ESTIMATE);
    ASSERT_NE(pr, nullptr);
    ASSERT_NE(pc, nullptr);

    F::execute(pr);
    F::execute(pc);

    compare_real_scaled(in, out, N * howmany, static_cast<double>(N),
                        dft_tolerance<F>(N));

    F::destroy_plan(pr);
    F::destroy_plan(pc);
    F::free_fn(in);
    F::free_fn(freq);
    F::free_fn(out);
}

/* =====================================================================
 * 12. Guru vs plan_dft_1d: same data, same strides, must match
 *     Targets: get_guru_dv_desc dimension/stride reversal
 * ===================================================================== */

TYPED_TEST(FftwWrapperCorrectnessTest,
           PTEST_GURU_MATCHES_PLAN_DFT_1D)
{
    using F = FftwTypes<TypeParam>;
    using complex_t [[maybe_unused]] = typename F::complex_t;
    using iodim_t = typename F::iodim_t;
    const int N = 16;

    auto *in    = F::alloc_complex(N);
    auto *out1  = F::alloc_complex(N);
    auto *out2  = F::alloc_complex(N);
    auto *ref   = F::alloc_complex(N);

    this->init_complex(in, N);

    auto p1 = F::plan_dft_1d(
        N, in, out1, FFTW_FORWARD, FFTW_ESTIMATE);
    ASSERT_NE(p1, nullptr);

    iodim_t dims[1] = {{N, 1, 1}};
    iodim_t howmany[1] = {{1, 1, 1}};
    auto p2 = F::plan_guru_dft(
        1, dims, 1, howmany, in, out2,
        FFTW_FORWARD, FFTW_ESTIMATE);
    ASSERT_NE(p2, nullptr);

    F::execute(p1);
    F::execute(p2);

    double tol = dft_tolerance<F>(N);
    compare_complex_arrays(out1, out2, N, tol);
    dft_reference_1d(in, ref, N, FFTW_FORWARD);
    compare_complex_arrays(ref, out1, N, tol);
    compare_complex_arrays(ref, out2, N, tol);

    F::destroy_plan(p1);
    F::destroy_plan(p2);
    F::free_fn(in);
    F::free_fn(out1);
    F::free_fn(out2);
    F::free_fn(ref);
}

/* =====================================================================
 * 13. Guru 2D: match against plan_dft_2d
 *     Targets: get_guru_dv_desc multi-dim reversal
 * ===================================================================== */

TYPED_TEST(FftwWrapperCorrectnessTest,
           PTEST_GURU_2D_MATCHES_PLAN_DFT_2D)
{
    using F = FftwTypes<TypeParam>;
    using complex_t [[maybe_unused]] = typename F::complex_t;
    using iodim_t = typename F::iodim_t;
    const int N0 = 4, N1 = 6;
    const int total = N0 * N1;

    auto *in    = F::alloc_complex(total);
    auto *out1  = F::alloc_complex(total);
    auto *out2  = F::alloc_complex(total);
    auto *ref   = F::alloc_complex(total);

    this->init_complex(in, total);

    auto p1 = F::plan_dft_2d(
        N0, N1, in, out1, FFTW_FORWARD, FFTW_ESTIMATE);
    ASSERT_NE(p1, nullptr);

    iodim_t dims[2] = {{N0, N1, N1}, {N1, 1, 1}};
    iodim_t howmany[1] = {{1, 1, 1}};
    auto p2 = F::plan_guru_dft(
        2, dims, 1, howmany, in, out2,
        FFTW_FORWARD, FFTW_ESTIMATE);
    ASSERT_NE(p2, nullptr);

    F::execute(p1);
    F::execute(p2);

    double tol = dft_tolerance<F>(total);
    compare_complex_arrays(out1, out2, total, tol);
    dft_reference_2d(in, ref, N0, N1, FFTW_FORWARD);
    compare_complex_arrays(ref, out1, total, tol);
    compare_complex_arrays(ref, out2, total, tol);

    F::destroy_plan(p1);
    F::destroy_plan(p2);
    F::free_fn(in);
    F::free_fn(out1);
    F::free_fn(out2);
    F::free_fn(ref);
}

/* =====================================================================
 * 14. plan_many with non-NULL inembed (padded layout)
 *     Input is N=8 but embedded in a 10-element row.  Output uses
 *     default (non-NULL identical) embedding.
 *     Targets: get_many_dv_desc inembed[rank-i] indexing
 * ===================================================================== */

TYPED_TEST(FftwWrapperCorrectnessTest,
           PTEST_MANY_DFT_CUSTOM_EMBED)
{
    using F = FftwTypes<TypeParam>;
    using complex_t [[maybe_unused]] = typename F::complex_t;
    const int N = 8;
    const int embed = 10;
    const int howmany = 2;

    auto *in  = F::alloc_complex(embed * howmany);
    auto *out = F::alloc_complex(N * howmany);
    auto *ref = F::alloc_complex(N);

    this->zero_complex(in, embed * howmany);
    this->init_complex_strided(in, N, 1, embed, howmany);

    int n_arr[] = {N};
    int inembed_arr[] = {embed};
    int onembed_arr[] = {N};
    auto p = F::plan_many_dft(
        1, n_arr, howmany,
        in, inembed_arr, 1, embed,
        out, onembed_arr, 1, N,
        FFTW_FORWARD, FFTW_ESTIMATE);
    ASSERT_NE(p, nullptr);
    F::execute(p);

    double tol = dft_tolerance<F>(N);
    for (int b = 0; b < howmany; b++)
    {
        dft_reference_1d(&in[b * embed], ref, N, FFTW_FORWARD);
        compare_complex_arrays(ref, &out[b * N], N, tol);
    }

    F::destroy_plan(p);
    F::free_fn(in);
    F::free_fn(out);
    F::free_fn(ref);
}

/* =====================================================================
 * 15. 2D C2C forward-backward roundtrip with asymmetric dims
 *     Targets: dimension reversal coherence between forward and backward
 * ===================================================================== */

TYPED_TEST(FftwWrapperCorrectnessTest,
           PTEST_2D_ASYMMETRIC_FORWARD_BACKWARD_ROUNDTRIP)
{
    using F = FftwTypes<TypeParam>;
    using complex_t [[maybe_unused]] = typename F::complex_t;
    const int N0 = 5, N1 = 8;
    const int total = N0 * N1;

    auto *in   = F::alloc_complex(total);
    auto *freq = F::alloc_complex(total);
    auto *back = F::alloc_complex(total);

    this->init_complex(in, total);

    auto pf = F::plan_dft_2d(
        N0, N1, in, freq, FFTW_FORWARD, FFTW_ESTIMATE);
    auto pb = F::plan_dft_2d(
        N0, N1, freq, back, FFTW_BACKWARD, FFTW_ESTIMATE);
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
 * 16. 1D BACKWARD direction: verify against reference
 *     Targets: sign mapping in init_flag (FORWARD vs BACKWARD)
 * ===================================================================== */

TYPED_TEST(FftwWrapperCorrectnessTest,
           PTEST_1D_BACKWARD_VS_REFERENCE)
{
    using F = FftwTypes<TypeParam>;
    using complex_t [[maybe_unused]] = typename F::complex_t;
    const int N = 16;

    auto *in  = F::alloc_complex(N);
    auto *out = F::alloc_complex(N);
    auto *ref = F::alloc_complex(N);

    this->init_complex(in, N);

    dft_reference_1d(in, ref, N, FFTW_BACKWARD);

    auto p = F::plan_dft_1d(
        N, in, out, FFTW_BACKWARD, FFTW_ESTIMATE);
    ASSERT_NE(p, nullptr);
    F::execute(p);

    double tol = dft_tolerance<F>(N);
    compare_complex_arrays(ref, out, N, tol);

    F::destroy_plan(p);
    F::free_fn(in);
    F::free_fn(out);
    F::free_fn(ref);
}

/* =====================================================================
 * 17. R2C 2D even-even dims: most common case, verify roundtrip
 *     Targets: get_r2c_dv_desc / get_c2r_dv_desc even-dimension path
 * ===================================================================== */

TYPED_TEST(FftwWrapperCorrectnessTest,
           PTEST_R2C_2D_EVEN_DIMS_ROUNDTRIP)
{
    using F = FftwTypes<TypeParam>;
    using real_t [[maybe_unused]] = typename F::real_t;
    using complex_t [[maybe_unused]] = typename F::complex_t;
    const int N0 = 6, N1 = 8;
    const int total = N0 * N1;
    const int Nc = N0 * (N1 / 2 + 1);

    auto *in   = F::alloc_real(total);
    auto *freq = F::alloc_complex(Nc);
    auto *out  = F::alloc_real(total);

    this->init_real_iota(in, total);

    auto pr = F::plan_dft_r2c_2d(
        N0, N1, in, freq, FFTW_ESTIMATE);
    auto pc = F::plan_dft_c2r_2d(
        N0, N1, freq, out, FFTW_ESTIMATE);
    ASSERT_NE(pr, nullptr);
    ASSERT_NE(pc, nullptr);

    F::execute(pr);
    F::execute(pc);

    compare_real_scaled(in, out, total, total, dft_tolerance<F>(total));

    F::destroy_plan(pr);
    F::destroy_plan(pc);
    F::free_fn(in);
    F::free_fn(freq);
    F::free_fn(out);
}

/* =====================================================================
 * 18. Out-of-place input buffer preservation
 *     FFTW guarantees out-of-place FFTW_ESTIMATE transforms preserve
 *     the input buffer.  Verify for C2C 1D, R2C 1D, and C2C 2D.
 * ===================================================================== */

TYPED_TEST(FftwWrapperCorrectnessTest,
           PTEST_OOP_C2C_INPUT_PRESERVED)
{
    using F = FftwTypes<TypeParam>;
    const int N = 32;
    auto *in  = F::alloc_complex(N);
    auto *out = F::alloc_complex(N);

    this->init_complex(in, N);
    std::vector<typename F::complex_t> orig(N);
    std::memcpy(orig.data(), in, sizeof(typename F::complex_t) * N);

    auto p = F::plan_dft_1d(N, in, out, FFTW_FORWARD, FFTW_ESTIMATE);
    ASSERT_NE(p, nullptr);
    F::execute(p);

    for (int i = 0; i < N; i++)
    {
        EXPECT_EQ(in[i][0], orig[i][0])
            << "C2C out-of-place modified input real at i=" << i;
        EXPECT_EQ(in[i][1], orig[i][1])
            << "C2C out-of-place modified input imag at i=" << i;
    }

    F::destroy_plan(p);
    F::free_fn(in);
    F::free_fn(out);
}

TYPED_TEST(FftwWrapperCorrectnessTest,
           PTEST_OOP_R2C_INPUT_PRESERVED)
{
    using F = FftwTypes<TypeParam>;
    using real_t = typename F::real_t;
    const int N = 32;
    int Nc = N / 2 + 1;
    auto *in  = F::alloc_real(N);
    auto *out = F::alloc_complex(Nc);

    this->init_real(in, N);
    std::vector<real_t> orig(in, in + N);

    auto p = F::plan_dft_r2c_1d(N, in, out, FFTW_ESTIMATE);
    ASSERT_NE(p, nullptr);
    F::execute(p);

    for (int i = 0; i < N; i++)
    {
        EXPECT_EQ(in[i], orig[i])
            << "R2C out-of-place modified input at i=" << i;
    }

    F::destroy_plan(p);
    F::free_fn(in);
    F::free_fn(out);
}

TYPED_TEST(FftwWrapperCorrectnessTest,
           PTEST_OOP_C2C_2D_INPUT_PRESERVED)
{
    using F = FftwTypes<TypeParam>;
    const int N0 = 4, N1 = 8, total = N0 * N1;
    auto *in  = F::alloc_complex(total);
    auto *out = F::alloc_complex(total);

    this->init_complex(in, total);
    std::vector<typename F::complex_t> orig(total);
    std::memcpy(orig.data(), in, sizeof(typename F::complex_t) * total);

    auto p = F::plan_dft_2d(N0, N1, in, out, FFTW_FORWARD, FFTW_ESTIMATE);
    ASSERT_NE(p, nullptr);
    F::execute(p);

    for (int i = 0; i < total; i++)
    {
        EXPECT_EQ(in[i][0], orig[i][0])
            << "C2C 2D out-of-place modified input real at i=" << i;
        EXPECT_EQ(in[i][1], orig[i][1])
            << "C2C 2D out-of-place modified input imag at i=" << i;
    }

    F::destroy_plan(p);
    F::free_fn(in);
    F::free_fn(out);
}
