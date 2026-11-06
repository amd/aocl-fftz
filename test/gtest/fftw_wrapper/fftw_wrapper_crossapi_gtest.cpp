// Copyright Advanced Micro Devices, Inc.
// SPDX-License-Identifier: BSD-3-Clause

/**
 * @file fftw_wrapper_crossapi_gtest.cpp
 * @brief Cross-API comparison tests: compute the same transform through
 *        different wrapper code paths (get_r2c_dv_desc vs
 *        get_many_r2c_dv_desc, etc.) and compare outputs.
 *        A divergence means one code path has a stride/dimension bug.
 *        Also: batched 2D R2C/C2R, interleaved strides, in-place
 *        multi-dim R2C, guru multi-dim howmany.
 */

#include <cstring>
#include <vector>
#include "fftw_wrapper_test_utils.h"

template<typename T>
class FftwWrapperCrossApiTest : public FftwWrapperTestBase<T>
{
};
TYPED_TEST_SUITE(FftwWrapperCrossApiTest, FftwTestTypes);

/* =====================================================================
 * 1. plan_dft_r2c_2d vs plan_many_dft_r2c(rank=2, howmany=1)
 *    get_r2c_dv_desc vs get_many_r2c_dv_desc -- DIFFERENT code,
 *    must produce identical output for out-of-place.
 * ===================================================================== */

TYPED_TEST(FftwWrapperCrossApiTest,
           PTEST_R2C_2D_PLAN_VS_MANY_OOP)
{
    using F = FftwTypes<TypeParam>;
    using real_t = typename F::real_t;
    using complex_t [[maybe_unused]] = typename F::complex_t;
    const int N0 = 5, N1 = 7;
    const int total = N0 * N1;
    const int Nc = N0 * (N1 / 2 + 1);

    auto *in   = F::alloc_real(total);
    auto *out1 = F::alloc_complex(Nc);
    auto *out2 = F::alloc_complex(Nc);

    this->init_real_iota(in, total);

    auto p1 = F::plan_dft_r2c_2d(
        N0, N1, in, out1, FFTW_ESTIMATE);
    ASSERT_NE(p1, nullptr);

    int n_arr[] = {N0, N1};
    auto p2 = F::plan_many_dft_r2c(
        2, n_arr, 1,
        in, nullptr, 1, total,
        out2, nullptr, 1, Nc,
        FFTW_ESTIMATE);
    ASSERT_NE(p2, nullptr);

    F::execute(p1);

    this->init_real_iota(in, total);
    F::execute(p2);

    compare_complex_arrays(out1, out2, Nc, dft_tolerance<F>(total));

    F::destroy_plan(p1);
    F::destroy_plan(p2);
    F::free_fn(in);
    F::free_fn(out1);
    F::free_fn(out2);
}

/* =====================================================================
 * 2. plan_dft_c2r_2d vs plan_many_dft_c2r(rank=2, howmany=1)
 *    get_c2r_dv_desc vs get_many_c2r_dv_desc -- DIFFERENT code.
 * ===================================================================== */

TYPED_TEST(FftwWrapperCrossApiTest,
           PTEST_C2R_2D_PLAN_VS_MANY_OOP)
{
    using F = FftwTypes<TypeParam>;
    using real_t = typename F::real_t;
    using complex_t [[maybe_unused]] = typename F::complex_t;
    const int N0 = 5, N1 = 7;
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

    int n_arr[] = {N0, N1};
    auto p2 = F::plan_many_dft_c2r(
        2, n_arr, 1,
        freq2, nullptr, 1, Nc,
        out2, nullptr, 1, total,
        FFTW_ESTIMATE);
    ASSERT_NE(p2, nullptr);

    F::execute(p1);
    F::execute(p2);

    compare_real_arrays(out1, out2, total, dft_tolerance<F>(total));

    F::destroy_plan(pr);
    F::destroy_plan(p1);
    F::destroy_plan(p2);
    F::free_fn(real_in);
    F::free_fn(freq);
    F::free_fn(freq2);
    F::free_fn(out1);
    F::free_fn(out2);
}

/* =====================================================================
 * 3. plan_dft_r2c_3d vs plan_many_dft_r2c(rank=3, howmany=1)
 *    3D variant -- most complex stride chain.
 * ===================================================================== */

TYPED_TEST(FftwWrapperCrossApiTest,
           PTEST_R2C_3D_PLAN_VS_MANY_OOP)
{
    using F = FftwTypes<TypeParam>;
    using real_t = typename F::real_t;
    using complex_t [[maybe_unused]] = typename F::complex_t;
    const int N0 = 3, N1 = 4, N2 = 5;
    const int total = N0 * N1 * N2;
    const int Nc = N0 * N1 * (N2 / 2 + 1);

    auto *in   = F::alloc_real(total);
    auto *out1 = F::alloc_complex(Nc);
    auto *out2 = F::alloc_complex(Nc);

    this->init_real_iota(in, total);

    auto p1 = F::plan_dft_r2c_3d(
        N0, N1, N2, in, out1, FFTW_ESTIMATE);
    ASSERT_NE(p1, nullptr);

    int n_arr[] = {N0, N1, N2};
    auto p2 = F::plan_many_dft_r2c(
        3, n_arr, 1,
        in, nullptr, 1, total,
        out2, nullptr, 1, Nc,
        FFTW_ESTIMATE);
    ASSERT_NE(p2, nullptr);

    F::execute(p1);

    this->init_real_iota(in, total);
    F::execute(p2);

    compare_complex_arrays(out1, out2, Nc, dft_tolerance<F>(total));

    F::destroy_plan(p1);
    F::destroy_plan(p2);
    F::free_fn(in);
    F::free_fn(out1);
    F::free_fn(out2);
}

/* =====================================================================
 * 4. Batched 2D R2C (rank=2, howmany=2): verify each batch against
 *    a single plan_dft_r2c_2d result.
 *    Targets: get_many_r2c_dv_desc vecs + 2D stride interaction.
 * ===================================================================== */

TYPED_TEST(FftwWrapperCrossApiTest,
           PTEST_MANY_R2C_2D_BATCH_VS_SINGLE)
{
    using F = FftwTypes<TypeParam>;
    using real_t = typename F::real_t;
    using complex_t [[maybe_unused]] = typename F::complex_t;
    const int N0 = 4, N1 = 6;
    const int total = N0 * N1;
    const int Nc = N0 * (N1 / 2 + 1);
    const int howmany = 2;

    auto *in_batch = F::alloc_real(total * howmany);
    auto *out_batch = F::alloc_complex(Nc * howmany);
    auto *in_single = F::alloc_real(total);
    auto *out_single = F::alloc_complex(Nc);

    this->init_real_strided(in_batch, total, 1, total, howmany);

    int n_arr[] = {N0, N1};
    auto pb = F::plan_many_dft_r2c(
        2, n_arr, howmany,
        in_batch, nullptr, 1, total,
        out_batch, nullptr, 1, Nc,
        FFTW_ESTIMATE);
    ASSERT_NE(pb, nullptr);
    F::execute(pb);

    double tol = dft_tolerance<F>(total);
    for (int b = 0; b < howmany; b++)
    {
        gather_real(in_single, &in_batch[b * total], total, 1);
        auto ps = F::plan_dft_r2c_2d(
            N0, N1, in_single, out_single, FFTW_ESTIMATE);
        ASSERT_NE(ps, nullptr);
        F::execute(ps);

        compare_complex_arrays(out_single, &out_batch[b * Nc], Nc, tol);
        F::destroy_plan(ps);
    }

    F::destroy_plan(pb);
    F::free_fn(in_batch);
    F::free_fn(out_batch);
    F::free_fn(in_single);
    F::free_fn(out_single);
}

/* =====================================================================
 * 5. Batched 2D C2R (rank=2, howmany=2): roundtrip R2C -> C2R,
 *    verify each batch recovers original * N.
 *    Targets: get_many_c2r_dv_desc vecs + 2D out_stride
 * ===================================================================== */

TYPED_TEST(FftwWrapperCrossApiTest,
           PTEST_MANY_C2R_2D_BATCH_ROUNDTRIP)
{
    using F = FftwTypes<TypeParam>;
    using real_t = typename F::real_t;
    using complex_t [[maybe_unused]] = typename F::complex_t;
    const int N0 = 4, N1 = 6;
    const int total = N0 * N1;
    const int Nc = N0 * (N1 / 2 + 1);
    const int howmany = 2;

    auto *in   = F::alloc_real(total * howmany);
    auto *freq = F::alloc_complex(Nc * howmany);
    auto *out  = F::alloc_real(total * howmany);

    this->init_real_strided(in, total, 1, total, howmany);

    int n_arr[] = {N0, N1};
    auto pr = F::plan_many_dft_r2c(
        2, n_arr, howmany,
        in, nullptr, 1, total,
        freq, nullptr, 1, Nc,
        FFTW_ESTIMATE);
    auto pc = F::plan_many_dft_c2r(
        2, n_arr, howmany,
        freq, nullptr, 1, Nc,
        out, nullptr, 1, total,
        FFTW_ESTIMATE);
    ASSERT_NE(pr, nullptr);
    ASSERT_NE(pc, nullptr);

    F::execute(pr);
    F::execute(pc);

    double tol = dft_tolerance<F>(total);
    for (int b = 0; b < howmany; b++)
    {
        compare_real_scaled(&in[b * total], &out[b * total], total, total, tol);
    }

    F::destroy_plan(pr);
    F::destroy_plan(pc);
    F::free_fn(in);
    F::free_fn(freq);
    F::free_fn(out);
}

/* =====================================================================
 * 6. Interleaved C2C: plan_many_dft with istride=2, ostride=2.
 *    Every other complex element belongs to the transform.
 *    Compare against non-interleaved plan_dft_1d result.
 * ===================================================================== */

TYPED_TEST(FftwWrapperCrossApiTest,
           PTEST_MANY_DFT_INTERLEAVED_STRIDE)
{
    using F = FftwTypes<TypeParam>;
    using complex_t [[maybe_unused]] = typename F::complex_t;
    const int N = 8;
    const int stride = 2;
    const int buf_size = N * stride;

    auto *in_inter = F::alloc_complex(buf_size);
    auto *out_inter = F::alloc_complex(buf_size);
    auto *in_contig = F::alloc_complex(N);
    auto *out_contig = F::alloc_complex(N);
    auto *out_line = F::alloc_complex(N);
    ASSERT_NE(in_inter, nullptr);
    ASSERT_NE(out_inter, nullptr);
    ASSERT_NE(in_contig, nullptr);
    ASSERT_NE(out_contig, nullptr);
    ASSERT_NE(out_line, nullptr);

    this->zero_complex(in_inter, buf_size);
    this->init_complex_strided(in_inter, N, stride, buf_size, 1);
    gather_complex(in_contig, in_inter, N, stride);

    int n_arr[] = {N};
    auto pi = F::plan_many_dft(
        1, n_arr, 1,
        in_inter, nullptr, stride, 0,
        out_inter, nullptr, stride, 0,
        FFTW_FORWARD, FFTW_ESTIMATE);
    ASSERT_NE(pi, nullptr);

    auto pc = F::plan_dft_1d(
        N, in_contig, out_contig,
        FFTW_FORWARD, FFTW_ESTIMATE);
    ASSERT_NE(pc, nullptr);

    F::execute(pi);
    F::execute(pc);

    gather_complex(out_line, out_inter, N, stride);
    compare_complex_arrays(out_contig, out_line, N, dft_tolerance<F>(N));

    F::destroy_plan(pi);
    F::destroy_plan(pc);
    F::free_fn(in_inter);
    F::free_fn(out_inter);
    F::free_fn(in_contig);
    F::free_fn(out_contig);
    F::free_fn(out_line);
}

/* =====================================================================
 * 7. 3D R2C in-place roundtrip.
 *    The buffer must be padded: last real dim = 2*(N2/2+1).
 *    Targets: get_r2c_dv_desc / get_c2r_dv_desc in-place 3D path.
 * ===================================================================== */

TYPED_TEST(FftwWrapperCrossApiTest,
           PTEST_R2C_3D_INPLACE_ROUNDTRIP)
{
    using F = FftwTypes<TypeParam>;
    using real_t = typename F::real_t;
    using complex_t [[maybe_unused]] = typename F::complex_t;
    const int N0 = 3, N1 = 4, N2 = 6;
    const int padded_N2 = 2 * (N2 / 2 + 1);
    const int buf_reals = N0 * N1 * padded_N2;
    const int Nc [[maybe_unused]] = N0 * N1 * (N2 / 2 + 1);

    auto *buf = (real_t *)F::malloc_fn(
        sizeof(real_t) * buf_reals);
    ASSERT_NE(buf, nullptr);

    std::memset(buf, 0, sizeof(real_t) * buf_reals);

    /* Contiguous logical block saved[p]=p+1, scattered into the padded
     * in-place buffer (last real dim N2 -> padded_N2). */
    auto *saved = F::alloc_real(N0 * N1 * N2);
    ASSERT_NE(saved, nullptr);
    this->init_real_iota(saved, N0 * N1 * N2);
    this->scatter_padded_real(buf, saved, static_cast<long>(N0) * N1, N2,
                              padded_N2);

    complex_t *cbuf = (complex_t *)buf;
    auto pr = F::plan_dft_r2c_3d(
        N0, N1, N2, buf, cbuf, FFTW_ESTIMATE);
    auto pc = F::plan_dft_c2r_3d(
        N0, N1, N2, cbuf, buf, FFTW_ESTIMATE);
    ASSERT_NE(pr, nullptr);
    ASSERT_NE(pc, nullptr);

    F::execute(pr);
    F::execute(pc);

    const int total = N0 * N1 * N2;
    double tol = dft_tolerance<F>(total);

    /* Gather the non-padded reals back and check the roundtrip (gain=total). */
    std::vector<real_t> result(total);
    this->gather_padded_real(result.data(), buf, static_cast<long>(N0) * N1,
                             N2, padded_N2);
    compare_real_scaled(saved, result.data(), total, total, tol);

    F::destroy_plan(pr);
    F::destroy_plan(pc);
    F::free_fn(buf);
    F::free_fn(saved);
}

/* =====================================================================
 * 8. 2D R2C in-place with ODD last dimension.
 *    padded_N1 = 2*(N1/2+1) which for odd N1 differs from N1+1.
 *    Roundtrip verifies stride correctness.
 * ===================================================================== */

TYPED_TEST(FftwWrapperCrossApiTest,
           PTEST_R2C_2D_INPLACE_ODD_ROUNDTRIP)
{
    using F = FftwTypes<TypeParam>;
    using real_t = typename F::real_t;
    using complex_t [[maybe_unused]] = typename F::complex_t;
    const int N0 = 4, N1 = 7;
    const int padded_N1 = 2 * (N1 / 2 + 1);
    const int buf_reals = N0 * padded_N1;
    const int Nc [[maybe_unused]] = N0 * (N1 / 2 + 1);

    auto *buf = (real_t *)F::malloc_fn(
        sizeof(real_t) * buf_reals);
    ASSERT_NE(buf, nullptr);

    std::memset(buf, 0, sizeof(real_t) * buf_reals);

    /* Contiguous logical block saved[p]=p+1, scattered into the padded
     * in-place buffer (odd last dim N1 -> padded_N1). */
    auto *saved = F::alloc_real(N0 * N1);
    ASSERT_NE(saved, nullptr);
    this->init_real_iota(saved, N0 * N1);
    this->scatter_padded_real(buf, saved, N0, N1, padded_N1);

    complex_t *cbuf = (complex_t *)buf;
    auto pr = F::plan_dft_r2c_2d(
        N0, N1, buf, cbuf, FFTW_ESTIMATE);
    auto pc = F::plan_dft_c2r_2d(
        N0, N1, cbuf, buf, FFTW_ESTIMATE);
    ASSERT_NE(pr, nullptr);
    ASSERT_NE(pc, nullptr);

    F::execute(pr);
    F::execute(pc);

    const int total = N0 * N1;
    double tol = dft_tolerance<F>(total);

    /* Gather the non-padded reals back and check the roundtrip (gain=total). */
    std::vector<real_t> result(total);
    this->gather_padded_real(result.data(), buf, N0, N1, padded_N1);
    compare_real_scaled(saved, result.data(), total, total, tol);

    F::destroy_plan(pr);
    F::destroy_plan(pc);
    F::free_fn(buf);
    F::free_fn(saved);
}

/* =====================================================================
 * 9. plan_many_dft rank=2 with howmany=2: each batch is a 2D C2C.
 *    Compare each batch output against single plan_dft_2d result.
 *    Targets: get_many_dv_desc vecs + 2D dim interaction.
 * ===================================================================== */

TYPED_TEST(FftwWrapperCrossApiTest,
           PTEST_MANY_DFT_2D_BATCH_VS_SINGLE)
{
    using F = FftwTypes<TypeParam>;
    using complex_t [[maybe_unused]] = typename F::complex_t;
    const int N0 = 3, N1 = 5;
    const int total = N0 * N1;
    const int howmany = 2;

    auto *in_batch  = F::alloc_complex(total * howmany);
    auto *out_batch = F::alloc_complex(total * howmany);
    auto *in_single  = F::alloc_complex(total);
    auto *out_single = F::alloc_complex(total);

    this->init_complex_strided(in_batch, total, 1, total, howmany);

    int n_arr[] = {N0, N1};
    auto pb = F::plan_many_dft(
        2, n_arr, howmany,
        in_batch, nullptr, 1, total,
        out_batch, nullptr, 1, total,
        FFTW_FORWARD, FFTW_ESTIMATE);
    ASSERT_NE(pb, nullptr);
    F::execute(pb);

    double tol = dft_tolerance<F>(total);
    for (int b = 0; b < howmany; b++)
    {
        gather_complex(in_single, &in_batch[b * total], total, 1);
        auto ps = F::plan_dft_2d(
            N0, N1, in_single, out_single,
            FFTW_FORWARD, FFTW_ESTIMATE);
        ASSERT_NE(ps, nullptr);
        F::execute(ps);

        compare_complex_arrays(out_single, &out_batch[b * total], total, tol);
        F::destroy_plan(ps);
    }

    F::destroy_plan(pb);
    F::free_fn(in_batch);
    F::free_fn(out_batch);
    F::free_fn(in_single);
    F::free_fn(out_single);
}

/* =====================================================================
 * 10. Guru with howmany_rank=2: multi-dimensional batching.
 *     3 batches along one axis, 2 along another.
 *     Compare each (batch_a, batch_b) result against plan_dft_1d.
 *     Targets: get_guru_dv_desc howmany_rank>1 reversal.
 * ===================================================================== */

TYPED_TEST(FftwWrapperCrossApiTest,
           PTEST_GURU_MULTIDIM_HOWMANY)
{
    using F = FftwTypes<TypeParam>;
    using complex_t [[maybe_unused]] = typename F::complex_t;
    using iodim_t = typename F::iodim_t;
    const int N = 8;
    const int B0 = 2, B1 = 3;
    const int total = N * B0 * B1;

    auto *in  = F::alloc_complex(total);
    auto *out = F::alloc_complex(total);
    auto *ref = F::alloc_complex(N);

    this->init_complex_strided(in, total, 1, total, 1);

    iodim_t dims[1] = {{N, 1, 1}};
    iodim_t howmany[2] = {
        {B0, N * B1, N * B1},
        {B1, N, N}
    };

    auto p = F::plan_guru_dft(
        1, dims, 2, howmany,
        in, out, FFTW_FORWARD, FFTW_ESTIMATE);
    ASSERT_NE(p, nullptr);
    F::execute(p);

    /* Per-transform size is N; the two batch dims (B0,B1) run independent
     * transforms and must not inflate the tolerance (see dft_tolerance). */
    double tol = dft_tolerance<F>(N);
    for (int b0 = 0; b0 < B0; b0++)
    {
        for (int b1 = 0; b1 < B1; b1++)
        {
            int offset = b0 * N * B1 + b1 * N;
            dft_reference_1d(
                &in[offset], ref, N, FFTW_FORWARD);
            compare_complex_arrays(ref, &out[offset], N, tol);
        }
    }

    F::destroy_plan(p);
    F::free_fn(in);
    F::free_fn(out);
    F::free_fn(ref);
}

/* =====================================================================
 * 11. plan_many_dft_r2c with non-NULL inembed (padded 2D real input).
 *     The real input is embedded in a wider array. The output uses
 *     default embedding.  Verify against single R2C plan.
 *     Targets: get_many_r2c_dv_desc inembed[rank-i] branch.
 * ===================================================================== */

TYPED_TEST(FftwWrapperCrossApiTest,
           PTEST_MANY_R2C_2D_CUSTOM_INEMBED)
{
    using F = FftwTypes<TypeParam>;
    using real_t = typename F::real_t;
    using complex_t [[maybe_unused]] = typename F::complex_t;
    const int N0 = 4, N1 = 6;
    const int E0 = 4, E1 = 8;
    const int Nc = N0 * (N1 / 2 + 1);
    const int total_real = N0 * N1;

    auto *in_padded = F::alloc_real(E0 * E1);
    auto *in_contig = F::alloc_real(total_real);
    auto *out1 = F::alloc_complex(Nc);
    auto *out2 = F::alloc_complex(Nc);

    /* Same iota logical block fed two ways: contiguous, and scattered into a
     * row-padded (stride E1) input for the custom-inembed path. */
    std::memset(in_padded, 0, sizeof(real_t) * E0 * E1);
    this->init_real_iota(in_contig, total_real);
    this->scatter_padded_real(in_padded, in_contig, N0, N1, E1);

    int n_arr[] = {N0, N1};
    int inembed_arr[] = {E0, E1};
    auto p1 = F::plan_many_dft_r2c(
        2, n_arr, 1,
        in_padded, inembed_arr, 1, E0 * E1,
        out1, nullptr, 1, Nc,
        FFTW_ESTIMATE);
    ASSERT_NE(p1, nullptr);

    auto p2 = F::plan_dft_r2c_2d(
        N0, N1, in_contig, out2, FFTW_ESTIMATE);
    ASSERT_NE(p2, nullptr);

    F::execute(p1);
    F::execute(p2);

    compare_complex_arrays(out1, out2, Nc, dft_tolerance<F>(total_real));

    F::destroy_plan(p1);
    F::destroy_plan(p2);
    F::free_fn(in_padded);
    F::free_fn(in_contig);
    F::free_fn(out1);
    F::free_fn(out2);
}

/* =====================================================================
 * 12. plan_dft_2d vs plan_many_dft(rank=2, howmany=1): C2C cross-API.
 *     get_dv_desc vs get_many_dv_desc -- different stride code.
 * ===================================================================== */

TYPED_TEST(FftwWrapperCrossApiTest,
           PTEST_C2C_2D_PLAN_VS_MANY)
{
    using F = FftwTypes<TypeParam>;
    using complex_t [[maybe_unused]] = typename F::complex_t;
    const int N0 = 5, N1 = 7;
    const int total = N0 * N1;

    auto *in   = F::alloc_complex(total);
    auto *out1 = F::alloc_complex(total);
    auto *out2 = F::alloc_complex(total);
    auto *ref  = F::alloc_complex(total);

    this->init_complex(in, total);

    auto p1 = F::plan_dft_2d(
        N0, N1, in, out1, FFTW_FORWARD, FFTW_ESTIMATE);
    ASSERT_NE(p1, nullptr);

    int n_arr[] = {N0, N1};
    auto p2 = F::plan_many_dft(
        2, n_arr, 1,
        in, nullptr, 1, total,
        out2, nullptr, 1, total,
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
