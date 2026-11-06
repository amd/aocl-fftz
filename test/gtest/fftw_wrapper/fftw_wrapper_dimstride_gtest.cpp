// Copyright Advanced Micro Devices, Inc.
// SPDX-License-Identifier: BSD-3-Clause

/** @file fftw_wrapper_dimstride_gtest.cpp
 * @brief GTest cases for FFTW wrapper dimension ordering and stride
 *        translation.
 */

/*
 * Tests focused on the FFTW-to-FFTZ dimension ordering and stride
 * translation.  The translator reverses dimension arrays:
 *   dims[i].n = n[rank - i - 1]
 * These tests use non-square / non-cubic arrays so that any
 * dimension-swap bug produces a detectably wrong result.
 */

#include <vector>
#include <cstring>
#include "fftw_wrapper_test_utils.h"

template<typename T>
class FftwWrapperDimStrideTest : public FftwWrapperTestBase<T>
{
};
TYPED_TEST_SUITE(FftwWrapperDimStrideTest, FftwTestTypes);

/* ====================================================================
 * Dimension-reversal tests (C2C)
 * ==================================================================== */

TYPED_TEST(FftwWrapperDimStrideTest, PTEST_DIM_ORDER_2D_ROW_MAJOR)
{
    using F = FftwTypes<TypeParam>;
    using complex_t [[maybe_unused]] = typename F::complex_t;
    const int n0 = 4, n1 = 8, total = n0 * n1;

    auto *in  = F::alloc_complex(total);
    auto *out = F::alloc_complex(total);
    this->init_complex_strided(in, total, 1, total, 1);

    auto p = F::plan_dft_2d(n0, n1, in, out, FFTW_FORWARD, FFTW_ESTIMATE);
    ASSERT_NE(p, nullptr);
    F::execute(p);

    /* Forward-vs-reference: a roundtrip alone cannot catch a dimension-swap
     * bug because forward and backward apply the same wrong shape and cancel
     * (back == in*total either way).  Comparing the forward spectrum against
     * an independent reference with the correct (n0,n1) shape does catch it. */
    auto *ref = F::alloc_complex(total);
    dft_reference_2d(in, ref, n0, n1, FFTW_FORWARD);
    compare_complex_arrays(ref, out, total, dft_tolerance<F>(total));

    auto *back = F::alloc_complex(total);
    auto pb = F::plan_dft_2d(n0, n1, out, back, FFTW_BACKWARD, FFTW_ESTIMATE);
    ASSERT_NE(pb, nullptr);
    F::execute(pb);

    compare_complex_scaled(in, back, total, static_cast<double>(total),
                           dft_tolerance<F>(total));

    F::destroy_plan(p);
    F::destroy_plan(pb);
    F::free_fn(in);
    F::free_fn(out);
    F::free_fn(ref);
    F::free_fn(back);
}

TYPED_TEST(FftwWrapperDimStrideTest, PTEST_DIM_ORDER_3D_ROW_MAJOR)
{
    using F = FftwTypes<TypeParam>;
    const int n0 = 2, n1 = 3, n2 = 4, total = n0 * n1 * n2;

    auto *in  = F::alloc_complex(total);
    auto *out = F::alloc_complex(total);
    this->init_complex_real_iota(in, total);

    auto pf = F::plan_dft_3d(n0, n1, n2, in, out, FFTW_FORWARD, FFTW_ESTIMATE);
    ASSERT_NE(pf, nullptr);
    F::execute(pf);

    /* Forward-vs-reference catches an axis-permutation bug the roundtrip
     * below cannot (non-cubic 2x3x4). */
    auto *ref = F::alloc_complex(total);
    dft_reference_3d(in, ref, n0, n1, n2, FFTW_FORWARD);
    compare_complex_arrays(ref, out, total, dft_tolerance<F>(total));

    auto *back = F::alloc_complex(total);
    auto pb = F::plan_dft_3d(n0, n1, n2, out, back, FFTW_BACKWARD,
                            FFTW_ESTIMATE);
    ASSERT_NE(pb, nullptr);
    F::execute(pb);

    compare_complex_scaled(in, back, total, total, dft_tolerance<F>(total));

    F::destroy_plan(pf);
    F::destroy_plan(pb);
    F::free_fn(in);
    F::free_fn(out);
    F::free_fn(ref);
    F::free_fn(back);
}

TYPED_TEST(FftwWrapperDimStrideTest, PTEST_DIM_ORDER_4D)
{
    using F = FftwTypes<TypeParam>;
    int n[] = {2, 3, 4, 5};
    int total = 2 * 3 * 4 * 5;

    auto *in   = F::alloc_complex(total);
    auto *out  = F::alloc_complex(total);
    auto *back = F::alloc_complex(total);
    this->init_complex_strided(in, total, 1, total, 1);

    auto pf = F::plan_dft(4, n, in, out, FFTW_FORWARD, FFTW_ESTIMATE);
    auto pb = F::plan_dft(4, n, out, back, FFTW_BACKWARD, FFTW_ESTIMATE);
    ASSERT_NE(pf, nullptr);
    ASSERT_NE(pb, nullptr);
    F::execute(pf);
    F::execute(pb);

    /* Forward-vs-reference catches a rank-4 axis-permutation bug. */
    auto *ref = F::alloc_complex(total);
    dft_reference_nd(in, ref, 4, n, FFTW_FORWARD);
    compare_complex_arrays(ref, out, total, dft_tolerance<F>(total));

    compare_complex_scaled(in, back, total, total, dft_tolerance<F>(total));

    F::destroy_plan(pf);
    F::destroy_plan(pb);
    F::free_fn(in);
    F::free_fn(out);
    F::free_fn(ref);
    F::free_fn(back);
}

/* plan_dft_2d(n0,n1,...) should produce identical output to
 * plan_dft(2,{n0,n1},...) */
TYPED_TEST(FftwWrapperDimStrideTest, PTEST_DIM_ORDER_ND_VIA_GENERIC)
{
    using F = FftwTypes<TypeParam>;
    const int n0 = 4, n1 = 6, total = n0 * n1;

    auto *in   = F::alloc_complex(total);
    auto *out1 = F::alloc_complex(total);
    auto *out2 = F::alloc_complex(total);

    this->init_complex_ramp(in, total);

    auto p1 = F::plan_dft_2d(n0, n1, in, out1, FFTW_FORWARD, FFTW_ESTIMATE);
    int n[] = {n0, n1};
    auto p2 = F::plan_dft(2, n, in, out2, FFTW_FORWARD, FFTW_ESTIMATE);
    ASSERT_NE(p1, nullptr);
    ASSERT_NE(p2, nullptr);

    F::execute(p1);
    F::execute(p2);

    compare_complex_arrays(out1, out2, total, dft_tolerance<F>(total));

    F::destroy_plan(p1);
    F::destroy_plan(p2);
    F::free_fn(in);
    F::free_fn(out1);
    F::free_fn(out2);
}

/* ====================================================================
 * Stride tests (many / advanced interface)
 * ==================================================================== */

TYPED_TEST(FftwWrapperDimStrideTest, PTEST_MANY_DFT_STRIDED_BATCH)
{
    using F = FftwTypes<TypeParam>;
    const int N = 8, howmany = 4;
    auto *in  = F::alloc_complex(N * howmany);
    auto *out = F::alloc_complex(N * howmany);
    auto *ref = F::alloc_complex(N);

    this->init_complex_real_iota(in, N * howmany);

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

TYPED_TEST(FftwWrapperDimStrideTest, PTEST_MANY_DFT_INTERLEAVED_BATCH)
{
    using F = FftwTypes<TypeParam>;
    const int N = 8, howmany = 3;
    int total = N * howmany;
    auto *in  = F::alloc_complex(total);
    auto *out = F::alloc_complex(total);

    this->init_complex_real_iota(in, total);

    int n[] = {N};
    auto p = F::plan_many_dft(1, n, howmany, in, nullptr, howmany, 1,
                             out, nullptr, howmany, 1,
                             FFTW_FORWARD, FFTW_ESTIMATE);
    ASSERT_NE(p, nullptr);
    F::execute(p);

    /* Verify every interleaved batch against an independent reference.
     * istride=idist... here stride=howmany, dist=1, so batch b element i is
     * at b + i*howmany; gather each line to reuse the contiguous reference. */
    auto *line = F::alloc_complex(N);
    auto *ref  = F::alloc_complex(N);
    for (int b = 0; b < howmany; b++)
    {
        gather_complex(line, in, N, howmany, b);
        dft_reference_1d(line, ref, N, FFTW_FORWARD);
        auto *out_line = F::alloc_complex(N);
        gather_complex(out_line, out, N, howmany, b);
        compare_complex_arrays(ref, out_line, N, dft_tolerance<F>(N));
        F::free_fn(out_line);
    }

    F::destroy_plan(p);
    F::free_fn(in);
    F::free_fn(out);
    F::free_fn(line);
    F::free_fn(ref);
}

TYPED_TEST(FftwWrapperDimStrideTest, PTEST_MANY_DFT_NULL_EMBED)
{
    using F = FftwTypes<TypeParam>;
    const int N = 8, howmany = 2;
    auto *in  = F::alloc_complex(N * howmany);
    auto *out = F::alloc_complex(N * howmany);
    auto *ref = F::alloc_complex(N);

    this->init_complex_real_iota_strided(in, N, 1, N, howmany);

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

/* Transform each column of a 2D array (classic FFTW tutorial pattern):
 * rank=1, n={nrows}, howmany=ncols, istride=ncols, idist=1 */
TYPED_TEST(FftwWrapperDimStrideTest, PTEST_MANY_DFT_2D_COLUMN_TRANSFORM)
{
    using F = FftwTypes<TypeParam>;
    const int nrows = 8, ncols = 4;
    int total = nrows * ncols;
    auto *in  = F::alloc_complex(total);
    auto *out = F::alloc_complex(total);

    /* Give every column distinct data so a bug that mixes or drops columns is
     * detectable (identical columns would hide it).  Each column c is one
     * length-nrows transform at stride=ncols, dist=1 -- exactly the batched
     * strided layout init_complex_strided lays out. */
    this->init_complex_strided(in, nrows, ncols, 1, ncols);

    int n[] = {nrows};
    auto p = F::plan_many_dft(1, n, ncols, in, nullptr, ncols, 1,
                             out, nullptr, ncols, 1,
                             FFTW_FORWARD, FFTW_ESTIMATE);
    ASSERT_NE(p, nullptr);
    F::execute(p);

    /* Each column is an independent length-nrows transform at stride=ncols. */
    auto *line = F::alloc_complex(nrows);
    auto *ref  = F::alloc_complex(nrows);
    auto *out_line = F::alloc_complex(nrows);
    for (int c = 0; c < ncols; c++)
    {
        gather_complex(line, in, nrows, ncols, c);
        dft_reference_1d(line, ref, nrows, FFTW_FORWARD);
        gather_complex(out_line, out, nrows, ncols, c);
        compare_complex_arrays(ref, out_line, nrows, dft_tolerance<F>(nrows));
    }

    F::destroy_plan(p);
    F::free_fn(in);
    F::free_fn(out);
    F::free_fn(line);
    F::free_fn(ref);
    F::free_fn(out_line);
}

/* ====================================================================
 * R2C / C2R stride and padding tests
 * ==================================================================== */

TYPED_TEST(FftwWrapperDimStrideTest, PTEST_R2C_2D_OUT_OF_PLACE_STRIDES)
{
    using F = FftwTypes<TypeParam>;
    using real_t = typename F::real_t;
    const int n0 = 4, n1 = 6;
    int total_real = n0 * n1;
    int Nc1 = n1 / 2 + 1;
    int total_complex = n0 * Nc1;

    auto *in  = F::alloc_real(total_real);
    auto *out = F::alloc_complex(total_complex);
    this->init_real_iota(in, total_real);

    auto p = F::plan_dft_r2c_2d(n0, n1, in, out, FFTW_ESTIMATE);
    ASSERT_NE(p, nullptr);
    F::execute(p);

    auto *ref = F::alloc_complex(total_complex);
    dft_reference_r2c_2d(in, ref, n0, n1);
    compare_complex_arrays(ref, out, total_complex, dft_tolerance<F>(total_real));

    F::destroy_plan(p);
    F::free_fn(in);
    F::free_fn(out);
    F::free_fn(ref);
}

TYPED_TEST(FftwWrapperDimStrideTest, PTEST_R2C_2D_INPLACE_PADDING)
{
    using F = FftwTypes<TypeParam>;
    using real_t = typename F::real_t;
    using complex_t [[maybe_unused]] = typename F::complex_t;
    const int n0 = 4, n1 = 6;
    int Nc1 = n1 / 2 + 1;
    int padded_row = 2 * Nc1;
    int total_padded = n0 * padded_row;

    auto *buf = (real_t *)F::malloc_fn(sizeof(real_t) * total_padded);
    ASSERT_NE(buf, nullptr);
    std::memset(buf, 0, sizeof(real_t) * total_padded);

    /* Build the contiguous unpadded input independently (an in-place transform
     * overwrites buf), then scatter it into the row-padded layout. */
    int total_real = n0 * n1;
    int total_complex = n0 * Nc1;
    auto *real_in = F::alloc_real(total_real);
    this->init_real_iota(real_in, total_real);
    this->scatter_padded_real(buf, real_in, n0, n1, padded_row);

    auto p = F::plan_dft_r2c_2d(n0, n1, buf, (complex_t *)buf, FFTW_ESTIMATE);
    ASSERT_NE(p, nullptr);

    F::execute(p);

    /* padded_row == 2*Nc1, so the in-place complex output is a contiguous
     * n0 x Nc1 array. */
    auto *ref = F::alloc_complex(total_complex);
    dft_reference_r2c_2d(real_in, ref, n0, n1);
    compare_complex_arrays(ref, reinterpret_cast<complex_t *>(buf),
                           total_complex, dft_tolerance<F>(total_real));

    F::destroy_plan(p);
    F::free_fn(buf);
    F::free_fn(real_in);
    F::free_fn(ref);
}

TYPED_TEST(FftwWrapperDimStrideTest, PTEST_C2R_2D_OUT_OF_PLACE_STRIDES)
{
    using F = FftwTypes<TypeParam>;
    using real_t = typename F::real_t;
    const int n0 = 4, n1 = 6;
    int Nc1 = n1 / 2 + 1;

    auto *real_in  = F::alloc_real(n0 * n1);
    auto *freq     = F::alloc_complex(n0 * Nc1);
    auto *real_out = F::alloc_real(n0 * n1);

    this->init_real_iota(real_in, n0 * n1);

    auto pr = F::plan_dft_r2c_2d(n0, n1, real_in, freq, FFTW_ESTIMATE);
    auto pc = F::plan_dft_c2r_2d(n0, n1, freq, real_out, FFTW_ESTIMATE);
    ASSERT_NE(pr, nullptr);
    ASSERT_NE(pc, nullptr);

    F::execute(pr);
    F::execute(pc);

    int total = n0 * n1;
    compare_real_scaled(real_in, real_out, total, total, dft_tolerance<F>(total));

    F::destroy_plan(pr);
    F::destroy_plan(pc);
    F::free_fn(real_in);
    F::free_fn(freq);
    F::free_fn(real_out);
}

TYPED_TEST(FftwWrapperDimStrideTest, PTEST_MANY_R2C_BATCHED)
{
    using F = FftwTypes<TypeParam>;
    using real_t = typename F::real_t;
    const int N = 16, howmany = 4;
    int Nc = N / 2 + 1;

    auto *in  = F::alloc_real(N * howmany);
    auto *out = F::alloc_complex(Nc * howmany);

    this->init_real_iota(in, N * howmany);

    int n[] = {N};
    auto p = F::plan_many_dft_r2c(1, n, howmany, in, nullptr, 1, N,
                                  out, nullptr, 1, Nc, FFTW_ESTIMATE);
    ASSERT_NE(p, nullptr);
    F::execute(p);

    /* Each contiguous batch is an independent length-N R2C transform. */
    auto *ref = F::alloc_complex(Nc);
    for (int b = 0; b < howmany; b++)
    {
        dft_reference_r2c_1d(&in[b * N], ref, N);
        compare_complex_arrays(ref, &out[b * Nc], Nc, dft_tolerance<F>(N));
    }

    F::destroy_plan(p);
    F::free_fn(in);
    F::free_fn(out);
    F::free_fn(ref);
}

/* Exercises the non-NULL onembed (padded complex output) branch of the many
 * R2C interface, which the other R2C/C2R tests never hit (they all pass
 * nullptr embeds).  Output rows are padded to (Nc + pad) complex per batch. */
TYPED_TEST(FftwWrapperDimStrideTest, PTEST_MANY_R2C_PADDED_ONEMBED)
{
    using F = FftwTypes<TypeParam>;
    using real_t = typename F::real_t;
    const int N = 16, howmany = 3, pad = 2;
    int Nc = N / 2 + 1;
    int oembed = Nc + pad;

    auto *in  = F::alloc_real(N * howmany);
    auto *out = F::alloc_complex(oembed * howmany);
    this->zero_complex(out, oembed * howmany);
    this->init_real_strided(in, N, 1, N, howmany);

    int n[] = {N};
    int onembed[] = {oembed};
    auto p = F::plan_many_dft_r2c(1, n, howmany, in, nullptr, 1, N,
                                  out, onembed, 1, oembed, FFTW_ESTIMATE);
    ASSERT_NE(p, nullptr);
    F::execute(p);

    auto *ref = F::alloc_complex(Nc);
    for (int b = 0; b < howmany; b++)
    {
        dft_reference_r2c_1d(&in[b * N], ref, N);
        /* only the first Nc entries of each padded output row are valid */
        compare_complex_arrays(ref, &out[b * oembed], Nc, dft_tolerance<F>(N));
    }

    F::destroy_plan(p);
    F::free_fn(in);
    F::free_fn(out);
    F::free_fn(ref);
}

TYPED_TEST(FftwWrapperDimStrideTest, PTEST_MANY_C2R_BATCHED)
{
    using F = FftwTypes<TypeParam>;
    using real_t = typename F::real_t;
    const int N = 16, howmany = 4;
    int Nc = N / 2 + 1;

    auto *real_in  = F::alloc_real(N * howmany);
    auto *freq     = F::alloc_complex(Nc * howmany);
    auto *real_out = F::alloc_real(N * howmany);

    this->init_real_iota(real_in, N * howmany);

    int n[] = {N};
    auto pr = F::plan_many_dft_r2c(1, n, howmany, real_in, nullptr, 1, N,
                                   freq, nullptr, 1, Nc, FFTW_ESTIMATE);
    auto pc = F::plan_many_dft_c2r(1, n, howmany, freq, nullptr, 1, Nc,
                                   real_out, nullptr, 1, N, FFTW_ESTIMATE);
    ASSERT_NE(pr, nullptr);
    ASSERT_NE(pc, nullptr);

    F::execute(pr);
    F::execute(pc);

    compare_real_scaled(real_in, real_out, N * howmany, N, dft_tolerance<F>(N));

    F::destroy_plan(pr);
    F::destroy_plan(pc);
    F::free_fn(real_in);
    F::free_fn(freq);
    F::free_fn(real_out);
}

/* ====================================================================
 * Guru interface stride tests
 * ==================================================================== */

TYPED_TEST(FftwWrapperDimStrideTest, PTEST_GURU_EXPLICIT_STRIDES)
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

TYPED_TEST(FftwWrapperDimStrideTest, PTEST_GURU_HOWMANY_VECTOR)
{
    using F = FftwTypes<TypeParam>;
    using iodim_t = typename F::iodim_t;
    const int N = 8, batch = 4;
    iodim_t dims[1] = {{N, 1, 1}};
    iodim_t howmany[1] = {{batch, N, N}};

    auto *in  = F::alloc_complex(N * batch);
    auto *out = F::alloc_complex(N * batch);
    auto *ref = F::alloc_complex(N);

    this->init_complex_real_iota(in, N * batch);

    auto p = F::plan_guru_dft(1, dims, 1, howmany, in, out,
                             FFTW_FORWARD, FFTW_ESTIMATE);
    ASSERT_NE(p, nullptr);
    F::execute(p);

    for (int b = 0; b < batch; b++)
    {
        dft_reference_1d(&in[b * N], ref, N, FFTW_FORWARD);
        compare_complex_arrays(ref, &out[b * N], N, dft_tolerance<F>(N));
    }

    F::destroy_plan(p);
    F::free_fn(in);
    F::free_fn(out);
    F::free_fn(ref);
}

TYPED_TEST(FftwWrapperDimStrideTest, PTEST_GURU64_DFT_BASIC)
{
    using F = FftwTypes<TypeParam>;
    using iodim64_t = typename F::iodim64_t;
    const int N = 4;
    iodim64_t dims[1] = {{N, 1, 1}};
    iodim64_t howmany[1] = {{1, 1, 1}};

    auto *in  = F::alloc_complex(N);
    auto *out = F::alloc_complex(N);
    auto *ref = F::alloc_complex(N);
    this->init_complex(in, N);

    auto p = F::plan_guru64_dft(1, dims, 1, howmany, in, out,
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

// 0 howmany stride (length-1 batch) -> 1 in get_guru_dv_desc().
TYPED_TEST(FftwWrapperDimStrideTest, PTEST_GURU_HOWMANY_ZERO_STRIDE)
{
    using F = FftwTypes<TypeParam>;
    using iodim_t = typename F::iodim_t;
    const int N = 16;
    iodim_t dims[1] = {{N, 1, 1}};
    iodim_t howmany[1] = {{1, 0, 0}};

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

// 64-bit twin: same 0 -> 1 howmany-stride mapping, via get_guru_64_dv_desc().
TYPED_TEST(FftwWrapperDimStrideTest, PTEST_GURU64_HOWMANY_ZERO_STRIDE)
{
    using F = FftwTypes<TypeParam>;
    using iodim64_t = typename F::iodim64_t;
    const int N = 8;
    iodim64_t dims[1] = {{N, 1, 1}};
    iodim64_t howmany[1] = {{1, 0, 0}};

    auto *in  = F::alloc_complex(N);
    auto *out = F::alloc_complex(N);
    auto *ref = F::alloc_complex(N);
    this->init_complex(in, N);

    auto p = F::plan_guru64_dft(1, dims, 1, howmany, in, out,
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

// Guru r2c path: same 0 -> 1 howmany-stride mapping (get_guru_dv_desc()).
TYPED_TEST(FftwWrapperDimStrideTest, PTEST_GURU_R2C_HOWMANY_ZERO_STRIDE)
{
    using F = FftwTypes<TypeParam>;
    using iodim_t = typename F::iodim_t;
    const int N = 16;
    const int Nc = N / 2 + 1;
    iodim_t dims[1] = {{N, 1, 1}};
    iodim_t howmany[1] = {{1, 0, 0}};

    auto *in  = F::alloc_real(N);
    auto *out = F::alloc_complex(Nc);
    auto *ref = F::alloc_complex(Nc);
    this->init_real_iota(in, N);

    auto p = F::plan_guru_dft_r2c(1, dims, 1, howmany, in, out,
                                  FFTW_ESTIMATE);
    ASSERT_NE(p, nullptr);
    F::execute(p);
    dft_reference_r2c_1d(in, ref, N);
    compare_complex_arrays(ref, out, Nc, dft_tolerance<F>(N));

    F::destroy_plan(p);
    F::free_fn(in);
    F::free_fn(out);
    F::free_fn(ref);
}

// 64-bit guru r2c twin: 0 -> 1 mapping via get_guru_64_dv_desc().
TYPED_TEST(FftwWrapperDimStrideTest, PTEST_GURU64_R2C_HOWMANY_ZERO_STRIDE)
{
    using F = FftwTypes<TypeParam>;
    using iodim64_t = typename F::iodim64_t;
    const int N = 8;
    const int Nc = N / 2 + 1;
    iodim64_t dims[1] = {{N, 1, 1}};
    iodim64_t howmany[1] = {{1, 0, 0}};

    auto *in  = F::alloc_real(N);
    auto *out = F::alloc_complex(Nc);
    auto *ref = F::alloc_complex(Nc);
    this->init_real_iota(in, N);

    auto p = F::plan_guru64_dft_r2c(1, dims, 1, howmany, in, out,
                                    FFTW_ESTIMATE);
    ASSERT_NE(p, nullptr);
    F::execute(p);
    dft_reference_r2c_1d(in, ref, N);
    compare_complex_arrays(ref, out, Nc, dft_tolerance<F>(N));

    F::destroy_plan(p);
    F::free_fn(in);
    F::free_fn(out);
    F::free_fn(ref);
}

// Guru c2r path: same 0 -> 1 howmany-stride mapping (get_guru_dv_desc()).
TYPED_TEST(FftwWrapperDimStrideTest, PTEST_GURU_C2R_HOWMANY_ZERO_STRIDE)
{
    using F = FftwTypes<TypeParam>;
    using iodim_t = typename F::iodim_t;
    const int N = 16;
    const int Nc = N / 2 + 1;
    iodim_t dims[1] = {{N, 1, 1}};
    iodim_t howmany[1] = {{1, 0, 0}};

    auto *real_in  = F::alloc_real(N);
    auto *freq     = F::alloc_complex(Nc);
    auto *real_out = F::alloc_real(N);
    this->init_real_iota(real_in, N);

    // Build a Hermitian spectrum with a plain r2c, then invert it through the
    // guru c2r path under test; an unnormalized roundtrip scales by N.
    auto pf = F::plan_dft_r2c_1d(N, real_in, freq, FFTW_ESTIMATE);
    ASSERT_NE(pf, nullptr);
    F::execute(pf);

    auto p = F::plan_guru_dft_c2r(1, dims, 1, howmany, freq, real_out,
                                  FFTW_ESTIMATE);
    ASSERT_NE(p, nullptr);
    F::execute(p);
    compare_real_scaled(real_in, real_out, N, N, dft_tolerance<F>(N));

    F::destroy_plan(pf);
    F::destroy_plan(p);
    F::free_fn(real_in);
    F::free_fn(freq);
    F::free_fn(real_out);
}

// 64-bit guru c2r twin: 0 -> 1 mapping via get_guru_64_dv_desc().
TYPED_TEST(FftwWrapperDimStrideTest, PTEST_GURU64_C2R_HOWMANY_ZERO_STRIDE)
{
    using F = FftwTypes<TypeParam>;
    using iodim64_t = typename F::iodim64_t;
    const int N = 8;
    const int Nc = N / 2 + 1;
    iodim64_t dims[1] = {{N, 1, 1}};
    iodim64_t howmany[1] = {{1, 0, 0}};

    auto *real_in  = F::alloc_real(N);
    auto *freq     = F::alloc_complex(Nc);
    auto *real_out = F::alloc_real(N);
    this->init_real_iota(real_in, N);

    auto pf = F::plan_dft_r2c_1d(N, real_in, freq, FFTW_ESTIMATE);
    ASSERT_NE(pf, nullptr);
    F::execute(pf);

    auto p = F::plan_guru64_dft_c2r(1, dims, 1, howmany, freq, real_out,
                                    FFTW_ESTIMATE);
    ASSERT_NE(p, nullptr);
    F::execute(p);
    compare_real_scaled(real_in, real_out, N, N, dft_tolerance<F>(N));

    F::destroy_plan(pf);
    F::destroy_plan(p);
    F::free_fn(real_in);
    F::free_fn(freq);
    F::free_fn(real_out);
}

/* ====================================================================
 * Roundtrip dimension-consistency tests
 * ==================================================================== */

TYPED_TEST(FftwWrapperDimStrideTest, PTEST_ROUNDTRIP_2D_NON_SQUARE)
{
    using F = FftwTypes<TypeParam>;
    const int n0 = 4, n1 = 8, total = n0 * n1;

    auto *in   = F::alloc_complex(total);
    auto *freq = F::alloc_complex(total);
    auto *back = F::alloc_complex(total);
    this->init_complex(in, total);

    auto pf = F::plan_dft_2d(n0, n1, in, freq, FFTW_FORWARD, FFTW_ESTIMATE);
    auto pb = F::plan_dft_2d(n0, n1, freq, back, FFTW_BACKWARD, FFTW_ESTIMATE);
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

TYPED_TEST(FftwWrapperDimStrideTest, PTEST_ROUNDTRIP_3D_NON_CUBIC)
{
    using F = FftwTypes<TypeParam>;
    const int n0 = 2, n1 = 3, n2 = 5, total = n0 * n1 * n2;

    auto *in   = F::alloc_complex(total);
    auto *freq = F::alloc_complex(total);
    auto *back = F::alloc_complex(total);
    this->init_complex(in, total);

    auto pf = F::plan_dft_3d(n0, n1, n2, in, freq, FFTW_FORWARD, FFTW_ESTIMATE);
    auto pb = F::plan_dft_3d(n0, n1, n2, freq, back, FFTW_BACKWARD,
                             FFTW_ESTIMATE);
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

TYPED_TEST(FftwWrapperDimStrideTest, PTEST_ROUNDTRIP_R2C_C2R_2D)
{
    using F = FftwTypes<TypeParam>;
    using real_t = typename F::real_t;
    const int n0 = 4, n1 = 6, total = n0 * n1;
    int Nc1 = n1 / 2 + 1;

    auto *in   = F::alloc_real(total);
    auto *freq = F::alloc_complex(n0 * Nc1);
    auto *out  = F::alloc_real(total);

    this->init_real_iota(in, total);

    auto pr = F::plan_dft_r2c_2d(n0, n1, in, freq, FFTW_ESTIMATE);
    auto pc = F::plan_dft_c2r_2d(n0, n1, freq, out, FFTW_ESTIMATE);
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

TYPED_TEST(FftwWrapperDimStrideTest, PTEST_ROUNDTRIP_MANY_DFT_2D)
{
    using F = FftwTypes<TypeParam>;
    const int n0 = 4, n1 = 6, total = n0 * n1, howmany = 3;
    int grand_total = total * howmany;

    auto *in   = F::alloc_complex(grand_total);
    auto *freq = F::alloc_complex(grand_total);
    auto *back = F::alloc_complex(grand_total);

    this->init_complex_strided(in, grand_total, 1, grand_total, 1);

    int n[] = {n0, n1};
    auto pf = F::plan_many_dft(2, n, howmany, in, nullptr, 1, total,
                              freq, nullptr, 1, total,
                              FFTW_FORWARD, FFTW_ESTIMATE);
    auto pb = F::plan_many_dft(2, n, howmany, freq, nullptr, 1, total,
                              back, nullptr, 1, total,
                              FFTW_BACKWARD, FFTW_ESTIMATE);
    ASSERT_NE(pf, nullptr);
    ASSERT_NE(pb, nullptr);

    F::execute(pf);
    F::execute(pb);

    compare_complex_scaled(in, back, grand_total, total, dft_tolerance<F>(total));

    F::destroy_plan(pf);
    F::destroy_plan(pb);
    F::free_fn(in);
    F::free_fn(freq);
    F::free_fn(back);
}
