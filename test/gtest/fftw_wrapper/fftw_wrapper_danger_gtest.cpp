// Copyright Advanced Micro Devices, Inc.
// SPDX-License-Identifier: BSD-3-Clause

/**
 * @file fftw_wrapper_danger_gtest.cpp
 * @brief Dangerous-pattern and high-dimensional tests targeting the FFTW
 *        wrapper's handling of edge cases: high-rank R2C (rank=4,5),
 *        negative guru strides, cleanup-then-plan, plan_dft with
 *        invalid sign, Parseval theorem for batched transforms,
 *        and execute_dft with new-array on a many plan.
 */

#include <cmath>
#include <cstring>
#include <vector>
#include "fftw_wrapper_test_utils.h"

/* ===================================================================== */
/*  Test fixture                                                         */
/* ===================================================================== */
template<typename T>
class FftwWrapperDangerTest : public FftwWrapperTestBase<T>
{
protected:
    using F         = FftwTypes<T>;
    using real_t    = typename F::real_t;
    using complex_t = typename F::complex_t;
    using plan_t    = typename F::plan_t;
};

TYPED_TEST_SUITE(FftwWrapperDangerTest, FftwTestTypes);

/* ===================================================================== */
/*  1) Rank-4 R2C out-of-place roundtrip                                 */
/*     Tests get_r2c_dv_desc with rank=4, verifying padding and stride   */
/*     chaining for i>=2 are correct.                                     */
/* ===================================================================== */
TYPED_TEST(FftwWrapperDangerTest, PTEST_R2C_RANK4_OOP_ROUNDTRIP)
{
    using F = FftwTypes<TypeParam>;
    using real_t    = typename F::real_t;
    using complex_t = typename F::complex_t;

    const int D0 = 2, D1 = 3, D2 = 2, D3 = 4;
    const int Nc3 = D3 / 2 + 1;
    int n_arr[] = {D0, D1, D2, D3};
    const int total_real = D0 * D1 * D2 * D3;
    const int total_cplx = D0 * D1 * D2 * Nc3;

    real_t    *in  = F::alloc_real(total_real);
    complex_t *mid = F::alloc_complex(total_cplx);
    real_t    *out = F::alloc_real(total_real);
    ASSERT_NE(in, nullptr);
    ASSERT_NE(mid, nullptr);
    ASSERT_NE(out, nullptr);

    this->fill_ramp_real(in, total_real);

    auto pr = F::plan_dft_r2c(4, n_arr, in, mid, FFTW_ESTIMATE);
    if (!pr)
    {
        F::free_fn(in); F::free_fn(mid); F::free_fn(out);
        GTEST_SKIP() << "4D R2C plan creation not supported";
    }

    /* Re-seed the input before executing: only FFTW_ESTIMATE is guaranteed
     * not to modify the input buffer, so restore the known ramp. */
    this->fill_ramp_real(in, total_real);
    F::execute(pr);

    auto pc = F::plan_dft_c2r(4, n_arr, mid, out, FFTW_ESTIMATE);
    if (!pc)
    {
        F::destroy_plan(pr);
        F::free_fn(in); F::free_fn(mid); F::free_fn(out);
        GTEST_SKIP() << "4D C2R plan creation not supported";
    }
    F::execute(pc);

    compare_real_scaled(in, out, total_real, total_real,
                        dft_tolerance<F>(total_real));

    F::destroy_plan(pr);
    F::destroy_plan(pc);
    F::free_fn(in); F::free_fn(mid); F::free_fn(out);
}

/* ===================================================================== */
/*  2) Rank-5 C2C roundtrip                                              */
/*     Pushes the dimension reversal to rank=5 and verifies correctness. */
/* ===================================================================== */
TYPED_TEST(FftwWrapperDangerTest, PTEST_C2C_RANK5_ROUNDTRIP)
{
    using F = FftwTypes<TypeParam>;
    using complex_t = typename F::complex_t;

    const int D0 = 2, D1 = 2, D2 = 3, D3 = 2, D4 = 2;
    int n_arr[] = {D0, D1, D2, D3, D4};
    const int total = D0 * D1 * D2 * D3 * D4;

    complex_t *in  = F::alloc_complex(total);
    complex_t *mid = F::alloc_complex(total);
    complex_t *out = F::alloc_complex(total);
    ASSERT_NE(in, nullptr);
    ASSERT_NE(mid, nullptr);
    ASSERT_NE(out, nullptr);

    this->init_complex_wave(in, total);

    auto pf = F::plan_dft(5, n_arr, in, mid, FFTW_FORWARD, FFTW_ESTIMATE);
    auto pb = F::plan_dft(5, n_arr, mid, out, FFTW_BACKWARD, FFTW_ESTIMATE);

    if (!pf || !pb)
    {
        if (pf) F::destroy_plan(pf);
        if (pb) F::destroy_plan(pb);
        F::free_fn(in); F::free_fn(mid); F::free_fn(out);
        GTEST_SKIP() << "5D plan creation not supported";
    }

    F::execute(pf);
    F::execute(pb);

    compare_complex_scaled(in, out, total, total, dft_tolerance<F>(total));

    F::destroy_plan(pf);
    F::destroy_plan(pb);
    F::free_fn(in); F::free_fn(mid); F::free_fn(out);
}

/* ===================================================================== */
/*  3) Rank-5 R2C roundtrip                                              */
/*     Tests the deepest R2C dimension-reversal + padding code path.     */
/* ===================================================================== */
TYPED_TEST(FftwWrapperDangerTest, PTEST_R2C_RANK5_ROUNDTRIP)
{
    using F = FftwTypes<TypeParam>;
    using real_t    = typename F::real_t;
    using complex_t = typename F::complex_t;

    const int D0 = 2, D1 = 2, D2 = 3, D3 = 2, D4 = 4;
    const int Nc4 = D4 / 2 + 1;
    int n_arr[] = {D0, D1, D2, D3, D4};
    const int total_real = D0 * D1 * D2 * D3 * D4;
    const int total_cplx = D0 * D1 * D2 * D3 * Nc4;

    real_t    *in  = F::alloc_real(total_real);
    complex_t *mid = F::alloc_complex(total_cplx);
    real_t    *out = F::alloc_real(total_real);
    ASSERT_NE(in, nullptr);
    ASSERT_NE(mid, nullptr);
    ASSERT_NE(out, nullptr);

    this->fill_ramp_real(in, total_real);

    auto pr = F::plan_dft_r2c(5, n_arr, in, mid, FFTW_ESTIMATE);
    if (!pr)
    {
        F::free_fn(in); F::free_fn(mid); F::free_fn(out);
        GTEST_SKIP() << "5D R2C plan creation not supported";
    }

    /* Re-seed the input before executing: the FFTW contract lets the planner
     * modify the input buffer (only FFTW_ESTIMATE is guaranteed not to), so
     * restore the known ramp so the roundtrip check below is well-defined. */
    this->fill_ramp_real(in, total_real);
    F::execute(pr);

    auto pc = F::plan_dft_c2r(5, n_arr, mid, out, FFTW_ESTIMATE);
    if (!pc)
    {
        F::destroy_plan(pr);
        F::free_fn(in); F::free_fn(mid); F::free_fn(out);
        GTEST_SKIP() << "5D C2R plan creation not supported";
    }
    F::execute(pc);

    compare_real_scaled(in, out, total_real, total_real,
                        dft_tolerance<F>(total_real));

    F::destroy_plan(pr);
    F::destroy_plan(pc);
    F::free_fn(in); F::free_fn(mid); F::free_fn(out);
}

/* ===================================================================== */
/*  4) Guru with negative output stride (reversed output).               */
/*     Native FFTW allows negative strides for reversed traversal, but    */
/*     AOCL-FFTZ's problem-descriptor validation requires strides > 0     */
/*     (see api/aoclfftz_api.c: "out_stride must be greater than zero").  */
/*     This is a documented AOCL-FFTZ limitation, so the wrapper must     */
/*     surface it gracefully by returning a NULL plan rather than         */
/*     crashing or producing garbage output.                             */
/* ===================================================================== */
TYPED_TEST(FftwWrapperDangerTest,
           KNOWN_DIVERGENCE_GURU_NEGATIVE_OUTPUT_STRIDE)
{
    using F = FftwTypes<TypeParam>;
    using complex_t = typename F::complex_t;
    using iodim_t   = typename F::iodim_t;

    const int N = 8;

    complex_t *in  = F::alloc_complex(N);
    complex_t *out = F::alloc_complex(N);
    ASSERT_NE(in, nullptr);
    ASSERT_NE(out, nullptr);

    this->init_complex_real_iota(in, N);

    iodim_t dims = {N, 1, -1};

    auto p = F::plan_guru_dft(
        1, &dims, 0, nullptr,
        in, &out[N - 1], FFTW_FORWARD, FFTW_ESTIMATE);

    /* Expected: AOCL-FFTZ rejects the negative stride during setup and the
     * wrapper returns NULL. Guard the execute/destroy path so the test stays
     * leak-free if a future FFTZ version starts supporting reversed strides. */
    EXPECT_EQ(p, nullptr)
        << "Negative output strides are a known AOCL-FFTZ limitation; "
        << "the wrapper is expected to return a NULL plan.";

    if (p)
    {
        F::execute(p);
        F::destroy_plan(p);
    }

    F::free_fn(in);
    F::free_fn(out);
}

/* ===================================================================== */
/*  5) Plan creation after fftw_cleanup(). FFTW docs state that          */
/*     cleanup invalidates existing plans but does NOT prevent new        */
/*     plan creation. Verify wrapper matches this behavior.              */
/* ===================================================================== */
TYPED_TEST(FftwWrapperDangerTest,
           PTEST_PLAN_AFTER_CLEANUP)
{
    using F = FftwTypes<TypeParam>;
    using complex_t = typename F::complex_t;

    const int N = 16;

    complex_t *in  = F::alloc_complex(N);
    complex_t *out = F::alloc_complex(N);
    ASSERT_NE(in, nullptr);
    ASSERT_NE(out, nullptr);

    auto p1 = F::plan_dft_1d(N, in, out, FFTW_FORWARD, FFTW_ESTIMATE);
    ASSERT_NE(p1, nullptr);
    F::destroy_plan(p1);

    F::cleanup();

    this->init_complex_real_iota(in, N);

    auto p2 = F::plan_dft_1d(N, in, out, FFTW_FORWARD, FFTW_ESTIMATE);
    EXPECT_NE(p2, nullptr)
        << "Plan creation failed after fftw_cleanup() -- "
        << "FFTW spec says new plans should still work";
    if (p2)
    {
        F::execute(p2);

        complex_t *ref = F::alloc_complex(N);
        ASSERT_NE(ref, nullptr);
        dft_reference_1d(in, ref, N, FFTW_FORWARD);

        compare_complex_arrays(ref, out, N, dft_tolerance<F>(N));
        F::free_fn(ref);
        F::destroy_plan(p2);
    }

    F::free_fn(in); F::free_fn(out);
}

/* ===================================================================== */
/*  6) plan_dft with sign=0 (neither FFTW_FORWARD nor FFTW_BACKWARD).   */
/*     In FFTW, this is undefined. The wrapper maps non-FORWARD to       */
/*     BACKWARD. Verify the wrapper doesn't crash.                       */
/* ===================================================================== */
TYPED_TEST(FftwWrapperDangerTest,
           PTEST_INVALID_SIGN_ZERO)
{
    using F = FftwTypes<TypeParam>;
    using complex_t = typename F::complex_t;

    const int N = 8;

    complex_t *in  = F::alloc_complex(N);
    complex_t *out = F::alloc_complex(N);
    ASSERT_NE(in, nullptr);
    ASSERT_NE(out, nullptr);

    this->init_complex_real_iota(in, N);

    auto p = F::plan_dft_1d(N, in, out, 0, FFTW_ESTIMATE);
    EXPECT_NE(p, nullptr)
        << "Plan with sign=0 returned NULL";

    if (p)
    {
        complex_t *bwd_out = F::alloc_complex(N);
        ASSERT_NE(bwd_out, nullptr);

        auto pb = F::plan_dft_1d(N, in, bwd_out,
                                 FFTW_BACKWARD, FFTW_ESTIMATE);
        ASSERT_NE(pb, nullptr);

        this->init_complex_real_iota(in, N);
        F::execute(p);

        this->init_complex_real_iota(in, N);
        F::execute(pb);

        compare_complex_arrays(bwd_out, out, N, dft_tolerance<F>(N));

        F::destroy_plan(pb);
        F::free_fn(bwd_out);
        F::destroy_plan(p);
    }

    F::free_fn(in); F::free_fn(out);
}

/* ===================================================================== */
/*  7) Parseval's theorem for batched 2D R2C.                            */
/*     sum(|x[n]|^2) = (1/N) * sum(|X[k]|^2)                           */
/*     for each batch independently.                                     */
/* ===================================================================== */
TYPED_TEST(FftwWrapperDangerTest,
           PTEST_PARSEVAL_BATCHED_2D_R2C)
{
    using F = FftwTypes<TypeParam>;
    using real_t    = typename F::real_t;
    using complex_t = typename F::complex_t;

    const int N0 = 4, N1 = 6;
    const int Nc1 = N1 / 2 + 1;
    const int total_r = N0 * N1;
    const int total_c = N0 * Nc1;
    const int K = 3;
    int n_arr[] = {N0, N1};

    real_t    *in  = F::alloc_real(total_r * K);
    complex_t *out = F::alloc_complex(total_c * K);
    ASSERT_NE(in, nullptr);
    ASSERT_NE(out, nullptr);

    /* K batches of (b+1)*sin(2*pi*i/total_r) laid out contiguously. */
    this->init_real_wave_strided(in, total_r, 1, total_r, K, WaveKind::Sin);

    auto p = F::plan_many_dft_r2c(
        2, n_arr, K,
        in, nullptr, 1, total_r,
        out, nullptr, 1, total_c,
        FFTW_ESTIMATE);
    ASSERT_NE(p, nullptr);

    /* Re-seed with the identical batched sine before executing (only
     * FFTW_ESTIMATE guarantees the planner left the input untouched). */
    this->init_real_wave_strided(in, total_r, 1, total_r, K, WaveKind::Sin);
    F::execute(p);

    for (int b = 0; b < K; ++b)
    {
        /* Time-domain energy of this batch straight from the (preserved,
         * out-of-place) input buffer. */
        double time_energy = energy_real(&in[b * total_r], total_r);

        double freq_energy = 0.0;
        for (int i0 = 0; i0 < N0; ++i0)
        {
            for (int k1 = 0; k1 < Nc1; ++k1)
            {
                int idx = b * total_c + i0 * Nc1 + k1;
                double re = out[idx][0];
                double im = out[idx][1];
                double mag2 = re * re + im * im;

                if (k1 > 0 && k1 < Nc1 - 1)
                {
                    freq_energy += 2.0 * mag2;
                }
                else if (k1 == 0 || (N1 % 2 == 0 && k1 == Nc1 - 1))
                {
                    freq_energy += mag2;
                }
                else
                {
                    freq_energy += 2.0 * mag2;
                }
            }
        }
        freq_energy /= static_cast<double>(total_r);

        EXPECT_NEAR(time_energy, freq_energy,
                    F::tolerance * total_r * total_r)
            << "Parseval violation for batch " << b;
    }

    F::destroy_plan(p);
    F::free_fn(in); F::free_fn(out);
}

/* ===================================================================== */
/*  8) execute_dft with new-array on a plan_many_dft plan.               */
/*     Tests that execute_dft correctly redirects to new buffers for      */
/*     a batched plan.                                                   */
/* ===================================================================== */
TYPED_TEST(FftwWrapperDangerTest,
           PTEST_EXECUTE_DFT_NEW_ARRAY_ON_MANY_PLAN)
{
    using F = FftwTypes<TypeParam>;
    using complex_t = typename F::complex_t;

    const int N = 8;
    const int K = 3;
    int n_arr[] = {N};

    complex_t *in1  = F::alloc_complex(N * K);
    complex_t *out1 = F::alloc_complex(N * K);
    complex_t *in2  = F::alloc_complex(N * K);
    complex_t *out2 = F::alloc_complex(N * K);
    ASSERT_NE(in1, nullptr);
    ASSERT_NE(out1, nullptr);
    ASSERT_NE(in2, nullptr);
    ASSERT_NE(out2, nullptr);

    auto fill = [&](complex_t *buf)
    {
        this->init_complex_linear(buf, N * K, 0.5, 0.3);
    };

    fill(in1);
    auto p = F::plan_many_dft(
        1, n_arr, K,
        in1, nullptr, 1, N,
        out1, nullptr, 1, N,
        FFTW_FORWARD, FFTW_ESTIMATE);
    ASSERT_NE(p, nullptr);

    fill(in1);
    F::execute(p);

    fill(in2);
    F::execute_dft(p, in2, out2);

    compare_complex_arrays(out1, out2, N * K, dft_tolerance<F>(N));

    F::destroy_plan(p);
    F::free_fn(in1);  F::free_fn(out1);
    F::free_fn(in2);  F::free_fn(out2);
}

/* ===================================================================== */
/*  9) Rank-4 R2C in-place roundtrip.                                    */
/*     Tests in-place padding logic for 4D R2C.                          */
/* ===================================================================== */
TYPED_TEST(FftwWrapperDangerTest,
           PTEST_R2C_RANK4_INPLACE_ROUNDTRIP)
{
    using F = FftwTypes<TypeParam>;
    using real_t    = typename F::real_t;
    using complex_t = typename F::complex_t;

    const int D0 = 2, D1 = 3, D2 = 2, D3 = 4;
    const int Nc3 = D3 / 2 + 1;
    const int padded_D3 = 2 * Nc3;
    int n_arr[] = {D0, D1, D2, D3};
    const int total_real = D0 * D1 * D2 * D3;
    const int total_padded = D0 * D1 * D2 * padded_D3;

    real_t *buf = F::alloc_real(total_padded);
    ASSERT_NE(buf, nullptr);

    /* Contiguous ramp orig[i]=i+1, scattered into the padded in-place buffer
     * whose last dimension is widened from D3 to padded_D3. */
    std::vector<real_t> orig(total_real);
    this->fill_affine_real(orig.data(), total_real, static_cast<real_t>(1),
                           static_cast<real_t>(1));
    this->scatter_padded_real(buf, orig.data(),
                              static_cast<long>(D0) * D1 * D2, D3, padded_D3);

    auto pr = F::plan_dft_r2c(
        4, n_arr, buf, (complex_t *)buf, FFTW_ESTIMATE);
    if (!pr)
    {
        F::free_fn(buf);
        GTEST_SKIP() << "4D in-place R2C plan not supported";
    }
    F::execute(pr);
    F::destroy_plan(pr);

    auto pc = F::plan_dft_c2r(
        4, n_arr, (complex_t *)buf, buf, FFTW_ESTIMATE);
    if (!pc)
    {
        F::free_fn(buf);
        GTEST_SKIP() << "4D in-place C2R plan not supported";
    }
    F::execute(pc);
    F::destroy_plan(pc);

    /* Gather the non-padded reals back and check the roundtrip scaled by
     * total_real (the unnormalized forward*inverse gain). */
    std::vector<real_t> result(total_real);
    this->gather_padded_real(result.data(), buf,
                             static_cast<long>(D0) * D1 * D2, D3, padded_D3);
    compare_real_scaled(orig.data(), result.data(), total_real,
                        static_cast<double>(total_real),
                        dft_tolerance<F>(total_real));

    F::free_fn(buf);
}

/* ===================================================================== */
/*  10) plan_many_dft_r2c with istride=2 (interleaved real signals).     */
/*      Each real signal is interleaved with another in memory.           */
/* ===================================================================== */
TYPED_TEST(FftwWrapperDangerTest,
           PTEST_INTERLEAVED_R2C_ISTRIDE2)
{
    using F = FftwTypes<TypeParam>;
    using real_t    = typename F::real_t;
    using complex_t = typename F::complex_t;

    const int N = 8;
    const int Nc = N / 2 + 1;
    const int K = 2;
    int n_arr[] = {N};

    real_t    *in  = F::alloc_real(N * K);
    complex_t *out = F::alloc_complex(Nc * K);
    ASSERT_NE(in, nullptr);
    ASSERT_NE(out, nullptr);

    /* Two interleaved real signals: batch b sits at stride K, amplitude (b+1),
     * shape cos(2*pi*i/N). */
    this->init_real_wave_strided(in, N, K, 1, K, WaveKind::Cos);

    auto p = F::plan_many_dft_r2c(
        1, n_arr, K,
        in,  nullptr, K, 1,
        out, nullptr, K, 1,
        FFTW_ESTIMATE);
    ASSERT_NE(p, nullptr)
        << "Interleaved R2C plan failed";
    F::execute(p);

    real_t    *in_one  = F::alloc_real(N);
    complex_t *out_one = F::alloc_complex(Nc);
    ASSERT_NE(in_one, nullptr);
    ASSERT_NE(out_one, nullptr);

    for (int b = 0; b < K; ++b)
    {
        this->init_real_wave(in_one, N, static_cast<real_t>(b + 1),
                             WaveKind::Cos);

        auto ps = F::plan_dft_r2c_1d(N, in_one, out_one, FFTW_ESTIMATE);
        ASSERT_NE(ps, nullptr);
        F::execute(ps);

        for (int k = 0; k < Nc; ++k)
        {
            EXPECT_NEAR(out[k * K + b][0], out_one[k][0],
                        dft_tolerance<F>(N))
                << "batch=" << b << " k=" << k << " (real)";
            EXPECT_NEAR(out[k * K + b][1], out_one[k][1],
                        dft_tolerance<F>(N))
                << "batch=" << b << " k=" << k << " (imag)";
        }
        F::destroy_plan(ps);
    }

    F::destroy_plan(p);
    F::free_fn(in); F::free_fn(out);
    F::free_fn(in_one); F::free_fn(out_one);
}

/* ===================================================================== */
/*  11) Shift-theorem: time-domain circular shift corresponds to         */
/*     frequency-domain phase rotation. Verify for 1D C2C.               */
/*     X[k] of shifted input = exp(-j*2*pi*k*shift/N) * X[k] of orig.   */
/* ===================================================================== */
TYPED_TEST(FftwWrapperDangerTest,
           PTEST_SHIFT_THEOREM_1D_C2C)
{
    using F = FftwTypes<TypeParam>;
    using complex_t = typename F::complex_t;

    const int N = 16;
    const int shift = 3;

    complex_t *in_orig    = F::alloc_complex(N);
    complex_t *in_shifted = F::alloc_complex(N);
    complex_t *out_orig   = F::alloc_complex(N);
    complex_t *out_shift  = F::alloc_complex(N);
    ASSERT_NE(in_orig, nullptr);
    ASSERT_NE(in_shifted, nullptr);
    ASSERT_NE(out_orig, nullptr);
    ASSERT_NE(out_shift, nullptr);

    /* Fill in_orig with a two-tone complex signal and in_shifted with its
     * circular shift.  Expressed once as a local lambda and re-run to re-seed
     * before executing (only FFTW_ESTIMATE guarantees the planner left the
     * inputs untouched). */
    auto init_signals = [&]()
    {
        /* Two-tone source: sin at the fundamental, cos at the 2nd harmonic. */
        this->init_complex_quadrature(in_orig, N, 1.0, 2.0);
        this->circular_shift_complex(in_shifted, in_orig, N, shift);
    };
    init_signals();

    auto p1 = F::plan_dft_1d(
        N, in_orig, out_orig, FFTW_FORWARD, FFTW_ESTIMATE);
    auto p2 = F::plan_dft_1d(
        N, in_shifted, out_shift, FFTW_FORWARD, FFTW_ESTIMATE);
    ASSERT_NE(p1, nullptr);
    ASSERT_NE(p2, nullptr);

    init_signals();
    F::execute(p1);
    F::execute(p2);

    expect_shift_theorem(out_orig, out_shift, N, shift, dft_tolerance<F>(N));

    F::destroy_plan(p1);
    F::destroy_plan(p2);
    F::free_fn(in_orig); F::free_fn(in_shifted);
    F::free_fn(out_orig); F::free_fn(out_shift);
}

/* ===================================================================== */
/*  12) plan_many_dft_r2c with 3D, howmany=2.                           */
/*      Tests 3D batched R2C stride calculation in the many path.        */
/* ===================================================================== */
TYPED_TEST(FftwWrapperDangerTest,
           PTEST_MANY_R2C_3D_BATCH2)
{
    using F = FftwTypes<TypeParam>;
    using real_t    = typename F::real_t;
    using complex_t = typename F::complex_t;

    const int N0 = 2, N1 = 3, N2 = 4;
    const int Nc2 = N2 / 2 + 1;
    int n_arr[] = {N0, N1, N2};
    const int total_r = N0 * N1 * N2;
    const int total_c = N0 * N1 * Nc2;
    const int K = 2;

    real_t    *in  = F::alloc_real(total_r * K);
    complex_t *out = F::alloc_complex(total_c * K);
    ASSERT_NE(in, nullptr);
    ASSERT_NE(out, nullptr);

    this->fill_affine_real(in, total_r * K, static_cast<real_t>(1.0),
                           static_cast<real_t>(0.2));

    auto pm = F::plan_many_dft_r2c(
        3, n_arr, K,
        in, nullptr, 1, total_r,
        out, nullptr, 1, total_c,
        FFTW_ESTIMATE);
    ASSERT_NE(pm, nullptr);

    this->fill_affine_real(in, total_r * K, static_cast<real_t>(1.0),
                           static_cast<real_t>(0.2));
    F::execute(pm);

    for (int b = 0; b < K; ++b)
    {
        real_t    *in_one  = F::alloc_real(total_r);
        complex_t *out_one = F::alloc_complex(total_c);
        ASSERT_NE(in_one, nullptr);
        ASSERT_NE(out_one, nullptr);

        /* Batch b is the slice [b*total_r, ..) of the global fill
         * in[i]=1.0+0.2*i, i.e. an affine ramp starting at 1.0+0.2*b*total_r. */
        this->fill_affine_real(in_one, total_r,
                               static_cast<real_t>(1.0 + 0.2 * b * total_r),
                               static_cast<real_t>(0.2));

        auto ps = F::plan_dft_r2c_3d(
            N0, N1, N2, in_one, out_one, FFTW_ESTIMATE);
        ASSERT_NE(ps, nullptr);
        F::execute(ps);

        compare_complex_arrays(out_one, &out[b * total_c], total_c,
                               dft_tolerance<F>(total_r));

        F::destroy_plan(ps);
        F::free_fn(in_one); F::free_fn(out_one);
    }

    F::destroy_plan(pm);
    F::free_fn(in); F::free_fn(out);
}
