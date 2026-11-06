// Copyright Advanced Micro Devices, Inc.
// SPDX-License-Identifier: BSD-3-Clause

/** @file fftw_wrapper_threading_gtest.cpp
 * @brief GTest cases for FFTW wrapper threading APIs.
 */

#include <vector>
#include <cmath>
#include <thread>
#include "fftw_wrapper_test_utils.h"

class FftwWrapperThreadingTest : public FftwWrapperTestBase<double>
{
};

/* ===== init_threads ===== */

TEST_F(FftwWrapperThreadingTest, TEST_INIT_THREADS)
{
    int ret = fftw_init_threads();
    EXPECT_NE(ret, 0);
}

TEST_F(FftwWrapperThreadingTest, TEST_INIT_THREADS_F)
{
    int ret = fftwf_init_threads();
    EXPECT_NE(ret, 0);
}

TEST_F(FftwWrapperThreadingTest, TEST_INIT_THREADS_RESETS_THREAD_COUNT)
{
    fftw_plan_with_nthreads(4);
    EXPECT_EQ(fftw_planner_nthreads(), 4);
    fftw_init_threads();
    EXPECT_EQ(fftw_planner_nthreads(), 1);
}

/* ===== plan_with_nthreads / planner_nthreads ===== */

TEST_F(FftwWrapperThreadingTest, TEST_PLAN_WITH_NTHREADS_1)
{
    fftw_plan_with_nthreads(1);
    EXPECT_EQ(fftw_planner_nthreads(), 1);
}

TEST_F(FftwWrapperThreadingTest, TEST_PLAN_WITH_NTHREADS_4)
{
    fftw_plan_with_nthreads(4);
    EXPECT_EQ(fftw_planner_nthreads(), 4);
    fftw_plan_with_nthreads(1);
}

TEST_F(FftwWrapperThreadingTest, TEST_PLAN_WITH_NTHREADS_8)
{
    fftw_plan_with_nthreads(8);
    EXPECT_EQ(fftw_planner_nthreads(), 8);
    fftw_plan_with_nthreads(1);
}

TEST_F(FftwWrapperThreadingTest, TEST_PLAN_WITH_NTHREADS_F)
{
    fftwf_plan_with_nthreads(4);
    EXPECT_EQ(fftwf_planner_nthreads(), 4);
    fftwf_plan_with_nthreads(1);
}

TEST_F(FftwWrapperThreadingTest, TEST_PLANNER_NTHREADS_RETURNS_CORRECT)
{
    for (int n : {1, 2, 4, 8, 16})
    {
        fftw_plan_with_nthreads(n);
        EXPECT_EQ(fftw_planner_nthreads(), n);
    }
    fftw_plan_with_nthreads(1);
}

/* The double and float APIs share the same global thread_num. */
TEST_F(FftwWrapperThreadingTest, TEST_PLANNER_NTHREADS_SHARED_STATE)
{
    fftw_plan_with_nthreads(4);
    EXPECT_EQ(fftwf_planner_nthreads(), 4)
        << "fftw_ and fftwf_ should share thread_num";
    fftwf_plan_with_nthreads(2);
    EXPECT_EQ(fftw_planner_nthreads(), 2)
        << "fftwf_ change should be visible via fftw_";
    fftw_plan_with_nthreads(1);
}

/* ===== cleanup_threads ===== */

TEST_F(FftwWrapperThreadingTest, TEST_CLEANUP_THREADS)
{
    fftw_plan_with_nthreads(4);
    fftw_cleanup_threads();
    EXPECT_EQ(fftw_planner_nthreads(), 1);
}

TEST_F(FftwWrapperThreadingTest, TEST_CLEANUP_THREADS_F)
{
    fftwf_plan_with_nthreads(4);
    fftwf_cleanup_threads();
    EXPECT_EQ(fftwf_planner_nthreads(), 1);
}

/* ===== set_timelimit (no-op stub) ===== */

TEST_F(FftwWrapperThreadingTest, KNOWN_DIVERGENCE_SET_TIMELIMIT_NOOP)
{
    fftw_set_timelimit(1.0);
    fftw_set_timelimit(-1.0);
    fftw_set_timelimit(0.0);
    fftwf_set_timelimit(1.0);
}

/* ===== threads_set_callback (no-op stub) ===== */

TEST_F(FftwWrapperThreadingTest,
       KNOWN_DIVERGENCE_THREADS_SET_CALLBACK_NOOP)
{
    fftw_threads_set_callback(nullptr, nullptr);
    fftwf_threads_set_callback(nullptr, nullptr);
}

/* ===== Multi-threaded plan + execute, verify correctness ===== */

TEST_F(FftwWrapperThreadingTest, TEST_MULTI_THREAD_PLAN_AND_EXECUTE_1D)
{
    const int N = 64;
    fftw_init_threads();
    fftw_plan_with_nthreads(4);

    fftw_complex *in  = fftw_alloc_complex(N);
    fftw_complex *out = fftw_alloc_complex(N);
    fftw_complex *ref = fftw_alloc_complex(N);

    this->init_complex(in, N);

    fftw_plan p = fftw_plan_dft_1d(N, in, out, FFTW_FORWARD, FFTW_ESTIMATE);
    ASSERT_NE(p, nullptr);
    fftw_execute(p);

    dft_reference_1d(in, ref, N, FFTW_FORWARD);
    compare_complex_arrays(ref, out, N, dft_tolerance<FftwTypes<double>>(N));

    fftw_destroy_plan(p);
    fftw_free(in);
    fftw_free(out);
    fftw_free(ref);
    fftw_cleanup_threads();
}

TEST_F(FftwWrapperThreadingTest, TEST_MULTI_THREAD_PLAN_AND_EXECUTE_2D)
{
    const int n0 = 4, n1 = 8, total = n0 * n1;
    fftw_init_threads();
    fftw_plan_with_nthreads(4);

    fftw_complex *in  = fftw_alloc_complex(total);
    fftw_complex *out = fftw_alloc_complex(total);
    fftw_complex *ref = fftw_alloc_complex(total);
    ASSERT_NE(in, nullptr);
    ASSERT_NE(out, nullptr);
    ASSERT_NE(ref, nullptr);
    this->init_complex_real_iota(in, total);

    fftw_plan p = fftw_plan_dft_2d(n0, n1, in, out, FFTW_FORWARD, FFTW_ESTIMATE);
    ASSERT_NE(p, nullptr);
    fftw_execute(p);

    dft_reference_2d(in, ref, n0, n1, FFTW_FORWARD);
    compare_complex_arrays(
        ref, out, total, dft_tolerance<FftwTypes<double>>(total));

    fftw_destroy_plan(p);
    fftw_free(in);
    fftw_free(out);
    fftw_free(ref);
    fftw_cleanup_threads();
}

/* Verify that different thread counts produce the same result. */
TEST_F(FftwWrapperThreadingTest, TEST_MULTI_THREAD_DIFFERENT_THREAD_COUNTS)
{
    const int N = 64;
    fftw_complex *in  = fftw_alloc_complex(N);
    this->init_complex_ramp(in, N);

    fftw_complex *ref = fftw_alloc_complex(N);
    dft_reference_1d(in, ref, N, FFTW_FORWARD);

    fftw_init_threads();
    for (int nt : {1, 2, 4, 8})
    {
        fftw_plan_with_nthreads(nt);
        fftw_complex *out = fftw_alloc_complex(N);
        fftw_plan p = fftw_plan_dft_1d(N, in, out, FFTW_FORWARD, FFTW_ESTIMATE);
        ASSERT_NE(p, nullptr) << "nthreads = " << nt;
        fftw_execute(p);
        compare_complex_arrays(ref, out, N, dft_tolerance<FftwTypes<double>>(N));
        fftw_destroy_plan(p);
        fftw_free(out);
    }

    fftw_free(in);
    fftw_free(ref);
    fftw_cleanup_threads();
}

/* Genuine multi-threaded correctness test: each worker owns a private plan
 * and private buffers, so there is no shared mutable state. This is the
 * canonical thread-safe FFTW usage pattern and MUST produce correct results.
 *
 * The planner itself is not assumed thread-safe, so all plans are created
 * serially on the main thread; only fftw_execute() runs concurrently. Workers
 * record a numeric max-error (max_error_complex has no GTest assertions and is
 * therefore safe to call off the main thread); the main thread does the
 * assertions after every worker has joined. */
TEST_F(FftwWrapperThreadingTest, TEST_CONCURRENT_INDEPENDENT_PLANS)
{
    const int N = 32;
    const int num_threads = 4;

    fftw_init_threads();
    fftw_plan_with_nthreads(1);

    std::vector<fftw_complex *> ins(num_threads);
    std::vector<fftw_complex *> outs(num_threads);
    std::vector<fftw_complex *> refs(num_threads);
    std::vector<fftw_plan>      plans(num_threads);

    for (int t = 0; t < num_threads; t++)
    {
        ins[t]  = fftw_alloc_complex(N);
        outs[t] = fftw_alloc_complex(N);
        refs[t] = fftw_alloc_complex(N);
        this->init_complex_batch(ins[t], N, t);
        dft_reference_1d(ins[t], refs[t], N, FFTW_FORWARD);
        plans[t] = fftw_plan_dft_1d(N, ins[t], outs[t],
                                    FFTW_FORWARD, FFTW_ESTIMATE);
        ASSERT_NE(plans[t], nullptr) << "plan " << t;
    }

    std::vector<double> errors(num_threads, 0.0);
    std::vector<std::thread> workers;
    workers.reserve(num_threads);
    for (int t = 0; t < num_threads; t++)
    {
        workers.emplace_back([&, t]() {
            fftw_execute(plans[t]);
            errors[t] = max_error_complex(refs[t], outs[t], N);
        });
    }
    for (auto &w : workers)
    {
        w.join();
    }

    for (int t = 0; t < num_threads; t++)
    {
        EXPECT_LT(errors[t], dft_tolerance<FftwTypes<double>>(N))
            << "thread " << t << " produced an incorrect transform";
        fftw_destroy_plan(plans[t]);
        fftw_free(ins[t]);
        fftw_free(outs[t]);
        fftw_free(refs[t]);
    }
    fftw_cleanup_threads();
}

/* Concurrent execution of a SHARED plan from multiple threads.
 *
 * Native FFTW guarantees fftw_execute_dft is safe to call concurrently on a
 * single shared plan as long as each thread uses its own in/out arrays.
 * AOCL-FFTZ's aoclfftz_execute_io() makes the same guarantee: it copies the
 * immutable base context per call and atomically arbitrates shared scratch
 * buffers. A mismatch is therefore a product regression and must fail CI.
 *
 * We use std::thread (not an OpenMP pragma) so the work is genuinely executed
 * concurrently regardless of whether the target was built with OpenMP. */
TEST_F(FftwWrapperThreadingTest, TEST_CONCURRENT_EXECUTE_SAME_PLAN)
{
    const int N = 16;
    fftw_complex *plan_in  = fftw_alloc_complex(N);
    fftw_complex *plan_out = fftw_alloc_complex(N);
    this->init_complex_real_iota(plan_in, N, 0.0);

    fftw_plan p = fftw_plan_dft_1d(N, plan_in, plan_out,
                                   FFTW_FORWARD, FFTW_ESTIMATE);
    ASSERT_NE(p, nullptr);

    const int num_threads = 4;
    std::vector<fftw_complex *> ins(num_threads), outs(num_threads);
    for (int t = 0; t < num_threads; t++)
    {
        ins[t]  = fftw_alloc_complex(N);
        outs[t] = fftw_alloc_complex(N);
        this->init_complex_arithmetic(ins[t], N, static_cast<double>(t));
    }

    /* Each thread calls fftw_execute_dft on the SAME plan with its own
     * buffers -- concurrently, via std::thread. */
    std::vector<std::thread> workers;
    workers.reserve(num_threads);
    for (int t = 0; t < num_threads; t++)
    {
        workers.emplace_back([&, t]() {
            fftw_execute_dft(p, ins[t], outs[t]);
        });
    }
    for (auto &w : workers)
    {
        w.join();
    }

    /* Count mismatches after joining: GTest assertions remain on the main
     * thread while execution itself is genuinely concurrent. */
    int mismatches = 0;
    const double tol = dft_tolerance<FftwTypes<double>>(N);
    for (int t = 0; t < num_threads; t++)
    {
        fftw_complex *ref = fftw_alloc_complex(N);
        dft_reference_1d(ins[t], ref, N, FFTW_FORWARD);
        if (max_error_complex(ref, outs[t], N) > tol)
        {
            ++mismatches;
        }
        fftw_free(ref);
        fftw_free(ins[t]);
        fftw_free(outs[t]);
    }

    fftw_destroy_plan(p);
    fftw_free(plan_in);
    fftw_free(plan_out);

    EXPECT_EQ(mismatches, 0)
        << "Concurrent fftw_execute_dft calls on a shared plan corrupted "
        << mismatches << " of " << num_threads << " independent outputs";
}
