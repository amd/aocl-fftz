// Copyright Advanced Micro Devices, Inc.
// SPDX-License-Identifier: BSD-3-Clause

/**
 * @file concurrent_setup_gtest.cpp
 *
 * @brief Test concurrent calls to the aoclfftz setup API.
 *
 * Many threads, spawned via OpenMP (the same threading runtime the rest of
 * the library uses), rendezvous on an `#pragma omp barrier` and then call
 * aoclfftz_setup_{d,f}() at the same instant. Four scenarios are exercised:
 *
 *   1. Homogeneous complex / double.
 *   2. Mixed real and complex (R2C + C2C) double-precision setups.
 *   3. Mixed float and double C2C setups.
 *   4. Fully mixed: all four of {C2C, R2C} x {f32, f64} interleaved.
 *
 *  @author Jeya R
 */

#include <omp.h>
#include <functional>
#include <vector>

#include "gtest/gtest.h"

extern "C"
{
#include "api/aoclfftz.h"
#include "api/types.h"
}

namespace {

// Per-test concurrency knobs.
constexpr FFTZ_INT32 thread_count = 16;
constexpr FFTZ_INT32 iter_count   = 32;
constexpr FFTZ_INT32 fft_size     = 16;

// Which (precision, fft_type) FFT variant an OpenMP thread should run.
enum class fft_variant
{
    COMPLEX_DOUBLE,
    COMPLEX_FLOAT,
    REAL_DOUBLE,
    REAL_FLOAT,
};

// Binds T (float / double) to the matching aoclfftz problem-descriptor
template <typename T>
struct setup_api;

template <>
struct setup_api<double>
{
    using prob_desc_t = aoclfftz_prob_desc_d;
    static FFTZ_VOID *setup(prob_desc_t *p)
    {
        return aoclfftz_setup_d(p);
    }
};

template <>
struct setup_api<float>
{
    using prob_desc_t = aoclfftz_prob_desc_f;
    static FFTZ_VOID *setup(prob_desc_t *p)
    {
        return aoclfftz_setup_f(p);
    }
};

// Fill `buf` with values from 0 to `n-1`, advancing by `stride` elements per
// write. Used by both inputs:
//   - real input    -> stride = 1 (contiguous reals)
//   - complex input -> stride = 2 (real parts only; imag parts stay zero)
template <typename T>
static FFTZ_VOID fill_data(T *buf, FFTZ_INT32 n, FFTZ_INT32 stride)
{
    for (FFTZ_INT32 i = 0; i < n; ++i)
    {
        buf[i * stride] = static_cast<T>(i);
    }
}

// Populate the descriptor with the fields.
template <typename T>
static FFTZ_VOID populate_problem_descriptor(
    typename setup_api<T>::prob_desc_t   &p,
    T                                    *in_ptr,
    T                                    *out_ptr,
    FFTZ_INT32                                fft_type,
    aoclfftz_dim_t                       &dim,
    aoclfftz_dim_t                       &vec)
{
    dim.n          = fft_size;
    dim.in_stride  = 1;
    dim.out_stride = 1;

    vec.n          = 1;
    vec.in_stride  = fft_size;
    vec.out_stride = fft_size;

    p          = {};
    p.in       = in_ptr;
    p.out      = out_ptr;
    p.vec_rank = 1;
    p.dim_rank = 1;
    p.dims     = &dim;
    p.vecs     = &vec;

    p.flags.fft_type            = fft_type;
    p.flags.fft_direction       = 0; // forward
    p.flags.storage_order       = 0;
    p.flags.fft_placement       = 1; // out-of-place
    p.flags.transpose_mode      = 0;
    p.flags.bit_reproducibility = 0;

    p.pthr_fft.num_threads        = 1;
    p.pthr_fft.dynamic_load_model = 0;

    p.cntrl_params.opt_level     = -1;
    p.cntrl_params.opt_off       = 1;
    p.cntrl_params.logger_mode   = AOCLFFTZ_LOG_NONE;
    p.cntrl_params.measure_stats = 0;
}

// Run iter_count rendezvous-then-setup/execute/destroy iterations against
// the supplied problem descriptor.
template <typename T>
static FFTZ_VOID run_fft(
    typename setup_api<T>::prob_desc_t &problem,
    FFTZ_INT32                                &had_failure)
{
    for (FFTZ_INT32 it = 0; it < iter_count; ++it)
    {
        #pragma omp barrier

        FFTZ_VOID *handle = setup_api<T>::setup(&problem);
        if (handle == nullptr)
        {
            had_failure = 1;
            continue;
        }
        if (aoclfftz_execute(handle) != AOCLFFTZ_SUCCESS)
        {
            had_failure = 1;
        }
        aoclfftz_destroy(handle);
    }
}

template <typename T>
static FFTZ_VOID run_complex(FFTZ_INT32 &had_failure)
{
    std::vector<T> in (2 * fft_size, T{0});
    std::vector<T> out(2 * fft_size, T{0});
    fill_data(in.data(), fft_size, /*stride=*/2);

    aoclfftz_dim_t                     dim;
    aoclfftz_dim_t                     vec;
    typename setup_api<T>::prob_desc_t problem;
    populate_problem_descriptor<T>(
        problem, in.data(), out.data(), /*fft_type=*/0, dim, vec);

    run_fft<T>(problem, had_failure);
}

template <typename T>
static FFTZ_VOID run_real(FFTZ_INT32 &had_failure)
{
    std::vector<T> in (fft_size,     T{0});
    std::vector<T> out(fft_size + 2, T{0});
    fill_data(in.data(), fft_size, /*stride=*/1);

    aoclfftz_dim_t                     dim;
    aoclfftz_dim_t                     vec;
    typename setup_api<T>::prob_desc_t problem;
    populate_problem_descriptor<T>(
        problem, in.data(), out.data(), /*fft_type=*/1, dim, vec);

    run_fft<T>(problem, had_failure);
}

static FFTZ_VOID dispatch_run(fft_variant variant, FFTZ_INT32 &had_failure)
{
    switch (variant)
    {
        case fft_variant::COMPLEX_DOUBLE:
            run_complex<double>(had_failure); break;
        case fft_variant::COMPLEX_FLOAT:
            run_complex<float> (had_failure); break;
        case fft_variant::REAL_DOUBLE:
            run_real   <double>(had_failure); break;
        case fft_variant::REAL_FLOAT:
            run_real   <float> (had_failure); break;
    }
}

// Run `thread_count` workers in parallel. `variant_for_thread(tid)`
// returns the fft_variant that thread `tid` should execute, letting
// callers interleave whichever variants they want. The test fails if any
// setup or execute call returned an error on any iteration on any thread.
static FFTZ_VOID run_concurrent_setup(
    const char *test_label,
    const std::function<fft_variant(FFTZ_INT32)> &variant_for_thread)
{
    FFTZ_INT32 any_failure = 0;

#pragma omp parallel num_threads(thread_count) reduction(|| : any_failure)
  {
    FFTZ_INT32 tid = omp_get_thread_num();
    fft_variant variant = variant_for_thread(tid);

    FFTZ_INT32 had_failure = 0;
    dispatch_run(variant, had_failure);

    any_failure = any_failure || had_failure;
  }

    EXPECT_EQ(any_failure, 0)
      << "[" << test_label << "] one or more concurrent setup / execute "
      << "calls failed (workload: " << thread_count << " threads x "
      << iter_count << " iterations).";
}

} // namespace

TEST(ConcurrentSetupTest, SimultaneousSetupDoesNotFail)
{
    run_concurrent_setup(
        "C2C double homogeneous",
        [](FFTZ_INT32) -> fft_variant { return fft_variant::COMPLEX_DOUBLE; });
}

// Mix R2C and C2C double-precision setups across threads.
TEST(ConcurrentSetupTest, MixedRealComplexSetupDoesNotFail)
{
    run_concurrent_setup(
        "R2C+C2C double interleaved",
        [](FFTZ_INT32 i) -> fft_variant {
            return (i % 2 == 0) ? fft_variant::COMPLEX_DOUBLE
                                : fft_variant::REAL_DOUBLE;
        });
}

// Mix float and double C2C setups across threads.
TEST(ConcurrentSetupTest, MixedFloatDoubleSetupDoesNotFail)
{
    run_concurrent_setup(
        "C2C f32+f64 interleaved",
        [](FFTZ_INT32 i) -> fft_variant {
            return (i % 2 == 0) ? fft_variant::COMPLEX_DOUBLE
                                : fft_variant::COMPLEX_FLOAT;
        });
}

// Fully mixed: all four of {C2C, R2C} x {f32, f64} interleaved across
// threads via i % 4.
TEST(ConcurrentSetupTest, MixedAllFourSetupDoesNotFail)
{
    run_concurrent_setup(
        "C2C+R2C x f32+f64 interleaved",
        [](FFTZ_INT32 i) -> fft_variant {
            switch (i % 4)
            {
                case 0:  return fft_variant::COMPLEX_DOUBLE;
                case 1:  return fft_variant::COMPLEX_FLOAT;
                case 2:  return fft_variant::REAL_DOUBLE;
                default: return fft_variant::REAL_FLOAT;
            }
        });
}
