// Copyright Advanced Micro Devices, Inc.
// SPDX-License-Identifier: BSD-3-Clause

/**
 * @file fftw_wrapper_test_utils.h
 * @brief Test utilities for FFTW wrapper: FftwTypes, FftwWrapperTestBase,
 *        reference DFT implementations, and array comparison helpers.
 */

#ifndef FFTW_WRAPPER_TEST_UTILS_H
#define FFTW_WRAPPER_TEST_UTILS_H

#include <cmath>
#include <cstring>
#include <cstdio>
#include <vector>
#include <algorithm>
#include <numeric>
#include <type_traits>
#include "gtest/gtest.h"

extern "C"
{
#include "api/fftw_wrapper.h"
}

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

static const std::vector<int> COMMON_SIZES = {
    1, 2, 4, 7, 8, 16, 32, 64, 128, 256, 1024, 4096
};

/* --------------------------------------------------------------------------
 * FftwTypes<T> -- precision trait: maps a real scalar type to the matching
 * FFTW wrapper functions and types (double -> fftw_*, float -> fftwf_*).
 * -------------------------------------------------------------------------- */
template<typename T> struct FftwTypes;

/* ----- double specialization ----- */
template<>
struct FftwTypes<double>
{
    using real_t    = double;
    using complex_t = fftw_complex;
    using plan_t    = fftw_plan;
    using iodim_t   = fftw_iodim;
    using iodim64_t = fftw_iodim64;

    static constexpr double tolerance = 1e-10;

    /* ---- plan creation ---- */
    static plan_t plan_dft(int rank, const int *n, complex_t *in,
                           complex_t *out, int sign, unsigned flags)
    {
        return fftw_plan_dft(rank, n, in, out, sign, flags);
    }
    static plan_t plan_dft_1d(int n, complex_t *in, complex_t *out,
                              int sign, unsigned flags)
    {
        return fftw_plan_dft_1d(n, in, out, sign, flags);
    }
    static plan_t plan_dft_2d(int n0, int n1, complex_t *in, complex_t *out,
                             int sign, unsigned flags)
    {
        return fftw_plan_dft_2d(n0, n1, in, out, sign, flags);
    }
    static plan_t plan_dft_3d(int n0, int n1, int n2, complex_t *in,
                             complex_t *out, int sign, unsigned flags)
    {
        return fftw_plan_dft_3d(n0, n1, n2, in, out, sign, flags);
    }
    static plan_t plan_many_dft(int rank, const int *n, int howmany,
                               complex_t *in, const int *inembed,
                               int istride, int idist,
                               complex_t *out, const int *onembed,
                               int ostride, int odist,
                               int sign, unsigned flags)
    {
        return fftw_plan_many_dft(rank, n, howmany, in, inembed, istride,
                                 idist, out, onembed, ostride, odist,
                                 sign, flags);
    }
    static plan_t plan_guru_dft(int rank, const iodim_t *dims,
                               int howmany_rank, const iodim_t *howmany_dims,
                               complex_t *in, complex_t *out,
                               int sign, unsigned flags)
    {
        return fftw_plan_guru_dft(rank, dims, howmany_rank, howmany_dims,
                                 in, out, sign, flags);
    }
    static plan_t plan_guru64_dft(int rank, const iodim64_t *dims,
                                 int howmany_rank,
                                 const iodim64_t *howmany_dims,
                                 complex_t *in, complex_t *out,
                                 int sign, unsigned flags)
    {
        return fftw_plan_guru64_dft(rank, dims, howmany_rank, howmany_dims,
                                   in, out, sign, flags);
    }

    /* ---- R2C plan creation ---- */
    static plan_t plan_dft_r2c(int rank, const int *n, real_t *in,
                              complex_t *out, unsigned flags)
    {
        return fftw_plan_dft_r2c(rank, n, in, out, flags);
    }
    static plan_t plan_dft_r2c_1d(int n, real_t *in, complex_t *out,
                                  unsigned flags)
    {
        return fftw_plan_dft_r2c_1d(n, in, out, flags);
    }
    static plan_t plan_dft_r2c_2d(int n0, int n1, real_t *in,
                                 complex_t *out, unsigned flags)
    {
        return fftw_plan_dft_r2c_2d(n0, n1, in, out, flags);
    }
    static plan_t plan_dft_r2c_3d(int n0, int n1, int n2, real_t *in,
                                 complex_t *out, unsigned flags)
    {
        return fftw_plan_dft_r2c_3d(n0, n1, n2, in, out, flags);
    }
    static plan_t plan_many_dft_r2c(int rank, const int *n, int howmany,
                                    real_t *in, const int *inembed,
                                    int istride, int idist,
                                    complex_t *out, const int *onembed,
                                    int ostride, int odist, unsigned flags)
    {
        return fftw_plan_many_dft_r2c(rank, n, howmany, in, inembed, istride,
                                     idist, out, onembed, ostride, odist,
                                     flags);
    }
    static plan_t plan_guru_dft_r2c(int rank, const iodim_t *dims,
                                   int howmany_rank,
                                   const iodim_t *howmany_dims,
                                   real_t *in, complex_t *out,
                                   unsigned flags)
    {
        return fftw_plan_guru_dft_r2c(rank, dims, howmany_rank, howmany_dims,
                                     in, out, flags);
    }
    static plan_t plan_guru64_dft_r2c(int rank, const iodim64_t *dims,
                                     int howmany_rank,
                                     const iodim64_t *howmany_dims,
                                     real_t *in, complex_t *out,
                                     unsigned flags)
    {
        return fftw_plan_guru64_dft_r2c(rank, dims, howmany_rank,
                                        howmany_dims, in, out, flags);
    }

    /* ---- C2R plan creation ---- */
    static plan_t plan_dft_c2r(int rank, const int *n, complex_t *in,
                              real_t *out, unsigned flags)
    {
        return fftw_plan_dft_c2r(rank, n, in, out, flags);
    }
    static plan_t plan_dft_c2r_1d(int n, complex_t *in, real_t *out,
                                 unsigned flags)
    {
        return fftw_plan_dft_c2r_1d(n, in, out, flags);
    }
    static plan_t plan_dft_c2r_2d(int n0, int n1, complex_t *in,
                                 real_t *out, unsigned flags)
    {
        return fftw_plan_dft_c2r_2d(n0, n1, in, out, flags);
    }
    static plan_t plan_dft_c2r_3d(int n0, int n1, int n2, complex_t *in,
                                 real_t *out, unsigned flags)
    {
        return fftw_plan_dft_c2r_3d(n0, n1, n2, in, out, flags);
    }
    static plan_t plan_many_dft_c2r(int rank, const int *n, int howmany,
                                   complex_t *in, const int *inembed,
                                   int istride, int idist,
                                   real_t *out, const int *onembed,
                                   int ostride, int odist, unsigned flags)
    {
        return fftw_plan_many_dft_c2r(rank, n, howmany, in, inembed, istride,
                                     idist, out, onembed, ostride, odist,
                                     flags);
    }
    static plan_t plan_guru_dft_c2r(int rank, const iodim_t *dims,
                                   int howmany_rank,
                                   const iodim_t *howmany_dims,
                                   complex_t *in, real_t *out,
                                   unsigned flags)
    {
        return fftw_plan_guru_dft_c2r(rank, dims, howmany_rank, howmany_dims,
                                     in, out, flags);
    }
    static plan_t plan_guru64_dft_c2r(int rank, const iodim64_t *dims,
                                     int howmany_rank,
                                     const iodim64_t *howmany_dims,
                                     complex_t *in, real_t *out,
                                     unsigned flags)
    {
        return fftw_plan_guru64_dft_c2r(rank, dims, howmany_rank,
                                        howmany_dims, in, out, flags);
    }

    /* ---- execution ---- */
    static void execute(plan_t p)
    {
        fftw_execute(p);
    }
    static void execute_dft(plan_t p, complex_t *in, complex_t *out)
    {
        fftw_execute_dft(p, in, out);
    }
    static void execute_dft_r2c(plan_t p, real_t *in, complex_t *out)
    {
        fftw_execute_dft_r2c(p, in, out);
    }
    static void execute_dft_c2r(plan_t p, complex_t *in, real_t *out)
    {
        fftw_execute_dft_c2r(p, in, out);
    }

    /* ---- destroy ---- */
    static void destroy_plan(plan_t p)
    {
        fftw_destroy_plan(p);
    }

    /* ---- memory ---- */
    static void *malloc_fn(size_t n)
    {
        return fftw_malloc(n);
    }
    static void free_fn(void *p)
    {
        fftw_free(p);
    }
    static real_t *alloc_real(size_t n)
    {
        return fftw_alloc_real(n);
    }
    static complex_t *alloc_complex(size_t n)
    {
        return fftw_alloc_complex(n);
    }
    static int alignment_of(real_t *p)
    {
        return fftw_alignment_of(p);
    }

    /* ---- threading ---- */
    static int init_threads()
    {
        return fftw_init_threads();
    }
    static void plan_with_nthreads(int n)
    {
        fftw_plan_with_nthreads(n);
    }
    static int planner_nthreads()
    {
        return fftw_planner_nthreads();
    }
    static void cleanup_threads()
    {
        fftw_cleanup_threads();
    }
    static void set_timelimit(double t)
    {
        fftw_set_timelimit(t);
    }
    static void threads_set_callback(
        void (*parallel_loop)(void *(*)(char *), char *, size_t,
                              int, void *),
        void *data)
    {
        fftw_threads_set_callback(parallel_loop, data);
    }

    /* ---- wisdom ---- */
    static void forget_wisdom()
    {
        fftw_forget_wisdom();
    }
    static int export_wisdom_to_filename(const char *f)
    {
        return fftw_export_wisdom_to_filename(f);
    }
    static void export_wisdom_to_file(FILE *f)
    {
        fftw_export_wisdom_to_file(f);
    }
    static char *export_wisdom_to_string()
    {
        return fftw_export_wisdom_to_string();
    }
    static int import_wisdom_from_filename(const char *f)
    {
        return fftw_import_wisdom_from_filename(f);
    }
    static int import_wisdom_from_file(FILE *f)
    {
        return fftw_import_wisdom_from_file(f);
    }
    static int import_wisdom_from_string(const char *s)
    {
        return fftw_import_wisdom_from_string(s);
    }
    static int import_system_wisdom()
    {
        return fftw_import_system_wisdom();
    }
    static void make_planner_thread_safe()
    {
        fftw_make_planner_thread_safe();
    }

    /* ---- misc ---- */
    static void cleanup()
    {
        fftw_cleanup();
    }
    static void flops(plan_t p, double *a, double *m, double *f)
    {
        fftw_flops(p, a, m, f);
    }
    static double estimate_cost(plan_t p)
    {
        return fftw_estimate_cost(p);
    }
    static double cost(plan_t p)
    {
        return fftw_cost(p);
    }
    static void print_plan(plan_t p)
    {
        fftw_print_plan(p);
    }
    static void fprint_plan(plan_t p, FILE *f)
    {
        fftw_fprint_plan(p, f);
    }
    static char *sprint_plan(plan_t p)
    {
        return fftw_sprint_plan(p);
    }
    static const char *version()
    {
        return fftw_version;
    }
};

/* ----- float specialization ----- */
template<>
struct FftwTypes<float>
{
    using real_t    = float;
    using complex_t = fftwf_complex;
    using plan_t    = fftwf_plan;
    using iodim_t   = fftwf_iodim;
    using iodim64_t = fftwf_iodim64;

    static constexpr double tolerance = 1e-4;

    /* ---- plan creation ---- */
    static plan_t plan_dft(int rank, const int *n, complex_t *in,
                           complex_t *out, int sign, unsigned flags)
    {
        return fftwf_plan_dft(rank, n, in, out, sign, flags);
    }
    static plan_t plan_dft_1d(int n, complex_t *in, complex_t *out,
                             int sign, unsigned flags)
    {
        return fftwf_plan_dft_1d(n, in, out, sign, flags);
    }
    static plan_t plan_dft_2d(int n0, int n1, complex_t *in, complex_t *out,
                             int sign, unsigned flags)
    {
        return fftwf_plan_dft_2d(n0, n1, in, out, sign, flags);
    }
    static plan_t plan_dft_3d(int n0, int n1, int n2, complex_t *in,
                             complex_t *out, int sign, unsigned flags)
    {
        return fftwf_plan_dft_3d(n0, n1, n2, in, out, sign, flags);
    }
    static plan_t plan_many_dft(int rank, const int *n, int howmany,
                               complex_t *in, const int *inembed,
                               int istride, int idist,
                               complex_t *out, const int *onembed,
                               int ostride, int odist,
                               int sign, unsigned flags)
    {
        return fftwf_plan_many_dft(rank, n, howmany, in, inembed, istride,
                                  idist, out, onembed, ostride, odist,
                                  sign, flags);
    }
    static plan_t plan_guru_dft(int rank, const iodim_t *dims,
                               int howmany_rank, const iodim_t *howmany_dims,
                               complex_t *in, complex_t *out,
                               int sign, unsigned flags)
    {
        return fftwf_plan_guru_dft(rank, dims, howmany_rank, howmany_dims,
                                  in, out, sign, flags);
    }
    static plan_t plan_guru64_dft(int rank, const iodim64_t *dims,
                                 int howmany_rank,
                                 const iodim64_t *howmany_dims,
                                 complex_t *in, complex_t *out,
                                 int sign, unsigned flags)
    {
        return fftwf_plan_guru64_dft(rank, dims, howmany_rank, howmany_dims,
                                    in, out, sign, flags);
    }

    static plan_t plan_dft_r2c(int rank, const int *n, real_t *in,
                              complex_t *out, unsigned flags)
    {
        return fftwf_plan_dft_r2c(rank, n, in, out, flags);
    }
    static plan_t plan_dft_r2c_1d(int n, real_t *in, complex_t *out,
                                 unsigned flags)
    {
        return fftwf_plan_dft_r2c_1d(n, in, out, flags);
    }
    static plan_t plan_dft_r2c_2d(int n0, int n1, real_t *in,
                                 complex_t *out, unsigned flags)
    {
        return fftwf_plan_dft_r2c_2d(n0, n1, in, out, flags);
    }
    static plan_t plan_dft_r2c_3d(int n0, int n1, int n2, real_t *in,
                                 complex_t *out, unsigned flags)
    {
        return fftwf_plan_dft_r2c_3d(n0, n1, n2, in, out, flags);
    }
    static plan_t plan_many_dft_r2c(int rank, const int *n, int howmany,
                                   real_t *in, const int *inembed,
                                   int istride, int idist,
                                   complex_t *out, const int *onembed,
                                   int ostride, int odist, unsigned flags)
    {
        return fftwf_plan_many_dft_r2c(rank, n, howmany, in, inembed, istride,
                                      idist, out, onembed, ostride, odist,
                                      flags);
    }
    static plan_t plan_guru_dft_r2c(int rank, const iodim_t *dims,
                                   int howmany_rank,
                                   const iodim_t *howmany_dims,
                                   real_t *in, complex_t *out,
                                   unsigned flags)
    {
        return fftwf_plan_guru_dft_r2c(rank, dims, howmany_rank, howmany_dims,
                                      in, out, flags);
    }
    static plan_t plan_guru64_dft_r2c(int rank, const iodim64_t *dims,
                                     int howmany_rank,
                                     const iodim64_t *howmany_dims,
                                     real_t *in, complex_t *out,
                                     unsigned flags)
    {
        return fftwf_plan_guru64_dft_r2c(rank, dims, howmany_rank,
                                        howmany_dims, in, out, flags);
    }

    static plan_t plan_dft_c2r(int rank, const int *n, complex_t *in,
                              real_t *out, unsigned flags)
    {
        return fftwf_plan_dft_c2r(rank, n, in, out, flags);
    }
    static plan_t plan_dft_c2r_1d(int n, complex_t *in, real_t *out,
                                 unsigned flags)
    {
        return fftwf_plan_dft_c2r_1d(n, in, out, flags);
    }
    static plan_t plan_dft_c2r_2d(int n0, int n1, complex_t *in,
                                 real_t *out, unsigned flags)
    {
        return fftwf_plan_dft_c2r_2d(n0, n1, in, out, flags);
    }
    static plan_t plan_dft_c2r_3d(int n0, int n1, int n2, complex_t *in,
                                 real_t *out, unsigned flags)
    {
        return fftwf_plan_dft_c2r_3d(n0, n1, n2, in, out, flags);
    }
    static plan_t plan_many_dft_c2r(int rank, const int *n, int howmany,
                                   complex_t *in, const int *inembed,
                                   int istride, int idist,
                                   real_t *out, const int *onembed,
                                   int ostride, int odist, unsigned flags)
    {
        return fftwf_plan_many_dft_c2r(rank, n, howmany, in, inembed, istride,
                                      idist, out, onembed, ostride, odist,
                                      flags);
    }
    static plan_t plan_guru_dft_c2r(int rank, const iodim_t *dims,
                                   int howmany_rank,
                                   const iodim_t *howmany_dims,
                                   complex_t *in, real_t *out,
                                   unsigned flags)
    {
        return fftwf_plan_guru_dft_c2r(rank, dims, howmany_rank, howmany_dims,
                                      in, out, flags);
    }
    static plan_t plan_guru64_dft_c2r(int rank, const iodim64_t *dims,
                                     int howmany_rank,
                                     const iodim64_t *howmany_dims,
                                     complex_t *in, real_t *out,
                                     unsigned flags)
    {
        return fftwf_plan_guru64_dft_c2r(rank, dims, howmany_rank,
                                        howmany_dims, in, out, flags);
    }

    /* ---- execution ---- */
    static void execute(plan_t p)
    {
        fftwf_execute(p);
    }
    static void execute_dft(plan_t p, complex_t *in, complex_t *out)
    {
        fftwf_execute_dft(p, in, out);
    }
    static void execute_dft_r2c(plan_t p, real_t *in, complex_t *out)
    {
        fftwf_execute_dft_r2c(p, in, out);
    }
    static void execute_dft_c2r(plan_t p, complex_t *in, real_t *out)
    {
        fftwf_execute_dft_c2r(p, in, out);
    }

    /* ---- destroy ---- */
    static void destroy_plan(plan_t p)
    {
        fftwf_destroy_plan(p);
    }

    /* ---- memory ---- */
    static void *malloc_fn(size_t n)
    {
        return fftwf_malloc(n);
    }
    static void free_fn(void *p)
    {
        fftwf_free(p);
    }
    static real_t *alloc_real(size_t n)
    {
        return fftwf_alloc_real(n);
    }
    static complex_t *alloc_complex(size_t n)
    {
        return fftwf_alloc_complex(n);
    }
    static int alignment_of(real_t *p)
    {
        return fftwf_alignment_of(p);
    }

    /* ---- threading ---- */
    static int init_threads()
    {
        return fftwf_init_threads();
    }
    static void plan_with_nthreads(int n)
    {
        fftwf_plan_with_nthreads(n);
    }
    static int planner_nthreads()
    {
        return fftwf_planner_nthreads();
    }
    static void cleanup_threads()
    {
        fftwf_cleanup_threads();
    }
    static void set_timelimit(double t)
    {
        fftwf_set_timelimit(t);
    }
    static void threads_set_callback(
        void (*parallel_loop)(void *(*)(char *), char *, size_t,
                              int, void *),
        void *data)
    {
        fftwf_threads_set_callback(parallel_loop, data);
    }

    /* ---- wisdom ---- */
    static void forget_wisdom()
    {
        fftwf_forget_wisdom();
    }
    static int export_wisdom_to_filename(const char *f)
    {
        return fftwf_export_wisdom_to_filename(f);
    }
    static void export_wisdom_to_file(FILE *f)
    {
        fftwf_export_wisdom_to_file(f);
    }
    static char *export_wisdom_to_string()
    {
        return fftwf_export_wisdom_to_string();
    }
    static int import_wisdom_from_filename(const char *f)
    {
        return fftwf_import_wisdom_from_filename(f);
    }
    static int import_wisdom_from_file(FILE *f)
    {
        return fftwf_import_wisdom_from_file(f);
    }
    static int import_wisdom_from_string(const char *s)
    {
        return fftwf_import_wisdom_from_string(s);
    }
    static int import_system_wisdom()
    {
        return fftwf_import_system_wisdom();
    }
    static void make_planner_thread_safe()
    {
        fftwf_make_planner_thread_safe();
    }

    /* ---- misc ---- */
    static void cleanup()
    {
        fftwf_cleanup();
    }
    static void flops(plan_t p, double *a, double *m, double *f)
    {
        fftwf_flops(p, a, m, f);
    }
    static double estimate_cost(plan_t p)
    {
        return fftwf_estimate_cost(p);
    }
    static double cost(plan_t p)
    {
        return fftwf_cost(p);
    }
    static void print_plan(plan_t p)
    {
        fftwf_print_plan(p);
    }
    static void fprint_plan(plan_t p, FILE *f)
    {
        fftwf_fprint_plan(p, f);
    }
    static char *sprint_plan(plan_t p)
    {
        return fftwf_sprint_plan(p);
    }
    static const char *version()
    {
        return fftwf_version;
    }
};

using FftwTestTypes = ::testing::Types<double, float>;

/* Selector for the sinusoid initializers: real part / wave shape. Defined at
 * namespace scope (not nested in the fixture) so call sites in the derived
 * typed suites can name it as a plain `WaveKind::Cos` without dependent-base
 * qualification. */
enum class WaveKind { Sin, Cos };

/* Free-function form of the "{start + i, 0}" complex ramp.  Defined at
 * namespace scope (not only as a fixture member) so the non-templated memory
 * smoke test and the mixed-precision test -- which fills a double *and* a float
 * buffer in one body and so cannot use the single-precision fixture member --
 * share the exact same initializer instead of open-coding the loop. */
template<typename complex_t>
inline void fill_complex_real_iota(complex_t *arr, int N, double start = 1.0)
{
    using elem_t = std::remove_reference_t<decltype(arr[0][0])>;
    for (int i = 0; i < N; i++)
    {
        arr[i][0] = static_cast<elem_t>(start + static_cast<double>(i));
        arr[i][1] = static_cast<elem_t>(0);
    }
}

/* --------------------------------------------------------------------------
 * FftwWrapperTestBase<T> -- common fixture used by typed test suites.
 * Provides helpers for allocating, initializing and comparing complex arrays.
 * -------------------------------------------------------------------------- */
template<typename T>
class FftwWrapperTestBase : public ::testing::Test
{
protected:
    using Types     = FftwTypes<T>;
    using real_t    = typename Types::real_t;
    using complex_t = typename Types::complex_t;
    using plan_t    = typename Types::plan_t;

    static complex_t *alloc_complex(size_t n)
    {
        return Types::alloc_complex(n);
    }

    static real_t *alloc_real(size_t n)
    {
        return Types::alloc_real(n);
    }

    static void free_buf(void *p)
    {
        Types::free_fn(p);
    }

    static void init_complex(complex_t *arr, int N)
    {
        for (int i = 0; i < N; i++)
        {
            arr[i][0] = static_cast<real_t>(i + 1.0) / N;
            arr[i][1] = static_cast<real_t>(N - i) / N;
        }
    }

    static void init_real(real_t *arr, int N)
    {
        for (int i = 0; i < N; i++)
        {
            arr[i] = static_cast<real_t>(i + 1.0) / N;
        }
    }

    /* Un-normalised natural ramp: arr[i] = start + i (start defaults to 1, so
     * arr = {1, 2, 3, ...}).  Shared so every plain real input is initialised
     * identically. */
    static void init_real_iota(real_t *arr, long N,
                               real_t start = static_cast<real_t>(1))
    {
        for (long i = 0; i < N; i++)
        {
            arr[i] = start + static_cast<real_t>(i);
        }
    }

    /* Smooth complex "wave" input (distinct real/imag frequencies) shared by
     * the brute-force roundtrip size-sweeps. */
    static void init_complex_wave(complex_t *arr, int N)
    {
        for (int i = 0; i < N; i++)
        {
            arr[i][0] = static_cast<real_t>(std::sin(2.0 * M_PI * i / N));
            arr[i][1] = static_cast<real_t>(std::cos(3.0 * M_PI * i / N));
        }
    }

    /* Batched / strided complex initializer: element i of batch b is written
     * at offset b*dist + i*stride (in complex-element units).  A single call
     * covers contiguous batched (stride=1, dist=N), interleaved
     * (stride=batch, dist=1) and single-batch (batch=1) layouts, so the test
     * data is laid out with exactly the strides handed to the planner. */
    static void init_complex_strided(complex_t *arr, int N, int stride,
                                     int dist, int batch)
    {
        for (int b = 0; b < batch; b++)
        {
            for (int i = 0; i < N; i++)
            {
                long idx = static_cast<long>(b) * dist +
                           static_cast<long>(i) * stride;
                arr[idx][0] = static_cast<real_t>((b + 1) * (i + 1));
                arr[idx][1] = static_cast<real_t>((b + 1) * (N - i));
            }
        }
    }

    /* Initialize a standalone scratch buffer as logical batch `batch_index` of
     * init_complex_strided(). */
    static void init_complex_batch(complex_t *arr, int N, int batch_index)
    {
        for (int i = 0; i < N; i++)
        {
            arr[i][0] = static_cast<real_t>((batch_index + 1) * (i + 1));
            arr[i][1] = static_cast<real_t>((batch_index + 1) * (N - i));
        }
    }

    /* Simple arithmetic complex ramp used by plan-reuse and batched-execute
     * checks: arr[i] = {seed + i + 1, seed - i}. */
    static void init_complex_arithmetic(complex_t *arr, int N,
                                        real_t seed = static_cast<real_t>(0))
    {
        for (int i = 0; i < N; i++)
        {
            arr[i][0] = static_cast<real_t>(seed + i + 1);
            arr[i][1] = static_cast<real_t>(seed - i);
        }
    }

    /* Batched/strided form of init_complex_arithmetic; batch b uses seed=b. */
    static void init_complex_arithmetic_strided(complex_t *arr, int N,
                                                int stride, int dist,
                                                int batch)
    {
        for (int b = 0; b < batch; b++)
        {
            for (int i = 0; i < N; i++)
            {
                long idx = static_cast<long>(b) * dist +
                           static_cast<long>(i) * stride;
                arr[idx][0] = static_cast<real_t>(b + i + 1);
                arr[idx][1] = static_cast<real_t>(b - i);
            }
        }
    }

    /* Real-valued complex ramp: arr[i] = {start + i, 0}.  Thin wrapper over the
     * namespace-scope fill_complex_real_iota so the loop lives in one place. */
    static void init_complex_real_iota(complex_t *arr, int N,
                                       real_t start = static_cast<real_t>(1))
    {
        fill_complex_real_iota(arr, N, static_cast<double>(start));
    }

    /* Batched/strided real-valued complex ramp; batch b starts at b + 1. */
    static void init_complex_real_iota_strided(complex_t *arr, int N,
                                               int stride, int dist,
                                               int batch)
    {
        for (int b = 0; b < batch; b++)
        {
            for (int i = 0; i < N; i++)
            {
                long idx = static_cast<long>(b) * dist +
                           static_cast<long>(i) * stride;
                arr[idx][0] = static_cast<real_t>(b + i + 1);
                arr[idx][1] = static_cast<real_t>(0);
            }
        }
    }

    /* Batched / strided real initializer (see init_complex_strided). */
    static void init_real_strided(real_t *arr, int N, int stride,
                                  int dist, int batch)
    {
        for (int b = 0; b < batch; b++)
        {
            for (int i = 0; i < N; i++)
            {
                long idx = static_cast<long>(b) * dist +
                           static_cast<long>(i) * stride;
                arr[idx] = static_cast<real_t>((b + 1) * (i + 1));
            }
        }
    }

    /* Initialize a standalone scratch buffer as logical batch `batch_index` of
     * init_real_strided(). */
    static void init_real_batch_iota(real_t *arr, int N, int batch_index)
    {
        for (int i = 0; i < N; i++)
        {
            arr[i] = static_cast<real_t>((batch_index + 1) * (i + 1));
        }
    }

    /* Affine "ramp" fill for a flat real buffer: arr[i] = i % mod + start. */
    static void fill_ramp_real(real_t *arr, long total, int mod = 7,
                               real_t start = static_cast<real_t>(0.5))
    {
        for (long i = 0; i < total; i++)
        {
            arr[i] = static_cast<real_t>(i % mod) + start;
        }
    }

    /* Flat affine real fill: arr[i] = start + step*i. */
    static void fill_affine_real(real_t *arr, long total, real_t start,
                                 real_t step)
    {
        for (long i = 0; i < total; i++)
        {
            arr[i] = start + step * static_cast<real_t>(i);
        }
    }

    /* Single sinusoid line: arr[i*stride] = amp * trig(2*pi*i/N), where trig
     * is sin (default) or cos.  Shared so the batched R2C/C2C sinusoid inputs
     * are written in exactly one place instead of an inline trig loop. */
    static void init_real_wave(real_t *arr, int N,
                               real_t amp = static_cast<real_t>(1),
                               WaveKind kind = WaveKind::Sin, int stride = 1)
    {
        for (int i = 0; i < N; i++)
        {
            double ph = 2.0 * M_PI * i / N;
            double w = (kind == WaveKind::Cos)
                           ? std::cos(ph) : std::sin(ph);
            arr[static_cast<long>(i) * stride] =
                static_cast<real_t>(static_cast<double>(amp) * w);
        }
    }

    /* Batched/strided form of init_real_wave: element i of batch b is written
     * at b*dist + i*stride with amplitude (b+1) -- the shared
     * "(b+1)*trig(2*pi*i/N)" batched-sinusoid pattern. */
    static void init_real_wave_strided(real_t *arr, int N, int stride, int dist,
                                       int batch, WaveKind kind = WaveKind::Sin)
    {
        for (int b = 0; b < batch; b++)
        {
            init_real_wave(&arr[static_cast<long>(b) * dist], N,
                           static_cast<real_t>(b + 1), kind, stride);
        }
    }

    /* Complex ramp {re = i+1, im = 0.5*i} shared by the dimension-order and
     * thread-count consistency checks. */
    static void init_complex_ramp(complex_t *arr, int N)
    {
        for (int i = 0; i < N; i++)
        {
            arr[i][0] = static_cast<real_t>(i + 1);
            arr[i][1] = static_cast<real_t>(i) * static_cast<real_t>(0.5);
        }
    }

    /* Fill every element with the same complex constant {re, im}.  Shared by
     * the flat-input and dummy-plan-buffer initializers. */
    static void fill_complex_constant(complex_t *arr, int N, real_t re,
                                      real_t im)
    {
        for (int i = 0; i < N; i++)
        {
            arr[i][0] = re;
            arr[i][1] = im;
        }
    }

    /* Descending complex ramp {re = len - i, im = imag_step * i}.  Shared by
     * the new-array execute checks that need input distinct from the plan
     * buffer. */
    static void init_complex_desc_ramp(complex_t *arr, int len, real_t imag_step)
    {
        for (int i = 0; i < len; i++)
        {
            arr[i][0] = static_cast<real_t>(len - i);
            arr[i][1] = imag_step * static_cast<real_t>(i);
        }
    }

    /* Complex tone whose imaginary part is a fixed multiple of the real part:
     * arr[i] = { sin(2*pi*freq*i/N), imag_scale * sin(2*pi*freq*i/N) }.
     * Shared by the plan-reuse and multi-thread stress inputs. */
    static void init_complex_sine_scaled(complex_t *arr, int N, double freq,
                                         double imag_scale)
    {
        for (int i = 0; i < N; i++)
        {
            double v = std::sin(2.0 * M_PI * freq * i / N);
            arr[i][0] = static_cast<real_t>(v);
            arr[i][1] = static_cast<real_t>(v * imag_scale);
        }
    }

    /* Quadrature complex tone with independent real/imag frequencies:
     * arr[i] = { sin(2*pi*freq_re*i/N), cos(2*pi*freq_im*i/N) }.  Shared by the
     * plan-reuse sweep and the shift-theorem source signal. */
    static void init_complex_quadrature(complex_t *arr, int N, double freq_re,
                                        double freq_im)
    {
        for (int i = 0; i < N; i++)
        {
            arr[i][0] = static_cast<real_t>(
                std::sin(2.0 * M_PI * freq_re * i / N));
            arr[i][1] = static_cast<real_t>(
                std::cos(2.0 * M_PI * freq_im * i / N));
        }
    }

    /* Amplitude-scaled cos/sin complex tone (fundamental frequency 2*pi*i/N):
     * arr[i*stride] = { amp*cos(2*pi*i/N), with_imag ? amp*sin(2*pi*i/N) : 0 }.
     * `stride` supports the interleaved-batch layout.  Shared by the batched
     * interleaved-C2C and large-batch-independence inputs. */
    static void init_complex_cossin(complex_t *arr, int N, real_t amp,
                                    bool with_imag, int stride = 1)
    {
        for (int i = 0; i < N; i++)
        {
            double ph = 2.0 * M_PI * i / N;
            long   idx = static_cast<long>(i) * stride;
            arr[idx][0] = static_cast<real_t>(
                static_cast<double>(amp) * std::cos(ph));
            arr[idx][1] = with_imag
                ? static_cast<real_t>(static_cast<double>(amp) * std::sin(ph))
                : static_cast<real_t>(0);
        }
    }

    /* Complex linear ramp through the origin: arr[i] = {re_step*i, im_step*i}.
     * Shared by the guru / new-array execute checks that just need an
     * index-proportional complex signal (the old inline "{i*a, i*b}" fill
     * lambdas). */
    static void init_complex_linear(complex_t *arr, long N, double re_step,
                                    double im_step)
    {
        for (long i = 0; i < N; i++)
        {
            arr[i][0] = static_cast<real_t>(re_step * static_cast<double>(i));
            arr[i][1] = static_cast<real_t>(im_step * static_cast<double>(i));
        }
    }

    /* Fill every element of a real buffer with the same constant. */
    static void fill_real_constant(real_t *arr, long N, double val)
    {
        for (long i = 0; i < N; i++)
        {
            arr[i] = static_cast<real_t>(val);
        }
    }

    /* Linear combination of two real buffers: z[i] = a*x[i] + b*y[i].  Shared
     * by the input side of the linearity checks. */
    static void combine_linear_real(real_t *z, const real_t *x,
                                    const real_t *y, double a, double b,
                                    long N)
    {
        for (long i = 0; i < N; i++)
        {
            z[i] = static_cast<real_t>(a * static_cast<double>(x[i]) +
                                       b * static_cast<double>(y[i]));
        }
    }

    /* Circular right-shift copy: dst[i] = src[(i - shift) mod N].  Shared by
     * the shift-theorem source setup. */
    static void circular_shift_complex(complex_t *dst, const complex_t *src,
                                       int N, int shift)
    {
        for (int i = 0; i < N; i++)
        {
            int s = ((i - shift) % N + N) % N;
            dst[i][0] = src[s][0];
            dst[i][1] = src[s][1];
        }
    }

    /* In-place R2C padding helpers: copy a contiguous (rows x last) real block
     * to / from a buffer whose last dimension is padded to `padded_last` reals
     * (FFTW's 2*(n/2+1) in-place layout).  `scatter` lays the logical block
     * into the padded buffer; `gather` extracts it back.  Shared by every
     * in-place R2C/C2R padding test so the padded index arithmetic lives in
     * one place. */
    static void scatter_padded_real(real_t *dst, const real_t *src, long rows,
                                    int last, int padded_last)
    {
        for (long r = 0; r < rows; r++)
        {
            for (int i = 0; i < last; i++)
            {
                dst[r * padded_last + i] = src[r * last + i];
            }
        }
    }

    static void gather_padded_real(real_t *dst, const real_t *src, long rows,
                                   int last, int padded_last)
    {
        for (long r = 0; r < rows; r++)
        {
            for (int i = 0; i < last; i++)
            {
                dst[r * last + i] = src[r * padded_last + i];
            }
        }
    }

    static void zero_complex(complex_t *arr, int N)
    {
        std::memset(arr, 0, sizeof(complex_t) * N);
    }

    static void zero_real(real_t *arr, int N)
    {
        std::memset(arr, 0, sizeof(real_t) * N);
    }
};

/* --------------------------------------------------------------------------
 * O(N^2) brute-force 1-D DFT reference.  Suitable for small N (<= 256).
 * Works on interleaved complex data: element k has real at [k][0], imag [k][1].
 * sign: FFTW_FORWARD (-1) or FFTW_BACKWARD (+1).
 * -------------------------------------------------------------------------- */
template<typename complex_t>
inline void dft_reference_1d(const complex_t *in, complex_t *out,
                            int N, int sign)
{
    for (int k = 0; k < N; k++)
    {
        double re = 0.0, im = 0.0;
        for (int j = 0; j < N; j++)
        {
            double angle = sign * 2.0 * M_PI * j * k / N;
            double cval = std::cos(angle);
            double sval = std::sin(angle);
            re += in[j][0] * cval - in[j][1] * sval;
            im += in[j][0] * sval + in[j][1] * cval;
        }
        out[k][0] = re;
        out[k][1] = im;
    }
}

/* R2C reference: real input of length N -> complex output of length N/2+1. */
template<typename Real, typename complex_t>
inline void dft_reference_r2c_1d(const Real *in, complex_t *out, int N)
{
    int Nc = N / 2 + 1;
    for (int k = 0; k < Nc; k++)
    {
        double re = 0.0, im = 0.0;
        for (int j = 0; j < N; j++)
        {
            double angle = -2.0 * M_PI * j * k / N;
            re += in[j] * std::cos(angle);
            im += in[j] * std::sin(angle);
        }
        out[k][0] = re;
        out[k][1] = im;
    }
}

/* --------------------------------------------------------------------------
 * N-dimensional DFT reference (separable form).
 *
 * An N-D DFT is separable: it equals successive 1-D DFTs applied along each
 * axis in turn.  Doing it this way costs O(total * sum(n[d])) instead of the
 * O(total^2) of a naive multi-dimensional sum, which keeps the reference cheap
 * enough to use as an independent oracle for the small shapes exercised by the
 * tests.  All computation is carried out in double regardless of the transform
 * precision so the reference stays "truth" for the float suite as well.
 *
 * Data is contiguous, row-major, interleaved complex (re at [2i], im at
 * [2i+1]); dims are n[0..rank-1] with the last dimension varying fastest.
 * sign is FFTW_FORWARD (-1) or FFTW_BACKWARD (+1).
 * -------------------------------------------------------------------------- */
inline void dft_reference_core_d(std::vector<double> &data, int rank,
                                 const int *n, int sign)
{
    long total = 1;
    for (int d = 0; d < rank; d++)
    {
        total *= n[d];
    }

    for (int d = 0; d < rank; d++)
    {
        const int Nd = n[d];
        long stride = 1;
        for (int e = d + 1; e < rank; e++)
        {
            stride *= n[e];
        }
        const long block = static_cast<long>(Nd) * stride;
        const long outer = total / block;

        std::vector<double> xr(Nd), xi(Nd), yr(Nd), yi(Nd);
        for (long bo = 0; bo < outer; bo++)
        {
            for (long si = 0; si < stride; si++)
            {
                const long base = bo * block + si;
                for (int k = 0; k < Nd; k++)
                {
                    xr[k] = data[2 * (base + static_cast<long>(k) * stride)];
                    xi[k] = data[2 * (base + static_cast<long>(k) * stride) + 1];
                }
                for (int k = 0; k < Nd; k++)
                {
                    double re = 0.0, im = 0.0;
                    for (int j = 0; j < Nd; j++)
                    {
                        double angle = sign * 2.0 * M_PI * j * k / Nd;
                        double c = std::cos(angle);
                        double s = std::sin(angle);
                        re += xr[j] * c - xi[j] * s;
                        im += xr[j] * s + xi[j] * c;
                    }
                    yr[k] = re;
                    yi[k] = im;
                }
                for (int k = 0; k < Nd; k++)
                {
                    data[2 * (base + static_cast<long>(k) * stride)]     = yr[k];
                    data[2 * (base + static_cast<long>(k) * stride) + 1] = yi[k];
                }
            }
        }
    }
}

/* Complex N-D DFT: contiguous row-major complex in -> complex out. */
template<typename complex_t>
inline void dft_reference_nd(const complex_t *in, complex_t *out,
                             int rank, const int *n, int sign)
{
    long total = 1;
    for (int d = 0; d < rank; d++)
    {
        total *= n[d];
    }
    std::vector<double> data(2 * total);
    for (long i = 0; i < total; i++)
    {
        data[2 * i]     = in[i][0];
        data[2 * i + 1] = in[i][1];
    }
    dft_reference_core_d(data, rank, n, sign);
    using out_real_t = std::remove_reference_t<decltype(out[0][0])>;
    for (long i = 0; i < total; i++)
    {
        out[i][0] = static_cast<out_real_t>(data[2 * i]);
        out[i][1] = static_cast<out_real_t>(data[2 * i + 1]);
    }
}

/* 2-D / 3-D convenience wrappers for the complex N-D reference. */
template<typename complex_t>
inline void dft_reference_2d(const complex_t *in, complex_t *out,
                             int n0, int n1, int sign)
{
    const int n[2] = {n0, n1};
    dft_reference_nd(in, out, 2, n, sign);
}

template<typename complex_t>
inline void dft_reference_3d(const complex_t *in, complex_t *out,
                             int n0, int n1, int n2, int sign)
{
    const int n[3] = {n0, n1, n2};
    dft_reference_nd(in, out, 3, n, sign);
}

/* R2C N-D reference: real in (dims n[]) -> complex out with the last
 * dimension reduced to n[rank-1]/2+1 (FFTW's non-redundant half). */
template<typename Real, typename complex_t>
inline void dft_reference_r2c_nd(const Real *in, complex_t *out,
                                 int rank, const int *n)
{
    long total = 1;
    for (int d = 0; d < rank; d++)
    {
        total *= n[d];
    }
    std::vector<double> data(2 * total);
    for (long i = 0; i < total; i++)
    {
        data[2 * i]     = static_cast<double>(in[i]);
        data[2 * i + 1] = 0.0;
    }
    dft_reference_core_d(data, rank, n, FFTW_FORWARD);

    const int last  = n[rank - 1];
    const int lastc = last / 2 + 1;
    const long outer = total / last;
    using out_real_t = std::remove_reference_t<decltype(out[0][0])>;
    for (long o = 0; o < outer; o++)
    {
        for (int k = 0; k < lastc; k++)
        {
            const long fi = o * last + k;
            const long oi = o * lastc + k;
            out[oi][0] = static_cast<out_real_t>(data[2 * fi]);
            out[oi][1] = static_cast<out_real_t>(data[2 * fi + 1]);
        }
    }
}

template<typename Real, typename complex_t>
inline void dft_reference_r2c_2d(const Real *in, complex_t *out, int n0, int n1)
{
    const int n[2] = {n0, n1};
    dft_reference_r2c_nd(in, out, 2, n);
}

template<typename Real, typename complex_t>
inline void dft_reference_r2c_3d(const Real *in, complex_t *out,
                                 int n0, int n1, int n2)
{
    const int n[3] = {n0, n1, n2};
    dft_reference_r2c_nd(in, out, 3, n);
}

/* C2R N-D reference: half-complex in (last dim n[rank-1]/2+1) -> real out
 * (dims n[]).  The stored half is Hermitian-extended to the full spectrum and
 * inverse-transformed; the (vanishing) imaginary part is dropped.  This is the
 * exact inverse of dft_reference_r2c_nd for Hermitian-consistent input and
 * matches FFTW's unnormalized c2r (result scaled by prod(n)). */
template<typename complex_t, typename Real>
inline void dft_reference_c2r_nd(const complex_t *in, Real *out,
                                 int rank, const int *n)
{
    long total = 1;
    for (int d = 0; d < rank; d++)
    {
        total *= n[d];
    }
    const int last  = n[rank - 1];
    const int lastc = last / 2 + 1;

    std::vector<double> data(2 * total);
    std::vector<int> idx(rank), midx(rank);
    for (long lin = 0; lin < total; lin++)
    {
        long rem = lin;
        for (int d = rank - 1; d >= 0; d--)
        {
            idx[d] = rem % n[d];
            rem /= n[d];
        }
        double re, im;
        if (idx[rank - 1] < lastc)
        {
            long half = 0;
            for (int d = 0; d < rank; d++)
            {
                half = half * (d == rank - 1 ? lastc : n[d]) + idx[d];
            }
            re = in[half][0];
            im = in[half][1];
        }
        else
        {
            for (int d = 0; d < rank; d++)
            {
                midx[d] = (n[d] - idx[d]) % n[d];
            }
            long half = 0;
            for (int d = 0; d < rank; d++)
            {
                half = half * (d == rank - 1 ? lastc : n[d]) + midx[d];
            }
            re =  in[half][0];
            im = -in[half][1];
        }
        data[2 * lin]     = re;
        data[2 * lin + 1] = im;
    }

    dft_reference_core_d(data, rank, n, FFTW_BACKWARD);
    for (long i = 0; i < total; i++)
    {
        out[i] = static_cast<Real>(data[2 * i]);
    }
}

template<typename complex_t, typename Real>
inline void dft_reference_c2r_2d(const complex_t *in, Real *out, int n0, int n1)
{
    const int n[2] = {n0, n1};
    dft_reference_c2r_nd(in, out, 2, n);
}

template<typename complex_t, typename Real>
inline void dft_reference_c2r_3d(const complex_t *in, Real *out,
                                 int n0, int n1, int n2)
{
    const int n[3] = {n0, n1, n2};
    dft_reference_c2r_nd(in, out, 3, n);
}

/* Element-wise comparison of two complex arrays. */
template<typename complex_t>
inline void compare_complex_arrays(const complex_t *expected,
                                  const complex_t *actual,
                                  int N, double tol)
{
    for (int i = 0; i < N; i++)
    {
        EXPECT_NEAR(expected[i][0], actual[i][0], tol)
            << "real mismatch at index " << i;
        EXPECT_NEAR(expected[i][1], actual[i][1], tol)
            << "imag mismatch at index " << i;
    }
}

/* Element-wise comparison against a constant complex value {re, im}.
 * Generalises compare_complex_zero for the delta->flat spectra whose bins are
 * all expected to equal the same constant (e.g. 1+0i). */
template<typename complex_t>
inline void compare_complex_constant(const complex_t *actual, int N,
                                     double re, double im, double tol)
{
    for (int i = 0; i < N; i++)
    {
        EXPECT_NEAR(static_cast<double>(actual[i][0]), re, tol)
            << "real mismatch at index " << i;
        EXPECT_NEAR(static_cast<double>(actual[i][1]), im, tol)
            << "imag mismatch at index " << i;
    }
}

/* Element-wise comparison against exact complex zero. */
template<typename complex_t>
inline void compare_complex_zero(const complex_t *actual, int N, double tol)
{
    compare_complex_constant(actual, N, 0.0, 0.0, tol);
}

/* Element-wise comparison of two real arrays. */
template<typename Real>
inline void compare_real_arrays(const Real *expected, const Real *actual,
                               int N, double tol)
{
    for (int i = 0; i < N; i++)
    {
        EXPECT_NEAR(static_cast<double>(expected[i]),
                    static_cast<double>(actual[i]), tol)
            << "mismatch at index " << i;
    }
}

/* Compare a transform result against the original input scaled by `scale`.
 * An un-normalised forward+backward roundtrip returns input * N; this bakes
 * that `EXPECT_NEAR(result, orig * scale, tol)` check into one call. */
template<typename complex_t>
inline void compare_complex_scaled(const complex_t *orig,
                                   const complex_t *result,
                                   int N, double scale, double tol)
{
    for (int i = 0; i < N; i++)
    {
        EXPECT_NEAR(static_cast<double>(result[i][0]),
                    static_cast<double>(orig[i][0]) * scale, tol)
            << "real mismatch at index " << i;
        EXPECT_NEAR(static_cast<double>(result[i][1]),
                    static_cast<double>(orig[i][1]) * scale, tol)
            << "imag mismatch at index " << i;
    }
}

template<typename Real>
inline void compare_real_scaled(const Real *orig, const Real *result,
                                int N, double scale, double tol)
{
    for (int i = 0; i < N; i++)
    {
        EXPECT_NEAR(static_cast<double>(result[i]),
                    static_cast<double>(orig[i]) * scale, tol)
            << "mismatch at index " << i;
    }
}

/* Cross-precision comparison: the two operands may have different element
 * types (e.g. fftw_complex vs fftwf_complex).  Used to check that the double
 * and single wrappers agree to within the looser single-precision tolerance. */
template<typename ComplexA, typename ComplexB>
inline void compare_complex_cross(const ComplexA *a, const ComplexB *b,
                                  int N, double tol)
{
    for (int i = 0; i < N; i++)
    {
        EXPECT_NEAR(static_cast<double>(a[i][0]),
                    static_cast<double>(b[i][0]), tol)
            << "real mismatch at index " << i;
        EXPECT_NEAR(static_cast<double>(a[i][1]),
                    static_cast<double>(b[i][1]), tol)
            << "imag mismatch at index " << i;
    }
}

/* Verify a complex spectrum equals the linear combination a*A + b*B, i.e. the
 * frequency-domain side of the DFT linearity theorem
 * (DFT(a*x + b*y) == a*DFT(x) + b*DFT(y)). */
template<typename complex_t>
inline void expect_complex_linear_combo(const complex_t *result,
                                        const complex_t *A, const complex_t *B,
                                        double a, double b, int N, double tol)
{
    for (int i = 0; i < N; i++)
    {
        EXPECT_NEAR(static_cast<double>(result[i][0]),
                    a * static_cast<double>(A[i][0]) +
                        b * static_cast<double>(B[i][0]), tol)
            << "real mismatch at index " << i;
        EXPECT_NEAR(static_cast<double>(result[i][1]),
                    a * static_cast<double>(A[i][1]) +
                        b * static_cast<double>(B[i][1]), tol)
            << "imag mismatch at index " << i;
    }
}

/* Verify the DFT shift theorem: a circular shift of the input by `shift`
 * samples multiplies bin k of the spectrum by exp(-j*2*pi*k*shift/N).  Checks
 * that `shifted` is `original` with that linear phase applied. */
template<typename complex_t>
inline void expect_shift_theorem(const complex_t *original,
                                 const complex_t *shifted,
                                 int N, int shift, double tol)
{
    for (int k = 0; k < N; k++)
    {
        double angle = -2.0 * M_PI * k * shift / N;
        double cos_a = std::cos(angle);
        double sin_a = std::sin(angle);
        double exp_re = static_cast<double>(original[k][0]) * cos_a -
                        static_cast<double>(original[k][1]) * sin_a;
        double exp_im = static_cast<double>(original[k][0]) * sin_a +
                        static_cast<double>(original[k][1]) * cos_a;
        EXPECT_NEAR(exp_re, static_cast<double>(shifted[k][0]), tol)
            << "shift theorem violation (real) at k=" << k;
        EXPECT_NEAR(exp_im, static_cast<double>(shifted[k][1]), tol)
            << "shift theorem violation (imag) at k=" << k;
    }
}

/* --------------------------------------------------------------------------
 * Signal energy (sum of squared magnitudes), shared by the Parseval-theorem
 * checks so the accumulation loop is written once.  Computed in double for a
 * stable oracle regardless of transform precision.
 * -------------------------------------------------------------------------- */
template<typename complex_t>
inline double energy_complex(const complex_t *a, long N)
{
    double e = 0.0;
    for (long i = 0; i < N; i++)
    {
        e += static_cast<double>(a[i][0]) * static_cast<double>(a[i][0]) +
             static_cast<double>(a[i][1]) * static_cast<double>(a[i][1]);
    }
    return e;
}

template<typename Real>
inline double energy_real(const Real *a, long N)
{
    double e = 0.0;
    for (long i = 0; i < N; i++)
    {
        e += static_cast<double>(a[i]) * static_cast<double>(a[i]);
    }
    return e;
}

/* --------------------------------------------------------------------------
 * Error-returning comparators.
 *
 * compare_*_arrays() assert internally, which is convenient but unusable from
 * a worker thread (gtest fatal assertions may not fire off the main thread)
 * and awkward inside a size sweep that wants to accumulate a worst-case error.
 * These variants simply return the maximum absolute deviation so the caller
 * decides how to check it.  `stride` is in element units.
 * -------------------------------------------------------------------------- */
template<typename complex_t>
inline double max_error_complex(const complex_t *a, const complex_t *b,
                                int N, int stride = 1)
{
    double m = 0.0;
    for (int i = 0; i < N; i++)
    {
        long idx = static_cast<long>(i) * stride;
        m = std::max(m, std::abs(static_cast<double>(a[idx][0]) -
                                 static_cast<double>(b[idx][0])));
        m = std::max(m, std::abs(static_cast<double>(a[idx][1]) -
                                 static_cast<double>(b[idx][1])));
    }
    return m;
}

template<typename Real>
inline double max_error_real(const Real *a, const Real *b,
                             int N, int stride = 1)
{
    double m = 0.0;
    for (int i = 0; i < N; i++)
    {
        long idx = static_cast<long>(i) * stride;
        m = std::max(m, std::abs(static_cast<double>(a[idx]) -
                                 static_cast<double>(b[idx])));
    }
    return m;
}

/* Scaled variants: worst-case deviation of `result` from `orig * scale`.
 * These are the error-returning twins of compare_*_scaled(), for the
 * un-normalised roundtrip checks (result ~= orig * N) that live inside size
 * sweeps or worker threads and so cannot use the assert-style comparator. */
template<typename complex_t>
inline double max_error_complex_scaled(const complex_t *orig,
                                       const complex_t *result,
                                       int N, double scale, int stride = 1)
{
    double m = 0.0;
    for (int i = 0; i < N; i++)
    {
        long idx = static_cast<long>(i) * stride;
        m = std::max(m, std::abs(static_cast<double>(result[idx][0]) -
                                 static_cast<double>(orig[idx][0]) * scale));
        m = std::max(m, std::abs(static_cast<double>(result[idx][1]) -
                                 static_cast<double>(orig[idx][1]) * scale));
    }
    return m;
}

template<typename Real>
inline double max_error_real_scaled(const Real *orig, const Real *result,
                                    int N, double scale, int stride = 1)
{
    double m = 0.0;
    for (int i = 0; i < N; i++)
    {
        long idx = static_cast<long>(i) * stride;
        m = std::max(m, std::abs(static_cast<double>(result[idx]) -
                                 static_cast<double>(orig[idx]) * scale));
    }
    return m;
}

/* --------------------------------------------------------------------------
 * Gather a single logical line out of a strided / batched / padded buffer
 * into a contiguous scratch buffer: dst[i] = src[offset + i*stride].
 *
 * This lets the contiguous references above be reused to verify interleaved,
 * column-transform and padded-in-place layouts without a bespoke reference
 * for each stride pattern.
 * -------------------------------------------------------------------------- */
template<typename complex_t>
inline void gather_complex(complex_t *dst, const complex_t *src, int N,
                           int stride, int offset = 0)
{
    for (int i = 0; i < N; i++)
    {
        long idx = static_cast<long>(offset) + static_cast<long>(i) * stride;
        dst[i][0] = src[idx][0];
        dst[i][1] = src[idx][1];
    }
}

template<typename Real>
inline void gather_real(Real *dst, const Real *src, int N,
                        int stride, int offset = 0)
{
    for (int i = 0; i < N; i++)
    {
        long idx = static_cast<long>(offset) + static_cast<long>(i) * stride;
        dst[i] = src[idx];
    }
}

/* --------------------------------------------------------------------------
 * Centralised tolerance policy.
 *
 * An un-normalised forward DFT sums `size` inputs, so each output element has
 * magnitude O(size * |input|) and its absolute round-off scales with that
 * magnitude. The suite therefore compares against `F::tolerance * size`, where
 * `size` is the number of points in a SINGLE transform (N for 1-D, the product
 * of the dims for multi-D).
 *
 * `size` is deliberately NOT the batch count: batching runs independent
 * transforms, so it does not increase any element's magnitude or error.
 * Scaling a batched comparison by `howmany` loosens the bound by that factor
 * for no physical reason -- use dft_tolerance with the per-transform size
 * instead.
 * -------------------------------------------------------------------------- */
template<typename F>
inline double dft_tolerance(long size)
{
    return F::tolerance * static_cast<double>(size);
}

/* --------------------------------------------------------------------------
 * RAII guards.
 *
 * The suite's prevalent `alloc -> ASSERT -> ... -> free` shape leaks every
 * buffer/plan on the ASSERT (or exception) path, which shows up as noise under
 * the project's own AddressSanitizer builds. These guards free on scope exit
 * regardless of how the scope is left, so the failure path stays leak-clean.
 *
 * Both implicitly convert to the underlying handle/pointer so they can be
 * handed to the existing wrapper calls without any change at the call site.
 * -------------------------------------------------------------------------- */
template<typename F>
class ScopedPlan
{
public:
    using plan_t = typename F::plan_t;

    ScopedPlan() : p_(nullptr) {}
    ScopedPlan(plan_t p) : p_(p) {}                 // NOLINT: intentional implicit
    ~ScopedPlan() { reset(); }

    ScopedPlan(const ScopedPlan &) = delete;
    ScopedPlan &operator=(const ScopedPlan &) = delete;

    ScopedPlan(ScopedPlan &&o) noexcept : p_(o.p_) { o.p_ = nullptr; }
    ScopedPlan &operator=(ScopedPlan &&o) noexcept
    {
        if (this != &o) { reset(); p_ = o.p_; o.p_ = nullptr; }
        return *this;
    }
    ScopedPlan &operator=(plan_t p)
    {
        if (p_ != p) { reset(); p_ = p; }
        return *this;
    }

    operator plan_t() const { return p_; }
    plan_t get() const { return p_; }

    void reset()
    {
        if (p_) { F::destroy_plan(p_); p_ = nullptr; }
    }

private:
    plan_t p_;
};

template<typename F, typename Elem>
class ScopedBuf
{
public:
    ScopedBuf() : p_(nullptr) {}
    explicit ScopedBuf(Elem *p) : p_(p) {}
    ~ScopedBuf() { if (p_) F::free_fn(p_); }

    ScopedBuf(const ScopedBuf &) = delete;
    ScopedBuf &operator=(const ScopedBuf &) = delete;

    ScopedBuf(ScopedBuf &&o) noexcept : p_(o.p_) { o.p_ = nullptr; }
    ScopedBuf &operator=(ScopedBuf &&o) noexcept
    {
        if (this != &o) { if (p_) F::free_fn(p_); p_ = o.p_; o.p_ = nullptr; }
        return *this;
    }

    operator Elem *() const { return p_; }
    Elem *get() const { return p_; }

private:
    Elem *p_;
};

/* Convenience aliases + factories so a guarded buffer reads as a one-liner:
 *     auto in = make_complex_buf<F>(N);   // freed automatically
 *     auto p  = ScopedPlan<F>(F::plan_dft_1d(...));                          */
template<typename F>
inline ScopedBuf<F, typename F::complex_t> make_complex_buf(size_t n)
{
    return ScopedBuf<F, typename F::complex_t>(F::alloc_complex(n));
}

template<typename F>
inline ScopedBuf<F, typename F::real_t> make_real_buf(size_t n)
{
    return ScopedBuf<F, typename F::real_t>(F::alloc_real(n));
}

#endif /* FFTW_WRAPPER_TEST_UTILS_H */
