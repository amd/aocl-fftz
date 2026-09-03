// Copyright Advanced Micro Devices, Inc.
// SPDX-License-Identifier: BSD-3-Clause

/** @file kernel_perf_gtest.cpp
 *
 *  @brief File that contains the GTest based kernel performance tests.
 *
 * This file contains the kernel performance test based on Google micro
 * benchmark.
 *
 *  @author Srirammaswamy Srinivasan
 *  @author Jeya R
 */

#include <gtest/gtest.h>
#include <benchmark/benchmark.h>
#include "api/aoclfftz_internal.h"
#include "kernel_gtest_base.h"
#include "utils/utils.h"

#define IN_STRIDE 1
#define OUT_STRIDE 1
#define VEC_IN_STRIDE 1
#define VEC_OUT_STRIDE 1
#define REAL 1
#define COMPLEX 0

template <typename T>
class PerformanceTest : public benchmark::Fixture {
    public:
    void SetUp(::benchmark::State& state) {}
    void TearDown(::benchmark::State& state) {}
    void kernel_profile(::benchmark::State& state)
    {
        FFTZ_UINT32 radix = state.range(0);
        FFTZ_INTP batches = state.range(1);
        FFTZ_UINT8 kernel_type = state.range(2);
        FFTZ_INTP in_stride = state.range(3);
        FFTZ_INTP out_stride = state.range(4);
        FFTZ_INTP v_in_stride = state.range(5) != 1 ? state.range(5) :
                           radix * in_stride;
        FFTZ_INTP v_out_stride = state.range(6) != 1 ? state.range(6) :
                            radix * out_stride;
        FFTZ_UINT8 is_bwd = false;
        FFTZ_UINT8 is_real = state.range(7);
        FFTZ_INT32 data_stride = is_real ? 1 : 2;

        wrapper_kernel_fp_list *table = get_kernel_table(kernel_type);
        kfft_ fft_kernel = get_kernel<T>(table, is_bwd, radix);
        if (fft_kernel == nullptr)
        {
            state.SkipWithError(std::string("Radix-" + std::to_string(radix) +
                get_kernel_type_as_string(kernel_type) +
                " kernel not found in the kernel table").c_str());
            return;
        }

        // ---------- prepare random input and output ----------
        FFTZ_INTP input_length  = radix * batches * in_stride;
        FFTZ_INTP output_length = radix * batches * out_stride;

        T *in = NULL;
        ALLOC_ALIGN_UNINIT(in, T, sizeof(T) * input_length * DATA_STRIDE);
        for (FFTZ_INTP idx = 0; idx < input_length * DATA_STRIDE; ++idx)
        {
            // range: [-10.0, 10.0) with 3 decimal precision
            in[idx] = (T)((rand() % 20000) / 1000.0) - 10.0;
        }
        T *out;
        ALLOC_ALIGN_INIT(out, T, output_length * DATA_STRIDE * sizeof(T));
        T *in_r = in;
        T *in_i = in + 1;
        T *out_r = out;
        T *out_i = out + 1;

        aoclfftz_strides_t strides;
        ALLOC_ALIGN_UNINIT(strides.in_strides, FFTZ_INTP,
                           radix * sizeof(FFTZ_INTP));
        ALLOC_ALIGN_UNINIT(strides.out_strides, FFTZ_INTP,
                           radix * sizeof(FFTZ_INTP));

        for (FFTZ_INTP i = 0; i < radix; i++)
        {
            strides.in_strides[i] = i * data_stride * in_stride;
            strides.out_strides[i] = i * data_stride * out_stride;
        }

        strides.v_in_stride = v_in_stride * data_stride;
        strides.v_out_stride = v_out_stride * data_stride;
        strides.v_in_sym_stride = strides.v_in_stride;
        strides.v_out_sym_stride = strides.v_out_stride;

        for (auto _ : state)
        {
            benchmark::DoNotOptimize(out_r);
            benchmark::DoNotOptimize(out_i);
            fft_kernel(in_r, in_i, out_r, out_i, batches, &strides, NULL,
                       is_bwd);
            benchmark::ClobberMemory();
        }

        FREE_ALIGN_ALLOCATED_MEM(in);
        FREE_ALIGN_ALLOCATED_MEM(out);
        FREE_ALIGN_ALLOCATED_MEM(strides.in_strides);
        FREE_ALIGN_ALLOCATED_MEM(strides.out_strides);
    }

    void twiddle_kernel_profile(::benchmark::State& state)
    {
        FFTZ_UINT32 radix = state.range(0);
        FFTZ_INTP batches = state.range(1);
        FFTZ_UINT8 kernel_type = state.range(2);
        FFTZ_INTP in_stride = state.range(3);
        FFTZ_INTP out_stride = state.range(4);
        FFTZ_INTP v_in_stride = state.range(5) != 1 ? state.range(5) :
                           radix * in_stride;
        FFTZ_INTP v_out_stride = state.range(6) != 1 ? state.range(6) :
                            radix * out_stride;
        FFTZ_UINT8 is_bwd = false;
        FFTZ_UINT8 is_real = state.range(7);
        FFTZ_INT32 data_stride = is_real ? 1 : 2;
        aoclfftz_twiddle_t tws;

        bool is_c2c_twid_kernel =
            kernel_type >= aocl_fftz_kernel_type::C2C_TWID_FWD_C &&
            kernel_type <= aocl_fftz_kernel_type::C2C_TWID_BWD_AVX512;
        bool is_r2c_c2r_twid_kernel =
            kernel_type >= aocl_fftz_kernel_type::C2C_TWID_R2C_C &&
            kernel_type <= aocl_fftz_kernel_type::C2C_TWID_C2R_AVX512;
        if (!is_c2c_twid_kernel && !is_r2c_c2r_twid_kernel)
        {
            state.SkipWithError(
                std::string("Given kernel is not a twiddle kernel.").c_str());
            return;
        }

        kfft_ tw_kernel = get_twiddle_kernel<T>(radix, is_bwd, kernel_type);
        if (tw_kernel == nullptr)
        {
            state.SkipWithError(
                std::string("Radix-" + std::to_string(radix) +
                            get_kernel_type_as_string(kernel_type) +
                            " twiddle kernel does not exist.")
                    .c_str());
            return;
        }

        // ---------- prepare random input and output ----------
        FFTZ_INTP input_length  = radix * batches * in_stride;
        FFTZ_INTP output_length = radix * batches * out_stride;

        T *in = NULL;
        ALLOC_ALIGN_UNINIT(in, T, sizeof(T) * input_length * DATA_STRIDE);
        for (FFTZ_INTP idx = 0; idx < input_length * DATA_STRIDE; ++idx)
        {
            // range: [-10.0, 10.0) with 3 decimal precision
            in[idx] = (T)((rand() % 20000) / 1000.0) - 10.0;
        }
        T *out;
        ALLOC_ALIGN_INIT(out, T, output_length * DATA_STRIDE * sizeof(T));
        T *in_r = in;
        T *in_i = in + 1;
        T *out_r = out;
        T *out_i = out + 1;

        aoclfftz_strides_t strides;
        ALLOC_ALIGN_UNINIT(strides.in_strides, FFTZ_INTP,
                           radix * sizeof(FFTZ_INTP));
        ALLOC_ALIGN_UNINIT(strides.out_strides, FFTZ_INTP,
                           radix * sizeof(FFTZ_INTP));

        for (FFTZ_INTP i = 0; i < radix; i++)
        {
            strides.in_strides[i] = i * data_stride * in_stride;
            strides.out_strides[i] = i * data_stride * out_stride;
        }

        strides.v_in_stride = v_in_stride * data_stride;
        strides.v_out_stride = v_out_stride * data_stride;
        strides.v_in_sym_stride = strides.v_in_stride;
        strides.v_out_sym_stride = strides.v_out_stride;

        FFTZ_VOID *twiddle_buffer = NULL;
        // setup the twiddle buffer
        ALLOC_ALIGN_UNINIT(twiddle_buffer, FFTZ_VOID,
                           data_stride * sizeof(T) * batches * radix);
        if (twiddle_buffer == nullptr)
        {
            state.SkipWithError(
                std::string(
                    "Internal error: Failed to allocate twiddle buffer.")
                    .c_str());
            goto cleanup;
        }
        {
            FFTZ_INTP rw = twiddle_kernel_register_width<T>(
                static_cast<aocl_fftz_kernel_type>(kernel_type));
            compute_twiddle_buffer_wrapper<T>(twiddle_buffer, radix, batches,
                                              rw, /*load_multi_cols=*/1);
        }

        tws.TW = twiddle_buffer;
        tws.twiddle_buf_ptr = twiddle_buffer;
        tws.load_multi_cols = 1; // true by default

        for (auto _ : state)
        {
            benchmark::DoNotOptimize(out_r);
            benchmark::DoNotOptimize(out_i);
            tw_kernel(in_r, in_i, out_r, out_i, batches, &strides,
                      &tws, is_bwd);
            benchmark::ClobberMemory();
        }

cleanup:
        FREE_ALIGN_ALLOCATED_MEM(twiddle_buffer);
        FREE_ALIGN_ALLOCATED_MEM(in);
        FREE_ALIGN_ALLOCATED_MEM(out);
        FREE_ALIGN_ALLOCATED_MEM(strides.in_strides);
        FREE_ALIGN_ALLOCATED_MEM(strides.out_strides);
    }
};

BENCHMARK_TEMPLATE_DEFINE_F(PerformanceTest, Kernel_d, FFTZ_DOUBLE)
                           (benchmark::State& state)
{
    kernel_profile(state);
}

BENCHMARK_TEMPLATE_DEFINE_F(PerformanceTest, Kernel_f, FFTZ_FLOAT)
                           (benchmark::State& state)
{
    kernel_profile(state);
}

BENCHMARK_TEMPLATE_DEFINE_F(PerformanceTest, Kernel_twiddle_d, FFTZ_DOUBLE)
                           (benchmark::State& state)
{
    twiddle_kernel_profile(state);
}

BENCHMARK_TEMPLATE_DEFINE_F(PerformanceTest, Kernel_twiddle_f, FFTZ_FLOAT)
                           (benchmark::State& state)
{
    twiddle_kernel_profile(state);
}

BENCHMARK_TEMPLATE_DEFINE_F(PerformanceTest, Kernel_real_d, FFTZ_DOUBLE)
                           (benchmark::State& state)
{
    kernel_profile(state);
}

BENCHMARK_TEMPLATE_DEFINE_F(PerformanceTest, Kernel_real_f, FFTZ_FLOAT)
                           (benchmark::State& state)
{
    kernel_profile(state);
}

                           // filter to benchmark specific test case :
                           // --benchmark_filter=
                           // "FFTZ_DOUBLE.*radix:6/batch:1/kernel_type:0/*"
                           // which runs Radix-6 C Double kernel with batch size
                           // of 1
                           BENCHMARK_REGISTER_F(PerformanceTest, Kernel_d)
                               ->ComputeStatistics(
                                   "min",
                                   [](const std::vector<double> &v) -> double {
                                       return *(std::min_element(std::begin(v),
                                                                 std::end(v)));
                                   })
                               ->ArgsProduct({
                                   // Covers all direct kernels from 2-16, 48
                                   {2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14,
                                    15, 16, 48},
                                   // Batch sizes to cover all possible cases in
                                   // C & AVX kernels
                                   benchmark::CreateDenseRange(1, 15, 1),
                                   // aocl_fftz_kernel_type -> C/AVX
                                   benchmark::CreateDenseRange(
                                       aocl_fftz_kernel_type::C2C_C,
                                       aocl_fftz_kernel_type::C2C_AVX512, 1),
                                   {IN_STRIDE},
                                   {OUT_STRIDE},
                                   {VEC_IN_STRIDE},
                                   {VEC_OUT_STRIDE},
                                   {COMPLEX},
                               })
                               ->ArgNames({"radix", "batch", "kernel_type",
                                           "in_stride", "out_stride",
                                           "v_in_stride", "v_out_stride",
                                           "is_real"});

                           BENCHMARK_REGISTER_F(PerformanceTest, Kernel_f)
                               ->ComputeStatistics(
                                   "min",
                                   [](const std::vector<double> &v) -> double {
                                       return *(std::min_element(std::begin(v),
                                                                 std::end(v)));
                                   })
                               ->ArgsProduct({
                                   // Covers all direct kernels from 2-16, 48
                                   {2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14,
                                    15, 16, 48},
                                   // Batch sizes to cover all possible cases in
                                   // C & AVX kernels
                                   benchmark::CreateDenseRange(1, 15, 1),
                                   // aocl_fftz_kernel_type -> C/AVX
                                   benchmark::CreateDenseRange(
                                       aocl_fftz_kernel_type::C2C_C,
                                       aocl_fftz_kernel_type::C2C_AVX512, 1),
                                   {IN_STRIDE},
                                   {OUT_STRIDE},
                                   {VEC_IN_STRIDE},
                                   {VEC_OUT_STRIDE},
                                   {COMPLEX},
                               })
                               ->ArgNames({"radix", "batch", "kernel_type",
                                           "in_stride", "out_stride",
                                           "v_in_stride", "v_out_stride",
                                           "is_real"});

                           BENCHMARK_REGISTER_F(PerformanceTest,
                                                Kernel_twiddle_d)
                               ->ComputeStatistics(
                                   "min",
                                   [](const std::vector<double> &v) -> double {
                                       return *(std::min_element(std::begin(v),
                                                                 std::end(v)));
                                   })
                               ->ArgsProduct({
                                   // Covers all direct kernels from 2-16
                                   benchmark::CreateDenseRange(2, 16, 1),
                                   {31},
                                   // Batch sizes to cover all possible cases in
                                   // C & AVX kernels
                                   // benchmark::CreateDenseRange(1, 15, 1),
                                   // aocl_fftz_kernel_type -> C/AVX,
                                   // FWD then BWD
                                   benchmark::CreateDenseRange(
                                       aocl_fftz_kernel_type::C2C_TWID_FWD_C,
                                       aocl_fftz_kernel_type::
                                           C2C_TWID_BWD_AVX512,
                                       1),
                                   {IN_STRIDE},
                                   {OUT_STRIDE},
                                   {VEC_IN_STRIDE},
                                   {VEC_OUT_STRIDE},
                                   {COMPLEX},
                               })
                               ->ArgNames({"radix", "batch", "kernel_type",
                                           "in_stride", "out_stride",
                                           "v_in_stride", "v_out_stride",
                                           "is_real"});

                           BENCHMARK_REGISTER_F(PerformanceTest,
                                                Kernel_twiddle_f)
                               ->ComputeStatistics(
                                   "min",
                                   [](const std::vector<double> &v) -> double {
                                       return *(std::min_element(std::begin(v),
                                                                 std::end(v)));
                                   })
                               ->ComputeStatistics(
                                   "min",
                                   [](const std::vector<double> &v) -> double {
                                       return *(std::min_element(std::begin(v),
                                                                 std::end(v)));
                                   })
                               ->ArgsProduct({
                                   // Covers all direct kernels from 2-16
                                   benchmark::CreateDenseRange(2, 16, 1),
                                   {31},
                                   // Batch sizes to cover all possible cases in
                                   // C & AVX kernels
                                   // benchmark::CreateDenseRange(1, 15, 1),
                                   // aocl_fftz_kernel_type -> C/AVX,
                                   // FWD then BWD
                                   benchmark::CreateDenseRange(
                                       aocl_fftz_kernel_type::C2C_TWID_FWD_C,
                                       aocl_fftz_kernel_type::
                                           C2C_TWID_BWD_AVX512,
                                       1),
                                   {IN_STRIDE},
                                   {OUT_STRIDE},
                                   {VEC_IN_STRIDE},
                                   {VEC_OUT_STRIDE},
                                   {COMPLEX},
                               })
                               ->ArgNames({"radix", "batch", "kernel_type",
                                           "in_stride", "out_stride",
                                           "v_in_stride", "v_out_stride",
                                           "is_real"});

                           // Second twiddle registration covering the R2C and
                           // C2R kinds, which the C2C FWD..BWD range above
                           // does not reach
                           BENCHMARK_REGISTER_F(PerformanceTest,
                                                Kernel_twiddle_d)
                               ->ComputeStatistics(
                                   "min",
                                   [](const std::vector<double> &v) -> double {
                                       return *(std::min_element(std::begin(v),
                                                                 std::end(v)));
                                   })
                               ->ArgsProduct({
                                   // Covers all direct kernels from 2-16
                                   benchmark::CreateDenseRange(2, 16, 1),
                                   {31},
                                   // aocl_fftz_kernel_type -> C/AVX,
                                   // R2C then C2R
                                   benchmark::CreateDenseRange(
                                       aocl_fftz_kernel_type::C2C_TWID_R2C_C,
                                       aocl_fftz_kernel_type::
                                           C2C_TWID_C2R_AVX512,
                                       1),
                                   {IN_STRIDE},
                                   {OUT_STRIDE},
                                   {VEC_IN_STRIDE},
                                   {VEC_OUT_STRIDE},
                                   {COMPLEX},
                               })
                               ->ArgNames({"radix", "batch", "kernel_type",
                                           "in_stride", "out_stride",
                                           "v_in_stride", "v_out_stride",
                                           "is_real"});

                           BENCHMARK_REGISTER_F(PerformanceTest,
                                                Kernel_twiddle_f)
                               ->ComputeStatistics(
                                   "min",
                                   [](const std::vector<double> &v) -> double {
                                       return *(std::min_element(std::begin(v),
                                                                 std::end(v)));
                                   })
                               ->ArgsProduct({
                                   // Covers all direct kernels from 2-16
                                   benchmark::CreateDenseRange(2, 16, 1),
                                   {31},
                                   // aocl_fftz_kernel_type -> C/AVX,
                                   // R2C then C2R
                                   benchmark::CreateDenseRange(
                                       aocl_fftz_kernel_type::C2C_TWID_R2C_C,
                                       aocl_fftz_kernel_type::
                                           C2C_TWID_C2R_AVX512,
                                       1),
                                   {IN_STRIDE},
                                   {OUT_STRIDE},
                                   {VEC_IN_STRIDE},
                                   {VEC_OUT_STRIDE},
                                   {COMPLEX},
                               })
                               ->ArgNames({"radix", "batch", "kernel_type",
                                           "in_stride", "out_stride",
                                           "v_in_stride", "v_out_stride",
                                           "is_real"});

                           BENCHMARK_REGISTER_F(PerformanceTest, Kernel_real_d)
                               ->ComputeStatistics(
                                   "min",
                                   [](const std::vector<double> &v) -> double {
                                       return *(std::min_element(std::begin(v),
                                                                 std::end(v)));
                                   })
                               ->ArgsProduct({
                                   benchmark::CreateDenseRange(2, 4, 1),
                                   // Batch sizes to cover all possible cases in
                                   // C & AVX
                                   benchmark::CreateDenseRange(1, 31, 1),
                                   benchmark::CreateDenseRange(
                                       aocl_fftz_kernel_type::R2HC_C,
                                       aocl_fftz_kernel_type::R2HCF_AVX512, 1),
                                   {IN_STRIDE},
                                   {OUT_STRIDE},
                                   {VEC_IN_STRIDE},
                                   {VEC_OUT_STRIDE},
                                   {REAL},
                               })
                               ->ArgNames({"radix", "batch", "kernel_type",
                                           "in_stride", "out_stride",
                                           "v_in_stride", "v_out_stride",
                                           "is_real"});

                           BENCHMARK_REGISTER_F(PerformanceTest, Kernel_real_f)
                               ->ComputeStatistics(
                                   "min",
                                   [](const std::vector<double> &v) -> double {
                                       return *(std::min_element(std::begin(v),
                                                                 std::end(v)));
                                   })
                               ->ArgsProduct({
                                   benchmark::CreateDenseRange(2, 4, 1),
                                   // Batch sizes to cover all possible cases in
                                   // C & AVX
                                   benchmark::CreateDenseRange(1, 31, 1),
                                   benchmark::CreateDenseRange(
                                       aocl_fftz_kernel_type::R2HC_C,
                                       aocl_fftz_kernel_type::R2HCF_AVX512, 1),
                                   {IN_STRIDE},
                                   {OUT_STRIDE},
                                   {VEC_IN_STRIDE},
                                   {VEC_OUT_STRIDE},
                                   {REAL},
                               })
                               ->ArgNames({"radix", "batch", "kernel_type",
                                           "in_stride", "out_stride",
                                           "v_in_stride", "v_out_stride",
                                           "is_real"});

                           BENCHMARK_MAIN();
