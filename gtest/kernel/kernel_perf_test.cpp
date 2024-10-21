/*
 * Copyright (C) 2024, Advanced Micro Devices. All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 * 1. Redistributions of source code must retain the above copyright notice,
 * this list of conditions and the following disclaimer.
 * 2. Redistributions in binary form must reproduce the above copyright notice,
 * this list of conditions and the following disclaimer in the documentation
 * and/or other materials provided with the distribution.
 * 3. Neither the name of the copyright holder nor the names of its
 * contributors may be used to endorse or promote products derived from this
 * software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
 * ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
 * LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
 * CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
 * SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
 * INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
 * CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
 * ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
 * POSSIBILITY OF SUCH DAMAGE.
 */

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
#include "kernel_gtest_base.h"
#include "utils/utils.h"

#define IN_STRIDE 1
#define OUT_STRIDE 1
#define VEC_IN_STRIDE 1
#define VEC_OUT_STRIDE 1

template <typename T>
class PerformanceTest : public benchmark::Fixture {
    public:
    void SetUp(::benchmark::State& state) {}
    void TearDown(::benchmark::State& state) {}
    void kernel_profile(::benchmark::State& state)
    {
        UINT32 radix = state.range(0);
        INTP batches = state.range(1);
        UINT8 kernel_type = state.range(2);
        INTP in_stride = state.range(3);
        INTP out_stride = state.range(4);
        INTP v_in_stride = state.range(5) != 1 ? state.range(5) :
                           radix * in_stride;
        INTP v_out_stride = state.range(6) != 1 ? state.range(6) :
                            radix * out_stride;
        UINT8 is_bwd = false;

        // Radix-13 AVX versions are yet to be implemented
        if (radix == 13 && (kernel_type != aocl_fftz_kernel_type::STANDARD_C &&
                            kernel_type != aocl_fftz_kernel_type::PERMUTED_C))
        {
            state.SkipWithMessage("Radix-13 only supported in C version, "
                                  "skipping AVX version");
            return;
        }

        wrapper_kernel_fp_list *table = get_kernel_table(kernel_type);
        kfft_ fft_kernel = get_kernel<T>(table, radix);
        if (fft_kernel == nullptr)
        {
            state.SkipWithError(std::string("Radix-" + std::to_string(radix) +
                get_kernel_type_as_string(kernel_type) +
                " kernel not found in the kernel table").c_str());
            return;
        }

        // ---------- prepare random input and output ----------
        INTP input_length  = radix * batches * in_stride;
        INTP output_length = radix * batches * out_stride;

        T *in = NULL;
        ALLOC_ALIGN_UNINIT(in, T, sizeof(T) * input_length * DATA_STRIDE);
        for (INTP idx = 0; idx < input_length * DATA_STRIDE; ++idx)
        {
            // range: [-10.0, 10.0) with 3 decimal precision
            in[idx] = (T)((rand() % 2000) / 200.0) - 10.0;
        }
        T *out;
        ALLOC_ALIGN_INIT(out, T, output_length * DATA_STRIDE * sizeof(T));
        T *in_r = in;
        T *in_i = in + 1;
        T *out_r = out;
        T *out_i = out + 1;

        aoclfftz_strides_t strides;
        ALLOC_ALIGN_UNINIT(strides.in_strides, INTP, radix * sizeof(INTP));
        ALLOC_ALIGN_UNINIT(strides.out_strides, INTP, radix * sizeof(INTP));

        for (INTP i = 0; i < radix; i++)
        {
            strides.in_strides[i] = i * DATA_STRIDE * in_stride;
            strides.out_strides[i] = i * DATA_STRIDE * out_stride;
        }

        strides.v_in_stride = v_in_stride * DATA_STRIDE;
        strides.v_out_stride = v_out_stride * DATA_STRIDE;

        for (auto _ : state)
        {
            benchmark::DoNotOptimize(out_r);
            benchmark::DoNotOptimize(out_i);
            fft_kernel(in_r, in_i, out_r, out_i, batches, &strides, is_bwd);
            benchmark::ClobberMemory();
        }

        FREE_ALIGN_ALLOCATED_MEM(in);
        FREE_ALIGN_ALLOCATED_MEM(out);
        FREE_ALIGN_ALLOCATED_MEM(strides.in_strides);
        FREE_ALIGN_ALLOCATED_MEM(strides.out_strides);
    }
};

BENCHMARK_TEMPLATE_DEFINE_F(PerformanceTest, Kernel_d, DOUBLE)
                           (benchmark::State& state)
{
    kernel_profile(state);
}

BENCHMARK_TEMPLATE_DEFINE_F(PerformanceTest, Kernel_f, FLOAT)
                           (benchmark::State& state)
{
    kernel_profile(state);
}

// filter to benchmark specific test case : --benchmark_filter=.FLOAT.*/9/5/4/
// runs Radix-9 AVX-256 Float variant with batch size of 5
BENCHMARK_REGISTER_F(PerformanceTest, Kernel_d)
    ->ComputeStatistics("min", [](const std::vector<double>& v) -> double {
        return *(std::min_element(std::begin(v), std::end(v)));
    })
    ->ArgsProduct({
                benchmark::CreateDenseRange(2, 16, 1),
                // Batch sizes to cover all possible cases in C & AVX-128/256
                benchmark::CreateDenseRange(1, 7, 1),
                // aocl_fftz_kernel_type -> STANDARD/PERMUTED C/AVX
                benchmark::CreateDenseRange(0, 5, 1),
                {IN_STRIDE}, {OUT_STRIDE},
                {VEC_IN_STRIDE}, {VEC_OUT_STRIDE},
    });

BENCHMARK_REGISTER_F(PerformanceTest, Kernel_f)
    ->ComputeStatistics("min", [](const std::vector<double>& v) -> double {
        return *(std::min_element(std::begin(v), std::end(v)));
    })
    ->ArgsProduct({
                benchmark::CreateDenseRange(2, 16, 1),
                benchmark::CreateDenseRange(1, 7, 1), // Batch sizes to cover all possible cases in C & AVX
                benchmark::CreateDenseRange(0, 5, 1),  // aocl_fftz_kernel_type -> STANDARD/PERMUTED C/AVX
                {IN_STRIDE}, {OUT_STRIDE},
                {VEC_IN_STRIDE}, {VEC_OUT_STRIDE},
    });

BENCHMARK_MAIN();
