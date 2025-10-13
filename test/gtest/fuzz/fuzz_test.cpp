/*
 * Copyright (C) 2025, Advanced Micro Devices. All rights reserved.
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

/** @file fuzz_test.cpp
 *
 *  @brief File that contains the Fuzz tests.
 *
 *  This file includes various fuzz tests for different configurations and
 *  problem sizes of the AOCL FFT library. The tests cover single and double
 *  precision, different problem dimensions, and various control parameters.
 *
 *  @author Jeya R
 *  @author Maheswar Rao S
 */

#include "test/gtest/fuzz/fuzz_test.h"

FUZZ_TEST_F(AoclfftzFuzzTestFloatLP64, fuzz_input_buffer_test)
    .WithDomains(problemsize);

FUZZ_TEST_F(AoclfftzFuzzTestFloatILP64, fuzz_input_buffer_test)
    .WithDomains(problemsize);

FUZZ_TEST_F(AoclfftzFuzzTestDoubleLP64, fuzz_input_buffer_test)
    .WithDomains(problemsize);

FUZZ_TEST_F(AoclfftzFuzzTestDoubleILP64, fuzz_input_buffer_test)
    .WithDomains(problemsize);

auto arbitrary_cntrl_params() {
    return fuzztest::StructOf<aoclfftz_cntrl_params_t>(
        // Define generators for each field in aoclfftz_cntrl_params_t
        fuzztest::Arbitrary<INT32>(),   // opt_level
        fuzztest::Arbitrary<INT32>(),   // opt_off
        fuzztest::ElementOf<aoclfftz_logger_mode>({
            AOCLFFTZ_LOG_NONE, AOCLFFTZ_LOG_ERROR, AOCLFFTZ_LOG_INFO,
            AOCLFFTZ_LOG_DEBUG, AOCLFFTZ_LOG_TRACE}),   // logger_mode
        fuzztest::Arbitrary<INT32>());  // measure_stats
}

// 1D FFT problems
FUZZ_TEST_F(AoclfftzFuzzTestFloatLP64, fuzz_problem_desc_test)
    .WithDomains(dims_and_vecs_1D(),
                //->flags are fuzzed for valid combinations not to waste fuzz cycles.
                // Flags: 0=C2C+Fwd+InPlace, 1=+OutPlace, 4=+Bwd, 5=+Bwd+OutPlace,
                // 8=Real+Fwd+InPlace, 9=+OutPlace, 12=+Bwd, 13=+Bwd+OutPlace
                fuzztest::ElementOf<UINT32>({0, 1, 4, 5, 8, 9, 12, 13}),
                fuzztest::Arbitrary<aoclfftz_smp_pfft_t>(),
                arbitrary_cntrl_params());

// Batched 1D problem
FUZZ_TEST_F(AoclfftzFuzzTestFloatILP64, fuzz_problem_desc_test)
    .WithDomains(dims_and_vecs_batched_1D(),
                //->flags are fuzzed for valid combinations not to waste fuzz cycles.
                // Flags: 0=C2C+Fwd+InPlace, 1=+OutPlace, 4=+Bwd, 5=+Bwd+OutPlace,
                // 8=Real+Fwd+InPlace, 9=+OutPlace, 12=+Bwd, 13=+Bwd+OutPlace
                fuzztest::ElementOf<UINT32>({0, 1, 4, 5, 8, 9, 12, 13}),
                fuzztest::Arbitrary<aoclfftz_smp_pfft_t>(),
                arbitrary_cntrl_params());

// 2D FFT problem
FUZZ_TEST_F(AoclfftzFuzzTestDoubleLP64, fuzz_problem_desc_test)
    .WithDomains(dims_and_vecs_2D(),
                // TODO: Add flag combination of real FFT once ND real FFT is supported
                // Flags: 0=C2C+Fwd+InPlace, 1=+OutPlace, 4=+Bwd, 5=+Bwd+OutPlace
                fuzztest::ElementOf<UINT32>({0, 1, 4, 5}), //->flags
                fuzztest::Arbitrary<aoclfftz_smp_pfft_t>(),
                arbitrary_cntrl_params());

// Multi batched/N-Dim FFT problem
FUZZ_TEST_F(AoclfftzFuzzTestDoubleILP64, fuzz_problem_desc_test)
    .WithDomains(dims_and_vecs_multi_batched_ND(),
                // TODO: Add flag combination of real FFT once ND real FFT is supported
                // Flags: 0=C2C+Fwd+InPlace, 1=+OutPlace, 4=+Bwd, 5=+Bwd+OutPlace
                fuzztest::ElementOf<UINT32>({0, 1, 4, 5}), //->flags
                fuzztest::Arbitrary<aoclfftz_smp_pfft_t>(),
                arbitrary_cntrl_params());
