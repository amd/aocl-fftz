/**
 * Copyright (C) 2023, Advanced Micro Devices. All rights reserved.
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

/** @file selector_direct_gtest.cpp
 *
 *  @brief File that contains the GTest based direct selector unit tests.
 *
 *  This file contains the entry point of the direct selector GTest.
 *
 *  @author Srirammaswamy Srinivasan
 */

#include <gtest/gtest.h>
#include "gtest/selector/selector_gtest_base.h"

TEST_F(AoclfftzSelectorTestFloatLP64, TEST_SELECTOR_DIRECT_FLOAT_LP64)
{
    std::string dims = "2:1:1";
    INT32 flags = 0b0000;
    INT32 opt_level = -1;
    aoclfftz_solution_t *sol =
        run_setup_and_get_solution(dims, flags, opt_level);
    aoclfftz_solution_t *ref_sol =
        generate_reference_solution(dims, flags, opt_level, SOLVER_DIRECT);
    COMPARE_SOLUTIONS(sol, ref_sol);
}

TEST_F(AoclfftzSelectorTestDoubleLP64, TEST_SELECTOR_DIRECT_DOUBLE_LP64)
{
    std::string dims = "5:1:2";
    INT32 flags = 0b0000;
    INT32 opt_level = -1;
    aoclfftz_solution_t *sol =
        run_setup_and_get_solution(dims, flags, opt_level);
    aoclfftz_solution_t *ref_sol =
        generate_reference_solution(dims, flags, opt_level, SOLVER_DIRECT);
    COMPARE_SOLUTIONS(sol, ref_sol);
}

TEST_F(AoclfftzSelectorTestFloatILP64, TEST_SELECTOR_DIRECT_FLOAT_ILP64)
{
    std::string dims = "8:3:1";
    INT32 flags = 0b0000;
    INT32 opt_level = -1;
    aoclfftz_solution_t *sol =
        run_setup_and_get_solution(dims, flags, opt_level);
    aoclfftz_solution_t *ref_sol =
        generate_reference_solution(dims, flags, opt_level, SOLVER_DIRECT);
    COMPARE_SOLUTIONS(sol, ref_sol);
}

TEST_F(AoclfftzSelectorTestDoubleILP64, TEST_SELECTOR_DIRECT_DOUBLE_ILP64)
{
    std::string dims = "1v13:5:3";
    INT32 flags = 0b0000;
    INT32 opt_level = -1;
    aoclfftz_solution_t *sol =
        run_setup_and_get_solution(dims, flags, opt_level);
    aoclfftz_solution_t *ref_sol =
        generate_reference_solution(dims, flags, opt_level, SOLVER_DIRECT);
    COMPARE_SOLUTIONS(sol, ref_sol);
}