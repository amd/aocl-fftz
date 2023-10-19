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

std::string dims_and_vecs[] = {
    // direct problems
    "2:1:1",
    "5:1:2",
    "8:3:1",
    "1v13:5:3",
    // batched direct problems
    "7v2:1:1",
    "4v15:1:4",
    "4:60:24v12:5:1",
    "10:45:24v3:5:2"
};

INT32 flags[] = {
    0b0000, // complex, forward, in-order, in-place problem
    0b0001, // complex, forward, in-order, out-of-place problem
    0b0100  // complex, backward, in-order, in-place problem
};

INT32 opt_levels[] = {
    -1 // no optimization at all
};

TEST_P(AoclfftzSelectorTestFloatLP64, TEST_SELECTOR_DIRECT_FLOAT_LP64)
{
    run_selector_test_and_verify_solutions(SOLVER_DIRECT);
}

TEST_P(AoclfftzSelectorTestDoubleLP64, TEST_SELECTOR_DIRECT_DOUBLE_LP64)
{
    run_selector_test_and_verify_solutions(SOLVER_DIRECT);
}

TEST_P(AoclfftzSelectorTestFloatILP64, TEST_SELECTOR_DIRECT_FLOAT_ILP64)
{
    run_selector_test_and_verify_solutions(SOLVER_DIRECT);
}

TEST_P(AoclfftzSelectorTestDoubleILP64, TEST_SELECTOR_DIRECT_DOUBLE_ILP64)
{
    run_selector_test_and_verify_solutions(SOLVER_DIRECT);
}

INSTANTIATE_TEST_SUITE_P(
    SelectorDirectParamTest, AoclfftzSelectorTestFloatLP64,
    ::testing::Combine(
        ::testing::ValuesIn(dims_and_vecs),
        ::testing::ValuesIn(flags),
        ::testing::ValuesIn(opt_levels),
        ::testing::Values(std::vector<aoclfftz_solver_type>()) // empty list
        ));

INSTANTIATE_TEST_SUITE_P(
    SelectorDirectParamTest, AoclfftzSelectorTestDoubleLP64,
    ::testing::Combine(
        ::testing::ValuesIn(dims_and_vecs),
        ::testing::ValuesIn(flags),
        ::testing::ValuesIn(opt_levels),
        ::testing::Values(std::vector<aoclfftz_solver_type>()) // empty list
        ));

INSTANTIATE_TEST_SUITE_P(
    SelectorDirectParamTest, AoclfftzSelectorTestFloatILP64,
    ::testing::Combine(
        ::testing::ValuesIn(dims_and_vecs),
        ::testing::ValuesIn(flags),
        ::testing::ValuesIn(opt_levels),
        ::testing::Values(std::vector<aoclfftz_solver_type>()) // empty list
        ));

INSTANTIATE_TEST_SUITE_P(
    SelectorDirectParamTest, AoclfftzSelectorTestDoubleILP64,
    ::testing::Combine(
        ::testing::ValuesIn(dims_and_vecs),
        ::testing::ValuesIn(flags),
        ::testing::ValuesIn(opt_levels),
        ::testing::Values(std::vector<aoclfftz_solver_type>()) // empty list
        ));