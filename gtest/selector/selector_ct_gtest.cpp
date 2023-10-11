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

/** @file selector_ct_gtest.cpp
 *
 *  @brief File that contains the GTest based CT selector unit tests.
 *
 *  This file contains the entry point of the CT selector GTest.
 *
 *  @author Srirammaswamy Srinivasan
 *  @author Murugan Vairavel
 */

#include <gtest/gtest.h>
#include "gtest/selector/selector_gtest_base.h"

std::string dims_and_vecs[] = {
    // CT problems
    "18:1:1",
    "48:2:5",
    "38:3:1",
    "57:4:7"
};

INT32 flags[] = {
    0b0000, // complex, forward, in-order, in-place problem
    0b0001, // complex, forward, in-order, out-of-place problem
    0b0100  // complex, backward, in-order, in-place problem
};

INT32 opt_levels[] = {
    -1 // no optimization at all
};

TEST_P(AoclfftzSelectorTestFloatLP64, TEST_SELECTOR_CT_FLOAT_LP64)
{
    run_selector_test_and_verify_solutions(SOLVER_CT);
}

TEST_P(AoclfftzSelectorTestDoubleLP64, TEST_SELECTOR_CT_DOUBLE_LP64)
{
    run_selector_test_and_verify_solutions(SOLVER_CT);
}

TEST_P(AoclfftzSelectorTestFloatILP64, TEST_SELECTOR_CT_FLOAT_ILP64)
{
    run_selector_test_and_verify_solutions(SOLVER_CT);
}

TEST_P(AoclfftzSelectorTestDoubleILP64, TEST_SELECTOR_CT_DOUBLE_ILP64)
{
    run_selector_test_and_verify_solutions(SOLVER_CT);
}

INSTANTIATE_TEST_SUITE_P(
    SelectorCTParamTest, AoclfftzSelectorTestFloatLP64,
    ::testing::Combine(
        ::testing::ValuesIn(dims_and_vecs),
        ::testing::ValuesIn(flags),
        ::testing::ValuesIn(opt_levels),
        ::testing::Values(std::vector<aoclfftz_solver_type>()) // empty list
        ));

INSTANTIATE_TEST_SUITE_P(
    SelectorCTParamTest, AoclfftzSelectorTestDoubleLP64,
    ::testing::Combine(
        ::testing::ValuesIn(dims_and_vecs),
        ::testing::ValuesIn(flags),
        ::testing::ValuesIn(opt_levels),
        ::testing::Values(std::vector<aoclfftz_solver_type>()) // empty list
        ));

INSTANTIATE_TEST_SUITE_P(
    SelectorCTParamTest, AoclfftzSelectorTestFloatILP64,
    ::testing::Combine(
        ::testing::ValuesIn(dims_and_vecs),
        ::testing::ValuesIn(flags),
        ::testing::ValuesIn(opt_levels),
        ::testing::Values(std::vector<aoclfftz_solver_type>()) // empty list
        ));

INSTANTIATE_TEST_SUITE_P(
    SelectorCTParamTest, AoclfftzSelectorTestDoubleILP64,
    ::testing::Combine(
        ::testing::ValuesIn(dims_and_vecs),
        ::testing::ValuesIn(flags),
        ::testing::ValuesIn(opt_levels),
        ::testing::Values(std::vector<aoclfftz_solver_type>()) // empty list
        ));