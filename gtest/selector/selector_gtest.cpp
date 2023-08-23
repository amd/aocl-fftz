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

/** @file selector_gtest.cpp
 *
 *  @brief File that contains the GTest based selector unit tests.
 *
 *  This file contains the entry point of the selector GTest.
 *
 *  @author Srirammaswamy Srinivasan
 */

#include <gtest/gtest.h>
#include "gtest/selector/selector_gtest_base.h"

aoclfftz_selector_test_params_t selector_params[] = {
    {"2:1:1", 0b0000, -1, {SOLVER_DIRECT}},
    {"15:3:4", 0b0000, -1, {SOLVER_DIRECT}}};

TEST_P(AoclfftzSelectorTestFloatLP64Parameterized, TEST_SELECTOR_FLOAT_LP64)
{
    run_selector_test();
}

TEST_P(AoclfftzSelectorTestDoubleLP64Parameterized, TEST_SELECTOR_DOUBLE_LP64)
{
    run_selector_test();
}

TEST_P(AoclfftzSelectorTestFloatILP64Parameterized, TEST_SELECTOR_FLOAT_ILP64)
{
    run_selector_test();
}

TEST_P(AoclfftzSelectorTestDoubleILP64Parameterized, TEST_SELECTOR_DOUBLE_ILP64)
{
    run_selector_test();
}

INSTANTIATE_TEST_CASE_P(SelectorParamTest,
                        AoclfftzSelectorTestFloatLP64Parameterized,
                        ::testing::ValuesIn(selector_params));

INSTANTIATE_TEST_CASE_P(SelectorParamTest,
                        AoclfftzSelectorTestDoubleLP64Parameterized,
                        ::testing::ValuesIn(selector_params));

INSTANTIATE_TEST_CASE_P(SelectorParamTest,
                        AoclfftzSelectorTestFloatILP64Parameterized,
                        ::testing::ValuesIn(selector_params));

INSTANTIATE_TEST_CASE_P(SelectorParamTest,
                        AoclfftzSelectorTestDoubleILP64Parameterized,
                        ::testing::ValuesIn(selector_params));