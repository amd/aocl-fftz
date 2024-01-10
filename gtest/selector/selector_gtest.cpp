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
    // flag: 0b0000 => complex, forward, in-order, in-place problem
    {"2:1:1", 0b0000, -1, {SOLVER_DIRECT}},
    {"15:3:4", 0b0000, -1, {SOLVER_DIRECT}},
    {"7v12:3:2", 0b0000, -1, {SOLVER_DIRECT}},
    {"4:20:30v10:2:3", 0b0000, -1, {SOLVER_DIRECT}},
    {"10:5:5v5:1:1", 0b0000, -1, {SOLVER_DIRECT}},
    // 1 level CT; factors: 2, 9
    {"18:1:1", 0b0000, -1, {SOLVER_CT, SOLVER_DIRECT, SOLVER_DIRECT}},
    // 1 level CT; factors: 2, 9
    {"2v18:1:1", 0b0000, -1,
     {SOLVER_BATCHED, SOLVER_CT, SOLVER_DIRECT, SOLVER_DIRECT}},
    // 2 level CT; factors: 5, 7, 11
    {"385:1:1", 0b0000, -1,
     {SOLVER_CT, SOLVER_DIRECT, SOLVER_BATCHED, SOLVER_CT, SOLVER_DIRECT,
      SOLVER_DIRECT}},
    // 1 level CT; factors: 2, 9
    {"2v18:1:1", 0b0000, -1,
     {SOLVER_BATCHED, SOLVER_CT, SOLVER_DIRECT, SOLVER_DIRECT}},
    // bluestein: 19 -> 39 (factors: 3, 13)
    {"19:1:1", 0b0000, -1,
     {SOLVER_BLUESTEIN, SOLVER_CT, SOLVER_DIRECT, SOLVER_DIRECT}},
    // bluestein: 17 -> 33 (factors: 3, 11)
    {"5v17:1:1", 0b0000, -1,
     {SOLVER_BATCHED, SOLVER_BLUESTEIN, SOLVER_CT, SOLVER_DIRECT,
      SOLVER_DIRECT}},
    // 1 level CT; factors: 2, 19; bluestein: 19 -> 39 (factors: 3, 13)
    {"38:2:3", 0b0000, -1,
     {SOLVER_CT, SOLVER_DIRECT, SOLVER_BATCHED, SOLVER_BLUESTEIN, SOLVER_CT,
      SOLVER_DIRECT, SOLVER_DIRECT}},
    // 1 level CT; factors: 3, 17; bluestein: 19 -> 39 (factors: 3, 13)
    {"3v51:1:2", 0b0000, -1,
     {SOLVER_BATCHED, SOLVER_CT, SOLVER_DIRECT, SOLVER_BATCHED,
      SOLVER_BLUESTEIN, SOLVER_CT, SOLVER_DIRECT, SOLVER_DIRECT}},
    // The solution list for ND tramsform is in the format:
    // SOLVER_NDIM,{solution list for 1D sol},{solution list for nD sol}
    // multi dimesional (2D) transform;
    {"2x6", 0b0000, -1, {SOLVER_NDIM, SOLVER_DIRECT, SOLVER_DIRECT}},
    // multi dimesional (2D) transform composite problem;
    {"18x21", 0b0000, -1,
     {SOLVER_NDIM, SOLVER_BATCHED, SOLVER_CT, SOLVER_DIRECT, SOLVER_DIRECT,
      SOLVER_BATCHED, SOLVER_CT, SOLVER_DIRECT, SOLVER_DIRECT}},
    // multi dimesional batched ND(3D) strided transform;*/
    {"4v6x2:3:8x2", 0b0000, -1,
    {SOLVER_BATCHED, SOLVER_NDIM, SOLVER_BATCHED, SOLVER_DIRECT, SOLVER_BATCHED, SOLVER_NDIM, SOLVER_DIRECT, SOLVER_DIRECT}},
    // multi dimesional batched (3D) transform;
    {"2x6v2x3x4", 0b0000, -1,
     {SOLVER_BATCHED, SOLVER_NDIM, SOLVER_DIRECT, SOLVER_BATCHED, SOLVER_NDIM, SOLVER_DIRECT, SOLVER_DIRECT}}};

TEST_P(AoclfftzSelectorTestFloatLP64, TEST_SELECTOR_FLOAT_LP64)
{
    run_selector_test_and_compare_solver_list();
}

TEST_P(AoclfftzSelectorTestDoubleLP64, TEST_SELECTOR_DOUBLE_LP64)
{
    run_selector_test_and_compare_solver_list();
}

TEST_P(AoclfftzSelectorTestFloatILP64, TEST_SELECTOR_FLOAT_ILP64)
{
    run_selector_test_and_compare_solver_list();
}

TEST_P(AoclfftzSelectorTestDoubleILP64, TEST_SELECTOR_DOUBLE_ILP64)
{
    run_selector_test_and_compare_solver_list();
}

INSTANTIATE_TEST_SUITE_P(SelectorParamTest, AoclfftzSelectorTestFloatLP64,
                         ::testing::ValuesIn(selector_params));

INSTANTIATE_TEST_SUITE_P(SelectorParamTest, AoclfftzSelectorTestDoubleLP64,
                         ::testing::ValuesIn(selector_params));

INSTANTIATE_TEST_SUITE_P(SelectorParamTest, AoclfftzSelectorTestFloatILP64,
                         ::testing::ValuesIn(selector_params));

INSTANTIATE_TEST_SUITE_P(SelectorParamTest, AoclfftzSelectorTestDoubleILP64,
                         ::testing::ValuesIn(selector_params));