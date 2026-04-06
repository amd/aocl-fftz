// Copyright Advanced Micro Devices, Inc.
// SPDX-License-Identifier: BSD-3-Clause

/** @file register_functions.h
 *
 *  @brief Register functions to function pointers.
 *
 *  This file contains a register function which registers the appropriate
 *  functions to function pointers in aoclfftz_bench_params_t object.
 *
 *  @author V. Murugan
 *  @author Srirammaswamy Srinivasan
 *  @author Jeya R
 */

#ifndef REGISTER_FUNCTIONS_H
#define REGISTER_FUNCTIONS_H

#include "test/bench/utils/compare.h"
#include "test/bench/utils/data_generation.h"
#include "test/bench/dft_reference.h"

INT32 register_functions(aoclfftz_bench_params_t *params);

#endif // REGISTER_FUNCTIONS_H
