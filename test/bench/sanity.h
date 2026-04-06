// Copyright Advanced Micro Devices, Inc.
// SPDX-License-Identifier: BSD-3-Clause

/** @file sanity.h
 *
 *  @brief functions for sanity mode of test bench.
 *
 *  This file contains function declarations of sanity mode related
 *  functions for test bench.
 *
 *  @author Avinash Thakur
 */

#ifndef SANITY_H
#define SANITY_H

#include "test/bench/aoclfftz_bench.h"

INT32 run_bench_on_sanity_mode(aoclfftz_bench_params_t *params);

#endif // SANITY_H

