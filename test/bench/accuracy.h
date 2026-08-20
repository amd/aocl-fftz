// Copyright Advanced Micro Devices, Inc.
// SPDX-License-Identifier: BSD-3-Clause

/** @file accuracy.h
 *
 *  @brief Accuracy test utility functions.
 *
 *  This file contains accuracy test related utility function prototypes for
 *  test bench.
 *
 *  @author V. Murugan
 *  @author Srirammaswamy Srinivasan
 *  @author Jeya R
 */

#ifndef ACCURACY_H
#define ACCURACY_H

#include "test/bench/aoclfftz_bench.h"
#include "test/bench/utils/data_generation.h"
#include "test/bench/bench_problem.h"
#include "utils/allocator.h"

FFTZ_INT32 run_bench_on_accuracy_mode(aoclfftz_bench_params_t *params);
FFTZ_INT32 run_linearity_test(aoclfftz_bench_params_t *params,
                              FFTZ_INTP *in_idx_map, FFTZ_INTP *out_idx_map,
                              FFTZ_VOID * handle, FFTZ_VOID *input_buffer);
FFTZ_INT32 run_impulse_transform_test(aoclfftz_bench_params_t *params,
                                 FFTZ_INTP *in_idx_map, FFTZ_INTP *out_idx_map,
                                 FFTZ_VOID * handle, FFTZ_VOID *input_buffer);
FFTZ_INT32 run_timeshift_test(aoclfftz_bench_params_t *params,
                              FFTZ_INTP *in_idx_map, FFTZ_INTP *out_idx_map,
                              FFTZ_VOID * handle, FFTZ_VOID *input_buffer);
#endif // ACCURACY_H
