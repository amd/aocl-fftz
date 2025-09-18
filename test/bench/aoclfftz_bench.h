/**
 * Copyright (C) 2023-2025, Advanced Micro Devices. All rights reserved.
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

/** @file aoclfftz_bench.h
 *
 *  @brief Data structures and presets for testing single-threaded core fft
 *  library.
 *
 *  This file contains the data structures and presets for testing the
 *  library APIs.
 *
 *  @author S. Biplab Raut
 *  @author V. Murugan
 *  @author Srirammaswamy Srinivasan
 *  @author Jeya R
 */

#ifndef AOCLFFTZ_BENCH_H
#define AOCLFFTZ_BENCH_H

#define BENCH_2_PI 6.2831853071795864769252867665590057683943388
#define BENCH_2_PIf 6.2831853071795864769252867665590057683943388f

#include "api/aoclfftz.h"
#include "api/types.h"

// Forward declarations
typedef struct aoclfftz_bench_params aoclfftz_bench_params_t;
typedef struct aoclfftz_bench_error aoclfftz_bench_error_t;
typedef struct aoclfftz_bench_sz_info aoclfftz_bench_sz_info_t;

// Function pointer types
typedef VOID (*dft_ref_) (aoclfftz_bench_params_t *params, VOID *in, VOID *out,
                            INTP *in_idx_map, INTP *out_idx_map);
typedef VOID *(*setup_problem_) (aoclfftz_bench_params_t *params);
typedef VOID (*prepare_input_data_) (VOID *input, INTP n, INTP *idx_map,
                                     INT32 input_type, INT32 data_stride);
typedef INT32 (*compare_) (aoclfftz_bench_params_t *params, VOID *a, VOID *b,
                           INTP batches, INTP n, INTP *a_map, INTP *b_map,
                           INT32 data_stride);

// Enumerators for test bench
typedef enum
{
    C2C = 0,
    R2C,
    C2R
} aoclfftz_bench_fft_type_t;

typedef enum
{
    FLOAT_P = 0,
    DOUBLE_P
} aoclfftz_bench_precision_t;

typedef enum
{
    LP64 = 0,
    ILP64
} aoclfftz_bench_data_model_t;

typedef enum
{
    IN_ORDER = 0,
    OUT_OF_ORDER
} aoclfftz_bench_order_t;

typedef enum
{
    IN_PLACE = 0,
    OUT_OF_PLACE
} aoclfftz_bench_res_placement_t;

typedef enum
{
    PERFORMANCE = 0,
    ACCURACY,
    SANITY
} aoclfftz_bench_type_t;

typedef enum
{
    FORWARD = 0,
    BACKWARD
} aoclfftz_bench_direction_t;

typedef enum
{
    PARSER_SUCCESS = 0,
    PARSER_ERROR,
    SIZE_REQUIRED_ERROR,
    NON_OPTION_ARGUMENTS_ERROR,
    SIZE_PARSING_ERROR,
    UNSUPPORTED_OPTION_ERROR,
    INVALID_ARGUMENT_ERROR,
    HELP_MENU = 100
} aoclfftz_bench_parser_status_t;

typedef enum
{
    SETUP_FAILURE = -4,
    EXECUTION_FAILURE,
    VERIFICATION_FAILURE,
    MEMORY_FAILURE,
    BENCH_SUCCESS
} aoclfftz_bench_status_t;

typedef enum
{
    RANDOM_INPUT = 0,
    IMPULSE_INPUT,
    SINUSOIDAL_SIGNAL_INPUT
} aoclfftz_bench_input_type_t;

typedef struct aoclfftz_bench_sz_info
{
    INTP n; // product of all dims->n
    INTP n_in; // n for C2C problems; n/2+1 on first dim for C2R
    INTP n_out; // n for C2C problems; n/2+1 on first dim for R2C
    INTP batches;
    UINTP input_size;
    UINTP output_size;
    UINTP input_bytes;
    UINTP output_bytes;
    INT32 in_data_stride; // 1 for C2C, 1 for R2C, 2 for C2R
    INT32 out_data_stride; // 1 for C2C, 2 for R2C, 1 for C2R
    INT32 dt_bytes; // sizeof(FLOAT) or sizeof(DOUBLE)
} aoclfftz_bench_sz_info_t;

// Structures for test bench
typedef struct aoclfftz_bench_params
{
    VOID *in;
    VOID *out;
    aoclfftz_bench_type_t bench_type;
    aoclfftz_bench_precision_t precision;
    aoclfftz_bench_data_model_t data_model;
    aoclfftz_bench_fft_type_t fft_type;
    aoclfftz_bench_order_t order;
    aoclfftz_bench_res_placement_t res_placement;
    aoclfftz_bench_direction_t dir;
    INT32 dim_rank;
    INT32 vec_rank;
    aoclfftz_dim_t_64_ *dims;
    aoclfftz_dim_t_64_ *vecs;
    INT32 num_iterations;
    INT32 seed;
    UCHAR use_random_seed;
    INT32 opt_level;
    DOUBLE tolerance;
    INT32 logger_mode;
    UINT32 num_threads;
    UINT32 dynamic_load_model;
    UCHAR selector_time;
    DOUBLE min_bench_time;
    INT32 measure_stats;
    INT32 bit_reproducibility;
    UINT32 aligned_alloc;
    aoclfftz_bench_sz_info_t sz_info;
    dft_ref_ dft_ref;
    setup_problem_ setup_problem;
    prepare_input_data_ prepare_input_data;
    compare_ compare;
} aoclfftz_bench_params_t;

typedef struct aoclfftz_bench_error
{
    DOUBLE max_abs_err;
    DOUBLE max_mag;
    INTP *max_err_coords;
    INTP *first_err_coords;
} aoclfftz_bench_error_t;

#endif // AOCLFFTZ_BENCH_H
