// Copyright Advanced Micro Devices, Inc.
// SPDX-License-Identifier: BSD-3-Clause

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
typedef FFTZ_VOID (*dft_ref_) (aoclfftz_bench_params_t *params, FFTZ_VOID *in,
                               FFTZ_VOID *out, FFTZ_INTP *in_idx_map,
                               FFTZ_INTP *out_idx_map);
typedef FFTZ_VOID *(*setup_problem_) (aoclfftz_bench_params_t *params);
typedef FFTZ_VOID (*prepare_input_data_) (FFTZ_VOID *input, FFTZ_INTP n,
                                          FFTZ_INTP *idx_map,
                                          FFTZ_INT32 input_type,
                                          FFTZ_INT32 data_stride);
typedef FFTZ_INT32 (*compare_) (aoclfftz_bench_params_t *params, FFTZ_VOID *a,
                                FFTZ_VOID *b, FFTZ_INTP batches, FFTZ_INTP n,
                                FFTZ_INTP *a_map, FFTZ_INTP *b_map,
                                FFTZ_INT32 data_stride);

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
    BENCH_SUCCESS = 0,
} aoclfftz_bench_status_t;

typedef enum
{
    RANDOM_INPUT = 0,
    IMPULSE_INPUT,
    SINUSOIDAL_SIGNAL_INPUT
} aoclfftz_bench_input_type_t;

typedef struct aoclfftz_bench_sz_info
{
    FFTZ_INTP n; // product of all dims->n
    FFTZ_INTP n_in; // n for C2C problems; n/2+1 on first dim for C2R
    FFTZ_INTP n_out; // n for C2C problems; n/2+1 on first dim for R2C
    FFTZ_INTP batches;
    FFTZ_UINTP input_size;
    FFTZ_UINTP output_size;
    FFTZ_UINTP input_bytes;
    FFTZ_UINTP output_bytes;
    FFTZ_INT32 in_data_stride; // 1 for C2C, 1 for R2C, 2 for C2R
    FFTZ_INT32 out_data_stride; // 1 for C2C, 2 for R2C, 1 for C2R
    FFTZ_INT32 dt_bytes; // sizeof(FFTZ_FLOAT) or sizeof(FFTZ_DOUBLE)
} aoclfftz_bench_sz_info_t;

// Structures for test bench
typedef struct aoclfftz_bench_params
{
    FFTZ_VOID *in;
    FFTZ_VOID *out;
    aoclfftz_bench_type_t bench_type;
    aoclfftz_bench_precision_t precision;
    aoclfftz_bench_data_model_t data_model;
    aoclfftz_bench_fft_type_t fft_type;
    aoclfftz_bench_order_t order;
    aoclfftz_bench_res_placement_t res_placement;
    aoclfftz_bench_direction_t dir;
    FFTZ_INT32 dim_rank;
    FFTZ_INT32 vec_rank;
    aoclfftz_dim_t_64_ *dims;
    aoclfftz_dim_t_64_ *vecs;
    FFTZ_INT32 num_iterations;
    FFTZ_INT32 seed;
    FFTZ_UCHAR use_random_seed;
    FFTZ_INT32 opt_level;
    FFTZ_DOUBLE tolerance;
    FFTZ_INT32 logger_mode;
    FFTZ_INT32 num_threads;
    FFTZ_UINT32 dynamic_load_model;
    FFTZ_UCHAR selector_time;
    FFTZ_DOUBLE min_bench_time;
    FFTZ_INT32 measure_stats;
    FFTZ_INT32 bit_reproducibility;
    FFTZ_UINT32 aligned_alloc;
    aoclfftz_bench_sz_info_t sz_info;
    dft_ref_ dft_ref;
    setup_problem_ setup_problem;
    prepare_input_data_ prepare_input_data;
    compare_ compare;
} aoclfftz_bench_params_t;

typedef struct aoclfftz_bench_error
{
    FFTZ_DOUBLE max_abs_err;
    FFTZ_DOUBLE max_mag;
    FFTZ_INTP *max_err_coords;
    FFTZ_INTP *first_err_coords;
} aoclfftz_bench_error_t;

#endif // AOCLFFTZ_BENCH_H
