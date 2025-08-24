/**
 * Copyright (C) 2024, Advanced Micro Devices. All rights reserved.
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

/** @file bench_problem.c
 *
 *  @brief Test bench problem descriptor functions.
 *
 *  This file contains the functions related to test bench params and
 *  problem descriptor.
 *
 *  @author V. Murugan
 *  @author Srirammaswamy Srinivasan
 *  @author Jeya R
 */

#include "test/bench/bench_problem.h"
#include "test/bench/aoclfftz_bench.h"

/**
 * @brief init bench params with default values
 *
 * @param bench_params aoclfftz_bench_params_t struct to store the values
 * @return VOID
 */
VOID init_bench_params(aoclfftz_bench_params_t *bench_params)
{
    bench_params->in = NULL;
    bench_params->out = NULL;
    bench_params->bench_type = PERFORMANCE;
    bench_params->precision = DOUBLE_P;
    bench_params->data_model = LP64;
    bench_params->fft_type = C2C;
    bench_params->order = IN_ORDER;
    bench_params->res_placement = OUT_OF_PLACE;
    bench_params->dir = FORWARD;
    bench_params->dim_rank = 0;
    bench_params->vec_rank = 0;
    bench_params->dims = NULL;
    bench_params->vecs = NULL;
    bench_params->num_iterations = 10;
    bench_params->seed = 0;
    bench_params->use_random_seed = 1;
    bench_params->opt_level = -1;
    bench_params->tolerance = 1E-10;
    bench_params->logger_mode = 0;
    bench_params->num_threads = 1;
    bench_params->dynamic_load_model = 0;
    bench_params->selector_time = 0;
    bench_params->min_bench_time = 100; // 100 ms
    bench_params->measure_stats = 0;
    bench_params->bit_reproducibility = 0;
    bench_params->aligned_alloc = 1;
}

/**
 * @brief prepare the bench params from the command line arguments
 *
 * @param argc command-line argument count
 * @param argv command-line argument value as a char array
 * @param bench_params aoclfftz_bench_params_t struct to store the values
 * @return INT32 aoclfftz_bench_parser_status_t type
 */
INT32 prepare_bench_params(INT32 argc, CHAR **argv,
                           aoclfftz_bench_params_t *bench_params)
{
    init_bench_params(bench_params);
    INT32 c = -1;

    UCHAR use_cust_tolerance = 0;
    // setting the data strides based on complex type and will be modified
    // later for real problems
    bench_params->sz_info.in_data_stride = 2;
    bench_params->sz_info.out_data_stride = 2;

    INT32 status = PARSER_SUCCESS;
    INT32 ret = PARSER_SUCCESS;
    // check for the dependent arguments
    UCHAR valid_iters_arg_found = 0;

    CHAR *str_buff = NULL;
    ALLOC_ALIGN_UNINIT(str_buff, CHAR, sizeof(CHAR) * 50);
    CHAR NULL_CHAR = '\0', *optarg = &NULL_CHAR;

    INT32 arg_idx = 1;
    INT32 non_opt_arg_cnt = 0;
    while (arg_idx < argc)
    {
        c = get_option(argv, arg_idx);
        // Check if there is at least one more command-line argument available
        // before accessing it.
        if (arg_idx + 1 < argc)
        {
            optarg = argv[arg_idx + 1];
        }
        switch (c)
        {
        case 'h':
            FREE_ALIGN_ALLOCATED_MEM(str_buff);
            return HELP_MENU;
        case 'p':
            if (!strcmp(optarg, "f"))
            {
                bench_params->precision = FLOAT_P;
            }
            else if (!strcmp(optarg, "d"))
            {
                bench_params->precision = DOUBLE_P;
            }
            else
            {
                printf("ERROR: Unknown precision\n");
                status = MAX(status, UNSUPPORTED_OPTION_ERROR);
            }
            break;
        case 'm':
            if (!strcmp(optarg, "l"))
            {
                bench_params->data_model = LP64;
            }
            else if (!strcmp(optarg, "i"))
            {
                bench_params->data_model = ILP64;
            }
            else
            {
                printf("ERROR: Unknown data model\n");
                status = MAX(status, UNSUPPORTED_OPTION_ERROR);
            }
            break;
        case 'b':
            if (!strcmp(optarg, "a"))
            {
                bench_params->bench_type = ACCURACY;
            }
            else if (!strcmp(optarg, "p"))
            {
                bench_params->bench_type = PERFORMANCE;
            }
            else
            {
                printf("ERROR: Unknown benchmark type\n");
                status = MAX(status, UNSUPPORTED_OPTION_ERROR);
            }
            break;
        case 'r':
            if (!strcmp(optarg, "i"))
            {
                bench_params->res_placement = IN_PLACE;
            }
            else if (!strcmp(optarg, "o"))
            {
                bench_params->res_placement = OUT_OF_PLACE;
            }
            else
            {
                printf("ERROR: Unknown result placement\n");
                status = MAX(status, UNSUPPORTED_OPTION_ERROR);
            }
            break;
        case 300:
            if (!strcmp(optarg, "o"))
            {
                bench_params->order = OUT_OF_ORDER;
            }
            else if (!strcmp(optarg, "i"))
            {
                bench_params->order = IN_ORDER;
            }
            else
            {
                printf("ERROR: Unknown output order\n");
                status = MAX(status, UNSUPPORTED_OPTION_ERROR);
            }
            break;
        case 301:
            if (!strcmp(optarg, "b"))
            {
                bench_params->dir = BACKWARD;
            }
            else if (!strcmp(optarg, "f"))
            {
                bench_params->dir = FORWARD;
            }
            else
            {
                printf("ERROR: Unknown FFT direction\n");
                status = MAX(status, UNSUPPORTED_OPTION_ERROR);
            }
            break;
        case 'f':
            if (!strcmp(optarg, "c2r"))
            {
                bench_params->fft_type = C2R;
            }
            else if (!strcmp(optarg, "r2c"))
            {
                bench_params->fft_type = R2C;
            }
            else if (!strcmp(optarg, "c2c"))
            {
                bench_params->fft_type = C2C;
            }
            else
            {
                printf("ERROR: Unknown fft type\n");
                status = MAX(status, UNSUPPORTED_OPTION_ERROR);
            }
            break;
        case 'i':
            valid_iters_arg_found = 1;
            VALIDATE_AND_GET_INT(optarg, str_buff, bench_params->num_iterations,
                                 ret, 0);
            if (ret != 0)
            {
                printf("WARNING: Invalid iterations value given, "
                       "iterations should be a positive integer, running "
                       "bench with default value\n");
                valid_iters_arg_found = 0;
            }
            break;
        case 's':
            VALIDATE_AND_GET_INT(optarg, str_buff, bench_params->seed, ret, 0);
            if (ret != 0)
            {
                printf("WARNING: Invalid seed value given, running bench with "
                       "random seed\n");
            }
            else
            {
                bench_params->use_random_seed = 0;
            }
            break;
        case 't':
            VALIDATE_AND_GET_DOUBLE(optarg, str_buff, bench_params->tolerance,
                                    ret, 0.0, 1.0);
            if (ret != 0)
            {
                printf("WARNING: Invalid custom tolerance value given, running "
                       "bench with default tolerance value\n");
            }
            else
            {
                use_cust_tolerance = 1;
            }
            break;
        case 'n':
            VALIDATE_AND_GET_INT(optarg, str_buff, bench_params->num_threads,
                                 ret, 1);
            if (ret != 0)
            {
                printf("WARNING: Invalid num threads value given, "
                       "running bench with default threads (1)\n");
            }
#ifndef MULTI_THREADING
            if (bench_params->num_threads > 1)
            {
                printf(
                    "WARNING: Multi-threaded FFT is not enabled, "
                    "running bench with single-threaded FFT\n");
                bench_params->num_threads = 1;
            }
#endif
            break;
        case 'o':
            VALIDATE_AND_GET_INT(optarg, str_buff, bench_params->opt_level, ret,
                                 -1);
            break;
        case 'l':
            VALIDATE_AND_GET_INT(optarg, str_buff, bench_params->logger_mode,
                                 ret, 0);
            break;
        case 302:
            if (atoi(optarg) != 0)
            {
                if (atoi(optarg) != 1)
                {
                    printf("WARNING: The provided value for dynamic_load_model "
                           "is not 1. Running the bench with dynamic_load_model"
                           "set to 1.\n");
                }
                bench_params->dynamic_load_model = 1;
            }

#ifndef MULTI_THREADING
            if (bench_params->dynamic_load_model != 0)
            {
                printf(
                    "WARNING: Multi-threaded FFT is not enabled, "
                    "running bench with single-threaded FFT\n");
                bench_params->dynamic_load_model = 0;
            }
#endif
            break;
        case 303:
            if (atoi(optarg) != 0)
            {
                if (atoi(optarg) != 1)
                {
                    printf("WARNING: The provided value for selector_time is "
                           "not 1. Running the bench with selector_time set to "
                           "1.\n");
                }
                bench_params->selector_time = 1;
            }
            break;
        case 304:
            // TODO: Modify this after adding support for measure-stats
            printf("WARNING: measure-stats option is currently not supported, "
                   "running bench with measure-stats disabled\n");
            break;
        case 305:
            // TODO: Modify this after adding support for bit-reproducibility
            printf("WARNING: bit-reproducibility option is currently not "
                   "supported, running bench without bit-reproducibility\n");
            break;
        case 306:
            if (atof(optarg) < 100)
            {
                printf("WARNING: min_bench_time value must be at least 100 ms, "
                       "running bench with default value(100 ms)\n");
                bench_params->min_bench_time = 100;
            }
            else
            {
                bench_params->min_bench_time = atof(optarg);
            }
            break;
        case 307:
            if (atoi(optarg) == 0)
            {
                bench_params->aligned_alloc = 0;
            }
            else if (atoi(optarg) == 1)
            {
                bench_params->aligned_alloc = 1;
            }
            else
            {
                printf("WARNING: Unknown value provided for aligned memory "
                       "allocation, defaulting to 1\n");
                bench_params->aligned_alloc = 1;
            }
            break;
        case 308:
            if (non_opt_arg_cnt == 0)
            {
                // Parse dimension sizes and vector sizes
                ret = find_dim_vec_ranks(argv[arg_idx], &bench_params->dim_rank,
                                         &bench_params->vec_rank);
                if (ret != PARSER_SUCCESS)
                {
                    status = MAX(status, SIZE_PARSING_ERROR);
                }
                else
                {
                    ret = allocate_and_fill_dims_vecs(
                        argv[arg_idx], bench_params->dim_rank,
                        bench_params->vec_rank, &bench_params->dims,
                        &bench_params->vecs, 1);
                    if (ret != PARSER_SUCCESS)
                    {
                        status = MAX(status, ret);
                    }
                }
                non_opt_arg_cnt++;
            }
            else
            {
                // Only one non-option argument should be provided
                // (which is problem size)
                status = MAX(status, NON_OPTION_ARGUMENTS_ERROR);
            }
            // should not move arg_idx by 2 places if it is a non option arg
            arg_idx--;
            break;
        case '?':
            status = MAX(status, INVALID_ARGUMENT_ERROR);
            break;
        default:
            status = MAX(status, INVALID_ARGUMENT_ERROR);
        }
        arg_idx += 2;
    }

    FREE_ALIGN_ALLOCATED_MEM(str_buff);

    // Problem size argument must be present
    if (non_opt_arg_cnt == 0)
    {
        status = MAX(status, SIZE_REQUIRED_ERROR);
    }
    else if (bench_params->dim_rank == 0 || bench_params->vec_rank == 0 ||
             bench_params->dims == NULL || bench_params->vecs == NULL)
    {
        status = MAX(status, SIZE_PARSING_ERROR);
    }
    // TODO: Support ND batched real problems
    else if (bench_params->vec_rank > 1 && bench_params->fft_type != C2C)
    {
        printf("ERROR: ND batched real problems are not supported");
        status = MAX(status, UNSUPPORTED_OPTION_ERROR);
    }
    // TODO: Support ND real problems
    else if (bench_params->dim_rank > 1 && bench_params->fft_type != C2C)
    {
        printf("ERROR: ND real problems are not supported");
        status = MAX(status, UNSUPPORTED_OPTION_ERROR);
    }

    if (status != PARSER_SUCCESS)
    {
        return status;
    }

    if (valid_iters_arg_found == 0)
    {
        if (bench_params->bench_type == ACCURACY)
        {
            bench_params->num_iterations = 1;
        }
        else // bench_type == PERFORMANCE
        {
            bench_params->num_iterations = 10;
        }
    }
    if (bench_params->selector_time != 0 &&
        bench_params->bench_type == ACCURACY)
    {
        printf("WARNING: selector-time won't be used in ACCURACY mode\n");
        bench_params->selector_time = 0;
    }
    if (!bench_params->use_random_seed && bench_params->num_iterations != 1 &&
        bench_params->bench_type == ACCURACY)
    {
        printf("WARNING: iterations will set to 1 since manual seed value is "
               "provided\n");
        bench_params->num_iterations = 1;
    }

    // NOTE: 'r2c/c2r backward' modes are internally converted to 'c2r/r2c
    // forward' modes
    // i.e. r2c backward  -> c2r forward or just 'c2r'
    //      c2r backward  -> r2c forward or just 'r2c'
    if (bench_params->fft_type == R2C && bench_params->dir == BACKWARD)
    {
        bench_params->fft_type = C2R;
        bench_params->dir = FORWARD;
    }
    else if (bench_params->fft_type == C2R && bench_params->dir == BACKWARD)
    {
        bench_params->fft_type = R2C;
        bench_params->dir = FORWARD;
    }

    // set the data strides for an 'r2c' problem
    if (bench_params->fft_type == R2C)
    {
        bench_params->sz_info.in_data_stride = 1;
        bench_params->sz_info.out_data_stride = 2;
    }
    else if (bench_params->fft_type == C2R)
    {
        bench_params->sz_info.in_data_stride = 2;
        bench_params->sz_info.out_data_stride = 1;
    }

    // change the min_bench_time unit from ms to ns
    bench_params->min_bench_time *= 1e6;
    if (!use_cust_tolerance)
    {
        if (bench_params->precision == FLOAT_P)
        {
            bench_params->tolerance = 1E-3;
        }
        else
        {
            bench_params->tolerance = 1E-10;
        }
    }

    // set default dims & vecs values
    set_default_dims_vecs(bench_params->dim_rank, bench_params->vec_rank,
                          bench_params->dims, bench_params->vecs,
                          bench_params->fft_type,
                          bench_params->res_placement == IN_PLACE,
                          bench_params->logger_mode);

    bench_params->sz_info.dt_bytes =
        (bench_params->precision == FLOAT_P) ? sizeof(FLOAT) : sizeof(DOUBLE);

    // get size info
    UINTP in_buffer_size = 0;
    UINTP out_buffer_size = 0;
    calculate_buffer_sizes(bench_params->dim_rank, bench_params->vec_rank,
                           bench_params->dims, bench_params->vecs,
                           &in_buffer_size, &out_buffer_size);
    bench_params->sz_info.input_size = in_buffer_size;
    bench_params->sz_info.output_size = out_buffer_size;

    bench_params->sz_info.input_bytes = in_buffer_size *
                                        bench_params->sz_info.in_data_stride *
                                        bench_params->sz_info.dt_bytes;
    bench_params->sz_info.output_bytes = out_buffer_size *
                                         bench_params->sz_info.out_data_stride *
                                         bench_params->sz_info.dt_bytes;

    // create input and output buffers
    UINT32 is_align = bench_params->aligned_alloc;
    INTP input_bytes = bench_params->sz_info.input_bytes;
    INTP output_bytes = bench_params->sz_info.output_bytes;
    // TODO: Check and remove this MAX logic if possible
    // In an R2C problem, input will have N real points and output will have N
    // complex points.
    // For in-place R2C problem, since the same buffer will be used for input
    // and output, it should be large enough to hold the N complex points.
    if (bench_params->fft_type != C2C &&
        bench_params->res_placement == IN_PLACE)
    {
        input_bytes = MAX(input_bytes, output_bytes);
    }
    ALLOC_UNINIT(bench_params->in, VOID, input_bytes, is_align);

    // use input buffer as output for in-place problems and create new output
    // buffer for out-of-place problems
    if (bench_params->res_placement == IN_PLACE)
    {
        bench_params->out = bench_params->in;
    }
    else
    {
        ALLOC_INIT(bench_params->out, VOID, output_bytes, is_align);
    }

#ifdef AOCL_ENABLE_LOG
    AOCLFFTZ_LOG_UNFORMATTED(TRACE, bench_params->logger_mode,
                             "Bench params prepared from parsing arguments");
#endif

    ret = register_functions(bench_params);
    status = MAX(status, ret);
    return status;
}

/**
 * @brief Setup FFT problem of FLOAT LP64 type
 *
 * @param params aoclfftz_bench_params_t type contains parsed arguments
 * @return VOID* handle object
 */
VOID *setup_problem_f(aoclfftz_bench_params_t *params)
{
#ifdef AOCL_ENABLE_LOG
    AOCLFFTZ_LOG_UNFORMATTED(TRACE, params->logger_mode, "ENTER");
#endif
#ifdef WIN32
    timer clk_tick;
#endif
    timeVal start_time, end_time;
    DOUBLE time_taken = 0.0;
    UINT32 is_align = params->aligned_alloc;

    aoclfftz_prob_desc_f *p_desc = NULL;
    ALLOC_UNINIT(p_desc, aoclfftz_prob_desc_f, sizeof(aoclfftz_prob_desc_f),
                 is_align);
    INIT_PD(params, p_desc, INT32, aoclfftz_dim_t);

    p_desc->in = (FLOAT *)params->in;
    p_desc->out = (FLOAT *)params->out;

    // call setup
    VOID *handle;
    if (params->selector_time != 0)
    {
        initTimer(clk_tick);
        getTime(start_time);
        handle = aoclfftz_setup_f(p_desc);
        getTime(end_time);
        time_taken = (DOUBLE)diffTime(clk_tick, start_time, end_time);
        if (handle != NULL)
        {
            CHAR time_unit[3];
            ADJUST_SELECTOR_TIME_UNIT(time_taken, time_unit);
            printf("\n=====================================\n");
            printf("      Selector time : %6.3lf %s\n", time_taken, time_unit);
            printf("=====================================\n");
        }
    }
    else
    {
        handle = aoclfftz_setup_f(p_desc);
    }
    DESTROY_PD(p_desc, is_align);
#ifdef AOCL_ENABLE_LOG
    AOCLFFTZ_LOG_UNFORMATTED(TRACE, params->logger_mode, "EXIT");
#endif
    return handle;
}

/**
 * @brief Setup FFT problem of DOUBLE LP64 type
 *
 * @param params aoclfftz_bench_params_t type contains parsed arguments
 * @return VOID* handle object
 */
VOID *setup_problem_d(aoclfftz_bench_params_t *params)
{
#ifdef AOCL_ENABLE_LOG
    AOCLFFTZ_LOG_UNFORMATTED(TRACE, params->logger_mode, "ENTER");
#endif
#ifdef WIN32
    timer clk_tick;
#endif
    timeVal start_time, end_time;
    DOUBLE time_taken = 0.0;
    UINT32 is_align = params->aligned_alloc;

    aoclfftz_prob_desc_d *p_desc = NULL;
    ALLOC_UNINIT(p_desc, aoclfftz_prob_desc_d, sizeof(aoclfftz_prob_desc_d),
                 is_align);
    INIT_PD(params, p_desc, INT32, aoclfftz_dim_t);

    p_desc->in = (DOUBLE *)params->in;
    p_desc->out = (DOUBLE *)params->out;

    // call setup
    VOID *handle;
    if (params->selector_time != 0)
    {
        initTimer(clk_tick);
        getTime(start_time);
        handle = aoclfftz_setup_d(p_desc);
        getTime(end_time);
        time_taken = (DOUBLE)diffTime(clk_tick, start_time, end_time);
        if (handle != NULL)
        {
            CHAR time_unit[3];
            ADJUST_SELECTOR_TIME_UNIT(time_taken, time_unit);
            printf("\n=====================================\n");
            printf("      Selector time : %6.3lf %s\n", time_taken, time_unit);
            printf("=====================================\n");
        }
    }
    else
    {
        handle = aoclfftz_setup_d(p_desc);
    }
    DESTROY_PD(p_desc, is_align);
#ifdef AOCL_ENABLE_LOG
    AOCLFFTZ_LOG_UNFORMATTED(TRACE, params->logger_mode, "EXIT");
#endif
    return handle;
}

/**
 * @brief Setup FFT problem of FLOAT ILP64 type
 *
 * @param params aoclfftz_bench_params_t type contains parsed arguments
 * @return VOID* handle object
 */
VOID *setup_problem_f_64_(aoclfftz_bench_params_t *params)
{
#ifdef AOCL_ENABLE_LOG
    AOCLFFTZ_LOG_UNFORMATTED(TRACE, params->logger_mode, "ENTER");
#endif
#ifdef WIN32
    timer clk_tick;
#endif
    timeVal start_time, end_time;
    DOUBLE time_taken = 0.0;
    UINT32 is_align = params->aligned_alloc;

    aoclfftz_prob_desc_f_64_ *p_desc = NULL;
    ALLOC_UNINIT(p_desc, aoclfftz_prob_desc_f_64_,
                 sizeof(aoclfftz_prob_desc_f_64_), is_align);
    INIT_PD(params, p_desc, INTP, aoclfftz_dim_t_64_);

    p_desc->in = (FLOAT *)params->in;
    p_desc->out = (FLOAT *)params->out;

    // call setup
    VOID *handle;
    if (params->selector_time != 0)
    {
        initTimer(clk_tick);
        getTime(start_time);
        handle = aoclfftz_setup_f_64_(p_desc);
        getTime(end_time);
        time_taken = (DOUBLE)diffTime(clk_tick, start_time, end_time);
        if (handle != NULL)
        {
            CHAR time_unit[3];
            ADJUST_SELECTOR_TIME_UNIT(time_taken, time_unit);
            printf("\n=====================================\n");
            printf("      Selector time : %6.3lf %s\n", time_taken, time_unit);
            printf("=====================================\n");
        }
    }
    else
    {
        handle = aoclfftz_setup_f_64_(p_desc);
    }
    DESTROY_PD(p_desc, is_align);
#ifdef AOCL_ENABLE_LOG
    AOCLFFTZ_LOG_UNFORMATTED(TRACE, params->logger_mode, "EXIT");
#endif
    return handle;
}

/**
 * @brief Setup FFT problem of DOUBLE ILP64 type
 *
 * @param params aoclfftz_bench_params_t type contains parsed arguments
 * @return VOID* handle object
 */
VOID *setup_problem_d_64_(aoclfftz_bench_params_t *params)
{
#ifdef AOCL_ENABLE_LOG
    AOCLFFTZ_LOG_UNFORMATTED(TRACE, params->logger_mode, "ENTER");
#endif
#ifdef WIN32
    timer clk_tick;
#endif
    timeVal start_time, end_time;
    DOUBLE time_taken = 0.0;
    UINT32 is_align = params->aligned_alloc;

    aoclfftz_prob_desc_d_64_ *p_desc = NULL;
    ALLOC_UNINIT(p_desc, aoclfftz_prob_desc_d_64_,
                 sizeof(aoclfftz_prob_desc_d_64_), params->aligned_alloc);
    INIT_PD(params, p_desc, INTP, aoclfftz_dim_t_64_);

    p_desc->in = (DOUBLE *)params->in;
    p_desc->out = (DOUBLE *)params->out;

    // call setup
    VOID *handle;
    if (params->selector_time != 0)
    {
        initTimer(clk_tick);
        getTime(start_time);
        handle = aoclfftz_setup_d_64_(p_desc);
        getTime(end_time);
        time_taken = (DOUBLE)diffTime(clk_tick, start_time, end_time);
        if (handle != NULL)
        {
            CHAR time_unit[3];
            ADJUST_SELECTOR_TIME_UNIT(time_taken, time_unit);
            printf("\n=====================================\n");
            printf("      Selector time : %6.3lf %s\n", time_taken, time_unit);
            printf("=====================================\n");
        }
    }
    else
    {
        handle = aoclfftz_setup_d_64_(p_desc);
    }
    DESTROY_PD(p_desc, is_align);
#ifdef AOCL_ENABLE_LOG
    AOCLFFTZ_LOG_UNFORMATTED(TRACE, params->logger_mode, "EXIT");
#endif
    return handle;
}

/**
 * @brief Free the structures used for FFT problem
 *
 * @param params aoclfftz_bench_params_t type contains parsed arguments
 * @return VOID
 */
VOID destroy_bench_param(aoclfftz_bench_params_t *params)
{
    if (params != NULL)
    {
        UINT32 is_align = params->aligned_alloc;
        FREE_ALLOCATED_MEM(params->in, is_align);
        if (params->res_placement == OUT_OF_PLACE)
        {
            FREE_ALLOCATED_MEM(params->out, is_align);
        }
        FREE_ALIGN_ALLOCATED_MEM(params->dims);
        FREE_ALIGN_ALLOCATED_MEM(params->vecs);
        FREE_ALIGN_ALLOCATED_MEM(params);
    }
}

INT32 get_option(CHAR **argv, INT32 arg_idx)
{
    CHAR *arg = argv[arg_idx];
    if (arg[0] == '-')
    {
        if (strcmp(arg, "--help") == 0 || arg[1] == 'h')
        {
            return 'h';
        }
        else if (strcmp(arg, "--precision") == 0 || arg[1] == 'p')
        {
            return 'p';
        }
        else if (strcmp(arg, "--data-model") == 0 || arg[1] == 'm')
        {
            return 'm';
        }
        else if (strcmp(arg, "--bench-type") == 0 || arg[1] == 'b')
        {
            return 'b';
        }
        else if (strcmp(arg, "--result-placement") == 0 || arg[1] == 'r')
        {
            return 'r';
        }
        else if (strcmp(arg, "--order") == 0)
        {
            return 300;
        }
        else if (strcmp(arg, "--dir") == 0)
        {
            return 301;
        }
        else if (strcmp(arg, "--fft-type") == 0 || arg[1] == 'f')
        {
            return 'f';
        }
        else if (strcmp(arg, "--iters") == 0 || arg[1] == 'i')
        {
            return 'i';
        }
        else if (strcmp(arg, "--seed") == 0 || arg[1] == 's')
        {
            return 's';
        }
        else if (strcmp(arg, "--tol") == 0 || arg[1] == 't')
        {
            return 't';
        }
        else if (strcmp(arg, "--num-threads") == 0 || arg[1] == 'n')
        {
            return 'n';
        }
        else if (strcmp(arg, "--dynamic-load-model") == 0)
        {
            return 302;
        }
        else if (strcmp(arg, "--opt-level") == 0 || arg[1] == 'o')
        {
            return 'o';
        }
        else if (strcmp(arg, "--logger-mode") == 0 || arg[1] == 'l')
        {
            return 'l';
        }
        else if (strcmp(arg, "--selector-time") == 0)
        {
            return 303;
        }
        else if (strcmp(arg, "--measure-stats") == 0)
        {
            return 304;
        }
        else if (strcmp(arg, "--bit-reproducibility") == 0)
        {
            return 305;
        }
        else if (strcmp(arg, "--min-bench-time") == 0)
        {
            return 306;
        }
        else if (strcmp(arg, "--aligned-alloc") == 0)
        {
            return 307;
        }
        else
        {
            // Unsupported option
            return '?';
        }
    }
    else
    {
        // Non option argument
        return 308;
    }
}

/**
 * @brief set the flag value based on bench params
 *
 * @param params bench params
 * @return INT32 encoded flag value
 */
UINT32 set_flag(aoclfftz_bench_params_t *params)
{
    UINT32 flag = 0;
    if (params->res_placement == IN_PLACE)
    {
        flag &= ~(1 << 0); // set 0th bit to 0
    }
    else
    {
        flag |= (1 << 0);  // set 0th bit to 1
    }
    if (params->order == IN_ORDER)
    {
        flag &= ~(1 << 1); // set 1st bit to 0
    }
    else
    {
        flag |= (1 << 1);  // set 1st bit to 1
    }
    if (params->fft_type == C2C)
    {
        flag &= ~(1 << 3); // set 3rd bit to 0
        if (params->dir == FORWARD)
        {
            flag &= ~(1 << 2); // set 2nd bit to 0
        }
        else
        {
            flag |= (1 << 2);  // set 2nd bit to 1
        }
    }
    else if (params->fft_type == R2C)
    {
        flag &= ~(1 << 2); // set 2nd bit to 0
        flag |= (1 << 3);  // set 3rd bit to 1
    }
    else // C2R
    {
        flag |= (1 << 2);  // set 2nd bit to 1
        flag |= (1 << 3);  // set 3rd bit to 1
    }
    return flag;
}

/**
 * @brief print the help menu contents to the output
 *
 * @return VOID
 */
VOID show_help_menu(VOID)
{
    printf(
        "\nUSAGE: aocl_fftz_bench [OPTIONS]... PROBLEM_SIZE\n\n"
        "PROBLEM_SIZE    for 1D problem :\n"
        "                  10 => one dimensional problem of size 10\n"
        "                  10:2:4 => one dimensional problem of size 10, input "
        "stride 2 and output stride 4\n"
        "                  2v10 => a length-2 vector of one dimensional "
        "problem with size 10\n"
        "                  2:1:2v10:2:4 => a strided length-2 vector of one "
        "dimensional problem with size 10 having input and output strides\n"
        "                for nD problem :\n"
        "                  2x3x4 => size of the three dimensional problem\n"
        "                  2x4v3x5 => a 2x4 length vector of two dimension "
        "problem with size 3x5\n"
        "                  2:2:1x4:1:2v10:3:2x5:5:2 => a strided 2x4 length "
        "vector of two dimension problem with size 10x5 having strides\n"
        "Available options :-\n"
        "-p, --precision          'd' for double (fp64), 'f' for float (fp32) "
        "[default: d]\n"
        "-m, --data-model         'l' for LP64, 'i' for ILP64 [default: l]\n"
        "-b, --bench-type         'a' for accuracy, 'p' for performance "
        "[default: p]\n"
        "-r, --result-placement   'i' for in-place, 'o' for out-of-place "
        "[default: o]\n"
        "--order                  'i' for in-order, 'o' for out-of-order "
        "[default: i]\n"
        "--dir                    FFT direction - 'f' for forward, 'b' for"
        "backward [default: f]\n"
        "-f, --fft-type           'c2c' for complex to complex, 'r2c' for real "
        "to complex, 'c2r' for complex to real [default: c2c]\n"
        "                         NOTE: 'c2r forward' is same as 'r2c "
        "backward' and 'c2r backward' is same as 'r2c forward'\n"
        "                               Hence c2r mode will be mapped to its "
        "r2c mode\n"
        "-i, --iters              number of iterations [default: 50 for "
        "`performance` mode and 1 for `accuracy` mode]\n"
        "-s, --seed               specify manual seed value, random seed will "
        "be used if this option is not specified\n"
        "                           NOTE: iters will set to 1 if seed is "
        "specified\n"
        "-t, --tol                error tolerance value ranges from 0.0 to 1.0 "
        "(inclusive) [default: 1E-10 for double, 1E-3 for float]\n"
        "-n, --num-threads        number of CPU threads for multi-threading "
        "FFT [default: 1]\n"
        "--dynamic-load-model     '1' to allow the library to determine how "
        "many threads to be used, '0' to use value given in --num-threads as "
        "maximum number of threads [default: 0]\n"
        "-o, --opt-level          optimization levels used for benchmarking\n"
        "                           -1 = no optimization\n"
        "                            0 = non-SIMD algorithmic optimization\n"
        "                            1 = AVX128 optimization\n"
        "                            2 = AVX256 optimization\n"
        "                            3 = AVX512 optimization\n"
        "                            [default: -1]\n"
        "-l, --logger-mode        logger mode: log level value ranges from 0 "
        "to 4 [default: 0]\n"
        "                            0 = no logging\n"
        "                            1 = error\n"
        "                            2 = info\n"
        "                            3 = debug\n"
        "                            4 = trace\n"
        "--selector-time          '1' to print the time taken for preparing "
        "the solution, '0' to disable it [default: 0]\n"
        "--min-bench-time         set minimum time to calculate performance "
        "iterations [default: 100 ms]\n"
        "--measure-stats          '1' to measure selector stats, '0' to "
        "disable it [default: 0]\n"
        "--bit-reproducibility    '1' to use bit reproducibility mode, '0' to "
        "disable it [default: 0]\n"
        "--aligned-alloc          '1' to use aligned memory allocation, '0' to "
        "disable it [default: 1]\n");
}
