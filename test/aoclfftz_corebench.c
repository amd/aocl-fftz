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

/** @file aoclfftz_corebench.c
 *
 *  @brief Tests the single-threaded core fft library for functional and
 *  performance tests.
 *
 *  This file contains the functions to setup and execute single-threaded core
 *  library for testing the library APIs.
 *
 *  @author S. Biplab Raut
 *  @author V. Murugan
 *  @author Srirammaswamy Srinivasan
 */

#include <math.h>
#include <time.h>
#include "aoclfftz_corebench.h"
#include "aoclfftz_corebench_utils.h"
#include "utils/utils.h"

/**
 * @brief print the help menu contents to the output
 *
 * @return VOID
 */
VOID show_help_menu()
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
        "                  2v3x4v5 => a 2x4 length vector of two dimensonal "
        "problem with size 4x5\n"
        "                  2:2:1v10:3:2x4:1:2v5:5:2 => a strided 2x4 length "
        "vector of two dimensonal problem with size 10x5 having strides\n"
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
        "to "
        "complex, 'c2r' for complex to real [default: c2c]\n"
        "-i, --iters              number of iterations [default: 50 for "
        "`performance` mode and 1 for `accuracy` mode]\n"
        "-w, --warmup-iters       number of warmup iterations, only used in "
        "performance mode [default: 10]\n"
        "-s, --seed               specify manual seed value, random seed will "
        "be used if this option is not specified\n"
        "                           NOTE: iters will set to 1 if seed is "
        "specified\n"
        "-t, --tol                error tolerance value ranges from 0.0 to 1.0 "
        "(inclusive) [default: 1E-10 for double, 1E-3 for float]\n"
        "-n, --num-threads        number of CPU threads for multi-threading "
        "FFT [default: 1]\n"
        "--dynamic-load-model     use it to allow the library to determine how "
        "many threads to be used (this option takes no value argument)\n"
        "-o, --opt-level          optimization levels used for benchmarking\n"
        "                           -1 = no optimization\n"
        "                            0 = non-SIMD algorithmic optimization\n"
        "                            1 = SSE2 optimization\n"
        "                            2 = AVX optimization\n"
        "                            3 = AVX2 optimization\n"
        "                            4 = AVX512 optimization\n"
        "                            5 = auto mode\n"
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
        "--measure-stats          '1' to measure selector stats, '0' to "
        "disable it [default: 0]\n"
        "--bit-reproducibility    '1' to use bit reproducibility mode, '0' to "
        "disable it [default: 0]\n");
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
        else if (strcmp(arg, "--warmup-iters") == 0 || arg[1] == 'w')
        {
            return 'w';
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
        else
        {
            // Unsupported option
            return '?';
        }
    }
    else
    {
        // Non option argument
        return 306;
    }
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
    INT32 c = -1;

    aoclfftz_bench_type_t bench_type = PERFORMANCE;
    aoclfftz_bench_precision_t precision = DOUBLE_P;
    aoclfftz_bench_data_model_t data_model = LP64;
    aoclfftz_bench_fft_type_t fft_type = COMPLEX_TO_COMPLEX;
    aoclfftz_bench_order_t order = IN_ORDER;
    aoclfftz_bench_res_placement_t res_placement = OUT_OF_PLACE;
    aoclfftz_bench_direction_t dir = FORWARD;
    INT32 dim_rank = 0;
    INT32 vec_rank = 0;
    aoclfftz_dim_t_64_ *dims = NULL;
    aoclfftz_dim_t_64_ *vecs = NULL;
    INT32 iters = 50;
    INT32 warmup_iters = 10;
    INT32 seed = 0;
    UCHAR use_random_seed = 1;
    INT32 opt_level = -1;
    INT32 logger_mode = 0;
    DOUBLE tolerance = 1E-10;
    UCHAR use_cust_tolerance = 0;
    INT32 num_threads = 1;
    INT32 dynamic_load_model = 0;
    UCHAR selector_time = 0;
    INT32 measure_stats = 0;
    INT32 bit_reproducibility = 0;
    INT32 status = PARSER_SUCCESS;
    INT32 ret = PARSER_SUCCESS;
    // check for the dependent arguments
    UCHAR valid_iters_arg_found = 0;
    UCHAR warmup_iters_arg_found = 0;

    CHAR *str_buff = (CHAR *)ALLOC_UNALIGN_UNINIT(sizeof(CHAR) * 50);

    INT32 arg_idx = 1;
    INT32 non_opt_arg_cnt = 0;
    while (arg_idx < argc)
    {
        c = get_option(argv, arg_idx);
        CHAR *optarg = argv[arg_idx + 1];
        switch (c)
        {
        case 'h':
            FREE_ALLOCATED_MEM(str_buff);
            FREE_ALLOCATED_MEM(dims);
            FREE_ALLOCATED_MEM(vecs);
            return HELP_MENU;
        case 'p':
            if (!strcmp(optarg, "f"))
            {
                precision = FLOAT_P;
            }
            else if (!strcmp(optarg, "d"))
            {
                precision = DOUBLE_P;
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
                data_model = LP64;
            }
            else if (!strcmp(optarg, "i"))
            {
                data_model = ILP64;
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
                bench_type = ACCURACY;
            }
            else if (!strcmp(optarg, "p"))
            {
                bench_type = PERFORMANCE;
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
                res_placement = IN_PLACE;
            }
            else if (!strcmp(optarg, "o"))
            {
                res_placement = OUT_OF_PLACE;
            }
            else
            {
                printf("ERROR: Unknown result placement\n");
                status = MAX(status, UNSUPPORTED_OPTION_ERROR);
            }
            // TODO: Remove this after adding support for in-place result
            // placement
            if (res_placement == IN_PLACE)
            {
                printf("ERROR: in-place result placement 'i' is currently "
                       "not supported\n");
                status = MAX(status, UNSUPPORTED_OPTION_ERROR);
            }
            break;
        case 300:
            if (!strcmp(optarg, "o"))
            {
                order = OUT_OF_ORDER;
            }
            else if (!strcmp(optarg, "i"))
            {
                order = IN_ORDER;
            }
            else
            {
                printf("ERROR: Unknown output order\n");
                status = MAX(status, UNSUPPORTED_OPTION_ERROR);
            }
            // TODO: Remove this after adding support for out-of-order
            // arrangement
            if (order == OUT_OF_ORDER)
            {
                printf("ERROR: out-order 'o' is currently not supported\n");
                status = MAX(status, UNSUPPORTED_OPTION_ERROR);
            }
            break;
        case 301:
            if (!strcmp(optarg, "b"))
            {
                dir = BACKWARD;
            }
            else if (!strcmp(optarg, "f"))
            {
                dir = FORWARD;
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
                fft_type = COMPLEX_TO_REAL;
            }
            else if (!strcmp(optarg, "r2c"))
            {
                fft_type = REAL_TO_COMPLEX;
            }
            else if (!strcmp(optarg, "c2c"))
            {
                fft_type = COMPLEX_TO_COMPLEX;
            }
            else
            {
                printf("ERROR: Unknown fft type\n");
                status = MAX(status, UNSUPPORTED_OPTION_ERROR);
            }
            // TODO: Remove this after adding support for 'c2r' and 'r2c' types
            if (fft_type == COMPLEX_TO_REAL || fft_type == REAL_TO_COMPLEX)
            {
                printf("ERROR: 'c2r' and 'r2c' types are currently not "
                       "supported\n");
                status = MAX(status, UNSUPPORTED_OPTION_ERROR);
            }
            break;
        case 'i':
            valid_iters_arg_found = 1;
            VALIDATE_AND_GET_INT(optarg, str_buff, iters, ret, 0);
            if (ret != 0)
            {
                printf("WARNING: Invalid iterations value given, "
                       "iterations should be a positive integer, running "
                       "bench with default value\n");
                valid_iters_arg_found = 0;
            }
            break;
        case 'w':
            warmup_iters_arg_found = 1;
            VALIDATE_AND_GET_INT(optarg, str_buff, warmup_iters, ret, 0);
            if (ret != 0)
            {
                printf("WARNING: Invalid warmup iterations value given, "
                       "iterations should be 0 or a positive integer, running "
                       "bench with default value\n");
                warmup_iters = 10;
            }
            break;
        case 's':
            VALIDATE_AND_GET_INT(optarg, str_buff, seed, ret, 0);
            if (ret != 0)
            {
                printf("WARNING: Invalid seed value given, running bench with "
                       "random seed\n");
            }
            else
            {
                use_random_seed = 0;
            }
            break;
        case 't':
            VALIDATE_AND_GET_DOUBLE(optarg, str_buff, tolerance, ret, 0.0, 1.0);
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
            VALIDATE_AND_GET_INT(optarg, str_buff, num_threads, ret, 1);
            if (ret != 0)
            {
                printf("WARNING: Invalid num threads value given, "
                       "running bench with default threads (1)\n");
                num_threads = 1;
            }
            // TODO: Remove this after adding multi-threaded FFT support
            if (num_threads > 1)
            {
                printf(
                    "WARNING: Multi-threaded FFT is currently not supported, "
                    "running bench with single-threaded FFT (1)\n");
                num_threads = 1;
            }
            break;
        case 'o':
            VALIDATE_AND_GET_INT(optarg, str_buff, opt_level, ret, -1);
            if (ret != 0)
            {
                printf("WARNING: Invalid opt level value given, running "
                       "bench with default value (-1: no-optimization)\n");
                opt_level = -1;
            }
            // TODO: Modify this after adding support for all optimization
            // levels
            else if (opt_level != -1)
            {
                printf("WARNING: only opt-level -1 is currently supported, "
                       "running bench with this option\n");
                opt_level = -1;
            }
            break;
        case 'l':
            VALIDATE_AND_GET_INT(optarg, str_buff, logger_mode, ret, 0);
            if (logger_mode > 4)
            {
                ret = 1;
            }
            if (ret != 0)
            {
                printf("WARNING: Invalid logger mode, running bench with "
                       "default logger mode (0)\n");
                logger_mode = 0;
            }
            break;
        case 302:
            // TODO: Modify this after adding support for dynamic-load-model
            printf("WARNING: dynamic-load-model option is currently not "
                   "supported, "
                   "running bench with dynamic-load-model disabled\n");
            break;
        case 303:
            if (atoi(optarg) != 0)
            {
                selector_time = 1;
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
            if (non_opt_arg_cnt == 0)
            {
                // Parse dimension sizes and vector sizes
                ret = find_dim_vec_ranks(argv[arg_idx], &dim_rank, &vec_rank);
                if (ret != PARSER_SUCCESS)
                {
                    status = MAX(status, SIZE_PARSING_ERROR);
                }
                else
                {
                    ret = allocate_and_fill_dims_vecs(
                        argv[arg_idx], dim_rank, vec_rank, &dims, &vecs, 1);
                    if (ret != PARSER_SUCCESS)
                    {
                        FREE_ALLOCATED_MEM(dims);
                        FREE_ALLOCATED_MEM(vecs);
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

    FREE_ALLOCATED_MEM(str_buff);

    // Problem size argument must be present
    if (non_opt_arg_cnt == 0)
    {
        status = MAX(status, SIZE_REQUIRED_ERROR);
    }

    else if (dim_rank == 0 || vec_rank == 0 || dims == NULL || vecs == NULL)
    {
        status = MAX(status, SIZE_PARSING_ERROR);
    }
    if (status != PARSER_SUCCESS)
    {
        FREE_ALLOCATED_MEM(dims);
        FREE_ALLOCATED_MEM(vecs);
        return status;
    }
    if (valid_iters_arg_found == 0)
    {
        if (bench_type == ACCURACY)
        {
            iters = 1;
        }
        else // bench_type == PERFORMANCE
        {
            iters = 50;
        }
    }
    if (warmup_iters_arg_found != 0 && bench_type == ACCURACY)
    {
        printf("WARNING: warmup iterations won't be used in ACCURACY mode\n");
        warmup_iters = 0;
    }
    if (selector_time != 0 && bench_type == ACCURACY)
    {
        printf("WARNING: selector-time won't be used in ACCURACY mode\n");
        selector_time = 0;
    }
    if (!use_random_seed && iters != 1 && bench_type == ACCURACY)
    {
        printf("WARNING: iterations will set to 1 since manual seed value is "
               "provided\n");
        iters = 1;
    }

    if (bench_params)
    {
        bench_params->dim_rank = dim_rank;
        bench_params->vec_rank = vec_rank;
        bench_params->dims = dims;
        bench_params->vecs = vecs;
        bench_params->precision = precision;
        bench_params->data_model = data_model;
        bench_params->bench_type = bench_type;
        bench_params->res_placement = res_placement;
        bench_params->order = order;
        bench_params->dir = dir;
        bench_params->fft_type = fft_type;
        bench_params->num_iterations = iters;
        bench_params->warmup_iterations = warmup_iters;
        bench_params->num_threads = num_threads;
        bench_params->dynamic_load_model = dynamic_load_model;
        bench_params->opt_level = opt_level;
        bench_params->logger_mode = logger_mode;
        bench_params->selector_time = selector_time;
        bench_params->measure_stats = measure_stats;
        bench_params->bit_reproducibility = bit_reproducibility;
        bench_params->seed = seed;
        bench_params->use_random_seed = use_random_seed;
        if (use_cust_tolerance)
        {
            bench_params->tolerance = tolerance;
        }
        else
        {
            if (precision == FLOAT_P)
            {
                bench_params->tolerance = 1E-3;
            }
            else
            {
                bench_params->tolerance = 1E-10;
            }
        }
        // create input and output buffers
        INT32 element_size = (bench_params->precision == FLOAT_P) ? 4 : 8;
        bench_params->in = ALLOC_UNALIGN_UNINIT(
            element_size * bench_params->dims[0].n *
            bench_params->dims[0].in_stride * T_DATA_STRIDE);
        bench_params->out = ALLOC_UNALIGN_INIT(
            bench_params->dims[0].n * bench_params->dims[0].out_stride *
            T_DATA_STRIDE, element_size);
    }
    else
    {
        status = MAX(status, PARSER_ERROR);
        return status;
    }

    AOCLFFTZ_LOG_UNFORMATTED(TRACE, bench_params->logger_mode,
                             "Bench params prepared from parsing arguments");

    ret = register_functions(bench_params->precision, bench_params->data_model);
    status = MAX(status, ret);
    return status;
}

/**
 * @brief Register the setup and other utility functions based on precision
 * and data-model
 *
 * @param precision data-type of input / output, FLOAT_P or DOUBLE_P
 * @param data_model data-model of integers, LP64 or ILP64
 * @return INT32
 */
INT32 register_functions(INT32 precision, INT32 data_model)
{
    if (precision == FLOAT_P)
    {
        prepare_input_data = prepare_input_data_f;
        dft_ref = dft_ref_f;
        compare = compare_f;
        if (data_model == LP64)
        {
            setup_problem = setup_problem_f;
            aoclfftz_execute = aoclfftz_execute_f;
            aoclfftz_destroy = aoclfftz_destroy_f;
        }
        else // data_model == ILP64
        {
            setup_problem = setup_problem_f_64_;
            aoclfftz_execute = aoclfftz_execute_f_64_;
            aoclfftz_destroy = aoclfftz_destroy_f_64_;
        }
    }
    else // precision == DOUBLE_P
    {
        prepare_input_data = prepare_input_data_d;
        dft_ref = dft_ref_d;
        compare = compare_d;
        if (data_model == LP64)
        {
            setup_problem = setup_problem_d;
            aoclfftz_execute = aoclfftz_execute_d;
            aoclfftz_destroy = aoclfftz_destroy_d;
        }
        else // data_model == ILP64
        {
            setup_problem = setup_problem_d_64_;
            aoclfftz_execute = aoclfftz_execute_d_64_;
            aoclfftz_destroy = aoclfftz_destroy_d_64_;
        }
    }
    return PARSER_SUCCESS;
}

/**
 * @brief Setup FFT problem of FLOAT LP64 type
 *
 * @param params aoclfftz_bench_params_t type contains parsed arguments
 * @return VOID* handle object
 */
VOID *setup_problem_f(aoclfftz_bench_params_t *params)
{
    AOCLFFTZ_LOG_UNFORMATTED(TRACE, params->logger_mode, "ENTER");
#ifdef WIN32
    timer clk_tick;
#endif
    timeVal start_time, end_time;
    UINT64 time_taken = 0.0;

    aoclfftz_prob_desc_f *p_desc =
        ALLOC_UNALIGN_UNINIT(sizeof(aoclfftz_prob_desc_f));
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
        time_taken = diffTime(clk_tick, start_time, end_time);
        if (handle != NULL)
        {
            printf("\n====================================\n");
            printf("    Selector time : %9.6lf ms\n", time_taken * 1E-6);
            printf("====================================\n");
        }
    }
    else
    {
        handle = aoclfftz_setup_f(p_desc);
    }
    DESTROY_PD(p_desc);
    AOCLFFTZ_LOG_UNFORMATTED(TRACE, params->logger_mode, "EXIT");
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
    AOCLFFTZ_LOG_UNFORMATTED(TRACE, params->logger_mode, "ENTER");
#ifdef WIN32
    timer clk_tick;
#endif
    timeVal start_time, end_time;
    UINT64 time_taken = 0.0;

    aoclfftz_prob_desc_d *p_desc =
        ALLOC_UNALIGN_UNINIT(sizeof(aoclfftz_prob_desc_d));
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
        time_taken = diffTime(clk_tick, start_time, end_time);
        if (handle != NULL)
        {
            printf("\n====================================\n");
            printf("    Selector time : %9.6lf ms\n", time_taken * 1E-6);
            printf("====================================\n");
        }
    }
    else
    {
        handle = aoclfftz_setup_d(p_desc);
    }
    DESTROY_PD(p_desc);
    AOCLFFTZ_LOG_UNFORMATTED(TRACE, params->logger_mode, "EXIT");
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
    AOCLFFTZ_LOG_UNFORMATTED(TRACE, params->logger_mode, "ENTER");
#ifdef WIN32
    timer clk_tick;
#endif
    timeVal start_time, end_time;
    UINT64 time_taken = 0.0;

    aoclfftz_prob_desc_f_64_ *p_desc =
        ALLOC_UNALIGN_UNINIT(sizeof(aoclfftz_prob_desc_f_64_));
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
        time_taken = diffTime(clk_tick, start_time, end_time);
        if (handle != NULL)
        {
            printf("\n====================================\n");
            printf("    Selector time : %9.6lf ms\n", time_taken * 1E-6);
            printf("====================================\n");
        }
    }
    else
    {
        handle = aoclfftz_setup_f_64_(p_desc);
    }
    DESTROY_PD(p_desc);
    AOCLFFTZ_LOG_UNFORMATTED(TRACE, params->logger_mode, "EXIT");
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
    AOCLFFTZ_LOG_UNFORMATTED(TRACE, params->logger_mode, "ENTER");
#ifdef WIN32
    timer clk_tick;
#endif
    timeVal start_time, end_time;
    UINT64 time_taken = 0.0;

    aoclfftz_prob_desc_d_64_ *p_desc =
        ALLOC_UNALIGN_UNINIT(sizeof(aoclfftz_prob_desc_d_64_));
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
        time_taken = diffTime(clk_tick, start_time, end_time);
        if (handle != NULL)
        {
            printf("\n====================================\n");
            printf("    Selector time : %9.6lf ms\n", time_taken * 1E-6);
            printf("====================================\n");
        }
    }
    else
    {
        handle = aoclfftz_setup_d_64_(p_desc);
    }
    DESTROY_PD(p_desc);
    AOCLFFTZ_LOG_UNFORMATTED(TRACE, params->logger_mode, "EXIT");
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
        FREE_ALLOCATED_MEM(params->in);
        FREE_ALLOCATED_MEM(params->out);
        FREE_ALLOCATED_MEM(params->dims);
        FREE_ALLOCATED_MEM(params->vecs);
        FREE_ALLOCATED_MEM(params);
    }
}

/**
 * @brief Run and benchmark the FFT problem
 *
 * @param params aoclfftz_bench_params_t type contains parsed arguments
 * @param handle handle object of VOID* type
 * @return INT32 status code
 */
INT32 run_problem_on_performance_mode(aoclfftz_bench_params_t *params,
                                      VOID *handle)
{
    AOCLFFTZ_LOG_UNFORMATTED(TRACE, params->logger_mode, "ENTER");
    INT32 status;
#ifdef WIN32
    timer clk_tick;
#endif
    timeVal start_time, end_time;
    UINT64 min_time = INT64_MAX, avg_time = 0, cur_time = 0;
    DOUBLE avg_mflops, max_mflops;
    INTP n = params->dims[0].n;

    // prepare random seed value
    if (params->use_random_seed)
    {
        params->seed = time(0);
    }

    AOCLFFTZ_LOG_FORMATTED(INFO, params->logger_mode, "seed   : %d",
                           params->seed);

    // prepare random input data
    prepare_input_data(params->in, n, params->dims[0].in_stride, RANDOM_INPUT);

    // warmup iterations (skipped from profiling)
    AOCLFFTZ_LOG_UNFORMATTED(TRACE, params->logger_mode, "WARM-UP START");
    for (INT32 i = 0; i < params->warmup_iterations; ++i)
    {
        AOCLFFTZ_LOG_FORMATTED(TRACE, params->logger_mode,
                               "WARM-UP Iteration: %d", i);
        status = aoclfftz_execute(handle);
    }
    AOCLFFTZ_LOG_UNFORMATTED(TRACE, params->logger_mode, "WARM-UP END");

    initTimer(clk_tick);
    for (INT32 i = 0; i < params->num_iterations; ++i)
    {
        AOCLFFTZ_LOG_FORMATTED(INFO, params->logger_mode, "Iteration: %d", i);
        getTime(start_time);
        status = aoclfftz_execute(handle);
        if (status != 0)
        {
            return EXECUTION_FAILURE;
        }
        getTime(end_time);
        cur_time = diffTime(clk_tick, start_time, end_time);
        avg_time = avg_time + cur_time;
        if (cur_time < min_time)
        {
            min_time = cur_time;
        }
    }
    avg_time = avg_time / params->num_iterations;
    max_mflops = (5.0 * n * log2(n)) / (min_time * 1E-3);
    avg_mflops = (5.0 * n * log2(n)) / (avg_time * 1E-3);

    // prepare suitable execution time unit
    DOUBLE time_multiplier = 1.0;
    CHAR time_unit[4];
    // units will be decided based on minimum of min_time and avg_time which is
    // min_time
    // print time in seconds
    if (min_time > 1E9)
    {
        time_multiplier = 1E-9;
        STRCPY(time_unit, 4, "s");
    }
    // print time in milli-seconds
    else if (min_time > 1E6)
    {
        time_multiplier = 1E-6;
        STRCPY(time_unit, 4, "ms");
    }
    // print time in micro-seconds
    else if (min_time > 1E3)
    {
        time_multiplier = 1E-3;
        STRCPY(time_unit, 4, "µs");
    }
    // print time in nano-seconds
    else
    {
        time_multiplier = 1.0;
        STRCPY(time_unit, 4, "ns");
    }

    printf("\n=====================================\n");
    printf("  Min Execution time : %6.3lf %s\n", min_time * time_multiplier,
           time_unit);
    printf("  Avg Execution time : %6.3lf %s\n", avg_time * time_multiplier,
           time_unit);
    printf("=====================================\n");
    printf("      Max MFLOPS : %9.6lf\n", max_mflops);
    printf("      Avg MFLOPS : %9.6lf\n", avg_mflops);
    printf("=====================================\n");
    AOCLFFTZ_LOG_UNFORMATTED(TRACE, params->logger_mode, "EXIT");
    return BENCH_SUCCESS;
}

#ifdef ENABLE_DFT_REFERENCE
/**
 * @brief run the FFT execute api and compare the output with DFT reference
 * output
 *
 * @param params bench params object
 * @return INT32 bench status code
 */
INT32 run_dft_reference_test(aoclfftz_bench_params_t *params)
{
    AOCLFFTZ_LOG_UNFORMATTED(TRACE, params->logger_mode, "ENTER");
    INT32 status = 0;
    INT32 compare_status = 0;
    INTP n = params->dims[0].n;
    INT32 element_size = (params->precision == FLOAT_P) ? 4 : 8;

    // setup FFT problem
    VOID *handle = setup_problem(params);
    if (handle == NULL)
    {
        return SETUP_FAILURE;
    }

    // create local buffer to store DFT reference output
    VOID *out_ref = ALLOC_UNALIGN_INIT(
        n * params->dims[0].out_stride * T_DATA_STRIDE, element_size);

    // initialize the random seed value based on current time
    if (params->use_random_seed)
    {
        srand(time(0));
    }

    for (INT32 i = 0; i < params->num_iterations && status == 0; i++)
    {
        // set the random seed value for each iteration
        if (params->use_random_seed)
        {
            params->seed = rand();
        }
        srand(params->seed);
        AOCLFFTZ_LOG_FORMATTED(INFO, params->logger_mode,
                               "Iteration: %d, Seed: %d", i, params->seed);

        // prepare random input data
        prepare_input_data(params->in, n, params->dims[0].in_stride,
                           RANDOM_INPUT);
        // execute the FFT problem
        status |= aoclfftz_execute(handle);
        // get the DFT reference output
        dft_ref(params->in, out_ref, n, params->dims[0].in_stride,
                params->dims[0].out_stride, params->dir);

        if (status != 0)
        {
            // destroy reference output buffer
            FREE_ALLOCATED_MEM(out_ref);
            // destroy handle
            aoclfftz_destroy(handle);
            return EXECUTION_FAILURE;
        }

        // compare the FFT output with DFT reference output
        compare_status = compare(
            params->out, out_ref, n * params->dims[0].out_stride,
            params->dims[0].out_stride, params->tolerance, params->logger_mode);
        if (compare_status != 0)
        {
            printf("\nResults mismatch on accuracy mode => DFT reference, "
                   "iteration: %d/%d, seed: %d\n",
                   i, params->num_iterations, params->seed);
            // destroy reference output buffer
            FREE_ALLOCATED_MEM(out_ref);
            // destroy handle
            aoclfftz_destroy(handle);
            return VERIFICATION_FAILURE;
        }
    }
    // destroy reference output buffer
    FREE_ALLOCATED_MEM(out_ref);
    // destroy handle
    aoclfftz_destroy(handle);
    AOCLFFTZ_LOG_UNFORMATTED(TRACE, params->logger_mode, "EXIT");
    return BENCH_SUCCESS;
}
#endif

/**
 * @brief run the FFT execute api verify the linearity property
 *
 * @param params bench params object
 * @return INT32 bench status code
 */
INT32 run_linearity_test(aoclfftz_bench_params_t *params)
{
    AOCLFFTZ_LOG_UNFORMATTED(TRACE, params->logger_mode, "ENTER");
    INT32 status = BENCH_SUCCESS;
    INT32 ret = AOCLFFTZ_SUCCESS;
    INTP n = params->dims[0].n;
    INT32 element_size = (params->precision == FLOAT_P) ? 4 : 8;
    INTP input_size = n * params->dims[0].in_stride;
    INTP output_size = n * params->dims[0].out_stride;

    VOID *factors, *in1, *in2, *out1, *out2, *out_combined, *handle;
    factors = in1 = in2 = out1 = out2 = out_combined = handle = NULL;

    // create buffer to store 2 complex constant values
    factors = ALLOC_UNALIGN_INIT(4, element_size);
    // create locals buffer to store inputs and outputs
    in1 = ALLOC_UNALIGN_UNINIT(element_size * input_size * T_DATA_STRIDE);
    in2 = ALLOC_UNALIGN_UNINIT(element_size * input_size * T_DATA_STRIDE);
    if (factors == NULL || in1 == NULL || in2 == NULL)
    {
        printf("run_linearity_test : input buffer creation failed\n");
        status = SETUP_FAILURE;
        goto exit_linearity_test;
    }
    out1 = ALLOC_UNALIGN_INIT(output_size * T_DATA_STRIDE, element_size);
    out2 = ALLOC_UNALIGN_INIT(output_size * T_DATA_STRIDE, element_size);
    out_combined =
        ALLOC_UNALIGN_INIT(output_size * T_DATA_STRIDE, element_size);
    if (out1 == NULL || out2 == NULL || out_combined == NULL)
    {
        printf("run_linearity_test : output buffer creation failed\n");
        status = SETUP_FAILURE;
        goto exit_linearity_test;
    }

    // setup FFT problem
    handle = setup_problem(params);
    if (handle == NULL)
    {
        status = SETUP_FAILURE;
        goto exit_linearity_test;
    }

    // initialize the random seed value based on current time
    if (params->use_random_seed)
    {
        srand(time(0));
    }

    for (INT32 i = 0; i < params->num_iterations; i++)
    {
        // set the random seed value for each iteration
        if (params->use_random_seed)
        {
            params->seed = rand();
        }
        srand(params->seed);
        AOCLFFTZ_LOG_FORMATTED(INFO, params->logger_mode,
                               "Iteration: %d, Seed: %d", i, params->seed);

        // prepare random input data
        prepare_input_data(in1, n, params->dims[0].in_stride, RANDOM_INPUT);
        prepare_input_data(in2, n, params->dims[0].in_stride, RANDOM_INPUT);

        // perform FFT for first input
        memcpy(params->in, in1, element_size * input_size * T_DATA_STRIDE);
        ret = aoclfftz_execute(handle);
        if (ret != AOCLFFTZ_SUCCESS)
        {
            status = EXECUTION_FAILURE;
            goto exit_linearity_test;
        }
        memcpy(out1, params->out, element_size * output_size * T_DATA_STRIDE);

        // perform FFT for second input
        memcpy(params->in, in2, element_size * input_size * T_DATA_STRIDE);
        ret = aoclfftz_execute(handle);
        if (ret != AOCLFFTZ_SUCCESS)
        {
            status = EXECUTION_FAILURE;
            goto exit_linearity_test;
        }
        memcpy(out2, params->out, element_size * output_size * T_DATA_STRIDE);

        // combine in1 and in2 and store the result in in1
        PREPARE_LINEAR_TEST_INPUTS(in1, in2, in1, factors, params->precision);
        // perform FFT for combined input
        memcpy(params->in, in1, element_size * input_size * T_DATA_STRIDE);
        ret = aoclfftz_execute(handle);
        if (ret != AOCLFFTZ_SUCCESS)
        {
            status = EXECUTION_FAILURE;
            goto exit_linearity_test;
        }
        memcpy(out_combined, params->out,
               element_size * output_size * T_DATA_STRIDE);

        // combine out1 and out2 and store the result in out1
        PREPARE_LINEAR_TEST_OUTPUTS(out1, out2, out1, factors,
                                    params->precision);
        // compare the outputs
        ret =
            compare(out1, out_combined, output_size, params->dims[0].out_stride,
                    params->tolerance, params->logger_mode);
        if (ret != AOCLFFTZ_SUCCESS)
        {
            printf("\nResults mismatch on accuracy mode => property: "
                   "linearity, iteration: %d/%d, seed: %d\n",
                   i, params->num_iterations, params->seed);
            status = VERIFICATION_FAILURE;
            goto exit_linearity_test;
        }
    }

exit_linearity_test:
    // destroy internal buffers
    FREE_ALLOCATED_MEM(in1);
    FREE_ALLOCATED_MEM(in2);
    FREE_ALLOCATED_MEM(out1);
    FREE_ALLOCATED_MEM(out2);
    FREE_ALLOCATED_MEM(out_combined);
    FREE_ALLOCATED_MEM(factors);
    // destroy handle
    aoclfftz_destroy(handle);
    AOCLFFTZ_LOG_UNFORMATTED(TRACE, params->logger_mode, "EXIT");
    return status;
}

/**
 * @brief run the FFT execute api verify the transformation property
 *
 * @param params bench params object
 * @return INT32 bench status code
 */
INT32 run_unit_impulse_transform_test(aoclfftz_bench_params_t *params)
{
    INT32 status = BENCH_SUCCESS;
    INT32 ret = AOCLFFTZ_SUCCESS;
    INTP n = params->dims[0].n;
    INT32 element_size = (params->precision == FLOAT_P) ? 4 : 8;
    INTP input_size = n * params->dims[0].in_stride;
    INTP output_size = n * params->dims[0].out_stride;

    // create a new bench params for reverse FFT direction
    aoclfftz_bench_params_t *params_reverse = NULL;
    ALLOC_AND_COPY_PARAMS(params_reverse, params);
    if (params_reverse == NULL)
    {
        printf("run_unit_impulse_transform_test : creating new bench params "
               "failed\n");
        return SETUP_FAILURE;
    }

    VOID *in, *handle, *handle_reverse;
    handle = handle_reverse = in = NULL;

    // create buffer to store input
    in = ALLOC_UNALIGN_UNINIT(element_size * input_size * T_DATA_STRIDE);
    if (in == NULL)
    {
        printf(
            "run_unit_impulse_transform_test : input buffer creation failed\n");
        return SETUP_FAILURE;
    }

    // reverse the FFT direction and swap input output strides
    if (params->dir == FORWARD)
    {
        params_reverse->dir = BACKWARD;
    }
    else
    {
        params_reverse->dir = FORWARD;
    }
    params_reverse->dims[0].in_stride = params->dims[0].out_stride;
    params_reverse->dims[0].out_stride = params->dims[0].in_stride;
    params_reverse->vecs[0].in_stride = params->vecs[0].out_stride;
    params_reverse->vecs[0].out_stride = params->vecs[0].in_stride;

    // create in and out buffers for params_reverse object
    params_reverse->in =
        ALLOC_UNALIGN_UNINIT(element_size * params_reverse->dims[0].n *
                             params_reverse->dims[0].in_stride * T_DATA_STRIDE);
    params_reverse->out = ALLOC_UNALIGN_INIT(
        params_reverse->dims[0].n * params_reverse->dims[0].out_stride *
        T_DATA_STRIDE, element_size);

    // setup FFT problem
    handle = setup_problem(params);
    if (handle == NULL)
    {
        status = SETUP_FAILURE;
        goto exit_unit_impulse_transform_test;
    }

    // setup reverse FFT problem
    handle_reverse = setup_problem(params_reverse);
    if (handle_reverse == NULL)
    {
        status = SETUP_FAILURE;
        goto exit_unit_impulse_transform_test;
    }

    // initialize the random seed value based on current time
    if (params->use_random_seed)
    {
        srand(time(0));
    }

    for (INT32 i = 0; i < params->num_iterations; i++)
    {
        // set the random seed value for each iteration
        if (params->use_random_seed)
        {
            params->seed = rand();
        }
        srand(params->seed);
        AOCLFFTZ_LOG_FORMATTED(INFO, params->logger_mode,
                               "Iteration: %d, Seed: %d", i, params->seed);

        // prepare impulse data
        prepare_input_data(in, n, params->dims[0].in_stride, IMPULSE_INPUT);

        // perform FFT
        memcpy(params->in, in, element_size * input_size * T_DATA_STRIDE);
        ret = aoclfftz_execute(handle);
        if (ret != AOCLFFTZ_SUCCESS)
        {
            status = EXECUTION_FAILURE;
            goto exit_unit_impulse_transform_test;
        }

        // perform reversed FFT
        memcpy(params_reverse->in, params->out,
               element_size * output_size * T_DATA_STRIDE);
        ret = aoclfftz_execute(handle_reverse);
        if (ret != AOCLFFTZ_SUCCESS)
        {
            status = EXECUTION_FAILURE;
            goto exit_unit_impulse_transform_test;
        }
        NORMALIZE_DATA(params_reverse->out, input_size, n, params->precision);

        // compare reversed output with the input
        ret = compare(in, params_reverse->out, input_size,
                      params->dims[0].in_stride, params->tolerance,
                      params->logger_mode);
        if (ret != AOCLFFTZ_SUCCESS)
        {
            printf("\nResults mismatch on accuracy mode => property: "
                   "transformation, iteration: %d/%d, seed: %d\n",
                   i, params->num_iterations, params->seed);
            status = VERIFICATION_FAILURE;
            goto exit_unit_impulse_transform_test;
        }
    }

exit_unit_impulse_transform_test:
    // destroy local buffer
    FREE_ALLOCATED_MEM(in);
    // destroy handles
    aoclfftz_destroy(handle);
    aoclfftz_destroy(handle_reverse);
    // destroy locally created bench param
    destroy_bench_param(params_reverse);
    AOCLFFTZ_LOG_UNFORMATTED(TRACE, params->logger_mode, "EXIT");
    return status;
}

/**
 * @brief run the FFT execute api verify the timeshift property
 *
 * @param params bench params object
 * @return INT32 bench status code
 */
INT32 run_timeshift_test(aoclfftz_bench_params_t *params)
{
    AOCLFFTZ_LOG_UNFORMATTED(TRACE, params->logger_mode, "ENTER");
    INT32 status = BENCH_SUCCESS;
    INT32 ret = AOCLFFTZ_SUCCESS;
    INTP n = params->dims[0].n;
    INT32 element_size = (params->precision == FLOAT_P) ? 4 : 8;
    INTP input_size = n * params->dims[0].in_stride;
    INTP output_size = n * params->dims[0].out_stride;

    VOID *in1, *in2, *out1, *out2, *handle;
    in1 = in2 = out1 = out2 = handle = NULL;
    // create buffers for inputs and outputs
    in1 = ALLOC_UNALIGN_UNINIT(element_size * input_size * T_DATA_STRIDE);
    in2 = ALLOC_UNALIGN_UNINIT(element_size * input_size * T_DATA_STRIDE);
    if (in1 == NULL || in2 == NULL)
    {
        printf("run_timeshift_test : input buffer creation failed\n");
        status = SETUP_FAILURE;
        goto exit_timeshift_test;
    }
    out1 = ALLOC_UNALIGN_INIT(output_size * T_DATA_STRIDE, element_size);
    out2 = ALLOC_UNALIGN_INIT(output_size * T_DATA_STRIDE, element_size);
    if (out1 == NULL || out2 == NULL)
    {
        printf("run_timeshift_test : output buffer creation failed\n");
        status = SETUP_FAILURE;
        goto exit_timeshift_test;
    }

    // setup FFT problem
    handle = setup_problem(params);
    if (handle == NULL)
    {
        status = SETUP_FAILURE;
        goto exit_timeshift_test;
    }

    // initialize the random seed value based on current time
    if (params->use_random_seed)
    {
        srand(time(0));
    }

    for (INT32 i = 0; i < params->num_iterations; i++)
    {
        // set the random seed value for each iteration
        if (params->use_random_seed)
        {
            params->seed = rand();
        }
        srand(params->seed);
        AOCLFFTZ_LOG_FORMATTED(INFO, params->logger_mode,
                               "Iteration: %d, Seed: %d", i, params->seed);

        // prepare the sinusoidal signal input data
        prepare_input_data(in1, n, params->dims[0].in_stride,
                           SINUSOIDAL_SIGNAL_INPUT);

        // Perform circular right shift by `m` times
        // range of m => [1, radix)
        INTP m = (INTP)(rand() % (n - 1)) + 1;
        PREPARE_TIMESHIFT_TEST_INPUTS(in1, in2, n, m, params->dims[0].in_stride,
                                      params->precision);

        // perform FFT for input
        memcpy(params->in, in1, element_size * input_size * T_DATA_STRIDE);
        ret = aoclfftz_execute(handle);
        if (ret != AOCLFFTZ_SUCCESS)
        {
            status = EXECUTION_FAILURE;
            goto exit_timeshift_test;
        }
        memcpy(out1, params->out, element_size * output_size * T_DATA_STRIDE);

        // perform FFT for shifted input
        memcpy(params->in, in2, element_size * input_size * T_DATA_STRIDE);
        status |= aoclfftz_execute(handle);
        memcpy(out2, params->out, element_size * output_size * T_DATA_STRIDE);
        PREPARE_TIMESHIFT_TEST_OUTPUTS(out1, out1, n, m,
                                       params->dims[0].out_stride, params->dir,
                                       params->precision);

        // compare the outputs
        ret = compare(out1, out2, output_size, params->dims[0].out_stride,
                      params->tolerance, params->logger_mode);
        if (ret != AOCLFFTZ_SUCCESS)
        {
            printf("\nResults mismatch on accuracy mode => property: "
                   "timeshift, iteration: %d/%d, seed: %d\n",
                   i, params->num_iterations, params->seed);
            status = VERIFICATION_FAILURE;
            goto exit_timeshift_test;
        }
    }

exit_timeshift_test:
    // destroy internal buffers
    FREE_ALLOCATED_MEM(in1);
    FREE_ALLOCATED_MEM(in2);
    FREE_ALLOCATED_MEM(out1);
    FREE_ALLOCATED_MEM(out2);
    // destroy handle
    aoclfftz_destroy(handle);
    AOCLFFTZ_LOG_UNFORMATTED(TRACE, params->logger_mode, "EXIT");
    return status;
}

/**
 * @brief run the test bench on accuracy mode by verifying the following DFT
 * properties.
 *          1. Linearity
 *          2. Unit Impulse Transformation
 *          3. Timeshift
 *
 * @param params bench params object
 * @return INT32 bench status code
 */
INT32 run_bench_on_accuracy_mode(aoclfftz_bench_params_t *params)
{
    INT32 status = BENCH_SUCCESS;

#ifdef ENABLE_DFT_REFERENCE
    status = run_dft_reference_test(params);
    HANDLE_BENCH_STATUS(status);
    if (status != BENCH_SUCCESS)
    {
        destroy_bench_param(params);
        return status;
    }
#endif

    // run property tests
    // 1. linearity property
    status = run_linearity_test(params);
    HANDLE_BENCH_STATUS(status);
    if (status != BENCH_SUCCESS)
    {
        destroy_bench_param(params);
        return status;
    }

    // 2. transformation test
    status = run_unit_impulse_transform_test(params);
    HANDLE_BENCH_STATUS(status);
    if (status != BENCH_SUCCESS)
    {
        destroy_bench_param(params);
        return status;
    }

    // 3. timeshift test
    status = run_timeshift_test(params);
    HANDLE_BENCH_STATUS(status);
    if (status != BENCH_SUCCESS)
    {
        destroy_bench_param(params);
        return status;
    }

    PRINT_SUCCESS("\nTest bench completed on accuracy mode\n\n");
    destroy_bench_param(params);
    return BENCH_SUCCESS;
}

/**
 * @brief run the test bench on performance mode and calculate MFLOPS.
 *
 * @param params bench params object
 * @return INT32 bench status code
 */
INT32 run_bench_on_performance_mode(aoclfftz_bench_params_t *params)
{
    INT32 status = 0;

    // setup the FFT problem
    VOID *handle = setup_problem(params);
    if (handle == NULL)
    {
        PRINT_FAILURE("\nTest bench failed [REASON: Setup problem failed]\n\n");
        aoclfftz_destroy(handle);
        destroy_bench_param(params);
        return SETUP_FAILURE;
    }

    // run the FFT problem
    status = run_problem_on_performance_mode(params, handle);
    if (status != 0)
    {
        PRINT_FAILURE(
            "\nTest bench failed [REASON: Execute problem failed]\n\n");
        aoclfftz_destroy(handle);
        destroy_bench_param(params);
        return EXECUTION_FAILURE;
    }

    // destroy the bench-param and handle objects
    aoclfftz_destroy(handle);
    destroy_bench_param(params);

    PRINT_SUCCESS("\nTest bench completed on performance mode\n\n");
    return BENCH_SUCCESS;
}

/**
 * @brief Entry function to test bench
 *
 * @param argc command-line argument count
 * @param argv command-line argument values as char array
 * @return INT32
 */
INT32 main(INT32 argc, CHAR *argv[])
{
    printf("\nAOCL-FFTZ version: %s\n\n", aoclfftz_version());

    // prepare bench params from user inputs
    aoclfftz_bench_params_t *params =
        ALLOC_UNALIGN_UNINIT(sizeof(aoclfftz_bench_params_t));
    INT32 status = prepare_bench_params(argc, argv, params);
    HANDLE_PARSER_ERROR_MESSAGE(status);
    if (status != PARSER_SUCCESS)
    {
        destroy_bench_param(params);
        return -1;
    }

    // log the user params in INFO mode
    LOG_BENCH_PARAMS(params);

    if (params->bench_type == PERFORMANCE)
    {
        printf("\nRunning bench on performance mode\n");
        run_bench_on_performance_mode(params);
    }
    else if (params->bench_type == ACCURACY)
    {
        printf("\nRunning bench on accuracy mode\n");
        run_bench_on_accuracy_mode(params);
    }
    else
    {
        printf("Invalid bench_type\n");
        return -1;
    }
    return 0;
}