// Copyright Advanced Micro Devices, Inc.
// SPDX-License-Identifier: BSD-3-Clause

/** @file register_functions.c
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

#include "test/bench/utils/register_functions.h"
#include "test/bench/bench_problem.h"

/**
 * @brief Register the setup, appropriate fftz API variants, and other utility
 * functions based on precision and data-model
 * @param params bench params object
 * @return FFTZ_INT32
 */
FFTZ_INT32 register_functions(aoclfftz_bench_params_t *params)
{
    if (params->precision == FLOAT_P)
    {
        params->prepare_input_data = prepare_input_data_f;
#ifdef ENABLE_DFT_REFERENCE
        params->dft_ref = dft_ref_f;
#endif
        params->compare = compare_f;
        if (params->data_model == LP64)
        {
            params->setup_problem = setup_problem_f;
        }
        else // data_model == ILP64
        {
            params->setup_problem = setup_problem_f_64_;
        }
    }
    else // precision == DOUBLE_P
    {
        params->prepare_input_data = prepare_input_data_d;
#ifdef ENABLE_DFT_REFERENCE
        params->dft_ref = dft_ref_d;
#endif
        params->compare = compare_d;
        if (params->data_model == LP64)
        {
            params->setup_problem = setup_problem_d;
        }
        else // data_model == ILP64
        {
            params->setup_problem = setup_problem_d_64_;
        }
    }
    return PARSER_SUCCESS;
}
