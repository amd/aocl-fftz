// Copyright Advanced Micro Devices, Inc.
// SPDX-License-Identifier: BSD-3-Clause

/** @file aoclfftz_api.c
 *
 *  @brief APIs and data structures implementation of the core (ST, MT) library.
 *
 *  This file contains the implementation of APIs and associated
 *  data structures that are responsible for setting up and executing the
 *  single-threaded and multi-threaded FFT operations.
 *
 *  @note Different variants of APIs and data structures are exposed to
 *  support float and double precision types in LP64 and ILP64 data models.
 *
 *  @author S. Biplab Raut
 */

#include "aoclfftz.h"
#include "types.h"
#include "selector/selector.h"
#include "core/executor.h"
#include "validate_problem.h"

// Setup function for float LP64 based Single-threaded and multi-threaded FFT
FFTZ_VOID *aoclfftz_setup_f(aoclfftz_prob_desc_f *problem)
{
    SET_PROBLEM_LOGGER_MODE(problem);
    VALIDATE_PROBLEM_DESCRIPTOR(problem);
    PRINT_LP64_PROBLEM_DESCRIPTOR(problem, DT_FLOAT);
    return setup_dft_f(problem);
}

// Setup function for double LP64 based Single-threaded and multi-threaded FFT
FFTZ_VOID *aoclfftz_setup_d(aoclfftz_prob_desc_d *problem)
{
    SET_PROBLEM_LOGGER_MODE(problem);
    VALIDATE_PROBLEM_DESCRIPTOR(problem);
    PRINT_LP64_PROBLEM_DESCRIPTOR(problem, DT_DOUBLE);
    return setup_dft_d(problem);
}

// Setup function for float ILP64 based Single-threaded and multi-threaded FFT
FFTZ_VOID *aoclfftz_setup_f_64_(aoclfftz_prob_desc_f_64_ *problem)
{
    SET_PROBLEM_LOGGER_MODE(problem);
    VALIDATE_PROBLEM_DESCRIPTOR(problem);
    PRINT_ILP64_PROBLEM_DESCRIPTOR(problem, DT_FLOAT);
    return setup_dft_f_64_(problem);
}

// Setup function for double ILP64 based Single-threaded and
// multi-threaded FFT
FFTZ_VOID *aoclfftz_setup_d_64_(aoclfftz_prob_desc_d_64_ *problem)
{
    SET_PROBLEM_LOGGER_MODE(problem);
    VALIDATE_PROBLEM_DESCRIPTOR(problem);
    PRINT_ILP64_PROBLEM_DESCRIPTOR(problem, DT_DOUBLE);
    return setup_dft_d_64_(problem);
}

// Execute function for Single-threaded and multi-threaded FFT
aoclfftz_error_type aoclfftz_execute(FFTZ_VOID *handle)
{
    if (handle == NULL)
    {
        return AOCLFFTZ_EXECUTION_FAILURE;
    }
    aoclfftz_executor_t *executor_obj = (aoclfftz_executor_t *)handle;
    return executor_obj->execute(executor_obj);
}

// Execute function for Single-threaded and multi-threaded FFT on different
// input, output buffers using the same solution from the handle
aoclfftz_error_type aoclfftz_execute_io(FFTZ_VOID *handle, FFTZ_VOID *in,
                                        FFTZ_VOID *out)
{
    if ((handle == NULL) | (in == NULL) | (out == NULL))
    {
        return AOCLFFTZ_EXECUTION_FAILURE;
    }
    // manipulate the in/out ptr in the structure based on direction
    aoclfftz_executor_t *executor_obj = (aoclfftz_executor_t *)handle;
    aoclfftz_solution_t *sol = executor_obj->solution;
    FFTZ_UINT32 dt_bytes = SOL_DT_SIZE(sol);

    sol->decomp_scheme->in_real = in;
    sol->decomp_scheme->in_imag = MOVE_ADDR(in, dt_bytes);
    sol->decomp_scheme->out_real = out;
    sol->decomp_scheme->out_imag = MOVE_ADDR(out, dt_bytes);

    // For 1D out-of-place problems, ct_buffer is NULL and ct_buf_real/imag
    // point to user's output buffer. We must update these pointers to new
    // buffers. For multi-dim or in-place problems, ct_buffer is internally
    // allocated, so ct_buf_real/imag should NOT be changed.
    if (sol->dft_bufs != NULL && sol->dft_bufs->ct_buffer == NULL)
    {
        sol->dft_bufs->ct_buf_real = sol->decomp_scheme->out_real;
        sol->dft_bufs->ct_buf_imag = sol->decomp_scheme->out_imag;
    }

    return aoclfftz_execute(handle);
}

// Destroy function for Single-threaded and multi-threaded FFT
FFTZ_VOID aoclfftz_destroy(FFTZ_VOID *handle)
{
    destroy_handle(handle);
    return;
}

// Function to return aocl-fftz library version string
const FFTZ_CHAR *aoclfftz_version(FFTZ_VOID)
{
    return (AOCLFFTZ_LIBRARY_VERSION " " AOCL_BUILD_VERSION);
}
