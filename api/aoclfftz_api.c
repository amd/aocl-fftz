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
#include "core/common/memory_manager.h"
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

// Execute function for Single-threaded and multi-threaded FFT.
// Uses the cached setup-time execution context.
aoclfftz_error_type aoclfftz_execute(FFTZ_VOID *handle)
{
    if (handle == NULL)
    {
        return AOCLFFTZ_EXECUTION_FAILURE;
    }

    aoclfftz_executor_t *executor_obj = (aoclfftz_executor_t *)handle;
    aoclfftz_mutable_ctx_t *base_ctx = &executor_obj->exec_metadata->base_ctx;
    return executor_obj->execute(executor_obj, base_ctx);
}

// Execute function for Single-threaded and multi-threaded FFT on different
// input, output buffers using the same solution from the handle.
// This function is safe for concurrent calls on a shared handle. A call either
// claims the setup-time scratch buffers if available or allocates fresh ones (C2C only).
aoclfftz_error_type aoclfftz_execute_io(FFTZ_VOID *handle, FFTZ_VOID *in,
                                        FFTZ_VOID *out)
{
    if ((handle == NULL) || (in == NULL) || (out == NULL))
    {
        return AOCLFFTZ_EXECUTION_FAILURE;
    }
    aoclfftz_executor_t *executor_obj = (aoclfftz_executor_t *)handle;
    aoclfftz_solution_t *sol = executor_obj->solution;
    FFTZ_INT32 is_real = IS_REAL(sol->decomp_scheme->flags);
    aoclfftz_immutable_metadata_t *exec_metadata = executor_obj->exec_metadata;
    aoclfftz_mutable_ctx_t ctx = exec_metadata->base_ctx;
    FFTZ_UINT32 dt_bytes = CTX_DT_SIZE(&ctx);
    ctx.in_real  = in;
    ctx.in_imag  = MOVE_ADDR(in, dt_bytes);
    ctx.out_real = out;
    ctx.out_imag = MOVE_ADDR(out, dt_bytes);

    // Real (R2C/C2R) solvers in this branch still consume in/out from
    // sol->decomp_scheme directly. Note: this part is NOT thread-safe.
    if (is_real)
    {
        sol->decomp_scheme->in_real  = ctx.in_real;
        sol->decomp_scheme->in_imag  = ctx.in_imag;
        sol->decomp_scheme->out_real = ctx.out_real;
        sol->decomp_scheme->out_imag = ctx.out_imag;

        return executor_obj->execute(executor_obj, &ctx);
    }

    FFTZ_INT32 expected_setup_buffers_acquired_val = 0;
    // Synchronize ownership with atomic builtins (wrapped in AOCLFFTZ_ATOMIC_*): they apply
    // atomicity per-access on the variable's address; declaring `setup_buffers_acquired` as
    // _Atomic risks a size/alignment mismatch when the parent struct is included in C++ (gtest).
    FFTZ_INT32 *setup_buffers_acquired = &exec_metadata->setup_buffers_acquired;
    FFTZ_INT32 is_owner = AOCLFFTZ_ATOMIC_CMP_XCHG(
        setup_buffers_acquired, &expected_setup_buffers_acquired_val, 1);
    // Try to acquire ownership of setup time allocated buffers: only one thread can hold
    // ownership at a time; others allocate fresh scratch slab.
    if (is_owner)
    {
        aoclfftz_error_type execute_status =
                                    executor_obj->execute(executor_obj, &ctx);
        AOCLFFTZ_ATOMIC_STORE(setup_buffers_acquired, 0);
        return execute_status;
    }

    // TODO: cache this slab in a per-core array (allocated lazily, freed at
    // destroy) to avoid repeated malloc/free on each execute_io invocation.
    FFTZ_VOID *scratch_slab = NULL;
    aoclfftz_error_type slab_status = alloc_per_call_scratch(exec_metadata,
                                                             &ctx,
                                                             &scratch_slab);
    if (slab_status != AOCLFFTZ_SUCCESS)
    {
        return slab_status;
    }

    aoclfftz_error_type ret = executor_obj->execute(executor_obj, &ctx);
    FREE_ALIGN_ALLOCATED_MEM(scratch_slab);
    return ret;
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
