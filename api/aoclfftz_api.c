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

/** @file aoclfftz_api.c
 *
 *  @brief APIs and data structures implementaion of the core (ST, MT) library.
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

//Setup function for float LP64 based Single-threaded and multi-threaded FFT
VOID *aoclfftz_setup_f(aoclfftz_prob_desc_f *problem)
{
    if (problem == NULL)
        return NULL;

    return setup_dft_f(problem);
}

//Execute function for float LP64 based Single-threaded and multi-threaded FFT
INT32 aoclfftz_execute_f(VOID *handle)
{
    if (handle == NULL)
    {
        return AOCLFFTZ_EXECUTION_FAILURE;
    }
    aoclfftz_executor_t *executor_obj = (aoclfftz_executor_t *)handle;
    return executor_obj->execute(executor_obj);
}

//Destroy function for float LP64 based Single-threaded and multi-threaded FFT
VOID aoclfftz_destroy_f(VOID *handle)
{
    destroy_handle(handle);
    return;
}

//Setup function for double LP64 based Single-threaded and multi-threaded FFT
VOID *aoclfftz_setup_d(aoclfftz_prob_desc_d *problem)
{
    if (problem == NULL)
        return NULL;

    return setup_dft_d(problem);
}

//Execute function for double LP64 based Single-threaded and multi-threaded FFT
INT32 aoclfftz_execute_d(VOID *handle)
{
    if (handle == NULL)
    {
        return AOCLFFTZ_EXECUTION_FAILURE;
    }
    aoclfftz_executor_t *executor_obj = (aoclfftz_executor_t *)handle;
    return executor_obj->execute(executor_obj);
}

//Destroy function for double LP64 based Single-threaded and multi-threaded FFT
VOID aoclfftz_destroy_d(VOID *handle)
{
    destroy_handle(handle);
    return;
}

//Setup function for float ILP64 based Single-threaded and multi-threaded FFT
VOID *aoclfftz_setup_f_64_(aoclfftz_prob_desc_f_64_ *problem)
{
    if (problem == NULL)
        return NULL;

    return setup_dft_f_64_(problem);
}

//Execute function for float ILP64 based Single-threaded and multi-threaded FFT
INT32 aoclfftz_execute_f_64_(VOID *handle)
{
    if (handle == NULL)
    {
        return AOCLFFTZ_EXECUTION_FAILURE;
    }
    aoclfftz_executor_t *executor_obj = (aoclfftz_executor_t *)handle;
    return executor_obj->execute(executor_obj);
}

//Destroy function for float ILP64 based Single-threaded and multi-threaded FFT
VOID aoclfftz_destroy_f_64_(VOID *handle)
{
    destroy_handle(handle);
    return;
}

//Setup function for double ILP64 based Single-threaded and
//multi-threaded FFT
VOID *aoclfftz_setup_d_64_(aoclfftz_prob_desc_d_64_ *problem)
{
    if (problem == NULL)
        return NULL;

    return setup_dft_d_64_(problem);
}

//Execute function for double ILP64 based Single-threaded and
//multi-threaded FFT
INT32 aoclfftz_execute_d_64_(VOID *handle)
{
    if (handle == NULL)
    {
        return AOCLFFTZ_EXECUTION_FAILURE;
    }
    aoclfftz_executor_t *executor_obj = (aoclfftz_executor_t *)handle;
    return executor_obj->execute(executor_obj);
}

//Destroy function for double ILP64 based Single-threaded and
//multi-threaded FFT
VOID aoclfftz_destroy_d_64_(VOID *handle)
{
    destroy_handle(handle);
    return;
}

//Function to return aocl-fftz library version string
const CHAR* aoclfftz_version(VOID)
{
    return (AOCLFFTZ_LIBRARY_VERSION " " AOCL_BUILD_VERSION);
}
