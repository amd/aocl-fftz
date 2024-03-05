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

/** @file executor.h
 *
 *  @brief Functions and data structures for executor module.
 *
 *  This file contains the functions and data structures that are used to
 *  execute a solution of kernels for the given input problem description.
 *
 *  @author S. Biplab Raut
 */

#ifndef AOCLFFTZ_EXECUTOR_H
#define AOCLFFTZ_EXECUTOR_H

#include "core/solvers/solver.h"
#include "core/kernels/kernel.h"

//Error return codes related to executor
//Add more codes at the top
typedef enum
{
    EXECUTOR_FAILURE = -1,
    EXECUTOR_SUCCESS         //Successful operation
} aoclfftz_executor_status;

//Executor data structure that is used to hold the solution and cost analysis
// at each decomposition level for the associated sub-problem
typedef struct aoclfftz_executor
{
    aoclfftz_solution_t *solution;
    execute_ execute;
    //cost_analysis_t *cost_analysis;
} aoclfftz_executor_t;

#endif //AOCLFFTZ_EXECUTOR_H
