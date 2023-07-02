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
 
 /** @file selector.h
 *  
 *  @brief Functions and data structures of the selector module.
 *
 *  This file contains the functions and data structures that are used to
 *  select a solution of kernels for the given input problem description.
 *
 *  @author S. Biplab Raut
 */
 
#ifndef AOCLFFTZ_SELECTOR_H
#define AOCLFFTZ_SELECTOR_H

#include "core/solvers/solver.h"
#include "core/kernels/kernel.h"

#define NUM_SOLVERS 16
#define NUM_KERNELS 512

//Error return codes related to selector
//Add more codes at the top
typedef enum
{
    SELECTOR_FAILURE = -1,
    SELECTOR_SUCCESS         //Successful operation
} aoclfftz_selector;

//Table of solvers that is populated with applicable solvers at setup time
//ct, direct, nDim, buf, permKer, batched, bluestein, PFA, rader, permCopy,
//trans
solver_t solvers_table[NUM_SOLVERS] = { { {0x0} } };

//Table of kernels that is populated with applicable kernels at setup time
kernel_t kernels_table[NUM_KERNELS] = { {0x0} };

//Computational cost analysis of solution of an executed problem/sub-problem
typedef struct cost_analysis
{
    INT64 operations;
    INT64 time;
} cost_analysis_t;

//Selector data structure that is used to hold the solution and cost analysis
// at each decomposition level for the associated sub-problem 
typedef struct
{
    aoclfftz_solution_t *solution;
    cost_analysis_t *cost_analysis;
 } aoclfftz_selector_t;

//Function declarations
INT32 register_solvers_kernels(solver_t *, kernel_t *, 
                               aoclfftz_solution_t *, INT32 cpu_flags);
INT32 setup_dft_f_(aoclfftz_selector_t *, solver_t *, kernel_t *);
INT32 setup_dft_d_(aoclfftz_selector_t *, solver_t *, kernel_t *);

#endif //AOCLFFTZ_SELECTOR_H
