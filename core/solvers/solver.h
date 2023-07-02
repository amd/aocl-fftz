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

 /** @file solver.h
 *
 *  @brief Solver data strcture and types.
 *
 *  This file contains the list of different solvers and the data structure to
 *  hold a specific solver for solving a given input problem or sub-problem.
 *
 *  @author S. Biplab Raut
 */

#ifndef AOCLFFTZ_SOLVER_H
#define AOCLFFTZ_SOLVER_H

#include "api/aoclfftz_internal.h"

 //Solver types implemented in the library for executing a given DFT problem
typedef enum
{
    SOLVER_CT = 1,
    SOLVER_DIRECT,
    SOLVER_NDIM,
    SOLVER_BUFFERED,
    SOLVER_PERM_KER,
    SOLVER_BATCHED,
    SOLVER_BLUESTEIN,
    SOLVER_PFA,
    SOLVER_RADER,
    SOLVER_PERM_COPY,
    SOLVER_TRANS,
    NUM_SOLVERS_END
} aoclfftz_solver_type;

//Solver data structure that holds solver object/pointer and its type
typedef struct solver
{
    aoclfftz_generic_solver_t solver;
    aoclfftz_solver_type solv_type;
} solver_t;

#endif //AOCLFFTZ_SOLVER_H