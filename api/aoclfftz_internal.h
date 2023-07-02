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
 
 /** @file aoclfftz_internal.h
 *  
 *  @brief Top-level data structures used across different modules that are not
 *  publicly exposed but are internal to the AOCL FFTZ library.
 *
 *  This file contains the internal library-wide data structures including
 *  top-level Handle and DFT module structures.
 *
 *  @note Different variants of data structures are defined to
 *  support float and double precision types in LP64 and ILP64 data models.
 *
 *  @author S. Biplab Raut
 */
 
#ifndef AOCLFFTZ_INTERNAL_H
#define AOCLFFTZ_INTERNAL_H

#include "types.h"
#include "aoclfftz.h"

#define AOCLFFTZ_INTERNAL_LIBRARY_VERSION "AOCL-FFTZ Internal 1.0"


//Forward declarations
typedef struct aoclfftz_solution aoclfftz_solution_t;
typedef struct aoclfftz_generic_solver aoclfftz_generic_solver_t;

//Base data structure acting as an abstract class that is derived by the
//top-level DFT data structure and implemented by all the solvers
struct aoclfftz_generic_solver
{
    INT32 * (*register_solver) (VOID *solver);
    INT32 * (*dft_solver) (VOID *prob_desc, aoclfftz_solution_t *solution);
    VOID (*destroy_solver) (aoclfftz_solution_t *solution);
    UINT32 logger_mode;
};

//Holds info on the main problem or decomposed sub-problem in current dimension
typedef struct {
    VOID *in;
    VOID *out;
    aoclfftz_smp_pfft *pfft;
    aoclfftz_cntrl_params *cntrl_params;
    ptrdiff_t n;
    ptrdiff_t i_stride;
    ptrdiff_t o_stride;
    ptrdiff_t vi_stride;
    ptrdiff_t vo_stride;
    INT32 vec_rank;
    INT32 dim_rank;
    UINT32 flags; //in-place, real, in-order, dir, ..
} aoclfftz_decomp_scheme_t;

//Solution data structure that is returned as a handle by the setup API and 
//used by the execute API.
typedef struct aoclfftz_solution
{
    aoclfftz_generic_solver_t *solver;
    aoclfftz_decomp_scheme_t *decomp_scheme;
    aoclfftz_solution_t *next_sol;
} aoclfftz_solution_t;

//float LP64
//DFT data structure that holds all other module objects and is the top-level
//data structure of the library.
typedef struct
{
    aoclfftz_prob_desc_f *prob_desc;
    aoclfftz_solution_t *sol_handle;
} aoclfftz_dft_f;

//double LP64
//DFT data structure that holds all other module objects and is the top-level
//data structure of the library.
typedef struct
{
    aoclfftz_prob_desc_d *prob_desc;
    aoclfftz_solution_t *sol_handle;
} aoclfftz_dft_d;

//float ILP64
//DFT data structure that holds all other module objects and is the top-level
//data structure of the library.
typedef struct
{
    aoclfftz_prob_desc_f_64_ *prob_desc;
    aoclfftz_solution_t *sol_handle;
} aoclfftz_dft_f_64_;

//double LP64
//DFT data structure that holds all other module objects and is the top-level
//data structure of the library.
typedef struct
{
    aoclfftz_prob_desc_d_64_ *prob_desc;
    aoclfftz_solution_t *sol_handle;
} aoclfftz_dft_d_64_;

#endif //AOCLFFTZ_INTERNAL_H