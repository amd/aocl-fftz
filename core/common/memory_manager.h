/**
 * Copyright (C) 2023-2025, Advanced Micro Devices. All rights reserved.
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

/** @file memory_manager.h
 *
 *  @brief Declares memory allocation and management functions of AOCL-FFTZ.
 *
 *  This file contains the function declarations for performing memory
 *  allocation management of different modules of the AOCL-FFTZ library.
 *
 *  @author S. Biplab Raut
 */

#ifndef MEMORY_MANAGER_H
#define MEMORY_MANAGER_H

#include "utils/allocator.h"
#include "api/aoclfftz_internal.h"
#include "selector/selector.h"

#define AOCL_SINGLE_MEM_REGION

aoclfftz_decomp_scheme_t *alloc_decomp_scheme(INT32 vec_rank, INT32 dim_rank);
INT32 alloc_bluestein_buffers(aoclfftz_bluestein_t *bluestein, INTP size);
aoclfftz_solution_t *alloc_solution(INT32 vec_rank, INT32 dim_rank);
aoclfftz_solution_t **alloc_sol_array(UINT32 n);

aoclfftz_selector_t *alloc_selector(INT32 vec_rank, INT32 dim_rank,
                                    VOID *scratch_space, UINT32 nthreads);

VOID *alloc_twiddle_buffer(UINTP size, UINT32 dt_prec);
VOID alloc_inplace_buffer(aoclfftz_solution_t *solution, VOID **buffer_ptr);

VOID destroy_selector(aoclfftz_selector_t *sel);
VOID destroy_selector_without_solution(aoclfftz_selector_t *sel);
VOID destroy_selector_without_scratch_space(aoclfftz_selector_t *sel);

VOID destroy_solution(aoclfftz_solution_t *sol, UINT8 destroy_buffers);
VOID destroy_solutions(aoclfftz_solution_t **sol, UINT32 n);
VOID destroy_decomp_scheme(aoclfftz_decomp_scheme_t *decomp_scheme);
VOID destroy_bluestein(aoclfftz_bluestein_t *bluestein);

#endif // MEMORY_MANAGER_H
