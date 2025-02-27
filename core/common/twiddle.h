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

/** @file twiddle.h
 *
 *  @brief Declarations for routines that computes and applies Twiddle factor.
 *
 *  This file contains the functions related to computing and multiplying
 *  twiddle factors to the values as needed between FFT stages
 *
 *  @author S. Biplab Raut
 */

#ifndef TWIDDLE_H
#define TWIDDLE_H

#include "api/aoclfftz_internal.h"

// Usage of additional buffers to hold the twiddle data is enabled by default.
#define IN_MEMORY_TWIDDLE_FACTORS 1

// Error return codes related to Twiddle factors multiplication
// Add more codes at the top
typedef enum
{
    TW_FAILURE = -1,
    TW_SUCCESS         // Successful operation
} twiddle_status;

#if IN_MEMORY_TWIDDLE_FACTORS == 1
VOID setup_twiddle_buffer(VOID* twiddle_buffer, INTP r, INTP m, UINT32 dt_prec);
#endif

INT32 twiddle_multiplier(aoclfftz_solution_t *sol);
INT32 twiddle_multiplier_inplace(aoclfftz_solution_t *sol);
INT32 twiddle_multiplier_for_real(aoclfftz_solution_t *sol, INTP p);

#endif // TWIDDLE_H
