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

/** @file bluestein_utils.h
 *
 *  @brief Declarations for utility functions related to Bluestein solver.
 *
 *  This file contains the functions for computing extended length, computing
 *  bluestein sequence, elementwise multiplication and normalize data.
 *
 *  @author Srirammaswamy Srinivasan
 */

#ifndef BLUESTEIN_UTILS_H
#define BLUESTEIN_UTILS_H

#include "api/aoclfftz_internal.h"
#include "core/kernels/kernel.h"

// Error return codes related to Bluestein sequence and elementwise
// multiplication
typedef enum
{
    BLUESTEIN_FAILURE = -1,
    BLUESTEIN_SUCCESS // Successful operation
} bluestein_status;

INTP get_extended_length(INTP n);
INT32 prepare_bluestein_sequence(aoclfftz_solution_t *sol, INTP m);
INT32 elementwise_multiplication(VOID *out, VOID *a, VOID *b, INTP n,
                                 UINT8 sign, UINT32 precision);
INT32 normalize_data(VOID *data, INTP n, DOUBLE normalize_factor,
                     UINT32 precision);

#endif // BLUESTEIN_UTILS_H