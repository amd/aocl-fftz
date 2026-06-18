// Copyright Advanced Micro Devices, Inc.
// SPDX-License-Identifier: BSD-3-Clause

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
VOID compute_twiddle_buffer(VOID *twiddle_buffer, INTP radix, INTP n_tw_batches,
                            UINT32 dt_prec);
VOID compute_twiddle_buffer_real(VOID *twiddle_buffer, INTP radix,
                                 INTP num_c2c_per_group, INTP num_groups,
                                 INTP freq_factor, UINT8 dir, UINT32 dt_prec);
VOID compute_sr_twiddle_buffer(VOID *twiddle_buffer, INTP n, UINT32 dt_prec);
#endif

#endif // TWIDDLE_H
