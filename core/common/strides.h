// Copyright Advanced Micro Devices, Inc.
// SPDX-License-Identifier: BSD-3-Clause

/** @file strides.h
 *
 *  @brief Declarations for stride utility functions.
 *
 *  This file contains the function declarations related to strided-memcpy,
 *  computing and manipulating stride array values for real, complex and
 *  half-complex data.
 *
 *  @author Srirammaswamy Srinivasan
 */

#ifndef STRIDES_H
#define STRIDES_H

#include "api/types.h"

FFTZ_VOID populate_stride_array(FFTZ_INTP *strides, FFTZ_INTP stride_val,
                                FFTZ_INTP n, FFTZ_UINT8 compute_half_complex,
                                FFTZ_UINT8 adjust_to_full_complex);
FFTZ_VOID prepare_real_c2c_kernel_strides(FFTZ_INTP *in, FFTZ_INTP *out,
                                          FFTZ_INTP radix, FFTZ_INTP n,
                                          FFTZ_INTP stride);
FFTZ_VOID prepare_fused_kernel_strides(FFTZ_INTP *strides, FFTZ_INTP radix,
                                       FFTZ_INTP offset);

#endif // STRIDES_H
