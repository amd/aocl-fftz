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

VOID populate_stride_array(INTP *strides, INTP stride_val, INTP n,
                           UINT8 compute_half_complex,
                           UINT8 adjust_to_full_complex);
VOID prepare_real_c2c_kernel_strides(INTP *in, INTP *out, INTP radix,
                                     INTP n, INTP stride);
VOID prepare_fused_kernel_strides(INTP *strides, INTP radix, INTP offset);

#endif // STRIDES_H
