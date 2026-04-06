// Copyright Advanced Micro Devices, Inc.
// SPDX-License-Identifier: BSD-3-Clause

/** @file transpose_kernels_c.c
 *
 *  @brief Implementations of transpose kernels
 *
 *  Implementations of transpose kernels that use different algorithms to
 *  perform the transpose operation.
 *
 *  @author Ashwin K. Godbole
 */

#include "core/kernels/transpose/transpose_kernels.h"
#include "utils/allocator.h"

#define TYPE_GENERIC_IMPLEMENTATION

#define TRANSPOSE_DT FLOAT
#include "core/kernels/transpose/c/transpose_square_inplace_generic.h"
#include "core/kernels/transpose/c/transpose_rectangle_inplace_generic.h"
#include "core/kernels/transpose/c/transpose_outofplace_generic.h"
#undef TRANSPOSE_DT

#define TRANSPOSE_DT DOUBLE
#include "core/kernels/transpose/c/transpose_square_inplace_generic.h"
#include "core/kernels/transpose/c/transpose_rectangle_inplace_generic.h"
#include "core/kernels/transpose/c/transpose_outofplace_generic.h"
#undef TRANSPOSE_DT

#define TRANSPOSE_DT aoclfftz_complex_f_t
#include "core/kernels/transpose/c/transpose_square_inplace_generic.h"
#include "core/kernels/transpose/c/transpose_rectangle_inplace_generic.h"
#include "core/kernels/transpose/c/transpose_outofplace_generic.h"
#undef TRANSPOSE_DT

#define TRANSPOSE_DT aoclfftz_complex_d_t
#include "core/kernels/transpose/c/transpose_square_inplace_generic.h"
#include "core/kernels/transpose/c/transpose_rectangle_inplace_generic.h"
#include "core/kernels/transpose/c/transpose_outofplace_generic.h"
#undef TRANSPOSE_DT

#undef TYPE_GENERIC_IMPLEMENTATION
