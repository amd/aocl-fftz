/**
 * Copyright (C) 2024, Advanced Micro Devices. All rights reserved.
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

/** @file transpose_kernels.h
 *
 *  @brief Declarations of all supported transpose kernels
 *
 *  This file contains declarations of all the supported transpose kernels. This
 *  includes different algorithms (iterative, recursive, etc.) as well as the
 *  implementations of each algorithm in the supported ISA variants (C, AVX128,
 *  AVX256 and AVX512) where possible. Every variant is also implemented for
 *  both, real and complex data matrices (single & double) precision data
 *  matrices.
 *
 *  This is the top-level header file for all transpose kernels variants. For
 *  all other files that require these transpose kernels, including this header
 *  is sufficient.
 *
 *  Individual header files (for generic C kernels) and regular kernel
 *  declarations (for ISA specific transpose kernels) are all to be included in
 *  this file only.
 *
 *  @author Ashwin K. Godbole
 */

#ifndef TRANSPOSE_KERNELS_H
#define TRANSPOSE_KERNELS_H

#include "api/aoclfftz_internal.h"
#include "core/kernels/transpose/transpose_utils.h"

/* -----------------------------------------------------------------------------
 * The general transpose kernel takes the following parameters
 *      *in                 : the input matrix
 *      *out                : the output matrix
 *      aoclfftz_dim_t_64_  : row_metadata
 *          .n              :     #rows
 *          .in_stride      :     input matrix leading dimension
 *          .out_stride     :     output matrix leading dimension
 *      aoclfftz_dim_t_64_  : column_metadata
 *          .n              :     #columns
 *          .in_stride      :     input matrix stride
 *          .out_stride     :     output matrix stride
 * -----------------------------------------------------------------------------
 * Kernel naming convention
 * -----------------------------------------------------------------------------
 * Each symbol and its associated meaning is tabulated below. In some cases,
 * the presence or absence of a symbol conveys some information about the
 * kernel; for e.g. an 's' in the name indicates that the kernel supports
 * arbitrarily strided matrices but the absence of the 's' indicates otherwise.
 * -----------------------------------------------------------------------------
 *      t       : transpose kernel
 * -----------------------------------------------------------------------------
 *      i       : in-place transpose
 *      o       : out-of-place transpose
 * -----------------------------------------------------------------------------
 *      s       : strided
 * -----------------------------------------------------------------------------
 *      q       : square transpose
 *      r       : rectangular transpose
 * -----------------------------------------------------------------------------
 *      FLOAT                   : FLOAT values
 *      DOUBLE                  : DOUBLE values
 *      aoclfftz_complex_f_t    : Complex(FLOAT) values
 *      aoclfftz_complex_d_t    : Complex(DOUBLE) values
 * -----------------------------------------------------------------------------
 *      c       : portable C code
 *      avx128  : avx128 intrinsic code
 *      avx256  : avx256 intrinsic code
 *      avx512  : avx512 intrinsic code
 * -----------------------------------------------------------------------------
 */

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

#endif // TRANSPOSE_KERNELS_H
