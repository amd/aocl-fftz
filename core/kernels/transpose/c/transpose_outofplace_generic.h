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

/** @file transpose_outofplace.h
 *
 *  @brief Declarations & definitions of type-generic C transpose kernels
 *
 *  Type-generic declarations and definitions of transpose kernels that use
 *  different algorithms to perform the transpose operation on matrices,
 *  out of place.
 *
 *  @author Ashwin K. Godbole
 */

// This header file has no include guards because its contents are type generic
// the file is expected to be included multiple times in other files to generate
// the required function declarations for the different supported datatypes

// TRANSPOSE_DT is expected to be one of the following:
//      FLOAT
//      DOUBLE
//      aoclfftz_complex_f_t
//      aoclfftz_complex_d_t
#ifndef TRANSPOSE_DT
#error "Datatype not specified for instantiation."
#endif

#include "api/aoclfftz_internal.h"
#include "utils/utils.h"
#include "core/kernels/transpose/transpose_utils.h"
#include "core/kernels/transpose/transpose_config.h"

// Kernel naming convention
// -----------------------------------------------------------------------------
// Each symbol and its associated meaning is tabulated below. In some cases,
// the presence or absence of a symbol conveys some information about the
// kernel; for e.g. an 's' in the name indicates that the kernel supports
// arbitrarily strided matrices but the absence of the 's' indicates otherwise.
// -----------------------------------------------------------------------------
//      t       : transpose kernel
// -----------------------------------------------------------------------------
//      i       : in-place transpose
//      o       : out-of-place transpose
// -----------------------------------------------------------------------------
//      s       : strided
// -----------------------------------------------------------------------------
//      q       : square transpose
//      r       : rectangular transpose
// -----------------------------------------------------------------------------
//      FLOAT                   : FLOAT values
//      DOUBLE                  : DOUBLE values
//      aoclfftz_complex_f_t    : Complex(FLOAT) values
//      aoclfftz_complex_d_t    : Complex(DOUBLE) values
// -----------------------------------------------------------------------------
//      c       : portable C code
//      avx128  : avx128 intrinsic code
//      avx256  : avx256 intrinsic code
//      avx512  : avx512 intrinsic code
// -----------------------------------------------------------------------------

// All transpose kernels must take the same args given by TRANSPOSE_KERNEL_ARGS.
// TRANSPOSE_KERNEL_ARGS expands to the following:
//     VOID *in_ptr,
//     VOID *out_ptr,
//     aoclfftz_dim_t_64_ row_metadata,
//     aoclfftz_dim_t_64_ column_metadata,
//     aoclfftz_transpose_aux_mem_t *aux_mem

#ifndef TYPE_GENERIC_IMPLEMENTATION
// Declarations

// iterative transpose for unit strided matrices
VOID FUNC(tos_iterative, TRANSPOSE_DT, c)(TRANSPOSE_KERNEL_ARGS);

#else
// implementations

VOID FUNC(tos_iterative, TRANSPOSE_DT, c)(TRANSPOSE_KERNEL_ARGS)
{
#ifdef AOCL_ENABLE_LOG
    AOCLFFTZ_LOG_UNFORMATTED(TRACE, TRACE, "Enter");
#endif

    TRANSPOSE_DT *in = (TRANSPOSE_DT *)in_ptr;
    TRANSPOSE_DT *out = (TRANSPOSE_DT *)out_ptr;

    for (INTP i = 0; i < row_metadata.n; ++i)
    {
        for (INTP j = 0; j < column_metadata.n; ++j)
        {
            out[LINEAR_IDX_2D(j, i, column_metadata.out_stride,
                           row_metadata.out_stride)] =
                in[LINEAR_IDX_2D(i, j, column_metadata.in_stride,
                               row_metadata.in_stride)];
        }
    }

#ifdef AOCL_ENABLE_LOG
    AOCLFFTZ_LOG_UNFORMATTED(TRACE, TRACE, "Exit");
#endif
}

#endif // TYPE_GENERIC_IMPLEMENTATION
