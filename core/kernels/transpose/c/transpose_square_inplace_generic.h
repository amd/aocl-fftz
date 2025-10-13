/**
 * Copyright (C) 2024-2025, Advanced Micro Devices. All rights reserved.
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

/** @file transpose_inplace_square_generic.h
 *
 *  @brief Declarations & definitions of type-generic C transpose kernels
 *
 *  Type-generic declarations and definitions of transpose kernels that use
 *  different algorithms to perform the transpose operation on square matrices,
 *  in place.
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
VOID FUNC(tiq_iterative, TRANSPOSE_DT, c)(TRANSPOSE_KERNEL_ARGS);

// iterative transpose for arbitrarily strided matrices
VOID FUNC(tisq_iterative, TRANSPOSE_DT, c)(TRANSPOSE_KERNEL_ARGS);

// copy a submatrix and its companion into local buffers, transpose them
// in-place and (swap) store the transposed data of each one in the other's
// location.
VOID FUNC(copy_transpose_swap_store, TRANSPOSE_DT, c)
                    (TRANSPOSE_DT *matrix, aoclfftz_dim_t_64_ row_metadata,
                    aoclfftz_dim_t_64_ column_metadata, INTP i, INTP j,
                    TRANSPOSE_DT *buffer1, TRANSPOSE_DT *buffer2);

// recursive function that performs the transpose of the given matrix
VOID FUNC(tiq_block_helper, TRANSPOSE_DT, c)(TRANSPOSE_DT *matrix,
                    aoclfftz_dim_t_64_ row_metadata,
                    aoclfftz_dim_t_64_ column_metadata, INTP i, INTP j,
                    UINT8 in_top_right, TRANSPOSE_DT *buffer1,
                    TRANSPOSE_DT *buffer2);

// recursive transpose for unit strided matrices
VOID FUNC(tiq_recursive_buf, TRANSPOSE_DT, c)(TRANSPOSE_KERNEL_ARGS);

#else
// implementations

VOID FUNC(tiq_iterative, TRANSPOSE_DT, c)(TRANSPOSE_KERNEL_ARGS)
{
    AOCLFFTZ_LOG(DEBUG, global_logger_mode, "Enter");

    TRANSPOSE_DT *in = (TRANSPOSE_DT *)in_ptr;

    for (INTP i = 0; i < row_metadata.n; ++i)
    {
        for (INTP j = i + 1; j < column_metadata.n; ++j)
        {
            TRANSPOSE_DT temp =
                in[LINEAR_IDX_2D(i, j, UNIT_STRIDE, row_metadata.in_stride)];

            in[LINEAR_IDX_2D(i, j, UNIT_STRIDE, row_metadata.in_stride)] =
                in[LINEAR_IDX_2D(j, i, UNIT_STRIDE, row_metadata.in_stride)];

            in[LINEAR_IDX_2D(j, i, UNIT_STRIDE, row_metadata.in_stride)] = temp;
        }
    }

    AOCLFFTZ_LOG(DEBUG, global_logger_mode, "Exit");
}

VOID FUNC(tisq_iterative, TRANSPOSE_DT, c)(TRANSPOSE_KERNEL_ARGS)
{
    AOCLFFTZ_LOG(DEBUG, global_logger_mode, "Enter");

    TRANSPOSE_DT *in = (TRANSPOSE_DT *)in_ptr;

    for (INTP i = 0; i < row_metadata.n; ++i)
    {
        for (INTP j = i + 1; j < column_metadata.n; ++j)
        {
            TRANSPOSE_DT temp = in[LINEAR_IDX_2D(
                i, j, column_metadata.in_stride, row_metadata.in_stride)];

            in[LINEAR_IDX_2D(i, j, column_metadata.in_stride,
                             row_metadata.in_stride)] =
                in[LINEAR_IDX_2D(j, i, column_metadata.in_stride,
                                 row_metadata.in_stride)];

            in[LINEAR_IDX_2D(j, i, column_metadata.in_stride,
                             row_metadata.in_stride)] = temp;
        }
    }

    AOCLFFTZ_LOG(DEBUG, global_logger_mode, "Exit");
}

VOID FUNC(copy_transpose_swap_store, TRANSPOSE_DT, c)
                    (TRANSPOSE_DT *matrix, aoclfftz_dim_t_64_ row_metadata,
                    aoclfftz_dim_t_64_ column_metadata, INTP i, INTP j,
                    TRANSPOSE_DT *buffer1, TRANSPOSE_DT *buffer2)
{
    INTP leading_dim = row_metadata.in_stride;
    INTP cols = column_metadata.n;

    /* TODO: check performance impact of replacing copy-then-inplace transpose
             with a direct out-of-place transpose */

    /* create references to matrices B and C */
    TRANSPOSE_DT *matrixB =
        &matrix[LINEAR_IDX_2D(i, j, UNIT_STRIDE, leading_dim)];
    TRANSPOSE_DT *matrixC =
        &matrix[LINEAR_IDX_2D(j, i, UNIT_STRIDE, leading_dim)];

    /* copy data from B and C to the temp buffers */
    for (INTP it = 0; it < cols; ++it)
    {
        /* copying from B to buffer1 */
        COPY_MATRIX_ELEMS(TRANSPOSE_DT, matrixB, it, 0, leading_dim, buffer1,
                          it, 0, cols, cols);

        /* copying from C to buffer2 */
        COPY_MATRIX_ELEMS(TRANSPOSE_DT, matrixC, it, 0, leading_dim, buffer2,
                          it, 0, cols, cols);
    }

    /* update the row metadata to match the dimensions of the temp buffer */
    row_metadata.in_stride = cols;

    /* transpose the data in-place within the external buffers */
    FUNC(tiq_iterative, TRANSPOSE_DT, c)(buffer1, NULL, row_metadata,
                                                column_metadata, NULL);

    FUNC(tiq_iterative, TRANSPOSE_DT, c)(buffer2, NULL, row_metadata,
                                                column_metadata, NULL);

    /* copy data from the temp buffers to B and C */
    for (INTP it = 0; it < cols; ++it)
    {
        /* copying from buffer2 to B */
        COPY_MATRIX_ELEMS(TRANSPOSE_DT, buffer2, it, 0, cols, matrixB, it, 0,
                          leading_dim, cols);

        /* copying from buffer1 to C */
        COPY_MATRIX_ELEMS(TRANSPOSE_DT, buffer1, it, 0, cols, matrixC, it, 0,
                          leading_dim, cols);
    }
}

// Important !!
//
//     - the destinations of elements in A & D, post transpose, are within the
//       source matrix
//
//     - the destinations of elements in B & C, post transpose are within the
//       other matrix (in C for B) (in B for C)
VOID FUNC(tiq_block_helper, TRANSPOSE_DT, c)(TRANSPOSE_DT *matrix,
                    aoclfftz_dim_t_64_ row_metadata,
                    aoclfftz_dim_t_64_ column_metadata, INTP i, INTP j,
                    UINT8 in_top_right, TRANSPOSE_DT *buffer1,
                    TRANSPOSE_DT *buffer2)
{
    INTP leading_dim = row_metadata.in_stride;
    INTP cols = column_metadata.n;

    TRANSPOSE_DT *sub_matrix =
        &matrix[LINEAR_IDX_2D(i, j, UNIT_STRIDE, leading_dim)];

    // if the size is smaller than min size for recursive square transpose
    if (cols <= CONCAT(REC_MIN_, TRANSPOSE_DT))
    {
        if (i == j)
        {
            // sub matrices on the main diagonal are transposed in place
            FUNC(tiq_iterative, TRANSPOSE_DT, c)(
                sub_matrix, NULL, row_metadata, column_metadata, NULL);
        }
        else
        {
            // off-diagonal sub matrices are transposed and swaped with their
            // corresponding transpose pair
            FUNC(copy_transpose_swap_store, TRANSPOSE_DT, c)(
                matrix, row_metadata, column_metadata, i, j, buffer1, buffer2);
        }
        return;
    }
    else
    {
        cols = cols / 2;
        column_metadata.n = cols;

        // Recursively split the matrix into 4 submatrices
        //
        //                   A ████ B ████
        //  ████████           ████   ████
        //  ████████    =>
        //  ████████         C ████ D ████
        //  ████████           ████   ████

        // Transpose sub matrix 'A'
        FUNC(tiq_block_helper, TRANSPOSE_DT, c)(matrix, row_metadata,
                                                       column_metadata, i, j,
                                                       in_top_right, buffer1,
                                                       buffer2);

        // Transpose sub matrix 'D'
        FUNC(tiq_block_helper, TRANSPOSE_DT, c)(matrix, row_metadata,
                                                       column_metadata,
                                                       i + cols, j + cols,
                                                       in_top_right, buffer1,
                                                       buffer2);

        // Since matrix B is the "top-right" matrix, set "in_top_right" = 1
        // Transpose sub matrix 'B'
        FUNC(tiq_block_helper, TRANSPOSE_DT, c)(matrix, row_metadata,
                                                       column_metadata, i,
                                                       j + cols, 1, buffer1,
                                                       buffer2);

        // NOTE:
        // -----
        // even though the logic looks different when compared to the iterative
        // transpose algorithm, it is actually the same. Our aim with the
        // recursive approach is still to transpose the matrix by swapping the
        // elements of the upper triangle along the main diagonal with the
        // lower trangle.
        //
        // ████████       ░███████
        // ████████  =>   ░░░█████
        // ████████       ░░░░░███
        // ████████       ░░░░░░░█


        // Transpose of sub matrix 'C' happens only if it is a part of a greater
        // matrix which itself is part of a "top-right" matrix.

        // The reason for this is that if 'C' is a submatrix of the
        // decomposition of a "top-right" matrix, then its transpose has to be
        // "swap-stored" with its companion's transpose.

        // On the other hand, if this 'C' is a submatrix of the decomposition of
        // a "non-top-right" matrix, then it automatically gets transposed when
        // its companion in the 'B' submatrix is transposed (due to the
        // "swap-store" transposition of off-diagonal companion matrices)
        if (in_top_right)
        {
            FUNC(tiq_block_helper, TRANSPOSE_DT, c)(matrix, row_metadata,
                                                           column_metadata,
                                                           i + cols, j,
                                                           in_top_right,
                                                           buffer1, buffer2);
        }

        // transpose the remaining row/column (if it exists) of the sub-matrices
        // (if they were recursively transposed)
        if (cols > CONCAT(REC_MIN_, TRANSPOSE_DT) && cols & 1)
        {
            TRANSPOSE_LAST_ROWCOL(TRANSPOSE_DT, matrix, cols, leading_dim, i, j,
                                  in_top_right);

            TRANSPOSE_LAST_ROWCOL(TRANSPOSE_DT, matrix, cols, leading_dim,
                                  i + cols, j + cols, in_top_right);

            TRANSPOSE_LAST_ROWCOL(TRANSPOSE_DT, matrix, cols, leading_dim, i,
                                  j + cols, 1);

            if (in_top_right)
            {
                TRANSPOSE_LAST_ROWCOL(TRANSPOSE_DT, matrix, cols, leading_dim,
                                      i + cols, j, in_top_right);
            }
        }
    }
}

VOID FUNC(tiq_recursive_buf, TRANSPOSE_DT, c)(TRANSPOSE_KERNEL_ARGS)
{
    AOCLFFTZ_LOG(DEBUG, global_logger_mode, "Enter");

    TRANSPOSE_DT *in = (TRANSPOSE_DT *)in_ptr;

    // allocate some temporary buffers
    TRANSPOSE_DT *buffer1, *buffer2;

    ALLOC_UNALIGN_INIT(buffer1, TRANSPOSE_DT,
                       CONCAT(REC_MIN_, TRANSPOSE_DT) *
                       CONCAT(REC_MIN_, TRANSPOSE_DT) *
                       sizeof(TRANSPOSE_DT));

    ALLOC_UNALIGN_INIT(buffer2, TRANSPOSE_DT,
                       CONCAT(REC_MIN_, TRANSPOSE_DT) *
                       CONCAT(REC_MIN_, TRANSPOSE_DT) *
                       sizeof(TRANSPOSE_DT));

    FUNC(tiq_block_helper, TRANSPOSE_DT, c)(
        in, row_metadata, column_metadata, 0, 0, 0, buffer1, buffer2);

    // transpose the remaining last row/column (if it exists) of the matrix (if
    // it was transposed recursively)
    if (column_metadata.n > CONCAT(REC_MIN_, TRANSPOSE_DT) &&
        column_metadata.n & 1)
    {
        TRANSPOSE_LAST_ROWCOL(TRANSPOSE_DT, in, column_metadata.n,
                              row_metadata.in_stride, 0, 0, 0);
    }

    // free the temporary buffers
    FREE_UNALIGN_ALLOCATED_MEM(buffer1);
    FREE_UNALIGN_ALLOCATED_MEM(buffer2);

    AOCLFFTZ_LOG(DEBUG, global_logger_mode, "Exit");
}

#endif // TYPE_GENERIC_IMPLEMENTATION
