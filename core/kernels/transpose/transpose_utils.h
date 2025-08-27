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

/** @file transpose_utils.h
 *
 *  @brief Helper macros for implementing transpose kernels
 *
 *  This file contains macros that make it easier to generate type generic
 *  transpose kernels, simplify matrix access logic, etc.
 *
 *  @author Ashwin K. Godbole
 */

#ifndef TRANSPOSE_UTILS_H
#define TRANSPOSE_UTILS_H

#include <string.h>

#define FUNC_(pre, type, post) pre##_##type##_##post
#define FUNC(pre, type, post) FUNC_(pre, type, post)

#define CONCAT_(x, y) x##y
#define CONCAT(x, y) CONCAT_(x, y)

#define UNIT_STRIDE 1

#define LINEAR_IDX_2D(i, j, column_stride, row_stride)                         \
    (((i) * (row_stride)) + ((j) * (column_stride)))

#define TRANSPOSE_KERNEL_ARGS                                                  \
    VOID *in_ptr, VOID *out_ptr, aoclfftz_dim_t_64_ row_metadata,              \
    aoclfftz_dim_t_64_ column_metadata, aoclfftz_transpose_aux_mem_t *aux_mem

// Assume that all the elements of the matrix are given numbers. such that
//      (0, 0) -> 0, (0, 1) -> 1, (0, 2) -> 2, ...
// This macro retrieves the linear index of the Nth element in the matrix
#define NTH_ELEMS_LINEAR_IDX_2D(N, cols, stride, leading_dim)                  \
    LINEAR_IDX_2D(((N) / (cols)), ((N) % (cols)), stride, leading_dim)

// Copy 'n_elems' number of elements from 'matrix' to 'buffer'
#define COPY_MATRIX_ELEMS(TYPE, matrix, m_i, m_j, m_leading_dim, buffer, b_i,  \
                          b_j, b_leading_dim, n_elems)                         \
    do                                                                         \
    {                                                                          \
        TYPE *write_loc =                                                      \
            &buffer[LINEAR_IDX_2D(b_i, b_j, UNIT_STRIDE, b_leading_dim)];      \
        TYPE *read_loc =                                                       \
            &matrix[LINEAR_IDX_2D(m_i, m_j, UNIT_STRIDE, m_leading_dim)];      \
        memcpy(write_loc, read_loc, sizeof(TYPE) * n_elems);                   \
    } while (0)

// Process the last row/column that remains after the recursive transpose of a
// square matrix with odd dimensions
//
// - If 'cols' is an odd number then one last row+column remains untransposed
//
// - For submatrices A and D, the vertical arm of the L-shaped area is swapped
//   with the horizontal arm (because A and D lie on the main diagonal)
//
// - For submatrix B, when the vertical arm of the L-shaped area is processed,
//   its elements get swapped with C's horizontal arm
//
// - This means that the elements of the horizontal arm of the L-shaped area of
//   submatrix B remain untransposed.
//
// - To fix this, we must also process the horizontal arm of the L-shaped area
//   of submatrix B (which gets swapped with the vertical arm of submatrix C)
//
//    ████████ █
//    ████████ █
//    ████████ █ <--- 'L' shaped untransposed area
//    ████████ █
//    ▄▄▄▄▄▄▄▄▄█

#define TRANSPOSE_LAST_ROWCOL(TYPE, matrix, cols, leading_dim, i, j,           \
                              in_top_right)                                    \
    {                                                                          \
        for (INTP it = 0; it < cols - 1; it++)                                 \
        {                                                                      \
            TYPE temp = matrix[LINEAR_IDX_2D(i + it, j + cols - 1,             \
                                             UNIT_STRIDE, leading_dim)];       \
            matrix[LINEAR_IDX_2D(i + it, j + cols - 1, UNIT_STRIDE,            \
                                 leading_dim)] =                               \
                matrix[LINEAR_IDX_2D(j + cols - 1, i + it, UNIT_STRIDE,        \
                                     leading_dim)];                            \
            matrix[LINEAR_IDX_2D(j + cols - 1, i + it, UNIT_STRIDE,            \
                                 leading_dim)] = temp;                         \
        }                                                                      \
        if (in_top_right)                                                      \
        {                                                                      \
            for (INTP it = 0; it <= cols - 1; it++)                            \
            {                                                                  \
                TYPE temp = matrix[LINEAR_IDX_2D(i + cols - 1, j + it,         \
                                                 UNIT_STRIDE, leading_dim)];   \
                matrix[LINEAR_IDX_2D(i + cols - 1, j + it, UNIT_STRIDE,        \
                                     leading_dim)] =                           \
                    matrix[LINEAR_IDX_2D(j + it, i + cols - 1, UNIT_STRIDE,    \
                                         leading_dim)];                        \
                matrix[LINEAR_IDX_2D(j + it, i + cols - 1, UNIT_STRIDE,        \
                                     leading_dim)] = temp;                     \
            }                                                                  \
        }                                                                      \
    }

#endif // TRANSPOSE_UTILS_H
