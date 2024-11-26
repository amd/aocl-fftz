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

#endif // TRANSPOSE_UTILS_H
