// Copyright Advanced Micro Devices, Inc.
// SPDX-License-Identifier: BSD-3-Clause

/** @file transpose_solver.c
 *
 *  @brief A solver that provides standalone transpose functionality
 *
 *  This file contains the functions that setup, execute and destroy
 *  the solver.
 *
 *  @author Ashwin K. Godbole
 */

#include "api/aoclfftz_internal.h"
#include "core/solvers/solver.h"
#include "core/solvers/transpose_solver.h"
#include "core/common/memory_manager.h"
#include "core/kernels/non_dft/transpose/transpose_kernels.h"

// Returns a pointer to the best suited transpose kernel for this problem
aoclfftz_transpose_kernel
get_transpose_kernel(aoclfftz_transpose_dtype type,
                     aoclfftz_dim_t_64_ row_metadata,
                     aoclfftz_dim_t_64_ column_metadata, FFTZ_UINT8 is_inplace,
                     FFTZ_UINT8 is_square)
{
    aoclfftz_transpose_kernel kernel = NULL;

    if (is_inplace)
    {
        if (is_square)
        {
            FFTZ_INTP selector_rec_mindim = 0;
            FFTZ_INTP kernel_rec_mindim = 0;
            SET_VAR(type, SEL_REC_MINDIM_, selector_rec_mindim);
            SET_VAR(type, REC_MIN_, kernel_rec_mindim);

            if (column_metadata.in_stride == 1)
            {
                // if the number of columns of the square matrix exceeds a
                // certain "upper bound", we determine that the matrix is best
                // transposed using the recursive algorithm.

                // also, since the recursive algorithm works well for matrices
                // whose number of columns is a power of 2, we add that
                // condition as well.

                if (column_metadata.n > selector_rec_mindim ||
                    (column_metadata.n > kernel_rec_mindim &&
                     IS_POW2(column_metadata.n)))
                {
                    SET_FNPTR(type, kernel, tiq_recursive_buf, c);
                    return kernel;
                }
                else
                {
                    SET_FNPTR(type, kernel, tiq_iterative, c);
                    return kernel;
                }
            }
            else // any strided
            {
                SET_FNPTR(type, kernel, tisq_iterative, c);
                return kernel;
            }
        }
        else // rectangle
        {
            if (column_metadata.in_stride == 1)
            {
                SET_FNPTR(type, kernel, tir_cycles, c);
                return kernel;
            }
            else
            {
                // default to using the revised cycles algorithm for now.
                SET_FNPTR(type, kernel, tisr_cycles, c);
                return kernel;
            }
        }
    }
    else // out of place
    {
        SET_FNPTR(type, kernel, tos_blocked, c);
        return kernel;
    }

    return kernel;
}
// -----------------------------------------------------------------------------

FFTZ_INT32 setup_transpose_solver(aoclfftz_solution_t *sol)
{
    AOCLFFTZ_LOG(TRACE, global_logger_mode, "Enter");

    // setup all the info
    aoclfftz_transpose_t *transpose = sol->dft_bufs->transpose;

    // Get the datatype of the incoming data
    aoclfftz_transpose_dtype dtype =
        (aoclfftz_transpose_dtype)(
            ((DT_PRECISION_FLAG(sol->decomp_scheme->flags) & 3) << 1) |
            ((IS_REAL(sol->decomp_scheme->flags) != 0)));

    transpose->col_info = sol->decomp_scheme->dims[0];
    transpose->row_info = sol->decomp_scheme->dims[1];

    if (transpose->row_info.in_stride !=
        (transpose->col_info.in_stride * transpose->col_info.n))
    {
        AOCLFFTZ_LOG(TRACE, global_logger_mode, "Exit (failure) : "
                                                     "row stride != column "
                                                     "stride * n");

        return SOLVER_FAILURE;
    }

    FFTZ_INT64 is_square = (transpose->row_info.n == transpose->col_info.n);
    if (! is_square)
    {
        transpose->aux_mem->size =
            transpose->row_info.n * transpose->col_info.n;
        ALLOC_ALIGN_UNINIT(transpose->aux_mem->data, FFTZ_UINT8,
                           transpose->aux_mem->size);
        if (transpose->aux_mem->data == NULL)
        {
            AOCLFFTZ_LOG(TRACE, global_logger_mode, "Exit (failure) : "
                                                    "aux_mem allocation "
                                                    "failed");
            return AOCLFFTZ_MEMORY_FAILURE;
        }
    }

    // Save the required kernel for the transpose operation
    transpose->kernel = get_transpose_kernel(
        dtype, transpose->row_info, transpose->col_info,
        !IS_OUT_OF_PLACE(sol->decomp_scheme->flags),
        is_square);

    AOCLFFTZ_LOG(TRACE, global_logger_mode, "Exit");
    return SOLVER_SUCCESS;
}

static FFTZ_INT32 execute_transpose_solver(aoclfftz_solution_t *sol,
                                           aoclfftz_mutable_ctx_t *ctx)
{
    AOCLFFTZ_LOG(TRACE, global_logger_mode, "Enter");


    if (sol->dft_bufs->transpose->aux_mem->size > 0)
    {
        memset(sol->dft_bufs->transpose->aux_mem->data, 0,
               sol->dft_bufs->transpose->aux_mem->size * sizeof(FFTZ_UINT8));
    }

    sol->dft_bufs->transpose->kernel((FFTZ_VOID *)ctx->in_real,
                                     (FFTZ_VOID *)ctx->out_real,
                                     sol->dft_bufs->transpose->row_info,
                                     sol->dft_bufs->transpose->col_info,
                                     sol->dft_bufs->transpose->aux_mem);

    AOCLFFTZ_LOG(TRACE, global_logger_mode, "Exit");
    return 0;
}

dft_solver_ register_execute_transpose_solver(void)
{
    return execute_transpose_solver;
}
