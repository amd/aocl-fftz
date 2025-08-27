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

/** @file dims_vecs_helper.c
 *
 *  @brief Helper functions for dims and vecs.
 *
 *  This file contains the helper functions for dims and vecs for the
 *  test bench and gtest.
 *
 *  @author S. Biplab Raut
 *  @author V. Murugan
 *  @author Srirammaswamy Srinivasan
 */

#include "dims_vecs_helper.h"
#include "utils/utils.h"

/**
 * @brief find the rank of the dimensions from the given string argument
 *
 * @param arg dimension string
 * @param dim_rank reference variable to store the dim-rank
 * @param vec_rank reference variable to store the vec-rank
 * @return INT32 status code
 */

// FIXME : this needs to be modified, club with allocate or exploit this result
INT32 find_dim_vec_ranks(CHAR *arg, INT32 *dim_rank, INT32 *vec_rank)
{
    INT32 dr = 1;
    INT32 vr = 1; // can we retain this as O when no vec ?
    for (INT32 i = 0; i < strlen(arg); i++)
    {
        if (arg[i] == 'x')
        {
            dr++;
        }
        else if (arg[i] == 'v')
        {
            vr = dr;
            dr = 1;
        }
    }
    (*dim_rank) = dr;
    (*vec_rank) = vr;
    return PARSER_SUCCESS;
}

/**
 * @brief allocate and fill the values of dims and vecs parsed from the
 * string argument.
 *
 * @param arg dimension string
 * @param dim_rank rank of the dimensions
 * @param vec_rank rank of the vectors
 * @param dims pointer to store the dims structure
 * @param vecs pointer to stroe the vecs structure
 * @param default_stride default stride value to be used for dims and vecs
 * @return INT32
 */
INT32 allocate_and_fill_dims_vecs(CHAR *arg, INT32 dim_rank, INT32 vec_rank,
                                  aoclfftz_dim_t_64_ **dims,
                                  aoclfftz_dim_t_64_ **vecs,
                                  INTP default_stride)
{
    ALLOC_ALIGN_INIT((*dims), aoclfftz_dim_t_64_,
                     dim_rank * sizeof(aoclfftz_dim_t_64_));
    ALLOC_ALIGN_INIT((*vecs), aoclfftz_dim_t_64_,
                     vec_rank * sizeof(aoclfftz_dim_t_64_));
    INT32 max_rank = dim_rank > vec_rank ? dim_rank : vec_rank;
    aoclfftz_dim_t_64_ *desc = NULL;
    ALLOC_ALIGN_INIT(desc, aoclfftz_dim_t_64_,
                     max_rank * sizeof(aoclfftz_dim_t_64_));

    INT32 is_stride = 0;
    INT32 rank_count = 0;
    INT32 vec_count = 0;
    INT32 start = 0;
    CHAR val_str[strlen(arg) + 1];
    INT32 status = PARSER_SUCCESS;
    for (INT32 i = 0; i < strlen(arg); ++i)
    {
        if (arg[i] == 'x' || arg[i] == 'X')
        {
            if (i + 1 >= strlen(arg) || !isdigit(arg[i + 1]))
            {
                printf("Integer value expected after 'x' character.\n");
                status = SIZE_PARSING_ERROR;
                goto exit_func;
            }
            if ((is_stride != 0 && is_stride != 2))
            {
                printf("Only in_stride is not accepted. "
                       "Please pass both in & out strides (or) no strides.\n");
                status = SIZE_PARSING_ERROR;
                goto exit_func;
            }
            is_stride = 0;
            rank_count++;
        }
        else if (arg[i] == 'v' || arg[i] == 'V')
        {
            if (i + 1 >= strlen(arg) || !isdigit(arg[i + 1]))
            {
                printf("Integer value expected after 'v' character.\n");
                status = SIZE_PARSING_ERROR;
                goto exit_func;
            }
            // by default the data is stored in desc always
            // once "v" is encountered, its moved to vecs and then desc is reset
            // FIXME : this needs to be fixed properly
            for (INT32 i = vec_rank - 1, j = 0; i >= 0; i--, j++)
            {
                (*vecs)[i].n = desc[j].n;
                (*vecs)[i].in_stride = (is_stride >= 1) ? desc[j].in_stride : 0;
                (*vecs)[i].out_stride =
                    (is_stride == 2) ? desc[j].out_stride : 0;
            }
            // reset buffer to store dims config
            memset(desc, 0, max_rank * sizeof(aoclfftz_dim_t_64_));
            rank_count = 0;
            is_stride = 0;
            vec_count++;
        }
        else if (arg[i] == ':')
        {
            if (i + 1 >= strlen(arg) || !isdigit(arg[i + 1]))
            {
                printf("Integer value expected after ':' character.\n");
                status = SIZE_PARSING_ERROR;
                goto exit_func;
            }
            is_stride++;
        }
        else if (isdigit(arg[i]))
        {
            start = i;
            val_str[0] = arg[i];
            while (isdigit(arg[++i]))
            {
                val_str[i - start] = arg[i];
            }
            val_str[i - start] = '\0';
            INTP val = atol(val_str);
            if (val < 0)
            {
                status = SIZE_PARSING_ERROR;
                goto exit_func;
            }
            if (is_stride == 0)
            {
                if (val == 0)
                {
                    printf("Invalid dim/vec size (zero) at rank : %d",
                           rank_count);
                    status = SIZE_PARSING_ERROR;
                    goto exit_func;
                }
                desc[rank_count].n = val;
            }
            else if (is_stride == 1)
            {
                desc[rank_count].in_stride = val;
            }
            else if (is_stride == 2)
            {
                desc[rank_count].out_stride = val;
            }
            i--;
        }
        else
        {
            status = SIZE_PARSING_ERROR;
            goto exit_func;
        }
    }

    // copy desc to dims in reverse
    for (INT32 i = dim_rank - 1, j = 0; i >= 0; i--, j++)
    {
        (*dims)[i].n = desc[j].n;
        (*dims)[i].in_stride = desc[j].in_stride;
        (*dims)[i].out_stride = desc[j].out_stride;
    }

    // Initialize vector to default value when no vector is encountered.
    if (vec_count == 0)
    {
        (*vecs)[0].n = 1;
        (*vecs)[0].in_stride = default_stride;
        (*vecs)[0].out_stride = default_stride;
    }

exit_func:
    FREE_ALIGN_ALLOCATED_MEM(desc);
    return status;
}

/**
 * @brief Set default stride values for dims and vecs.
 *
 * This function calculates appropriate default stride values for dims and vecs
 * when they are not explicitly provided. The stride calculations vary based on
 * the FFT type and whether the operation is in-place or out-of-place.
 *
 * Stride Calculation Rules:
 *
 * For dims:
 * - dims[0]: Unit stride (1) for both input and output
 * - dims[1]: Stride calculations based on FFT type:
 *   * consider, n = n, is = in_stride, os = out_stride of dims[0]
 *   ------------------|--------------------|--------------------
 *    Type             | in_stride          | out_stride
 *   ------------------|--------------------|--------------------
 *    C2C              | n * is             | n * os
 *    R2C out-of-place | n * is             | (n/2 + 1) * os
 *    R2C in-place     | (n/2 + 1) * is * 2 | (n/2 + 1) * os     (where is = os)
 *    C2R out-of-place | (n/2 + 1) * is     | n * os
 *    C2R in-place     | (n/2 + 1) * is     | (n/2 + 1) * os * 2 (where is = os)
 *   ------------------|--------------------|--------------------
 * - dims[2] or above: dims[i-1].n * dims[i-1].stride
 *                     (where stride = in_stride or out_stride)
 *
 * For vecs:
 * - vecs[0]: This is same as dims[1] configuration
 *            where, n = n, is = in_stride, os = out_stride of dims[dim_rank-1]
 * - vecs[1] or above: vecs[i-1].n * vecs[i-1].stride
 *                     (where stride = in_stride or out_stride)
 *
 * @param dim_rank rank of the dimensions
 * @param vec_rank rank of the vectors
 * @param dims dims structure to set default strides
 * @param vecs vecs structure to set default strides
 * @param type fft_type -> 0: C2C, 1: R2C, 2: C2R
 * @param is_in_place 1: in-place, 0: out-of-place
 * @return VOID
 */
VOID set_default_dims_vecs(INT32 dim_rank, INT32 vec_rank,
                            aoclfftz_dim_t_64_ *dims, aoclfftz_dim_t_64_ *vecs,
                            aoclfftz_bench_fft_type_t type, UINT8 is_in_place,
                            UINT8 logger_mode)
{
    // Set default strides for dims if not explicitly provided
    for (INT32 i = 0; i < dim_rank; i++)
    {
        INTP def_in_stride = 0;
        INTP def_out_stride = 0;
        if (i == 0)
        {
            def_in_stride = 1;
            def_out_stride = 1;
        }
        else if (i == 1)
        {
            if (type == R2C && is_in_place)
            {
                def_in_stride = (dims[0].n / 2 + 1) * dims[0].in_stride * 2;
                def_out_stride = (dims[0].n / 2 + 1) * dims[0].out_stride;
            }
            else if (type == R2C && !is_in_place)
            {
                def_in_stride = dims[0].n * dims[0].in_stride;
                def_out_stride = (dims[0].n / 2 + 1) * dims[0].out_stride;
            }
            else if (type == C2R && is_in_place)
            {
                def_in_stride = (dims[0].n / 2 + 1) * dims[0].in_stride;
                def_out_stride = (dims[0].n / 2 + 1) * dims[0].out_stride * 2;
            }
            else if (type == C2R && !is_in_place)
            {
                def_in_stride = (dims[0].n / 2 + 1) * dims[0].in_stride;
                def_out_stride = dims[0].n * dims[0].out_stride;
            }
            else /* C2C */
            {
                def_in_stride = dims[0].n * dims[0].in_stride;
                def_out_stride = dims[0].n * dims[0].out_stride;
            }
        }
        else /* i > 1 */
        {
            def_in_stride = dims[i - 1].n * dims[i - 1].in_stride;
            def_out_stride = dims[i - 1].n * dims[i - 1].out_stride;
        }
        if (dims[i].in_stride == 0)
        {
            dims[i].in_stride = def_in_stride;
            AOCLFFTZ_LOG_FORMATTED(
                INFO, logger_mode,
                "in stride for dim[%d] is set to default value", i);
        }
        if (dims[i].out_stride == 0)
        {
            dims[i].out_stride = def_out_stride;
            AOCLFFTZ_LOG_FORMATTED(
                INFO, logger_mode,
                "out stride for dim[%d] is set to default value", i);
        }
    }

    // set strides for vecs if not provided
    for (INT32 i = 0; i < vec_rank; i++)
    {
        INTP def_in_stride = 0;
        INTP def_out_stride = 0;
        if (i == 0)
        {
            aoclfftz_dim_t_64_ last_dim = dims[dim_rank - 1];
            if (type == R2C && is_in_place)
            {
                def_in_stride = (last_dim.n / 2 + 1) * last_dim.in_stride * 2;
                def_out_stride = (last_dim.n / 2 + 1) * last_dim.out_stride;
            }
            else if (type == R2C && !is_in_place)
            {
                def_in_stride = last_dim.n * last_dim.in_stride;
                def_out_stride = (last_dim.n / 2 + 1) * last_dim.out_stride;
            }
            else if (type == C2R && is_in_place)
            {
                def_in_stride = (last_dim.n / 2 + 1) * last_dim.in_stride;
                def_out_stride = (last_dim.n / 2 + 1) * last_dim.out_stride * 2;
            }
            else if (type == C2R && !is_in_place)
            {
                def_in_stride = (last_dim.n / 2 + 1) * last_dim.in_stride;
                def_out_stride = last_dim.n * last_dim.out_stride;
            }
            else /* C2C */
            {
                def_in_stride = last_dim.n * last_dim.in_stride;
                def_out_stride = last_dim.n * last_dim.out_stride;
            }
        }
        else /* i > 0 */
        {
            def_in_stride = vecs[i - 1].n * vecs[i - 1].in_stride;
            def_out_stride = vecs[i - 1].n * vecs[i - 1].out_stride;
        }
        if (vecs[i].in_stride == 0)
        {
            AOCLFFTZ_LOG_FORMATTED(
                INFO, logger_mode,
                "in stride for vec[%d] is set to default value", i);
            vecs[i].in_stride = def_in_stride;
        }
        if (vecs[i].out_stride == 0)
        {
            AOCLFFTZ_LOG_FORMATTED(
                INFO, logger_mode,
                "out stride for vec[%d] is set to default value", i);
            vecs[i].out_stride = def_out_stride;
        }
    }
}
