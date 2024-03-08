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

/** @file dims_vecs_helper.c
 *
 *  @brief Helper functions for dims and vecs.
 *
 *  This file contains the helper functions for dims and vecs for test bench.
 *
 *  @author S. Biplab Raut
 *  @author V. Murugan
 *  @author Srirammaswamy Srinivasan
 */

#include <stdio.h>
#include "dims_vecs_helper.h"

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
    ALLOC_ALIGN_INIT((*dims),
        aoclfftz_dim_t_64_, dim_rank * sizeof(aoclfftz_dim_t_64_));
    ALLOC_ALIGN_INIT((*vecs),
        aoclfftz_dim_t_64_, vec_rank * sizeof(aoclfftz_dim_t_64_));
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
                (*vecs)[i].in_stride =
                    (is_stride >= 1) ? desc[j].in_stride : 0;
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
            status =  SIZE_PARSING_ERROR;
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

    // validate & set strides for dims if not provided
    for (INT32 i = 0; i < dim_rank; i++)
    {
        INTP min_stride = (i == 0) ?
                1 : ((*dims)[i - 1]. n * (*dims)[i - 1].in_stride);
        if ((*dims)[i].in_stride == 0)
        {
            (*dims)[i].in_stride = min_stride;
        }
        else if ((*dims)[i].in_stride < min_stride)
        {
            printf("Invalid in stride value : %td provided for (%d) dim."
                    "minimum value expected : %td\n",
                    (*dims)[i].in_stride, i+1, min_stride);
            status = SIZE_PARSING_ERROR;
            goto exit_func;
        }
        min_stride = (i == 0) ?
                1 : ((*dims)[i - 1]. n * (*dims)[i - 1].out_stride);
        if ((*dims)[i].out_stride == 0)
        {
            (*dims)[i].out_stride = min_stride;
        }
        else if ((*dims)[i].out_stride < min_stride)
        {
            printf("Invalid out stride value : %td provided for (%d) dim."
                    "minimum value expected : %td\n",
                    (*dims)[i].out_stride, i+1, min_stride);
            status = SIZE_PARSING_ERROR;
            goto exit_func;
        }
    }

    // validate & set strides for vecs if not provided
    for (INT32 i = 0; i < vec_rank; i++)
    {
        INTP min_stride = (i == 0) ?
                (*dims)[dim_rank -1].n * (*dims)[dim_rank - 1].in_stride :
                    ((*vecs) [i - 1]. n * (*vecs) [i - 1].in_stride);
        if ((*vecs)[i].in_stride == 0)
        {
            // stride of fcd should atleast be the length of dims
            (*vecs)[i].in_stride = min_stride;
        }
        else if ((*vecs)[i].in_stride < min_stride)
        {
            printf("Invalid in stride value : %td provided for (%d) vec."
                    "minimum value expected : %td\n",
                    (*vecs)[i].in_stride, i+1, min_stride);
            status = SIZE_PARSING_ERROR;
            goto exit_func;
        }
        min_stride = (i == 0) ?
                (*dims)[dim_rank -1].n * (*dims)[dim_rank - 1].out_stride :
                    ((*vecs) [i - 1]. n * (*vecs) [i - 1].out_stride);
        if ((*vecs)[i].out_stride == 0)
        {
            // stride of fcd should atleast be the length of dims
            (*vecs)[i].out_stride = min_stride;
        }
        else if ((*vecs)[i].out_stride < min_stride)
        {
            printf("Invalid out stride value : %td provided for (%d) vec."
                    "minimum value expected : %td\n",
                    (*vecs)[i].out_stride, i+1, min_stride);
            status = SIZE_PARSING_ERROR;
            goto exit_func;
        }
    }

    // Initialize vector size to default value when no vector is encountered.
    if (vec_count == 0)
    {
        (*vecs)[0].n = 1;
        (*vecs)[0].in_stride = default_stride;
        (*vecs)[0].out_stride = default_stride;
    }

exit_func :
    FREE_ALIGN_ALLOCATED_MEM(desc);
    return status;
}
