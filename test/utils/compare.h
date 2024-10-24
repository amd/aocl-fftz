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

/** @file compare.h
 *
 *  @brief Data compare functions.
 *
 *  This file contains declarations of data compare functions for test bench.
 *
 *  @author Srirammaswamy Srinivasan
 *  @author Jeya R
 */

#ifndef COMPARE_H
#define COMPARE_H

#ifdef WIN32
#include <direct.h> /* for _getcwd */
#endif
#include "test/aoclfftz_bench.h"
#include "utils/allocator.h"
#include "utils/utils.h"
#include "test/utils/register_functions.h"


#define OUTPUT_LOG_FILE "output_dump.txt"

#ifdef WIN32
#define SSCANF sscanf_s
#define STRCPY(dst, dst_size, src) strcpy_s(dst, dst_size, src)
#define FOPEN(file_pointer, file_name, open_mode)   \
    fopen_s(&file_pointer, file_name, open_mode)
#define GETCWD(buffer, size) _getcwd(buffer, size)
#define DIRECTORY_SEPARATOR "\\"
#else
#define SSCANF sscanf
#define STRCPY(dst, dst_size, src) strcpy(dst, src)
#define FOPEN(file_pointer, file_name, open_mode)   \
    (file_pointer = fopen(file_name, open_mode))
#define GETCWD(buffer, size) getcwd(buffer, size)
#define DIRECTORY_SEPARATOR "/"
#endif


/**
 * @brief increment the nD counter by 1 value, used to map 1D index to nD
 *
 */
#define INCREMENT_ND_COUNTER(cur_dims, max_dims, rank)                         \
    {                                                                          \
        UINT8 incremented = 0;                                                 \
        INTP cur_rank = 0;                                                     \
        do                                                                     \
        {                                                                      \
            if (cur_dims[cur_rank] < max_dims[cur_rank].n - 1)                 \
            {                                                                  \
                cur_dims[cur_rank]++;                                          \
                incremented = 1;                                               \
            }                                                                  \
            else                                                               \
            {                                                                  \
                for (INTP i = 0; i <= cur_rank; i++)                           \
                {                                                              \
                    cur_dims[i] = 0;                                           \
                }                                                              \
                cur_rank++;                                                    \
            }                                                                  \
        } while (!incremented && cur_rank < rank);                             \
    }

/**
 * @brief reset the nD counter values to 0
 *
 */
#define RESET_ND_COUNTER(cur_dims, rank)                                       \
    {                                                                          \
        for (INT32 i = 0; i < rank; i++)                                       \
        {                                                                      \
            cur_dims[i] = 0;                                                   \
        }                                                                      \
    }

/**
 * @brief copy the nD counter from src to dst
 *
 */
#define COPY_ND_COORDS(dst, src, rank)                                         \
    {                                                                          \
        for (INTP i = 0; i < rank; i++)                                        \
        {                                                                      \
            dst[i] = src[i];                                                   \
        }                                                                      \
    }

/**
 * @brief print the nD index (with nD batches) to stdout
 *
 */
#define PRINT_ND_COUNTER(dims, vecs, dim_rank, vec_rank)                       \
    {                                                                          \
        /* vecs */                                                             \
        printf("[");                                                           \
        for (INT32 j = vec_rank - 1; j >= 0; j--)                              \
        {                                                                      \
            if (j < vec_rank - 1)                                              \
                printf(",");                                                   \
            printf("%td", vecs[j]);                                            \
        }                                                                      \
        printf("]v[");                                                         \
        /* dims */                                                             \
        for (INT32 j = dim_rank - 1; j >= 0; j--)                              \
        {                                                                      \
            if (j < dim_rank - 1)                                              \
                printf(",");                                                   \
            printf("%td", dims[j]);                                            \
        }                                                                      \
        printf("]");                                                           \
    }

/**
 * @brief print the nD index (with nD batches) to file
 *
 */
#define PRINT_ND_COUNTER_TO_FILE(out_file, dims, vecs, dim_rank, vec_rank)     \
    {                                                                          \
        /* vecs */                                                             \
        fprintf(out_file, "[");                                                \
        for (INT32 j = vec_rank - 1; j >= 0; j--)                              \
        {                                                                      \
            if (j < vec_rank - 1)                                              \
                fprintf(out_file, ",");                                        \
            fprintf(out_file, "%td", vecs[j]);                                 \
        }                                                                      \
        fprintf(out_file, "]v[");                                              \
        /* dims */                                                             \
        for (INT32 j = dim_rank - 1; j >= 0; j--)                              \
        {                                                                      \
            if (j < dim_rank - 1)                                              \
                fprintf(out_file, ",");                                        \
            fprintf(out_file, "%td", dims[j]);                                 \
        }                                                                      \
        fprintf(out_file, "]");                                                \
    }


INT32 compare_f(aoclfftz_bench_params_t *params, VOID *a, VOID *b, INTP batches,
                INTP n, INTP *idx_map);
INT32 compare_d(aoclfftz_bench_params_t *params, VOID *a, VOID *b, INTP batches,
                INTP n, INTP *idx_map);

#endif // COMPARE_H
