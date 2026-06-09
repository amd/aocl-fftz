// Copyright Advanced Micro Devices, Inc.
// SPDX-License-Identifier: BSD-3-Clause

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
#include "test/bench/aoclfftz_bench.h"
#include "utils/allocator.h"
#include "utils/utils.h"
#include "test/bench/utils/register_functions.h"


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
        for (INT32 i = 0; i < rank; i++)                                       \
        {                                                                      \
            if (++cur_dims[i] < max_dims[i].n) {                               \
                break;                                                         \
            }                                                                  \
            cur_dims[i] = 0;                                                   \
        }                                                                      \
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
        fprintf(stderr, "[");                                                  \
        for (INT32 j = vec_rank - 1; j >= 0; j--)                              \
        {                                                                      \
            if (j < vec_rank - 1)                                              \
                fprintf(stderr, ",");                                          \
            fprintf(stderr, "%td", vecs[j]);                                   \
        }                                                                      \
        fprintf(stderr, "]v[");                                                \
        /* dims */                                                             \
        for (INT32 j = dim_rank - 1; j >= 0; j--)                              \
        {                                                                      \
            if (j < dim_rank - 1)                                              \
                fprintf(stderr, ",");                                          \
            fprintf(stderr, "%td", dims[j]);                                   \
        }                                                                      \
        fprintf(stderr, "]");                                                  \
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
                INTP n, INTP *a_map, INTP *b_map, INT32 data_stride);
INT32 compare_d(aoclfftz_bench_params_t *params, VOID *a, VOID *b, INTP batches,
                INTP n, INTP *a_map, INTP *b_map, INT32 data_stride);

#endif // COMPARE_H
