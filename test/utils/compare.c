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

/** @file compare.c
 *
 *  @brief Data compare functions.
 *
 *  This file contains declarations of data compare functions for test bench.
 *
 *  @author Srirammaswamy Srinivasan
 */

#include <string.h>
#include <math.h>
#include <stdio.h>
#include "test/utils/compare.h"
/**
 * @brief Compare the two data of length n (FLOAT type)
 *        Used to compare output with reference output.
 *
 * @param params aoclfftz_bench_params_t struct containing the req info
 * @param a first data
 * @param b second data
 * @param batches batch/vec size
 * @param n problem size
 * @param idx_map index map
 * @return INT32 if the data points are same, return 1 else 0
 */
INT32 compare_f(aoclfftz_bench_params_t *params, VOID *a, VOID *b, INTP batches,
                INTP n, INTP *idx_map)
{
    DOUBLE tol = params->tolerance;
    INT32 logger_mode = params->logger_mode;
    FLOAT *a_f = (FLOAT *)a;
    FLOAT *b_f = (FLOAT *)b;
    INT32 status = BENCH_SUCCESS;

    INT32 dim_rank = params->dim_rank;
    aoclfftz_dim_t_64_ *dims = params->dims;
    INT32 vec_rank = params->vec_rank;
    aoclfftz_dim_t_64_ *vecs = params->vecs;
    INTP *dim_counter = NULL;
    INTP *vec_counter = NULL;
    UINT32 is_aligned = params->aligned_alloc;
    ALLOC_INIT(dim_counter, INTP, dim_rank * sizeof(INTP), is_aligned);
    ALLOC_INIT(vec_counter, INTP, vec_rank * sizeof(INTP), is_aligned);
    if (dim_counter == NULL || vec_counter == NULL)
    {
        return MEMORY_FAILURE;
    }
    FLOAT max_abs_err = 0.0;
    FLOAT max_mag = 0.0;
    INTP max_err_idx = -1;
    INTP first_err_idx = INT64_MAX;
    FLOAT first_abs_err = 0.0;
    INTP *d_maxerr_coords = NULL;
    INTP *d_err_coords = NULL;
    INTP *b_maxerr_coords = NULL;
    INTP *b_err_coords = NULL;
    ALLOC_INIT(d_maxerr_coords, INTP, dim_rank * sizeof(INTP), is_aligned);
    ALLOC_INIT(d_err_coords, INTP, dim_rank * sizeof(INTP), is_aligned);
    ALLOC_INIT(b_maxerr_coords, INTP, vec_rank * sizeof(INTP), is_aligned);
    ALLOC_INIT(b_err_coords, INTP, vec_rank * sizeof(INTP), is_aligned);
    if (d_maxerr_coords == NULL || d_err_coords == NULL ||
        b_maxerr_coords == NULL || b_err_coords == NULL)
    {
        return MEMORY_FAILURE;
    }
    INTP N = batches * n;
    for (INTP i = 0; i < N; i++)
    {
        INTP idx = idx_map[i];
        FLOAT abs_err = fmaxf(
            fabsf(a_f[idx * T_DATA_STRIDE] - b_f[idx * T_DATA_STRIDE]),
            fabsf(a_f[idx * T_DATA_STRIDE + 1] - b_f[idx * T_DATA_STRIDE + 1]));
        FLOAT mag = fminf(fmaxf(fabsf(a_f[idx * T_DATA_STRIDE]),
                              fabsf(a_f[idx * T_DATA_STRIDE + 1])),
                         fmaxf(fabsf(b_f[idx * T_DATA_STRIDE]),
                              fabsf(b_f[idx * T_DATA_STRIDE + 1])));
        if (abs_err > max_abs_err)
        {
            max_err_idx = idx;
            max_abs_err = abs_err;
            COPY_ND_COORDS(d_maxerr_coords, dim_counter, dim_rank);
            COPY_ND_COORDS(b_maxerr_coords, vec_counter, vec_rank);
            if (idx < first_err_idx && abs_err > tol)
            {
                first_err_idx = idx;
                first_abs_err = abs_err;
                COPY_ND_COORDS(d_err_coords, dim_counter, dim_rank);
                COPY_ND_COORDS(b_err_coords, vec_counter, vec_rank);
            }
        }
        if (mag > max_mag)
        {
            max_mag = mag;
        }
        INCREMENT_ND_COUNTER(dim_counter, dims, dim_rank);
        if (i % n == n - 1)
        {
            INCREMENT_ND_COUNTER(vec_counter, vecs, vec_rank);
        }
    }

    FLOAT rel_err;
    if (max_abs_err == 0.0 && max_mag == 0.0)
    {
        rel_err = 0.0;
    }
    else if (max_mag == 0.0)
    {
        rel_err = max_abs_err;
    }
    else
    {
        rel_err = max_abs_err / max_mag;
    }
#ifdef AOCL_ENABLE_LOG
    AOCLFFTZ_LOG_FORMATTED(INFO, logger_mode, "Error = %.6g", rel_err);
#endif
    if (rel_err > tol)
    {
        printf("Relative error  = %.6g\n", rel_err);
        printf("Tolerance       = %.6g\n", tol);
        if (first_err_idx < INT64_MAX)
        {
            printf("First absolute error at index %td -> ", first_err_idx);
            PRINT_ND_COUNTER(d_err_coords, b_err_coords, dim_rank,
                             vec_rank);
            printf("\n  expected = %.6g + %.6gj\n",
                   b_f[first_err_idx * T_DATA_STRIDE],
                   b_f[first_err_idx * T_DATA_STRIDE + 1]);
            printf("  got      = %.6g + %.6gj\n",
                   a_f[first_err_idx * T_DATA_STRIDE],
                   a_f[first_err_idx * T_DATA_STRIDE + 1]);
        }
        printf("  max abs error = %.6g\n", first_abs_err);
        printf("Max absolute error at index %td -> ", max_err_idx);
        PRINT_ND_COUNTER(d_maxerr_coords, b_maxerr_coords, dim_rank, vec_rank);
        printf("\n  expected = %.6g + %.6gj\n",
               b_f[max_err_idx * T_DATA_STRIDE],
               b_f[max_err_idx * T_DATA_STRIDE + 1]);
        printf("  got      = %.6g + %.6gj\n", a_f[max_err_idx * T_DATA_STRIDE],
               a_f[max_err_idx * T_DATA_STRIDE + 1]);
        printf("  max abs error = %.6g\n", max_abs_err);
        if (logger_mode >= DEBUG)
        {
            RESET_ND_COUNTER(dim_counter, dim_rank);
            RESET_ND_COUNTER(vec_counter, vec_rank);
            // Using printf instead of logger to avoid file and line prefix
            printf("\n\t%5s%26s%32s\n", "Index", "Expected", "Actual");
            for (INTP i = 0, c = 100; i < N && c > 0; i++, c--)
            {
                INTP idx = idx_map[i];
                printf("%7td -> ", idx);
                PRINT_ND_COUNTER(dim_counter, vec_counter, dim_rank, vec_rank);
                printf(" : %12.6f + %12.6fj  vs  %12.6f + %12.6fj\n",
                       a_f[idx * T_DATA_STRIDE], a_f[idx * T_DATA_STRIDE + 1],
                       b_f[idx * T_DATA_STRIDE], b_f[idx * T_DATA_STRIDE + 1]);

                INCREMENT_ND_COUNTER(dim_counter, dims, dim_rank);
                if (i % n == n - 1)
                {
                    INCREMENT_ND_COUNTER(vec_counter, vecs, vec_rank);
                }
            }
            RESET_ND_COUNTER(dim_counter, dim_rank);
            RESET_ND_COUNTER(vec_counter, vec_rank);
            // Write full output to a file
            FILE *out_file = NULL;
            FOPEN(out_file, OUTPUT_LOG_FILE, "w");
            fprintf(out_file, "\t%10s%30s%48s\n", "Index", "Expected",
                    "Actual");
            for (INTP i = 0; i < N; i++)
            {
                INTP idx = idx_map[i];
                fprintf(out_file, "%7td -> ", idx);
                PRINT_ND_COUNTER_TO_FILE(out_file, dim_counter, vec_counter,
                                         dim_rank, vec_rank);
                fprintf(out_file, " : %12.6f + %12.6fj  vs  %12.6f + %12.6fj\n",
                        a_f[idx * T_DATA_STRIDE], a_f[idx * T_DATA_STRIDE + 1],
                        b_f[idx * T_DATA_STRIDE], b_f[idx * T_DATA_STRIDE + 1]);

                INCREMENT_ND_COUNTER(dim_counter, dims, dim_rank);
                if (i % n == n - 1)
                {
                    INCREMENT_ND_COUNTER(vec_counter, vecs, vec_rank);
                }
            }
            fclose(out_file);
            CHAR path[PATH_SIZE_MAX];
            CHAR *ret = GETCWD(path, sizeof(path));
            if (ret == NULL)
            {
                STRCPY(path, PATH_SIZE_MAX, "current_dir");
            }
            printf("\nFull output log can be found in %s%s%s\n", path,
                    DIRECTORY_SEPARATOR, OUTPUT_LOG_FILE);
        }
        else
        {
            printf("\nUse debug logger mode [--logger-mode 3 (or) -l 3] to get "
                   "detailed error log\n");
        }
        status = VERIFICATION_FAILURE;
    }

    FREE_ALLOCATED_MEM(dim_counter, is_aligned);
    FREE_ALLOCATED_MEM(vec_counter, is_aligned);
    FREE_ALLOCATED_MEM(d_maxerr_coords, is_aligned);
    FREE_ALLOCATED_MEM(d_err_coords, is_aligned);
    FREE_ALLOCATED_MEM(b_maxerr_coords, is_aligned);
    FREE_ALLOCATED_MEM(b_err_coords, is_aligned);

    return status;
}

/**
 * @brief Compare the two data of length n (DOUBLE type)
 *        Used to compare output with reference output.
 *
 * @param params aoclfftz_bench_params_t struct containing the req info
 * @param a first data
 * @param b second data
 * @param batches batch/vec size
 * @param n problem size
 * @param idx_map index map
 * @return INT32 if the data points are same, return 1 else 0
 */
INT32 compare_d(aoclfftz_bench_params_t *params, VOID *a, VOID *b, INTP batches,
                INTP n, INTP *idx_map)
{
    DOUBLE tol = params->tolerance;
    INT32 logger_mode = params->logger_mode;
    DOUBLE *a_d = (DOUBLE *)a;
    DOUBLE *b_d = (DOUBLE *)b;
    INT32 status = BENCH_SUCCESS;
    INT32 dim_rank = params->dim_rank;
    aoclfftz_dim_t_64_ *dims = params->dims;
    INT32 vec_rank = params->vec_rank;
    aoclfftz_dim_t_64_ *vecs = params->vecs;
    INTP *dim_counter = NULL;
    INTP *vec_counter = NULL;
    UINT32 is_aligned = params->aligned_alloc;
    ALLOC_INIT(dim_counter, INTP, dim_rank * sizeof(INTP), is_aligned);
    ALLOC_INIT(vec_counter, INTP, vec_rank * sizeof(INTP), is_aligned);

    DOUBLE max_abs_err = 0.0;
    DOUBLE max_mag = 0.0;
    INTP max_err_idx = -1;
    INTP first_err_idx = INT64_MAX;
    DOUBLE first_abs_err = 0.0;
    INTP *d_maxerr_coords = NULL;
    INTP *d_err_coords = NULL;
    INTP *b_maxerr_coords = NULL;
    INTP *b_err_coords = NULL;
    ALLOC_INIT(d_maxerr_coords, INTP, dim_rank * sizeof(INTP), is_aligned);
    ALLOC_INIT(d_err_coords, INTP, dim_rank * sizeof(INTP), is_aligned);
    ALLOC_INIT(b_maxerr_coords, INTP, vec_rank * sizeof(INTP), is_aligned);
    ALLOC_INIT(b_err_coords, INTP, vec_rank * sizeof(INTP), is_aligned);

    INTP N = batches * n;
    for (INTP i = 0; i < N; i++)
    {
        INTP idx = idx_map[i];
        DOUBLE abs_err = fmax(
            fabs(a_d[idx * T_DATA_STRIDE] - b_d[idx * T_DATA_STRIDE]),
            fabs(a_d[idx * T_DATA_STRIDE + 1] - b_d[idx * T_DATA_STRIDE + 1]));
        DOUBLE mag = fmin(fmax(fabs(a_d[idx * T_DATA_STRIDE]),
                               fabs(a_d[idx * T_DATA_STRIDE + 1])),
                          fmax(fabs(b_d[idx * T_DATA_STRIDE]),
                               fabs(b_d[idx * T_DATA_STRIDE + 1])));
        if (abs_err > max_abs_err)
        {
            max_err_idx = idx;
            max_abs_err = abs_err;
            COPY_ND_COORDS(d_maxerr_coords, dim_counter, dim_rank);
            COPY_ND_COORDS(b_maxerr_coords, vec_counter, vec_rank);
            if (idx < first_err_idx && abs_err > tol)
            {
                first_err_idx = idx;
                first_abs_err = abs_err;
                COPY_ND_COORDS(d_err_coords, dim_counter, dim_rank);
                COPY_ND_COORDS(b_err_coords, vec_counter, vec_rank);
            }
        }
        if (mag > max_mag)
        {
            max_mag = mag;
        }
        INCREMENT_ND_COUNTER(dim_counter, dims, dim_rank);
        if (i % n == n - 1)
        {
            INCREMENT_ND_COUNTER(vec_counter, vecs, vec_rank);
        }
    }
    DOUBLE rel_err;
    if (max_abs_err == 0.0 && max_mag == 0.0)
    {
        rel_err = 0.0;
    }
    else if (max_mag == 0.0)
    {
        rel_err = max_abs_err;
    }
    else
    {
        rel_err = max_abs_err / max_mag;
    }
#ifdef AOCL_ENABLE_LOG
    AOCLFFTZ_LOG_FORMATTED(INFO, logger_mode, "Error = %.6g", rel_err);
#endif
    if (rel_err > tol)
    {
        printf("Relative error  = %.6g\n", rel_err);
        printf("Tolerance       = %.6g\n", tol);
        if (first_err_idx < INT64_MAX)
        {
            printf("First absolute error at index %td -> ", first_err_idx);
            PRINT_ND_COUNTER(d_err_coords, b_err_coords, dim_rank,
                             vec_rank);
            printf("\n  expected = %.6g + %.6gj\n",
                   b_d[first_err_idx * T_DATA_STRIDE],
                   b_d[first_err_idx * T_DATA_STRIDE + 1]);
            printf("  got      = %.6g + %.6gj\n",
                   a_d[first_err_idx * T_DATA_STRIDE],
                   a_d[first_err_idx * T_DATA_STRIDE + 1]);
        }
        printf("  max abs error = %.6g\n", first_abs_err);
        printf("Max absolute error at index %td -> ", max_err_idx);
        PRINT_ND_COUNTER(d_maxerr_coords, b_maxerr_coords, dim_rank, vec_rank);
        printf("\n  expected = %.6g + %.6gj\n",
               b_d[max_err_idx * T_DATA_STRIDE],
               b_d[max_err_idx * T_DATA_STRIDE + 1]);
        printf("  got      = %.6g + %.6gj\n", a_d[max_err_idx * T_DATA_STRIDE],
               a_d[max_err_idx * T_DATA_STRIDE + 1]);
        printf("  max abs error = %.6g\n", max_abs_err);
        if (logger_mode >= DEBUG)
        {
            RESET_ND_COUNTER(dim_counter, dim_rank);
            RESET_ND_COUNTER(vec_counter, vec_rank);
            // Using printf instead of logger to avoid file and line prefix
            printf("\n\t%10s%30s%48s\n", "Index", "Expected", "Actual");
            for (INTP i = 0, c = 100; i < N && c > 0; i++, c--)
            {
                INTP idx = idx_map[i];
                // vecs
                printf("%7td -> ", idx);
                PRINT_ND_COUNTER(dim_counter, vec_counter, dim_rank, vec_rank);
                printf(" : %20.14lf + %20.14lfj  vs  %20.14lf + %20.14lfj\n",
                       a_d[idx * T_DATA_STRIDE], a_d[idx * T_DATA_STRIDE + 1],
                       b_d[idx * T_DATA_STRIDE], b_d[idx * T_DATA_STRIDE + 1]);

                INCREMENT_ND_COUNTER(dim_counter, dims, dim_rank);
                if (i % n == n - 1)
                {
                    INCREMENT_ND_COUNTER(vec_counter, vecs, vec_rank);
                }
            }
            RESET_ND_COUNTER(dim_counter, dim_rank);
            RESET_ND_COUNTER(vec_counter, vec_rank);
            // Write full output to a file
            FILE *out_file = NULL;
            FOPEN(out_file, OUTPUT_LOG_FILE, "w");
            fprintf(out_file, "\t%10s%30s%48s\n", "Index", "Expected",
                    "Actual");
            for (INTP i = 0; i < N; i++)
            {
                INTP idx = idx_map[i];
                fprintf(out_file, "%7td -> ", idx);
                PRINT_ND_COUNTER_TO_FILE(out_file, dim_counter, vec_counter,
                                         dim_rank, vec_rank);
                fprintf(out_file,
                        " : %20.14lf + %20.14lfj  vs  %20.14lf + %20.14lfj\n",
                        a_d[idx * T_DATA_STRIDE], a_d[idx * T_DATA_STRIDE + 1],
                        b_d[idx * T_DATA_STRIDE], b_d[idx * T_DATA_STRIDE + 1]);

                INCREMENT_ND_COUNTER(dim_counter, dims, dim_rank);
                if (i % n == n - 1)
                {
                    INCREMENT_ND_COUNTER(vec_counter, vecs, vec_rank);
                }
            }
            fclose(out_file);
            CHAR path[PATH_SIZE_MAX];
            CHAR *ret = GETCWD(path, sizeof(path));
            if (ret == NULL)
            {
                STRCPY(path, PATH_SIZE_MAX, "current_dir");
            }
            printf("\nFull output log can be found in %s%s%s\n", path,
                    DIRECTORY_SEPARATOR, OUTPUT_LOG_FILE);
        }
        else
        {
            printf("\nUse debug logger mode [--logger-mode 3 (or) -l 3] to get "
                   "detailed error log\n");
        }
        status = VERIFICATION_FAILURE;
    }

    FREE_ALLOCATED_MEM(dim_counter, is_aligned);
    FREE_ALLOCATED_MEM(vec_counter, is_aligned);
    FREE_ALLOCATED_MEM(d_maxerr_coords, is_aligned);
    FREE_ALLOCATED_MEM(d_err_coords, is_aligned);
    FREE_ALLOCATED_MEM(b_maxerr_coords, is_aligned);
    FREE_ALLOCATED_MEM(b_err_coords, is_aligned);

    return status;
}
