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

/** @file compare.c
 *
 *  @brief Data compare functions.
 *
 *  This file contains declarations of data compare functions for test bench.
 *
 *  @author Srirammaswamy Srinivasan
 *  @author Jeya R
 */

#include "test/bench/utils/compare.h"
#include "test/bench/utils/bench_utils.h"
#include <stdio.h>
/**
 * @brief Compare the two data of length n (FLOAT type)
 *        Used to compare output with reference output.
 *
 *        This function is used to compare the output of the FFT implementation
 *        against a reference output. It checks each element of the two arrays
 *        (optionally using index maps for strided or permuted access) and
 *        reports if any element differs by more than the allowed tolerance.
 *
 * @param params   Pointer to aoclfftz_bench_params_t containing test configuration and tolerance.
 * @param a        Pointer to the first data array (FLOAT type).
 * @param b        Pointer to the second data array (FLOAT type).
 * @param batches  Number of batches (or vectors) to compare.
 * @param n        Number of elements per batch.
 * @param a_map    Optional index map for the first array (NULL for linear access).
 * @param b_map    Optional index map for the second array (NULL for linear access).
 * @param data_stride Stride between elements in the data arrays.
 * @return         BENCH_SUCCESS (0) if all data points match within tolerance, or an error code.
 */
INT32 compare_f(aoclfftz_bench_params_t *params, VOID *a, VOID *b, INTP batches,
                INTP n, INTP *a_map, INTP *b_map, INT32 data_stride)
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
    FLOAT max_abs_err = 0.0;
    FLOAT max_mag = 0.0;
    INTP max_err_idx = -1;
    INTP first_err_idx = INT64_MAX;
    FLOAT first_abs_err = 0.0;
    INTP *d_maxerr_coords = NULL;
    INTP *d_err_coords = NULL;
    INTP *b_maxerr_coords = NULL;
    INTP *b_err_coords = NULL;
    UINT32 is_aligned = params->aligned_alloc;

    ALLOC_INIT(dim_counter, INTP, dim_rank * sizeof(INTP), is_aligned);
    ALLOC_INIT(vec_counter, INTP, vec_rank * sizeof(INTP), is_aligned);
    ALLOC_INIT(d_maxerr_coords, INTP, dim_rank * sizeof(INTP), is_aligned);
    ALLOC_INIT(d_err_coords, INTP, dim_rank * sizeof(INTP), is_aligned);
    ALLOC_INIT(b_maxerr_coords, INTP, vec_rank * sizeof(INTP), is_aligned);
    ALLOC_INIT(b_err_coords, INTP, vec_rank * sizeof(INTP), is_aligned);

    if (dim_counter == NULL || vec_counter == NULL ||
        d_maxerr_coords == NULL || d_err_coords == NULL ||
        b_maxerr_coords == NULL || b_err_coords == NULL)
    {
        status = MEMORY_FAILURE;
        goto cleanup;
    }

    INTP N = batches * n;
    for (INTP i = 0; i < N * data_stride; i++)
    {
        INTP idx_a = a_map ? a_map[i / data_stride] : i / data_stride;
        INTP idx_b = b_map ? b_map[i / data_stride] : i / data_stride;
        FLOAT abs_err =
            fabsf(a_f[idx_a * data_stride + (i % data_stride)] -
                 b_f[idx_b * data_stride + (i % data_stride)]);
        FLOAT mag = fminf((fabsf(a_f[idx_a * data_stride + (i % data_stride)])),
                          (fabsf(b_f[idx_b * data_stride + (i % data_stride)])));
        if (abs_err > max_abs_err)
        {
            max_err_idx = idx_b;
            max_abs_err = abs_err;
            COPY_ND_COORDS(d_maxerr_coords, dim_counter, dim_rank);
            COPY_ND_COORDS(b_maxerr_coords, vec_counter, vec_rank);
            if (idx_b < first_err_idx && abs_err > tol)
            {
                first_err_idx = idx_b;
                first_abs_err = abs_err;
                COPY_ND_COORDS(d_err_coords, dim_counter, dim_rank);
                COPY_ND_COORDS(b_err_coords, vec_counter, vec_rank);
            }
        }
        if (mag > max_mag)
        {
            max_mag = mag;
        }
        // Increment the dim_counter after reaching the end of single element
        if (i % data_stride == data_stride - 1)
        {
            INCREMENT_ND_COUNTER(dim_counter, dims, dim_rank);
            // Increment the vec_counter after reaching the end of one batch
            if ((i / data_stride) % n == n - 1)
            {
                INCREMENT_ND_COUNTER(vec_counter, vecs, vec_rank);
            }
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
                   a_f[first_err_idx * data_stride],
                   a_f[first_err_idx * data_stride  + 1]);
            printf("  got      = %.6g + %.6gj\n",
                   b_f[first_err_idx * data_stride],
                   b_f[first_err_idx * data_stride  + 1]);
        }
        printf("  max abs error = %.6g\n", first_abs_err);
        printf("Max absolute error at index %td -> ", max_err_idx);
        PRINT_ND_COUNTER(d_maxerr_coords, b_maxerr_coords, dim_rank, vec_rank);
        printf("\n  expected = %.6g + %.6gj\n",
               a_f[max_err_idx * data_stride],
               a_f[max_err_idx * data_stride  + 1]);
        printf("  got      = %.6g + %.6gj\n", b_f[max_err_idx * data_stride],
               b_f[max_err_idx * data_stride  + 1]);
        printf("  max abs error = %.6g\n", max_abs_err);
        if (logger_mode >= AOCLFFTZ_LOG_DEBUG)
        {
            RESET_ND_COUNTER(dim_counter, dim_rank);
            RESET_ND_COUNTER(vec_counter, vec_rank);
            // Using printf instead of logger to avoid file and line prefix
            if (data_stride == 1)
            {
                printf("\n\t%5s%20s%17s\n", "Index", "Expected", "Actual");
            }
            else
            {
                printf("\n\t%5s%26s%32s\n", "Index", "Expected", "Actual");
            }
            for (INTP i = 0, c = 100; i < N && c > 0; i++, c--)
            {
                INTP idx = b_map ? b_map[i] : i;
                printf("%7td -> ", idx);
                PRINT_ND_COUNTER(dim_counter, vec_counter, dim_rank, vec_rank);
                if (data_stride == 1)
                {
                    printf(" : %12.6f  vs  %12.6f\n",
                           a_f[idx * data_stride], b_f[idx * data_stride]);
                }
                else
                {
                    printf(" : %12.6f + %12.6fj  vs  %12.6f + %12.6fj\n",
                        a_f[idx * data_stride], a_f[idx * data_stride + 1],
                        b_f[idx * data_stride], b_f[idx * data_stride + 1]);
                }
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
            if (data_stride == 1)
            {
                fprintf(out_file, "\t%5s%20s%17s\n", "Index", "Expected",
                        "Actual");
            }
            else
            {
                fprintf(out_file, "\t%5s%26s%32s\n", "Index", "Expected",
                        "Actual");
            }
            for (INTP i = 0; i < N; i++)
            {
                INTP idx = b_map ? b_map[i] : i;
                fprintf(out_file, "%7td -> ", idx);
                PRINT_ND_COUNTER_TO_FILE(out_file, dim_counter, vec_counter,
                                         dim_rank, vec_rank);
                if (data_stride == 1)
                {
                    fprintf(out_file,
                            " : %12.6f vs  %12.6f\n",
                            a_f[idx * data_stride], b_f[idx * data_stride]);
                }
                else
                {
                    fprintf(out_file,
                            " : %12.6f + %12.6fj  vs  %12.6f + %12.6fj\n",
                            a_f[idx * data_stride], a_f[idx * data_stride + 1],
                            b_f[idx * data_stride], b_f[idx * data_stride + 1]);
                }
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

cleanup:
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
 * This function is typically used to compare the output of a computation with a reference output.
 * It supports optional index mapping for strided or non-contiguous data layouts.
 *
 * @param params Pointer to aoclfftz_bench_params_t containing test configuration and tolerance.
 * @param a Pointer to the first data buffer.
 * @param b Pointer to the second data buffer.
 * @param batches Number of batches (vector size).
 * @param n Number of elements per batch (problem size).
 * @param a_map Optional index map for buffer a (NULL for linear access).
 * @param b_map Optional index map for buffer b (NULL for linear access).
 * @param data_stride Stride between elements (1 for real, 2 for complex).
 * @return INT32 BENCH_SUCCESS if all data points match within tolerance, VERIFICATION_FAILURE otherwise.
 */
INT32 compare_d(aoclfftz_bench_params_t *params, VOID *a, VOID *b, INTP batches,
                INTP n, INTP *a_map, INTP *b_map, INT32 data_stride)
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
    DOUBLE max_abs_err = 0.0;
    DOUBLE max_mag = 0.0;
    INTP max_err_idx = -1;
    INTP first_err_idx = INT64_MAX;
    DOUBLE first_abs_err = 0.0;
    INTP *d_maxerr_coords = NULL;
    INTP *d_err_coords = NULL;
    INTP *b_maxerr_coords = NULL;
    INTP *b_err_coords = NULL;
    ALLOC_INIT(dim_counter, INTP, dim_rank * sizeof(INTP), is_aligned);
    ALLOC_INIT(vec_counter, INTP, vec_rank * sizeof(INTP), is_aligned);
    ALLOC_INIT(d_maxerr_coords, INTP, dim_rank * sizeof(INTP), is_aligned);
    ALLOC_INIT(d_err_coords, INTP, dim_rank * sizeof(INTP), is_aligned);
    ALLOC_INIT(b_maxerr_coords, INTP, vec_rank * sizeof(INTP), is_aligned);
    ALLOC_INIT(b_err_coords, INTP, vec_rank * sizeof(INTP), is_aligned);

    if (dim_counter == NULL || vec_counter == NULL ||
        d_maxerr_coords == NULL || d_err_coords == NULL ||
        b_maxerr_coords == NULL || b_err_coords == NULL)
    {
        status = MEMORY_FAILURE;
        goto cleanup;
    }

    INTP N = batches * n;
    for (INTP i = 0; i < N * data_stride; i++)
    {
        INTP idx_a = a_map ? a_map[i / data_stride] : i / data_stride;
        INTP idx_b = b_map ? b_map[i / data_stride] : i / data_stride;
        DOUBLE abs_err = fabs(a_d[idx_a * data_stride + (i % data_stride)] -
                              b_d[idx_b * data_stride + (i % data_stride)]);
        DOUBLE mag = fmin((fabs(a_d[idx_a * data_stride + (i % data_stride)])),
                          (fabs(b_d[idx_b * data_stride + (i % data_stride)])));
        if (abs_err > max_abs_err)
        {
            max_err_idx = idx_b;
            max_abs_err = abs_err;
            COPY_ND_COORDS(d_maxerr_coords, dim_counter, dim_rank);
            COPY_ND_COORDS(b_maxerr_coords, vec_counter, vec_rank);
            if (idx_b < first_err_idx && abs_err > tol)
            {
                first_err_idx = idx_b;
                first_abs_err = abs_err;
                COPY_ND_COORDS(d_err_coords, dim_counter, dim_rank);
                COPY_ND_COORDS(b_err_coords, vec_counter, vec_rank);
            }
        }
        if (mag > max_mag)
        {
            max_mag = mag;
        }
        // Increment the dim_counter after reaching the end of single element
        if (i % data_stride == data_stride - 1)
        {
            INCREMENT_ND_COUNTER(dim_counter, dims, dim_rank);
            // Increment the vec_counter after reaching the end of one batch
            if ((i / data_stride) % n == n - 1)
            {
                INCREMENT_ND_COUNTER(vec_counter, vecs, vec_rank);
            }
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
        AOCLFFTZ_ERROR_FORMATTED("Relative error  = %.6g\n", rel_err);
        AOCLFFTZ_ERROR_FORMATTED("Tolerance       = %.6g\n", tol);
        if (first_err_idx < INT64_MAX)
        {
            printf("First absolute error at index %td -> ", first_err_idx);
            PRINT_ND_COUNTER(d_err_coords, b_err_coords, dim_rank,
                             vec_rank);
            printf("\n  expected = %.6g + %.6gj\n",
                   a_d[first_err_idx * data_stride],
                   a_d[first_err_idx * data_stride + 1]);
            printf("  got      = %.6g + %.6gj\n",
                   b_d[first_err_idx * data_stride],
                   b_d[first_err_idx * data_stride + 1]);
        }
        printf("  max abs error = %.6g\n", first_abs_err);
        printf("Max absolute error at index %td -> ", max_err_idx);
        PRINT_ND_COUNTER(d_maxerr_coords, b_maxerr_coords, dim_rank, vec_rank);
        fflush(stdout);
        printf("\n  expected = %.6g + %.6gj\n",
               a_d[max_err_idx * data_stride],
               a_d[max_err_idx * data_stride + 1]);
        printf("  got      = %.6g + %.6gj\n",
               b_d[max_err_idx * data_stride],
               b_d[max_err_idx * data_stride + 1]);
        printf("  max abs error = %.6g\n", max_abs_err);
        if (logger_mode >= AOCLFFTZ_LOG_DEBUG)
        {
            RESET_ND_COUNTER(dim_counter, dim_rank);
            RESET_ND_COUNTER(vec_counter, vec_rank);
            // Using printf instead of logger to avoid file and line prefix
            if (data_stride == 1)
            {
                printf("\n\t%10s%20s%24s\n", "Index", "Expected", "Actual");
            }
            else
            {
                printf("\n\t%10s%30s%48s\n", "Index", "Expected", "Actual");
            }
            for (INTP i = 0, c = 100; i < N && c > 0; i++, c--)
            {
                INTP idx = b_map ? b_map[i] : i;
                // vecs
                printf("%7td -> ", idx);
                PRINT_ND_COUNTER(dim_counter, vec_counter, dim_rank, vec_rank);
                if (data_stride == 1)
                {
                    printf(" : %20.14lf  vs  %20.14lf\n",
                           a_d[idx * data_stride], b_d[idx * data_stride]);
                }
                else
                {
                    printf(
                        " : %20.14lf + %20.14lfj  vs  %20.14lf + %20.14lfj\n",
                        a_d[idx * data_stride], a_d[idx * data_stride + 1],
                        b_d[idx * data_stride], b_d[idx * data_stride + 1]);
                }
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
            if (data_stride == 1)
            {
                fprintf(out_file, "\t%10s%20s%24s\n", "Index", "Expected",
                        "Actual");
            }
            else
            {
                fprintf(out_file, "\t%10s%30s%48s\n", "Index", "Expected",
                        "Actual");
            }
            for (INTP i = 0; i < N; i++)
            {
                INTP idx = b_map ? b_map[i] : i;
                fprintf(out_file, "%7td -> ", idx);
                PRINT_ND_COUNTER_TO_FILE(out_file, dim_counter, vec_counter,
                                         dim_rank, vec_rank);
                if (data_stride == 1)
                {
                    fprintf(out_file,
                            " : %20.14lf vs  %20.14lf\n",
                            a_d[idx * data_stride], b_d[idx * data_stride]);
                }
                else
                {
                    fprintf(
                        out_file,
                        " : %20.14lf + %20.14lfj  vs  %20.14lf + %20.14lfj\n",
                        a_d[idx * data_stride], a_d[idx * data_stride + 1],
                        b_d[idx * data_stride], b_d[idx * data_stride + 1]);
                }
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
            AOCLFFTZ_ERROR_UNFORMATTED("\nUse debug logger mode "
                                       "[--logger-mode 3 (or) -l 3] to get "
                                       "detailed error log\n");
        }
        status = VERIFICATION_FAILURE;
    }

cleanup:
    FREE_ALLOCATED_MEM(dim_counter, is_aligned);
    FREE_ALLOCATED_MEM(vec_counter, is_aligned);
    FREE_ALLOCATED_MEM(d_maxerr_coords, is_aligned);
    FREE_ALLOCATED_MEM(d_err_coords, is_aligned);
    FREE_ALLOCATED_MEM(b_maxerr_coords, is_aligned);
    FREE_ALLOCATED_MEM(b_err_coords, is_aligned);

    return status;
}
