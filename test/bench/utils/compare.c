// Copyright Advanced Micro Devices, Inc.
// SPDX-License-Identifier: BSD-3-Clause

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
#include "utils/utils.h"
/**
 * @brief Compare the two data of length n (FFTZ_FLOAT type)
 *        Used to compare output with reference output.
 *
 *        This function is used to compare the output of the FFT implementation
 *        against a reference output. It checks each element of the two arrays
 *        (optionally using index maps for strided or permuted access) and
 *        reports if any element differs by more than the allowed tolerance.
 *
 * @param params   Pointer to aoclfftz_bench_params_t containing test
 * configuration and tolerance.
 * @param a        Pointer to the first data array (FFTZ_FLOAT type).
 * @param b        Pointer to the second data array (FFTZ_FLOAT type).
 * @param batches  Number of batches (or vectors) to compare.
 * @param n        Number of elements per batch.
 * @param a_map    Optional index map for the first array (NULL for linear
 * access).
 * @param b_map    Optional index map for the second array (NULL for linear
 * access).
 * @param data_stride Stride between elements in the data arrays.
 * @return         BENCH_SUCCESS (0) if all data points match within tolerance,
 * or an error code.
 */
FFTZ_INT32 compare_f(aoclfftz_bench_params_t *params, FFTZ_VOID *a,
                     FFTZ_VOID *b, FFTZ_INTP batches, FFTZ_INTP n,
                     FFTZ_INTP *a_map, FFTZ_INTP *b_map, FFTZ_INT32 data_stride)
{
    FFTZ_DOUBLE tol = params->tolerance;
    FFTZ_FLOAT *a_f = (FFTZ_FLOAT *)a;
    FFTZ_FLOAT *b_f = (FFTZ_FLOAT *)b;
    FFTZ_INT32 status = BENCH_SUCCESS;

    FFTZ_INT32 dim_rank = params->dim_rank;
    aoclfftz_dim_t_64_ *dims = params->dims;
    FFTZ_INT32 vec_rank = params->vec_rank;
    aoclfftz_dim_t_64_ *vecs = params->vecs;
    FFTZ_INTP *dim_counter = NULL;
    FFTZ_INTP *vec_counter = NULL;
    FFTZ_FLOAT max_abs_err = 0.0;
    FFTZ_FLOAT max_mag = 0.0;
    FFTZ_INTP max_err_idx = -1;
    FFTZ_INTP first_err_idx = INT64_MAX;
    FFTZ_FLOAT first_abs_err = 0.0;
    FFTZ_INTP *d_maxerr_coords = NULL;
    FFTZ_INTP *d_err_coords = NULL;
    FFTZ_INTP *b_maxerr_coords = NULL;
    FFTZ_INTP *b_err_coords = NULL;
    FFTZ_UINT32 is_aligned = params->aligned_alloc;
    FFTZ_INT32 logger_mode = params->logger_mode;

    ALLOC_INIT(dim_counter, FFTZ_INTP, dim_rank * sizeof(FFTZ_INTP),
               is_aligned);
    ALLOC_INIT(vec_counter, FFTZ_INTP, vec_rank * sizeof(FFTZ_INTP),
               is_aligned);
    ALLOC_INIT(d_maxerr_coords, FFTZ_INTP, dim_rank * sizeof(FFTZ_INTP),
               is_aligned);
    ALLOC_INIT(d_err_coords, FFTZ_INTP, dim_rank * sizeof(FFTZ_INTP),
               is_aligned);
    ALLOC_INIT(b_maxerr_coords, FFTZ_INTP, vec_rank * sizeof(FFTZ_INTP),
               is_aligned);
    ALLOC_INIT(b_err_coords, FFTZ_INTP, vec_rank * sizeof(FFTZ_INTP),
               is_aligned);

    if (dim_counter == NULL || vec_counter == NULL ||
        d_maxerr_coords == NULL || d_err_coords == NULL ||
        b_maxerr_coords == NULL || b_err_coords == NULL)
    {
        status = MEMORY_FAILURE;
        goto cleanup;
    }

    FFTZ_INTP N = batches * n;
    for (FFTZ_INTP i = 0; i < N * data_stride; i++)
    {
        FFTZ_INTP idx_a = a_map ? a_map[i / data_stride] : i / data_stride;
        FFTZ_INTP idx_b = b_map ? b_map[i / data_stride] : i / data_stride;
        FFTZ_FLOAT abs_err =
            fabsf(a_f[idx_a * data_stride + (i % data_stride)] -
                 b_f[idx_b * data_stride + (i % data_stride)]);
        FFTZ_FLOAT mag = fminf(
            (fabsf(a_f[idx_a * data_stride + (i % data_stride)])),
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

    FFTZ_FLOAT rel_err;
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
    AOCLFFTZ_LOG(INFO, logger_mode, "Error = %.6g", rel_err);
    if (rel_err > tol)
    {
        AOCLFFTZ_ERROR("Accuracy failure");
        fprintf(stderr, "Relative error  = %.6g\n", rel_err);
        fprintf(stderr, "Tolerance       = %.6g\n", tol);
        if (first_err_idx < INT64_MAX)
        {
            fprintf(stderr, "First absolute error at index %td -> ",
                    first_err_idx);
            PRINT_ND_COUNTER(d_err_coords, b_err_coords, dim_rank,
                             vec_rank);
            fprintf(stderr, "\n  expected = %.6g + %.6gj\n",
                    a_f[first_err_idx * data_stride],
                    a_f[first_err_idx * data_stride  + 1]);
            fprintf(stderr, "  got      = %.6g + %.6gj\n",
                    b_f[first_err_idx * data_stride],
                    b_f[first_err_idx * data_stride  + 1]);
        }
        fprintf(stderr, "  max abs error = %.6g\n", first_abs_err);
        fprintf(stderr, "Max absolute error at index %td -> ", max_err_idx);
        PRINT_ND_COUNTER(d_maxerr_coords, b_maxerr_coords, dim_rank, vec_rank);
        fprintf(stderr, "\n  expected = %.6g + %.6gj\n",
                a_f[max_err_idx * data_stride],
                a_f[max_err_idx * data_stride  + 1]);
        fprintf(stderr, "  got      = %.6g + %.6gj\n",
                b_f[max_err_idx * data_stride],
                b_f[max_err_idx * data_stride  + 1]);
        fprintf(stderr, "  max abs error = %.6g\n", max_abs_err);
        if (logger_mode >= AOCLFFTZ_LOG_DEBUG)
        {
            RESET_ND_COUNTER(dim_counter, dim_rank);
            RESET_ND_COUNTER(vec_counter, vec_rank);
            // Using printf instead of logger to avoid file and line prefix
            if (data_stride == 1)
            {
                fprintf(stderr, "\n\t%5s%20s%17s\n",
                        "Index", "Expected", "Actual");
            }
            else
            {
                fprintf(stderr, "\n\t%5s%26s%32s\n",
                        "Index", "Expected", "Actual");
            }
            for (FFTZ_INTP i = 0, c = 100; i < N && c > 0; i++, c--)
            {
                FFTZ_INTP idx = b_map ? b_map[i] : i;
                fprintf(stderr, "%7td -> ", idx);
                PRINT_ND_COUNTER(dim_counter, vec_counter, dim_rank, vec_rank);
                if (data_stride == 1)
                {
                    fprintf(stderr, " : %12.6f  vs  %12.6f\n",
                            a_f[idx * data_stride], b_f[idx * data_stride]);
                }
                else
                {
                    fprintf(stderr,
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
            for (FFTZ_INTP i = 0; i < N; i++)
            {
                FFTZ_INTP idx = b_map ? b_map[i] : i;
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
            FFTZ_CHAR path[PATH_SIZE_MAX];
            FFTZ_CHAR *ret = GETCWD(path, sizeof(path));
            if (ret == NULL)
            {
                STRCPY(path, PATH_SIZE_MAX, "current_dir");
            }
            fprintf(stderr, "\nFull output log can be found in %s%s%s\n", path,
                    DIRECTORY_SEPARATOR, OUTPUT_LOG_FILE);
        }
        else
        {
            fprintf(stderr, "\nUse debug logger mode "
                   "[--logger-mode 3 (or) -l 3] to get detailed error log\n");
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
 * @brief Compare the two data of length n (FFTZ_DOUBLE type)
 *        Used to compare output with reference output.
 *
 * This function is typically used to compare the output of a computation with a
 * reference output. It supports optional index mapping for strided or
 * non-contiguous data layouts.
 *
 * @param params Pointer to aoclfftz_bench_params_t containing test
 * configuration and tolerance.
 * @param a Pointer to the first data buffer.
 * @param b Pointer to the second data buffer.
 * @param batches Number of batches (vector size).
 * @param n Number of elements per batch (problem size).
 * @param a_map Optional index map for buffer a (NULL for linear access).
 * @param b_map Optional index map for buffer b (NULL for linear access).
 * @param data_stride Stride between elements (1 for real, 2 for complex).
 * @return FFTZ_INT32 BENCH_SUCCESS if all data points match within tolerance,
 * VERIFICATION_FAILURE otherwise.
 */
FFTZ_INT32 compare_d(aoclfftz_bench_params_t *params, FFTZ_VOID *a,
                     FFTZ_VOID *b, FFTZ_INTP batches, FFTZ_INTP n,
                     FFTZ_INTP *a_map, FFTZ_INTP *b_map, FFTZ_INT32 data_stride)
{
    FFTZ_DOUBLE tol = params->tolerance;
    FFTZ_DOUBLE *a_d = (FFTZ_DOUBLE *)a;
    FFTZ_DOUBLE *b_d = (FFTZ_DOUBLE *)b;
    FFTZ_INT32 status = BENCH_SUCCESS;
    FFTZ_INT32 dim_rank = params->dim_rank;
    aoclfftz_dim_t_64_ *dims = params->dims;
    FFTZ_INT32 vec_rank = params->vec_rank;
    aoclfftz_dim_t_64_ *vecs = params->vecs;
    FFTZ_INTP *dim_counter = NULL;
    FFTZ_INTP *vec_counter = NULL;
    FFTZ_UINT32 is_aligned = params->aligned_alloc;
    FFTZ_DOUBLE max_abs_err = 0.0;
    FFTZ_DOUBLE max_mag = 0.0;
    FFTZ_INTP max_err_idx = -1;
    FFTZ_INTP first_err_idx = INT64_MAX;
    FFTZ_DOUBLE first_abs_err = 0.0;
    FFTZ_INTP *d_maxerr_coords = NULL;
    FFTZ_INTP *d_err_coords = NULL;
    FFTZ_INTP *b_maxerr_coords = NULL;
    FFTZ_INTP *b_err_coords = NULL;
    FFTZ_INT32 logger_mode = params->logger_mode;
    ALLOC_INIT(dim_counter, FFTZ_INTP, dim_rank * sizeof(FFTZ_INTP),
               is_aligned);
    ALLOC_INIT(vec_counter, FFTZ_INTP, vec_rank * sizeof(FFTZ_INTP),
               is_aligned);
    ALLOC_INIT(d_maxerr_coords, FFTZ_INTP, dim_rank * sizeof(FFTZ_INTP),
               is_aligned);
    ALLOC_INIT(d_err_coords, FFTZ_INTP, dim_rank * sizeof(FFTZ_INTP),
               is_aligned);
    ALLOC_INIT(b_maxerr_coords, FFTZ_INTP, vec_rank * sizeof(FFTZ_INTP),
               is_aligned);
    ALLOC_INIT(b_err_coords, FFTZ_INTP, vec_rank * sizeof(FFTZ_INTP),
               is_aligned);

    if (dim_counter == NULL || vec_counter == NULL ||
        d_maxerr_coords == NULL || d_err_coords == NULL ||
        b_maxerr_coords == NULL || b_err_coords == NULL)
    {
        status = MEMORY_FAILURE;
        goto cleanup;
    }

    FFTZ_INTP N = batches * n;
    for (FFTZ_INTP i = 0; i < N * data_stride; i++)
    {
        FFTZ_INTP idx_a = a_map ? a_map[i / data_stride] : i / data_stride;
        FFTZ_INTP idx_b = b_map ? b_map[i / data_stride] : i / data_stride;
        FFTZ_DOUBLE abs_err =
            fabs(a_d[idx_a * data_stride + (i % data_stride)] -
                              b_d[idx_b * data_stride + (i % data_stride)]);
        FFTZ_DOUBLE mag = fmin(
            (fabs(a_d[idx_a * data_stride + (i % data_stride)])),
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
    FFTZ_DOUBLE rel_err;
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
    AOCLFFTZ_LOG(INFO, logger_mode, "Error = %.6g", rel_err);
    if (rel_err > tol)
    {
        AOCLFFTZ_ERROR("Accuracy failure");
        fprintf(stderr, "Relative error  = %.6g\n", rel_err);
        fprintf(stderr, "Tolerance       = %.6g\n", tol);
        if (first_err_idx < INT64_MAX)
        {
            fprintf(stderr, "First absolute error at index %td -> ",
                    first_err_idx);
            PRINT_ND_COUNTER(d_err_coords, b_err_coords, dim_rank, vec_rank);
            fprintf(stderr, "\n  expected = %.6g + %.6gj\n",
                    a_d[first_err_idx * data_stride],
                    a_d[first_err_idx * data_stride + 1]);
            fprintf(stderr, "  got      = %.6g + %.6gj\n",
                    b_d[first_err_idx * data_stride],
                    b_d[first_err_idx * data_stride + 1]);
        }
        fprintf(stderr, "  max abs error = %.6g\n", first_abs_err);
        fprintf(stderr, "Max absolute error at index %td -> ", max_err_idx);
        PRINT_ND_COUNTER(d_maxerr_coords, b_maxerr_coords, dim_rank, vec_rank);
        fflush(stdout);
        fprintf(stderr, "\n  expected = %.6g + %.6gj\n",
                a_d[max_err_idx * data_stride],
                a_d[max_err_idx * data_stride + 1]);
        fprintf(stderr, "  got      = %.6g + %.6gj\n",
                b_d[max_err_idx * data_stride],
                b_d[max_err_idx * data_stride + 1]);
        fprintf(stderr, "  max abs error = %.6g\n", max_abs_err);
        if (logger_mode >= AOCLFFTZ_LOG_DEBUG)
        {
            RESET_ND_COUNTER(dim_counter, dim_rank);
            RESET_ND_COUNTER(vec_counter, vec_rank);
            // Using printf instead of logger to avoid file and line prefix
            if (data_stride == 1)
            {
                fprintf(stderr, "\n\t%10s%20s%24s\n", "Index", "Expected",
                        "Actual");
            }
            else
            {
                fprintf(stderr, "\n\t%10s%30s%48s\n", "Index", "Expected",
                        "Actual");
            }
            for (FFTZ_INTP i = 0, c = 100; i < N && c > 0; i++, c--)
            {
                FFTZ_INTP idx = b_map ? b_map[i] : i;
                // vecs
                fprintf(stderr, "%7td -> ", idx);
                PRINT_ND_COUNTER(dim_counter, vec_counter, dim_rank, vec_rank);
                if (data_stride == 1)
                {
                    fprintf(stderr, " : %20.14lf  vs  %20.14lf\n",
                           a_d[idx * data_stride], b_d[idx * data_stride]);
                }
                else
                {
                    fprintf(stderr,
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
            for (FFTZ_INTP i = 0; i < N; i++)
            {
                FFTZ_INTP idx = b_map ? b_map[i] : i;
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
            FFTZ_CHAR path[PATH_SIZE_MAX];
            FFTZ_CHAR *ret = GETCWD(path, sizeof(path));
            if (ret == NULL)
            {
                STRCPY(path, PATH_SIZE_MAX, "current_dir");
            }
            fprintf(stderr, "\nFull output log can be found in %s%s%s\n", path,
                    DIRECTORY_SEPARATOR, OUTPUT_LOG_FILE);
        }
        else
        {
            fprintf(stderr, "\nUse debug logger mode "
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
