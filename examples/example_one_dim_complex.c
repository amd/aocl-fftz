// Copyright Advanced Micro Devices, Inc.
// SPDX-License-Identifier: BSD-3-Clause

/** @file example_one_dim_complex.c
 *
 *  @brief Example for AOCL-FFTZ APIs to compute 1D complex FFT forward problem
 *
 *  The following test program shows the sample usage and calling sequence of
 *  AOCL-FFTZ APIs to compute 1D complex FFT problem for DOUBLE datatype on LP64
 *  systems. This file also contains helper macros and functions to be used for
 *  initialization of the problem descriptor.
 *
 *  @note To run the program:
 *  example_one_dim_complex
 *
 *  @author Partiksha
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "aoclfftz.h"
#include "helpers.h"

int main()
{
    /* STEP 1: Create and initialize prob_desc params. */

    /* Available problem descriptor types-
     *      `aoclfftz_prob_desc_f` for FLOAT LP64
     *      `aoclfftz_prob_desc_d` for DOUBLE LP64
     *      `aoclfftz_prob_desc_f_64_` for FLOAT ILP64
     *      `aoclfftz_prob_desc_d_64_` for DOUBLE ILP64
     *
     * Available dims/vecs types-
     *      `aoclfftz_dim_t` for LP64
     *      `aoclfftz_dim_t_64_` for ILP64
     *
     * type definition for DOUBLE LP64 is used here.
     */
    aoclfftz_dim_t dims[] = {{.n = 40, .in_stride = 2, .out_stride = 2}};
    aoclfftz_dim_t vecs[] = {{.n = 3, .in_stride = 80, .out_stride = 80}};
    aoclfftz_prob_desc_d problem = {
        .dim_rank = 1, // the number of signal/frequency dimensions, must be >= 1.
        .vec_rank = 1, // the number of batch/vector dimensions, must be >= 1.
        .dims = dims,
        .vecs = vecs,
        .flags = {
            0, // fft_type       : complex(0), real(1)
            0, // fft_direction  : forward(0), backward(1)
            0, // storage_order  : inorder(0), out-of-order(1)
            0, // fft_placement  : inplace(0), out-of-place(1)
            0  // transpose_mode : FFT(0), standalone transpose(1)
        },
        .pthr_fft = {
                    /*
                     * num_threads = 1 for Single Threaded Execution
                     * num_threads > 1 for Multithreaded Execution (Make sure that Library
                     * is build with ENABLE_MULTI_THREADING=ON for multithreading execution)
                     */
                     .num_threads = 1,
                     .dynamic_load_model = 0},
        .cntrl_params =
            {
                .opt_level = 0,
                .opt_off = 1,
                .logger_mode = 0,
                .measure_stats = 0,
            },
    };
    UINT8 is_out_of_place = problem.flags.fft_placement;

    /* STEP 1.1: Set default vecs and dims strides values.
                 Strides values are already set for the first dimension and
                 `set_default_dims_vecs()` can be used to set the strides
                 to default values for higher dimensions. */

    /* STEP 1.2: Calculate input/output buffer sizes.
                 `calculate_buffer_sizes()` can be used to calculate buffer sizes
                 for higher dimension problem size. */
    UINTP in_buffer_size =  ((dims[0].n - 1) * (dims[0].in_stride)) +
                            ((vecs[0].n - 1) * (vecs[0].in_stride)) + 1;
    UINTP out_buffer_size = ((dims[0].n - 1) * (dims[0].out_stride)) +
                            ((vecs[0].n - 1) * (vecs[0].out_stride)) + 1;

    /* STEP 1.3: Allocate memory input/output buffers and
                 create new output buffer in case of out-of-place problem. */
    DOUBLE *in = NULL;
    DOUBLE *out = NULL;
    ALLOC(in, DOUBLE, (in_buffer_size * DATA_STRIDE(problem.flags.fft_type)));
    if (is_out_of_place)
    {
        ALLOC(out, DOUBLE, (out_buffer_size * DATA_STRIDE(problem.flags.fft_type)));
    }

    /* STEP 1.4: Prepare input for FFT calculation and
                 Use input buffer as output for in-place buffer. */
    PREPARE_RANDOM_INPUT(in, in_buffer_size, problem.flags.fft_type, DOUBLE);
    problem.in = in;
    problem.out = !is_out_of_place ? in : out;

    /* STEP 2: Invoke appropriate setup API to generate solution. */
    VOID *aoclfftz_handle = aoclfftz_setup_d(&problem);

    /* STEP 3: Invoke execute API. */
    if (aoclfftz_handle)
    {
        printf("\nSetup Successful\n");
        INT32 res = AOCLFFTZ_SUCCESS;
        res = aoclfftz_execute(aoclfftz_handle);

        if (res == AOCLFFTZ_EXECUTION_FAILURE)
        {
            printf("\nExecution Failure\n");
        }
        else
        {
            printf("\nExecution successful\n");
        }
    }
    else
    {
        printf("\nSetup Failure\n");
    }

    /* STEP 4: Invoke destroy API to free `aoclfftz_handle` and
               free the allocated memory. */
    aoclfftz_destroy(aoclfftz_handle);
    free(problem.in);
    if (is_out_of_place)
    {
        free(problem.out);
    }
}
