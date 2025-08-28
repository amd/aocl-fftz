/**
 * Copyright (C) 2025, Advanced Micro Devices. All rights reserved.
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

#define DATA_STRIDE 2 // 1 - for real problems,
                      // 2 - for complex problems
#define ALLOC(ptr, type, n_ele)                      \
{                                                    \
    ptr = (type *)malloc(sizeof(type) * n_ele);      \
    if(ptr == NULL)                                  \
    {                                                \
        printf("\nMemory allocation failed\n");      \
        exit(EXIT_FAILURE);                          \
    }                                                \
    else                                             \
    {                                                \
        memset(ptr, 0, sizeof(type) * n_ele);        \
    }                                                \
}

// This function can be used for real (1D) and complex(1D and ND) problems
VOID calculate_buffer_sizes(aoclfftz_dim_t *dims, INT32 dim_rank,
                            aoclfftz_dim_t *vecs, INT32 vec_rank,
                            UINTP *in_buffer_size, UINTP *out_buffer_size)
{
    // Example: for an 1D problem with 1D batch
    // Problem size : 3:6:6v4:1:1
    // Data arrangement considered :
    // [1,      2,      3,      4]<0, 0>[5,      6,     7,     8]<0, 0>[9,     10,     11,     12]
    // <-problem size*dim_stride->
    // <-----------vec stride---------->
    // <---------------------------------------Batches------------------------------------------->
    UINTP in_size = 1;
    UINTP out_size = 1;
    for (INT32 i = 0; i < dim_rank; i++)
    {
        in_size += ((dims[i].n - 1) * (dims[i].in_stride));
        out_size += ((dims[i].n - 1) * (dims[i].out_stride));
    }
    for (INT32 i = 0; i < vec_rank; i++)
    {
        in_size += ((vecs[i].n - 1) * (vecs[i].in_stride));
        out_size += ((vecs[i].n - 1) * (vecs[i].out_stride));
    }
    *in_buffer_size = in_size;
    *out_buffer_size = out_size;
}

// This function can be used for real (1D) and complex(1D and ND) problems
VOID prepare_random_input(DOUBLE *in, UINTP input_size)
{
    INTP idx = 0;
    for (idx = 0; idx < input_size * DATA_STRIDE; ++idx)
    {
        in[idx] = (20.0 / RAND_MAX) * rand() - 10.0;
    }
}

// This function calculates the strides for complex inputs for (1D and ND)
VOID calculate_vecs_n_dims_strides_for_complex(aoclfftz_dim_t *dims,
                INT32 dim_rank, aoclfftz_dim_t *vecs, INT32 vec_rank)
{
    // Example: for an 1D problem with 1D batch
    // Problem size : 3v4:2:2
    // Data arrangement considered :
    // [1, -, 3, -, 4, -, 6, -]    [8, -, 10, -, 12, -, 14 -]   [16, -, 18, -, 20, -, 22, -]
    // <-------vec_stride----------->
    // vec_stride = number of elements of batch-1 * dim_stride
    INT32 d;
    for (d = 0; d < dim_rank; d++)
    {
        if (d == 0)
        {
            dims[0].in_stride = (dims[0].in_stride <= 1) ?
                                    1 : dims[0].in_stride;
            dims[0].out_stride = (dims[0].out_stride <= 1) ?
                                    1 : dims[0].out_stride;
            continue;
        }
        dims[d].in_stride = dims[d - 1].in_stride * dims[d - 1].n;
        dims[d].out_stride = dims[d - 1].out_stride * dims[d - 1].n;
    }
    UINT32 default_vec_in_stride  = dims[dim_rank - 1].in_stride
                                    * dims[dim_rank - 1].n;
    UINT32 default_vec_out_stride = dims[dim_rank - 1].out_stride
                                    * dims[dim_rank - 1].n;
    for (d = 0; d < vec_rank; d++)
    {
        if (d == 0)
        {
            vecs[0].in_stride = (vecs[0].in_stride <= 1) ?
                                    default_vec_in_stride : vecs[0].in_stride;
            vecs[0].out_stride = (vecs[0].out_stride <= 1) ?
                                    default_vec_out_stride : vecs[0].out_stride;
            continue;
        }
        vecs[d].in_stride = vecs[d - 1].in_stride * vecs[d - 1].n;
        vecs[d].out_stride = vecs[d - 1].out_stride * vecs[d - 1].n;
    }
}

int main()
{
    // Create and initialize prob_desc params

    /* Available problem descriptor types-
     * `aoclfftz_prob_desc_f` for FLOAT LP64
     * `aoclfftz_prob_desc_d` for DOUBLE LP64
     * `aoclfftz_prob_desc_f_64_` for FLOAT ILP64
     * `aoclfftz_prob_desc_d_64_` for DOUBLE ILP64
     *
     * type definition for DOUBLE LP64 is used here.
     */
    aoclfftz_prob_desc_d *problem;
    ALLOC(problem, aoclfftz_prob_desc_d, 1);
    problem->dim_rank = 1; // the number of signal/frequency dimensions, must be >= 1,
    problem->vec_rank = 1; // the number of batch dimensions, must be >= 1.

    // create the initialise dims and vecs

    /* Available dimension types-
     * `aoclfftz_dim_t` for LP64
     * `aoclfftz_dim_t_64_` for ILP64
     *
     * According to problem descriptor,
     * type definition for LP64 is used here.
     */
    ALLOC(problem->dims, aoclfftz_dim_t, problem->dim_rank);
    ALLOC(problem->vecs, aoclfftz_dim_t, problem->vec_rank);

    // complex, forward, in-order, in-place, FFT problem
    aoclfftz_flags_t flags = {
        0, // fft_type       : complex(0), real(1)
        0, // fft_direction  : forward(0), backward(1)
        0, // storage_order  : inorder(0), out-of-order(1)
        0, // fft_placement  : inplace(0), out-of-place(1)
        0  // transpose_mode : FFT(0), standalone transpose(1)
    };
    problem->flags = flags;
    UINT8 is_inplace = !problem->flags.fft_placement;
    problem->pthr_fft.dynamic_load_model = 0;
    problem->pthr_fft.num_threads = 1;
    /*
    * num_threads = 1 for Single Threaded Execution
    * num_threads > 1 for Multithreaded Execution (Make sure that Library
    * is build with ENABLE_MULTI_THREADING=ON for multithreading execution)
    */
    problem->cntrl_params.logger_mode = 0;
    problem->cntrl_params.measure_stats = 0;
    problem->cntrl_params.opt_level = -1;
    problem->cntrl_params.opt_off = 1;
    // 1D problem size: 3v40:2:2
    // dims/vecs in_stride and out_stride must be same for in-place problems
    problem->dims[0].n = 40;
    problem->dims[0].in_stride = 2;  // 0th dimension in_stride
    problem->dims[0].out_stride = 2; // 0th dimension out_stride
    problem->vecs[0].n = 3;
    // Calculate in_strides and out_strides for `dim_rank` dimensions and
    // `vec_rank`vector
    calculate_vecs_n_dims_strides_for_complex(problem->dims, problem->dim_rank,
                                              problem->vecs, problem->vec_rank);

    // Calculate input/output buffer sizes
    UINTP in_buffer_size = 0;
    UINTP out_buffer_size = 0;
    calculate_buffer_sizes(problem->dims, problem->dim_rank, problem->vecs,
                           problem->vec_rank, &in_buffer_size, &out_buffer_size);

    DOUBLE *in = NULL;
    DOUBLE *out = NULL;
    ALLOC(in, DOUBLE, (in_buffer_size * DATA_STRIDE));
    if (!is_inplace)
    {
        // create new output buffer for out-of-place problems
        ALLOC(out, DOUBLE, (out_buffer_size * DATA_STRIDE));
    }

    // prepare input for FFT calculation
    prepare_random_input(in, in_buffer_size);

    problem->in = in;
    if (is_inplace)
    {
        // Use input buffer as output for in-place buffer
        problem->out = in;
    }
    else
    {
        problem->out = out;
    }

    // setup call
    VOID *aoclfftz_handle = aoclfftz_setup_d(problem);

    if (aoclfftz_handle)
    {
        printf("\nSetup Successful\n");
        // execute call
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
    // destroy handle
    aoclfftz_destroy(aoclfftz_handle);
    free(problem->dims);
    free(problem->vecs);
    free(problem->in);
    if (!is_inplace)
    {
        free(problem->out);
    }
    free(problem);
}
