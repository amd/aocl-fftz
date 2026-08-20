// Copyright Advanced Micro Devices, Inc.
// SPDX-License-Identifier: BSD-3-Clause

// A test program to validate the bit-reproducibility mode of AOCL-FFTZ library
//
// To build:
// gcc -o test_bit_repro test_bit_repro.c -I<path/to/api/headers>
// -Wl,-rpath=<path/to/lib/dir> -L<path/to/lib/dir> -laocl_fftz -lm
//
// author: Ashwin K. Godbole

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "aoclfftz.h"

#define DATA_STRIDE_c 2
#define DATA_STRIDE_r 1
#define DATA_STRIDE(x) DATA_STRIDE_##x

void usage(char* program_name)
{
    printf("Usage:\n");
    printf("This program validates the bit-reproducibility mode of AOCL-FFTZ "
           "library\n");
    printf("%s <signal_type> <size> <nthreads> <check_n_times>\n",
           program_name);
    printf("\nWhere:\n");
    printf("<signal_datatype>: 'D' or 'd' for double\n");
    printf("                  'F' or 'f' for float\n");
    printf("<signal_type>   : 'C' or 'c' for complex\n");
    printf("                  'R' or 'r' for real\n");
    printf("<size>          : Size of the FFT\n");
    printf("                  For 1D FFT: <D1>\n");
    printf("                  For ND FFT: <D1>x<D2>x...<DN>\n");
    printf("<nthreads>      : Number of threads to use\n");
    printf("<check_n_times> : Number of times to run the FFT and check for "
           "bit-reproducibility\n");
    printf("\nbuild instructions:\n");
    printf(
        "gcc -o test_bit_repro test_bit_repro.c -I<path/to/api/headers> "
        "-Wl,-rpath=<path/to/lib/dir> -L<path/to/lib/dir> -laocl_fftz -lm\n");
}

#define ALLOC(ptr, type, n_ele)                                                \
    {                                                                          \
        ptr = (type *)calloc(n_ele, sizeof(type));                             \
        if (ptr == NULL)                                                       \
        {                                                                      \
            printf("\nMemory allocation failed\n");                            \
            exit(EXIT_FAILURE);                                                \
        }                                                                      \
    }

// TODO: may overallocate for real sizes, verify
FFTZ_VOID calculate_buffer_sizes(aoclfftz_dim_t_64_ *dims, FFTZ_INT32 dim_rank,
                                 aoclfftz_dim_t_64_ *vecs, FFTZ_INT32 vec_rank,
                                 FFTZ_UINTP *in_buffer_size,
                                 FFTZ_UINTP *out_buffer_size,
                                 FFTZ_UINT8 is_real)
{
    FFTZ_UINTP in_size = 1;
    FFTZ_UINTP out_size = 1;
    for (FFTZ_INT32 i = 0; i < dim_rank; i++)
    {
        in_size += ((dims[i].n - 1) * (dims[i].in_stride));
        out_size += ((dims[i].n - 1) * (dims[i].out_stride));
    }
    for (FFTZ_INT32 i = 0; i < vec_rank; i++)
    {
        in_size += ((vecs[i].n - 1) * (vecs[i].in_stride));
        out_size += ((vecs[i].n - 1) * (vecs[i].out_stride));
    }
    *in_buffer_size = in_size;
    *out_buffer_size = is_real ? out_size * 2 : out_size;
}

// This function calculates the strides for complex inputs for (1D and ND)
FFTZ_VOID calculate_vecs_n_dims_strides_for_complex(aoclfftz_dim_t_64_ *dims,
                                               FFTZ_INT32 dim_rank,
                                               aoclfftz_dim_t_64_ *vecs,
                                               FFTZ_INT32 vec_rank)
{
    FFTZ_INT32 d;
    for (d = 0; d < dim_rank; d++)
    {
        if (d == 0)
        {
            dims[0].in_stride =
                (dims[0].in_stride <= 1) ? 1 : dims[0].in_stride;
            dims[0].out_stride =
                (dims[0].out_stride <= 1) ? 1 : dims[0].out_stride;
            continue;
        }
        dims[d].in_stride = dims[d - 1].in_stride * dims[d - 1].n;
        dims[d].out_stride = dims[d - 1].out_stride * dims[d - 1].n;
    }
    FFTZ_UINT32 default_vec_in_stride =
        dims[dim_rank - 1].in_stride * dims[dim_rank - 1].n;
    FFTZ_UINT32 default_vec_out_stride =
        dims[dim_rank - 1].out_stride * dims[dim_rank - 1].n;
    for (d = 0; d < vec_rank; d++)
    {
        if (d == 0)
        {
            vecs[0].in_stride = (vecs[0].in_stride <= 1) ? default_vec_in_stride
                                                         : vecs[0].in_stride;
            vecs[0].out_stride = (vecs[0].out_stride <= 1)
                                     ? default_vec_out_stride
                                     : vecs[0].out_stride;
            continue;
        }
        vecs[d].in_stride = vecs[d - 1].in_stride * vecs[d - 1].n;
        vecs[d].out_stride = vecs[d - 1].out_stride * vecs[d - 1].n;
    }
}

// This function calculates the in_stride and out_strides for dims and vecs.
// It assumes dim_rank and vec_rank to be 1.
FFTZ_VOID calculate_1D_vecs_dims_strides_for_real(aoclfftz_dim_t_64_ *dims,
                                                  FFTZ_INT32 dim_rank,
                                                  aoclfftz_dim_t_64_ *vecs,
                                                  FFTZ_INT32 vec_rank,
                                                  FFTZ_UINT8 direction,
                                                  FFTZ_UINT8 is_inplace)
{
    // Initializing dims
    dims[0].in_stride = (dims[0].in_stride <= 1) ? 1 : dims[0].in_stride;
    dims[0].out_stride = (dims[0].out_stride <= 1) ? 1 : dims[0].out_stride;

    // Initializing vecs
    FFTZ_INT32 def_vec_in_stride = 1;
    FFTZ_INT32 def_vec_out_stride = 1;

    if (direction && !is_inplace)
    {
        def_vec_in_stride = dims[0].n * dims[0].in_stride;
        def_vec_out_stride = (dims[0].n / 2 + 1) * dims[0].out_stride;
    }
    else if (direction && is_inplace)
    {
        def_vec_in_stride = (dims[0].n / 2 + 1) * dims[0].in_stride * 2;
        def_vec_out_stride = (dims[0].n / 2 + 1) * dims[0].out_stride;
    }
    else if (!direction && !is_inplace)
    {
        def_vec_in_stride = (dims[0].n / 2 + 1) * dims[0].in_stride;
        def_vec_out_stride = dims[0].n * dims[0].out_stride;
    }
    else if (!direction && is_inplace)
    {
        def_vec_in_stride = (dims[0].n / 2 + 1) * dims[0].in_stride;
        def_vec_out_stride = (dims[0].n / 2 + 1) * dims[0].out_stride * 2;
    }

    vecs[0].in_stride = (vecs[0].in_stride <= 1) ?
                            def_vec_in_stride : vecs[0].in_stride;
    vecs[0].out_stride = (vecs[0].out_stride <= 1) ?
                            def_vec_out_stride : vecs[0].out_stride;
}

FFTZ_VOID calculate_vecs_n_dims_strides(aoclfftz_dim_t_64_ *dims,
                                        FFTZ_INT32 dim_rank,
                                        aoclfftz_dim_t_64_ *vecs,
                                        FFTZ_INT32 vec_rank,
                                        FFTZ_UINT8 direction,
                                        FFTZ_UINT8 is_out_place,
                                        FFTZ_UINT8 is_real)
{
    if (is_real)
    {
        calculate_1D_vecs_dims_strides_for_real(dims, dim_rank, vecs, vec_rank,
                                                direction, !is_out_place);
    }
    else
    {
        calculate_vecs_n_dims_strides_for_complex(dims, dim_rank, vecs,
                                                  vec_rank);
    }
}

#define MAX_DIM_RANK 3
typedef struct cmd_args
{
    char valid;
    char signal_datatype;
    FFTZ_INT32 signal_type;
    FFTZ_INT32 nthreads;
    FFTZ_INT32 check_n_times;
    FFTZ_INT32 dim_rank;
    FFTZ_UINTP size[MAX_DIM_RANK];
} cmd_args;

cmd_args parse_args(int argc, char** argv)
{
    cmd_args args = {0};
    args.valid = 0;

    char signal_datatype = argv[1][0];
    if (signal_datatype == 'D' || signal_datatype == 'd')
    {
        args.signal_datatype = 'd';
        args.valid = 1;
    }
    else if (signal_datatype == 'F' || signal_datatype == 'f')
    {
        args.signal_datatype = 'f';
        args.valid = 1;
    }
    if (!args.valid)
    {
        printf("Invalid signal datatype: %c\n\n", signal_datatype);
        return args;
    }

    char signal_type = argv[2][0];

    if (signal_type == 'C' || signal_type == 'c')
    {
        args.signal_type = 0;
        args.valid = 1;
    }
    else if (signal_type == 'R' || signal_type == 'r')
    {
        args.signal_type = 1;
        args.valid = 1;
    }

    if (!args.valid)
    {
        printf("Invalid signal type: %c\n\n", signal_type);
        return args;
    }

    char* token;
    char* rest = argv[3];
    FFTZ_INT32 dim_index = 0;

    // Parse the size argument to extract dimensions
    while ((token = strtok_r(rest, "x", &rest)))
    {
        if (dim_index >= MAX_DIM_RANK)
        {
            args.valid = 0;
        }
        else
        {
            args.size[dim_index] = (FFTZ_UINTP)atoi(token);
        }
        dim_index++;
    }
    if (dim_index > 1 && args.signal_type == 1)
    {
        args.valid = 0;
    }

    if (!args.valid)
    {
        printf("Too many dimensions provided (max supported : %d) : %d\n\n",
               MAX_DIM_RANK, dim_index);
        return args;
    }

    args.dim_rank = dim_index;
    args.nthreads = atoi(argv[4]);
    args.valid = args.valid && (args.nthreads > 0);
    if (!args.valid)
    {
        printf("Invalid number of threads: %d\n\n", args.nthreads);
        return args;
    }

    if (args.nthreads > 1)
    {
        printf("Note: Ensure that the AOCL-FFTZ library is built with "
               "ENABLE_MULTI_THREADING=ON for multithreading execution\n\n");
    }

    args.check_n_times = atoi(argv[5]);
    args.valid = args.valid && (args.check_n_times > 0);
    if (!args.valid)
    {
        printf("Invalid no. of checks: %d\n\n", args.check_n_times);
        return args;
    }

    return args;
}

#define HEX_UINT64 "%tx"
#define HEX_UINT32 "%x"
#define HEX_(type) HEX_##type
#define HEX(type) HEX_(type)

#define print_complex_array_hex(name, type, arr, size)                         \
    {                                                                          \
        printf("%s = [", name);                                                \
        for (FFTZ_UINTP i = 0; i < size; i++) { \
              printf("(" HEX(type) ", " HEX(type) ") ", (type)arr[2 * i],      \
                     (type)arr[2 * i + 1]);                                    \
        }                                                                      \
        printf("]\n");                                                         \
    }

// This function can be used for real (1D) and complex(1D and ND) problems
#define prepare_random_input(in, input_size, datatype_char, signaltype_char)   \
    {                                                                          \
        FFTZ_INTP idx = 0; \
        for (idx = 0; idx < input_size * DATA_STRIDE(signaltype_char); ++idx)  \
        {                                                                      \
            in[idx] = (20.0 / RAND_MAX) * rand() - 10.0;                       \
        }                                                                      \
    }

#define problem_type(datatype_char) aoclfftz_prob_desc_##datatype_char##_64_
#define setup_api(datatype_char) aoclfftz_setup_##datatype_char##_64_
#define type_d FFTZ_DOUBLE
#define type_f FFTZ_FLOAT
#define data_type(datatype_char) type_##datatype_char
#define int_type_d FFTZ_UINT64
#define int_type_f FFTZ_UINT32
#define int_type(datatype_char) int_type_##datatype_char

#define check_bit_reproducibility(args, datatype_char, signaltype_char)        \
    {                                                                          \
        /* Create and initialize prob_desc params (using FFTZ_DOUBLE ILP64) */ \
        problem_type(datatype_char) * problem;                                 \
        ALLOC(problem, problem_type(datatype_char), 1);                        \
                                                                               \
        /* note: hard-coded vec_rank for simplicity */                         \
        problem->vec_rank = 1;                                                 \
                                                                               \
        problem->dim_rank = args.dim_rank;                                     \
                                                                               \
        /* create the initialise dims and vecs */                              \
        ALLOC(problem->dims, aoclfftz_dim_t_64_, problem->dim_rank);           \
        ALLOC(problem->vecs, aoclfftz_dim_t_64_, problem->vec_rank);           \
                                                                               \
        /* note: hard-coding the problem type */                               \
        /*       forward, in-order, out-of-place */                            \
        problem->flags.fft_type = args.signal_type;                            \
        problem->flags.fft_direction = 0;  /* forward */                       \
        problem->flags.storage_order = 0;  /* in-order */                      \
        problem->flags.fft_placement = 1;  /* out-of-place */                  \
        problem->flags.transpose_mode = 0; /* no-transpose */                  \
        problem->flags.bit_reproducibility =                             \
            1; /* enable bit-reproducibility */                                \
                                                                               \
        problem->pthr_fft.dynamic_load_model = 0;                              \
        problem->pthr_fft.num_threads = args.nthreads;                         \
        problem->cntrl_params.logger_mode = 0;                                 \
        problem->cntrl_params.measure_stats = 0;                               \
        problem->cntrl_params.opt_level = -1;                                  \
        problem->cntrl_params.opt_off = 1;                                     \
                                                                               \
        problem->vecs[0].n = 1;                                                \
                                                                               \
        /* Initialize the dims */                                              \
        for (FFTZ_INT32 i = 0; i < problem->dim_rank; i++) \
        {                                                                      \
            problem->dims[i].n = args.size[i];                                 \
        }                                                                      \
        problem->dims[0].in_stride = 1;                                        \
        problem->dims[0].out_stride = 1;                                       \
                                                                               \
        calculate_vecs_n_dims_strides(                                         \
            problem->dims, problem->dim_rank, problem->vecs,                   \
            problem->vec_rank, problem->flags.fft_direction,                   \
            problem->flags.fft_placement, args.signal_type);                   \
                                                                               \
        FFTZ_UINTP in_buffer_size = 0; \
        FFTZ_UINTP out_buffer_size = 0; \
        calculate_buffer_sizes(problem->dims, problem->dim_rank,               \
                               problem->vecs, problem->vec_rank,               \
                               &in_buffer_size, &out_buffer_size,              \
                               args.signal_type);                              \
                                                                               \
        data_type(datatype_char) *in = NULL;                                   \
        data_type(datatype_char) *out = NULL;                                  \
        data_type(datatype_char) *out_i = NULL;                                \
        ALLOC(in, data_type(datatype_char),                                    \
              (in_buffer_size * DATA_STRIDE(signaltype_char)));                \
        ALLOC(out, data_type(datatype_char),                                   \
              (out_buffer_size * DATA_STRIDE(signaltype_char)));               \
        ALLOC(out_i, data_type(datatype_char),                                 \
              (out_buffer_size * DATA_STRIDE(signaltype_char)));               \
                                                                               \
        /* prepare input for FFT calculation */                                \
        prepare_random_input(in, in_buffer_size, datatype_char,                \
                             signaltype_char);                                 \
                                                                               \
        /* for bit-reproducibility check, we run a 'setup and execute' pair */ \
        /* for (args.check_n_times + 1) times. */                              \
        /* the output of the first run is considered as reference output */    \
        /* the output of subsequent runs is compared with the reference        \
         * output*/                                                            \
                                                                               \
        {                                                                      \
            problem->in = in;                                                  \
            problem->out = out;                                                \
                                                                               \
            FFTZ_VOID *aoclfftz_handle = setup_api(datatype_char)(problem); \
            if (aoclfftz_handle)                                               \
            {                                                                  \
                FFTZ_INT32 res = aoclfftz_execute(aoclfftz_handle); \
                                                                               \
                if (res == AOCLFFTZ_EXECUTION_FAILURE)                         \
                {                                                              \
                    printf("\nExecution Failure\n");                           \
                }                                                              \
            }                                                                  \
            else                                                               \
            {                                                                  \
                printf("\nSetup Failure\n");                                   \
            }                                                                  \
                                                                               \
            aoclfftz_destroy(aoclfftz_handle);                                 \
        }                                                                      \
                                                                               \
        /* now run `n_check_times` runs and compare the output to the          \
         * reference */                                                        \
        /* output (out) */                                                     \
        for (FFTZ_INT32 i = 0; i < args.check_n_times; i++) \
        {                                                                      \
            problem->in = in;                                                  \
            problem->out = out_i;                                              \
                                                                               \
            FFTZ_VOID *aoclfftz_handle = setup_api(datatype_char)(problem); \
            if (aoclfftz_handle)                                               \
            {                                                                  \
                FFTZ_INT32 res = aoclfftz_execute(aoclfftz_handle); \
                                                                               \
                if (res == AOCLFFTZ_EXECUTION_FAILURE)                         \
                {                                                              \
                    printf("\nExecution Failure\n");                           \
                }                                                              \
                else                                                           \
                {                                                              \
                    /* do a bitwise compare of every element of out and out_i  \
                     */                                                        \
                    /* this has to be done for both components of the complex  \
                     * number */                                               \
                    FFTZ_INT32 mis_match = 0; \
                    for (FFTZ_UINTP j = 0; \
                         j < out_buffer_size * DATA_STRIDE(signaltype_char);   \
                         j++)                                                  \
                    {                                                          \
                        int_type(datatype_char) current =                      \
                            (int_type(datatype_char))out[j];                   \
                        int_type(datatype_char) reference =                    \
                            (int_type(datatype_char))out_i[j];                 \
                        if (current != reference)                              \
                        {                                                      \
                            printf("Mismatch at index %zu: "                   \
                                    HEX(int_type(datatype_char))               \
                                    " vs "                                     \
                                    HEX(int_type(datatype_char))               \
                                    "\n", j,                                   \
                                   current, reference);                        \
                            mis_match = 1;                                     \
                            break;                                             \
                        }                                                      \
                    }                                                          \
                                                                               \
                    if (mis_match)                                             \
                    {                                                          \
                        printf("Run %5d: FAIL\n", i + 1);                      \
                        print_complex_array_hex("Current   Output",            \
                                                int_type(datatype_char),       \
                                                out_i, out_buffer_size);       \
                        print_complex_array_hex("Reference Output",            \
                                                int_type(datatype_char), out,  \
                                                out_buffer_size);              \
                    }                                                          \
                    else                                                       \
                    {                                                          \
                        printf("Run %5d: PASS\n", i + 1);                      \
                    }                                                          \
                }                                                              \
            }                                                                  \
            else                                                               \
            {                                                                  \
                printf("\nSetup Failure\n");                                   \
            }                                                                  \
                                                                               \
            aoclfftz_destroy(aoclfftz_handle);                                 \
        }                                                                      \
                                                                               \
        problem->in = NULL;                                                    \
        problem->out = NULL;                                                   \
        free(problem->dims);                                                   \
        free(problem->vecs);                                                   \
        free(in);                                                              \
        free(out);                                                             \
        free(out_i);                                                           \
        free(problem);                                                         \
    }

int main(int argc, char** argv)
{
    if (argc != 6)
    {
        printf("Invalid number of arguments (expected %d): %d\n\n", 6,
               argc - 1);
        usage(argv[0]);
        return -1;
    }

    // parse command line arguments
    cmd_args args = parse_args(argc, argv);
    if (!args.valid)
    {
        usage(argv[0]);
        return -1;
    }

    if (args.signal_datatype == 'd')
    {
        if (args.signal_type == 0) // complex
        {
            check_bit_reproducibility(args, d, c);
        }
        else
        {
            check_bit_reproducibility(args, d, r);
        }
    }
    else if (args.signal_datatype == 'f')
    {
        if (args.signal_type == 0) // complex
        {
            check_bit_reproducibility(args, f, c);
        }
        else
        {
            check_bit_reproducibility(args, f, r);
        }
    }

    return 0;
}
