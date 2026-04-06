// Copyright Advanced Micro Devices, Inc.
// SPDX-License-Identifier: BSD-3-Clause
/** @file size_and_index_mapper.c
 *
 *  @brief Problem size and dims related utility functions.
 *
 *  This file contains the test bench utility functions related to problem size,
 *  dims and strides for test bench.
 *
 *  @author V. Murugan
 *  @author Srirammaswamy Srinivasan
 *  @author Jeya R
 */
#include <string.h>
#include "test/bench/utils/size_and_index_mapper.h"
#include "utils/allocator.h"

/**
 * @brief calculates the total size without strides
 *
 * @param dims holds dims related info
 * @param rank rank of the provided dims
 * @return INTP length of the input
 */
INTP calculate_size(aoclfftz_dim_t_64_ *dims, INT32 rank)
{
    INTP len = 1;
    for (INT32 i = 0; i < rank; i++)
    {
        len = len * dims[i].n;
    }
    return len;
}

/**
 * @brief Function to find the required buffer size for memory allocation.
 * calculates the total length of input & output buffers with strides included
 *
 * @param dim_rank rank of the dims array
 * @param vec_rank rank of the vecs array
 * @param dims aoclfftz_dim_t_64_ struct holding problem size and strides
 * @param vecs aoclfftz_dim_t_64_ struct holding batch size and strides
 * @param in_buffer_size register to store input buffer size
 * @param out_buffer_size register to store output buffer size
 * @param fft_type FFT type (C2C, R2C, or C2R)
 * @return VOID
 */
VOID calculate_buffer_sizes(INT32 dim_rank,  INT32 vec_rank,
                            aoclfftz_dim_t_64_ *dims, aoclfftz_dim_t_64_ *vecs,
                            UINTP *in_buffer_size, UINTP *out_buffer_size,
                            aoclfftz_bench_fft_type_t fft_type)
{
    // Example: for an 1D problem with 1D batch
    // Problem size : 3:6:6v4:1:1
    // Data arrangement considered :
    // [1, 2, 3, 4]<0, 0>[5, 6, 7, 8]<0, 0>[9, 10, 11, 12]
    // <---vec stride--->
    // <-------------(Batches -1)---------><--- Problem size * dim stride --->
    // ((Batches -1) * (vec_stride)) + (Problem size * dim stride)
    in_buffer_size[0] = 0;
    out_buffer_size[0] = 0;
    UINTP in_size = 1;
    UINTP out_size = 1;
    for (INT32 i = 0; i < dim_rank; i++)
    {
        INTP in_len = dims[i].n;
        INTP out_len = dims[i].n;
        // Half-complex format adjustment for innermost dimension (i=0):
        // - C2R: Input is half-complex with (n/2 + 1) complex values
        // - R2C: Output is half-complex with (n/2 + 1) complex values
        if (i == 0)
        {
            if (fft_type == C2R)
            {
                in_len = (dims[i].n / 2) + 1;
            }
            else if (fft_type == R2C)
            {
                out_len = (dims[i].n / 2) + 1;
            }
        }
        in_size += ((in_len - 1) * (dims[i].in_stride));
        out_size += ((out_len - 1) * (dims[i].out_stride));
    }
    for (INT32 i = 0; i < vec_rank; i++)
    {
        in_size += ((vecs[i].n - 1) * (vecs[i].in_stride));
        out_size += ((vecs[i].n - 1) * (vecs[i].out_stride));
    }
    in_buffer_size[0] = in_size;
    out_buffer_size[0] = out_size;
}

/**
 * @brief prepare the index map to map non strided indices to strides ones
 *
 * index map is used to simplify the property tests for strided problems
 *
 * Example: for an 1D problem with 1D batch
 * Problem size : 3:6:6v4:1:1
 * Data arrangement considered :
 * [1, 2, 3, 4]<0, 0>[5, 6, 7, 8]<0, 0>[9, 10, 11, 12]
 * <---vec stride--->
 * <------------(Batches - 1)--------->
 *
 * data buffer   => [1, 2, 3, 4, 0, 0, 5, 6, 7, 8,  0,  0,  9, 10, 11, 12]
 * (here data in the strides are considered as 0)
 * indices       => [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]
 * valid indices => [0, 1, 2, 3,       6, 7, 8, 9,         12, 13, 14, 15]
 * (ignoring the indices of strides)
 *
 * index map for this configuration in key, value representation :
 * key : valid data index for the given problem without strides
 * value : valid data index for the given problem with strides
 * [(0,0), (1,1), (2,2), (3,3), (4,6), (5,7), (6,8), (7,9), (8,12), (9,13), (10,14), (11,15)]
 * simplified index map (in array representation, where array index is the key) :
 * [0, 1, 2, 3, 6, 7, 8, 9, 12, 13, 14, 15]
 *
 * @param dim_rank rank of the dims array
 * @param vec_rank rank of the vecs array
 * @param dims aoclfftz_dim_t_64_ struct holding problem size and strides
 * @param vecs aoclfftz_dim_t_64_ struct holding batch size and strides
 * @param in_idx_map Buffer to store input index map
 * @param out_idx_map Buffer to store output index map
 * @param fft_type FFT type (C2C, R2C, or C2R)
 * @param is_aligned Flag indicating if memory allocation should be aligned
 * @return VOID
 */
VOID prepare_index_map(INT32 dim_rank, INT32 vec_rank, aoclfftz_dim_t_64_ *dims,
                       aoclfftz_dim_t_64_ *vecs, INTP *in_idx_map,
                       INTP *out_idx_map, aoclfftz_bench_fft_type_t fft_type,
                       UINT32 is_aligned)
{
    // combine dims and vecs
    aoclfftz_dim_t_64_ *combined_dims = NULL;
    ALLOC_UNINIT(combined_dims, aoclfftz_dim_t_64_, sizeof(aoclfftz_dim_t_64_) *
                 (dim_rank + vec_rank), is_aligned);
    memcpy(combined_dims, dims, (sizeof(aoclfftz_dim_t_64_) * dim_rank));
    memcpy((combined_dims + dim_rank), vecs,
           (sizeof(aoclfftz_dim_t_64_) * vec_rank));
    INT32 combined_rank = dim_rank + vec_rank;

    UINT8 is_half_complex_input = fft_type == C2R ? 1 : 0;
    UINT8 is_half_complex_output = fft_type == R2C ? 1 : 0;

    // Compute index map for input
    INTP src_idx = 0;
    INTP dst_idx = 0;
    compute_index_map(in_idx_map, &src_idx, dst_idx, combined_dims,
                      combined_rank, 1 /* input */, is_half_complex_input);
    // Compute index map for output
    src_idx = 0;
    dst_idx = 0;
    compute_index_map(out_idx_map, &src_idx, dst_idx, combined_dims,
                      combined_rank, 0 /* output */, is_half_complex_output);
    FREE_ALLOCATED_MEM(combined_dims, is_aligned);
}

/**
 * @brief recursive algorithm to compute index map
 *
 * @param in_idx_map buffer to store input index map
 * @param out_idx_map buffer to store output index map
 * @param src_idx source index for input and output buffers
 * @param dst_in_idx destination index for input buffer
 * @param dst_out_idx destination index for output buffer
 * @param dims aoclfftz_dim_t_64_ struct which holds problem size and strides
 * @param rank current rank of the nD problem
 * @return VOID
 */
VOID compute_index_map(INTP *idx_map, INTP *src_idx, INTP dst_idx,
                       aoclfftz_dim_t_64_ *dims, INT32 rank, UINT8 is_input,
                       UINT8 is_half_complex)
{
    if (rank == 0)
    {
        idx_map[*src_idx] = dst_idx;
        (*src_idx)++;
    }
    else
    {
        INTP n = dims[rank - 1].n;
        // TODO: Check and update this logic for Real ND problems
        // Here rank = 1 represents the innermost dimension of the problem
        // which should be treated as half-complex for C2R and R2C transforms
        if (is_half_complex && rank == 1)
        {
            n = (n / 2) + 1;
        }
        INTP stride =
            is_input ? dims[rank - 1].in_stride : dims[rank - 1].out_stride;
        for (INTP i = 0; i < n; i++)
        {
            compute_index_map(idx_map, src_idx, dst_idx, dims, rank - 1,
                              is_input, is_half_complex);
            dst_idx += stride;
        }
    }
}
