// Copyright Advanced Micro Devices, Inc.
// SPDX-License-Identifier: BSD-3-Clause

/** @file size_and_index_mapper.h
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

#ifndef SIZE_AND_INDEX_MAPPER_H
#define SIZE_AND_INDEX_MAPPER_H

#include "api/aoclfftz.h"
#include "test/bench/aoclfftz_bench.h"

INTP calculate_size(aoclfftz_dim_t_64_ *dims, INT32 rank);
VOID calculate_buffer_sizes(INT32 dim_rank,  INT32 vec_rank,
                            aoclfftz_dim_t_64_ *dims, aoclfftz_dim_t_64_ *vecs,
                            UINTP *in_buffer_size, UINTP *out_buffer_size,
                            aoclfftz_bench_fft_type_t fft_type);
VOID prepare_index_map(INT32 dim_rank, INT32 vec_rank, aoclfftz_dim_t_64_ *dims,
                       aoclfftz_dim_t_64_ *vecs, INTP *in_idx_map,
                       INTP *out_idx_map, aoclfftz_bench_fft_type_t fft_type,
                       UINT32 is_aligned);
VOID compute_index_map(INTP *idx_map, INTP *src_idx, INTP dst_idx,
                       aoclfftz_dim_t_64_ *dims, INT32 rank, UINT8 is_input,
                       UINT8 is_half_complex);

#endif // SIZE_AND_INDEX_MAPPER_H
