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

FFTZ_INTP calculate_size(aoclfftz_dim_t_64_ *dims, FFTZ_INT32 rank);
FFTZ_VOID calculate_buffer_sizes(FFTZ_INT32 dim_rank, FFTZ_INT32 vec_rank,
                                 aoclfftz_dim_t_64_ *dims,
                                 aoclfftz_dim_t_64_ *vecs,
                                 FFTZ_UINTP *in_buffer_size,
                                 FFTZ_UINTP *out_buffer_size,
                                 aoclfftz_bench_fft_type_t fft_type);
FFTZ_VOID prepare_index_map(FFTZ_INT32 dim_rank, FFTZ_INT32 vec_rank,
                            aoclfftz_dim_t_64_ *dims, aoclfftz_dim_t_64_ *vecs,
                            FFTZ_INTP *in_idx_map, FFTZ_INTP *out_idx_map,
                            aoclfftz_bench_fft_type_t fft_type,
                            FFTZ_UINT32 is_aligned);
FFTZ_VOID compute_index_map(FFTZ_INTP *idx_map, FFTZ_INTP *src_idx,
                            FFTZ_INTP dst_idx, aoclfftz_dim_t_64_ *dims,
                            FFTZ_INT32 rank, FFTZ_UINT8 is_input,
                            FFTZ_UINT8 is_half_complex);

#endif // SIZE_AND_INDEX_MAPPER_H
