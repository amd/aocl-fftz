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
                            UINTP *in_buffer_size, UINTP *out_buffer_size);
VOID prepare_index_map(INT32 dim_rank, INT32 vec_rank, aoclfftz_dim_t_64_ *dims,
                       aoclfftz_dim_t_64_ *vecs, INTP *in_idx_map,
                       INTP *out_idx_map, aoclfftz_bench_fft_type_t fft_type,
                       UINT32 is_aligned);
VOID compute_index_map(INTP *idx_map, INTP *src_idx, INTP dst_idx,
                       aoclfftz_dim_t_64_ *dims, INT32 rank, UINT8 is_input,
                       UINT8 is_half_complex);

#endif // SIZE_AND_INDEX_MAPPER_H
