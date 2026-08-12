// Copyright Advanced Micro Devices, Inc.
// SPDX-License-Identifier: BSD-3-Clause

/** @file dims_vecs_helper.h
 *
 *  @brief Helper functions for dims and vecs.
 *
 *  This file contains the helper function prototypes for dims and vecs for
 *  test bench.
 *
 *  @author S. Biplab Raut
 *  @author V. Murugan
 *  @author Srirammaswamy Srinivasan
 */

#ifndef DIMS_VECS_HELPER_H
#define DIMS_VECS_HELPER_H

#include <ctype.h>
#include "api/aoclfftz.h"
#include "test/bench/utils/size_and_index_mapper.h"
#include "utils/allocator.h"

EXPORT_SYM_DYN
FFTZ_INT32 allocate_and_fill_dims_vecs(FFTZ_CHAR *arg, FFTZ_INT32 dim_rank,
                                       FFTZ_INT32 vec_rank,
                                       aoclfftz_dim_t_64_ **dims,
                                       aoclfftz_dim_t_64_ **vecs,
                                       FFTZ_INTP default_stride);
EXPORT_SYM_DYN
FFTZ_VOID set_default_dims_vecs(FFTZ_INT32 dim_rank, FFTZ_INT32 vec_rank,
                                aoclfftz_dim_t_64_ *dims,
                                aoclfftz_dim_t_64_ *vecs,
                                aoclfftz_bench_fft_type_t type,
                                FFTZ_UINT8 is_in_place, FFTZ_UINT8 logger_mode);

EXPORT_SYM_DYN
FFTZ_INT32 find_dim_vec_ranks(FFTZ_CHAR *arg, FFTZ_INT32 *dim_rank,
                              FFTZ_INT32 *vec_rank);

#endif // DIMS_VECS_HELPER_H
