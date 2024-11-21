/*
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

/** @file fuzz_utils.h
 *
 *  @brief Utility functions and macros for fuzz testing
 *
 *  This file contains utility functions and macros related to fuzz tests.
 *
 *  @author Jeya R
 *  @author Maheswar Rao S
 */

#ifndef FUZZ_UTILS_H
#define FUZZ_UTILS_H
#include <cstring>
#include <stdio.h>
#include <algorithm>
#include "gtest/gtest.h"
#include "fuzztest/fuzztest.h"
extern "C"
{
#include "api/aoclfftz_internal.h"
#include "api/aoclfftz.h"
#include "api/types.h"
#include "utils/allocator.h"
#include "test/bench/utils/size_and_index_mapper.h"
#include "test/bench/accuracy.h"
#include "test/bench/utils/bench_utils.h"
#include "test/bench/aoclfftz_bench.h"
#include "test/bench/bench_problem.h"
#include "test/utils/dims_vecs_helper.h"
#include "test/bench/utils/register_functions.h"
}

#define ALIGNED_ALLOC 1
// Range limiter for fuzzing input values
#define RANGE_LIMITER 100000

/**
 * @brief Initialize benchmark parameters.
 *
 * @tparam dt_t Data type of input and output (supported: FLOAT and DOUBLE)
 * @tparam dm_t Data model for data length and strides
 * supported: LP64[INT32] and ILP64[INTP])
 * @param params Pointer to benchmark parameters structure
 * @param dim_rank Dimension rank
 * @param vec_rank Vector rank
 * @param dims Pointer to dimensions structure
 * @param vecs Pointer to vectors structure
 */
template <typename dt_t, typename dm_t>
void init_bench_params(aoclfftz_bench_params_t* params, INT32 dim_rank,
            INT32 vec_rank, aoclfftz_dim_t_64_* dims, aoclfftz_dim_t_64_* vecs)
{
    if (params)
    {
        params->dim_rank = dim_rank;
        params->vec_rank = vec_rank;
        params->dims = dims;
        params->vecs = vecs;
        if (typeid(dt_t) == typeid(FLOAT))
        {
            params->precision = FLOAT_P;
        }
        else
        {
            params->precision = DOUBLE_P;
        }
        if (typeid(dm_t) == typeid(INT32))
        {
            params->data_model = LP64;
        }
        else
        {
            params->data_model = ILP64;
        }
        params->bench_type = ACCURACY;
        params->res_placement = OUT_OF_PLACE;
        params->order = IN_ORDER;
        params->dir = FORWARD;
        params->fft_type = COMPLEX_TO_COMPLEX;
        params->num_iterations = 1;
        params->num_threads = 1;
        params->dynamic_load_model = 0;
        params->opt_level = -1;
        params->logger_mode = 0;
        params->selector_time = 0;
        params->measure_stats = 0;
        params->bit_reproducibility = 0;
        params->aligned_alloc = ALIGNED_ALLOC;
        params->seed = 0;
        params->use_random_seed = 1;
        if (params->precision == FLOAT_P)
        {
            params->tolerance = 1E-3;
        }
        else
        {
            params->tolerance = 1E-10;
        }
        // create input and output buffers
        INT32 dt_bytes = (params->precision == FLOAT_P) ?
                                        sizeof(FLOAT) : sizeof(DOUBLE);
        UINT32 is_align = params->aligned_alloc;
        ALLOC_UNINIT(params->in, VOID, params->sz_info.input_size *
                                        dt_bytes * T_DATA_STRIDE, is_align);
        ALLOC_INIT(params->out, VOID, params->sz_info.output_size *
                                        dt_bytes * T_DATA_STRIDE, is_align);
    }
}

/**
 * @brief Constructs dimensions and vector information for the given problem
 * from the given array.
 *
 * @param dims_vecs static array which have information about
 * dimensions & vectors.
 * The array is consumed differently for every problem domain as follows:
 * 1D FFT :
 * {dim_rank, vec_rank, vec[0].n, vec[0].in_stride, vec[0].out_stride,
 *                               dim[0].n, dim[0].in_stride, dim[0].out_stride}
 * Batched 1D FFT :
 * {dim_rank, vec_rank, vec[0].n, vec[0].in_stride, vec[0].out_stride,
 *                              dim[0].n, dim[0].in_stride, dim[0].out_stride}
 * 2D FFT :
 * {dim_rank, vec_rank, dim[0].n, dim[0].in_stride, dim[0].out_stride,
 *                              dim[1].n, dim[1].in_stride, dim[1].out_stride}
 * Multi batched FFT :
 * {dim_rank, vec_rank, vec[vec_rank - 1].n, place holder, place holder,
 *                              place holder, place holder, place holder}
 * ND FFT :
 * {dim_rank, vec_rank, dim[dim_rank - 1].n, place holder, place holder,
 *                                  place holder, place holder, place holder}
 * @param dims pointer to store the dims structure
 * @param vecs pointer to store the vecs structure
 */
VOID construct_dims_and_vecs(const std::array<INTP,8>&dims_vecs,
                             aoclfftz_dim_t_64_ **dims,
                             aoclfftz_dim_t_64_ **vecs)
{
    INT32 dim_rank = dims_vecs[0];
    INT32 vec_rank = dims_vecs[1];
    ALLOC_ALIGN_INIT((*dims),
        aoclfftz_dim_t_64_, dim_rank * sizeof(aoclfftz_dim_t_64_));
    ALLOC_ALIGN_INIT((*vecs),
        aoclfftz_dim_t_64_, vec_rank * sizeof(aoclfftz_dim_t_64_));
    (*dims)[0].n = dims_vecs[2];
    (*dims)[0].in_stride = dims_vecs[3];
    (*dims)[0].out_stride = dims_vecs[4];
    (*vecs)[0].n = dims_vecs[5];
    (*vecs)[0].in_stride = dims_vecs[6];
    (*vecs)[0].out_stride = dims_vecs[7];
    if (dim_rank == 2)
    {
        (*dims)[0].n = dims_vecs[2];
        (*dims)[0].in_stride = dims_vecs[3];
        (*dims)[0].out_stride = dims_vecs[4];
        (*dims)[1].n = dims_vecs[5];
        (*dims)[1].in_stride = dims_vecs[6];
        (*dims)[1].out_stride = dims_vecs[7];
        (*vecs)[0].n = 1;
        (*vecs)[0].in_stride = 1;
        (*vecs)[0].out_stride = 1;
    }
    INT32 i;
    //special cases
    if (vec_rank > 1)
    {
        (*vecs)[0].n = 2;
        (*vecs)[0].in_stride = 1;
        (*vecs)[0].out_stride = 1;
        for (i = 1; i < vec_rank; i++)
        {
            (*vecs)[i].n = 2;
            (*vecs)[i].in_stride = (*vecs)[i - 1].n * (*vecs)[i - 1].in_stride;
            (*vecs)[i].out_stride = (*vecs)[i - 1].n *
                                    (*vecs)[i - 1].out_stride;
        }
        //Overwrite it with the fuzzed value for the last vector size
        (*vecs)[i - 1].n = dims_vecs[2];
    }
    if (dim_rank > 2)
    {
        (*vecs)[0].n = 1;
        (*vecs)[0].in_stride = 1;
        (*vecs)[0].out_stride = 1;
        for (i = 1; i < dim_rank; i++)
        {
            (*dims)[i].n = 2;
            (*dims)[i].in_stride = (*dims)[i - 1].n * (*dims)[i - 1].in_stride;
            (*dims)[i].out_stride = (*dims)[i - 1].n *
                                    (*dims)[i - 1].out_stride;
        }
        // Overwrite it with the fuzzed value for the last dim size
        (*dims)[i - 1].n = dims_vecs[2];
    }
}

/**
 * @brief Generates restricted dimension strides for 1D FFT problems.
 *
 * This function returns an array representing the dimension strides
 * for 1D FFT problems.
 * The array format is: {dim_rank, vec_rank, dims[0].n, dims[0].in_stride,
 *  dims[0].out_stride, vecs[0].n, vecs[0].in_stride, vecs[0].out_stride}.
 * The dimension strides are restricted to prevent out-of-memory (OOM) issues.
 *
 * @param dim_size Dimension size. Range: [1, INT32_MAX / 8]
 * @return Array of dimension strides with restricted dim.in_stride and
 * dim.out_stride from 1 to INT32_MAX / dim_size.
 */
auto restricted_dim_stride()
{
    return fuzztest::FlatMap([](INT32 dim_size)
    {
        return fuzztest::ArrayOf(fuzztest::Just(static_cast<INTP>(1)),
                       fuzztest::Just(static_cast<INTP>(1)),
                       fuzztest::Just(static_cast<INTP>(dim_size)),
                       fuzztest::InRange(static_cast<INTP>(1),
                               static_cast<INTP>(INT32_MAX / dim_size)),
                       fuzztest::InRange(static_cast<INTP>(1),
                               static_cast<INTP>(INT32_MAX / dim_size)),
                       fuzztest::Just(static_cast<INTP>(1)),
                       fuzztest::Just(static_cast<INTP>(1)),
                       fuzztest::Just(static_cast<INTP>(1)));
    },
    // Max. Range for dim size is scaled down to prevent OOM
    fuzztest::InRange(1, INT32_MAX / 8));
}

/**
 * @brief Generates restricted dimension sizes for 1D FFT problems.
 *
 * This function returns an array representing the dimension sizes for
 * 1D FFT problems.
 * The array format is: {dim_rank, vec_rank, dims[0].n, dims[0].in_stride,
 *  dims[0].out_stride, vecs[0].n, vecs[0].in_stride, vecs[0].out_stride}.
 * The dimension sizes are restricted to prevent OOM issues.
 *
 * @param dim_stride Dimension stride. Range: [1, INT32_MAX]
 * @return Array of dimension sizes with restricted dims.n from
 * 1 to INT32_MAX / dim_stride.
 */
auto restricted_dim_size()
{
    return fuzztest::FlatMap([](INT32 dim_stride)
    {
        INTP dim_max_size = INT32_MAX / dim_stride;
        if (dim_stride <= 8)
        {
            // Scaled down to prevent OOM
            dim_max_size = INT32_MAX / (dim_stride * 8);
        }
        return fuzztest::ArrayOf(fuzztest::Just(static_cast<INTP>(1)),
                       fuzztest::Just(static_cast<INTP>(1)),
                       // Scaled down to prevent OOM
                       fuzztest::InRange(static_cast<INTP>(1),
                               static_cast<INTP>(dim_max_size)),
                       fuzztest::InRange(static_cast<INTP>(1),
                               static_cast<INTP>(dim_stride)),
                       fuzztest::Just(static_cast<INTP>(dim_stride)),
                       fuzztest::Just(static_cast<INTP>(1)),
                       fuzztest::Just(static_cast<INTP>(1)),
                       fuzztest::Just(static_cast<INTP>(1)));
    },
    fuzztest::InRange(1, INT32_MAX));
}

/**
 * @brief Generates dimension and vector sizes for 1D FFT problems.
 * @return At runtime either 1D FFT with restricted dim_stride or
 * restricted dim_size array gets generated.
 */
auto dims_and_vecs_1D()
{
    return fuzztest::OneOf(restricted_dim_stride(), restricted_dim_size());
}

/**
 * @brief Generates restricted vector strides for batched 1D FFT problems.
 *
 * This function returns an array representing the vector strides for
 * batched 1D FFT problems.
 * The array format is: {dim_rank, vec_rank, dims[0].n, dims[0].in_stride,
 *  dims[0].out_stride, vecs[0].n, vecs[0].in_stride, vecs[0].out_stride}.
 * The vector strides are restricted to prevent OOM issues.
 *
 * @param dim_size Dimension size. Range: [1, 100]
 * @param dim_instride Dimension input stride. Range: [1, 10]
 * @param dim_outstride Dimension output stride. Range: [1, 10]
 * @return Array of vector strides with restricted vecs.in_stride and
 * vecs.out_stride based on dim_size.
 */
auto restricted_vec_stride()
{
    return fuzztest::FlatMap([](INT32 dim_size, INT32 dim_instride,
                                INT32 dim_outstride)
    {
        INT32 vector_instride = dim_size * dim_instride;
        INT32 vector_outstride = dim_size * dim_outstride;
        INT32 vector_size_max = std::min(((INT32_MAX - vector_instride) /
                                          vector_instride),
                                         ((INT32_MAX - vector_outstride) /
                                         vector_outstride));
        return fuzztest::ArrayOf(fuzztest::Just(static_cast<INTP>(1)),
                       fuzztest::Just(static_cast<INTP>(1)),
                       fuzztest::Just(static_cast<INTP>(dim_size)),
                       fuzztest::Just(static_cast<INTP>(dim_instride)),
                       fuzztest::Just(static_cast<INTP>(dim_outstride)),
                       fuzztest::InRange(static_cast<INTP>(2),
                               static_cast<INTP>(vector_size_max)),
                       fuzztest::Just(static_cast<INTP>(vector_instride)),
                       fuzztest::Just(static_cast<INTP>(vector_outstride)));
    },
    fuzztest::InRange(1, 100), fuzztest::InRange(1, 10),
    fuzztest::InRange(1, 10));
}

/**
 * @brief Generates restricted vector sizes for batched 1D FFT problems.
 *
 * This function returns an array representing the vector sizes for
 * batched 1D FFT problems.
 * The array format is: {dim_rank, vec_rank, dims[0].n, dims[0].in_stride,
 *  dims[0].out_stride, vecs[0].n, vecs[0].in_stride, vecs[0].out_stride}.
 * The vector sizes are restricted to prevent OOM issues.
 *
 * @param dim_size Dimension size. Range: [1, 100]
 * @param dim_instride Dimension input stride. Range: [1, 10]
 * @param dim_outstride Dimension output stride. Range: [1, 10]
 * @param vec_size Vector size. Range: [1, 10]
 * @return Array of vector sizes with restricted vecs.n based on
 * dim_size and dim_stride.
 */
auto restricted_vec_size()
{
    return fuzztest::FlatMap([](INT32 dim_size, INT32 dim_instride,
                                INT32 dim_outstride, INT32 vec_size)
    {
        INT32 vector_instride_max =
                           (INT32_MAX - (dim_size * dim_instride)) / vec_size;
        INT32 vector_outstride_max =
                           (INT32_MAX - (dim_size * dim_outstride)) / vec_size;
        return fuzztest::ArrayOf(fuzztest::Just(static_cast<INTP>(1)),
                       fuzztest::Just(static_cast<INTP>(1)),
                       fuzztest::Just(static_cast<INTP>(dim_size)),
                       fuzztest::Just(static_cast<INTP>(dim_instride)),
                       fuzztest::Just(static_cast<INTP>(dim_outstride)),
                       fuzztest::Just(static_cast<INTP>(vec_size)),
                       fuzztest::InRange(static_cast<INTP>(dim_size *
                                                           dim_instride),
                               static_cast<INTP>(vector_instride_max)),
                       fuzztest::InRange(static_cast<INTP>(dim_size *
                                                           dim_outstride),
                               static_cast<INTP>(vector_outstride_max)));
    },
    fuzztest::InRange(1, 100), fuzztest::InRange(1, 10),
    fuzztest::InRange(1, 10), fuzztest::InRange(1, 10));
}

/**
 * @brief Generates dimension and vector sizes for batched 1D FFT problems.
 * @return At runtime either batched 1D FFT with restricted vec_stride or
 * restricted vec_size array gets generated.
 */
auto dims_and_vecs_batched_1D()
{
    return fuzztest::OneOf(restricted_vec_size(), restricted_vec_stride());
}

/**
 * @brief Generates restricted dimension sizes for 2D FFT problems.
 *
 * This function returns an array representing the dimension sizes for
 * 2D FFT problems.
 * The array format is: {dim_rank, vec_rank, dims[0].n, dims[0].in_stride,
 * dims[0].out_stride, dims[1].n, dims[1].in_stride, dims[1].out_stride}.
 * The dimension sizes are restricted to prevent OOM issues.
 *
 * @param dim1_size for the first dimension.Range: [1, 10]
 * @param dim1_instride for the first dimension. Range: [1, 10]
 * @param dim1_outstride for the first dimension. Range: [1, 10]
 * @return Array of dimension sizes for the first dimension with
 * restricted dims.n for the second dimension.
 */
auto restricted_dim1_size()
{
    return fuzztest::FlatMap([](INT32 dim1_size, INT32 dim1_instride,
                                INT32 dim1_outstride)
    {
        INT32 dim2_instride = dim1_size * dim1_instride;
        INT32 dim2_outstride = dim1_size * dim1_outstride;
        INT32 dim2_size_max = std::min((INT32_MAX / (dim2_instride *
                                                    dim2_instride)),
                                       (INT32_MAX / (dim2_outstride *
                                                    dim2_outstride)));
        return fuzztest::ArrayOf(fuzztest::Just(static_cast<INTP>(2)),
                       fuzztest::Just(static_cast<INTP>(1)),
                       fuzztest::Just(static_cast<INTP>(dim1_size)),
                       fuzztest::Just(static_cast<INTP>(dim1_instride)),
                       fuzztest::Just(static_cast<INTP>(dim1_outstride)),
                       fuzztest::InRange(static_cast<INTP>(2),
                               static_cast<INTP>(dim2_size_max)),
                       fuzztest::Just(static_cast<INTP>(dim2_instride)),
                       fuzztest::Just(static_cast<INTP>(dim2_outstride)));
    },
    fuzztest::InRange(1, 10), fuzztest::InRange(1, 10),
    fuzztest::InRange(1, 10));
}

/**
 * @brief Generates restricted dimension sizes for 2D FFT problems.
 *
 * This function returns an array representing the dimension sizes for
 * 2D FFT problems.
 * The array format is: {dim_rank, vec_rank, dims[0].n, dims[0].in_stride,
 * dims[0].out_stride, dims[1].n, dims[1].in_stride, dims[1].out_stride}.
 * The dimension sizes are restricted to prevent OOM issues.
 *
 * @param dim1_size for the first dimension. Range: [1, INT32_MAX / 110]
 * @param dim1_instride for the first dimension. Range: [1, 10]
 * @param dim1_outstride for the first dimension. Range: [1, 10]
 * @return Array of dimension sizes for the second dimension with
 * restricted dims.n for the first dimension.
 */
auto restricted_dim2_size()
{
    return fuzztest::FlatMap([](INT32 dim1_size, INT32 dim1_instride,
                                INT32 dim1_outstride)
    {
        INT32 dim2_instride = dim1_size * dim1_instride;
        INT32 dim2_outstride = dim1_size * dim1_outstride;
        return fuzztest::ArrayOf(fuzztest::Just(static_cast<INTP>(2)),
                       fuzztest::Just(static_cast<INTP>(1)),
                       fuzztest::Just(static_cast<INTP>(dim1_size)),
                       fuzztest::Just(static_cast<INTP>(dim1_instride)),
                       fuzztest::Just(static_cast<INTP>(dim1_outstride)),
                       fuzztest::InRange(static_cast<INTP>(1),
                               static_cast<INTP>(10)),
                       fuzztest::Just(static_cast<INTP>(dim2_instride)),
                       fuzztest::Just(static_cast<INTP>(dim2_outstride)));
    },
    fuzztest::InRange(1, INT32_MAX / 110), fuzztest::InRange(1, 10),
    fuzztest::InRange(1, 10));
}

/**
 * @brief Generates dimension and vector sizes for 2D FFT problems.
 * @return At runtime either 2D FFT with restricted dim1_size or
 * restricted dim2_size array gets generated.
 */
auto dims_and_vecs_2D()
{
    return fuzztest::OneOf(restricted_dim1_size(), restricted_dim2_size());
}

/**
 * @brief Generates dimension and vector sizes for
 * multi-batched 1D FFT problems.
 *
 * This function returns an array representing the dimension and
 * vector sizes for multi-batched 1D FFT problems.
 * The array format is: {dim_rank, vec_rank, vec[vec_rank - 1].n, place holder,
 * place holder, place holder, place holder, place holder}
 * The vector sizes are restricted to prevent OOM issues.
 *
 * @param vec_rank Vector rank. Range: [2, 28]
 * @return Array of dimension and vector sizes with restricted
 * vecs.n based on vec_rank.
 */
auto dims_and_vecs_multi_batched_1D()
{
    return fuzztest::FlatMap([](INT32 vec_rank)
    {
        INT32 vec_size_max = (INT32_MAX / pow(2, vec_rank + 1)) - 2;
        return fuzztest::ArrayOf(fuzztest::Just(static_cast<INTP>(1)),
                       fuzztest::Just(static_cast<INTP>(vec_rank)),
                       fuzztest::InRange(static_cast<INTP>(1),
                               static_cast<INTP>(vec_size_max)),
                       fuzztest::Just(static_cast<INTP>(1)),
                       fuzztest::Just(static_cast<INTP>(1)),
                       fuzztest::Just(static_cast<INTP>(2)),
                       fuzztest::Just(static_cast<INTP>(1)),
                       fuzztest::Just(static_cast<INTP>(1)));
    },
    fuzztest::InRange(2, 28));
}

/**
 * @brief Generates dimension and vector sizes for ND FFT problems.
 *
 * This function returns an array representing the dimension and
 * vector sizes for ND FFT problems.
 * The array format is: {dim_rank, vec_rank, dim[dim_rank - 1].n, place holder,
 * place holder, place holder, place holder, place holder}
 * The dimension sizes are restricted to prevent OOM issues.
 *
 * @param dim_rank Dimension rank. Range: [3, 28]
 * @return Array of dimension and vector sizes with restricted
 * dims.n based on dim_rank.
 */
auto dims_and_vecs_ND()
{
    return fuzztest::FlatMap([](INT32 dim_rank)
    {
        INT32 dim_size_max = (INT32_MAX / pow(2, dim_rank + 1) * 8);
        return fuzztest::ArrayOf(fuzztest::Just(static_cast<INTP>(dim_rank)),
                       fuzztest::Just(static_cast<INTP>(1)),
                       fuzztest::InRange(static_cast<INTP>(2),
                               static_cast<INTP>(dim_size_max)),
                       fuzztest::Just(static_cast<INTP>(1)),
                       fuzztest::Just(static_cast<INTP>(1)),
                       fuzztest::Just(static_cast<INTP>(2)),
                       fuzztest::Just(static_cast<INTP>(1)),
                       fuzztest::Just(static_cast<INTP>(1)));
    },
    fuzztest::InRange(3, 28));
}

/**
 * @brief Generates dimension and vector sizes for
 * multi-batched ND FFT problems.
 * @return At runtime either multibatched 1D FFT or ND FFT array gets generated
 */
auto dims_and_vecs_multi_batched_ND()
{
    return fuzztest::OneOf(dims_and_vecs_ND(), dims_and_vecs_multi_batched_1D());
}

#endif // FUZZ_UTILS_H
