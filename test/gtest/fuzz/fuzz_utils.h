// Copyright Advanced Micro Devices, Inc.
// SPDX-License-Identifier: BSD-3-Clause

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
#include <sys/sysinfo.h>
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
// Size limit to prevent OOM SIGKILL for fuzz tests
#define MAX_BUFFER_SIZE 1000000.0
#define MAX_DIM1_SIZE 100000
#define MAX_DIM_STRIDE 10000
#define COMPLEX_DATA_STRIDE 2
#define REAL_DATA_STRIDE 1

double get_max_buffer_size()
{
    struct sysinfo si;
    if (sysinfo(&si) != 0) {
        return MAX_BUFFER_SIZE; // fallback to default limit
    }

    // Get available memory in bytes
    double avail_memory_bytes = (double)si.freeram * (double)si.mem_unit;
    // Calculate available memory divided by 64
    double memory_based_limit = avail_memory_bytes / 64.0;
    // Return minimum of (available_memory/64, MAX_BUFFER_SIZE)
    return (memory_based_limit < MAX_BUFFER_SIZE) ? memory_based_limit
                                                  : MAX_BUFFER_SIZE;
}
/**
 * @brief Initialize benchmark parameters.
 *
 * @tparam dt_t Data type of input and output (
     supported: FFTZ_FLOAT and FFTZ_DOUBLE)
 * @tparam dm_t Data model for data length and strides
 * supported: LP64[FFTZ_INT32] and ILP64[FFTZ_INTP])
 * @param params Pointer to benchmark parameters structure
 * @param dim_rank Dimension rank
 * @param vec_rank Vector rank
 * @param dims Pointer to dimensions structure
 * @param vecs Pointer to vectors structure
 */
template <typename dt_t, typename dm_t>
void init_bench_params(aoclfftz_bench_params_t* params, FFTZ_INT32 dim_rank,
                       FFTZ_INT32 vec_rank, aoclfftz_dim_t_64_* dims,
                       aoclfftz_dim_t_64_* vecs)
{
    if (params)
    {
        params->dim_rank = dim_rank;
        params->vec_rank = vec_rank;
        params->dims = dims;
        params->vecs = vecs;
        if (typeid(dt_t) == typeid(FFTZ_FLOAT))
        {
            params->precision = FLOAT_P;
            params->sz_info.dt_bytes = sizeof(FFTZ_FLOAT);
            params->tolerance = 1E-3;
        }
        else
        {
            params->precision = DOUBLE_P;
            params->sz_info.dt_bytes = sizeof(FFTZ_DOUBLE);
            params->tolerance = 1E-10;
        }
        if (typeid(dm_t) == typeid(FFTZ_INT32))
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
        params->fft_type = C2C;
        params->num_iterations = 1;
        params->num_threads = 1;
        params->dynamic_load_model = 0;
        params->opt_level = 3;
        params->logger_mode = 0;
        params->selector_time = 0;
        params->measure_stats = 0;
        params->bit_reproducibility = 0;
        params->aligned_alloc = ALIGNED_ALLOC;
        params->seed = 0;
        params->use_random_seed = 1;
        // setting the data strides based on complex type and will be modified
        // later for real problems
        params->sz_info.in_data_stride = COMPLEX_DATA_STRIDE;
        params->sz_info.out_data_stride = COMPLEX_DATA_STRIDE;
        // create input and output buffers
        params->in = NULL;
        params->out = NULL;
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
FFTZ_VOID construct_dims_and_vecs(const std::array<FFTZ_INTP,8>&dims_vecs,
                             aoclfftz_dim_t_64_ **dims,
                             aoclfftz_dim_t_64_ **vecs)
{
    FFTZ_INT32 dim_rank = dims_vecs[0];
    FFTZ_INT32 vec_rank = dims_vecs[1];
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
    FFTZ_INT32 i;
    //special cases
    if (vec_rank > 1)
    {
        (*vecs)[0].n = 2;
        (*vecs)[0].in_stride = 2;
        (*vecs)[0].out_stride = 2;

        for (i = 1; i < vec_rank; i++)
        {
            (*vecs)[i].n = 2;
            (*vecs)[i].in_stride = (*vecs)[i - 1].n * (*vecs)[i - 1].in_stride;
            (*vecs)[i].out_stride = (*vecs)[i - 1].n *
                                    (*vecs)[i - 1].out_stride;
        }
        //Overwrite it with the fuzzed value for the last vector size
        (*vecs)[i - 1].n = dims_vecs[2];
        (*dims)[0].n = 2; // minimum value to prevent OOM
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
 * @param dim_size Dimension size. Range: [1, MAX_DIM1_SIZE]
 * @return Array of dimension strides with restricted dim.in_stride and
 * dim.out_stride from 1 to get_max_buffer_size() / dim_size.
 */
auto restricted_dim_stride()
{
    return fuzztest::FlatMap(
        [](FFTZ_INT32 dim_size)
        {
            return fuzztest::ArrayOf(
                fuzztest::Just(static_cast<FFTZ_INTP>(1)),
                fuzztest::Just(static_cast<FFTZ_INTP>(1)),
                fuzztest::Just(static_cast<FFTZ_INTP>(dim_size)),
                fuzztest::InRange(
                    static_cast<FFTZ_INTP>(1),
                    static_cast<FFTZ_INTP>(get_max_buffer_size() / dim_size)),
                fuzztest::InRange(
                    static_cast<FFTZ_INTP>(1),
                    static_cast<FFTZ_INTP>(get_max_buffer_size() / dim_size)),
                fuzztest::Just(static_cast<FFTZ_INTP>(1)),
                fuzztest::Just(static_cast<FFTZ_INTP>(1)),
                fuzztest::Just(static_cast<FFTZ_INTP>(1)));
        },
        // Max. Range for dim size is scaled down to prevent OOM
        fuzztest::InRange(1, MAX_DIM1_SIZE));
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
 * @param dim_stride Dimension stride. Range: [1, MAX_DIM_STRIDE]
 * @return Array of dimension sizes with restricted dims.n from
 * 1 to get_max_buffer_size() / dim_stride.
 */
auto restricted_dim_size()
{
    return fuzztest::FlatMap([](FFTZ_INT32 dim_stride)
    {
        FFTZ_INTP dim_max_size = get_max_buffer_size() / dim_stride;
        return fuzztest::ArrayOf(fuzztest::Just(static_cast<FFTZ_INTP>(1)),
                       fuzztest::Just(static_cast<FFTZ_INTP>(1)),
                       fuzztest::InRange(static_cast<FFTZ_INTP>(1),
                               static_cast<FFTZ_INTP>(dim_max_size)),
                       fuzztest::InRange(static_cast<FFTZ_INTP>(1),
                               static_cast<FFTZ_INTP>(dim_stride)),
                       fuzztest::Just(static_cast<FFTZ_INTP>(dim_stride)),
                       fuzztest::Just(static_cast<FFTZ_INTP>(1)),
                       fuzztest::Just(static_cast<FFTZ_INTP>(1)),
                       fuzztest::Just(static_cast<FFTZ_INTP>(1)));
    },
    fuzztest::InRange(1, MAX_DIM_STRIDE));
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
    return fuzztest::FlatMap(
        [](FFTZ_INT32 dim_size, FFTZ_INT32 dim_instride,
           FFTZ_INT32 dim_outstride)
        {
            FFTZ_INT32 vector_instride = dim_size * dim_instride;
            FFTZ_INT32 vector_outstride = dim_size * dim_outstride;
            FFTZ_INT32 vector_size_max = std::min(
                ((get_max_buffer_size() - vector_instride) / vector_instride),
                ((get_max_buffer_size() - vector_outstride) /
                 vector_outstride));
            return fuzztest::ArrayOf(
                fuzztest::Just(static_cast<FFTZ_INTP>(1)),
                fuzztest::Just(static_cast<FFTZ_INTP>(1)),
                fuzztest::Just(static_cast<FFTZ_INTP>(dim_size)),
                fuzztest::Just(static_cast<FFTZ_INTP>(dim_instride)),
                fuzztest::Just(static_cast<FFTZ_INTP>(dim_outstride)),
                fuzztest::InRange(static_cast<FFTZ_INTP>(2),
                                  static_cast<FFTZ_INTP>(vector_size_max)),
                fuzztest::Just(static_cast<FFTZ_INTP>(vector_instride)),
                fuzztest::Just(static_cast<FFTZ_INTP>(vector_outstride)));
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
    return fuzztest::FlatMap(
        [](FFTZ_INT32 dim_size, FFTZ_INT32 dim_instride,
           FFTZ_INT32 dim_outstride, FFTZ_INT32 vec_size)
        {
            FFTZ_INT32 vector_instride_max =
                (get_max_buffer_size() - (dim_size * dim_instride)) / vec_size *
                100;
            FFTZ_INT32 vector_outstride_max =
                (get_max_buffer_size() - (dim_size * dim_outstride)) /
                vec_size * 100;
            return fuzztest::ArrayOf(
                fuzztest::Just(static_cast<FFTZ_INTP>(1)),
                fuzztest::Just(static_cast<FFTZ_INTP>(1)),
                fuzztest::Just(static_cast<FFTZ_INTP>(dim_size)),
                fuzztest::Just(static_cast<FFTZ_INTP>(dim_instride)),
                fuzztest::Just(static_cast<FFTZ_INTP>(dim_outstride)),
                fuzztest::Just(static_cast<FFTZ_INTP>(vec_size)),
                fuzztest::InRange(
                    static_cast<FFTZ_INTP>(dim_size * dim_instride),
                    static_cast<FFTZ_INTP>(vector_instride_max)),
                fuzztest::InRange(
                    static_cast<FFTZ_INTP>(dim_size * dim_outstride),
                    static_cast<FFTZ_INTP>(vector_outstride_max)));
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
    return fuzztest::FlatMap(
        [](FFTZ_INT32 dim1_size, FFTZ_INT32 dim1_instride,
           FFTZ_INT32 dim1_outstride)
        {
            FFTZ_INT32 dim2_instride = dim1_size * dim1_instride;
            FFTZ_INT32 dim2_outstride = dim1_size * dim1_outstride;
            FFTZ_INT32 dim2_size_max = std::min(
                (get_max_buffer_size() / (dim2_instride * dim2_instride)),
                (get_max_buffer_size() / (dim2_outstride * dim2_outstride)));
            return fuzztest::ArrayOf(
                fuzztest::Just(static_cast<FFTZ_INTP>(2)),
                fuzztest::Just(static_cast<FFTZ_INTP>(1)),
                fuzztest::Just(static_cast<FFTZ_INTP>(dim1_size)),
                fuzztest::Just(static_cast<FFTZ_INTP>(dim1_instride)),
                fuzztest::Just(static_cast<FFTZ_INTP>(dim1_outstride)),
                fuzztest::InRange(static_cast<FFTZ_INTP>(2),
                                  static_cast<FFTZ_INTP>(dim2_size_max)),
                fuzztest::Just(static_cast<FFTZ_INTP>(dim2_instride)),
                fuzztest::Just(static_cast<FFTZ_INTP>(dim2_outstride)));
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
 * @param dim1_size for the first dimension. Range: [1, get_max_buffer_size() /
 * 110]
 * @param dim1_instride for the first dimension. Range: [1, 10]
 * @param dim1_outstride for the first dimension. Range: [1, 10]
 * @return Array of dimension sizes for the second dimension with
 * restricted dims.n for the first dimension.
 */
auto restricted_dim2_size()
{
    return fuzztest::FlatMap(
        [](FFTZ_INT32 dim1_size, FFTZ_INT32 dim1_instride,
           FFTZ_INT32 dim1_outstride)
        {
            FFTZ_INT32 dim2_instride = dim1_size * dim1_instride;
            FFTZ_INT32 dim2_outstride = dim1_size * dim1_outstride;
            return fuzztest::ArrayOf(
                fuzztest::Just(static_cast<FFTZ_INTP>(2)),
                fuzztest::Just(static_cast<FFTZ_INTP>(1)),
                fuzztest::Just(static_cast<FFTZ_INTP>(dim1_size)),
                fuzztest::Just(static_cast<FFTZ_INTP>(dim1_instride)),
                fuzztest::Just(static_cast<FFTZ_INTP>(dim1_outstride)),
                fuzztest::InRange(static_cast<FFTZ_INTP>(1),
                                  static_cast<FFTZ_INTP>(10)),
                fuzztest::Just(static_cast<FFTZ_INTP>(dim2_instride)),
                fuzztest::Just(static_cast<FFTZ_INTP>(dim2_outstride)));
        },
        fuzztest::InRange(1, static_cast<int>(get_max_buffer_size() / 110)),
        fuzztest::InRange(1, 10), fuzztest::InRange(1, 10));
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
 * @param vec_rank Vector rank. Range: [2, 15]
 * @return Array of dimension and vector sizes with restricted
 * vecs.n based on vec_rank.
 */
auto dims_and_vecs_multi_batched_1D()
{
    return fuzztest::FlatMap([](FFTZ_INT32 vec_rank)
    {
        FFTZ_INT32 vec_size_max =
            (get_max_buffer_size() / pow(2, vec_rank + 1)) - 2;
        return fuzztest::ArrayOf(fuzztest::Just(static_cast<FFTZ_INTP>(1)),
                       fuzztest::Just(static_cast<FFTZ_INTP>(vec_rank)),
                       fuzztest::InRange(static_cast<FFTZ_INTP>(1),
                               static_cast<FFTZ_INTP>(vec_size_max)),
                       fuzztest::Just(static_cast<FFTZ_INTP>(1)),
                       fuzztest::Just(static_cast<FFTZ_INTP>(1)),
                       fuzztest::Just(static_cast<FFTZ_INTP>(2)),
                       fuzztest::Just(static_cast<FFTZ_INTP>(1)),
                       fuzztest::Just(static_cast<FFTZ_INTP>(1)));
    },
    fuzztest::InRange(2, 15));
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
 * @param dim_rank Dimension rank. Range: [3, 15]
 * @return Array of dimension and vector sizes with restricted
 * dims.n based on dim_rank.
 */
auto dims_and_vecs_ND()
{
    return fuzztest::FlatMap(
        [](FFTZ_INT32 dim_rank)
        {
            FFTZ_INT32 dim_size_max =
                (get_max_buffer_size() / pow(2, dim_rank + 1));
            return fuzztest::ArrayOf(
                fuzztest::Just(static_cast<FFTZ_INTP>(dim_rank)),
                fuzztest::Just(static_cast<FFTZ_INTP>(1)),
                fuzztest::InRange(static_cast<FFTZ_INTP>(2),
                                  static_cast<FFTZ_INTP>(dim_size_max)),
                fuzztest::Just(static_cast<FFTZ_INTP>(1)),
                fuzztest::Just(static_cast<FFTZ_INTP>(1)),
                fuzztest::Just(static_cast<FFTZ_INTP>(2)),
                fuzztest::Just(static_cast<FFTZ_INTP>(1)),
                fuzztest::Just(static_cast<FFTZ_INTP>(1)));
        },
        fuzztest::InRange(3, 15));
}

/**
 * @brief Generates dimension and vector sizes for
 * multi-batched ND FFT problems.
 * @return At runtime either multibatched 1D FFT or ND FFT array gets generated
 */
auto dims_and_vecs_multi_batched_ND()
{
    return fuzztest::OneOf(dims_and_vecs_ND(),
                           dims_and_vecs_multi_batched_1D());
}

#endif // FUZZ_UTILS_H
