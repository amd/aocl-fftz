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

/** @file fuzz_test.h
 *
 *  @brief Base file for fuzz tests.
 *
 *  This file contains the classes and functions used
 *  for running fuzz tests using Google Fuzz test.
 *
 *  @author Jeya R
 *  @author Maheswar Rao S
 */

#ifndef FUZZ_TEST_H
#define FUZZ_TEST_H

#include "test/gtest/fuzz/fuzz_utils.h"

/**
 * @brief Base class for the AOCLFFTZ Fuzz Tests
 *
 * @tparam dt_t data-type of input and output (supported: FLOAT and DOUBLE)
 * @tparam dm_t data-model for data length and strides (supported: LP64[INT32]
 * and ILP64[INTP])
 * @tparam dim_t type of aoclfftz_dim struct
 * @tparam prob_desc_t type of problem descriptor struct
 */
template <class dt_t, class dm_t, class dim_t, class prob_desc_t>
class AoclfftzFuzzTestBase
{
 public:
    VOID *(*aoclfftz_setup)(prob_desc_t *problem);
    AoclfftzFuzzTestBase()
    {
        aoclfftz_setup = nullptr;
        if (std::is_same<dt_t, FLOAT>::value)
        {
            if (std::is_same<dm_t, INT32>::value)
            {
                aoclfftz_setup = reinterpret_cast<VOID *(*)(prob_desc_t *)>
                                                        (aoclfftz_setup_f);
            }
            else
            {
                aoclfftz_setup = reinterpret_cast<VOID *(*)(prob_desc_t *)>
                                                        (aoclfftz_setup_f_64_);
            }
        }
        else if (std::is_same<dt_t, DOUBLE>::value)
        {
            if (std::is_same<dm_t, INT32>::value)
            {
                aoclfftz_setup = reinterpret_cast<VOID *(*)(prob_desc_t *)>
                                                        (aoclfftz_setup_d);
            }
            else
            {
                aoclfftz_setup = reinterpret_cast<VOID *(*)(prob_desc_t *)>
                                                        (aoclfftz_setup_d_64_);
            }
        }
    }
/**
* @brief Fuzz the input buffer of the problem descriptor and verify the

* correctness of the FFT output using signal properties
*
* @param problem_size string holding the input problem

*/
VOID fuzz_input_buffer_test(const std::string& problem_size)
{
    std::string prob = problem_size;
    INT32 status = PARSER_SUCCESS;
    INT32 dim_rank = 0;
    INT32 vec_rank = 0;
    aoclfftz_bench_params_t *params = NULL;
    ALLOC_ALIGN_UNINIT(params, aoclfftz_bench_params_t,
                         sizeof(aoclfftz_bench_params_t));
    status = find_dim_vec_ranks((CHAR *)prob.c_str(), &dim_rank, &vec_rank);
    if (status != PARSER_SUCCESS)
    {
        AOCLFFTZ_ERROR("Failed on finding dim_rank and vec_rank\n");
        return;
    }

    aoclfftz_dim_t_64_ *dims = NULL;
    aoclfftz_dim_t_64_ *vecs = NULL;
    status = allocate_and_fill_dims_vecs(
        (CHAR *)prob.c_str(), dim_rank, vec_rank, &dims, &vecs, 1);
    if (status != PARSER_SUCCESS)
    {
        FREE_ALIGN_ALLOCATED_MEM(dims);
        FREE_ALIGN_ALLOCATED_MEM(vecs);
        return ;
    }

    init_bench_params<dt_t, dm_t>(params, dim_rank, vec_rank, dims, vecs);
    set_default_dims_vecs(dim_rank, vec_rank, dims, vecs,
                    params->fft_type,
                    params->res_placement == IN_PLACE,
                    params->logger_mode);
    UINTP input_size = 0;
    UINTP output_size = 0;

    params->sz_info.n = calculate_size(dims, dim_rank);
    params->sz_info.batches = calculate_size(vecs, vec_rank);
    INTP size = params->sz_info.n * params->sz_info.batches ;
    calculate_buffer_sizes(dim_rank, vec_rank, dims,
                            vecs, &input_size, &output_size);

    params->sz_info.input_size = input_size;
    params->sz_info.output_size = output_size;
    params->sz_info.input_bytes = input_size *
                                        params->sz_info.in_data_stride *
                                        params->sz_info.dt_bytes;
    params->sz_info.output_bytes = output_size *
                                         params->sz_info.out_data_stride *
                                         params->sz_info.dt_bytes;
    params->sz_info.n_in = params->sz_info.n;
    params->sz_info.n_out = params->sz_info.n;
    INTP *in_idx_map = NULL;
    ALLOC_ALIGN_UNINIT(in_idx_map, INTP, size * sizeof(INTP));
    INTP *out_idx_map = NULL;
    ALLOC_ALIGN_UNINIT(out_idx_map, INTP, size * sizeof(INTP));
    // Preparing the index map
    prepare_index_map(dim_rank, vec_rank, dims,
                    vecs, in_idx_map, out_idx_map, params->fft_type, ALIGNED_ALLOC);
    // Allocate in/out buffers
    ALLOC_UNINIT(params->in, VOID, params->sz_info.input_bytes,
                 params->aligned_alloc);
    ALLOC_INIT(params->out, VOID, params->sz_info.output_bytes,
               params->aligned_alloc);

    register_functions(params);
    std::vector<dt_t> inBuf(size * params->sz_info.in_data_stride);
    absl::BitGen prng;
    if (params->precision == FLOAT_P)
    {
        for (int i = 0; i < size * params->sz_info.in_data_stride; ++i)
        {
            inBuf[i] = fuzztest::InRange(-10000.0f,
                                         10000.0f).GetRandomValue(prng);
        }
    }
    else
    {
        for (int i = 0; i < size * params->sz_info.in_data_stride; ++i)
        {
            inBuf[i] = fuzztest::InRange(-10000.0,
                                         10000.0).GetRandomValue(prng);
        }
    }
    dt_t *input = NULL;
    ALLOC_ALIGN_UNINIT(input, dt_t, sizeof(dt_t) * DATA_STRIDE * input_size);
    for (INTP idx = 0; idx < size; idx = idx + 1)
    {
        (input)[in_idx_map[idx] * DATA_STRIDE] = inBuf[idx * DATA_STRIDE];
        (input)[in_idx_map[idx] * DATA_STRIDE + 1] =
                                            inBuf[idx * DATA_STRIDE + 1];
    }

    // Setup and run tests
    VOID *handle = params->setup_problem(params);
    EXPECT_FALSE(handle == NULL);
    // Verifying the correctness of FFT output against signal properties
    INT32 result = run_linearity_test(params, in_idx_map, out_idx_map, handle,
                                         (VOID *)input);
    EXPECT_EQ(result, BENCH_SUCCESS);
    result = run_impulse_transform_test(params, in_idx_map, out_idx_map, handle,
                                        (VOID *)input);
    EXPECT_EQ(result, BENCH_SUCCESS);
    result = run_timeshift_test(params, in_idx_map, out_idx_map, handle,
                                 (VOID *)input);
    EXPECT_EQ(result, BENCH_SUCCESS);

    aoclfftz_destroy(handle);
    FREE_ALIGN_ALLOCATED_MEM(in_idx_map);
    FREE_ALIGN_ALLOCATED_MEM(out_idx_map);
    FREE_ALLOCATED_MEM(params->in, params->aligned_alloc);
    FREE_ALLOCATED_MEM(params->out, params->aligned_alloc);
    FREE_ALIGN_ALLOCATED_MEM(input);
    FREE_ALIGN_ALLOCATED_MEM(params);
    FREE_ALIGN_ALLOCATED_MEM(dims);
    FREE_ALIGN_ALLOCATED_MEM(vecs);
}

/**
* @brief fuzz problem descriptor passed as an argument to the setup API of the
* library which are provided as function argument and check the typical flow
* of the library setup->execute->destroy
*
* @param dims_and_vecs array which has values for the dims and vecs of the
* FFT problem
* @param flags structure member of problem descriptor
* @param pthr_fft structure member of problem descriptor
* @param cntrl_params structure member of problem descriptor
*/
VOID fuzz_problem_desc_test(const std::array<INTP, 8>& dims_and_vecs,
                            UINT32 flags, aoclfftz_smp_pfft_t pthr_fft,
                            aoclfftz_cntrl_params_t cntrl_params)
{
    // Allocate memory for benchmark parameters
    aoclfftz_bench_params_t *params = NULL;
    ALLOC_ALIGN_UNINIT(params, aoclfftz_bench_params_t,
                         sizeof(aoclfftz_bench_params_t));
    if (params == NULL)
    {
        printf("Failed to allocate memory for params\n");
        return;
    }
    // Extract dimension and vector ranks
    INT32 dim_rank = dims_and_vecs[0];
    INT32 vec_rank = dims_and_vecs[1];
    aoclfftz_dim_t_64_ *dims = NULL;
    aoclfftz_dim_t_64_ *vecs = NULL;
    construct_dims_and_vecs(dims_and_vecs, &dims, &vecs);

    // Initialize benchmark parameters
    init_bench_params<dt_t, dm_t>(params, dim_rank, vec_rank, dims, vecs);
    if (IS_REAL(flags) && params->dir == FORWARD)
    {
        params->fft_type = R2C;
        params->sz_info.in_data_stride = REAL_DATA_STRIDE;
        params->sz_info.out_data_stride = COMPLEX_DATA_STRIDE;
    }
    else if (IS_REAL(flags) && params->dir == BACKWARD)
    {
        params->fft_type = C2R;
        params->sz_info.in_data_stride = COMPLEX_DATA_STRIDE;
        params->sz_info.out_data_stride = REAL_DATA_STRIDE;
    }

    set_default_dims_vecs(dim_rank, vec_rank, dims, vecs,
                    params->fft_type,
                    params->res_placement == IN_PLACE,
                    params->logger_mode);
    UINTP input_size = 0;
    UINTP output_size = 0;
    params->sz_info.n = calculate_size(dims, dim_rank);
    params->sz_info.batches = calculate_size(vecs, vec_rank);
    INTP size = params->sz_info.n * params->sz_info.batches;
    calculate_buffer_sizes(dim_rank, vec_rank, dims,
                            vecs, &input_size, &output_size);
    params->sz_info.input_size = input_size;
    params->sz_info.output_size = output_size;
    params->sz_info.input_bytes = input_size *
                                        params->sz_info.in_data_stride *
                                        params->sz_info.dt_bytes;
    params->sz_info.output_bytes = output_size *
                                         params->sz_info.out_data_stride *
                                         params->sz_info.dt_bytes;
    params->sz_info.n_in = params->sz_info.n;
    params->sz_info.n_out = params->sz_info.n;
    INTP n0 = params->dims[0].n;
    INTP n0_hc = n0 / 2 + 1; // half-complex size for R2C/C2R

    // Adjust sizes for half-complex transforms:
    // R2C: N real → (N/2+1) complex (Hermitian symmetry reduces storage)
    // C2R: (N/2+1) complex → N real
    if (params->fft_type == R2C)
    {
        params->sz_info.n_out = (params->sz_info.n * n0_hc) / n0;
    }
    else if (params->fft_type == C2R)
    {
        params->sz_info.n_in = (params->sz_info.n * n0_hc) / n0;
    }

    INTP *in_idx_map = NULL;
    ALLOC_ALIGN_UNINIT(in_idx_map, INTP, size * sizeof(INTP));
    INTP *out_idx_map = NULL;
    ALLOC_ALIGN_UNINIT(out_idx_map, INTP, size * sizeof(INTP));

    if (in_idx_map == NULL || out_idx_map == NULL)
    {
        printf("Failed to allocate memory for in_idx_map/out_idx_map\n");
        return;
    }

    // Prepare index maps
    prepare_index_map(dim_rank, vec_rank, dims,
                      vecs, in_idx_map, out_idx_map, params->fft_type,
                      ALIGNED_ALLOC);

    register_functions(params);

    // create input and output buffers
    UINT32 is_align = params->aligned_alloc;
    INTP input_bytes = params->sz_info.input_bytes;
    INTP output_bytes = params->sz_info.output_bytes;
    if (params->fft_type != C2C &&
        params->res_placement == IN_PLACE)
    {
        input_bytes = MAX(input_bytes, output_bytes);
    }
    ALLOC_UNINIT(params->in, VOID, input_bytes, is_align);
    if (params->in == NULL)
    {
        printf("Failed to allocate memory for input buffer\n");
        return;
    }

    // use input buffer as output for in-place problems or create new output
    // buffer for out-of-place problems
    if (params->res_placement == IN_PLACE)
    {
        params->out = params->in;
    }
    else
    {
        ALLOC_INIT(params->out, VOID, output_bytes, is_align);
        if (params->out == NULL)
        {
            printf("Failed to allocate memory for output buffer\n");
            return;
        }
    }

    // Generate input buffer
    std::vector<dt_t> inBuf(size * params->sz_info.in_data_stride);
    absl::BitGen prng;
    if (std::is_same<dt_t, FLOAT>::value)
    {
        for (int i = 0; i < size * params->sz_info.in_data_stride; ++i)
        {
            inBuf[i] = fuzztest::InRange(-10.0f, 10.0f).GetRandomValue(prng);
        }
    }
    else
    {
        for (int i = 0; i < size * params->sz_info.in_data_stride; ++i)
        {
            inBuf[i] = fuzztest::InRange(-10.0, 10.0).GetRandomValue(prng);
        }
    }
    dt_t *input = NULL;
    INT32 in_data_stride = params->sz_info.in_data_stride;
    INT32 out_data_stride = params->sz_info.out_data_stride;
    ALLOC_ALIGN_UNINIT(input, dt_t, params->sz_info.input_bytes);
    if (input == NULL)
    {
        printf("Failed to allocate memory for input buffer\n");
        return;
    }
    INTP n = params->sz_info.n_in * params->sz_info.batches;
    for (INTP idx = 0; idx < n * in_data_stride; ++idx)
    {
        (input)[in_idx_map[idx / in_data_stride] * in_data_stride +
                        (idx % in_data_stride)] = inBuf[idx];
    }

    // Set additional parameters from fuzzed input
    params->order = IS_OUT_OF_ORDER(flags) ? OUT_OF_ORDER : IN_ORDER;
    params->num_threads = pthr_fft.num_threads;
    params->dynamic_load_model = pthr_fft.dynamic_load_model;
    params->logger_mode = cntrl_params.logger_mode;
    params->measure_stats = cntrl_params.measure_stats;

    // Setup problem and run tests
    VOID *handle = params->setup_problem(params);
    if (handle != NULL)
    {
        INT32 result = run_impulse_transform_test(
            params, in_idx_map, out_idx_map, handle, (VOID *)input);
        EXPECT_EQ(result, BENCH_SUCCESS);
    }
    else
    {
        AOCLFFTZ_ERROR("Setup failed resulting handle as NULL\n");
    }

    aoclfftz_destroy(handle);
    FREE_ALIGN_ALLOCATED_MEM(in_idx_map);
    FREE_ALIGN_ALLOCATED_MEM(out_idx_map);
    FREE_ALLOCATED_MEM(params->in, params->aligned_alloc);
    FREE_ALLOCATED_MEM(params->out, params->aligned_alloc);
    FREE_ALIGN_ALLOCATED_MEM(input);
    FREE_ALIGN_ALLOCATED_MEM(params);
    FREE_ALIGN_ALLOCATED_MEM(dims);
    FREE_ALIGN_ALLOCATED_MEM(vecs);
}
};

auto problemsize = fuzztest::ElementOf<std::string>(
                    {{"2:1:1"},             // Direct
                     {"18:2:3"},            // CT
                     {"10:100:200v50:2:2"}, // Batched CT
                     {"78x124x24x3"},       // 4D holding composite size (CT)
                     {"3x4x5:3:4"},         // Basic 3D case
                     {"3x4:25:30x2:12:11v2x5"},// 3D Strided Batched ND Problem
                     {"2x2v20"},     // 2D Batched 1D Composite
                     {"2x4:2:4x2"},  // InStride != OutStride
                     {"19:1:1"}});   // Bluestein

class AoclfftzFuzzTestFloatLP64 : public AoclfftzFuzzTestBase<FLOAT, INT32,
                             aoclfftz_dim_t, aoclfftz_prob_desc_f>{};

class AoclfftzFuzzTestFloatILP64 : public AoclfftzFuzzTestBase<FLOAT, INTP,
                             aoclfftz_dim_t_64_, aoclfftz_prob_desc_f_64_>{};

class AoclfftzFuzzTestDoubleLP64 : public AoclfftzFuzzTestBase<DOUBLE, INT32,
                             aoclfftz_dim_t, aoclfftz_prob_desc_d>{};

class AoclfftzFuzzTestDoubleILP64 : public AoclfftzFuzzTestBase<DOUBLE, INTP,
                             aoclfftz_dim_t_64_, aoclfftz_prob_desc_d_64_>{};

#endif // FUZZ_TEST_H
