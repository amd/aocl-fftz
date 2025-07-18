/**
 * Copyright (C) 2023, Advanced Micro Devices. All rights reserved.
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

/** @file selector_gtest.h
 *
 *  @brief Base file for GTest selector tests.
 *
 *  This file contains the classes and functions used
 *  for running selector unit tests using GTest.
 *
 *  @author Srirammaswamy Srinivasan
 */

#ifndef AOCLFFTZ_SELECTOR_GTEST_BASE_H
#define AOCLFFTZ_SELECTOR_GTEST_BASE_H

#include <gtest/gtest.h>
#include <math.h>
#include "api/aoclfftz.h"
extern "C"
{
#include "test/gtest/aoclfftz_core_wrapper.h"
#include "test/utils/dims_vecs_helper.h"
}
#include "test/gtest/gtest_types.h"

/****** Utility functions for Bluestein solver ******/
INTP get_extended_length_ref(INTP n)
{
    INTP m = 2 * n - 1;
    // check all the factors of m is within the supported kernels
    // i.e. prime numbers in range 2 to 16
    // if not, adjust the m to a nearest larger number
    // which satisfies the above condition
    INTP supported_primes[] = {2, 3, 5, 7, 11, 13};
    UINT32 prime_count = sizeof(supported_primes) / sizeof(supported_primes[0]);
    for (INTP next_m = m, quo = 0; quo != 1; next_m++)
    {
        quo = m = next_m;
        for (UINT32 i = 0; i < prime_count; i++)
        {
            while (quo % supported_primes[i] == 0)
            {
                quo /= supported_primes[i];
            }
            if (quo == 1)
            {
                break; // solvable m value
            }
        }
    }
    return m;
}

VOID prepare_bluestein_sequence_ref(VOID *B, INTP m, INTP n, UINT32 precision)
{
    /*            Bluestein sequence B of length m
        <------ (n) -----><-- (m-2n-1) --><----- (n-1) ----->
        |     values      |     zeros     | reversed values |

                                                                */
    INTP n2 = n << 1;
    if (precision == DT_FLOAT)
    {
        FLOAT *B_f = (FLOAT *)B;
        // fill the sequence values
        for (INTP i = 0; i < n; i++)
        {
            INTP m = (i * i) % n2;
            FLOAT angle = (AOCLFFTZ_2_PIf * m) / n2;
            B_f[i * DATA_STRIDE] = cos(angle);
            B_f[i * DATA_STRIDE + 1] = sin(angle);
        }
        // zero padding
        memset(B_f + n * DATA_STRIDE, 0,
               (m - n - 1) * DATA_STRIDE * sizeof(FLOAT));
        // reverse the first n values (from 2 to n)
        for (INTP i = 1; i < n; i++)
        {
            B_f[(m - i) * DATA_STRIDE] = B_f[i * DATA_STRIDE];
            B_f[(m - i) * DATA_STRIDE + 1] = B_f[i * DATA_STRIDE + 1];
        }
    }
    else
    {
        DOUBLE *B_d = (DOUBLE *)B;
        // fill the sequence values
        for (INTP i = 0; i < n; i++)
        {
            INTP m = (i * i) % n2;
            DOUBLE angle = (AOCLFFTZ_2_PI * m) / n2;
            B_d[i * DATA_STRIDE] = cos(angle);
            B_d[i * DATA_STRIDE + 1] = sin(angle);
        }
        // zero padding
        memset(B_d + n * DATA_STRIDE, 0,
               (m - n - 1) * DATA_STRIDE * sizeof(DOUBLE));
        // reverse the first n values (from 2 to n)
        for (INTP i = 1; i < n; i++)
        {
            B_d[(m - i) * DATA_STRIDE] = B_d[i * DATA_STRIDE];
            B_d[(m - i) * DATA_STRIDE + 1] = B_d[i * DATA_STRIDE + 1];
        }
    }
}

template <class prob_desc_t>
VOID calculate_input_output_sizes(prob_desc_t *p_desc, INTP *in_buffer_size,
                                  INTP *out_buffer_size)
{
    // Data arrangement considered :
    // [1, 2, 3, 4]<0, 0>[5, 6, 7, 8]<0, 0>[9, 10, 11, 12]
    // <---vec stride--->
    // <-------------(Batches -1)---------><--- Problem size * dim stride --->
    // ((Batches -1) * (vec_stride)) + (Problem size * dim stride)

    INT32 dim_rank = p_desc->dim_rank;
    INT32 vec_rank = p_desc->vec_rank;
    in_buffer_size[0] = 0;
    out_buffer_size[0] = 0;
    INT32 in_size = 1; // rank-0 problem where its a constant ?
    INT32 out_size = 1;

    for (INT32 i = 0; i < dim_rank; i++)
    {
        in_size += ((p_desc->dims[i].n - 1) * (p_desc->dims[i].in_stride));
        out_size += ((p_desc->dims[i].n - 1) * (p_desc->dims[i].out_stride));
    }

    for (INT32 i = 0; i < vec_rank; i++)
    {
        in_size += ((p_desc->vecs[i].n - 1) * (p_desc->vecs[i].in_stride));
        out_size += ((p_desc->vecs[i].n - 1) * (p_desc->vecs[i].out_stride));
    }

    in_buffer_size[0] = in_size;
    out_buffer_size[0] = out_size;
}

/**
 * @brief Base class for the AOCLFFTZ Selector GTest
 *
 * @tparam dt_t data-type of input and output (supported: FLOAT and DOUBLE)
 * @tparam dm_t data-model for data length and strides (supported: LP64[INT32]
 * and ILP64[INTP])
 * @tparam dim_t type of aoclfftz_dim struct
 * @tparam prob_desc_t type of problem descriptor struct
 */
template <class dt_t, class dm_t, class dim_t, class prob_desc_t>
class AoclfftzSelectorTestBase
    : public ::testing::TestWithParam<aoclfftz_selector_test_params_t>
{
  protected:
    // function pointer for aoclfftz_setup_* API
    VOID *(*aoclfftz_setup)(prob_desc_t *problem);

    UINT64 random_seed; // random seed value used for random data generation
    // following pointers are created in class level scope to hold the objects
    // created inside member functions
    // class level scope is used for better memory management
    VOID *handle;        // to store the handle object returns from setup API
    prob_desc_t *p_desc; // to store the problem descriptor
    aoclfftz_solution_t *ref_solution; // to store the reference solution object

    AoclfftzSelectorTestBase()
    {
        p_desc = NULL;
        handle = NULL;
        ref_solution = NULL;
    }

    ~AoclfftzSelectorTestBase()
    {
        destroy_handle_wrapper(handle);
        destroy_solution_wrapper(ref_solution);
        // destroy problem descriptor
        if (p_desc != NULL)
        {
            INT32 is_out_place = IS_OUT_OF_PLACE(p_desc->flags);
            FREE_ALIGN_ALLOCATED_MEM(p_desc->in);
            if (is_out_place)
            {
                FREE_ALIGN_ALLOCATED_MEM(p_desc->out);
            }
            FREE_ALIGN_ALLOCATED_MEM(p_desc->dims);
            FREE_ALIGN_ALLOCATED_MEM(p_desc->vecs);
            FREE_ALIGN_ALLOCATED_MEM(p_desc);
        }
    }

    void SetUp() override
    {
        random_seed = std::chrono::duration_cast<std::chrono::microseconds>(
                          std::chrono::system_clock::now().time_since_epoch())
                          .count();
        srand(random_seed);
    }

    /**
     * @brief Entry function to run selector tests and verify the properties of
     * all the solutions in the list
     *
     * @param solver_type solver type of the first solution in reference
     * solution
     */
    void run_selector_test_and_verify_solutions(aoclfftz_solver_type solver)
    {
        std::string dims_and_vecs = std::get<0>(GetParam());
        INT32 flags = std::get<1>(GetParam());
        INT32 opt_level = std::get<2>(GetParam());
        std::vector<aoclfftz_solver_type> solver_list = std::get<3>(GetParam());
        aoclfftz_solution_t *sol =
            run_setup_and_get_solution(dims_and_vecs, flags, opt_level);
        generate_default_ref_solution(&ref_solution, dims_and_vecs, flags,
                                      opt_level, solver);
        ASSERT_NE(sol, nullptr) << "Failed at: run_setup_and_get_solution\n";
        ASSERT_NE(ref_solution, nullptr)
            << "Failed at: generate_reference_solution\n";
        ASSERT_TRUE(verify_solution(sol, ref_solution))
            << "Failed at: verify_solution\n";
    }

    /**
     * @brief Entry function to run selector tests and compare the solver types
     * in the soutions list
     */
    void run_selector_test_and_compare_solver_list()
    {
        std::string dims_and_vecs = std::get<0>(GetParam());
        INT32 flags = std::get<1>(GetParam());
        INT32 opt_level = std::get<2>(GetParam());
        std::vector<aoclfftz_solver_type> solver_list = std::get<3>(GetParam());
        aoclfftz_solution_t *sol =
            run_setup_and_get_solution(dims_and_vecs, flags, opt_level);
        EXPECT_TRUE(compare_solver_list(sol, solver_list));
    }

    /**
     * @brief A wrapper to create a problem descriptor, calls the setup API and
     * get the solution object
     *
     * @param dims_and_vecs problem descriptor string which contains dims and
     * vecs info
     * @param flags in-place:bit-0, real:bit-1, out-of-order:bit-2, dir:bit-3
     * @param opt_level optimization level
     * @return aoclfftz_solution_t*
     */
    aoclfftz_solution_t *run_setup_and_get_solution(std::string dims_and_vecs,
                                                    INT32 flags,
                                                    INT32 opt_level)
    {
        INT32 status = PARSER_SUCCESS;
        INT32 dim_rank = 0;
        INT32 vec_rank = 0;
        INT32 is_in_place = !IS_OUT_OF_PLACE(flags);
        status = find_dim_vec_ranks((CHAR *)dims_and_vecs.c_str(), &dim_rank,
                                    &vec_rank);
        if (status != PARSER_SUCCESS)
        {
            AOCLFFTZ_LOG_UNFORMATTED(ERR, ERR, "invalid dims/vecs rank");
            return NULL;
        }
        aoclfftz_dim_t_64_ *dims = NULL;
        aoclfftz_dim_t_64_ *vecs = NULL;
        status = allocate_and_fill_dims_vecs(
            (CHAR *)dims_and_vecs.c_str(), dim_rank, vec_rank, &dims, &vecs, 1);
        if (status != PARSER_SUCCESS)
        {
            AOCLFFTZ_LOG_UNFORMATTED(ERR, ERR, "dims and vecs parsing failed");
            FREE_ALIGN_ALLOCATED_MEM(dims);
            FREE_ALIGN_ALLOCATED_MEM(vecs);
            return NULL;
        }

        ALLOC_ALIGN_UNINIT(p_desc, prob_desc_t, sizeof(prob_desc_t));
        p_desc->dim_rank = dim_rank;
        p_desc->vec_rank = vec_rank;
        ALLOC_ALIGN_UNINIT(p_desc->dims, dim_t, dim_rank * sizeof(dim_t));
        for (INT32 i = 0; i < dim_rank; i++)
        {
            p_desc->dims[i].n = (dm_t)dims[i].n;
            p_desc->dims[i].in_stride = (dm_t)dims[i].in_stride;
            // Strides must be equal for inplace problems
            if (is_in_place)
            {
                p_desc->dims[i].out_stride = (dm_t)dims[i].in_stride;
            }
            else
            {
                p_desc->dims[i].out_stride = (dm_t)dims[i].out_stride;
            }
        }
        ALLOC_ALIGN_UNINIT(p_desc->vecs, dim_t, vec_rank * sizeof(dim_t));
        for (INT32 i = 0; i < vec_rank; i++)
        {
            p_desc->vecs[i].n = (dm_t)vecs[i].n;
            p_desc->vecs[i].in_stride = (dm_t)vecs[i].in_stride;
            // Strides must be equal for inplace problems
            if (is_in_place)
            {
                p_desc->vecs[i].out_stride = (dm_t)vecs[i].in_stride;
            }
            else
            {
                p_desc->vecs[i].out_stride = (dm_t)vecs[i].out_stride;
            }
        }
        FREE_ALIGN_ALLOCATED_MEM(dims);
        FREE_ALIGN_ALLOCATED_MEM(vecs);

        INTP input_size, output_size;
        calculate_input_output_sizes<prob_desc_t>(p_desc, &input_size,
                                                  &output_size);
        dt_t *in = NULL;
        dt_t *out = NULL;
        ALLOC_ALIGN_UNINIT(in, dt_t, input_size * DATA_STRIDE * sizeof(dt_t));
        // input and output buffers must be same for inplace problems
        if (is_in_place)
        {
            out = in;
        }
        else
        {
            ALLOC_ALIGN_INIT(out, dt_t,
                             output_size * DATA_STRIDE * sizeof(dt_t));
        }
        for (int i = 0; i < input_size * DATA_STRIDE; i++)
            in[i] = (rand() % 1000) / 100.0;

        p_desc->in = in;
        p_desc->out = out;
        // in/out-of place:0-bit, in/out-of order:1-bit, dir:2-bit,
        // real/comp:3-bit..
        p_desc->flags = flags;

        p_desc->cntrl_params.opt_level = opt_level;
        if (opt_level == -1)
        {
            p_desc->cntrl_params.opt_off = 1;
        }
        p_desc->cntrl_params.logger_mode = 0;
        p_desc->cntrl_params.measure_stats = 0;
        p_desc->pthr_fft.num_threads = 1;
        p_desc->pthr_fft.dynamic_load_model = 0;

        handle = aoclfftz_setup(p_desc);
        if (handle == NULL)
        {
            AOCLFFTZ_LOG_UNFORMATTED(ERR, ERR, "aoclfftz_setup failed");
            return NULL;
        }
        aoclfftz_selector_t *sel = (aoclfftz_selector_t *)handle;
        return sel->solution;
    }

    /**
     * @brief Generate the solution object based on given params
     *
     * @param sol aoclfftz_solution_t**
     * @param dims_and_vecs problem descriptor string which contains dims and
     * vecs info
     * @param flags in/out-of place:0-bit, in/out-of order:1-bit, dir:2-bit,
     * real/comp:3-bit
     * @param opt_level optimization level
     * @param solver_type aoclfftz_solver_type
     */
    VOID generate_default_ref_solution(aoclfftz_solution_t **ref_sol,
                                       std::string dims_and_vecs, INT32 flags,
                                       INT32 opt_level,
                                       aoclfftz_solver_type solver_type)
    {
        INT32 status = PARSER_SUCCESS;
        INT32 dim_rank = 0;
        INT32 vec_rank = 0;
        INT32 is_in_place = !IS_OUT_OF_PLACE(flags);
        status = find_dim_vec_ranks((CHAR *)dims_and_vecs.c_str(), &dim_rank,
                                    &vec_rank);
        if (status != PARSER_SUCCESS)
        {
            AOCLFFTZ_LOG_UNFORMATTED(ERR, ERR, "invalid dims/vecs rank");
            return;
        }
        aoclfftz_dim_t_64_ *dims;
        aoclfftz_dim_t_64_ *vecs;
        status = allocate_and_fill_dims_vecs(
            (CHAR *)dims_and_vecs.c_str(), dim_rank, vec_rank, &dims, &vecs, 1);
        if (status != PARSER_SUCCESS)
        {
            AOCLFFTZ_LOG_UNFORMATTED(ERR, ERR, "dims and vecs parsing failed");
            FREE_ALIGN_ALLOCATED_MEM(dims);
            FREE_ALIGN_ALLOCATED_MEM(vecs);
            return;
        }

        INT32 new_dim_rank = 1;
        SHRINK_DIM_RANK(dims, dim_rank, new_dim_rank);

        // creating a solution object to store the reference values
        aoclfftz_solution_t *sol;
        sol = alloc_solution_wrapper(vec_rank, new_dim_rank);
        if (sol == NULL)
        {
            AOCLFFTZ_LOG_UNFORMATTED(ERR, ERR, "sol creation failed");
            FREE_ALIGN_ALLOCATED_MEM(dims);
            FREE_ALIGN_ALLOCATED_MEM(vecs);
            return;
        }
        sol->decomp_scheme->dim_rank = new_dim_rank;
        INT32 cnt = 0;
        for (INT32 idx = 0; idx < dim_rank; ++idx)
        {
            if (dims[idx].n != 1)
            {
                sol->decomp_scheme->dims[cnt].n = dims[idx].n;
                sol->decomp_scheme->dims[cnt].in_stride = dims[idx].in_stride;
                // Strides must be equal for inplace problems
                if (is_in_place)
                {
                    sol->decomp_scheme->dims[cnt].out_stride =
                                        dims[idx].in_stride;
                }
                else
                {
                    sol->decomp_scheme->dims[cnt].out_stride =
                                        dims[idx].out_stride;
                }
                cnt++;
            }
        }
        // Sets value for atleast one of dims array element in a 1D problem or
        // ND problem where the dim rank is reduced to one and the size of the
        // dimensions are one.
        // Example dims:1x1x1  or dims:1
        if (cnt == 0)
        {
            sol->decomp_scheme->dims[0].n = dims[0].n;
            sol->decomp_scheme->dims[0].in_stride = dims[0].in_stride;
            // Strides must be equal for inplace problems
            if (is_in_place)
            {
                sol->decomp_scheme->dims[0].out_stride = dims[0].in_stride;
            }
            else
            {
                sol->decomp_scheme->dims[0].out_stride = dims[0].out_stride;
            }
        }
        sol->decomp_scheme->vec_rank = vec_rank;
        for (INT32 idx = 0; idx < sol->decomp_scheme->vec_rank; ++idx)
        {
            sol->decomp_scheme->vecs[idx].n = vecs[idx].n;
            sol->decomp_scheme->vecs[idx].in_stride = vecs[idx].in_stride;
            // Strides must be equal for inplace problems
            if (is_in_place)
            {
                sol->decomp_scheme->vecs[idx].out_stride = vecs[idx].in_stride;
            }
            else
            {
                sol->decomp_scheme->vecs[idx].out_stride = vecs[idx].out_stride;
            }
        }
        FREE_ALIGN_ALLOCATED_MEM(dims);
        FREE_ALIGN_ALLOCATED_MEM(vecs);
        fuse_vecs_wrapper(sol);
        sol->decomp_scheme->flags = flags;
        sol->decomp_scheme->cntrl_params->opt_level = opt_level;
        sol->decomp_scheme->cntrl_params->opt_off = opt_level == -1 ? 1 : 0;
        sol->decomp_scheme->cntrl_params->logger_mode = 0;
        sol->decomp_scheme->cntrl_params->measure_stats = 0;
        sol->decomp_scheme->thread_info->pthr_fft->num_threads = 1;
        sol->decomp_scheme->thread_info->pthr_fft->dynamic_load_model = 0;
        sol->decomp_scheme->thread_info->avl_threads = 1;
        sol->decomp_scheme->thread_info->n_threads = 1;
        sol->solver->solver_type = solver_type;
        sol->next_sol = NULL;
        if (typeid(dt_t) == typeid(FLOAT))
        {
            SET_PRECISION(sol->decomp_scheme->flags, DT_FLOAT);
        }
        else if (typeid(dt_t) == typeid(DOUBLE))
        {
            SET_PRECISION(sol->decomp_scheme->flags, DT_DOUBLE);
        }
        *ref_sol = sol;
    }

    /**
     * @brief Verify the properties of solvers for all the child solutions
     * in `sol`
     *
     * @param sol solution to verify
     * @return bool
     */
    bool verify_properties(aoclfftz_solution_t *sol)
    {
        bool ret = true;
        aoclfftz_solution_t *cur_a = sol;

        while (cur_a != NULL)
        {
            if (cur_a->solver->solver_type == SOLVER_DIRECT)
            {
                ret &= (cur_a->strides_grp->strides->in_strides[1]) ==
                       (cur_a->decomp_scheme->dims[0].in_stride * DATA_STRIDE);
                ret &= (cur_a->strides_grp->strides->out_strides[1]) ==
                       (cur_a->decomp_scheme->dims[0].out_stride * DATA_STRIDE);
                ret &= cur_a->strides_grp->strides->v_in_stride ==
                       cur_a->decomp_scheme->vecs[0].in_stride * DATA_STRIDE;
                ret &= cur_a->strides_grp->strides->v_out_stride ==
                       cur_a->decomp_scheme->vecs[0].out_stride * DATA_STRIDE;
                if (ret == false)
                {
                    AOCLFFTZ_LOG_UNFORMATTED(
                        ERR, ERR, "Strides do not match !"
                        "Failed at level 2 compare [Direct solver]");
                    return false;
                }
                cur_a = cur_a->next_sol;
            }
            else if (cur_a->solver->solver_type == SOLVER_BATCHED)
            {
                aoclfftz_solution_t *next_sol = cur_a->next_sol;
                if (next_sol == NULL)
                {
                    AOCLFFTZ_LOG_UNFORMATTED(
                        ERR, ERR, "No solution after Batched ! "
                            "Failed at level 2 compare [Batched solver]");
                    return false;
                }
                ret &= ((next_sol->solver->solver_type == SOLVER_CT) ||
                        (next_sol->solver->solver_type == SOLVER_BLUESTEIN) ||
                        (next_sol->solver->solver_type == SOLVER_NDIM) ||
                        (next_sol->solver->solver_type == SOLVER_SIZEONE) ||
                        (next_sol->solver->solver_type == SOLVER_DIRECT));
                ret &= next_sol->decomp_scheme->vec_rank == 1;
                ret &= next_sol->decomp_scheme->vecs[0].n == 1;
                if (ret == false)
                {
                    AOCLFFTZ_LOG_UNFORMATTED(
                        ERR, ERR, "Failed at level 2 compare [Batched solver]");
                    return false;
                }
                cur_a = next_sol;
            }
            else if (cur_a->solver->solver_type == SOLVER_CT)
            {
                aoclfftz_solution_t *sol_r = cur_a->next_sol;
                if (sol_r == NULL)
                {
                    AOCLFFTZ_LOG_UNFORMATTED(
                        ERR, ERR, "Solution-r not present !"
                        "Failed at level 2 compare [CT solver]");
                    return false;
                }
                aoclfftz_solution_t *sol_m = cur_a->next_sol->next_sol;
                if (sol_m == NULL)
                {
                    AOCLFFTZ_LOG_UNFORMATTED(
                        ERR, ERR, "Solution-m not present !"
                        "Failed at level 2 compare [CT solver]");
                    return false;
                }
                ret &= (sol_r->solver->solver_type == SOLVER_DIRECT);
                ret &= ((sol_m->solver->solver_type == SOLVER_DIRECT) ||
                        (sol_m->solver->solver_type == SOLVER_BATCHED));

                // verify the dims and vecs values of solution-r and solution-m
                ret &= (cur_a->decomp_scheme->dims[0].n ==
                        sol_r->decomp_scheme->dims[0].n *
                            sol_m->decomp_scheme->dims[0].n);
                ret &= (sol_r->decomp_scheme->dims[0].n ==
                        sol_m->decomp_scheme->vecs[0].n);
                ret &= (sol_m->decomp_scheme->dims[0].n ==
                        sol_r->decomp_scheme->vecs[0].n);

                // verify the strides and vec strides of solution-r
                ret &= (sol_r->decomp_scheme->dims[0].in_stride ==
                        sol_m->decomp_scheme->dims[0].n *
                            cur_a->decomp_scheme->dims[0].out_stride);
                ret &= (sol_r->decomp_scheme->dims[0].out_stride ==
                        sol_m->decomp_scheme->dims[0].n *
                            cur_a->decomp_scheme->dims[0].out_stride);
                ret &= (sol_r->decomp_scheme->vecs[0].in_stride ==
                        cur_a->decomp_scheme->dims[0].out_stride);
                ret &= (sol_r->decomp_scheme->vecs[0].out_stride ==
                        cur_a->decomp_scheme->dims[0].out_stride);

                // verify the strides and vec strides of solution-m
                ret &= (sol_m->decomp_scheme->dims[0].in_stride ==
                        sol_r->decomp_scheme->dims[0].n *
                            cur_a->decomp_scheme->dims[0].in_stride);
                ret &= (sol_m->decomp_scheme->dims[0].out_stride ==
                        (IS_OUT_OF_PLACE(cur_a->decomp_scheme->flags)
                             ? cur_a->decomp_scheme->dims[0].out_stride
                             : sol_r->decomp_scheme->dims[0].n *
                                   cur_a->decomp_scheme->dims[0].out_stride));
                ret &= (sol_m->decomp_scheme->vecs[0].in_stride ==
                        cur_a->decomp_scheme->dims[0].in_stride);
                ret &= (sol_m->decomp_scheme->vecs[0].out_stride ==
                        (IS_OUT_OF_PLACE(cur_a->decomp_scheme->flags)
                             ? sol_m->decomp_scheme->dims[0].n *
                                   cur_a->decomp_scheme->dims[0].out_stride
                             : cur_a->decomp_scheme->dims[0].out_stride));

                if (ret == false)
                {
                    AOCLFFTZ_LOG_UNFORMATTED(
                        ERR, ERR, "Failed at level 2 compare [CT solver]");
                    return false;
                }
                cur_a = sol_m;
            }
            else if (cur_a->solver->solver_type == SOLVER_BLUESTEIN)
            {
                INTP n = cur_a->decomp_scheme->dims[0].n;
                if (cur_a->next_sol == NULL)
                {
                    AOCLFFTZ_LOG_UNFORMATTED(
                        ERR, ERR, "No solution after Bluestein !"
                        "Failed at level 2 compare [Bluestein solver]");
                }
                UINT8 dt_prec = DT_PRECISION_FLAG(cur_a->decomp_scheme->flags);
                UINT32 dt_bytes = DT_PRECISION_BYTES(dt_prec);
                INTP m = cur_a->next_sol->decomp_scheme->dims[0].n;
                VOID *B = cur_a->dft_bufs->bluestein->B;
                VOID *B_ref = NULL;
                ALLOC_ALIGN_UNINIT(B_ref, VOID, m * DATA_STRIDE * dt_bytes);
                prepare_bluestein_sequence_ref(B_ref, m, n, dt_prec);
                ret &= get_extended_length_ref(n) == m;
                ret &= ((B != NULL) &&
                        (memcmp(B, B_ref, m * DATA_STRIDE * dt_bytes) == 0));
                FREE_ALIGN_ALLOCATED_MEM(B_ref);
                if (ret == false)
                {
                    AOCLFFTZ_LOG_UNFORMATTED(
                        ERR, ERR,
                        "Failed at level 2 compare [Bluestein solver]");
                    return false;
                }
                cur_a = cur_a->next_sol;
            }
            else if (cur_a->solver->solver_type == SOLVER_NDIM)
            {
                aoclfftz_solution_t *sol_1d = cur_a->next_sol;
                if (sol_1d == NULL)
                {
                    AOCLFFTZ_LOG_UNFORMATTED(
                        ERR, ERR, "sol_1d is NULL [NDim solver]");
                    return false;
                }
                aoclfftz_solution_t *sol_nd = cur_a->dft_bufs->nd_sol;
                if (sol_nd == NULL)
                {
                    AOCLFFTZ_LOG_UNFORMATTED(
                        ERR, ERR, "sol_nd is NULL [NDim solver]");
                    return false;
                }

                // sol_nd can only be either Direct (if parent is 2D) or
                // Batched (parent > 2D case)
                ret &= ((sol_nd->solver->solver_type == SOLVER_DIRECT) ||
                        (sol_nd->solver->solver_type == SOLVER_BATCHED));

                // sol_1d can only be Direct in contiguous memory scenario
                ret &= ((sol_1d->solver->solver_type == SOLVER_DIRECT) ||
                        (sol_1d->solver->solver_type == SOLVER_BATCHED));

                INT32 cur_dim_rank = cur_a->decomp_scheme->dim_rank;
                INT32 rank_nd = sol_nd->decomp_scheme->dim_rank;

                // verify the dims & vecs of solution-nd
                ret &= (sol_nd->decomp_scheme->dim_rank ==
                        cur_a->decomp_scheme->dim_rank - 1);
                ret &= (sol_nd->decomp_scheme->vec_rank == 1);

                ret &= (sol_nd->decomp_scheme->vecs[0].n ==
                        cur_a->decomp_scheme->dims[cur_dim_rank - 1].n);
                ret &= (sol_nd->decomp_scheme->vecs[0].in_stride ==
                        cur_a->decomp_scheme->dims[cur_dim_rank - 1].in_stride);
                ret &=
                    (sol_nd->decomp_scheme->vecs[0].out_stride ==
                     cur_a->decomp_scheme->dims[cur_dim_rank - 1].out_stride);

                ret &= (memcmp(sol_nd->decomp_scheme->dims,
                               cur_a->decomp_scheme->dims,
                               sizeof(aoclfftz_dim_t_64_) * rank_nd) == 0);

                // verify the dims & vecs of solution-1d
                ret &= !(IS_OUT_OF_PLACE(sol_1d->decomp_scheme->flags));

                ret &= (sol_1d->decomp_scheme->dim_rank == 1);
                ret &= (sol_1d->decomp_scheme->dims[0].n ==
                        cur_a->decomp_scheme->dims[cur_dim_rank - 1].n);
                ret &= (sol_1d->decomp_scheme->dims[0].in_stride ==
                        sol_1d->decomp_scheme->dims[0].out_stride);
                ret &=
                    (sol_1d->decomp_scheme->dims[0].in_stride ==
                     cur_a->decomp_scheme->dims[cur_dim_rank - 1].out_stride);

                INTP total_size = 1;
                for (INT32 i = 0; i < cur_dim_rank; i++)
                {
                    total_size *= cur_a->decomp_scheme->dims[i].n;
                }

                // TODO: add a different simple logic for fusable_dims ?
                ret &= ((sol_1d->decomp_scheme->vec_rank == 1) ||
                        (sol_1d->decomp_scheme->vec_rank <
                         cur_a->decomp_scheme->dim_rank));

                // combined size of dims & vecs of 1D should be equal to
                // the combined size of dims of cur solution
                INTP total_len_1d = sol_1d->decomp_scheme->dims[0].n;
                for (INT32 i = 0; i < sol_1d->decomp_scheme->vec_rank; i++)
                {
                    total_len_1d *= sol_1d->decomp_scheme->vecs[i].n;
                }

                ret &= (total_len_1d == total_size);

                // combined size of vecs of 1D should be equal to
                // the combined size of dims of cur solution/outermost
                // dimension size
                ret &= (total_len_1d / sol_1d->decomp_scheme->dims[0].n ==
                        total_size /
                        cur_a->decomp_scheme->dims[cur_dim_rank - 1].n);
                ret &= ((sol_1d->decomp_scheme->vecs[0].in_stride ==
                        cur_a->decomp_scheme->dims[0].out_stride));
                ret &= ((sol_1d->decomp_scheme->vecs[0].out_stride ==
                        cur_a->decomp_scheme->dims[0].out_stride));

                INT32 i,j;
                // for every vector in 1D solution
                for (i = 0, j = 0;
                     i < sol_1d->decomp_scheme->vec_rank && j < cur_dim_rank -1
                     && ret; i++)
                {
                    INT32 k;
                    // if the current solution dimension does not match the 1D
                    // solution vector check if it was fused by the library
                    if (cur_a->decomp_scheme->dims[j].n !=
                         sol_1d->decomp_scheme->vecs[i].n)
                    {
                        // check for fused dimension size:
                        INTP fused_dim_size = cur_a->decomp_scheme->dims[j].n;
                        for (k = j+1; k < cur_dim_rank - 1; k++)
                        {
                            fused_dim_size = fused_dim_size *
                                             cur_a->decomp_scheme->dims[k].n;
                            if (sol_1d->decomp_scheme->vecs[i].n ==
                                                                fused_dim_size)
                            {
                                // break if the size of the fused dim matches
                                // the size of the 1D solution vector size
                                // (which would have probably be fused in the
                                //  library)
                                break;
                            }
                        }
                        if (k == cur_dim_rank -1)
                        {
                            //vector size didnt match
                            ret = false;
                        }
                        else
                        {
                            //check if its fusable
                            if ((sol_1d->decomp_scheme->vecs[i].n *
                                 sol_1d->decomp_scheme->vecs[i].in_stride ==
                                cur_a->decomp_scheme->dims[k].n *
                                 cur_a->decomp_scheme->dims[k].out_stride) &&
                                (sol_1d->decomp_scheme->vecs[i].n *
                                 sol_1d->decomp_scheme->vecs[i].out_stride ==
                                cur_a->decomp_scheme->dims[k].n *
                                 cur_a->decomp_scheme->dims[k].out_stride))
                            {
                                j = k+1;
                            }
                            else
                            {
                                ret = false;
                            }
                        }
                    }
                    else
                    {
                        j++;
                    }
                }
                ret &= (i == sol_1d->decomp_scheme->vec_rank);
                ret &= (j == cur_dim_rank - 1);
                if (ret == false)
                {
                    AOCLFFTZ_LOG_UNFORMATTED(
                        ERR, ERR, "Failed at level 2 compare [NDim solver]");
                    return false;
                }

                // traverse along ND solution and verify all of its children
                cur_a = sol_nd;
                ret = verify_properties(cur_a);
                if (ret == false)
                {
                    AOCLFFTZ_LOG_UNFORMATTED(
                        ERR, ERR, "Error along ND solution [NDim solver]");
                    return false;
                }

                // traverse along 1D solution and verify its children
                cur_a = sol_1d;
            }
            else if (cur_a->solver->solver_type == SOLVER_SIZEONE)
            {
                ret &= (cur_a->decomp_scheme->dims[0].n == 1);
                ret &= (cur_a->next_sol == NULL);
                if (ret == false)
                {
                    AOCLFFTZ_LOG_UNFORMATTED(
                        ERR, ERR, "Failed at Level 2 compare [SizeOne solver]");
                    return false;
                }
                cur_a = cur_a->next_sol;
            }
            else
            {
                AOCLFFTZ_LOG_UNFORMATTED(ERR, ERR, "Invalid solver type");
                return false;
            }
        };

        return ret;
    }

    /**
     * @brief compare the first solutions of two solution lists `a` and `b`
     * and verify the properties of solvers for all the each child solutions
     * in `a` belonging to the decomposed sub-problems
     *
     * @param sol_a first solution object
     * @param sol_b second solution object
     * @return bool
     */
    bool verify_solution(aoclfftz_solution_t *sol_a, aoclfftz_solution_t *sol_b)
    {
        if (sol_a == NULL || sol_b == NULL)
        {
            AOCLFFTZ_LOG_UNFORMATTED(ERR, ERR, "sol_a or sol_b is empty");
            return false;
        }
        bool ret = true;
        aoclfftz_solution_t *cur_a = sol_a;
        aoclfftz_solution_t *cur_b = sol_b;

        /*************************************************************
         *  Level 1: Compare the first solution of sol_a with sol_b  *
         *************************************************************/

        // ********** check solver **********
        ret &= (cur_a->solver->solver_type == cur_b->solver->solver_type);
        // ********** decomp scheme **********
        // dims and vecs
        ret &=
            (cur_a->decomp_scheme->dim_rank == cur_b->decomp_scheme->dim_rank);
        for (INT32 i = 0; i < cur_a->decomp_scheme->dim_rank; ++i)
        {
            ret &= (cur_a->decomp_scheme->dims[i].n ==
                    cur_b->decomp_scheme->dims[i].n);
            ret &= (cur_a->decomp_scheme->dims[i].in_stride ==
                    cur_b->decomp_scheme->dims[i].in_stride);
            ret &= (cur_a->decomp_scheme->dims[i].out_stride ==
                    cur_b->decomp_scheme->dims[i].out_stride);
        }

        ret &=
            (cur_a->decomp_scheme->vec_rank == cur_b->decomp_scheme->vec_rank);
        for (INT32 i = 0; i < cur_a->decomp_scheme->vec_rank; ++i)
        {
            ret &= (cur_a->decomp_scheme->vecs[i].n ==
                    cur_b->decomp_scheme->vecs[i].n);
            ret &= (cur_a->decomp_scheme->vecs[i].in_stride ==
                    cur_b->decomp_scheme->vecs[i].in_stride);
            ret &= (cur_a->decomp_scheme->vecs[i].out_stride ==
                    cur_b->decomp_scheme->vecs[i].out_stride);
        }
        // flags
        ret &= (cur_a->decomp_scheme->flags == cur_b->decomp_scheme->flags);
        // cntrl params
        ret &= (cur_a->decomp_scheme->cntrl_params->opt_level ==
                cur_b->decomp_scheme->cntrl_params->opt_level);
        ret &= (cur_a->decomp_scheme->cntrl_params->opt_off ==
                cur_b->decomp_scheme->cntrl_params->opt_off);

        // following values from decomp_scheme are skipped from comparison
        // 1. cntrl_params->logger_mode
        // 2. cntrl_params->measure_stat
        // 3. thread_info->pthr_fft->num_threads
        // 4. thread_info->pthr_fft->dynamic_load_model
        // 5. thread_info->avl_threads
        // 6. thread_info->n_threads

        // ********** twiddle **********
        // either both should be NULL or both should have values
        ret &=
            (((cur_a->twiddle->TW != NULL) && (cur_b->twiddle->TW != NULL)) ||
             ((cur_a->twiddle->TW == NULL) && (cur_b->twiddle->TW == NULL)));

        if (ret == false)
        {
            AOCLFFTZ_LOG_UNFORMATTED(ERR, ERR, "Level 1 compare failed");
            return false;
        }

        /*****************************************************************
         * Level 2: Verify the properties of solvers for all the child   *
         *          solutions belonging to the decomposed sub-problems.  *
         *****************************************************************/
        ret &= verify_properties(cur_a);

        return ret;
    }

    /**
     * @brief compare the solution object list's dft_solver using the given
     * solver list
     *
     * @param sol solution list
     * @param solver_list list of solvers
     * @return bool
     */
    bool compare_solver_list(aoclfftz_solution_t *sol,
                             std::vector<aoclfftz_solver_type> solver_list)
    {
        if (sol == NULL)
        {
            AOCLFFTZ_LOG_UNFORMATTED(ERR, ERR, "sol is empty");
            return false;
        }
        bool ret = true;
        UINT32 node_count = 0;
        aoclfftz_solution_t *cur_sol = sol;
        aoclfftz_solution_t *nd_sol = NULL;
        do // for ND support
        {
            if (nd_sol != NULL)
            {
                cur_sol = nd_sol;
                nd_sol = NULL;
            }
            while (cur_sol != NULL && ret && node_count <= solver_list.size())
            {
                switch (solver_list[node_count])
                {
                case SOLVER_DIRECT:
                    ret &= (cur_sol->solver->solver_type == SOLVER_DIRECT);
                    break;
                case SOLVER_BATCHED:
                    ret &= (cur_sol->solver->solver_type == SOLVER_BATCHED);
                    break;
                case SOLVER_CT:
                    ret &= (cur_sol->solver->solver_type == SOLVER_CT);
                    break;
                case SOLVER_BLUESTEIN:
                    ret &= (cur_sol->solver->solver_type == SOLVER_BLUESTEIN);
                    break;
                case SOLVER_NDIM:
                    ret &= (cur_sol->solver->solver_type == SOLVER_NDIM);
                    nd_sol = cur_sol->dft_bufs->nd_sol;
                    break;
                case SOLVER_SIZEONE:
                    ret &= (cur_sol->solver->solver_type == SOLVER_SIZEONE);
                    break;
                default:
                    ret = false;
                    break;
                }
                node_count++;
                cur_sol = cur_sol->next_sol;
            }
        } while (nd_sol != NULL && ret);
        return ret && (node_count == solver_list.size());
    }
};

/**
 * @brief A test class derived from AoclfftzSelectorTestBase for
 * FLOAT data-type and LP64 data-model
 *
 */
class AoclfftzSelectorTestFloatLP64
    : public AoclfftzSelectorTestBase<FLOAT, INT32, aoclfftz_dim_t,
                                      aoclfftz_prob_desc_f>
{
  protected:
    AoclfftzSelectorTestFloatLP64()
    {
        aoclfftz_setup = aoclfftz_setup_f;
    }
};

/**
 * @brief A derived class from AoclfftzSelectorTestBase for DOUBLE
 * data-type and LP64 data-model
 *
 */
class AoclfftzSelectorTestDoubleLP64
    : public AoclfftzSelectorTestBase<DOUBLE, INT32, aoclfftz_dim_t,
                                      aoclfftz_prob_desc_d>
{
  protected:
    AoclfftzSelectorTestDoubleLP64()
    {
        aoclfftz_setup = aoclfftz_setup_d;
    }
};

/**
 * @brief A derived class from AoclfftzSelectorTestBase for FLOAT
 * data-type and ILP64 data-model
 *
 */
class AoclfftzSelectorTestFloatILP64
    : public AoclfftzSelectorTestBase<FLOAT, INTP, aoclfftz_dim_t_64_,
                                      aoclfftz_prob_desc_f_64_>
{
  protected:
    AoclfftzSelectorTestFloatILP64()
    {
        aoclfftz_setup = aoclfftz_setup_f_64_;
    }
};

/**
 * @brief A derived class from AoclfftzSelectorTestBase for DOUBLE data-type and
 * ILP64 data-model
 *
 */
class AoclfftzSelectorTestDoubleILP64
    : public AoclfftzSelectorTestBase<DOUBLE, INTP, aoclfftz_dim_t_64_,
                                      aoclfftz_prob_desc_d_64_>
{
  protected:
    AoclfftzSelectorTestDoubleILP64()
    {
        aoclfftz_setup = aoclfftz_setup_d_64_;
    }
};

#endif // AOCLFFTZ_SELECTOR_GTEST_BASE_H
