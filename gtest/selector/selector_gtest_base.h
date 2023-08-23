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
#include "api/aoclfftz.h"
extern "C"
{
#include "core/common/memory_manager.h"
}
#include "gtest/gtest_types.h"
#include "test/aoclfftz_corebench_utils.h"

/**
 * @brief a macro for Gtest SetUp function to be used in non-parameterized test
 * classes
 *
 */
#define SETUP_SELECTOR_TEST_CLASS                                              \
    void SetUp() override                                                      \
    {                                                                          \
        random_seed = std::chrono::duration_cast<std::chrono::microseconds>(   \
                          std::chrono::system_clock::now().time_since_epoch()) \
                          .count();                                            \
        srand(random_seed);                                                    \
    }

/**
 * @brief a macro for Gtest SetUp function and run_selector_test function to be
 * used in parameterized test classes
 *
 */
#define SETUP_SELECTOR_TEST_WITH_PARAMS_CLASS                                  \
    void SetUp() override                                                      \
    {                                                                          \
        random_seed = std::chrono::duration_cast<std::chrono::microseconds>(   \
                          std::chrono::system_clock::now().time_since_epoch()) \
                          .count();                                            \
        srand(random_seed);                                                    \
    }                                                                          \
    void run_selector_test()                                                   \
    {                                                                          \
        run_selector_test_with_param(GetParam());                              \
    }

/**
 * @brief a helper macro function to compare the solution objects
 *
 */
#define COMPARE_SOLUTIONS(sol, ref_sol)                                        \
    ASSERT_NE(sol, nullptr) << "Failed at: run_setup_and_get_solution\n";      \
    ASSERT_NE(ref_sol, nullptr) << "Failed at: generate_reference_solution\n"; \
    ASSERT_TRUE(compare_solution(sol, ref_sol))                                \
        << "Failed at: compare_solution";

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
{
  protected:
    // function pointers for aoclfftz_setup_* and aoclfftz_execute_* APIs
    VOID *(*aoclfftz_setup)(prob_desc_t *);
    INT32 (*aoclfftz_execute)(VOID *);

    UINT64 random_seed; // random seed value used for random data generation
    // following pointers are created in class level scope to hold the objects
    // created inside member functions
    // class level scope is used for better memory management
    VOID *handle;        // to store the handle object returns from setup API
    prob_desc_t *p_desc; // to store the problem descriptor
    aoclfftz_solution_t *ref_solution; // to store the reference solution object
    aoclfftz_cntrl_params cntrl_params; // control params object
    aoclfftz_smp_pfft pthr_fft;         // SMP FFT object

    AoclfftzSelectorTestBase()
    {
        p_desc = NULL;
        handle = NULL;
        ref_solution = NULL;
        cntrl_params = {0x0};
        pthr_fft = {0x0};
    }

    ~AoclfftzSelectorTestBase()
    {
        destroy_handle(handle);
        destroy_solution(ref_solution);
        // destroy problem descriptor
        if (p_desc != NULL)
        {
            FREE_ALLOCATED_MEM(p_desc->in);
            FREE_ALLOCATED_MEM(p_desc->out);
            FREE_ALLOCATED_MEM(p_desc->dims);
            FREE_ALLOCATED_MEM(p_desc->vecs);
            FREE_ALLOCATED_MEM(p_desc);
        }
    }

    /**
     * @brief Entry function to run the generic selector test
     *
     * @param params selector test params contains dims, flags and solver list
     */
    void run_selector_test_with_param(aoclfftz_selector_test_params_t params)
    {
        const auto [dims_and_vecs, flags, opt_level, solver_list] = params;
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
        status = find_dim_vec_ranks((CHAR *)dims_and_vecs.c_str(), &dim_rank,
                                    &vec_rank);
        if (status != PARSER_SUCCESS)
        {
            return NULL;
        }
        aoclfftz_dim_t_64_ *dims = NULL;
        aoclfftz_dim_t_64_ *vecs = NULL;
        status = allocate_and_fill_dims_vecs(
            (CHAR *)dims_and_vecs.c_str(), dim_rank, vec_rank, &dims, &vecs, 1);
        if (status != PARSER_SUCCESS)
        {
            return NULL;
        }

        p_desc = (prob_desc_t *)ALLOC_UNALIGN_UNINIT(sizeof(prob_desc_t));
        p_desc->dim_rank = dim_rank;
        p_desc->vec_rank = vec_rank;
        p_desc->dims = (dim_t *)ALLOC_UNALIGN_UNINIT(dim_rank * sizeof(dim_t));
        for (INT32 i = 0; i < dim_rank; i++)
        {
            p_desc->dims[i].n = (dm_t)dims[i].n;
            p_desc->dims[i].in_stride = (dm_t)dims[i].in_stride;
            p_desc->dims[i].out_stride = (dm_t)dims[i].out_stride;
        }
        p_desc->vecs = (dim_t *)ALLOC_UNALIGN_UNINIT(dim_rank * sizeof(dim_t));
        for (INT32 i = 0; i < vec_rank; i++)
        {
            p_desc->vecs[i].n = (dm_t)vecs[i].n;
            p_desc->vecs[i].in_stride = (dm_t)vecs[i].in_stride;
            p_desc->vecs[i].out_stride = (dm_t)vecs[i].out_stride;
        }
        FREE_ALLOCATED_MEM(dims);
        FREE_ALLOCATED_MEM(vecs);

        dm_t n = p_desc->dims[0].n;
        dt_t *in = (dt_t *)ALLOC_UNALIGN_UNINIT(2 * n * sizeof(dt_t));
        dt_t *out = (dt_t *)calloc(2 * n, sizeof(dt_t));
        for (int i = 0; i < 2 * n; i++)
            in[i] = (rand() % 1000) / 100.0;

        p_desc->in = in;
        p_desc->out = out;
        // in-place:0-bit, real:1-bit, out-of-order:2-bit, dir:3-bit
        p_desc->flags = flags;

        p_desc->cntrl_params.opt_level = opt_level;
        if (opt_level == -1)
        {
            p_desc->cntrl_params.opt_off = 1;
        }

        p_desc->pthr_fft.num_threads = 1;
        p_desc->pthr_fft.dynamic_load_model = 0;

        handle = aoclfftz_setup(p_desc);
        if (handle == NULL)
        {
            return NULL;
        }
        aoclfftz_selector_t *sel = (aoclfftz_selector_t *)handle;
        return sel->solution;
    }

    /**
     * @brief Get the solution object based on given params
     *
     * @param dims_and_vecs problem descriptor string which contains dims and
     * vecs info
     * @param flags in-place:0-bit, real:1-bit, out-of-order:2-bit, dir:3-bit
     * @param kernel_r pointer to the first FFT kernel
     * @param kernel_m pointer to the second FFT kernel (only used in CT-solver,
     * NULL otherwise)
     * @param exec pointer to the solver's execute function
     * @param opt_level optimization level
     * @return aoclfftz_solution_t*
     */
    aoclfftz_solution_t *
    generate_reference_solution(std::string dims_and_vecs, INT32 flags,
                                INT32 opt_level, kfft_ kernel_r, kfft_ kernel_m,
                                dft_solver_ exec)
    {
        INT32 status = PARSER_SUCCESS;
        INT32 dim_rank = 0;
        INT32 vec_rank = 0;
        status = find_dim_vec_ranks((CHAR *)dims_and_vecs.c_str(), &dim_rank,
                                    &vec_rank);
        if (status != PARSER_SUCCESS)
        {
            return NULL;
        }
        aoclfftz_dim_t_64_ *dims;
        aoclfftz_dim_t_64_ *vecs;
        status = allocate_and_fill_dims_vecs(
            (CHAR *)dims_and_vecs.c_str(), dim_rank, vec_rank, &dims, &vecs, 1);
        if (status != PARSER_SUCCESS)
        {
            return NULL;
        }
        // creating a solution object to store the reference values
        ref_solution = alloc_solution(vec_rank, dim_rank);
        if (ref_solution == NULL)
        {
            return NULL;
        }
        ref_solution->decomp_scheme->dim_rank = dim_rank;
        for (INT32 idx = 0; idx < ref_solution->decomp_scheme->dim_rank; ++idx)
        {
            ref_solution->decomp_scheme->dims[idx].n = dims[idx].n;
            ref_solution->decomp_scheme->dims[idx].in_stride =
                dims[idx].in_stride;
            ref_solution->decomp_scheme->dims[idx].out_stride =
                dims[idx].out_stride;
        }
        ref_solution->decomp_scheme->vec_rank = vec_rank;
        for (INT32 idx = 0; idx < ref_solution->decomp_scheme->vec_rank; ++idx)
        {
            ref_solution->decomp_scheme->vecs[idx].n = vecs[idx].n;
            ref_solution->decomp_scheme->vecs[idx].in_stride =
                vecs[idx].in_stride;
            ref_solution->decomp_scheme->vecs[idx].out_stride =
                vecs[idx].out_stride;
        }
        FREE_ALLOCATED_MEM(dims);
        FREE_ALLOCATED_MEM(vecs);
        ref_solution->strides->in_stride =
            ref_solution->decomp_scheme->dims[0].in_stride;
        ref_solution->strides->out_stride =
            ref_solution->decomp_scheme->dims[0].out_stride;
        ref_solution->strides->v_in_stride =
            ref_solution->decomp_scheme->vecs[0].in_stride;
        ref_solution->strides->v_out_stride =
            ref_solution->decomp_scheme->vecs[0].out_stride;
        ref_solution->decomp_scheme->flags = flags;
        ref_solution->decomp_scheme->cntrl_params = &cntrl_params;
        ref_solution->decomp_scheme->cntrl_params->opt_level = opt_level;
        ref_solution->decomp_scheme->cntrl_params->opt_off =
            opt_level == -1 ? 1 : 0;
        ref_solution->decomp_scheme->cntrl_params->logger_mode = 0;
        ref_solution->decomp_scheme->cntrl_params->measure_stats = 0;
        ref_solution->decomp_scheme->pthr_fft = &pthr_fft;
        ref_solution->decomp_scheme->pthr_fft->num_threads = 1;
        ref_solution->decomp_scheme->pthr_fft->dynamic_load_model = 0;
        ref_solution->solver->kernel_r = kernel_r;
        ref_solution->solver->kernel_m = kernel_m;
        ref_solution->solver->execute_solver = exec;
        ref_solution->next_sol = NULL;
        if (typeid(dt_t) == typeid(FLOAT))
        {
            SET_PRECISION(ref_solution->decomp_scheme->flags, DT_FLOAT);
        }
        else if (typeid(dt_t) == typeid(DOUBLE))
        {
            SET_PRECISION(ref_solution->decomp_scheme->flags, DT_DOUBLE);
        }

        return ref_solution;
    }

    /**
     * @brief compare the two solution object `a` and `b`
     *
     * @param sol_a first solution object
     * @param sol_b second solution object
     * @return bool
     */
    bool compare_solution(aoclfftz_solution_t *sol_a,
                          aoclfftz_solution_t *sol_b)
    {
        if (sol_a == NULL || sol_b == NULL)
        {
            return false;
        }
        bool ret = true;
        aoclfftz_solution_t *cur_a = sol_a;
        aoclfftz_solution_t *cur_b = sol_b;
        while (cur_a != NULL && cur_b != NULL && ret)
        {
            // ********** check solver **********
            ret &= (cur_a->solver->kernel_r == cur_b->solver->kernel_r);
            ret &= (cur_a->solver->kernel_m == cur_b->solver->kernel_m);
            ret &= (cur_a->solver->execute_solver ==
                    cur_b->solver->execute_solver);

            // ********** check strides **********
            ret &= (cur_a->strides->in_stride == cur_b->strides->in_stride);
            ret &= (cur_a->strides->out_stride == cur_b->strides->out_stride);
            ret &= (cur_a->strides->v_in_stride == cur_b->strides->v_in_stride);
            ret &=
                (cur_a->strides->v_out_stride == cur_b->strides->v_out_stride);

            // ********** decomp scheme **********
            // dims and vecs
            ret &= (cur_a->decomp_scheme->dim_rank ==
                    cur_b->decomp_scheme->dim_rank);
            for (INT32 i = 0; i < cur_a->decomp_scheme->dim_rank; ++i)
            {
                ret &= (cur_a->decomp_scheme->dims[i].n ==
                        cur_b->decomp_scheme->dims[i].n);
                ret &= (cur_a->decomp_scheme->dims[i].in_stride ==
                        cur_b->decomp_scheme->dims[i].in_stride);
                ret &= (cur_a->decomp_scheme->dims[i].out_stride ==
                        cur_b->decomp_scheme->dims[i].out_stride);
            }
            ret &= (cur_a->decomp_scheme->vec_rank ==
                    cur_b->decomp_scheme->vec_rank);
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
            // 3. pthr_fft->num_threads
            // 4. pthr_fftdynamic_load_model

            // ********** twiddle **********
            // either both should be NULL or both should have values
            ret &= (((cur_a->twiddle->TW != NULL) &&
                     (cur_b->twiddle->TW != NULL)) ||
                    ((cur_a->twiddle->TW == NULL) &&
                     (cur_b->twiddle->TW == NULL)));

            // go to next solution
            cur_a = cur_a->next_sol;
            cur_b = cur_b->next_sol;
        }
        if (cur_a != NULL || cur_b != NULL)
        {
            ret = false;
        }
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
            return false;
        }
        bool ret = true;
        UINT32 node_count = 0;
        aoclfftz_solution_t *cur_sol = sol;
        while (cur_sol != NULL)
        {
            switch (solver_list[node_count])
            {
            case SOLVER_DIRECT:
                ret &= (cur_sol->solver->solver_type == SOLVER_DIRECT);
                ret &= (cur_sol->solver->execute_solver == executor_direct_dft);
                break;
            case SOLVER_CT:
                ret &= (cur_sol->solver->solver_type == SOLVER_CT);
                ret &= (cur_sol->solver->execute_solver == executor_ct_dft);
                break;
            default:
                ret = false;
                break;
            }
            node_count++;
            cur_sol = cur_sol->next_sol;
        }
        return ret && (node_count == solver_list.size());
    }
};

/**
 * @brief A test class derived from AoclfftzSelectorTestBase for FLOAT data-type
 * and LP64 data-model
 *
 */
class AoclfftzSelectorTestFloatLP64
    : public AoclfftzSelectorTestBase<FLOAT, INT32, aoclfftz_dim_t,
                                      aoclfftz_prob_desc_f>,
      public ::testing::Test
{
  protected:
    SETUP_SELECTOR_TEST_CLASS;
    AoclfftzSelectorTestFloatLP64()
    {
        aoclfftz_setup = aoclfftz_setup_f;
        aoclfftz_execute = aoclfftz_execute_f;
    }
};

/**
 * @brief A test class derived from AoclfftzSelectorTestBase for DOUBLE
 * data-type and LP64 data-model
 *
 */
class AoclfftzSelectorTestDoubleLP64
    : public AoclfftzSelectorTestBase<DOUBLE, INT32, aoclfftz_dim_t,
                                      aoclfftz_prob_desc_d>,
      public ::testing::Test
{
  protected:
    SETUP_SELECTOR_TEST_CLASS;
    AoclfftzSelectorTestDoubleLP64()
    {
        aoclfftz_setup = aoclfftz_setup_d;
        aoclfftz_execute = aoclfftz_execute_d;
    }
};

/**
 * @brief A test class derived from AoclfftzSelectorTestBase for FLOAT data-type
 * and ILP64 data-model
 *
 */
class AoclfftzSelectorTestFloatILP64
    : public AoclfftzSelectorTestBase<FLOAT, INTP, aoclfftz_dim_t_64_,
                                      aoclfftz_prob_desc_f_64_>,
      public ::testing::Test
{
  protected:
    SETUP_SELECTOR_TEST_CLASS;
    AoclfftzSelectorTestFloatILP64()
    {
        aoclfftz_setup = aoclfftz_setup_f_64_;
        aoclfftz_execute = aoclfftz_execute_f_64_;
    }
};

/**
 * @brief A parameterized test class derived from AoclfftzSelectorTestBase for
 * DOUBLE data-type and ILP64 data-model
 *
 */
class AoclfftzSelectorTestDoubleILP64
    : public AoclfftzSelectorTestBase<DOUBLE, INTP, aoclfftz_dim_t_64_,
                                      aoclfftz_prob_desc_d_64_>,
      public ::testing::Test
{
  protected:
    SETUP_SELECTOR_TEST_CLASS;
    AoclfftzSelectorTestDoubleILP64()
    {
        aoclfftz_setup = aoclfftz_setup_d_64_;
        aoclfftz_execute = aoclfftz_execute_d_64_;
    }
};

/**
 * @brief A parameterized test class derived from AoclfftzSelectorTestBase for
 * FLOAT data-type and LP64 data-model
 *
 */
class AoclfftzSelectorTestFloatLP64Parameterized
    : public AoclfftzSelectorTestBase<FLOAT, INT32, aoclfftz_dim_t,
                                      aoclfftz_prob_desc_f>,
      public ::testing::TestWithParam<aoclfftz_selector_test_params_t>
{
  protected:
    SETUP_SELECTOR_TEST_WITH_PARAMS_CLASS;
    AoclfftzSelectorTestFloatLP64Parameterized()
    {
        aoclfftz_setup = aoclfftz_setup_f;
        aoclfftz_execute = aoclfftz_execute_f;
    }
};

/**
 * @brief A parameterized derived class from AoclfftzSelectorTestBase for DOUBLE
 * data-type and LP64 data-model
 *
 */
class AoclfftzSelectorTestDoubleLP64Parameterized
    : public AoclfftzSelectorTestBase<DOUBLE, INT32, aoclfftz_dim_t,
                                      aoclfftz_prob_desc_d>,
      public ::testing::TestWithParam<aoclfftz_selector_test_params_t>
{
  protected:
    SETUP_SELECTOR_TEST_WITH_PARAMS_CLASS;
    AoclfftzSelectorTestDoubleLP64Parameterized()
    {
        aoclfftz_setup = aoclfftz_setup_d;
        aoclfftz_execute = aoclfftz_execute_d;
    }
};

/**
 * @brief A parameterized derived class from AoclfftzSelectorTestBase for FLOAT
 * data-type and ILP64 data-model
 *
 */
class AoclfftzSelectorTestFloatILP64Parameterized
    : public AoclfftzSelectorTestBase<FLOAT, INTP, aoclfftz_dim_t_64_,
                                      aoclfftz_prob_desc_f_64_>,
      public ::testing::TestWithParam<aoclfftz_selector_test_params_t>
{
  protected:
    SETUP_SELECTOR_TEST_WITH_PARAMS_CLASS;
    AoclfftzSelectorTestFloatILP64Parameterized()
    {
        aoclfftz_setup = aoclfftz_setup_f_64_;
        aoclfftz_execute = aoclfftz_execute_f_64_;
    }
};

/**
 * @brief A derived class from AoclfftzSelectorTestBase for DOUBLE data-type and
 * ILP64 data-model
 *
 */
class AoclfftzSelectorTestDoubleILP64Parameterized
    : public AoclfftzSelectorTestBase<DOUBLE, INTP, aoclfftz_dim_t_64_,
                                      aoclfftz_prob_desc_d_64_>,
      public ::testing::TestWithParam<aoclfftz_selector_test_params_t>
{
  protected:
    SETUP_SELECTOR_TEST_WITH_PARAMS_CLASS;
    AoclfftzSelectorTestDoubleILP64Parameterized()
    {
        aoclfftz_setup = aoclfftz_setup_d_64_;
        aoclfftz_execute = aoclfftz_execute_d_64_;
    }
};

#endif // AOCLFFTZ_SELECTOR_GTEST_BASE_H