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

 /** @file selector.c
 *
 *  @brief Selects and returns a suitable solution for the input problem.
 *
 *  This file contains the implementation of functions that are used to
 *  select a solution of kernels for the given input problem description.
 *
 *  @author S. Biplab Raut
 */

#include "selector/selector.h"
#include "utils/utils.h"
#include "core/common/memory_manager.h"
#include "core/kernels/kernel.h"


//Tables of kernels that are populated with applicable kernels at setup time.
//There are 8 sets of kernels contained in the kernels_table which are :
//c, perm_c, avx128, perm_avx128, avx256, perm_avx256, avx512, perm_avx512
//Each set is having 64 enteries populated based on support for the kernels of those radices.
//Permuted and Non-permuted kernels are stored separately.
kernel_t kernels_table[NUM_KERNEL_TABLES][NUM_KERNELS_IN_TABLE] = { {{0x0},}, };

//Register all applicable solvers and kernels into the respective tables
//based on the input problem and cpu arch flags
INT32 register_solvers_kernels(
                    kernel_t kertab[NUM_KERNEL_TABLES][NUM_KERNELS_IN_TABLE],
                    INT32 dt, INT32 cpu_flags)
{
    INT32 ret = SELECTOR_FAILURE;

    //Register Solvers
    ret = register_solvers(dt, cpu_flags);
    if (ret != SOLVER_SUCCESS)
        return SELECTOR_FAILURE;
    
    //Register Kernels
    ret = register_kernels(kertab, dt, cpu_flags);

    return ret;
}


INT32 smallest_prime_divisor(INT32 n)
{
    if (n <= 1)
        return n;
    
    INT32 divisor = 2;

    for (divisor = 2; (divisor * divisor) <= n; divisor++)
    {
        if (n % divisor == 0)
            return divisor;
    }

    return n;
}

INT32 check_prime_solvability_bluestein(
                            aoclfftz_decomp_scheme_t *decomp_scheme,
                            INT32 is_FFT_ker_supported)
{
    INT32 n = decomp_scheme->dims[0].n;
    INT32 dim_rank = decomp_scheme->dim_rank;
    INT32 vec_rank = decomp_scheme->vec_rank;
    INT32 smallest_prime_factor = 1;

    if (dim_rank != 1 || vec_rank != 0)
        return 0;

    smallest_prime_factor = smallest_prime_divisor(n);  
    if (smallest_prime_factor == n &&
        is_FFT_ker_supported != 0)
        return 1;

    return 0;
}

INT32 check_FFT_kernel_support(INT32 n)
{
    INT32 is_supported = 0, i;
    for (i = 0; i < NUM_KERNELS_IN_TABLE; i++)
    {
        if (kernels_table[STANDARD_KERNEL_TBL_IDX][i].radix == 0)
            break;
        if (kernels_table[STANDARD_KERNEL_TBL_IDX][i].radix == n)
        {
            is_supported = 1;
            break;
        }
    }

    return is_supported;
}

//Fixed decision logic and CPI based selector mode execution for the
//single-precision input problem based on the applicable tables
//of solvers and kernels
INT32 selector_fixed_mode_dft_(aoclfftz_selector_t *sel, kernel_t *kertab)
{
    aoclfftz_generic_solver_t* solver_obj = sel->solution->solver;
    INT32 ret = SELECTOR_FAILURE;
    INT32 vec_rank = sel->solution->decomp_scheme->vec_rank;
    INT32 dim_rank = sel->solution->decomp_scheme->dim_rank;
    INT32 is_FFT_ker_supported =
            check_FFT_kernel_support(sel->solution->decomp_scheme->dims[0].n);
    INT32 is_solvable_by_bluestein = 
            check_prime_solvability_bluestein(sel->solution->decomp_scheme,
                                              is_FFT_ker_supported);
    INT32 level1_cond1 = 0;
    INT32 level1_cond2 = 0;
    INT32 level2_cond = 0;

    SET_SELECTOR_MODE(sel->solution->decomp_scheme->flags, 
                      AOCLFFTZ_FIXED_SELECTOR_MODE);
    
    //SOLVER_BATCHED
    level1_cond1 = (vec_rank > 1);
    //SOLVER_NDIM
    level1_cond1 |= ((dim_rank > 1) << 1);
    //SOLVER_BLUESTEIN
    level1_cond1 |= (is_solvable_by_bluestein << 2);
    //SOLVER_BUFFERED
    level1_cond2 = IS_IN_PLACE(sel->solution->decomp_scheme->flags);
    //SOLVER_BUFFERED -> ToDo: Conditions to work with AOCLFFTZ_FIXED_SELECTOR_MODE
    level1_cond2 &= ((sel->solution->decomp_scheme->dims[0].n > 
                            MAX_GUARANTEED_CACHEABLE_SIZE) &&
                     (GET_SELECTOR_MODE(sel->solution->decomp_scheme->flags) ==
                            AOCLFFTZ_AUTO_SELECTOR_MODE));
    //SOLVER_PERM_KER
    level1_cond2 |= 
                   (IS_OUT_OF_ORDER(sel->solution->decomp_scheme->flags) << 1);
    //SOLVER_DIRECT
    level2_cond = is_FFT_ker_supported;
    //SOLVER_PFA
    //SOLVER_RADER

    /** Level 1 decisions : Solvers **/
    //Batched/vector FFT Solver
    if (level1_cond1 & 0x1)
    {
        solver_obj->solver_type = SOLVER_BATCHED;
        if (set_solver_fp(solver_obj) != SOLVER_SUCCESS)
            return SELECTOR_FAILURE;

        //call batched solver master
        ret = selector_batched_dft(sel, kertab);
        return ret;
    }
    //Multi-dimentional FFT Solver
    if (level1_cond1 & 0x2)
    {
        solver_obj->solver_type = SOLVER_NDIM;
        if (set_solver_fp(solver_obj) != SOLVER_SUCCESS)
            return SELECTOR_FAILURE;

        //call ndim solver master
        ret = selector_ndim_dft(sel, kertab);
        return ret;
    }
    //Large Primes - Bluestein FFT Solver
    if (level1_cond1 & 0x4)
    {
        solver_obj->solver_type = SOLVER_BLUESTEIN;
        if (set_solver_fp(solver_obj) != SOLVER_SUCCESS)
            return SELECTOR_FAILURE;

        //call bluestein solver master
        ret = selector_bluestein_dft(sel, kertab);
        return ret;
    }
    //Buffered FFT Solver
    if (level1_cond2 & 0x1)
    {
        solver_obj->solver_type = SOLVER_BUFFERED;
        if (set_solver_fp(solver_obj) != SOLVER_SUCCESS)
            return SELECTOR_FAILURE;

        //call buffered solver master
        ret = selector_buffered_dft(sel, kertab);
        return ret;
    }
    //Permuted (out-of-order output) FFT Solver
    if (level1_cond2 & 0x2)
    {
        solver_obj->solver_type = SOLVER_PERM_KER;
        if (set_solver_fp(solver_obj) != SOLVER_SUCCESS)
            return SELECTOR_FAILURE;

        //call permuted solver master
        ret = selector_permuted_dft(sel, kertab);
        return ret;
    }

    /** Level 2 decisions : CT Solver and Kernels **/
    if (level2_cond & 0x1)
    {
        solver_obj->solver_type = SOLVER_DIRECT;
        if (set_solver_fp(solver_obj) != SOLVER_SUCCESS)
            return SELECTOR_FAILURE;

        //Call Direct Solver master
        ret = selector_direct_dft(sel, kertab);
        return ret;
    }
    else
    {
        solver_obj->solver_type = SOLVER_CT;
        if (set_solver_fp(solver_obj) != SOLVER_SUCCESS)
            return SELECTOR_FAILURE;

        //Call CT Solver master
        ret = selector_ct_dft(sel, kertab);
        return ret;
    }

    return ret;
}

//Setup to find a solution for the input problem based on the
//applicable tables of solvers and kernels
//Common function for both single-precision and double-precision
INT32 setup_dft_(aoclfftz_selector_t *sel, kernel_t *kertab)
{
    INT32 ret;
    
#if AOCLFFTZ_SELECTOR_AUTO_TUNER_MODE == 0
    //Fixed decision logic and CPI based selector mode
    ret = selector_fixed_mode_dft_(sel, kertab);
#else
    //Auto tuner based selector mode
    ret = selector_autotuner_mode_dft_(sel, kertab);
#endif

    return ret;
}

//Selector interface function that performs setup for finding solution for a
//single-precision LP64 problem
VOID *setup_dft_f(aoclfftz_prob_desc_f *problem)
{
    INT32 ret = 0;
    INT32 cpu_flags = 0;
    aoclfftz_cntrl_params cntrl_params = problem->cntrl_params;
    aoclfftz_selector_t* sel_obj = NULL;
    
    //allocate selector object
    sel_obj = alloc_selector(problem->vec_rank, problem->dim_rank);
    if (sel_obj == NULL)
        return NULL;

    //Find CPU feature flags that will be used by dynamic dispatcher
    cpu_flags = setup_dynamic_dispatcher(cntrl_params.opt_off,
                                  cntrl_params.opt_level,
                                  cntrl_params.logger_mode);

    //Register solvers and kernels for solving the problem based on
    //input problem datatype, CPU flags and dynamic dispatcher FMV selection
    ret = register_solvers_kernels(kernels_table,
                                   DT_FLOAT, cpu_flags);
    if (ret == 0)
        return NULL;

    //Initialize decomposition scheme data object
    INIT_DECOMP_SCHEME(sel_obj, problem);
    SET_PRECISION(sel_obj->solution->decomp_scheme->flags, DT_FLOAT);

    //Select the best solution for the given input problem
    ret = setup_dft_(sel_obj, (kernel_t *)kernels_table);
    if (ret != SELECTOR_SUCCESS)
        return NULL;

    return sel_obj->solution;
}

//Selector interface function that performs setup for finding solution for a
//double-precision LP64 problem
VOID *setup_dft_d(aoclfftz_prob_desc_d *problem)
{
    INT32 ret = 0;
    INT32 cpu_flags = 0;
    aoclfftz_cntrl_params cntrl_params = problem->cntrl_params;
    aoclfftz_selector_t* sel_obj = NULL;

    //allocate selector object
    sel_obj = alloc_selector(problem->vec_rank, problem->dim_rank);
    if (sel_obj == NULL)
        return NULL;

    //Find CPU feature flags that will be used by dynamic dispatcher
    cpu_flags = setup_dynamic_dispatcher(cntrl_params.opt_off,
        cntrl_params.opt_level,
        cntrl_params.logger_mode);

    //Register solvers and kernels for solving the problem based on
    //input problem datatype, CPU flags and dynamic dispatcher FMV selection
    ret = register_solvers_kernels(kernels_table,
                                   DT_DOUBLE, cpu_flags);
    if (ret == 0)
        return NULL;

    //Initialize decomposition scheme data object
    INIT_DECOMP_SCHEME(sel_obj, problem);
    SET_PRECISION(sel_obj->solution->decomp_scheme->flags, DT_FLOAT);

    //Select the best solution for the given input problem
    ret = setup_dft_(sel_obj, (kernel_t *)kernels_table);
    if (ret != SELECTOR_SUCCESS)
        return NULL;

    return sel_obj->solution;
}

//Selector interface function that performs setup for finding solution for a
//single-precision ILP64 problem
VOID *setup_dft_f_64_(aoclfftz_prob_desc_f_64_ *problem)
{
    INT32 ret = 0;
    INT32 cpu_flags = 0;
    aoclfftz_cntrl_params cntrl_params = problem->cntrl_params;
    aoclfftz_selector_t* sel_obj = NULL;

    //allocate selector object
    sel_obj = alloc_selector(problem->vec_rank, problem->dim_rank);
    if (sel_obj == NULL)
        return NULL;

    //Find CPU feature flags that will be used by dynamic dispatcher
    cpu_flags = setup_dynamic_dispatcher(cntrl_params.opt_off,
        cntrl_params.opt_level,
        cntrl_params.logger_mode);

    //Register solvers and kernels for solving the problem based on
    //input problem datatype, CPU flags and dynamic dispatcher FMV selection
    ret = register_solvers_kernels(kernels_table,
                                   DT_FLOAT, cpu_flags);
    if (ret == 0)
        return NULL;

    //Initialize decomposition scheme data object
    INIT_DECOMP_SCHEME(sel_obj, problem);
    SET_PRECISION(sel_obj->solution->decomp_scheme->flags, DT_FLOAT);

    //Select the best solution for the given input problem
    ret = setup_dft_(sel_obj, (kernel_t *)kernels_table);
    if (ret != SELECTOR_SUCCESS)
        return NULL;

    return sel_obj->solution;
}

//Selector interface function that performs setup for finding solution for a
//double-precision ILP64 problem
VOID *setup_dft_d_64_(aoclfftz_prob_desc_d_64_ *problem)
{
    INT32 ret = 0;
    INT32 cpu_flags = 0;
    aoclfftz_cntrl_params cntrl_params = problem->cntrl_params;
    aoclfftz_selector_t* sel_obj = NULL;

    //allocate selector object
    sel_obj = alloc_selector(problem->vec_rank, problem->dim_rank);
    if (sel_obj == NULL)
        return NULL;

    //Find CPU feature flags that will be used by dynamic dispatcher
    cpu_flags = setup_dynamic_dispatcher(cntrl_params.opt_off,
        cntrl_params.opt_level,
        cntrl_params.logger_mode);

    //Register solvers and kernels for solving the problem based on
    //input problem datatype, CPU flags and dynamic dispatcher FMV selection
    ret = register_solvers_kernels(kernels_table,
                                   DT_DOUBLE, cpu_flags);
    if (ret == 0)
        return NULL;

    //Initialize decomposition scheme data object
    INIT_DECOMP_SCHEME(sel_obj, problem);
    SET_PRECISION(sel_obj->solution->decomp_scheme->flags, DT_FLOAT);

    //Select the best solution for the given input problem
    ret = setup_dft_(sel_obj, (kernel_t *)kernels_table);
    if (ret != SELECTOR_SUCCESS)
        return NULL;

    return sel_obj->solution;
}

VOID destroy_handle(VOID *handle)
{
    destroy_solution((aoclfftz_solution_t*)handle);
}