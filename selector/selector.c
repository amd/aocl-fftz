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
#include "core/common/twiddle.h"

// Function pointer to communicate the exact selector model for executing the problem
// Not thread-safe
typedef INT32(*selector_model_func_)(aoclfftz_selector_t*);
selector_model_func_ sel_fp = NULL;

// Different tables of kernels contains 4 sets of kernels which are :
// c, avx128, avx256, avx512
// Each set is having 64 entries populated based on support for the kernels
// of those radices.
// DFT kernels (Not thread-safe)
kernel_t kernels_dft_table[NUM_KERNELS_IN_TABLE] = { {0x0} };
// RDFT kernels (Not thread-safe)
kernel_t kernels_rdft_table[NUM_KERNELS_IN_TABLE] = { {0x0} };
// TDFT kernels (fused twiddle + dft kernels) (Not thread-safe)
kernel_t kernels_twid_dft_table[NUM_KERNELS_IN_TABLE] = { {0x0} };

//Register all applicable solvers and kernels into the respective tables
//based on the input problem and cpu arch flags
INT32 register_solvers_kernels(kernel_t kertab_dft[NUM_KERNELS_IN_TABLE],
                               kernel_t kertab_rdft[NUM_KERNELS_IN_TABLE],
                               kernel_t kertab_twid_dft[NUM_KERNELS_IN_TABLE],
                               INT32 dt,
                               INT32 dir, INT32 is_real, INT32 cpu_flags)
{
    INT32 ret = SELECTOR_FAILURE;

    // Register Solvers
    ret = register_solvers(dt, is_real, cpu_flags);
    if (ret != SOLVER_SUCCESS)
    {
        return SELECTOR_FAILURE;
    }

    //Register Kernels
    if (is_real)
    {
        ret = register_kernels(kertab_rdft, dt, dir, is_real, cpu_flags);
    }
    else
    {
        ret = register_kernels(kertab_dft, dt, dir, 0, cpu_flags);
        ret |= register_twid_kernels(kertab_twid_dft, dt, dir, 0, cpu_flags);
    }

    return ret;
}

INT32 check_FFT_kernel_support(INTP n, kernel_t *kernels_table)
{
    INT32 is_supported = 0, i;
    for (i = 0; i < NUM_KERNELS_IN_TABLE; i++)
    {
        if (kernels_table[i].radix == 0)
        {
            break;
        }
        if (kernels_table[i].radix == n)
        {
            is_supported = 1;
            break;
        }
    }

    return is_supported;
}

INTP check_CT_solvability(INTP n, kernel_t *kertab)
{
    INT32 ker_cat = 0;
    UINT32 radix = 0;
    for (ker_cat = 0; ker_cat < NUM_KERNELS_IN_TABLE; ker_cat++)
    {
        radix = kertab[ker_cat].radix;
        if (radix == 0) // End of suitable kernels in the list
        {
            break;
        }
        // Check if this radix can factorize the problem
        if ((n % radix) == 0)
        {
            return 1;
        }
    }
    return 0;
}

INT32 check_prime_solvability_bluestein(aoclfftz_decomp_scheme_t *decomp_scheme,
                                        INT32 is_FFT_ker_supported,
                                        kernel_t *kertab)
{
    INTP n = decomp_scheme->dims[0].n;
    INTP batch = decomp_scheme->vecs[0].n;
    INT32 dim_rank = decomp_scheme->dim_rank;
    INT32 vec_rank = decomp_scheme->vec_rank;

    if (n == 1 || dim_rank != 1 || vec_rank != 1 || batch != 1)
    {
        return 0;
    }

    // n is solvable by bluestein solver if is not solvable by direct
    // and CT solvers
    INT32 is_solvable_by_CT = check_CT_solvability(n, kertab);
    if (is_FFT_ker_supported == 0 && is_solvable_by_CT == 0)
    {
        return 1;
    }

    return 0;
}

INT32 is_prime(INT32 n)
{
    for (INT32 i = 2; i * i <= n; i++)
    {
        if (n % i == 0)
        {
            return 0;
        }
    }

    return 1;
}

// Check if the input problem contains a prime factor that requires a
// bluestein solver to solve.
INT32 check_bluestein_problem(aoclfftz_decomp_scheme_t *decomp_scheme)
{
    INT32 dim_rank = decomp_scheme->dim_rank;
    for (int i = 0; i < dim_rank; i++)
    {
        INTP n = decomp_scheme->dims[i].n;
        // Check for prime factors greater than 13 in (n)
        for (INT32 i = 17; i <= n; i++)
        {
            if (n % i == 0 && is_prime(i))
            {
                return 1;
            }
        }
    }

    return 0;
}

// Fixed decision logic and CPI based selector mode execution for the
// input problem based on the applicable tables of solvers and kernels
INT32 selector_fixed_mode_dft_(aoclfftz_selector_t *sel)
{
    aoclfftz_generic_solver_t *solver_obj = sel->solution->solver;
    kernel_t *kertab = kernels_dft_table;
    INT32 ret = SELECTOR_FAILURE;
    INT32 dim_rank = sel->solution->decomp_scheme->dim_rank;
    UINT32 avl_threads = sel->solution->decomp_scheme->thread_info->avl_threads;

    // TODO: Should be removed after supporting MT in bluestein solver
    if (check_bluestein_problem(sel->solution->decomp_scheme) &&
        (avl_threads > 1))
    {
        AOCLFFTZ_LOG_UNFORMATTED(INFO, INFO, "Multi Threaded execution is"
            " not supported for Bluestein solver, falling back to single"
            " threaded Execution");
        sel->solution->decomp_scheme->thread_info->avl_threads = 1;
        avl_threads = 1;
    }
    INT32 is_FFT_ker_supported =
            check_FFT_kernel_support(sel->solution->decomp_scheme->dims[0].n,
                                     kertab);
    INT32 is_solvable_by_bluestein =
            check_prime_solvability_bluestein(sel->solution->decomp_scheme,
                                              is_FFT_ker_supported, kertab);
    INT32 level1_cond1 = 0;
    INT32 level1_cond2 = 0;
    INT32 level2_cond = 0;
    INT32 standalone_transpose_cond = 0;

    SET_SELECTOR_MODE(sel->solution->decomp_scheme->flags,
                      AOCLFFTZ_FIXED_SELECTOR);

    if (sel->solution->decomp_scheme->vec_rank > 1)
    {
        fuse_vecs(sel->solution);
    }
    //SOLVER_BATCHED
    level1_cond1 =
            ((sel->solution->decomp_scheme->dims[0].n != 1) && /* size one */
            ((sel->solution->decomp_scheme->vec_rank > 1) ||  /* ND Batched */
            /* 1D Batched 1D Non-direct cases*/
            ((sel->solution->decomp_scheme->vecs[0].n > 1) &&
                                    !is_FFT_ker_supported) ||
            /* 1D Batched ND case*/
            (dim_rank > 1 &&
                sel->solution->decomp_scheme->vecs[0].n > 1)));
    // SOLVER_NDIM
    level1_cond1 |= ((dim_rank > 1) << 1);
    // SOLVER_BLUESTEIN
    level1_cond1 |= (is_solvable_by_bluestein << 2);
    // SOLVER_BUFFERED
    level1_cond2 = !(IS_OUT_OF_PLACE(sel->solution->decomp_scheme->flags));
    // SOLVER_BUFFERED
    // TODO: Conditions to work with AOCLFFTZ_FIXED_SELECTOR
    level1_cond2 &= ((sel->solution->decomp_scheme->dims[0].n >
                      MAX_GUARANTEED_CACHEABLE_SIZE) &&
                     (GET_SELECTOR_MODE(sel->solution->decomp_scheme->flags) ==
                      AOCLFFTZ_AUTO_SELECTOR));
    // SOLVER_PERM_KER
    level1_cond2 |= (IS_OUT_OF_ORDER(sel->solution->decomp_scheme->flags) << 1);
    // SOLVER_DIRECT
    level2_cond = is_FFT_ker_supported;
    // SOLVER_PFA
    // SOLVER_RADER

    // SOLVER_TRANSPOSE
    standalone_transpose_cond =
        GET_STANDALONE_TRANSPOSE(sel->solution->decomp_scheme->flags);

    /** Level 1 decisions : Solvers **/
    // Batched/vector FFT Solver
    if (level1_cond1 & 0x1)
    {
        if (avl_threads <= 1)
        {
            solver_obj->solver_type = SOLVER_BATCHED;
        }
        else
        {
            solver_obj->solver_type = SOLVER_MT_BATCHED;
        }

        if (set_solver_fp(solver_obj) != SOLVER_SUCCESS)
        {
            return SELECTOR_FAILURE;
        }

        // call batched solver master
        ret = selector_batched_dft(sel, kertab);
        return ret;
    }
    // Transpose Solver (Standalone)
    if (standalone_transpose_cond)
    {
        solver_obj->solver_type = SOLVER_TRANSPOSE;
        if (set_solver_fp(solver_obj) != SOLVER_SUCCESS)
            return SELECTOR_FAILURE;

        // call the transpose solver selector
        ret = selector_transpose(sel);
        return ret;
    }
    // Multi-dimentional FFT Solver
    if (level1_cond1 & 0x2)
    {
        solver_obj->solver_type = SOLVER_NDIM;
        if (set_solver_fp(solver_obj) != SOLVER_SUCCESS)
        {
            return SELECTOR_FAILURE;
        }

        // call ndim solver master
        ret = selector_ndim_dft(sel, kertab);
        return ret;
    }
    // Large Primes - Bluestein FFT Solver
    if (level1_cond1 & 0x4)
    {
        solver_obj->solver_type = SOLVER_BLUESTEIN;
        if (set_solver_fp(solver_obj) != SOLVER_SUCCESS)
        {
            return SELECTOR_FAILURE;
        }

        // call bluestein solver master
        ret = selector_bluestein_dft(sel, kertab);
        return ret;
    }
    // Buffered FFT Solver
    if (level1_cond2 & 0x1)
    {
        solver_obj->solver_type = SOLVER_BUFFERED;
        if (set_solver_fp(solver_obj) != SOLVER_SUCCESS)
        {
            return SELECTOR_FAILURE;
        }

        // call buffered solver master
        ret = selector_buffered_dft(sel, kertab);
        return ret;
    }
    // Permuted (out-of-order output) FFT Solver
    if (level1_cond2 & 0x2)
    {
        solver_obj->solver_type = SOLVER_PERM_KER;
        if (set_solver_fp(solver_obj) != SOLVER_SUCCESS)
        {
            return SELECTOR_FAILURE;
        }

        // call permuted solver master
        ret = selector_permuted_dft(sel, kertab);
        return ret;
    }
    /** Level 2 decisions : CT Solver and Kernels **/
    // SizeOne FFT Solver
    if (sel->solution->decomp_scheme->dims[0].n == 1)
    {
        solver_obj->solver_type = SOLVER_SIZEONE;
        if (set_solver_fp(solver_obj) != SOLVER_SUCCESS)
        {
            return SELECTOR_FAILURE;
        }

        // Call SizeOne Solver master
        ret = selector_sizeone_dft(sel, kertab);
        return ret;
    }
    else if (level2_cond & 0x1)
    {
        if (avl_threads <= 1)
        {
            solver_obj->solver_type = SOLVER_DIRECT;
        }
        else
        {
            solver_obj->solver_type = SOLVER_MT_DIRECT;
        }
        if (set_solver_fp(solver_obj) != SOLVER_SUCCESS)
        {
            return SELECTOR_FAILURE;
        }

        // Call Direct Solver master
        ret = selector_direct_dft(sel, kertab);
        return ret;
    }
    else
    {
        solver_obj->solver_type = SOLVER_CT;
        if (set_solver_fp(solver_obj) != SOLVER_SUCCESS)
        {
            return SELECTOR_FAILURE;
        }

        // Call CT Solver master
        ret = selector_ct_dft(sel, kertab);
        return ret;
    }

    return ret;
}

// Fixed decision logic and CPI based selector mode execution for the
// input problem based on the applicable tables of solvers and kernels
// wherein the successive stage dfts are fused with twiddle multiplications
INT32 selector_fixed_mode_fused_twid_dft_(aoclfftz_selector_t* sel)
{
    aoclfftz_generic_solver_t* solver_obj = sel->solution->solver;
    kernel_t *kertab = kernels_dft_table;
    INT32 ret = SELECTOR_FAILURE;
    INT32 dim_rank = sel->solution->decomp_scheme->dim_rank;
    UINT32 avl_threads = sel->solution->decomp_scheme->thread_info->avl_threads;

    // TODO: Should be removed after supporting MT in N-D and bluestein solver
    if ((dim_rank > 1 || check_bluestein_problem(sel->solution->decomp_scheme))
        && avl_threads > 1)
    {
        AOCLFFTZ_LOG_UNFORMATTED(INFO, INFO, "Multi Threaded execution is"
            " not suported for N-D & Bluestein solver, falling back to Single"
            " Threaded execution");
        sel->solution->decomp_scheme->thread_info->avl_threads = 1;
        avl_threads = 1;
    }
    INT32 is_FFT_ker_supported =
        check_FFT_kernel_support(sel->solution->decomp_scheme->dims[0].n,
            kertab);
    INT32 is_solvable_by_bluestein =
        check_prime_solvability_bluestein(sel->solution->decomp_scheme,
            is_FFT_ker_supported, kertab);
    INT32 level1_cond1 = 0;
    INT32 level1_cond2 = 0;
    INT32 level2_cond = 0;
    INT32 standalone_transpose_cond = 0;

    SET_SELECTOR_MODE(sel->solution->decomp_scheme->flags,
        AOCLFFTZ_FIXED_SELECTOR);

    if (sel->solution->decomp_scheme->vec_rank > 1)
    {
        fuse_vecs(sel->solution);
    }
    //SOLVER_BATCHED
    level1_cond1 =
        ((sel->solution->decomp_scheme->dims[0].n != 1) && /* size one */
            ((sel->solution->decomp_scheme->vec_rank > 1) ||  /* ND Batched */
                /* 1D Batched 1D Non-direct cases*/
                ((sel->solution->decomp_scheme->vecs[0].n > 1) &&
                    !is_FFT_ker_supported) ||
                /* 1D Batched ND case*/
                (dim_rank > 1 &&
                    sel->solution->decomp_scheme->vecs[0].n > 1)));
    // SOLVER_NDIM
    level1_cond1 |= ((dim_rank > 1) << 1);
    // SOLVER_BLUESTEIN
    level1_cond1 |= (is_solvable_by_bluestein << 2);
    // SOLVER_BUFFERED
    level1_cond2 = !(IS_OUT_OF_PLACE(sel->solution->decomp_scheme->flags));
    // SOLVER_BUFFERED
    // TODO: Conditions to work with AOCLFFTZ_FIXED_SELECTOR
    level1_cond2 &= ((sel->solution->decomp_scheme->dims[0].n >
        MAX_GUARANTEED_CACHEABLE_SIZE) &&
        (GET_SELECTOR_MODE(sel->solution->decomp_scheme->flags) ==
            AOCLFFTZ_AUTO_SELECTOR));
    // SOLVER_PERM_KER
    level1_cond2 |= (IS_OUT_OF_ORDER(sel->solution->decomp_scheme->flags) << 1);
    // SOLVER_DIRECT
    level2_cond = is_FFT_ker_supported;
    // SOLVER_PFA
    // SOLVER_RADER

    // SOLVER_TRANSPOSE
    standalone_transpose_cond =
        GET_STANDALONE_TRANSPOSE(sel->solution->decomp_scheme->flags);

    /** Level 1 decisions : Solvers **/
    // Batched/vector FFT Solver
    if (level1_cond1 & 0x1)
    {
        if (avl_threads <= 1)
        {
            solver_obj->solver_type = SOLVER_BATCHED;
        }
        else
        {
            solver_obj->solver_type = SOLVER_MT_BATCHED;
        }

        if (set_solver_fp(solver_obj) != SOLVER_SUCCESS)
        {
            return SELECTOR_FAILURE;
        }

        // call batched solver master
        ret = selector_batched_dft(sel, kertab);
        return ret;
    }
    // Transpose Solver (Standalone)
    if (standalone_transpose_cond)
    {
        solver_obj->solver_type = SOLVER_TRANSPOSE;
        if (set_solver_fp(solver_obj) != SOLVER_SUCCESS)
            return SELECTOR_FAILURE;

        // call the transpose solver selector
        ret = selector_transpose(sel);
        return ret;
    }
    // Multi-dimentional FFT Solver
    if (level1_cond1 & 0x2)
    {
        solver_obj->solver_type = SOLVER_NDIM;
        if (set_solver_fp(solver_obj) != SOLVER_SUCCESS)
        {
            return SELECTOR_FAILURE;
        }

        // call ndim solver master
        ret = selector_ndim_dft(sel, kertab);
        return ret;
    }
    // Large Primes - Bluestein FFT Solver
    if (level1_cond1 & 0x4)
    {
        solver_obj->solver_type = SOLVER_BLUESTEIN;
        if (set_solver_fp(solver_obj) != SOLVER_SUCCESS)
        {
            return SELECTOR_FAILURE;
        }

        // call bluestein solver master
        ret = selector_bluestein_dft(sel, kertab);
        return ret;
    }
    // Buffered FFT Solver
    if (level1_cond2 & 0x1)
    {
        solver_obj->solver_type = SOLVER_BUFFERED;
        if (set_solver_fp(solver_obj) != SOLVER_SUCCESS)
        {
            return SELECTOR_FAILURE;
        }

        // call buffered solver master
        ret = selector_buffered_dft(sel, kertab);
        return ret;
    }
    // Permuted (out-of-order output) FFT Solver
    if (level1_cond2 & 0x2)
    {
        solver_obj->solver_type = SOLVER_PERM_KER;
        if (set_solver_fp(solver_obj) != SOLVER_SUCCESS)
        {
            return SELECTOR_FAILURE;
        }

        // call permuted solver master
        ret = selector_permuted_dft(sel, kertab);
        return ret;
    }
    /** Level 2 decisions : CT Solver and Kernels **/
    // SizeOne FFT Solver
    if (sel->solution->decomp_scheme->dims[0].n == 1)
    {
        solver_obj->solver_type = SOLVER_SIZEONE;
        if (set_solver_fp(solver_obj) != SOLVER_SUCCESS)
        {
            return SELECTOR_FAILURE;
        }

        // Call SizeOne Solver master
        ret = selector_sizeone_dft(sel, kertab);
        return ret;
    }
    else if (level2_cond & 0x1)
    {
        if (avl_threads <= 1)
        {
            solver_obj->solver_type = SOLVER_DIRECT;
        }
        else
        {
            solver_obj->solver_type = SOLVER_MT_DIRECT;
        }
        if (set_solver_fp(solver_obj) != SOLVER_SUCCESS)
        {
            return SELECTOR_FAILURE;
        }

        //Direct sub-problems will use normal dft kernels
        //since they are leaf sub-problems always
        kertab = kernels_dft_table;

        // Call Direct Solver master
        ret = selector_direct_dft(sel, kertab);
        return ret;
    }
    else
    {
        solver_obj->solver_type = SOLVER_CT;
        if (set_solver_fp(solver_obj) != SOLVER_SUCCESS)
        {
            return SELECTOR_FAILURE;
        }

        //All the CT sub-problems will use fused twiddle dft kernels
        //since they are non-leaf sub-problems always
        kertab = kernels_twid_dft_table;

        // Call CT Solver master
        ret = selector_ct_dft(sel, kertab);
        return ret;
    }

    return ret;
}

// Fixed decision logic and CPI based selector mode execution for the
// real input problem based on the applicable tables
// of real solvers and real kernels
INT32 selector_fixed_mode_rdft_(aoclfftz_selector_t *sel,
                                aoclfftz_realhelper_t *realhelper)
{
    aoclfftz_generic_solver_t *solver_obj = sel->solution->solver;
    kernel_t *kertab = kernels_rdft_table;
    INT32 logger_mode = sel->solution->decomp_scheme->cntrl_params->logger_mode;
    INT32 ret = SELECTOR_FAILURE;
    INT32 vec_rank = sel->solution->decomp_scheme->vec_rank;
    INT32 dim_rank = sel->solution->decomp_scheme->dim_rank;
    INT32 is_FFT_ker_supported =
            check_FFT_kernel_support(sel->solution->decomp_scheme->dims[0].n,
                                     kertab);
    INT32 is_solvable_by_bluestein = 0;
    INT32 level1_cond1 = 0;
    INT32 level1_cond2 = 0;
    INT32 level2_cond = 0;

    SET_SELECTOR_MODE(sel->solution->decomp_scheme->flags,
                      AOCLFFTZ_FIXED_SELECTOR);

    // SOLVER_BATCHED
    level1_cond1 =
        ((sel->solution->decomp_scheme->dims[0].n != 1) && /* size one */
         ((vec_rank > 1) ||                                /* ND Batched */
          /* 1D Batched 1D Non-direct cases*/
          ((sel->solution->decomp_scheme->vecs[0].n > 1) &&
           !is_FFT_ker_supported &&
           (sel->solution->decomp_scheme->dims[0].n ==
            realhelper->problem_size)) ||
          /* 1D Batched ND case*/
          (dim_rank > 1 && sel->solution->decomp_scheme->vecs[0].n > 1)));
    // SOLVER_NDIM
    level1_cond1 |= ((dim_rank > 1) << 1);
    // SOLVER_BLUESTEIN
    level1_cond1 |= (is_solvable_by_bluestein << 2);
    // SOLVER_BUFFERED
    // Buffered solver will be used for all CT problems as of now
    level1_cond2 = !realhelper->is_buffered_invoked && !is_FFT_ker_supported;
    // SOLVER_PERM_KER
    level1_cond2 |= (IS_OUT_OF_ORDER(sel->solution->decomp_scheme->flags) << 1);
    // SOLVER_DIRECT
    level2_cond = is_FFT_ker_supported;
    // SOLVER_PFA
    // SOLVER_RADER

    /** Level 1 decisions : Solvers **/
    // Batched/vector FFT Solver
    if (level1_cond1 & 0x1)
    {
        solver_obj->solver_type = SOLVER_REAL_BATCHED;
        if (set_solver_fp(solver_obj) != SOLVER_SUCCESS)
        {
            return SELECTOR_FAILURE;
        }

        ret = selector_batched_rdft(sel, kertab, realhelper);
        return ret;
    }
    // Multi-dimentional FFT Solver
    if (level1_cond1 & 0x2)
    {
        AOCLFFTZ_LOG_UNFORMATTED(ERR, logger_mode,
            "Multi-dimentional RealFFT is not supported");
        return SELECTOR_FAILURE;
    }
    // Large Primes - Bluestein FFT Solver
    if (level1_cond1 & 0x4)
    {
        AOCLFFTZ_LOG_UNFORMATTED(ERR, logger_mode,
            "Large Prime RealFFT is not supported");
        return SELECTOR_FAILURE;
    }
    // Buffered FFT Solver
    if (level1_cond2 & 0x1)
    {
        solver_obj->solver_type = SOLVER_REAL_BUFFERED;
        if (set_solver_fp(solver_obj) != SOLVER_SUCCESS)
        {
            return SELECTOR_FAILURE;
        }

        ret = selector_buffered_rdft(sel, kertab, realhelper);
        return ret;
    }
    // Permuted (out-of-order output) FFT Solver
    if (level1_cond2 & 0x2)
    {
        AOCLFFTZ_LOG_UNFORMATTED(ERR, logger_mode,
            "Permuted RealFFT is not supported");
        return SELECTOR_FAILURE;
    }
    /** Level 2 decisions : CT Solver and Kernels **/
    // Size one problem
    if (sel->solution->decomp_scheme->dims[0].n == 1)
    {
        solver_obj->solver_type = SOLVER_SIZEONE;
        if (set_solver_fp(solver_obj) != SOLVER_SUCCESS)
        {
            return SELECTOR_FAILURE;
        }

        // No setup for SizeOne problem
        return SELECTOR_SUCCESS;
    }
    else if (level2_cond & 0x1)
    {
        solver_obj->solver_type = SOLVER_REAL_DIRECT;
        if (set_solver_fp(solver_obj) != SOLVER_SUCCESS)
        {
            return SELECTOR_FAILURE;
        }

        // Call Direct Solver master
        ret = selector_direct_rdft(sel, kertab, realhelper);
        return ret;
    }
    else
    {
        solver_obj->solver_type = SOLVER_REAL_CT;
        if (set_solver_fp(solver_obj) != SOLVER_SUCCESS)
        {
            return SELECTOR_FAILURE;
        }

        // Call CT Solver master
        ret = selector_ct_rdft(sel, kertab, realhelper);
        return ret;
    }

    return ret;
}

INT32 selector_autotuner_mode_dft_(aoclfftz_selector_t* sel)
{
    INT32 logger_mode = sel->solution->decomp_scheme->cntrl_params->logger_mode;
    AOCLFFTZ_LOG_UNFORMATTED(INFO, logger_mode,
        "Autotuner selector is not yet available for evaluation");
    return SELECTOR_FAILURE;
}

// Main selector driver that invokes the complementary/alternate selector
// algorithms/models and decides on the final selector based on its suitability
// and performance.
// Also provides a cleaner approach to init and handle various related
// solvers and kernel tables
INT32 selector_driver_dft_(aoclfftz_selector_t* sel)
{
    INT32 ret = SELECTOR_FAILURE;
    aoclfftz_selector_t *sel_models[AOCLFFTZ_SELECTOR_MODELS] = { 0x0, };
    UINT32 best_model_id = 0;
    cost_analysis_t best_cost = {INT64_MAX, INT64_MAX};
    INT32 vec_rank = sel->solution->decomp_scheme->vec_rank;
    INT32 dim_rank = sel->solution->decomp_scheme->dim_rank;

    // FIXED SELECTOR BASED MODEL : start
#ifdef AOCLFFTZ_FIXED_SELECTOR_MODE
    // Allocate selector object
    sel_models[AOCLFFTZ_FIXED_SELECTOR] =
        alloc_selector(vec_rank, dim_rank, sel->scratch_space);
    if (sel_models[AOCLFFTZ_FIXED_SELECTOR] != NULL)
    {
        COPY_DECOMP_SCHEME(sel_models[AOCLFFTZ_FIXED_SELECTOR]->solution->decomp_scheme,
            sel->solution->decomp_scheme);
        SET_PRECISION(sel_models[AOCLFFTZ_FIXED_SELECTOR]->solution->decomp_scheme->flags,
            DT_PRECISION_FLAG(sel->solution->decomp_scheme->flags));
        SET_STANDALONE_TRANSPOSE(sel_models[AOCLFFTZ_FIXED_SELECTOR]->solution->decomp_scheme->flags,
            GET_STANDALONE_TRANSPOSE(sel->solution->decomp_scheme->flags));

        // Fixed decision logic and CPI based selector mode
        //ret = selector_fixed_mode_dft_(sel_models[AOCLFFTZ_FIXED_SELECTOR]);
        sel_fp = selector_fixed_mode_dft_;
        ret = selector_model_dft_(sel_models[AOCLFFTZ_FIXED_SELECTOR]);
        if (ret != SELECTOR_FAILURE)
        {
            *(sel_models[AOCLFFTZ_FIXED_SELECTOR]->cost_analysis) = *(sel->cost_analysis);
        }
    }
#endif
    // FIXED SELECTOR BASED MODEL : end

    // FIXED SELECTOR + FUSED TWIDDLE DFT BASED MODEL : start
    // Allocate selector object
#ifdef AOCLFFTZ_FIXED_SELECTOR_FUSED_TWID_DFT_MODE
    sel_models[AOCLFFTZ_FIXED_SELECTOR_FUSED_TWID_DFT] =
        alloc_selector(vec_rank, dim_rank, sel->scratch_space);
    if (sel_models[AOCLFFTZ_FIXED_SELECTOR_FUSED_TWID_DFT] != NULL)
    {
        COPY_DECOMP_SCHEME(sel_models[AOCLFFTZ_FIXED_SELECTOR_FUSED_TWID_DFT]->solution->decomp_scheme,
            sel->solution->decomp_scheme);
        SET_PRECISION(sel_models[AOCLFFTZ_FIXED_SELECTOR_FUSED_TWID_DFT]->solution->decomp_scheme->flags,
            DT_PRECISION_FLAG(sel->solution->decomp_scheme->flags));
        SET_STANDALONE_TRANSPOSE(sel_models[AOCLFFTZ_FIXED_SELECTOR_FUSED_TWID_DFT]->solution->decomp_scheme->flags,
            GET_STANDALONE_TRANSPOSE(sel->solution->decomp_scheme->flags));

        // Fixed decision logic and CPI based selector mode
        //ret = selector_fixed_mode_fused_twid_dft_(sel_models[AOCLFFTZ_FIXED_SELECTOR_FUSED_TWID_DFT]);
        sel_fp = selector_fixed_mode_fused_twid_dft_;
        ret = selector_model_dft_(sel_models[AOCLFFTZ_FIXED_SELECTOR_FUSED_TWID_DFT]);
        if (ret != SELECTOR_FAILURE)
        {
            *(sel_models[AOCLFFTZ_FIXED_SELECTOR_FUSED_TWID_DFT]->cost_analysis) = *(sel->cost_analysis);
        }
    }
#endif
    // FIXED SELECTOR + FUSED TWIDDLE DFT BASED MODEL : end

    // AUTO TUNED SELECTOR BASED MODEL : start
    // Allocate selector object
#ifdef AOCLFFTZ_AUTO_SELECTOR_MODE
    sel_models[AOCLFFTZ_AUTO_SELECTOR] =
        alloc_selector(vec_rank, dim_rank, sel->scratch_space);
    if (sel_models[AOCLFFTZ_AUTO_SELECTOR] != NULL)
    {
        COPY_DECOMP_SCHEME(sel_models[AOCLFFTZ_AUTO_SELECTOR]->solution->decomp_scheme,
            sel->solution->decomp_scheme);
        SET_PRECISION(sel_models[AOCLFFTZ_AUTO_SELECTOR]->solution->decomp_scheme->flags,
            DT_PRECISION_FLAG(sel->solution->decomp_scheme->flags));
        SET_STANDALONE_TRANSPOSE(sel_models[AOCLFFTZ_AUTO_SELECTOR]->solution->decomp_scheme->flags,
            GET_STANDALONE_TRANSPOSE(sel->solution->decomp_scheme->flags));

        // Fixed decision logic and CPI based selector mode
        //ret = selector_autotuner_mode_dft_(sel_models[AOCLFFTZ_AUTO_SELECTOR]);
        sel_fp = selector_autotuner_mode_dft_;
        ret = selector_model_dft_(sel_models[AOCLFFTZ_AUTO_SELECTOR]);
        if (ret != SELECTOR_FAILURE)
        {
            *(sel_models[AOCLFFTZ_AUTO_SELECTOR]->cost_analysis) = *(sel->cost_analysis);
        }
    }
#endif
    // AUTO TUNED SELECTOR BASED MODEL : end

    // Compare and find the best performant model for the input problem
    for (UINT32 model_id = 0; model_id < AOCLFFTZ_SELECTOR_MODELS; model_id++)
    {
        if (sel_models[model_id] != NULL)
        {
            if (sel_models[model_id]->cost_analysis->ops < best_cost.ops)
            {
                best_cost = *(sel_models[model_id]->cost_analysis);
                best_model_id = model_id;
            }
        }
    }
    // Update the primary selector object with the best selector model based solution
    aoclfftz_solution_t *org_sol = sel->solution;
    sel->solution = sel_models[best_model_id]->solution;
    *(sel->cost_analysis) = *(sel_models[best_model_id]->cost_analysis);

    // Destroy and Free unnecessary objects
    destroy_solution(org_sol);
    for (UINT32 model_id = 0; model_id < AOCLFFTZ_SELECTOR_MODELS; model_id++)
    {
        if (model_id != best_model_id)
            destroy_selector_without_scratch_space(sel_models[model_id]);
        else
            destroy_selector_without_solution(sel_models[model_id]);
    }

    return ret;
}

// Specific Selector model function that invokes the internal recursive
// selector function pointer passing the selector boject along.
// Common function for both single-precision and double-precision
INT32 selector_model_dft_(aoclfftz_selector_t *sel)
{
    INT32 ret = SELECTOR_FAILURE;

    ret = sel_fp(sel);

    return ret;
}

// Setup to find a solution for the input problem based on the
// applicable tables of real-solvers and real-kernels
// Common function for both single-precision and double-precision
INT32 setup_rdft_(aoclfftz_selector_t *sel,
                  aoclfftz_realhelper_t *realhelper)
{
    INT32 ret;

    // Fixed decision logic and CPI based selector mode
    ret = selector_fixed_mode_rdft_(sel, realhelper);

    return ret;
}

// Selector interface function that performs setup for finding solution for a
// single-precision LP64 problem
VOID *setup_dft_f(aoclfftz_prob_desc_f *problem)
{
    INT32 ret = 0;
    INT32 cpu_flags = 0;
    aoclfftz_cntrl_params_t cntrl_params = problem->cntrl_params;
    aoclfftz_selector_t *sel_obj = NULL;

    // shrink dim_rank
    // used in n dim case where size one problems are removed
    INT32 dim_rank = 1;
    SHRINK_DIM_RANK(problem->dims, problem->dim_rank, dim_rank);

    // allocate selector object
    sel_obj = alloc_selector(problem->vec_rank, dim_rank, NULL);
    if (sel_obj == NULL)
    {
        return NULL;
    }

    // Find CPU feature flags that will be used by dynamic dispatcher
    cpu_flags = setup_dynamic_dispatcher(cntrl_params.opt_off,
                                         cntrl_params.opt_level,
                                         cntrl_params.logger_mode);

    //Register solvers and kernels for solving the problem based on
    //input problem datatype, CPU flags and dynamic dispatcher FMV selection
    ret = register_solvers_kernels(kernels_dft_table,
                                   kernels_rdft_table,
                                   kernels_twid_dft_table, DT_FLOAT,
                                   FFT_DIR(problem->flags),
                                   IS_REAL(problem->flags),
                                   cpu_flags);
    if (ret != 0)
    {
        goto exit_setup_dft_f;
    }

    // Initialize decomposition scheme data object
    INIT_DECOMP_SCHEME(sel_obj, problem, dim_rank);
    SET_PRECISION(sel_obj->solution->decomp_scheme->flags, DT_FLOAT);

    // Note: Currently the 8th bit of the flags member of both, the problem and
    // the selector represent the same thing -> a standalone transpose
    // operation. Hence the reuse of the same macro is valid. If this no longer
    // holds for whatever reason, change this "GET_STANDALONE_TRANSPOSE" macro
    // accordingly.
    if (GET_STANDALONE_TRANSPOSE(problem->flags))
    {
        SET_STANDALONE_TRANSPOSE(sel_obj->solution->decomp_scheme->flags, 1);
    }

    // Select the best solution for the given input problem
    PREPARE_AND_SETUP_DFT(sel_obj, ret);
    if (ret != SELECTOR_SUCCESS)
    {
        goto exit_setup_dft_f;
    }

    post_process_solution(sel_obj->solution);
    return sel_obj;

exit_setup_dft_f:
    destroy_selector(sel_obj);
    return NULL;
}

// Selector interface function that performs setup for finding solution for a
// double-precision LP64 problem
VOID *setup_dft_d(aoclfftz_prob_desc_d *problem)
{
    INT32 ret = 0;
    INT32 cpu_flags = 0;
    aoclfftz_cntrl_params_t cntrl_params = problem->cntrl_params;
    aoclfftz_selector_t *sel_obj = NULL;

    // shrink dim_rank
    // used in n dim case where size one problems are removed
    INT32 dim_rank = 1;
    SHRINK_DIM_RANK(problem->dims, problem->dim_rank, dim_rank);

    // allocate selector object
    sel_obj = alloc_selector(problem->vec_rank, dim_rank, NULL);
    if (sel_obj == NULL)
    {
        return NULL;
    }

    // Find CPU feature flags that will be used by dynamic dispatcher
    cpu_flags = setup_dynamic_dispatcher(cntrl_params.opt_off,
                                         cntrl_params.opt_level,
                                         cntrl_params.logger_mode);

    //Register solvers and kernels for solving the problem based on
    //input problem datatype, CPU flags and dynamic dispatcher FMV selection
    ret = register_solvers_kernels(kernels_dft_table,
                                   kernels_rdft_table,
                                   kernels_twid_dft_table, DT_DOUBLE,
                                   FFT_DIR(problem->flags),
                                   IS_REAL(problem->flags),
                                   cpu_flags);
    if (ret != 0)
    {
        goto exit_setup_dft_d;
    }

    // Initialize decomposition scheme data object
    INIT_DECOMP_SCHEME(sel_obj, problem, dim_rank);
    SET_PRECISION(sel_obj->solution->decomp_scheme->flags, DT_DOUBLE);

    // Note: Currently the 8th bit of the flags member of both, the problem and
    // the selector represent the same thing -> a standalone transpose
    // operation. Hence the reuse of the same macro is valid. If this no longer
    // holds for whatever reason, change this "GET_STANDALONE_TRANSPOSE" macro
    // accordingly.
    if (GET_STANDALONE_TRANSPOSE(problem->flags))
    {
        SET_STANDALONE_TRANSPOSE(sel_obj->solution->decomp_scheme->flags, 1);
    }

    // Select the best solution for the given input problem
    PREPARE_AND_SETUP_DFT(sel_obj, ret);
    if (ret != SELECTOR_SUCCESS)
    {
        goto exit_setup_dft_d;
    }

    post_process_solution(sel_obj->solution);
    return sel_obj;

exit_setup_dft_d:
    destroy_selector(sel_obj);
    return NULL;
}

// Selector interface function that performs setup for finding solution for a
// single-precision ILP64 problem
VOID *setup_dft_f_64_(aoclfftz_prob_desc_f_64_ *problem)
{
    INT32 ret = 0;
    INT32 cpu_flags = 0;
    aoclfftz_cntrl_params_t cntrl_params = problem->cntrl_params;
    aoclfftz_selector_t *sel_obj = NULL;

    // shrink dim_rank
    // used in n dim case where size one problems are removed
    INT32 dim_rank = 1;
    SHRINK_DIM_RANK(problem->dims, problem->dim_rank, dim_rank);

    // allocate selector object
    sel_obj = alloc_selector(problem->vec_rank, dim_rank, NULL);
    if (sel_obj == NULL)
    {
        return NULL;
    }

    // Find CPU feature flags that will be used by dynamic dispatcher
    cpu_flags = setup_dynamic_dispatcher(cntrl_params.opt_off,
                                         cntrl_params.opt_level,
                                         cntrl_params.logger_mode);

    //Register solvers and kernels for solving the problem based on
    //input problem datatype, CPU flags and dynamic dispatcher FMV selection
    ret = register_solvers_kernels(kernels_dft_table,
                                   kernels_rdft_table,
                                   kernels_twid_dft_table, DT_FLOAT,
                                   FFT_DIR(problem->flags),
                                   IS_REAL(problem->flags),
                                   cpu_flags);
    if (ret != 0)
    {
        goto exit_setup_dft_f_64_;
    }

    // Initialize decomposition scheme data object
    INIT_DECOMP_SCHEME(sel_obj, problem, dim_rank);
    SET_PRECISION(sel_obj->solution->decomp_scheme->flags, DT_FLOAT);

    // Note: Currently the 8th bit of the flags member of both, the problem and
    // the selector represent the same thing -> a standalone transpose
    // operation. Hence the reuse of the same macro is valid. If this no longer
    // holds for whatever reason, change this "GET_STANDALONE_TRANSPOSE" macro
    // accordingly.
    if (GET_STANDALONE_TRANSPOSE(problem->flags))
    {
        SET_STANDALONE_TRANSPOSE(sel_obj->solution->decomp_scheme->flags, 1);
    }

    // Select the best solution for the given input problem
    PREPARE_AND_SETUP_DFT(sel_obj, ret);
    if (ret != SELECTOR_SUCCESS)
    {
        goto exit_setup_dft_f_64_;
    }

    post_process_solution(sel_obj->solution);
    return sel_obj;

exit_setup_dft_f_64_:
    destroy_selector(sel_obj);
    return NULL;
}

// Selector interface function that performs setup for finding solution for a
// double-precision ILP64 problem
VOID *setup_dft_d_64_(aoclfftz_prob_desc_d_64_ *problem)
{
    INT32 ret = 0;
    INT32 cpu_flags = 0;
    aoclfftz_cntrl_params_t cntrl_params = problem->cntrl_params;
    aoclfftz_selector_t *sel_obj = NULL;

    // shrink dim_rank
    // used in n dim case where size one problems are removed
    INT32 dim_rank = 1;
    SHRINK_DIM_RANK(problem->dims, problem->dim_rank, dim_rank);

    // allocate selector object
    sel_obj = alloc_selector(problem->vec_rank, dim_rank, NULL);
    if (sel_obj == NULL)
    {
        return NULL;
    }

    // Find CPU feature flags that will be used by dynamic dispatcher
    cpu_flags = setup_dynamic_dispatcher(cntrl_params.opt_off,
                                         cntrl_params.opt_level,
                                         cntrl_params.logger_mode);

    //Register solvers and kernels for solving the problem based on
    //input problem datatype, CPU flags and dynamic dispatcher FMV selection
    ret = register_solvers_kernels(kernels_dft_table,
                                   kernels_rdft_table,
                                   kernels_twid_dft_table, DT_DOUBLE,
                                   FFT_DIR(problem->flags),
                                   IS_REAL(problem->flags),
                                   cpu_flags);
    if (ret != 0)
    {
        goto exit_setup_dft_d_64_;
    }

    // Initialize decomposition scheme data object
    INIT_DECOMP_SCHEME(sel_obj, problem, dim_rank);
    SET_PRECISION(sel_obj->solution->decomp_scheme->flags, DT_DOUBLE);

    // Note: Currently the 8th bit of the flags member of both, the problem and
    // the selector represent the same thing -> a standalone transpose
    // operation. Hence the reuse of the same macro is valid. If this no longer
    // holds for whatever reason, change this "GET_STANDALONE_TRANSPOSE" macro
    // accordingly.
    if (GET_STANDALONE_TRANSPOSE(problem->flags))
    {
        SET_STANDALONE_TRANSPOSE(sel_obj->solution->decomp_scheme->flags, 1);
    }

    // Select the best solution for the given input problem
    PREPARE_AND_SETUP_DFT(sel_obj, ret);
    if (ret != SELECTOR_SUCCESS)
    {
        goto exit_setup_dft_d_64_;
    }

    post_process_solution(sel_obj->solution);
    return sel_obj;

exit_setup_dft_d_64_:
    destroy_selector(sel_obj);
    return NULL;
}

VOID destroy_handle(VOID *handle)
{
    destroy_selector((aoclfftz_selector_t *)handle);
}

VOID fuse_vecs(aoclfftz_solution_t *sol)
{
    INT32 last_fused_idx = 0, is_fusable = 0, fused_rank = 0;
    INTP fused_size = sol->decomp_scheme->vecs[0].n;
    INT32 vec_rank = sol->decomp_scheme->vec_rank;
    aoclfftz_dim_t_64_ *vecs = sol->decomp_scheme->vecs;
    for (INT32 i = 1; i < vec_rank; i++)
    {
        // expected stride is the regular stride we obtain by n * stride of
        // prev dim
        INTP expected_in_stride = vecs[i-1].n * vecs[i-1].in_stride;
        INTP expected_out_stride = vecs[i-1].n * vecs[i-1].out_stride;
        INTP actual_in_stride = vecs[i].in_stride;
        INTP actual_out_stride = vecs[i].out_stride;
        // mark the vector for fusing and compute the new size for the fused
        // vector
        if (expected_in_stride == actual_in_stride &&
             expected_out_stride == actual_out_stride)
        {
            is_fusable = 1;
            fused_size = fused_size * vecs[i].n;
        }
        else
        {
            if (is_fusable)
            {
                vecs[fused_rank].n = fused_size;
            }
            else
            {
                vecs[fused_rank].n = vecs[last_fused_idx].n;
            }
            vecs[fused_rank].in_stride = vecs[last_fused_idx].in_stride;
            vecs[fused_rank].out_stride = vecs[last_fused_idx].out_stride;
            fused_size = vecs[i].n;
            last_fused_idx = i;
            fused_rank++;
            is_fusable = 0;
        }
    }
    //initalize last vector
    vecs[fused_rank].n = fused_size;
    vecs[fused_rank].out_stride = vecs[last_fused_idx].out_stride;
    vecs[fused_rank].in_stride = vecs[last_fused_idx].in_stride;
    sol->decomp_scheme->vec_rank = fused_rank + 1;
}

VOID setup_twiddle_buffer_complex(aoclfftz_solution_t *solution)
{
#if IN_MEMORY_TWIDDLE_FACTORS == 1
    aoclfftz_solution_t *curr = solution;
    UINT32 dt_prec = DT_PRECISION_FLAG(solution->decomp_scheme->flags);
    while (curr != NULL)
    {
        if (curr->solver->solver_type == SOLVER_CT)
        {
            INTP r = curr->next_sol[0]->decomp_scheme->dims[0].n;
            INTP m = curr->next_sol[0]->next_sol[0]->decomp_scheme->dims[0].n;

            VOID *TW = alloc_twiddle_buffer(r * m, dt_prec);
            if (TW != NULL)
            {
                compute_twiddle_buffer(TW, r, m, dt_prec);
                curr->next_sol[0]->twiddle->TW = TW;
                curr->next_sol[0]->twiddle->twiddle_buf_ptr = TW;
            }
        }
        curr = curr->next_sol ? curr->next_sol[0] : NULL;
    }
#endif
}

VOID setup_twiddle_buffer_real(aoclfftz_solution_t *solution)
{
#if IN_MEMORY_TWIDDLE_FACTORS == 1
    UINT32 dt_prec = DT_PRECISION_FLAG(solution->decomp_scheme->flags);
    if (FFT_DIR(solution->decomp_scheme->flags) == FORWARD_FFT_DIR)
    {
        // aoclfftz_solution_t *curr = solution;
        aoclfftz_solution_t *prev = solution;
        FOR_EACH_SOLUTION(curr, solution)
        {
            // Always inherit the parent's twiddle buffer reference.
            curr->twiddle->TW = prev->twiddle->TW;
            if (curr->solver->solver_type == SOLVER_REAL_CT)
            {
                // goto next direct node to setup twiddle buffer
                while (curr != NULL &&
                       curr->solver->solver_type != SOLVER_REAL_DIRECT)
                {
                    curr->twiddle->TW = prev->twiddle->TW;
                    curr = curr->next_sol[0];
                }
                INTP n = curr->decomp_scheme->dims[0].n;
                INTP no_of_groups = curr->solver->batches[R2HCF_KERNEL] > 0
                                        ? curr->solver->batches[R2HCF_KERNEL]
                                        : curr->solver->batches[R2HC_KERNEL];
                // No. of c2c kernels per group that require twiddle computation
                INTP group_size = curr->solver->batches[C2C_KERNEL] / no_of_groups;
                INTP tw_buf_size = n * group_size * DATA_STRIDE;
                // Allocate Twiddle buffer to store twiddle values for every
                // radix-n c2c kernel of a group
                VOID *TW = alloc_twiddle_buffer(tw_buf_size, dt_prec);
                if (TW != NULL)
                {
                    INTP p = (curr->decomp_scheme->vecs[0].n *
                              curr->decomp_scheme->dims[0].n) /
                             no_of_groups;
                    compute_twiddle_buffer_real(TW, group_size, n, p, FORWARD_FFT_DIR,
                                                dt_prec);
                    curr->twiddle->TW = TW;
                    curr->twiddle->twiddle_buf_ptr = TW;
                }
            }
            prev = curr;
        }
    }
    else
    {
        aoclfftz_solution_t *prev = solution;
        FOR_EACH_SOLUTION(curr, (solution->next_sol) ? solution->next_sol[0] : NULL)
        {
            if (curr->solver->solver_type == SOLVER_REAL_CT &&
                prev->solver->solver_type == SOLVER_REAL_DIRECT)
            {
                INTP n = prev->decomp_scheme->dims[0].n;
                INTP no_of_groups = prev->solver->batches[R2HCF_KERNEL] > 0
                                        ? prev->solver->batches[R2HCF_KERNEL]
                                        : prev->solver->batches[R2HC_KERNEL];
                // No. of c2c kernels per group that require twiddle computation
                INTP group_sz = prev->solver->batches[C2C_KERNEL] / no_of_groups;
                INTP tw_buf_sz = n * group_sz * DATA_STRIDE;
                // Allocate Twiddle buffer to store twiddle factors for every
                // radix-n c2c kernel of a group
                VOID *TW = alloc_twiddle_buffer(tw_buf_sz, dt_prec);
                if (TW != NULL)
                {
                    INTP p = (prev->decomp_scheme->vecs[0].n *
                              prev->decomp_scheme->dims[0].n) /
                             no_of_groups;
                    compute_twiddle_buffer_real(TW, group_sz, n, p,
                                                BACKWARD_FFT_DIR, dt_prec);
                    prev->twiddle->TW = TW;
                    prev->twiddle->twiddle_buf_ptr = TW;
                }
            }
            // Always inherit the parent's twiddle buffer reference.
            curr->twiddle->TW = prev->twiddle->TW;
            prev = curr;
        }
    }
#endif
}

aoclfftz_solution_t *deep_copy_solution_tree(aoclfftz_solution_t* src)
{
    if (!src)
    {
        return NULL;
    }

    // Copy the solution object and its strides
    INT32 vec_rank = src->decomp_scheme->vec_rank;
    INT32 dim_rank = src->decomp_scheme->dim_rank;
    aoclfftz_solution_t* dst = alloc_solution(vec_rank, dim_rank);
    COPY_SOLUTION_OBJ(dst, src);
    COPY_STRIDES(dst, src);

    // Recursively copy nd_sol if present (for NDIM solvers)
    dst->dft_bufs->nd_sol = src->dft_bufs->nd_sol ?
                        deep_copy_solution_tree(src->dft_bufs->nd_sol) : NULL;

    // Initiate deep copy of next_sol recursively until leaf node
    if (src->next_sol)
    {
        int n = (src->solver->solver_type == SOLVER_MT_BATCHED) ?
                              src->decomp_scheme->thread_info->n_threads : 1;
        dst->next_sol = alloc_sol_array(n);

        for (int i = 0; i < n; i++)
        {
            dst->next_sol[i] = deep_copy_solution_tree(src->next_sol[0]);
        }
    }
    else
    {
        dst->next_sol = NULL;
    }
    return dst;
}

VOID post_process_solution(aoclfftz_solution_t *sol)
{
    while (sol != NULL)
    {
        if (sol->solver->solver_type == SOLVER_MT_BATCHED)
        {
            // call post process recursively to find the innermost MT batched
            // solution
            post_process_solution(sol->next_sol[0]);

            // replicate the solution in next_sol[0] to array of next_sols in
            // each MT batched solution
            for (int i = 1; i < sol->decomp_scheme->thread_info->n_threads; i++)
            {
                sol->next_sol[i] = deep_copy_solution_tree(sol->next_sol[0]);
            }
            break;
        }
        if (sol->solver->solver_type == SOLVER_NDIM && sol->dft_bufs->nd_sol)
        {
            post_process_solution(sol->dft_bufs->nd_sol);
        }
        sol = sol->next_sol ? sol->next_sol[0] : NULL;
    }
}
